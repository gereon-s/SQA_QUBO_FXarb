# coding: utf-8
"""
FX Arbitrage Detection Using Simulated Quantum Annealing

This script implements FX arbitrage detection by fomulating the problem as a
Quadratic Unconstrained Binary Optimization (QUBO) problem and solving it
using Simulated Quantum Annealing (SQA).

Key Parts:
- Models FX markets as a directed graph where currencies are nodes and exchange rates are edges
- Uses negative log transformation to convert multiplicative arbitrage into additive cycles
- Applies quantum-inspired optimization to find profitable trading cycles

Mathematical Foundation:
- Edge weights: w(u,v) = -log(exchange_rate * (1 - transaction_cost))
- Arbitrage cycle profit: product of rates > 1 ⟺ sum of negative log weights < 0
- QUBO formulation ensures flow conservation while maximizing cycle profitability

Dependencies:
    – networkx, numpy, pandas, scipy, dimod, numba, joblib, optuna

Files needed to run this script:
    * sqa_implementation.py (custom SQA sampler implementation)
    * fx_data_march_april_2025.csv (FX mid-rates indexed by timestamp)
"""
from __future__ import annotations

import math
import os
import logging
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union
from logging.handlers import RotatingFileHandler

import networkx as nx
import numpy as np
import pandas as pd
from dimod import BinaryQuadraticModel, BINARY

from sqa_implementation import SimulatedQuantumAnnealingSampler

###############################################################################
# 0. Logging Configuration ###################################################
###############################################################################

LOGGER = logging.getLogger("main")
if not LOGGER.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

###############################################################################
# A. Constants ################################################################
###############################################################################

# Guard against numerical overflow in log calculations
# When exchange rates approach zero, -log(rate) approaches infinity
SENTINEL = 1_000.0  # Cap edge weights to prevent solver instability

###############################################################################
# B. QUBO (Quadratic Unconstrained Binary Optimization) Builder ##############
###############################################################################


class QUBOBuilder:
    """
    Constructs a QUBO formulation for the FX arbitrage detection problem.

    The QUBO formulation balances three objectives:
    1. Maximize cycle profitability (negative log weights encourage profitable paths)
    2. Enforce flow conservation (penalty for unbalanced in/out flows at each node)
    3. Penalize empty solutions (encourages non-trivial cycle detection)

    Mathematical Formulation:
    minimize: Σᵢⱼ wᵢⱼ·xᵢⱼ + P·Σₙ(inflow_n - outflow_n)² + C·|selected_edges|

    where:
    - xᵢⱼ ∈ {0,1}: binary variable indicating if edge (i,j) is selected
    - wᵢⱼ: normalized edge weight (negative log of effective exchange rate)
    - P: flow conservation penalty multiplier
    - C: cycle penalty multiplier (prevents empty solutions)
    """

    REQUIRED = {"objective_scale", "flow_penalty_mul", "cycle_penalty_mul"}

    def __init__(self, params: Dict[str, float]):
        """Initialize QUBO builder with penalty parameters."""
        missing = self.REQUIRED - params.keys()
        if missing:
            raise ValueError(f"Missing QUBO parameters: {missing}")
        self.params = params

    def build_qubo(self, network: "ArbitrageNetwork") -> BinaryQuadraticModel:
        """
        Construct a Binary Quadratic Model (BQM) for the arbitrage detection problem.

        The BQM formulation uses adaptive scaling based on the maximum edge weight
        to ensure numerical stability across different market conditions.

        Returns:
            BinaryQuadraticModel: QUBO
        """
        # Extract scaling parameters
        obj_scale = self.params["objective_scale"]  # Overall problem scale
        P_mul = self.params["flow_penalty_mul"]  # Flow conservation strength
        C_mul = self.params["cycle_penalty_mul"]  # Empty solution penalty

        # Initialize empty Binary Quadratic Model
        bqm = BinaryQuadraticModel({}, {}, 0.0, BINARY)

        # 1. Filter edges with finite weights (avoid numerical overflow)
        edges: List[Tuple[str, str, Dict[str, Any]]] = [
            (u, v, d)
            for u, v, d in network.G.edges(data=True)
            if abs(d["weight"]) < SENTINEL  # Only include stable edge weights
        ]

        if not edges:
            return bqm  # Return empty BQM if no valid edges

        # 2. Adaptive weight normalization
        # Scale edge weights relative to maximum to maintain solver stability
        weights = [abs(d["weight"]) for *_, d in edges]
        max_w = max(weights) if weights else 1.0
        if max_w == 0:  # Prevent division by zero
            max_w = 1.0

        # Calculate penalty strengths
        flow_P = P_mul * obj_scale  # Flow conservation penalty
        cycle_pen = C_mul * obj_scale  # Empty solution penalty

        # 3. Add binary variables for each edge
        edge_var = {(u, v): f"x_{u}_{v}" for u, v, _ in edges}
        for (u, v), var in edge_var.items():
            # Linear coefficient: normalized weight encourages profitable edges
            # Subtract cycle penalty to discourage empty solutions
            norm_w = network.G[u][v]["weight"] / max_w  # Normalize to [-1,1] range
            linear_coeff = norm_w * obj_scale - cycle_pen
            bqm.add_variable(var, linear_coeff)

        # Offset adjustment: penalize empty edge selection
        bqm.offset += len(edge_var) * cycle_pen

        # 4. Flow conservation constraints
        # For each currency node, ensure balanced in/out flow in selected cycles
        in_edges: Dict[str, List[str]] = {n: [] for n in network.currencies}
        out_edges: Dict[str, List[str]] = {n: [] for n in network.currencies}

        # Categorize edges by their impact on each currency's flow balance
        for (u, v), var in edge_var.items():
            if u in out_edges:
                out_edges[u].append(var)  # Outgoing edge from currency u
            if v in in_edges:
                in_edges[v].append(var)  # Incoming edge to currency v

        # Add flow conservation penalty terms
        for node in network.currencies:
            ins, outs = in_edges[node], out_edges[node]

            # Linear penalty terms: discourage unbalanced flows
            for vi in ins + outs:
                bqm.add_linear(vi, flow_P)

            # Quadratic penalty terms enforce flow balance
            # Same-direction interactions (encourage balanced selection)
            for group in (ins, outs):
                for i, vi in enumerate(group):
                    for vj in group[i + 1 :]:
                        bqm.add_quadratic(vi, vj, 2 * flow_P)

            # Cross in/out interaction (discourage flow imbalance)
            for vi in ins:
                for vj in outs:
                    bqm.add_quadratic(vi, vj, -2 * flow_P)

        return bqm


###############################################################################
# C. Smart Parameter Space ####################################################
###############################################################################


class ParameterSpace:
    """
    Parameter sampler for QUBO and SQA hyperparameters.

    Uses coupled penalty design principles:
    - Flow penalties can be "loose" (allow some imbalance) or "strict"
    - Cycle penalties prevent trivial empty solutions
    - SQA parameters follow quantum annealing params
    """

    def __init__(self, *, allow_loose: bool = True, objective_hi: float = 10_000.0):
        """
        Initialize parameter space sampler.

        Args:
            allow_loose: Allow loose flow conservation (50% chance)
            objective_hi: Maximum objective scale for problem normalization
        """
        self.allow_loose = allow_loose
        self.objective_hi = objective_hi

    def _log_uniform(self, low: float, high: float) -> float:
        """Sample from log-uniform distribution for scale-invariant parameters."""
        return float(np.exp(np.random.uniform(math.log(low), math.log(high))))

    def sample(self) -> Dict[str, Any]:
        """
        Generate a random parameter configuration.

        Parameter design rationale:
        - objective_scale: Overall problem magnitude (log-uniform for scale invariance)
        - flow_penalty_mul: Balance between strict/loose flow conservation
        - cycle_penalty_mul: Prevent empty solutions without over-constraining
        - SQA parameters: Follow quantum annealing temperature/field schedules

        Returns:
            Dict containing all parameters needed for QUBO construction and SQA solving
        """
        params: Dict[str, Any] = {}

        # QUBO scaling parameters
        params["objective_scale"] = self._log_uniform(1.0, self.objective_hi)

        # Flow conservation strategy: loose vs strict
        if self.allow_loose and bool(np.random.randint(0, 2)):
            # Loose: allow some flow imbalance (Idea: may find more diverse cycles)
            params["flow_penalty_mul"] = self._log_uniform(0.001, 0.1)
        else:
            # Strict: enforce tight flow conservation (cleaner cycles)
            params["flow_penalty_mul"] = self._log_uniform(0.1, 2.0)

        # Empty solution penalty (uniform sampling)
        params["cycle_penalty_mul"] = np.random.uniform(0.0, 0.5)

        # Simulated Quantum Annealing parameters
        params.update(
            # Sampling parameters
            num_reads=int(
                np.random.choice([100, 500, 1000, 2000, 5000])
            ),  # Number of independent runs
            num_sweeps=int(
                np.random.choice([250, 500, 1000, 2000])
            ),  # Monte Carlo steps per run
            num_trotter_slices=int(
                np.random.choice([8, 16, 32, 64])
            ),  # Quantum simulation fidelity
            # Annealing schedule (transverse field simulates quantum tunneling)
            transverse_field_start=self._log_uniform(
                1.0, 10.0
            ),  # Initial quantum fluctuation
            transverse_field_end=self._log_uniform(0.005, 0.5),  # Final classical limit
            # Temperature schedule (controls thermal fluctuations)
            beta_start=self._log_uniform(0.001, 1.0),  # Initial temperature (high)
            beta_end=self._log_uniform(1.0, 20.0),  # Final temperature (low)
            # Problem-specific constraints
            seed=int(np.random.randint(0, 2**32)),  # Reproducibility
            min_cycle_length=int(
                np.random.choice([3, 4])
            ),  # Minimum profitable cycle size
            max_cycle_length=int(
                np.random.choice([4, 5, 6, 7, 8])
            ),  # Maximum trackable cycle size
        )

        # Ensure logical consistency
        if params["max_cycle_length"] < params["min_cycle_length"]:
            params["max_cycle_length"] = params["min_cycle_length"]

        return params


###############################################################################
# 1. Core Data Structures #####################################################
###############################################################################


class ArbitrageNetwork:
    """
    Represents FX market as a directed graph for arbitrage detection.

    Model:
    - Nodes: Currency codes (USD, EUR, GBP, etc.)
    - Edges: Exchange rates with transaction costs
    - Edge weights: w(u,v) = -log(effective_rate) where effective_rate = rate * (1 - cost)

    Arbitrage Detection Principle:
    A cycle has arbitrage potential if the product of exchange rates > 1
    Equivalently: sum of edge weights < 0 (since weights are negative logs)
    """

    def __init__(self) -> None:
        self.G: nx.DiGraph = nx.DiGraph()  # Directed graph for exchange rates
        self.currencies: set[str] = set()  # Set of all currency codes

    def add_rate(
        self,
        base: str,
        quote: str,
        rate: float,
        transaction_costs: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> None:
        """
        Add bidirectional exchange rate to the network.

        Creates two directed edges:
        1. base → quote with weight -log(rate * (1 - cost))
        2. quote → base with weight -log((1/rate) * (1 - cost))

        Args:
            base: Base currency code
            quote: Quote currency code
            rate: Exchange rate (base/quote)
            transaction_costs: Optional dict of (from,to) → cost percentage
        """
        if rate <= 0:
            raise ValueError(f"Invalid rate for {base}/{quote}: {rate}")

        # Extract transaction costs (default to 0)
        tc = transaction_costs or {}
        cost_fwd = tc.get((base, quote), 0.0)  # Cost for base → quote
        cost_bwd = tc.get((quote, base), 0.0)  # Cost for quote → base

        # Validate transaction costs
        if not (0 <= cost_fwd < 1) or not (0 <= cost_bwd < 1):
            raise ValueError("Transaction cost must be ∈[0,1).")

        # Calculate effective rates after transaction costs
        eff_fwd = rate * (1 - cost_fwd)  # Effective forward rate
        eff_bwd = (1 / rate) * (1 - cost_bwd)  # Effective reverse rate

        # Convert to negative log weights (avoid log(0) with small epsilon check)
        w_fwd = -math.log(eff_fwd) if eff_fwd > 1e-15 else SENTINEL
        w_bwd = -math.log(eff_bwd) if eff_bwd > 1e-15 else SENTINEL

        # Add bidirectional edges to graph
        self.G.add_edge(base, quote, weight=w_fwd)
        self.G.add_edge(quote, base, weight=w_bwd)
        self.currencies.update({base, quote})

    def cycle_profit_pct(self, cycle: List[str]) -> float:
        """
        Calculate exact percentage profit for a trading cycle.

        Mathematical derivation:
        - Start with 1 unit of first currency
        - Multiply by exchange rates around the cycle
        - Final amount = product of all rates
        - Profit percentage = (final_amount - 1) * 100

        Args:
            cycle: List of currencies forming a closed loop

        Returns:
            Profit percentage (e.g., 0.37 means 0.37% profit)
        """
        if not cycle:
            return 0.0

        # Calculate product of exchange rates around the cycle
        prod = 1.0
        for u, v in zip(cycle, cycle[1:] + [cycle[0]]):  # Close the cycle
            if not self.G.has_edge(u, v) or "weight" not in self.G[u][v]:
                LOGGER.warning(
                    f"Edge {u}->{v} not found in profit calculation for cycle {cycle}"
                )
                return -float("inf")  # Invalid cycle

            # Convert back from log weight to exchange rate
            prod *= math.exp(-self.G[u][v]["weight"])

        # Return profit as percentage
        return (prod - 1.0) * 100.0


###############################################################################
# 2. Arbitrage Detection Engine ###############################################
###############################################################################

# Required parameters for arbitrage detector
a_REQUIRED = {
    "objective_scale",
    "flow_penalty_mul",
    "cycle_penalty_mul",  # QUBO parameters
    "min_cycle_length",
    "max_cycle_length",  # Cycle constraints
    "num_reads",
    "num_sweeps",
    "num_trotter_slices",  # SQA sampling
    "transverse_field_start",
    "transverse_field_end",  # Quantum field schedule
    "beta_start",
    "beta_end",  # Temperature schedule
}


class ArbitrageDetector:
    """
    Main engine for detecting FX arbitrage opportunities using SQA.

    Pipeline:
    1. Convert FX rates to weighted directed graph
    2. Formulate arbitrage detection as QUBO problem
    3. Solve using Simulated Quantum Annealing
    4. Extract profitable cycles from quantum solutions
    5. Calculate exact profit percentages
    """

    def __init__(
        self,
        params: Dict[str, Any],
        transaction_cost_bps: float = 0.0002,  # Default 0.02% (2 basis points)
    ) -> None:
        """
        Initialize arbitrage detector with specified parameters.

        Args:
            params: Dictionary containing all required QUBO and SQA parameters
            transaction_cost_bps: Transaction cost as decimal (0.0002 = 0.02%)
        """
        # Validate required parameters
        missing = a_REQUIRED - params.keys()
        if missing:
            raise ValueError(f"Missing detector parameters: {missing}")

        self.params = params
        self.transaction_cost = transaction_cost_bps

        # Initialize core components
        self.network = ArbitrageNetwork()
        self.qubo_builder = QUBOBuilder(params)
        self.sampler = SimulatedQuantumAnnealingSampler()

        self.logger = self._setup_logger()
        self.current_timestamp = None

    def _setup_logger(self) -> logging.Logger:
        """Configure logging with rotation to prevent disk space issues."""
        lg = logging.getLogger("arbitrage_detector")
        if not lg.handlers:
            lg.setLevel(logging.INFO)
            log_file = "arbitrage_detector.log"
            try:
                # Use rotating file handler to manage log size
                fh = RotatingFileHandler(
                    log_file, maxBytes=5 * 1024 * 1024, backupCount=3
                )
                fh.setFormatter(
                    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
                )
                lg.addHandler(fh)
            except Exception as e:
                # Fallback to console if file logging fails
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
                    )
                )
                lg.addHandler(console_handler)
                lg.error(f"Failed to set up file logging: {e}. Using console output.")
        return lg

    def add_market_data(self, rates: Dict[str, float], timestamp=None) -> None:
        """
        Update network with latest market snapshot.

        Completely rebuilds the arbitrage network with new rates.
        This approach ensures clean state for each time point.

        Args:
            rates: Dictionary mapping currency pairs to exchange rates
                  Format: {"USD/EUR": 0.85, "EUR/GBP": 0.90, ...}
            timestamp: Optional timestamp for tracking
        """
        self.current_timestamp = timestamp
        self.network = ArbitrageNetwork()  # Fresh network for new data

        # Build transaction cost mapping
        tc_dict: Dict[Tuple[str, str], float] = {}
        for pair_str in rates:
            if "/" not in pair_str:
                self.logger.warning(f"Invalid pair format: {pair_str}. Skipping.")
                continue
            base, quote = pair_str.split("/")
            # Apply symmetric transaction costs
            tc_dict[(base, quote)] = self.transaction_cost
            tc_dict[(quote, base)] = self.transaction_cost

        # Add all exchange rates to network
        for pair_str, rate in rates.items():
            if "/" not in pair_str:
                continue
            base, quote = pair_str.split("/")
            try:
                self.network.add_rate(base, quote, rate, tc_dict)
            except ValueError as exc:
                self.logger.error("Failed to add rate for %s: %s", pair_str, exc)

    def find_arbitrage(self, allow_empty: bool = False) -> List[Dict[str, Any]]:
        """
        Detect arbitrage opportunities using SQA.

        Algorithm:
        1. Build QUBO formulation from current network
        2. Sample solutions using SQA
        3. Extract cycles from binary solutions
        4. Filter cycles by length and profitability
        5. Return ranked opportunities

        Args:
            allow_empty: Whether to include solutions with no selected edges

        Returns:
            List of arbitrage opportunities sorted by profit (descending)
        """
        # Check if network has any edges
        if not self.network.G.edges:
            self.logger.info("No edges in network. Skipping arbitrage detection.")
            return []

        # Build QUBO formulation
        bqm = self.qubo_builder.build_qubo(self.network)
        if not bqm.variables:
            self.logger.info(
                "QUBO has no variables (all weights were sentinel). Skipping."
            )
            return []

        # Solve using SQA
        response = self.sampler.sample(
            bqm,
            num_reads=self.params["num_reads"],
            num_sweeps=self.params["num_sweeps"],
            num_trotter_slices=self.params["num_trotter_slices"],
            beta_start=self.params["beta_start"],
            beta_end=self.params["beta_end"],
            transverse_field_start=self.params["transverse_field_start"],
            transverse_field_end=self.params["transverse_field_end"],
            seed=self.params.get("seed"),
        )

        # Log best energy found
        min_E = (
            float(np.min(response.record["energy"]))
            if len(response) > 0
            and response.record is not None
            and "energy" in response.record.dtype.names
            and len(response.record["energy"]) > 0
            else float("inf")
        )
        self.logger.info("Best QUBO energy found: %.6f", min_E)

        # Extract cycles from quantum solutions
        return self._extract_cycles_from_response(response, allow_empty)

    def _extract_cycles_from_response(
        self, response, allow_empty: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Extract and validate profitable cycles from SQA response.

        Process:
        1. Parse variable names to recover edge information
        2. For each sample, build subgraph from selected edges
        3. Find all simple cycles in subgraph
        4. Filter by length constraints and profitability
        5. Eliminate duplicates using canonical cycle representation
        """
        var_names = list(response.bqm.variables)

        # Parse edge information from variable names (format: "x_BASE_QUOTE")
        try:
            edges = [tuple(name[2:].split("_", 1)) for name in var_names]
            if any(len(e) != 2 for e in edges):
                self.logger.error(
                    "Malformed variable names found. Cannot extract edges."
                )
                return []
        except Exception as e:
            self.logger.error(f"Error parsing variable names to edges: {e}")
            return []

        var_to_idx = {v: i for i, v in enumerate(var_names)}

        opps: List[Dict[str, Any]] = []
        seen: set[Tuple[str, ...]] = set()  # Track seen cycles to avoid duplicates

        if response.record is None or len(response.record) == 0:
            self.logger.info("SQA sampler returned no samples.")
            return []

        # Process each sample from the SQA response
        for sample_vec, energy, occ in zip(
            response.record["sample"],
            response.record["energy"],
            response.record["num_occurrences"],
        ):
            # Extract selected edges (where binary variable = 1)
            chosen_edges_from_sample: List[Tuple[str, str]] = []
            for i, var_name in enumerate(var_names):
                if sample_vec[var_to_idx[var_name]] == 1:
                    chosen_edges_from_sample.append(edges[i])

            if not chosen_edges_from_sample and not allow_empty:
                continue

            # Build subgraph from selected edges
            subg = nx.DiGraph()
            subg.add_edges_from(chosen_edges_from_sample)

            # Find all simple cycles in the subgraph
            for cycle in nx.simple_cycles(subg):
                L = len(cycle)
                min_len = self.params.get("min_cycle_length", 3)
                max_len = self.params.get("max_cycle_length", 5)

                # Filter by cycle length constraints
                if not (min_len <= L <= max_len):
                    continue

                # Create canonical representation to avoid duplicates
                if L > 0:
                    canon = min(tuple(cycle[i:] + cycle[:i]) for i in range(L))
                else:
                    continue

                if canon in seen:
                    continue
                seen.add(canon)

                # Calculate exact profit percentage
                profit = self.network.cycle_profit_pct(cycle)

                # Filter out unprofitable cycles
                if profit <= 1e-5:
                    continue

                # Store opportunity with metadata
                opps.append(
                    {
                        "cycle": cycle,
                        "profit": round(profit, 6),
                        "energy": float(energy),
                        "timestamp": self.current_timestamp,
                        "num_occurrences": int(occ),
                    }
                )

        # Return opportunities sorted by profit (highest first)
        return sorted(opps, key=lambda x: x["profit"], reverse=True)


###############################################################################
# 3. Data Management and Experimentation #####################################
###############################################################################


class MarketDataManager:
    """
    Handles loading and preprocessing of FX market data.

    Expects CSV format with:
    - DateTime index (first column)
    - Currency pair columns (e.g., "USD/EUR", "EUR/GBP")
    - Exchange rates as values
    """

    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        self.path = csv_path
        self.logger = logging.getLogger("market_data_manager")

    def fetch(self) -> pd.DataFrame:
        """
        Load and validate FX data from CSV file.

        Returns:
            DataFrame with DatetimeIndex and currency pair columns
        """
        try:
            # Load CSV with first column as datetime index
            df = pd.read_csv(self.path, index_col=0, parse_dates=True)

            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                    if not isinstance(df.index, pd.DatetimeIndex):
                        raise ValueError(
                            "CSV index could not be converted to DatetimeIndex."
                        )
                except Exception as e:
                    self.logger.error(f"Failed to parse index as datetime: {e}")
                    raise ValueError("CSV index must be datetime.") from e

            # Handle NaT values in index
            if df.index.hasnans:
                self.logger.warning(
                    "CSV index contains NaT values. Dropping invalid rows."
                )
                df = df[~df.index.isna()]

            return df
        except Exception as e:
            self.logger.error(f"Error processing CSV file {self.path}: {e}")
            raise


class ArbitrageExperiment:
    """
    Orchestrates arbitrage detection experiments across time series data.

    Processes market data chronologically, detecting arbitrage opportunities
    at each timestamp and aggregating results for analysis.
    """

    def __init__(self, params: Dict[str, Any], results_dir: str = "results"):
        """
        Initialize experiment with detector parameters.

        Args:
            params: Complete parameter dictionary for QUBO and SQA
            results_dir: Directory for saving experiment results
        """
        self.params = params
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        # Initialize detector with specified parameters
        self.detector = ArbitrageDetector(params)
        self.logger = logging.getLogger("arbitrage_experiment")

    def run(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Execute arbitrage detection across entire time series.

        For each timestamp:
        1. Extract valid exchange rates (drop NaN values)
        2. Update detector with new market data
        3. Run arbitrage detection
        4. Aggregate results

        Args:
            df: DataFrame with datetime index and currency pair columns

        Returns:
            List of all arbitrage opportunities found across time series
        """
        all_opps: List[Dict[str, Any]] = []

        if df.empty:
            self.logger.info("Input DataFrame is empty. No experiment run.")
            return []

        # Process each timestamp sequentially
        for ts, row in df.iterrows():
            self.logger.debug(f"Processing timestamp: {ts}")

            # Extract valid rates (drop NaN values)
            rates = row.dropna().to_dict()
            if not rates:
                self.logger.debug(f"No valid rates for {ts}. Skipping.")
                continue

            # Update detector and find arbitrage
            self.detector.add_market_data(rates, ts)
            try:
                opps_at_ts = self.detector.find_arbitrage(allow_empty=False)
                all_opps.extend(opps_at_ts)
            except Exception as e:
                self.logger.error(
                    f"Error during arbitrage detection for {ts}: {e}",
                    exc_info=True,
                )

        return all_opps

    def save(self, opps: List[Dict[str, Any]], fname: str = "opps.pkl") -> None:
        """Save experiment results to pickle file for later analysis."""
        if not fname.endswith(".pkl"):
            fname += ".pkl"
        path = os.path.join(self.results_dir, fname)
        try:
            with open(path, "wb") as fh:
                pickle.dump(opps, fh)
            self.logger.info("Saved %d opportunities to %s", len(opps), path)
        except Exception as e:
            self.logger.error(f"Failed to save opportunities to {path}: {e}")


###############################################################################
# 4. Convenience Functions ###################################################
###############################################################################


def sample_parameters() -> Dict[str, Union[int, float]]:
    """
    Generate a random parameter configuration using intelligent sampling.

    This is a wrapper around ParameterSpace.sample() for
    quick parameter generation in hyperparameter optimization or
    Monte Carlo experiments.

    Returns:
        Dictionary containing all parameters needed for arbitrage detection
    """
    return ParameterSpace().sample()


def main():
    """
    Main execution function demonstrating the FX arbitrage detection system.

    Workflow:
    1. Load historical FX data from CSV
    2. Configure detection parameters (hand-tuned baseline)
    3. Run arbitrage detection on data subset
    4. Save and analyze results

    """
    # ------------------------------------------------------------------
    # 1. Load Market Data ----------------------------------------------
    # ------------------------------------------------------------------
    # Path to FX dataset
    csv_path = "data/fx_data_march.csv"
    if os.path.exists(csv_path):
        LOGGER.info(f"Loading market data from {csv_path}...")
    else:
        LOGGER.error(f"Market data CSV file not found at {csv_path}.")

    try:
        market_data_manager = MarketDataManager(csv_path)
        df = market_data_manager.fetch()
    except FileNotFoundError:
        LOGGER.error(f"Market data file not found: {csv_path}. Exiting.")
        return
    except ValueError as exc_val:
        LOGGER.error(f"Error processing market data: {exc_val}. Exiting.")
        return
    except Exception as exc:
        LOGGER.error(
            f"Unexpected error loading market data: {exc}. Exiting.",
            exc_info=True,
        )
        return

    if df.empty and csv_path.endswith("fx_data_march_april_2025.csv"):
        LOGGER.warning("Market data is empty.")

    # ------------------------------------------------------------------
    # 2. Configure Detection Parameters -------------------------------
    # ------------------------------------------------------------------
    # \tuning files override the default parameters

    # Arbitrary baseline parameters
    default_params = {
        # QUBO formulation parameters
        "objective_scale": 50.0,  # Overall problem scaling
        "flow_penalty_mul": 0.3,  # Flow conservation strength (moderate)
        "cycle_penalty_mul": 0.05,  # Empty solution penalty (light)
        # Cycle detection constraints
        "min_cycle_length": 3,  # Minimum: triangular arbitrage
        "max_cycle_length": 5,  # Maximum: computational tractability
        # SQA sampling parameters
        "num_reads": 1000,  # Number of independent samples
        "num_sweeps": 500,  # Monte Carlo steps per sample
        "num_trotter_slices": 16,  # Quantum simulation accuracy
        # Annealing schedule (quantum → classical transition)
        "transverse_field_start": 5.0,  # Strong quantum fluctuations
        "transverse_field_end": 0.05,  # Weak quantum fluctuations
        "beta_start": 0.01,  # High temperature (exploration)
        "beta_end": 10.0,  # Low temperature (exploitation)
        # Reproducibility
        "seed": 42,  # Fixed seed for consistent results
    }

    current_params = default_params

    # Initialize experiment with configured params
    exp = ArbitrageExperiment(current_params, results_dir="SQA_results")

    # ------------------------------------------------------------------
    # 3. Run Arbitrage Detection ---------------------------------------
    # ------------------------------------------------------------------
    # Process subset of data for demonstration
    slice_size = min(50, len(df))
    if len(df) == 0:
        LOGGER.warning("DataFrame is empty. Smoke test will run on no data.")

    # Execute arbitrage detection on data subset
    opps = exp.run(df.head(slice_size))

    # Save results for later analysis
    exp.save(opps, fname="arbitrage_opportunities.pkl")

    # ------------------------------------------------------------------
    # 4. Analyze and Report Results ------------------------------------
    # ------------------------------------------------------------------
    if opps:
        # Calculate summary statistics
        profits = [o["profit"] for o in opps]
        LOGGER.info(
            "Found %d arbitrage opportunities. Mean profit = %.4f%%. Max profit = %.4f%%",
            len(opps),
            np.mean(profits) if profits else 0.0,
            np.max(profits) if profits else 0.0,
        )

        # Log sample opportunities for inspection
        for i, opp in enumerate(opps[: min(3, len(opps))]):
            LOGGER.info(
                f"Example Opportunity {i+1}: "
                f"Cycle={opp['cycle']}, "
                f"Profit={opp['profit']:.4f}%, "
                f"Timestamp={opp['timestamp']}"
            )
    else:
        LOGGER.warning(
            "No arbitrage cycles found with current parameters and data slice. "
            "Consider adjusting QUBO parameters or using larger dataset."
        )


if __name__ == "__main__":
    main()
