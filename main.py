# coding: utf-8
"""
Main script for detecting FX‑arbitrage cycles with an adaptive QUBO and
Simulated Quantum Annealing

Dependencies:
    – networkx, numpy, pandas, scipy, dimod, numba, joblib, optuna

Files needed to run this script:
    * sqa_implementation.py
    * a CSV file with FX mid‑rates indexed by timestamp (one column per pair)
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
# 0.  Logging helpers #########################################################
###############################################################################

LOGGER = logging.getLogger("main")
if not LOGGER.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

###############################################################################
# A. Constants  ###############################################################
###############################################################################

SENTINEL = 1_000.0  # numerical overflow guard for log weights

###############################################################################
# B. QUBO builder #############################################################
###############################################################################


class QUBOBuilder:
    """Adaptive‑scaling QUBO builder with coupled penalties and empty‑bias.
    Based on dimod's BinaryQuadraticModel.
    This class is used to build a QUBO for the SQAsampler.
    """

    REQUIRED = {"objective_scale", "flow_penalty_mul", "cycle_penalty_mul"}

    # ------------------------------------------------------------------
    def __init__(self, params: Dict[str, float]):
        missing = self.REQUIRED - params.keys()
        if missing:
            raise ValueError(f"Missing QUBO parameters: {missing}")
        self.params = params

    # ------------------------------------------------------------------
    def build_qubo(self, network: "ArbitrageNetwork") -> BinaryQuadraticModel:
        """Construct a BQM whose scale adapts to the data itself."""
        obj_scale = self.params["objective_scale"]
        P_mul = self.params["flow_penalty_mul"]
        C_mul = self.params["cycle_penalty_mul"]

        bqm = BinaryQuadraticModel({}, {}, 0.0, BINARY)

        # 1. finite‑weight edges only -----------------------------------
        edges: List[Tuple[str, str, Dict[str, Any]]] = [
            (u, v, d)
            for u, v, d in network.G.edges(data=True)
            if abs(d["weight"]) < SENTINEL
        ]
        if not edges:
            return bqm  # empty BQM

        # Ensure max_w is not zero if edges exist but all weights are zero
        weights = [abs(d["weight"]) for *_, d in edges]
        max_w = max(weights) if weights else 1.0
        if max_w == 0:  # Avoid division by zero if all valid weights are zero
            max_w = 1.0

        flow_P = P_mul * obj_scale
        cycle_pen = C_mul * obj_scale

        # 2. variables ---------------------------------------------------
        edge_var = {(u, v): f"x_{u}_{v}" for u, v, _ in edges}
        for (u, v), var in edge_var.items():
            # Ensure G[u][v] exists; it should if (u,v,_) in edges
            norm_w = (
                network.G[u][v]["weight"] / max_w
            )  # ∈[‑1,1] (approx, due to SENTINEL filtering)
            bqm.add_variable(var, norm_w * obj_scale - cycle_pen)
        bqm.offset += len(edge_var) * cycle_pen  # penalise empty set

        # 3. flow‑conservation penalty ----------------------------------
        in_edges: Dict[str, List[str]] = {n: [] for n in network.currencies}
        out_edges: Dict[str, List[str]] = {n: [] for n in network.currencies}
        for (u, v), var in edge_var.items():
            # Ensure u and v are in network.currencies; they should be by ArbitrageNetwork construction
            if u in out_edges:
                out_edges[u].append(var)
            if v in in_edges:
                in_edges[v].append(var)

        for node in network.currencies:
            ins, outs = in_edges[node], out_edges[node]

            for vi in ins + outs:  # linear terms
                bqm.add_linear(vi, flow_P)

            # same‑direction interactions
            for group in (ins, outs):
                for i, vi in enumerate(group):
                    for vj in group[i + 1 :]:
                        bqm.add_quadratic(vi, vj, 2 * flow_P)
            # cross in/out interaction
            for vi in ins:
                for vj in outs:
                    bqm.add_quadratic(vi, vj, -2 * flow_P)
        return bqm


###############################################################################
# C. Smart Parameter Space ####################################################
###############################################################################


class ParameterSpace:
    """Random sampler based on coupled‑penalty design."""

    def __init__(self, *, allow_loose: bool = True, objective_hi: float = 10_000.0):
        self.allow_loose = allow_loose
        self.objective_hi = objective_hi

    # ------------------------------------------------------------------
    def _log_uniform(self, low: float, high: float) -> float:
        return float(np.exp(np.random.uniform(math.log(low), math.log(high))))

    # ------------------------------------------------------------------
    def sample(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        params["objective_scale"] = self._log_uniform(1.0, self.objective_hi)

        if self.allow_loose and bool(np.random.randint(0, 2)):  # type: ignore
            params["flow_penalty_mul"] = self._log_uniform(0.001, 0.1)
        else:
            params["flow_penalty_mul"] = self._log_uniform(0.1, 2.0)

        params["cycle_penalty_mul"] = np.random.uniform(0.0, 0.5)  # type: ignore

        # annealing & misc ------------------------------------------------
        params.update(
            num_reads=int(np.random.choice([100, 500, 1000, 2000, 5000])),
            num_sweeps=int(np.random.choice([250, 500, 1000, 2000])),
            num_trotter_slices=int(np.random.choice([8, 16, 32, 64])),
            transverse_field_start=self._log_uniform(1.0, 10.0),
            transverse_field_end=self._log_uniform(0.005, 0.5),
            beta_start=self._log_uniform(0.001, 1.0),
            beta_end=self._log_uniform(1.0, 20.0),
            seed=int(np.random.randint(0, 2**32)),
            min_cycle_length=int(np.random.choice([3, 4])),
            max_cycle_length=int(np.random.choice([4, 5, 6, 7, 8])),
        )
        if params["max_cycle_length"] < params["min_cycle_length"]:
            params["max_cycle_length"] = params["min_cycle_length"]
        return params


###############################################################################
# 1.  Core data structures ####################################################
###############################################################################


class ArbitrageNetwork:
    """Directed graph with edge weights = –log(effective FX rate)."""

    def __init__(self) -> None:
        self.G: nx.DiGraph = nx.DiGraph()
        self.currencies: set[str] = set()

    # ------------------------------------------------------------------
    def add_rate(
        self,
        base: str,
        quote: str,
        rate: float,
        transaction_costs: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> None:
        if rate <= 0:
            raise ValueError(f"Invalid rate for {base}/{quote}: {rate}")

        tc = transaction_costs or {}
        cost_fwd = tc.get((base, quote), 0.0)
        cost_bwd = tc.get((quote, base), 0.0)
        if not (0 <= cost_fwd < 1) or not (0 <= cost_bwd < 1):
            raise ValueError("Transaction cost must be ∈[0,1).")

        eff_fwd = rate * (1 - cost_fwd)
        eff_bwd = (1 / rate) * (1 - cost_bwd)

        w_fwd = -math.log(eff_fwd) if eff_fwd > 1e-15 else SENTINEL  # Use SENTINEL
        w_bwd = -math.log(eff_bwd) if eff_bwd > 1e-15 else SENTINEL  # Use SENTINEL

        self.G.add_edge(base, quote, weight=w_fwd)
        self.G.add_edge(quote, base, weight=w_bwd)
        self.currencies.update({base, quote})

    # ------------------------------------------------------------------
    def cycle_profit_pct(self, cycle: List[str]) -> float:
        """Exact % gain for a closed cycle; +0.37 ⇒ +0.37%."""
        if not cycle:
            return 0.0
        prod = 1.0
        for u, v in zip(cycle, cycle[1:] + [cycle[0]]):
            if not self.G.has_edge(u, v) or "weight" not in self.G[u][v]:
                # This case should ideally not happen if cycle comes from graph edges
                # but good for robustness
                LOGGER.warning(
                    f"Edge {u}->{v} not found or no weight in profit calculation for cycle {cycle}"
                )
                return -float("inf")  # Invalid cycle
            prod *= math.exp(-self.G[u][v]["weight"])
        return (prod - 1.0) * 100.0


###############################################################################
# 2.  Arbitrage detector ######################################################
###############################################################################


a_REQUIRED = {
    "objective_scale",
    "flow_penalty_mul",
    "cycle_penalty_mul",
    "min_cycle_length",
    "max_cycle_length",
    "num_reads",
    "num_sweeps",
    "num_trotter_slices",
    "transverse_field_start",
    "transverse_field_end",
    "beta_start",
    "beta_end",
}


class ArbitrageDetector:
    """Builds the adaptive QUBO and calls the SQA sampler."""

    def __init__(
        self,
        params: Dict[str, Any],
        transaction_cost_bps: float = 0.0002,  # Default 0.02% transaction cost
    ) -> None:
        missing = a_REQUIRED - params.keys()
        if missing:
            raise ValueError(f"Missing detector parameters: {missing}")

        self.params = params
        self.transaction_cost = transaction_cost_bps

        self.network = ArbitrageNetwork()
        self.qubo_builder = QUBOBuilder(params)
        self.sampler = SimulatedQuantumAnnealingSampler()

        self.logger = self._setup_logger()
        self.current_timestamp = None

    # ------------------------------------------------------------------
    def _setup_logger(self) -> logging.Logger:
        lg = logging.getLogger("arbitrage_detector")
        if not lg.handlers:
            lg.setLevel(logging.INFO)
            log_file = "arbitrage_detector.log"
            try:
                fh = RotatingFileHandler(
                    log_file, maxBytes=5 * 1024 * 1024, backupCount=3
                )
                fh.setFormatter(
                    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
                )
                lg.addHandler(fh)
            except Exception as e:  # Handle potential permission issues, etc.
                # Fallback to basic console logging for the detector's logger
                # if file handler fails
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
                    )
                )
                lg.addHandler(console_handler)
                lg.error(
                    f"Failed to set up RotatingFileHandler for {log_file}: {e}. Using console output for this logger."
                )

        return lg

    # ------------------------------------------------------------------
    def add_market_data(self, rates: Dict[str, float], timestamp=None) -> None:
        """Overwrite the internal graph with the latest snapshot."""
        self.current_timestamp = timestamp
        self.network = ArbitrageNetwork()

        tc_dict: Dict[Tuple[str, str], float] = {}
        for pair_str in rates:
            if "/" not in pair_str:
                self.logger.warning(f"Invalid pair format: {pair_str}. Skipping.")
                continue
            base, quote = pair_str.split("/")
            tc_dict[(base, quote)] = self.transaction_cost
            tc_dict[(quote, base)] = self.transaction_cost

        for pair_str, rate in rates.items():
            if "/" not in pair_str:
                # Already warned above, but good to be safe
                continue
            base, quote = pair_str.split("/")
            try:
                self.network.add_rate(base, quote, rate, tc_dict)
            except ValueError as exc:
                self.logger.error("Add rate failed for %s: %s", pair_str, exc)

    # ------------------------------------------------------------------
    def find_arbitrage(self, allow_empty: bool = False) -> List[Dict[str, Any]]:
        if not self.network.G.edges:
            self.logger.info("No edges in the network. Skipping arbitrage detection.")
            return []

        bqm = self.qubo_builder.build_qubo(self.network)
        if not bqm.variables:
            self.logger.info(
                "QUBO has no variables (e.g. all edge weights were sentinel). Skipping sampling."
            )
            return []

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
        min_E = (
            float(np.min(response.record["energy"]))
            if len(response) > 0
            and response.record is not None
            and "energy" in response.record.dtype.names
            and len(response.record["energy"]) > 0
            else float("inf")
        )
        self.logger.info("Best raw BQM energy: %.6f", min_E)

        # ----------------------------------------------------------------
        # Extract cycles --------------------------------------------------
        # ----------------------------------------------------------------
        var_names = list(bqm.variables)
        try:
            edges = [tuple(name[2:].split("_", 1)) for name in var_names]
            if any(len(e) != 2 for e in edges):  # Basic validation
                self.logger.error(
                    "Malformed variable names found. Cannot extract edges."
                )
                return []
        except Exception as e:
            self.logger.error(f"Error parsing variable names to edges: {e}")
            return []

        var_to_idx = {v: i for i, v in enumerate(var_names)}

        opps: List[Dict[str, Any]] = []
        seen: set[Tuple[str, ...]] = set()

        if (
            response.record is None or len(response.record) == 0
        ):  # Check if response is empty
            self.logger.info("SQA Sampler returned no samples.")
            return []

        for sample_vec, energy, occ in zip(
            response.record["sample"],
            response.record["energy"],
            response.record["num_occurrences"],
        ):
            chosen_edges_from_sample: List[Tuple[str, str]] = []
            for i, var_name in enumerate(var_names):
                if sample_vec[var_to_idx[var_name]] == 1:
                    chosen_edges_from_sample.append(edges[i])

            if not chosen_edges_from_sample and not allow_empty:
                continue

            subg = nx.DiGraph()
            subg.add_edges_from(chosen_edges_from_sample)

            for cycle in nx.simple_cycles(subg):
                L = len(cycle)
                min_len = self.params.get("min_cycle_length", 3)
                max_len = self.params.get("max_cycle_length", 5)

                if not (min_len <= L <= max_len):
                    continue

                # Canonical form to avoid duplicates from different starting nodes
                if L > 0:
                    canon = min(tuple(cycle[i:] + cycle[:i]) for i in range(L))
                else:  # should not happen as we check L > 0 above
                    continue

                if canon in seen:
                    continue
                seen.add(canon)

                profit = self.network.cycle_profit_pct(cycle)
                if profit <= 1e-5:  # Profit threshold to filter out negligible cycles
                    continue
                opps.append(
                    {
                        "cycle": cycle,
                        "profit": round(profit, 6),
                        "energy": float(energy),
                        "timestamp": self.current_timestamp,
                        "num_occurrences": int(occ),
                    }
                )

        return sorted(opps, key=lambda x: x["profit"], reverse=True)


###############################################################################
# 3.  Experiments & helpers ###################################################
###############################################################################


class MarketDataManager:
    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        self.path = csv_path
        self.logger = logging.getLogger("market_data_manager")

    def fetch(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.path, index_col=0, parse_dates=True)
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                    if not isinstance(df.index, pd.DatetimeIndex):
                        raise ValueError(
                            "CSV index could not be converted to DatetimeIndex."
                        )
                except Exception as e:
                    self.logger.error(
                        f"Failed to parse or convert index to DatetimeIndex: {e}"
                    )
                    raise ValueError("CSV index must be datetime.") from e

            # Check for NaN in index which can happen if parse_dates fails silently on some entries
            if df.index.hasnans:
                self.logger.warning(
                    f"CSV index contains NaT values after parsing. Rows with NaT index will be dropped."
                )
                df = df[~df.index.isna()]

            return df
        except Exception as e:
            self.logger.error(f"Error reading or processing CSV file {self.path}: {e}")
            raise


class ArbitrageExperiment:
    def __init__(self, params: Dict[str, Any], results_dir: str = "results"):
        self.params = params
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.detector = ArbitrageDetector(params)
        self.logger = logging.getLogger("arbitrage_experiment")

    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        all_opps: List[Dict[str, Any]] = []
        if df.empty:
            self.logger.info("Input DataFrame is empty. No experiment run.")
            return []

        for ts, row in df.iterrows():
            self.logger.debug(f"Processing data for timestamp: {ts}")
            # Drop NaNs from the row to get valid rates
            rates = row.dropna().to_dict()
            if not rates:
                self.logger.debug(f"No valid rates for timestamp: {ts}. Skipping.")
                continue

            self.detector.add_market_data(rates, ts)
            try:
                # Potentially expose allow_empty if needed for experiments
                opps_at_ts = self.detector.find_arbitrage(allow_empty=False)
                all_opps.extend(opps_at_ts)
            except Exception as e:
                self.logger.error(
                    f"Error during arbitrage detection for timestamp {ts}: {e}",
                    exc_info=True,
                )
        return all_opps

    # ------------------------------------------------------------------
    def save(self, opps: List[Dict[str, Any]], fname: str = "opps.pkl") -> None:
        if not fname.endswith(".pkl"):
            fname += ".pkl"
        path = os.path.join(self.results_dir, fname)
        try:
            with open(path, "wb") as fh:
                pickle.dump(opps, fh)
            self.logger.info("Saved %d opportunities → %s", len(opps), path)
        except Exception as e:
            self.logger.error(f"Failed to save opportunities to {path}: {e}")


###############################################################################
# 4.  Convenience wrappers ####################################################
###############################################################################


def sample_parameters() -> Dict[str, Union[int, float]]:
    return ParameterSpace().sample()


def main():
    # ------------------------------------------------------------------
    # 1.  Load market data ---------------------------------------------
    # ------------------------------------------------------------------
    # Dataset from Refinitive could not be uploaded to GitHub due to licensing issues.
    csv_path = "data/fx_data_march_april_2025.csv"
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
    except Exception as exc:  # Catch any other loading error
        LOGGER.error(
            f"An unexpected error occurred while loading market data: {exc}. Exiting.",
            exc_info=True,
        )
        return

    if df.empty and csv_path.endswith("fx_data_march_april_2025.csv"):
        LOGGER.warning("Market data is empty.")

    # ------------------------------------------------------------------
    # 2.  Use a *hand‑picked* starting parameter set --------------------
    # ------------------------------------------------------------------
    default_params = {
        "objective_scale": 50.0,
        "flow_penalty_mul": 0.3,
        "cycle_penalty_mul": 0.05,
        "min_cycle_length": 3,
        "max_cycle_length": 5,
        "num_reads": 1000,
        "num_sweeps": 500,
        "num_trotter_slices": 16,
        "transverse_field_start": 5.0,
        "transverse_field_end": 0.05,
        "beta_start": 0.01,
        "beta_end": 10.0,
        "seed": 42,
    }

    current_params = default_params

    exp = ArbitrageExperiment(current_params, results_dir="SQA_results")

    slice_size = min(50, len(df))
    if len(df) == 0:
        LOGGER.warning("DataFrame is empty. Smoke test will run on no data.")

    opps = exp.run(df.head(slice_size))
    exp.save(opps, fname="arbitrage_opportunities.pkl")

    if opps:
        profits = [o["profit"] for o in opps]
        LOGGER.info(
            "Found %d arbitrage opportunities. Mean profit = %.4f%%. Max profit = %.4f%%",
            len(opps),
            np.mean(profits) if profits else 0.0,
            np.max(profits) if profits else 0.0,
        )
        # Log some example opportunities
        for i, opp in enumerate(opps[: min(3, len(opps))]):  # Log first 3
            LOGGER.info(
                f"Example Opp {i+1}: Cycle={opp['cycle']}, Profit={opp['profit']:.4f}%, Timestamp={opp['timestamp']}"
            )
    else:
        LOGGER.warning(
            "No arbitrage cycles found with the current parameters and data slice."
        )


if __name__ == "__main__":
    main()
