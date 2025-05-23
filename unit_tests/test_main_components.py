import sys
import types
import os
import math
import logging
import pickle
from unittest.mock import patch, MagicMock, mock_open

import pytest
import numpy as np
import pandas as pd
import networkx as nx
import dimod
from dimod import BinaryQuadraticModel, BINARY, SPIN

# Mock joblib
dummy_joblib = types.ModuleType("joblib")


class MockDelayedFunc:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return (self.func, args, kwargs)  # Returns tuple


dummy_joblib.delayed = MockDelayedFunc


class MockParallel:
    def __init__(self, n_jobs=None, verbose=0, **kwargs):
        pass

    def __call__(self, tasks_generator):  # tasks_generator yields (func, args, kwargs)
        results = []
        for task_tuple in tasks_generator:
            func, args_tuple, kwargs_dict = task_tuple
            results.append(func(*args_tuple, **kwargs_dict))
        return results


dummy_joblib.Parallel = MockParallel
sys.modules["joblib"] = dummy_joblib


dummy_numba = types.ModuleType("numba")
dummy_numba.njit = lambda *args, **kwargs: (lambda f: f)
sys.modules["numba"] = dummy_numba

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import (
    QUBOBuilder,
    ParameterSpace,
    ArbitrageNetwork,
    ArbitrageDetector,
    MarketDataManager,
    ArbitrageExperiment,
    SENTINEL,
    a_REQUIRED,
)


# --- Fake Sampler/Response ---
class FakeResponse:
    """Mocks a dimod.SampleSet-like object for testing sampler interactions."""

    def __init__(self, samples_data, variables, energies, num_occurrences, info=None):
        """
        Initializes FakeResponse.

        Args:
            samples_data (list[dict]): List of sample dictionaries.
            variables (list): List of variable names in order.
            energies (list): List of energies corresponding to samples.
            num_occurrences (list): List of occurrences for each sample.
            info (dict, optional): Additional information for the response.
        """
        self.variables = list(variables)  # Store as a plain list of variable names
        self.vartype = dimod.BINARY
        self.info = info if info is not None else {}
        num_vars = len(self.variables)
        num_samples = len(samples_data)
        record_dtype_fields = [
            ("sample", np.int8, (num_vars,) if num_vars > 0 else (0,)),
            ("energy", np.float64),
            ("num_occurrences", np.int32),
        ]
        self.record = np.empty(num_samples, dtype=record_dtype_fields)
        if num_samples > 0:
            samples_matrix = np.array(
                [
                    [sample_dict.get(var, 0) for var in self.variables]
                    for sample_dict in samples_data
                ],
                dtype=np.int8,
            )
            self.record["sample"] = samples_matrix
            self.record["energy"] = np.asarray(energies, dtype=np.float64)
            self.record["num_occurrences"] = np.asarray(num_occurrences, dtype=np.int32)

    def __len__(self):
        """Returns the number of samples in the fake response."""
        return len(self.record)

    @property
    def first(self):
        """Returns a view of the first sample, mimicking dimod.SampleSet.first."""
        if not self:
            raise ValueError("SampleSet is empty")

        class MockSampleView:  # Simple class to mimic dimod.SampleView attribute access
            def __init__(self, s, e, n):
                self.sample = s
                self.energy = e
                self.num_occurrences = n

        vals = self.record["sample"][0]
        s_dict = {self.variables[i]: vals[i] for i in range(len(self.variables))}
        return MockSampleView(
            s_dict, self.record["energy"][0], self.record["num_occurrences"][0]
        )

    def data(self, fields=None, sorted_by="energy", **kwargs):
        """Yields sample data, mimicking dimod.SampleSet.data()."""

        class MockSampleViewData:  # Simple class for yielded data
            def __init__(self, s, e, n):
                self.sample = s
                self.energy = e
                self.num_occurrences = n

        for i in range(len(self.record)):
            vals = self.record["sample"][i]
            s_dict = {self.variables[j]: vals[j] for j in range(len(self.variables))}
            yield MockSampleViewData(
                s_dict, self.record["energy"][i], self.record["num_occurrences"][i]
            )


class FakeSampler:
    """Mocks a dimod Sampler for testing."""

    def __init__(self, r: FakeResponse):
        """
        Initializes FakeSampler.

        Args:
            r (FakeResponse): The FakeResponse object to be returned by sample().
        """
        self.response_to_return = r

    def sample(self, b, **kw):
        """Mocks the sample method, returning the pre-configured FakeResponse."""
        return self.response_to_return


# --- Fixtures for Tests ---
@pytest.fixture
def base_qubo_params():
    """Provides a basic set of parameters for QUBOBuilder tests."""
    return {"objective_scale": 1.0, "flow_penalty_mul": 1.0, "cycle_penalty_mul": 0.1}


@pytest.fixture
def full_detector_params():
    """Provides a set of parameters for ArbitrageDetector tests."""
    p = {
        "objective_scale": 50.0,
        "flow_penalty_mul": 0.3,
        "cycle_penalty_mul": 0.05,
        "min_cycle_length": 3,
        "max_cycle_length": 5,
        "num_reads": 10,
        "num_sweeps": 50,
        "num_trotter_slices": 8,
        "transverse_field_start": 5.0,
        "transverse_field_end": 0.05,
        "beta_start": 0.01,
        "beta_end": 10.0,
        "seed": 42,
    }
    for (
        k
    ) in (
        a_REQUIRED
    ):  # Ensure all keys required by ArbitrageDetector.__init__ are present
        if k not in p:
            print(f"Warning: {k} not in full_detector_params, setting to default.")
            p[k] = 1.0  # Add a default if missing from the explicit list
    return p


class TestQUBOBuilder:
    """Tests for the QUBOBuilder class."""

    def test_init_valid_params(self, base_qubo_params):
        """
        Purpose: Test QUBOBuilder initialization with valid parameters.
        Input: A dictionary of valid parameters.
        Expected: QUBOBuilder instance is created successfully and params are stored.
        """
        b = QUBOBuilder(base_qubo_params)
        assert b.params == base_qubo_params

    def test_init_missing_params(self):
        """
        Purpose: Test QUBOBuilder initialization with missing required parameters.
        Input: A dictionary missing some required parameters.
        Expected: Raises ValueError.
        """
        with pytest.raises(ValueError):
            QUBOBuilder({"objective_scale": 1.0})

    def test_build_qubo_empty_network(self, base_qubo_params):
        """
        Purpose: Test BQM construction for an empty arbitrage network.
        Input: Empty ArbitrageNetwork.
        Expected: Empty BQM (no linear/quadratic terms, zero offset).
        """
        b = QUBOBuilder(base_qubo_params)
        net = ArbitrageNetwork()
        bqm = b.build_qubo(net)
        assert not bqm.linear and not bqm.quadratic and bqm.offset == 0.0

    def test_build_qubo_one_edge_pair(self):
        """
        Purpose: Test BQM construction for a network with a single currency pair.
                 Verifies objective, cycle penalty, and flow penalty contributions to linear terms and offset.
        Input: Network with A/B rate, specific QUBO params.
        Expected: Correct linear biases and offset in the BQM based on calculations.
        """
        p = {"objective_scale": 10.0, "flow_penalty_mul": 1.0, "cycle_penalty_mul": 0.1}
        b = QUBOBuilder(p)
        net = ArbitrageNetwork()
        net.add_rate("A", "B", 2.0)
        bqm = b.build_qubo(net)
        assert "x_A_B" in bqm.variables and "x_B_A" in bqm.variables
        assert pytest.approx(bqm.linear["x_A_B"]) == 9.0
        assert pytest.approx(bqm.linear["x_B_A"]) == 29.0
        assert pytest.approx(bqm.offset) == 2.0

    def test_build_qubo_flow_penalty(self):
        """
        Purpose: Test BQM construction focusing on flow conservation penalty terms.
                 Verifies linear and quadratic coefficients for a simple A-B-C network.
        Input: Network with A/B and B/C rates (weights set to 0 for simplicity), specific QUBO params.
        Expected: Correct linear and quadratic flow penalty coefficients.
        """
        p = {"objective_scale": 1.0, "flow_penalty_mul": 2.0, "cycle_penalty_mul": 0.0}
        b = QUBOBuilder(p)
        net = ArbitrageNetwork()
        net.add_rate("A", "B", 1.0)
        net.add_rate("B", "C", 1.0)
        bqm = b.build_qubo(net)
        assert pytest.approx(bqm.linear["x_A_B"]) == 4.0
        assert pytest.approx(bqm.linear["x_B_A"]) == 4.0
        assert pytest.approx(bqm.linear["x_B_C"]) == 4.0
        assert pytest.approx(bqm.linear["x_C_B"]) == 4.0
        assert pytest.approx(bqm.quadratic[("x_A_B", "x_C_B")]) == 4.0
        assert pytest.approx(bqm.quadratic[("x_B_A", "x_B_C")]) == 4.0
        assert pytest.approx(bqm.quadratic[("x_A_B", "x_B_A")]) == -8.0
        assert pytest.approx(bqm.quadratic[("x_A_B", "x_B_C")]) == -4.0
        assert pytest.approx(bqm.quadratic[("x_C_B", "x_B_A")]) == -4.0
        assert pytest.approx(bqm.quadratic[("x_B_C", "x_C_B")]) == -8.0

    def test_build_qubo_with_sentinel_weight(self, base_qubo_params):
        """
        Purpose: Test that edges with SENTINEL weights are excluded from the BQM.
        Input: Network where one direction of an edge has a SENTINEL weight.
        Expected: The variable corresponding to the SENTINEL edge is not in the BQM.
        """
        b = QUBOBuilder(base_qubo_params)
        net = ArbitrageNetwork()
        net.add_rate("A", "B", 1e-300)
        assert net.G["A"]["B"]["weight"] == SENTINEL
        bqm = b.build_qubo(net)
        assert "x_A_B" not in bqm.variables and "x_B_A" in bqm.variables


class TestParameterSpace:
    """Tests for the ParameterSpace class, focusing on parameter generation."""

    def test_sample_structure_and_types(self):
        """
        Purpose: Verify the structure and basic types of parameters generated by sample().
        Input: None (uses default ParameterSpace).
        Expected: Generated params is a dict, contains all required keys, and critical
                  parameters have correct types (float, int). Max cycle length >= min.
        """
        ps = ParameterSpace()
        p = ps.sample()
        assert isinstance(p, dict)
        assert a_REQUIRED.issubset(p.keys())
        assert isinstance(p["objective_scale"], float) and isinstance(
            p["num_reads"], int
        )
        assert p["max_cycle_length"] >= p["min_cycle_length"]

    def test_log_uniform_range(self):
        """
        Purpose: Test the _log_uniform helper method to ensure generated values are within bounds.
        Input: Defined low and high bounds.
        Expected: All sampled values fall within [low, high].
        """
        ps = ParameterSpace()
        lo, hi = 1.0, 100.0
        samples = [ps._log_uniform(lo, hi) for _ in range(100)]
        assert all(lo <= s <= hi for s in samples)


class TestArbitrageNetwork:
    """Tests for the ArbitrageNetwork class, handling graph representation and profit calculation."""

    def test_init(self):
        """
        Purpose: Test ArbitrageNetwork initialization.
        Input: None.
        Expected: Graph G is a DiGraph, and currencies set is empty.
        """
        net = ArbitrageNetwork()
        assert isinstance(net.G, nx.DiGraph) and not net.currencies

    def test_add_rate_valid(self):
        """
        Purpose: Test adding a valid FX rate without transaction costs.
        Input: Base currency, quote currency, and a positive rate.
        Expected: Edges for both directions (base->quote, quote->base) are added to the graph
                  with weights equal to -log(effective_rate). Currencies are registered.
        """
        net = ArbitrageNetwork()
        net.add_rate("U", "E", 0.9)
        assert net.G.has_edge("U", "E") and net.G.has_edge("E", "U")
        assert "U" in net.currencies and "E" in net.currencies
        assert pytest.approx(net.G["U"]["E"]["weight"]) == -math.log(0.9)
        assert pytest.approx(net.G["E"]["U"]["weight"]) == -math.log(1 / 0.9)

    def test_add_rate_with_transaction_costs(self):
        """
        Purpose: Test adding an FX rate with specified transaction costs.
        Input: Base, quote, rate, and a transaction cost dictionary.
        Expected: Edge weights reflect the rate adjusted by transaction costs.
        """
        net = ArbitrageNetwork()
        tc = {("U", "E"): 0.001, ("E", "U"): 0.001}
        net.add_rate("U", "E", 0.9, tc)
        fwd = 0.9 * 0.999
        bwd = (1 / 0.9) * 0.999
        assert pytest.approx(net.G["U"]["E"]["weight"]) == -math.log(fwd)
        assert pytest.approx(net.G["E"]["U"]["weight"]) == -math.log(bwd)

    def test_add_rate_invalid_rate_or_tc(self):
        """
        Purpose: Test adding rates with invalid inputs (zero/negative rate, invalid TC).
        Input: Various invalid rate or transaction cost values.
        Expected: ValueError is raised for each invalid case.
        """
        net = ArbitrageNetwork()
        with pytest.raises(ValueError):
            net.add_rate("U", "E", 0)
        with pytest.raises(ValueError):
            net.add_rate("U", "E", -1.0)
        with pytest.raises(ValueError):
            net.add_rate("U", "E", 0.9, {("U", "E"): 1.0})
        with pytest.raises(ValueError):
            net.add_rate("U", "E", 0.9, {("U", "E"): -0.1})

    def test_add_rate_near_zero_effective_rate(self):
        """
        Purpose: Test adding a rate that results in a near-zero (or effectively zero)
                 rate after considering precision, leading to a SENTINEL weight.
        Input: A very small rate (e.g., 1e-300).
        Expected: The forward edge weight becomes SENTINEL, while the backward edge
                  (with a very large rate) should have a finite weight.
        """
        net = ArbitrageNetwork()
        net.add_rate("U", "E", 1e-300)
        assert (
            net.G["U"]["E"]["weight"] == SENTINEL
            and net.G["E"]["U"]["weight"] != SENTINEL
        )

    def test_cycle_profit_pct(self):
        """
        Purpose: Test the calculation of profit percentage for a given cycle.
        Input: Networks with pre-defined cycles (one non-profitable, one profitable).
        Expected: Correct profit percentages are calculated.
        """
        net = ArbitrageNetwork()
        net.add_rate("U", "E", 0.9)
        net.add_rate("E", "J", 130)
        net.add_rate("J", "U", 0.0075)
        assert (
            pytest.approx(net.cycle_profit_pct(["U", "E", "J"])) == -12.25
        )  # (0.9 * 130 * 0.0075 - 1) * 100
        net_p = ArbitrageNetwork()
        net_p.add_rate("A", "B", 2)
        net_p.add_rate("B", "C", 2)
        net_p.add_rate("C", "A", 0.3)
        assert (
            pytest.approx(net_p.cycle_profit_pct(["A", "B", "C"])) == 20.0
        )  # (2 * 2 * 0.3 - 1) * 100

    def test_cycle_profit_empty_or_invalid_edge(self):
        """
        Purpose: Test profit calculation for an empty cycle or a cycle with missing edges.
        Input: An empty list (for empty cycle) and a list representing a cycle where
               an edge does not exist in the network.
        Expected: 0.0 for an empty cycle, -infinity for a cycle with a missing edge.
        """
        net = ArbitrageNetwork()
        assert net.cycle_profit_pct([]) == 0.0
        net.add_rate("A", "B", 1)
        assert net.cycle_profit_pct(["A", "B", "C"]) == -float("inf")


class TestArbitrageDetector:
    """Tests for the ArbitrageDetector class, covering initialization, data handling, and arbitrage finding."""

    def test_init_valid_params(self, full_detector_params):
        """
        Purpose: Test ArbitrageDetector initialization with a valid and complete set of parameters.
        Input: `full_detector_params` fixture.
        Expected: Instance created, params stored, and internal QUBOBuilder initialized with all passed params.
        """
        det = ArbitrageDetector(full_detector_params, 0.001)
        assert det.params == full_detector_params
        assert det.qubo_builder.params == full_detector_params

    def test_init_missing_params(self, full_detector_params):
        """
        Purpose: Test ArbitrageDetector initialization with missing required parameters.
        Input: `full_detector_params` with one required key removed.
        Expected: Raises ValueError.
        """
        p_inc = {k: v for k, v in full_detector_params.items() if k != "num_reads"}
        with pytest.raises(ValueError):
            ArbitrageDetector(p_inc)

    def test_add_market_data(self, full_detector_params):
        """
        Purpose: Test adding market data to the detector.
        Input: A dictionary of rates and a timestamp.
        Expected: Detector's internal network is updated with new rates (including transaction costs),
                  and current_timestamp is set.
        """
        det = ArbitrageDetector(full_detector_params, 0.001)
        ts = pd.Timestamp("20230101")
        det.add_market_data({"U/E": 0.9, "E/J": 130}, ts)
        assert det.current_timestamp == ts and det.network.G.has_edge("U", "E")
        assert pytest.approx(det.network.G["U"]["E"]["weight"]) == -math.log(
            0.9 * 0.999
        )

    def test_add_market_data_invalid_pair(self, full_detector_params, caplog):
        """
        Purpose: Test adding market data with an invalid pair string format.
        Input: Rates dictionary with a malformed pair string (e.g., missing '/').
        Expected: A warning is logged, and no edges are added for the invalid pair.
        """
        det = ArbitrageDetector(full_detector_params)
        with caplog.at_level(logging.WARNING, logger="arbitrage_detector"):
            det.add_market_data({"UE": 0.9})
        assert "Invalid pair format: UE" in caplog.text and not det.network.G.edges

    def test_find_arbitrage_no_edges(self, full_detector_params, caplog):
        """
        Purpose: Test arbitrage finding when the network has no edges.
        Input: Detector initialized but no market data added.
        Expected: Returns an empty list of opportunities, and an info message is logged.
        """
        det = ArbitrageDetector(full_detector_params)
        with caplog.at_level(logging.INFO, logger="arbitrage_detector"):
            opps = det.find_arbitrage()
        assert not opps and "No edges in the network" in caplog.text

    def test_find_arbitrage_empty_bqm(self, full_detector_params, caplog):
        """
        Purpose: Test arbitrage finding when the QUBO constructed has no variables.
                 This can happen if all edge weights in the network are SENTINEL.
        Input: Detector with market data leading to an empty BQM (mocked).
        Expected: Returns an empty list, logs "QUBO has no variables".
        """
        det = ArbitrageDetector(full_detector_params)
        empty_bqm = BinaryQuadraticModel({}, {}, 0.0, BINARY)
        assert not empty_bqm.variables
        det.qubo_builder.build_qubo = MagicMock(return_value=empty_bqm)
        det.add_market_data({"A/B": 1.0})  # Add data to trigger BQM build attempt
        with caplog.at_level(logging.INFO, logger="arbitrage_detector"):
            opps = det.find_arbitrage()
        assert not opps and "QUBO has no variables" in caplog.text
        det.qubo_builder.build_qubo.assert_called_once()

    def test_find_arbitrage_successful_cycle(self, full_detector_params):
        """
        Purpose: Test finding a known profitable arbitrage cycle.
        Input: Market data forming a 3-cycle with 20% profit. Sampler is mocked to return this cycle.
        Expected: One opportunity found with the correct cycle and profit.
        """
        p3c = {**full_detector_params, "min_cycle_length": 3, "max_cycle_length": 3}
        det = ArbitrageDetector(p3c, 0.0)
        det.add_market_data({"A/B": 2, "B/C": 2, "C/A": 0.3}, "t1")
        bqm = det.qubo_builder.build_qubo(det.network)
        v_list = list(bqm.variables)
        s_dict = {v: 0 for v in v_list}
        for edge_var in ["x_A_B", "x_B_C", "x_C_A"]:
            if edge_var in s_dict:
                s_dict[edge_var] = 1
        resp = FakeResponse([s_dict], v_list, [bqm.energy(s_dict)], [1])
        det.sampler = FakeSampler(resp)
        opps = det.find_arbitrage()
        assert len(opps) == 1 and tuple(opps[0]["cycle"]) in {
            ("A", "B", "C"),
            ("B", "C", "A"),
            ("C", "A", "B"),
        }
        assert pytest.approx(opps[0]["profit"]) == 20.0

    def test_find_arbitrage_filter_by_length_and_profit(self, full_detector_params):
        """
        Purpose: Test cycle filtering based on length and profit constraints.
        Input:
            1. A 3-cycle with detector configured for min/max length 4 (should be filtered).
            2. A 3-cycle with negative profit (should be filtered).
        Expected: No opportunities found in both cases.
        """
        # Test length filter
        p4 = {**full_detector_params, "min_cycle_length": 4, "max_cycle_length": 4}
        det_l = ArbitrageDetector(p4, 0.0)
        det_l.add_market_data({"A/B": 2, "B/C": 2, "C/A": 0.3})  # len 3
        bq_l = det_l.qubo_builder.build_qubo(det_l.network)
        v_l = list(bq_l.variables)
        s_l = {v: 0 for v in v_l}
        s_l.update({"x_A_B": 1, "x_B_C": 1, "x_C_A": 1})
        s_l = {v: s_l.get(v, 0) for v in v_l}
        det_l.sampler = FakeSampler(FakeResponse([s_l], v_l, [0.0], [1]))
        assert not det_l.find_arbitrage()

        # Test profit filter
        det_p = ArbitrageDetector(full_detector_params, 0.0)
        det_p.add_market_data({"A/B": 1, "B/C": 1, "C/A": 0.9})  # neg profit
        bq_p = det_p.qubo_builder.build_qubo(det_p.network)
        v_p = list(bq_p.variables)
        s_p = {v: 0 for v in v_p}
        s_p.update({"x_A_B": 1, "x_B_C": 1, "x_C_A": 1})
        s_p = {v: s_p.get(v, 0) for v in v_p}
        det_p.sampler = FakeSampler(FakeResponse([s_p], v_p, [0.0], [1]))
        assert not det_p.find_arbitrage()

    def test_find_arbitrage_sampler_no_samples(self, full_detector_params, caplog):
        """
        Purpose: Test behavior when the sampler returns an empty response (no samples).
        Input: Sampler mocked to return an empty FakeResponse.
        Expected: No opportunities found, and an info message is logged.
        """
        det = ArbitrageDetector(full_detector_params)
        det.add_market_data({"A/B": 2.0})
        bqm = det.qubo_builder.build_qubo(det.network)
        v_list = list(bqm.variables) if bqm.variables else ["v"]
        det.sampler = FakeSampler(FakeResponse([], v_list, [], []))
        with caplog.at_level(logging.INFO, logger="arbitrage_detector"):
            opps = det.find_arbitrage()
        assert not opps and "SQA Sampler returned no samples." in caplog.text


class TestMarketDataManager:
    """Tests for MarketDataManager, focusing on CSV loading and parsing."""

    @patch("os.path.exists", return_value=False)
    def test_init_file_not_found(self, mock_ex):
        """
        Purpose: Test MarketDataManager initialization when the CSV file does not exist.
        Input: Path to a non-existent CSV file.
        Expected: FileNotFoundError is raised.
        """
        with pytest.raises(FileNotFoundError):
            MarketDataManager("d.csv")

    @patch("pandas.read_csv")  # Innermost decorator's mock is first arg
    @patch(
        "os.path.exists", return_value=True
    )  # Outermost decorator's mock is second arg
    def test_fetch_valid_csv(self, mock_os_exists, mock_pandas_read_csv):
        """
        Purpose: Test fetching data from a valid CSV file.
        Input: Mocks for os.path.exists (True) and pandas.read_csv (returns a valid DataFrame).
        Expected: DataFrame is returned as mocked, and pandas.read_csv is called correctly.
        """
        mock_df = pd.DataFrame({"A/B": [1.0]}, index=pd.to_datetime(["20230101"]))
        mock_pandas_read_csv.return_value = mock_df
        df = MarketDataManager("d.csv").fetch()
        pd.testing.assert_frame_equal(df, mock_df)
        mock_pandas_read_csv.assert_called_once_with(
            "d.csv", index_col=0, parse_dates=True
        )
        mock_os_exists.assert_called_once_with(
            "d.csv"
        )  # Verify os.path.exists was checked

    @patch("pandas.read_csv")
    @patch("os.path.exists", return_value=True)
    def test_fetch_bad_index(self, mock_os_exists, mock_pandas_read_csv):
        """
        Purpose: Test fetching data from a CSV where the index cannot be converted to DatetimeIndex.
        Input: pandas.read_csv mocked to return a DataFrame with a non-datetime-convertible index.
        Expected: ValueError is raised.
        """
        mock_df = pd.DataFrame({"A/B": [1.0]}, index=["bad_date_str"])
        # Ensure the index is actually not a DatetimeIndex after initial read
        mock_df.index = pd.Index(["bad_date_str"])
        mock_pandas_read_csv.return_value = mock_df
        # If pd.to_datetime is called internally and fails, that's what we're testing
        with pytest.raises(ValueError, match="CSV index must be datetime"):
            MarketDataManager("d.csv").fetch()

    @patch("pandas.read_csv")
    @patch("os.path.exists", return_value=True)
    def test_fetch_index_with_nat(self, mock_os_exists, mock_pandas_read_csv, caplog):
        """
        Purpose: Test fetching data from a CSV where some index entries parse to NaT (Not a Time).
        Input: pandas.read_csv mocked to return a DataFrame with NaT values in its DatetimeIndex.
        Expected: Rows with NaT in the index are dropped, and a warning is logged.
        """
        idx = pd.to_datetime(
            ["20230101", "bad_date_str_causing_NaT", "20230103"], errors="coerce"
        )  # Creates NaT
        mock_df_with_nat = pd.DataFrame({"A/B": [1.0, 2.0, 3.0]}, index=idx)
        mock_pandas_read_csv.return_value = (
            mock_df_with_nat.copy()
        )  # Use copy as DataFrames are mutable
        with caplog.at_level(logging.WARNING, logger="market_data_manager"):
            df = MarketDataManager("d.csv").fetch()
        assert "CSV index contains NaT values" in caplog.text
        expected_df = mock_df_with_nat[
            ~mock_df_with_nat.index.isna()
        ]  # Filter out NaT rows
        pd.testing.assert_frame_equal(df, expected_df)


class TestArbitrageExperiment:
    """Tests for ArbitrageExperiment, covering experiment runs and result saving."""

    @patch(
        "main.ArbitrageDetector"
    )  # Patch where ArbitrageDetector is imported in main.py
    def test_init(self, MockAD, full_detector_params):
        """
        Purpose: Test ArbitrageExperiment initialization.
        Input: Experiment parameters and a results directory.
        Expected: ArbitrageDetector is initialized with params, results_dir is set, and directory is created.
        """
        r_dir = "test_experiment_results_dir"
        if os.path.exists(r_dir):
            os.rmdir(r_dir)  # Clean if leftover from previous failed run
        exp = ArbitrageExperiment(full_detector_params, results_dir=r_dir)
        MockAD.assert_called_once_with(full_detector_params)
        assert exp.results_dir == r_dir and os.path.exists(r_dir)
        os.rmdir(r_dir)  # Cleanup

    @patch("main.ArbitrageDetector")
    def test_run_empty_df(self, MockAD, full_detector_params, caplog):
        """
        Purpose: Test running an experiment with an empty DataFrame.
        Input: Empty pandas DataFrame.
        Expected: No opportunities found, info message logged, detector methods not called.
        """
        r_dir = "test_res_run_empty_df"
        exp = ArbitrageExperiment(full_detector_params, results_dir=r_dir)
        with caplog.at_level(logging.INFO, logger="arbitrage_experiment"):
            res = exp.run(pd.DataFrame())
        assert not res and "Input DataFrame is empty" in caplog.text
        MockAD.return_value.add_market_data.assert_not_called()
        if os.path.exists(r_dir):
            os.rmdir(r_dir)  # Cleanup

    @patch("main.ArbitrageDetector")
    def test_run_with_data(self, MockAD, full_detector_params):
        """
        Purpose: Test running an experiment with valid market data.
        Input: DataFrame with multiple timestamps and rates. Detector's find_arbitrage is mocked.
        Expected: Opportunities are aggregated from detector calls for each timestamp.
                  Detector's add_market_data and find_arbitrage are called for each row.
        """
        mock_det = MockAD.return_value
        mock_det.find_arbitrage.side_effect = [[{"profit": 1.0}], []]
        r_dir = "test_res_run_with_data"
        exp = ArbitrageExperiment(full_detector_params, results_dir=r_dir)
        df = pd.DataFrame(
            {"A/B": [1.0, 1.1], "B/C": [1.0, 1.1], "C/A": [1.0, 0.8]},
            index=pd.to_datetime(["20230101", "20230102"]),
        )
        res = exp.run(df)
        assert len(res) == 1 and res[0]["profit"] == 1.0
        assert (
            mock_det.add_market_data.call_count == 2
            and mock_det.find_arbitrage.call_count == 2
        )
        if os.path.exists(r_dir):
            os.rmdir(r_dir)  # Cleanup

    @patch("pickle.dump")  # Innermost decorator's mock
    @patch("builtins.open", new_callable=mock_open)  # Outermost decorator's mock
    def test_save_and_io_error(
        self, mock_builtin_open, mock_pickle_dump, full_detector_params, caplog
    ):
        """
        Purpose: Test saving experiment results to a pickle file, and handling IOErrors during save.
        Input: List of opportunity dictionaries.
        Expected:
            1. Successful save: pickle.dump called with correct data and file handle. Log message confirms.
            2. IOError on save: Error message logged, pickle.dump not called after error.
        """
        r_dir = "custom_save_results_dir"
        os.makedirs(r_dir, exist_ok=True)
        exp = ArbitrageExperiment(full_detector_params, results_dir=r_dir)
        pkl_p = os.path.join(r_dir, "test_opps.pkl")  # Define path once

        # Test successful save
        mock_builtin_open.reset_mock()
        mock_pickle_dump.reset_mock()  # Clean state for this part of test
        exp.save([{"profit": 1.0}], "test_opps")  # .pkl added by save method in main.py
        mock_builtin_open.assert_called_once_with(pkl_p, "wb")
        mock_pickle_dump.assert_called_once_with(
            [{"profit": 1.0}], mock_builtin_open()
        )  # mock_open() gives the file handle
        assert f"Saved 1 opportunities → {pkl_p}" in caplog.text

        # Test IO error
        mock_builtin_open.reset_mock()
        mock_pickle_dump.reset_mock()  # Clean state
        mock_builtin_open.side_effect = IOError("Disk full")  # Simulate IOError on open
        with caplog.at_level(logging.ERROR, logger="arbitrage_experiment"):
            exp.save([{"profit": 1.0}], "test_opps.pkl")

        mock_builtin_open.assert_called_once_with(
            pkl_p, "wb"
        )  # open is still attempted
        mock_pickle_dump.assert_not_called()  # dump should not be called if open failed
        assert f"Failed to save opportunities to {pkl_p}: Disk full" in caplog.text

        if os.path.exists(r_dir):
            os.rmdir(r_dir)  # Cleanup


@pytest.fixture(autouse=True)
def configure_logging_for_tests(caplog):
    """Ensures loggers used in the application are set to DEBUG for capture by pytest."""
    loggers_to_configure = [
        "main",
        "arbitrage_detector",
        "market_data_manager",
        "arbitrage_experiment",
    ]
    original_levels = {
        name: logging.getLogger(name).level for name in loggers_to_configure
    }
    for name in loggers_to_configure:
        logging.getLogger(name).setLevel(logging.DEBUG)

    caplog.set_level(logging.DEBUG)  # Set pytest's own capture level
    yield
    for name in loggers_to_configure:
        logging.getLogger(name).setLevel(original_levels.get(name, logging.WARNING))
