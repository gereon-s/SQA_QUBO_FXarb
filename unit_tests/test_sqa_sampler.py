import sys
import types
import time
import os

import pytest
import numpy as np
import dimod
from dimod import BinaryQuadraticModel, SPIN, BINARY, SampleSet
from unittest.mock import patch

# Joblib mock for testing purposes
dummy_joblib = types.ModuleType("joblib")


class MockDelayedFunc:
    """Mocks joblib.delayed to capture the function and its arguments."""

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        # Returns a tuple that MockParallel expects
        return (self.func, args, kwargs)


dummy_joblib.delayed = MockDelayedFunc  # type: ignore


class MockParallel:
    """Mocks joblib.Parallel to execute tasks serially in the same process."""

    def __init__(self, n_jobs=None, verbose=0, **kwargs):
        # n_jobs and other parameters are ignored by mock
        pass

    def __call__(self, tasks_generator):
        # tasks_generator returns tuples of (function, args_tuple, kwargs_dict)
        results = []
        for task_tuple in tasks_generator:
            func_to_call, args_tuple, kwargs_dict = task_tuple
            results.append(func_to_call(*args_tuple, **kwargs_dict))
        return results


dummy_joblib.Parallel = MockParallel  # type: ignore
sys.modules["joblib"] = dummy_joblib


# Stub out numba for testing SQA sampler without JIT compilation
dummy_numba = types.ModuleType("numba")
dummy_numba.njit = lambda *args, **kwargs: (
    lambda f: f
)  # njit decorator becomes identity
sys.modules["numba"] = dummy_numba

# Import the SimulatedQuantumAnnealingSampler class and the function to be tested
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sqa_implementation import (
    SimulatedQuantumAnnealingSampler,
    _run_single_sqa_read_numba,
)


@pytest.fixture
def simple_bqm_binary():
    """
    Provides a simple Binary Quadratic Model with BINARY vartype.
    - Variables: x0, x1
    - Linear biases: h(x0)=1.0, h(x1)=1.0
    - Quadratic bias: J(x0,x1)=2.0
    - Offset: 0.0
    - Ground state: x0=0, x1=0, Energy=0.0
    """
    return BinaryQuadraticModel(
        {"x0": 1.0, "x1": 1.0}, {("x0", "x1"): 2.0}, 0.0, BINARY
    )


@pytest.fixture
def simple_bqm_spin():
    """
    Provides a simple Binary Quadratic Model with SPIN vartype.
    - Variables: s0, s1
    - Linear biases: h(s0)=-1.0, h(s1)=-1.0
    - Quadratic bias: J(s0,s1)=0.5
    - Offset: 0.0
    - Ground state: s0=1, s1=1, Energy=-1.5
    """
    return BinaryQuadraticModel(
        {"s0": -1.0, "s1": -1.0}, {("s0", "s1"): 0.5}, 0.0, SPIN
    )


class TestSimulatedQuantumAnnealingSampler:
    """
    Tests for the SimulatedQuantumAnnealingSampler class.
    These tests primarily mock the underlying `_run_single_sqa_read_numba`
    function to focus on the sampler's logic for:
    - Parameter handling and storage.
    - BQM vartype conversion (BINARY to SPIN for internal processing).
    - Conversion of results back to the original BQM's vartype.
    - Aggregation and sorting of samples from multiple reads.
    - Handling of edge cases like empty BQMs or non-finite energies.
    """

    def test_init(self):
        """
        Purpose: Test the initial state of the SQA sampler upon instantiation.
        Input: None.
        Expected: The `parameters` and `timing` attributes of the sampler
                  should be None before any sampling is performed.
        """
        sampler = SimulatedQuantumAnnealingSampler()
        assert sampler.parameters is None, "Initial parameters should be None"
        assert sampler.timing is None, "Initial timing should be None"

    @patch("sqa_implementation._run_single_sqa_read_numba")
    def test_sample_binary_bqm(self, mock_sqa_read, simple_bqm_binary):
        """
        Purpose: Test sampling from a BQM with BINARY vartype.
                 Ensures correct conversion to spin for internal SQA,
                 and correct conversion back to binary for the output SampleSet.
        Input:
            - `simple_bqm_binary`: providing  BINARY BQM.
            - `mock_sqa_read`: Mock for the numba-optimized SQA read function.
        Expected:
            - The mock SQA read function is called with appropriate spin-model parameters.
            - The returned SampleSet contains samples in BINARY format (0,1).
            - Energies in the SampleSet correspond to the original BINARY BQM.
            - Sampler parameters and timing information are recorded.
        """
        sampler = SimulatedQuantumAnnealingSampler()

        # Pre-calculate expected spin BQM and energy for the mock
        # Input BQM (simple_bqm_binary) is BINARY. Sampler converts it to SPIN internally.
        bqm_internal_spin = simple_bqm_binary.change_vartype(SPIN, inplace=False)
        # SQA read mock to return the ground state of simple_bqm_binary (x0=0, x1=0)
        # This corresponds to spins s0=-1, s1=-1
        mock_returned_spins = np.array([-1, -1])
        mock_returned_spin_energy = bqm_internal_spin.energy(
            {var: s for var, s in zip(bqm_internal_spin.variables, mock_returned_spins)}
        )
        mock_sqa_read.return_value = (mock_returned_spins, mock_returned_spin_energy)

        # Call the sampler
        response = sampler.sample(simple_bqm_binary, num_reads=1, seed=123)

        # Assertions
        mock_sqa_read.assert_called_once()
        args_call, _ = mock_sqa_read.call_args
        assert args_call[2] == len(
            simple_bqm_binary.variables
        ), "num_variables mismatch"

        # Check that the per-read seed is correctly derived from the main seed
        expected_rng = np.random.default_rng(123)  # Main seed for this sample run
        expected_first_read_seed = expected_rng.integers(
            0, 2**32, 1, dtype=np.uint32
        )[0]
        assert args_call[8] == int(expected_first_read_seed), "Per-read seed mismatch"

        assert isinstance(response, SampleSet), "Response should be a SampleSet"
        assert len(response) == 1, "Expected one sample for one read"

        first_sample_view = response.first
        assert first_sample_view.sample == {
            "x0": 0,
            "x1": 0,
        }, "Sample not converted to BINARY correctly"
        assert pytest.approx(first_sample_view.energy) == simple_bqm_binary.energy(
            {"x0": 0, "x1": 0}
        ), "Energy mismatch for BINARY BQM"

        assert sampler.parameters is not None, "Sampler parameters not set"
        assert sampler.parameters["seed"] == 123, "Main seed in parameters mismatch"
        assert (
            sampler.timing is not None and "run_time_seconds" in sampler.timing
        ), "Timing info not recorded"

    @patch("sqa_implementation._run_single_sqa_read_numba")
    def test_sample_spin_bqm(self, mock_sqa_read, simple_bqm_spin):
        """
        Purpose: Test sampling from a BQM that is already in SPIN vartype.
        Input:
            - `simple_bqm_spin`:  providing SPIN BQM.
            - `mock_sqa_read`: Mock for the SQA read function.
        Expected:
            - The sampler uses the SPIN BQM directly for internal processing.
            - The returned SampleSet contains samples in SPIN format (-1,1).
            - Energies match the input SPIN BQM (plus its offset).
        """
        sampler = SimulatedQuantumAnnealingSampler()
        assert simple_bqm_spin.vartype == dimod.SPIN, "Fixture should be SPIN"

        # Mock SQA to return the ground state of simple_bqm_spin (s0=1, s1=1)
        mock_returned_spins = np.array([1, 1])

        # Calculate raw spin energy (without BQM offset) for the mock
        linear_energy_contrib = sum(
            simple_bqm_spin.linear[v] * s
            for v, s in zip(simple_bqm_spin.variables, mock_returned_spins)
        )
        quadratic_energy_contrib = sum(
            simple_bqm_spin.quadratic[k]
            * mock_returned_spins[simple_bqm_spin.variables.index(k[0])]
            * mock_returned_spins[simple_bqm_spin.variables.index(k[1])]
            for k in simple_bqm_spin.quadratic
        )
        raw_mock_energy = linear_energy_contrib + quadratic_energy_contrib

        mock_sqa_read.return_value = (mock_returned_spins, raw_mock_energy)

        response = sampler.sample(simple_bqm_spin, num_reads=1, seed=456)

        mock_sqa_read.assert_called_once()
        first_sample_view = response.first
        assert first_sample_view.sample == {"s0": 1, "s1": 1}, "SPIN sample mismatch"
        # Sampler adds BQM offset to raw energy for SPIN BQMs
        assert (
            pytest.approx(first_sample_view.energy)
            == raw_mock_energy + simple_bqm_spin.offset
        ), "SPIN BQM energy mismatch"

    def test_determinism_with_seed(self, simple_bqm_binary):
        """
        Purpose: Verify that the SQA sampler produces deterministic results when a seed is provided.
                 This tests the chain: main seed -> RNG for read_seeds -> per-read_seed for SQA.
        Input: A BINARY BQM, fixed SQA parameters including a main seed.
               `_run_single_sqa_read_numba` is mocked with a version that uses its own seed argument.
        Expected: Two independent sampling runs with the same parameters yield identical SampleSets.
        """
        assert simple_bqm_binary.vartype == dimod.BINARY, "Fixture should be BINARY"

        # This mock uses the per-read seed passed to it
        def mock_read_uses_per_read_seed(
            h_vec,
            J_mat,
            num_vars,
            num_sweeps,
            num_trotter_slices,
            betas,
            h_transverse_factors,
            initial_spins_for_read,
            per_read_seed_val,
        ):
            np.random.seed(per_read_seed_val)  # Use the per-read seed
            # Simulate deterministic spin choice based on seed
            final_spins = initial_spins_for_read[0].copy()  # Use first trotter slice
            if len(final_spins) > 0 and per_read_seed_val % 2 == 0:
                final_spins[0] *= -1  # Flip first spin if seed is even
            # Calculate raw spin energy
            energy = h_vec @ final_spins + 0.5 * final_spins @ J_mat @ final_spins
            return final_spins, energy

        with patch(
            "sqa_implementation._run_single_sqa_read_numba",
            side_effect=mock_read_uses_per_read_seed,
        ):
            common_params = {
                "num_reads": 2,
                "num_sweeps": 10,
                "num_trotter_slices": 2,
                "seed": 789,
            }

            sampler1 = SimulatedQuantumAnnealingSampler()
            response1 = sampler1.sample(simple_bqm_binary, **common_params)

            sampler2 = SimulatedQuantumAnnealingSampler()  # A fresh instance
            response2 = sampler2.sample(simple_bqm_binary, **common_params)

            assert len(response1) == len(response2), "Number of samples differ"
            # SampleSet.data() iterates in energy-sorted order by default
            # To check true determinism of reads before sorting, we need to compare based on original read order
            # or ensure mock returns are sufficiently distinct if seeds are different.
            # Here: if all reads are identical = sorted list is identical.
            for datum1, datum2 in zip(response1.data(), response2.data()):
                assert (
                    datum1.sample == datum2.sample
                ), "Samples differ between deterministic runs"
                assert pytest.approx(datum1.energy) == datum2.energy, "Energies differ"
                assert (
                    datum1.num_occurrences == datum2.num_occurrences
                ), "Occurrences differ"

    @patch("sqa_implementation._run_single_sqa_read_numba")
    def test_sample_multiple_reads_sorting(self, mock_sqa_read, simple_bqm_binary):
        """
        Purpose: Test that results from multiple SQA reads are correctly aggregated
                 and the final SampleSet is sorted by energy (lowest first).
        Input: A BINARY BQM, num_reads=2.
               `_run_single_sqa_read_numba` is mocked to return two distinct spin samples,
               with the one corresponding to higher binary energy returned by the "first" mock call.
        Expected: The final SampleSet contains two samples, correctly converted to BINARY,
                  and sorted by their BINARY BQM energies.
        """
        sampler = SimulatedQuantumAnnealingSampler()
        assert simple_bqm_binary.vartype == dimod.BINARY, "Fixture should be BINARY"

        bqm_internal_spin = simple_bqm_binary.change_vartype(SPIN, inplace=False)

        # Spin config for BINARY x0=0, x1=0 (Energy_bin=0.0)
        spins1 = np.array([-1, -1])
        energy1_spin_raw = (
            bqm_internal_spin.energy(
                {v: s for v, s in zip(bqm_internal_spin.variables, spins1)}
            )
            - bqm_internal_spin.offset
        )

        # Spin config for BINARY x0=1, x1=0 (Energy_bin=1.0)
        spins2 = np.array([1, -1])
        energy2_spin_raw = (
            bqm_internal_spin.energy(
                {v: s for v, s in zip(bqm_internal_spin.variables, spins2)}
            )
            - bqm_internal_spin.offset
        )

        # Mock SQA to return results in an order not sorted by final binary energy
        mock_sqa_read.side_effect = [
            (spins2, energy2_spin_raw),
            (spins1, energy1_spin_raw),
        ]

        response = sampler.sample(simple_bqm_binary, num_reads=2)
        assert len(response) == 2, "Expected two samples"

        # Verify all samples in the response are in BINARY format
        for datum in response.data():
            for var_name, value in datum.sample.items():
                assert value in (
                    0,
                    1,
                ), f"Sample {datum.sample} contains non-binary value {value}"

        # SampleSet.from_samples sorts by energy by default.
        # `response.first` should correspond to the lowest energy binary configuration.
        assert response.first.sample == {
            "x0": 0,
            "x1": 0,
        }, "Lowest energy sample mismatch"
        assert pytest.approx(response.first.energy) == simple_bqm_binary.energy(
            {"x0": 0, "x1": 0}
        ), "Energy of first sample incorrect"

        # Check the second sample in the sorted list
        data_as_list = list(response.data())
        assert data_as_list[1].sample == {"x0": 1, "x1": 0}, "Second sample mismatch"
        assert pytest.approx(data_as_list[1].energy) == simple_bqm_binary.energy(
            {"x0": 1, "x1": 0}
        ), "Energy of second sample incorrect"

    def test_empty_bqm(self):
        """
        Purpose: Test the sampler's behavior when given an empty BQM (no variables).
        Input: An empty BINARY BQM.
        Expected: returns a SampleSet with one empty sample ({})
                  and energy equal to the BQM's offset (0.0 in this case).
        """
        sampler = SimulatedQuantumAnnealingSampler()
        empty_bqm = BinaryQuadraticModel(
            {}, {}, 0.0, BINARY
        )  # Explicitly no linear/quadratic
        response = sampler.sample(empty_bqm, num_reads=1)

        assert len(response) == 1, "Expected one sample for empty BQM"
        assert list(response.variables) == [], "Variables list should be empty"
        assert response.first.sample == {}, "Sample should be an empty dict"
        assert (
            pytest.approx(response.first.energy) == 0.0
        ), "Energy should be BQM offset"

    @patch("sqa_implementation._run_single_sqa_read_numba")
    def test_non_finite_energy_from_read(self, mock_sqa_read, simple_bqm_binary):
        """
        Purpose: Test how the sampler handles non-finite (inf, NaN) energies
                 returned by the underlying SQA read function.
        Input: `_run_single_sqa_read_numba` is mocked to return np.inf, then np.nan.
        Expected: Samples associated with non-finite energies should be discarded,
                  leading to an empty SampleSet if all reads yield non-finite energy.
        """
        sampler = SimulatedQuantumAnnealingSampler()
        assert simple_bqm_binary.vartype == dimod.BINARY, "Fixture should be BINARY"

        mock_valid_spins = np.array([-1, -1])  # A valid spin configuration

        # Test with np.inf
        mock_sqa_read.return_value = (mock_valid_spins, np.inf)
        response_inf = sampler.sample(simple_bqm_binary, num_reads=1)
        assert len(response_inf) == 0, "SampleSet should be empty for np.inf energy"

        # Test with np.nan
        mock_sqa_read.return_value = (mock_valid_spins, np.nan)
        response_nan = sampler.sample(simple_bqm_binary, num_reads=1)
        assert len(response_nan) == 0, "SampleSet should be empty for np.nan energy"


def test_run_single_sqa_read_numba_logic_check():
    """
    Purpose: Perform a basic python logic check of the `_run_single_sqa_read_numba` function.
             This test executes the function's python code directly because the Numba JIT
             compilation is bypassed during unit tests.
             It does not test the Numba JIT compilation itself or performance aspects.
    Input: Manually defined parameters for a simple 1-variable BQM (h=[-1], J={}).
           The ground state is spin=+1, energy=-1.0.
    Expected:
        - The function returns a 1D numpy array for `final_spins` and a float for `final_energy`.
        - The `final_energy` returned by the function must match the energy calculated
          from `h_vec`, `J_mat`, and the `final_spins`.
        - The `final_energy` should be reasonably low, ideally the ground state energy,
          or at least not worse than an arbitrary valid state.
    """
    h_vec = np.array([-1.0])
    J_mat = np.array([[0.0]])
    num_variables = 1
    num_sweeps = 10  # Minimal sweeps for a quick test run
    num_trotter_slices = 2  # Minimal slices
    betas = np.linspace(0.1, 1.0, num_sweeps)
    h_transverse_factors = np.linspace(5.0, 0.1, num_sweeps)  # Gamma schedule
    # Initial spins: one trotter slice starts at -1, other at +1
    initial_spins = np.array(
        [[-1.0], [1.0]], dtype=np.float64
    )  # Numba expects specific dtypes
    seed_val = 123

    final_spins, final_energy = _run_single_sqa_read_numba(
        h_vec,
        J_mat,
        num_variables,
        num_sweeps,
        num_trotter_slices,
        betas,
        h_transverse_factors,
        initial_spins,
        seed_val,
    )

    assert final_spins.shape == (num_variables,), "Final spins shape mismatch"

    # Verify that the energy returned by the function is consistent with the spins returned
    calculated_energy = h_vec @ final_spins + 0.5 * (final_spins @ J_mat @ final_spins)
    assert (
        pytest.approx(final_energy) == calculated_energy
    ), "Reported energy differs from calculated"

    # For this simple 1-variable problem (h=-1), energy states are:
    # spin = -1 => E = (-1)*(-1) = 1.0
    # spin = +1 => E = (-1)*(+1) = -1.0 (ground state)
    # The SQA should find an energy less than or equal to the higher energy state.
    assert final_energy <= 1.0, "Final energy not better than or equal to a known state"
