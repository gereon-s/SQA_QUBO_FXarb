# sqa_implementation.py
"""
Simulated Quantum Annealing (SQA) Sampler Implementation using Numba and Joblib.
This code implements a simulated quantum annealing sampler using Numba for
performance optimization and Joblib for parallel processing.
The sampler is designed to work with binary quadratic models (BQMs) and
supports various parameters for tuning the annealing process.
The code includes:
- A function to run a single SQA read using Numba for performance optimization.
- A class `SimulatedQuantumAnnealingSampler` that implements the SQA sampler.
- The class includes methods for sampling from a binary quadratic model (BQM),
  converting between binary and spin variables, and handling energy offsets.
- The class supports parallel processing using Joblib for multiple reads.
- The class captures timing information for performance analysis.
"""

import time
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from numba import njit
import joblib
from dimod import BinaryQuadraticModel, SampleSet, BINARY, SPIN


###############################################################################
# 1.  Numba-optimized function to run a single SQA read #######################
###############################################################################


@njit(fastmath=True, cache=True)
def _run_single_sqa_read_numba(
    h_vec: np.ndarray,
    J_mat: np.ndarray,
    num_variables: int,
    num_sweeps: int,
    num_trotter_slices: int,
    betas: np.ndarray,
    h_transverse_factors: np.ndarray,
    initial_spins: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, float]:
    np.random.seed(seed)
    spins = initial_spins.astype(np.float64).copy()
    slices_plus = np.roll(np.arange(num_trotter_slices), -1)
    slices_minus = np.roll(np.arange(num_trotter_slices), 1)

    for sweep in range(num_sweeps):
        beta = betas[sweep]
        gamma = h_transverse_factors[sweep]
        arg = beta * gamma / num_trotter_slices
        if arg < 1e-12:
            J_perp = 1000.0
        else:
            t = np.tanh(arg)
            if t <= 0:
                J_perp = 1000.0
            else:
                J_perp = -0.5 * np.log(t)
                if not np.isfinite(J_perp):
                    J_perp = 1000.0

        for p in range(num_trotter_slices):
            idxs = np.random.permutation(num_variables)
            for i in idxs:
                # Quantum coupling (β already in J_perp)
                h_q = J_perp * (spins[slices_plus[p], i] + spins[slices_minus[p], i])
                # Classical field
                h_c = h_vec[i] + np.dot(J_mat[i], spins[p])
                # Euclidean-action change ΔS
                delta_s = 2.0 * spins[p, i] * ((beta / num_trotter_slices) * h_c + h_q)
                # Metropolis: accept if ΔS ≤ 0 or rand < exp(-ΔS)
                if (delta_s <= 0.0) or (np.random.rand() < np.exp(-delta_s)):
                    spins[p, i] *= -1

    best_energy = np.inf
    best_slice = 0
    for p in range(num_trotter_slices):
        s = spins[p]
        e = h_vec @ s + 0.5 * s @ J_mat @ s
        if e < best_energy:
            best_energy = e
            best_slice = p
    return spins[best_slice].copy(), best_energy


###############################################################################
# 1.  Simulated Quantum Annealing Sampler Class ###############################
###############################################################################


class SimulatedQuantumAnnealingSampler:
    #
    def __init__(self):
        self.parameters: Optional[Dict[str, Any]] = None
        self.timing: Optional[Dict[str, Any]] = None

    def sample(self, bqm: BinaryQuadraticModel, **parameters: Any) -> SampleSet:
        start_time = time.perf_counter()
        num_reads = int(parameters.get("num_reads", 10))
        num_sweeps = int(parameters.get("num_sweeps", 1000))
        num_trotter_slices = int(parameters.get("num_trotter_slices", 10))
        beta_start = float(parameters.get("beta_start", 0.01))
        beta_end = float(parameters.get("beta_end", 10.0))
        gamma_start = float(parameters.get("transverse_field_start", 5.0))
        gamma_end = float(parameters.get("transverse_field_end", 0.01))
        seed = int(parameters.get("seed", int(time.time() * 1e6) % 2**32))
        num_workers = int(parameters.get("num_workers", 1))  # default to 1

        self.parameters = {
            "num_reads": num_reads,
            "num_sweeps": num_sweeps,
            "num_trotter_slices": num_trotter_slices,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "transverse_field_start": gamma_start,
            "transverse_field_end": gamma_end,
            "seed": seed,
            "num_workers": num_workers,
        }

        variables = list(bqm.variables)
        num_variables = len(variables)

        # Convert to SPIN and capture offset
        if bqm.vartype == BINARY:
            bqm_spin = bqm.change_vartype(SPIN, inplace=False)
        else:
            bqm_spin = bqm
        energy_offset = getattr(bqm_spin, "offset", 0.0)

        # Build h_vec and J_mat
        h_vec = np.zeros(num_variables)
        J_mat = np.zeros((num_variables, num_variables))
        var_index = {v: i for i, v in enumerate(variables)}
        for v, bias in bqm_spin.linear.items():
            h_vec[var_index[v]] = bias
        for (u, v), bias in bqm_spin.quadratic.items():
            i, j = var_index[u], var_index[v]
            if i != j:
                J_mat[i, j] = bias
                J_mat[j, i] = bias

        # Schedules
        betas = np.linspace(beta_start, beta_end, num_sweeps)
        gammas = np.linspace(gamma_start, gamma_end, num_sweeps)

        # Initial states & seeds
        rng = np.random.default_rng(seed)
        read_seeds = rng.integers(0, 2**32, num_reads, dtype=np.uint32)
        initial_states = rng.choice(
            [-1, 1], size=(num_reads, num_trotter_slices, num_variables)
        )

        # Parallel reads
        parallel = joblib.Parallel(n_jobs=num_workers)
        jobs = joblib.delayed(_run_single_sqa_read_numba)
        results = parallel(
            jobs(
                h_vec,
                J_mat,
                num_variables,
                num_sweeps,
                num_trotter_slices,
                betas,
                gammas,
                initial_states[i],
                int(read_seeds[i]),
            )
            for i in range(num_reads)
        )

        # Post-process
        spins_list, energies_raw = zip(*results) if results else ([], [])
        sorted_idx = np.argsort(energies_raw)

        samples_final, energies_final, num_occ = [], [], []
        for idx in sorted_idx:
            spin = spins_list[idx]
            e = energies_raw[idx]
            if not np.isfinite(e):
                continue
            if bqm.vartype == BINARY:
                bits = (spin.astype(int) + 1) // 2
                sample = {variables[i]: int(bits[i]) for i in range(num_variables)}
                energy = bqm.energy(sample)
            else:
                sample = {variables[i]: int(spin[i]) for i in range(num_variables)}
                energy = e + energy_offset  # add offset
            samples_final.append(sample)
            energies_final.append(energy)
            num_occ.append(1)

        self.timing = {"run_time_seconds": time.perf_counter() - start_time}
        info = {"parameters": self.parameters, "timing": self.timing}

        if not samples_final:
            return SampleSet.from_samples_bqm([], bqm)

        return SampleSet.from_samples(
            samples_final,
            bqm.vartype,
            energies_final,
            info=info,
            num_occurrences=num_occ,
            sort_labels=False,
        )
