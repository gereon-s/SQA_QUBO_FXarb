# sqa_implementation.py
"""
Simulated Quantum Annealing (SQA) Sampler Implementation

This code implements a Simulated Quantum Annealing sampler that mimics the
behavior of quantum annealers for solving combinatorial optimization problems.

Theory:
Simulates quantum annealing by introducing imaginary time slices (Trotter decomposition)
to approximate quantum superposition and tunneling effects. The algorithm evolves a system
of coupled classical spins that represent quantum states at different imaginary times.

Key Physics Concepts:
- Quantum Tunneling: Simulated via transverse field coupling between time slices
- Thermal Fluctuations: Controlled by temperature schedule (beta parameter)
- Quantum Superposition: Approximated by multiple coupled spin configurations
- Adiabatic Evolution: Slow reduction of quantum field to reach ground state

Mathematical Model:
The effective Hamiltonian combines:
1. Classical Problem: H_classical = Σᵢ hᵢσᵢ + Σᵢⱼ Jᵢⱼσᵢσⱼ
2. Quantum Coupling: H_quantum = -Γ Σᵢₚ σᵢₚσᵢ₍ₚ₊₁₎ (between time slices)
3. Total Energy: H_total = H_classical + H_quantum

Performance Optimizations:
- Numba JIT compilation for critical numerical loops
- Joblib parallelization across multiple independent runs
- Efficient matrix operations and memory layout
- Numerical stability handling

Dependencies:
    - numba: JIT compilation for performance-critical functions
    - joblib: Parallel processing for multiple annealing runs
    - dimod: Interface compatibility with D-Wave ecosystem
    - numpy: Numerical operations and random number generation
"""

import time
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from numba import njit
import joblib
from dimod import BinaryQuadraticModel, SampleSet, BINARY, SPIN

###############################################################################
# 1. Core SQA Algorithm - Numba Optimized ####################################
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
    """
    Execute a single Simulated Quantum Annealing run using Numba acceleration.

    This function implements the core SQA algorithm with Trotter decomposition,
    simulating quantum effects through coupling between imaginary time slices.

    Physics Implementation:
    - Each time slice represents a quantum state replica
    - Transverse field creates coupling between adjacent slices (quantum tunneling)
    - Classical field implements the original optimization problem
    - Metropolis algorithm ensures proper thermal equilibrium

    Mathematical Details:
    - Action change: ΔS = 2σᵢₚ[(β/P)hᵢᶜ + J⊥(σᵢₚ₋₁ + σᵢₚ₊₁)]
    - Quantum coupling: J⊥ = -½ln(tanh(βΓ/P)) where Γ is transverse field
    - Acceptance probability: P(accept) = min(1, exp(-ΔS))

    Args:
        h_vec: Linear coefficients (problem-specific fields)
        J_mat: Quadratic coefficients (interaction matrix)
        num_variables: Number of optimization variables
        num_sweeps: Monte Carlo steps for annealing schedule
        num_trotter_slices: Number of imaginary time slices (quantum accuracy)
        betas: Temperature schedule (inverse temperature β = 1/kT)
        h_transverse_factors: Transverse field schedule (quantum → classical)
        initial_spins: Starting spin configuration for all slices
        seed: Random seed for reproducible results

    Returns:
        Tuple of (best_spin_configuration, lowest_energy_found)
    """
    # Initialize random seed for this specific run
    np.random.seed(seed)

    # Working spin array: [slice_index, variable_index] = spin_value
    spins = initial_spins.astype(np.float64).copy()

    # Precompute slice connectivity for periodic boundary conditions
    # Quantum coupling connects adjacent time slices in a ring topology
    slices_plus = np.roll(np.arange(num_trotter_slices), -1)  # p+1 (next slice)
    slices_minus = np.roll(np.arange(num_trotter_slices), 1)  # p-1 (previous slice)

    # Main annealing loop: evolve system from quantum to classical
    for sweep in range(num_sweeps):
        # Current annealing parameters
        beta = betas[sweep]  # Inverse temperature (higher = colder)
        gamma = h_transverse_factors[
            sweep
        ]  # Transverse field strength (quantum effect)

        # Calculate quantum coupling strength J_perp
        # Derivation: J⊥ = -½ln(tanh(βΓ/P)) from quantum-to-classical mapping
        arg = beta * gamma / num_trotter_slices
        if arg < 1e-12:  # Avoid numerical issues when transverse field → 0
            J_perp = 1000.0  # Strong coupling (classical limit)
        else:
            t = np.tanh(arg)
            if t <= 0:  # Numerical safety check
                J_perp = 1000.0
            else:
                J_perp = -0.5 * np.log(t)
                if not np.isfinite(J_perp):  # Handle overflow/underflow
                    J_perp = 1000.0

        # Update all time slices
        for p in range(num_trotter_slices):
            # Random variable ordering to avoid systematic bias
            idxs = np.random.permutation(num_variables)

            # Single-spin-flip updates within current time slice
            for i in idxs:
                # Quantum field: coupling to neighboring time slices
                # simulates quantum tunneling between eigenstates
                h_q = J_perp * (spins[slices_plus[p], i] + spins[slices_minus[p], i])

                # Classical field: original optimization problem
                # Linear term + quadratic interactions within same time slice
                h_c = h_vec[i] + np.dot(J_mat[i], spins[p])

                # Euclidean action change for spin flip: σᵢₚ → -σᵢₚ
                # Factor of 2 accounts for σᵢₚ → -σᵢₖ transition
                delta_s = 2.0 * spins[p, i] * ((beta / num_trotter_slices) * h_c + h_q)

                # Metropolis acceptance criterion
                # Always accept energy-lowering moves (ΔS ≤ 0)
                # Accept energy-raising moves with thermal probability exp(-ΔS)
                if (delta_s <= 0.0) or (np.random.rand() < np.exp(-delta_s)):
                    spins[p, i] *= -1  # Flip spin

    # Extract best solution from all time slices
    # Each slice represents a potential solution to the original problem
    best_energy = np.inf
    best_slice = 0

    for p in range(num_trotter_slices):
        s = spins[p]
        # Calculate classical energy: E = h·s + ½s·J·s
        e = h_vec @ s + 0.5 * s @ J_mat @ s
        if e < best_energy:
            best_energy = e
            best_slice = p

    return spins[best_slice].copy(), best_energy


###############################################################################
# 2. Main SQA Sampler Class ###################################################
###############################################################################


class SimulatedQuantumAnnealingSampler:
    """
    Simulated Quantum Annealing sampler for binary optimization.

    This sampler provides a quantum-inspired approach to solving Binary Quadratic Models
    (BQMs) by simulating the quantum annealing process. The implementation uses classical
    computation to approximate quantum effects through imaginary time evolution and Trotter
    decomposition.

    Key Features:
    - Quantum tunneling simulation via transverse field coupling
    - Parallel processing for multiple independent annealing runs
    - Automatic vartype conversion (BINARY ↔ SPIN)
    - Performance optimization with Numba JIT compilation
    - Compatible with dimod ecosystem (D-Wave Ocean SDK)

    Usage Pattern:
    ```python
    sampler = SimulatedQuantumAnnealingSampler()
    response = sampler.sample(bqm, num_reads=1000, num_sweeps=500)
    best_solution = response.first.sample
    ```
    """

    def __init__(self):
        """Initialize SQA sampler with empty state."""
        self.parameters: Optional[Dict[str, Any]] = None  # Store last run parameters
        self.timing: Optional[Dict[str, Any]] = None  # Performance metrics

    def sample(self, bqm: BinaryQuadraticModel, **parameters: Any) -> SampleSet:
        """
        Sample solutions from a Binary Quadratic Model using SQA.

        This method guides the complete SQA process:
        1. Parameter validation and preprocessing
        2. Problem conversion to Ising spin representation
        3. Parallel execution of multiple annealing runs
        4. Solution post-processing and ranking
        5. Result packaging in dimod-compatible format

        Physics Parameters:
        - beta_start/end: Temperature schedule (exploration → exploitation)
        - transverse_field_start/end: Quantum field schedule (quantum → classical)
        - num_trotter_slices: Quantum simulation accuracy (more slices = better approximation)

        Computational Parameters:
        - num_reads: Independent annealing runs (more reads = better solution diversity)
        - num_sweeps: Monte Carlo steps per run (longer annealing = better convergence)
        - num_workers: Parallel processing threads (speedup for multiple reads)

        Args:
            bqm: Binary Quadratic Model to solve
            **parameters: SQA algorithm parameters

        Returns:
            SampleSet containing solutions ranked by energy with metadata
        """
        start_time = time.perf_counter()

        # Extract and validate parameters with sensible defaults
        num_reads = int(parameters.get("num_reads", 10))  # Parallel runs
        num_sweeps = int(parameters.get("num_sweeps", 1000))  # Annealing length
        num_trotter_slices = int(
            parameters.get("num_trotter_slices", 10)
        )  # Quantum accuracy

        # Temperature schedule: β = 1/(kT), higher β = lower temperature
        beta_start = float(
            parameters.get("beta_start", 0.01)
        )  # High temp (exploration)
        beta_end = float(parameters.get("beta_end", 10.0))  # Low temp (exploitation)

        # Transverse field schedule: Γ controls quantum tunneling strength
        gamma_start = float(
            parameters.get("transverse_field_start", 5.0)
        )  # Strong quantum
        gamma_end = float(parameters.get("transverse_field_end", 0.01))  # Weak quantum

        # Computational parameters
        seed = int(
            parameters.get("seed", int(time.time() * 1e6) % 2**32)
        )  # Reproducibility
        num_workers = int(parameters.get("num_workers", 1))  # Parallelization

        # Store parameters for debugging
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

        # Extract variable information
        variables = list(bqm.variables)
        num_variables = len(variables)

        # Convert BQM to SPIN representation for SQA algorithm
        # SPIN variables σ ∈ {-1, +1} are natural for Ising model formulation
        if bqm.vartype == BINARY:
            bqm_spin = bqm.change_vartype(SPIN, inplace=False)
        else:
            bqm_spin = bqm
        energy_offset = getattr(
            bqm_spin, "offset", 0.0
        )  # Preserve constant energy terms

        # Build coefficient arrays for efficient numerical computation
        h_vec = np.zeros(num_variables)  # Linear coefficients
        J_mat = np.zeros((num_variables, num_variables))  # Quadratic coefficients
        var_index = {v: i for i, v in enumerate(variables)}  # Variable → index mapping

        # Populate linear terms: Σᵢ hᵢσᵢ
        for v, bias in bqm_spin.linear.items():
            h_vec[var_index[v]] = bias

        # Populate quadratic terms: Σᵢⱼ Jᵢⱼσᵢσⱼ (symmetric matrix)
        for (u, v), bias in bqm_spin.quadratic.items():
            i, j = var_index[u], var_index[v]
            if i != j:  # Avoid self-loops
                J_mat[i, j] = bias
                J_mat[j, i] = bias  # Ensure symmetry

        # Create annealing schedules
        # Linear interpolation between start and end values over sweep iterations
        betas = np.linspace(beta_start, beta_end, num_sweeps)  # Temperature schedule
        gammas = np.linspace(
            gamma_start, gamma_end, num_sweeps
        )  # Transverse field schedule

        # Initialize random states and seeds for parallel runs
        rng = np.random.default_rng(seed)
        read_seeds = rng.integers(0, 2**32, num_reads, dtype=np.uint32)  # Unique seeds

        # Random initial spin configurations: shape = (num_reads, num_slices, num_variables)
        initial_states = rng.choice(
            [-1, 1], size=(num_reads, num_trotter_slices, num_variables)
        )

        # Execute parallel SQA runs using joblib
        # Each worker runs independent annealing with different random seed
        parallel = joblib.Parallel(n_jobs=num_workers)
        jobs = joblib.delayed(_run_single_sqa_read_numba)  # Numba-compiled function

        results = parallel(
            jobs(
                h_vec,  # Problem linear coefficients
                J_mat,  # Problem quadratic coefficients
                num_variables,  # Problem size
                num_sweeps,  # Annealing schedule length
                num_trotter_slices,  # Quantum simulation accuracy
                betas,  # Temperature schedule
                gammas,  # Transverse field schedule
                initial_states[i],  # Random initial state for this run
                int(read_seeds[i]),  # Unique random seed for this run
            )
            for i in range(num_reads)
        )

        # Post-process results: convert spins to samples and handle energy offsets
        spins_list, energies_raw = zip(*results) if results else ([], [])
        sorted_idx = np.argsort(energies_raw)  # Sort by energy (best solutions first)

        samples_final, energies_final, num_occ = [], [], []

        for idx in sorted_idx:
            spin = spins_list[idx]
            e = energies_raw[idx]

            # Skip invalid solutions (numerical errors)
            if not np.isfinite(e):
                continue

            # Convert spin configuration to requested vartype and calculate exact energy
            if bqm.vartype == BINARY:
                # Convert SPIN {-1, +1} → BINARY {0, 1}: x = (σ + 1)/2
                bits = (spin.astype(int) + 1) // 2
                sample = {variables[i]: int(bits[i]) for i in range(num_variables)}
                energy = bqm.energy(sample)  # Use original BQM for exact energy
            else:
                # Keep SPIN representation
                sample = {variables[i]: int(spin[i]) for i in range(num_variables)}
                energy = (
                    e + energy_offset
                )  # Add constant offset from vartype conversion

            samples_final.append(sample)
            energies_final.append(energy)
            num_occ.append(1)  # Each sample appears once

        # Record timing information for analysis
        self.timing = {"run_time_seconds": time.perf_counter() - start_time}
        info = {"parameters": self.parameters, "timing": self.timing}

        # Handle edge case: no valid solutions found
        if not samples_final:
            return SampleSet.from_samples_bqm([], bqm)

        # Results to dimod-compatible SampleSet format
        return SampleSet.from_samples(
            samples_final,  # Solution samples
            bqm.vartype,  # Variable type (BINARY or SPIN)
            energies_final,  # Corresponding energies
            info=info,  # Algorithm metadata
            num_occurrences=num_occ,  # Sample frequencies
            sort_labels=False,  # Preserve energy-based ordering
        )
