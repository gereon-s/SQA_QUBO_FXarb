# FX Arbitrage Detection Using Simulated Quantum Annealing

**Bachelor Thesis Implementation**  
**Author:** `Gereon S`  
**Date:** May 2024

---

## Abstract

This project implements and evaluates a system for detecting foreign exchange (FX) arbitrage opportunities using quantum-inspired optimization techniques. The system formulates arbitrage detection as a Quadratic Unconstrained Binary Optimization (QUBO) problem and employs a custom Simulated Quantum Annealing (SQA) sampler for solution finding. Performance is benchmarked against the traditional Bellman-Ford algorithm for negative cycle detection.

**Key Contributions:**
- Adaptive QUBO formulation with dynamic penalty scaling
- Custom SQA implementation optimized with Numba JIT compilation
- Comprehensive hyperparameter optimization using Optuna with time-series cross-validation
- Statistical comparison framework between quantum-inspired and classical approaches

---

## Project Structure

```
REPO/
├── main.py                    # Core implementation: ArbitrageDetector, QUBOBuilder
├── sqa_implementation.py      # Custom Simulated Quantum Annealing sampler
├── data/                      # FX time-series data (placeholder files)
│   ├── fx_data_march.csv
│   └── fx_data_april.csv
├── tuning/                    # Hyperparameter optimization
│   ├── tuner.py              # Initial Optuna optimization
│   ├── tuner_round2.py       # Refined optimization with seeding
│   ├── tunerCV.py           # Time-series cross-validation tuning
│   └── tuning_output/
│       ├── round1/          # Initial tuning results
│       ├── round2/          # Refined tuning results
│       └── CV/              # Cross-validation results
├── oos_test_march/           # March out-of-sample testing
│   ├── run_holdout_march.py     # SQA evaluation
│   ├── benchmark_bf_march.py    # Bellman-Ford benchmark
│   └── output/
├── oos_test_april/           # April out-of-sample testing
│   ├── run_holdout_april.py     # SQA evaluation
│   ├── benchmark_bf_april.py    # Bellman-Ford benchmark
│   └── output/
├── analysis/                 # Statistical analysis and visualization
│   ├── march/               # March-specific analysis
│   ├── april/               # April-specific analysis
│   └── comparison/          # Cross-month comparison
└── unit_tests/              # Unit testing framework
```

---

## Core Components

### ArbitrageNetwork Class

Represents the FX market as a directed graph where nodes are currencies and edge weights are derived from `-log(effective_fx_rate)`, incorporating transaction costs. This transformation converts the arbitrage detection problem into finding negative-weight cycles.

### QUBOBuilder Class

Constructs the QUBO formulation with adaptive scaling mechanisms:

```python
# Objective function: maximize profit from arbitrage cycles
objective = sum(log_profit[edge] * x[edge] for edge in edges)

# Constraint penalties
flow_penalty = λ₁ * sum((in_degree[node] - out_degree[node])² for node in nodes)
cycle_penalty = λ₂ * sum(x[edge] for edge in edges)

# Final QUBO formulation
minimize: -objective + flow_penalty + cycle_penalty
```

**Key Features:**
- Adaptive penalty scaling based on `objective_scale` parameter
- Flow conservation constraints ensuring valid cycles
- Transaction cost integration in edge weight calculation

### SimulatedQuantumAnnealingSampler Class

Custom SQA implementation featuring:

```python
@njit(fastmath=True, cache=True)
def _run_single_sqa_read_numba(h_vec, J_mat, num_variables, num_sweeps, 
                               num_trotter_slices, betas, h_transverse_factors, 
                               initial_spins, seed):
    # Numba-optimized SQA core loop
    # Metropolis updates across Trotter slices
    # Returns best spin configuration and energy
```

**Technical Features:**
- Numba JIT compilation for performance optimization
- Joblib parallelization for multiple reads
- Trotter slice simulation of quantum annealing dynamics
- Compatible with dimod BinaryQuadraticModel interface

---

## Methodology

### QUBO Formulation

The arbitrage detection problem is formulated as a binary optimization:

**Variables:** `x[i,j] ∈ {0,1}` indicating whether edge (i,j) is selected

**Objective:** Maximize total log-profit from selected edges

**Constraints:**
1. **Flow Conservation:** For each currency node, in-degree equals out-degree
2. **Cycle Validity:** Selected edges must form valid cycles within length bounds

### Simulated Quantum Annealing Process

The SQA algorithm simulates quantum annealing through:

1. **Quantum-to-Classical Mapping:** Problem mapped to classical system with Trotter dimension
2. **Annealing Schedule:** Gradual reduction of transverse field strength
3. **Temperature Schedule:** Increasing inverse temperature β over sweeps
4. **Metropolis Updates:** Accept/reject spin flips based on energy change

### Performance Evaluation

Comparison metrics between SQA and Bellman-Ford:

| Metric | Description | Statistical Test |
|--------|-------------|-----------------|
| Detection Rate | Percentage of timestamps with arbitrage found | Fisher's Exact Test |
| Profit Distribution | Statistical analysis of found opportunities | Mann-Whitney U Test |
| Runtime Efficiency | Computational performance comparison | Mann-Whitney U Test |
| Cycle Characteristics | Length and structure analysis | Descriptive Statistics |

---

## Dependencies

```python
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
networkx>=2.6.0
scipy>=1.7.0
dimod>=0.10.0

# Performance optimization
numba>=0.54.0
joblib>=1.1.0

# Hyperparameter tuning
optuna>=2.10.0

# Analysis and visualization
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0

# Testing
pytest>=6.2.0
```

---

## Usage

### Data Preparation

Format FX data as CSV with datetime index:

```csv
Timestamp,USD/EUR,EUR/JPY,GBP/USD,JPY/GBP
2025-03-01 00:00:00,0.9123,160.12,1.2789,0.0062
2025-03-01 00:01:00,0.9124,160.15,1.2791,0.0062
```

Place files in `data/` directory as `fx_data_march.csv` and `fx_data_april.csv`.

### Execution Pipeline

**Step 1: Unit Testing**
```bash
pytest unit_tests/
```

**Step 2: Hyperparameter Optimization**
```bash
cd tuning/
python tuner.py # Global first search
python tuner_round2.py # Second narrow search
python tunerCV.py  # Time-series cross-validation search
```

**Step 3: Out-of-Sample Evaluation**
```bash
# March evaluation
cd oos_test_march/
python run_holdout_march.py      # SQA method
python benchmark_bf_march.py     # Bellman-Ford benchmark

# April evaluation
cd oos_test_april/
python run_holdout_april.py      # SQA method  
python benchmark_bf_april.py     # Bellman-Ford benchmark
```

**Step 4: Statistical Analysis**
```bash
cd analysis/march/
python analysis_march.py

cd ../april/
python analysis_april.py

# Comparative analysis
cd ../comparison/
jupyter notebook compare.ipynb
```

---

## Implementation Details

### Adaptive Scaling Mechanism

The QUBOBuilder implements adaptive penalty scaling:

```python
# Scale penalties relative to objective magnitude
max_weight = max(abs(edge_weight) for edge_weight in graph_edges)
flow_penalty = flow_penalty_multiplier * objective_scale
cycle_penalty = cycle_penalty_multiplier * objective_scale

# Normalize edge weights to prevent numerical issues
normalized_weight = edge_weight / max_weight * objective_scale
```

### Cross-Validation Strategy

Time-series cross-validation with rolling windows:

```python
window_size = 3  # days
step_size = 2    # days
n_folds = 6      # total folds
```

### Statistical Testing Framework

Non-parametric tests for robust comparison:
- **Mann-Whitney U Test:** For profit and runtime distributions
- **Fisher's Exact Test:** For detection rate comparisons
- **Effect Size Calculation:** Cohen's d for practical significance

---

## Data License Notice

The actual FX market data used in this research is licensed from Refinitiv and cannot be redistributed.
Refinitiv Codebook IDE code attached.

---

## References

Detailed references and mathematical derivations are provided in the full thesis document.

---

## Usage
This code is part of a Bachelor thesis submitted for academic evaluation. 
