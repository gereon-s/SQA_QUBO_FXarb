# coding: utf-8
"""
Second tuning round for the FX adaptive arbitrage detector.
This round uses a more refined search space and a different
pruning strategy.

"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import timedelta
import multiprocessing as mp

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.storages.journal import JournalFileBackend
import pandas as pd
from tqdm.auto import tqdm

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from main import ArbitrageDetector, MarketDataManager
except ImportError as e:
    raise ImportError(
        "Could not import required classes from 'main.py'. "
        "Ensure 'main.py' is in the same directory or your PYTHONPATH."
        f" Original error: {e}"
    )


# ─────────────────────────────────────────────────────────────
# 0 · Paths & constants
# ─────────────────────────────────────────────────────────────

# --- PATH DEFINITION FOR CSV ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_CSV: Path = PROJECT_ROOT / "data" / "fx_data_march.csv"

RESULTS_DIR = Path("tuning_output/round2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # create if not exist


FIDELITY_STEPS: list[int] = [4, 8, 16, 32]
FIDELITY_SAMPLE: int = 500
STALL_LIMIT: int = 200
N_TRIALS: int = 100
WINDOW_DAYS: int = 2


# Cached window shared by all worker threads
WEEK_DATA = None  # type: ignore

# ─────────────────────────────────────────────────────────────
# 1 · Data loading helper
# ─────────────────────────────────────────────────────────────


def load_window(csv_path: Path, days: int = WINDOW_DAYS) -> pd.DataFrame:
    df = MarketDataManager(str(csv_path)).fetch()
    start = df.index.min()
    end = start + timedelta(days=days)
    return df.loc[start:end]


# ─────────────────────────────────────────────────────────────
# 2 · Optuna helpers
# ─────────────────────────────────────────────────────────────


class TQDMProgressBar:
    def __init__(self, total: int) -> None:
        self._bar = tqdm(total=total, desc="Trials")

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        self._bar.update(1)


def suggest_detector_params(trial: optuna.trial.Trial) -> dict[str, float | int]:
    """Return a parameter dict that matches ArbitrageDetector.a_REQUIRED."""

    params = {
        # Energy scales
        "objective_scale": trial.suggest_float("objective_scale", 0.5, 2.0, log=True),
        "flow_penalty_mul": trial.suggest_float(
            "flow_penalty_mul", 1e-4, 2e-3, log=True
        ),
        "cycle_penalty_mul": trial.suggest_float("cycle_penalty_mul", 0.15, 0.40),
        # Schedule
        "transverse_field_start": trial.suggest_float(
            "transverse_field_start", 5.0, 10.0
        ),
        "transverse_field_end": trial.suggest_float(
            "transverse_field_end", 0.05, 0.20, log=True
        ),
        "beta_start": trial.suggest_float("beta_start", 0.05, 1.0),
        "beta_end": trial.suggest_float("beta_end", 1.0, 3.0),
        # Discrete knobs
        "num_reads": trial.suggest_categorical("num_reads", [100, 250, 500]),
        "num_trotter_slices": trial.suggest_categorical(
            "num_trotter_slices", [16, 32, 64]
        ),
        "min_cycle_length": 3,
        "max_cycle_length": trial.suggest_int("max_cycle_length", 5, 6),
        # Fixed per‑trial seed for reproducibility
        "seed": trial.number,
    }
    return params


# ─────────────────────────────────────────────────────────────
# 3 · Objective function
# ─────────────────────────────────────────────────────────────


def objective(trial: optuna.trial.Trial) -> float:
    global WEEK_DATA

    if WEEK_DATA is None:
        WEEK_DATA = load_window(DATA_CSV)

    params = suggest_detector_params(trial)

    # Fidelity ladder
    for sweeps in FIDELITY_STEPS:
        params["num_sweeps"] = sweeps
        det = ArbitrageDetector(params)
        profit_sum = 0.0
        ts_cnt = 0
        stall = 0

        for _, row in WEEK_DATA.head(FIDELITY_SAMPLE).iterrows():
            det.add_market_data(row.dropna().to_dict())
            profit = sum(op["profit"] for op in det.find_arbitrage())
            profit_sum += profit
            ts_cnt += 1
            stall = 0 if profit else stall + 1
            if stall >= STALL_LIMIT:
                break

        avg_profit = profit_sum / ts_cnt if ts_cnt else 0.0
        trial.report(avg_profit, step=sweeps)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Full‑window evaluation
    params["num_sweeps"] = FIDELITY_STEPS[-1]
    det = ArbitrageDetector(params)
    prof_tot = 0.0
    ts_tot = 0
    stall = 0

    for _, row in WEEK_DATA.iterrows():
        det.add_market_data(row.dropna().to_dict())
        profit = sum(op["profit"] for op in det.find_arbitrage())
        prof_tot += profit
        ts_tot += 1
        stall = 0 if profit else stall + 1
        if stall >= STALL_LIMIT:
            break

    trial.set_user_attr("total_profit", prof_tot)
    trial.set_user_attr("num_timestamps", ts_tot)

    return prof_tot / ts_tot if ts_tot else 0.0


# ─────────────────────────────────────────────────────────────
# 4 · Study runner
# ─────────────────────────────────────────────────────────────


def main() -> None:
    storage = JournalStorage(
        JournalFileBackend(str(RESULTS_DIR / "study_round2.jsonl"))
    )

    study = optuna.create_study(
        direction="maximize",
        study_name="fx_adaptive_tuning_round2",
        storage=storage,
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, seed=42),
        load_if_exists=True,
    )

    pbar = TQDMProgressBar(total=N_TRIALS)
    n_jobs = max(1, mp.cpu_count() - 1)
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=n_jobs,
        callbacks=[pbar],
        show_progress_bar=False,
    )

    best = study.best_trial
    print("Best avg profit:", best.value)
    print("Best params:", best.params)
    print("User attrs:", best.user_attrs)


if __name__ == "__main__":
    main()
