# coding: utf-8
"""
Tuner with **time‑series cross‑validation** for the FX adaptive arbitrage detector.

Key points
──────────
* Rolling‑window CV (2‑day window, 1‑day step, default 6 folds).
* Same fidelity ladder, early‑stall exit, and deterministic `seed` as in round‑2.
* Reduced to 60 trials by default because each trial is ~6× heavier.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import timedelta
import multiprocessing as mp

import numpy as np
import optuna
from optuna.storages import JournalStorage
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

RESULTS_DIR = Path("tuning_output/CV")  # REPO/tuning/tuning_output/round1/
RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # create if not exist


# --- CONFIG ---
WINDOW_DAYS: int = 3  # length of each fold window
STEP_DAYS: int = 2  # shift between folds
N_FOLDS: int = 6  # how many folds per trial (rolling)

FIDELITY_STEPS: list[int] = [4, 8, 16, 32]
FIDELITY_SAMPLE: int = 500
STALL_LIMIT: int = 200

N_TRIALS: int = 60

# Data cache shared by worker threads
DATA_DF = None  # type: ignore
FOLD_STARTS: list[pd.Timestamp] = []

# ─────────────────────────────────────────────────────────────
# 1 · Data helpers
# ─────────────────────────────────────────────────────────────


def load_full_data(csv_path: Path) -> pd.DataFrame:
    mgr = MarketDataManager(str(csv_path))
    return mgr.fetch()


def prepare_folds(df: pd.DataFrame) -> list[pd.Timestamp]:
    """Return fold start timestamps beginning at day‑1 and stepping forward.

    Ensures all folds lie **before** the hold‑out split (day‑16)."""
    starts: list[pd.Timestamp] = []
    cur = df.index.min()  # day‑1
    # add folds until we either hit N_FOLDS or overlap day‑16
    while len(starts) < N_FOLDS and cur + timedelta(
        days=WINDOW_DAYS
    ) <= df.index.min() + timedelta(days=15):
        starts.append(cur)
        cur += timedelta(days=STEP_DAYS)
    return starts


# ─────────────────────────────────────────────────────────────
# 2 · Optuna utilities
# ─────────────────────────────────────────────────────────────


class TQDMProgressBar:
    def __init__(self, total: int) -> None:
        self._bar = tqdm(total=total, desc="Trials")

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        self._bar.update(1)


def suggest_detector_params(trial: optuna.trial.Trial) -> dict[str, float | int]:
    """Return params matching ArbitrageDetector.a_REQUIRED."""
    return {
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
        # Reproducibility
        "seed": trial.number,
    }


# ─────────────────────────────────────────────────────────────
# 3 · Objective
# ─────────────────────────────────────────────────────────────


def run_on_window(params: dict, window: pd.DataFrame) -> float:
    """Return average profit over a given window at highest fidelity."""
    params = params.copy()
    params["num_sweeps"] = FIDELITY_STEPS[-1]
    det = ArbitrageDetector(params)

    prof, cnt, stall = 0.0, 0, 0
    for _, row in window.iterrows():
        det.add_market_data(row.dropna().to_dict())
        p = sum(op["profit"] for op in det.find_arbitrage())
        prof += p
        cnt += 1
        stall = 0 if p else stall + 1
        if stall >= STALL_LIMIT:
            break
    return prof / cnt if cnt else 0.0


def objective(trial: optuna.trial.Trial) -> float:
    global DATA_DF, FOLD_STARTS

    # lazy-load data and folds
    if DATA_DF is None:
        DATA_DF = load_full_data(DATA_CSV)
        FOLD_STARTS = prepare_folds(DATA_DF)

    params = suggest_detector_params(trial)
    fold_scores: list[float] = []

    for fold_idx, start in enumerate(FOLD_STARTS, 1):
        window = DATA_DF.loc[start : start + timedelta(days=WINDOW_DAYS)]

        # fidelity ladder (cheap first passes)
        for sweeps in FIDELITY_STEPS:
            params["num_sweeps"] = sweeps
            det = ArbitrageDetector(params)
            psum, cnt, stall = 0.0, 0, 0
            for _, row in window.head(FIDELITY_SAMPLE).iterrows():
                det.add_market_data(row.dropna().to_dict())
                p = sum(op["profit"] for op in det.find_arbitrage())
                psum += p
                cnt += 1
                stall = 0 if p else stall + 1
                if stall >= STALL_LIMIT:
                    break

        # full window at max sweeps
        fold_scores.append(run_on_window(params, window))
        trial.report(np.mean(fold_scores), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_scores))


def evaluate_on_holdout(params: dict, df: pd.DataFrame, start_day: int = 16) -> float:
    window = df.loc[df.index.min() + timedelta(days=start_day - 1) : df.index.max()]
    return run_on_window(params, window)


# ─────────────────────────────────────────────────────────────
# 4 · Study runner
# ─────────────────────────────────────────────────────────────


def main() -> None:
    storage = JournalStorage(JournalFileBackend(str(RESULTS_DIR / "CV_study.jsonl")))

    study = optuna.create_study(
        direction="maximize",
        study_name="fx_adaptive_tuning_cv",
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

    holdout = evaluate_on_holdout(best.params, DATA_DF)
    print("Best CV profit:", best.value)
    print("Holdout profit:", holdout)
    print("Best trial number:", best.number)
    print("Best params:", best.params)
    print("User attrs:", best.user_attrs)


if __name__ == "__main__":
    main()
