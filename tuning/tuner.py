# coding: utf-8
"""
Hyper‑parameter tuning for the adaptive FX‑arbitrage detector using **Optuna**.
First global search space for initial exploration.

A fidelity‑based pruning scheme evaluates three ``num_sweeps`` levels before
committing to the full data window.
"""
from __future__ import annotations

import multiprocessing as mp
from datetime import timedelta
from pathlib import Path
from typing import Dict, Any
import os
import sys

import optuna
import pandas as pd
from optuna.pruners import SuccessiveHalvingPruner
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from main import ArbitrageDetector, MarketDataManager
except ImportError as e:
    raise ImportError(
        "Could not import required classes from 'main.py'. "
        "Ensure 'main.py' is in the same directory or your PYTHONPATH."
        f" Original error: {e}"
    )

from tqdm.auto import tqdm


# ────────────────────────────────────────────────────────────────────────────
# 0.  Helper
# ────────────────────────────────────────────────────────────────────────────


class TQDMProgressBar:
    def __init__(self, total: int) -> None:
        self._bar = tqdm(total=total, desc="Trials")

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        # advance by one each time a trial completes
        self._bar.update(1)


# ────────────────────────────────────────────────────────────────────────────
# 0.  Paths & constants
# ────────────────────────────────────────────────────────────────────────────

# --- PATH DEFINITION FOR CSV ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_CSV: Path = PROJECT_ROOT / "data" / "fx_data_march.csv"

RESULTS_DIR = Path("tuning_output/round1")  # REPO/tuning/tuning_output/round1/
RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # create if not exist

# --- CONSTANTS ---
N_TRIALS = 40
WINDOW_DAYS = 2
FIDELITY_SAMPLE = 50

# ────────────────────────────────────────────────────────────────────────────
# 1.  Data helper -----------------------------------------------------------
# ────────────────────────────────────────────────────────────────────────────


def load_window(csv_path: Path, days: int = WINDOW_DAYS) -> pd.DataFrame:
    df = MarketDataManager(str(csv_path)).fetch()
    start = df.index.min()
    end = start + timedelta(days=days)
    return df.loc[start:end]


WEEK_DATA: pd.DataFrame | None = None  # global cache

# ────────────────────────────────────────────────────────────────────────────
# 2.  Optuna objective ------------------------------------------------------
# ────────────────────────────────────────────────────────────────────────────


def suggest_detector_params(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Return a full parameter dict for *ArbitrageDetector* compatible with SmartQUBO."""

    # ‑‑ core QUBO scale ------------------------------------------------
    obj_scale = trial.suggest_float("objective_scale", 1.0, 10_000.0, log=True)

    # ‑‑ flow penalty branch -------------------------------------------
    loose = trial.suggest_categorical("loose_constraints", [False, True])
    if loose:
        flow_mul = trial.suggest_float("flow_penalty_mul_loose", 0.001, 0.1, log=True)
    else:
        flow_mul = trial.suggest_float("flow_penalty_mul", 0.1, 2.0, log=True)

    params: Dict[str, Any] = {
        "objective_scale": obj_scale,
        "flow_penalty_mul": flow_mul,
        "cycle_penalty_mul": trial.suggest_float("cycle_penalty_mul", 0.0, 0.5),
        # cycle length constraints
        "min_cycle_length": trial.suggest_categorical("min_cycle", [3, 4]),
        "max_cycle_length": trial.suggest_categorical("max_cycle", [4, 5, 6, 7, 8]),
        # SQA schedule --------------------------------------------------
        "num_reads": trial.suggest_categorical("num_reads", [100, 500, 1000]),
        "num_trotter_slices": trial.suggest_categorical("trotter", [8, 16, 32, 64]),
        "transverse_field_start": trial.suggest_float(
            "gamma_start", 1.0, 10.0, log=True
        ),
        "transverse_field_end": trial.suggest_float("gamma_end", 0.005, 0.5, log=True),
        "beta_start": trial.suggest_float("beta_start", 0.001, 1.0, log=True),
        "beta_end": trial.suggest_float("beta_end", 1.0, 20.0, log=True),
        "seed": trial.number,  # deterministic per trial
    }
    if params["max_cycle_length"] < params["min_cycle_length"]:
        params["max_cycle_length"] = params["min_cycle_length"]
    return params


FIDELITY_STEPS = [100, 250, 500]  # num_sweeps levels


def objective(trial: optuna.trial.Trial) -> float:
    global WEEK_DATA
    if WEEK_DATA is None:
        WEEK_DATA = load_window(DATA_CSV)  # DATA_CSV is now correctly defined

    params = suggest_detector_params(trial)

    # ‑‑ fidelity loop --------------------------------------------------
    for sweeps in FIDELITY_STEPS:
        params["num_sweeps"] = sweeps
        det = ArbitrageDetector(params)
        profit_sum, ts_cnt = 0.0, 0
        for _, row in WEEK_DATA.head(FIDELITY_SAMPLE).iterrows():
            det.add_market_data(row.dropna().to_dict())
            profit_sum += sum(op["profit"] for op in det.find_arbitrage())
            ts_cnt += 1
        avg_profit = profit_sum / ts_cnt if ts_cnt else 0.0
        trial.report(avg_profit, step=sweeps)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # ‑‑ final full‑window evaluation ----------------------------------
    # The last value from the fidelity loop is implicitly the one for the highest fidelity
    # on the FIDELITY_SAMPLE data.
    # The original code then re-evaluates on the *full* WEEK_DATA with the highest fidelity.
    params["num_sweeps"] = FIDELITY_STEPS[-1]  # Ensure highest fidelity for full window
    det = ArbitrageDetector(params)  # Re-initialize detector for full window eval
    prof_tot, ts_tot = 0.0, 0
    for _, row in WEEK_DATA.iterrows():  # Iterate over the full WEEK_DATA
        det.add_market_data(row.dropna().to_dict())
        prof_tot += sum(op["profit"] for op in det.find_arbitrage())
        ts_tot += 1

    trial.set_user_attr("total_profit", prof_tot)
    trial.set_user_attr("num_timestamps", ts_tot)
    return prof_tot / ts_tot if ts_tot else 0.0


# ────────────────────────────────────────────────────────────────────────────
# 3.  Run study -------------------------------------------------------------
# ────────────────────────────────────────────────────────────────────────────


def main() -> None:

    pruner = SuccessiveHalvingPruner(min_resource=FIDELITY_STEPS[0], reduction_factor=3)

    storage = JournalStorage(
        JournalFileBackend(str(RESULTS_DIR / "study_round1.jsonl"))
    )

    study = optuna.create_study(
        direction="maximize",
        study_name="fx_adaptive_tuning",  # Original name
        storage=storage,
        pruner=pruner,
        # load_if_exists=False, # Original was False. If you want to resume, set to True.
        # For strict adherence to original, keeping it False.
        load_if_exists=False,
        sampler=optuna.samplers.TPESampler(
            seed=42
        ),  # Added seed to TPESampler for reproducibility as per best practice
    )

    pbar = TQDMProgressBar(total=N_TRIALS)

    n_jobs = max(1, mp.cpu_count() - 1)
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=n_jobs,
        callbacks=[pbar],
        show_progress_bar=False,  # Original setting
    )

    # ‑‑ save CSV summary ----------------------------------------------
    # This part remains as per your original logic
    trial_df = study.trials_dataframe(
        attrs=(
            "number",
            "value",
            "params",
            "user_attrs",
            "datetime_start",
            "datetime_complete",
            "state",  # Added state as it's often useful
        )
    )
    trial_df.to_csv(RESULTS_DIR / "tuning_trials.csv", index=False)  # Original filename

    best = study.best_trial
    summary = {
        **best.params,
        "avg_profit": best.value,
        **best.user_attrs,
        "trial_number": best.number,
        "state": (
            best.trial.state.name if hasattr(best.trial, "state") else "UNKNOWN"
        ),  # Get state name
    }
    pd.DataFrame([summary]).to_csv(
        RESULTS_DIR / "tuning_summary.csv", index=False
    )  # Original filename
    print("Best trial:", summary)


if __name__ == "__main__":
    main()
