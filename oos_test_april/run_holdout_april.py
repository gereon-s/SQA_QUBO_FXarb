#!/usr/bin/env python
"""
Run the FX arbitrage detector (SQA) on the holdout dataset using champion parameters.
Outputs all SQA opportunities found at each timestamp to a standardized CSV format.
"""

from pathlib import Path
import time
import json
import sys

import sys
import os

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import ArbitrageDetector, MarketDataManager


# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_CSV = PROJECT_ROOT / "data" / "fx_data_april.csv"
PARAMS_JSON = PROJECT_ROOT / "tuning" / "tuning_output" / "CV" / "best_params_CV.json"
# WINDOW_START_DAY = 17 # No longer used, as entire dataset is processed
DETERMINISTIC_SEED = 42  # For reproducibility of SQA runs

SQA_RESULTS_CSV = Path("output/sqa_results_april.csv")


def main_script() -> None:
    # Load SQA parameters
    try:
        with open(PARAMS_JSON) as f:
            params = json.load(f)
    except FileNotFoundError:
        print(f"Error: Parameters file not found at {PARAMS_JSON}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {PARAMS_JSON}", file=sys.stderr)
        sys.exit(1)

    params["seed"] = DETERMINISTIC_SEED  # Override seed for deterministic run
    # Ensure num_sweeps is set to highest fidelity if it's in params
    if "num_sweeps" in params:
        params["num_sweeps"] = 32
    else:  # If not in params file, add it
        print(
            "Warning: 'num_sweeps' not in params file, setting to 32.", file=sys.stderr
        )
        params["num_sweeps"] = 32

    # Load market data
    try:
        df_manager = MarketDataManager(str(DATA_CSV))
        df = df_manager.fetch()
    except FileNotFoundError:
        print(f"Error: Data CSV file not found at {DATA_CSV}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading market data: {e}", file=sys.stderr)
        sys.exit(1)

    # Select data window
    if df.empty:
        print("Error: Loaded market data DataFrame is empty.", file=sys.stderr)
        sys.exit(1)

    window = df
    # The original logic for selecting a holdout window based on WINDOW_START_DAY
    # has been replaced by using the entire DataFrame.

    # Initialize detector
    try:
        det = ArbitrageDetector(params)
    except Exception as e:
        print(f"Error initializing ArbitrageDetector: {e}", file=sys.stderr)
        sys.exit(1)

    # For stderr summary
    grand_total_sqa_profit = 0.0
    timestamps_with_any_sqa_profit = 0
    processed_ts_count = 0

    print(
        f"Starting SQA run on entire dataset. Output will be saved to {SQA_RESULTS_CSV}"
    )

    with open(SQA_RESULTS_CSV, "w") as outfile:
        header_fields = [
            "timestamp",
            "method",
            "profit_pct",
            "cycle_nodes",
            "cycle_length",
            "step_runtime_sec",
            "num_nodes_in_graph",
            "num_edges_in_graph",
            "error_message",
        ]
        outfile.write(",".join(header_fields) + "\n")

        if window.empty:
            print("No data to process.", file=sys.stderr)
        else:
            for timestamp, row in window.iterrows():
                processed_ts_count += 1
                market_data_for_step = row.dropna().to_dict()

                # Add market data and get graph info for this step
                log_num_nodes = 0
                log_num_edges = 0
                try:
                    det.add_market_data(market_data_for_step)
                    # Assumes det.network.G structure to get graph info
                    if hasattr(det, "network") and hasattr(det.network, "G"):
                        current_graph = det.network.G
                        log_num_nodes = current_graph.number_of_nodes()
                        log_num_edges = current_graph.number_of_edges()
                    else:
                        print(
                            "Warning: Cannot access det.network.G to get graph size. Defaulting to 0.",
                            file=sys.stderr,
                        )
                except Exception as e:
                    print(
                        f"Error in add_market_data or getting graph info for {timestamp}: {e}",
                        file=sys.stderr,
                    )
                    # attempt to find arbitrage if possible, or log error and continue

                start_find_time = time.perf_counter()
                try:
                    opportunities = (
                        det.find_arbitrage()
                    )  # List of {"profit": X, "cycle": [...]}
                except Exception as e:
                    print(
                        f"Error in det.find_arbitrage() for {timestamp}: {e}",
                        file=sys.stderr,
                    )
                    opportunities = []  # Treat as no opportunities found on error
                sqa_runtime_sec_for_step = time.perf_counter() - start_find_time

                log_timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                log_error_str = ""  # Default error message for the CSV row

                current_step_had_profit_flag = False
                step_total_profit_console = 0.0  # For live console printing

                if opportunities:
                    for op in opportunities:
                        profit_value = op.get("profit", 0.0)
                        cycle_nodes_list = op.get("cycle", [])
                        cycle_len = (
                            len(cycle_nodes_list)
                            if isinstance(cycle_nodes_list, list) and cycle_nodes_list
                            else 0
                        )

                        grand_total_sqa_profit += profit_value
                        step_total_profit_console += profit_value

                        if (
                            profit_value > 1e-9
                        ):  # Using a small threshold for "profitable"
                            current_step_had_profit_flag = True

                        formatted_profit = f"{profit_value:.18g}"
                        formatted_runtime = f"{sqa_runtime_sec_for_step:.18g}"
                        nodes_str = str(
                            cycle_nodes_list
                        )  # Pandas will handle quoting if needed on read

                        csv_values = [
                            log_timestamp_str,
                            "SQA",
                            formatted_profit,
                            f'"{nodes_str}"',
                            str(cycle_len),
                            formatted_runtime,
                            str(log_num_nodes),
                            str(log_num_edges),
                            log_error_str,
                        ]
                        outfile.write(",".join(csv_values) + "\n")
                else:
                    # No SQA opportunity found, write a placeholder row
                    csv_values = [
                        log_timestamp_str,
                        "SQA",
                        "0.0",
                        '"[]"',
                        "0",
                        f"{sqa_runtime_sec_for_step:.18g}",
                        str(log_num_nodes),
                        str(log_num_edges),
                        "No SQA opportunity found",
                    ]
                    outfile.write(",".join(csv_values) + "\n")

                if current_step_had_profit_flag:
                    timestamps_with_any_sqa_profit += 1

                if (
                    processed_ts_count % 50 == 0 or processed_ts_count == 1
                ):  # Print progress
                    print(
                        f"Processed SQA for {log_timestamp_str} - Step Profit: {step_total_profit_console:.6f} - Runtime: {sqa_runtime_sec_for_step:.4f}s"
                    )

    avg_profit_overall_sqa = (
        grand_total_sqa_profit / processed_ts_count if processed_ts_count > 0 else 0.0
    )
    detection_rate_sqa = (
        (timestamps_with_any_sqa_profit / processed_ts_count) * 100
        if processed_ts_count > 0
        else 0
    )

    sys.stderr.write(f"\n# SQA Summary Stats (Standardized Output Run):\n")
    sys.stderr.write(f"Deterministic Seed Used: {DETERMINISTIC_SEED}\n")
    sys.stderr.write(
        f"Total SQA Profit (sum of all cycles): {grand_total_sqa_profit:.6f}\n"
    )
    sys.stderr.write(
        f"Average SQA Profit per Timestamp (Overall): {avg_profit_overall_sqa:.6f}\n"
    )
    sys.stderr.write(
        f"SQA Detection Rate (Timestamps with any profit > 0): {detection_rate_sqa:.2f}%\n"
    )
    sys.stderr.write(f"Number of Timestamps Processed: {processed_ts_count}\n")
    sys.stderr.write(f"Detailed SQA results saved to: {SQA_RESULTS_CSV}\n")


if __name__ == "__main__":
    main_script()
