"""
Benchmark Bellman–Ford (BF) for FX arbitrage detection on the holdout dataset.
Outputs results to a standardized CSV format, compatible with SQA output.
"""

from pathlib import Path
from datetime import timedelta
import time
import logging
import pandas as pd
import networkx as nx
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import MarketDataManager, ArbitrageNetwork

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_CSV = PROJECT_ROOT / "data" / "fx_data_april.csv"
# WINDOW_START_DAY = 17 # No longer used, as entire dataset is processed
TRANSACTION_COST_BPS = 0.0002  # 0.02% transaction cost
DEFAULT_SOURCE_NODE = "USD"  # For nx.find_negative_cycle

BF_RESULTS_CSV = Path("output/bf_results_april.csv")

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger("benchmark_bf_standardized")


def run_bellman_ford_standardized(
    window: pd.DataFrame, transaction_cost: float
) -> list[dict]:
    all_timestamp_results = []
    processed_count_log = 0  # For logging progress

    LOGGER.info(
        "Starting Bellman-Ford standardized run on %d timestamps...", len(window)
    )

    for ts, row in window.iterrows():
        processed_count_log += 1
        rates_for_step = row.dropna().to_dict()

        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        method_name = "BF"
        profit_val = 0.0
        cycle_nodes_str = '"[]"'
        cycle_len_val = 0
        runtime_val = 0.0
        error_val = ""
        current_num_nodes = 0
        current_num_edges = 0

        if not rates_for_step:
            error_val = "No rate data for timestamp"
            # Still append a row to keep timestamps consistent
            all_timestamp_results.append(
                {
                    "timestamp": ts_str,
                    "method": method_name,
                    "profit_pct": profit_val,
                    "cycle_nodes": cycle_nodes_str,
                    "cycle_length": cycle_len_val,
                    "step_runtime_sec": runtime_val,
                    "num_nodes_in_graph": current_num_nodes,
                    "num_edges_in_graph": current_num_edges,
                    "error_message": error_val,
                }
            )
            if processed_count_log % 50 == 0 or processed_count_log == 1:
                LOGGER.info(f"Processed BF for {ts_str} - No rates")
            continue

        try:
            network = ArbitrageNetwork()  # Fresh network for each timestamp
            tc_dict: dict[tuple[str, str], float] = {}
            all_currencies = set()
            valid_rates = {}

            # Pre-process rates and identify currencies
            for pair, rate_val in rates_for_step.items():
                try:

                    base, quote = pair.split("/")
                    if isinstance(rate_val, (int, float)) and rate_val > 1e-15:
                        all_currencies.add(base)
                        all_currencies.add(quote)
                        valid_rates[pair] = float(rate_val)
                except ValueError:
                    # LOGGER.warning("Skipping invalid pair format: %s at %s", pair, ts)
                    continue

            if not all_currencies:
                error_val = "No valid currency data after parsing"
            else:
                for base in all_currencies:
                    for quote in all_currencies:
                        if base != quote:
                            tc_dict[(base, quote)] = transaction_cost

                for pair, rate_val in valid_rates.items():
                    try:
                        base, quote = pair.split("/")
                        network.add_rate(base, quote, rate_val, tc_dict)
                    except Exception:  # Catch any error during add_rate
                        # LOGGER.warning("Add rate failed for %s at %s", pair, ts)
                        pass

                G = network.G
                current_num_nodes = G.number_of_nodes()
                current_num_edges = G.number_of_edges()

                if not G or not G.nodes:
                    error_val = "Empty graph after processing rates"
                else:
                    source_node = DEFAULT_SOURCE_NODE
                    if source_node not in G:
                        sorted_nodes_list = sorted(list(G.nodes()))
                        source_node = (
                            sorted_nodes_list[0] if sorted_nodes_list else None
                        )

                    if source_node:
                        start_time_bf_core = time.perf_counter()
                        found_cycle_nodes_raw = None
                        try:
                            found_cycle_nodes_raw = nx.find_negative_cycle(
                                G, source=source_node, weight="weight"
                            )
                        except nx.NetworkXError:  # No negative cycle
                            pass
                        except Exception as find_exc:
                            error_val = f"find_negative_cycle error: {find_exc}"
                        runtime_val = time.perf_counter() - start_time_bf_core

                        if found_cycle_nodes_raw:
                            is_valid_for_profit = (
                                isinstance(found_cycle_nodes_raw, list)
                                and len(found_cycle_nodes_raw) >= 4  # e.g. A-B-C-A
                                and found_cycle_nodes_raw[0]
                                == found_cycle_nodes_raw[-1]
                            )

                            if is_valid_for_profit:
                                cycle_for_profit_calc = found_cycle_nodes_raw[:-1]
                                try:
                                    temp_profit = network.cycle_profit_pct(
                                        cycle_for_profit_calc
                                    )
                                    if temp_profit > 1e-9:  # profitability threshold
                                        profit_val = temp_profit
                                        cycle_nodes_str = (
                                            f'"{str(cycle_for_profit_calc)}"'
                                        )
                                        cycle_len_val = len(cycle_for_profit_calc)
                                    else:  # Valid cycle but not profitable enough
                                        # profit_val remains 0.0
                                        cycle_nodes_str = f'"{str(cycle_for_profit_calc)}"'  # Log it anyway
                                        cycle_len_val = len(cycle_for_profit_calc)
                                        if not error_val:
                                            error_val = "BF cycle profit negligible"
                                except Exception as profit_exc:
                                    if not error_val:
                                        error_val = (
                                            f"BF Profit calc error: {profit_exc}"
                                        )
                                    profit_val = 0.0
                                    cycle_nodes_str = (
                                        f'"{str(found_cycle_nodes_raw)}"'  # Log raw
                                    )
                            elif not error_val:
                                error_val = f"BF reported invalid cycle: {found_cycle_nodes_raw}"
                    elif not error_val:
                        error_val = "No suitable source node for BF"
        except Exception as e:
            if not error_val:
                error_val = f"Outer processing error: {e}"
            # Attempt to get graph size even on error if G might exist
            if "G" in locals() and G is not None:
                current_num_nodes = G.number_of_nodes()
                current_num_edges = G.number_of_edges()

        all_timestamp_results.append(
            {
                "timestamp": ts_str,
                "method": method_name,
                "profit_pct": profit_val,
                "cycle_nodes": cycle_nodes_str,
                "cycle_length": cycle_len_val,
                "step_runtime_sec": runtime_val,
                "num_nodes_in_graph": current_num_nodes,
                "num_edges_in_graph": current_num_edges,
                "error_message": error_val,
            }
        )

        if processed_count_log % 50 == 0 or processed_count_log == 1:
            LOGGER.info(
                f"Processed BF for {ts_str} - Profit: {profit_val:.6f} - Runtime: {runtime_val:.6f}s"
            )

    LOGGER.info("Finished Bellman-Ford standardized run.")
    return all_timestamp_results


def main() -> None:
    LOGGER.info("Loading data from %s...", DATA_CSV)
    try:
        mgr = MarketDataManager(str(DATA_CSV))
        df = mgr.fetch()
    except Exception as e:
        LOGGER.error("Failed to load or parse data: %s", e, exc_info=True)
        return
    LOGGER.info("Data loaded successfully: %d timestamps.", len(df))

    # --- Process the entire dataset ---
    window = df
    LOGGER.info(
        "Processing entire dataset: %s to %s (%d ts)",
        window.index.min() if not window.empty else "N/A",
        window.index.max() if not window.empty else "N/A",
        len(window),
    )
    if window.empty:
        LOGGER.warning("Dataset is empty.")
        return
    # The original logic for selecting a holdout window based on WINDOW_START_DAY
    # has been replaced by using the entire DataFrame.

    list_of_result_dicts = run_bellman_ford_standardized(window, TRANSACTION_COST_BPS)

    if list_of_result_dicts:
        df_bf_standardized = pd.DataFrame(list_of_result_dicts)
        df_bf_standardized.to_csv(BF_RESULTS_CSV, index=False)
        LOGGER.info(f"Saved standardized BF results to {BF_RESULTS_CSV}")

        bf_total_profit_final = df_bf_standardized[
            df_bf_standardized["profit_pct"] > 0
        ][
            "profit_pct"
        ].sum()  # Make sure profit_pct is numeric for sum
        LOGGER.info(
            f"BF Final Total Profit (from standardized CSV, where profit > 0): {bf_total_profit_final:.6f}"
        )
    else:
        LOGGER.warning("No BF results generated for standardized CSV.")


if __name__ == "__main__":
    main()
