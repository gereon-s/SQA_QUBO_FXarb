#!/usr/bin/env python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval

# --- Configuration ---
SQA_RESULTS_CSV = "../../oos_test_march/output/sqa_results_march.csv"
BF_RESULTS_CSV = "../../oos_test_march/output/bf_results_march.csv"


# --- Helper Functions ---
def safe_literal_eval(val):
    try:
        # Ensure that val is string before passing to literal_eval
        if pd.isna(val):  # Handle NaN before it becomes a float/string
            return []
        if not isinstance(val, str):
            val = str(
                val
            )  # Convert if it's some other type that might be a list string
        return literal_eval(val)
    except (ValueError, SyntaxError, TypeError):
        return []


def analyze_results():
    try:
        df_sqa = pd.read_csv(SQA_RESULTS_CSV)
        df_bf = pd.read_csv(BF_RESULTS_CSV)
    except FileNotFoundError as e:
        print(f"Error: One or both results files not found. {e}")
        print(
            "Please generate 'sqa_results_standardized.csv' and 'bf_results_standardized.csv' first."
        )
        return

    print("--- Data Loading and Initial Processing (Standardized Format) ---")

    df_all = pd.concat([df_sqa, df_bf], ignore_index=True)

    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
    # Standardize 'cycle_nodes' to 'cycle_nodes_list' for processing
    df_all["cycle_nodes_list"] = (
        df_all["cycle_nodes"].fillna('"[]"').apply(safe_literal_eval)
    )
    df_all["cycle_length"] = pd.to_numeric(
        df_all["cycle_length"], errors="coerce"
    ).fillna(0)
    df_all["profit_pct"] = pd.to_numeric(df_all["profit_pct"], errors="coerce").fillna(
        0
    )
    df_all["step_runtime_sec"] = pd.to_numeric(
        df_all["step_runtime_sec"], errors="coerce"
    ).fillna(0)
    df_all["num_nodes_in_graph"] = pd.to_numeric(
        df_all["num_nodes_in_graph"], errors="coerce"
    ).fillna(0)
    df_all["num_edges_in_graph"] = pd.to_numeric(
        df_all["num_edges_in_graph"], errors="coerce"
    ).fillna(0)

    print(f"Loaded {len(df_sqa)} SQA records and {len(df_bf)} BF records.")
    print(f"Total records in combined DataFrame: {len(df_all)}")

    print("\n--- Calculating Key Metrics (from combined DataFrame) ---")

    total_unique_timestamps_in_holdout = df_all["timestamp"].nunique()
    print(
        f"Total unique timestamps in holdout data: {total_unique_timestamps_in_holdout}"
    )

    summary_metrics_list = []  # Use list of dicts for DataFrame creation
    for method_name, group_df in df_all.groupby("method"):
        # Total profit is sum of all individual cycle profits for this method
        total_profit_for_method = group_df["profit_pct"].sum()

        # Timestamps where this method found ANY profit > 0
        # Filter group_df first, then count unique timestamps
        profitable_ops_for_method = group_df[
            group_df["profit_pct"] > 1e-9
        ]  # Use small threshold
        num_ts_with_profit_for_method = profitable_ops_for_method["timestamp"].nunique()

        detection_rate_for_method = (
            (num_ts_with_profit_for_method / total_unique_timestamps_in_holdout) * 100
            if total_unique_timestamps_in_holdout > 0
            else 0
        )
        avg_profit_overall_for_method = (
            total_profit_for_method / total_unique_timestamps_in_holdout
            if total_unique_timestamps_in_holdout > 0
            else 0
        )
        avg_profit_per_active_ts_for_method = (
            total_profit_for_method / num_ts_with_profit_for_method
            if num_ts_with_profit_for_method > 0
            else 0
        )

        avg_profit_per_single_cycle_for_method = (
            profitable_ops_for_method["profit_pct"].mean()
            if not profitable_ops_for_method.empty
            else 0
        )

        unique_runtimes_df = group_df.drop_duplicates(subset=["timestamp"])
        avg_runtime_per_step_for_method = unique_runtimes_df["step_runtime_sec"].mean()
        total_runtime_for_method = unique_runtimes_df["step_runtime_sec"].sum()

        avg_cycle_len_profitable_for_method = (
            profitable_ops_for_method["cycle_length"].mean()
            if not profitable_ops_for_method.empty
            else 0
        )

        summary_metrics_list.append(
            {
                "Method": method_name,
                "Total Timestamps in Window": total_unique_timestamps_in_holdout,
                "Timestamps with Any Profitable Opp.": num_ts_with_profit_for_method,
                "Detection Rate (%)": f"{detection_rate_for_method:.2f}",
                "Total Profit (Sum all cycles) (%)": f"{total_profit_for_method:.6f}",
                "Avg. Profit / Window TS (Overall) (%)": f"{avg_profit_overall_for_method:.6f}",
                "Avg. Profit / Active TS (When found) (%)": f"{avg_profit_per_active_ts_for_method:.6f}",
                "Avg. Profit / Single Profitable Cycle (%)": (
                    f"{avg_profit_per_single_cycle_for_method:.6f}"
                    if not np.isnan(avg_profit_per_single_cycle_for_method)
                    else "N/A"
                ),
                "Total Execution Time (s)": f"{total_runtime_for_method:.2f}",
                "Avg. Execution Time / TS (s)": f"{avg_runtime_per_step_for_method:.6f}",
                "Avg. Profitable Cycle Length (hops)": (
                    f"{avg_cycle_len_profitable_for_method:.2f}"
                    if not np.isnan(avg_cycle_len_profitable_for_method)
                    else "N/A"
                ),
            }
        )

    df_summary = pd.DataFrame(summary_metrics_list)
    print("\n--- Overall Performance Summary ---")
    print(
        df_summary.to_string(index=False)
    )  # Print without method as index for clarity

    # --- 5. Statistical Tests ---
    print("\n--- Statistical Tests ---")
    sqa_cycle_profits = df_all[
        (df_all["method"] == "SQA") & (df_all["profit_pct"] > 1e-9)
    ]["profit_pct"]
    bf_cycle_profits = df_all[
        (df_all["method"] == "BF") & (df_all["profit_pct"] > 1e-9)
    ]["profit_pct"]

    if not sqa_cycle_profits.empty and not bf_cycle_profits.empty:
        try:
            u_stat, p_val_profit_dist = stats.mannwhitneyu(
                sqa_cycle_profits,
                bf_cycle_profits,
                alternative="two-sided",
                nan_policy="omit",
            )
            print(
                f"Mann-Whitney U for INDIVIDUAL cycle profit distributions (profit > 1e-9): U={u_stat:.2f}, p-value={p_val_profit_dist:.4f}"
            )
            if p_val_profit_dist < 0.05:
                print(
                    "  Suggests a significant difference in individual cycle profit distributions."
                )
            else:
                print(
                    "  No significant difference detected in individual cycle profit distributions."
                )
        except ValueError as e:
            print(f"  Could not perform Mann-Whitney U for profits: {e}")
    else:
        print("  Not enough data for profit distribution comparison.")

    sqa_step_runtimes = (
        df_all[df_all["method"] == "SQA"]
        .drop_duplicates(subset=["timestamp"])["step_runtime_sec"]
        .dropna()
    )
    bf_step_runtimes = (
        df_all[df_all["method"] == "BF"]
        .drop_duplicates(subset=["timestamp"])["step_runtime_sec"]
        .dropna()
    )

    if not sqa_step_runtimes.empty and not bf_step_runtimes.empty:
        try:
            u_stat_rt, p_val_rt_dist = stats.mannwhitneyu(
                sqa_step_runtimes,
                bf_step_runtimes,
                alternative="two-sided",
                nan_policy="omit",
            )
            print(
                f"Mann-Whitney U for per-step runtime distributions: U={u_stat_rt:.2f}, p-value={p_val_rt_dist:.4f}"
            )
            if p_val_rt_dist < 0.05:
                print("  Suggests a significant difference in runtime distributions.")
                print(
                    f"  Median SQA runtime: {sqa_step_runtimes.median():.6f}s, Median BF runtime: {bf_step_runtimes.median():.6f}s"
                )
            else:
                print("  No significant difference detected in runtime distributions.")
        except ValueError as e:
            print(f"  Could not perform Mann-Whitney U for runtimes: {e}")
    else:
        print("  Not enough data for runtime distribution comparison.")

    # Detection Rate (Timestamps with ANY profit > 1e-9)
    sqa_ts_w_profit_count = df_all[
        (df_all["method"] == "SQA") & (df_all["profit_pct"] > 1e-9)
    ]["timestamp"].nunique()
    bf_ts_w_profit_count = df_all[
        (df_all["method"] == "BF") & (df_all["profit_pct"] > 1e-9)
    ]["timestamp"].nunique()

    sqa_no_profit_ts_count = total_unique_timestamps_in_holdout - sqa_ts_w_profit_count
    bf_no_profit_ts_count = total_unique_timestamps_in_holdout - bf_ts_w_profit_count

    if total_unique_timestamps_in_holdout > 0:
        contingency_table = np.array(
            [
                [sqa_ts_w_profit_count, sqa_no_profit_ts_count],
                [bf_ts_w_profit_count, bf_no_profit_ts_count],
            ]
        )
        print(
            "\nContingency Table for Detection Rates (timestamps with any profit > 1e-9):"
        )
        print(
            f"        Found | No Opp (Total TS: {total_unique_timestamps_in_holdout})"
        )
        print(f"SQA:    {sqa_ts_w_profit_count:5d} | {sqa_no_profit_ts_count:5d}")
        print(f"BF :    {bf_ts_w_profit_count:5d} | {bf_no_profit_ts_count:5d}")
        try:
            odds_ratio, p_val_detection = stats.fisher_exact(contingency_table)
            print(
                f"Fisher's Exact Test for detection rates: Odds Ratio={odds_ratio:.2f}, p-value={p_val_detection:.4f}"
            )
            if p_val_detection < 0.05:
                print("  Suggests a significant difference in detection rates.")
            else:
                print("  No significant difference detected in detection rates.")
        except ValueError as e:
            print(f"  Could not perform Fisher's exact test: {e}")
    else:
        print("  Not enough timestamp data for detection rate test.")

    # --- 6. Visualizations ---
    print("\n--- Generating Visualizations (will save to files) ---")
    plt.style.use("seaborn-v0_8-whitegrid")

    df_plot_profitable = df_all[
        df_all["profit_pct"] > 1e-9
    ]  # Use a small threshold for plotting

    if not df_plot_profitable.empty:
        plt.figure(figsize=(12, 6))
        sns.histplot(
            data=df_plot_profitable,
            x="profit_pct",
            hue="method",
            kde=True,
            stat="density",
            element="step",
            common_norm=False,
        )
        plt.title("Distribution of Individual Profitable Arbitrage Cycles")
        plt.xlabel("Profit Percentage (%)")
        plt.savefig("plots/std_individual_cycle_profit_distribution.png")
        plt.close()
        print("Saved std_individual_cycle_profit_distribution.png")

        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df_plot_profitable, x="method", y="profit_pct")
        plt.title("Box Plot of Individual Profitable Arbitrage Cycles")
        plt.ylabel("Profit Percentage (%)")
        plt.savefig("plots/std_individual_cycle_profit_boxplot.png")
        plt.close()
        print("Saved std_individual_cycle_profit_boxplot.png")
    else:
        print("No profitable operations to plot for profit distributions.")

    df_runtimes_unique_step = df_all.drop_duplicates(subset=["timestamp", "method"])
    if not df_runtimes_unique_step.empty:
        plt.figure(figsize=(12, 6))
        sns.histplot(
            data=df_runtimes_unique_step,
            x="step_runtime_sec",
            hue="method",
            kde=True,
            stat="density",
            element="step",
            log_scale=(True, False),
            common_norm=False,
        )
        plt.title("Distribution of Per-Timestamp Runtimes (X-axis Log Scale)")
        plt.xlabel("Runtime (seconds)")
        plt.savefig("plots/std_runtime_distribution_per_step.png")
        plt.close()
        print("Saved std_runtime_distribution_per_step.png")

        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df_runtimes_unique_step, x="method", y="step_runtime_sec")
        plt.title("Box Plot of Per-Timestamp Runtimes")
        plt.ylabel("Runtime (seconds)")
        plt.yscale("log")
        plt.savefig("plots/std_runtime_boxplot_logscale_per_step.png")
        plt.close()
        print("Saved std_runtime_boxplot_logscale_per_step.png")
    else:
        print("No runtime data to plot.")

    if not df_plot_profitable.empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=df_plot_profitable[df_plot_profitable["cycle_length"] > 0],
            x="cycle_length",
            hue="method",
            discrete=True,
            stat="density",
            shrink=0.8,
            multiple="dodge",
            common_norm=False,
        )
        plt.title("Distribution of Profitable Cycle Lengths (Hops)")

        min_len_all = df_plot_profitable[df_plot_profitable["cycle_length"] > 0][
            "cycle_length"
        ].min()
        max_len_all = df_plot_profitable[df_plot_profitable["cycle_length"] > 0][
            "cycle_length"
        ].max()
        if (
            pd.notna(min_len_all)
            and pd.notna(max_len_all)
            and max_len_all >= min_len_all
        ):
            plt.xticks(np.arange(int(min_len_all), int(max_len_all) + 1))
        plt.xlabel("Number of Hops in Cycle")
        plt.savefig("plots/std_all_cycle_length_distribution.png")
        plt.close()
        print("Saved std_all_cycle_length_distribution.png")
    else:
        print("No profitable operations to plot for cycle lengths.")

    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    analyze_results()
