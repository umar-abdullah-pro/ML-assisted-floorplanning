"""Comprehensive Comparative Study Runner for the 4-way Evaluation Plan:
1. Baseline (Fixed Shapes, Fixed Step)
2. SDS Shaping Only (Fixed Step)
3. Q-Learning Only (Fixed Shapes)
4. Proposed Combined (Q-Learning + SDS Shaping)

Evaluates on multiple MCNC benchmarks across multiple seeds, computes metrics
(Height, Wirelength HPWL, Legality, Runtime), and saves publication-quality summary charts.
"""

from __future__ import annotations
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks.mcnc_benchmarks import build_mcnc_benchmark, MCNC_SPECS
from q_learning.experiment_runner import compare_all_methods
from q_learning.placement_search import SearchMode


def run_comparative_benchmark_suite(
    benchmark_names: list[str] = ["apte", "xerox", "hp"],
    iterations_per_run: int = 80,
    seeds: list[int] = [42, 123],
    output_dir: str = "output/comparative_study",
) -> pd.DataFrame:
    """Run the 4-way comparative study on a suite of benchmarks and aggregate results."""
    os.makedirs(output_dir, exist_ok=True)
    all_rows = []

    print("=" * 90)
    print("STARTING 4-WAY COMPARATIVE STUDY ACROSS BENCHMARKS")
    print("Methods: [1] Baseline | [2] SDS Shaping Only | [3] Q-Learning Only | [4] Proposed (Both)")
    print("=" * 90)

    for bench in benchmark_names:
        print(f"\nEvaluating Benchmark: {bench.upper()}...")
        spec = MCNC_SPECS[bench]
        outline_w = spec["W_table2"]
        outline_h = outline_w * 1.20  # standard fixed-outline aspect ratio

        for seed in seeds:
            blocks, gh, gv, W = build_mcnc_benchmark(bench, target_experiment="table2", seed=seed)
            # Standard netlist connecting neighboring blocks in circuit
            nets = [[i, (i % len(blocks)) + 1] for i in range(1, len(blocks))]
            if len(blocks) >= 4:
                nets.append([1, len(blocks) // 2, len(blocks)])

            exp_res = compare_all_methods(
                benchmark_name=bench,
                blocks=blocks,
                nets=nets,
                outline_width=outline_w,
                outline_height=outline_h,
                max_iterations=iterations_per_run,
                seed=seed,
                verbose=False,
            )

            for row in exp_res.summary_table:
                row_copy = dict(row)
                row_copy["Circuit"] = bench.upper()
                row_copy["Seed"] = seed
                all_rows.append(row_copy)

    df = pd.DataFrame(all_rows)

    # Reorder columns
    cols = ["Circuit", "Seed", "Method", "Legal", "Height", "Width", "HPWL", "Cost", "Legal Steps", "Time (s)"]
    df = df[cols]

    # Save to CSV
    csv_path = os.path.join(output_dir, "comparative_study_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw results table to {csv_path}")

    # Compute Averages Grouped by Circuit and Method
    agg_df = df.groupby(["Circuit", "Method"])[["Height", "HPWL", "Cost", "Time (s)"]].mean().reset_index()

    print("\n" + "=" * 90)
    print("AGGREGATED SUMMARY TABLE (AVERAGED OVER SEEDS)")
    print("=" * 90)
    print(agg_df.to_string(index=False))

    # Generate Comparative Bar Charts
    plot_comparative_charts(agg_df, output_dir)
    return df


def plot_comparative_charts(agg_df: pd.DataFrame, output_dir: str) -> None:
    """Generate side-by-side grouped bar charts for Height and HPWL."""
    circuits = agg_df["Circuit"].unique()
    methods = [
        SearchMode.BASELINE.value,
        SearchMode.SHAPING_ONLY.value,
        SearchMode.Q_LEARNING_ONLY.value,
        SearchMode.COMBINED_PROPOSED.value,
    ]
    method_labels = ["Baseline", "SDS Shaping", "Q-Learning", "Proposed (Both)"]
    colors = ["#90A4AE", "#42A5F5", "#AB47BC", "#2E7D32"]

    # 1. Height Comparison Chart
    fig_h, ax_h = plt.subplots(figsize=(10, 5), dpi=150)
    x = np.arange(len(circuits))
    width = 0.20

    for idx, (m_val, m_label, col) in enumerate(zip(methods, method_labels, colors)):
        sub = agg_df[agg_df["Method"] == m_val]
        vals = [sub[sub["Circuit"] == c]["Height"].values[0] if len(sub[sub["Circuit"] == c]) > 0 else 0.0 for c in circuits]
        offset = (idx - 1.5) * width
        bars = ax_h.bar(x + offset, vals, width, label=m_label, color=col)
        # Value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax_h.annotate(f"{height:.0f}",
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3), textcoords="offset points",
                          ha="center", va="bottom", fontsize=7)

    ax_h.set_title("Layout Height Comparison Across Methods (Lower is Better)", fontsize=12, fontweight="bold", pad=12)
    ax_h.set_xticks(x)
    ax_h.set_xticklabels(circuits, fontweight="bold", fontsize=11)
    ax_h.set_ylabel("Final Layout Height", fontweight="bold")
    ax_h.grid(axis="y", linestyle=":", alpha=0.6)
    ax_h.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    chart_h_path = os.path.join(output_dir, "comparison_layout_height.png")
    plt.savefig(chart_h_path)
    plt.close(fig_h)

    # 2. HPWL Comparison Chart
    fig_w, ax_w = plt.subplots(figsize=(10, 5), dpi=150)
    for idx, (m_val, m_label, col) in enumerate(zip(methods, method_labels, colors)):
        sub = agg_df[agg_df["Method"] == m_val]
        vals = [sub[sub["Circuit"] == c]["HPWL"].values[0] if len(sub[sub["Circuit"] == c]) > 0 else 0.0 for c in circuits]
        offset = (idx - 1.5) * width
        bars = ax_w.bar(x + offset, vals, width, label=m_label, color=col)
        for bar in bars:
            hpwl = bar.get_height()
            ax_w.annotate(f"{hpwl:.0f}",
                          xy=(bar.get_x() + bar.get_width() / 2, hpwl),
                          xytext=(0, 3), textcoords="offset points",
                          ha="center", va="bottom", fontsize=7)

    ax_w.set_title("Wirelength (HPWL) Comparison Across Methods (Lower is Better)", fontsize=12, fontweight="bold", pad=12)
    ax_w.set_xticks(x)
    ax_w.set_xticklabels(circuits, fontweight="bold", fontsize=11)
    ax_w.set_ylabel("Half-Perimeter Wirelength (HPWL)", fontweight="bold")
    ax_w.grid(axis="y", linestyle=":", alpha=0.6)
    ax_w.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    chart_w_path = os.path.join(output_dir, "comparison_hpwl_wirelength.png")
    plt.savefig(chart_w_path)
    plt.close(fig_w)

    print(f"Saved charts to {chart_h_path} and {chart_w_path}")


if __name__ == "__main__":
    run_comparative_benchmark_suite()

