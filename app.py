"""Streamlit Web Application: Q-Learning-Assisted Slack-Driven Optimization for Fixed-Outline VLSI Floorplanning.
Combines:
1. Optimal Slack-Driven Block Shaping (SDS) [Yan & Chu, ISPD 2012]
2. Q-Learning Step-Size Controller [Meng et al., CSF 2025]
"""

from __future__ import annotations
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sds.block import Block
from sds.basic_sds import BasicSDSSolver
from sds.full_sds import FullSDSSolver
from sds.optimality import OptimalityChecker
from sds.visualizer import plot_floorplan, plot_convergence
from benchmarks.figure2_example import build_figure2_benchmark
from benchmarks.mcnc_benchmarks import build_mcnc_benchmark, MCNC_SPECS
from benchmarks.synthetic_benchmarks import generate_hb_style_benchmark
from q_learning.placement_search import (
    FixedOutlinePlacementOptimizer,
    OptimizerConfig,
    SearchMode,
)
from q_learning.experiment_runner import compare_all_methods


# Page configuration
st.set_page_config(
    page_title="VLSI Floorplanning: SDS & Q-Learning",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ Q-Learning Assisted Slack-Driven VLSI Floorplanning")
st.markdown(
    """
    **Dual-Paper Implementation & Comparative Platform:**
    * **Paper 1 (Shaping):** *Optimal Slack-Driven Block Shaping in Fixed-Outline Floorplanning* (Yan & Chu, ISPD 2012)
    * **Paper 2 (Controller):** *CSF: Fixed-Outline Floorplanning Based on Conjugate Subgradient Assisted by Q-Learning* (Meng et al., 2025)
    """
)

# Sidebar
st.sidebar.header("🔧 Benchmark & Design Setup")

benchmark_choice = st.sidebar.selectbox(
    "Select Benchmark Circuit",
    [
        "Figure 2 (ISPD 2012 Counterexample)",
        "MCNC: apte (9 blocks)",
        "MCNC: xerox (10 blocks)",
        "MCNC: hp (11 blocks)",
        "MCNC: ami33a (33 blocks)",
        "Synthetic Scalable Benchmark (HB-Style)",
    ],
    index=1,
)

# Custom params for synthetic benchmark
if "Synthetic" in benchmark_choice:
    num_soft = st.sidebar.slider("Number of Soft Blocks", 5, 60, 20)
    num_hard = st.sidebar.slider("Number of Hard Macros", 2, 20, 6)
    seed = st.sidebar.number_input("Random Seed", value=42, step=1)
    blocks, gh, gv, default_w = generate_hb_style_benchmark(num_soft=num_soft, num_hard=num_hard, seed=seed)
    bench_name = f"Synthetic_{num_soft}S_{num_hard}H"
    default_h = default_w
    nets = [[i, (i % len(blocks)) + 1] for i in range(1, len(blocks))]
elif "Figure 2" in benchmark_choice:
    blocks, gh, gv, default_w = build_figure2_benchmark()
    bench_name = "Figure2_Example"
    default_h = 4.0
    nets = [[1, 2], [2, 3], [3, 4]]
else:
    # MCNC
    clean_name = benchmark_choice.split(":")[1].split("(")[0].strip()
    blocks, gh, gv, default_w = build_mcnc_benchmark(clean_name, target_experiment="table2", seed=42)
    bench_name = clean_name
    default_h = default_w * 1.15
    nets = [[i, (i % len(blocks)) + 1] for i in range(1, len(blocks))]

# Outline Bounds
st.sidebar.subheader("📐 Fixed-Outline Chip Constraints")
outline_w = st.sidebar.number_input("Upper-Bound Width (W)", value=float(round(default_w, 2)), step=1.0)
outline_h = st.sidebar.number_input("Target Height (H)", value=float(round(default_h, 2)), step=1.0)

st.sidebar.subheader("⚙️ Search & Q-Learning Settings")
max_iters = st.sidebar.slider("Optimization Iterations", 20, 300, 100, step=10)
lr = st.sidebar.slider("Q-Learning Rate (α)", 0.05, 0.50, 0.15, step=0.05)
gamma = st.sidebar.slider("Discount Factor (γ)", 0.70, 0.99, 0.90, step=0.05)
epsilon = st.sidebar.slider("Initial Exploration (ε)", 0.0, 0.6, 0.30, step=0.05)

run_button = st.sidebar.button("🚀 Run Comparative Study", type="primary", use_container_width=True)

# Main App Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Comparative Study (4 Modes)",
    "🗺️ Layout Visualizer",
    "📈 Convergence Curves",
    "🧠 Q-Table & Step-Size Diagnostics",
    "🔬 SDS Theoretical Inspector",
])


# Session State storage for persistent results
if "experiment_results" not in st.session_state:
    st.session_state.experiment_results = None

if run_button or st.session_state.experiment_results is None:
    with st.spinner(f"Running 4-way comparative evaluation on {bench_name}..."):
        t0 = time.perf_counter()
        exp_res = compare_all_methods(
            benchmark_name=bench_name,
            blocks=blocks,
            nets=nets,
            outline_width=outline_w,
            outline_height=outline_h,
            max_iterations=max_iters,
            seed=42,
            verbose=False,
        )
        st.session_state.experiment_results = exp_res


exp_res = st.session_state.experiment_results

# TAB 1: Comparative Study
with tab1:
    st.header(f"Experimental Evaluation on `{exp_res.benchmark_name}` ({exp_res.num_blocks} Blocks)")
    st.markdown(
        rf"""
        Fixed Outline Constraints: **Width $W \le {exp_res.outline_width:.2f}$** | **Height $H \le {exp_res.outline_height:.2f}$**
        """
    )

    df_summary = pd.DataFrame(exp_res.summary_table)

    # Styling helper
    def highlight_legal(row):
        is_legal = row["Legal"] == "Yes"
        color = "#E8F5E9" if is_legal else "#FFEBEE"
        return [f"background-color: {color}"] * len(row)

    st.dataframe(df_summary.style.apply(highlight_legal, axis=1), use_container_width=True)

    # Key metrics comparison cards
    col1, col2, col3, col4 = st.columns(4)
    baseline_res = exp_res.results[SearchMode.BASELINE.name]
    shaping_res = exp_res.results[SearchMode.SHAPING_ONLY.name]
    ql_res = exp_res.results[SearchMode.Q_LEARNING_ONLY.name]
    prop_res = exp_res.results[SearchMode.COMBINED_PROPOSED.name]

    with col1:
        st.metric(
            "1. Baseline",
            f"H = {baseline_res.best_height:.2f}",
            f"{'Legal' if baseline_res.best_is_legal else 'Violates Outline'}",
            delta_color="normal" if baseline_res.best_is_legal else "inverse",
        )
        st.caption(f"HPWL: {baseline_res.best_hpwl:.2f} | Time: {baseline_res.runtime_seconds:.2f}s")

    with col2:
        diff_h = (baseline_res.best_height - shaping_res.best_height) / max(baseline_res.best_height, 1.0) * 100
        st.metric(
            "2. SDS Shaping Only",
            f"H = {shaping_res.best_height:.2f}",
            f"{diff_h:+.1f}% vs Baseline",
            delta_color="inverse" if diff_h > 0 else "normal",
        )
        st.caption(f"Legal: {shaping_res.best_is_legal} | Time: {shaping_res.runtime_seconds:.2f}s")

    with col3:
        diff_ql = (baseline_res.best_height - ql_res.best_height) / max(baseline_res.best_height, 1.0) * 100
        st.metric(
            "3. Q-Learning Only",
            f"H = {ql_res.best_height:.2f}",
            f"{diff_ql:+.1f}% vs Baseline",
            delta_color="inverse" if diff_ql > 0 else "normal",
        )
        st.caption(f"Legal: {ql_res.best_is_legal} | Time: {ql_res.runtime_seconds:.2f}s")

    with col4:
        diff_prop = (baseline_res.best_height - prop_res.best_height) / max(baseline_res.best_height, 1.0) * 100
        st.metric(
            "4. Proposed (Q-Learn + SDS)",
            f"H = {prop_res.best_height:.2f}",
            f"{diff_prop:+.1f}% vs Baseline",
            delta_color="inverse" if diff_prop > 0 else "normal",
        )
        st.caption(f"Legal: {prop_res.best_is_legal} | Time: {prop_res.runtime_seconds:.2f}s")

    # Comparative charts
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Layout Height Comparison")
        fig_h, ax_h = plt.subplots(figsize=(6, 3.5), dpi=120)
        bars_h = ax_h.bar(
            ["Baseline", "Shaping Only", "Q-Learning", "Proposed (Both)"],
            [r.best_height for r in [baseline_res, shaping_res, ql_res, prop_res]],
            color=["#90A4AE", "#42A5F5", "#AB47BC", "#2E7D32"],
        )
        ax_h.axhline(exp_res.outline_height, color="#D32F2F", linestyle="--", label="Target Height Bound (H)")
        ax_h.set_ylabel("Height")
        ax_h.legend(loc="upper right")
        ax_h.grid(axis="y", linestyle=":", alpha=0.6)
        st.pyplot(fig_h)
        plt.close(fig_h)

    with chart_col2:
        st.subheader("HPWL Wirelength Comparison")
        fig_w, ax_w = plt.subplots(figsize=(6, 3.5), dpi=120)
        bars_w = ax_w.bar(
            ["Baseline", "Shaping Only", "Q-Learning", "Proposed (Both)"],
            [r.best_hpwl for r in [baseline_res, shaping_res, ql_res, prop_res]],
            color=["#90A4AE", "#42A5F5", "#AB47BC", "#2E7D32"],
        )
        ax_w.set_ylabel("HPWL")
        ax_w.grid(axis="y", linestyle=":", alpha=0.6)
        st.pyplot(fig_w)
        plt.close(fig_w)


# TAB 2: Layout Visualizer
with tab2:
    st.header("Floorplan Layout Visualization")
    layout_mode = st.radio(
        "Select Layout to Display",
        ["Proposed (Q-Learning + SDS Shaping)", "Baseline (Fixed Shapes, Fixed Step)", "SDS Shaping Only", "Q-Learning Only"],
        horizontal=True,
    )

    mode_map = {
        "Proposed (Q-Learning + SDS Shaping)": prop_res,
        "Baseline (Fixed Shapes, Fixed Step)": baseline_res,
        "SDS Shaping Only": shaping_res,
        "Q-Learning Only": ql_res,
    }
    sel_res = mode_map[layout_mode]

    fig_layout = plot_floorplan(
        sel_res.best_blocks,
        sel_res.best_width,
        sel_res.best_height,
        exp_res.outline_width,
        title=f"{layout_mode} - W={sel_res.best_width:.2f}, H={sel_res.best_height:.2f} (Legal: {sel_res.best_is_legal})",
    )
    st.pyplot(fig_layout)
    plt.close(fig_layout)


# TAB 3: Convergence Curves
with tab3:
    st.header("Optimization Trajectory & Convergence")
    fig_conv, ax_c = plt.subplots(figsize=(9, 4.5), dpi=130)

    for mode_name, color, label in [
        (SearchMode.BASELINE.name, "#90A4AE", "Baseline (Fixed Shapes, Fixed Step)"),
        (SearchMode.SHAPING_ONLY.name, "#42A5F5", "SDS Shaping Only"),
        (SearchMode.Q_LEARNING_ONLY.name, "#AB47BC", "Q-Learning Only"),
        (SearchMode.COMBINED_PROPOSED.name, "#2E7D32", "Proposed (Q-Learning + SDS)"),
    ]:
        res_obj = exp_res.results[mode_name]
        iters = [rec.iteration for rec in res_obj.history]
        costs = [rec.cost for rec in res_obj.history]
        ax_c.plot(iters, costs, label=label, color=color, linewidth=1.8 if "Proposed" in label else 1.2)

    ax_c.set_xlabel("Iteration", fontweight="bold")
    ax_c.set_ylabel("Composite Cost (Height + Violations + HPWL)", fontweight="bold")
    ax_c.grid(True, linestyle=":", alpha=0.6)
    ax_c.legend(loc="upper right")
    st.pyplot(fig_conv)
    plt.close(fig_conv)


# TAB 4: Q-Learning Diagnostics
with tab4:
    st.header("Tabular Q-Learning Agent Diagnostics")
    if prop_res.q_controller:
        agent = prop_res.q_controller
        st.subheader("Learned Q-Table Matrix")
        st.markdown(
            """
            * **Actions:** `Small` (1 local perturbation), `Medium` (2-3 perturbations), `Large` (4-6 coarse leaps).
            * **State Space:** Progress Trend (`Improving`, `Unchanged`, `Worsening`) $\\times$ Outline Legality (`Legal`, `Illegal`).
            """
        )
        q_summary = agent.get_q_matrix_summary()
        df_q = pd.DataFrame(q_summary).T
        st.dataframe(df_q.style.background_gradient(cmap="Greens", axis=None), use_container_width=True)

        st.subheader("Action Distribution Chosen by Agent")
        action_counts = {}
        for a in agent.action_history:
            action_counts[a.name] = action_counts.get(a.name, 0) + 1
        df_actions = pd.DataFrame(list(action_counts.items()), columns=["Action", "Count"])
        st.bar_chart(df_actions.set_index("Action"))
    else:
        st.info("Run with Q-learning enabled to view learned Q-table.")


# TAB 5: SDS Theoretical Inspector (ISPD 2012 Deep Dive)
with tab5:
    st.header("Deep-Dive Inspector: Yan & Chu (ISPD 2012) SDS Engine")
    st.markdown(
        r"""
        Test the standalone SDS algorithm on the current circuit to inspect:
        - **4-Line Packing:** Left Line ($x=0$), Right Line ($x=W$), Bottom Line ($y=0$), Top Line ($y=y_{n+1}$).
        - **Slack Values:** Horizontal slack $s_i^h$ and vertical slack $s_i^v$ for every block.
        - **The 7 Disjoint Subsets (I to VII):** Identification of target soft blocks (Subsets IV & VI) and intersection blocks (Subset III).
        - **Optimality Proofs:** Verification of **Lemma 2** (Hard VCP) and **Lemma 3** ($\le 1$ soft HCP/VCP).
        """
    )

    if st.button("🔍 Run Full SDS Analysis on Circuit", type="secondary"):
        with st.spinner("Analyzing constraint graphs and slacks..."):
            inspector_solver = BasicSDSSolver(blocks, gh, gv, upper_bound_width=outline_w)
            inspector_solver.compute_slacks()
            subsets = inspector_solver.classify_subsets()

            checker = OptimalityChecker(inspector_solver.blocks, gh, gv, upper_bound_width=outline_w)
            report = checker.check(inspector_solver.x_ll, inspector_solver.y_bl, subsets)

            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Initial Layout Width", f"{inspector_solver.layout_width:.2f}")
            with col_s2:
                st.metric("Initial Layout Height", f"{inspector_solver.layout_height:.2f}")
            with col_s3:
                st.metric("Optimality Status", "Optimal" if report.is_optimal else "Trapped / Needs GP")

            st.markdown(f"**Optimality Explanation:** {report.explanation}")

            # Subsets breakdown
            st.subheader("Block Subset Distribution (Section 3.1)")
            subset_rows = []
            for i, b in inspector_solver.blocks.items():
                subset_rows.append({
                    "Block": b.name,
                    "Type": "Hard" if b.is_hard else "Soft",
                    "Width": round(b.w, 2),
                    "Height": round(b.h, 2),
                    "Slack H": round(b.slack_h, 4),
                    "Slack V": round(b.slack_v, 4),
                    "Subset": b.subset,
                })
            st.dataframe(pd.DataFrame(subset_rows), use_container_width=True)
