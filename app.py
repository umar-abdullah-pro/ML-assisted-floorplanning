"""Streamlit Web Application: Q-Learning-Assisted Slack-Driven Optimization for Fixed-Outline VLSI Floorplanning.
Combines:
1. Optimal Slack-Driven Block Shaping (SDS) [Yan & Chu, ISPD 2012]
2. Q-Learning Step-Size Controller [Meng et al., CSF 2025]
3. Custom Floorplan Input Studio (Table Editor, JSON/CSV Upload, Custom Netlist)
"""

from __future__ import annotations
import math
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sds.block import Block
from sds.basic_sds import BasicSDSSolver
from sds.optimality import OptimalityChecker
from sds.visualizer import plot_floorplan
from sds.sequence_pair import sequence_pair_to_constraint_graphs
from sds.custom_input import (
    DEFAULT_CUSTOM_BLOCKS,
    DEFAULT_CUSTOM_NETS,
    parse_blocks_from_dataframe,
    parse_netlist_text,
    parse_custom_json,
    export_custom_to_json,
)
from benchmarks.figure2_example import build_figure2_benchmark
from benchmarks.mcnc_benchmarks import build_mcnc_benchmark
from benchmarks.synthetic_benchmarks import generate_hb_style_benchmark
from q_learning.placement_search import SearchMode
from q_learning.experiment_runner import compare_all_methods


# Page configuration
st.set_page_config(
    page_title="VLSI Floorplanning | SDS & Q-Learning",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS using translucent theme-adaptive styling so Light & Dark modes both look crisp
st.markdown(
    """
    <style>
    /* Layout rhythm & typography */
    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 2.5rem;
        max-width: 1380px;
    }
    footer {visibility: hidden;}

    /* Sidebar polish */
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.72;
        margin-top: 1.1rem;
        margin-bottom: 0.4rem;
    }

    /* Hero header card */
    .hero-banner {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.08) 0%, rgba(148, 163, 184, 0.06) 100%);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-left: 4px solid #3B82F6;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
    }
    .hero-kicker {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #3B82F6;
        margin-bottom: 0.25rem;
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: inherit;
        margin: 0 0 0.4rem 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 0.92rem;
        color: inherit;
        opacity: 0.82;
        margin: 0;
        line-height: 1.5;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: rgba(148, 163, 184, 0.07);
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 10px;
        padding: 0.9rem 1.05rem;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.78rem;
        font-weight: 600;
        opacity: 0.72;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.45rem;
        font-weight: 700;
        color: inherit;
        font-variant-numeric: tabular-nums;
    }

    /* Tabs */
    div[data-baseweb="tab-list"] {
        gap: 0.35rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.25);
        padding-bottom: 0.1rem;
        margin-bottom: 1rem;
    }
    button[data-baseweb="tab"] {
        font-size: 0.9rem;
        font-weight: 600;
        color: inherit;
        opacity: 0.72;
        padding: 0.55rem 1rem;
        border-radius: 6px 6px 0 0;
        background-color: transparent;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        opacity: 1;
        color: #3B82F6;
        background-color: rgba(59, 130, 246, 0.12);
    }

    /* Expander card */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    /* Status badge pill */
    .status-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        align-items: center;
        background-color: rgba(148, 163, 184, 0.07);
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 8px;
        padding: 0.65rem 1rem;
        margin-bottom: 1rem;
        font-size: 0.88rem;
        color: inherit;
    }
    .badge {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-legal {
        background-color: rgba(16, 185, 129, 0.16);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .badge-illegal {
        background-color: rgba(239, 68, 68, 0.16);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .badge-neutral {
        background-color: rgba(59, 130, 246, 0.16);
        color: #3B82F6;
        border: 1px solid rgba(59, 130, 246, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Banner
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-kicker">Fixed-Outline Physical Design Workbench</div>
        <h1 class="hero-title">Q-Learning Assisted Slack-Driven VLSI Floorplanning</h1>
        <p class="hero-subtitle">
            <strong>Shaping Engine:</strong> Optimal Slack-Driven Block Shaping (Yan &amp; Chu, ISPD 2012)
            &nbsp;&bull;&nbsp;
            <strong>Step-Size Controller:</strong> Tabular Q-Learning Search Regulator (Meng et al., CSF 2025)
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "custom_blocks_df" not in st.session_state:
    st.session_state.custom_blocks_df = pd.DataFrame(DEFAULT_CUSTOM_BLOCKS)
if "custom_nets_text" not in st.session_state:
    st.session_state.custom_nets_text = DEFAULT_CUSTOM_NETS
if "experiment_results" not in st.session_state:
    st.session_state.experiment_results = None

# Sidebar Setup
st.sidebar.subheader("Benchmark & Design Setup")

benchmark_choice = st.sidebar.selectbox(
    "Select Benchmark Circuit",
    [
        "Custom Floorplan (Build / Upload)",
        "MCNC: apte (9 blocks)",
        "MCNC: xerox (10 blocks)",
        "MCNC: hp (11 blocks)",
        "MCNC: ami33a (33 blocks)",
        "Synthetic Scalable Benchmark (HB-Style)",
        "Figure 2 (ISPD 2012 Counterexample)",
    ],
    index=0,
)

gh = None
gv = None

# Branch on benchmark selection
if benchmark_choice == "Custom Floorplan (Build / Upload)":
    bench_name = "Custom_Floorplan"
    is_custom = True
elif "Synthetic" in benchmark_choice:
    is_custom = False
    num_soft = st.sidebar.slider("Number of Soft Blocks", 5, 60, 20)
    num_hard = st.sidebar.slider("Number of Hard Macros", 2, 20, 6)
    seed = st.sidebar.number_input("Random Seed", value=42, step=1)
    blocks, gh, gv, default_w = generate_hb_style_benchmark(num_soft=num_soft, num_hard=num_hard, seed=seed)
    bench_name = f"Synthetic_{num_soft}S_{num_hard}H"
    default_h = default_w
    nets = [[i, (i % len(blocks)) + 1] for i in range(1, len(blocks))]
elif "Figure 2" in benchmark_choice:
    is_custom = False
    blocks, gh, gv, default_w = build_figure2_benchmark()
    bench_name = "Figure2_Example"
    default_h = 4.0
    nets = [[1, 2], [2, 3], [3, 4]]
else:
    is_custom = False
    clean_name = benchmark_choice.split(":")[1].split("(")[0].strip()
    blocks, gh, gv, default_w = build_mcnc_benchmark(clean_name, target_experiment="table2", seed=42)
    bench_name = clean_name
    default_h = default_w * 1.15
    nets = [[i, (i % len(blocks)) + 1] for i in range(1, len(blocks))]


# CUSTOM FLOORPLAN STUDIO
if is_custom:
    with st.expander("Custom Floorplan Studio — Design, Edit & Connect Blocks", expanded=True):
        st.caption(
            "Configure soft and hard blocks, aspect ratio bounds, and netlist connectivity via the table editor, file upload, or JSON."
        )

        input_method = st.radio(
            "Input Method",
            ["Interactive Table Editor", "Upload JSON / CSV File", "Paste JSON Code"],
            horizontal=True,
        )

        if input_method == "Interactive Table Editor":
            pcol1, pcol2, pcol3, _ = st.columns([1.4, 1.4, 1.0, 3.2])
            with pcol1:
                if st.button("Load 8-Block SoC Preset", width="stretch"):
                    st.session_state.custom_blocks_df = pd.DataFrame(DEFAULT_CUSTOM_BLOCKS)
                    st.session_state.custom_nets_text = DEFAULT_CUSTOM_NETS
                    st.rerun()
            with pcol2:
                if st.button("Load 4-Block Fig. 2 Preset", width="stretch"):
                    st.session_state.custom_blocks_df = pd.DataFrame([
                        {"Name": "B1", "Type": "Soft", "Area": 4.0, "Fixed_Width": 0.0, "Fixed_Height": 0.0, "Min_AR": 0.25, "Max_AR": 4.0},
                        {"Name": "B2", "Type": "Soft", "Area": 4.0, "Fixed_Width": 0.0, "Fixed_Height": 0.0, "Min_AR": 0.25, "Max_AR": 4.0},
                        {"Name": "B3", "Type": "Soft", "Area": 4.0, "Fixed_Width": 0.0, "Fixed_Height": 0.0, "Min_AR": 0.25, "Max_AR": 4.0},
                        {"Name": "B4", "Type": "Soft", "Area": 4.0, "Fixed_Width": 0.0, "Fixed_Height": 0.0, "Min_AR": 0.25, "Max_AR": 4.0},
                    ])
                    st.session_state.custom_nets_text = "B1, B2\nB2, B3\nB3, B4"
                    st.rerun()
            with pcol3:
                if st.button("Reset Table", width="stretch"):
                    st.session_state.custom_blocks_df = pd.DataFrame([
                        {"Name": "Block1", "Type": "Soft", "Area": 100.0, "Fixed_Width": 0.0, "Fixed_Height": 0.0, "Min_AR": 0.33, "Max_AR": 3.0}
                    ])
                    st.session_state.custom_nets_text = ""
                    st.rerun()

            column_config = {
                "Name": st.column_config.TextColumn("Block Name", required=True),
                "Type": st.column_config.SelectboxColumn("Type", options=["Soft", "Hard"], required=True),
                "Area": st.column_config.NumberColumn("Area (A)", min_value=1.0, max_value=100000.0, step=1.0, format="%.1f"),
                "Fixed_Width": st.column_config.NumberColumn("Fixed Width (Hard)", min_value=0.0, max_value=10000.0, step=0.5, format="%.2f"),
                "Fixed_Height": st.column_config.NumberColumn("Fixed Height (Hard)", min_value=0.0, max_value=10000.0, step=0.5, format="%.2f"),
                "Min_AR": st.column_config.NumberColumn("Min AR (h/w)", min_value=0.1, max_value=5.0, step=0.05, format="%.2f"),
                "Max_AR": st.column_config.NumberColumn("Max AR (h/w)", min_value=0.2, max_value=10.0, step=0.05, format="%.2f"),
            }

            edited_df = st.data_editor(
                st.session_state.custom_blocks_df,
                column_config=column_config,
                num_rows="dynamic",
                width="stretch",
                key="custom_blocks_editor",
            )
            st.session_state.custom_blocks_df = edited_df

        elif input_method == "Upload JSON / CSV File":
            uploaded_file = st.file_uploader("Upload Floorplan Specification (.json or .csv)", type=["json", "csv"])
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith(".json"):
                        content = uploaded_file.read().decode("utf-8")
                        parsed_df, parsed_nets = parse_custom_json(content)
                        st.session_state.custom_blocks_df = parsed_df
                        st.session_state.custom_nets_text = parsed_nets
                        st.success(f"Loaded {len(parsed_df)} blocks from {uploaded_file.name}.")
                    else:
                        parsed_df = pd.read_csv(uploaded_file)
                        st.session_state.custom_blocks_df = parsed_df
                        st.success(f"Loaded {len(parsed_df)} blocks from {uploaded_file.name}.")
                    st.dataframe(st.session_state.custom_blocks_df, width="stretch")
                except Exception as e:
                    st.error(f"Error parsing file: {e}")

        else:
            json_sample = export_custom_to_json(
                st.session_state.custom_blocks_df,
                st.session_state.custom_nets_text,
                {},
            )
            pasted_json = st.text_area("JSON Floorplan Definition", value=json_sample, height=210)
            if st.button("Apply Pasted JSON"):
                try:
                    parsed_df, parsed_nets = parse_custom_json(pasted_json)
                    st.session_state.custom_blocks_df = parsed_df
                    st.session_state.custom_nets_text = parsed_nets
                    st.success("JSON applied successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")

        st.markdown("**Netlist Connectivity (HPWL Wirelength)**")
        custom_nets_str = st.text_area(
            "Enter comma-separated block names per net (one net per line)",
            value=st.session_state.custom_nets_text,
            height=95,
            key="custom_nets_input_area",
        )
        st.session_state.custom_nets_text = custom_nets_str

        # Parse blocks and compute statistics
        blocks, name_to_id, total_area = parse_blocks_from_dataframe(st.session_state.custom_blocks_df)
        nets = parse_netlist_text(st.session_state.custom_nets_text, name_to_id)

        sug_w = math.ceil(math.sqrt(total_area * 1.15))
        sug_h = sug_w
        default_w = float(sug_w)
        default_h = float(sug_h)

        stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns([1.3, 1.3, 1.6, 1.1, 1.1])
        with stat_col1:
            st.metric("Total Silicon Area", f"{total_area:.1f}")
        with stat_col2:
            num_s = sum(1 for b in blocks.values() if not b.is_hard)
            num_h = len(blocks) - num_s
            st.metric("Block Count", f"{len(blocks)} ({num_s}S / {num_h}H)")
        with stat_col3:
            st.metric("Suggested Outline (15% WS)", f"{sug_w} x {sug_h}")
        with stat_col4:
            json_export = export_custom_to_json(st.session_state.custom_blocks_df, st.session_state.custom_nets_text, name_to_id)
            st.download_button("Export JSON", data=json_export, file_name="custom_floorplan.json", mime="application/json", width="stretch")
        with stat_col5:
            csv_export = st.session_state.custom_blocks_df.to_csv(index=False)
            st.download_button("Export CSV", data=csv_export, file_name="custom_blocks.csv", mime="text/csv", width="stretch")


# Sidebar: Outline Bounds & Settings
st.sidebar.divider()
st.sidebar.subheader("Fixed-Outline Chip Constraints")
outline_w = st.sidebar.number_input("Upper-Bound Width (W)", value=float(round(default_w, 2)), step=1.0)
outline_h = st.sidebar.number_input("Target Height (H)", value=float(round(default_h, 2)), step=1.0)

st.sidebar.divider()
st.sidebar.subheader("Search & Q-Learning Settings")
max_iters = st.sidebar.slider("Optimization Iterations", 20, 300, 100, step=10)
lr = st.sidebar.slider("Q-Learning Rate (alpha)", 0.05, 0.50, 0.15, step=0.05)
gamma = st.sidebar.slider("Discount Factor (gamma)", 0.70, 0.99, 0.90, step=0.05)
epsilon = st.sidebar.slider("Initial Exploration (epsilon)", 0.0, 0.60, 0.30, step=0.05)

st.sidebar.markdown("")
run_button = st.sidebar.button("Run Comparative Study", type="primary", width="stretch")

# Main App Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Comparative Study",
    "Layout Visualizer",
    "Convergence Curves",
    "Q-Table & Diagnostics",
    "SDS Theoretical Inspector",
])


# Run Optimization
if run_button or st.session_state.experiment_results is None:
    with st.spinner(f"Running 4-way comparative evaluation on {bench_name}..."):
        exp_res = compare_all_methods(
            benchmark_name=bench_name,
            blocks=blocks,
            nets=nets,
            outline_width=outline_w,
            outline_height=outline_h,
            max_iterations=max_iters,
            seed=42,
            verbose=False,
            learning_rate=lr,
            discount_factor=gamma,
            epsilon_init=epsilon,
        )
        st.session_state.experiment_results = exp_res


exp_res = st.session_state.experiment_results

# Detect active Streamlit theme (Light / Dark) safely
is_dark = (getattr(st.context.theme, "type", None) == "dark")
chart_theme = {
    "fig_bg": "#111827" if is_dark else "#FFFFFF",
    "ax_bg": "#0F172A" if is_dark else "#FFFFFF",
    "text_col": "#F1F5F9" if is_dark else "#1E293B",
    "muted_col": "#CBD5E1" if is_dark else "#334155",
    "tick_col": "#94A3B8" if is_dark else "#475569",
    "grid_col": "#334155" if is_dark else "#CBD5E1",
    "legend_bg": "#1E293B" if is_dark else "#FFFFFF",
    "legend_edge": "#334155" if is_dark else "#E2E8F0",
}


def _style_axes(ax: plt.Axes, fig: plt.Figure) -> None:
    """Apply consistent minimal theme-aware styling to Matplotlib figures and axes."""
    fig.patch.set_facecolor(chart_theme["fig_bg"])
    ax.set_facecolor(chart_theme["ax_bg"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(chart_theme["grid_col"])
    ax.spines["bottom"].set_color(chart_theme["grid_col"])
    ax.tick_params(colors=chart_theme["tick_col"], labelsize=9)
    ax.grid(axis="y", linestyle=":", color=chart_theme["grid_col"], alpha=0.6)


# TAB 1: Comparative Study
with tab1:
    st.markdown(
        f"""
        <div class="status-strip">
            <span><strong>Active Benchmark:</strong> {exp_res.benchmark_name}</span>
            <span class="badge badge-neutral">{exp_res.num_blocks} Blocks</span>
            <span><strong>Width Bound:</strong> W &le; {exp_res.outline_width:.2f}</span>
            <span><strong>Height Target:</strong> H &le; {exp_res.outline_height:.2f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_summary = pd.DataFrame(exp_res.summary_table)

    def highlight_legal(row):
        if row["Legal"] == "Yes":
            if is_dark:
                return ["background-color: rgba(16, 185, 129, 0.18); color: #6EE7B7"] * len(row)
            return ["background-color: #ECFDF5; color: #065F46"] * len(row)
        if is_dark:
            return ["background-color: rgba(239, 68, 68, 0.18); color: #FCA5A5"] * len(row)
        return ["background-color: #FEF2F2; color: #991B1B"] * len(row)

    styled_df = (
        df_summary.style
        .apply(highlight_legal, axis=1)
        .format({
            "Height": "{:.2f}",
            "Width": "{:.2f}",
            "HPWL": "{:.2f}",
            "Cost": "{:.4f}",
            "Time (s)": "{:.3f}",
        })
    )
    st.dataframe(styled_df, width="stretch", hide_index=True)

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
            "Legal" if baseline_res.best_is_legal else "Violates Outline",
            delta_color="normal" if baseline_res.best_is_legal else "inverse",
        )
        st.caption(f"HPWL: {baseline_res.best_hpwl:.2f}  |  Time: {baseline_res.runtime_seconds:.3f}s")

    with col2:
        pct_h = (shaping_res.best_height - baseline_res.best_height) / max(baseline_res.best_height, 1.0) * 100
        st.metric(
            "2. SDS Shaping Only",
            f"H = {shaping_res.best_height:.2f}",
            f"{pct_h:+.1f}% vs Baseline",
            delta_color="inverse",
        )
        st.caption(f"Legal: {'Yes' if shaping_res.best_is_legal else 'No'}  |  HPWL: {shaping_res.best_hpwl:.2f}  |  {shaping_res.runtime_seconds:.3f}s")

    with col3:
        pct_ql = (ql_res.best_height - baseline_res.best_height) / max(baseline_res.best_height, 1.0) * 100
        st.metric(
            "3. Q-Learning Only",
            f"H = {ql_res.best_height:.2f}",
            f"{pct_ql:+.1f}% vs Baseline",
            delta_color="inverse",
        )
        st.caption(f"Legal: {'Yes' if ql_res.best_is_legal else 'No'}  |  HPWL: {ql_res.best_hpwl:.2f}  |  {ql_res.runtime_seconds:.3f}s")

    with col4:
        pct_prop = (prop_res.best_height - baseline_res.best_height) / max(baseline_res.best_height, 1.0) * 100
        st.metric(
            "4. Proposed (Q-Learning + SDS)",
            f"H = {prop_res.best_height:.2f}",
            f"{pct_prop:+.1f}% vs Baseline",
            delta_color="inverse",
        )
        st.caption(f"Legal: {'Yes' if prop_res.best_is_legal else 'No'}  |  HPWL: {prop_res.best_hpwl:.2f}  |  {prop_res.runtime_seconds:.3f}s")

    st.markdown("")

    # Comparative charts
    chart_col1, chart_col2 = st.columns(2)
    method_labels = ["Baseline", "SDS Shaping", "Q-Learning", "Proposed"]
    palette = ["#94A3B8", "#3B82F6", "#8B5CF6", "#059669"]

    with chart_col1:
        st.subheader("Final Layout Height Comparison")
        fig_h, ax_h = plt.subplots(figsize=(6, 3.6), dpi=130)
        heights_vals = [r.best_height for r in [baseline_res, shaping_res, ql_res, prop_res]]
        bars_h = ax_h.bar(method_labels, heights_vals, color=palette, width=0.55, zorder=3)
        ax_h.axhline(exp_res.outline_height, color="#FB7185" if is_dark else "#E11D48", linestyle="--", linewidth=1.4, label=f"Target Height (H={exp_res.outline_height:.1f})", zorder=4)
        for bar in bars_h:
            val = bar.get_height()
            ax_h.annotate(
                f"{val:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=chart_theme["text_col"],
            )
        ax_h.set_ylabel("Height", fontsize=9.5, fontweight="bold", color=chart_theme["muted_col"])
        ax_h.set_ylim(0, max(max(heights_vals), exp_res.outline_height) * 1.18)
        _style_axes(ax_h, fig_h)
        leg_h = ax_h.legend(loc="upper right", frameon=True, facecolor=chart_theme["legend_bg"], edgecolor=chart_theme["legend_edge"], fontsize=8.5)
        for txt in leg_h.get_texts():
            txt.set_color(chart_theme["text_col"])
        plt.tight_layout()
        st.pyplot(fig_h)
        plt.close(fig_h)

    with chart_col2:
        st.subheader("HPWL Wirelength Comparison")
        fig_w, ax_w = plt.subplots(figsize=(6, 3.6), dpi=130)
        hpwl_vals = [r.best_hpwl for r in [baseline_res, shaping_res, ql_res, prop_res]]
        bars_w = ax_w.bar(method_labels, hpwl_vals, color=palette, width=0.55, zorder=3)
        for bar in bars_w:
            val = bar.get_height()
            ax_w.annotate(
                f"{val:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=chart_theme["text_col"],
            )
        ax_w.set_ylabel("Half-Perimeter Wirelength (HPWL)", fontsize=9.5, fontweight="bold", color=chart_theme["muted_col"])
        ax_w.set_ylim(0, max(max(hpwl_vals), 1.0) * 1.18)
        _style_axes(ax_w, fig_w)
        plt.tight_layout()
        st.pyplot(fig_w)
        plt.close(fig_w)


# TAB 2: Layout Visualizer
with tab2:
    st.subheader("Floorplan Layout Visualization")
    layout_mode = st.radio(
        "Select Method Layout",
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

    badge_cls = "badge-legal" if sel_res.best_is_legal else "badge-illegal"
    badge_txt = "LEGAL (Fits Outline)" if sel_res.best_is_legal else "VIOLATES OUTLINE"
    st.markdown(
        f"""
        <div class="status-strip">
            <span class="badge {badge_cls}">{badge_txt}</span>
            <span><strong>Width:</strong> {sel_res.best_width:.2f} / {exp_res.outline_width:.2f}</span>
            <span><strong>Height:</strong> {sel_res.best_height:.2f} / {exp_res.outline_height:.2f}</span>
            <span><strong>HPWL:</strong> {sel_res.best_hpwl:.2f}</span>
            <span><strong>Overlaps:</strong> 0 (Sequence-Pair Guaranteed)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig_layout = plot_floorplan(
        sel_res.best_blocks,
        sel_res.best_width,
        sel_res.best_height,
        exp_res.outline_width,
        title=f"{layout_mode}  |  W={sel_res.best_width:.2f}, H={sel_res.best_height:.2f}",
        target_height=exp_res.outline_height,
        dark_mode=is_dark,
    )
    st.pyplot(fig_layout)
    plt.close(fig_layout)


# TAB 3: Convergence Curves
with tab3:
    st.subheader("Optimization Trajectory & Cost Convergence")
    fig_conv, ax_c = plt.subplots(figsize=(9.5, 4.4), dpi=140)

    for mode_name, color, label, lw in [
        (SearchMode.BASELINE.name, "#94A3B8", "Baseline (Fixed Shapes, Fixed Step)", 1.4),
        (SearchMode.SHAPING_ONLY.name, "#3B82F6", "SDS Shaping Only", 1.6),
        (SearchMode.Q_LEARNING_ONLY.name, "#8B5CF6", "Q-Learning Only", 1.6),
        (SearchMode.COMBINED_PROPOSED.name, "#10B981" if is_dark else "#059669", "Proposed (Q-Learning + SDS)", 2.2),
    ]:
        res_obj = exp_res.results[mode_name]
        iters = [rec.iteration for rec in res_obj.history]
        costs = [rec.cost for rec in res_obj.history]
        ax_c.plot(iters, costs, label=label, color=color, linewidth=lw, alpha=0.92)

    ax_c.set_xlabel("Iteration", fontsize=10, fontweight="bold", color=chart_theme["muted_col"])
    ax_c.set_ylabel("Composite Cost (Height + Violations + HPWL)", fontsize=10, fontweight="bold", color=chart_theme["muted_col"])
    _style_axes(ax_c, fig_conv)
    ax_c.grid(True, linestyle=":", color=chart_theme["grid_col"], alpha=0.6)
    leg_c = ax_c.legend(loc="upper right", frameon=True, facecolor=chart_theme["legend_bg"], edgecolor=chart_theme["legend_edge"], fontsize=9)
    for txt in leg_c.get_texts():
        txt.set_color(chart_theme["text_col"])
    plt.tight_layout()
    st.pyplot(fig_conv)
    plt.close(fig_conv)


# TAB 4: Q-Learning Diagnostics
with tab4:
    st.subheader("Tabular Q-Learning Agent Diagnostics")
    if prop_res.q_controller:
        agent = prop_res.q_controller
        st.markdown(
            "State space tracks **Progress Trend** (`IMPROVING`, `UNCHANGED`, `WORSENING`) $\\times$ "
            "**Outline Legality** (`LEGAL`, `ILLEGAL`). Actions control perturbation step size: "
            "`SMALL` (1 move), `MEDIUM` (2–3 moves), `LARGE` (4–6 moves)."
        )
        q_summary = agent.get_q_matrix_summary()
        df_q = pd.DataFrame(q_summary).T
        st.dataframe(
            df_q.style.background_gradient(cmap="Blues", axis=None).format("{:.3f}"),
            width="stretch",
        )

        q_col1, q_col2 = st.columns(2)
        with q_col1:
            st.markdown("**Action Selection Distribution**")
            ordered_actions = ["SMALL", "MEDIUM", "LARGE"]
            action_counts = {a: 0 for a in ordered_actions}
            for a in agent.action_history:
                action_counts[a.name] = action_counts.get(a.name, 0) + 1

            fig_act, ax_act = plt.subplots(figsize=(5.5, 3.3), dpi=130)
            act_vals = [action_counts[k] for k in ordered_actions]
            bars_a = ax_act.bar(ordered_actions, act_vals, color=["#64748B", "#3B82F6", "#059669"], width=0.5, zorder=3)
            for bar in bars_a:
                v = bar.get_height()
                ax_act.annotate(
                    f"{int(v)}",
                    xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color=chart_theme["text_col"],
                )
            ax_act.set_ylabel("Selections", fontsize=9, fontweight="bold", color=chart_theme["muted_col"])
            ax_act.set_ylim(0, max(max(act_vals), 1) * 1.18)
            _style_axes(ax_act, fig_act)
            plt.tight_layout()
            st.pyplot(fig_act)
            plt.close(fig_act)

        with q_col2:
            st.markdown("**Step Reward Progression**")
            fig_rew, ax_rew = plt.subplots(figsize=(5.5, 3.3), dpi=130)
            rewards = agent.reward_history
            ax_rew.plot(range(1, len(rewards) + 1), rewards, color="#3B82F6" if is_dark else "#2563EB", linewidth=1.2, alpha=0.9)
            ax_rew.axhline(0.0, color="#64748B", linestyle="--", linewidth=1.0)
            ax_rew.set_xlabel("Iteration", fontsize=9, fontweight="bold", color=chart_theme["muted_col"])
            ax_rew.set_ylabel("Reward", fontsize=9, fontweight="bold", color=chart_theme["muted_col"])
            _style_axes(ax_rew, fig_rew)
            ax_rew.grid(True, linestyle=":", color=chart_theme["grid_col"], alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig_rew)
            plt.close(fig_rew)
    else:
        st.info("Run with Q-learning enabled to view learned Q-table diagnostics.")


# TAB 5: SDS Theoretical Inspector
with tab5:
    st.subheader("SDS Engine Theoretical Inspector (Yan & Chu, ISPD 2012)")
    st.markdown(
        r"""
        Inspect the standalone Slack-Driven Shaping mechanics on the active circuit:
        - **4-Line Packing:** Left Line ($x=0$), Right Line ($x=W$), Bottom Line ($y=0$), Top Line ($y=y_{n+1}$).
        - **Slack Values:** Horizontal slack $s_i^h$ and vertical slack $s_i^v$ per block.
        - **7 Disjoint Subsets (I–VII):** Target soft blocks (Subsets IV & VI) and intersection blocks (Subset III).
        - **Optimality Conditions:** Verification of **Lemma 2** (Hard VCP exists) and **Lemma 3** ($\le 1$ soft HCP or VCP).
        """
    )

    if st.button("Run Full SDS Analysis", type="secondary"):
        with st.spinner("Analyzing constraint graphs and slacks..."):
            if gh is None or gv is None:
                seq_init = list(range(1, len(blocks) + 1))
                insp_gh, insp_gv = sequence_pair_to_constraint_graphs(seq_init, seq_init, len(blocks), apply_reduction=True)
            else:
                insp_gh, insp_gv = gh, gv

            inspector_solver = BasicSDSSolver(blocks, insp_gh, insp_gv, upper_bound_width=outline_w)
            inspector_solver.compute_slacks()
            subsets = inspector_solver.classify_subsets()

            checker = OptimalityChecker(inspector_solver.blocks, insp_gh, insp_gv, upper_bound_width=outline_w)
            report = checker.check(inspector_solver.x_ll, inspector_solver.y_bl, subsets)

            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Initial Layout Width", f"{inspector_solver.layout_width:.2f}")
            with col_s2:
                st.metric("Initial Layout Height", f"{inspector_solver.layout_height:.2f}")
            with col_s3:
                st.metric("Optimality Status", "Optimal" if report.is_optimal else "Needs GP Step")

            st.info(f"**Optimality Diagnosis:** {report.explanation}")

            st.markdown("**Block Subset Classification (Section 3.1)**")
            subset_rows = []
            for i, b in inspector_solver.blocks.items():
                subset_rows.append({
                    "Block": b.name,
                    "Type": "Hard" if b.is_hard else "Soft",
                    "Width": b.w,
                    "Height": b.h,
                    "Slack H": b.slack_h,
                    "Slack V": b.slack_v,
                    "Subset": b.subset,
                })
            df_subsets = pd.DataFrame(subset_rows).style.format({
                "Width": "{:.2f}",
                "Height": "{:.2f}",
                "Slack H": "{:.4f}",
                "Slack V": "{:.4f}",
            })
            st.dataframe(df_subsets, width="stretch", hide_index=True)
