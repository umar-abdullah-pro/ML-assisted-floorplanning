"""Visualization utilities for VLSI floorplanning and SDS convergence curves."""

from __future__ import annotations
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .block import Block
from .basic_sds import IterationRecord


def plot_floorplan(
    blocks: Dict[int, Block],
    layout_width: float,
    layout_height: float,
    upper_bound_width: float,
    title: str = "Floorplan Layout",
    save_path: Optional[str] = None,
    show_aspect_ratios: bool = True,
    target_height: Optional[float] = None,
    dark_mode: bool = False,
) -> plt.Figure:
    """Plot floorplan layout showing all blocks, bounding outline, and dimensions."""
    fig, ax = plt.subplots(figsize=(8.2, 6.4), dpi=150)

    if dark_mode:
        fig_bg = "#111827"
        ax_bg = "#0B1120"
        outline_fill = "#0F172A"
        outline_edge = "#FB7185"
        text_primary = "#F1F5F9"
        text_muted = "#CBD5E1"
        spine_col = "#334155"
        legend_bg = "#1E293B"
        legend_edge = "#334155"
    else:
        fig_bg = "#FFFFFF"
        ax_bg = "#F8FAFC"
        outline_fill = "#FFFFFF"
        outline_edge = "#E11D48"
        text_primary = "#0F172A"
        text_muted = "#334155"
        spine_col = "#CBD5E1"
        legend_bg = "#FFFFFF"
        legend_edge = "#E2E8F0"

    fig.patch.set_facecolor(fig_bg)
    ax.set_facecolor(ax_bg)

    outline_h = target_height if target_height is not None else layout_height

    # Fixed outline rectangle (W x H)
    outline = patches.Rectangle(
        (0, 0),
        upper_bound_width,
        outline_h,
        linewidth=1.8,
        edgecolor=outline_edge,
        facecolor=outline_fill,
        linestyle="--",
        label=f"Fixed Outline (W={upper_bound_width:.2f}, H={outline_h:.2f})",
        zorder=1,
    )
    ax.add_patch(outline)

    # Curated modern muted palette for soft blocks
    soft_palette = [
        ("#DBEAFE", "#2563EB"),  # Blue
        ("#D1FAE5", "#059669"),  # Emerald
        ("#EDE9FE", "#7C3AED"),  # Violet
        ("#FEF3C7", "#D97706"),  # Amber
        ("#CFFAFE", "#0891B2"),  # Cyan
        ("#FCE7F3", "#DB2777"),  # Pink
        ("#E0E7FF", "#4F46E5"),  # Indigo
        ("#ECFCCB", "#65A30D"),  # Lime
        ("#FFEDD5", "#EA580C"),  # Orange
        ("#E0F2FE", "#0284C7"),  # Sky
    ]
    hard_face = "#E2E8F0"
    hard_edge = "#475569"

    for i, b in blocks.items():
        if b.is_hard:
            face_col = hard_face
            edge_col = hard_edge
            alpha = 0.92
        else:
            face_col, edge_col = soft_palette[i % len(soft_palette)]
            alpha = 0.90

        rect = patches.Rectangle(
            (b.x, b.y),
            b.w,
            b.h,
            linewidth=1.25,
            edgecolor=edge_col,
            facecolor=face_col,
            alpha=alpha,
            zorder=2,
        )
        ax.add_patch(rect)

        # Text label (always dark ink on light block fills for crisp readability)
        cx = b.x + b.w / 2.0
        cy = b.y + b.h / 2.0
        label_text = f"{b.name}"
        if show_aspect_ratios:
            label_text += f"\n{b.w:.1f} x {b.h:.1f}\nr={b.aspect_ratio():.2f}"

        fs = 8 if len(blocks) < 18 else (6.5 if len(blocks) < 35 else 5.5)
        ax.text(
            cx, cy, label_text,
            ha="center", va="center",
            fontsize=fs,
            fontweight="bold" if b.is_hard else "normal",
            color="#0F172A",
            zorder=3,
        )

    max_x = max(upper_bound_width, layout_width)
    max_y = max(outline_h, layout_height)
    margin_x = max(max_x * 0.08, 1.0)
    margin_y = max(max_y * 0.08, 1.0)
    ax.set_xlim(-margin_x * 0.4, max_x + margin_x)
    ax.set_ylim(-margin_y * 0.4, max_y + margin_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (Width)", fontsize=10, fontweight="bold", color=text_muted)
    ax.set_ylabel("Y (Height)", fontsize=10, fontweight="bold", color=text_muted)
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=text_primary, pad=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(spine_col)
    ax.spines["bottom"].set_color(spine_col)
    ax.tick_params(colors=text_muted, labelsize=9)
    ax.grid(True, linestyle=":", color=spine_col, alpha=0.55, zorder=0)
    legend = ax.legend(loc="upper right", frameon=True, facecolor=legend_bg, edgecolor=legend_edge, fontsize=8.5)
    for txt in legend.get_texts():
        txt.set_color(text_primary)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_convergence(
    history: List[IterationRecord],
    optimal_height: Optional[float] = None,
    title: str = "SDS Layout-Height Convergence",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot layout height vs iteration number, replicating Figure 5 in Yan & Chu (ISPD 2012).
    Includes convergence markers when (y - Y) / Y drops below 10%, 5%, 1%, and 0.1%.
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    iters = [rec.iteration for rec in history]
    heights = [rec.layout_height for rec in history]
    Y = optimal_height if optimal_height is not None else heights[-1]

    ax.plot(iters, heights, color="#D32F2F", linewidth=1.8, label="Layout Height $y_{n+1}$")

    # Threshold markers: 10%, 5%, 1%, 0.1%
    thresholds = [
        (0.10, "10%", "#1976D2"),
        (0.05, "5%", "#388E3C"),
        (0.01, "1%", "#F57C00"),
        (0.001, "0.1%", "#7B1FA2"),
    ]

    for pct, label, col in thresholds:
        target_val = Y * (1.0 + pct)
        # Find first iteration where height <= target_val
        for rec in history:
            if rec.layout_height <= target_val + 1e-6:
                ax.axhline(target_val, color=col, linestyle="--", alpha=0.4, linewidth=0.9)
                ax.plot(rec.iteration, rec.layout_height, marker="o", markersize=6, color=col)
                ax.annotate(
                    f"< {label} (Iter {rec.iteration})",
                    xy=(rec.iteration, rec.layout_height),
                    xytext=(rec.iteration + len(iters) * 0.03, rec.layout_height + (heights[0] - Y) * 0.03),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.2),
                    fontsize=8,
                    fontweight="bold",
                    color=col,
                )
                break

    ax.axhline(Y, color="#212121", linestyle=":", linewidth=1.5, label=f"Optimal Height $Y={Y:.4f}$")

    ax.set_xlabel("Iteration Number", fontsize=11, fontweight="bold")
    ax.set_ylabel("Layout Height", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig
