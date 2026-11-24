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
) -> plt.Figure:
    """Plot floorplan layout showing all blocks, bounding outline, and dimensions."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)

    # Outline rectangle (W x H)
    outline = patches.Rectangle(
        (0, 0),
        upper_bound_width,
        layout_height,
        linewidth=2.0,
        edgecolor="#D32F2F",
        facecolor="none",
        linestyle="--",
        label=f"Fixed Outline (W={upper_bound_width:.2f}, H={layout_height:.2f})",
        zorder=1,
    )
    ax.add_patch(outline)

    # Palette
    try:
        import matplotlib
        soft_cmap = matplotlib.colormaps["Pastel1"]
    except Exception:
        soft_cmap = plt.cm.get_cmap("Pastel1")
    hard_color = "#9E9E9E"

    for i, b in blocks.items():
        if b.is_hard:
            face_col = hard_color
            edge_col = "#424242"
            alpha = 0.8
        else:
            face_col = soft_cmap(i % 12)
            edge_col = "#1565C0"
            alpha = 0.75

        rect = patches.Rectangle(
            (b.x, b.y),
            b.w,
            b.h,
            linewidth=1.2,
            edgecolor=edge_col,
            facecolor=face_col,
            alpha=alpha,
            zorder=2,
        )
        ax.add_patch(rect)

        # Text label
        cx = b.x + b.w / 2.0
        cy = b.y + b.h / 2.0
        label_text = f"{b.name}"
        if show_aspect_ratios:
            label_text += f"\n{b.w:.2f}x{b.h:.2f}\nr={b.aspect_ratio():.2f}"

        # Font size adjusted by block size
        fs = 8 if len(blocks) < 20 else 6
        ax.text(
            cx, cy, label_text,
            ha="center", va="center",
            fontsize=fs,
            fontweight="bold" if b.is_hard else "normal",
            color="#212121",
            zorder=3,
        )

    margin_x = max(upper_bound_width * 0.08, 1.0)
    margin_y = max(layout_height * 0.08, 1.0)
    ax.set_xlim(-margin_x * 0.5, upper_bound_width + margin_x)
    ax.set_ylim(-margin_y * 0.5, layout_height + margin_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (Width)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Y (Height)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)

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
