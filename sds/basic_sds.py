"""Basic Slack-Driven Shaping (Basic SDS) algorithm implementation.
Corresponds to Section 3 and Figure 1 of Yan & Chu (ISPD 2012).
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

from .block import Block
from .constraint_graph import ConstraintGraph


@dataclass
class IterationRecord:
    """Record of a single SDS iteration for analysis and visualization."""
    iteration: int
    layout_width: float
    layout_height: float
    beta: float
    subset_counts: Dict[str, int]
    target_iv_count: int
    target_vi_count: int
    max_delta_h: float
    max_delta_v: float


@dataclass
class SDSResult:
    """Result returned by SDS solver."""
    feasible: bool
    optimal: bool
    layout_width: float
    layout_height: float
    total_iterations: int
    runtime_seconds: float
    blocks: Dict[int, Block]
    history: List[IterationRecord] = field(default_factory=list)
    final_subsets: Dict[str, List[int]] = field(default_factory=dict)
    message: str = ""


class BasicSDSSolver:
    """Basic Slack-Driven Shaping solver for fixed-outline floorplanning."""

    def __init__(
        self,
        blocks: Dict[int, Block],
        gh: ConstraintGraph,
        gv: ConstraintGraph,
        upper_bound_width: float,
        slack_epsilon: float = 1e-6,
        delta_threshold: float = 0.0001,
    ):
        """Initialize solver.

        Args:
            blocks: Dict of blocks indexed 1..n.
            gh: Horizontal constraint graph (with source 0 and sink n+1).
            gv: Vertical constraint graph (with source 0 and sink n+1).
            upper_bound_width: Target upper bound W on layout width.
            slack_epsilon: Numerical tolerance for zero slack (s == 0).
            delta_threshold: Minimum change on block dimension (0.0001 in paper).
        """
        self.blocks = {i: b.clone() for i, b in blocks.items()}
        self.num_blocks = len(self.blocks)
        self.gh = gh
        self.gv = gv
        self.W = float(upper_bound_width)
        self.slack_eps = slack_epsilon
        self.delta_threshold = delta_threshold

        # Positions and slacks
        self.x_ll: Dict[int, float] = {}
        self.x_rl: Dict[int, float] = {}
        self.y_bl: Dict[int, float] = {}
        self.y_tl: Dict[int, float] = {}
        self.layout_width: float = 0.0
        self.layout_height: float = 0.0

        # Subsets I through VII
        self.subsets: Dict[str, List[int]] = {
            "I": [], "II": [], "III": [], "IV": [], "V": [], "VI": [], "VII": []
        }

    def _pack_ll(self) -> float:
        """Pack blocks to Left Line (x = 0). Returns x_{n+1}."""
        w_dict = {i: self.blocks[i].w for i in self.blocks}
        self.x_ll = self.gh.compute_longest_path_forward(w_dict, start_val=0.0)
        self.layout_width = self.x_ll[self.gh.sink]
        return self.layout_width

    def _pack_rl(self) -> None:
        """Pack blocks to Right Line (x = W)."""
        w_dict = {i: self.blocks[i].w for i in self.blocks}
        self.x_rl = self.gh.compute_longest_path_backward(w_dict, end_val=self.W)

    def _pack_bl(self) -> float:
        """Pack blocks to Bottom Line (y = 0). Returns y_{n+1}."""
        h_dict = {i: self.blocks[i].h for i in self.blocks}
        self.y_bl = self.gv.compute_longest_path_forward(h_dict, start_val=0.0)
        self.layout_height = self.y_bl[self.gv.sink]
        return self.layout_height

    def _pack_tl(self) -> None:
        """Pack blocks to Top Line (y = y_{n+1})."""
        h_dict = {i: self.blocks[i].h for i in self.blocks}
        self.y_tl = self.gv.compute_longest_path_backward(h_dict, end_val=self.layout_height)

    def compute_slacks(self) -> None:
        """Pack to all 4 lines and compute horizontal and vertical slacks for all blocks."""
        self._pack_bl()
        self._pack_tl()
        self._pack_ll()
        self._pack_rl()

        for i, b in self.blocks.items():
            delta_x = self.x_rl[i] - self.x_ll[i]
            delta_y = self.y_tl[i] - self.y_bl[i]

            b.slack_h = max(0.0, delta_x) if delta_x > self.slack_eps else 0.0
            b.slack_v = max(0.0, delta_y) if delta_y > self.slack_eps else 0.0
            b.x = self.x_ll[i]
            b.y = self.y_bl[i]

    def classify_subsets(self) -> Dict[str, List[int]]:
        """Classify blocks into 7 disjoint subsets (Section 3.1):
        Subset I: Hard blocks
        Subset II: Soft, sh > 0, sv > 0 (not on HCP nor VCP)
        Subset III: Soft, sh = 0, sv = 0 (intersection of HCP and VCP)
        Subset IV: Soft, sh > 0, sv = 0, wi < W_i^max (target for width increase)
        Subset V: Soft, sh > 0, sv = 0, wi >= W_i^max
        Subset VI: Soft, sh = 0, sv > 0, hi < H_i^max (target for height increase)
        Subset VII: Soft, sh = 0, sv > 0, hi >= H_i^max
        """
        for s in self.subsets:
            self.subsets[s].clear()

        for i, b in self.blocks.items():
            if b.is_hard:
                self.subsets["I"].append(i)
                b.subset = "I"
            else:
                sh_zero = (b.slack_h <= self.slack_eps)
                sv_zero = (b.slack_v <= self.slack_eps)

                if not sh_zero and not sv_zero:
                    self.subsets["II"].append(i)
                    b.subset = "II"
                elif sh_zero and sv_zero:
                    self.subsets["III"].append(i)
                    b.subset = "III"
                elif not sh_zero and sv_zero:
                    # On VCP, not on HCP
                    if b.w < b.w_max - self.slack_eps:
                        self.subsets["IV"].append(i)
                        b.subset = "IV"
                    else:
                        self.subsets["V"].append(i)
                        b.subset = "V"
                else:
                    # sh_zero and not sv_zero: on HCP, not on VCP
                    if b.h < b.h_max - self.slack_eps:
                        self.subsets["VI"].append(i)
                        b.subset = "VI"
                    else:
                        self.subsets["VII"].append(i)
                        b.subset = "VII"

        return self.subsets

    def compute_deltas(self, beta: float) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Compute delta_j^h for Subset IV and delta_j^v for Subset VI using linear DP."""
        deltas_h: Dict[int, float] = {}
        deltas_v: Dict[int, float] = {}

        # 1. Delta_h for Subset IV (widening)
        subset_iv = self.subsets["IV"]
        if subset_iv:
            # DP weights: W_k^max - w_k for soft blocks, 0 for hard / dummy
            weights_w = {}
            for i in range(1, self.num_blocks + 1):
                b = self.blocks[i]
                weights_w[i] = 0.0 if b.is_hard else max(0.0, b.w_max - b.w)

            dp_max_h = self.gh.compute_dp_max_path(weights_w)

            for j in subset_iv:
                b = self.blocks[j]
                rem_w = max(0.0, b.w_max - b.w)
                denom = dp_max_h.get(j, rem_w)
                if denom > 1e-12 and rem_w > 0.0:
                    delta = (rem_w * b.slack_h) / denom
                    delta = min(delta, rem_w)
                    if delta >= self.delta_threshold:
                        deltas_h[j] = delta

        # 2. Delta_v for Subset VI (heightening)
        subset_vi = self.subsets["VI"]
        if subset_vi:
            # DP weights: H_k^max - h_k for soft blocks, 0 for hard / dummy
            weights_h = {}
            for i in range(1, self.num_blocks + 1):
                b = self.blocks[i]
                weights_h[i] = 0.0 if b.is_hard else max(0.0, b.h_max - b.h)

            dp_max_v = self.gv.compute_dp_max_path(weights_h)

            for j in subset_vi:
                b = self.blocks[j]
                rem_h = max(0.0, b.h_max - b.h)
                denom = dp_max_v.get(j, rem_h)
                if denom > 1e-12 and rem_h > 0.0:
                    delta = beta * ((rem_h * b.slack_v) / denom)
                    delta = min(delta, rem_h)
                    if delta >= self.delta_threshold:
                        deltas_v[j] = delta

        return deltas_h, deltas_v

    def solve(
        self,
        reset_to_min: bool = True,
        max_iterations: int = 5000,
        early_stop_height: Optional[float] = None,
        verbose: bool = False,
    ) -> SDSResult:
        """Run the Basic Slack-Driven Shaping algorithm (Figure 1)."""
        start_time = time.perf_counter()

        # Step 1: Initialize soft blocks to minimum width (Section 3.3)
        if reset_to_min:
            for b in self.blocks.values():
                if not b.is_hard:
                    b.reset_to_min_width()

        # Step 2: Pack to LL and check initial feasibility
        init_width = self._pack_ll()
        if init_width > self.W + self.slack_eps:
            elapsed = time.perf_counter() - start_time
            return SDSResult(
                feasible=False,
                optimal=False,
                layout_width=init_width,
                layout_height=self._pack_bl(),
                total_iterations=0,
                runtime_seconds=elapsed,
                blocks=self.blocks,
                message=f"Infeasible: Initial width {init_width:.4f} exceeds upper-bound W={self.W:.4f}",
            )

        history: List[IterationRecord] = []
        beta = 1.0
        stalled_count = 0
        last_height = float("inf")

        for it in range(1, max_iterations + 1):
            # Step 7-10: Compute slacks
            self.compute_slacks()

            # Check early stopping condition (e.g. fixed outline satisfied)
            if early_stop_height is not None and self.layout_height <= early_stop_height:
                if verbose:
                    print(f"Early stop reached at iteration {it}: height {self.layout_height:.4f} <= {early_stop_height:.4f}")
                break

            # Step 11: Find target soft blocks
            self.classify_subsets()

            # Adaptive damping beta control:
            # "whenever we detect that y_{n+1} does not change for more than two iterations,
            # we will set beta = 90% for the next iteration." (Section 3.3)
            if abs(last_height - self.layout_height) < self.delta_threshold:
                stalled_count += 1
            else:
                stalled_count = 0
                beta = 1.0

            if stalled_count >= 2:
                beta = 0.9

            last_height = self.layout_height

            # Step 12-14: Compute shaping amounts
            deltas_h, deltas_v = self.compute_deltas(beta)

            record = IterationRecord(
                iteration=it,
                layout_width=self.layout_width,
                layout_height=self.layout_height,
                beta=beta,
                subset_counts={k: len(v) for k, v in self.subsets.items()},
                target_iv_count=len(deltas_h),
                target_vi_count=len(deltas_v),
                max_delta_h=max(deltas_h.values(), default=0.0),
                max_delta_v=max(deltas_v.values(), default=0.0),
            )
            history.append(record)

            if verbose and (it <= 5 or it % 50 == 0):
                print(
                    f"Iter {it:4d}: W={self.layout_width:8.4f}/{self.W:8.4f}, "
                    f"H={self.layout_height:8.4f}, beta={beta:.2f}, "
                    f"IV={len(deltas_h):2d}, VI={len(deltas_v):2d}"
                )

            # Step 15: Stopping condition: no target soft block can be shaped
            if not deltas_h and not deltas_v:
                if verbose:
                    print(f"Basic SDS converged at iteration {it}: no target soft blocks.")
                break

            # Apply updates
            for j, dh in deltas_h.items():
                self.blocks[j].set_width(self.blocks[j].w + dh)

            for j, dv in deltas_v.items():
                self.blocks[j].set_height(self.blocks[j].h + dv)

        # Final packing and classification
        self.compute_slacks()
        self.classify_subsets()
        elapsed = time.perf_counter() - start_time

        return SDSResult(
            feasible=True,
            optimal=False,  # Checked by OptimalityChecker / Full SDS
            layout_width=self.layout_width,
            layout_height=self.layout_height,
            total_iterations=len(history),
            runtime_seconds=elapsed,
            blocks=self.blocks,
            history=history,
            final_subsets=dict(self.subsets),
            message="Basic SDS completed.",
        )
