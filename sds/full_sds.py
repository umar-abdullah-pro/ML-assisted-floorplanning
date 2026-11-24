"""Full Slack-Driven Shaping (SDS) algorithm implementation.
Corresponds to Section 5 and Figure 4 of Yan & Chu (ISPD 2012).
Combines Basic SDS, Optimality Condition Verification, and Geometric Programming fallback.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .block import Block
from .constraint_graph import ConstraintGraph
from .basic_sds import BasicSDSSolver, SDSResult, IterationRecord
from .optimality import OptimalityChecker, OptimalityReport
from .convex_solver import solve_floorplan_convex


@dataclass
class FullSDSResult:
    """Complete result from Full SDS algorithm."""
    feasible: bool
    optimal: bool
    layout_width: float
    layout_height: float
    total_iterations: int
    runtime_seconds: float
    gp_fallback_used: bool
    optimality_report: Optional[OptimalityReport]
    blocks: Dict[int, Block]
    history: List[IterationRecord] = field(default_factory=list)
    message: str = ""


class FullSDSSolver:
    """Full Slack-Driven Shaping solver guaranteed to reach global optimality."""

    def __init__(
        self,
        blocks: Dict[int, Block],
        gh: ConstraintGraph,
        gv: ConstraintGraph,
        upper_bound_width: float,
        slack_epsilon: float = 1e-6,
        delta_threshold: float = 0.0001,
    ):
        self.blocks = {i: b.clone() for i, b in blocks.items()}
        self.gh = gh
        self.gv = gv
        self.W = float(upper_bound_width)
        self.slack_eps = slack_epsilon
        self.delta_threshold = delta_threshold

    def solve(
        self,
        reset_to_min: bool = True,
        max_outer_loops: int = 10,
        max_basic_iterations: int = 5000,
        verbose: bool = False,
    ) -> FullSDSResult:
        """Run Full SDS algorithm (Figure 4).

        1. Run Basic SDS until no target soft block can be shaped.
        2. Check optimality (Lemmas 2 and 3).
        3. If not optimal, apply Geometric Programming / Convex Optimization to
           intersection soft blocks (Subset III) and continue until optimal.
        """
        start_time = time.perf_counter()
        total_history: List[IterationRecord] = []
        gp_used = False
        optimality_rep: Optional[OptimalityReport] = None
        is_optimal_solution = False

        current_blocks = {i: b.clone() for i, b in self.blocks.items()}

        for outer in range(1, max_outer_loops + 1):
            basic_solver = BasicSDSSolver(
                blocks=current_blocks,
                gh=self.gh,
                gv=self.gv,
                upper_bound_width=self.W,
                slack_epsilon=self.slack_eps,
                delta_threshold=self.delta_threshold,
            )

            # Only reset on outer loop 1 if requested
            should_reset = (outer == 1 and reset_to_min)
            basic_res = basic_solver.solve(
                reset_to_min=should_reset,
                max_iterations=max_basic_iterations,
                verbose=verbose,
            )

            total_history.extend(basic_res.history)
            current_blocks = basic_res.blocks

            if not basic_res.feasible:
                elapsed = time.perf_counter() - start_time
                return FullSDSResult(
                    feasible=False,
                    optimal=False,
                    layout_width=basic_res.layout_width,
                    layout_height=basic_res.layout_height,
                    total_iterations=len(total_history),
                    runtime_seconds=elapsed,
                    gp_fallback_used=gp_used,
                    optimality_report=None,
                    blocks=current_blocks,
                    history=total_history,
                    message=basic_res.message,
                )

            # Check optimality conditions (Section 4)
            checker = OptimalityChecker(
                blocks=current_blocks,
                gh=self.gh,
                gv=self.gv,
                upper_bound_width=self.W,
                slack_epsilon=self.slack_eps,
            )
            optimality_rep = checker.check(
                x_ll=basic_solver.x_ll,
                y_bl=basic_solver.y_bl,
                subsets=basic_solver.subsets,
            )

            if verbose:
                print(f"[SDS Outer Loop {outer}] Basic SDS finished at H={basic_res.layout_height:.4f}, W={basic_res.layout_width:.4f}")
                print(f"[SDS Optimality] Optimal: {optimality_rep.is_optimal}. {optimality_rep.explanation}")

            if optimality_rep.is_optimal:
                # Certified optimal!
                is_optimal_solution = True
                break

            # If not optimal, apply Geometric Programming step to intersection soft blocks (Subset III)
            gp_used = True
            subset_iii = optimality_rep.subset_iii_blocks
            if verbose:
                print(f"[SDS Fallback] Running convex optimization on intersection soft blocks: {subset_iii}...")

            convex_res = solve_floorplan_convex(
                blocks=current_blocks,
                gh=self.gh,
                gv=self.gv,
                upper_bound_width=self.W,
                target_block_ids=subset_iii if subset_iii else None,
            )

            if convex_res.is_optimal:
                # Update block dimensions
                for i, w_val in convex_res.widths.items():
                    if not current_blocks[i].is_hard:
                        current_blocks[i].set_width(w_val)

                if verbose:
                    print(f"[SDS Fallback] Convex step found optimal height: {convex_res.optimal_height:.4f}.")

                # Figure 4, Lines 22-23: Geometric programming found optimal solution => L is optimal!
                is_optimal_solution = True
                break
            else:
                if verbose:
                    print(f"[SDS Fallback] Convex solver status: {convex_res.status}. Continuing...")

        elapsed = time.perf_counter() - start_time

        # Final packing
        final_solver = BasicSDSSolver(
            blocks=current_blocks,
            gh=self.gh,
            gv=self.gv,
            upper_bound_width=self.W,
            slack_epsilon=self.slack_eps,
            delta_threshold=self.delta_threshold,
        )
        final_solver.compute_slacks()
        final_solver.classify_subsets()

        is_opt = is_optimal_solution

        return FullSDSResult(
            feasible=True,
            optimal=is_opt,
            layout_width=final_solver.layout_width,
            layout_height=final_solver.layout_height,
            total_iterations=len(total_history),
            runtime_seconds=elapsed,
            gp_fallback_used=gp_used,
            optimality_report=optimality_rep,
            blocks=final_solver.blocks,
            history=total_history,
            message="Optimal shaping solution found." if is_opt else "SDS completed (near-optimal).",
        )
