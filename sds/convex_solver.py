"""Exact Geometric Programming / Convex Optimization solver using CVXPY.
Used as the GP fallback in Section 5 of Yan & Chu (ISPD 2012) and for exact ground-truth benchmarking.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import cvxpy as cp
import numpy as np

from .block import Block
from .constraint_graph import ConstraintGraph


@dataclass
class ConvexSolverResult:
    status: str
    is_optimal: bool
    optimal_height: float
    optimal_width: float
    widths: Dict[int, float]
    heights: Dict[int, float]
    x_coords: Dict[int, float]
    y_coords: Dict[int, float]
    solver_runtime: float


def solve_floorplan_convex(
    blocks: Dict[int, Block],
    gh: ConstraintGraph,
    gv: ConstraintGraph,
    upper_bound_width: float,
    target_block_ids: Optional[List[int]] = None,
    solver: Optional[str] = None,
) -> ConvexSolverResult:
    """Solve Problem 1 (or the intersection-block subproblem) via Disciplined Convex Programming (DCP).

    Formulation:
        Minimize y_{n+1}
        Subject to:
            x_{n+1} <= W
            x_j >= x_i + w_i    for all (i, j) in Gh
            y_j >= y_i + h_i    for all (i, j) in Gv
            h_i >= A_i * inv_pos(w_i)   for soft blocks
            W_i^min <= w_i <= W_i^max
            H_i^min <= h_i <= H_i^max
            x_0 = 0, y_0 = 0

    Args:
        blocks: Dict of blocks 1..n.
        gh: Horizontal constraint graph.
        gv: Vertical constraint graph.
        upper_bound_width: Upper bound W on width.
        target_block_ids: If provided, only allow these soft blocks to change shape;
                          all other blocks remain fixed at their current dimensions.
        solver: CVXPY solver name (e.g. 'CLARABEL', 'SCS', or default).
    """
    n = len(blocks)
    total_nodes = n + 2  # 0 to n+1

    # Position variables
    x = cp.Variable(total_nodes, name="x")
    y = cp.Variable(total_nodes, name="y")

    # Width and height expressions for each block 1..n
    w_expr = {}
    h_expr = {}

    constraints = [
        x[gh.source] == 0,
        y[gv.source] == 0,
        x[gh.sink] <= upper_bound_width,
    ]

    target_set = set(target_block_ids) if target_block_ids is not None else set(blocks.keys())

    for i, b in blocks.items():
        if b.is_hard or (i not in target_set):
            # Fixed block
            w_val = float(b.w)
            h_val = float(b.h)
            w_expr[i] = w_val
            h_expr[i] = h_val
        else:
            # Soft block variable
            w_var = cp.Variable(pos=True, name=f"w_{i}")
            h_var = cp.Variable(pos=True, name=f"h_{i}")

            constraints.append(w_var >= b.w_min)
            constraints.append(w_var <= b.w_max)
            constraints.append(h_var >= b.h_min)
            constraints.append(h_var <= b.h_max)
            # Convex relaxation: h_i >= A_i / w_i.
            # At minimum height y_{n+1}, this constraint is guaranteed tight.
            constraints.append(h_var >= b.area * cp.inv_pos(w_var))

            w_expr[i] = w_var
            h_expr[i] = h_var

    # Horizontal non-overlapping constraints: x_j >= x_i + w_i
    for u, v in gh.edges:
        if u == gh.source:
            constraints.append(x[v] >= x[u])
        elif u == gh.sink:
            pass
        else:
            constraints.append(x[v] >= x[u] + w_expr[u])

    # Vertical non-overlapping constraints: y_j >= y_i + h_i
    for u, v in gv.edges:
        if u == gv.source:
            constraints.append(y[v] >= y[u])
        elif u == gv.sink:
            pass
        else:
            constraints.append(y[v] >= y[u] + h_expr[u])

    prob = cp.Problem(cp.Minimize(y[gv.sink]), constraints)

    try:
        prob.solve(solver=solver)
    except Exception:
        # Fallback to standard solver
        prob.solve()

    is_optimal = prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

    res_w = {}
    res_h = {}
    for i, b in blocks.items():
        if isinstance(w_expr[i], (int, float)):
            res_w[i] = float(w_expr[i])
            res_h[i] = float(h_expr[i])
        else:
            w_val = float(w_expr[i].value) if w_expr[i].value is not None else b.w
            res_w[i] = w_val
            res_h[i] = b.area / w_val if w_val > 0 else b.h

    res_x = {i: float(x[i].value) if x[i].value is not None else 0.0 for i in range(total_nodes)}
    res_y = {i: float(y[i].value) if y[i].value is not None else 0.0 for i in range(total_nodes)}

    opt_height = float(prob.value) if prob.value is not None else float("inf")
    opt_width = res_x.get(gh.sink, 0.0)

    return ConvexSolverResult(
        status=prob.status,
        is_optimal=is_optimal,
        optimal_height=opt_height,
        optimal_width=opt_width,
        widths=res_w,
        heights=res_h,
        x_coords=res_x,
        y_coords=res_y,
        solver_runtime=getattr(prob.solver_stats, "solve_time", 0.0) or 0.0,
    )

