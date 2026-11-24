"""High-level API for integrating SDS into fixed-outline floorplanning and placement optimizers.
Convenient for Simulated Annealing, Q-Learning controllers, and RL agents.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, Any

from .block import Block
from .constraint_graph import ConstraintGraph
from .sequence_pair import sequence_pair_to_constraint_graphs
from .full_sds import FullSDSSolver, FullSDSResult


def compute_hpwl(
    blocks: Dict[int, Block],
    nets: List[List[Union[int, str]]],
) -> float:
    """Compute Half-Perimeter Wirelength (HPWL) based on block centers.

    Args:
        blocks: Dict of blocks with coordinates x, y and dimensions w, h.
        nets: List of nets, where each net is a list of block IDs or names.

    Returns:
        Total HPWL wirelength.
    """
    id_map = {}
    for i, b in blocks.items():
        id_map[i] = b
        id_map[b.name] = b

    total_hpwl = 0.0
    for net in nets:
        if len(net) <= 1:
            continue
        xs = []
        ys = []
        for item in net:
            if item in id_map:
                b = id_map[item]
                cx = b.x + b.w / 2.0
                cy = b.y + b.h / 2.0
                xs.append(cx)
                ys.append(cy)
        if xs and ys:
            total_hpwl += (max(xs) - min(xs)) + (max(ys) - min(ys))

    return total_hpwl


def shape_floorplan(
    block_specs: List[Dict[str, Any]],
    seq_x: Optional[List[int]] = None,
    seq_y: Optional[List[int]] = None,
    gh: Optional[ConstraintGraph] = None,
    gv: Optional[ConstraintGraph] = None,
    upper_bound_width: float = 100.0,
    outline_height: Optional[float] = None,
    nets: Optional[List[List[Any]]] = None,
    max_iterations: int = 2000,
) -> Dict[str, Any]:
    """All-in-one function to perform Optimal Slack-Driven Shaping on a floorplan candidate.

    Args:
        block_specs: List of dicts specifying blocks:
            [
                {"id": 1, "area": 100.0, "is_hard": False, "aspect_ratio_range": (1/3, 3.0)},
                {"id": 2, "w_fixed": 10.0, "h_fixed": 15.0, "is_hard": True},
                ...
            ]
        seq_x: Sequence pair X permutation (if Gh and Gv are not supplied).
        seq_y: Sequence pair Y permutation (if Gh and Gv are not supplied).
        gh: Pre-constructed horizontal constraint graph (optional).
        gv: Pre-constructed vertical constraint graph (optional).
        upper_bound_width: Fixed outline chip width W.
        outline_height: Fixed outline chip height H (optional, for legality checking).
        nets: Optional netlist for HPWL wirelength computation.
        max_iterations: Maximum SDS iterations.

    Returns:
        dict containing:
            - 'feasible': bool (fits width W)
            - 'legal': bool (fits inside W x H outline without overlap)
            - 'layout_width': float
            - 'layout_height': float
            - 'hpwl': float (if nets provided)
            - 'optimal': bool
            - 'iterations': int
            - 'runtime_seconds': float
            - 'blocks': Dict[int, Block]
            - 'block_positions': Dict[int, Dict[str, float]]
    """
    # 1. Instantiate blocks
    blocks: Dict[int, Block] = {}
    for spec in block_specs:
        b_id = spec["id"]
        is_hard = spec.get("is_hard", False)
        b = Block(
            id=b_id,
            name=spec.get("name", f"B{b_id}"),
            area=spec.get("area", 1.0),
            is_hard=is_hard,
            aspect_ratio_range=spec.get("aspect_ratio_range", (1.0 / 3.0, 3.0)),
            w_fixed=spec.get("w_fixed"),
            h_fixed=spec.get("h_fixed"),
            w_min=spec.get("w_min"),
            w_max=spec.get("w_max"),
            h_min=spec.get("h_min"),
            h_max=spec.get("h_max"),
        )
        blocks[b_id] = b

    n = len(blocks)

    # 2. Derive constraint graphs if not provided
    if gh is None or gv is None:
        if seq_x is None or seq_y is None:
            raise ValueError("Either (gh, gv) or (seq_x, seq_y) must be provided.")
        gh, gv = sequence_pair_to_constraint_graphs(seq_x, seq_y, n, apply_reduction=(n <= 250))

    # 3. Run Full SDS solver
    solver = FullSDSSolver(blocks, gh, gv, upper_bound_width=upper_bound_width)
    result = solver.solve(
        max_basic_iterations=max_iterations,
        reset_to_min=True,
    )

    # 4. Check legality against outline height
    is_legal = result.feasible
    if outline_height is not None:
        is_legal = is_legal and (result.layout_height <= outline_height + 1e-4)

    # 5. Compute HPWL if netlist is supplied
    hpwl_val = 0.0
    if nets is not None:
        hpwl_val = compute_hpwl(result.blocks, nets)

    block_positions = {}
    for i, b in result.blocks.items():
        block_positions[i] = {
            "x": b.x,
            "y": b.y,
            "w": b.w,
            "h": b.h,
            "area": b.area,
            "aspect_ratio": b.aspect_ratio(),
            "is_hard": b.is_hard,
        }

    return {
        "feasible": result.feasible,
        "legal": is_legal,
        "layout_width": result.layout_width,
        "layout_height": result.layout_height,
        "hpwl": hpwl_val,
        "optimal": result.optimal,
        "iterations": result.total_iterations,
        "runtime_seconds": result.runtime_seconds,
        "gp_fallback_used": result.gp_fallback_used,
        "blocks": result.blocks,
        "block_positions": block_positions,
    }

