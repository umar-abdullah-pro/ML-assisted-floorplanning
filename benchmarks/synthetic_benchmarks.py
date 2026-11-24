"""Synthetic scalable floorplan benchmark generator.
Generates test cases matching the characteristics of HB benchmarks in Section 6.2 (Table 4)
with both hard and soft blocks, realistic area distributions, and fixed outlines.
"""

from __future__ import annotations
import math
import random
from typing import Dict, Tuple

from sds.block import Block
from sds.constraint_graph import ConstraintGraph
from sds.sequence_pair import sequence_pair_to_constraint_graphs


def generate_hb_style_benchmark(
    num_soft: int = 50,
    num_hard: int = 20,
    aspect_ratio_range: Tuple[float, float] = (1.0 / 3.0, 3.0),
    area_min: float = 10.0,
    area_max: float = 200.0,
    hard_macro_multiplier: float = 4.0,
    seed: int = 42,
) -> Tuple[Dict[int, Block], ConstraintGraph, ConstraintGraph, float]:
    """Generate a floorplanning instance with hard and soft blocks matching Section 6.2.

    Upper-bound width W is set as sqrt(110% of total block area), as specified in Section 6.2.

    Args:
        num_soft: Number of soft blocks.
        num_hard: Number of hard macro blocks.
        aspect_ratio_range: Range of aspect ratio for soft blocks (default [1/3, 3]).
        area_min: Minimum area for soft blocks.
        area_max: Maximum area for soft blocks.
        hard_macro_multiplier: Scaling factor for hard macro sizes.
        seed: Random seed.

    Returns:
        (blocks, Gh, Gv, W)
    """
    rng = random.Random(seed)
    total_blocks = num_soft + num_hard
    blocks: Dict[int, Block] = {}

    block_id = 1
    total_area = 0.0

    # 1. Soft blocks
    for _ in range(num_soft):
        area = rng.uniform(area_min, area_max)
        b = Block(
            id=block_id,
            name=f"S{block_id}",
            area=area,
            is_hard=False,
            aspect_ratio_range=aspect_ratio_range,
        )
        blocks[block_id] = b
        total_area += area
        block_id += 1

    # 2. Hard blocks (macros with fixed dimensions)
    for _ in range(num_hard):
        area = rng.uniform(area_min * hard_macro_multiplier, area_max * hard_macro_multiplier)
        # Random aspect ratio for hard block fixed shape
        r = rng.uniform(0.5, 2.0)
        w = math.sqrt(area / r)
        h = math.sqrt(area * r)
        b = Block(
            id=block_id,
            name=f"H{block_id}",
            is_hard=True,
            w_fixed=w,
            h_fixed=h,
        )
        blocks[block_id] = b
        total_area += area
        block_id += 1

    # Sequence pair generation
    seq_x = list(range(1, total_blocks + 1))
    seq_y = list(range(1, total_blocks + 1))
    rng.shuffle(seq_x)
    rng.shuffle(seq_y)

    gh, gv = sequence_pair_to_constraint_graphs(seq_x, seq_y, total_blocks, apply_reduction=(total_blocks <= 250))

    # Paper Section 6.2: "we set the upper-bound width W as the square root of 110% of the total block area"
    W = math.sqrt(1.10 * total_area)

    # Ensure initial width at min width does not exceed W for feasibility
    w_min_dict = {i: blocks[i].w_min for i in blocks}
    initial_w = gh.compute_longest_path_forward(w_min_dict)[gh.sink]
    if initial_w > W:
        W = initial_w * 1.05

    return blocks, gh, gv, W

