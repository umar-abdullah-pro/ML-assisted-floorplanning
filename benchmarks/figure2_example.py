"""Exact reproduction of the motivational counterexample from Figure 2 of Yan & Chu (ISPD 2012).

Layout has 4 soft blocks (1, 2, 3, 4):
- Area A_i = 4, W_i^min = 1, W_i^max = 4 for all i in {1, 2, 3, 4}.
- Upper bound width W = 5.
- Initial layout: w1 = w3 = 4 (h1 = h3 = 1), w2 = w4 = 1 (h2 = h4 = 4).
- Basic SDS gets stuck at y_{n+1} = 5.0 because all blocks are in Subset III.
- Optimality checker identifies soft_hcps = 2, soft_vcps = 2, hard_vcps = 0 (Lemmas 2 & 3 fail).
- Full SDS / Geometric programming achieves optimal y_{n+1} = 3.2 with w1 = w2 = w3 = w4 = 2.5.
"""

from typing import Dict, Tuple
from sds.block import Block
from sds.constraint_graph import ConstraintGraph


def build_figure2_benchmark() -> Tuple[Dict[int, Block], ConstraintGraph, ConstraintGraph, float]:
    """Construct the benchmark from Figure 2.

    Returns:
        (blocks, Gh, Gv, W)
    """
    blocks = {
        1: Block(id=1, name="B1", area=4.0, is_hard=False, w_min=1.0, w_max=4.0, h_min=1.0, h_max=4.0),
        2: Block(id=2, name="B2", area=4.0, is_hard=False, w_min=1.0, w_max=4.0, h_min=1.0, h_max=4.0),
        3: Block(id=3, name="B3", area=4.0, is_hard=False, w_min=1.0, w_max=4.0, h_min=1.0, h_max=4.0),
        4: Block(id=4, name="B4", area=4.0, is_hard=False, w_min=1.0, w_max=4.0, h_min=1.0, h_max=4.0),
    }

    # Set initial shapes as in Figure 2-(a):
    blocks[1].set_width(4.0)  # h1 = 1.0
    blocks[2].set_width(1.0)  # h2 = 4.0
    blocks[3].set_width(4.0)  # h3 = 1.0
    blocks[4].set_width(1.0)  # h4 = 4.0

    gh = ConstraintGraph(num_blocks=4, name="Gh_Fig2")
    # Paths in Gh: 0 -> 4 -> 1 -> 5, and 0 -> 3 -> 2 -> 5
    gh.add_edge(0, 4)
    gh.add_edge(4, 1)
    gh.add_edge(1, 5)

    gh.add_edge(0, 3)
    gh.add_edge(3, 2)
    gh.add_edge(2, 5)
    gh.finalize_source_sink()

    gv = ConstraintGraph(num_blocks=4, name="Gv_Fig2")
    # Paths in Gv: 0 -> 3 -> 4 -> 5, and 0 -> 2 -> 1 -> 5
    gv.add_edge(0, 3)
    gv.add_edge(3, 4)
    gv.add_edge(4, 5)

    gv.add_edge(0, 2)
    gv.add_edge(2, 1)
    gv.add_edge(1, 5)
    gv.finalize_source_sink()

    W = 5.0
    return blocks, gh, gv, W

