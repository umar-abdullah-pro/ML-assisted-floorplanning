"""Sequence-Pair representation and conversion to constraint graphs Gh and Gv."""

from __future__ import annotations
from typing import List, Tuple, Set
from .constraint_graph import ConstraintGraph


def transitive_reduction(edges: Set[Tuple[int, int]], nodes: List[int]) -> Set[Tuple[int, int]]:
    """Perform transitive reduction on a DAG to eliminate redundant edges."""
    # Build adj list
    adj = {u: set() for u in nodes}
    for u, v in edges:
        adj[u].add(v)

    # In a DAG, an edge (u, v) is redundant if there exists an alternate path of length >= 2 from u to v
    # For moderate n, reachability from each node via DFS
    reduced: Set[Tuple[int, int]] = set()

    for u in nodes:
        for v in adj[u]:
            # Check if v is reachable from u without the direct edge (u, v)
            visited = set()
            stack = [w for w in adj[u] if w != v]
            reachable = False
            while stack:
                curr = stack.pop()
                if curr == v:
                    reachable = True
                    break
                if curr not in visited:
                    visited.add(curr)
                    stack.extend(adj[curr])
            if not reachable:
                reduced.add((u, v))

    return reduced


def sequence_pair_to_constraint_graphs(
    seq_x: List[int],
    seq_y: List[int],
    num_blocks: int,
    apply_reduction: bool = True,
) -> Tuple[ConstraintGraph, ConstraintGraph]:
    """Convert a Sequence Pair (X, Y) into horizontal and vertical constraint graphs.

    Rule:
    - (..., a, ..., b, ...) in both X and Y  =>  a is to the left of b (a -> b in Gh)
    - (..., b, ..., a, ...) in X and (..., a, ..., b, ...) in Y => a is below b (a -> b in Gv)

    Args:
        seq_x: Permutation of block IDs [1..num_blocks].
        seq_y: Permutation of block IDs [1..num_blocks].
        num_blocks: Number of blocks.
        apply_reduction: Whether to apply transitive reduction.

    Returns:
        (Gh, Gv) constraint graphs finalized with source 0 and sink n+1.
    """
    pos_x = {block_id: idx for idx, block_id in enumerate(seq_x)}
    pos_y = {block_id: idx for idx, block_id in enumerate(seq_y)}

    h_edges: Set[Tuple[int, int]] = set()
    v_edges: Set[Tuple[int, int]] = set()

    for i in range(1, num_blocks + 1):
        for j in range(i + 1, num_blocks + 1):
            if pos_x[i] < pos_x[j] and pos_y[i] < pos_y[j]:
                # i before j in both X and Y => i left of j
                h_edges.add((i, j))
            elif pos_x[i] > pos_x[j] and pos_y[i] > pos_y[j]:
                # j before i in both X and Y => j left of i
                h_edges.add((j, i))
            elif pos_x[i] < pos_x[j] and pos_y[i] > pos_y[j]:
                # i before j in X, j before i in Y => j below i (j -> i in Gv)
                v_edges.add((j, i))
            else:
                # j before i in X, i before j in Y => i below j (i -> j in Gv)
                v_edges.add((i, j))

    block_nodes = list(range(1, num_blocks + 1))
    if apply_reduction and num_blocks <= 250:
        h_edges = transitive_reduction(h_edges, block_nodes)
        v_edges = transitive_reduction(v_edges, block_nodes)

    gh = ConstraintGraph(num_blocks, name="Gh")
    gv = ConstraintGraph(num_blocks, name="Gv")

    for u, v in h_edges:
        gh.add_edge(u, v)
    for u, v in v_edges:
        gv.add_edge(u, v)

    gh.finalize_source_sink()
    gv.finalize_source_sink()

    return gh, gv

