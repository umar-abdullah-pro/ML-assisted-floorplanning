"""Constraint Graph representation and algorithms (longest path, DP, topological sort)."""

from __future__ import annotations
from collections import deque
from typing import Dict, List, Set, Tuple, Optional


class ConstraintGraph:
    """Directed Acyclic Graph (DAG) representing horizontal (Gh) or vertical (Gv) constraints.

    Vertices are indexed 0 to n+1:
        0: Source dummy node (x0 = 0 or y0 = 0, dimension = 0)
        1..n: Circuit blocks
        n+1: Sink dummy node (dimension = 0)
    """

    def __init__(self, num_blocks: int, name: str = "G"):
        self.num_blocks = num_blocks
        self.name = name
        self.source = 0
        self.sink = num_blocks + 1
        self.nodes = list(range(self.sink + 1))

        self.edges: Set[Tuple[int, int]] = set()
        self.in_edges: Dict[int, List[int]] = {i: [] for i in self.nodes}
        self.out_edges: Dict[int, List[int]] = {i: [] for i in self.nodes}
        self._topo_order: Optional[List[int]] = None

    def add_edge(self, u: int, v: int) -> None:
        """Add directed constraint edge u -> v (u before v)."""
        if u < 0 or u > self.sink or v < 0 or v > self.sink:
            raise ValueError(f"Vertex indices out of range [0, {self.sink}]: ({u}, {v})")
        if u == v:
            return
        if (u, v) not in self.edges:
            self.edges.add((u, v))
            self.out_edges[u].append(v)
            self.in_edges[v].append(u)
            self._topo_order = None

    def finalize_source_sink(self) -> None:
        """Connect source (0) to all blocks with no incoming block-edges,
        and all blocks with no outgoing block-edges to sink (n+1)."""
        for i in range(1, self.num_blocks + 1):
            # Incoming edges excluding dummy source
            block_in = [u for u in self.in_edges[i] if u != self.source]
            if not block_in and (self.source, i) not in self.edges:
                self.add_edge(self.source, i)

            # Outgoing edges excluding dummy sink
            block_out = [v for v in self.out_edges[i] if v != self.sink]
            if not block_out and (i, self.sink) not in self.edges:
                self.add_edge(i, self.sink)

        # In case graph has 0 blocks (corner case)
        if self.num_blocks == 0 and (self.source, self.sink) not in self.edges:
            self.add_edge(self.source, self.sink)

        # Invalidate cached topo order
        self._topo_order = None

    def topological_sort(self) -> List[int]:
        """Compute and cache topological sort of the DAG using Kahn's algorithm."""
        if self._topo_order is not None:
            return self._topo_order

        in_deg = {i: len(self.in_edges[i]) for i in self.nodes}
        queue = deque([i for i in self.nodes if in_deg[i] == 0])
        order: List[int] = []

        while queue:
            u = queue.popleft()
            order.append(u)
            for v in self.out_edges[u]:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    queue.append(v)

        if len(order) != len(self.nodes):
            raise ValueError(f"Cycle detected in constraint graph '{self.name}'. Order has {len(order)} of {len(self.nodes)} nodes.")

        self._topo_order = order
        return self._topo_order

    def reverse_topological_sort(self) -> List[int]:
        """Return reversed topological sort order."""
        return list(reversed(self.topological_sort()))

    def compute_longest_path_forward(self, weights: Dict[int, float], start_val: float = 0.0) -> Dict[int, float]:
        """Compute earliest position (ASAP longest path from source).
        For edge u -> v: pos[v] >= pos[u] + weights[u].
        """
        topo = self.topological_sort()
        pos = {i: 0.0 for i in self.nodes}
        pos[self.source] = start_val

        for u in topo:
            u_end = pos[u] + weights.get(u, 0.0)
            for v in self.out_edges[u]:
                if u_end > pos[v]:
                    pos[v] = u_end
        return pos

    def compute_longest_path_backward(self, weights: Dict[int, float], end_val: float) -> Dict[int, float]:
        """Compute latest position (ALAP longest path backwards from sink).
        For edge u -> v: pos[u] <= pos[v] - weights[u].
        """
        rev_topo = self.reverse_topological_sort()
        pos = {i: float("inf") for i in self.nodes}
        pos[self.sink] = end_val

        for v in rev_topo:
            for u in self.in_edges[v]:
                candidate = pos[v] - weights.get(u, 0.0)
                if candidate < pos[u]:
                    pos[u] = candidate
        return pos

    def compute_dp_max_path(self, weights: Dict[int, float]) -> Dict[int, float]:
        """Dynamic programming algorithm (Equations 5, 6, 7 in the paper).
        Computes max_{p in P_i} sum_{k in p} weights[k] in linear time O(|V| + |E|).

        Weights of source 0 and sink n+1 are forced to 0.0.
        """
        w = {i: float(weights.get(i, 0.0)) for i in self.nodes}
        w[self.source] = 0.0
        w[self.sink] = 0.0

        topo = self.topological_sort()
        rev_topo = self.reverse_topological_sort()

        # D_in[i] = max_{p in P_i^in} sum_{k in p} w_k (Eq 5)
        d_in: Dict[int, float] = {i: 0.0 for i in self.nodes}
        for u in topo:
            if u == self.source:
                d_in[u] = w[u]  # 0.0
            else:
                max_pred = max((d_in[p] for p in self.in_edges[u]), default=0.0)
                d_in[u] = max_pred + w[u]

        # D_out[i] = max_{p in P_i^out} sum_{k in p} w_k (Eq 6)
        d_out: Dict[int, float] = {i: 0.0 for i in self.nodes}
        for u in rev_topo:
            if u == self.sink:
                d_out[u] = w[u]  # 0.0
            else:
                max_succ = max((d_out[s] for s in self.out_edges[u]), default=0.0)
                d_out[u] = max_succ + w[u]

        # max_{p in P_i} sum_{k in p} w_k = d_in[i] + d_out[i] - w[i] (Eq 7)
        max_path: Dict[int, float] = {}
        for i in range(1, self.num_blocks + 1):
            val = d_in[i] + d_out[i] - w[i]
            max_path[i] = max(val, w[i], 1e-12)

        return max_path

    def count_paths_in_critical_subgraph(
        self,
        active_nodes: Set[int],
        active_edges: Set[Tuple[int, int]],
        avoid_nodes: Optional[Set[int]] = None,
    ) -> int:
        """Count number of directed paths from source (0) to sink (n+1) in the critical subgraph.
        Optionally avoids a given set of nodes (e.g. Subset III for Hard Critical Path check).
        Runs in linear time O(|V| + |E|).
        """
        if avoid_nodes is None:
            avoid_nodes = set()

        if self.source in avoid_nodes or self.sink in avoid_nodes:
            return 0
        if self.source not in active_nodes or self.sink not in active_nodes:
            return 0

        valid_nodes = active_nodes - avoid_nodes
        rev_topo = self.reverse_topological_sort()

        path_count: Dict[int, int] = {i: 0 for i in self.nodes}
        path_count[self.sink] = 1

        for u in rev_topo:
            if u not in valid_nodes:
                continue
            for v in self.out_edges[u]:
                if (u, v) in active_edges and v in valid_nodes:
                    path_count[u] += path_count[v]

        return path_count[self.source]

    def has_path_in_critical_subgraph(
        self,
        active_nodes: Set[int],
        active_edges: Set[Tuple[int, int]],
        avoid_nodes: Optional[Set[int]] = None,
    ) -> bool:
        """Return True if at least one path exists from source to sink in critical subgraph avoiding avoid_nodes."""
        return self.count_paths_in_critical_subgraph(active_nodes, active_edges, avoid_nodes) > 0

    def find_all_paths(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        active_nodes: Optional[Set[int]] = None,
        active_edges: Optional[Set[Tuple[int, int]]] = None,
    ) -> List[List[int]]:
        """Find all paths from start to end (used for verification and small graphs)."""
        if start is None:
            start = self.source
        if end is None:
            end = self.sink

        paths: List[List[int]] = []

        def dfs(curr: int, current_path: List[int]):
            if curr == end:
                paths.append(list(current_path))
                return
            for nxt in self.out_edges[curr]:
                if active_nodes is not None and nxt not in active_nodes:
                    continue
                if active_edges is not None and (curr, nxt) not in active_edges:
                    continue
                current_path.append(nxt)
                dfs(nxt, current_path)
                current_path.pop()

        dfs(start, [start])
        return paths

