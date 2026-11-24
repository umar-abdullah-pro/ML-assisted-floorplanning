"""Optimality Conditions checker for Slack-Driven Shaping (Section 4, Lemmas 2 and 3)."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from .block import Block
from .constraint_graph import ConstraintGraph


@dataclass
class OptimalityReport:
    """Report detailing the optimality verification of a shaping solution."""
    is_optimal: bool
    lemma2_satisfied: bool  # Exists >= 1 Hard VCP
    lemma3_satisfied: bool  # <= 1 Soft HCP or <= 1 Soft VCP
    total_vcps: int
    hard_vcps: int
    soft_vcps: int
    total_hcps: int
    hard_hcps: int
    soft_hcps: int
    subset_iii_blocks: List[int]  # Intersection soft blocks
    explanation: str


class OptimalityChecker:
    """Checks the optimality conditions for a shaping solution L."""

    def __init__(
        self,
        blocks: Dict[int, Block],
        gh: ConstraintGraph,
        gv: ConstraintGraph,
        upper_bound_width: float,
        slack_epsilon: float = 1e-6,
    ):
        self.blocks = blocks
        self.gh = gh
        self.gv = gv
        self.W = upper_bound_width
        self.slack_eps = slack_epsilon

    def check(
        self,
        x_ll: Dict[int, float],
        y_bl: Dict[int, float],
        subsets: Dict[str, List[int]],
    ) -> OptimalityReport:
        """Evaluate Lemma 2 and Lemma 3 on solution L.

        Args:
            x_ll: Packed x-coordinates (from LL packing).
            y_bl: Packed y-coordinates (from BL packing).
            subsets: Classified subsets I-VII.

        Returns:
            OptimalityReport with complete analysis.
        """
        subset_iii = set(subsets.get("III", []))

        # 1. Build Vertical Critical Subgraph in Gv
        # A node u is critical in Gv if b.slack_v <= eps (for dummy source/sink, always critical)
        v_crit_nodes: Set[int] = {self.gv.source, self.gv.sink}
        for i, b in self.blocks.items():
            if b.slack_v <= self.slack_eps:
                v_crit_nodes.add(i)

        v_crit_edges: Set[Tuple[int, int]] = set()
        for u in v_crit_nodes:
            u_dim = 0.0 if u in (self.gv.source, self.gv.sink) else self.blocks[u].h
            for v in self.gv.out_edges[u]:
                if v in v_crit_nodes:
                    # Edge is tight if y_u + h_u == y_v
                    if abs(y_bl[u] + u_dim - y_bl[v]) <= self.slack_eps:
                        v_crit_edges.add((u, v))

        total_vcps = self.gv.count_paths_in_critical_subgraph(v_crit_nodes, v_crit_edges)
        hard_vcps = self.gv.count_paths_in_critical_subgraph(v_crit_nodes, v_crit_edges, avoid_nodes=subset_iii)
        soft_vcps = max(0, total_vcps - hard_vcps)

        # 2. Build Horizontal Critical Subgraph in Gh
        # HCPs only exist if layout width x_{n+1} == W (Section 3)
        layout_w = x_ll.get(self.gh.sink, 0.0)
        hcp_exists = abs(layout_w - self.W) <= self.slack_eps

        if not hcp_exists:
            total_hcps = 0
            hard_hcps = 0
            soft_hcps = 0
        else:
            h_crit_nodes: Set[int] = {self.gh.source, self.gh.sink}
            for i, b in self.blocks.items():
                if b.slack_h <= self.slack_eps:
                    h_crit_nodes.add(i)

            h_crit_edges: Set[Tuple[int, int]] = set()
            for u in h_crit_nodes:
                u_dim = 0.0 if u in (self.gh.source, self.gh.sink) else self.blocks[u].w
                for v in self.gh.out_edges[u]:
                    if v in h_crit_nodes:
                        if abs(x_ll[u] + u_dim - x_ll[v]) <= self.slack_eps:
                            h_crit_edges.add((u, v))

            total_hcps = self.gh.count_paths_in_critical_subgraph(h_crit_nodes, h_crit_edges)
            hard_hcps = self.gh.count_paths_in_critical_subgraph(h_crit_nodes, h_crit_edges, avoid_nodes=subset_iii)
            soft_hcps = max(0, total_hcps - hard_hcps)

        # Lemma 2: If there exists >= 1 Hard VCP, L is optimal.
        lemma2_satisfied = (hard_vcps > 0)

        # Lemma 3: If there exists <= 1 Soft HCP or <= 1 Soft VCP, L is optimal.
        lemma3_satisfied = (soft_hcps <= 1 or soft_vcps <= 1)

        is_optimal = lemma2_satisfied or lemma3_satisfied

        # Formulate human-readable explanation
        reasons = []
        if lemma2_satisfied:
            reasons.append(f"Lemma 2 satisfied: Found {hard_vcps} Hard VCP(s) (no soft intersection blocks).")
        if lemma3_satisfied:
            reasons.append(f"Lemma 3 satisfied: soft_hcps={soft_hcps} (<= 1) or soft_vcps={soft_vcps} (<= 1).")
        if not is_optimal:
            reasons.append(
                f"Non-optimal condition detected: soft_hcps={soft_hcps} > 1, soft_vcps={soft_vcps} > 1, "
                f"and 0 Hard VCPs. Intersection soft blocks {list(subset_iii)} can be improved via GP."
            )

        return OptimalityReport(
            is_optimal=is_optimal,
            lemma2_satisfied=lemma2_satisfied,
            lemma3_satisfied=lemma3_satisfied,
            total_vcps=total_vcps,
            hard_vcps=hard_vcps,
            soft_vcps=soft_vcps,
            total_hcps=total_hcps,
            hard_hcps=hard_hcps,
            soft_hcps=soft_hcps,
            subset_iii_blocks=sorted(list(subset_iii)),
            explanation="; ".join(reasons),
        )

