"""Optimal Slack-Driven Block Shaping (SDS) Package.
Replication of ISPD 2012 paper:
'Optimal Slack-Driven Block Shaping Algorithm in Fixed-Outline Floorplanning'
by Jackey Z. Yan and Chris Chu.
"""

from .block import Block
from .constraint_graph import ConstraintGraph
from .sequence_pair import sequence_pair_to_constraint_graphs
from .basic_sds import BasicSDSSolver, SDSResult, IterationRecord
from .optimality import OptimalityChecker, OptimalityReport
from .convex_solver import solve_floorplan_convex, ConvexSolverResult
from .full_sds import FullSDSSolver, FullSDSResult
from .visualizer import plot_floorplan, plot_convergence
from .api import shape_floorplan, compute_hpwl

__all__ = [
    "Block",
    "ConstraintGraph",
    "sequence_pair_to_constraint_graphs",
    "BasicSDSSolver",
    "SDSResult",
    "IterationRecord",
    "OptimalityChecker",
    "OptimalityReport",
    "solve_floorplan_convex",
    "ConvexSolverResult",
    "FullSDSSolver",
    "FullSDSResult",
    "plot_floorplan",
    "plot_convergence",
    "shape_floorplan",
    "compute_hpwl",
]

