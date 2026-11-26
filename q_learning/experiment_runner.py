"""Experiment runner to compare all 4 modes on standard benchmarks (Evaluation Plan)."""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from sds.block import Block
from .placement_search import (
    FixedOutlinePlacementOptimizer,
    OptimizerConfig,
    OptimizationResult,
    SearchMode,
)


@dataclass
class BenchmarkExperimentResult:
    benchmark_name: str
    num_blocks: int
    outline_width: float
    outline_height: float
    results: Dict[str, OptimizationResult]
    summary_table: List[Dict[str, Any]]


def compare_all_methods(
    benchmark_name: str,
    blocks: Dict[int, Block],
    nets: Optional[List[List[Any]]],
    outline_width: float,
    outline_height: float,
    max_iterations: int = 100,
    seed: int = 42,
    verbose: bool = False,
    learning_rate: float = 0.15,
    discount_factor: float = 0.90,
    epsilon_init: float = 0.35,
) -> BenchmarkExperimentResult:
    """Run all 4 comparative modes on a given benchmark:
    1. Baseline (Fixed Shapes, Fixed Step)
    2. SDS Shaping Only (Fixed Step)
    3. Q-Learning Only (Fixed Shapes)
    4. Proposed (Q-Learning + SDS Shaping)
    """
    config = OptimizerConfig(
        outline_width=outline_width,
        outline_height=outline_height,
        max_iterations=max_iterations,
        sds_inner_iterations=120,
        q_learning_rate=learning_rate,
        q_discount_factor=discount_factor,
        q_epsilon_init=epsilon_init,
        seed=seed,
    )

    optimizer = FixedOutlinePlacementOptimizer(blocks, nets, config)
    modes_to_test = [
        SearchMode.BASELINE,
        SearchMode.SHAPING_ONLY,
        SearchMode.Q_LEARNING_ONLY,
        SearchMode.COMBINED_PROPOSED,
    ]

    results: Dict[str, OptimizationResult] = {}
    summary_table: List[Dict[str, Any]] = []

    for mode in modes_to_test:
        if verbose:
            print(f"Running mode: {mode.value}...")
        res = optimizer.run_optimization(mode)
        results[mode.name] = res

        summary_table.append({
            "Method": mode.value,
            "Legal": "Yes" if res.best_is_legal else "No",
            "Height": round(res.best_height, 2),
            "Width": round(res.best_width, 2),
            "HPWL": round(res.best_hpwl, 2),
            "Cost": round(res.best_cost, 4),
            "Legal Steps": f"{res.legal_iterations_count}/{res.total_iterations}",
            "Time (s)": round(res.runtime_seconds, 3),
        })

    return BenchmarkExperimentResult(
        benchmark_name=benchmark_name,
        num_blocks=len(blocks),
        outline_width=outline_width,
        outline_height=outline_height,
        results=results,
        summary_table=summary_table,
    )

