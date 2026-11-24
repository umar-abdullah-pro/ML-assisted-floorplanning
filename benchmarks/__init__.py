"""Benchmark instances and generators for Slack-Driven Shaping."""

from .figure2_example import build_figure2_benchmark
from .mcnc_benchmarks import build_mcnc_benchmark, MCNC_SPECS
from .synthetic_benchmarks import generate_hb_style_benchmark

__all__ = [
    "build_figure2_benchmark",
    "build_mcnc_benchmark",
    "MCNC_SPECS",
    "generate_hb_style_benchmark",
]

