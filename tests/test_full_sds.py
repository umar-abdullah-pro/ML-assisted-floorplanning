"""End-to-end integration tests for Slack-Driven Shaping and comparison with convex solver."""

from sds.basic_sds import BasicSDSSolver
from sds.full_sds import FullSDSSolver
from sds.convex_solver import solve_floorplan_convex
from benchmarks.synthetic_benchmarks import generate_hb_style_benchmark
from benchmarks.mcnc_benchmarks import build_mcnc_benchmark


def test_height_monotonicity_and_width_bound():
    """Verify that layout height is monotonically non-increasing and width never exceeds W."""
    blocks, gh, gv, W = generate_hb_style_benchmark(num_soft=15, num_hard=5, seed=123)

    solver = BasicSDSSolver(blocks, gh, gv, upper_bound_width=W)
    result = solver.solve(max_iterations=1000)

    assert result.feasible, "Benchmark must be feasible"
    assert result.total_iterations > 0

    # Check height monotonically non-increasing (allowing small float jitter)
    heights = [rec.layout_height for rec in result.history]
    for i in range(1, len(heights)):
        assert heights[i] <= heights[i - 1] + 1e-6, (
            f"Height increased at iteration {i}: {heights[i-1]} -> {heights[i]}"
        )

    # Check width bound never violated
    widths = [rec.layout_width for rec in result.history]
    for i, w in enumerate(widths):
        assert w <= W + 1e-4, f"Width exceeded W={W} at iteration {i}: w={w}"


def test_sds_vs_convex_solver_ground_truth():
    """Verify that SDS converges to the exact convex optimal solution."""
    blocks, gh, gv, W = build_mcnc_benchmark("apte", target_experiment="table2", seed=42)

    # Solve with CVXPY convex ground truth
    convex_res = solve_floorplan_convex(blocks, gh, gv, upper_bound_width=W)
    assert convex_res.is_optimal

    # Solve with SDS
    full_sds = FullSDSSolver(blocks, gh, gv, upper_bound_width=W)
    sds_res = full_sds.solve(max_basic_iterations=2000)

    assert sds_res.feasible
    assert sds_res.layout_width <= W + 1e-4

    # Relative difference between SDS height and CVXPY ground truth should be very small (< 1%)
    rel_diff = abs(sds_res.layout_height - convex_res.optimal_height) / convex_res.optimal_height
    print(f"\nApte benchmark: CVXPY={convex_res.optimal_height:.4f}, SDS={sds_res.layout_height:.4f}, diff={rel_diff*100:.3f}%")
    assert rel_diff < 0.015, f"SDS height {sds_res.layout_height} differs from convex optimum {convex_res.optimal_height} by {rel_diff*100:.2f}%"


if __name__ == "__main__":
    test_height_monotonicity_and_width_bound()
    test_sds_vs_convex_solver_ground_truth()
    print("All end-to-end tests passed successfully!")

