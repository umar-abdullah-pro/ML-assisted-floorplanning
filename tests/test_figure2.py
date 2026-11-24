"""Test exact reproduction of Figure 2 from Yan & Chu (ISPD 2012)."""

from benchmarks.figure2_example import build_figure2_benchmark
from sds.basic_sds import BasicSDSSolver
from sds.full_sds import FullSDSSolver
from sds.optimality import OptimalityChecker


def test_figure2_basic_sds_and_optimality():
    """Verify that Basic SDS gets stuck at height 5.0 and OptimalityChecker flags it."""
    blocks, gh, gv, W = build_figure2_benchmark()

    # Step 1: Run Basic SDS
    solver = BasicSDSSolver(blocks, gh, gv, upper_bound_width=W)
    solver.compute_slacks()
    subsets = solver.classify_subsets()

    # Check that all 4 blocks are in Subset III
    assert set(subsets["III"]) == {1, 2, 3, 4}
    assert len(subsets["IV"]) == 0
    assert len(subsets["VI"]) == 0
    assert abs(solver.layout_height - 5.0) < 1e-6
    assert abs(solver.layout_width - 5.0) < 1e-6

    # Run solver on Fig 2-(a) shapes: since no target soft blocks exist, terminates immediately with height 5.0
    res = solver.solve(reset_to_min=False)
    assert abs(res.layout_height - 5.0) < 1e-6

    # Step 2: Check optimality conditions (Section 4)
    checker = OptimalityChecker(solver.blocks, gh, gv, upper_bound_width=W)
    report = checker.check(solver.x_ll, solver.y_bl, solver.subsets)

    assert not report.is_optimal, "Figure 2 layout at height 5.0 must NOT be classified as optimal"
    assert not report.lemma2_satisfied, "Should have 0 Hard VCPs"
    assert not report.lemma3_satisfied, "Should have > 1 soft HCP and > 1 soft VCP"
    assert report.soft_hcps == 2, f"Expected 2 soft HCPs, got {report.soft_hcps}"
    assert report.soft_vcps == 2, f"Expected 2 soft VCPs, got {report.soft_vcps}"
    assert set(report.subset_iii_blocks) == {1, 2, 3, 4}


def test_figure2_full_sds_optimal():
    """Verify that Full SDS achieves the theoretical optimal layout height 3.2."""
    blocks, gh, gv, W = build_figure2_benchmark()

    full_solver = FullSDSSolver(blocks, gh, gv, upper_bound_width=W)
    result = full_solver.solve(reset_to_min=False)

    assert result.feasible, "Result must be feasible"
    assert result.optimal, "Full SDS must report optimal solution"
    assert result.gp_fallback_used, "Figure 2 requires GP/convex fallback"

    # Verify optimal height 3.2 (Figure 2-(b) in paper)
    assert abs(result.layout_height - 3.2) < 1e-3, f"Expected optimal height 3.2, got {result.layout_height}"
    assert result.layout_width <= W + 1e-6, f"Width {result.layout_width} must not exceed W={W}"

    # Verify optimal widths w1 = w2 = w3 = w4 = 2.5
    for i in (1, 2, 3, 4):
        b = result.blocks[i]
        assert abs(b.w - 2.5) < 1e-3, f"Expected block {i} width 2.5, got {b.w}"
        assert abs(b.h - 1.6) < 1e-3, f"Expected block {i} height 1.6, got {b.h}"


if __name__ == "__main__":
    test_figure2_basic_sds_and_optimality()
    test_figure2_full_sds_optimal()
    print("Figure 2 tests passed successfully!")
