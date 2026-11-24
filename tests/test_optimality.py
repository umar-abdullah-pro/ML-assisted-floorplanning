"""Unit tests for Lemma 2 and Lemma 3 optimality conditions."""

from sds.block import Block
from sds.constraint_graph import ConstraintGraph
from sds.basic_sds import BasicSDSSolver
from sds.optimality import OptimalityChecker


def test_lemma2_hard_vcp():
    """Verify that a layout with at least 1 Hard VCP is certified optimal under Lemma 2."""
    # 2 blocks: Block 1 is HARD, Block 2 is SOFT
    # Path 1 in Gv: 0 -> 1 -> 3 (all hard blocks => Hard VCP!)
    # Path 2 in Gv: 0 -> 2 -> 3 (soft block)
    blocks = {
        1: Block(id=1, name="H1", is_hard=True, w_fixed=2.0, h_fixed=5.0),
        2: Block(id=2, name="S2", is_hard=False, area=8.0, aspect_ratio_range=(0.5, 2.0)),
    }
    gh = ConstraintGraph(num_blocks=2, name="Gh")
    gh.add_edge(0, 1)
    gh.add_edge(0, 2)
    gh.add_edge(1, 3)
    gh.add_edge(2, 3)
    gh.finalize_source_sink()

    gv = ConstraintGraph(num_blocks=2, name="Gv")
    gv.add_edge(0, 1)
    gv.add_edge(1, 3)
    gv.add_edge(0, 2)
    gv.add_edge(2, 3)
    gv.finalize_source_sink()

    W = 10.0
    solver = BasicSDSSolver(blocks, gh, gv, upper_bound_width=W)
    solver.compute_slacks()
    solver.classify_subsets()

    checker = OptimalityChecker(solver.blocks, gh, gv, upper_bound_width=W)
    report = checker.check(solver.x_ll, solver.y_bl, solver.subsets)

    assert report.is_optimal
    assert report.lemma2_satisfied
    assert report.hard_vcps >= 1


def test_lemma3_single_soft_hcp():
    """Verify that a layout with <= 1 soft HCP is certified optimal under Lemma 3."""
    # 2 soft blocks in series in Gh (0 -> 1 -> 2 -> 3) => exactly 1 HCP!
    # In Gv, parallel (0 -> 1 -> 3, 0 -> 2 -> 3)
    blocks = {
        1: Block(id=1, name="S1", area=4.0, aspect_ratio_range=(1/3, 3.0)),
        2: Block(id=2, name="S2", area=4.0, aspect_ratio_range=(1/3, 3.0)),
    }
    gh = ConstraintGraph(num_blocks=2, name="Gh")
    gh.add_edge(0, 1)
    gh.add_edge(1, 2)
    gh.add_edge(2, 3)
    gh.finalize_source_sink()

    gv = ConstraintGraph(num_blocks=2, name="Gv")
    gv.add_edge(0, 1)
    gv.add_edge(1, 3)
    gv.add_edge(0, 2)
    gv.add_edge(2, 3)
    gv.finalize_source_sink()

    W = 5.0
    solver = BasicSDSSolver(blocks, gh, gv, upper_bound_width=W)
    res = solver.solve()

    checker = OptimalityChecker(res.blocks, gh, gv, upper_bound_width=W)
    report = checker.check(solver.x_ll, solver.y_bl, solver.subsets)

    assert report.is_optimal
    assert report.lemma3_satisfied
    assert report.soft_hcps <= 1


if __name__ == "__main__":
    test_lemma2_hard_vcp()
    test_lemma3_single_soft_hcp()
    print("Optimality condition tests passed successfully!")

