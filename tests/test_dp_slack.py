"""Unit tests for Dynamic Programming slack distribution and longest path."""

from sds.constraint_graph import ConstraintGraph


def test_longest_path_forward_backward():
    cg = ConstraintGraph(num_blocks=3, name="test_cg")
    # 0 -> 1 -> 2 -> 3 -> 4
    cg.add_edge(0, 1)
    cg.add_edge(1, 2)
    cg.add_edge(2, 3)
    cg.add_edge(3, 4)
    cg.finalize_source_sink()

    weights = {1: 10.0, 2: 20.0, 3: 30.0}

    # Forward
    pos_fwd = cg.compute_longest_path_forward(weights, start_val=0.0)
    assert pos_fwd[0] == 0.0
    assert pos_fwd[1] == 0.0
    assert pos_fwd[2] == 10.0
    assert pos_fwd[3] == 30.0
    assert pos_fwd[4] == 60.0

    # Backward from 60.0
    pos_bwd = cg.compute_longest_path_backward(weights, end_val=60.0)
    assert pos_bwd[4] == 60.0
    assert pos_bwd[3] == 30.0
    assert pos_bwd[2] == 10.0
    assert pos_bwd[1] == 0.0


def test_dp_max_path_linear():
    """Verify linear DP computes max_{p in P_i} sum_{k in p} w_k correctly."""
    cg = ConstraintGraph(num_blocks=4, name="test_dp")
    # Two paths:
    # Path A: 0 -> 1 -> 2 -> 5 (weights: w1=5, w2=10 => sum=15)
    # Path B: 0 -> 3 -> 4 -> 5 (weights: w3=2, w4=3 => sum=5)
    cg.add_edge(0, 1)
    cg.add_edge(1, 2)
    cg.add_edge(2, 5)

    cg.add_edge(0, 3)
    cg.add_edge(3, 4)
    cg.add_edge(4, 5)
    cg.finalize_source_sink()

    weights = {1: 5.0, 2: 10.0, 3: 2.0, 4: 3.0}
    max_paths = cg.compute_dp_max_path(weights)

    assert abs(max_paths[1] - 15.0) < 1e-6
    assert abs(max_paths[2] - 15.0) < 1e-6
    assert abs(max_paths[3] - 5.0) < 1e-6
    assert abs(max_paths[4] - 5.0) < 1e-6


def test_dp_max_path_diamond():
    """Verify DP on diamond graph with cross paths."""
    cg = ConstraintGraph(num_blocks=4, name="diamond")
    # 0 -> 1 -> 2 -> 5
    # 0 -> 1 -> 3 -> 5
    # 0 -> 4 -> 2 -> 5
    cg.add_edge(0, 1)
    cg.add_edge(1, 2)
    cg.add_edge(1, 3)
    cg.add_edge(0, 4)
    cg.add_edge(4, 2)
    cg.add_edge(2, 5)
    cg.add_edge(3, 5)
    cg.finalize_source_sink()

    # Paths:
    # 1 -> 2: sum = 4 + 10 = 14
    # 1 -> 3: sum = 4 + 2 = 6
    # 4 -> 2: sum = 8 + 10 = 18
    weights = {1: 4.0, 2: 10.0, 3: 2.0, 4: 8.0}
    max_paths = cg.compute_dp_max_path(weights)

    # For node 1: max path is 1->2 (14)
    assert abs(max_paths[1] - 14.0) < 1e-6
    # For node 2: max path is 4->2 (18)
    assert abs(max_paths[2] - 18.0) < 1e-6
    # For node 3: max path is 1->3 (6)
    assert abs(max_paths[3] - 6.0) < 1e-6
    # For node 4: max path is 4->2 (18)
    assert abs(max_paths[4] - 18.0) < 1e-6


if __name__ == "__main__":
    test_longest_path_forward_backward()
    test_dp_max_path_linear()
    test_dp_max_path_diamond()
    print("All DP tests passed successfully!")
