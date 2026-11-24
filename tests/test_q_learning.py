"""Unit tests for Q-learning step size controller and placement search."""

from sds.block import Block
from q_learning.q_controller import QLearningController, StepAction, ProgressTrend, LegalityState
from q_learning.placement_search import (
    FixedOutlinePlacementOptimizer,
    OptimizerConfig,
    SearchMode,
)


def test_q_learning_controller_updates():
    """Verify Q-learning agent initializes, selects actions, and performs Bellman updates."""
    agent = QLearningController(learning_rate=0.2, discount_factor=0.9, epsilon_init=0.0)

    # State: IMPROVING & LEGAL
    state1 = (ProgressTrend.IMPROVING, LegalityState.LEGAL)
    state2 = (ProgressTrend.UNCHANGED, LegalityState.LEGAL)

    # Initial Q-values are 0
    s_idx = agent.state_to_idx[state1]
    assert agent.q_table[s_idx, StepAction.SMALL.value] == 0.0

    # With epsilon=0, action selection picks greedily
    action = agent.select_action(state1)
    assert isinstance(action, StepAction)

    # Update with reward 10.0
    new_val = agent.update(state1, StepAction.SMALL, reward=10.0, next_state=state2)
    assert new_val == 0.2 * 10.0  # 2.0
    assert agent.q_table[s_idx, StepAction.SMALL.value] == 2.0


def test_placement_search_all_modes():
    """Verify that placement optimizer runs successfully in all 4 modes."""
    # 4 small blocks
    blocks = {
        1: Block(id=1, name="B1", area=16.0, is_hard=False),
        2: Block(id=2, name="B2", area=16.0, is_hard=False),
        3: Block(id=3, name="B3", w_fixed=4.0, h_fixed=4.0, is_hard=True),
        4: Block(id=4, name="B4", area=16.0, is_hard=False),
    }
    nets = [[1, 2], [2, 3], [3, 4]]

    config = OptimizerConfig(
        outline_width=15.0,
        outline_height=15.0,
        max_iterations=15,
        sds_inner_iterations=30,
        seed=42,
    )

    optimizer = FixedOutlinePlacementOptimizer(blocks, nets, config)

    # Test all 4 modes
    for mode in SearchMode:
        res = optimizer.run_optimization(mode)
        assert res.total_iterations == 15
        assert len(res.history) == 15
        assert res.best_width > 0
        assert res.best_height > 0
        assert res.best_hpwl >= 0


if __name__ == "__main__":
    test_q_learning_controller_updates()
    test_placement_search_all_modes()
    print("All Q-learning and placement search tests passed successfully!")

