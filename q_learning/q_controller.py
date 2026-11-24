"""Tabular Q-Learning Controller for adaptive step-size selection in VLSI floorplanning.
Inspired by CSF (Meng et al., arXiv:2504.03796).
"""

from __future__ import annotations
import random
from enum import Enum
from typing import Dict, List, Tuple
import numpy as np


class StepAction(Enum):
    SMALL = 0   # 1 perturbation: fine-tuning / local refinement
    MEDIUM = 1  # 2 perturbations: balanced exploration
    LARGE = 2   # 4 perturbations: coarse leap / escaping local traps


class ProgressTrend(Enum):
    IMPROVING = 0
    UNCHANGED = 1
    WORSENING = 2


class LegalityState(Enum):
    LEGAL = 0
    ILLEGAL = 1


# A state is a tuple: (ProgressTrend, LegalityState)
QState = Tuple[ProgressTrend, LegalityState]


class QLearningController:
    """Tabular Q-learning agent that dynamically selects placement search step size."""

    def __init__(
        self,
        learning_rate: float = 0.15,
        discount_factor: float = 0.90,
        epsilon_init: float = 0.35,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.99,
        seed: int = 42,
    ):
        self.alpha = float(learning_rate)
        self.gamma = float(discount_factor)
        self.epsilon = float(epsilon_init)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.rng = random.Random(seed)

        # Actions
        self.actions = [StepAction.SMALL, StepAction.MEDIUM, StepAction.LARGE]
        self.num_actions = len(self.actions)

        # States: 3 progress trends x 2 legality states = 6 states
        self.states: List[QState] = [
            (p, l) for p in ProgressTrend for l in LegalityState
        ]
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for i, s in enumerate(self.states)}

        # Q-table: shape (num_states, num_actions)
        self.q_table = np.zeros((len(self.states), self.num_actions), dtype=np.float64)

        # Tracking metrics
        self.action_history: List[StepAction] = []
        self.reward_history: List[float] = []

    def get_state(
        self,
        cost_diff: float,
        is_legal: bool,
        tolerance: float = 1e-4,
    ) -> QState:
        """Map continuous optimization indicators into discrete QState.

        Args:
            cost_diff: cost_{t-1} - cost_t (positive means cost decreased / improved).
            is_legal: whether current layout fits completely within fixed outline (W x H).
            tolerance: threshold for unchanged status.
        """
        if cost_diff > tolerance:
            trend = ProgressTrend.IMPROVING
        elif cost_diff < -tolerance:
            trend = ProgressTrend.WORSENING
        else:
            trend = ProgressTrend.UNCHANGED

        legality = LegalityState.LEGAL if is_legal else LegalityState.ILLEGAL
        return (trend, legality)

    def select_action(self, state: QState) -> StepAction:
        """Select step size using epsilon-greedy strategy."""
        s_idx = self.state_to_idx[state]

        if self.rng.random() < self.epsilon:
            # Exploration
            action = self.rng.choice(self.actions)
        else:
            # Exploitation: pick action with max Q-value
            q_vals = self.q_table[s_idx]
            max_val = np.max(q_vals)
            # Break ties randomly
            best_actions = [a for a, q in zip(self.actions, q_vals) if q == max_val]
            action = self.rng.choice(best_actions)

        self.action_history.append(action)
        return action

    def update(
        self,
        state: QState,
        action: StepAction,
        reward: float,
        next_state: QState,
    ) -> float:
        """Perform Bellman equation update on Q-table.

        Q(s, a) <- Q(s, a) + alpha * [R + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        s_idx = self.state_to_idx[state]
        a_idx = action.value
        next_s_idx = self.state_to_idx[next_state]

        old_val = self.q_table[s_idx, a_idx]
        best_future = np.max(self.q_table[next_s_idx])

        new_val = old_val + self.alpha * (reward + self.gamma * best_future - old_val)
        self.q_table[s_idx, a_idx] = new_val

        self.reward_history.append(reward)
        self.decay_epsilon()
        return new_val

    def decay_epsilon(self) -> None:
        """Decay exploration rate epsilon towards epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def compute_reward(
        self,
        prev_cost: float,
        curr_cost: float,
        was_legal: bool,
        is_legal: bool,
        width_violation: float,
        height_violation: float,
    ) -> float:
        """Compute structured reward signal based on objective improvement and legality.

        Positive reward:
        - Cost reduction (lower height / lower HPWL)
        - Transitioning from illegal to legal (+bonus)
        - Staying legal (+small reward)

        Negative reward:
        - Cost increase
        - Boundary violations
        """
        # Normalized relative cost improvement: (prev - curr) / max(prev, 1.0)
        rel_improvement = (prev_cost - curr_cost) / max(abs(prev_cost), 1.0)
        reward = rel_improvement * 10.0

        # Legality rewards/penalties
        if is_legal:
            if not was_legal:
                reward += 15.0  # Big bonus for legalizing the floorplan!
            else:
                reward += 2.0   # Reward maintaining legality
        else:
            # Penalty proportional to relative outline violation
            viol_penalty = (width_violation + height_violation) * 5.0
            reward -= min(viol_penalty, 20.0)

        return float(reward)

    def get_q_matrix_summary(self) -> Dict[str, Dict[str, float]]:
        """Return human-readable dictionary of Q-table for Streamlit display and diagnostics."""
        summary = {}
        action_names = ["Small (1)", "Medium (2-3)", "Large (4-6)"]
        for s_idx, state in self.idx_to_state.items():
            trend_str = state[0].name
            leg_str = state[1].name
            state_label = f"{trend_str} & {leg_str}"
            summary[state_label] = {
                action_names[a]: float(self.q_table[s_idx, a]) for a in range(self.num_actions)
            }
        return summary

