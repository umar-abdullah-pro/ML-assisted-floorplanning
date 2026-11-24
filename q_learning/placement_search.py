"""Placement search engine combining Sequence-Pair local search, Q-learning, and Slack-Driven Shaping."""

from __future__ import annotations
import copy
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

from sds.block import Block
from sds.constraint_graph import ConstraintGraph
from sds.sequence_pair import sequence_pair_to_constraint_graphs
from sds.basic_sds import BasicSDSSolver
from sds.full_sds import FullSDSSolver
from sds.api import compute_hpwl
from .q_controller import QLearningController, StepAction, QState


class SearchMode(Enum):
    BASELINE = "Baseline (Fixed Shapes, Fixed Step)"
    SHAPING_ONLY = "SDS Shaping Only (Fixed Step)"
    Q_LEARNING_ONLY = "Q-Learning Step Size Only (Fixed Shapes)"
    COMBINED_PROPOSED = "Proposed (Q-Learning + SDS Shaping)"


@dataclass
class OptimizerConfig:
    """Configuration for fixed-outline placement search."""
    outline_width: float
    outline_height: float
    max_iterations: int = 150
    weight_height: float = 1.0
    weight_width: float = 0.5
    weight_hpwl: float = 0.5
    penalty_boundary: float = 10.0
    sds_inner_iterations: int = 150
    use_full_sds: bool = False  # True for Full SDS, False for fast Basic SDS
    temperature_init: float = 10.0  # Simulated annealing acceptance
    temperature_decay: float = 0.96
    seed: int = 42


@dataclass
class SearchStepRecord:
    iteration: int
    step_action: StepAction
    cost: float
    layout_width: float
    layout_height: float
    hpwl: float
    is_legal: bool
    reward: float
    accepted: bool
    temperature: float


@dataclass
class OptimizationResult:
    mode: SearchMode
    best_cost: float
    best_width: float
    best_height: float
    best_hpwl: float
    best_is_legal: bool
    best_seq_x: List[int]
    best_seq_y: List[int]
    best_blocks: Dict[int, Block]
    total_iterations: int
    legal_iterations_count: int
    runtime_seconds: float
    history: List[SearchStepRecord] = field(default_factory=list)
    q_controller: Optional[QLearningController] = None


class FixedOutlinePlacementOptimizer:
    """Fixed-outline placement search engine supporting the 4 evaluation modes."""

    def __init__(
        self,
        blocks: Dict[int, Block],
        nets: Optional[List[List[Any]]],
        config: OptimizerConfig,
    ):
        self.orig_blocks = {i: b.clone() for i, b in blocks.items()}
        self.num_blocks = len(blocks)
        self.nets = nets if nets is not None else []
        self.config = config
        self.rng = random.Random(config.seed)

    def _perturb_sequence_pair(
        self,
        seq_x: List[int],
        seq_y: List[int],
        action: StepAction,
    ) -> Tuple[List[int], List[int]]:
        """Apply perturbations to (X, Y) corresponding to selected step size."""
        new_x = list(seq_x)
        new_y = list(seq_y)
        n = self.num_blocks

        # Number of perturbation moves based on action
        if action == StepAction.SMALL:
            num_moves = 1
        elif action == StepAction.MEDIUM:
            num_moves = self.rng.randint(2, 3)
        else:  # LARGE
            num_moves = self.rng.randint(4, 6)

        for _ in range(num_moves):
            move_type = self.rng.randint(0, 3)
            i, j = self.rng.sample(range(n), 2)

            if move_type == 0:
                # Swap two in X
                new_x[i], new_x[j] = new_x[j], new_x[i]
            elif move_type == 1:
                # Swap two in Y
                new_y[i], new_y[j] = new_y[j], new_y[i]
            elif move_type == 2:
                # Swap two in both X and Y
                new_x[i], new_x[j] = new_x[j], new_x[i]
                new_y[i], new_y[j] = new_y[j], new_y[i]
            else:
                # Invert subsequence
                lo, hi = min(i, j), max(i, j)
                if self.rng.random() < 0.5:
                    new_x[lo:hi+1] = reversed(new_x[lo:hi+1])
                else:
                    new_y[lo:hi+1] = reversed(new_y[lo:hi+1])

        return new_x, new_y

    def _evaluate_placement(
        self,
        seq_x: List[int],
        seq_y: List[int],
        blocks_dict: Dict[int, Block],
        apply_shaping: bool,
    ) -> Tuple[float, float, float, float, bool, Dict[int, Block]]:
        """Evaluate a sequence pair placement with or without SDS block shaping.

        Returns:
            (cost, width, height, hpwl, is_legal, evaluated_blocks)
        """
        gh, gv = sequence_pair_to_constraint_graphs(
            seq_x, seq_y, self.num_blocks, apply_reduction=(self.num_blocks <= 250)
        )

        eval_blocks = {i: b.clone() for i, b in blocks_dict.items()}

        if apply_shaping:
            if self.config.use_full_sds:
                sds = FullSDSSolver(eval_blocks, gh, gv, upper_bound_width=self.config.outline_width)
                sds_res = sds.solve(max_basic_iterations=self.config.sds_inner_iterations, reset_to_min=True)
                layout_w = sds_res.layout_width
                layout_h = sds_res.layout_height
                eval_blocks = sds_res.blocks
            else:
                # Fast Basic SDS
                sds = BasicSDSSolver(eval_blocks, gh, gv, upper_bound_width=self.config.outline_width)
                sds_res = sds.solve(max_iterations=self.config.sds_inner_iterations, reset_to_min=True)
                layout_w = sds_res.layout_width
                layout_h = sds_res.layout_height
                eval_blocks = sds_res.blocks
        else:
            # Baseline: blocks packed at their current/fixed dimensions without shaping
            w_dict = {i: eval_blocks[i].w for i in eval_blocks}
            h_dict = {i: eval_blocks[i].h for i in eval_blocks}
            x_pos = gh.compute_longest_path_forward(w_dict, start_val=0.0)
            y_pos = gv.compute_longest_path_forward(h_dict, start_val=0.0)
            layout_w = x_pos[gh.sink]
            layout_h = y_pos[gv.sink]
            for i, b in eval_blocks.items():
                b.x = x_pos[i]
                b.y = y_pos[i]

        # HPWL
        hpwl = compute_hpwl(eval_blocks, self.nets) if self.nets else 0.0

        # Boundary violations
        viol_w = max(0.0, layout_w - self.config.outline_width)
        viol_h = max(0.0, layout_h - self.config.outline_height)
        is_legal = (viol_w <= 1e-4) and (viol_h <= 1e-4)

        # Composite cost function:
        # Cost = w_h * (H / H_out) + w_w * (W / W_out) + w_hpwl * (HPWL / HPWL_ref) + Penalty * Violations
        norm_h = layout_h / max(self.config.outline_height, 1.0)
        norm_w = layout_w / max(self.config.outline_width, 1.0)
        norm_hpwl = hpwl / max(self.config.outline_width + self.config.outline_height, 1.0) if hpwl > 0 else 0.0

        cost = (
            self.config.weight_height * norm_h
            + self.config.weight_width * norm_w
            + self.config.weight_hpwl * norm_hpwl
            + self.config.penalty_boundary * (viol_w / self.config.outline_width + viol_h / self.config.outline_height)
        )

        return cost, layout_w, layout_h, hpwl, is_legal, eval_blocks

    def run_optimization(self, mode: SearchMode) -> OptimizationResult:
        """Run placement optimization using the specified evaluation mode."""
        start_time = time.perf_counter()

        use_q_learning = mode in (SearchMode.Q_LEARNING_ONLY, SearchMode.COMBINED_PROPOSED)
        use_shaping = mode in (SearchMode.SHAPING_ONLY, SearchMode.COMBINED_PROPOSED)

        # Initial sequence pair: 1..n
        curr_seq_x = list(range(1, self.num_blocks + 1))
        curr_seq_y = list(range(1, self.num_blocks + 1))
        self.rng.shuffle(curr_seq_x)
        self.rng.shuffle(curr_seq_y)

        # Initial blocks
        curr_blocks = {i: b.clone() for i, b in self.orig_blocks.items()}

        # Evaluate initial state
        curr_cost, curr_w, curr_h, curr_hpwl, curr_legal, curr_blocks = self._evaluate_placement(
            curr_seq_x, curr_seq_y, curr_blocks, apply_shaping=use_shaping
        )

        best_cost = curr_cost
        best_w = curr_w
        best_h = curr_h
        best_hpwl = curr_hpwl
        best_legal = curr_legal
        best_seq_x = list(curr_seq_x)
        best_seq_y = list(curr_seq_y)
        best_blocks = {i: b.clone() for i, b in curr_blocks.items()}

        # Q-controller initialization
        q_controller = QLearningController(seed=self.config.seed) if use_q_learning else None
        prev_cost = curr_cost
        was_legal = curr_legal

        history: List[SearchStepRecord] = []
        temperature = float(self.config.temperature_init)
        legal_count = 1 if curr_legal else 0

        # Initial state for Q-agent
        curr_state = (
            q_controller.get_state(cost_diff=0.0, is_legal=curr_legal)
            if q_controller
            else None
        )

        for it in range(1, self.config.max_iterations + 1):
            # Step size selection:
            if use_q_learning and q_controller and curr_state is not None:
                action = q_controller.select_action(curr_state)
            else:
                # Fixed baseline step size: small perturbation (1 move)
                action = StepAction.SMALL

            # Propose candidate sequence pair
            cand_x, cand_y = self._perturb_sequence_pair(curr_seq_x, curr_seq_y, action)

            # Evaluate candidate
            cand_cost, cand_w, cand_h, cand_hpwl, cand_legal, cand_blocks = self._evaluate_placement(
                cand_x, cand_y, curr_blocks, apply_shaping=use_shaping
            )

            # Simulated annealing acceptance criterion
            delta_cost = cand_cost - curr_cost
            if delta_cost < 0:
                accept = True
            else:
                accept_prob = random.random() < (0.0 if temperature <= 1e-6 else min(1.0, 2.71828 ** (-delta_cost / max(temperature, 1e-4))))
                accept = accept_prob

            reward = 0.0
            if use_q_learning and q_controller and curr_state is not None:
                # Compute reward
                viol_w = max(0.0, cand_w - self.config.outline_width) / max(self.config.outline_width, 1.0)
                viol_h = max(0.0, cand_h - self.config.outline_height) / max(self.config.outline_height, 1.0)
                reward = q_controller.compute_reward(
                    prev_cost=curr_cost,
                    curr_cost=cand_cost,
                    was_legal=curr_legal,
                    is_legal=cand_legal,
                    width_violation=viol_w,
                    height_violation=viol_h,
                )
                next_state = q_controller.get_state(
                    cost_diff=(curr_cost - cand_cost),
                    is_legal=cand_legal,
                )
                q_controller.update(curr_state, action, reward, next_state)
                curr_state = next_state

            if accept:
                curr_seq_x = cand_x
                curr_seq_y = cand_y
                curr_cost = cand_cost
                curr_w = cand_w
                curr_h = cand_h
                curr_hpwl = cand_hpwl
                curr_legal = cand_legal
                curr_blocks = cand_blocks

                # Update best-so-far
                # Preference: legal solutions always beat illegal solutions
                is_better = False
                if cand_legal and not best_legal:
                    is_better = True
                elif cand_legal == best_legal:
                    if cand_cost < best_cost:
                        is_better = True

                if is_better:
                    best_cost = cand_cost
                    best_w = cand_w
                    best_h = cand_h
                    best_hpwl = cand_hpwl
                    best_legal = cand_legal
                    best_seq_x = list(cand_x)
                    best_seq_y = list(cand_y)
                    best_blocks = {i: b.clone() for i, b in cand_blocks.items()}

            if curr_legal:
                legal_count += 1

            record = SearchStepRecord(
                iteration=it,
                step_action=action,
                cost=curr_cost,
                layout_width=curr_w,
                layout_height=curr_h,
                hpwl=curr_hpwl,
                is_legal=curr_legal,
                reward=reward,
                accepted=accept,
                temperature=temperature,
            )
            history.append(record)

            temperature *= self.config.temperature_decay

        elapsed = time.perf_counter() - start_time

        return OptimizationResult(
            mode=mode,
            best_cost=best_cost,
            best_width=best_w,
            best_height=best_h,
            best_hpwl=best_hpwl,
            best_is_legal=best_legal,
            best_seq_x=best_seq_x,
            best_seq_y=best_seq_y,
            best_blocks=best_blocks,
            total_iterations=self.config.max_iterations,
            legal_iterations_count=legal_count,
            runtime_seconds=elapsed,
            history=history,
            q_controller=q_controller,
        )

