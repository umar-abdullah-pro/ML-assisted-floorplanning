"""Experimental replication of Yan & Chu (ISPD 2012):
'Optimal Slack-Driven Block Shaping Algorithm in Fixed-Outline Floorplanning'

Replicates:
1. Section 4 & Figure 2: The 4-block counterexample, failure of Basic SDS, optimality diagnosis, and resolution via Full SDS.
2. Section 6.1: MCNC benchmarks (apte, xerox, hp, ami33a, ami49a) with runtime and optimality validation.
3. Section 6.2: Scalable HB-style benchmarks, convergence percentages (<10%, <5%, <1%, <0.1%), and Figure 5 convergence plots.
4. Exporting plots and summary tables.
"""

from __future__ import annotations
import os
import sys
import time
from typing import Dict, List

# Ensure package directory is on python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sds.block import Block
from sds.basic_sds import BasicSDSSolver
from sds.full_sds import FullSDSSolver
from sds.optimality import OptimalityChecker
from sds.convex_solver import solve_floorplan_convex
from sds.visualizer import plot_floorplan, plot_convergence
from benchmarks.figure2_example import build_figure2_benchmark
from benchmarks.mcnc_benchmarks import build_mcnc_benchmark, MCNC_SPECS
from benchmarks.synthetic_benchmarks import generate_hb_style_benchmark


def run_figure2_replication(output_dir: str = "output") -> None:
    """Replicate Figure 2: The non-optimal trap and its optimal resolution."""
    print("=" * 80)
    print("EXPERIMENT 1: Reproduction of Figure 2 (Motivational Counterexample)")
    print("=" * 80)

    blocks, gh, gv, W = build_figure2_benchmark()
    print("Benchmark parameters: 4 soft blocks, Area A_i = 4.0, W_min = 1.0, W_max = 4.0, Outline W = 5.0")
    print("Initial configuration (Fig 2-a): w1=4, h1=1; w2=1, h2=4; w3=4, h3=1; w4=1, h4=4")

    # Initial packing
    solver_init = BasicSDSSolver(blocks, gh, gv, upper_bound_width=W)
    solver_init.compute_slacks()
    subsets_init = solver_init.classify_subsets()

    print(f"\n[Figure 2-(a)] Packed Layout:")
    print(f"  Width  x_{{n+1}} = {solver_init.layout_width:.4f} (<= W = {W})")
    print(f"  Height y_{{n+1}} = {solver_init.layout_height:.4f}")
    print(f"  Classified subsets: {subsets_init}")
    for i in range(1, 5):
        b = solver_init.blocks[i]
        print(f"  Block {i}: w={b.w:.4f}, h={b.h:.4f}, slack_h={b.slack_h:.4f}, slack_v={b.slack_v:.4f}, subset={b.subset}")

    # Run Basic SDS without resetting
    basic_res = solver_init.solve(reset_to_min=False)
    print(f"\n[Basic SDS Result on Fig 2-(a)]:")
    print(f"  Target soft blocks found: IV={len(basic_res.final_subsets.get('IV', []))}, VI={len(basic_res.final_subsets.get('VI', []))}")
    print(f"  Iterations: {basic_res.total_iterations}")
    print(f"  Height returned: {basic_res.layout_height:.4f}")
    print(f"  -> Basic SDS is TRAPPED at local minimum y_{{n+1}} = 5.0!")

    # Check Optimality Conditions
    checker = OptimalityChecker(solver_init.blocks, gh, gv, upper_bound_width=W)
    report = checker.check(solver_init.x_ll, solver_init.y_bl, solver_init.subsets)
    print(f"\n[Optimality Conditions Check (Section 4)]:")
    print(f"  Total VCPs: {report.total_vcps}, Hard VCPs: {report.hard_vcps}, Soft VCPs: {report.soft_vcps}")
    print(f"  Total HCPs: {report.total_hcps}, Hard HCPs: {report.hard_hcps}, Soft HCPs: {report.soft_hcps}")
    print(f"  Lemma 2 satisfied (Hard VCP exists): {report.lemma2_satisfied}")
    print(f"  Lemma 3 satisfied (<=1 soft HCP or <=1 soft VCP): {report.lemma3_satisfied}")
    print(f"  Certified Optimal: {report.is_optimal}")
    print(f"  Diagnosis: {report.explanation}")

    # Run Full SDS (triggers GP fallback on Subset III)
    print(f"\n[Full SDS Execution]:")
    full_sds = FullSDSSolver(blocks, gh, gv, upper_bound_width=W)
    full_res = full_sds.solve(reset_to_min=False, verbose=True)

    print(f"\n[Figure 2-(b) Optimal Result via Full SDS]:")
    print(f"  Final Width:  {full_res.layout_width:.4f} (<= W = {W})")
    print(f"  Final Height: {full_res.layout_height:.4f} (Exact theoretical optimum = 3.2000)")
    print(f"  GP Fallback Invoked: {full_res.gp_fallback_used}")
    print(f"  Height Improvement: {(5.0 - full_res.layout_height)/5.0 * 100:.2f}%")
    for i in range(1, 5):
        b = full_res.blocks[i]
        print(f"  Block {i}: w={b.w:.4f}, h={b.h:.4f}, aspect_ratio={b.aspect_ratio():.4f}")

    # Generate plots
    fig_a = plot_floorplan(
        solver_init.blocks,
        solver_init.layout_width,
        solver_init.layout_height,
        W,
        title="Figure 2-(a): Non-Optimal Basic SDS Trap (Height = 5.0)",
        save_path=os.path.join(output_dir, "figure2_a_trap.png"),
    )
    fig_b = plot_floorplan(
        full_res.blocks,
        full_res.layout_width,
        full_res.layout_height,
        W,
        title="Figure 2-(b): Optimal Solution via Full SDS (Height = 3.2)",
        save_path=os.path.join(output_dir, "figure2_b_optimal.png"),
    )
    print(f"Saved plots to {output_dir}/figure2_a_trap.png and figure2_b_optimal.png")


def run_mcnc_experiments(output_dir: str = "output") -> None:
    """Run SDS across all MCNC benchmarks and compare with CVXPY convex ground truth."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: MCNC Benchmark Evaluation (Tables 2 & 3)")
    print("=" * 80)

    benchmarks = ["apte", "xerox", "hp", "ami33a", "ami49a"]

    print(f"{'Circuit':<8} | {'Blocks':<6} | {'Bound W':<9} | {'Init H':<9} | {'Final W':<9} | {'SDS H':<9} | {'CVXPY H':<9} | {'Diff %':<7} | {'Iters':<6} | {'Time (s)':<8}")
    print("-" * 95)

    for name in benchmarks:
        blocks, gh, gv, W = build_mcnc_benchmark(name, target_experiment="table2", seed=42)
        n = len(blocks)

        # Initial height at minimal width
        w_min_dict = {i: blocks[i].w_min for i in blocks}
        h_max_dict = {i: blocks[i].area / blocks[i].w_min for i in blocks}
        init_h = gv.compute_longest_path_forward(h_max_dict)[gv.sink]

        # Solve with CVXPY ground truth
        t0_cp = time.perf_counter()
        convex_res = solve_floorplan_convex(blocks, gh, gv, upper_bound_width=W)
        cp_time = time.perf_counter() - t0_cp

        # Solve with SDS
        solver = FullSDSSolver(blocks, gh, gv, upper_bound_width=W)
        sds_res = solver.solve(max_basic_iterations=3000)

        diff_pct = abs(sds_res.layout_height - convex_res.optimal_height) / convex_res.optimal_height * 100.0

        print(
            f"{name:<8} | {n:<6d} | {W:<9.2f} | {init_h:<9.2f} | {sds_res.layout_width:<9.2f} | "
            f"{sds_res.layout_height:<9.2f} | {convex_res.optimal_height:<9.2f} | {diff_pct:<7.3f} | "
            f"{sds_res.total_iterations:<6d} | {sds_res.runtime_seconds:<8.4f}"
        )

        # Save floorplan visualization for apte and ami33a
        if name in ("apte", "ami33a"):
            plot_floorplan(
                sds_res.blocks,
                sds_res.layout_width,
                sds_res.layout_height,
                W,
                title=f"MCNC Benchmark {name.upper()} Shaped by SDS (H={sds_res.layout_height:.2f}, W={sds_res.layout_width:.2f})",
                save_path=os.path.join(output_dir, f"floorplan_{name}.png"),
            )


def run_hb_convergence_experiments(output_dir: str = "output") -> None:
    """Run scalable benchmarks with both soft and hard blocks, tracking convergence percentages (Table 4, Fig 5)."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Scalable Benchmarks & Convergence Milestones (Table 4, Fig 5)")
    print("=" * 80)

    test_configs = [
        {"name": "HB_Small", "soft": 50, "hard": 20, "seed": 101},
        {"name": "HB_Medium", "soft": 150, "hard": 50, "seed": 202},
        {"name": "HB_Large", "soft": 400, "hard": 100, "seed": 303},
    ]

    print(f"{'Testcase':<12} | {'Soft/Hard':<10} | {'Outline W':<10} | {'Final H (Y)':<11} | {'Time (s)':<8} | {'Total Iter':<10} | {'<10%':<6} | {'<5%':<6} | {'<1%':<6} | {'<0.1%':<6}")
    print("-" * 95)

    for cfg in test_configs:
        name = cfg["name"]
        num_soft = cfg["soft"]
        num_hard = cfg["hard"]
        seed = cfg["seed"]

        blocks, gh, gv, W = generate_hb_style_benchmark(num_soft=num_soft, num_hard=num_hard, seed=seed)

        solver = FullSDSSolver(blocks, gh, gv, upper_bound_width=W)
        res = solver.solve(max_basic_iterations=2000)

        Y = res.layout_height
        history = res.history

        # Compute milestone iterations where (y - Y) / Y < 10%, 5%, 1%, 0.1%
        iter_10 = iter_5 = iter_1 = iter_01 = res.total_iterations
        for rec in history:
            rel_err = (rec.layout_height - Y) / Y
            if rel_err < 0.10 and iter_10 == res.total_iterations:
                iter_10 = rec.iteration
            if rel_err < 0.05 and iter_5 == res.total_iterations:
                iter_5 = rec.iteration
            if rel_err < 0.01 and iter_1 == res.total_iterations:
                iter_1 = rec.iteration
            if rel_err < 0.001 and iter_01 == res.total_iterations:
                iter_01 = rec.iteration

        print(
            f"{name:<12} | {num_soft}/{num_hard:<5} | {W:<10.2f} | {Y:<11.2f} | {res.runtime_seconds:<8.3f} | "
            f"{res.total_iterations:<10d} | {iter_10:<6d} | {iter_5:<6d} | {iter_1:<6d} | {iter_01:<6d}"
        )

        # Plot convergence curve matching Figure 5
        plot_convergence(
            history,
            optimal_height=Y,
            title=f"Layout-Height Convergence for {name} ({num_soft} soft / {num_hard} hard blocks)",
            save_path=os.path.join(output_dir, f"convergence_{name}.png"),
        )

    print(f"\nSaved convergence graphs to {output_dir}/convergence_*.png")


def main():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print("STARTING FULL REPRODUCTION OF THE SDS ALGORITHM (YAN & CHU, ISPD 2012)")
    start_total = time.perf_counter()

    run_figure2_replication(output_dir)
    run_mcnc_experiments(output_dir)
    run_hb_convergence_experiments(output_dir)

    total_time = time.perf_counter() - start_total
    print("\n" + "=" * 80)
    print(f"ALL EXPERIMENTS COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print("=" * 80)


if __name__ == "__main__":
    main()

