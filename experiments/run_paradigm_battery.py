"""Drosophila Neuroethological Paradigm Battery Runner.

Executes canonical behavioral paradigms in Arena, captures multi-modal telemetry
via ParadigmDataLogger, computes cohort statistics, and outputs summary tables and reports.

Can be run from the command line:
    python run_paradigm_battery.py --paradigms all --trials 3 --steps 500
    python run_paradigm_battery.py --paradigms t-maze,heat-maze,buridan --trials 3 --steps 500

Or imported as a module:
    from experiments.run_paradigm_battery import run_battery
    results = run_battery(paradigms="t-maze,heat-maze", trials=2, steps=100)
"""

import os
import sys
import argparse
import time
from typing import Dict, List, Optional, Any, Union

# Ensure flybrain root is on sys.path
SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SIM_DIR not in sys.path:
    sys.path.insert(0, SIM_DIR)

from arena import Arena
from maze import ExperimentRegistry
from data_logger import ParadigmDataLogger


PRIMARY_METRIC_MAP = {
    't_maze': ('PI', 'performance_index'),
    'tmaze': ('PI', 'performance_index'),
    'y_maze': ('SAR', 'spontaneous_alternation_rate'),
    'ymaze': ('SAR', 'spontaneous_alternation_rate'),
    'heat_maze': ('escape_latency_ms', 'escape_latency_ms'),
    'heatmaze': ('escape_latency_ms', 'escape_latency_ms'),
    'buridan': ('Centrophobism', 'centrophobism_index'),
    'visual_operant': ('LI', 'operant_learning_index'),
    'visualoperant': ('LI', 'operant_learning_index'),
    'wind_tunnel': ('source_reached', 'upwind_progress_mm'),
    'windtunnel': ('source_reached', 'upwind_progress_mm'),
    'looming_escape': ('escape_initiated', 'time_to_collision_at_jump_ms'),
    'loomingescape': ('escape_initiated', 'time_to_collision_at_jump_ms'),
    'optomotor': ('optomotor_gain', 'optomotor_gain'),
    'gap_crossing': ('crossing_success', 'gap_width_mm'),
    'gapcrossing': ('crossing_success', 'gap_width_mm'),
    'circadian_dam': ('sleep_minutes', 'total_sleep_minutes'),
    'circadiandam': ('sleep_minutes', 'total_sleep_minutes'),
    'courtship': ('courtship_index', 'courtship_index'),
    'labyrinth': ('goal_reached', 'path_tortuosity'),
}


def resolve_paradigms(paradigms: Union[str, List[str]]) -> List[str]:
    """Resolves paradigm names from 'all' or comma-separated string / list."""
    all_available = ExperimentRegistry.list_paradigms()
    if isinstance(paradigms, str):
        cleaned = paradigms.strip()
        if cleaned.lower() == 'all':
            return all_available
        items = [p.strip() for p in cleaned.split(',') if p.strip()]
    else:
        items = list(paradigms)

    resolved: List[str] = []
    for item in items:
        norm = item.lower().replace('-', '_')
        if norm in all_available:
            resolved.append(norm)
        else:
            canonical = ExperimentRegistry._canonical_name(item)
            if canonical in ExperimentRegistry._registry:
                match = next((p for p in all_available if ExperimentRegistry._canonical_name(p) == canonical), item)
                resolved.append(match)
            else:
                resolved.append(item)
    return resolved


def format_summary_table(results: Dict[str, Any]) -> str:
    """Formats a human-readable text table of battery cohort metrics."""
    headers = ["Paradigm", "Trials", "Steps", "Primary Metric", "Mean \u00b1 SEM", "p-value", "Cohen's d"]
    col_widths = [18, 8, 8, 20, 20, 10, 11]

    def make_row(cols):
        return "| " + " | ".join(f"{str(c):<{w}}" for c, w in zip(cols, col_widths)) + " |"

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    lines = [
        sep,
        make_row(headers),
        sep,
    ]

    for p_name, data in results.items():
        trials_count = len(data.get("summaries", []))
        steps_count = data.get("steps", 0)
        p_metric = data.get("primary_metric_name", "N/A")
        stats = data.get("primary_metric_stats", {})

        if stats and stats.get("n", 0) > 0:
            mean = stats.get("mean", 0.0)
            sem = stats.get("sem", 0.0)
            pval = stats.get("p_value", 1.0)
            cohen_d = stats.get("cohens_d", 0.0)
            val_str = f"{mean:.3f} \u00b1 {sem:.3f}"
            pval_str = f"{pval:.4f}"
            cohen_str = f"{cohen_d:.3f}"
        else:
            val_str = "N/A"
            pval_str = "N/A"
            cohen_str = "N/A"

        row = [
            p_name,
            str(trials_count),
            str(steps_count),
            p_metric,
            val_str,
            pval_str,
            cohen_str,
        ]
        lines.append(make_row(row))

    lines.append(sep)
    return "\n".join(lines)


def run_battery(
    paradigms: Union[str, List[str]] = "all",
    trials: int = 3,
    steps: int = 500,
    output_dir: str = "./experiment_data/battery",
    dt: float = 0.02,
    brain_type: str = "connectome",
    connectome_mode: str = "surrogate",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Runs a battery of neuroethological paradigms in Arena, logging with ParadigmDataLogger.

    Args:
        paradigms: 'all' or list/comma-string of paradigm names.
        trials: Number of trials per paradigm (default: 3).
        steps: Simulation steps per trial (default: 500).
        output_dir: Directory where per-paradigm telemetry and reports are saved.
        dt: Timestep duration in seconds (default: 0.02).
        brain_type: 'connectome' or 'modular'.
        connectome_mode: 'surrogate' or 'rpc'.
        verbose: Print progress and summary table to stdout.

    Returns:
        Dict containing paradigm results, summary table string, and output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    resolved_list = resolve_paradigms(paradigms)

    if verbose:
        print("=" * 80)
        print("STARTING DROSOPHILA NEUROETHOLOGICAL PARADIGM BATTERY")
        print(f"Paradigms ({len(resolved_list)}): {', '.join(resolved_list)}")
        print(f"Trials per Paradigm: {trials} | Steps per Trial: {steps} | dt: {dt}s")
        print(f"Brain Architecture: {brain_type} ({connectome_mode})")
        print(f"Output Directory: {output_dir}")
        print("=" * 80)

    battery_results: Dict[str, Any] = {}

    for p_idx, p_name in enumerate(resolved_list, 1):
        norm_name = p_name.lower().replace('-', '_')
        p_dir = os.path.join(output_dir, norm_name)
        os.makedirs(p_dir, exist_ok=True)

        if verbose:
            print(f"\n[{p_idx}/{len(resolved_list)}] Executing Paradigm: {p_name} ({trials} trials x {steps} steps)...")

        logger = ParadigmDataLogger(paradigm_name=p_name, output_dir=p_dir, dt=dt)

        for t_idx in range(1, trials + 1):
            trial_id = f"{logger.norm_paradigm}_trial_{t_idx:03d}"
            logger.start_trial(trial_id=trial_id)

            arena = Arena(
                paradigm=p_name,
                brain_type=brain_type,
                connectome_mode=connectome_mode
            )

            last_step_out: Dict[str, Any] = {}
            for s in range(steps):
                step_out = arena.step(dt=dt)
                logger.log_step(step_out)
                last_step_out = step_out

            trial_metrics = last_step_out.get('paradigm_metrics', {})
            logger.end_trial(trial_metrics)

        # Generate cohort statistical markdown report
        report_content = logger.generate_cohort_report()
        cohort_metrics = logger.compute_cohort_metrics()

        # Identify primary metric
        primary_candidates = PRIMARY_METRIC_MAP.get(norm_name, ('PI', 'performance_index'))
        prim_metric_name = "N/A"
        prim_stats: Dict[str, float] = {}

        for cand in primary_candidates:
            if cand in cohort_metrics:
                prim_metric_name = cand
                prim_stats = cohort_metrics[cand]
                break

        if prim_metric_name == "N/A" and cohort_metrics:
            prim_metric_name = next(iter(cohort_metrics.keys()))
            prim_stats = cohort_metrics[prim_metric_name]

        battery_results[p_name] = {
            "paradigm": p_name,
            "trials": trials,
            "steps": steps,
            "output_dir": p_dir,
            "summaries": logger.trial_summaries,
            "cohort_metrics": cohort_metrics,
            "primary_metric_name": prim_metric_name,
            "primary_metric_stats": prim_stats,
            "report_path": logger.last_report_path,
            "report": report_content,
        }

        if verbose and prim_stats:
            print(f"  -> Completed. {prim_metric_name} = {prim_stats['mean']:.3f} \u00b1 {prim_stats['sem']:.3f} (p={prim_stats['p_value']:.4f})")

    summary_table = format_summary_table(battery_results)

    if verbose:
        print("\n" + "=" * 80)
        print("BATTERY EXECUTION SUMMARY")
        print("=" * 80)
        print(summary_table)
        print(f"\nAll telemetry, summaries, and reports saved to: {output_dir}\n")

    return {
        "paradigms": battery_results,
        "summary_table": summary_table,
        "output_dir": output_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Drosophila Neuroethological Paradigm Battery")
    parser.add_argument(
        "--paradigms",
        type=str,
        default="all",
        help="Paradigms to run: 'all' or comma-separated list (e.g. t-maze,heat-maze,buridan)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials per paradigm (default: 3)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of simulation steps per trial (default: 500)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiment_data/battery",
        help="Base directory for logs and reports (default: ./experiment_data/battery)"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Simulation step size dt (default: 0.02)"
    )
    parser.add_argument(
        "--brain-type",
        type=str,
        default="connectome",
        choices=["connectome", "modular"],
        help="Brain controller architecture (default: connectome)"
    )
    parser.add_argument(
        "--connectome-mode",
        type=str,
        default="surrogate",
        choices=["surrogate", "rpc"],
        help="Connectome bridge mode (default: surrogate)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose logging"
    )

    args = parser.parse_args()

    run_battery(
        paradigms=args.paradigms,
        trials=args.trials,
        steps=args.steps,
        output_dir=args.output_dir,
        dt=args.dt,
        brain_type=args.brain_type,
        connectome_mode=args.connectome_mode,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
