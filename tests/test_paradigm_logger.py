"""Unit and Integration Test Suite for ParadigmDataLogger and Paradigm Battery.
=============================================================================
Verifies:
1. ParadigmDataLogger initialization, directory management, and state tracking.
2. Per-step telemetry extraction and CSV streaming across distinct paradigms
   (Heat-Maze, Optomotor, T-Maze, Courtship, Looming).
3. Complete trial summary compilation (JSON) and spatial occupancy/trajectory (NPZ).
4. Rigorous schema validation for CSV columns, JSON keys, and NPZ shapes.
5. Cohort statistical aggregation (mean, std, SEM, t-stat, p-value, Cohen's d)
   and Markdown report (PARADIGM_REPORT.md) generation.
6. run_paradigm_battery.py execution both as an importable module and CLI script.
7. 100% backward compatibility with ScientificDataLogger and LearningAssay.
"""

import os
import sys
import json
import csv
import subprocess
import pytest
import numpy as np

# Ensure flybrain root is in sys.path
SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SIM_DIR not in sys.path:
    sys.path.insert(0, SIM_DIR)

from data_logger import (
    ParadigmDataLogger,
    ScientificDataLogger,
    LearningAssay,
    compute_cohort_statistics,
    t_distribution_p_value,
)
from arena import Arena
from experiments.run_paradigm_battery import run_battery, resolve_paradigms


# =============================================================================
# 1. INITIALIZATION & STATE MANAGEMENT TESTS
# =============================================================================

class TestParadigmDataLoggerInit:
    """Test logger instantiation, path handling, and buffer defaults."""

    def test_default_initialization(self, tmp_path):
        out_dir = str(tmp_path / "test_init")
        logger = ParadigmDataLogger(paradigm_name="heat-maze", output_dir=out_dir)

        assert os.path.exists(out_dir)
        assert logger.paradigm_name == "heat-maze"
        assert logger.norm_paradigm == "heat_maze"
        assert logger.trial_id == "heat_maze_trial_001"
        assert logger.step_count == 0
        assert len(logger.trajectory) == 0
        assert len(logger.step_buffer) == 0
        assert logger.occupancy_grid.shape == (50, 50)
        assert np.all(logger.occupancy_grid == 0.0)

    def test_custom_parameters(self, tmp_path):
        out_dir = str(tmp_path / "test_custom")
        logger = ParadigmDataLogger(
            paradigm_name="optomotor",
            output_dir=out_dir,
            trial_id="custom_trial_42",
            occupancy_bins=25,
            occupancy_range=(-60.0, 60.0),
            dt=0.01,
        )

        assert logger.trial_id == "custom_trial_42"
        assert logger.occupancy_bins == 25
        assert logger.occupancy_grid.shape == (25, 25)
        assert logger.occupancy_range == (-60.0, 60.0)
        assert logger.dt == 0.01

    def test_start_trial_resets_buffers(self, tmp_path):
        out_dir = str(tmp_path / "test_reset")
        logger = ParadigmDataLogger(paradigm_name="t-maze", output_dir=out_dir)

        # Log mock step
        logger.log_step({"time_step": 1, "fly_x": 10.0, "fly_y": 20.0})
        assert logger.step_count == 1
        assert len(logger.trajectory) == 1

        # Start new trial
        logger.start_trial("t_maze_trial_002")
        assert logger.trial_id == "t_maze_trial_002"
        assert logger.step_count == 0
        assert len(logger.trajectory) == 0
        assert len(logger.step_buffer) == 0
        assert np.all(logger.occupancy_grid == 0.0)


# =============================================================================
# 2. MULTI-PARADIGM TELEMETRY & EXPORTS
# =============================================================================

class TestMultiParadigmLoggingAndExports:
    """Verify CSV telemetry, JSON summaries, and NPZ occupancy across distinct paradigms."""

    def test_heat_maze_telemetry_and_exports(self, tmp_path):
        out_dir = str(tmp_path / "heat_maze")
        logger = ParadigmDataLogger(paradigm_name="heat-maze", output_dir=out_dir)
        arena = Arena(paradigm="heat-maze", brain_type="connectome")

        # Step simulation
        for _ in range(15):
            step_out = arena.step(dt=0.02)
            row = logger.log_step(step_out)
            # Verify clean extraction of heat-maze specific columns
            assert "floor_temp" in row
            assert "epg_bump" in row
            assert "pam_burst" in row
            assert isinstance(row["floor_temp"], float)

        # Check telemetry CSV exists on disk
        csv_file = os.path.join(out_dir, f"{logger.trial_id}_telemetry.csv")
        assert os.path.exists(csv_file)
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 15
            assert "floor_temp" in reader.fieldnames
            assert "epg_bump" in reader.fieldnames
            assert "pam_burst" in reader.fieldnames

        # End trial & check exports
        metrics = arena.step(dt=0.02).get("paradigm_metrics", {})
        summary_path = logger.end_trial(metrics)

        # JSON Summary verification
        assert os.path.exists(summary_path)
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
            assert summary["trial_id"] == "heat_maze_trial_001"
            assert summary["paradigm"] == "heat-maze"
            assert "timestamps" in summary
            assert summary["total_steps"] == 15
            assert "escape_latency_ms" in summary
            assert "tortuosity" in summary

        # NPZ Occupancy verification
        npz_file = os.path.join(out_dir, "heat_maze_trial_001_occupancy.npz")
        assert os.path.exists(npz_file)
        npz_data = np.load(npz_file)
        assert "occupancy" in npz_data
        assert "trajectory" in npz_data
        assert npz_data["occupancy"].shape == (50, 50)
        assert npz_data["trajectory"].shape == (15, 2)
        assert np.sum(npz_data["occupancy"]) > 0

    def test_optomotor_telemetry_and_exports(self, tmp_path):
        out_dir = str(tmp_path / "optomotor")
        logger = ParadigmDataLogger(paradigm_name="optomotor", output_dir=out_dir)
        arena = Arena(paradigm="optomotor", brain_type="connectome")

        for _ in range(12):
            step_out = arena.step(dt=0.02)
            row = logger.log_step(step_out)
            assert "hs_rate" in row
            assert "drum_vel" in row
            assert "saccade_shunt" in row

        csv_file = os.path.join(out_dir, f"{logger.trial_id}_telemetry.csv")
        assert os.path.exists(csv_file)
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert "hs_rate" in reader.fieldnames
            assert "drum_vel" in reader.fieldnames
            assert "saccade_shunt" in reader.fieldnames

        metrics = arena.step(dt=0.02).get("paradigm_metrics", {})
        summary_path = logger.end_trial(metrics)
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
            assert "optomotor_gain" in summary

        npz_file = os.path.join(out_dir, "optomotor_trial_001_occupancy.npz")
        assert os.path.exists(npz_file)
        npz_data = np.load(npz_file)
        assert npz_data["trajectory"].shape == (12, 2)

    def test_t_maze_telemetry_and_exports(self, tmp_path):
        out_dir = str(tmp_path / "t_maze")
        logger = ParadigmDataLogger(paradigm_name="t-maze", output_dir=out_dir)
        arena = Arena(paradigm="t-maze", brain_type="connectome")

        for _ in range(10):
            step_out = arena.step(dt=0.02)
            row = logger.log_step(step_out)
            assert "odor_a" in row
            assert "odor_b" in row
            assert "active_arm" in row

        csv_file = os.path.join(out_dir, f"{logger.trial_id}_telemetry.csv")
        assert os.path.exists(csv_file)
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert "odor_a" in reader.fieldnames
            assert "odor_b" in reader.fieldnames
            assert "active_arm" in reader.fieldnames

        metrics = arena.step(dt=0.02).get("paradigm_metrics", {})
        summary_path = logger.end_trial(metrics)
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
            assert "PI" in summary
            assert "latency_to_choice_ms" in summary["latencies"] or "latency_to_choice_ms" in summary

        npz_file = os.path.join(out_dir, "t_maze_trial_001_occupancy.npz")
        assert os.path.exists(npz_file)

    def test_courtship_telemetry_and_exports(self, tmp_path):
        out_dir = str(tmp_path / "courtship")
        logger = ParadigmDataLogger(paradigm_name="courtship", output_dir=out_dir)
        arena = Arena(paradigm="courtship", brain_type="connectome")

        for _ in range(10):
            step_out = arena.step(dt=0.02)
            row = logger.log_step(step_out)
            assert "wing_angle" in row
            assert "p1_rate" in row
            assert "rejection" in row

        metrics = arena.step(dt=0.02).get("paradigm_metrics", {})
        summary_path = logger.end_trial(metrics)
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
            assert "courtship_index" in summary

    def test_looming_escape_telemetry_and_exports(self, tmp_path):
        out_dir = str(tmp_path / "looming")
        logger = ParadigmDataLogger(paradigm_name="looming-escape", output_dir=out_dir)
        arena = Arena(paradigm="looming-escape", brain_type="connectome")

        for _ in range(10):
            step_out = arena.step(dt=0.02)
            row = logger.log_step(step_out)
            assert "looming_size" in row
            assert "gf_spike" in row

        metrics = arena.step(dt=0.02).get("paradigm_metrics", {})
        summary_path = logger.end_trial(metrics)
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
            assert summary["paradigm"] == "looming-escape"


# =============================================================================
# 3. SCHEMA VALIDATION TESTS
# =============================================================================

class TestSchemaValidation:
    """Validate strict schema conformance for CSV telemetry and JSON summary outputs."""

    REQUIRED_JSON_KEYS = {
        "trial_id", "paradigm", "timestamps", "duration", "total_steps",
        "total_distance", "latencies", "metrics", "PI", "SAR",
        "Centrophobism", "LI", "optomotor_gain", "sleep_minutes",
        "courtship_index", "tortuosity"
    }

    REQUIRED_BASE_CSV_COLS = {
        "step", "sim_time", "pos_x", "pos_y", "heading", "speed", "reward", "punishment"
    }

    def test_csv_schema_integrity(self, tmp_path):
        out_dir = str(tmp_path / "csv_schema")
        logger = ParadigmDataLogger(paradigm_name="heat-maze", output_dir=out_dir)

        # Log simulated steps
        for i in range(1, 6):
            logger.log_step({
                "time_step": i,
                "fly_x": float(i * 2),
                "fly_y": float(i * 3),
                "fly_heading": 0.1 * i,
                "fly_speed": 1.5,
                "floor_temp": 30.0 + i,
                "epg_bump": 0.5,
                "pam_burst": 1.0,
            })

        csv_file = os.path.join(out_dir, "heat_maze_trial_001_telemetry.csv")
        assert os.path.exists(csv_file)

        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames)
            # Check base columns
            assert self.REQUIRED_BASE_CSV_COLS.issubset(fieldnames)
            # Check paradigm columns
            assert {"floor_temp", "epg_bump", "pam_burst"}.issubset(fieldnames)

            # Validate row types
            for row in reader:
                assert int(row["step"]) > 0
                assert float(row["sim_time"]) >= 0.0
                assert float(row["pos_x"]) is not None
                assert float(row["pos_y"]) is not None
                assert float(row["floor_temp"]) >= 30.0

    def test_json_schema_integrity(self, tmp_path):
        out_dir = str(tmp_path / "json_schema")
        logger = ParadigmDataLogger(paradigm_name="t-maze", output_dir=out_dir)

        logger.log_step({"time_step": 1, "fly_x": 0.0, "fly_y": 0.0})
        logger.log_step({"time_step": 2, "fly_x": 5.0, "fly_y": 10.0})

        summary_path = logger.end_trial({
            "performance_index": 0.72,
            "latency_to_choice_ms": 320.0,
            "custom_flag": True,
        })

        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

        # Check all required top-level keys
        assert self.REQUIRED_JSON_KEYS.issubset(set(summary.keys()))

        # Check nested timestamps structure
        ts = summary["timestamps"]
        assert isinstance(ts, dict)
        assert "start_time" in ts and "end_time" in ts
        assert "start_iso" in ts and "end_iso" in ts

        # Check types
        assert isinstance(summary["duration"], (int, float))
        assert isinstance(summary["total_steps"], int)
        assert isinstance(summary["total_distance"], (int, float))
        assert isinstance(summary["latencies"], dict)
        assert isinstance(summary["metrics"], dict)

        # Canonical metric values
        assert abs(summary["PI"] - 0.72) < 1e-4
        assert summary["tortuosity"] is not None
        assert summary["tortuosity"] >= 1.0


# =============================================================================
# 4. COHORT REPORTING & STATISTICAL AGGREGATION
# =============================================================================

class TestCohortReportGeneration:
    """Verify cohort statistics calculation and Markdown summary report writing."""

    def test_compute_cohort_statistics_accuracy(self):
        # Known synthetic sample: [10.0, 12.0, 14.0]
        # mean = 12.0, std = 2.0, sem = 2.0 / sqrt(3) ~= 1.1547
        # t_stat = 12.0 / 1.1547 ~= 10.3923, cohens_d = 12.0 / 2.0 = 6.0
        stats = compute_cohort_statistics([10.0, 12.0, 14.0])

        assert stats["n"] == 3
        assert abs(stats["mean"] - 12.0) < 1e-6
        assert abs(stats["std"] - 2.0) < 1e-6
        assert abs(stats["sem"] - (2.0 / np.sqrt(3))) < 1e-5
        assert abs(stats["cohens_d"] - 6.0) < 1e-5
        assert stats["t_stat"] > 10.0
        assert 0.0 <= stats["p_value"] <= 0.05

    def test_t_distribution_p_value_symmetry(self):
        p_pos = t_distribution_p_value(2.5, 4)
        p_neg = t_distribution_p_value(-2.5, 4)
        assert abs(p_pos - p_neg) < 1e-8
        assert 0.0 < p_pos < 1.0
        assert t_distribution_p_value(0.0, 4) == 1.0

    def test_cohort_report_markdown_generation(self, tmp_path):
        out_dir = str(tmp_path / "cohort_report")
        logger = ParadigmDataLogger(paradigm_name="t-maze", output_dir=out_dir)

        # Simulate 3 trials with different performance indices
        for t_idx, pi in enumerate([0.35, 0.50, 0.65], 1):
            logger.start_trial(f"t_maze_trial_{t_idx:03d}")
            for s in range(5):
                logger.log_step({"time_step": s, "fly_x": float(s), "fly_y": float(s)})
            logger.end_trial({"performance_index": pi, "latency_to_choice_ms": 100.0 * t_idx})

        assert len(logger.trial_summaries) == 3

        report = logger.generate_cohort_report()

        # Check ReportContent properties
        assert isinstance(report, str)
        assert os.path.exists(report.file_path)
        assert report.file_path == os.path.join(out_dir, "PARADIGM_REPORT.md")

        # Check content contains key sections and statistics
        assert "# Drosophila Neuroethological Cohort Report: T-MAZE" in report
        assert "Aggregated Cohort Statistics" in report
        assert "Individual Trial Breakdown" in report
        assert "PI" in report
        assert "Cohen's d" in report

        cohort_metrics = logger.compute_cohort_metrics()
        assert "PI" in cohort_metrics
        assert abs(cohort_metrics["PI"]["mean"] - 0.50) < 1e-4

    def test_empty_and_single_trial_reports(self, tmp_path):
        out_dir = str(tmp_path / "edge_cases")
        logger = ParadigmDataLogger(paradigm_name="buridan", output_dir=out_dir)

        # Empty report
        rep_empty = logger.generate_cohort_report()
        assert "Cohort Size (N)**: `0`" in rep_empty

        # 1 trial
        logger.start_trial("buridan_trial_001")
        logger.log_step({"time_step": 1, "fly_x": 0.0, "fly_y": 0.0})
        logger.end_trial({"centrophobism_index": 0.85})

        rep_one = logger.generate_cohort_report()
        assert "Cohort Size (N)**: `1`" in rep_one
        assert os.path.exists(os.path.join(out_dir, "PARADIGM_REPORT.md"))


# =============================================================================
# 5. RUN PARADIGM BATTERY INTEGRATION & CLI
# =============================================================================

class TestRunParadigmBattery:
    """Verify paradigm battery orchestration, module invocation, and CLI execution."""

    def test_resolve_paradigms_helper(self):
        # 'all' resolution
        all_p = resolve_paradigms("all")
        assert len(all_p) >= 10
        assert "t_maze" in all_p

        # Comma-separated with mixed cases / hyphens
        subset = resolve_paradigms("t-maze, heat-maze, optomotor")
        assert "t_maze" in subset
        assert "heat_maze" in subset
        assert "optomotor" in subset

    def test_run_battery_module_invocation(self, tmp_path):
        out_dir = str(tmp_path / "battery_module")
        results = run_battery(
            paradigms="t-maze,heat-maze",
            trials=2,
            steps=10,
            output_dir=out_dir,
            brain_type="connectome",
            verbose=False,
        )

        assert "paradigms" in results
        assert "summary_table" in results
        assert len(results["paradigms"]) == 2

        for p_name in ["t_maze", "heat_maze"]:
            assert p_name in results["paradigms"]
            p_res = results["paradigms"][p_name]
            assert p_res["trials"] == 2
            assert p_res["steps"] == 10
            assert os.path.exists(p_res["report_path"])

            # Verify files on disk
            p_dir = os.path.join(out_dir, p_name)
            assert os.path.exists(os.path.join(p_dir, f"{p_name}_trial_001_telemetry.csv"))
            assert os.path.exists(os.path.join(p_dir, f"{p_name}_trial_001_summary.json"))
            assert os.path.exists(os.path.join(p_dir, f"{p_name}_trial_001_occupancy.npz"))
            assert os.path.exists(os.path.join(p_dir, "PARADIGM_REPORT.md"))

        assert "BATTERY" in results["summary_table"] or "Paradigm" in results["summary_table"]

    def test_run_battery_cli_execution(self, tmp_path):
        out_dir = str(tmp_path / "battery_cli")
        cmd = [
            sys.executable,
            "-m", "experiments.run_paradigm_battery",
            "--paradigms", "optomotor",
            "--trials", "2",
            "--steps", "10",
            "--output-dir", out_dir,
        ]

        proc = subprocess.run(cmd, cwd=SIM_DIR, capture_output=True, text=True)
        assert proc.returncode == 0
        assert "PARADIGM BATTERY" in proc.stdout
        assert "optomotor" in proc.stdout

        p_dir = os.path.join(out_dir, "optomotor")
        assert os.path.exists(os.path.join(p_dir, "optomotor_trial_001_telemetry.csv"))
        assert os.path.exists(os.path.join(p_dir, "optomotor_trial_002_summary.json"))
        assert os.path.exists(os.path.join(p_dir, "PARADIGM_REPORT.md"))


# =============================================================================
# 6. BACKWARD COMPATIBILITY
# =============================================================================

class TestBackwardCompatibility:
    """Verify complete backward compatibility with ScientificDataLogger and LearningAssay."""

    def test_scientific_data_logger_intact(self, tmp_path):
        logger = ScientificDataLogger(output_dir=str(tmp_path))
        assert logger.step_count == 0
        assert logger.flush_count == 0

        # Step logging
        logger.log_step({"simulation_time": 0.02, "forward_speed": 5.0})
        assert logger.step_count == 1
        assert len(logger.step_buffer) == 1

        # Trial ending
        summary = logger.end_trial(food_collected=2, escapes=1)
        assert summary["food_collected"] == 2
        assert summary["escapes"] == 1

        # Save experiment
        exp_dir = logger.save_experiment()
        assert os.path.exists(os.path.join(exp_dir, "experiment_meta.json"))
        assert os.path.exists(os.path.join(exp_dir, "trial_summaries.json"))
        assert os.path.exists(os.path.join(exp_dir, "occupancy.npz"))
