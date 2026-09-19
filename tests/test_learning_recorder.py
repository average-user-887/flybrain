"""Tests for durable learning records (learning_recorder.py).

Covers:
1. JsonlWriter append-only semantics, durability flags and timestamp rotation
   that never deletes data.
2. LearningRecorder session manifest, trial de-duplication and summaries.
3. RecorderThread draining a runner-like object under its lock, including the
   real ContinuousExperimentRunner from neurofly_daemon.
4. Data-directory resolution precedence (flag > env > default).
"""

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from learning_recorder import (  # noqa: E402
    SCHEMA_VERSION,
    JsonlWriter,
    LearningRecorder,
    RecorderThread,
    read_jsonl,
    resolve_data_dir,
    summarise_runner,
)


# =============================================================================
# 1. JsonlWriter
# =============================================================================

class TestJsonlWriter:
    def test_append_is_one_json_object_per_line(self, tmp_path):
        path = tmp_path / "a.jsonl"
        with JsonlWriter(path, fsync=False) as w:
            w.append({"n": 1, "arr": np.arange(3), "f": np.float32(1.5)})
            w.append({"n": 2, "unicode": "Drosophila é"})
        rows = read_jsonl(path)
        assert rows == [{"n": 1, "arr": [0, 1, 2], "f": 1.5}, {"n": 2, "unicode": "Drosophila é"}]
        assert w.lines_written == 2

    def test_reopen_appends_never_truncates(self, tmp_path):
        path = tmp_path / "a.jsonl"
        with JsonlWriter(path, fsync=False) as w:
            w.append({"n": 1})
        with JsonlWriter(path, fsync=False) as w:
            w.append({"n": 2})
        assert [r["n"] for r in read_jsonl(path)] == [1, 2]

    def test_rotation_keeps_every_line(self, tmp_path):
        path = tmp_path / "r.jsonl"
        w = JsonlWriter(path, max_bytes=200, fsync=False)
        for i in range(40):
            w.append({"i": i, "pad": "x" * 20})
        w.close()
        assert w.rotations >= 2
        rotated = w.rotated_files()
        assert rotated and all(p.name.startswith("r.jsonl.") for p in rotated)
        seen = []
        for p in rotated + [path]:
            seen.extend(r["i"] for r in read_jsonl(p))
        assert sorted(seen) == list(range(40))
        # Every rotated file respects the cap; nothing was deleted.
        assert all(p.stat().st_size <= 200 for p in rotated)

    def test_fsync_path_does_not_raise(self, tmp_path):
        with JsonlWriter(tmp_path / "s.jsonl", fsync=True) as w:
            w.append({"ok": True})
        assert read_jsonl(tmp_path / "s.jsonl") == [{"ok": True}]


# =============================================================================
# 2. LearningRecorder
# =============================================================================

class TestLearningRecorder:
    def test_session_manifest_and_log(self, tmp_path):
        rec = LearningRecorder(tmp_path / "d", session={"paradigm": "t-maze", "port": 1}, fsync=False)
        manifest = json.loads((tmp_path / "d" / "session.json").read_text())
        assert manifest["schema_version"] == SCHEMA_VERSION
        assert manifest["paradigm"] == "t-maze" and manifest["port"] == 1
        assert manifest["session_id"] == rec.session_id
        assert "argv" not in manifest  # never persist command lines (tokens)
        assert "hostname" not in manifest
        log = read_jsonl(tmp_path / "d" / "sessions.jsonl")
        assert len(log) == 1 and log[0]["session_id"] == rec.session_id
        rec.close()
        rec2 = LearningRecorder(tmp_path / "d", fsync=False)
        assert len(read_jsonl(tmp_path / "d" / "sessions.jsonl")) == 2
        rec2.close()

    def test_trials_are_deduplicated_by_trial_number(self, tmp_path):
        rec = LearningRecorder(tmp_path, fsync=False)
        history = [
            {"trial": 1, "paradigm": "t-maze", "step": 10, "metric": 0.5, "timestamp": 1.0},
            {"trial": 2, "paradigm": "t-maze", "step": 20, "metric": 0.6, "timestamp": 2.0},
        ]
        assert rec.record_trials(history) == 2
        assert rec.record_trials(history) == 0
        history.append({"trial": 3, "paradigm": "t-maze", "step": 30, "metric": 0.7, "timestamp": 3.0})
        assert rec.record_trials(history[1:]) == 1          # tolerates truncated history
        assert rec.record_trials([{"no_trial": True}]) == 0  # malformed entries are skipped
        rec.close()
        rows = read_jsonl(tmp_path / "trials.jsonl")
        assert [r["trial"] for r in rows] == [1, 2, 3]
        for r in rows:
            assert r["type"] == "trial"
            assert r["schema_version"] == SCHEMA_VERSION
            assert r["session_id"] == rec.session_id
            assert "recorded_at" in r and "metric" in r and "paradigm" in r

    def test_restart_does_not_drop_trials_that_restart_at_one(self, tmp_path):
        first = LearningRecorder(tmp_path, fsync=False)
        first.record_trials([{"trial": 1, "metric": 0.1}, {"trial": 2, "metric": 0.2}])
        first.close()
        second = LearningRecorder(tmp_path, fsync=False)
        assert second.prior_trial_lines == 2
        assert second.record_trials([{"trial": 1, "metric": 0.9}]) == 1
        second.close()
        rows = read_jsonl(tmp_path / "trials.jsonl")
        assert [r["trial"] for r in rows] == [1, 2, 1]
        assert rows[0]["session_id"] != rows[2]["session_id"]

    def test_summary_record(self, tmp_path):
        rec = LearningRecorder(tmp_path, fsync=False)
        rec.record_summary({"step": 100, "paradigm": "open-arena", "mb_weights_mean": 0.5})
        rec.close()
        rows = read_jsonl(tmp_path / "telemetry_summary.jsonl")
        assert rows[0]["type"] == "telemetry_summary" and rows[0]["step"] == 100


# =============================================================================
# 3. RecorderThread
# =============================================================================

class _FakeRunner:
    def __init__(self):
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.total_steps = 0
        self.sim_speed = 10.0
        self.active_paradigm_id = "t-maze"
        self.current_trial = 1
        self.trial_history = []
        self.learning_curve = []
        self.latest_telemetry = {}

    def complete_trial(self, metric):
        with self.lock:
            self.total_steps += 100
            self.learning_curve.append(metric)
            self.trial_history.append({"trial": self.current_trial, "paradigm": self.active_paradigm_id,
                                       "step": self.total_steps, "metric": metric, "timestamp": time.time()})
            self.current_trial += 1
            self.latest_telemetry = {"fly": {"x": 1.0, "y": 2.0, "heading": 0.1, "speed": 3.0, "state": "WANDER"},
                                     "plasticity": {"mb_weights_mean": 0.5, "mb_weights_std": 0.05},
                                     "metrics": {"performance_index": metric}}


class TestRecorderThread:
    def test_poll_once_drains_trials_and_summary(self, tmp_path):
        runner = _FakeRunner()
        rec = LearningRecorder(tmp_path, fsync=False)
        thread = RecorderThread(runner, rec, poll_interval=0.05, summary_interval=999)
        runner.complete_trial(0.3)
        runner.complete_trial(0.4)
        out = thread.poll_once(force_summary=True)
        assert out == {"trials": 2, "summaries": 1}
        assert thread.poll_once() == {"trials": 0, "summaries": 0}
        thread.stop()  # flushes a final summary and closes files
        trials = read_jsonl(tmp_path / "trials.jsonl")
        summaries = read_jsonl(tmp_path / "telemetry_summary.jsonl")
        assert [t["trial"] for t in trials] == [1, 2]
        assert len(summaries) == 2
        s = summaries[0]
        assert s["paradigm"] == "t-maze" and s["trials_completed"] == 2 and s["step"] == 200
        assert s["fly"]["x"] == 1.0 and s["mb_weights_mean"] == 0.5
        assert s["learning_curve_tail"] == [0.3, 0.4]
        assert s["metrics"] == {"performance_index": 0.4}

    def test_background_thread_records_within_poll_interval(self, tmp_path):
        runner = _FakeRunner()
        rec = LearningRecorder(tmp_path, fsync=False)
        thread = RecorderThread(runner, rec, poll_interval=0.05, summary_interval=0.1)
        thread.start()
        try:
            runner.complete_trial(0.5)
            deadline = time.time() + 3
            while time.time() < deadline and rec.trials.lines_written < 1:
                time.sleep(0.02)
            assert rec.trials.lines_written == 1
            time.sleep(0.3)
            assert rec.summaries.lines_written >= 2
        finally:
            thread.stop()
        assert thread.errors == 0
        thread.stop()  # idempotent

    def test_summarise_runner_tolerates_empty_telemetry(self):
        runner = _FakeRunner()
        s = summarise_runner(runner)
        assert s["fly"] == {} and s["mb_weights_mean"] is None and s["learning_curve_tail"] == []

    def test_with_real_daemon_runner(self, tmp_path):
        """The poller works against the real runner without touching its loop."""
        from neurofly_daemon import ContinuousExperimentRunner

        runner = ContinuousExperimentRunner(initial_paradigm="open-arena", sim_speed=50.0,
                                            output_dir=tmp_path / "outputs")
        rec = LearningRecorder(tmp_path / "learning", fsync=False)
        thread = RecorderThread(runner, rec, poll_interval=0.05, summary_interval=999)
        # Step the arena a few times by hand and inject one milestone through the
        # runner's own public bookkeeping.
        for _ in range(3):
            res = runner.arena.step(0.02)
            runner.total_steps += 1
            runner.latest_telemetry = runner._assemble_telemetry(res)
        with runner.lock:
            runner.trial_history.append({"trial": runner.current_trial, "paradigm": runner.active_paradigm_id,
                                         "step": runner.total_steps, "metric": 0.5, "timestamp": time.time()})
            runner.current_trial += 1
        out = thread.poll_once(force_summary=True)
        thread.stop()
        assert out["trials"] == 1 and out["summaries"] == 1
        trial = read_jsonl(tmp_path / "learning" / "trials.jsonl")[0]
        assert trial["paradigm"] == "open-arena" and trial["step"] == 3
        summary = read_jsonl(tmp_path / "learning" / "telemetry_summary.jsonl")[0]
        assert summary["step"] == 3 and "x" in summary["fly"] and summary["mb_weights_mean"] is not None


# =============================================================================
# 4. Data-directory resolution
# =============================================================================

class TestResolveDataDir:
    def test_precedence(self, tmp_path):
        root = tmp_path / "proj"
        assert resolve_data_dir(root, None, environ={}) == (root / "outputs" / "learning").resolve()
        env_dir = tmp_path / "from_env"
        assert resolve_data_dir(root, None, environ={"NEUROFLY_DATA_DIR": str(env_dir)}) == env_dir.resolve()
        flag_dir = tmp_path / "from_flag"
        assert resolve_data_dir(root, str(flag_dir), environ={"NEUROFLY_DATA_DIR": str(env_dir)}) == flag_dir.resolve()
        assert resolve_data_dir(root, "", environ={"NEUROFLY_DATA_DIR": "  "}) == (root / "outputs" / "learning").resolve()
