"""Tests for durable learning records (learning_recorder.py).

Covers:
1. JsonlWriter append-only semantics, durability flags and timestamp rotation
   that never deletes data.
2. LearningRecorder session manifest, trial de-duplication and summaries.
3. RecorderThread draining a runner-like object under its lock, including the
   real ContinuousExperimentRunner from neurofly_daemon.
4. Data-directory resolution precedence (flag > env > default).
"""

import copy
import json
import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from learning_recorder import (  # noqa: E402
    SCHEMA_VERSION,
    DurabilityUncertainError,
    JsonlWriter,
    LearningRecorder,
    ObservationConflictError,
    ObservationJournalError,
    RecorderOwnershipError,
    RecorderThread,
    read_jsonl,
    resolve_data_dir,
    summarise_runner,
)


def _terminal_envelope(*, producer_name="t_maze", daemon_run_id="daemon-1",
                       run_id=None, instance_id="instance-1", segment_id="segment-1"):
    import maze
    import online_metrics as om
    from observation_envelopes import build_observation_envelope
    from provenance import RunManifest

    assay_ids = {
        "t_maze": "t-maze", "gap_crossing": "gap-crossing",
    }
    producer = maze.ExperimentRegistry.get(producer_name)
    fly = SimpleNamespace(
        pos=SimpleNamespace(x=30.0, y=30.0), heading=0.0, speed=1.0,
        angular_velocity=0.0, assay_escape_remaining=0.0, behavioral_state="SURGE")
    producer.step(fly, 0.02)
    snapshot = producer.freeze_observation("manual_reset")
    manifest = RunManifest.create(
        backend="modular", assay=assay_ids[producer_name], instance_id=instance_id,
        seed=3, graph=None, dynamics={}, learned_parameter_locations={},
        source={"commit": "fixture"})
    identity = manifest.identity()
    if run_id is not None:
        identity["run_id"] = run_id
    identity.update(
        daemon_run_id=daemon_run_id, activation=4, brain_id="brain-1",
        graph_io_extension={"version": None, "sha256": None})
    return build_observation_envelope(
        snapshot, identity=identity,
        provenance={
            "gf_source": "geometric", "stimulus_entry_stage": None,
            "motor_assists_enabled": True, "controller_states": ["SURGE", "CAST"],
            "controller_specific": {"raw_motor_command": {"yaw": 0.0}},
        },
        segment_id=segment_id, segment_start_sim_s=100.0, validity="valid",
        terminal_pose_post_step={"x_mm": 30.0, "y_mm": 30.0, "heading_rad": 0.0})


def _next_presentation(envelope):
    value = copy.deepcopy(envelope)
    value["presentation_index"] += 1
    value["presentation_id"] = f"{value['segment_id']}:{value['presentation_index']}"
    return value


def _trial_bytes(data_dir):
    return {path.name: path.read_bytes() for path in sorted(Path(data_dir).glob("trials.jsonl*"))}


def _observation_rows(data_dir):
    rows = []
    for path in sorted(Path(data_dir).glob("trials.jsonl*")):
        rows.extend(row for row in read_jsonl(path) if row.get("type") == "observation")
    return rows


def _json_leaves(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _json_leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from _json_leaves(item)
    else:
        yield value


def _owner_process(data_dir, connection, start_event):
    """Independent-process ownership probe (must stay module-level for spawn)."""
    start_event.wait(10)
    try:
        recorder = LearningRecorder(Path(data_dir), fsync=False)
    except Exception as exc:
        connection.send(("error", type(exc).__name__, str(exc)))
        connection.close()
        return
    connection.send(("owned", os.getpid()))
    command = connection.recv()
    if command == "close":
        recorder.close()
    elif command == "crash":
        os._exit(17)
    connection.close()


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


class TestDurableObservationJournal:
    def test_fresh_retry_restart_and_actual_stored_payload(self, tmp_path):
        envelope = _terminal_envelope()
        (tmp_path / "trials.jsonl").write_text('{"legacy":true}\n', encoding="utf-8")
        rec = LearningRecorder(tmp_path, fsync=False)
        fresh = rec.record_observation(envelope)
        retry = rec.record_observation(copy.deepcopy(envelope))
        assert fresh["durable"] is True and fresh["idempotent"] is False
        assert retry["durable"] is True and retry["idempotent"] is True
        assert {key: fresh[key] for key in ("observation_key", "payload_sha256", "file", "line", "offset")} == {
            key: retry[key] for key in ("observation_key", "payload_sha256", "file", "line", "offset")}
        rows = _observation_rows(tmp_path)
        assert len(rows) == 1 and rows[0]["observation"] == envelope
        assert rows[0]["observation_key"] == fresh["observation_key"]
        leaves = list(_json_leaves(rows[0]["observation"]))
        assert None in leaves and False in leaves
        assert any(not isinstance(value, bool) and isinstance(value, (int, float)) and value == 0
                   for value in leaves)
        assert read_jsonl(tmp_path / "trials.jsonl")[0] == {"legacy": True}
        rec.close()

        restarted = LearningRecorder(tmp_path, fsync=False)
        receipt = restarted.record_observation(envelope)
        restarted.close()
        assert receipt["idempotent"] is True
        assert receipt["payload_sha256"] == fresh["payload_sha256"]
        assert len(_observation_rows(tmp_path)) == 1


    def test_two_presentations_and_full_identity_are_not_trial_deduplicated(self, tmp_path):
        rec = LearningRecorder(tmp_path, fsync=False)
        first = _terminal_envelope(run_id="run-a")
        second = _next_presentation(first)
        rec.record_trials([{"trial": 1, "metric": 0.5}])
        rec.record_observation(first)
        rec.record_observation(second)
        rec.record_observation(_terminal_envelope(daemon_run_id="daemon-2", run_id="run-a"))
        rec.record_observation(_terminal_envelope(run_id="run-b"))
        rec.record_observation(_terminal_envelope(run_id="run-a", instance_id="instance-2"))
        rec.record_observation(_terminal_envelope(
            producer_name="gap_crossing", daemon_run_id="daemon-gap",
            run_id="run-gap", instance_id="instance-gap", segment_id="segment-gap"))
        rec.close()
        rows = _observation_rows(tmp_path)
        assert len(rows) == 6
        assert {row["observation_key"]["presentation_id"] for row in rows[:2]} == {
            "segment-1:0", "segment-1:1"}
        assert read_jsonl(tmp_path / "trials.jsonl")[0]["type"] == "trial"

    def test_changed_payload_conflicts_and_wrapper_live_nonfinite_do_not_mutate(self, tmp_path):
        rec = LearningRecorder(tmp_path, fsync=False)
        envelope = _terminal_envelope(run_id="run-conflict")
        rec.record_observation(envelope)
        before = _trial_bytes(tmp_path)
        changed = copy.deepcopy(envelope)
        changed["provenance"]["controller_specific"]["raw_motor_command"]["yaw"] = 1.0
        with pytest.raises(ObservationConflictError, match="different observation payload"):
            rec.record_observation(changed)

        live = copy.deepcopy(envelope)
        for record in live["records"].values():
            record["final"] = False
        live["completeness"] = None
        live["end_reason"] = None
        live["measurement_end_rel_s"] = None
        live["measurement_end_sim_s"] = None
        live["terminal_pose_post_step"] = None
        with pytest.raises(ObservationJournalError, match="frozen terminal"):
            rec.record_observation(live)
        with pytest.raises(ObservationJournalError, match="last_terminal wrapper"):
            rec.record_observation({"last_terminal": envelope})
        nonfinite = copy.deepcopy(envelope)
        nonfinite["provenance"]["controller_specific"]["bad"] = float("nan")
        with pytest.raises(ObservationJournalError, match="finite"):
            rec.record_observation(nonfinite)
        empty_key = copy.deepcopy(envelope)
        empty_key["identity"]["daemon_run_id"] = ""
        with pytest.raises(ObservationJournalError, match="daemon_run_id|identity"):
            rec.record_observation(empty_key)
        assert _trial_bytes(tmp_path) == before
        rec.close()

    def test_write_flush_and_file_fsync_failures_retry_without_duplicates(self, tmp_path, monkeypatch):
        scenarios = ("write", "flush", "fsync")
        for scenario in scenarios:
            data_dir = tmp_path / scenario
            rec = LearningRecorder(data_dir, fsync=False)
            envelope = _terminal_envelope(run_id=f"run-{scenario}")
            attr = {"write": "_write_strict", "flush": "_flush_strict",
                    "fsync": "_fsync_strict"}[scenario]
            original = getattr(rec.trials, attr)

            def fail(*args, **kwargs):
                raise OSError(f"injected {scenario} failure")

            monkeypatch.setattr(rec.trials, attr, fail)
            with pytest.raises(OSError, match=f"injected {scenario}"):
                rec.record_observation(envelope)
            monkeypatch.setattr(rec.trials, attr, original)
            if scenario == "fsync":
                # A failed file fsync may have dropped the dirty pages while the
                # cache still shows the line; the same process never acknowledges it.
                with pytest.raises(DurabilityUncertainError, match="uncertain"):
                    rec.record_observation(envelope)
                assert len(_observation_rows(data_dir)) == 1
                rec.close()
                rec = LearningRecorder(data_dir, fsync=False)
            receipt = rec.record_observation(envelope)
            assert receipt["durable"] is True
            assert len(_observation_rows(data_dir)) == 1
            rec.close()

    def test_complete_line_with_lost_fsync_ack_recovers_after_restart(self, tmp_path, monkeypatch):
        envelope = _terminal_envelope(run_id="run-restart-ambiguous")
        rec = LearningRecorder(tmp_path, fsync=False)

        def fail_fsync():
            raise OSError("lost file fsync acknowledgement")

        monkeypatch.setattr(rec.trials, "_fsync_strict", fail_fsync)
        with pytest.raises(OSError, match="lost file fsync acknowledgement"):
            rec.record_observation(envelope)
        assert len(_observation_rows(tmp_path)) == 1
        rec.close()

        restarted = LearningRecorder(tmp_path, fsync=False)
        receipt = restarted.record_observation(envelope)
        restarted.close()
        assert receipt["durable"] is True and receipt["idempotent"] is True
        assert len(_observation_rows(tmp_path)) == 1

    def test_persistent_directory_failure_refuses_then_rotation_recovers(self, tmp_path, monkeypatch):
        creation_dir = tmp_path / "creation"
        rec = LearningRecorder(creation_dir, fsync=False)
        envelope = _terminal_envelope(run_id="run-dir")
        original = rec.trials._fsync_parent

        def fail_dir():
            raise OSError("injected directory fsync failure")

        monkeypatch.setattr(rec.trials, "_fsync_parent", fail_dir)
        for _ in range(2):
            with pytest.raises(OSError, match="directory fsync"):
                rec.record_observation(envelope)
        assert _observation_rows(creation_dir) == []
        monkeypatch.setattr(rec.trials, "_fsync_parent", original)
        assert rec.record_observation(envelope)["durable"] is True
        rec.close()

        rotation_dir = tmp_path / "rotation"
        rotating = LearningRecorder(rotation_dir, max_bytes=128, fsync=False)
        first = _terminal_envelope(run_id="run-rotate")
        second = _next_presentation(first)
        rotating.record_observation(first)
        original = rotating.trials._fsync_parent
        calls = 0

        def fail_after_rename():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected rotation directory fsync failure")
            original()

        monkeypatch.setattr(rotating.trials, "_fsync_parent", fail_after_rename)
        with pytest.raises(OSError, match="rotation directory fsync"):
            rotating.record_observation(second)
        monkeypatch.setattr(rotating.trials, "_fsync_parent", original)
        assert rotating.record_observation(second)["durable"] is True
        assert len(rotating.trials.rotated_files()) >= 1
        assert len(_observation_rows(rotation_dir)) == 2
        assert rotating.record_observation(first)["idempotent"] is True
        rotating.close()

    @pytest.mark.parametrize("tail", [b'{"type":"trial"}', b'{broken\n'])
    def test_partial_or_malformed_tail_refuses_with_location_and_preserves_bytes(
            self, tmp_path, tail):
        rec = LearningRecorder(tmp_path, fsync=False)
        path = tmp_path / "trials.jsonl"
        path.write_bytes(tail)
        before = path.read_bytes()
        match = "partial JSONL row" if not tail.endswith(b"\n") else "invalid JSON"
        with pytest.raises(ObservationJournalError, match=match) as caught:
            rec.record_observation(_terminal_envelope(run_id="run-tail"))
        assert "trials.jsonl:line 1:byte 0" in str(caught.value)
        assert path.read_bytes() == before
        rec.close()

    def test_concurrent_same_recorder_calls_share_one_transaction(self, tmp_path):
        rec = LearningRecorder(tmp_path, fsync=False)
        envelope = _terminal_envelope(run_id="run-concurrent")
        barrier = threading.Barrier(3)
        receipts = []
        errors = []

        def call():
            barrier.wait()
            try:
                receipts.append(rec.record_observation(envelope))
            except Exception as exc:  # pragma: no cover - asserted empty below
                errors.append(exc)

        threads = [threading.Thread(target=call) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=5)
        rec.close()
        assert not errors and len(receipts) == 2
        assert sorted(receipt["idempotent"] for receipt in receipts) == [False, True]
        assert len(_observation_rows(tmp_path)) == 1


class TestRecorderDirectoryOwnership:
    @pytest.mark.parametrize("channel", ["observation", "trials", "summary"])
    def test_close_waits_for_started_operation_before_releasing_directory(
            self, tmp_path, monkeypatch, channel):
        recorder = LearningRecorder(tmp_path)
        entered = threading.Event()
        proceed = threading.Event()
        closing = threading.Event()
        closed = threading.Event()
        errors = []
        results = []
        envelope = _terminal_envelope(run_id="close-race")
        if channel == "observation":
            target, attribute = recorder, "_prepare_observation"
            write = lambda: recorder.record_observation(envelope)
        elif channel == "trials":
            target, attribute = recorder.trials, "append"
            write = lambda: recorder.record_trials([{"trial": 1}])
        else:
            target, attribute = recorder.summaries, "append"
            write = lambda: recorder.record_summary({"step": 1})
        original = getattr(target, attribute)

        def blocked(*args, **kwargs):
            entered.set()
            if not proceed.wait(5):
                raise RuntimeError("close-race probe timed out")
            return original(*args, **kwargs)

        monkeypatch.setattr(target, attribute, blocked)

        def write_worker():
            try:
                results.append(write())
            except BaseException as exc:
                errors.append(exc)

        def close_worker():
            closing.set()
            try:
                recorder.close()
            except BaseException as exc:
                errors.append(exc)
            finally:
                closed.set()

        writer = threading.Thread(target=write_worker)
        closer = threading.Thread(target=close_worker)
        writer.start()
        try:
            assert entered.wait(2)
            closer.start()
            assert closing.wait(2)
            assert not closed.wait(0.05)
            with pytest.raises(RecorderOwnershipError, match="already owned"):
                LearningRecorder(tmp_path)
        finally:
            proceed.set()
            writer.join(5)
            if closer.ident is not None:
                closer.join(5)
        assert not writer.is_alive() and not closer.is_alive()
        assert not errors and len(results) == 1 and closed.is_set()
        with LearningRecorder(tmp_path) as replacement:
            before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
            with pytest.raises(RecorderOwnershipError, match="closed"):
                write()
            assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before
            if channel == "observation":
                assert replacement.record_observation(envelope)["idempotent"] is True
                assert len(_observation_rows(tmp_path)) == 1

    def test_second_object_and_alias_refuse_without_mutating_existing_bytes(self, tmp_path):
        data_dir = tmp_path / "owned"
        owner = LearningRecorder(data_dir, fsync=False)
        before = _trial_bytes(data_dir) | {
            "session.json": (data_dir / "session.json").read_bytes(),
            "sessions.jsonl": (data_dir / "sessions.jsonl").read_bytes(),
        }
        with pytest.raises(RecorderOwnershipError, match="already owned"):
            LearningRecorder(data_dir, fsync=False)
        alias = tmp_path / "alias"
        alias.symlink_to(data_dir, target_is_directory=True)
        with pytest.raises(RecorderOwnershipError, match="already owned"):
            LearningRecorder(alias, fsync=False)
        after = _trial_bytes(data_dir) | {
            "session.json": (data_dir / "session.json").read_bytes(),
            "sessions.jsonl": (data_dir / "sessions.jsonl").read_bytes(),
        }
        assert after == before
        owner.close()

    def test_different_directories_close_restart_and_use_after_close(self, tmp_path):
        first = LearningRecorder(tmp_path / "first", fsync=False)
        other = LearningRecorder(tmp_path / "other", fsync=False)
        envelope = _terminal_envelope(run_id="owned-restart")
        receipt = first.record_observation(envelope)
        lock_path = tmp_path / "first" / ".neurofly-recorder.lock"
        lock_inode = lock_path.stat().st_ino
        first.close()
        first.close()
        assert lock_path.is_file() and lock_path.stat().st_ino == lock_inode
        with pytest.raises(RecorderOwnershipError, match="closed"):
            first.record_observation(envelope)
        restarted = LearningRecorder(tmp_path / "first", fsync=False)
        assert lock_path.stat().st_ino == lock_inode
        assert restarted.record_observation(envelope)["idempotent"] is True
        assert restarted.record_observation(envelope)["payload_sha256"] == receipt["payload_sha256"]
        restarted.close()
        other.close()

    def test_initialisation_failure_releases_owner(self, tmp_path, monkeypatch):
        original = LearningRecorder._write_session

        def fail(_self):
            raise OSError("injected session initialisation failure")

        monkeypatch.setattr(LearningRecorder, "_write_session", fail)
        with pytest.raises(OSError, match="initialisation failure"):
            LearningRecorder(tmp_path, fsync=False)
        monkeypatch.setattr(LearningRecorder, "_write_session", original)
        recorder = LearningRecorder(tmp_path, fsync=False)
        recorder.close()

    def test_inherited_object_is_rejected(self, tmp_path, monkeypatch):
        recorder = LearningRecorder(tmp_path, fsync=False)
        owner_pid = os.getpid()
        monkeypatch.setattr("learning_recorder.os.getpid", lambda: owner_pid + 1)
        with pytest.raises(RecorderOwnershipError, match="cannot be used"):
            recorder.record_summary({"step": 1})
        with pytest.raises(RecorderOwnershipError, match="cannot be used"):
            recorder.close()
        monkeypatch.undo()
        recorder.close()

    @pytest.mark.skipif(os.name not in ("posix", "nt"), reason="no supported lock backend")
    def test_two_actual_processes_race_and_abrupt_exit_releases_lock(self, tmp_path):
        ctx = multiprocessing.get_context("spawn")
        start = ctx.Event()
        parent_a, child_a = ctx.Pipe()
        parent_b, child_b = ctx.Pipe()
        processes = [
            ctx.Process(target=_owner_process, args=(str(tmp_path), child_a, start)),
            ctx.Process(target=_owner_process, args=(str(tmp_path), child_b, start)),
        ]
        for process in processes:
            process.start()
        start.set()
        replies = [parent_a.recv(), parent_b.recv()]
        assert sorted(reply[0] for reply in replies) == ["error", "owned"]
        assert next(reply for reply in replies if reply[0] == "error")[1] == "RecorderOwnershipError"
        assert len(read_jsonl(tmp_path / "sessions.jsonl")) == 1
        owned_index = next(i for i, reply in enumerate(replies) if reply[0] == "owned")
        (parent_a, parent_b)[owned_index].send("crash")
        for process in processes:
            process.join(10)
            assert not process.is_alive()
        assert processes[owned_index].exitcode == 17
        restarted = LearningRecorder(tmp_path, fsync=False)
        restarted.close()


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
