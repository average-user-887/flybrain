import copy
from types import SimpleNamespace

import pytest

import maze
from learning_recorder import LearningRecorder
from observation_envelopes import build_observation_envelope
from observation_publication import (
    ObservationPayloadConflict,
    ObservationPublicationError,
    ObservationPublicationQueue,
    ObservationQueueFull,
)
from provenance import RunManifest


def terminal(producer_name="t_maze", *, run_id="run-1", instance_id="instance-1",
             segment_id="segment-1", brain_id="brain-1"):
    assay_ids = {"t_maze": "t-maze", "gap_crossing": "gap-crossing"}
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
    identity.update(run_id=run_id, daemon_run_id="daemon-1", activation=4,
                    brain_id=brain_id,
                    graph_io_extension={"version": None, "sha256": None})
    return build_observation_envelope(
        snapshot, identity=identity,
        provenance={
            "gf_source": "geometric", "stimulus_entry_stage": None,
            "motor_assists_enabled": True, "controller_states": ["SURGE", "CAST"],
        },
        segment_id=segment_id, segment_start_sim_s=100.0, validity="valid",
        terminal_pose_post_step={"x_mm": 30.0, "y_mm": 30.0, "heading_rad": 0.0})


def claim_and_record(queue, recorder):
    claim = queue.claim_oldest()
    receipt = recorder.record_observation(claim["observation"])
    return claim, receipt


def test_order_capacity_duplicate_conflict_and_mutation_isolation(tmp_path):
    queue = ObservationPublicationQueue(capacity=2)
    first = terminal()
    second = terminal("gap_crossing", run_id="run-2", instance_id="instance-2",
                      segment_id="segment-2", brain_id="brain-2")
    original = copy.deepcopy(first)
    result = queue.enqueue(first)
    first["identity"]["run_id"] = "caller-mutated"
    result["observation"]["identity"]["run_id"] = "return-mutated"
    assert queue.enqueue(original)["status"] == "pending"
    queue.enqueue(second)
    with pytest.raises(ObservationQueueFull):
        queue.enqueue(terminal(run_id="run-3", instance_id="instance-3", segment_id="segment-3"))
    assert queue.pending_status()["pending_count"] == 2

    changed = copy.deepcopy(original)
    changed["provenance"]["controller_states"].append("changed")
    with pytest.raises(ObservationPayloadConflict):
        queue.enqueue(changed)
    claim = queue.claim_oldest()
    assert claim["observation"] == original
    assert claim["observation_key"]["run_id"] == "run-1"
    assert queue.pending_status()["entries"][0]["phase"] == "claimed"


def test_failed_head_blocks_later_and_stale_receipts_cannot_publish(tmp_path):
    queue = ObservationPublicationQueue(capacity=3)
    first = terminal()
    second = terminal(run_id="run-2", instance_id="instance-2", segment_id="segment-2")
    queue.enqueue(first)
    queue.enqueue(second)
    old = queue.claim_oldest()
    queue.mark_failed(old["attempt_token"], {"phase": "file_fsync", "message": "ambiguous"})
    assert queue.claim_oldest() is None
    queue.retry_failed()
    retry = queue.claim_oldest()
    assert retry["attempt_token"] != old["attempt_token"]
    assert retry["observation"] == old["observation"]
    fake = {
        "durable": True, "idempotent": False,
        "observation_key": retry["observation_key"],
        "payload_sha256": retry["payload_sha256"],
        "file": "trials.jsonl", "line": 1, "offset": 0,
    }
    with pytest.raises(ObservationPublicationError, match="stale"):
        queue.acknowledge(old["attempt_token"], fake)
    assert queue.pending_status()["pending_count"] == 2


@pytest.mark.parametrize("change", ["key", "digest", "file", "line", "offset", "durable"])
def test_receipt_key_digest_location_and_durable_are_atomic(change):
    queue = ObservationPublicationQueue()
    queue.enqueue(terminal())
    claim = queue.claim_oldest()
    receipt = {
        "durable": True, "idempotent": False,
        "observation_key": claim["observation_key"],
        "payload_sha256": claim["payload_sha256"],
        "file": "trials.jsonl", "line": 1, "offset": 0,
    }
    if change == "key":
        receipt["observation_key"] = dict(receipt["observation_key"], run_id="wrong")
    elif change == "digest":
        receipt["payload_sha256"] = "0" * 64
    elif change == "file":
        receipt["file"] = "../trials.jsonl"
    elif change == "line":
        receipt["line"] = 0
    elif change == "offset":
        receipt["offset"] = -1
    else:
        receipt["durable"] = 1
    with pytest.raises(ObservationPublicationError):
        queue.acknowledge(claim["attempt_token"], receipt)
    assert queue.pending_status()["pending_count"] == 1
    assert queue.last_terminal(assay="t-maze", backend="modular",
                               instance_id="instance-1", brain_id="brain-1") is None


def test_real_recorder_fresh_and_ambiguous_ack_retry(tmp_path):
    queue = ObservationPublicationQueue()
    envelope = terminal()
    queue.enqueue(envelope)
    recorder = LearningRecorder(tmp_path, fsync=False)
    first_claim, lost_receipt = claim_and_record(queue, recorder)
    queue.mark_failed(first_claim["attempt_token"], {"phase": "directory_fsync", "ambiguous": True})
    queue.retry_failed()
    retry_claim, retry_receipt = claim_and_record(queue, recorder)
    assert retry_claim["observation"] == first_claim["observation"]
    assert retry_receipt["durable"] is True and retry_receipt["idempotent"] is True
    durable = queue.acknowledge(retry_claim["attempt_token"], retry_receipt)
    assert durable["receipt"] == retry_receipt
    assert lost_receipt["payload_sha256"] == retry_receipt["payload_sha256"]
    assert queue.pending_status()["pending_count"] == 0
    recorder.close()

    durable["observation"]["identity"]["run_id"] = "mutated-return"
    durable["receipt"]["file"] = "mutated-return"
    last = queue.last_terminal(assay="t-maze", backend="modular",
                               instance_id="instance-1", brain_id="brain-1")
    assert last["observation"] == envelope
    assert last["receipt"] == retry_receipt
    assert queue.enqueue(copy.deepcopy(envelope))["status"] == "durable"


def test_later_failure_does_not_change_prior_owner_terminal(tmp_path):
    queue = ObservationPublicationQueue(capacity=2)
    recorder = LearningRecorder(tmp_path, fsync=False)
    first = terminal()
    queue.enqueue(first)
    claim, receipt = claim_and_record(queue, recorder)
    queue.acknowledge(claim["attempt_token"], receipt)
    before = queue.last_terminal(assay="t-maze", backend="modular",
                                 instance_id="instance-1", brain_id="brain-1")

    queue.enqueue(terminal("gap_crossing", run_id="run-2", instance_id="instance-2",
                           segment_id="segment-2", brain_id="brain-2"))
    failed = queue.claim_oldest()
    queue.mark_failed(failed["attempt_token"], {"phase": "write", "message": "disk full"})
    assert queue.last_terminal(assay="t-maze", backend="modular",
                               instance_id="instance-1", brain_id="brain-1") == before
    assert queue.last_terminal(assay="gap-crossing", backend="modular",
                               instance_id="instance-2", brain_id="brain-2") is None
    recorder.close()


def test_coordinator_operations_perform_no_file_io(monkeypatch):
    envelope = terminal()
    queue = ObservationPublicationQueue()

    def refuse_io(*args, **kwargs):
        raise AssertionError("coordinator attempted file I/O")

    monkeypatch.setattr("builtins.open", refuse_io)
    queue.enqueue(envelope)
    claim = queue.claim_oldest()
    queue.mark_failed(claim["attempt_token"], {"phase": "write", "retryable": True})
    queue.retry_failed()
    claim = queue.claim_oldest()
    receipt = {
        "durable": True, "idempotent": False,
        "observation_key": claim["observation_key"],
        "payload_sha256": claim["payload_sha256"],
        "file": "trials.jsonl", "line": 1, "offset": 0,
    }
    queue.acknowledge(claim["attempt_token"], receipt)
    assert queue.pending_status()["pending_count"] == 0
