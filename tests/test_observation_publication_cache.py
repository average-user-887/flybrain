"""Bounded recent-durability cache: misses re-enter recorder validation."""

import copy
import gc
import json
import sqlite3
import tracemalloc

import pytest

from learning_recorder import LearningRecorder, ObservationConflictError, ObservationJournalError
from observation_publication import (
    DEFAULT_DURABLE_CACHE_CAPACITY,
    ObservationPayloadConflict,
    ObservationPublicationQueue,
    ObservationQueueFull,
)
from tests.test_observation_publication import terminal
from tests.test_observation_publication_bridge import attached, terminal_for


def observation_rows(data_dir):
    rows = []
    for path in sorted(data_dir.glob("trials.jsonl*")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("type") == "observation":
                rows.append(row)
    return rows


def publish(queue, recorder, envelope):
    """Enqueue, write and acknowledge one envelope; return enqueue status and receipt."""
    status = queue.enqueue(envelope)["status"]
    claim = queue.claim_oldest()
    receipt = recorder.record_observation(claim["observation"])
    queue.acknowledge(claim["attempt_token"], receipt)
    return status, receipt


def segment(n):
    return terminal(segment_id=f"segment-{n}")


def changed(envelope):
    value = copy.deepcopy(envelope)
    value["provenance"]["controller_states"].append("changed")
    return value


def synthetic_receipt(claim):
    return {
        "durable": True, "idempotent": False,
        "observation_key": claim["observation_key"],
        "payload_sha256": claim["payload_sha256"],
        "file": "trials.jsonl", "line": 1, "offset": 0,
    }


def test_capacity_is_validated_and_default_is_bounded():
    assert ObservationPublicationQueue().durable_cache_capacity == DEFAULT_DURABLE_CACHE_CAPACITY
    for bad in (0, -1, True, 1.5, None):
        with pytest.raises(ValueError, match="durable_cache_capacity"):
            ObservationPublicationQueue(durable_cache_capacity=bad)


def test_old_exact_retry_revalidates_without_a_new_ledger_row(tmp_path):
    queue = ObservationPublicationQueue(durable_cache_capacity=2)
    recorder = LearningRecorder(tmp_path, fsync=False)
    old = segment(0)
    _, original = publish(queue, recorder, old)
    publish(queue, recorder, segment(1))
    publish(queue, recorder, segment(2))  # evicts segment-0
    assert len(queue._durable_digests) == 2
    rows = observation_rows(tmp_path)
    assert len(rows) == 3

    # Visible shift: an evicted key is no longer "durable" at once; it is pending
    # until the recorder confirms the existing row.
    retry = queue.enqueue(copy.deepcopy(old))
    assert retry["status"] == "enqueued"
    assert queue.pending_status()["pending_count"] == 1
    claim = queue.claim_oldest()
    receipt = recorder.record_observation(claim["observation"])
    assert receipt["idempotent"] is True
    assert {k: receipt[k] for k in ("file", "line", "offset", "payload_sha256")} == \
        {k: original[k] for k in ("file", "line", "offset", "payload_sha256")}
    durable = queue.acknowledge(claim["attempt_token"], receipt)
    assert durable["status"] == "durable"
    assert observation_rows(tmp_path) == rows
    # Re-confirmed, so cached again and answered immediately.
    assert queue.enqueue(copy.deepcopy(old))["status"] == "durable"
    assert len(queue._durable_digests) == 2
    recorder.close()


def test_old_conflicting_retry_fails_before_any_durable_ack(tmp_path):
    queue = ObservationPublicationQueue(durable_cache_capacity=1)
    recorder = LearningRecorder(tmp_path, fsync=False)
    old = segment(0)
    publish(queue, recorder, old)
    recent = segment(1)
    publish(queue, recorder, recent)  # evicts segment-0
    before_rows = observation_rows(tmp_path)
    before_terminal = queue.last_terminal(assay="t-maze", backend="modular",
                                          instance_id="instance-1", brain_id="brain-1")

    # A cache hit with a changed payload is still refused at enqueue.
    with pytest.raises(ObservationPayloadConflict):
        queue.enqueue(changed(recent))

    # An evicted key with a changed payload is refused by the recorder instead.
    assert queue.enqueue(changed(old))["status"] == "enqueued"
    claim = queue.claim_oldest()
    with pytest.raises(ObservationConflictError):
        recorder.record_observation(claim["observation"])
    failed = queue.mark_failed(claim["attempt_token"], {"error_type": "ObservationConflictError"})
    assert failed["phase"] == "failed"
    # Nothing durable was produced, nothing was written, the old evidence stands.
    assert observation_rows(tmp_path) == before_rows
    assert queue.last_terminal(assay="t-maze", backend="modular",
                               instance_id="instance-1", brain_id="brain-1") == before_terminal
    assert claim["observation_key"] not in [dict(zip(
        ("daemon_run_id", "run_id", "instance_id", "segment_id", "presentation_id"), key))
        for key in queue._durable_digests]
    # A retry of the conflicting payload fails the same way; it never becomes durable.
    queue.retry_failed()
    claim = queue.claim_oldest()
    with pytest.raises(ObservationConflictError):
        recorder.record_observation(claim["observation"])
    queue.mark_failed(claim["attempt_token"], {"error_type": "ObservationConflictError"})
    assert queue.pending_status()["entries"][0]["phase"] == "failed"
    assert observation_rows(tmp_path) == before_rows
    recorder.close()


def test_old_conflicting_retry_through_runner_halts_without_publishing(tmp_path):
    runner, recorder, thread = attached(tmp_path)
    with runner.lock:
        runner.observation_publication = ObservationPublicationQueue(durable_cache_capacity=1)
    old = terminal_for(runner, segment_id="segment-old")
    for envelope in (old, terminal_for(runner, segment_id="segment-new")):
        with runner.lock:
            runner.enqueue_terminal_observation(envelope)
        assert thread.poll_once()["observations"] == 1
    rows = observation_rows(tmp_path / "records")
    with runner.lock:
        before = runner.observation_publication_status()["last_terminal"]
        assert runner.enqueue_terminal_observation(changed(old))["status"] == "enqueued"
    with pytest.raises(ObservationConflictError):
        thread.poll_once()
    with runner.lock:
        status = runner.observation_publication_status()
    head = status["pending"]["entries"][0]
    assert head["phase"] == "failed"
    assert head["failure"]["error_type"] == "ObservationConflictError"
    assert status["last_terminal"] == before
    assert runner.last_error
    assert observation_rows(tmp_path / "records") == rows
    assert thread.stop() is False


def test_recorder_failure_during_revalidation_keeps_the_retry_pending(tmp_path, monkeypatch):
    queue = ObservationPublicationQueue(durable_cache_capacity=1)
    recorder = LearningRecorder(tmp_path, fsync=False)
    old = segment(0)
    publish(queue, recorder, old)
    publish(queue, recorder, segment(1))
    rows = observation_rows(tmp_path)

    assert queue.enqueue(copy.deepcopy(old))["status"] == "enqueued"
    claim = queue.claim_oldest()
    original = recorder._refresh_index_locked

    def broken_index():
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(recorder, "_refresh_index_locked", broken_index)
    with pytest.raises(ObservationJournalError, match="nothing was acknowledged"):
        recorder.record_observation(claim["observation"])
    queue.mark_failed(claim["attempt_token"], {"error_type": "ObservationJournalError"})
    status = queue.pending_status()
    assert status["pending_count"] == 1 and status["entries"][0]["phase"] == "failed"
    assert len(queue._durable_digests) == 1  # the miss was not cached as durable

    monkeypatch.setattr(recorder, "_refresh_index_locked", original)
    queue.retry_failed()
    claim = queue.claim_oldest()
    receipt = recorder.record_observation(claim["observation"])
    assert receipt["idempotent"] is True
    queue.acknowledge(claim["attempt_token"], receipt)
    assert queue.pending_status()["pending_count"] == 0
    assert observation_rows(tmp_path) == rows
    recorder.close()


def test_queue_capacity_old_retry_needs_a_slot_and_pending_is_never_evicted(tmp_path):
    queue = ObservationPublicationQueue(capacity=2, durable_cache_capacity=1)
    recorder = LearningRecorder(tmp_path, fsync=False)
    old, recent = segment(0), segment(1)
    publish(queue, recorder, old)
    publish(queue, recorder, recent)
    waiting = [segment(2), segment(3)]
    for envelope in waiting:
        assert queue.enqueue(envelope)["status"] == "enqueued"

    # A cached key needs no slot; an evicted one does, and is refused, not dropped.
    assert queue.enqueue(copy.deepcopy(recent))["status"] == "durable"
    with pytest.raises(ObservationQueueFull):
        queue.enqueue(copy.deepcopy(old))
    status = queue.pending_status()
    assert status["pending_count"] == 2
    assert [e["observation_key"]["segment_id"] for e in status["entries"]] == \
        ["segment-2", "segment-3"]

    # Acknowledging one pending entry evicts only cached durable keys; the
    # other pending entry survives every eviction.
    claim = queue.claim_oldest()
    queue.acknowledge(claim["attempt_token"], recorder.record_observation(claim["observation"]))
    assert len(queue._durable_digests) == 1
    status = queue.pending_status()
    assert [e["observation_key"]["segment_id"] for e in status["entries"]] == ["segment-3"]
    assert queue.enqueue(copy.deepcopy(old))["status"] == "enqueued"
    recorder.close()


def test_cache_size_never_exceeds_cap_and_memory_stays_flat(monkeypatch):
    cap, n = 8, 3000
    queue = ObservationPublicationQueue(durable_cache_capacity=cap, terminal_capacity=4)
    base = terminal()

    def ack(i):
        envelope = copy.deepcopy(base)
        envelope["identity"]["run_id"] = f"{i:032x}"
        queue.enqueue(envelope)
        claim = queue.claim_oldest()
        queue.acknowledge(claim["attempt_token"], synthetic_receipt(claim))

    def refuse_io(*args, **kwargs):
        raise AssertionError("coordinator attempted file I/O")

    for i in range(cap * 4):  # fill the cache and warm every path
        ack(i)
    monkeypatch.setattr("builtins.open", refuse_io)
    gc.collect()
    tracemalloc.start()
    try:
        start = tracemalloc.get_traced_memory()[0]
        for i in range(cap * 4, cap * 4 + n):
            ack(i)
            assert len(queue._durable_digests) <= cap
        gc.collect()  # count only what the queue keeps, not uncollected cycles
        growth = tracemalloc.get_traced_memory()[0] - start
    finally:
        tracemalloc.stop()
    assert len(queue._durable_digests) == cap
    last = cap * 4 + n - 1
    assert [key[1] for key in queue._durable_digests] == \
        [f"{i:032x}" for i in range(last - cap + 1, last + 1)]
    # The unbounded map retained about 290 bytes per acknowledgement (about
    # 870 kB here). Allow a small constant for allocator noise, not growth.
    assert growth < 64 * 1024, growth
