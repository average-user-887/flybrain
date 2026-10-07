"""Retained memory of ObservationPublicationQueue per acknowledged observation.

Not collected by pytest (no ``test_`` prefix). Deterministic and CPU only: one
real terminal envelope is built once, then N copies that differ only in their
``run_id`` (a 32-character hex string like a real one) are enqueued, claimed and
acknowledged with a well-formed synthetic receipt. No recorder and no file I/O
are involved, so the measurement is the queue's own retention.

    python tests/bench_publication_digest_memory.py --counts 1000 10000 50000 \\
        [--impl-dir DIR]

``--impl-dir`` puts a directory holding another ``observation_publication.py``
first on ``sys.path`` so an older implementation is measured on identical data.
Prints one JSON object per count. ``retained_bytes`` is the tracemalloc traced
size after N acknowledgements minus the size before the first one (after a full
garbage collection), so it is what the queue keeps, not peak working memory.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _base_envelope():
    from types import SimpleNamespace

    import maze
    from observation_envelopes import build_observation_envelope
    from provenance import RunManifest

    producer = maze.ExperimentRegistry.get("t_maze")
    fly = SimpleNamespace(
        pos=SimpleNamespace(x=30.0, y=30.0), heading=0.0, speed=1.0,
        angular_velocity=0.0, assay_escape_remaining=0.0, behavioral_state="SURGE")
    producer.step(fly, 0.02)
    snapshot = producer.freeze_observation("manual_reset")
    manifest = RunManifest.create(
        backend="modular", assay="t-maze", instance_id="instance-1", seed=3, graph=None,
        dynamics={}, learned_parameter_locations={}, source={"commit": "fixture"})
    identity = manifest.identity()
    identity.update(daemon_run_id="daemon-1", activation=4, brain_id="brain-1",
                    graph_io_extension={"version": None, "sha256": None})
    return build_observation_envelope(
        snapshot, identity=identity,
        provenance={
            "gf_source": "geometric", "stimulus_entry_stage": None,
            "motor_assists_enabled": True, "controller_states": ["SURGE", "CAST"],
        },
        segment_id="segment-1", segment_start_sim_s=100.0, validity="valid",
        terminal_pose_post_step={"x_mm": 30.0, "y_mm": 30.0, "heading_rad": 0.0})


def acknowledge_one(queue, envelope):
    queue.enqueue(envelope)
    claim = queue.claim_oldest()
    receipt = {
        "durable": True, "idempotent": False,
        "observation_key": claim["observation_key"],
        "payload_sha256": claim["payload_sha256"],
        "file": "trials.jsonl", "line": 1, "offset": 0,
    }
    queue.acknowledge(claim["attempt_token"], receipt)


def measure(count: int, base) -> dict:
    from observation_publication import ObservationPublicationQueue

    def envelope(i):
        value = copy.deepcopy(base)
        value["identity"]["run_id"] = f"{i:032x}"
        return value

    queue = ObservationPublicationQueue()
    acknowledge_one(queue, envelope(0))  # warm every code path and the owner slot
    gc.collect()
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    for i in range(1, count):
        # Built inside the window so its allocation and release both net out.
        acknowledge_one(queue, envelope(i))
    gc.collect()
    after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    acknowledged = count - 1
    digests = getattr(queue, "_durable_digests")
    return {
        "impl": str(Path(sys.modules["observation_publication"].__file__).resolve()),
        "acknowledged": acknowledged,
        "retained_bytes": after - before,
        "retained_bytes_per_ack": round((after - before) / acknowledged, 1),
        "durable_digest_entries": len(digests),
        "durable_cache_capacity": getattr(queue, "durable_cache_capacity", None),
        "last_terminal_entries": len(queue._last_terminal),
        "pending": queue.pending_status()["pending_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", type=int, nargs="+", default=[1000, 10000, 50000])
    parser.add_argument("--impl-dir", type=Path)
    args = parser.parse_args()
    if args.impl_dir is not None:
        sys.path.insert(0, str(args.impl_dir.resolve()))
    sys.path.insert(1 if args.impl_dir is not None else 0, str(ROOT))
    base = _base_envelope()
    for count in args.counts:
        print(json.dumps(measure(count, base), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
