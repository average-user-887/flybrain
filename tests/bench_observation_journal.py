"""Long-history benchmark for LearningRecorder.record_observation (not collected by pytest).

Builds a retained trials ledger of a given size from real, fully valid observation
rows (plus one legacy trial row per observation), split across rotated files like a
long run, then opens a recorder on it and times fresh durable observation writes.

    python tests/bench_observation_journal.py --work outputs/bench-journal \\
        --sizes-mb 1 4 16 64 128 --writes 5 [--impl-dir DIR]

``--impl-dir`` puts a directory holding another ``learning_recorder.py`` first on
``sys.path`` so an older implementation can be measured on identical data.
Prints one JSON object per history size.
"""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _terminal_envelope(run_id: str):
    """Same fixture as tests/test_learning_recorder.py, without importing the recorder."""
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
    identity["run_id"] = run_id
    identity.update(daemon_run_id="daemon-1", activation=4, brain_id="brain-1",
                    graph_io_extension={"version": None, "sha256": None})
    return build_observation_envelope(
        snapshot, identity=identity,
        provenance={
            "gf_source": "geometric", "stimulus_entry_stage": None,
            "motor_assists_enabled": True, "controller_states": ["SURGE", "CAST"],
            "controller_specific": {"raw_motor_command": {"yaw": 0.0}},
        },
        segment_id="segment-1", segment_start_sim_s=100.0, validity="valid",
        terminal_pose_post_step={"x_mm": 30.0, "y_mm": 30.0, "heading_rad": 0.0})


def _template():
    token = "BENCHRUNTOKEN"
    envelope = _terminal_envelope(token)
    key = {"daemon_run_id": envelope["identity"]["daemon_run_id"], "run_id": token,
           "instance_id": envelope["identity"]["instance_id"],
           "segment_id": envelope["segment_id"], "presentation_id": envelope["presentation_id"]}
    row = {"type": "observation", "schema_version": 1, "session_id": "bench",
           "recorded_at": 0.0, "observation_key": key, "observation": envelope}
    text = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)
    return envelope, token, text


def build_history(data_dir: Path, size_mb: float, max_bytes: int, template: str, token: str) -> int:
    data_dir.mkdir(parents=True)
    target = int(size_mb * 1024 * 1024)
    written = count = part = 0
    fh = None
    current = 0
    while written < target:
        if fh is None or current >= max_bytes:
            if fh is not None:
                fh.close()
                part += 1
            fh = open(data_dir / f"trials.jsonl.20260101T{part:06d}", "w", encoding="utf-8")
            current = 0
        obs = template.replace(token, f"hist-{count:08d}") + "\n"
        trial = json.dumps({"type": "trial", "trial": count + 1, "session_id": "bench",
                            "metric": 0.5}, separators=(",", ":")) + "\n"
        fh.write(obs + trial)
        current += len(obs) + len(trial)
        written += len(obs) + len(trial)
        count += 1
    fh.close()
    # The newest part is the active file.
    newest = data_dir / f"trials.jsonl.20260101T{part:06d}"
    newest.rename(data_dir / "trials.jsonl")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True)
    parser.add_argument("--sizes-mb", type=float, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--writes", type=int, default=5)
    parser.add_argument("--impl-dir")
    parser.add_argument("--label", default="current")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if args.impl_dir:
        sys.path.insert(0, str(Path(args.impl_dir).resolve()))
    import learning_recorder

    envelope, token, template = _template()
    work = Path(args.work).resolve()
    for size in args.sizes_mb:
        data_dir = work / f"{args.label}-{size:g}mb"
        if data_dir.exists():
            shutil.rmtree(data_dir)
        rows = build_history(data_dir, size, learning_recorder.DEFAULT_MAX_BYTES, template, token)
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t0 = time.perf_counter()
        rec = learning_recorder.LearningRecorder(data_dir, fsync=True)
        open_s = time.perf_counter() - t0
        rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        per_write = []
        for i in range(args.writes):
            fresh = json.loads(json.dumps(envelope))
            fresh["identity"]["run_id"] = f"fresh-{i}"
            t0 = time.perf_counter()
            receipt = rec.record_observation(fresh)
            per_write.append(time.perf_counter() - t0)
            assert receipt["durable"] is True and receipt["idempotent"] is False
        t0 = time.perf_counter()
        assert rec.record_observation(fresh)["idempotent"] is True
        retry_s = time.perf_counter() - t0
        rec.close()
        files = sorted(p.name for p in data_dir.glob("trials.jsonl*"))
        print(json.dumps({
            "impl": args.label, "history_mb": size, "observation_rows": rows,
            "ledger_files": len(files), "open_s": round(open_s, 4),
            "write_median_ms": round(statistics.median(per_write) * 1000, 2),
            "write_max_ms": round(max(per_write) * 1000, 2),
            "idempotent_retry_ms": round(retry_s * 1000, 2),
            "maxrss_growth_mib_at_open": round((rss1 - rss0) / 1024, 1),
        }), flush=True)
        shutil.rmtree(data_dir)


if __name__ == "__main__":
    main()
