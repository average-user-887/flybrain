"""Embodied 3D body recording (body.nfbody) for 1x browser replay."""
import gzip
import json

import pytest

from neurofly_body import cli
from neurofly_body.body_recording import BodyRecorder, frame_interval_steps, read_body_recording
from neurofly_body.decoder import DNa02CPGDecoder
from neurofly_body.runner import EmbodiedConfig, run_embodied
from tests.test_embodied_runner import FakeBody, FakeGraph

SKELETON = {"segments": ["thorax", "head"], "parents": [-1, 0], "units": "mm"}


def _record(step_index, yaw=0.1, events=()):
    return {
        "neural": {"locomotion_dn": {"DNa02_L_rate_hz": 12.3456, "GF_L_spikes": 0}},
        "body": {"thorax": {"yaw_rad": yaw}, "contacts": {"found": [1.0, 0.0]}},
        "motor": {"applied_cpg_drive": [1.0, 0.5], "decoder": {"events": list(events)}},
    }


def _write(path, steps=10, fps=100.0):
    recorder = BodyRecorder(path, fps=fps, neural_dt_ms=2.0)
    recorder.header(skeleton=SKELETON, provenance={"seed": 1})
    for step in range(1, steps + 1):
        recorder.step(step, _record(step, events=[{"event": "takeoff"}] if step == 3 else []),
                      [[step * 0.123456, 0.0, 0.5], [0.0, 0.0, 0.6]], total_spikes=2)
    return recorder.close()


def test_frame_interval_must_be_whole_neural_steps():
    assert frame_interval_steps(50.0, 2.0) == 10
    assert frame_interval_steps(500.0, 2.0) == 1
    for bad in (0.0, -1.0, float("nan"), 60.0, 1000.0):
        with pytest.raises(ValueError):
            frame_interval_steps(bad, 2.0)


def test_frames_carry_window_totals_and_verify(tmp_path):
    summary = _write(tmp_path / "a.nfbody")
    assert summary["frames"] == 2
    rec = read_body_recording(tmp_path / "a.nfbody")
    assert rec["header"]["skeleton"] == SKELETON
    first, second = rec["frames"]
    assert (first["t"], second["t"]) == (0.01, 0.02)
    assert first["pos"] == [0.6173, 0.0, 0.5, 0.0, 0.0, 0.6]      # rounded to 0.1 um
    assert first["spikes"] == 10 and second["spikes"] == 10       # spikes summed over the frame window
    assert first["events"] == [{"event": "takeoff"}] and "events" not in second
    assert first["dn"] == {"DNa02_L": 12.346}
    assert first["contacts"] == [1, 0]


def test_recording_is_byte_identical(tmp_path):
    _write(tmp_path / "a.nfbody")
    _write(tmp_path / "b.nfbody")
    assert (tmp_path / "a.nfbody").read_bytes() == (tmp_path / "b.nfbody").read_bytes()


def test_tampered_frame_is_rejected(tmp_path):
    _write(tmp_path / "a.nfbody")
    lines = gzip.decompress((tmp_path / "a.nfbody").read_bytes()).splitlines(keepends=True)
    lines[1] = lines[1].replace(b'"yaw":0.1', b'"yaw":0.2')
    (tmp_path / "b.nfbody").write_bytes(gzip.compress(b"".join(lines)))
    with pytest.raises(ValueError):
        read_body_recording(tmp_path / "b.nfbody")


def test_recorder_never_overwrites(tmp_path):
    _write(tmp_path / "a.nfbody")
    with pytest.raises(FileExistsError):
        BodyRecorder(tmp_path / "a.nfbody", fps=50.0, neural_dt_ms=2.0)


def test_runner_needs_segment_positions_only_when_recording(tmp_path):
    config = EmbodiedConfig(duration_s=0.006, output_dir=tmp_path / "plain", mode="intact")
    assert "recording" not in run_embodied(config, FakeGraph(), FakeBody(), decoder=DNa02CPGDecoder())
    config = EmbodiedConfig(duration_s=0.006, output_dir=tmp_path / "rec", mode="intact", record_fps=250.0)
    with pytest.raises(Exception, match="segment positions"):
        run_embodied(config, FakeGraph(), FakeBody(), decoder=DNa02CPGDecoder())


def test_config_rejects_fps_off_the_neural_grid(tmp_path):
    with pytest.raises(ValueError):
        EmbodiedConfig(duration_s=0.006, output_dir=tmp_path / "x", record_fps=60.0).validated()


pytest.importorskip("flygym")


def test_flygym_run_writes_a_replayable_recording(tmp_path):
    out = tmp_path / "run"
    assert cli.main(["run", "--controller", "modular", "--duration", "0.1", "--output", str(out),
                     "--seed", "1"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    rec = read_body_recording(out / "body.nfbody")
    assert summary["recording"]["frames"] == len(rec["frames"]) == 5      # 50 fps default
    segments = rec["header"]["skeleton"]["segments"]
    assert all(len(frame["pos"]) == 3 * len(segments) for frame in rec["frames"])
    assert rec["header"]["provenance"]["neural_backend"]["controller_kind"] == "modular-baseline"
