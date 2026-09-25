"""Deterministic lockstep driver for neural and physical backends."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .decoder import DNa02CPGDecoder, DNCommandDecoder
from .interfaces import BodyBackend, NeuralBackend


# Neural reply fields that measure the host, not the simulation.  They go to
# timing.jsonl so telemetry.jsonl depends only on the seed, config and code.
WALL_CLOCK_FIELDS = ("elapsed_ms",)


@dataclass(frozen=True)
class EmbodiedConfig:
    duration_s: float
    output_dir: Path
    mode: str = "intact"
    neural_dt_ms: float = 2.0
    physics_dt_s: float = 0.0001
    world_angular_velocity_rad_s: float = 4.0
    contrast: float = 1.0
    seed: int = 0

    def validated(self) -> "EmbodiedConfig":
        if self.mode not in {"intact", "output-disconnected"}:
            raise ValueError("mode must be 'intact' or 'output-disconnected'")
        finite = (
            self.duration_s,
            self.neural_dt_ms,
            self.physics_dt_s,
            self.world_angular_velocity_rad_s,
            self.contrast,
        )
        if not all(math.isfinite(float(value)) for value in finite):
            raise ValueError("all timing and stimulus values must be finite")
        if self.duration_s <= 0 or self.neural_dt_ms <= 0 or self.physics_dt_s <= 0:
            raise ValueError("duration and timesteps must be positive")
        if not 0.0 <= self.contrast <= 1.0:
            raise ValueError("contrast must be between 0 and 1")
        steps = self.duration_s / (self.neural_dt_ms / 1000.0)
        substeps = (self.neural_dt_ms / 1000.0) / self.physics_dt_s
        if not math.isclose(steps, round(steps), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("duration must contain an integer number of neural steps")
        if not math.isclose(substeps, round(substeps), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("neural_dt_ms must contain an integer number of physics substeps")
        return self


def validate_real_v3_status(status: dict[str, Any]) -> None:
    """Fail closed unless the backend identifies the intended real graph run."""
    required = {
        "synthetic": False,
        "lif_dynamics_version": "v3",
        "engineered_assistance_enabled": False,
        "transmitter_policy": "v3-modulatory-only",
    }
    problems = []
    for key, expected in required.items():
        if status.get(key) != expected:
            problems.append(f"{key}={status.get(key)!r}, expected {expected!r}")
    if not status.get("graph_sha256"):
        problems.append("graph_sha256 is missing")
    if not status.get("optomotor_io_map_sha256"):
        problems.append("optomotor_io_map_sha256 is missing")
    if problems:
        raise RuntimeError(
            "refusing embodied run: backend is not the verified real v3 configuration ("
            + "; ".join(problems)
            + ")"
        )


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _assert_finite(value: Any, path: str = "record") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise FloatingPointError(f"non-finite value at {path}: {value!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(_to_builtin(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


class _RunOutput:
    """Exclusive output reservation and crash-visible manifest lifecycle."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.telemetry_path = self.output_dir / "telemetry.jsonl"
        self.timing_path = self.output_dir / "timing.jsonl"
        self.manifest_path = self.output_dir / "manifest.json"
        self.summary_path = self.output_dir / "summary.json"
        self._stream = self.telemetry_path.open("x", encoding="utf-8", buffering=1)
        self._timing = self.timing_path.open("x", encoding="utf-8", buffering=1)
        self._digest = hashlib.sha256()
        self.manifest: dict[str, Any] = {}

    def set_manifest(self, manifest: dict[str, Any]) -> None:
        self.manifest = manifest
        _write_json(self.manifest_path, manifest)

    def write(self, record: dict[str, Any]) -> None:
        record = _to_builtin(record)
        _assert_finite(record)
        line = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        self._digest.update(line.encode("utf-8"))
        self._stream.write(line)

    def write_timing(self, record: dict[str, Any]) -> None:
        self._timing.write(json.dumps(_to_builtin(record), sort_keys=True) + "\n")

    @property
    def trajectory_sha256(self) -> str:
        """SHA-256 of telemetry.jsonl as written so far."""
        return self._digest.hexdigest()

    def complete(self, summary: dict[str, Any]) -> None:
        self._stream.flush()
        self._timing.flush()
        _write_json(self.summary_path, summary)
        self.manifest.update(
            status="complete",
            completed_at=datetime.now(timezone.utc).isoformat(),
            records=int(summary["records"]),
            trajectory_sha256=summary["trajectory_sha256"],
        )
        _write_json(self.manifest_path, self.manifest)

    def fail(self, error: BaseException) -> None:
        self.manifest.update(
            status="failed",
            failed_at=datetime.now(timezone.utc).isoformat(),
            error={"type": type(error).__name__, "message": str(error)},
        )
        _write_json(self.manifest_path, self.manifest)

    def close(self) -> None:
        self._stream.close()
        self._timing.close()


def run_embodied(
    config: EmbodiedConfig,
    neural: NeuralBackend,
    body: BodyBackend,
    *,
    decoder: DNCommandDecoder | DNa02CPGDecoder | None = None,
    invocation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a coupled experiment and return the written summary.

    The graph is stepped once per neural tick.  Its command is then held while
    the body/controller executes an integer number of fixed physics substeps.

    ``telemetry.jsonl`` holds only simulated quantities, so the same code, seed,
    config and brain backend give a byte-identical file; its SHA-256 is the
    run's ``trajectory_sha256``.  Wall-clock measurements go to ``timing.jsonl``.
    ``invocation`` (the CLI arguments) is stored so ``replay-check`` can re-run it.
    """
    config = config.validated()
    decoder = decoder or DNCommandDecoder()
    substeps = int(round((config.neural_dt_ms / 1000.0) / config.physics_dt_s))
    n_steps = int(round(config.duration_s / (config.neural_dt_ms / 1000.0)))
    if not math.isclose(body.physics_dt_s, config.physics_dt_s, abs_tol=1e-12):
        raise RuntimeError(
            f"body physics_dt_s={body.physics_dt_s} differs from requested {config.physics_dt_s}"
        )

    output = _RunOutput(config.output_dir)
    started = time.perf_counter()
    status: dict[str, Any] = {}
    previous_neural_ms = -math.inf
    previous_body_time = -math.inf
    total_spikes = 0
    driven_steps = 0
    decoded_steps = 0
    events: list[dict[str, Any]] = []
    last_record: dict[str, Any] | None = None
    try:
        neural.reset()
        status = _to_builtin(neural.get_status())
        if status.get("controller_kind") != "modular-baseline":
            validate_real_v3_status(status)
        missing = [key for key in decoder.required_status if not status.get(key)]
        if missing:
            raise RuntimeError(
                f"refusing embodied run: decoder {decoder.name} needs {missing} from the graph backend"
            )
        body_obs = _to_builtin(body.reset(config.seed))
        _assert_finite(body_obs, "initial_body")
        previous_body_time = float(body_obs["body_sim_time_s"])
        decoder.reset()

        manifest = {
            "schema": "neurofly-embodied-run-v1",
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "package_version": __version__,
            "command_mode": config.mode,
            "config": _to_builtin(asdict(config)),
            "lockstep": {
                "neural_dt_ms": config.neural_dt_ms,
                "physics_dt_s": config.physics_dt_s,
                "physics_substeps_per_neural_step": substeps,
                "ordering": "measure body -> compute retinal slip -> graph -> decode -> body substeps",
            },
            "sensory_feedback": {
                "equation": "retinal_slip_rad_s = world_angular_velocity_rad_s - body_yaw_velocity_rad_s",
                "contrast": config.contrast,
            },
            "neural_backend": status,
            "body_backend": _to_builtin(body.describe()),
            "decoder": decoder.describe(),
            "control": {
                "intact": "decoded command reaches FlyGym CPG",
                "output-disconnected": "same graph and feedback run, decoded command replaced by [0,0]",
            },
            "determinism": {
                "trajectory_sha256": "SHA-256 of telemetry.jsonl; set when the run completes",
                "excluded_wall_clock_fields": list(WALL_CLOCK_FIELDS),
                "replay_requires": "same code, config, seed, graph and brain_backend",
            },
            "invocation": _to_builtin(invocation or {}),
            "provenance": {
                "python": sys.version,
                "platform": platform.platform(),
                "argv": list(sys.argv),
                "pid": os.getpid(),
            },
            "limitations": [
                "The DN-to-CPG decoder is an engineered bridge, not a biological VNC model; its gains are assumptions.",
                "The model does not establish full behavioral reproduction.",
                "Passive v3 graph activity may yield sparse or zero DNa02 spikes.",
                "Contact force and torque values remain in raw MuJoCo model units.",
            ],
        }
        output.set_manifest(manifest)

        for step_index in range(n_steps):
            step_started = time.perf_counter()
            body_yaw_velocity = float(body_obs["thorax"]["yaw_velocity_rad_s"])
            slip = config.world_angular_velocity_rad_s - body_yaw_velocity
            reply = _to_builtin(
                neural.step(
                    {
                        "optomotor_slip_rad_s": slip,
                        "optomotor_contrast": config.contrast,
                    },
                    duration_ms=config.neural_dt_ms,
                )
            )
            _assert_finite(reply, "neural")
            timing = {"step": step_index + 1}
            for field in WALL_CLOCK_FIELDS:
                if field in reply:
                    timing[f"neural_{field}"] = reply.pop(field)
            for field in ("sim_ms", "total_step_spikes"):
                if field not in reply:
                    raise RuntimeError(f"neural reply is missing required field {field!r}")
            neural_ms = float(reply["sim_ms"])
            expected_neural_ms = (step_index + 1) * config.neural_dt_ms
            if neural_ms <= previous_neural_ms:
                raise RuntimeError("neural clock did not advance monotonically")
            if not math.isclose(neural_ms, expected_neural_ms, abs_tol=1e-6):
                raise RuntimeError(
                    f"neural clock {neural_ms} ms differs from lockstep {expected_neural_ms} ms"
                )
            previous_neural_ms = neural_ms

            decoded = decoder.decode_reply(reply, config.neural_dt_ms)
            decoded_command = (
                float(decoded["left_cpg_drive"]),
                float(decoded["right_cpg_drive"]),
            )
            if decoded_command != (0.0, 0.0):
                decoded_steps += 1
            for event in decoded["events"]:
                events.append({"step": step_index + 1, **event})
            applied_command = (
                (0.0, 0.0) if config.mode == "output-disconnected" else decoded_command
            )
            if applied_command != (0.0, 0.0):
                driven_steps += 1
            body_obs = _to_builtin(body.step(applied_command, substeps))
            _assert_finite(body_obs, "body")
            body_time = float(body_obs["body_sim_time_s"])
            expected_body_time = previous_body_time + substeps * config.physics_dt_s
            if body_time <= previous_body_time:
                raise RuntimeError("body clock did not advance monotonically")
            if not math.isclose(body_time, expected_body_time, abs_tol=1e-9):
                raise RuntimeError(
                    f"body clock increment {body_time - previous_body_time} differs from lockstep"
                )
            previous_body_time = body_time

            total_spikes += int(reply["total_step_spikes"])
            last_record = {
                "schema": "neurofly-embodied-step-v1",
                "step": step_index + 1,
                "run_time_s": (step_index + 1) * config.neural_dt_ms / 1000.0,
                "sensory": {
                    "world_angular_velocity_rad_s": config.world_angular_velocity_rad_s,
                    "body_yaw_velocity_rad_s": body_yaw_velocity,
                    "retinal_slip_rad_s": slip,
                    "contrast": config.contrast,
                },
                "neural": reply,
                "motor": {
                    "decoder": decoded,
                    "decoded_cpg_drive": list(decoded_command),
                    "applied_cpg_drive": list(applied_command),
                    "output_connected": config.mode == "intact",
                },
                "body": body_obs,
            }
            output.write(last_record)
            timing["step_wall_ms"] = (time.perf_counter() - step_started) * 1000.0
            output.write_timing(timing)

        if hasattr(body, "save_video"):
            body.save_video()  # type: ignore[attr-defined]
        summary = {
            "schema": "neurofly-embodied-summary-v1",
            "status": "complete",
            "mode": config.mode,
            "records": n_steps,
            "duration_s": config.duration_s,
            "wall_time_s": time.perf_counter() - started,
            "total_graph_spikes": total_spikes,
            "steps_with_nonzero_applied_drive": driven_steps,
            "steps_with_nonzero_decoded_drive": decoded_steps,
            "any_neural_motor_output": decoded_steps > 0,
            "decoder": decoder.name,
            "motor_events": events,
            "final_thorax": None if last_record is None else last_record["body"]["thorax"],
            "trajectory_sha256": output.trajectory_sha256,
            "real_time_factor": config.duration_s / max(time.perf_counter() - started, 1e-12),
            "artifacts": {
                "manifest": "manifest.json",
                "telemetry": "telemetry.jsonl",
                "timing": "timing.jsonl",
                "summary": "summary.json",
            },
        }
        _assert_finite(summary, "summary")
        output.complete(summary)
        return summary
    except BaseException as error:
        if not output.manifest:
            output.manifest = {
                "schema": "neurofly-embodied-run-v1",
                "status": "running",
                "config": _to_builtin(asdict(config)),
                "traceback": traceback.format_exc(),
            }
        output.fail(error)
        raise
    finally:
        output.close()
        body.close()
