"""FlyGym 2.1 / MuJoCo articulated-body adapter."""

from __future__ import annotations

import math
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np


def _yaw_from_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrapped_delta(current: float, previous: float) -> float:
    return math.atan2(math.sin(current - previous), math.cos(current - previous))


class FlyGymBody:
    """Stock FlyGym locomotion model driven by its hybrid turning controller."""

    FLY_NAME = "neurofly"

    def __init__(
        self,
        *,
        physics_dt_s: float = 0.0001,
        warmup_s: float = 0.05,
        video_path: Path | None = None,
    ) -> None:
        if physics_dt_s <= 0 or warmup_s < 0:
            raise ValueError("physics_dt_s must be positive and warmup_s non-negative")
        installed = version("flygym")
        if installed != "2.1.0":
            raise RuntimeError(f"neurofly_body requires FlyGym 2.1.0, found {installed}")

        # Imports are intentionally lazy: unit tests and ``--help`` do not require
        # FlyGym, MuJoCo, OpenGL, or the demo package to be installed.
        from flygym import Simulation
        from flygym.anatomy import BodySegment
        from flygym.compose import FlatGroundWorld
        from flygym.utils.math import Rotation3D
        from flygym_demo.complex_terrain import (
            HybridControllerObservation,
            HybridTurningController,
            LocomotionAction,
            PreprogrammedSteps,
            apply_locomotion_action,
            make_locomotion_fly,
        )

        self._BodySegment = BodySegment
        self._HybridControllerObservation = HybridControllerObservation
        self._LocomotionAction = LocomotionAction
        self._apply_action = apply_locomotion_action
        self._preprogrammed_steps = PreprogrammedSteps()
        self._video_path = Path(video_path) if video_path is not None else None
        self._warmup_s = float(warmup_s)

        fly = make_locomotion_fly(name=self.FLY_NAME, add_adhesion=True, colorize=True)
        self._camera = None
        if self._video_path is not None:
            self._camera = fly.add_tracking_camera(
                name="body_cam",
                pos_offset=(-0.5, -7.5, 0.0),
                rotation=Rotation3D("euler", (1.57, 0.0, 0.0)),
                fovy=30.0,
            )
        world = FlatGroundWorld()
        world.add_fly(
            fly,
            spawn_pos=[0.0, 0.0, 0.5],
            spawn_rot=Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]),
        )
        self.fly = fly
        self.sim = Simulation(world, timestep=float(physics_dt_s))
        self.controller = HybridTurningController(
            timestep=self.sim.timestep,
            preprogrammed_steps=self._preprogrammed_steps,
            output_dof_order=fly.get_actuated_jointdofs_order("position"),
        )
        self.renderer = None
        if self._camera is not None:
            self.renderer = self.sim.set_renderer(
                [self._camera], playback_speed=1.0, output_fps=25, buffer_frames=True
            )

        body_order = fly.get_bodysegs_order()
        self._thorax_idx = body_order.index(BodySegment("c_thorax"))
        self._joint_names = [str(item) for item in fly.get_jointdofs_order()]
        self._actuated_joint_names = [
            str(item) for item in fly.get_actuated_jointdofs_order("position")
        ]
        self._previous_yaw: float | None = None
        self._last_observation_time: float | None = None
        self._video_saved = False

    @property
    def physics_dt_s(self) -> float:
        return float(self.sim.timestep)

    def reset(self, seed: int) -> dict[str, Any]:
        self.sim.reset()
        self.controller.reset(seed=int(seed))
        initial_action = self._LocomotionAction(
            joint_angles=self._preprogrammed_steps.default_pose_by_dof_order(
                self.fly.get_actuated_jointdofs_order("position")
            ),
            adhesion_onoff=np.ones(6, dtype=bool),
        )
        self._apply_action(self.sim, self.FLY_NAME, initial_action)
        if self._warmup_s:
            self.sim.warmup(self._warmup_s)
        self._previous_yaw = None
        self._last_observation_time = None
        return self.observe()

    def step(self, cpg_drive: tuple[float, float], substeps: int) -> dict[str, Any]:
        command = np.asarray(cpg_drive, dtype=float)
        if command.shape != (2,) or not np.isfinite(command).all():
            raise ValueError("cpg_drive must contain two finite values")
        if (command < 0).any():
            raise ValueError("cpg_drive values cannot be negative")
        if substeps <= 0:
            raise ValueError("substeps must be positive")
        for _ in range(int(substeps)):
            controller_obs = self._HybridControllerObservation.from_sim(
                self.sim, self.FLY_NAME
            )
            action = self.controller.step(command, controller_obs)
            self._apply_action(self.sim, self.FLY_NAME, action)
            self.sim.step()
            if self.renderer is not None:
                self.sim.render_as_needed()
        return self.observe()

    def observe(self) -> dict[str, Any]:
        positions = np.asarray(self.sim.get_body_positions(self.FLY_NAME), dtype=float)
        rotations = np.asarray(self.sim.get_body_rotations(self.FLY_NAME), dtype=float)
        thorax_pos = positions[self._thorax_idx]
        thorax_quat = rotations[self._thorax_idx]
        yaw = _yaw_from_wxyz(thorax_quat)
        now = float(self.sim.time)
        if self._previous_yaw is None or self._last_observation_time is None:
            yaw_velocity = 0.0
        else:
            dt = now - self._last_observation_time
            if dt <= 0:
                raise RuntimeError("FlyGym body clock did not advance monotonically")
            yaw_velocity = _wrapped_delta(yaw, self._previous_yaw) / dt
        self._previous_yaw = yaw
        self._last_observation_time = now

        contact_found, forces, torques, contact_pos, normals, tangents = (
            self.sim.get_ground_contact_info(self.FLY_NAME)
        )
        return {
            "body_sim_time_s": now,
            "thorax": {
                "position_mm": thorax_pos.tolist(),
                "quaternion_wxyz": thorax_quat.tolist(),
                "yaw_rad": yaw,
                "yaw_velocity_rad_s": yaw_velocity,
            },
            "joint_angles_rad": np.asarray(
                self.sim.get_joint_angles(self.FLY_NAME), dtype=float
            ).tolist(),
            "joint_velocities_rad_s": np.asarray(
                self.sim.get_joint_velocities(self.FLY_NAME), dtype=float
            ).tolist(),
            "contacts": {
                "found": np.asarray(contact_found, dtype=float).tolist(),
                "forces_mujoco_model_units": np.asarray(forces, dtype=float).tolist(),
                "torques_mujoco_model_units": np.asarray(torques, dtype=float).tolist(),
                "positions_mm": np.asarray(contact_pos, dtype=float).tolist(),
                "normals_world": np.asarray(normals, dtype=float).tolist(),
                "tangents_world": np.asarray(tangents, dtype=float).tolist(),
            },
            "cpg_magnitudes": np.asarray(
                self.controller.cpg_network.curr_magnitudes, dtype=float
            ).tolist(),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "adapter": "neurofly_body.flygym_body.FlyGymBody",
            "flygym_version": version("flygym"),
            "mujoco_version": version("mujoco"),
            "model": "FlyGym stock NeuroMechFly articulated locomotion model",
            "controller": "flygym_demo.complex_terrain.HybridTurningController",
            "physics_dt_s": self.physics_dt_s,
            "warmup_s": self._warmup_s,
            "joint_order": self._joint_names,
            "actuated_joint_order": self._actuated_joint_names,
            "units": {
                "clock": "s",
                "body_position": "mm",
                "joint_angle": "rad",
                "joint_velocity": "rad/s",
                "angular_velocity": "rad/s",
                "contact_force_and_torque": "raw MuJoCo model units (not converted)",
            },
        }

    def save_video(self) -> None:
        if self.renderer is not None and self._video_path is not None and not self._video_saved:
            self.renderer.save_video(self._video_path)
            self._video_saved = True

    def close(self) -> None:
        self.sim.close()
