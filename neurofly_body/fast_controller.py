"""Low-overhead drop-in for FlyGym's hybrid turning controller loop.

FlyGym 2.1's ``HybridTurningController`` spends ~85% of each 0.1 ms physics
substep in Python: rebuilding ``BodySegment``/``JointDOF`` objects, dictionary
lookups, ``np.isin`` contact filtering and SciPy spline dispatch. MuJoCo's
``mj_step`` is only ~11%. This module precomputes every index map once and
evaluates the same arithmetic, in the same floating-point order, per substep.
The per-leg maths runs in a numba kernel (no fastmath); a vectorised NumPy
path with the same results is used when numba is unavailable.

It is a performance change only: for the same seed and commands it writes
bit-identical actuator and adhesion controls into ``mj_data.ctrl``, so the
physics trajectory is bit-identical to the stock loop
(``tests/test_fast_controller.py`` checks this against the stock controller).
The controller's state (CPG network, correction arrays, persistence counter)
stays in the stock ``HybridTurningController`` object, which remains the
single source of truth.
"""

from __future__ import annotations

import mujoco as mj
import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - numba is a core dependency
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator


class _PeriodicCubicLegs:
    """Evaluates the per-leg periodic ``CubicSpline``s for all legs at once.

    Reproduces ``PPoly.__call__`` with ``extrapolate='periodic'`` bit for bit:
    the phase is wrapped exactly as SciPy wraps it, the interval is located
    with SciPy's ``x[i] <= x < x[i+1]`` rule, and the polynomial is summed in
    SciPy's ``evaluate_poly1`` order (``res += c[k] * z * 1.0``, lowest power
    first). All legs must share the same breakpoints (they do in FlyGym 2.1).
    """

    def __init__(self, splines) -> None:
        self.x = np.asarray(splines[0].x, dtype=np.float64)
        for spline in splines[1:]:
            if not np.array_equal(spline.x, self.x):
                raise ValueError("per-leg splines must share breakpoints")
        self.x0 = self.x[0]
        self.period = self.x[-1] - self.x[0]
        # (legs, order+1, intervals, dofs); per interval, highest power first.
        self.c = np.ascontiguousarray(np.stack([np.asarray(s.c, dtype=np.float64) for s in splines]))
        self.last = len(self.x) - 2
        self.leg_index = np.arange(len(splines))

    def __call__(self, phases: np.ndarray) -> np.ndarray:
        x = self.x0 + (phases - self.x0) % self.period
        i = np.clip(np.searchsorted(self.x, x, side="right") - 1, 0, self.last)
        s = (x - self.x[i])[:, np.newaxis]
        c = self.c[self.leg_index, :, i, :]  # (legs, order+1, dofs)
        k = c.shape[1]
        res = 0.0 + c[:, k - 1] * 1.0 * 1.0
        z = 1.0 * s
        for kp in range(1, k):
            res = res + c[:, k - kp - 1] * z * 1.0
            if kp < k - 1:
                z = z * s
        return res


def _interp_rows(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """``np.interp(x[r], xp[r], fp)`` for every row ``r``, in NumPy's arithmetic.

    NumPy computes ``slope * (x - xp[j]) + fp[j]`` with
    ``slope = (fp[j+1] - fp[j]) / (xp[j+1] - xp[j])``, returns ``fp[j]`` when
    ``x == xp[j]``, ``fp[-1]`` at or past the right end and ``fp[0]`` left of it.
    """
    rows = np.arange(len(x))
    last = xp.shape[1] - 1
    j = np.clip(_search_right(xp, x) - 1, 0, last - 1)
    x_lo = xp[rows, j]
    x_hi = xp[rows, j + 1]
    slope = (fp[j + 1] - fp[j]) / (x_hi - x_lo)
    out = slope * (x - x_lo) + fp[j]
    out = np.where(x == x_lo, fp[j], out)
    out = np.where(x >= xp[:, last], fp[last], out)
    return np.where(x < xp[:, 0], fp[0], out)


def _search_right(xp: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (xp <= x[:, np.newaxis]).sum(axis=1)


@njit(cache=True)
def _legs_kernel(
    phases, magnitudes, retraction, stumble, retracting, stumbling_mask,
    r_up, r_down, s_up, s_down, max_correction,
    grid, x0, period, coeffs, neutral, gain_points, gain_values, correction,
    swing_start, swing_end, enable_adhesion, two_pi,
    perm, actuator_ids, adhesion_ids, ctrl, net_gain,
):  # pragma: no cover - compiled
    """Per-leg part of ``HybridController.step`` plus writing ``mj_data.ctrl``.

    Scalar IEEE arithmetic in the same order as the NumPy/SciPy code it
    replaces (no fastmath, so no reassociation or FMA contraction).
    """
    n_legs, n_dofs = neutral.shape
    order = coeffs.shape[1]
    last_interval = grid.shape[0] - 2
    n_points = gain_points.shape[1]
    angles = np.empty(n_legs * n_dofs)
    for leg in range(n_legs):
        if retracting[leg]:
            retraction[leg] = retraction[leg] + r_up
        else:
            retraction[leg] = max(0.0, retraction[leg] - r_down)
        if stumbling_mask[leg]:
            stumble[leg] = stumble[leg] + s_up
        else:
            stumble[leg] = max(0.0, stumble[leg] - s_down)
        if retraction[leg] > 0:
            net = retraction[leg]
            stumble[leg] = 0.0
        else:
            net = stumble[leg]
        net = min(max(net, 0.0), max_correction)

        # Periodic CubicSpline (PPoly) evaluation, SciPy's order of operations.
        phase = phases[leg]
        x = x0 + (phase - x0) % period
        interval = 0
        while interval < last_interval and grid[interval + 1] <= x:
            interval += 1
        s = x - grid[interval]

        # np.interp of the phase-gain curve (NumPy's formula and edge rules).
        wrapped = phase % two_pi
        if wrapped < gain_points[leg, 0]:
            gain = gain_values[0]
        elif wrapped >= gain_points[leg, n_points - 1]:
            gain = gain_values[n_points - 1]
        else:
            j = 0
            while gain_points[leg, j + 1] <= wrapped:
                j += 1
            if wrapped == gain_points[leg, j]:
                gain = gain_values[j]
            else:
                slope = (gain_values[j + 1] - gain_values[j]) / (
                    gain_points[leg, j + 1] - gain_points[leg, j]
                )
                gain = slope * (wrapped - gain_points[leg, j]) + gain_values[j]
        leg_gain = net * gain
        net_gain[leg] = leg_gain

        for dof in range(n_dofs):
            res = 0.0 + coeffs[leg, order - 1, interval, dof] * 1.0 * 1.0
            z = 1.0 * s
            for kp in range(1, order):
                res = res + coeffs[leg, order - kp - 1, interval, dof] * z * 1.0
                if kp < order - 1:
                    z = z * s
            offset = res - neutral[leg, dof]
            angle = neutral[leg, dof] + magnitudes[leg] * offset
            angles[leg * n_dofs + dof] = angle + leg_gain * correction[leg, dof]

        if enable_adhesion:
            swinging = swing_start[leg] < wrapped and wrapped < swing_end[leg]
            ctrl[adhesion_ids[leg]] = 0.0 if swinging else 1.0
        else:
            ctrl[adhesion_ids[leg]] = 0.0
    for index in range(perm.shape[0]):
        ctrl[actuator_ids[index]] = angles[perm[index]]


class FastHybridLoop:
    """Runs ``from_sim -> controller.step -> apply_action -> mj_step`` fast.

    ``controller`` must be a stock ``HybridTurningController`` built with the
    fly's actuated position DOF order; its state is read and updated in place.
    """

    def __init__(
        self, sim, fly_name: str, controller, preprogrammed_steps, *, use_numba: bool | None = None
    ) -> None:
        from flygym.compose import ActuatorType
        from flygym_demo.complex_terrain.common import dof_spec_to_jointdof
        from flygym_demo.complex_terrain.hybrid_controller import (
            _CORRECTION_VECTORS,
            _DETECTED_STUMBLING_LINKS,
            _RIGHT_LEG_CORRECTION_SIGN,
        )

        if controller.output_dof_order is None:
            raise ValueError("FastHybridLoop needs a controller with an explicit output_dof_order")
        self.sim = sim
        self.model = sim.mj_model
        self.data = sim.mj_data
        self.c = controller
        self.steps = preprogrammed_steps
        legs = tuple(controller.legs)
        self.legs = legs

        fly = sim.world.fly_lookup[fly_name]
        seg_cls = type(fly).BODY_SEGMENT_CLASS
        body_order = fly.get_bodysegs_order()
        body_ids = sim._internal_bodyids_by_fly[fly_name]
        thorax_idx = body_order.index(seg_cls("c_thorax"))
        self.thorax_body = int(body_ids[thorax_idx])
        self.tarsus5_bodies = np.array(
            [body_ids[body_order.index(seg_cls(f"{leg}_tarsus5"))] for leg in legs],
            dtype=np.intp,
        )

        # Stumbling contact detection: geom id -> output row, ground geom ids.
        n_links = len(_DETECTED_STUMBLING_LINKS)
        self.n_links = n_links
        geom_ids_by_segment = sim._internal_geomid_by_bodyseg_by_fly[fly_name]
        segments = [seg_cls(f"{leg}_{link}") for leg in legs for link in _DETECTED_STUMBLING_LINKS]
        from flygym.anatomy import BodySegment
        segments = [s if isinstance(s, BodySegment) else BodySegment(s) for s in segments]
        self.geom_to_row = {int(geom_ids_by_segment[seg]): i for i, seg in enumerate(segments)}
        self.n_rows = len(segments)
        self.ground_geoms = frozenset(int(g) for g in np.asarray(sim._internal_ground_geom_ids).ravel())
        self._wrench = np.zeros(6, dtype=float)

        # Output permutation: flat (leg, dof) index -> actuator order.
        dofs_per_leg = self.steps.dofs_per_leg
        n_dofs = len(dofs_per_leg)
        position = {}
        for leg_idx, leg in enumerate(legs):
            for dof_idx, spec in enumerate(dofs_per_leg):
                position[dof_spec_to_jointdof(leg, spec)] = leg_idx * n_dofs + dof_idx
        self.perm = np.array([position[dof] for dof in controller.output_dof_order], dtype=np.intp)
        self.n_dofs = n_dofs

        self.actuator_ids = np.asarray(
            sim._intern_actuatorids_by_type_by_fly[ActuatorType.POSITION][fly_name]
        )
        self.adhesion_ids = np.asarray(sim._intern_adhesionactuatorids_by_fly[fly_name])
        if len(self.actuator_ids) != len(self.perm):
            raise ValueError("actuator count does not match the controller DOF order")
        if len(self.adhesion_ids) != len(legs):
            raise ValueError("adhesion actuator count does not match the leg count")

        # Per-leg constants, computed with the same expressions as the stock code.
        self.splines = _PeriodicCubicLegs([self.steps._psi_funcs[leg] for leg in legs])
        self.neutral = np.stack([self.steps.neutral_pos[leg][:, 0] for leg in legs])
        correction = []
        gain_points = []
        adhesion_window = []
        for leg in legs:
            vector = _CORRECTION_VECTORS[leg[1]]
            if leg.startswith("r"):
                vector = vector * _RIGHT_LEG_CORRECTION_SIGN
            correction.append(vector)
            swing_start, swing_end = self.steps.swing_period[leg]
            ext = controller.swing_extension
            gain_points.append(
                np.array(
                    [
                        swing_start,
                        np.mean([swing_start, swing_end]),
                        swing_end + ext,
                        np.mean([swing_end, 2 * np.pi]),
                        2 * np.pi,
                    ]
                )
            )
            adhesion_window.append((swing_start, swing_end + ext))
        self.correction = np.stack(correction)
        self.gain_points = np.stack(gain_points)
        self.gain_values = np.array([0.0, 0.8, 0.0, -0.1, 0.0])
        self.swing_start = np.array([w[0] for w in adhesion_window])
        self.swing_end = np.array([w[1] for w in adhesion_window])
        self._last_command = None
        self.use_numba = HAVE_NUMBA if use_numba is None else bool(use_numba)
        self._net_gain = np.zeros(len(legs), dtype=float)
        self._ctrl_args = (
            self.splines.x,
            float(self.splines.x0),
            float(self.splines.period),
            self.splines.c,
            np.ascontiguousarray(self.neutral),
            np.ascontiguousarray(self.gain_points),
            self.gain_values,
            np.ascontiguousarray(self.correction),
            self.swing_start,
            self.swing_end,
        )

    # -- observation ---------------------------------------------------------
    def _stumbling_forces(self) -> np.ndarray:
        forces = np.zeros((self.n_rows, 3), dtype=float)
        data = self.data
        ncon = data.ncon
        if ncon == 0:
            return forces
        contacts = data.contact
        geom1 = contacts.geom1[:ncon]
        geom2 = contacts.geom2[:ncon]
        exclude = contacts.exclude[:ncon]
        rows = self.geom_to_row
        ground = self.ground_geoms
        wrench = self._wrench
        for contact_id in range(ncon):
            g1 = int(geom1[contact_id])
            g2 = int(geom2[contact_id])
            if exclude[contact_id]:
                continue
            if not ((g1 in rows and g2 in ground) or (g2 in rows and g1 in ground)):
                continue
            mj.mj_contactForce(self.model, data, contact_id, wrench)
            frame = contacts.frame[contact_id].reshape(3, 3)
            world_force = frame.T @ wrench[:3]
            if g1 in rows:
                forces[rows[g1]] -= world_force
            if g2 in rows:
                forces[rows[g2]] += world_force
        return forces

    # -- one substep ---------------------------------------------------------
    def _set_command(self, command: np.ndarray) -> None:
        """HybridTurningController.step's side-specific CPG modulation.

        Stock code rebuilds these arrays every substep; they only depend on the
        command, so they are rebuilt only when the command changes or something
        else (e.g. ``controller.reset``) has replaced them.
        """
        net = self.c.cpg_network
        key = (float(command[0]), float(command[1]))
        cached = self._last_command
        if (
            cached is not None
            and cached[0] == key
            and net.intrinsic_amps is cached[1]
            and net.intrinsic_freqs is cached[2]
        ):
            return
        net.intrinsic_amps = np.repeat(np.abs(command[:, np.newaxis]), 3, axis=1).ravel()
        freqs = self.c._base_intrinsic_freqs.copy()
        freqs[:3] *= 1 if command[0] >= 0 else -1
        freqs[3:] *= 1 if command[1] >= 0 else -1
        net.intrinsic_freqs = freqs
        self._last_command = (key, net.intrinsic_amps, net.intrinsic_freqs)

    def substep(self, command: np.ndarray) -> None:
        c = self.c
        data = self.data
        xpos = data.xpos
        thorax_z = float(xpos[self.thorax_body, 2])
        tarsus5_z = np.array(xpos[self.tarsus5_bodies, 2], dtype=float)
        stumbling = self._stumbling_forces().reshape(len(self.legs), self.n_links, 3)
        heading = data.xmat[self.thorax_body].reshape(3, 3)[:, 0].copy()
        self._set_command(command)

        # HybridController.step, vectorised over legs (legs are independent).
        end_effector_z = thorax_z - tarsus5_z
        order = np.argsort(end_effector_z)
        ordered = end_effector_z[order]
        retract_leg = int(order[-1]) if ordered[-1] > ordered[-3] + c.retraction_height_threshold else None
        counter = c.retraction_persistence_counter
        if retract_leg is not None and (
            c.retraction_correction[retract_leg] > c.retraction_persistence_initiation_threshold
        ):
            counter[retract_leg] = 1
        counter[counter > 0] += 1
        counter[counter > c.retraction_persistence_steps] = 0

        force_proj = np.dot(stumbling, heading)
        stumbling_mask = (force_proj < c.stumbling_force_threshold).any(axis=1)
        c.cpg_network.step()

        dt = c.timestep
        retraction = c.retraction_correction
        stumble = c.stumbling_correction
        retracting = counter > 0
        if retract_leg is not None:
            retracting[retract_leg] = True
        phases = c.cpg_network.curr_phases
        magnitudes = c.cpg_network.curr_magnitudes

        if self.use_numba:
            net_gain = self._net_gain
            _legs_kernel(
                phases, magnitudes, retraction, stumble, retracting, stumbling_mask,
                c.retraction_rates[0] * dt, c.retraction_rates[1] * dt,
                c.stumbling_rates[0] * dt, c.stumbling_rates[1] * dt,
                float(c.max_correction), *self._ctrl_args, bool(c.enable_adhesion), 2 * np.pi,
                self.perm, self.actuator_ids, self.adhesion_ids, data.ctrl, net_gain,
            )
            net_gain = net_gain.copy()
        else:
            net_gain = self._legs_numpy(
                phases, magnitudes, retraction, stumble, retracting, stumbling_mask
            )

        c.last_info = {
            "net_corrections": net_gain,
            "retraction_correction": retraction.copy(),
            "stumbling_correction": stumble.copy(),
            "stumbling_mask": stumbling_mask,
            "leg_to_correct_retraction": retract_leg,
        }
        mj.mj_step(self.model, data)

    def _legs_numpy(self, phases, magnitudes, retraction, stumble, retracting, stumbling_mask):
        """Vectorised NumPy equivalent of ``_legs_kernel`` (used without numba)."""
        c = self.c
        dt = c.timestep
        retraction[:] = np.where(
            retracting,
            retraction + c.retraction_rates[0] * dt,
            np.maximum(0, retraction - c.retraction_rates[1] * dt),
        )
        stumble[:] = np.where(
            stumbling_mask,
            stumble + c.stumbling_rates[0] * dt,
            np.maximum(0, stumble - c.stumbling_rates[1] * dt),
        )
        retracted = retraction > 0
        stumble[retracted] = 0
        net_correction = np.where(retracted, retraction, stumble)

        offset = self.splines(phases) - self.neutral
        leg_angles = self.neutral + magnitudes[:, np.newaxis] * offset

        net_correction = np.clip(net_correction, 0, c.max_correction)
        wrapped = phases % (2 * np.pi)
        phase_gain = _interp_rows(wrapped, self.gain_points, self.gain_values)
        net_gain = net_correction * phase_gain
        leg_angles = leg_angles + net_gain[:, np.newaxis] * self.correction
        if c.enable_adhesion:
            adhesion = ~((self.swing_start < wrapped) & (wrapped < self.swing_end))
        else:
            adhesion = np.zeros(len(self.legs), dtype=bool)
        data = self.data
        data.ctrl[self.actuator_ids] = leg_angles.ravel()[self.perm]
        data.ctrl[self.adhesion_ids] = adhesion
        return net_gain
