#!/usr/bin/env python3
"""Deterministic wall-progress fixtures (remediation plan, work package 3).

Each fixture declares, before it runs, the physical constraint the arena must
enforce and the progress criterion that tells a working solver from a trapping one.
Three verdicts are kept apart:

* arena containment - solver/physics: no escape, no unexplained jump, no numerical
  trapping (``containment_audit.MotionProbe``);
* controller task success - what the controller achieved (reaching a cue, making
  progress, or pushing a wall / resting). Reported, never counted as a physics defect;
* biological validity - not established by any fixture here (see ``BIOLOGY``).

Physics fixtures drive the body with scripted *known escape commands* that bypass
every brain, so a failure to move points at the solver, not at a policy. Behaviour
fixtures run the modular controller (and restored trained checkpoints) and report
immobility or wall pushing as task outcomes. All run well beyond 12 simulated s.

Run as a script to write receipts::

    ./.venv/bin/python tests/fixtures/wall_progress.py --out outputs/wp3-wall-progress \
        --retained-brains /path/to/COPY/of/brains
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import containment_audit as audit  # noqa: E402
from arena import Arena, RectRegion  # noqa: E402
from maze import ExperimentParadigm, WallSegment  # noqa: E402

DT = 0.02
V = 1.5                 # scripted walking speed, mm/s (modular brain walks 0.1-3.5)
ALIGN_TOL = 0.2         # rad: an escape command counts once the heading is this close
ESCAPE_RATIO = 0.9      # aligned escape: realized / attempted motion along the escape direction
SLIDE_RATIO = 0.5       # oblique contact: realized / attempted tangential motion
STILL_TOL_MM = 1e-6     # head-on pushing: per-step creep allowed at contact
SYMMETRY_TOL_MM = 1e-9  # symmetric head-on pushes must not break symmetry numerically
OFF = dict(audit.ASSISTS_OFF)

BIOLOGY = {
    "status": "not_assessed",
    "reason": ("Physics fixtures use scripted commands; behaviour fixtures use the hand-built modular "
               "controller with heuristic gains. Nothing here is compared with measured Drosophila "
               "wall-following, centrophobism or obstacle-negotiation data, and the engineered motor "
               "assists (wall_avoidance_reflex, contact_turn) are not fitted to fly physiology."),
}


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
class FixtureParadigm(ExperimentParadigm):
    """Stimulus-free enclosure: only walls and the containment region act."""

    def __init__(self, name: str, dims: Tuple[float, float], walls: Sequence[WallSegment] = ()):
        super().__init__(name=name, description="WP3 physics fixture", dimensions=dims, walls=list(walls))

    def step(self, fly: Any, dt: float = 1.0) -> Dict[str, Any]:
        self.time_elapsed_ms += dt * 1000.0
        return {}

    def sample_stimuli(self, x: Any, y: Optional[float] = None, heading: Optional[float] = None) -> Dict[str, Any]:
        return {"temperature": 25.0, "wind": (0.0, 0.0)}

    def reset_trial(self) -> Dict[str, Any]:
        return {}

    def get_metrics(self) -> Dict[str, Any]:
        return {}


def box(x0: float, y0: float, x1: float, y1: float) -> List[WallSegment]:
    return [WallSegment((x0, y0), (x1, y0)), WallSegment((x1, y0), (x1, y1)),
            WallSegment((x1, y1), (x0, y1)), WallSegment((x0, y1), (x0, y0))]


def corridor(width: float, length: float = 80.0, y0: float = 10.0) -> List[WallSegment]:
    return [WallSegment((0.0, y0), (length, y0)), WallSegment((length, y0 + width), (0.0, y0 + width))]


@dataclass
class Setup:
    arena: Arena
    region: Optional[audit.Region]      # independent legal region for the escape check
    walls: List[WallSegment]


def _place(arena: Arena, x: float, y: float, heading: float, speed: float = 0.0) -> None:
    f = arena.fly
    f.pos.x, f.pos.y, f.heading, f.speed, f.angular_velocity = x, y, heading, speed, 0.0


def walls_only(walls: List[WallSegment], pose, assists: bool, dims=(200.0, 200.0)) -> Setup:
    """Collision-engine walls only: the containment failsafe is moved far away."""
    arena = Arena(paradigm=FixtureParadigm("fixture_walls", dims, walls), seed=7, num_flies=1,
                  num_predators=0, motor_assists=None if assists else OFF)
    arena.containment = RectRegion(-1e4, -1e4, 1e4, 1e4)
    _place(arena, *pose)
    return Setup(arena, None, list(walls))


def failsafe_only(dims, pose, assists: bool) -> Setup:
    """No wall segments: only the containment failsafe holds the body."""
    arena = Arena(paradigm=FixtureParadigm("fixture_region", dims, []), seed=7, num_flies=1,
                  num_predators=0, motor_assists=None if assists else OFF)
    _place(arena, *pose)
    return Setup(arena, audit.Rect(0.0, 0.0, dims[0], dims[1]), [])


def real_paradigm(pid: str, pose, assists: bool, seed: int = 7) -> Setup:
    arena = audit.make_arena(pid, seed, assists=assists)
    if pose is not None:
        _place(arena, *pose)
    walls = list(getattr(arena.paradigm, "walls", []) or []) if arena.paradigm else []
    return Setup(arena, audit.legal_region(pid), walls)


# ---------------------------------------------------------------------------
# Scripted motor commands (known escape commands). Signature: (t, fly, sensory) -> (yaw, speed)
# ---------------------------------------------------------------------------
Command = Callable[[float, Any, Dict[str, float]], Tuple[float, float]]


def hold(speed: float, yaw: float = 0.0) -> Command:
    return lambda t, f, s: (yaw, speed)


def steer_to(target: float, speed: float) -> Command:
    """Turn in place toward ``target`` then walk; the canonical known escape command."""
    def cmd(t, f, s):
        err = wrap(target - f.heading)
        return max(-2.5, min(2.5, 4.0 * err)), (speed if abs(err) < ALIGN_TOL else 0.0)
    return cmd


def weave(amplitude: float, period: float, speed: float) -> Command:
    return lambda t, f, s: (amplitude * math.sin(2.0 * math.pi * t / period), speed)


def tropotaxis(gain: float, speed: float) -> Command:
    """Pure bilateral comparison: zero yaw whenever both antennae sense the same."""
    return lambda t, f, s: (gain * (s.get("left_a", 0.0) - s.get("right_a", 0.0)), speed)


# ---------------------------------------------------------------------------
# Runs and checks
# ---------------------------------------------------------------------------
@dataclass
class Row:
    t: float
    phase: str
    x: float
    y: float
    heading: float
    speed: float
    kind: str
    rec: Dict[str, Any]


@dataclass
class Run:
    fixture: "Fixture"
    assists: bool
    rows: List[Row]
    probe: audit.MotionProbe
    arena: Arena

    def phase(self, name: str) -> List[Row]:
        return [r for r in self.rows if r.phase == name]


@dataclass
class Check:
    label: str
    kind: str   # 'physics' (solver verdict), 'progress' (solver progress under a known command) or 'task'
    fn: Callable[[Run], Optional[str]]
    assists_off_only: bool = False


def physics_check() -> Check:
    def fn(run: Run) -> Optional[str]:
        rep = run.probe.containment_report()
        return None if rep["ok"] else f"containment: {rep}"
    return Check("no escape, unexplained jump or numerical trapping", "physics", fn)


def still_when_pushing(phase: str, min_s: float = 2.0) -> Check:
    """Head-on push: once pushing, the body rests at contact with no creep or jitter."""
    def fn(run: Run) -> Optional[str]:
        rows = [r for r in run.phase(phase) if r.kind == "wall_pushing"]
        if len(rows) * DT < min_s:
            return f"only {len(rows) * DT:.2f} s classified wall_pushing in '{phase}' (expected >= {min_s} s)"
        rows = rows[5:]   # the arrival step itself is partly realized
        creep = max(r.rec.get("realized_mm", 0.0) for r in rows)
        xs, ys = [r.x for r in rows], [r.y for r in rows]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        if creep > STILL_TOL_MM or span > STILL_TOL_MM:
            return f"creep {creep:.3g} mm/step, span {span:.3g} mm while pushing head-on"
        return None
    return Check(f"'{phase}': rests at contact (>= {min_s} s wall_pushing, creep <= {STILL_TOL_MM} mm)", "progress", fn)


def no_jitter(phase: str) -> Check:
    """No ping-pong at contact: consecutive realized steps never reverse direction."""
    def fn(run: Run) -> Optional[str]:
        rows = [r for r in run.phase(phase) if r.rec.get("in_contact")]
        flips = 0
        for a, b in zip(rows, rows[1:]):
            if b.rec["step"] != a.rec["step"] + 1:
                continue
            ma, mb = a.rec["realized_mm"], b.rec["realized_mm"]
            if ma > 1e-7 and mb > 1e-7 and a.rec["realized_dx"] * b.rec["realized_dx"] + a.rec["realized_dy"] * b.rec["realized_dy"] < 0:
                flips += 1
        return None if flips == 0 else f"{flips} direction reversals between consecutive contact steps"
    return Check(f"'{phase}': no contact jitter (no step-to-step reversals)", "physics", fn)


def progress_along(phase: str, direction: Tuple[float, float], ratio: float, label: str,
                   min_s: float = 1.0) -> Check:
    ux, uy = direction
    n = math.hypot(ux, uy)
    ux, uy = ux / n, uy / n

    def fn(run: Run) -> Optional[str]:
        rows = [r for r in run.phase(phase) if r.rec.get("attempted_mm", 0.0) > 0.0]
        if len(rows) * DT < min_s:
            return f"only {len(rows) * DT:.2f} s of attempted motion in '{phase}'"
        att = sum(r.rec["attempted_dx"] * ux + r.rec["attempted_dy"] * uy for r in rows)
        real = sum(r.rec["realized_dx"] * ux + r.rec["realized_dy"] * uy for r in rows)
        if att <= 1e-9:
            return f"attempted motion along ({ux:.2f},{uy:.2f}) is {att:.3g} mm"
        if real < ratio * att:
            return f"realized {real:.3f} of {att:.3f} mm attempted ({real / att:.2%} < {ratio:.0%})"
        return None
    return Check(f"'{phase}': {label} (realized >= {ratio:.0%} of attempted)", "progress", fn)


def ends_beyond(phase: str, predicate: Callable[[float, float], bool], label: str) -> Check:
    def fn(run: Run) -> Optional[str]:
        rows = run.phase(phase)
        return None if rows and predicate(rows[-1].x, rows[-1].y) else f"end of '{phase}' at ({rows[-1].x:.3f}, {rows[-1].y:.3f})"
    return Check(f"'{phase}': {label}", "progress", fn)


def symmetric_x(x0: float) -> Check:
    def fn(run: Run) -> Optional[str]:
        dev = max(abs(r.x - x0) for r in run.rows)
        return None if dev <= SYMMETRY_TOL_MM else f"x deviated {dev:.3g} mm from the symmetry axis x={x0}"
    return Check(f"symmetric head-on push stays on x={x0} (<= {SYMMETRY_TOL_MM} mm)", "physics", fn, assists_off_only=True)


def reverse_sign_kept(phase: str) -> Check:
    def fn(run: Run) -> Optional[str]:
        bad = [r for r in run.phase(phase) if r.speed > 0.0]
        return None if not bad else f"{len(bad)} steps with positive speed while reversing (first t={bad[0].t:.2f})"
    return Check(f"'{phase}': reverse speed keeps its sign through contact", "physics", fn)


def task_goal(label: str, predicate: Callable[[Run], bool]) -> Check:
    return Check(label, "task", lambda run: None if predicate(run) else "not achieved")


@dataclass
class Fixture:
    name: str
    category: str                   # 'physics' (scripted) or 'behaviour' (controller drives)
    constraint: str                 # predeclared physical constraint
    progress: str                   # predeclared progress / outcome criterion
    build: Callable[[bool], Setup]
    duration_s: float
    phases: Optional[List[Tuple[str, float, Command]]] = None
    checks: List[Check] = field(default_factory=list)


def run_fixture(fx: Fixture, assists: bool) -> Tuple[Run, Dict[str, Any]]:
    setup = fx.build(assists)
    arena, fly = setup.arena, setup.arena.fly
    probe = audit.MotionProbe(DT, region=setup.region, walls=setup.walls, radius=fly.radius)
    clock = {"t": 0.0, "phase": "brain"}

    if fx.phases:
        bounds, acc = [], 0.0
        for name, dur, cmd in fx.phases:
            acc += dur
            bounds.append((acc, name, cmd))

        def scripted(*args, **kwargs):
            f = kwargs.get("fly") or arena.fly
            sensory = kwargs.get("sensory", args[0] if args else {})
            for end, name, cmd in bounds:
                if clock["t"] < end - 1e-9:
                    break
            yaw, speed = cmd(clock["t"], f, sensory)
            return float(yaw), float(speed), "SCRIPTED", f.heading, f.heading
        arena.compute_steering = scripted  # type: ignore[assignment]

    rows: List[Row] = []
    steps = int(round(fx.duration_s / DT))
    for i in range(steps):
        t = i * DT
        clock["t"] = t
        if fx.phases:
            clock["phase"] = next(name for end, name, _ in bounds if t < end - 1e-9) if t < bounds[-1][0] - 1e-9 else bounds[-1][1]
        arena.step(DT)
        rec = dict(fly.motor_record)
        kind = probe.observe(rec, fly.pos.x, fly.pos.y)
        rows.append(Row(t, clock["phase"], fly.pos.x, fly.pos.y, fly.heading, fly.speed, kind, rec))

    run = Run(fx, assists, rows, probe, arena)
    results = []
    for chk in fx.checks:
        if chk.assists_off_only and assists:
            results.append(dict(label=chk.label, kind=chk.kind, passed=None, detail="applies with assists off"))
            continue
        err = chk.fn(run)
        results.append(dict(label=chk.label, kind=chk.kind, passed=err is None, detail=err))
    physics_pass = all(r["passed"] is not False for r in results if r["kind"] == "physics")
    progress_pass = all(r["passed"] is not False for r in results if r["kind"] == "progress")
    report = dict(
        fixture=fx.name, category=fx.category, assists="on" if assists else "off",
        constraint=fx.constraint, progress_criterion=fx.progress, sim_seconds=round(steps * DT, 3),
        arena_containment=dict(probe.containment_report(), solver_progress_pass=progress_pass if fx.category == "physics" else None),
        controller_outcome=dict(probe.controller_report(),
                                task_checks=[r for r in results if r["kind"] == "task"]),
        biological_validity=BIOLOGY,
        motor_provenance=arena.motor_provenance(),
        checks=results, physics_pass=physics_pass, progress_pass=progress_pass,
    )
    return run, report


# ---------------------------------------------------------------------------
# Fixture catalogue
# ---------------------------------------------------------------------------
HEAD_ON = "Rests at contact without creep, jitter or lateral drift while pushed; wall pushing is a behavioural classification. Known escape (turn to the open side, walk 1.5 mm/s) realizes >= 90 % of its aligned motion."


def _push_escape(push_heading: float, escape_heading: float, push_s: float = 10.0, esc_s: float = 5.0):
    return [("push", push_s, steer_to(push_heading, V)), ("escape", esc_s, steer_to(escape_heading, V)),
            ("repush", 10.0, steer_to(push_heading, V))]


def _escape_dir(h: float) -> Tuple[float, float]:
    return (math.cos(h), math.sin(h))


def physics_fixtures() -> List[Fixture]:
    fx: List[Fixture] = []

    def head_on(name, build, push_h, esc_h, constraint, extra=(), esc_s=5.0):
        fx.append(Fixture(name, "physics", constraint, HEAD_ON, build, 20.0 + esc_s, _push_escape(push_h, esc_h, esc_s=esc_s),
                          [physics_check(), still_when_pushing("push"), no_jitter("push"), no_jitter("repush"),
                           progress_along("repush", _escape_dir(push_h), 0.0, "returns to the wall", min_s=0.5),
                           progress_along("escape", _escape_dir(esc_h), ESCAPE_RATIO, "known escape moves away"), *extra]))

    head_on("head_on_wall_segment", lambda a: walls_only(box(0, 0, 40, 40), (33.0, 20.0, 0.0), a), 0.0, math.pi,
            "Wall segment x=40: body centre stays >= 1.5 mm from it (0.05 mm tolerance), no tunnelling.",
            [Check("head-on push keeps y=20 (no lateral drift)", "physics",
                                  lambda run: None if max(abs(r.y - 20.0) for r in run.phase("push")) <= SYMMETRY_TOL_MM
                                  else "lateral drift", assists_off_only=True)])
    head_on("head_on_containment_edge", lambda a: failsafe_only((80.0, 80.0), (74.0, 40.0, 0.0), a), 0.0, math.pi,
            "Containment rectangle 80x80 (no wall segments): failsafe keeps the body inside x <= 78.5.")
    head_on("concave_corner_walls", lambda a: walls_only(box(0, 0, 40, 40), (35.0, 35.0, math.pi / 4), a),
            math.pi / 4, 5 * math.pi / 4,
            "Two wall segments meeting at (40,40): body stays inside both (wedge solver).")
    head_on("concave_corner_containment", lambda a: failsafe_only((80.0, 80.0), (76.0, 76.0, math.pi / 4), a),
            math.pi / 4, 5 * math.pi / 4,
            "Containment rectangle corner (80,80): failsafe keeps the body inside both edges.")
    head_on("concave_corner_labyrinth_L", lambda a: real_paradigm("labyrinth", (30.0, 65.0, 3 * math.pi / 4), a),
            3 * math.pi / 4, -math.pi / 4,
            "Labyrinth walls 1 and 2 meet at (25,70) (x>25, y<70 side): no penetration of either.")
    head_on("circular_wall_courtship", lambda a: real_paradigm("courtship", (10.0, 14.0, math.pi / 2), a),
            math.pi / 2, -math.pi / 2,
            "Courtship chamber circle r=8.5 at (10,10): failsafe keeps |p-c| <= 7.0.")
    head_on("convex_endcap_head_on", lambda a: real_paradigm("labyrinth", (125.0, 76.0, -math.pi / 2), a),
            -math.pi / 2, math.pi / 2,
            "Free end (125,70) of labyrinth wall 9 hit exactly head-on: rounded end cap holds the body 1.5 mm off.",
            [Check("head-on end-cap push keeps x=125", "physics",
                   lambda run: None if max(abs(r.x - 125.0) for r in run.phase("push")) <= SYMMETRY_TOL_MM
                   else "lateral drift at the end cap", assists_off_only=True)], esc_s=4.0)

    fx.append(Fixture(
        "convex_endcap_slide", "physics",
        "Labyrinth wall 9 (x=125, y 20..70) and its free end: no penetration while sliding and rounding the end.",
        "Oblique walk (heading 95.7 deg, no yaw) slides along the wall with >= 50 % of the attempted tangential "
        "motion realized, passes the free end (y > 72) without a jump.",
        lambda a: real_paradigm("labyrinth", (126.6, 58.0, math.pi / 2 + 0.1), a), 16.0,
        [("slide", 16.0, hold(V))],
        [physics_check(), progress_along("slide", (0.0, 1.0), SLIDE_RATIO, "slides along +y"),
         ends_beyond("slide", lambda x, y: y > 72.0, "passes the free end (y > 72)")]))
    fx.append(Fixture(
        "convex_pillar_slide", "physics",
        "Multisensory pillar (35,35) r=6 (containment hole plus 8-segment wall polygon): no penetration.",
        "Walk tangentially with a slight inward heading and no yaw: >= 50 % of the attempted motion is realized "
        "overall and the body moves >= 10 mm along +y (slides round, not stuck on polygon vertices).",
        lambda a: real_paradigm("multisensory-sandbox", (27.4, 35.0, math.pi / 2 - 0.2), a), 16.0,
        [("slide", 16.0, hold(V))],
        [physics_check(), progress_along("slide", (0.0, 1.0), SLIDE_RATIO, "moves along +y around the pillar"),
         ends_beyond("slide", lambda x, y: y > 45.0, "gets >= 10 mm along +y")]))

    for name, build, constraint in (
        ("narrow_corridor_gap_crossing", lambda a: real_paradigm("gap-crossing", (15.0, 10.0, 0.0), a),
         "Gap-crossing track y in [7.5, 12.5] (2 mm of lateral freedom), containment failsafe only."),
        ("narrow_corridor_walls_3.2mm", lambda a: walls_only(corridor(3.2), (5.0, 11.6, 0.0), a),
         "Parallel wall segments 3.2 mm apart (0.2 mm clearance): no penetration of either."),
        ("narrow_corridor_walls_3.05mm", lambda a: walls_only(corridor(3.05), (5.0, 11.525, 0.0), a),
         "Parallel wall segments 3.05 mm apart: both walls inside the solver's contact band at once."),
    ):
        fx.append(Fixture(
            name, "physics", constraint,
            "Weaving walk (yaw 3 sin(2 pi t) rad/s, 1.5 mm/s): realized progress along the corridor is >= 50 % "
            "of the attempted axial motion, with no numerical trapping between the walls.",
            build, 20.0, [("weave", 20.0, weave(3.0, 1.0, V))],
            [physics_check(), progress_along("weave", (1.0, 0.0), SLIDE_RATIO, "progresses along the corridor")]))

    rev = [("reverse_push", 10.0, hold(-1.0)), ("forward_escape", 6.0, hold(1.0))]
    rev_checks = lambda: [physics_check(), still_when_pushing("reverse_push"), reverse_sign_kept("reverse_push"),
                          progress_along("forward_escape", (-1.0, 0.0), ESCAPE_RATIO, "walking forward leaves the wall")]
    fx.append(Fixture("reverse_into_wall_segment", "physics",
                      "Wall segment x=40 behind a fly facing -x and walking backwards: no penetration.",
                      "Backing into the wall rests at contact with the reverse sign kept; walking forward escapes (>= 90 %).",
                      lambda a: walls_only(box(0, 0, 40, 40), (35.0, 20.0, math.pi), a), 16.0, rev, rev_checks()))
    fx.append(Fixture("reverse_into_containment_edge", "physics",
                      "Containment edge x=80 behind a fly walking backwards: failsafe holds x <= 78.5.",
                      "As for the wall segment.",
                      lambda a: failsafe_only((80.0, 80.0), (74.0, 40.0, math.pi), a), 16.0, rev, rev_checks()))
    fx.append(Fixture("reverse_oblique_slide", "physics",
                      "Wall segment x=40 approached backwards at 17 deg: no penetration.",
                      "Backing obliquely slides along +y with >= 50 % of the attempted tangential motion; sign kept.",
                      lambda a: walls_only(box(0, 0, 40, 60), (38.0, 10.0, math.pi + 0.3), a), 16.0,
                      [("reverse_slide", 16.0, hold(-1.0))],
                      [physics_check(), reverse_sign_kept("reverse_slide"),
                       progress_along("reverse_slide", (0.0, 1.0), SLIDE_RATIO, "slides along +y")]))
    return fx


def _goal_reached(run: Run) -> bool:
    return bool((run.arena.paradigm.get_metrics() or {}).get("goal_reached"))


def behaviour_fixtures() -> List[Fixture]:
    goal = "Task (reported, not asserted): reach the labyrinth goal zone around (130,85) behind wall 10 within 30 s."
    return [
        Fixture("cue_behind_wall_modular", "behaviour",
                "Labyrinth wall 10 (y=80, x 100..140) separates the fly from the odour goal: no crossing or penetration.",
                goal, lambda a: real_paradigm("labyrinth", (128.0, 74.0, math.pi / 2), a), 30.0, None,
                [physics_check(), task_goal("reaches the goal zone", _goal_reached),
                 task_goal("no wall pushing longer than 2 s",
                           lambda run: run.probe.max_seconds("wall_pushing") <= 2.0)]),
        Fixture("equal_antenna_scripted_tropotaxis", "behaviour",
                "Wall 10 directly between the fly (x=130) and the goal (130,85): no penetration, no symmetry breaking by the solver.",
                "Controller: pure tropotaxis, zero yaw for equal antenna signals. Expected outcome without assists: "
                "sustained head-on wall pushing (a task failure, not a physics defect). Any escape with assists on "
                "must be credited to the logged wall_avoidance_reflex.",
                lambda a: real_paradigm("labyrinth", (130.0, 74.0, math.pi / 2), a), 20.0,
                [("tropotaxis", 20.0, tropotaxis(20.0, V))],
                [physics_check(), symmetric_x(130.0),
                 task_goal("leaves the wall (pushing < 50 % of the trial)",
                           lambda run: run.probe.seconds("wall_pushing") < 10.0)]),
        Fixture("equal_antenna_modular", "behaviour",
                "Same pose with the modular controller (odour symmetric about x=130).",
                goal, lambda a: real_paradigm("labyrinth", (130.0, 74.0, math.pi / 2), a), 30.0, None,
                [physics_check(), task_goal("reaches the goal zone", _goal_reached),
                 task_goal("no wall pushing longer than 2 s",
                           lambda run: run.probe.max_seconds("wall_pushing") <= 2.0)]),
    ]


# ---------------------------------------------------------------------------
# Restored trained checkpoints (always from copies)
# ---------------------------------------------------------------------------
def copy_brains(src: Path, dst: Path) -> Dict[str, str]:
    """Copy checkpoint JSONs (never the event ledgers) and return their sha256."""
    dst.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for f in sorted(Path(src).glob("*.json")):
        shutil.copy2(f, dst / f.name)
        hashes[f.name] = hashlib.sha256((dst / f.name).read_bytes()).hexdigest()
    return hashes


def run_restored(brain_dir: Path, paradigm: str, seconds: float, assists: bool) -> Dict[str, Any]:
    """Restore a trained brain from ``brain_dir`` (a copy) and run it at its spawn."""
    from experiment_brains import ExperimentBrain
    before = hashlib.sha256((brain_dir / f"{paradigm}.json").read_bytes()).hexdigest()
    brain = ExperimentBrain(paradigm, brain_dir)
    arena = brain.arena
    if not assists:
        arena.motor_assists.update(OFF)
    fly = arena.fly
    walls = list(getattr(arena.paradigm, "walls", []) or []) if arena.paradigm else []
    probe = audit.MotionProbe(DT, region=audit.legal_region(paradigm), walls=walls, radius=fly.radius)
    steps = int(round(seconds / DT))
    for _ in range(steps):
        arena.step(DT)
        probe.observe(dict(fly.motor_record), fly.pos.x, fly.pos.y)
    after = hashlib.sha256((brain_dir / f"{paradigm}.json").read_bytes()).hexdigest()
    metrics = arena.paradigm.get_metrics() if arena.paradigm else {}
    return dict(
        fixture=f"restored_checkpoint:{paradigm}", category="restored_checkpoint",
        assists="on" if assists else "off", paradigm=paradigm, restored=brain.restored,
        brain_steps=brain.steps, brain_trials=brain.trials, learning_enabled=brain.learning_enabled,
        checkpoint_sha256=before, checkpoint_unchanged=before == after, sim_seconds=round(steps * DT, 3),
        constraint="Paradigm legal region and wall segments (containment_audit.legal_region).",
        progress_criterion="Physics: no escape, jump or numerical trapping. Task outcome reported from paradigm metrics.",
        arena_containment=probe.containment_report(),
        controller_outcome=dict(probe.controller_report(), paradigm_metrics=_jsonable(metrics)),
        biological_validity=BIOLOGY, motor_provenance=arena.motor_provenance(),
        physics_pass=probe.physics_ok,
    )


def _jsonable(obj: Any) -> Any:
    try:
        return json.loads(json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    except (TypeError, ValueError):
        return str(obj)


# ---------------------------------------------------------------------------
# Receipt writer
# ---------------------------------------------------------------------------
def _row_md(r: Dict[str, Any]) -> str:
    c, o = r["arena_containment"], r["controller_outcome"]
    t = o.get("time_s", {})
    prov = r["motor_provenance"]["assist_totals"]
    solver = r.get("progress_pass")
    solver_txt = "-" if r.get("category") != "physics" else ("PASS" if solver else "FAIL")
    return (f"| {r['fixture']} | {r['assists']} | {r['sim_seconds']:.0f} | {'PASS' if r['physics_pass'] else 'FAIL'} "
            f"| {c['escapes']}/{c['unexplained_jumps']}/{c['max_numerical_trapping_s']:.2f} | {solver_txt} "
            f"| {o['net_displacement_mm']:.1f} | {o.get('realized_fraction') or 0:.2f} | {t.get('wall_pushing', 0):.1f} "
            f"| {o['max_continuous_s'].get('wall_pushing', 0):.1f} | {sum(t.get(k, 0) for k in ('rest', 'feeding', 'gap_probing', 'courtship_rest', 'tethered')):.1f} "
            f"| {o['near_wall_s']:.1f} | {sum(o['stall_episodes'].values())} "
            f"| {prov['wall_reflex_steps']}/{prov['contact_turn_events']} |")


MD_HEAD = ("| fixture | assists | sim s | containment | esc/jump/trap s | solver progress | net mm | realized frac "
           "| push s | max push s | rest s | near-wall s | stalls | reflex/turn steps |\n|" + "---|" * 14)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "wp3-wall-progress")
    ap.add_argument("--retained-brains", type=Path, default=None,
                    help="directory of COPIED checkpoint JSONs (copied again into --out; never written)")
    ap.add_argument("--restored-seconds", type=float, default=60.0)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    reports: List[Dict[str, Any]] = []
    for fx in physics_fixtures() + behaviour_fixtures():
        for assists in (False, True):
            _, rep = run_fixture(fx, assists)
            reports.append(rep)
            print(f"{fx.name:36s} assists={rep['assists']:3s} physics={'PASS' if rep['physics_pass'] else 'FAIL'} "
                  f"progress={'PASS' if rep['progress_pass'] else 'FAIL'}", flush=True)
    brain_hashes = {}
    if args.retained_brains:
        work = args.out / "retained-brain-copies"
        brain_hashes = copy_brains(args.retained_brains, work)
        for f in sorted(work.glob("*.json")):
            for assists in (True, False):
                # Fresh copy per run so a run can never see another's writes.
                run_dir = args.out / "runs" / f"{f.stem}-{'on' if assists else 'off'}"
                run_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, run_dir / f.name)
                rep = run_restored(run_dir, f.stem, args.restored_seconds, assists)
                reports.append(rep)
                print(f"restored {f.stem:28s} assists={rep['assists']:3s} physics={'PASS' if rep['physics_pass'] else 'FAIL'} "
                      f"push={rep['controller_outcome']['time_s'].get('wall_pushing', 0):.1f}s", flush=True)
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        rev, dirty = "unknown", None
    receipt = dict(generated_at=time.time(), revision=rev, working_tree_dirty=dirty, dt=DT,
                   thresholds=dict(escape_ratio=ESCAPE_RATIO, slide_ratio=SLIDE_RATIO, still_tol_mm=STILL_TOL_MM,
                                   symmetry_tol_mm=SYMMETRY_TOL_MM, stall_window_s=audit.STALL_WINDOW_S,
                                   stall_net_mm=audit.STALL_NET_MM, absorbed_ratio=audit.ABSORBED_RATIO,
                                   numerical_trap_limit_s=audit.NUMERICAL_TRAP_LIMIT_S,
                                   jump_slack_mm=audit.JUMP_SLACK_MM),
                   retained_brain_sha256=brain_hashes, reports=reports)
    (args.out / "wall_progress_receipt.json").write_text(json.dumps(_jsonable(receipt), indent=2, allow_nan=False))
    lines = ["# WP3 wall-progress fixtures", "",
             f"Revision `{rev}` (dirty working tree: {dirty}); dt = {DT} s. Containment = arena physics; "
             "solver progress = known escape command outcome; the remaining columns are controller outcomes; "
             "biological validity: not assessed.", "", MD_HEAD]
    lines += [_row_md(r) for r in reports]
    (args.out / "wall_progress_table.md").write_text("\n".join(lines) + "\n")
    bad_physics = [r["fixture"] + "/" + r["assists"] for r in reports if not r["physics_pass"]]
    bad_solver = [r["fixture"] for r in reports if r.get("category") == "physics" and r["assists"] == "off" and not r["progress_pass"]]
    print(f"\nwrote {args.out / 'wall_progress_receipt.json'} and wall_progress_table.md")
    print(f"containment failures: {bad_physics or 'none'}; solver-progress failures (assists off): {bad_solver or 'none'}")
    return 1 if bad_physics or bad_solver else 0


if __name__ == "__main__":
    sys.exit(main())
