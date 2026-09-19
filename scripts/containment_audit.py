#!/usr/bin/env python3
"""Containment audit: script-controlled physics harness for every NeuroFly paradigm.

Steps the headless simulation (the same ``Arena.step(dt)`` loop the 24/7 daemon
runs) for every paradigm the daemon exposes, over many seeds, start positions and
headings, against an *independent* description of each paradigm's legal region
(defined in this file, not read back from ``arena.py``). Two verdicts are kept apart
(decision contract item 4: physics constrains, it does not steer toward success).

Arena containment (physics; any failure is a solver defect):
  (a) escape   - the fly body is never outside the paradigm's legal region and never
                 penetrates or tunnels through a wall segment;
  (b) teleport - no per-step displacement larger than the physical maximum, and no
                 step longer than the attempted motor step (``unexplained_jump``);
  (e) trapping - attempted motion is never absorbed without a touching boundary, or
                 while it points away from every touching boundary, for longer than
                 ``NUMERICAL_TRAP_LIMIT_S`` (``numerical_trapping``).

Controller outcome (behaviour; reported, never a physics verdict):
  (c) pinned   - in contact with a boundary while pushing into it (commanded motion
                 absorbed, or travel direction pointing into the wall), longest run;
  (d) stuck    - clamped at one coordinate while commanded motion is absorbed;
  plus wall-pushing time, rest/tethered/feeding/gap-probing time, time near walls,
  rolling progress and repeated stall episodes (see ``MotionProbe``).

Usage:
    ./.venv/bin/python scripts/containment_audit.py               # full audit, default assists
    ./.venv/bin/python scripts/containment_audit.py --no-assists  # engineered assists disabled
    ./.venv/bin/python scripts/containment_audit.py --quick       # short smoke run
    ./.venv/bin/python scripts/containment_audit.py --paradigm multisensory-sandbox

Exit status is 1 when any paradigm fails arena containment (add ``--strict-behaviour``
to also fail on wall pushing). The pytest wrapper
``tests/test_containment_all_paradigms.py`` runs a reduced matrix.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arena import Arena  # noqa: E402
from maze import WallSegment  # noqa: E402


# ---------------------------------------------------------------------------
# Paradigm catalogue: the 14 ids the daemon's /api/paradigms endpoint advertises.
# ---------------------------------------------------------------------------
DAEMON_PARADIGMS: List[str] = [
    "open-arena",
    "t-maze",
    "y-maze",
    "heat-maze",
    "buridan",
    "visual-operant",
    "wind-tunnel",
    "looming-escape",
    "optomotor",
    "gap-crossing",
    "circadian-dam",
    "courtship",
    "labyrinth",
    "multisensory-sandbox",
]

FLY_RADIUS = 1.5
# Fastest commanded ground speed in the modular brain is the 3.5 mm/s LC4 escape sprint.
MAX_FLY_SPEED_MM_S = 3.5
# A step is a teleport if the body moves further than this in one tick. The collision
# solver may legitimately push the body out of a wall by a fraction of a millimetre,
# so the threshold is a small absolute distance on top of the physical maximum.
TELEPORT_SLACK_MM = 0.25
CONTACT_GAP_MM = 0.05        # body edge within this distance of a boundary = in contact
WALL_TOLERANCE_MM = 0.05     # allowed numerical penetration of a wall surface
ABSORBED_FRACTION = 0.5      # commanded motion absorbed by the boundary -> pushing into it
APPROACH_COS = 0.3           # cos(angle between travel and the into-wall direction) that counts as pushing


# ---------------------------------------------------------------------------
# Independent legal-region geometry
# ---------------------------------------------------------------------------
class Region:
    """A closed 2D region. ``gap`` is the signed distance from a point to the boundary
    (positive inside), so the fly body of radius r is legal when ``gap >= r``."""

    def gap(self, x: float, y: float) -> float:
        raise NotImplementedError

    def normal(self, x: float, y: float) -> Tuple[float, float]:
        """Unit vector pointing from the nearest boundary into the region."""
        raise NotImplementedError

    def bbox(self) -> Tuple[float, float, float, float]:
        raise NotImplementedError

    def sample(self, rng: random.Random, margin: float) -> Tuple[float, float]:
        xmin, ymin, xmax, ymax = self.bbox()
        for _ in range(10000):
            x = rng.uniform(xmin, xmax)
            y = rng.uniform(ymin, ymax)
            if self.gap(x, y) >= margin:
                return x, y
        raise RuntimeError("could not sample a legal start position")


class Rect(Region):
    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float):
        self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax

    def gap(self, x: float, y: float) -> float:
        return min(x - self.xmin, self.xmax - x, y - self.ymin, self.ymax - y)

    def normal(self, x: float, y: float) -> Tuple[float, float]:
        sides = [(x - self.xmin, (1.0, 0.0)), (self.xmax - x, (-1.0, 0.0)),
                 (y - self.ymin, (0.0, 1.0)), (self.ymax - y, (0.0, -1.0))]
        return min(sides, key=lambda s: s[0])[1]

    def bbox(self):
        return (self.xmin, self.ymin, self.xmax, self.ymax)


class Circle(Region):
    def __init__(self, cx: float, cy: float, r: float):
        self.cx, self.cy, self.r = cx, cy, r

    def gap(self, x: float, y: float) -> float:
        return self.r - math.hypot(x - self.cx, y - self.cy)

    def normal(self, x: float, y: float) -> Tuple[float, float]:
        dx, dy = self.cx - x, self.cy - y
        d = math.hypot(dx, dy)
        return (dx / d, dy / d) if d > 1e-9 else (1.0, 0.0)

    def bbox(self):
        return (self.cx - self.r, self.cy - self.r, self.cx + self.r, self.cy + self.r)


class OrientedRect(Region):
    """Rectangle running from (x0, y0) along ``angle`` for ``length`` with half-width ``half_w``."""

    def __init__(self, x0: float, y0: float, angle: float, length: float, half_w: float):
        self.x0, self.y0, self.length, self.half_w = x0, y0, length, half_w
        self.ux, self.uy = math.cos(angle), math.sin(angle)

    def gap(self, x: float, y: float) -> float:
        dx, dy = x - self.x0, y - self.y0
        u = dx * self.ux + dy * self.uy
        v = -dx * self.uy + dy * self.ux
        return min(u, self.length - u, self.half_w - abs(v))

    def normal(self, x: float, y: float) -> Tuple[float, float]:
        dx, dy = x - self.x0, y - self.y0
        u = dx * self.ux + dy * self.uy
        v = -dx * self.uy + dy * self.ux
        sides = [(u, (self.ux, self.uy)), (self.length - u, (-self.ux, -self.uy)),
                 (self.half_w - v, (self.uy, -self.ux)), (self.half_w + v, (-self.uy, self.ux))]
        return min(sides, key=lambda s: s[0])[1]

    def bbox(self):
        pts = [(self.x0 + self.ux * u - self.uy * v, self.y0 + self.uy * u + self.ux * v)
               for u in (0.0, self.length) for v in (-self.half_w, self.half_w)]
        return (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))


class Union(Region):
    def __init__(self, members: Sequence[Region]):
        self.members = list(members)

    def gap(self, x: float, y: float) -> float:
        return max(m.gap(x, y) for m in self.members)

    def normal(self, x: float, y: float) -> Tuple[float, float]:
        return max(self.members, key=lambda m: m.gap(x, y)).normal(x, y)

    def bbox(self):
        boxes = [m.bbox() for m in self.members]
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))


class Difference(Region):
    """``outer`` minus a list of circular holes."""

    def __init__(self, outer: Region, holes: Sequence[Circle]):
        self.outer = outer
        self.holes = list(holes)

    def gap(self, x: float, y: float) -> float:
        g = self.outer.gap(x, y)
        for h in self.holes:
            g = min(g, math.hypot(x - h.cx, y - h.cy) - h.r)
        return g

    def normal(self, x: float, y: float) -> Tuple[float, float]:
        best_g, best_n = self.outer.gap(x, y), self.outer.normal(x, y)
        for h in self.holes:
            dx, dy = x - h.cx, y - h.cy
            d = math.hypot(dx, dy)
            if d - h.r < best_g:
                best_g, best_n = d - h.r, ((dx / d, dy / d) if d > 1e-9 else (1.0, 0.0))
        return best_n

    def bbox(self):
        return self.outer.bbox()


def legal_region(paradigm_id: str) -> Region:
    """Legal body-centre region for each daemon paradigm, matching the geometry the
    dashboard (web/app.js) draws so that Python and JS agree on what 'inside' means."""
    pid = paradigm_id
    if pid == "open-arena":
        return Rect(0.0, 0.0, 100.0, 100.0)
    if pid == "t-maze":
        return Union([Rect(63.0, 10.0, 77.0, 57.0), Rect(10.0, 43.0, 130.0, 57.0)])
    if pid == "y-maze":
        return Circle(60.0, 60.0, 48.0)          # arms live inside this; walls are the truth
    if pid == "heat-maze":
        return Circle(60.0, 60.0, 55.0)
    if pid == "buridan":
        return Circle(60.0, 60.0, 50.0)          # platform; beyond it is the water moat
    if pid == "visual-operant":
        return Rect(0.0, 0.0, 80.0, 80.0)
    if pid == "wind-tunnel":
        return Rect(0.0, 0.0, 200.0, 60.0)
    if pid == "looming-escape":
        return Rect(0.0, 0.0, 80.0, 80.0)
    if pid == "optomotor":
        return Rect(0.0, 0.0, 90.0, 90.0)
    if pid == "gap-crossing":
        return Rect(5.0, 7.5, 95.0, 12.5)
    if pid == "circadian-dam":
        return Rect(5.0, 1.0, 60.0, 9.0)
    if pid == "courtship":
        return Circle(10.0, 10.0, 8.5)
    if pid == "labyrinth":
        return Rect(0.0, 0.0, 140.0, 100.0)
    if pid == "multisensory-sandbox":
        pillars = [Circle(35.0, 35.0, 6.0), Circle(-35.0, 35.0, 6.0),
                   Circle(-35.0, -35.0, 6.0), Circle(35.0, -35.0, 6.0)]
        return Difference(Circle(0.0, 0.0, 75.0), pillars)
    raise KeyError(paradigm_id)


def spawn_region(paradigm_id: str) -> Region:
    """Where random start poses are drawn from. Identical to the legal region except
    for the Y-maze, whose legal outer circle also covers the space between the arms;
    starts must be inside the three arms or the hub."""
    if paradigm_id == "y-maze":
        arms = [OrientedRect(60.0, 60.0, a, 40.0, 6.0)
                for a in (math.pi / 2.0, 7.0 * math.pi / 6.0, 11.0 * math.pi / 6.0)]
        return Union(arms + [Circle(60.0, 60.0, 10.0)])
    return legal_region(paradigm_id)


ASSISTS_OFF = {name: False for name in Arena.MOTOR_ASSISTS}


def make_arena(paradigm_id: str, seed: int, assists: bool = True) -> Arena:
    motor_assists = None if assists else dict(ASSISTS_OFF)
    if paradigm_id == "open-arena":
        return Arena(paradigm=None, seed=seed, num_predators=0, motor_assists=motor_assists)
    return Arena(paradigm=paradigm_id, brain_type="modular", num_flies=1, num_predators=0, seed=seed,
                 motor_assists=motor_assists)


# ---------------------------------------------------------------------------
# Motion probe: per-step instrumentation that separates physics from behaviour
# ---------------------------------------------------------------------------
# Thresholds are declared here, before any fixture runs, and are not tuned to results.
STALL_WINDOW_S = 1.0        # rolling window for net progress
STALL_NET_MM = 0.2          # net displacement over the window below this = stalled
ABSORBED_RATIO = 0.25       # realized < 25 % of the attempted step = motion absorbed
REST_SPEED_MM_S = 0.05      # |commanded speed| below this is an intentional rest
JUMP_SLACK_MM = 0.001       # realized step may exceed the attempted step by this much (solver skin 1e-4 mm)
NUMERICAL_TRAP_LIMIT_S = 0.2  # absorbed escape-direction motion tolerated this long
STATE_KINDS = {"FEED": "feeding", "PROBE": "gap_probing", "COURTSHIP": "courtship_rest"}

# Step kinds. Physics defects: numerical_trapping, unexplained_jump, escape.
# Everything else is a controller/task outcome, reported but never a physics failure.
PHYSICS_DEFECT_KINDS = ("numerical_trapping", "unexplained_jump")


def classify_step(rec: Dict, dt: float) -> str:
    """Classify one ``fly.motor_record`` (see ``Arena._finish_motor_record``).

    * tethered / feeding / gap_probing / courtship_rest / rest: intentional immobility;
    * free, sliding: the attempted step was (mostly) realized;
    * wall_pushing: motion absorbed while the attempted step points into a boundary
      the body touches - a behavioural outcome, not a solver defect;
    * numerical_trapping: motion absorbed with no touching boundary, or while the
      attempt points away from every touching boundary - a solver defect;
    * unexplained_jump: the body moved further than it attempted - a solver defect.
    """
    attempted = float(rec.get("attempted_mm", 0.0))
    realized = float(rec.get("realized_mm", 0.0))
    # A pose set from outside (reset, restore, test) that overlaps geometry is pushed
    # out by the static overlap check; that correction is logged and explained.
    overlap = float(rec.get("overlap_correction_mm", 0.0))
    if rec.get("tethered"):
        return "unexplained_jump" if realized > overlap + 1e-9 else "tethered"
    if realized > attempted + overlap + JUMP_SLACK_MM:
        return "unexplained_jump"
    kind = STATE_KINDS.get(str(rec.get("state", "")))
    if kind:
        return kind
    if abs(float(rec.get("controller_speed", 0.0))) < REST_SPEED_MM_S or attempted < 1e-9:
        return "rest"
    normals = rec.get("contact_normals") or []
    in_contact = bool(rec.get("in_contact")) or bool(normals)
    if realized >= ABSORBED_RATIO * attempted:
        return "sliding" if in_contact else "free"
    if not normals and rec.get("in_contact"):
        normals = [tuple(rec.get("wall_normal", (0.0, 0.0)))]
    ax, ay = float(rec.get("attempted_dx", 0.0)), float(rec.get("attempted_dy", 0.0))
    into = any(ax * n[0] + ay * n[1] < -1e-6 * max(attempted, 1e-9) for n in normals)
    if in_contact and into:
        return "wall_pushing"
    return "numerical_trapping"


@dataclass
class MotionProbe:
    """Accumulates motor records into separate containment / controller summaries.

    ``region`` and ``walls`` are an independent legal-region description (this file's
    ``legal_region``) used for the escape check; pass ``region=None`` to skip it.
    """
    dt: float
    region: Optional[Region] = None
    walls: List[WallSegment] = field(default_factory=list)
    radius: float = FLY_RADIUS
    kinds: Dict[str, int] = field(default_factory=dict)
    escapes: int = 0
    first_defect: Optional[str] = None
    max_step_mm: float = 0.0
    path_mm: float = 0.0
    attempted_path_mm: float = 0.0
    near_wall_steps: int = 0
    contact_steps: int = 0
    overlap_corrections: int = 0
    max_run: Dict[str, int] = field(default_factory=dict)
    stall_episodes: Dict[str, int] = field(default_factory=dict)
    min_rolling_progress_mm: Optional[float] = None
    steps: int = 0
    _run_kind: Optional[str] = None
    _run_len: int = 0
    _positions: List[Tuple[float, float]] = field(default_factory=list)
    _window_kinds: List[str] = field(default_factory=list)
    _stalled: bool = False
    start: Optional[Tuple[float, float]] = None
    end: Optional[Tuple[float, float]] = None

    def observe(self, rec: Dict, x: float, y: float) -> str:
        if self.start is None:
            self.start = (x - float(rec.get("realized_dx", 0.0)), y - float(rec.get("realized_dy", 0.0)))
            self._positions.append(self.start)
        self.steps += 1
        kind = classify_step(rec, self.dt)
        self.kinds[kind] = self.kinds.get(kind, 0) + 1
        realized = float(rec.get("realized_mm", 0.0))
        self.path_mm += realized
        self.attempted_path_mm += float(rec.get("attempted_mm", 0.0))
        self.max_step_mm = max(self.max_step_mm, realized)
        self.near_wall_steps += int(bool(rec.get("near_wall")))
        self.contact_steps += int(bool(rec.get("in_contact")))
        self.overlap_corrections += int(float(rec.get("overlap_correction_mm", 0.0)) > 0.0)

        if self.region is not None:
            outside = self.region.gap(x, y) < self.radius - WALL_TOLERANCE_MM
            penetrated = any(w.distance_to_point(x, y) < self.radius - WALL_TOLERANCE_MM for w in self.walls)
            if outside or penetrated:
                self.escapes += 1
                self._note(f"step {rec.get('step')}: escape at ({x:.3f}, {y:.3f})")
        if kind in PHYSICS_DEFECT_KINDS:
            self._note(f"step {rec.get('step')}: {kind} attempted={rec.get('attempted_mm', 0.0):.4f} "
                       f"realized={realized:.4f} at ({x:.3f}, {y:.3f})")

        # Longest continuous run of each kind
        if kind == self._run_kind:
            self._run_len += 1
        else:
            self._run_kind, self._run_len = kind, 1
        self.max_run[kind] = max(self.max_run.get(kind, 0), self._run_len)

        # Rolling progress and stall episodes
        window = max(1, int(round(STALL_WINDOW_S / self.dt)))
        self._positions.append((x, y))
        self._window_kinds.append(kind)
        if len(self._positions) > window + 1:
            self._positions.pop(0)
            self._window_kinds.pop(0)
        if len(self._positions) == window + 1:
            net = math.dist(self._positions[0], self._positions[-1])
            self.min_rolling_progress_mm = net if self.min_rolling_progress_mm is None else min(self.min_rolling_progress_mm, net)
            moving = [k for k in self._window_kinds if k not in ("rest", "tethered", "feeding", "gap_probing", "courtship_rest")]
            stalled = net < STALL_NET_MM and len(moving) == len(self._window_kinds)
            if stalled and not self._stalled:
                cause = max(set(moving), key=moving.count)
                if cause in ("free", "sliding"):
                    cause = "low_net_progress"   # moving but circling/oscillating in place
                self.stall_episodes[cause] = self.stall_episodes.get(cause, 0) + 1
            self._stalled = stalled
        self.end = (x, y)
        return kind

    def _note(self, msg: str) -> None:
        if self.first_defect is None:
            self.first_defect = msg

    def seconds(self, kind: str) -> float:
        return self.kinds.get(kind, 0) * self.dt

    def max_seconds(self, kind: str) -> float:
        return self.max_run.get(kind, 0) * self.dt

    @property
    def physics_ok(self) -> bool:
        return (self.escapes == 0 and self.kinds.get("unexplained_jump", 0) == 0
                and self.max_seconds("numerical_trapping") <= NUMERICAL_TRAP_LIMIT_S)

    def containment_report(self) -> Dict:
        return {
            "ok": self.physics_ok,
            "escapes": self.escapes,
            "unexplained_jumps": self.kinds.get("unexplained_jump", 0),
            "numerical_trapping_steps": self.kinds.get("numerical_trapping", 0),
            "max_numerical_trapping_s": round(self.max_seconds("numerical_trapping"), 3),
            "max_step_mm": round(self.max_step_mm, 5),
            "overlap_corrections": self.overlap_corrections,
            "first_defect": self.first_defect,
        }

    def controller_report(self) -> Dict:
        net = math.dist(self.start, self.end) if self.start and self.end else 0.0
        return {
            "sim_seconds": round(self.steps * self.dt, 3),
            "path_mm": round(self.path_mm, 3),
            "attempted_path_mm": round(self.attempted_path_mm, 3),
            "realized_fraction": round(self.path_mm / self.attempted_path_mm, 4) if self.attempted_path_mm > 0 else None,
            "net_displacement_mm": round(net, 3),
            "min_rolling_progress_mm_per_s": None if self.min_rolling_progress_mm is None else round(self.min_rolling_progress_mm / STALL_WINDOW_S, 4),
            "time_s": {k: round(v * self.dt, 3) for k, v in sorted(self.kinds.items())},
            "max_continuous_s": {k: round(v * self.dt, 3) for k, v in sorted(self.max_run.items())},
            "near_wall_s": round(self.near_wall_steps * self.dt, 3),
            "contact_s": round(self.contact_steps * self.dt, 3),
            "stall_episodes": dict(sorted(self.stall_episodes.items())),
        }


def _segments_cross(ax, ay, bx, by, cx, cy, dx, dy) -> bool:
    """Proper intersection test between segments AB and CD."""
    def orient(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)
    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay)
    o4 = orient(cx, cy, dx, dy, bx, by)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


# ---------------------------------------------------------------------------
# Per-run bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    paradigm: str
    seed: int
    start: Tuple[float, float, float]
    steps: int
    dt: float
    escapes: int = 0
    wall_penetrations: int = 0
    wall_crossings: int = 0
    teleports: int = 0
    max_step_mm: float = 0.0
    contact_steps: int = 0
    max_pinned_s: float = 0.0
    max_stuck_coord_s: float = 0.0
    path_length_mm: float = 0.0
    first_escape: Optional[str] = None
    first_teleport: Optional[str] = None
    first_pinned: Optional[str] = None
    # Trace for post-mortem: (step, x, y, heading, commanded speed, displacement)
    trace: List[Tuple[int, float, float, float, float, float]] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        # Physics only. Wall pushing is reported (pushing_exceeded) but is a controller
        # outcome; see the decision contract, item 4.
        return self.physics_failed

    pinned_failed: bool = False
    stuck_failed: bool = False
    probe: Optional["MotionProbe"] = None
    assists: bool = True

    @property
    def physics_failed(self) -> bool:
        """Arena containment defects only: escape, unexplained jump, numerical trapping."""
        return (self.escapes > 0 or self.teleports > 0
                or (self.probe is not None and not self.probe.physics_ok))

    @property
    def pushing_exceeded(self) -> bool:
        """Controller outcome: pushed into a boundary longer than the limit. Under the
        decision contract (item 4) this is a task outcome, not a physics defect."""
        return self.pinned_failed or self.stuck_failed


def run_case(
    paradigm_id: str,
    seed: int,
    start: Tuple[float, float, float],
    steps: int,
    dt: float,
    pinned_seconds: float,
    keep_trace: bool = False,
    assists: bool = True,
) -> RunResult:
    arena = make_arena(paradigm_id, seed, assists=assists)
    region = legal_region(paradigm_id)
    walls: List[WallSegment] = list(getattr(arena.paradigm, "walls", []) or []) if arena.paradigm else []
    fly = arena.fly
    r = getattr(fly, "radius", FLY_RADIUS)

    fly.pos.x, fly.pos.y, fly.heading = float(start[0]), float(start[1]), float(start[2])
    fly.speed = 1.2
    fly.angular_velocity = 0.0

    # Capture the brain's commanded speed each step (before collision resolution).
    commanded = {"v": 0.0, "backward": False}
    original_steering = arena.compute_steering

    def tapped_steering(*args, **kwargs):
        out = original_steering(*args, **kwargs)
        commanded["v"] = abs(float(out[1]))
        commanded["backward"] = float(out[1]) < 0.0
        return out

    arena.compute_steering = tapped_steering  # type: ignore[assignment]

    res = RunResult(paradigm_id, seed, start, steps, dt, assists=assists)
    res.probe = MotionProbe(dt, region=region, walls=walls, radius=r)
    teleport_mm = MAX_FLY_SPEED_MM_S * dt + TELEPORT_SLACK_MM
    pinned_run = 0
    stuck_run = 0
    prev_x, prev_y = fly.pos.x, fly.pos.y

    def nearest_boundary(x: float, y: float) -> Tuple[float, float, float]:
        """(distance, nx, ny) of the closest boundary; the normal points away from it."""
        g = region.gap(x, y)
        nx, ny = region.normal(x, y)
        for w in walls:
            px, py, _ = w.project_point(x, y)
            d = math.hypot(x - px, y - py)
            if d < g:
                g = d
                nx, ny = ((x - px) / d, (y - py) / d) if d > 1e-9 else (w.nx, w.ny)
        return g, nx, ny

    for i in range(steps):
        arena.step(dt)
        x, y = fly.pos.x, fly.pos.y
        res.probe.observe(fly.motor_record, x, y)
        disp = math.hypot(x - prev_x, y - prev_y)
        res.path_length_mm += disp
        res.max_step_mm = max(res.max_step_mm, disp)

        # (a) legal region and walls
        outside = region.gap(x, y) < r - WALL_TOLERANCE_MM
        penetrated = any(w.distance_to_point(x, y) < r - WALL_TOLERANCE_MM for w in walls)
        crossed = any(_segments_cross(prev_x, prev_y, x, y, w.p1[0], w.p1[1], w.p2[0], w.p2[1]) for w in walls)
        if outside or penetrated or crossed:
            res.escapes += 1
            res.wall_penetrations += int(penetrated)
            res.wall_crossings += int(crossed)
            if res.first_escape is None:
                why = "outside region" if outside else ("wall penetration" if penetrated else "wall crossing")
                res.first_escape = f"step {i}: {why} at ({x:.2f}, {y:.2f}) gap={region.gap(x, y):.2f}"

        # (b) teleport
        if disp > teleport_mm:
            res.teleports += 1
            if res.first_teleport is None:
                res.first_teleport = f"step {i}: {disp:.3f} mm from ({prev_x:.2f}, {prev_y:.2f}) to ({x:.2f}, {y:.2f})"

        # (c) pinned: in contact and pushing into the boundary (motion absorbed, or the
        #     direction of travel points into the wall)
        dist, nx, ny = nearest_boundary(x, y)
        gap = dist - r
        in_contact = gap <= CONTACT_GAP_MM
        if in_contact:
            res.contact_steps += 1
        intended = commanded["v"] * dt
        travel = fly.heading + (math.pi if commanded["backward"] else 0.0)
        approach = -(math.cos(travel) * nx + math.sin(travel) * ny)
        absorbed = in_contact and intended > 1e-6 and disp < (1.0 - ABSORBED_FRACTION) * intended
        pushing = absorbed or (in_contact and intended > 1e-6 and approach > APPROACH_COS)
        if pushing:
            pinned_run += 1
            res.max_pinned_s = max(res.max_pinned_s, pinned_run * dt)
            if pinned_run * dt > pinned_seconds and res.first_pinned is None:
                res.first_pinned = f"step {i}: at ({x:.2f}, {y:.2f}) heading {fly.heading:.2f} cmd {commanded['v']:.2f} mm/s"
        else:
            pinned_run = 0

        # (d) clamped at one coordinate while the boundary absorbs the commanded motion
        same_coord = (x == prev_x) != (y == prev_y)  # exactly one axis frozen
        if same_coord and absorbed:
            stuck_run += 1
            res.max_stuck_coord_s = max(res.max_stuck_coord_s, stuck_run * dt)
        else:
            stuck_run = 0

        if keep_trace:
            res.trace.append((i, x, y, fly.heading, commanded["v"], disp))
        prev_x, prev_y = x, y

    res.pinned_failed = res.max_pinned_s > pinned_seconds
    res.stuck_failed = res.max_stuck_coord_s > pinned_seconds
    return res


# ---------------------------------------------------------------------------
# Matrix driver
# ---------------------------------------------------------------------------
def default_spawn(paradigm_id: str, seed: int) -> Tuple[float, float, float]:
    arena = make_arena(paradigm_id, seed)
    return (arena.fly.pos.x, arena.fly.pos.y, arena.fly.heading)


def start_matrix(paradigm_id: str, seed: int, random_starts: int) -> List[Tuple[float, float, float]]:
    """The paradigm's own spawn (what the daemon uses) plus random legal poses that are
    clear of every wall segment."""
    rng = random.Random(seed * 7919 + len(paradigm_id))
    region = spawn_region(paradigm_id)
    arena = make_arena(paradigm_id, seed)
    walls: List[WallSegment] = list(getattr(arena.paradigm, "walls", []) or []) if arena.paradigm else []
    starts = [(arena.fly.pos.x, arena.fly.pos.y, arena.fly.heading)]
    margin = FLY_RADIUS + 0.2
    for _ in range(random_starts):
        for _attempt in range(10000):
            x, y = region.sample(rng, margin)
            if all(w.distance_to_point(x, y) >= margin for w in walls):
                break
        else:
            raise RuntimeError(f"could not sample a wall-clear start for {paradigm_id}")
        starts.append((x, y, rng.uniform(0.0, 2.0 * math.pi)))
    return starts


@dataclass
class ParadigmSummary:
    paradigm: str
    assists: bool = True
    runs: int = 0
    steps: int = 0
    escapes: int = 0
    teleports: int = 0
    unexplained_jumps: int = 0
    numerical_trapping_steps: int = 0
    max_numerical_trapping_s: float = 0.0
    max_step_mm: float = 0.0
    max_pinned_s: float = 0.0
    max_stuck_s: float = 0.0
    wall_pushing_s: float = 0.0
    max_wall_pushing_s: float = 0.0
    rest_s: float = 0.0
    near_wall_s: float = 0.0
    path_mm: float = 0.0
    contact_steps: int = 0
    stall_episodes: Dict[str, int] = field(default_factory=dict)
    assist_totals: Dict[str, float] = field(default_factory=dict)
    failed_runs: int = 0
    pushing_runs: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def contact_pct(self) -> float:
        return 100.0 * self.contact_steps / max(1, self.steps)

    @property
    def ok(self) -> bool:
        """Arena containment only (escape, unexplained jump, numerical trapping)."""
        return self.failed_runs == 0


def audit(
    paradigms: Sequence[str],
    seeds: Sequence[int],
    random_starts: int,
    steps: int,
    dt: float,
    pinned_seconds: float,
    progress: Optional[Callable[[str], None]] = None,
    assists: bool = True,
) -> Dict[str, ParadigmSummary]:
    out: Dict[str, ParadigmSummary] = {}
    for pid in paradigms:
        summary = ParadigmSummary(pid, assists=assists)
        for seed in seeds:
            for start in start_matrix(pid, seed, random_starts):
                res = run_case(pid, seed, start, steps, dt, pinned_seconds, assists=assists)
                probe = res.probe
                summary.runs += 1
                summary.steps += res.steps
                summary.escapes += res.escapes
                summary.teleports += res.teleports
                summary.unexplained_jumps += probe.kinds.get("unexplained_jump", 0)
                summary.numerical_trapping_steps += probe.kinds.get("numerical_trapping", 0)
                summary.max_numerical_trapping_s = max(summary.max_numerical_trapping_s, probe.max_seconds("numerical_trapping"))
                summary.max_step_mm = max(summary.max_step_mm, res.max_step_mm)
                summary.max_pinned_s = max(summary.max_pinned_s, res.max_pinned_s)
                summary.max_stuck_s = max(res.max_stuck_coord_s, summary.max_stuck_s)
                summary.wall_pushing_s += probe.seconds("wall_pushing")
                summary.max_wall_pushing_s = max(summary.max_wall_pushing_s, probe.max_seconds("wall_pushing"))
                summary.rest_s += sum(probe.seconds(k) for k in ("rest", "feeding", "gap_probing", "courtship_rest"))
                summary.near_wall_s += probe.near_wall_steps * dt
                summary.path_mm += probe.path_mm
                summary.contact_steps += res.contact_steps
                for cause, n in probe.stall_episodes.items():
                    summary.stall_episodes[cause] = summary.stall_episodes.get(cause, 0) + n
                summary.pushing_runs += int(res.pushing_exceeded)
                if res.failed:
                    summary.failed_runs += 1
                    for note in (res.first_escape, res.first_teleport, probe.first_defect):
                        if note and len(summary.notes) < 6:
                            summary.notes.append(f"seed {seed} start ({start[0]:.1f},{start[1]:.1f},{start[2]:.2f}) -> {note}")
        out[pid] = summary
        if progress:
            progress(f"{pid:22s} physics {'OK ' if summary.ok else 'FAIL'} escapes={summary.escapes} "
                     f"jumps={summary.unexplained_jumps} trap_steps={summary.numerical_trapping_steps} | "
                     f"outcome: max_push={summary.max_wall_pushing_s:.2f}s max_pinned={summary.max_pinned_s:.2f}s "
                     f"contact={summary.contact_pct:.1f}% stalls={summary.stall_episodes}")
    return out


def format_table(results: Dict[str, ParadigmSummary], pinned_seconds: float) -> str:
    head = (f"{'paradigm':22s} {'runs':>4s} {'steps':>7s} | {'escapes':>7s} {'teleport':>8s} {'jumps':>5s} "
            f"{'trap_s':>6s} {'max_step':>8s} {'physics':>7s} | {'push_s':>7s} {'max_push':>8s} {'max_pin':>7s} "
            f"{'stuck_s':>7s} {'contact%':>8s} {'stalls':>6s}")
    lines = ["ARENA CONTAINMENT (physics) | CONTROLLER OUTCOME (behaviour, not a physics verdict)", head, "-" * len(head)]
    for pid, s in results.items():
        lines.append(
            f"{pid:22s} {s.runs:4d} {s.steps:7d} | {s.escapes:7d} {s.teleports:8d} {s.unexplained_jumps:5d} "
            f"{s.max_numerical_trapping_s:6.2f} {s.max_step_mm:8.3f} {'OK' if s.ok else 'FAIL':>7s} | "
            f"{s.wall_pushing_s:7.1f} {s.max_wall_pushing_s:8.2f} {s.max_pinned_s:7.2f} {s.max_stuck_s:7.2f} "
            f"{s.contact_pct:8.1f} {sum(s.stall_episodes.values()):6d}"
        )
    lines.append(f"(physics fails on escape, teleport, unexplained jump or numerical trapping > {NUMERICAL_TRAP_LIMIT_S:.1f} s;"
                 f" pushing longer than {pinned_seconds:.1f} s is reported as a controller outcome)")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paradigm", action="append", help="restrict to one or more paradigm ids")
    ap.add_argument("--seeds", type=int, default=3, help="number of RNG seeds per paradigm")
    ap.add_argument("--starts", type=int, default=3, help="random legal start poses per seed (plus the default spawn)")
    ap.add_argument("--steps", type=int, default=2000, help="simulation steps per run (2000 x 0.02 s = 40 s)")
    ap.add_argument("--dt", type=float, default=0.02, help="simulation tick in seconds (daemon uses 0.02)")
    ap.add_argument("--pinned-seconds", type=float, default=2.0, help="continuous wall-pushing time reported as a controller outcome")
    ap.add_argument("--no-assists", action="store_true",
                    help="disable the engineered motor assists (wall_avoidance_reflex, contact_turn)")
    ap.add_argument("--strict-behaviour", action="store_true",
                    help="also exit 1 when pushing exceeds --pinned-seconds (a controller outcome, not physics)")
    ap.add_argument("--quick", action="store_true", help="1 seed, 1 random start, 500 steps")
    args = ap.parse_args(argv)

    if args.quick:
        args.seeds, args.starts, args.steps = 1, 1, 500

    paradigms = args.paradigm or DAEMON_PARADIGMS
    seeds = list(range(1, args.seeds + 1))
    t0 = time.perf_counter()
    print(f"motor assists: {'OFF' if args.no_assists else 'ON (default)'}")
    results = audit(paradigms, seeds, args.starts, args.steps, args.dt, args.pinned_seconds,
                    progress=lambda msg: print(msg, flush=True), assists=not args.no_assists)
    print()
    print(format_table(results, args.pinned_seconds))
    failures = [s for s in results.values() if not s.ok]
    for s in failures:
        for note in s.notes:
            print(f"  [{s.paradigm}] {note}")
    pushing = [s for s in results.values() if s.pushing_runs]
    print(f"\n{len(results) - len(failures)}/{len(results)} paradigms pass arena containment; "
          f"{len(pushing)} with wall pushing > {args.pinned_seconds:.1f} s (controller outcome)  "
          f"({time.perf_counter() - t0:.1f} s)")
    return 1 if failures or (args.strict_behaviour and pushing) else 0


if __name__ == "__main__":
    sys.exit(main())
