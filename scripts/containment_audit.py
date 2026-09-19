#!/usr/bin/env python3
"""Containment audit: script-controlled physics harness for every NeuroFly paradigm.

Steps the headless simulation (the same ``Arena.step(dt)`` loop the 24/7 daemon
runs) for every paradigm the daemon exposes, over many seeds, start positions and
headings, and checks four invariants against an *independent* description of each
paradigm's legal region (defined in this file, not read back from ``arena.py``):

  (a) escape   - the fly body is never outside the paradigm's legal region and never
                 penetrates or tunnels through a wall segment;
  (b) teleport - no per-step displacement larger than the physical maximum;
  (c) pinned   - the fly is never wall-pinned: in contact with a boundary while pushing
                 into it (commanded motion absorbed by the boundary, or travel direction
                 pointing into the wall) for longer than ``--pinned-seconds`` of
                 continuous simulated time. Sliding parallel to a wall at full speed is
                 thigmotaxis and is allowed;
  (d) stuck    - the fly never sits clamped at one coordinate (x or y bit-identical
                 across steps) while its commanded motion is absorbed by that boundary.

Usage:
    ./.venv/bin/python scripts/containment_audit.py            # full audit
    ./.venv/bin/python scripts/containment_audit.py --quick    # short smoke run
    ./.venv/bin/python scripts/containment_audit.py --paradigm multisensory-sandbox

Exit status is 1 when any paradigm fails, so the script doubles as a CI gate. The
pytest wrapper ``tests/test_containment_all_paradigms.py`` runs a reduced matrix.
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


def make_arena(paradigm_id: str, seed: int) -> Arena:
    if paradigm_id == "open-arena":
        return Arena(paradigm=None, seed=seed, num_predators=0)
    return Arena(paradigm=paradigm_id, brain_type="modular", num_flies=1, num_predators=0, seed=seed)


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
        return self.escapes > 0 or self.teleports > 0 or self.pinned_failed or self.stuck_failed

    pinned_failed: bool = False
    stuck_failed: bool = False


def run_case(
    paradigm_id: str,
    seed: int,
    start: Tuple[float, float, float],
    steps: int,
    dt: float,
    pinned_seconds: float,
    keep_trace: bool = False,
) -> RunResult:
    arena = make_arena(paradigm_id, seed)
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

    res = RunResult(paradigm_id, seed, start, steps, dt)
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
    runs: int = 0
    steps: int = 0
    escapes: int = 0
    teleports: int = 0
    max_step_mm: float = 0.0
    max_pinned_s: float = 0.0
    max_stuck_s: float = 0.0
    contact_steps: int = 0
    failed_runs: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def contact_pct(self) -> float:
        return 100.0 * self.contact_steps / max(1, self.steps)

    @property
    def ok(self) -> bool:
        return self.failed_runs == 0


def audit(
    paradigms: Sequence[str],
    seeds: Sequence[int],
    random_starts: int,
    steps: int,
    dt: float,
    pinned_seconds: float,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, ParadigmSummary]:
    out: Dict[str, ParadigmSummary] = {}
    for pid in paradigms:
        summary = ParadigmSummary(pid)
        for seed in seeds:
            for start in start_matrix(pid, seed, random_starts):
                res = run_case(pid, seed, start, steps, dt, pinned_seconds)
                summary.runs += 1
                summary.steps += res.steps
                summary.escapes += res.escapes
                summary.teleports += res.teleports
                summary.max_step_mm = max(summary.max_step_mm, res.max_step_mm)
                summary.max_pinned_s = max(summary.max_pinned_s, res.max_pinned_s)
                summary.max_stuck_s = max(res.max_stuck_coord_s, summary.max_stuck_s)
                summary.contact_steps += res.contact_steps
                if res.failed:
                    summary.failed_runs += 1
                    for note in (res.first_escape, res.first_teleport, res.first_pinned):
                        if note and len(summary.notes) < 6:
                            summary.notes.append(f"seed {seed} start ({start[0]:.1f},{start[1]:.1f},{start[2]:.2f}) -> {note}")
        out[pid] = summary
        if progress:
            progress(f"{pid:22s} {'OK ' if summary.ok else 'FAIL'} escapes={summary.escapes} teleports={summary.teleports} "
                     f"max_pinned={summary.max_pinned_s:.2f}s contact={summary.contact_pct:.1f}%")
    return out


def format_table(results: Dict[str, ParadigmSummary], pinned_seconds: float) -> str:
    head = f"{'paradigm':22s} {'runs':>4s} {'steps':>7s} {'escapes':>7s} {'teleport':>8s} {'max_step':>8s} {'max_pin_s':>9s} {'stuck_s':>7s} {'contact%':>8s}  result"
    lines = [head, "-" * len(head)]
    for pid, s in results.items():
        lines.append(
            f"{pid:22s} {s.runs:4d} {s.steps:7d} {s.escapes:7d} {s.teleports:8d} {s.max_step_mm:8.3f} "
            f"{s.max_pinned_s:9.2f} {s.max_stuck_s:7.2f} {s.contact_pct:8.1f}  {'OK' if s.ok else 'FAIL'}"
        )
    lines.append(f"(pinned/stuck limit {pinned_seconds:.1f} s of continuous contact with absorbed motion)")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paradigm", action="append", help="restrict to one or more paradigm ids")
    ap.add_argument("--seeds", type=int, default=3, help="number of RNG seeds per paradigm")
    ap.add_argument("--starts", type=int, default=3, help="random legal start poses per seed (plus the default spawn)")
    ap.add_argument("--steps", type=int, default=2000, help="simulation steps per run (2000 x 0.02 s = 40 s)")
    ap.add_argument("--dt", type=float, default=0.02, help="simulation tick in seconds (daemon uses 0.02)")
    ap.add_argument("--pinned-seconds", type=float, default=2.0, help="max tolerated continuous wall-pinned time")
    ap.add_argument("--quick", action="store_true", help="1 seed, 1 random start, 500 steps")
    args = ap.parse_args(argv)

    if args.quick:
        args.seeds, args.starts, args.steps = 1, 1, 500

    paradigms = args.paradigm or DAEMON_PARADIGMS
    seeds = list(range(1, args.seeds + 1))
    t0 = time.perf_counter()
    results = audit(paradigms, seeds, args.starts, args.steps, args.dt, args.pinned_seconds,
                    progress=lambda msg: print(msg, flush=True))
    print()
    print(format_table(results, args.pinned_seconds))
    failures = [s for s in results.values() if not s.ok]
    for s in failures:
        for note in s.notes:
            print(f"  [{s.paradigm}] {note}")
    print(f"\n{len(results) - len(failures)}/{len(results)} paradigms pass  ({time.perf_counter() - t0:.1f} s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
