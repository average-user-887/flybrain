"""Containment audit across every daemon paradigm, plus regression tests for the
root causes it exposed.

The heavy lifting lives in ``scripts/containment_audit.py`` (run it directly for the
full matrix and the per-paradigm table). Here each paradigm gets a reduced matrix so
the suite stays fast; scale it up with ``NEUROFLY_AUDIT_STEPS`` / ``NEUROFLY_AUDIT_SEEDS``.
"""

import math
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import containment_audit as audit  # noqa: E402
from arena import Arena, CircleRegion, HoledRegion, RectRegion, UnionRegion  # noqa: E402

AUDIT_STEPS = int(os.environ.get("NEUROFLY_AUDIT_STEPS", "600"))
AUDIT_SEEDS = int(os.environ.get("NEUROFLY_AUDIT_SEEDS", "1"))
AUDIT_STARTS = int(os.environ.get("NEUROFLY_AUDIT_STARTS", "1"))
DT = 0.02
PINNED_LIMIT_S = 2.0


@pytest.mark.parametrize("paradigm_id", audit.DAEMON_PARADIGMS)
def test_paradigm_containment_invariants(paradigm_id):
    """No escapes, no teleports, no wall-pinning, no coordinate clamping, for the
    paradigm's own spawn and random legal start poses."""
    results = audit.audit([paradigm_id], list(range(1, AUDIT_SEEDS + 1)), AUDIT_STARTS,
                          AUDIT_STEPS, DT, PINNED_LIMIT_S)
    summary = results[paradigm_id]
    detail = "\n".join(summary.notes)
    assert summary.escapes == 0, f"{paradigm_id}: {summary.escapes} escapes\n{detail}"
    assert summary.teleports == 0, f"{paradigm_id}: {summary.teleports} teleports (max step {summary.max_step_mm:.3f} mm)\n{detail}"
    assert summary.max_pinned_s <= PINNED_LIMIT_S, f"{paradigm_id}: wall-pinned for {summary.max_pinned_s:.2f} s\n{detail}"
    assert summary.max_stuck_s <= PINNED_LIMIT_S, f"{paradigm_id}: clamped at one coordinate for {summary.max_stuck_s:.2f} s\n{detail}"
    # The fly must actually have moved: a frozen fly trivially satisfies the above.
    assert summary.steps == summary.runs * AUDIT_STEPS


class TestContainmentRegions:
    def test_paradigm_key_is_exact_not_substring(self):
        """'heat_maze' contains 't_maze'; substring matching once clamped the heat-maze
        fly into the T-maze corridor (a 49 mm teleport on the first step)."""
        heat = Arena(paradigm="heat-maze")
        assert Arena.paradigm_key(heat.paradigm) == "heat_maze"
        assert isinstance(heat.containment, CircleRegion)
        assert isinstance(Arena(paradigm="t-maze").containment, UnionRegion)
        assert Arena.paradigm_key(Arena(paradigm="multisensory-sandbox").paradigm) == "multisensory"

    def test_heat_maze_start_is_not_clamped_into_t_maze(self):
        arena = Arena(paradigm="heat-maze")
        fly = arena.fly
        fly.pos.x, fly.pos.y, fly.heading = 21.7, 72.8, 1.0
        arena.step(DT)
        assert math.hypot(fly.pos.x - 21.7, fly.pos.y - 72.8) < 0.2

    def test_multisensory_spawns_at_origin_inside_circular_arena(self):
        """The multisensory arena is origin-centred (r=75, the frame web/app.js draws);
        the old spawn at (80, 80) was outside it and the fly was then held on the
        160x160 rectangle at x=158.5."""
        arena = Arena(paradigm="multisensory-sandbox")
        assert (arena.fly.pos.x, arena.fly.pos.y) == (0.0, 0.0)
        assert isinstance(arena.containment, HoledRegion)
        assert arena.world_bounds == (-75.0, -75.0, 75.0, 75.0)
        fly = arena.fly
        fly.pos.x, fly.pos.y, fly.heading = 80.0, 80.0, 0.5
        arena.step(DT)
        assert math.hypot(fly.pos.x, fly.pos.y) <= 75.0 - fly.radius + 1e-6
        assert fly.pos.x < 158.0

    def test_multisensory_pillars_are_excluded(self):
        arena = Arena(paradigm="multisensory-sandbox")
        x, y = arena.containment.clamp(35.5, 35.2, 1.5)
        assert math.hypot(x - 35.0, y - 35.0) >= 6.0 + 1.5 - 1e-9

    def test_region_geometry_matches_dashboard(self):
        expected = {
            "gap-crossing": (5.0, 7.5, 95.0, 12.5),
            "circadian-dam": (5.0, 1.0, 60.0, 9.0),
            "wind-tunnel": (0.0, 0.0, 200.0, 60.0),
            "labyrinth": (0.0, 0.0, 140.0, 100.0),
        }
        for pid, box in expected.items():
            region = Arena(paradigm=pid).containment
            assert isinstance(region, RectRegion)
            assert region.bbox() == box
        court = Arena(paradigm="courtship").containment
        assert isinstance(court, CircleRegion) and court.radius == 8.5
        buridan = Arena(paradigm="buridan").containment
        assert isinstance(buridan, CircleRegion) and buridan.radius == 50.0


class TestWallPerception:
    def test_reflex_turns_away_from_wall_before_contact(self):
        """A fly heading into the T-maze stem wall from 2 mm away must receive an
        away-from-wall yaw larger than any brain-level turn command."""
        arena = Arena(paradigm="t-maze")
        fly = arena.fly
        fly.pos.x, fly.pos.y = 63.0 + 1.5 + 2.0, 25.0   # body edge 2 mm from the x=63 wall
        fly.heading = math.radians(160.0)                # heading west-north-west into it
        turn = arena.wall_avoidance_turn(fly, 0.0)
        # Shortest way out is clockwise (160 -> 90 -> 0 deg), stronger than the brain's 0.45 clip
        assert turn < -0.45
        # Mirror pose below the wall normal turns the other way
        fly.heading = math.radians(200.0)
        assert arena.wall_avoidance_turn(fly, 0.0) > 0.45
        # Far from every wall nothing is added
        fly.pos.x, fly.pos.y, fly.heading = 70.0, 25.0, math.pi / 2.0
        assert arena.wall_avoidance_turn(fly, 0.1) == pytest.approx(0.1)

    def test_reflex_uses_direction_of_travel_when_walking_backwards(self):
        """Under heat or laser the modular brain walks backwards (speed < 0). The fly
        then backs into walls it is facing away from; the reflex must act on the
        direction of travel, not the heading."""
        arena = Arena(paradigm="visual-operant")           # no walls: region bottom edge y=0
        fly = arena.fly
        fly.pos.x, fly.pos.y = 40.0, 1.5 + 1.0             # body edge 1 mm above the bottom edge
        fly.heading, fly.speed = math.radians(70.0), -0.5  # facing up, walking backwards (down)
        turn = arena.wall_avoidance_turn(fly, 0.0)
        assert abs(turn) > 0.45
        fly.speed = 0.5                                     # same pose walking forwards: moving away
        assert arena.wall_avoidance_turn(fly, 0.0) == pytest.approx(0.0)

    def test_reflex_is_silent_when_sliding_parallel(self):
        arena = Arena(paradigm="t-maze")
        fly = arena.fly
        fly.pos.x, fly.pos.y, fly.heading = 64.6, 25.0, math.pi / 2.0   # touching x=63, heading north
        assert arena.wall_avoidance_turn(fly, 0.2) == pytest.approx(0.2)

    def test_failsafe_clamp_turns_and_slows_instead_of_pinning(self):
        arena = Arena(paradigm="looming-escape")   # no wall segments: only the failsafe holds it
        fly = arena.fly
        fly.pos.x, fly.pos.y, fly.heading, fly.speed = 79.5, 40.0, 0.0, 2.0   # outside x=80-1.5, heading +x
        clamped = arena.enforce_containment(fly, DT)
        assert clamped
        assert fly.pos.x == pytest.approx(78.5)
        assert fly.speed == pytest.approx(0.0, abs=1e-9)      # head-on: no speed into the wall survives
        assert fly.heading != 0.0                              # torque applied, not a silent clamp

    def test_no_pin_when_driven_into_rectangle_corner(self):
        """Drive the fly repeatedly toward a corner of a wall-less arena; it must keep
        moving rather than freeze against the failsafe."""
        arena = Arena(paradigm="looming-escape", seed=3)
        fly = arena.fly
        fly.pos.x, fly.pos.y, fly.heading = 76.0, 76.0, math.pi / 4.0
        frozen = 0
        for _ in range(400):
            px, py = fly.pos.x, fly.pos.y
            arena.step(DT)
            moved = math.hypot(fly.pos.x - px, fly.pos.y - py)
            frozen = frozen + 1 if (moved < 1e-4 and fly.speed > 0.05) else 0
            assert frozen * DT <= PINNED_LIMIT_S
            assert 1.5 - 1e-6 <= fly.pos.x <= 78.5 + 1e-6 and 1.5 - 1e-6 <= fly.pos.y <= 78.5 + 1e-6


def test_reset_fly_to_spawn_uses_paradigm_spawn():
    arena = Arena(paradigm="multisensory-sandbox")
    arena.fly.pos.x, arena.fly.pos.y = 40.0, -20.0
    arena.reset_fly_to_spawn()
    assert (arena.fly.pos.x, arena.fly.pos.y, arena.fly.heading) == (0.0, 0.0, 0.0)
    t = Arena(paradigm="t-maze")
    t.fly.pos.x = 100.0
    t.reset_fly_to_spawn()
    assert (t.fly.pos.x, t.fly.pos.y) == (70.0, 18.0)
