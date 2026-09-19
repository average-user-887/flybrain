"""Work package 3: wall-progress fixtures, solver regressions and motor-assist provenance.

Physics fixtures drive the body with scripted known escape commands, so a failure
points at the collision solver. Behaviour fixtures run a controller and report
immobility or wall pushing as task outcomes; only their physics is asserted.
See ``tests/fixtures/wall_progress.py`` for the predeclared criteria.
"""
import math
import os
import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import containment_audit as audit  # noqa: E402
from arena import Arena  # noqa: E402
from maze import CollisionEngine, WallSegment  # noqa: E402
from tests.fixtures import wall_progress as wp  # noqa: E402

PHYSICS = {fx.name: fx for fx in wp.physics_fixtures()}
BEHAVIOUR = {fx.name: fx for fx in wp.behaviour_fixtures()}
DT = wp.DT


def _failed(report, kinds):
    return [c for c in report["checks"] if c["kind"] in kinds and c["passed"] is False]


@pytest.mark.parametrize("name", sorted(PHYSICS))
def test_physics_fixture_solver_only(name):
    """Assists off: the solver alone must contain the body and let the known escape
    command make progress. Every fixture runs longer than 12 simulated seconds."""
    fx = PHYSICS[name]
    assert fx.duration_s > 12.0
    run, report = wp.run_fixture(fx, assists=False)
    assert report["motor_provenance"]["assist_totals"]["wall_reflex_steps"] == 0
    assert report["motor_provenance"]["assist_totals"]["contact_turn_events"] == 0
    bad = _failed(report, ("physics", "progress"))
    assert not bad, f"{name}: {bad}\ncontainment={report['arena_containment']}"


@pytest.mark.parametrize("name", sorted(PHYSICS))
def test_physics_fixture_with_default_assists_keeps_containment(name):
    _, report = wp.run_fixture(PHYSICS[name], assists=True)
    bad = _failed(report, ("physics",))
    assert not bad, f"{name}: {bad}\ncontainment={report['arena_containment']}"


@pytest.mark.parametrize("assists", [False, True], ids=["assists-off", "assists-on"])
@pytest.mark.parametrize("name", sorted(BEHAVIOUR))
def test_behaviour_fixture_reports_three_verdicts(name, assists):
    run, report = wp.run_fixture(BEHAVIOUR[name], assists=assists)
    assert report["sim_seconds"] > 12.0
    assert not _failed(report, ("physics",)), report["arena_containment"]
    # The three verdicts are separate fields; task checks are recorded, not asserted.
    assert set(report) >= {"arena_containment", "controller_outcome", "biological_validity"}
    assert report["biological_validity"]["status"] == "not_assessed"
    assert all(c["passed"] in (True, False) for c in report["controller_outcome"]["task_checks"])


def test_equal_antenna_push_is_a_task_outcome_and_escape_is_credited_to_the_assist():
    """Equal antenna signals give zero yaw, so a pure tropotaxis controller pushes the
    wall head-on. Without assists that is classified wall_pushing (not trapping) and
    the solver keeps it exactly symmetric; with assists the escape is logged as reflex yaw."""
    fx = BEHAVIOUR["equal_antenna_scripted_tropotaxis"]
    run, off = wp.run_fixture(fx, assists=False)
    assert run.probe.seconds("wall_pushing") > 10.0
    assert run.probe.kinds.get("numerical_trapping", 0) == 0
    assert max(abs(r.x - 130.0) for r in run.rows) <= wp.SYMMETRY_TOL_MM
    task = {c["label"]: c["passed"] for c in off["controller_outcome"]["task_checks"]}
    assert task["leaves the wall (pushing < 50 % of the trial)"] is False
    run_on, on = wp.run_fixture(fx, assists=True)
    totals = on["motor_provenance"]["assist_totals"]
    assert totals["wall_reflex_steps"] > 0 and totals["wall_reflex_abs_yaw_rad"] > 0.0
    assert run_on.probe.seconds("wall_pushing") < run.probe.seconds("wall_pushing")


# ---------------------------------------------------------------------------
# Solver regressions found by the fixtures
# ---------------------------------------------------------------------------
def _scripted(arena, yaw, speed):
    arena.compute_steering = lambda *a, **k: (yaw, speed, "SCRIPTED", arena.fly.heading, arena.fly.heading)


def test_leaving_a_wall_moves_exactly_the_commanded_step():
    """The start-of-step check used to invent prev = pos - v*dt; a fly in contact and
    walking away at 45 deg then moved 2x its normal step (0.057 instead of 0.028 mm)."""
    arena = Arena(paradigm="labyrinth", seed=1, motor_assists=dict(audit.ASSISTS_OFF))
    fly = arena.fly
    fly.pos.x, fly.pos.y, fly.heading = 25.0 + 1.5 + 1e-4, 40.0, math.radians(45.0)
    _scripted(arena, 0.0, 2.0)
    x0 = fly.pos.x
    arena.step(DT)
    assert fly.pos.x - x0 == pytest.approx(2.0 * DT * math.cos(math.radians(45.0)), abs=1e-9)
    assert fly.motor_record["realized_mm"] == pytest.approx(fly.motor_record["attempted_mm"], abs=1e-9)
    assert fly.motor_record["overlap_correction_mm"] == 0.0


def test_reverse_speed_sign_survives_the_overlap_check():
    """|v| from the start-of-step check turned a reversing fly's speed positive."""
    arena = Arena(paradigm="labyrinth", seed=1, motor_assists=dict(audit.ASSISTS_OFF))
    fly = arena.fly
    # Overlapping x=25 while backing obliquely into it: the tangential share survives.
    fly.pos.x, fly.pos.y, fly.heading, fly.speed = 25.0 + 1.4, 40.0, 0.5, -2.0
    seen = []
    arena.compute_steering = lambda *a, **k: (seen.append(fly.speed) or (0.0, -2.0, "REVERSE", fly.heading, fly.heading))
    arena.step(DT)
    assert seen[0] < 0.0
    assert fly.motor_record["overlap_correction_mm"] == pytest.approx(0.1 + 1e-4, abs=1e-6)


def test_second_wall_inside_contact_band_does_not_snap_the_body_to_the_corner():
    """Sliding along wall A within 0.05 mm of wall B used to place the body on the
    corner intersection: a 0.049 mm (up to ~0.2 mm at shallow vertices) jump."""
    engine = CollisionEngine([WallSegment((0, 40), (40, 40)), WallSegment((40, 40), (40, 0))])
    r = 1.5
    x0, y0 = 38.451, 40.0 - r - 1e-4           # touching the top wall, 1.549 from the side wall
    v = (1.06, 1.06)
    x, y, vx, vy, hit, normals = engine.resolve(x0 + v[0] * DT, y0 + v[1] * DT, v[0], v[1], radius=r,
                                                prev_x=x0, prev_y=y0, dt=DT)
    assert hit
    assert math.hypot(x - x0, y - y0) <= math.hypot(*v) * DT + 1e-9
    assert x < 40.0 - r and y <= 40.0 - r
    assert len(normals) == 1                   # only the touched wall constrains velocity


def test_parallel_walls_do_not_recentre_the_body():
    """Former det=0 branch put the body midway between parallel walls."""
    engine = CollisionEngine(wp.corridor(3.05))
    r = 1.5
    x0, y0 = 5.0, 10.0 + r + 1e-4               # touching the lower wall
    x, y, *_ = engine.resolve(x0 + 0.03, y0 - 0.001, 1.5, -0.05, radius=r, prev_x=x0, prev_y=y0, dt=DT)
    assert y == pytest.approx(y0, abs=1e-6)


# ---------------------------------------------------------------------------
# Motor-assist option: explicit, named, logged, default unchanged
# ---------------------------------------------------------------------------
def test_motor_assists_default_on_and_validated():
    arena = Arena(paradigm="t-maze")
    assert arena.motor_assists == {"wall_avoidance_reflex": True, "contact_turn": True}
    with pytest.raises(ValueError):
        Arena(paradigm="t-maze", motor_assists={"wall_magnet": False})


def test_disabled_reflex_leaves_controller_yaw_untouched():
    arena = Arena(paradigm="t-maze", motor_assists={"wall_avoidance_reflex": False})
    fly = arena.fly
    fly.pos.x, fly.pos.y, fly.heading = 63.0 + 1.5 + 2.0, 25.0, math.radians(160.0)
    assert arena.wall_avoidance_turn(fly, 0.0) == 0.0
    assert arena.wall_avoidance_turn(fly, 0.3) == 0.3


def test_disabled_contact_turn_only_removes_into_wall_speed():
    arena = Arena(paradigm="looming-escape", motor_assists={"contact_turn": False})
    fly = arena.fly
    fly.pos.x, fly.pos.y, fly.heading, fly.speed = 79.5, 40.0, 0.0, 2.0
    assert arena.enforce_containment(fly, DT)
    assert fly.heading == 0.0 and fly.speed == pytest.approx(0.0, abs=1e-9)
    assert arena.last_failsafe[3] == 0.0


def test_every_step_logs_attempt_realization_contacts_and_assists():
    arena = Arena(paradigm="t-maze")
    fly = arena.fly
    fly.pos.x, fly.pos.y, fly.heading = 63.0 + 1.5 + 0.5, 25.0, math.radians(170.0)
    result = arena.step(DT)
    rec = result["motor"]
    for key in ("controller_yaw", "controller_speed", "wall_reflex_yaw", "attempted_mm", "realized_mm",
                "contact_normals", "solver_correction_mm", "failsafe_correction_mm", "contact_turn_rad",
                "overlap_correction_mm", "wall_gap_mm", "near_wall", "in_contact", "tethered"):
        assert key in rec
    assert rec["wall_reflex_yaw"] != 0.0                     # the assist acted and is visible
    prov = arena.motor_provenance()
    assert prov["assist_totals"]["wall_reflex_steps"] == 1
    assert prov["assist_totals"]["steps"] == 1


def test_open_arena_logs_edge_clamp_and_contact_turn():
    arena = Arena(seed=1, num_predators=0)
    fly = arena.fly
    fly.pos.x, fly.pos.y, fly.heading = 98.5, 50.0, 0.0
    arena.compute_steering = lambda *a, **k: (0.0, 2.0, "SCRIPTED", fly.heading, fly.heading)
    arena.step(DT)
    rec = fly.motor_record
    assert rec["failsafe_correction_mm"] > 0.0 and rec["contact_normals"] == [(-1.0, 0.0)]
    assert rec["contact_turn_rad"] != 0.0
    off = Arena(seed=1, num_predators=0, motor_assists=dict(audit.ASSISTS_OFF))
    f2 = off.fly
    f2.pos.x, f2.pos.y, f2.heading = 98.5, 50.0, 0.0
    off.compute_steering = lambda *a, **k: (0.0, 2.0, "SCRIPTED", f2.heading, f2.heading)
    off.step(DT)
    assert f2.heading == 0.0 and f2.motor_record["contact_turn_rad"] == 0.0


def test_classify_step_separates_pushing_from_trapping():
    base = dict(controller_speed=1.0, attempted_mm=0.03, attempted_dx=0.03, attempted_dy=0.0, state="WANDER")
    pushing = dict(base, realized_mm=0.0, contact_normals=[(-1.0, 0.0)], in_contact=True)
    trapped_free = dict(base, realized_mm=0.0, contact_normals=[], in_contact=False)
    trapped_away = dict(base, realized_mm=0.0, contact_normals=[(1.0, 0.0)], in_contact=True)
    jump = dict(base, realized_mm=0.2)
    rest = dict(base, controller_speed=0.0, attempted_mm=0.0, realized_mm=0.0)
    probe = dict(base, state="PROBE", realized_mm=0.0)
    assert audit.classify_step(pushing, DT) == "wall_pushing"
    assert audit.classify_step(trapped_free, DT) == "numerical_trapping"
    assert audit.classify_step(trapped_away, DT) == "numerical_trapping"
    assert audit.classify_step(jump, DT) == "unexplained_jump"
    assert audit.classify_step(rest, DT) == "rest"
    assert audit.classify_step(probe, DT) == "gap_probing"
    assert audit.classify_step(dict(rest, tethered=True), DT) == "tethered"


# ---------------------------------------------------------------------------
# Restored trained checkpoints (copies only)
# ---------------------------------------------------------------------------
RESTORED = ("t-maze", "labyrinth", "courtship", "multisensory-sandbox")


@pytest.fixture(scope="module")
def trained_brain_dir(tmp_path_factory):
    """Train small brains with the teaching protocol, save, and return the directory."""
    from experiment_brains import ExperimentBrain
    d = tmp_path_factory.mktemp("trained-brains")
    for pid in RESTORED:
        brain = ExperimentBrain(pid, d)
        brain.start_teaching(pairs=2)
        while brain.teaching:
            brain.teaching_step(DT)
        brain.arena.step(DT)
        brain.save()
    return d


@pytest.mark.parametrize("assists", [True, False], ids=["assists-on", "assists-off"])
@pytest.mark.parametrize("paradigm", RESTORED)
def test_restored_trained_checkpoint_keeps_containment(trained_brain_dir, tmp_path, paradigm, assists):
    work = tmp_path / "copy"
    work.mkdir()
    shutil.copy2(trained_brain_dir / f"{paradigm}.json", work / f"{paradigm}.json")
    report = wp.run_restored(work, paradigm, 15.0, assists)
    assert report["restored"] and report["brain_steps"] > 0
    assert report["checkpoint_unchanged"]
    assert report["physics_pass"], report["arena_containment"]
    assert report["sim_seconds"] > 12.0


RETAINED = os.environ.get("NEUROFLY_RETAINED_BRAINS")


@pytest.mark.skipif(not RETAINED, reason="set NEUROFLY_RETAINED_BRAINS to a COPY of retained checkpoints")
@pytest.mark.parametrize("assists", [True, False], ids=["assists-on", "assists-off"])
def test_users_retained_brains_keep_containment(tmp_path, assists):
    src = Path(RETAINED)
    for ckpt in sorted(src.glob("*.json")):
        work = tmp_path / f"{ckpt.stem}-{assists}"
        work.mkdir()
        shutil.copy2(ckpt, work / ckpt.name)
        report = wp.run_restored(work, ckpt.stem, 15.0, assists)
        assert report["restored"], ckpt.name
        assert report["physics_pass"], (ckpt.name, report["arena_containment"])
