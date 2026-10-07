"""Python arena geometry is the sole authority for live 3D assay geometry."""

import json
import math

from arena import Arena
from neurofly_daemon import ContinuousExperimentRunner, assay_geometry_telemetry
from tests.transition_control_helpers import transition_command


ASSAYS = (
    "open-arena", "t-maze", "y-maze", "heat-maze", "buridan", "visual-operant",
    "wind-tunnel", "looming-escape", "optomotor", "gap-crossing", "circadian-dam",
    "courtship", "labyrinth", "multisensory-sandbox",
)

EXPECTED_WALLS = {
    "open-arena": 0, "t-maze": 8, "y-maze": 9, "heat-maze": 32, "buridan": 0,
    "visual-operant": 0, "wind-tunnel": 4, "looming-escape": 0, "optomotor": 0,
    "gap-crossing": 0, "circadian-dam": 0, "courtship": 0, "labyrinth": 16,
    "multisensory-sandbox": 64,
}


def arena_for(assay):
    return Arena(paradigm=None if assay == "open-arena" else assay)


def test_all_14_assays_publish_finite_geometry_from_their_python_arena():
    for assay in ASSAYS:
        arena = arena_for(assay)
        geometry = assay_geometry_telemetry(arena)
        assert geometry["schema"] == "neurofly.assay-geometry.v1", assay
        assert geometry["source"] == "python-arena", assay
        assert geometry["coordinate_frame"] == "arena-mm", assay
        assert geometry["bounds"] == list(arena.world_bounds), assay
        assert len(geometry["walls"]) == EXPECTED_WALLS[assay], assay
        assert geometry["surfaces"], assay
        json.dumps(geometry, allow_nan=False)


def test_divergent_assays_preserve_exact_physical_dimensions():
    buridan = assay_geometry_telemetry(arena_for("buridan"))
    assert buridan["containment"] == {"kind": "circle", "center": [60.0, 60.0], "radius": 50.0}
    assert buridan["surfaces"] == [
        {"shape": "circle", "center": [60.0, 60.0], "radius": 50.0, "role": "containment"}
    ]

    labyrinth = assay_geometry_telemetry(arena_for("labyrinth"))
    assert {"p1": [100.0, 70.0], "p2": [100.0, 80.0]} in labyrinth["walls"]
    assert {"p1": [100.0, 70.0], "p2": [100.0, 85.0]} not in labyrinth["walls"]

    gap = assay_geometry_telemetry(arena_for("gap-crossing"))
    assert gap["surfaces"] == [
        {"shape": "rectangle", "bounds": [0.0, 7.5, 45.0, 12.5], "role": "start_track"},
        {"shape": "rectangle", "bounds": [48.5, 7.5, 100.0, 12.5], "role": "landing_track"},
    ]


def test_circadian_tubes_and_containment_shapes_are_not_browser_inventions():
    dam = assay_geometry_telemetry(arena_for("circadian-dam"))
    assert len(dam["surfaces"]) == 16
    assert dam["surfaces"][0]["bounds"] == [0.0, 0.0, 65.0, 10.0]
    assert dam["surfaces"][-1]["bounds"] == [0.0, 150.0, 65.0, 160.0]

    heat = assay_geometry_telemetry(arena_for("heat-maze"))
    assert heat["containment"]["radius"] == 55.0
    courtship = assay_geometry_telemetry(arena_for("courtship"))
    assert courtship["containment"]["radius"] == 8.5
    multi = assay_geometry_telemetry(arena_for("multisensory-sandbox"))
    assert multi["containment"]["kind"] == "holed"
    assert len(multi["containment"]["holes"]) == 4


def test_runner_packets_embed_the_geometry_contract_across_all_14_assays(tmp_path):
    runner = ContinuousExperimentRunner(initial_paradigm=ASSAYS[0], output_dir=tmp_path,
                                        checkpoint_interval=3600)
    for assay in ASSAYS:
        if assay != ASSAYS[0]:
            # Since 730059e a switch needs a durable observation recorder and
            # completes as a queued transaction; the shared helper supplies both.
            ack = transition_command(runner, {"action": "switch_paradigm", "paradigm": assay})
            assert ack["status"] == "ok", assay
        # The queued switch publishes the new assay with the next frame, so
        # check the frame a client actually receives.
        packet = json.loads(runner._publish_snapshot().data)
        geometry = packet["scene"]["geometry"]
        assert packet["paradigm"] == assay
        assert runner.latest_telemetry["scene"]["geometry"] == geometry
        assert geometry == assay_geometry_telemetry(runner.arena)
        assert len(geometry["walls"]) == EXPECTED_WALLS[assay]
        assert all(math.isfinite(value) for value in geometry["bounds"])
