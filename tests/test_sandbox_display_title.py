"""CARD75: the sandbox's display title is text only.

The title "Multisensory Sandbox · Heuristic Body Proxy" replaced "Multisensory Limb & Body
Benchmark". Every identifier stays the same: registry IDs, the snake_case paradigm key derived
from the name's prefix, the live-control limitation key, and the key stored in world snapshots,
so earlier snapshots that say 'multisensory' still restore.
"""
import pytest

from arena import Arena
from assay_controls import LIMITS, describe
from maze import ExperimentRegistry, MultisensoryLimbBenchmark

TITLE = "Multisensory Sandbox · Heuristic Body Proxy"


@pytest.mark.parametrize("registry_id", ["multisensory_benchmark", "multisensory_sandbox", "multisensory"])
def test_registry_ids_unchanged_and_title_is_neutral(registry_id):
    paradigm = ExperimentRegistry.get(registry_id)
    assert isinstance(paradigm, MultisensoryLimbBenchmark)
    assert paradigm.name == TITLE
    assert "Benchmark" not in paradigm.name and "Limb" not in paradigm.name
    assert Arena.paradigm_key(paradigm) == "multisensory"


def test_dashboard_id_keeps_its_key_controls_and_snapshot_contract():
    arena = Arena(paradigm="multisensory-sandbox", seed=4, num_flies=1, num_predators=0)
    assert arena.paradigm.name == TITLE
    assert describe(arena)["limitation"] == LIMITS["multisensory"]
    snap = arena.snapshot_world()
    assert snap["paradigm"] == "multisensory"
    # A snapshot written before the rename carries the same key and still restores.
    other = Arena(paradigm="multisensory-sandbox", seed=4, num_flies=1, num_predators=0)
    other.restore_world(snap)
