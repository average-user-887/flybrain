"""Test suite for Phase 4 Capability Matrix and Batch Ingress Pathways.

Reference: docs/CAPABILITY_MATRIX.md and docs/ROADMAP.md
"""
from pathlib import Path
import pytest
from experiment_registry import PARADIGMS
from brainlab.graph_identity import DEFAULT_CONNECTOME_DIR, DEFAULT_GRAPH_DIR
from brainlab.cosim_server import ConnectomeServer


def test_capability_matrix_coverage():
    """Verify that docs/CAPABILITY_MATRIX.md exists and covers all 14 paradigms."""
    matrix_file = Path(__file__).resolve().parents[1] / 'docs/CAPABILITY_MATRIX.md'
    assert matrix_file.is_file(), "docs/CAPABILITY_MATRIX.md must exist"
    content = matrix_file.read_text(encoding='utf-8')

    for paradigm in PARADIGMS:
        assert f"`{paradigm}`" in content, f"Paradigm {paradigm} must be documented in capability matrix"

    # Verify batch descriptions
    assert "Batch A: Visual-Motor Pathway" in content
    assert "Batch B: Olfactory & Mechanosensory Pathway" in content
    assert "Batch C: Thermal & Spatial Pathway" in content
    assert "Batch D: Complex & Composite Assays" in content


def test_server_sensory_batch_ingress_mapping():
    """Verify that ConnectomeServer maps all Batch channels when real data is loaded."""
    if not (DEFAULT_GRAPH_DIR / 'graph.npz').is_file():
        pytest.skip('MaleCNS graph.npz not available in standard path')

    server = ConnectomeServer()
    indices = server.sensory_indices

    # Batch A (Visual / Looming)
    assert len(indices["visual_l"]) > 0
    assert len(indices["visual_r"]) > 0
    assert len(indices["visual_looming"]) > 0  # LC4 and LPLC2

    # Batch B (Olfactory / Wind)
    assert len(indices["orn_food"]) > 0
    assert len(indices["orn_danger"]) > 0
    assert len(indices["jon_wind"]) > 0
    assert len(indices["courtship_cva"]) > 0

    # Batch C (Thermal & Proprioceptive)
    assert len(indices["feco_proprio"]) > 0
    assert len(indices["thermo_receptors"]) > 0

    # Motor descending channels
    assert len(server.dn_indices["dna02_l"]) > 0
    assert len(server.dn_indices["dna02_r"]) > 0
    assert len(server.dn_indices["dnp09"]) > 0
    assert len(server.dn_indices["mdn"]) > 0
    assert len(server.dn_indices["dnp01"]) > 0
