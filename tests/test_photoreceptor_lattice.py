"""Hex-lattice -> plane mapping of the photoreceptor grating encoder.

The released MaleCNS ``assignedOlHex1/2`` lattice has nearest neighbours
(+-1, 0), (0, +-1) and +-(1, 1).  ``'axial-v1'`` (the original v4/v5 frame)
places +-(1, -1) at unit distance instead, which is a shear of that lattice;
``'malecns-hex-v2'`` is the post-finding correction.  v1 stays the default and
must stay bit-identical to the published encoder.
"""
import math

import numpy as np
import pytest

from brainlab.photoreceptor_io import (DEFAULT_LATTICE, INTEROMMATIDIAL_DEG,
                                       PhotoreceptorGratingEncoder, PhotoreceptorIOMap,
                                       hex_to_plane_deg, lattice_matrix)

MALECNS_NEIGHBOURS = [(1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, -1)]


def _dist(lattice, offset):
    return float(np.linalg.norm(lattice_matrix(lattice) @ np.asarray(offset, float)))


def test_v2_places_the_six_malecns_neighbours_on_a_unit_hexagon():
    angles = []
    for o in MALECNS_NEIGHBOURS:
        assert _dist('malecns-hex-v2', o) == pytest.approx(1.0, abs=1e-12)
        p = lattice_matrix('malecns-hex-v2') @ np.asarray(o, float)
        angles.append(math.degrees(math.atan2(p[1], p[0])) % 360)
    assert sorted(round(a, 6) for a in angles) == [0, 60, 120, 180, 240, 300]
    # (1, -1) is second ring, not a neighbour
    assert _dist('malecns-hex-v2', (1, -1)) == pytest.approx(math.sqrt(3), abs=1e-12)


def test_v1_is_a_shear_not_an_equivalent_basis():
    # v1 puts the true neighbour (1, 1) at sqrt(3) and the non-neighbour (1, -1) at 1
    assert _dist('axial-v1', (1, 1)) == pytest.approx(math.sqrt(3), abs=1e-12)
    assert _dist('axial-v1', (1, -1)) == pytest.approx(1.0, abs=1e-12)
    s = lattice_matrix('axial-v1') @ np.linalg.inv(lattice_matrix('malecns-hex-v2'))
    np.testing.assert_allclose(s, [[1, 2 / math.sqrt(3)], [0, 1]], atol=1e-12)
    # a rotation/reflection would have both singular values 1
    np.testing.assert_allclose(np.linalg.svd(s)[1], [math.sqrt(3), 1 / math.sqrt(3)], atol=1e-12)


def test_v1_matches_the_published_inline_formula_exactly():
    rng = np.random.default_rng(0)
    cart = rng.integers(1, 40, size=(500, 2))
    u = (cart[:, 0] + cart[:, 1] / 2.0) * INTEROMMATIDIAL_DEG
    w = (math.sqrt(3.0) / 2.0 * cart[:, 1]) * INTEROMMATIDIAL_DEG
    assert np.array_equal(hex_to_plane_deg(cart, 'axial-v1'), np.stack([u, w], axis=1))
    assert DEFAULT_LATTICE == 'axial-v1'


def test_worked_column_18_20():
    c = np.array([[18, 20], [19, 21], [17, 21]])
    v1, v2 = hex_to_plane_deg(c, 'axial-v1'), hex_to_plane_deg(c, 'malecns-hex-v2')
    assert np.linalg.norm(v1[1] - v1[0]) == pytest.approx(5 * math.sqrt(3))
    assert np.linalg.norm(v2[1] - v2[0]) == pytest.approx(5.0)
    assert np.linalg.norm(v1[2] - v1[0]) == pytest.approx(5.0)
    assert np.linalg.norm(v2[2] - v2[0]) == pytest.approx(5 * math.sqrt(3))


def test_unknown_lattice_is_refused():
    with pytest.raises(ValueError):
        lattice_matrix('hex')


def _io(n=200, seed=1):
    rng = np.random.default_rng(seed)
    cart = {e: rng.integers(1, 30, size=(n, 2)) for e in 'LR'}
    nodes = {'L': np.arange(n), 'R': np.arange(n, 2 * n)}
    pos = {e: hex_to_plane_deg(cart[e], 'axial-v1') for e in 'LR'}
    return PhotoreceptorIOMap(r_nodes=nodes, r_position_deg=pos, r_cartridge=cart,
                              undriven={'L': 0, 'R': 0}, majority_assigned={'L': 0, 'R': 0},
                              trace={}, sha256='synthetic'), 2 * n


def test_encoder_default_is_v1_and_bit_identical_to_explicit_v1():
    io, n = _io()
    a, b = np.zeros(n, np.float32), np.zeros(n, np.float32)
    PhotoreceptorGratingEncoder(io).encode(a, 137.0, 45.0, 1.0)
    PhotoreceptorGratingEncoder(io, lattice='axial-v1').encode(b, 137.0, 45.0, 1.0)
    assert np.array_equal(a, b)
    assert 'lattice' not in PhotoreceptorGratingEncoder(io).describe()


def test_v2_changes_the_drive_except_along_the_unsheared_axis():
    io, n = _io()
    v1, v2 = (PhotoreceptorGratingEncoder(io, lattice=l) for l in ('axial-v1', 'malecns-hex-v2'))
    a, b = np.zeros(n, np.float32), np.zeros(n, np.float32)
    v1.encode(a, 137.0, 0.0, 1.0)
    v2.encode(b, 137.0, 0.0, 1.0)
    assert not np.allclose(a, b)
    # the shear fixes w, so a grating drifting along w (90 deg) is unchanged
    a[:], b[:] = 0, 0
    v1.encode(a, 137.0, 90.0, 1.0)
    v2.encode(b, 137.0, 90.0, 1.0)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-5)
    assert v2.describe()['lattice'] == 'malecns-hex-v2'
