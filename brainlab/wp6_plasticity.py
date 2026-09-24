"""WP6: Visual-Heading Plasticity Rule (ER4d + ER2 -> EPG compass synapses).

Specification: docs/WP6_PLASTICITY_SPEC.md
Machine-readable companion: docs/wp6_plastic_subset.json

Declared plastic subset:
- Presynaptic types: ER4d (26 neurons), ER2_a, ER2_b, ER2_c, ER2_d (41 neurons) - 67 total
- Postsynaptic types: EPG, EPGt - 50 neurons
- Directed edges: exactly 3,081 (0.012% of the full 25.58M MaleCNS graph)
- Initial weights: all negative (every ER ring neuron is predicted GABAergic: 282/282)
- Modulatory population: EL (18 octopaminergic neurons)
- Allowed update: depression-only (|w| decreases toward 0); sign flip is forbidden.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from .graph_identity import DEFAULT_CONNECTOME_DIR, DEFAULT_GRAPH_DIR, GraphUnavailable, resolve_connectome_dir, resolve_graph_dir


class VisualHeadingPlasticityRule:
    """Two-factor presynaptic depression rule gated by octopaminergic EL modulatory signal (WP6).

    Follows Rule R1 from docs/WP6_PLASTICITY_SPEC.md §3.2.
    """
    __test__ = False  # Not a pytest test class

    name = 'visual-heading-plasticity-v1'
    subset_id = 'ER_to_EPG_visual_subset'
    test_only = False

    PRE_TYPES = ('ER4d', 'ER2_a', 'ER2_b', 'ER2_c', 'ER2_d')
    POST_TYPES = ('EPG', 'EPGt')
    MODULATOR_TYPES = ('EL',)

    def __init__(
        self,
        edges: Sequence[int] | np.ndarray,
        initial_weights: Sequence[float] | np.ndarray,
        *,
        el_nodes: Optional[Sequence[int] | np.ndarray] = None,
        pre_nodes: Optional[Sequence[int] | np.ndarray] = None,
        post_nodes: Optional[Sequence[int] | np.ndarray] = None,
        eta: float = 0.02,
        tau_pre: float = 0.20,
        tau_mod: float = 0.50,
        tau_rec: float = 60.0,
        dt: float = 0.002,
        s_sat: float = 50.0,
        r_sat: float = 30.0,
        g_min: float = 0.0,
        g_max: float = 1.0,
        test_only: bool = False,
    ):
        self.edges = np.asarray(edges, dtype=np.int64)
        self.initial_weights = np.asarray(initial_weights, dtype=np.float32)
        if len(self.edges) != len(self.initial_weights):
            raise ValueError(f"Length mismatch: {len(self.edges)} edges vs {len(self.initial_weights)} initial weights")

        self.g0 = np.abs(self.initial_weights)
        self.el_nodes = np.asarray(el_nodes, dtype=np.int64) if el_nodes is not None else np.zeros(0, dtype=np.int64)
        self.pre_nodes = np.asarray(pre_nodes, dtype=np.int64) if pre_nodes is not None else np.zeros(0, dtype=np.int64)
        self.post_nodes = np.asarray(post_nodes, dtype=np.int64) if post_nodes is not None else np.zeros(0, dtype=np.int64)

        self.eta = float(eta)
        self.tau_pre = float(tau_pre)
        self.tau_mod = float(tau_mod)
        self.tau_rec = float(tau_rec)
        self.dt = float(dt)
        self.s_sat = float(s_sat)
        self.r_sat = float(r_sat)
        self.g_min = float(g_min)
        self.g_max = float(g_max)
        self.test_only = bool(test_only)

        # Dynamic state traces
        self.pre_trace = np.zeros(len(self.edges), dtype=np.float32)
        self.mod_trace = 0.0

    @classmethod
    def from_connectome(
        cls,
        graph_dir: Optional[Path | str] = None,
        connectome_dir: Optional[Path | str] = None,
        **kwargs,
    ) -> 'VisualHeadingPlasticityRule':
        """Resolve the exact 3,081 ER->EPG edges and initial weights from the MaleCNS v1.0 graph."""
        import pyarrow.feather as feather

        gdir, _ = resolve_graph_dir(graph_dir)
        cdir, _ = resolve_connectome_dir(connectome_dir)

        graph_path = gdir / 'graph.npz'
        neurons_path = cdir / 'normalized/neurons.feather'

        if not graph_path.is_file() or not neurons_path.is_file():
            raise GraphUnavailable(f"MaleCNS graph or annotations missing in {gdir} / {cdir}")

        nodes = feather.read_table(neurons_path).to_pandas()
        graph = np.load(graph_path)
        ptr = graph['ptr']
        post = graph['post']
        weight = graph['weight']

        pre_nodes = sorted(nodes[nodes.cell_type.isin(cls.PRE_TYPES)].node_index)
        post_nodes_set = set(nodes[nodes.cell_type.isin(cls.POST_TYPES)].node_index)
        post_nodes = sorted(post_nodes_set)
        el_nodes = sorted(nodes[nodes.cell_type.isin(cls.MODULATOR_TYPES)].node_index)

        plastic_edges: List[int] = []
        for pre_idx in pre_nodes:
            p_start = ptr[pre_idx]
            p_end = ptr[pre_idx + 1]
            for edge_idx in range(p_start, p_end):
                if post[edge_idx] in post_nodes_set:
                    plastic_edges.append(edge_idx)

        edges_arr = np.array(plastic_edges, dtype=np.int64)
        initial_weights = weight[edges_arr].astype(np.float32)

        return cls(
            edges=edges_arr,
            initial_weights=initial_weights,
            el_nodes=el_nodes,
            pre_nodes=pre_nodes,
            post_nodes=post_nodes,
            **kwargs,
        )

    @classmethod
    def from_shared(
        cls,
        shared: Any,
        connectome_dir: Optional[Path | str] = None,
        **kwargs,
    ) -> 'VisualHeadingPlasticityRule':
        """Construct rule from a SharedGraph instance."""
        if getattr(getattr(shared, 'identity', None), 'synthetic', False):
            raise GraphUnavailable("VisualHeadingPlasticityRule cannot be resolved on a synthetic graph")
        return cls.from_connectome(
            graph_dir=None,
            connectome_dir=connectome_dir,
            **kwargs,
        )

    def describe(self) -> Dict[str, Any]:
        """Return full metadata matching docs/wp6_plastic_subset.json and manifest schema."""
        return {
            'name': self.name,
            'id': self.subset_id,
            'test_only': self.test_only,
            'n_edges': int(len(self.edges)),
            'pre_types': list(self.PRE_TYPES),
            'post_types': list(self.POST_TYPES),
            'pre_neurons': int(len(self.pre_nodes)),
            'post_neurons': int(len(self.post_nodes)),
            'modulator_types': list(self.MODULATOR_TYPES),
            'modulator_neurons': int(len(self.el_nodes)),
            'eta': self.eta,
            'tau_pre': self.tau_pre,
            'tau_mod': self.tau_mod,
            'tau_rec': self.tau_rec,
            'dt': self.dt,
            's_sat': self.s_sat,
            'r_sat': self.r_sat,
            'g_min': self.g_min,
            'g_max': self.g_max,
            'allowed_update': 'depression only (|w| decreases toward 0); sign flip forbidden',
            'spec': 'docs/WP6_PLASTICITY_SPEC.md',
            'primary_sources': ['Fisher 2019', 'Kim 2019', 'Fisher 2022', 'Plitt 2025 (preprint)'],
        }

    def reset(self) -> None:
        """Reset eligibility and modulatory traces."""
        self.pre_trace.fill(0.0)
        self.mod_trace = 0.0

    def substeps(self, duration_ms: float) -> int:
        """How many rule steps of ``dt`` make up one brain step of ``duration_ms``.

        The traces and rate normalisations assume one ``update`` per ``dt`` of
        simulated time (spec §3.3, 2 ms), so a caller stepping the brain in larger
        chunks must split them into this many ``dt`` sub-steps.
        """
        n = round(duration_ms / (self.dt * 1000.0))
        if n < 1 or not math.isclose(n * self.dt * 1000.0, duration_ms, abs_tol=1e-9):
            raise ValueError(f'step of {duration_ms} ms is not a whole multiple of the '
                             f'plasticity rule dt ({self.dt * 1000.0} ms)')
        return n

    def update(
        self,
        delta: np.ndarray,
        pre_counts: np.ndarray,
        post_counts: np.ndarray,
        full_counts: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> None:
        """Update plastic weight deltas in place.

        delta is added to the initial negative weights w0 in GraphInstance._materialize:
            w_eff = w0 + delta
        Since w0 < 0 and depression decreases magnitude |w_eff| toward 0:
            delta >= 0
            delta <= |w0| = g0
        Sign flip is strictly forbidden.
        """
        # 1. Update presynaptic eligibility trace a_i
        s_hat_i = pre_counts.astype(np.float32) / (self.dt * self.s_sat)
        alpha_pre = 1.0 - math.exp(-self.dt / self.tau_pre)
        self.pre_trace += alpha_pre * (np.clip(s_hat_i, 0.0, 1.0) - self.pre_trace)

        # 2. Update modulatory trace m (from EL population if full_counts is available)
        if full_counts is not None and len(self.el_nodes) > 0:
            el_spikes = float(full_counts[self.el_nodes].sum())
            el_rate = el_spikes / (len(self.el_nodes) * self.dt)
            m_hat = min(1.0, el_rate / self.r_sat)
            alpha_mod = 1.0 - math.exp(-self.dt / self.tau_mod)
            self.mod_trace += alpha_mod * (m_hat - self.mod_trace)
        elif 'modulator_rate' in kwargs:
            m_hat = min(1.0, float(kwargs['modulator_rate']) / self.r_sat)
            alpha_mod = 1.0 - math.exp(-self.dt / self.tau_mod)
            self.mod_trace += alpha_mod * (m_hat - self.mod_trace)
        elif self.mod_trace == 0.0 and len(self.el_nodes) == 0:
            # Fallback for synthetic/isolated test harnesses when no modulator is present
            self.mod_trace = 1.0

        # 3. Depression and recovery update
        # g = current conductance magnitude = |w0| - delta
        g = np.maximum(0.0, self.g0 - delta)
        # Delta g_e = -eta * a_i * m * g_e + (g0 - g_e) * dt / tau_rec
        # Since delta = g0 - g, Delta delta = -Delta g_e
        delta_delta = (
            self.eta * self.pre_trace * self.mod_trace * g
            - delta * (self.dt / self.tau_rec)
        )
        delta += delta_delta

        # 4. Invariants: depression only, sign never flips
        # w_eff = w0 + delta <= 0 ==> delta <= g0 * (1 - g_min)
        # delta >= 0 (no potentiation beyond initial anatomy)
        max_delta = self.g0 * (1.0 - self.g_min)
        np.clip(delta, 0.0, max_delta, out=delta)
