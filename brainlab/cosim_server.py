"""
Drosophila Whole-Brain RPC Co-Simulation Server (MaleCNS v1.0)
============================================================
Exposes high-performance HTTP endpoints for running the 166,700-neuron,
25.58M-synapse MaleCNS v1.0 spiking engine on local or remote compute hosts.

Endpoints:
- GET  /status : Returns system health, dataset info, neuron count, spike telemetry.
- POST /reset  : Resets membrane potentials and queues to resting state (-52 mV).
- POST /step   : Ingests continuous sensory inputs, injects currents into sensory
                 nodes, steps LIF graph kernel, and decodes descending neuron spikes.
"""

import argparse
import json
import math
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

# Root path resolution
ROOT = Path(__file__).resolve().parents[1]
try:
    from .brain import Brain
except ImportError:
    try:
        from brainlab.brain import Brain
    except ImportError:
        try:
            from brain import Brain
        except ImportError:
            Brain = None


try:
    from .graph_identity import (DN_CHANNELS, SYNTHETIC_LABEL, GraphUnavailable, dynamics_pin,
                                 DYNAMICS_VERSIONS, active_dynamics_version,
                                 resolve_connectome_dir, resolve_graph_dir, sha256_json,
                                 synthetic_test_graph, verify_graph)
except ImportError:
    from brainlab.graph_identity import (DN_CHANNELS, SYNTHETIC_LABEL, GraphUnavailable, dynamics_pin,
                                         DYNAMICS_VERSIONS, active_dynamics_version,
                                         resolve_connectome_dir, resolve_graph_dir, sha256_json,
                                         synthetic_test_graph, verify_graph)

try:
    from .transmitter_policy import (POLICIES, POLICY_LEGACY, POLICY_V3, UNCLEAR_MODES,
                                     describe as describe_transmitter_policy)
except ImportError:
    from brainlab.transmitter_policy import (POLICIES, POLICY_LEGACY, POLICY_V3, UNCLEAR_MODES,
                                             describe as describe_transmitter_policy)

# Engineered inputs that bypass sensory pathways; reported, never hidden (WP5).
# Gated by ConnectomeServer(engineered_assistance=...) / --no-engineered-assistance;
# every /step reply lists the ones actually applied on that step.
ENGINEERED_ASSISTANCE = [
    'looming_trigger injects 80 directly into DNp01 (not via LC4/LPLC2)',
    'tonic drive 14.5*(1+0.5*(1-energy)) injected directly into DNb01',
]
MAPPING_WARNINGS = [
    'visual_l/visual_r drive R1-R6 photoreceptors of one eye each (annotated rootSide, 100 lowest body IDs); '
    'this is a non-directional luminance drive, not HS/optic-flow input. The eye-specific optomotor '
    'pathway is the optional optomotor_slip_rad_s input (brainlab/io_map.py, docs/WP5_OPTOMOTOR.md).',
]


class ConnectomeServer:
    """Fixed-weight graph server. Refuses to start without the verified graph.

    ``allow_synthetic=True`` (CLI ``--synthetic-test-graph``) is the only way to
    run on a synthetic graph; every status and step reply then carries
    ``synthetic: true`` and the synthetic label.
    """

    def __init__(self, graph_path: Optional[Path] = None, metadata_path: Optional[Path] = None, *,
                 graph_dir: Optional[Path] = None, connectome_dir: Optional[Path] = None,
                 allow_synthetic: bool = False, engineered_assistance: bool = True,
                 optomotor_seed: int = 0, dynamics: Optional[str] = None,
                 transmitter_policy: Optional[str] = None, unclear_mode: str = 'excitatory'):
        """Create a fixed-weight controller with an explicit LIF/policy pairing.

        ``dynamics`` defaults to the declared process setting. v1/v2 keep the
        prepared graph unchanged; v3 defaults to ``v3-modulatory-only``, which
        derives a distinct in-memory graph identity without altering graph.npz.
        """
        if graph_path is not None:
            graph_path = Path(graph_path)
            if graph_path.name != "graph.npz":
                raise GraphUnavailable(f"Expected a prepared graph.npz, got {graph_path}")
            graph_dir = graph_path.parent
        if metadata_path is not None:
            connectome_dir = Path(metadata_path).parent.parent
        self.graph_dir, self.graph_dir_source = resolve_graph_dir(graph_dir)
        self.connectome_dir, _ = resolve_connectome_dir(connectome_dir)
        self.graph_path = self.graph_dir / "graph.npz"
        self.metadata_path = self.connectome_dir / "normalized/neurons.feather"
        self.allow_synthetic = allow_synthetic
        self.engineered_assistance = bool(engineered_assistance)
        self.optomotor_seed = int(optomotor_seed)
        self.dynamics = dynamics if dynamics is not None else active_dynamics_version()
        if self.dynamics not in DYNAMICS_VERSIONS:
            raise ValueError(f"Unknown dynamics version {self.dynamics!r}; "
                             f"declared: {sorted(DYNAMICS_VERSIONS)}")
        self.transmitter_policy = (transmitter_policy if transmitter_policy is not None
                                   else (POLICY_V3 if self.dynamics == 'v3' else POLICY_LEGACY))
        if self.transmitter_policy not in POLICIES:
            raise ValueError(f"Unknown transmitter policy {self.transmitter_policy!r}; "
                             f"declared: {list(POLICIES)}")
        if unclear_mode not in UNCLEAR_MODES:
            raise ValueError(f"Unknown unclear_mode {unclear_mode!r}; declared: {list(UNCLEAR_MODES)}")
        if self.dynamics != 'v3' and self.transmitter_policy != POLICY_LEGACY:
            raise ValueError(f"LIF dynamics {self.dynamics!r} requires the legacy prepared weights; "
                             f"{POLICY_V3!r} is only valid with dynamics='v3'")
        self.unclear_mode = unclear_mode
        self.transmitter_policy_report = describe_transmitter_policy(
            self.transmitter_policy, self.unclear_mode)
        self.optomotor = None          # (io map, encoder, decoder) on the real graph only
        self.shared_graph = None        # retains transformed v3 arrays for this controller
        self.brain = None
        self.identity = None
        self.n_neurons = 0
        self.n_edges = 0
        self.start_time = time.time()
        self.total_steps = 0
        self.is_synthetic = False
        self.unmapped_channels: List[str] = []

        # Node index mapping for Descending Neurons (verified against cell types
        # by graph_identity.verify_graph before any step is served).
        self.dn_indices = {name: list(indices) for name, indices in DN_CHANNELS.items()}

        # Sensory node index mapping
        self.sensory_indices = {
            "orn_food": [],                       # DM1/DM2 food odorants
            "orn_danger": [],                     # DA2/geosmin/acid
            "visual_l": [],                       # see MAPPING_WARNINGS
            "visual_r": [],                       # see MAPPING_WARNINGS
            "visual_looming": [],                 # LC4 / LPLC2 looming
            "jon_wind": [],                       # Johnston's organ wind mechanoreception
            "feco_proprio": [],                   # Leg chordotonal organs & sensilla
            "courtship_cva": [],                  # ORN_DA1 pheromone
            "thermo_receptors": [],               # Antennal thermosensory receptor neurons (TRN)
        }

        self._load_brain()
        self._load_metadata()
        self.sensory_map_sha256 = sha256_json(self.sensory_indices)

    def _load_brain(self):
        try:
            self.identity = verify_graph(self.graph_dir, self.connectome_dir)
            self.identity.graph_path_source = self.graph_dir_source
        except GraphUnavailable as error:
            if not self.allow_synthetic:
                raise
            print(f"[ConnectomeServer] {error}", flush=True)
            self._init_synthetic_brain()
            return
        print(f"[ConnectomeServer] Loading verified MaleCNS v1.0 graph {self.identity.graph_sha256[:12]} "
              f"from {self.graph_path} ({self.graph_dir_source})...", flush=True)
        if self.transmitter_policy == POLICY_V3:
            # This mirrors scripts/wp5_optomotor.py: validate/load the pinned
            # graph, apply the declared policy only in memory, and use the
            # transformed identity. graph.npz remains byte-for-byte intact.
            try:
                from experiment_registry import SharedGraph
                from .transmitter_policy import apply_to_shared
            except ImportError:
                from experiment_registry import SharedGraph
                from brainlab.transmitter_policy import apply_to_shared
            self.shared_graph = SharedGraph.load(self.graph_dir, self.connectome_dir)
            self.shared_graph, self.transmitter_policy_report = apply_to_shared(
                self.shared_graph, policy=self.transmitter_policy,
                unclear_mode=self.unclear_mode, connectome_dir=self.connectome_dir)
            self.identity = self.shared_graph.identity
            self.identity.graph_path_source = self.graph_dir_source
            self.brain = Brain(arrays=self.shared_graph.arrays, validate=False, dynamics=self.dynamics)
        else:
            self.brain = Brain(self.graph_path, dynamics=self.dynamics)
        self.n_neurons = self.brain.n
        self.n_edges = len(self.brain.post)
        print(f"[ConnectomeServer] Brain loaded: {self.n_neurons:,} neurons, {self.n_edges:,} synapses.", flush=True)

    def _init_synthetic_brain(self):
        """Explicit test option only: a small in-memory graph, never written to disk."""
        self.is_synthetic = True
        arrays, self.identity, io_map = synthetic_test_graph(n=2500, k_out=15, seed=42)
        # A synthetic graph has no released transmitter table. It is kept
        # explicitly synthetic rather than pretending a MaleCNS policy was
        # applied; the requested policy is still carried in every identity.
        self.transmitter_policy_report = describe_transmitter_policy(
            self.transmitter_policy, self.unclear_mode)
        self.transmitter_policy_report.update(
            applied_to_weights=False,
            note='synthetic test graph has no MaleCNS transmitter metadata; no policy transform applied')
        self.brain = Brain(arrays=arrays, dynamics=self.dynamics)
        self.n_neurons = self.brain.n
        self.n_edges = len(self.brain.post)
        self.dn_indices = io_map
        print(f"[ConnectomeServer] {SYNTHETIC_LABEL}: {self.n_neurons} neurons, {self.n_edges} synapses.", flush=True)

    def _load_metadata(self):
        if self.is_synthetic:
            self.unmapped_channels = sorted(self.sensory_indices)
            return
        import pyarrow.feather as feather
        df = feather.read_table(self.metadata_path).to_pandas()
        # No invented fallback indices: an absent cell type leaves the channel
        # empty and listed in unmapped_channels.
        self.sensory_indices["orn_food"] = df[df['cell_type'] == 'ORN_DM1']['node_index'].tolist()[:50]
        self.sensory_indices["orn_danger"] = df[df['cell_type'] == 'ORN_DA2']['node_index'].tolist()[:50]
        # Eye-specific photoreceptors by annotated rootSide (R1-R6 have no soma
        # side), 100 lowest body IDs per eye; never dataframe halves.
        ann_path = self.connectome_dir / "annotations.feather"
        if ann_path.is_file():
            ann = feather.read_table(ann_path, columns=['bodyId', 'rootSide']).to_pandas()
            root_side = dict(zip(ann.bodyId.astype('int64'), ann.rootSide))
            retina = df[df['cell_type'] == 'R1-R6'].sort_values('source_id')
            sides = retina['source_id'].map(root_side)
            self.sensory_indices["visual_l"] = retina[sides == 'L']['node_index'].tolist()[:100]
            self.sensory_indices["visual_r"] = retina[sides == 'R']['node_index'].tolist()[:100]
        try:
            from .io_map import DNa02YawDecoder, OptomotorEncoder, resolve_optomotor_io
        except ImportError:
            from brainlab.io_map import DNa02YawDecoder, OptomotorEncoder, resolve_optomotor_io
        try:
            io = resolve_optomotor_io(self.connectome_dir)
            self.optomotor = (io, OptomotorEncoder(io, np.random.default_rng(self.optomotor_seed)),
                              DNa02YawDecoder(io))
        except GraphUnavailable as error:
            print(f"[ConnectomeServer] optomotor IO map unavailable: {error}", flush=True)
        self.sensory_indices["jon_wind"] = df[df['cell_type'].str.contains('JO-', na=False)]['node_index'].tolist()[:50]
        self.sensory_indices["feco_proprio"] = df[df['cell_type'].str.contains('SNta', na=False)]['node_index'].tolist()[:50]
        self.sensory_indices["visual_looming"] = df[df['cell_type'].isin(['LC4', 'LPLC2'])]['node_index'].tolist()
        self.sensory_indices["courtship_cva"] = df[df['cell_type'] == 'ORN_DA1']['node_index'].tolist()[:50]
        if ann_path.is_file():
            ann_thermo = feather.read_table(ann_path, columns=['bodyId', 'class']).to_pandas()
            thermo_ids = set(ann_thermo[ann_thermo['class'] == 'thermosensory']['bodyId'].astype('int64'))
            self.sensory_indices["thermo_receptors"] = df[df['source_id'].isin(thermo_ids)]['node_index'].tolist()
        self.unmapped_channels = sorted(k for k, v in self.sensory_indices.items() if not v)
        print(f"[ConnectomeServer] Sensory indices mapped from {self.metadata_path.name}; "
              f"unmapped: {self.unmapped_channels}", flush=True)

    def identity_fields(self) -> Dict[str, Any]:
        ident = self.identity
        return {
            "backend": "synthetic-test-graph" if self.is_synthetic else "connectome-fixed",
            "synthetic": self.is_synthetic,
            "label": ident.label,
            "graph_sha256": ident.graph_sha256,
            "neuron_map_sha256": ident.neuron_map_sha256,
            "io_map_sha256": ident.io_map_sha256,
            "sensory_map_sha256": getattr(self, "sensory_map_sha256", None),
            # graph_sha256 identifies the exact effective CSR arrays; paired
            # with this declaration it cannot confuse prepared and v3
            # policy-transformed fast weights.
            "transmitter_policy": self.transmitter_policy,
            "transmitter_policy_report": self.transmitter_policy_report,
            "engineered_assistance_enabled": self.engineered_assistance,
            "optomotor_io_map_sha256": self.optomotor[0].sha256 if self.optomotor else None,
            # A dynamics change is a new controller version (docs/LIF_DYNAMICS_SPEC.md):
            # telemetry must never leave which engine produced a spike ambiguous.
            "lif_dynamics_version": self.brain.dynamics,
            "lif_dynamics_pin": dynamics_pin(self.brain.dynamics),
            "controller_version": ('synthetic-test-v1' if self.is_synthetic
                                   else f'brainlab-lif-{self.brain.dynamics}'),
        }

    def reset(self):
        """Resets membrane potentials and conductances to resting state."""
        self.brain.v.fill(-52.0)
        self.brain.g.fill(0.0)
        self.brain.refractory.fill(0)
        self.brain.queue.fill(0)
        self.brain.queue_count.fill(0)
        self.brain.counts.fill(0)
        self.brain.active.fill(0)
        self.brain.active_flag.fill(0)
        self.brain.nactive.fill(0)
        self.brain.cursor = 0
        self.brain.total_spikes = 0
        self.brain.sim_ms = 0.0
        if self.optomotor is not None:
            self.optomotor[2].reset()

    def step(self, sensory: Dict[str, Any], duration_ms: float = 2.0) -> Dict[str, Any]:
        """
        Translates arena sensory observations into synaptic currents,
        steps the LIF connectome graph, and decodes descending motor commands.
        """
        currents = np.zeros(self.n_neurons, dtype=np.float32)

        # 1. Olfactory drive
        mean_odor = float(sensory.get("mean_odor", 0.0))
        if mean_odor > 0.01:
            i_food = float(min(45.0, mean_odor * 30.0))
            for idx in self.sensory_indices["orn_food"]:
                if idx < self.n_neurons:
                    currents[idx] += i_food

        # 2. Visual Optic Flow (HS/VS)
        hs_l = float(sensory.get("hs_left_shunted", sensory.get("hs_left_raw", 0.0)))
        hs_r = float(sensory.get("hs_right_shunted", sensory.get("hs_right_raw", 0.0)))
        for idx in self.sensory_indices["visual_l"]:
            if idx < self.n_neurons:
                currents[idx] += float(max(0.0, hs_l * 12.0))
        for idx in self.sensory_indices["visual_r"]:
            if idx < self.n_neurons:
                currents[idx] += float(max(0.0, hs_r * 12.0))

        applied_assistance: List[str] = []

        # 2b. Eye-specific optomotor input (WP5): retinal slip -> T4/T5 by eye.
        optomotor_slip = sensory.get("optomotor_slip_rad_s")
        if optomotor_slip is not None and self.optomotor is not None:
            io, encoder, _ = self.optomotor
            encoder.encode(currents, self.brain.sim_ms, float(optomotor_slip),
                           float(sensory.get("optomotor_contrast", 1.0)))

        # 3. Visual Looming (LC4/LPLC2 -> Giant Fiber)
        looming = bool(sensory.get("looming_trigger", False)) or float(sensory.get("looming_expansion_rate", 0.0)) > 0.1
        if looming:
            for idx in self.sensory_indices["visual_looming"]:
                if idx < self.n_neurons:
                    currents[idx] += 30.0

        if self.engineered_assistance and sensory.get("looming_trigger", False):
            applied_assistance.append(ENGINEERED_ASSISTANCE[0])
            for idx in self.dn_indices["dnp01"]:
                if idx < self.n_neurons:
                    currents[idx] += 80.0  # Supra-threshold escape trigger

        # 3b. Antennal Thermoreception (TRN -> SEZ)
        temp_val = float(sensory.get("temperature_excess", max(0.0, float(sensory.get("temperature_c", 25.0)) - 25.0)))
        if temp_val > 0.5:
            i_thermo = float(min(40.0, temp_val * 4.0))
            for idx in self.sensory_indices["thermo_receptors"]:
                if idx < self.n_neurons:
                    currents[idx] += i_thermo

        # 3c. Courtship Pheromone (cVA -> ORN_DA1)
        cva_val = float(sensory.get("pheromone_cva", sensory.get("courtship_cva", 0.0)))
        if cva_val > 0.01:
            i_cva = float(min(40.0, cva_val * 35.0))
            for idx in self.sensory_indices["courtship_cva"]:
                if idx < self.n_neurons:
                    currents[idx] += i_cva

        # 4. Wind mechanoreception (Johnston's organ drag)
        wind_speed = float(sensory.get("wpn_wind_speed", 0.0))
        if wind_speed > 5.0:
            i_wind = float(min(35.0, wind_speed * 0.15))
            for idx in self.sensory_indices["jon_wind"]:
                if idx < self.n_neurons:
                    currents[idx] += i_wind

        # 5. Baseline tonic peduncular current (BPN straight walking drive)
        # Keeps Drosophila motor system in active exploration state
        if self.engineered_assistance:
            applied_assistance.append(ENGINEERED_ASSISTANCE[1])
            bpn_tonic = 14.5 * (1.0 + 0.5 * (1.0 - float(sensory.get("energy_reserve", 1.0))))
            for idx in self.dn_indices["dnb01"]:
                if idx < self.n_neurons:
                    currents[idx] += bpn_tonic

        # 6. Step the LIF connectome kernel
        spike_counts, elapsed_s = self.brain.step(currents, duration_ms)
        self.total_steps += 1

        # 7. Decode Descending Neuron Activity (spikes / duration -> Hz)
        sec = duration_ms / 1000.0

        # DNa02 fine yaw steering
        spk_dna02_l = sum(spike_counts[i] for i in self.dn_indices["dna02_l"] if i < self.n_neurons)
        spk_dna02_r = sum(spike_counts[i] for i in self.dn_indices["dna02_r"] if i < self.n_neurons)
        dna02_rate_l = spk_dna02_l / sec
        dna02_rate_r = spk_dna02_r / sec
        # Steering convention shared with the daemon and neurofly_body: DNa02 drives
        # ipsilateral turning, and yaw is + counter-clockwise (a left turn), so the
        # steering signal is L - R.
        dna02_diff = float(dna02_rate_l - dna02_rate_r)

        # DNp09 pursuit forward drive
        spk_dnp09 = sum(spike_counts[i] for i in self.dn_indices["dnp09"] if i < self.n_neurons)
        dnp09_rate = float(spk_dnp09 / (len(self.dn_indices["dnp09"]) * sec)) if self.dn_indices["dnp09"] else 0.0

        # BPN / DNb01 straight walking cadence
        spk_dnb01 = sum(spike_counts[i] for i in self.dn_indices["dnb01"] if i < self.n_neurons)
        bpn_rate = float(spk_dnb01 / (len(self.dn_indices["dnb01"]) * sec)) if self.dn_indices["dnb01"] else 0.0

        # MDN Moonwalker backward walking
        spk_mdn = sum(spike_counts[i] for i in self.dn_indices["mdn"] if i < self.n_neurons)
        mdn_rate = float(spk_mdn / (len(self.dn_indices["mdn"]) * sec)) if self.dn_indices["mdn"] else 0.0

        # DNp01 Giant Fiber spikes
        dnp01_gf_spikes = int(sum(spike_counts[i] for i in self.dn_indices["dnp01"] if i < self.n_neurons))

        optomotor_reply = None
        if self.optomotor is not None:
            motor = self.optomotor[2].decode(spike_counts, duration_ms)
            optomotor_reply = {
                "slip_rad_s": None if optomotor_slip is None else float(optomotor_slip),
                "yaw_rad_s": motor["yaw_rad_s"], "contributions": motor["contributions"],
                "rate_l": motor["rate_l"], "rate_r": motor["rate_r"],
                "io_map_sha256": self.optomotor[0].sha256,
                "decoder": "yaw = 0.02*(rate DNa02_L - rate DNa02_R), + = counter-clockwise; unclipped",
            }

        return {
            "status": "ok",
            **self.identity_fields(),
            "server_step": self.total_steps,
            "elapsed_ms": elapsed_s * 1000.0,
            "sim_ms": self.brain.sim_ms,
            "total_step_spikes": int(spike_counts.sum()),
            "dna02_diff": dna02_diff,
            "dna02_rate_l": dna02_rate_l,
            "dna02_rate_r": dna02_rate_r,
            "dnp09_rate": dnp09_rate,
            "bpn_rate": bpn_rate,
            "mdn_rate": mdn_rate,
            "dnp01_gf_spikes": dnp01_gf_spikes,
            "engineered_assistance_applied": applied_assistance,
            "optomotor": optomotor_reply,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": "online",
            **self.identity_fields(),
            "graph_path": self.identity.graph_path,
            "graph_path_source": self.identity.graph_path_source,
            "unmapped_channels": self.unmapped_channels,
            "mapping_warnings": MAPPING_WARNINGS,
            "engineered_assistance": ENGINEERED_ASSISTANCE,
            "dataset": "malecns_v1" if not self.is_synthetic else "synthetic_test_graph",
            "num_neurons": self.n_neurons,
            "num_synapses": self.n_edges,
            "uptime_s": time.time() - self.start_time,
            "total_steps": self.total_steps,
            "total_spikes": self.brain.total_spikes if self.brain else 0,
            "sim_ms": self.brain.sim_ms if self.brain else 0.0,
            "brunel_scaled": True
        }


class CoSimHTTPHandler(BaseHTTPRequestHandler):
    server: Any  # Typing helper for self.server.connectome

    def send_json(self, code: int, payload: Dict[str, Any]):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/status" or self.path == "/api/status":
            return self.send_json(200, self.server.connectome.get_status())
        return self.send_json(404, {"error": f"Path not found: {self.path}"})

    def do_POST(self):
        if self.path == "/reset":
            self.server.connectome.reset()
            return self.send_json(200, {"status": "reset_complete",
                                        **self.server.connectome.identity_fields()})

        if self.path == "/step":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                request_data = json.loads(body.decode("utf-8")) if body else {}
                sensory = request_data.get("sensory", {})
                duration_ms = float(request_data.get("duration_ms", 2.0))
                result = self.server.connectome.step(sensory, duration_ms)
                return self.send_json(200, result)
            except Exception as e:
                return self.send_json(500, {"error": str(e)})

        return self.send_json(404, {"error": f"Path not found: {self.path}"})

    def log_message(self, format, *args):
        # Silence routine request access logs to preserve stdout clarity
        pass


def run_server(host: str = "0.0.0.0", port: int = 8768, *, graph_dir: Optional[Path] = None,
               connectome_dir: Optional[Path] = None, allow_synthetic: bool = False,
               engineered_assistance: bool = True):
    connectome = ConnectomeServer(graph_dir=graph_dir, connectome_dir=connectome_dir,
                                  allow_synthetic=allow_synthetic, engineered_assistance=engineered_assistance)
    server = ThreadingHTTPServer((host, port), CoSimHTTPHandler)
    server.connectome = connectome  # type: ignore
    print(f"[ConnectomeServer] Serving on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ConnectomeServer] Shutting down...", flush=True)
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MaleCNS v1.0 Spiking Co-Simulation Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host address")
    parser.add_argument("--port", type=int, default=8768, help="Listening port")
    parser.add_argument("--graph-dir", type=Path, help="Directory holding graph.npz (else $NEUROFLY_GRAPH_DIR)")
    parser.add_argument("--connectome-dir", type=Path, help="connectome_data/malecns_v1 (else $NEUROFLY_CONNECTOME_DIR)")
    parser.add_argument("--synthetic-test-graph", action="store_true",
                        help="TEST ONLY: serve a labelled synthetic graph if the real graph is unavailable")
    parser.add_argument("--no-engineered-assistance", action="store_true",
                        help="Disable the direct DNp01 looming injection and tonic DNb01 drive (causal tests)")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, graph_dir=args.graph_dir,
               connectome_dir=args.connectome_dir, allow_synthetic=args.synthetic_test_graph,
               engineered_assistance=not args.no_engineered_assistance)
