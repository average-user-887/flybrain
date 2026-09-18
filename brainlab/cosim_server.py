"""
Drosophila Whole-Brain RPC Co-Simulation Server (MaleCNS v1.0)
============================================================
Exposes high-performance HTTP endpoints for running the 166,700-neuron,
25.58M-synapse MaleCNS v1.0 spiking engine on Ryzen / remote compute nodes.

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


class ConnectomeServer:
    def __init__(self, graph_path: Optional[Path] = None, metadata_path: Optional[Path] = None):
        self.graph_path = graph_path or (ROOT / "outputs/brainlab/malecns_v1/graph.npz")
        self.metadata_path = metadata_path or (ROOT / "connectome_data/malecns_v1/normalized/neurons.feather")
        self.brain = None
        self.n_neurons = 0
        self.n_edges = 0
        self.start_time = time.time()
        self.total_steps = 0
        self.is_synthetic = False

        # Node index mapping for Descending Neurons
        self.dn_indices = {
            "dnp01": [0, 6],                      # Giant Fiber escape
            "dna02_l": [332],                     # Fine steering yaw (left)
            "dna02_r": [131957],                  # Fine steering yaw (right)
            "dna01": [406, 704],                  # Course holding
            "dnp09": [725, 1087],                 # Pursuit forward drive
            "dnb01": [608, 703],                  # Straight walking cadence (BPN proxy)
            "mdn": [706, 1196, 1240, 2194]        # Moonwalker backward walking
        }

        # Sensory node index mapping
        self.sensory_indices = {
            "orn_food": [],                       # DM1/DM2 food odorants
            "orn_danger": [],                     # DA2/geosmin/acid
            "visual_l": [],                       # Left compound eye / optic flow
            "visual_r": [],                       # Right compound eye / optic flow
            "visual_looming": [],                 # LC4 / LPLC2 looming
            "jon_wind": [],                       # Johnston's organ wind mechanoreception
            "feco_proprio": []                    # Leg chordotonal organs & sensilla
        }

        self._load_brain()
        self._load_metadata()

    def _load_brain(self):
        if self.graph_path.exists():
            print(f"[ConnectomeServer] Loading MaleCNS v1.0 graph from {self.graph_path}...", flush=True)
            self.brain = Brain(self.graph_path)
            self.n_neurons = self.brain.n
            self.n_edges = len(self.brain.post)
            print(f"[ConnectomeServer] Brain loaded: {self.n_neurons:,} neurons, {self.n_edges:,} synapses.", flush=True)
        else:
            print(f"[ConnectomeServer] Graph file not found at {self.graph_path}. Initializing synthetic surrogate graph...", flush=True)
            self._init_synthetic_brain()

    def _init_synthetic_brain(self):
        """Creates a minimal synthetic graph for test environments."""
        self.is_synthetic = True
        n = 2500
        np.random.seed(42)
        k_out = 15
        post = np.random.randint(0, n, size=n * k_out, dtype=np.int32)
        ptr = np.arange(0, (n + 1) * k_out, k_out, dtype=np.int64)
        weight = np.random.randn(n * k_out).astype(np.float32) * 0.35
        ids = np.arange(n, dtype=np.int64)

        synthetic_dir = ROOT / "outputs/brainlab"
        synthetic_dir.mkdir(parents=True, exist_ok=True)
        synth_path = synthetic_dir / "synthetic_test_graph.npz"
        np.savez(synth_path, ptr=ptr, post=post, weight=weight, ids=ids)

        self.brain = Brain(synth_path)
        self.n_neurons = self.brain.n
        self.n_edges = len(self.brain.post)
        self.dn_indices = {
            "dnp01": [0, 1],
            "dna02_l": [10],
            "dna02_r": [11],
            "dna01": [12, 13],
            "dnp09": [14, 15],
            "dnb01": [16, 17],
            "mdn": [18, 19]
        }
        print(f"[ConnectomeServer] Synthetic brain initialized: {self.n_neurons} neurons, {self.n_edges} synapses.", flush=True)

    def _load_metadata(self):
        if self.is_synthetic or not self.metadata_path.exists():
            return
        try:
            import pyarrow.feather as feather
            df = feather.read_table(self.metadata_path).to_pandas()
            # Olfactory
            dm1_nodes = df[df['cell_type'] == 'ORN_DM1']['node_index'].tolist()
            da2_nodes = df[df['cell_type'] == 'ORN_DA2']['node_index'].tolist()
            self.sensory_indices["orn_food"] = dm1_nodes[:50] if dm1_nodes else [100, 101]
            self.sensory_indices["orn_danger"] = da2_nodes[:50] if da2_nodes else [102, 103]

            # Visual
            r_nodes = df[df['cell_type'] == 'R1-R6']['node_index'].tolist()
            if r_nodes:
                half = len(r_nodes) // 2
                self.sensory_indices["visual_l"] = r_nodes[:half][:100]
                self.sensory_indices["visual_r"] = r_nodes[half:][:100]

            # Johnston's organ
            jo_nodes = df[df['cell_type'].str.contains('JO-', na=False)]['node_index'].tolist()
            self.sensory_indices["jon_wind"] = jo_nodes[:50] if jo_nodes else [104, 105]

            # Proprioception
            snt_nodes = df[df['cell_type'].str.contains('SNta', na=False)]['node_index'].tolist()
            self.sensory_indices["feco_proprio"] = snt_nodes[:50] if snt_nodes else [106, 107]

            print(f"[ConnectomeServer] Annotated sensory indices mapped from {self.metadata_path.name}.", flush=True)
        except Exception as err:
            print(f"[ConnectomeServer] Metadata loading skipped ({err}); using default fallback index mapping.", flush=True)

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

        # 3. Visual Looming (LC4/LPLC2 -> Giant Fiber)
        if sensory.get("looming_trigger", False):
            for idx in self.dn_indices["dnp01"]:
                if idx < self.n_neurons:
                    currents[idx] += 80.0  # Supra-threshold escape trigger

        # 4. Wind mechanoreception (Johnston's organ drag)
        wind_speed = float(sensory.get("wpn_wind_speed", 0.0))
        if wind_speed > 5.0:
            i_wind = float(min(35.0, wind_speed * 0.15))
            for idx in self.sensory_indices["jon_wind"]:
                if idx < self.n_neurons:
                    currents[idx] += i_wind

        # 5. Baseline tonic peduncular current (BPN straight walking drive)
        # Keeps Drosophila motor system in active exploration state
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
        dna02_diff = float(dna02_rate_r - dna02_rate_l)

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

        return {
            "status": "ok",
            "elapsed_ms": elapsed_s * 1000.0,
            "sim_ms": self.brain.sim_ms,
            "total_step_spikes": int(spike_counts.sum()),
            "dna02_diff": dna02_diff,
            "dna02_rate_l": dna02_rate_l,
            "dna02_rate_r": dna02_rate_r,
            "dnp09_rate": dnp09_rate,
            "bpn_rate": bpn_rate,
            "mdn_rate": mdn_rate,
            "dnp01_gf_spikes": dnp01_gf_spikes
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": "online",
            "dataset": "malecns_v1" if not self.is_synthetic else "synthetic_surrogate",
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
            return self.send_json(200, {"status": "reset_complete"})

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


def run_server(host: str = "0.0.0.0", port: int = 8768):
    connectome = ConnectomeServer()
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
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
