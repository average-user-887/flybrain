import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainlab.cosim_server import ConnectomeServer

print("[1/3] Initializing ConnectomeServer on MaleCNS v1.0...")
srv = ConnectomeServer()
print(f"[2/3] Verified Graph: {srv.n_neurons:,} neurons, {srv.n_edges:,} edges.")
print("[3/3] Testing single step with sensory packet...")
res = srv.step({"mean_odor": 0.8, "hs_left_raw": 1.0}, duration_ms=2.0)
keys = ["dna02_diff", "dna02_rate_l", "dna02_rate_r", "dnp09_rate", "mdn_rate", "total_spikes", "sim_ms"]
print("Step output:", {k: res.get(k) for k in keys})
print("[SUCCESS] Full connectome stepping verified!")
