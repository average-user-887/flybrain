"""Check the redaction on the real (FlyGym + MaleCNS) studio runs' files."""
import sys
from pathlib import Path

from neurofly_studio.redact import leaks, redact_bytes

runs = Path(sys.argv[1])
for run in sorted(runs.iterdir()):
    for name in ("manifest.json", "summary.json", "body.nfbody", "timing.jsonl", "telemetry.jsonl"):
        f = run / name
        if not f.is_file():
            continue
        raw = f.read_bytes()
        before, after = len(leaks(raw, name)), leaks(redact_bytes(name, raw), name)
        print(run.name, name, "leaks before", before, "after", len(after), after[:2])
