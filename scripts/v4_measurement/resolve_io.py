from _paths import REPO, SCRATCH  # repo root; scratch = $NEUROFLY_V4_SCRATCH or outputs/v4_measurement
import json
import os
import sys

sys.path.insert(0, str(REPO))
from brainlab.photoreceptor_io import resolve_photoreceptor_io

io = resolve_photoreceptor_io(pin=None)
print(json.dumps(io.describe(), indent=2, default=str))
print('PIN', io.sha256)
