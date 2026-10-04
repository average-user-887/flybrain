import json
import os
import sys

sys.path.insert(0, '<redacted-path>/Documents/ChatGPT/flybrain/<redacted-path>/.wt-graded')
from brainlab.photoreceptor_io import resolve_photoreceptor_io

io = resolve_photoreceptor_io(pin=None)
print(json.dumps(io.describe(), indent=2, default=str))
print('PIN', io.sha256)
