import os

# The CPU kernel is the reference every test's expected values were derived
# from.  Brain() defaults to the GPU when one is usable, so pin the reference
# here; tests/test_cuda_engine.py selects the GPU explicitly.
os.environ.setdefault('NEUROFLY_BRAIN_BACKEND', 'cpu')
