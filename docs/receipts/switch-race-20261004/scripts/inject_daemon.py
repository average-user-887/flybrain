"""Fixed daemon with a one-shot fault hook: touching FLAG makes the next graph step raise."""
import os, sys
import neurofly_daemon as nd

FLAG = "<scratch>/race/inject.flag"
_orig = nd.GraphArenaController.__call__


def faulty(self, *a, **k):
    if os.path.exists(FLAG):
        os.remove(FLAG)
        raise RuntimeError("injected test fault (one-shot)")
    return _orig(self, *a, **k)


nd.GraphArenaController.__call__ = faulty
sys.exit(nd.run_daemon())
