"""RETIRED (2026-10-05): manual full-graph ConnectomeServer smoke test.

The old script loaded the whole MaleCNS graph and stepped it at module top level,
so merely importing it (or asking for ``--help``) did the heavy work. It is now
inert: importing it does nothing, and running it prints this notice and exits 2.
The original source is pinned at its last historical commit, listed in
docs/RETIREMENT_INDEX.md.
"""
import sys

RETIRED_NOTICE = """\
scripts/test_srv.py is RETIRED and does not run. It loaded and stepped the full
MaleCNS graph as a side effect of being imported, whatever the arguments (even --help).

Use instead:
  neurofly status          verifies the graph identity and the compute backend
  pytest tests/            ConnectomeServer is exercised by tests in tests/

Last historical commit and details: docs/RETIREMENT_INDEX.md
"""


def main(argv=None) -> int:
    """Print the retirement notice (``--help`` shows it too) and refuse to run."""
    import argparse
    argparse.ArgumentParser(prog="scripts/test_srv.py", description=RETIRED_NOTICE,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_known_args(argv)
    print(RETIRED_NOTICE, file=sys.stderr, end="")
    return 2


if __name__ == "__main__":
    sys.exit(main())
