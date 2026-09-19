# Contributing to Project NeuroFly

Thanks for your interest. NeuroFly is a research codebase; the rules below keep
it honest and reproducible.

## Setup

```bash
git clone <this repository> neurofly && cd neurofly
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
PYTHONPATH=. .venv/bin/python -m pytest -q tests/
```

The test suite needs no connectome download; it runs on synthetic graphs and
the in-process behavioural models. Expect it to finish in under a minute.

## Ground rules

1. **Do not overstate what the simulation does.** The README's "What this
   simulates today" section is the reference. If your change adds a model,
   describe it at the same level: what is computed, what parameters were
   chosen and why, and what is *not* modelled. Claims about neuron or synapse
   counts must point at code that actually loads them.
2. **Keep tests green and add tests for new behaviour.** CI runs
   `pytest tests/` on Python 3.12 for every pull request.
3. **No private infrastructure in the tree.** Hostnames, LAN or overlay IP
   addresses, SSH key names, container ids, usernames and machine-specific
   paths must not be committed. Use environment variables (`NEUROFLY_*`) or
   CLI flags with neutral defaults. `docs/RELEASE_AUDIT.md` lists the
   patterns the release audit greps for.
4. **No secrets.** The admin token for public mode is read from
   `NEUROFLY_ADMIN_TOKEN`; never add a `--token` style CLI flag (it would
   leak through process listings and shell history) and never commit `.env`
   files.
5. **Respect data licenses.** MaleCNS v1.0 is CC BY 4.0: keep the attribution
   in `NOTICE` and the provenance records under `data-provenance/`. Do not
   commit the raw connectome tables (`connectome_data/` is git-ignored).
6. **Generated data stays out of git.** `outputs/`, `runs/`, and daemon
   learning records are ignored. Small, curated example outputs are fine if
   they are documented.

## Pull requests

- One topic per PR. Describe what changed and how you checked it.
- Physics changes in `maze.py`/`arena.py` must be mirrored in `web/app.js`
  (the browser runs its own copy of the arena), and vice versa.
- Changes to `neurofly_daemon.py`'s HTTP surface need a test in
  `tests/test_stream_gateway.py`; changes to the on-disk record format need
  a `schema_version` bump and an update to `docs/DATA_SCHEMA.md`.
- Documentation-only changes are welcome, including corrections to
  scientific claims.

## Reporting problems

Open an issue with the command you ran, the Python version, and the relevant
part of the output. For anything security-related about the public stream
(auth bypass, resource exhaustion), please report privately to the
maintainers before opening a public issue.
