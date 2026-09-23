# Earlier modular simulator

This page records the older NeuroFly code that remains in the repository. It
is not a description of the current embodied MaleCNS MVP, and its models are
not used by `python -m neurofly_body`.

The earlier Python arena and behavioral modules remain in `arena.py`,
`maze.py`, `circuit.py`, `central_complex.py`, `surge_cast.py`, `vision.py`,
`mechanosensory.py`, `metabolic.py` and `locomotion.py`. They include
phenomenological controllers and a configurable arena. Their parameters and
behavioral validity should be assessed from those implementations and their
own records; their presence does not imply that they are connectome-derived.

The browser UI and continuous daemon also remain in `web/`,
`neurofly_daemon.py` and `stream_gateway.py`. The daemon's modular brain,
dashboard, and assay tools are a separate runtime from the fixed-weight
connectome-to-FlyGym loop described in [EMBODIED_MVP.md](EMBODIED_MVP.md).

The older modules, dashboards, experiment runners and documentation are kept
for development continuity. They are not part of the current MVP quick start,
and earlier learning, assay or performance descriptions are not evidence for
the embodied integration.
