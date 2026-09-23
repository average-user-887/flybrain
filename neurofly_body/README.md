# NeuroFly embodied MVP

This isolated package couples the verified MaleCNS fixed-weight graph to the stock
FlyGym 2.1 articulated fly. It does not modify or depend on the web dashboard or
the 14 assay controller.

```bash
python -m neurofly_body run \
  --duration 0.5 \
  --output outputs/embodied/intact-001 \
  --mode intact
```

The output directory must not already exist. The command refuses synthetic graphs,
non-v3 dynamics, the wrong transmitter policy, engineered assistance, missing graph
identity, or a missing verified optomotor map. It writes `manifest.json`, one complete
neural/body record per 2 ms to `telemetry.jsonl`, and `summary.json`.

For the causal control use a fresh directory and the same seed/stimulus:

```bash
python -m neurofly_body run \
  --duration 0.5 \
  --output outputs/embodied/disconnected-001 \
  --mode output-disconnected
```

The control still advances the same real graph and body and logs the decoded motor
signal, but replaces the command reaching the FlyGym CPG with exact zeros. There is
no tonic or fallback walking drive. `--video` optionally writes `body.mp4` using the
offscreen MuJoCo renderer; on a headless host, `MUJOCO_GL=egl` is selected by default.

Scientific scope: DNa02 rates are crossed into the stock left/right CPG magnitudes by
an explicitly labelled engineered decoder. This establishes a testable neural-output
to joint/body causal seam; it is not a biological VNC implementation and does not by
itself establish full behavioral reproduction.
