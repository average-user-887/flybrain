# NeuroFly verification rule

For dashboard, rendering, controls, telemetry or experiment-switching changes,
the final functional verification must be in the running browser. Unit tests,
API calls and headless rendering checks supplement that check; they do not replace it.

- Reproduce the reported interaction in the user's live UI before changing code.
- After the final code change, click the real controls and check the resulting
  arena, selected assay, live clock/step counter, tools panel and browser errors.
- For experiment switching, visit all 14 assays, test the reported transition,
  rapid consecutive selections and synchronization between open dashboard tabs.
- Reload existing dashboard tabs to load the deployed JavaScript. Preserve saved
  brains and restore the user's starting assay, speed and pause state after testing.
- Record what was actually tested and any remaining limitations. Do not claim a
  browser pass from API responses, static screenshots or model tests alone.
