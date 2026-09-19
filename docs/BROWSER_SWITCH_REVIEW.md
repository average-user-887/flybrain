# Browser switching regression — 19 September 2026

The user reported that switching experiments stopped working after the behavior
repairs. The previous browser review visited assays but missed a transition bug.

## Reproduction and correction

In the actual dashboard at `http://127.0.0.1:8780/index.html`, selecting Buridan
after the wind tunnel produced:

```
TypeError: stripes[0] is not iterable (cannot read property undefined)
    at ScientificBioArena.renderBuridan
    at ScientificBioArena.render
    at loop
```

The click initialized a new local scene before the daemon's matching telemetry
arrived. The renderer read the old wind-tunnel packet, whose landmarks array was
empty. The exception prevented the next animation frame from being scheduled.

Live selection now sends a command and keeps the existing scene until the new
assay's telemetry arrives. Initialization discards mismatched telemetry. Buridan
also validates the landmark array and has a safe geometric fallback. Rapid
requests are serialized, retaining the latest pending selection. Read-only
observers follow the daemon's actual assay too.

## Final browser verification

Both existing dashboard tabs were reloaded with the fix. The verification used
actual catalog clicks through the browser, DOM readings of the selected assay
and advancing step counter, screenshots of the rendered arenas, and console
inspection. No application internals or direct API commands drove these checks.

| Selected assay | Observed advancing steps |
|---|---|
| Open arena | 54518 → 54520 |
| T-maze | 54545 → 54547 |
| Y-maze | 54572 → 54573 |
| Heat maze | 54866 → 54868 |
| Buridan | 54896 → 54898 |
| Visual operant | 54921 → 54923 |
| Wind tunnel | 55249 → 55251 |
| Looming escape | 55276 → 55278 |
| Optomotor | 55301 → 55303 |
| Gap crossing | 55594 → 55596 |
| Circadian DAM | 55614 → 55616 |
| Courtship | 55641 → 55643 |
| Labyrinth | 56011 → 56012 |
| Multisensory sandbox | 56036 → 56037 |

All 14 also passed a reverse-order selection pass with Assay Tools open: the
displayed model description matched each selected experiment. Eight consecutive
rapid selections ended on the requested Buridan assay. The second dashboard
followed it; switching to courtship from that second tab updated both views.
There were **no new browser errors or warnings** after the reload. The original
pre-fix exception remained in the browser log and was distinguished by timestamp.

The wind tunnel was restored at 1×, running, with its saved brain preserved.
This verifies switching and rendering, not biological validity or task learning.

Supplementary regression coverage renders all **182 directed transitions**
between distinct assays, checks an empty Buridan landmark list, ensures a live
click cannot initialize a local scene, and exercises overlapping queued commands.
The project `AGENTS.md` now requires a final live-browser verification for UI work.
