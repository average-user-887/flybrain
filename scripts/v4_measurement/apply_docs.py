from _paths import REPO, SCRATCH  # repo root; scratch = $NEUROFLY_V4_SCRATCH or outputs/v4_measurement
import pathlib

W = REPO
T = SCRATCH

# --- WP5 §13 --------------------------------------------------------------
wp5 = W / 'docs/WP5_OPTOMOTOR.md'
text = wp5.read_text()
section = (T / 'wp5_13.md').read_text()
if '## 13. v4 —' not in text:
    text = text.rstrip('\n') + '\n\n' + section.lstrip('\n')

banner_old = """> **Caveat on §11 (2026-09-24, not a change to the recorded result):**"""
banner_new = """> **Update 2026-09-27 — §13 is a different experiment, and it changes how to read
> the HS failure above.** LIF **v4** adds declared graded (non-spiking) transmission
> (`docs/LIF_DYNAMICS_SPEC.md` §7). HS is a graded cell, so under v4 it emits no
> spikes at all and the v3-1 HS check becomes *not applicable* rather than passing.
> §13 also removes the `io_map.py` encoder's direction-selectivity assumption by
> driving R1-R6 only. Result, in one line: under v4 the signal crosses the whole
> graph to DNa02 for the first time, and it carries **no direction selectivity** —
> the predicted outcome, because one global `tau_syn = 5 ms` cannot build a delay
> line. The preregistered optomotor protocol was **not** run under v4, by the
> declared stop rule. Everything in §1–§12 stands as written.
>
> **Caveat on §11 (2026-09-24, not a change to the recorded result):**"""
if banner_new.split('\n')[0] not in text:
    text = text.replace(banner_old, banner_new, 1)
wp5.write_text(text)
print('WP5 updated:', len(text.splitlines()), 'lines')

# --- ROADMAP --------------------------------------------------------------
rm = W / 'docs/ROADMAP.md'
r = rm.read_text()
row = ("| P1 graded transmission (v4) | Pre-registered and measured 2026-09-27. "
       "LIF v4 adds declared graded (non-spiking) cell classes; a photoreceptor-only "
       "grating now reaches DNa02 (10-24 Hz) where v3 died at the first synapse, but "
       "T4/T5 carry **no direction selectivity** (flat tuning, <=0.26 mV, subtypes not "
       "anti-parallel) against the animal's DSI 0.7-0.9. The declared gate on the "
       "expensive protocol was not passed, so it was not run. Named blocker: one global "
       "tau_syn = 5 ms cannot express a 20-50 ms delay line. Default stays v3. | "
       "`docs/LIF_DYNAMICS_SPEC.md` §7, `docs/receipts/lif_dynamics_v4.json`, "
       "`docs/WP5_OPTOMOTOR.md` §13 |\n")
anchor = "| P1 optomotor on v3 |"
i = r.index(anchor)
j = r.index('\n', i) + 1
if 'P1 graded transmission (v4)' not in r:
    r = r[:j] + row + r[j:]
rm.write_text(r)
print('ROADMAP updated')
