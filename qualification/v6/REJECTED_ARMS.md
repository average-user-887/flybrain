# v6 rejected arms (never deleted)

| arm | stage | spec sha256 | why rejected | numbers |
|---|---|---|---|---|
| S1-A1 | S1 fit, phototransduction with ONE adaptation constant for gain and speed | 67d684c3…7177 (S1_fit_spec.json) | the single constant cannot give both the dark-flash amplitude and a 20 ms BG0 time to peak: tau_p0 ran to its 1 ms bound, T3 = 10 ms against 20 +- 4; the release fit then crashed on an L cell clamped at E_inh (V_dark = -70, division by zero) | G 19.6, Ka 0.026, tau_p0 1.0 (bound); T1 39.2 mV (44 +- 4), T2 0.44 (0.39 +- 0.09), T3 10 ms (20 +- 4); held-out H1 15 ms (accept 30-50), H2 0.87 |
