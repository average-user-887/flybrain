# S2 fit spec addendum 1 (before any fit result; the first launch crashed while building the readout)
MaleCNS `assignedOlHex` is empty for every L4 (0/1770) and Tm3 (0/2054) cell, so the frozen spec's cutout had no central L4 or Tm3.
Change: cutout columns for the 13 types come from the Path B column map (phase1/cell_columns.npz, pinned in the Path B verdict, b61741d2...), R1-R6 still from photoreceptor_io cartridges.
L4 is absent from that map too, so L4 is EXCLUDED from the objective (12 types) and reported as not evaluable; its tau_m parameter is left at the default 20 ms.
Nothing else changes (observation model, bounds, objective, optimiser, arms, stop rule).
