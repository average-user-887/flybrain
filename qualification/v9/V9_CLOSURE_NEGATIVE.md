# v9 (option a, pinned flash intensity): closed as a documented negative, 10 Oct 2026

Decision (owner, 10 Oct 2026): v9 is frozen as a documented negative and will not be run.

Why:
- Mansour 2026 gives no calibrated flash intensity, only "a saturating 10-ms bright light flash". The intensity was pinned at E_sat = 100 by a rule committed before any probe (V9_flash_pin.json, commit e3151840).
- The CPU probe (V9_probe.json, C_R_stage_feasibility) evaluated a 72-point R-stage grid at E_sat. 0 of 72 points pass the R stage.
  - The onset gate G3_R (<= 10 ms) holds only for tau_p0 <= 2 ms, where the flash time-to-peak is 11-17 ms and G1 fails.
  - G1 (time-to-peak 23-30 ms) holds only for tau_p0 = 8 ms, where onset is 13-14 ms and G3_R fails.
  - The two gates never pass together (n_G1_and_G3_R = 0), and time-to-peak never rises with brighter flashes.
- Pinning the flash level therefore cannot rescue the v8 F1_R outcome. The R-stage failure is a structural conflict between onset and time-to-peak in the photoreceptor stage, not a flash-level artefact.
- The probe is descriptive and on a coarse grid; it does not rule out options (b) refit photoreceptor adaptation or (c) a structural change (e.g. adaptive dead time). Both are science-rule changes that need the owner and are not pursued in this goal.

Frozen files (sha256):
- V9_input_prereg_draft.json  691cf499220507ebafb866001825ac8960966b9606608b1ad268db9aa8db99ab
- V9_probe.json               3cfe90d392b96850f34ae4e0dbb3463e7e643b828b5737304f45e39fe9318a3b
- V9_flash_pin.json           58fedcbf3d54b6022181bf2bbbe95d09f417b5b0a6c2df6ba5779f10653f9edb
- scripts/v9/probe_v9.py      2d370fee55c0e9e9fa74feb7601ec46e4c42bcf5c38f08895117844875626139

The draft prereg stays a draft. No fit was run, and no held-out data (Behnia) was evaluated.
