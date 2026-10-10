# Archive-share synchronization receipt

Source revision: `41a08339bc29a47e0ed226683e8c7a28fd6a3b11` (accepted remediation plan).
Destination: `snapshots/75f6d3a/` on the project's archive share.
(The archive share is a private SMB share; its host and mount point are not published.)

Gemini's existing root snapshot and `CLAUDE_HANDOFF.md` are preserved. Comparison
found 40 identical tracked files, 18 differing files and 223 local tracked files
absent from the legacy share. No automatic merge or overwrite was performed.
The older handoff's operational-readiness claims are superseded by the current
audit; the legacy source remains available for comparison.

The snapshot contains a complete Git-tracked source export in `source/`,
`source.tar`, and a per-file SHA-256 manifest (`SHA256.json`). Both remediation
documents were compared byte-for-byte after copying. A Git bundle accompanies
the snapshot to retain the current branch history, including this receipt.

This is source/document synchronization, not deployment. Ignored datasets,
checkpoints, virtual environments, local credentials and untracked work are not
included. Existing brains and services were not changed. Real graph acquisition
and independent learning instances remain planned work, not delivered capabilities.

Entry point on the share: `CURRENT_HANDOFF.md`.
