# HP server synchronization receipt

Source revision: `75f6d3aa264bd64ebcc4759658d54f290e8608be` (accepted remediation plan).
Destination: `/mnt/hpserver-storage/neurofly/snapshots/75f6d3a/`.
SMB location: `//<archive-host>/Storage/neurofly`.

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
