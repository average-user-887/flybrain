# History-rewrite SHA note for the Path A / escape-circuit preregs (9 Oct 2026 rewrite)

The repository history was rewritten on 9 Oct 2026. Frozen Path A preregs, contracts,
hash lists and results committed before the rewrite pin commit SHAs from the old
history. Those files are frozen evidence and are carried **byte-for-byte unchanged**:
editing them would break their recorded hashes. This note maps each pinned old SHA to
its equivalent on the published history instead.

None of the pinned SHAs below appear in the public rewrite commit map, because they were
never on a public ref (they lived only on local working branches). Each one does have a
published equivalent: the same commit carried onto the rewritten history by rebase. Every
pair was checked with `git patch-id --stable` (identical patch), and
`git range-diff` reports the whole 17-commit stack as unchanged (`=`).

| Pinned old SHA | Commit | Published equivalent |
|---|---|---|
| `d30adaf0e6a409458d82d45d6e8f06b02844d43f` | Path A: MaleCNS pathway mapping and frozen known-circuit contract | `877a26b1c3545c5fd432e1886b4a0201d4423d28` |
| `6bc971fbd52e6328367ffcc325462923fbf84662` | Path A: bind every result row to contract, engine, graph and source | `f3f31896355117ff36b7f5e61723f45d1b51a1db` |
| `51cc2c9f42cae37d4bbabdef10bbe1c8359a2bb4` | Path A: CF analyser G3 rejects missing, null or malformed input hashes | `75c6cc995085f10df17c1a1e3c8131be5c562f76` |
| `33110a4f3292ab3ce8a5c64ee8a17b2626e1cec1` | Path A: hash list of the original frozen A2 count files for the PVLP151 CF | `809911db1039ad90ac611c2089f460339fafc813` |
| `80bb93c411da2035a2b4083ef3832d121d6cf23f` | Path A: freeze the PVLP151->GF direct-edge prereg with external pins (not run) | `715ec8025863f73ff1fe5ec059be05561e465421` |
| `457d703038d6e7b8b77c71cd3ee933e00979a552` | Path A: checker-only amendment closing two G6 false passes (not run) | `c7e1c35996f42f9424fb6997a889aa9aa679b390` |
| `a1456f3383b8ea87bc0c089ef9ef05492a746efb` | Freeze audited 75-row GF diagnostic preparation without launching | `ca245e3ce798a9bc8f326cc7a8eea1e213216180` |

The stack's old base `59d22550` maps to `7847aab4` in the rewrite commit map
(`HISTORY_REWRITE_COMMIT_MAP_20261009.txt`); the stack is now based on the published
`master` (`6fdff749`). Short SHAs in the frozen files (for example `6bc971f`, `80bb93c`,
`d30adaf`, `51cc2c9`, `33110a4`) are prefixes of the full SHAs above.

Because the old SHAs were never published, a reader of the public repository cannot
resolve them directly. Use this table. The code at each published equivalent is the same
code the old SHA identified. Only the parent history differs. Run evidence produced by
the old commits (for example the accepted PVLP151 counterfactual run) remains valid: the
equivalent commit carries an identical patch.

## The "Ryzen Storage disk" wording

Two frozen hash lists (`qualification/pathA/pvlp151_cf_original_counts.sha256.json` and
`qualification/pathA/pvlp151_cf_clamp_counts.sha256.json`) describe where the hashed run
outputs were stored as "the Ryzen Storage disk". "Ryzen" names the CPU family of the
workstation that ran the evaluation, and "Storage" names the data disk. The phrase is
descriptive only. It contains no hostname, user name, path, address or other personal
data. It is left unchanged because the files are frozen.
