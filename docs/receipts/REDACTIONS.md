# Redaction log

Date: 5 October 2026. Branch `claude/sanitize-tree`, based on `origin/master` at `93f0b18`.

An external audit on 5 October 2026 found personal data and private-infrastructure
names in the published tree: the local username in home-directory paths, AI-agent
job/scratch paths, the workstation's hostname, internal host and share names, and
a tracked `.claude/settings.json`. This file records what was removed from the
current tree. Git history is unchanged. By owner decision of 5 October 2026
(`docs/OWNER_DECISIONS.md`) it is not rewritten: older commits still contain the
original strings, and their author/committer metadata is an accepted residual.

## Rule

Receipts are scientific evidence, so only the personal/infrastructure **substrings**
were replaced. No number, result, hash of other evidence, field name, key order or
line was added, removed or reordered. Every changed JSON receipt was re-parsed and
compared leaf by leaf against `origin/master`: same structure, every non-string
value identical, and every changed line contains one of the placeholders below.

## Placeholders

| placeholder | stands for |
|---|---|
| `<repo>` | the repository root of the checkout that produced the receipt (the main checkout or a local worktree of it; the commit recorded in each receipt identifies the code) |
| `<home>` | the operator's home directory |
| `<scratch>` | an agent job / temporary scratch directory outside the repository |
| `<reference-host>` | the hostname of the reference NVIDIA workstation (GTX 1660 Ti) |

The receipt scripts under `docs/receipts/switch-race-20261004/scripts/` are records of
what was run. After redaction they contain these placeholders and must have them
substituted before they will run again.

## Hash citations

Before editing, the SHA-256 of every changed file (old version) was searched for,
as a full hash and as a 12-character prefix, in the old and new trees; references to
each receipt's file name near "sha256", "locked", "digest" or "pin" were also checked.
**No changed file's own hash is cited anywhere in the repository.** No
`*.locked.md` declaration contained personal data, and none was modified. Receipts
that record the hashes of *other* files (for example `photoreceptor_encoder.json`
under `declaration_lock`) keep those hashes unchanged. Old and new SHA-256 values are
listed below for every redacted receipt, so anyone holding an old copy can verify
it against `origin/master` history.

## Receipts redacted

| file | class of data redacted | old sha256 | new sha256 |
|---|---|---|---|
| `docs/receipts/body_speedup/cloud_after.json` | agent-scratch path | `f0dd6c499bcaa0451050a0992ff275b58df25e4acecc758c5da9db01f3654ca5` | `23fdbba23b0822dcc6e049392659fa3104b3aa61a79cbbe40fdf83fec17982a4` |
| `docs/receipts/body_speedup/cloud_before.json` | agent-scratch path | `44c30ae2d370b918a8dfb0265855ce66e95bc828c9cbe74641de0ebd1cb58510` | `239cb6b8325e4c286c70d09b73d3db6ef1174a1a7bdc4d67b54d823da40454d7` |
| `docs/receipts/embodied_mvp_verification.md` | agent-scratch path, home path (checkout) | `47725a5308c9092e3baf8a5ec3824ef5bebe39b3473c4cd0dfa8ee71753315ea` | `19ded9a9f483bf485eaf72a1c300cb24557d5520c6351dfc01f14dc921ce9da5` |
| `docs/receipts/graph_identity.json` | home path (checkout) | `e060c30a30d5e65345c9331ea15f7c72e5fec520f49e56441d1683e6f395ba98` | `67405fe4e467abfe1fc749acdbed57beb78c58a1e0d6e79ebe4598c98f35900c` |
| `docs/receipts/integration/firefox_headless_receipt.json` | home path (checkout) | `5e343b517e0a5817416bb89e5c0af606b67689e845cb7399aac003cf3a150eed` | `d7f1678baf81ee2ab959f5f22f3581e5150c07a4894cdaf03afffbc2c338f20b` |
| `docs/receipts/integration/firefox_headless_receipt_5c9519e.json` | home path (checkout) | `24985432258bc68305865c06ede87bfc254a908037d41d6a3c78f04499154f12` | `df5277fc2f6dab158e3c950a1fab19303f419701f9da02329916a6233850431d` |
| `docs/receipts/lif_dynamics_diagnosis.json` | home path (checkout), hostname | `39bcbe35e5f794851f2ed2792b35cb4d34584292b3e6e2f9716fc914fed770e4` | `2622c77e3349f800b0c3adde5dc11d5e40cf678b3ead272739c5b3592dda0c49` |
| `docs/receipts/lif_dynamics_v2.json` | home path (checkout), hostname | `941b11139b03fcd542ddcf89032cea4daee4962956ce0a0a5192aa2f5fed0135` | `1e81378bbea876ba63bab508a80e04a3254866360252a55ecb16c9df337a67e2` |
| `docs/receipts/lif_dynamics_v3.json` | home path (checkout), hostname | `dc709e1ad71b52698f86f7b88a4e9ea259fc5f3517c4f8bc293b7376bbe2dd7f` | `aaa569139dde9f5c93dda4cdd2affb584b0490b8dad02d4bdc3e282954f0aeec` |
| `docs/receipts/live-signoff-fresh/live_ui_signoff.json` | home path (checkout) | `657a90bb10c620cf0aba5dd6b4c56ae3257c936c7ea1a69792eb44dea6b84b0b` | `369b7eb9f9a97cedbfe12fe382ef87aac706661f8958333e7a89de946e481d5c` |
| `docs/receipts/live-signoff/live_ui_signoff.json` | home path (checkout) | `aa2b89e4dec745b5045d43d147868763ec717462f6f4eac1cee6608a3a6a1b68` | `fb22bf77fd6d0082b255bcbafbac05fb6a1a52c42f043608ce60d1a794a98f99` |
| `docs/receipts/photoreceptor_encoder.json` | hostname | `3b998a4719bd3b74b335a7d9b72a4fb82f3da045db3efd5c522fef9abbc460ba` | `0ef245744930e70972ef1ddf1f9241d808d802e42999bcb0b22cd9c0f115b8d5` |
| `docs/receipts/ryzen/bench-ryzen-1f4a58a.json` | hostname | `64a29e4d8633d45171ae0bd08e671578ee27a817ff79747df6c2112302574246` | `868a52d457bdd124f0b9baa2071ab7532ece00664ff44e0ef62471e237e20ce3` |
| `docs/receipts/switch-race-20261004/browser/fixed-run1/live_ui_signoff.json` | agent-scratch path | `32e161f7dba99217cf6220bbc738e637c6cc4a59b37ecfd308fe678e95a551e3` | `f290b962085786b5383437152e5bdb6772238fb1d382a2b1fbee19da89e9eca0` |
| `docs/receipts/switch-race-20261004/browser/fixed-run2/live_ui_signoff.json` | agent-scratch path | `73f326e34333ba835c63c66eda4b72dc324406e5baece0356b5f909a85b05b6f` | `8730ba94c6fb3bbe2861413f210ca8857eeee7cde9197691f24010773d821dc4` |
| `docs/receipts/switch-race-20261004/browser/fixed-run3/live_ui_signoff.json` | agent-scratch path | `cb4de356d8b3f9c1ac5b93ad6b13916840bc743ca4590bcdb5dd49f354970432` | `98bb32439538dcb8cd000ec32e9f93a5bd2235a01dabeac8ad36cdfe9d035d34` |
| `docs/receipts/switch-race-20261004/browser/fixed-run4/live_ui_signoff.json` | agent-scratch path | `fb67a425dd39e0de7dbae10e8a512fce0d280450865e99438ffd6704d68602d2` | `a6d5357b0b3fadb5ba249a10f3f8dfce0528a3967809111817d826f0e141fd42` |
| `docs/receipts/switch-race-20261004/browser/fixedb-run1/live_ui_signoff.json` | agent-scratch path | `e7cce4454f2f51df1ae3e2e60f2d77a04c8cb13dfe00064baec344ec5ae50fca` | `11e3b3b1f887e87135c8d5fd858c844736f8b6532fd3f9bdd22fea4435ecd5e4` |
| `docs/receipts/switch-race-20261004/browser/fixedb-run2/live_ui_signoff.json` | agent-scratch path | `112ab6621fc10cb5b712133617cf8f21c208bd259dfc1eae9a27bf2c8052a92a` | `3eecc9d9d9a1cadde32c8f50246d09a23ec3db75ceca5ccda39453c8dbde0246` |
| `docs/receipts/switch-race-20261004/browser/master-run1/live_ui_signoff.json` | agent-scratch path | `fc5e3e73e24c0b740d5c86676565fa0a3124ef9543e02abc0478153d4b8cfae2` | `b79416d0dc3554923f97a5aedec371bf440f94f3322e73104cbff955f77be143` |
| `docs/receipts/switch-race-20261004/browser/master-run2/live_ui_signoff.json` | agent-scratch path | `571527fe1e14e8c9c635293832296df8aa915a6ab2de47094c83091f0fa7b538` | `5412e6652cc08a7f39669d2dbae4cb0fb893a808299fc010f3665c2de2c3483e` |
| `docs/receipts/switch-race-20261004/scripts/browser_runs.sh` | agent-scratch path, home path (checkout) | `8486650e842ef9cd3bcd81141ab690b26b3fc44502a2534cf62eef1f311ae1bb` | `fedf35e40eca0da55e67f43a720aef650fabbcb207b9afe71c1a3b28be032e1e` |
| `docs/receipts/switch-race-20261004/scripts/halt_browser_check.py` | agent-scratch path | `e5b042266a69f7c0a30b381bc73e5d9fbd9605fe632f418a4d01803a1c4896cb` | `5a213045144e74fab57a9a458033584103ee4d0b0c564bd3882bbefa1ca3e85f` |
| `docs/receipts/switch-race-20261004/scripts/inject_daemon.py` | agent-scratch path | `df6579852266e720927c7bec0df0fea70d5c2acd61a36f424d3c7863c5d048f2` | `6f03483c9a6dbe8d6a3e8a05c2578581a326b83a40444cf1e92c347d2efbcc64` |
| `docs/receipts/switch-race-20261004/scripts/start_daemon.sh` | agent-scratch path, home path (checkout) | `9ffac9cbcc72146fbc1a8285c4423f2ec985f96e2c4dadcffbcad70093fd9073` | `34f85270ecc81e0a095738f70252b1bbfbc9f25e3280ed92d8092d8bdebe40c5` |
| `docs/receipts/switch-race-20261004/scripts/start_inject.sh` | agent-scratch path, home path (checkout) | `7623272835909f69e01591e39cd46408763d556e793128ad1568de08fdb7f212` | `0ae2fa3b9ee129a85a526d52d08f83c4585d82379748c7880cb4fcd4fe6dc884` |
| `docs/receipts/switch-race-20261004/scripts/trial.sh` | agent-scratch path, home path (checkout) | `e9212f2d92b0365aa42fecaf703a2e1c4387e53ba5071ade6799485e86936638` | `665c277059a9752fb2c61002be28d0209f3f163d95378c282a0ed8b3d6e78369` |
| `docs/receipts/validation/optomotor-yaw-v3-2.md` | home path (checkout) | `c7db624abf227e936d6fd96e894e4e9b3f4108e62a16f18565c020ed71ad034a` | `8cdc81fde59c1d459cd85976689e0db9e974c027a33f0cd8a9c7be32b89de57b` |
| `docs/receipts/wp4_registry_resources.json` | home path (checkout), hostname | `676f59553232a126c4c0d46e6d95c713bb5c0d94e850cb8c7c29e5c0b151a477` | `7ee1da5a148cf1eb39167d89ec6a11a96153dc4ba194caa9986c65cb66ce67d8` |
| `docs/receipts/wp5_optomotor.json` | home path (checkout), hostname | `f81d07a3a70c6de4a5ca9a488637ce3344c66f330b8e9c4934caa6fe10aaf42f` | `7fa3b02f462f98a4240d998019f1afbbd87baccd17913396ae384feb105f2d00` |

Classes: *home path (checkout)*: an absolute path to a checkout under the
operator's home directory (username and directory layout); *agent-scratch path*:
a per-job scratch path of an AI coding agent; *hostname*: the workstation hostname,
which embeds the username.

## Other files changed

| file | change |
|---|---|
| `scripts/v4_measurement/*.py` (19 scripts) and new `scripts/v4_measurement/_paths.py` | Hard-coded home-directory checkout paths and agent scratch paths replaced: the repository root is resolved from the script location, and intermediate JSON goes to `$NEUROFLY_V4_SCRATCH` (default: gitignored `outputs/v4_measurement/`). Behaviour otherwise unchanged. |
| `docs/RELEASE_PLAN_v0.4.md` | Internal host names, the archive-share mount path and the home-relative checkout path replaced with generic wording ("reference NVIDIA host", "AMD test host", "the project's archive share"). Gate G9 corrected. |
| `docs/ARCHIVE_SYNC.md` (renamed from a file named after the internal server) | The internal server name and share mount path were removed; the snapshot revision and contents are unchanged. |
| `docs/archive/CODEX_HANDOFF.md`, `docs/archive/CLAUDE_HANDOFF.md`, `docs/archive/ROADMAP_v1.0.md` | Internal host names, the archive-share mount path, the overlay-network name, a personal laptop folder name and the home-relative checkout path replaced with generic wording or placeholders. |
| `docs/LEARNING_OBSERVATORY.md` | Agent worktree path replaced with "a separate local worktree". |
| `docs/RELEASE_AUDIT.md` | Home-directory layout and a mapped drive letter in the findings table replaced with placeholders. |
| `.claude/settings.json` | Untracked (`git rm --cached`); `/.claude/` was already in `.gitignore`. |
| `scripts/check_private_infra.sh`, new `scripts/check_private_infra.py`, `scripts/private_infra_allowlist.txt`, `tests/test_private_infra_guard.py`, `.github/workflows/tests.yml` | Strengthened guard, run locally and in CI; see below. |

## Guard

`scripts/check_private_infra.sh` (a wrapper around `scripts/check_private_infra.py`)
fails on: RFC 1918 addresses; `/home/<user>/`, `/Users/<user>/` and `C:\Users\...`
paths; `/mnt/` and `/media/<user>/` paths, UNC shares and mapped share drives; AI-agent
job, session, scratch and worktree paths; e-mail addresses other than noreply
addresses (`noreply@anthropic.com`, `*@users.noreply.github.com`, `noreply@github.com`);
JSON `host`/`hostname` fields holding a real machine name; and a list of known
private tokens (username, hostname, internal host names), stored as SHA-256 hashes
so the guard does not republish them. It scans tracked and untracked, non-ignored
files in a git checkout, and walks the directory with the same exclusions in an
exported tree. Genuine exceptions go in `scripts/private_infra_allowlist.txt`.
Hashing only hides these short tokens from casual reading, because a short token can
be guessed. The tokens are already in the published history.

## Artifacts written from now on (branch `claude/no-hostnames`)

The redaction above cleaned files already in the tree. The code that produced them
also had to stop writing personal data, or every new receipt, run manifest,
recording and bundle a user shares would leak their machine's name and home
directory again. All writers now go through one module, `neurofly/privacy.py`:

- `host_description()` replaces `platform.node()` (the hostname). It records the OS
  and architecture, OS release, CPU model, CPU count, GPU model(s) (or `"none"`),
  Python version and key library versions. Hardware model names are public facts;
  no hostname, account name or path is recorded.
- `redact_local()` / `portable_path()` write absolute local paths with the placeholders
  in the table above: `<repo>/...`, `<home>/...`, `<scratch>/...` (system temporary
  directory), and `<local-path>/<last component>` for any other path under `/home/`,
  `/Users/`, `/media/`, `/run/media/` or `/mnt/`. The account and machine names are
  also removed from any remaining path-like string (`<user>`, `<host>`). Strings that
  are not paths are never rewritten. In memory, paths stay absolute (they are used to
  load files); only what is written changes.

**Compatibility.** The `host` field keeps its name in run manifests, validation
receipts, recording sidecars and the WP5 / LIF-diagnosis / registry-measurement
receipts, but now holds the host-description object instead of a string.
`RunManifest.read` accepts both shapes, and the receipt-assembly scripts copy `host`
through unchanged. The benchmark receipt's `host.hostname` key was read by nothing and
is no longer written; `host.description` holds the host description. Run manifests
write `source.root` as `<repo>`, and graph identities write `graph_path` /
`neuron_map_path` as placeholders; the graph is identified by its SHA-256, not its
path. The embodied `replay-check` maps `<repo>`, `<home>` and `<scratch>` back to this
machine's directories and treats an unmappable `<local-path>` graph directory as "use
the default location". Committed receipts are not regenerated.

Writers changed: `provenance.py` (`RunManifest`, `source_revision`),
`brainlab/graph_identity.py` (`GraphIdentity.to_dict`), `brainlab/cosim_server.py`
(status), `brainlab/runs.py`, `brainlab/measure_registry.py`, `neurofly/recording.py`,
`neurofly_body/runner.py` and `cli.py` (manifest, summary, `body.nfbody` header,
`replay_check.json`), `neurofly_daemon.py` (checkpoint JSON), `validation/harness.py`,
`scripts/benchmark.py`, `scripts/lif_dynamics_diagnosis.py`, `scripts/timing_stress.py`,
`scripts/wp5_optomotor.py`, `scripts/wp5_photoreceptor_probe.py`.
`tests/test_no_personal_data.py` checks each, comparing against this machine's
hostname and account name at test time, and fails if any code outside
`neurofly/privacy.py` reads the hostname or account name.
