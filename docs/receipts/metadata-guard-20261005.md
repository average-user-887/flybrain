# Commit-metadata privacy guard — 5 October 2026

## Finding

A review of published commit `713ba82` found a private agent session link in its
commit message, in a session trailer. The tree guard (`scripts/check_private_infra.sh`)
had passed, because it only scans file contents. Commit messages, author and committer
were not checked anywhere, locally or in CI.

## Residual: published history is not rewritten

The owner ruled on 5 October 2026 that published history is not rewritten
(`docs/OWNER_DECISIONS.md`, "Personal data in git history: no rewrite"). Commits already
on GitHub therefore stay as they are. The guard recognises them only through a frozen,
exact exception manifest (below). It never trusts a commit just because some remote ref
already contains it.

Recounted on `origin/master` at `713ba82` (244 commits), using the new guard
(`--commits origin/master`) and cross-checked with `git log --grep`:

| Range | Commits | With an agent session link in the message | Failing any metadata rule |
|---|---:|---:|---:|
| all of `origin/master` | 244 | **145** | 223 |
| `5c54b03..713ba82` | 61 | **44** | 46 |

Every one of the 145 messages carries the link in a session trailer. The owner ruling
cited 105 such messages when the audit was taken. The other 40 were published after the
audit and before this guard existed. The "failing any rule" column also counts the
identity residual that the ruling already covers: a local machine e-mail and hostname
in author/committer, and the owner's personal e-mail on web-interface merges made
before e-mail privacy was enabled.

## Frozen historical exception manifest

`scripts/private_infra_commit_exceptions.txt` lists, by full SHA, every already-published
object whose metadata fails a rule, with exactly the rule classes it fails. It records no
matched values.

- **Scope:** all 88 refs the GitHub repository advertised on 5 October 2026 (55 branches,
  33 `refs/pull/N/head`, tag `v0.3.0`). The one exclusion is `claude/metadata-guard`, which
  is this change, and its commit is clean anyway. The sha256 of the
  `git show-ref --dereference` listing of that scope is recorded in the manifest header.
- **Build:** a fresh mirror clone, scanned with the guard itself:
  `--no-commit-exceptions --emit-exceptions --commits '--exclude=refs/heads/claude/metadata-guard --all refs/tags/v0.3.0'`.
  274 commits plus 1 annotated tag were scanned, and 232 objects fail and are listed.
- **Meaning:** an entry acknowledges immutable existing exposure. It does not approve the
  disclosure and does not make it private.
- **What an entry exempts:** only its listed classes, and only for that object's metadata.
  A commit not listed is always checked, wherever it is reachable from. No tree is ever
  exempted, and there are no e-mail or domain exceptions.
- **Pinning:** the file is pinned by `COMMIT_EXCEPTIONS_SHA256` in
  `check_private_infra.py`. Any edit fails closed until the pin is changed too, which
  makes the change visible in review.

## What is checked now

- `check_private_infra.sh --commits REV_OR_RANGE` scans the metadata of every commit in
  the range. A single revision means its whole ancestry. It also scans annotated tags
  named in the range.
- The fields scanned are the full message, author, committer and tagger names and
  e-mails, and other non-signature headers (for example an embedded mergetag).
- The rules applied:
  - the tree rules: private IPs, home/share/agent-scratch paths, e-mail, denied tokens
    (stored as hashes);
  - `agent-session`: Claude Code session, Claude chat/share, ChatGPT/Codex task and chat,
    and Gemini chat links, and bare `session_<id>` identifiers;
  - `session-trailer`: trailers such as `Claude-Session:` or `Codex-Task:`;
  - `identity-email`: only `*@users.noreply.github.com`, `noreply@anthropic.com` and
    `noreply@github.com` are accepted, both as identities and inside messages;
  - `hostname`: LAN-style names (`.local`, `.lan`, `.home`, `.internal`, ...).
- The allow-list file does not apply to metadata.
- Findings print the SHA, the field (with a line number for message and header) and the
  rule, never the matched text.
- `check_private_infra.sh --tree-rev REV` scans the committed tree of REV, which is what
  a push publishes, and not the checkout.
- The guard fails closed with exit 2 on a bad or empty range, a missing object, a
  malformed object, a manifest that is malformed or does not match its pin, or any
  scanner error.

## Where it runs

- **CI** runs in a separate workflow, `.github/workflows/privacy.yml`:
  - It runs on every push (all branches and tags), every pull request and manual runs.
  - It is never cancelled by a newer push.
  - Branch deletions are skipped because they publish nothing.
  - `scripts/ci_privacy_tip.sh` picks the published tip: the PR head (not GitHub's test
    merge), the push's `after`, or `github.sha` for manual runs. It fails closed if that
    id is missing, malformed, all zeros or absent from the clone.
  - The job runs the tree guard on the checkout, then `--tree-rev` and `--commits` on
    the tip. There is no before..after window and no remote-ref exemption.
- **Pre-push hook** (`scripts/hooks/pre-push`):
  - For every pushed ref, whether or not it is checked out, it runs `--tree-rev <sha>`
    and `--commits <sha>`.
  - Deletions are validated but send nothing.
  - Malformed input, a missing object or a scanner error refuses the push.
  - Install it once per clone:

    ```bash
    git config core.hooksPath scripts/hooks
    ```

- **By hand, before pushing:**

  ```bash
  scripts/check_private_infra.sh --tree-rev HEAD
  scripts/check_private_infra.sh --commits HEAD
  ```

A finding is fixed by rewording the unpublished commit (`git commit --amend`, or an
interactive rebase) and by setting `user.email` to a noreply address.

## Tests

`tests/test_private_infra_commits.py` plants commits in throw-away repositories and a
bare remote. It covers:

- each kind of leak, detected by SHA, field and rule;
- a bad new author, committer or body, rejected both in-process and through the real
  hook;
- annotated tag metadata;
- a ref that is not checked out but has a dirty tree, rejected;
- two new refs sharing a bad commit, rejected even after one of them reaches the remote;
- a clean forward update and a new branch over frozen published history, accepted, and
  a new leak on top of it, rejected;
- deletion input handled without being treated as a new object, and malformed hook input
  refused;
- fail-closed behaviour on bad ranges, missing commits and blobs, a broken manifest pin
  and malformed manifest lines;
- exceptions that cover only the listed commit and classes, and never a tree;
- that no planted private text appears in any diagnostic.

All planted strings are assembled from pieces, so the tree guard stays clean on the
test file itself.

## Limits

- The guard cannot reach history that is already published (owner ruling above).
- The hook can be bypassed with `git push --no-verify`. CI is the backstop.
- A failing check blocks a merge only if branch protection on `master` requires the
  privacy check. That is a repository setting, not something this change can set.
- CI runs the guard code and manifest from the checked-out tree. A change to the guard,
  the manifest or its pin is therefore a reviewed change like any other, and needs
  owner review.
- New denied tokens added later may also flag published commits for a class their
  manifest entry does not list. The guard then fails closed until the manifest is
  re-reviewed.
