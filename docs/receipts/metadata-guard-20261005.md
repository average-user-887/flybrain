# Commit-metadata privacy guard — 5 October 2026

## Finding

A review of published commit `713ba82` found a private agent session link in its
commit message, in a session trailer. The tree guard (`scripts/check_private_infra.sh`)
had passed, because it only scans file contents. Commit messages, author and committer
were not checked anywhere, locally or in CI.

## Residual: published history is not rewritten

The owner ruled on 5 October 2026 that published history is not rewritten
(`docs/OWNER_DECISIONS.md`, "Personal data in git history: no rewrite"). Commits already
on `master` therefore stay as they are and are an accepted residual. The guard checks
only commits that are new to a push or pull request.

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

## What is checked now

`scripts/check_private_infra.py --commits RANGE` (also via the `.sh` wrapper) scans,
for every commit in a git revision range, the full message, the author name and e-mail
and the committer name and e-mail. It applies:

- the tree rules: private IPs, home/share/agent-scratch paths, e-mail, denied tokens
  (stored as hashes);
- `agent-session`: Claude Code session, Claude chat/share, ChatGPT/Codex task and chat,
  and Gemini chat links, and bare `session_<id>` identifiers;
- `session-trailer`: trailers such as `Claude-Session:` or `Codex-Task:`;
- `identity-email`: author/committer e-mail must be `*@users.noreply.github.com`,
  `noreply@anthropic.com` or `noreply@github.com`. The same short list is the only one
  accepted for e-mail addresses written inside messages;
- `hostname`: LAN-style names (`.local`, `.lan`, `.home`, `.internal`, ...).

Findings print the commit SHA, the field (with the line number for messages) and the
rule. They never print the matched text. The tree scan now also catches `agent-session`
and `session-trailer`, and it redacts agent-session matches.

## Where it runs

- **CI** (`.github/workflows/tests.yml`, step "Commit metadata guard"). The checkout
  uses full history. `scripts/ci_commit_range.sh` picks the range:
  - pull request: `base.sha..head.sha`, which is the PR's own commits and not GitHub's
    test merge;
  - push: `before..after`;
  - first push of a branch (`before` is all zeros) or a force push whose old tip is
    missing: every commit of `after` that no other branch already contains;
  - manual runs: skipped, because they introduce no commits.
- **Pre-push hook** (`scripts/hooks/pre-push`). It runs the tree guard and checks
  exactly the outgoing range of each pushed ref (`remote..local`, or for a new branch
  `local --not --remotes=<remote>`). It skips deletions. Install it once per clone:

  ```bash
  git config core.hooksPath scripts/hooks
  ```

- **By hand, before pushing:**

  ```bash
  scripts/check_private_infra.sh --commits origin/master..HEAD
  ```

A finding is fixed by rewording the unpublished commit (`git commit --amend`, or an
interactive rebase) and by setting `user.email` to a noreply address.

## Tests

`tests/test_private_infra_commits.py` plants commits in throw-away repositories: a
session trailer, a Codex task link, a personal author e-mail, a LAN hostname, a denied
token in the committer name, and clean commits. It checks that each one is detected by
SHA, field and rule, that no planted text appears in the output, and that clean and
empty ranges pass. It also tests the CI range rules (including first and force pushes)
and the real pre-push hook against a bare remote. All planted strings are assembled
from pieces, so the tree guard stays clean on the test file itself.

## Limits

- The guard cannot reach history that is already published (owner ruling above).
- The hook can be bypassed with `git push --no-verify`. CI is the backstop for pushes
  and pull requests.
- The hook's tree check scans the working tree, not each pushed commit's tree. CI
  checks the tree that is actually pushed.
- A failing pull-request check blocks a merge only if branch protection on `master`
  requires the `pytest (Python 3.12)` check. That is a repository setting, not
  something this change can set.
- Merges made in GitHub's web interface run no local hook. Their metadata is checked by
  CI on the resulting push to `master`. With e-mail privacy on, they carry a noreply
  address.
