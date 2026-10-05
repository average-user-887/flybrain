#!/usr/bin/env bash
# Print the git revision range of the commits a CI event introduces, for
# scripts/check_private_infra.sh --commits. Prints nothing when the event adds no commits.
# Run inside the checkout (full history: actions/checkout with fetch-depth: 0).
#
# Inputs (environment, as set by .github/workflows/tests.yml):
#   EVENT     github.event_name
#   PR_BASE   github.event.pull_request.base.sha     (pull_request)
#   PR_HEAD   github.event.pull_request.head.sha     (pull_request)
#   BEFORE    github.event.before                    (push; all zeros for a new branch)
#   AFTER     github.sha                             (push)
#   REF_NAME  github.ref_name                        (push)
#   REMOTE    remote name of the checkout (default origin)
#
#   pull_request  PR_BASE..PR_HEAD: the PR's own commits, not GitHub's test merge commit
#   push          BEFORE..AFTER; if BEFORE is all zeros (first push of a branch) or not in
#                 the clone (force push), every commit of AFTER that no *other* branch of
#                 the remote already contains
#   anything else (workflow_dispatch, schedule): no commits are introduced -> empty
set -euo pipefail
remote="${REMOTE:-origin}"
case "${EVENT:-}" in
    pull_request)
        echo "${PR_BASE:?}..${PR_HEAD:?}" ;;
    push)
        before="${BEFORE:-}"
        if [ -n "${before//0/}" ] && git cat-file -e "${before}^{commit}" 2>/dev/null; then
            echo "${before}..${AFTER:?}"
        else
            echo "${AFTER:?} --not --exclude=${remote}/${REF_NAME:?} --remotes=${remote}"
        fi ;;
    *) ;;
esac
