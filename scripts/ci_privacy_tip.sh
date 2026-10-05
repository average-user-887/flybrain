#!/usr/bin/env bash
# Print the object a CI event publishes, for the privacy guard:
#   check_private_infra.sh --tree-rev TIP   (its committed tree)
#   check_private_infra.sh --commits  TIP   (metadata of its whole ancestry; commits that
#       were published before the guard are recognised only via the frozen manifest
#       scripts/private_infra_commit_exceptions.txt, never via other remote refs)
# Run inside the checkout (full history: actions/checkout with fetch-depth: 0).
#
# Inputs (environment, as set by .github/workflows/tests.yml):
#   EVENT     github.event_name
#   PR_HEAD   github.event.pull_request.head.sha   (pull_request: the PR's own tip, not
#                                                   GitHub's synthetic test merge)
#   AFTER     github.event.after                   (push: the new tip, may be a tag object)
#   SHA       github.sha                           (any other event)
#
# Fails closed (exit 1, nothing printed) if the input is missing, not an object id,
# all zeros, or not present in the clone.
set -euo pipefail
case "${EVENT:-}" in
    pull_request) tip="${PR_HEAD:-}" ;;
    push)         tip="${AFTER:-}" ;;
    "")           echo "ci_privacy_tip: EVENT is not set" >&2; exit 1 ;;
    *)            tip="${SHA:-}" ;;
esac
if ! [[ "$tip" =~ ^[0-9a-f]{40}([0-9a-f]{24})?$ ]] || [ -z "${tip//0/}" ]; then
    echo "ci_privacy_tip: no valid object id for event ${EVENT}" >&2
    exit 1
fi
if ! git cat-file -e "${tip}^{commit}" 2>/dev/null; then
    echo "ci_privacy_tip: ${tip:0:12} is not in the clone" >&2
    exit 1
fi
echo "$tip"
