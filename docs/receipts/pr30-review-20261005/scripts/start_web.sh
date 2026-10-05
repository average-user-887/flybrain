#!/bin/bash
# Static dashboard server for the PR #30 review worktree on 127.0.0.1:8795.
WT=<repo>
exec <home>/Documents/ChatGPT/flybrain/.venv/bin/python -u -m http.server 8795 --bind 127.0.0.1 \
  --directory "$WT/web" > <scratch>/web.log 2>&1
