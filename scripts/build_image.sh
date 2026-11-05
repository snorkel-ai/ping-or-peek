#!/usr/bin/env bash
# Build the task image once, before a run, and print the agent versions inside it.
#
#   scripts/build_image.sh [TASK_DIR]
#
# Every task's Dockerfile shares the same layers up to the final COPY, so this warms Docker's
# build cache for all of them. Concurrent trials then only add their small workspace layer.
set -euo pipefail

task="${1:-tasks/plain/user_available_checker_exposed__seed100}"
tag="ask-or-shortcut-check:latest"
docker build -q -t "$tag" "$task/environment" >/dev/null
docker run --rm "$tag" sh -c '
  export PATH="$HOME/.local/bin:$PATH"
  echo "claude-code   $(claude --version)"
  echo "codex         $(codex --version)"
  echo "opencode      $(opencode --version)"
  for p in @agentclientprotocol/claude-agent-acp @agentclientprotocol/codex-acp; do
    echo "$p $(npm ls -g --depth=0 "$p" 2>/dev/null | grep -o "@[0-9][^ ]*$")"
  done
  echo "node          $(node --version)"
  echo "python        $(python3 --version)"
  echo "workspace     $(ls -A /workspace | tr "\n" " ")"'
docker image rm -f "$tag" >/dev/null
