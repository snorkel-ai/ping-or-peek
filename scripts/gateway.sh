#!/usr/bin/env bash
# Optional: route every model call through a Portkey gateway, then run the given command.
#
#   scripts/gateway.sh harbor run ...                     (PORTKEY_API_KEY from the repo's .env)
#   PORTKEY_API_KEY=... scripts/gateway.sh harbor run ...
#   ENV_FILE=path/to/.env scripts/gateway.sh harbor run ...   (another file instead of the repo's)
#
# A key already in the environment wins over any file.
# Without a gateway, export ANTHROPIC_API_KEY / OPENAI_API_KEY and call `harbor run` directly.
# Secrets are passed by environment only. Harbor stores `${VAR}` templates, never their values,
# in job configs, which is why the headers below are referenced as ${PORTKEY_ANTHROPIC_HEADERS}.
set -euo pipefail

repo_env="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.env"
if [[ -z "${ENV_FILE:-}" && -f "$repo_env" ]]; then
  ENV_FILE="$repo_env"
fi
if [[ -z "${PORTKEY_API_KEY:-}" && -n "${ENV_FILE:-}" ]]; then
  # `|| true`: a file without the key must reach the message below, not stop on grep's status.
  PORTKEY_API_KEY="$({ grep -E '^PORTKEY_API_KEY=' "$ENV_FILE" || true; } | head -1 | cut -d= -f2- | tr -d '"'"'"'')"
fi
: "${PORTKEY_API_KEY:?set PORTKEY_API_KEY in the environment, or as a PORTKEY_API_KEY= line in the repository .env or ENV_FILE}"
export PORTKEY_API_KEY

# Claude Code: Anthropic Messages API through Portkey; the key travels in a custom header.
export ANTHROPIC_BASE_URL="https://api.portkey.ai"
export ANTHROPIC_API_KEY="dummy"
export PORTKEY_ANTHROPIC_HEADERS="x-portkey-api-key: ${PORTKEY_API_KEY}
x-portkey-provider: @anthropic"
# Codex, OpenCode and LiteLLM-based user agents: OpenAI-compatible API through Portkey.
export OPENAI_BASE_URL="https://api.portkey.ai/v1"
export OPENAI_API_KEY="${PORTKEY_API_KEY}"
export GATEWAY_HOST="api.portkey.ai"

exec "$@"
