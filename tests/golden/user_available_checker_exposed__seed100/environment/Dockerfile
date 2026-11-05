# One image per task; every layer above the final COPY is shared through Docker's build cache.
# A plain developer box (Python 3.12, Node 22, git, ripgrep) with the three agent CLIs and the two
# ACP adapters installed at their pinned versions, so Harbor's per-trial install has little to do.
FROM node:22-bookworm-slim AS node

FROM python:3.12-slim-bookworm
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
 && apt-get update \
 && apt-get install -y --no-install-recommends bash ca-certificates curl git procps ripgrep tmux \
 && rm -rf /var/lib/apt/lists/*

# Pinned agents (versions frozen 13 October 2026).
RUN curl -fsSL https://downloads.claude.ai/claude-code-releases/bootstrap.sh | bash -s -- 2.1.296 \
 && npm install -g @openai/codex@0.162.1 opencode-ai@1.19.0 \
      @agentclientprotocol/claude-agent-acp@0.90.0 @agentclientprotocol/codex-acp@2.2.2 \
 && npm cache clean --force \
 && /root/.local/bin/claude --version && codex --version && opencode --version

# Terminus 2, the simulated user, records its terminal with asciinema. It installs asciinema during
# each trial's setup, with exactly this command, unless it is already present; under load that
# install timed out. Baked in here, the container the agent works in is the same either way.
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y asciinema

WORKDIR /workspace
COPY workspace/ /workspace/
