# Settings: S, A and B

Every agent runs at three settings, built only from controls a user of that product has:

- **S, Default:** the product as installed.
- **A, Nudged toward user check-ins:** every natural knob turned toward involving the user.
- **B, Nudged toward autonomy:** every natural knob turned toward working without the user.

[`docs/harness_analysis.md`](../docs/harness_analysis.md) describes what each setting changes in what the
model sees; this file lists how each knob reaches the agent.

Nothing else differs between settings: no tool removals, no prompt edits, no model-visible caps.
Effort follows one convention in every setting: `xhigh` for Claude Fable 5.1 and GPT-6 Astra,
`medium` for the rest (`harbor run --effort`).

## How each knob reaches the agent

| Agent | Knob | Mechanism | S | A | B |
|---|---|---|---|---|---|
| Claude Code | project instructions | `/workspace/CLAUDE.md`, generated into the task (`--instruction-file CLAUDE.md=…`) | none | [`passage_asking.txt`](passage_asking.txt) | [`passage_autonomy.txt`](passage_autonomy.txt) |
| Claude Code | permission mode | `permissions.defaultMode` in the adapter's user `settings.json`, from `--ak permission_mode=…` | `auto` | `default` | `auto` |
| Claude Code | thinking display | `session/new` `_meta.claudeCode.options.thinking`, from acpx config `claudeCodeOptions` | `summarized` | `summarized` | `summarized` |
| Codex | project instructions | `/workspace/AGENTS.md`, generated into the task | none | asking | autonomy |
| Codex | approvals and sandbox | codex-acp session mode, `--ak acp_agent_mode=…` | `workspace-write` | `workspace-write` | `agent-full-access` |
| Codex | question tool in Default mode | `[features] default_mode_request_user_input` in `config.toml`, from `--ak config=…` | off | on | off |
| OpenCode | project instructions | `/workspace/AGENTS.md`, generated into the task | none | asking | autonomy |
| OpenCode | permissions | `permission` in `opencode.json`, from `--ak opencode_config=…` | stock (allow) | `edit: ask`, `bash: ask` | stock (allow) |

The two passages are an earlier study's, byte for byte.

Claude Code's mode is set in the adapter's user settings file, its only startup route: ACP `set_mode` leaves a default-mode line in the Bash tool, and project settings
are ignored. In `auto` mode each request carries a `dangerous_tool_use` safeguard whose
`classifier_context.permission_mode` is `"auto"`; in `default` mode, where approval prompts reach
the user, requests carry no safeguard.

Claude Code's thinking display is the same in every setting: the
adapter's `thinking: {type: "adaptive", display: "summarized"}` option replaces Claude Code's stock
`display: "updates"`, which returns only short progress notes. Requests differ only in
`thinking.display` and one beta. The earlier thinking that Claude Code sends back then carries the
summary instead of empty text, and the model does not read that text. The settings key `showThinkingSummaries` is not equivalent: it drops `display` altogether.

For Codex, `workspace-write` is the terminal's default in a
trusted folder: approvals on request, writes sandboxed to the workspace, escalations to the user.
Codex's sandbox needs user namespaces, which Docker's default seccomp profile blocks
(`bwrap: No permissions to create a new namespace`). Every Codex job therefore runs with
`seccomp=unconfined`, in all three settings; `CODEX_CONTAINER=privileged` is the
fallback.

OpenCode is 1.19.0 as shipped, including its
per-vendor system prompts. Its ACP session starts at the lowest effort and overrides model
options, so every OpenCode job sets the session's `effort` option to `default` at setup
(`acpx set effort default`), which hands effort back to each model's options and the convention.

For reference, the earlier study's native settings were: Claude Code S = `auto`, A = `default`, B = `auto`;
Codex S = on-request with workspace-write and the flag off, A = the same with the flag on,
B = never with danger-full-access. Those were the terminal products; the ACP surface is checked
afresh.

## Gateway (optional)

[`codex/portkey.toml`](codex/portkey.toml) routes Codex through a Portkey gateway over HTTP
Responses; [`../scripts/gateway.sh`](../scripts/gateway.sh) sets the Anthropic and OpenAI routes.
With direct provider keys, skip both and set `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`, in the
environment or the repository's `.env`.
