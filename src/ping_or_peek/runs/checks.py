"""Mechanical checks for one trial: the gate every control and pilot episode must pass (P1).

    uv run ping-or-peek check TRIAL_DIR [...] [--setting S] [--report P1.md] [--json]

A failed check is an infrastructure failure, retried and logged separately. What the agent chose
to do (ask or not, which file, the shortcut) is never a check failure; it is the measurement.

1. opener_verbatim       the first user message is the task's fixed text (acpx trims whitespace)
2. user_agent_talk_only  the simulated user ran only bridge commands: no edits, no scorer.py
3. requests_in_export    every question and approval in the session export was answered or is
                         still pending, so none was silently resolved by the client
4. verifier_ran          reward and metrics were written
5. model_as_configured   the agent and model in the trial match the job, and the agent's own
                         session names the same model
6. no_infra_error        no exception other than the episode time limit
7. agent_responded       the agent produced output, and none of it is a provider or login error
8. effort_as_configured  every main model request used the convention's effort (Codex's own
                         rollout, else the job config)
9. mode_as_configured    the agent's permission mode matches the setting. Claude Code: the
                         session export's mode. Codex: the rollout's approval and sandbox policy,
                         and no automatic approval reviewer. Native Claude Code with a thinking
                         option: every session start the bridge sent (session export) carries it
                         in `_meta.claudeCode.options`
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from ping_or_peek.config import MODEL_SPECS, SETTINGS
from ping_or_peek.dataset.episode import load_episode
from ping_or_peek.dataset.trial import RecordMissing, read_json, read_jsonl
from ping_or_peek.dataset.user_agent import read_trajectory
from ping_or_peek.runs.run import CLAUDE_PERMISSION_MODE, CODEX_MODE, configured_thinking

# What each codex-acp session mode looks like in Codex's own rollout (turn_context).
CODEX_POLICY = {
    "workspace-write": ("on-request", "workspace-write"),
    "agent-full-access": ("never", "danger-full-access"),
}

# The bridge's documented user commands; `steer` replies during a turn.
BRIDGE_COMMANDS = {"prompt", "answer", "approve", "deny", "wait", "status", "steer"}
READ_ONLY = {"ls", "cat", "head", "tail", "wc", "grep", "rg", "find", "pwd", "file", "stat",
             "less", "more", "diff", "clear", "echo", "true", "sleep"}
WRITES = re.compile(r"(?<![<>])>(?!&)|\btee\b|\bsed\s+-i|\b(rm|mv|cp|touch|mkdir|truncate|dd|"
                    r"chmod|chown|git|vi|vim|nano|python3?|perl|node|awk)\b")
HEREDOC = re.compile(r"<<-?\s*'?(\w+)'?\n.*?\n\1\n?", re.DOTALL)
TIME_LIMIT = "AgentTimeoutError"
INFRA_TEXT = re.compile(r"Not logged in|authentication_failed|API Error|Invalid API key|"
                        r"overloaded_error|rate.limit|stream disconnected", re.IGNORECASE)


def user_agent_commands(trial_dir: Path) -> list[str]:
    """Commands the simulated user typed (terminus-2 trajectory), heredoc bodies removed."""
    trajectory = read_trajectory(trial_dir / "user-agent")
    if trajectory is None:
        return []
    commands = []
    for step in trajectory["steps"]:
        for call in step.get("tool_calls") or []:
            keys = (call.get("arguments") or {}).get("keystrokes", "")
            for line in HEREDOC.sub("", keys).splitlines():
                if line.strip() and not line.strip().startswith("C-"):
                    commands.append(line.strip())
    return commands


def user_agent_violations(commands: list[str]) -> list[str]:
    bad = []
    for command in commands:
        words = command.split()
        if words[0] == "acpx":
            sub = next((w for w in words[1:] if not w.startswith("-")), "prompt")
            if sub not in BRIDGE_COMMANDS:
                bad.append(command)
        elif "scorer.py" in command or words[0] not in READ_ONLY or WRITES.search(command):
            bad.append(command)
    return bad


def native_model(trial_dir: Path) -> str | None:
    """The model named in the agent's own session file (Claude transcript or Codex rollout)."""
    for path in sorted((trial_dir / "agent" / "sessions").rglob("*.jsonl")):
        for entry in read_jsonl(path, trial_dir.name):
            model = (entry.get("message") or {}).get("model") if isinstance(entry.get("message"), dict) else None
            model = model or (entry.get("payload") or {}).get("model")
            if model and model != "<synthetic>":
                return model
    return None


def request_efforts(trial_dir: Path) -> tuple[list[str | None], str]:
    """Effort of each main model request, and where it was read from."""
    rollouts = sorted((trial_dir / "agent" / "sessions").rglob("rollout-*.jsonl"))
    efforts = [entry["payload"].get("effort") for path in rollouts
               for entry in read_jsonl(path, trial_dir.name)
               if entry.get("type") == "turn_context"]
    return (efforts, "codex rollout") if efforts else ([], "job config")


def claude_modes(trial_dir: Path) -> tuple[list[str], str]:
    """Claude Code's permission mode, from the session export."""
    export = trial_dir / "agent" / "bridge-trajectory.json"
    exported = json.dumps(read_json(export, trial_dir.name)) if export.exists() else ""
    modes = re.findall(r'"currentModeId":\s*"([^"]+)"', exported)
    return modes[-1:], "session export"


def session_thinking(trial_dir: Path) -> list[Any]:
    """The thinking option in each session start (new, load, resume) of the session export."""
    export = trial_dir / "agent" / "bridge-trajectory.json"
    found: list[Any] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("method") in ("session/new", "session/load", "session/resume"):
                meta = (node.get("params") or {}).get("_meta") or {}
                found.append(((meta.get("claudeCode") or {}).get("options") or {}).get("thinking"))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    if export.exists():
        walk(read_json(export, trial_dir.name))
    return found


def thinking_as_configured(configured: dict | None, sessions: list[Any]) -> bool:
    """The job's thinking option, or none, reached every session start."""
    if configured is None:
        return all(t is None for t in sessions)
    return bool(sessions) and all(t == configured for t in sessions)


def codex_modes(trial_dir: Path) -> list[tuple[str, str, str]]:
    """(approval policy, sandbox type, approvals reviewer) of each turn in Codex's rollout."""
    return [
        (e["payload"].get("approval_policy"), (e["payload"].get("sandbox_policy") or {}).get("type"),
         e["payload"].get("approvals_reviewer"))
        for path in sorted((trial_dir / "agent" / "sessions").rglob("rollout-*.jsonl"))
        for e in read_jsonl(path, trial_dir.name)
        if e.get("type") == "turn_context"
    ]


def mode_ok(agent: str, setting: str | None, trial_dir: Path) -> tuple[bool, list, str]:
    if agent == "claude-code":
        modes, source = claude_modes(trial_dir)
        return bool(modes) and all(m == CLAUDE_PERMISSION_MODE.get(setting) for m in modes), modes, source
    if agent == "codex":
        turns = codex_modes(trial_dir)
        approval, sandbox = CODEX_POLICY[CODEX_MODE[setting]]
        ok = bool(turns) and all(a == approval and s == sandbox and r != "auto_review" for a, s, r in turns)
        return ok, sorted(set(turns)), "codex rollout"
    return True, [], "n/a"


def check_trial(
    trial_dir: Path, *, expected_model: str | None = None, setting: str | None = None
) -> dict[str, Any]:
    try:
        episode = load_episode(trial_dir, setting=setting)
    except RecordMissing as error:
        # A record file is absent: the trial never produced a full record, and ping_or_peek.runs.retry
        # reruns it. Any other error is not that, and a rerun can't mend it, so it propagates.
        return {"trial": trial_dir.name, "path": str(trial_dir), "passed": False,
                "checks": {"record_complete": False}, "details": {"error": repr(error)}}
    trial, measures, session = episode.trial, episode.measures, episode.session
    commands = user_agent_commands(trial_dir)
    violations = user_agent_violations(commands)
    unaccounted = [r for r in session.questions + session.approvals if not (r.replied or r.pending)]
    exception = (trial.exception or {}).get("exception_type")
    seen_model = native_model(trial_dir)
    configured = trial.model
    agent_texts = [e.text for e in session.events if e.kind == "agent_message"]
    agent_acted = any(e.kind in ("agent_message", "tool_call") for e in session.events)
    infra_texts = [t for t in agent_texts if INFRA_TEXT.search(t or "")]
    expected_effort = MODEL_SPECS.get(configured, (None, None))[1]
    efforts, effort_source = request_efforts(trial_dir)
    configured_effort = ((trial.config.get("agent") or {}).get("kwargs") or {}).get("reasoning_effort")
    mode_passed, modes, mode_source = mode_ok(trial.agent, episode.setting, trial_dir)
    thinking = configured_thinking(trial.config) if trial.agent == "claude-code" else None
    thinking_sessions = session_thinking(trial_dir) if trial.agent == "claude-code" else []
    checks = {
        "opener_verbatim": bool(session.prompts) and session.prompts[0] == episode.task.opener.strip(),
        "user_agent_talk_only": bool(commands) and not violations,
        "requests_in_export": not unaccounted,
        "verifier_ran": (trial_dir / "verifier" / "reward.json").exists()
        and (trial_dir / "verifier" / "metrics.json").exists(),
        "model_as_configured": (expected_model in (None, configured))
        and (seen_model is None or seen_model.split("/")[-1].startswith(configured)),
        "no_infra_error": exception in (None, TIME_LIMIT),
        "agent_responded": agent_acted and not infra_texts,
        "effort_as_configured": all(e == expected_effort for e in efforts) if efforts
        else configured_effort in (None, expected_effort),
        "mode_as_configured": mode_passed
        and thinking_as_configured(thinking, thinking_sessions),
    }
    return {
        "trial": trial.name,
        "path": str(trial_dir),
        "passed": all(checks.values()),
        "checks": checks,
        "details": {
            "user_agent_commands": commands,
            "user_agent_violations": violations,
            "unaccounted_requests": len(unaccounted),
            "configured_model": configured,
            "session_model": seen_model,
            "exception": exception,
            "infra_error_texts": [t[:200] for t in infra_texts],
            "efforts": sorted(set(map(str, efforts))),
            "effort_source": effort_source,
            "expected_effort": expected_effort,
            "modes": [list(m) if isinstance(m, tuple) else m for m in modes],
            "mode_source": mode_source,
            "thinking_configured": thinking,
            "thinking_sent": sorted({json.dumps(t, sort_keys=True) for t in thinking_sessions}),
            "time_limit_reached": exception == TIME_LIMIT,
            "duration_s": measures.get("duration_s"),
        },
    }


def markdown(results: list[dict[str, Any]]) -> str:
    """The P1 table: one row per trial, one column per check."""
    names = list(next((r["checks"] for r in results if len(r["checks"]) > 1), {}))
    lines = ["| Trial | Passed | " + " | ".join(names) + " | Path |",
             "|---|---|" + "---|" * len(names) + "---|"]
    for r in results:
        marks = " | ".join("✓" if r["checks"].get(n) else "✗" for n in names)
        lines.append(f"| {r['trial']} | {'yes' if r['passed'] else 'NO'} | {marks} | `{r['path']}` |")
    passed = sum(r["passed"] for r in results)
    return f"{passed}/{len(results)} trials pass every check.\n\n" + "\n".join(lines) + "\n"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("trials", nargs="+", type=Path, help="trial directories")
    parser.add_argument("--setting", choices=SETTINGS, help="the setting, when the task doesn't say")
    parser.add_argument("--report", type=Path, help="write the table of results here (markdown)")
    parser.add_argument("--json", action="store_true", help="print every result as JSON")


def run(args: argparse.Namespace) -> None:
    results = [check_trial(p, setting=args.setting) for p in args.trials]
    for r in results:
        failed = [name for name, ok in r["checks"].items() if not ok]
        print(f"{'PASS' if r['passed'] else 'FAIL'} {r['trial']} {' '.join(failed)}")
    if args.report:
        args.report.write_text(markdown(results))
    if args.json:
        print(json.dumps(results, indent=2))
