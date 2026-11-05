"""What the simulated user did in the container, read from its own trajectory.

The persona may look at the workspace, but it talks only through acpx, never edits files, never
opens scorer.py or the verifier, and never starts, closes or switches sessions. These checks read
the shell commands the user agent ran (Harbor's ATIF trajectory in `user-agent/`).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .acp import RecordError
from .trial import read_json

WRITE = re.compile(
    r"(^|[^<>&0-9])>{1,2}(?!&)|\btee\b|\bsed\s+-i\b|\b(rm|mv|cp|touch|truncate|chmod)\s"
)
VERIFIER = re.compile(r"/tests\b|/logs/verifier|verify\.py|expected\.json")
SESSION_COMMANDS = re.compile(
    r"^acpx\s+(?:--\S+\s+)*(sessions?\s+(new|close|ensure|prune|import)|--agent\b)"
)
# Each is True for a violation found, False when checked and clean, and None when there was
# nothing to check (no trajectory, or one that records no commands): never read as passing.
(
    EDITED_WORKSPACE,
    OPENED_SCORER,
    TOUCHED_VERIFIER,
    MANAGED_SESSIONS,
) = PERSONA_RULES = (
    "persona_edited_workspace",
    "persona_opened_scorer",
    "persona_touched_verifier",
    "persona_managed_sessions",
)
# A breach of one of these makes the episode an infrastructure attempt, in the order the export
# reports the first; session management is kept for review.
INFRASTRUCTURE_RULES = (EDITED_WORKSPACE, TOUCHED_VERIFIER, OPENED_SCORER)


def commands(trajectory: dict[str, Any]) -> list[str]:
    """The shell input of every tool call the user agent made, in order.

    Terminus sends keystrokes; Claude Code and Codex send a command. Codex in code mode sends
    JavaScript that calls `tools.exec_command({"cmd": ...})`; those commands are extracted.
    """
    found = []
    for step in trajectory["steps"]:
        for call in step.get("tool_calls") or []:
            args = call.get("arguments") or {}
            text = args.get("keystrokes") or args.get("command") or args.get("cmd") or ""
            if isinstance(text, list):
                text = " ".join(map(str, text))
            if not text and isinstance(args.get("input"), str):
                found += code_mode_commands(args["input"])
            elif text.strip():
                found.append(text)
    return found


def code_mode_commands(source: str) -> list[str]:
    """The `cmd` of every `exec_command({...})` call in Codex code-mode JavaScript."""
    decoder = json.JSONDecoder()
    found = []
    for match in re.finditer(r"exec_command\(\s*", source):
        try:
            value, _ = decoder.raw_decode(source, match.end())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("cmd"), str):
            found.append(value["cmd"])
    return found


def shell_lines(command: str) -> list[str]:
    """Command lines, with heredoc bodies (the messages sent through acpx) removed."""
    lines, end = [], None
    for line in command.splitlines():
        if end is not None:
            if line.strip() == end:
                end = None
            continue
        heredoc = re.search(r"<<-?\s*['\"]?(\w+)['\"]?", line)
        end = heredoc.group(1) if heredoc else None
        if line.strip():
            lines.append(line.strip())
    return lines


def read_trajectory(user_agent_dir: Path) -> dict[str, Any] | None:
    """The simulated user's trajectory, or None when it left none. Raises RecordError, naming the
    trial and the file, for one that isn't JSON, whose steps aren't a list of objects, or whose
    tool calls (a step may have none) aren't objects with object arguments."""
    path = Path(user_agent_dir) / "trajectory.json"
    if not path.exists():
        return None
    trial = Path(user_agent_dir).parent.name
    trajectory = read_json(path, trial)

    def broken(what: str) -> RecordError:
        return RecordError(f"{trial}: {path}, the simulated user's trajectory, {what}")

    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        raise broken("has no list of steps")
    for number, step in enumerate(steps):
        if not isinstance(step, dict):
            raise broken(f"has step {number}, which isn't an object")
        calls = step.get("tool_calls") or []
        if not isinstance(calls, list) or not all(isinstance(c, dict) for c in calls):
            raise broken(f"has step {number}, whose tool calls aren't a list of objects")
        if not all(isinstance(c.get("arguments") or {}, dict) for c in calls):
            raise broken(
                f"has step {number}, a tool call of which has arguments that aren't an object"
            )
    return trajectory


def persona_checks(user_agent_dir: Path) -> dict[str, Any]:
    """The persona rules (`PERSONA_RULES`), checked on the simulated user's commands.
    `user_agent_record` is False when it left no trajectory; each rule is None when there was
    nothing to check. Raises RecordError as `read_trajectory` does."""
    trajectory = read_trajectory(user_agent_dir)
    if trajectory is None:
        return {"user_agent_record": False, **dict.fromkeys(PERSONA_RULES)}
    lines = [line for c in commands(trajectory) for line in shell_lines(c)]
    if not lines:
        return {"user_agent_record": True, **dict.fromkeys(PERSONA_RULES)}
    acpx = [line for line in lines if line.startswith("acpx")]
    other = [line for line in lines if not line.startswith("acpx")]
    subcommands: dict[str, int] = {}
    for line in acpx:
        words = [w for w in line.split()[1:] if not w.startswith("-")]
        name = words[0] if words else "?"
        subcommands[name] = subcommands.get(name, 0) + 1
    return {
        "user_agent_record": True,
        "persona_acpx_commands": subcommands,
        "persona_other_commands": len(other),
        EDITED_WORKSPACE: any(WRITE.search(line) for line in other),
        OPENED_SCORER: any("scorer.py" in line for line in other),
        TOUCHED_VERIFIER: any(VERIFIER.search(line) for line in lines),
        MANAGED_SESSIONS: any(SESSION_COMMANDS.search(line) for line in acpx),
    }
