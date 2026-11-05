"""Does a trial's record hold every input the measures and labels need? `ping-or-peek check-record`.

Run on the first trials of each kind (Claude Code, Codex, OpenCode; available and away) before
labeling at scale. Each check is a fact about the files, not a judgment about the agent.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import acp
from .acp import RecordError
from .trial import Trial, _task_metadata, find_trials, load_trial, read_json, read_jsonl
from .user_agent import PERSONA_RULES, persona_checks


def check_record(path: Path) -> dict[str, Any]:
    """Each input the measures and labels need, and whether the trial's record holds it. It reads
    the record with labeling's readers (the `Trial` accessors and `acp.parse_export`), and reports
    each rule they enforce that the record breaks as a failed check, not raised, so a run over
    many trials reports them all."""
    checks: dict[str, Any] = {}

    def check(name: str, ok: bool | None, detail: Any = None) -> None:
        checks[name] = {"ok": ok, "detail": detail}

    def recorded(read: Callable[[], Any]) -> tuple[bool, Any]:
        try:
            return True, read()
        except RecordError as error:
            return False, str(error)

    try:
        trial = load_trial(path)  # raises RecordMissing without a result.json
    except RecordError:  # each rule the record breaks is reported below
        result = recorded(lambda: read_json(path / "result.json", path.name))
        if not result[0]:  # nothing else can be read without it
            check("result.json: readable", False, result[1])
            return {
                "trial": path.name,
                "path": str(path),
                "missing": list(checks),
                "checks": checks,
            }
        config_file = path / "config.json"
        config = (
            recorded(lambda: read_json(config_file, path.name)) if config_file.exists() else None
        )
        if config is not None:
            check("config.json: readable", config[0], None if config[0] else config[1])
        trial = Trial(path, result[1], config[1] if config and config[0] else {})

    task = recorded(lambda: _task_metadata(trial))
    trial.task_metadata = task[1] if task[0] else {}
    agent = recorded(lambda: trial.agent)
    check("result.json: model", *recorded(lambda: trial.model))
    check("result.json: agent", *agent)
    check(
        "result.json: agent execution timing",
        trial.seconds("agent_execution") is not None,
        trial.seconds("agent_execution"),
    )
    agent_cfg = (trial.result.get("config") or trial.config).get("agent") or {}
    check(
        "config: reasoning effort",
        "effort" in json.dumps(agent_cfg).lower(),
        agent_cfg.get("kwargs"),
    )
    check("task: condition and seed", *task)
    check("acp export: present", trial.acp_export.exists(), str(trial.acp_export))
    session = None
    if trial.acp_export.exists():
        parsed = recorded(lambda: acp.parse_export(trial.acp_export))
        check("acp export: follows ACP", parsed[0], None if parsed[0] else parsed[1])
        session = parsed[1] if parsed[0] else None
    if session is not None:
        exported = read_json(trial.acp_export, trial.name)  # parse_export has read it
        kinds: dict[str, int] = {}
        for e in session.events:
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        check(
            "acp export: opener (first user prompt)",
            bool(session.prompts),
            (session.prompts or [""])[0][:120],
        )
        check("acp export: turn ends", bool(session.stop_reasons), session.stop_reasons)
        check(  # an agent may say nothing (e.g. it waits on an approval the away user never gives)
            "acp export: agent messages",
            kinds.get("agent_message", 0) > 0 or None,
            kinds.get("agent_message", 0),
        )
        thoughts = [e for e in session.events if e.kind == "agent_thought"]
        check(
            "acp export: thought (reasoning) text",
            any(e.text.strip() for e in thoughts) or None,
            f"{len(thoughts)} thought events, {sum(bool(e.text.strip()) for e in thoughts)} with text",
        )
        check("acp export: tool calls", bool(session.tool_calls), len(session.tool_calls))
        outputs = sum(1 for c in session.tool_calls.values() if c.content or c.raw_output)
        check(
            "acp export: tool outputs",
            outputs > 0,
            f"{outputs} of {len(session.tool_calls)} calls carry output",
        )
        check(
            "acp export: question requests (elicitations)",
            None if not session.questions else True,
            len(session.questions),
        )
        check(
            "acp export: approval requests",
            None if not session.approvals else True,
            len(session.approvals),
        )
        replied = [r for r in session.questions + session.approvals if r.replied]
        check(
            "acp export: replies to requests",
            None if not (session.questions or session.approvals) else bool(replied),
            len(replied),
        )
        check("acp export: after-turn events", None, sum(e.after_turn for e in session.events))
        check("acp export: unpaired messages", not session.unparsed, len(session.unparsed))
        check(
            "acp export: per-message timestamps",
            session.has_timestamps or None,
            session.has_timestamps,
        )
        log = exported.get("interaction_log") or []
        if log:
            logged = {
                json.dumps(e["message"], sort_keys=True) for e in log if e.get("type") == "message"
            }
            journal = [json.dumps(m, sort_keys=True) for m in exported.get("history", [])]
            missing = [m for m in journal if m not in logged]
            check(
                "bridge interaction log: holds every journaled message",
                not missing,
                f"{len(logged)} logged, {len(journal)} journaled, {len(missing)} journaled but not logged",
            )
            entry_types: dict[str, int] = {}
            for e in log:
                entry_types[e.get("type")] = entry_types.get(e.get("type"), 0) + 1
            check("bridge interaction log: entry types", True, entry_types)
        state = (exported.get("session") or {}).get("state") or {}
        check(
            "acp export: input_requests log",
            "input_requests" in state or None,
            list(state)[:30],
        )
        check("acp export: event kinds", True, kinds)
    # Codex's rollout is required: codex-acp sends no command output over ACP. Claude Code's
    # transcript is supporting evidence; OpenCode keeps none in the trial.
    native = trial.native_session_files()
    needed = {"codex": True, "claude-code": None}.get(agent[1]) if agent[0] else None
    check(
        "native session files (required for Codex)",
        bool(native) if needed else (bool(native) or None),
        [str(p.relative_to(trial.path)) for p in native][:5],
    )
    readable = recorded(lambda: [read_jsonl(p, trial.name) for p in native])
    check(
        "native session files: readable",
        readable[0] if native else None,
        None if readable[0] else readable[1],
    )
    persona = recorded(lambda: persona_checks(path / "user-agent"))
    # The export's definition (user_agent_record); a RecordError means the trajectory exists.
    user_record = not persona[0] or persona[1]["user_agent_record"]
    check(
        "user agent record",
        user_record,
        [str(p.relative_to(trial.path)) for p in trial.user_agent_files()][:10]
        if user_record
        else "no user-agent/trajectory.json: the export lists the trial as an infrastructure "
        "attempt",
    )
    if not user_record:
        steps = (None, None)  # no trajectory: reported above
    elif not persona[0]:
        steps = (False, persona[1])  # the RecordError
    elif any(persona[1][rule] is None for rule in PERSONA_RULES):
        steps = (False, "records no commands, so the persona rules went unchecked")
    else:
        steps = (True, None)
    check("user agent record: its steps", *steps)
    metrics = recorded(trial.verifier_metrics)
    check(
        "verifier: metrics",
        metrics[0] and bool(metrics[1]),
        sorted(metrics[1]) if metrics[0] else metrics[1],
    )
    exception = recorded(lambda: (trial.exception or {}).get("exception_type"))
    # The time limit is a behavioral outcome (an away user leaves requests pending until then).
    check(
        "exception (other than the time limit)",
        exception[0] and exception[1] in (None, "AgentTimeoutError"),
        exception[1],
    )
    missing = [name for name, c in checks.items() if c["ok"] is False]
    return {"trial": trial.name, "path": str(path), "missing": missing, "checks": checks}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "paths", nargs="+", type=Path, help="trial directories, or jobs holding them"
    )


def run(args: argparse.Namespace) -> None:
    for path in find_trials(args.paths):
        print(json.dumps(check_record(path), indent=2, default=str))
