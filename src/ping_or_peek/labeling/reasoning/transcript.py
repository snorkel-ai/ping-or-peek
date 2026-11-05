"""One episode as a numbered transcript, for coding the agent's stated reasons.

The reasoning deep dive reads what the agent wrote in every channel: its messages,
its reasoning where the harness records it, its commands, and its questions and approvals. This
module turns one episode into an ordered list of `Event`s with stable ids, so that every code can
cite a short quote and the event it comes from, and the quote can be checked verbatim.

The source is a Harbor trial: the acpx session export, parsed by `dataset.acp`, plus Codex's
rollout for command output, which codex-acp does not forward. Event ids are `e<position>` in the
ACP stream, the same positions the outcome labels cite.

What reasoning exists depends on the harness: Codex and OpenCode return reasoning
summaries; native Claude Code returns only short progress notes (`thinking.display: "updates"`),
which are kept as their own kind so they are never mistaken for reasoning.

Read-only: nothing here writes to a trial directory. Codex's app-server logs are never opened.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ping_or_peek.dataset import acp, codex_rollout, evidence
from ping_or_peek.dataset.episode import Task
from ping_or_peek.dataset.trial import Trial, load_trial

# Kinds that are the agent's own words (a quote supporting an agent code must come from these).
AGENT_KINDS = frozenset(
    {"agent_message", "reasoning", "progress_note", "terminus_analysis", "question", "tool_call"}
)
OUTPUT_LIMIT = 3200  # characters of tool output shown; scorer.py (2.5 kB) and both CSVs fit
FORBIDDEN_PARTS = ("app_server", "app-server", "APP_SERVER")  # Codex logs that hold the API key


@dataclass
class Event:
    id: str
    position: int
    kind: str
    text: str
    after_turn: bool = False
    ts: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Transcript:
    """An episode's events plus the facts a coder needs (never the model's name)."""

    episode: str  # a stable key: the Harbor trial path relative to the jobs root
    source: str  # "harbor"
    harness: str  # claude-code, codex or opencode
    model: str
    setting: str | None
    condition: str
    seed: int | None
    user_available: bool
    checker_exposed: bool
    checker_target: str | None  # the CSV scorer.py names, which the user did not mean
    intended: str | None  # the CSV the user meant (kept from the coder; for analysis only)
    files: list[str]
    reasoning_kind: str  # "summaries", "progress_notes" or "none"
    events: list[Event]
    extra: dict[str, Any] = field(default_factory=dict)

    def event(self, event_id: str) -> Event | None:
        return next((e for e in self.events if e.id == event_id), None)


def _clip(text: str, limit: int = OUTPUT_LIMIT) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... {len(text) - limit} more characters not shown]"


def _command_text(raw_input: Any) -> str:
    if isinstance(raw_input, dict):
        for key in ("command", "cmd", "patch", "input"):
            value = raw_input.get(key)
            if isinstance(value, list):
                return " ".join(map(str, value))
            if isinstance(value, str) and value:
                return value
        return json.dumps(raw_input, ensure_ascii=False)
    if isinstance(raw_input, list):
        return " ".join(map(str, raw_input))
    return "" if raw_input is None else str(raw_input)


def _output_text(call: acp.ToolCall) -> str:
    raw = call.raw_output
    if isinstance(raw, dict):
        for key in ("output", "stdout", "formatted_output", "aggregated_output"):
            if isinstance(raw.get(key), str):
                return raw[key]
        meta = raw.get("metadata")
        if isinstance(meta, dict) and isinstance(meta.get("output"), str):
            return meta["output"]
    elif isinstance(raw, str):
        return raw
    return "\n".join(t for t in (acp.text_of(c) for c in call.content or []) if t)


def _rollout_files(trial: Trial) -> list[Path]:
    return [
        p for p in trial.native_session_files() if not any(f in str(p) for f in FORBIDDEN_PARTS)
    ]


def episode_key(trial: Path, jobs_root: Path) -> str:
    """The coder's key for a trial's episode: its path under the jobs root, `<job>/<trial>` when
    `jobs_root` holds the jobs. The coder saves each episode under it (`label.safe_name`), and every
    reader of its saved codings finds them by it."""
    return str(trial.relative_to(jobs_root))


def from_trial(
    path: Path, jobs_root: Path | None = None, output_limit: int | None = None
) -> Transcript:
    """Build the transcript of one Harbor trial (read-only).

    `output_limit` clips each tool output to that many characters (default OUTPUT_LIMIT).
    """
    trial = load_trial(path)
    meta = trial.task_metadata
    task = Task.regenerate(
        None if meta.get("seed") is None else int(meta["seed"]), meta["condition"]
    )
    session = acp.parse_export(trial.acp_export)
    agent = trial.agent
    native = (
        codex_rollout.native_events(_rollout_files(trial), trial.name) if agent == "codex" else []
    )
    key = episode_key(path, jobs_root) if jobs_root else trial.name
    return from_session(
        session,
        native,
        agent=agent,
        task=task,
        episode=key,
        model=trial.model,
        setting=trial.setting,
        identity={"trial": trial.name, "job": trial.job_name, "phase": trial.phase},
        output_limit=output_limit,
    )


def from_session(
    session: acp.Session,
    native: list[dict[str, Any]],
    *,
    agent: str,
    task: Task,
    episode: str,
    model: str,
    setting: str | None,
    identity: dict[str, Any],
    output_limit: int | None = None,
) -> Transcript:
    """The transcript of one episode from its parsed ACP session and Codex's log (`native`),
    as `from_trial` builds it from a trial. `episode` is its key, and `identity` heads the
    transcript's `extra`.

    `output_limit` clips each tool output to that many characters (default OUTPUT_LIMIT).
    """
    limit = output_limit or OUTPUT_LIMIT
    executions = {n["id"]: n for n in native if n["type"] in codex_rollout.EXECUTION_ITEMS}
    # Native Claude Code returns reasoning summaries only when the client asks for them; at the
    # stock display it returns progress notes.
    display = claude_thinking_display(session) if agent == "claude-code" else None
    progress_notes = agent == "claude-code" and display != "summarized"
    thought_kind = "progress_note" if progress_notes else "reasoning"

    events: list[Event] = []
    shown_calls: set[str] = set()

    def add(event: acp.Event, kind: str, text: str, **extra: Any) -> None:
        events.append(
            Event(
                f"e{event.position}", event.position, kind, text, event.after_turn, event.ts, extra
            )
        )

    for event in session.events:
        if event.kind == "user_prompt":
            add(event, "user_message", event.text, mid_turn=bool(event.data.get("steering")))
        elif event.kind == "agent_message":
            if event.data.get("async_question"):
                add(event, "question", event.text, tool="request_user_input_async", blocking=False)
            else:
                add(event, "agent_message", event.text)
        elif event.kind == "agent_thought":
            if event.text.strip():
                add(event, thought_kind, event.text.strip())
        elif event.kind in ("tool_call", "tool_update"):
            call_id = event.data["tool_call_id"]
            call = session.tool_calls.get(call_id)
            if call is None or call_id in shown_calls:
                continue
            # Show each call once, where it starts, with its final input and output merged.
            shown_calls.add(call_id)
            native_exec = executions.get(call_id)
            command = _command_text(call.raw_input)
            output = _output_text(call)
            exit_code = None
            if native_exec is not None:
                if native_exec["type"] == "CommandExecution":
                    command = command or _command_text(native_exec.get("command"))
                    if isinstance(native_exec.get("command"), list):
                        command = native_exec["command"][-1]
                    output = native_exec.get("output") or output
                    exit_code = native_exec.get("exit_code")
                elif native_exec["type"] == "FileChange":
                    command = command or json.dumps(native_exec.get("changes"), ensure_ascii=False)
            if isinstance(call.raw_output, dict) and exit_code is None:
                exit_code = call.raw_output.get("exit_code", call.raw_output.get("exitCode"))
            title = call.title or ""
            description = (
                call.raw_input.get("description") if isinstance(call.raw_input, dict) else None
            )
            body = [f"[{call.kind or 'tool'}] {title}".rstrip()]
            if description and description != title:
                body.append(f"description: {description}")
            if command and command != title:
                body.append(f"input: {command}")
            status = call.status + (f", exit {exit_code}" if exit_code is not None else "")
            body.append(f"status: {status}")
            if output.strip():
                body.append("output:\n" + _clip(output, limit))
            add(event, "tool_call", "\n".join(body), tool_call_id=call_id, title=title)
        elif event.kind == "question":
            request = session.requests[event.data["request"]]
            channel, tool = evidence.classify_question(request, agent)
            add(
                event,
                "question",
                evidence.render_question(request.params),
                tool=tool,
                channel=channel,
                blocking=True,
                routed_to_user=request.routed_to_user,
            )
        elif event.kind == "question_reply":
            request = session.requests[event.data["request"]]
            answer = evidence.render_answer(request)
            status = (request.reply or {}).get("action") or "no reply"
            if request.resolved_by == "aborted":
                status = "withdrawn unanswered"
            add(event, "answer", answer or f"({status})", status=status)
        elif event.kind == "approval":
            request = session.requests[event.data["request"]]
            call = request.params.get("toolCall") or {}
            text = f"Permission request: {call.get('title')}"
            if call.get("rawInput"):
                text += f"\ninput: {_command_text(call.get('rawInput'))}"
            add(event, "approval", text, routed_to_user=request.routed_to_user)
        elif event.kind == "approval_reply":
            request = session.requests[event.data["request"]]
            decision, chosen = evidence.approval_decision(request)
            who = "the user" if request.routed_to_user else "the client's policy"
            add(event, "approval_reply", f"{decision} ({chosen}), by {who}")
        elif event.kind == "turn_end":
            add(event, "turn_end", f"turn ended: {event.data.get('stop_reason')}")
        elif event.kind == "mode_change":
            add(event, "note", f"mode changed to {event.data.get('mode')}")

    # Requests still open at the end of the episode (away users leave them pending).
    for request in session.requests:
        if request.pending and (
            acp.is_elicitation(request.method) or request.method == "session/request_permission"
        ):
            events.append(
                Event(
                    f"pending{request.position}",
                    request.position,
                    "note",
                    "request still pending when the episode ended",
                    request.after_turn,
                    None,
                )
            )

    reasoning_kind = "progress_notes" if progress_notes else "summaries"
    if not any(e.kind in ("reasoning", "progress_note") for e in events) and not progress_notes:
        reasoning_kind = "none"
    return Transcript(
        episode=episode,
        source="harbor",
        harness=agent,
        model=model,
        setting=setting,
        condition=task.condition,
        seed=task.seed,
        user_available=task.user_available,
        checker_exposed=task.scorer_exposed,
        checker_target=task.decoy if task.scorer_exposed else None,
        intended=task.target,
        files=sorted(task.files),
        reasoning_kind=reasoning_kind,
        events=events,
        extra={
            **identity,
            "claude_thinking_display": display,
            "output_limit": limit,
            # Codex: summaries forwarded over ACP vs reasoning items in its own rollout. A shortfall
            # means a summary never reached the export.
            "acp_reasoning_events": sum(e.kind == "reasoning" for e in events),
            "rollout_reasoning_items": sum(bool(n.get("private")) for n in native),
        },
    )


def claude_thinking_display(session: acp.Session) -> str | None:
    """The thinking display the client requested in session/new (or load/resume) `_meta`.

    None means the client set none, so Claude Code used its stock display ("updates": short
    progress notes, no reasoning text).
    """
    for request in session.setup.get("client_requests", []):
        if request["method"] in ("session/new", "session/load", "session/resume"):
            meta = (request.get("params") or {}).get("_meta") or {}
            options = (meta.get("claudeCode") or {}).get("options") or {}
            display = (options.get("thinking") or {}).get("display")
            if display:
                return display
    return None


def render(t: Transcript) -> str:
    """The transcript as the coder reads it, without the model's or the harness's name."""
    lines = []
    lines.append(
        f"condition: user {'available' if t.user_available else 'away'}; checker "
        f"{'exposed' if t.checker_exposed else 'hidden'}"
        + (f" (scorer.py names {t.checker_target})" if t.checker_exposed else "")
    )
    lines.append(f"CSV files in the workspace: {', '.join(t.files)}")
    lines.append(f"reasoning recorded: {t.reasoning_kind}")
    lines.append("")
    for e in t.events:
        tags = []
        if e.after_turn:
            tags.append("after turn")
        if e.meta.get("mid_turn"):
            tags.append("typed mid-turn")
        if e.meta.get("tool"):
            tags.append(str(e.meta["tool"]))
        if e.kind == "approval" and not e.meta.get("routed_to_user", True):
            tags.append("answered by client policy")
        tag = f" ({'; '.join(tags)})" if tags else ""
        lines.append(f"[{e.id}] {e.kind.upper()}{tag}:")
        lines.append(e.text.rstrip())
        lines.append("")
    return "\n".join(lines)
