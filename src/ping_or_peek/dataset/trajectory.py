"""Write one rollout's trace as an ATIF trajectory, with Harbor's models (`labeling.atif` reads it
back and documents the format).

The steps follow Harbor's ACP agent (`_convert_events_to_trajectory` in
`harbor/agents/installed/acp.py`), whose helpers name each tool, normalize its arguments and turn
its output into text: a reasoning or message chunk, or a new tool call, after a tool call has
completed starts a new agent step. Unlike Harbor's agent, which has no session of its own to
record, each prompt is a user step and each turn end a system step, and every other ACP event
(questions, approvals, client operations, replies, plans, mode changes) stays in the step it
came in, in that step's `extra.acp_events`.

`check_reads_back` refuses a trajectory unless `atif.from_trajectory` reads back the session the
labelers read.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from harbor.agents.installed.acp import (
    _normalize_tool_arguments,
    _resolve_tool_name,
    _stringify_tool_output,
)
from harbor.models.trajectories import (
    Agent,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from pydantic import ValidationError

from ping_or_peek.labeling import atif
from ping_or_peek.labeling.reasoning.transcript import claude_thinking_display

from . import acp, evidence
from .acp import APPROVAL, OTHER, QUESTION
from .build import (
    BY_USER,
    VIEW_VOCABULARY,
    approval_answered_by,
    model_family,
    question_answered_by,
    trace_values,
)

NOTES = (
    "Converted from the ACP session the client recorded, grouped into steps as Harbor's ACP "
    "agent groups them. extra.acp_events holds each step's ACP events, extra.agent_log Codex's "
    "own log, extra.verifier the verifier's facts about the final files."
)


@dataclasses.dataclass
class _Tool:
    name: str
    arguments: dict[str, Any] = dataclasses.field(default_factory=dict)
    chunks: list[str] = dataclasses.field(default_factory=list)
    terminals: list[dict[str, Any]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class _Step:
    """An agent step being collected, as Harbor's `_AcpStepState`."""

    timestamp: str | None = None
    message: list[str] = dataclasses.field(default_factory=list)
    reasoning: list[str] = dataclasses.field(default_factory=list)
    tools: dict[str, _Tool] = dataclasses.field(default_factory=dict)
    records: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    completed: bool = False


def checked(view: dict[str, Any], name: str) -> dict[str, Any]:
    """`view`, once each of its closed fields holds a value of VIEW_VOCABULARY[name] or null.
    Raises ValueError naming the field, never the value."""
    for field, values in VIEW_VOCABULARY[name].items():
        if view.get(field) is not None and view[field] not in values:
            raise ValueError(f"{field} of a {name} view is outside build.VIEW_VOCABULARY")
    return view


def question_view(request: acp.Request, agent: str) -> dict[str, Any]:
    """A question-tool request as the user saw it, with the reply, as the labelers read it."""
    _, tool = evidence.classify_question(request, agent)
    view = {
        "tool": tool,
        "text": evidence.render_question(request.params),
        "answer": evidence.render_answer(request) or None,
        "status": evidence.question_status(request),
        "answered_by": question_answered_by(request),
    }
    return checked(view, QUESTION)


def approval_view(request: acp.Request) -> dict[str, Any]:
    """A permission request's options, the option chosen, the decision recorded (even when the
    request was then withdrawn), its status, and who answered."""
    decision, chosen = evidence.approval_decision(request)
    view = {
        "options": [o.get("name") for o in request.params.get("options", [])],
        "answer": chosen or None,
        "decision": decision,
        "status": evidence.approval_status(request),
        "answered_by": approval_answered_by(request),
    }
    return checked(view, APPROVAL)


def async_view(contact: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """A Codex async question, shaped like a question-tool request: the user contact
    `evidence.build_items` makes of it, and the options Codex offered."""
    view = {
        "tool": contact["tool"],
        "text": contact["question"],
        "options": [o for q in data.get("questions") or [] for o in q.get("options") or []],
        "answer": contact["answer"] or None,
        "status": contact["status"],
        "reason": contact["reason"],
        # The reply, when there is one, is the user's next message.
        "answered_by": BY_USER if contact["status"] == evidence.ANSWERED else None,
        "reply_position": contact["reply_position"],
        "answered_mid_turn": contact["answered_mid_turn"],
        "rollout_id": contact["rollout_id"],
    }
    return checked(view, QUESTION)


def to_trajectory(
    session: acp.Session,
    native: list[dict[str, Any]],
    *,
    episode_id: str,
    agent: str,
    agent_version: str,
    model: str,
    user_mode: str,
    verifier: dict[str, Any],
) -> dict[str, Any]:
    """The rollout's ATIF trajectory as a JSON document, validated with Harbor's models.
    `user_mode` is "available" or "away", as the labelers read the episode. Raises ValueError,
    naming the rule and never a recorded value, when a closed value is outside build.VOCABULARY
    (`trace_values`) or the trajectory fails Harbor's validation (by its fields' paths)."""
    trace_values(session, native, agent, model_family(model))  # closed values, by the registry
    try:
        return _trajectory(
            session,
            native,
            episode_id=episode_id,
            agent=agent,
            agent_version=agent_version,
            model=model,
            user_mode=user_mode,
            verifier=verifier,
        )
    except ValidationError as error:  # its text quotes the recorded input
        raise ValueError(
            f"{episode_id}: not a valid ATIF-v1.8 trajectory at {atif.invalid_fields(error)}"
        ) from None


def _trajectory(
    session: acp.Session,
    native: list[dict[str, Any]],
    *,
    episode_id: str,
    agent: str,
    agent_version: str,
    model: str,
    user_mode: str,
    verifier: dict[str, Any],
) -> dict[str, Any]:
    _, contacts = evidence.build_items(session, agent=agent, user_mode=user_mode, native=native)
    asked = {c["agent_position"]: c for c in contacts if c["channel"] == evidence.ASYNC_QUESTION[0]}
    terminals = session.terminal_results()
    shown: set[tuple[str, str]] = set()  # (tool call, terminal) already in an observation
    steps: list[Step] = []
    current: _Step | None = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        steps.append(
            Step(
                step_id=len(steps) + 1,
                timestamp=current.timestamp,
                source=atif.AGENT,
                message="".join(current.message),
                reasoning_content="".join(current.reasoning) or None,
                tool_calls=[
                    ToolCall(tool_call_id=i, function_name=t.name, arguments=t.arguments)
                    for i, t in current.tools.items()
                ]
                or None,
                observation=Observation(
                    results=[
                        ObservationResult(
                            source_call_id=i,
                            content="\n\n".join(t.chunks) or None,
                            extra={"terminals": t.terminals} if t.terminals else None,
                        )
                        for i, t in current.tools.items()
                        if t.chunks or t.terminals
                    ]
                )
                if any(t.chunks or t.terminals for t in current.tools.values())
                else None,
                extra={"acp_events": current.records},
            )
        )
        current = None

    def step(timestamp: str | None) -> _Step:
        nonlocal current
        if current is None:
            current = _Step(timestamp)
        return current

    def alone(source: str, message: str, event: acp.Event) -> None:
        flush()
        record = {"position": event.position, "kind": event.kind, "timestamp": event.ts}
        steps.append(
            Step(
                step_id=len(steps) + 1,
                timestamp=event.ts,
                source=source,
                message=message,
                extra={"acp_events": [{**record, "data": event.data}]},
            )
        )

    for event in session.events:
        record = {"position": event.position, "kind": event.kind, "timestamp": event.ts}
        if event.kind == atif.PROMPT:
            alone(atif.USER, event.text, event)
        elif event.kind == atif.TURN_END:
            alone(atif.SYSTEM, "", event)
        elif event.kind in (atif.MESSAGE, atif.THOUGHT):
            if current is not None and current.completed:
                flush()
            if event.position in asked:  # a Codex async question, in place of its message
                view = async_view(asked[event.position], event.data)
                record.update(kind=QUESTION, delivery=atif.ASYNC, text=event.text)
                step(event.ts).records.append({**record, "data": event.data, "question": view})
                continue
            collecting = step(event.ts)
            texts = collecting.message if event.kind == atif.MESSAGE else collecting.reasoning
            texts.append(event.text)
            collecting.records.append({**record, "data": event.data, "length": len(event.text)})
        elif event.kind in atif.TOOL_KINDS:
            update = event.data["update"]
            call_id = event.data["tool_call_id"]
            if current is not None and current.completed and call_id not in current.tools:
                flush()
            tool = step(event.ts).tools.setdefault(call_id, _Tool(_resolve_tool_name(update)))
            if update.get("rawInput") is not None:
                tool.arguments = _normalize_tool_arguments(update["rawInput"])
            text = _stringify_tool_output(update.get("rawOutput"), update.get("content"))
            if text:
                tool.chunks.append(text)
            # A terminal block has no text of its own: the output, exit code and truncation flag
            # are the client terminal's, which Harbor's helper doesn't read.
            for block in update.get("content") or []:
                if not isinstance(block, dict) or block.get("type") != "terminal":
                    continue
                key = (call_id, block.get("terminalId"))
                if key in shown or key[1] not in terminals:
                    continue
                shown.add(key)
                terminal = terminals[key[1]]
                if terminal.get("output"):
                    tool.chunks.append(terminal["output"])
                tool.terminals.append(
                    {
                        "terminal_id": key[1],
                        "exit_code": terminal.get("exit_code"),
                        "truncated": terminal.get("truncated"),
                    }
                )
            current.completed |= update.get("status") == acp.TOOL_COMPLETED
            current.records.append({**record, "data": event.data})
        elif event.kind == OTHER:
            continue  # harness bookkeeping: its time goes in the root's `acp.bookkeeping`
        else:
            record["data"] = event.data
            if event.kind in atif.REQUEST_KINDS:
                index = event.data["request"]
                request = session.requests[index]
                record["request"] = {"index": index, **dataclasses.asdict(request)}
                if event.kind == QUESTION:
                    record["question"] = question_view(request, agent)
                elif event.kind == APPROVAL:
                    record["approval"] = approval_view(request)
            step(event.ts).records.append(record)
    flush()
    trajectory = Trajectory(
        schema_version="ATIF-v1.8",
        session_id=episode_id,
        agent=Agent(name=agent, version=agent_version, model_name=model),
        steps=steps,
        notes=NOTES,
        extra={
            "acp": {
                "events": len(session.events),
                "cwd": session.cwd,
                "thinking_display": claude_thinking_display(session),
                "has_direction": session.has_direction,
                "has_timestamps": session.has_timestamps,
                "bookkeeping": [
                    [e.position, e.ts] for e in session.events if e.kind == OTHER and e.ts
                ],
            },
            **({"agent_log": native} if native else {}),
            "verifier": verifier,
        },
    )
    return trajectory.to_json_dict()


def _ordered(value: Any) -> str:
    """`value` as JSON text, dataclasses as their fields, every dict in its own key order: the
    labelers' inputs follow that order, so two values match only if it does too."""
    return json.dumps(value, ensure_ascii=False, default=dataclasses.asdict)


def check_reads_back(
    document: dict[str, Any],
    session: acp.Session,
    native: list[dict[str, Any]],
    verifier: dict[str, Any],
) -> None:
    """Raise ValueError unless the trajectory, as a reader gets it (serialized, validated and
    read back), holds the session the labelers read, every dict in its key order: every event
    but bookkeeping, and the position, kind and time of every event; the requests with their
    replies, the tool calls' merged state, the prompts, stop reasons, working directory, thinking
    display, and whether the stream recorded directions and times; and the Codex log and
    verifier facts as given."""
    text = json.dumps(document, ensure_ascii=False)
    try:
        Trajectory.model_validate_json(text)
    except ValidationError as error:  # its text quotes the recorded input
        raise ValueError(
            f"{document.get('session_id')}: not a valid ATIF-v1.8 trajectory at "
            f"{atif.invalid_fields(error)}"
        ) from None
    try:
        read = atif.from_trajectory(text)
    except ValueError as error:
        raise ValueError(
            f"{document.get('session_id')}: the trajectory doesn't read back: {error}"
        ) from None
    got, want = read.session, session

    def stream(s: acp.Session) -> list[tuple[int, str, str | None]]:
        return [(e.position, e.kind, e.ts) for e in s.events]

    differ = [
        name
        for name, a, b in (
            ("events", [e for e in got.events if e.kind != OTHER],
             [e for e in want.events if e.kind != OTHER]),
            ("stream", stream(got), stream(want)),
            ("requests", got.requests, want.requests),
            ("tool calls", got.tool_calls, want.tool_calls),
            ("prompts", got.prompts, want.prompts),
            ("stop reasons", got.stop_reasons, want.stop_reasons),
            ("working directory", got.cwd, want.cwd),
            ("thinking display", claude_thinking_display(got), claude_thinking_display(want)),
            ("directions and times", (got.has_direction, got.has_timestamps),
             (want.has_direction, want.has_timestamps)),
            ("agent log", read.native, native),
            ("verifier", read.verifier, verifier),
        )
        if _ordered(a) != _ordered(b)
    ]  # fmt: skip
    if differ:
        raise ValueError(
            f"{document.get('session_id')}: the trajectory doesn't read back: {differ}"
        )
