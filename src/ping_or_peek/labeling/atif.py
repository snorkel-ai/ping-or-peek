"""Read the dataset's trace, an ATIF trajectory, back into what the labelers read.

The dataset's `trace` is an Agent Trajectory Interchange Format document (ATIF-v1.8, Harbor's
`harbor.models.trajectories`), written by `dataset.trajectory` from one rollout's parsed ACP
session (`acp.Session`), Codex's own log and the verifier's facts. The simulated user's facts are
the row's `task.user_facts`, not the trajectory's.
Its steps group the ACP events as Harbor's ACP agent does: a user step per prompt, agent steps
holding messages, reasoning and tool calls, and a system step per turn end. What ATIF has no
field for is in its `extra` fields:

- each step's `extra.acp_events`: the ACP events the step was made from, in stream order, each
  with its `position` in the stream, `kind` (`acp.Event.kind`), `timestamp` and `data` as parsed.
  An agent message or thought gives its text's `length` in the step's `message` or
  `reasoning_content`, which hold the texts joined. A question, approval or client operation
  holds its `request` (every `acp.Request` field, and its `index` among the requests); a question
  and an approval add a `question` or `approval` view as the user saw it. A Codex async question
  (request_user_input_async), which reached the client as agent text, is a `question` with
  `delivery` "async", its ACP `text`, and no message;
- the root `extra`: `acp` (how many events the stream had, its working directory, the thinking
  display the client asked for, whether its entries carried directions and times, and the time
  of each bookkeeping event), `agent_log` (Codex's log items, `codex_rollout.native_events`) and
  `verifier` (`episode.filesystem_facts`).

Harness bookkeeping (usage, session titles, command lists) is left out but for its times: those
stream positions read back as `other` events with no content, which no labeler reads, at the
time they were logged, which `episode.log_order_matches_timestamps` reads. The read-back
`acp.Session` holds what the labelers read and no more: no `unparsed` messages, no bridge
interaction log, and of the client's own requests only its working directory and thinking
display.

This module reads plain JSON and needs no Harbor, so the labelers and CI can use it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ping_or_peek.dataset import acp
from ping_or_peek.dataset.acp import (
    APPROVAL,
    APPROVAL_REPLY,
    CLIENT_OP,
    CLIENT_OP_RESULT,
    MODE_CHANGE,
    OTHER,
    PLAN,
    QUESTION,
    QUESTION_REPLY,
    TOOL_CALL,
    TOOL_UPDATE,
    TURN_END,
)

MESSAGE, THOUGHT, PROMPT = acp.AGENT_MESSAGE, acp.AGENT_THOUGHT, acp.USER_PROMPT
# Kinds of an `acp_events` record: `acp.Event.kind`, less `other`.
TOOL_KINDS = (TOOL_CALL, TOOL_UPDATE)
REQUEST_KINDS = (QUESTION, APPROVAL, CLIENT_OP)
OTHER_KINDS = (QUESTION_REPLY, APPROVAL_REPLY, CLIENT_OP_RESULT, PLAN, MODE_CHANGE)
KINDS = frozenset((MESSAGE, THOUGHT, PROMPT, TURN_END, *TOOL_KINDS, *REQUEST_KINDS, *OTHER_KINDS))
SESSION_UPDATE = {kind: update for update, kind in acp.TOOL_UPDATES.items()}
ASYNC = "async"  # a question's `delivery` when it reached the client as agent text
SOURCES = (SYSTEM, USER, AGENT) = ("system", "user", "agent")  # a step's source (Harbor's `Step`)
TEXTS = ("message", "reasoning_content")  # the step fields an agent step's records cut


@dataclass
class Trace:
    """What one trajectory holds of a rollout, in the labelers' terms."""

    session: acp.Session
    native: list[dict[str, Any]]  # Codex's log items; empty for other agents
    verifier: dict[str, Any]  # the verifier's facts about the final files


def invalid_fields(error: Any) -> list[str]:
    """Where a pydantic `ValidationError` found a document invalid, as each field's path, without
    the input values its text quotes."""
    return sorted({".".join(map(str, e["loc"])) for e in error.errors()})


def records(document: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """(step, record) for every `acp_events` record, in stream order."""
    return [(s, r) for s in document["steps"] for r in (s.get("extra") or {}).get("acp_events", [])]


def thinking_setup(cwd: str, display: str | None) -> dict[str, Any]:
    """`acp.Session.setup` holding what the labelers read of the client's session/new request:
    its working directory and the thinking display it asked Claude Code for."""
    params: dict[str, Any] = {"cwd": cwd}
    if display is not None:
        params["_meta"] = {"claudeCode": {"options": {"thinking": {"display": display}}}}
    return {"client_requests": [{"method": "session/new", "params": params}]}


def from_trajectory(document: dict[str, Any] | str) -> Trace:
    """The session, Codex log and verifier facts a trajectory holds. Raises ValueError, naming
    the rule, when the document isn't an ATIF trajectory, lacks the root `extra` this format
    adds (`acp`, `verifier`) or a field atif reads, holds a record of a kind outside `KINDS`, two
    records at one stream position or one past the stream's end, a reply or question whose
    request it doesn't hold, or text its records don't account for exactly. A message names a
    record or step by its order in the document, never a value the trace records."""
    try:
        return _read(document)
    except (KeyError, TypeError, IndexError, AttributeError) as error:  # its text may quote one
        raise ValueError(
            "the trajectory lacks a field atif reads, or holds one of the wrong type "
            f"({type(error).__name__})"
        ) from None


def _read(document: dict[str, Any] | str) -> Trace:
    if isinstance(document, str):
        document = json.loads(document)
    if not isinstance(document, dict) or not str(document.get("schema_version")).startswith("ATIF"):
        raise ValueError(
            "a trace is an ATIF trajectory, which revision 4a958f8 and earlier, whose trace is an "
            "event list, don't hold"
        )
    root = document.get("extra") or {}
    for name in ("acp", "verifier"):
        if name not in root:
            raise ValueError(
                f"a trace's root extra holds {name!r}; this trajectory's extra lacks it"
            )
    stream = root["acp"]
    by_position: dict[int, tuple[dict[str, Any], dict[str, Any], int]] = {}
    for n, (step, record) in enumerate(records(document)):
        position = record["position"]
        if type(position) is not int:
            raise ValueError(f"record {n} of the trace: its position isn't a stream position")
        if record["kind"] not in KINDS:
            raise ValueError(
                f"record {n} of the trace has a kind none of the kinds atif reads: {sorted(KINDS)}"
            )
        if position in by_position:
            raise ValueError(
                f"records {by_position[position][2]} and {n} of the trace hold one stream position"
            )
        if not 0 <= position < stream["events"]:
            raise ValueError(f"record {n} of the trace is past the stream's end")
        by_position[position] = (step, record, n)
    bookkeeping = dict(map(tuple, stream.get("bookkeeping", [])))
    order = {id(step): i for i, step in enumerate(document["steps"])}
    read = {id(step): dict.fromkeys(TEXTS, 0) for step in document["steps"]}

    def cut(step: dict[str, Any], field: str, length: int) -> str:
        text, start = step.get(field) or "", read[id(step)][field]
        if start + length > len(text):
            raise ValueError(
                f"step {order[id(step)]} of the trace: {field} is shorter than its records"
            )
        read[id(step)][field] = start + length
        return text[start : start + length]

    events: list[acp.Event] = []
    requests: dict[int, acp.Request] = {}
    tool_calls: dict[str, acp.ToolCall] = {}
    turn, after_turn = 0, False

    def add(kind: str, text: str = "", ts: str | None = None, **data: Any) -> acp.Event:
        event = acp.Event(len(events), kind, turn, after_turn, text, data, ts)
        events.append(event)
        return event

    def request(index: Any, n: int) -> acp.Request:
        if type(index) is not int or index not in requests:
            raise ValueError(f"record {n} of the trace names a request no earlier record holds")
        return requests[index]

    for position in range(stream["events"]):
        if position not in by_position:
            add(OTHER, ts=bookkeeping.get(position))
            continue
        step, record, n = by_position[position]
        kind, ts, data = record["kind"], record["timestamp"], record["data"]
        if record.get("request") is not None:
            fields = dict(record["request"])
            index = fields.pop("index")
            if type(index) is not int:
                raise ValueError(f"record {n} of the trace holds a request with no number")
            requests[index] = acp.Request(**fields)
        if "request" in data and kind != MESSAGE:
            request(data["request"], n)
        if kind == PROMPT:
            if not data.get("steering"):
                turn, after_turn = turn + 1, False
            add(kind, cut(step, "message", len(step["message"])), ts, **data)
        elif kind == QUESTION and record.get("delivery") == ASYNC:
            add(MESSAGE, record["text"], ts, **data)
        elif kind == MESSAGE:
            add(kind, cut(step, "message", record["length"]), ts, **data)
        elif kind == THOUGHT:
            add(kind, cut(step, "reasoning_content", record["length"]), ts, **data)
        elif kind in TOOL_KINDS:
            # The parser's own merge, so each call's state is what the labelers read.
            update = {"sessionUpdate": SESSION_UPDATE[kind], **data["update"]}
            try:
                acp._add_update(update, events, tool_calls, add, ts)
            except acp.RecordError as error:  # a published trace isn't a trial's record
                raise ValueError(f"record {n} of the trace: {error}") from None
        elif kind == QUESTION:
            add(kind, request(data["request"], n).params.get("message", ""), ts, **data)
        else:  # approvals, client operations, replies, plans, mode changes, turn ends
            add(kind, "", ts, **data)
            if kind == TURN_END:
                after_turn = True
    for i, step in enumerate(document["steps"]):
        for field in TEXTS:
            if read[id(step)][field] != len(step.get(field) or ""):
                raise ValueError(
                    f"step {i} of the trace: {field} holds text its records leave unread"
                )
    if sorted(requests) != list(range(len(requests))):
        raise ValueError(f"the trace's {len(requests)} requests aren't numbered from 0 on")
    session = acp.Session(
        events=events,
        prompts=[e.text for e in events if e.kind == PROMPT],
        requests=[requests[i] for i in sorted(requests)],
        tool_calls=tool_calls,
        stop_reasons=[acp.stop_reason(e.data) for e in events if e.kind == TURN_END],
        setup=thinking_setup(stream["cwd"], stream["thinking_display"]),
        has_direction=stream["has_direction"],
        has_timestamps=stream["has_timestamps"],
        unparsed=[],
    )
    return Trace(session, root.get("agent_log", []), root["verifier"])
