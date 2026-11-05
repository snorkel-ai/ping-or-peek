"""Read an `acpx sessions export` file into the ordered events of one conversation.

Harbor's simulated user talks to the agent under test through acpx, an Agent Client Protocol
(ACP) client. The export's `history` holds every ACP JSON-RPC message between acpx and the agent,
in the order acpx saw them. This module turns that stream into typed events:

- the user's prompts, and the end of each agent turn with its stop reason;
- the agent's messages and thoughts (reasoning), with streamed chunks merged;
- tool calls and their updates;
- the file and terminal operations the agent asked acpx to run on its behalf;
- questions (ACP elicitations) and approvals (ACP permission requests), each with its reply.

With the ask-user bridge, the export also carries an `interaction_log`: every wire message with
its direction and time, plus which questions and approvals were handed to the simulated user and
how each was resolved. The parser prefers that log, and falls back to `history` for stock acpx.

Nothing is interpreted here. `evidence.py` decides what the events mean for the experiment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Closed vocabularies of the ACP schema (@agentclientprotocol/sdk 1.5.0, schema.json): a tool
# call's kind and status, why a turn stopped, and the file operations a client runs.
TOOL_KINDS = (
    TOOL_READ,
    TOOL_EDIT,
    TOOL_DELETE,
    TOOL_MOVE,
    TOOL_SEARCH,
    TOOL_EXECUTE,
    TOOL_THINK,
    TOOL_FETCH,
    TOOL_SWITCH_MODE,
    TOOL_OTHER,
) = (
    "read",
    "edit",
    "delete",
    "move",
    "search",
    "execute",
    "think",
    "fetch",
    "switch_mode",
    "other",
)
TOOL_STATUSES = (TOOL_PENDING, TOOL_IN_PROGRESS, TOOL_COMPLETED, TOOL_FAILED) = (
    "pending",
    "in_progress",
    "completed",
    "failed",
)
STOP_REASONS = ("end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled")
FS_METHODS = (READ_TEXT_FILE, WRITE_TEXT_FILE) = ("fs/read_text_file", "fs/write_text_file")

# Methods the client (acpx) sends to the agent. Everything else that has a method comes from the
# agent. Used only when a journal entry does not record its direction.
CLIENT_METHODS = frozenset(
    {
        "initialize",
        "authenticate",
        "session/new",
        "session/load",
        "session/resume",
        "session/list",
        "session/prompt",
        "session/cancel",
        "session/set_mode",
        "session/set_model",
        "session/set_config_option",
        "session/status",
        "_session/steering",  # codex-acp: a user message delivered mid-turn
    }
)


class RecordError(Exception):
    """A trial's record breaks a rule the analysis relies on, such as a session with no working
    directory. A rerun can't mend it, so it reaches callers such as `ping_or_peek.runs.checks` by name; only
    a record that is absent (`trial.RecordMissing`) counts as one the trial never produced."""


# Event kinds (`Event.kind`), and the kind of each tool-call update (`sessionUpdate`).
(
    USER_PROMPT,
    TURN_END,
    AGENT_MESSAGE,
    AGENT_THOUGHT,
    TOOL_CALL,
    TOOL_UPDATE,
    PLAN,
    MODE_CHANGE,
    QUESTION,
    QUESTION_REPLY,
    APPROVAL,
    APPROVAL_REPLY,
    CLIENT_OP,
    CLIENT_OP_RESULT,
    OTHER,
) = EVENT_KINDS = (
    "user_prompt",
    "turn_end",
    "agent_message",
    "agent_thought",
    "tool_call",
    "tool_update",
    "plan",
    "mode_change",
    "question",
    "question_reply",
    "approval",
    "approval_reply",
    "client_op",
    "client_op_result",
    "other",
)
TOOL_UPDATES = {"tool_call": TOOL_CALL, "tool_call_update": TOOL_UPDATE}


@dataclass
class Event:
    """One step of the conversation, in stream order.

    `kind` is one of: user_prompt, turn_end, agent_message, agent_thought, tool_call,
    tool_update, plan, mode_change, question, question_reply, approval, approval_reply,
    client_op, client_op_result, other.
    """

    position: int
    kind: str
    turn: int
    after_turn: bool
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    ts: str | None = None


@dataclass
class Request:
    """A request the agent sent to the client, and the client's reply (None while pending)."""

    method: str
    params: dict[str, Any]
    position: int
    turn: int
    after_turn: bool
    ts: str | None = None
    reply: dict[str, Any] | None = None
    reply_error: dict[str, Any] | None = None
    reply_position: int | None = None
    reply_ts: str | None = None
    replied: bool = False  # a reply can legitimately be null, so track arrival separately
    routed_to_user: bool = False  # handed to the simulated user (bridge), not answered by policy
    resolved_by: str | None = None  # "caller" (the user), "aborted", or None

    @property
    def pending(self) -> bool:
        return not self.replied


@dataclass
class ToolCall:
    """The merged state of one tool call: its first announcement plus every later update."""

    tool_call_id: str
    title: str = ""
    kind: str = ""
    status: str = ""
    raw_input: Any = None
    raw_output: Any = None
    content: list[Any] = field(default_factory=list)
    locations: list[Any] = field(default_factory=list)
    position: int = 0
    last_position: int = 0
    completed_position: int | None = None  # the update that first reported completed or failed


@dataclass
class Session:
    """Everything parsed from one export."""

    events: list[Event]
    prompts: list[str]
    requests: list[Request]
    tool_calls: dict[str, ToolCall]
    stop_reasons: list[str]
    setup: dict[str, Any]
    has_direction: bool
    has_timestamps: bool
    unparsed: list[dict[str, Any]]
    state: dict[str, Any] = field(default_factory=dict)  # the export's session record
    interaction: list[dict[str, Any]] = field(default_factory=list)  # bridge log, non-message

    @property
    def caller_input(self) -> bool:
        """Whether acpx ran in the bridge's caller-input mode for this session.

        Stock acpx answers permission requests itself, by policy (approve-all in Harbor), and
        cancels elicitations. Which requests reached the user is per request: `routed_to_user`.
        """
        return bool(self.interaction) or "input_requests" in self.state

    @property
    def cwd(self) -> str:
        """The agent's working directory, from the session/new (or load) request. Raises
        RecordError when no such request records one: write detection resolves paths against it."""
        for request in self.setup.get("client_requests", []):
            if request["method"] in ("session/new", "session/load", "session/resume"):
                cwd = (request.get("params") or {}).get("cwd")
                if cwd:
                    return cwd.rstrip("/")
        raise RecordError("the session records no cwd in a session/new, load or resume request")

    def requests_of(self, *prefixes: str) -> list[Request]:
        return [r for r in self.requests if r.method.startswith(prefixes)]

    @property
    def questions(self) -> list[Request]:
        return [r for r in self.requests if is_elicitation(r.method)]

    @property
    def approvals(self) -> list[Request]:
        return self.requests_of("session/request_permission")

    def terminal_results(self) -> dict[str, dict[str, Any]]:
        """Output, exit code and whether the client truncated the output, of each client-run
        terminal, by terminal id."""
        results: dict[str, dict[str, Any]] = {}
        for r in self.requests_of("terminal/"):
            terminal = r.params.get("terminalId") or (r.reply or {}).get("terminalId")
            entry = results.setdefault(terminal, {})
            if r.method == "terminal/create":
                entry["command"] = r.params.get("command")
            reply = r.reply if isinstance(r.reply, dict) else {}
            if "output" in reply:
                entry["output"] = reply["output"]
            exit_status = reply.get("exitStatus") or (reply if "exitCode" in reply else None)
            if exit_status:
                entry["exit_code"] = exit_status.get("exitCode")
            if isinstance(reply.get("truncated"), bool):  # the client shortened the output
                entry["truncated"] = reply["truncated"]
        return results


def is_elicitation(method: str) -> bool:
    return "elicitation" in method


def _unwrap(entry: dict[str, Any]) -> tuple[dict[str, Any], str | None, str | None]:
    """Return (message, direction, timestamp) for one journal entry.

    Stock acpx stores the bare JSON-RPC message. A journal that records direction and time may
    wrap it as {"message": ..., "direction": ..., "ts": ...} or add an "_acpx" field.
    """
    if "message" in entry and isinstance(entry["message"], dict) and "jsonrpc" not in entry:
        meta = entry
        message = entry["message"]
    else:
        meta = entry.get("_acpx") or {}
        message = entry
    direction = meta.get("direction")
    ts = meta.get("ts") or meta.get("timestamp")
    return message, direction, ts


def stop_reason(data: dict[str, Any]) -> str:
    """A turn's stop reason as `Session.stop_reasons` lists it, from its turn_end event's data:
    the prompt reply's, else "error" when the prompt failed, else "unknown"."""
    return data.get("stop_reason") or ("error" if data.get("error") else "unknown")


def text_of(content: Any) -> str:
    """The text of an ACP content block, or of a list of them."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(text_of(c) for c in content)
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        if content.get("type") == "content":  # ToolCallContent wrapping a content block
            return text_of(content.get("content"))
        if content.get("type") == "resource":
            return (content.get("resource") or {}).get("text", "")
    return ""


def _is_reply_shape(method: str, result: Any) -> bool:
    """Whether `result` looks like the reply to `method`. Disambiguates colliding ids."""
    if not isinstance(result, dict):
        return method in (WRITE_TEXT_FILE, "terminal/release", "terminal/kill") or (
            method.startswith("session/set")
        )
    keys = set(result)
    expected = {
        "session/prompt": {"stopReason"},
        "session/new": {"sessionId"},
        "initialize": {"protocolVersion"},
        READ_TEXT_FILE: {"content"},
        "terminal/create": {"terminalId"},
        "terminal/output": {"output"},
        "terminal/wait_for_exit": {"exitCode", "signal"},
        "session/request_permission": {"outcome"},
    }
    if method in expected:
        return bool(keys & expected[method])
    if is_elicitation(method):
        return "action" in keys
    return True


class _Pending:
    """Open requests in one direction, keyed by JSON-RPC id."""

    def __init__(self) -> None:
        self.by_id: dict[Any, tuple[str, Any]] = {}

    def add(self, msg_id: Any, method: str, target: Any) -> None:
        self.by_id[json.dumps(msg_id)] = (method, target)

    def get(self, msg_id: Any) -> tuple[str, Any] | None:
        return self.by_id.get(json.dumps(msg_id))

    def pop(self, msg_id: Any) -> tuple[str, Any] | None:
        return self.by_id.pop(json.dumps(msg_id), None)


def read_record_text(path: Path, trial: str) -> str:
    """One of `trial`'s plain-text record files. Raises RecordError, naming the trial and the
    file, when it is absent or isn't text."""
    try:
        return Path(path).read_text()
    except FileNotFoundError as error:
        raise RecordError(f"{trial}: {path} is absent") from error
    except UnicodeDecodeError as error:
        raise RecordError(f"{trial}: {path} is not text ({error})") from error


def parse_export(path: Path) -> Session:
    """Parse one `acpx sessions export` JSON file. Raises RecordError, naming the trial and the
    file, for one that isn't a JSON object, breaks ACP or records no working directory, so every
    reader sees the same valid session."""
    trial = Path(path).parent.parent.name  # <trial>/agent/<export>
    try:
        exported = json.loads(read_record_text(path, trial))
    except json.JSONDecodeError as error:
        raise RecordError(f"{trial}: {path} is not JSON ({error})") from error
    if not isinstance(exported, dict):
        raise RecordError(f"{trial}: {path} is not a JSON object")
    try:
        return _parse_exported(exported)
    except RecordError as error:
        raise RecordError(f"{trial}: {path}: {error}") from error


def _parse_exported(exported: dict[str, Any]) -> Session:
    for key in ("interaction_log", "history"):
        entries = exported.get(key, [])
        if not isinstance(entries, list):
            raise RecordError(f"its {key} is not a list")
        for number, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise RecordError(f"entry {number} of its {key} isn't an object")
            message = entry.get("message")
            if (
                key == "interaction_log"
                and entry.get("type") == "message"
                and not isinstance(message, dict)
            ):
                raise RecordError(f"message {number} of its {key} isn't an object")
    log = exported.get("interaction_log") or []
    if log:
        wire = [
            {"message": e["message"], "direction": e.get("direction"), "ts": e.get("at")}
            for e in log
            if e.get("type") == "message"
        ]
        session = parse_history(wire)
        session.interaction = [e for e in log if e.get("type") != "message"]
        _mark_routed(session)
    else:
        session = parse_history(exported.get("history", []))
    session.state = (exported.get("session") or {}).get("state") or {}
    _ = session.cwd  # validated here, where the record is read, not first by write detection
    return session


def _mark_routed(session: Session) -> None:
    """Match the bridge's input_requested/input_resolved entries to the agent's requests.

    Questions match by tool call id, else message; approvals by the tool call's id. When several
    requests share a key, they are matched in order.
    """
    resolved = {e["id"]: e for e in session.interaction if e.get("type") == "input_resolved"}
    candidates = [
        r
        for r in session.requests
        if r.method == "session/request_permission" or is_elicitation(r.method)
    ]
    for entry in (e for e in session.interaction if e.get("type") == "input_requested"):
        view = entry.get("input") or {}
        if view.get("kind") not in ("question", "permission"):
            continue  # e.g. an async question forwarded as a message: no request to match
        if view.get("kind") == "permission":
            key = ((view.get("toolCall") or {}).get("toolCallId"), None)
        else:
            key = (view.get("toolCallId"), view.get("message"))
        for request in candidates:
            if request.routed_to_user:
                continue
            if request.method == "session/request_permission":
                mine = ((request.params.get("toolCall") or {}).get("toolCallId"), None)
            else:
                mine = (request.params.get("toolCallId"), request.params.get("message"))
            if (key[0] and key[0] == mine[0]) or (not key[0] and key[1] == mine[1]):
                request.routed_to_user = True
                request.resolved_by = (resolved.get(view.get("id")) or {}).get("by")
                request.ts = request.ts or view.get("requestedAt")
                break


def parse_history(history: list[dict[str, Any]]) -> Session:
    events: list[Event] = []
    requests: list[Request] = []
    tool_calls: dict[str, ToolCall] = {}
    prompts: list[str] = []
    stop_reasons: list[str] = []
    setup: dict[str, Any] = {"client_requests": []}
    unparsed: list[dict[str, Any]] = []
    from_client = _Pending()  # requests acpx sent to the agent
    from_agent = _Pending()  # requests the agent sent to acpx
    turn, after_turn = 0, False
    has_direction = has_timestamps = False

    def add(kind: str, text: str = "", ts: str | None = None, **data: Any) -> Event:
        event = Event(len(events), kind, turn, after_turn, text, data, ts)
        events.append(event)
        return event

    for entry in history:
        message, direction, ts = _unwrap(entry)
        has_direction |= direction is not None
        has_timestamps |= ts is not None
        method = message.get("method")
        msg_id = message.get("id")

        if method is not None:
            outbound = direction == "outbound" if direction else method in CLIENT_METHODS
            if outbound:
                params = message.get("params") or {}
                if method == "session/prompt":
                    turn += 1
                    after_turn = False
                    text = text_of(params.get("prompt"))
                    prompts.append(text)
                    add(USER_PROMPT, text, ts)
                elif method == "_session/steering":  # typed while the agent works; no new turn
                    text = text_of(params.get("prompt"))
                    prompts.append(text)
                    add(USER_PROMPT, text, ts, steering=True)
                elif method != "session/cancel":
                    setup["client_requests"].append({"method": method, "params": params})
                if msg_id is not None:
                    from_client.add(msg_id, method, params)
                continue

            params = message.get("params") or {}
            if msg_id is None:  # a notification from the agent
                if method == "session/update":
                    _add_update(params.get("update") or {}, events, tool_calls, add, ts)
                else:
                    add(OTHER, "", ts, method=method, params=params)
                continue

            request = Request(method, params, len(events), turn, after_turn, ts)
            requests.append(request)
            from_agent.add(msg_id, method, request)
            if is_elicitation(method):
                add(QUESTION, params.get("message", ""), ts, request=len(requests) - 1)
            elif method == "session/request_permission":
                add(APPROVAL, "", ts, request=len(requests) - 1)
            else:
                add(CLIENT_OP, "", ts, request=len(requests) - 1, method=method)
            continue

        # A response. Find which open request it answers.
        if msg_id is None:
            unparsed.append(message)
            continue
        result, error = message.get("result"), message.get("error")
        client_side = from_client.get(msg_id)
        agent_side = from_agent.get(msg_id)
        if direction:
            # inbound = the agent answering acpx; outbound = acpx answering the agent
            answers_agent = direction == "outbound"
        elif client_side and agent_side:
            answers_agent = _is_reply_shape(agent_side[0], result) and not _is_reply_shape(
                client_side[0], result
            )
        else:
            answers_agent = agent_side is not None
        if answers_agent and agent_side:
            from_agent.pop(msg_id)
            request = agent_side[1]
            request.reply, request.reply_error, request.replied = result, error, True
            request.reply_position, request.reply_ts = len(events), ts
            index = requests.index(request)
            if is_elicitation(request.method):
                add(QUESTION_REPLY, "", ts, request=index)
            elif request.method == "session/request_permission":
                add(APPROVAL_REPLY, "", ts, request=index)
            else:
                add(CLIENT_OP_RESULT, "", ts, request=index, method=request.method)
        elif client_side:
            from_client.pop(msg_id)
            method_sent, _ = client_side
            if method_sent == "session/prompt":
                reason = (result or {}).get("stopReason") if isinstance(result, dict) else None
                event = add(TURN_END, "", ts, stop_reason=reason, error=error)
                stop_reasons.append(stop_reason(event.data))
                after_turn = True
            else:
                setup.setdefault("replies", {}).setdefault(method_sent, []).append(
                    result if error is None else {"error": error}
                )
        else:
            unparsed.append(message)

    return Session(
        events=events,
        prompts=prompts,
        requests=requests,
        tool_calls=tool_calls,
        stop_reasons=stop_reasons,
        setup=setup,
        has_direction=has_direction,
        has_timestamps=has_timestamps,
        unparsed=unparsed,
    )


def _async_marker(meta: Any) -> dict[str, Any] | None:
    """The part of a chunk's _meta marking async delivery, e.g. {"delivery": "async", ...}.

    codex-acp with the display-only patch forwards Codex's async messages with
    `_meta.codex = {delivery: "async", questions}`; only those with questions are questions.
    """
    if isinstance(meta, dict):
        if meta.get("delivery") == "async":
            return meta
        for value in meta.values():
            found = _async_marker(value)
            if found is not None:
                return found
    return None


def _add_update(update, events, tool_calls, add, ts) -> None:
    """Merge one `session/update` notification into the event list."""
    kind = update.get("sessionUpdate")
    if kind in ("agent_message_chunk", "agent_thought_chunk"):
        text = text_of(update.get("content"))
        merged = AGENT_MESSAGE if kind == "agent_message_chunk" else AGENT_THOUGHT
        content = update.get("content")
        marker = _async_marker(update.get("_meta")) or _async_marker(
            content.get("_meta") if isinstance(content, dict) else None
        )
        is_async = marker is not None
        last = events[-1] if events else None
        if (
            last is not None
            and last.kind == merged
            and not is_async
            and not last.data.get("async_message")
        ):
            last.text += text
            last.data["chunks"] += 1
        elif is_async:  # an async question forwarded as agent text stays its own event
            add(
                merged,
                text,
                ts,
                chunks=1,
                async_question=bool(marker.get("questions")),
                async_message=True,
                questions=marker.get("questions"),
            )
        else:
            add(merged, text, ts, chunks=1)
    elif kind in TOOL_UPDATES:
        if not update.get("toolCallId"):
            raise RecordError(f"a {kind} update without a toolCallId, which ACP requires")
        call_id = update["toolCallId"]
        call = tool_calls.get(call_id)
        if call is None:
            call = ToolCall(call_id, position=len(events))
            tool_calls[call_id] = call
        for key, attr in (
            ("title", "title"),
            ("kind", "kind"),
            ("status", "status"),
            ("rawInput", "raw_input"),
            ("rawOutput", "raw_output"),
            ("locations", "locations"),
        ):
            if update.get(key) is not None:
                setattr(call, attr, update[key])
        if update.get("content") is not None:
            call.content = update["content"]
        call.last_position = len(events)
        if (
            update.get("status") in (TOOL_COMPLETED, TOOL_FAILED)
            and call.completed_position is None
        ):
            call.completed_position = len(events)
        add(
            TOOL_UPDATES[kind],
            "",
            ts,
            tool_call_id=call_id,
            update={k: v for k, v in update.items() if k != "sessionUpdate"},
        )
    elif kind == "plan":
        add(PLAN, "", ts, entries=update.get("entries", []))
    elif kind == "current_mode_update":
        add(MODE_CHANGE, "", ts, mode=update.get("currentModeId"))
    elif kind == "user_message_chunk":
        add(OTHER, text_of(update.get("content")), ts, update="user_message_chunk")
    else:
        add(OTHER, "", ts, update=kind, data=update)
