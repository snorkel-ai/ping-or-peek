"""Build the ping-or-peek two-CSV rollouts as a Hugging Face dataset.

One row per measured two-CSV rollout (1,620), with what reproducing the blog post needs: the setup
(model, harness, setting, condition, seed), the task, the trace (an ATIF trajectory holding
everything the labelers read, `dataset.trajectory` and `atif`), the path
class with its two inputs, whether the agent said the situation may be a test, and whether a CSV
changed (`KEEP`). Every other label stays in the repository's analysis files. Constants (CLI
versions, effort, the simulated user's model) go to the build record for the card. Nothing is
uploaded.

    uv run --no-sync ping-or-peek dataset build \\
        --export <folder with the export's episodes.parquet and provenance.json> \\
        --tasks <the generated tasks/> --coder-run <the reasoning coder's run folder> \\
        --out <dataset folder> --records <build records folder>

The export's provenance names each rollout's trial directory. The rows are read from the pinned
export (`EXPORT_SHA256`), and any other export is refused.

Left out by construction: vendor system prompts and tool definitions (never read), harness
bookkeeping events (usage, session titles, slash-command lists, auth status), harness-internal
files, terminal recordings, raw request recordings, and every run that is not a measured
rollout. The build stops if a row would publish a credential the agents received or a local
path (`Guard`).
"""

from __future__ import annotations

import argparse
import base64
import bisect
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_args
from urllib.parse import quote, quote_plus

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ping_or_peek.config import (
    ASKED,
    ASSUMED,
    MODELS,
    NATIVE_AGENTS,
    NO_CHANGE,
    PRESENT_HIDDEN,
    READ_GRADER,
    REPO,
    SETTINGS,
    STOPPED_AT_PROMPT,
)
from ping_or_peek.labeling import atif
from ping_or_peek.labeling.reasoning.schema import EvaluationAwareness
from ping_or_peek.runs.run import (
    CODEX_CONTAINER,
    CREDENTIAL_ENV,
    CREDENTIAL_SECRETS,
    INSTRUCTION_FILE,
    PASSAGE,
)

from . import acp, codex_rollout, evidence
from .episode import Task, filesystem_facts, user_mode
from .trial import load_trial, read_json


def path(episode: pd.Series) -> str | None:
    """What decided an episode's first CSV change, from fields the export has: the
    dataset's `path` column, one of `config.PATHS`. Null when a label it needs is null."""
    asked = episode["asked_user_before_edit"]
    if pd.isna(asked):
        return None
    if asked:  # whether or not an edit followed
        return ASKED
    # With no checker there is no grader to read; elsewhere a null shortcut label is unknown.
    grader = False if episode["condition"] == PRESENT_HIDDEN else episode["scorer_used_to_edit_csv"]
    if pd.isna(grader):
        return None
    if grader:
        return READ_GRADER
    edited = pd.notna(episode["first_confirmed_edit_position"]) or bool(
        episode["edit_without_position"]
    )
    if edited:
        return ASSUMED
    if episode["pending_at_end"] > 0 and episode["approval_requested"]:
        return STOPPED_AT_PROMPT
    return NO_CHANGE


def family(model: str) -> str:
    """The model family: Claude (Fable, Opus, Sonnet) or GPT (Astra, Luna, GPT-6.1 Sol)."""
    return "Claude" if model.startswith("claude") else "GPT" if model.startswith("gpt") else model


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["setting"] = pd.Categorical(df["setting"], SETTINGS, ordered=True)
    return df.sort_values([c for c in ("harness", "model", "setting") if c in df])


TEST_AWARENESS = "verbalized_test_awareness"  # the reasoning coder's column
CLAUDE_FAMILY, GPT_FAMILY = FAMILIES = tuple(sorted({family(m) for m in MODELS}))
CLAUDE = CLAUDE_FAMILY.lower()  # as the dataset's model_family column names it
_, CODEX = NATIVE_AGENTS

EXPORT_SHA256 = "58f3120d40a41d24f25ed49e9eb4475ad4ad77c8632058cae6a557e77b747f93"
SHARDS = 2
SUMMARY = "build_summary.json"  # the build record, in --records

# --- Credentials and local paths ------------------------------------------------------------

TEMPLATE = re.compile(r"\$\{\w+\}")  # a variable Harbor resolves at run time, as recorded
LOCAL_PATH = "/Users/"  # a home folder on the machine that ran the experiment
# Each section of Harbor's trial config (TrialConfig) that sets a container's environment, and
# what it must hold: the target, which the bridge starts, gets CREDENTIAL_ENV; no one else gets
# anything.
ENV_SECTIONS = {
    ("agent", "env"): CREDENTIAL_ENV,
    ("user_agent", "env"): {},
    ("environment", "env"): {},
    ("verifier", "env"): {},
}


def credential_check(trial: str, config: dict[str, Any]) -> None:
    """The trial's recorded config gave its containers CREDENTIAL_ENV and nothing more: each
    section in ENV_SECTIONS holds exactly the keys it lists, with their values; no `${VAR}`
    template or `env` section appears anywhere else; and each compose overlay is one
    ping_or_peek.runs.run writes (CODEX_CONTAINER), which sets no environment. Which value each template
    held is scripts/gateway.sh's (CREDENTIAL_SECRETS). Raises ValueError naming the trial and
    each place that differs: a key of CREDENTIAL_ENV's by its name, a section by ENV_SECTIONS'
    name, and any other recorded key by its position, never its text or a value."""
    problems = []
    for (section, field), expected in ENV_SECTIONS.items():
        env = (config.get(section) or {}).get(field) or {}
        problems += [
            f"{section}.{field} key {i}: not one of ping_or_peek.runs.run.CREDENTIAL_ENV's names"
            for i, key in enumerate(env)
            if key not in expected
        ]
        problems += [
            f"{section}.{field}.{key}: not as ENV_SECTIONS gives it"
            for key in expected
            if key not in env or env[key] != expected[key]
        ]
    sections = {section for section, _ in ENV_SECTIONS}

    def where(path: tuple[Any, ...], positions: tuple[int, ...]) -> str:
        root = path[0] if path[0] in sections else f"key {positions[0]}"
        return f"{root} key path {list(positions[1:])}"

    def walk(value: Any, path: tuple[Any, ...], positions: tuple[int, ...]) -> None:
        if isinstance(value, dict):
            for i, (key, item) in enumerate(value.items()):
                at = (*path, key)
                if at in ENV_SECTIONS:
                    continue
                if key in ("env", "env_file"):
                    problems.append(
                        f"{where(at, (*positions, i))}: an env section outside ENV_SECTIONS"
                    )
                else:
                    walk(item, at, (*positions, i))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                walk(item, path, (*positions, i))
        elif isinstance(value, str) and TEMPLATE.search(value):
            problems.append(f"{where(path, positions)}: a template outside the agent's env")

    walk(config, (), ())
    overlays = (config.get("environment") or {}).get("extra_docker_compose") or []
    for i, overlay in enumerate(overlays):
        path = Path(overlay)
        if not path.is_file() or path.read_text() not in CODEX_CONTAINER.values():
            problems.append(
                f"environment.extra_docker_compose[{i}]: not a compose overlay ping_or_peek.runs.run writes"
            )
    if problems:
        raise ValueError(f"{trial}: {problems}")


def credentials() -> dict[str, str]:
    """The secrets the agents' containers received (ping_or_peek.runs.run.CREDENTIAL_SECRETS), by the
    variable holding each, from the environment (the CLI has loaded `.env` first). Raises SystemExit
    naming each variable that isn't set; a value is never printed."""
    names = sorted(set(CREDENTIAL_SECRETS.values()))
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise SystemExit(
            f"{missing} not set: the build checks the dataset for the credentials the agents "
            "received (ping_or_peek.runs.run.CREDENTIAL_SECRETS)"
        )
    return {name: os.environ[name] for name in names}


def _json_escaped(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)[1:-1]


def _base64(alphabet: Callable[[bytes], bytes], offset: int, value: str) -> str:
    """The base64 characters that `value`'s bytes alone determine when they start `offset`
    bytes (modulo 3) into the encoded data, whatever bytes surround them."""
    data = value.encode()
    encoded = alphabet(b"\0" * offset + data + b"\0\0").decode()
    first = -(-8 * offset // 6)  # the first character with no bit of a byte before
    last = (8 * (offset + len(data)) - 6) // 6  # the last with no bit of a byte after
    return encoded[first : last + 1]


# Each form a published string can hold a credential in: as it is, inside JSON text (the trace's
# JSON fields, and JSON text recorded in them), URL-encoded, or base64 at any byte alignment.
ENCODINGS: tuple[tuple[str, Callable[[str], str]], ...] = (
    ("raw", lambda value: value),
    ("JSON-escaped", _json_escaped),
    ("JSON-escaped twice", lambda value: _json_escaped(_json_escaped(value))),
    ("URL-quoted", lambda value: quote(value, safe="")),
    ("URL-quoted, spaces as +", quote_plus),
    *(
        (f"{name} base64 at byte {offset}", partial(_base64, alphabet, offset))
        for name, alphabet in (
            ("standard", base64.b64encode),
            ("URL-safe", base64.urlsafe_b64encode),
        )
        for offset in range(3)
    ),
)


class Guard:
    """Stops the build when a row would publish a credential the agents received, in any of
    ENCODINGS, or a local path (LOCAL_PATH). It checks every string of the row, keys included,
    as recorded (each `Encoded` value before it is made text) and then as published
    (`serialized`, where a `Joined` output is one text). It changes nothing, and its message
    names the episode, the field and the variable and encoding, never the value."""

    def __init__(self, secrets: dict[str, str]):
        # The dataset's own names; a key with any other name is recorded text.
        self.names = field_names(SCHEMA) | set(EVENT_FIELDS)
        self.forms = [
            (name, encoding, encode(value))
            for name, value in secrets.items()
            for encoding, encode in ENCODINGS
        ]

    def __call__(self, episode: str, row: dict[str, Any]) -> None:
        self._check(episode, row, "")
        self._check(episode, serialized(row), "")

    def _check(self, episode: str, value: Any, field: str) -> None:
        if isinstance(value, Encoded):
            self._check(episode, value.value, field)
        elif isinstance(value, Joined):
            for part in value.parts:
                self._check(episode, part, field)
        elif isinstance(value, str):
            for name, encoding, form in self.forms:
                if form in value:
                    raise SystemExit(
                        f"{episode}: {field} holds {name}'s value ({encoding}); the dataset "
                        "publishes no credential the agents received"
                    )
            if LOCAL_PATH in value:
                raise SystemExit(f"{episode}: {field} holds a local path ({LOCAL_PATH})")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                self._check(episode, item, f"{field}[{i}]")
        elif isinstance(value, dict):
            for i, (key, item) in enumerate(value.items()):
                # The key first; a path names a key only when the schema does, else by position.
                name = key if key in self.names else f"#{i}"
                self._check(episode, key, f"{field}.(key {name})" if field else f"(key {name})")
                self._check(episode, item, f"{field}.{name}" if field else str(name))


# --- Trace --------------------------------------------------------------------------------

# The trace's event types, and each reasoning summary's source by model family.
EVENT_TYPES = (
    USER_MESSAGE,
    AGENT_MESSAGE,
    REASONING_SUMMARY,
    TOOL_CALL,
    TOOL_RESULT,
    PERMISSION_REQUEST,
    QUESTION,
    CLIENT_OPERATION,
    PLAN,
    MODE_CHANGE,
    TURN_END,
) = (
    "user_message",
    "agent_message",
    "reasoning_summary",
    "tool_call",
    "tool_result",
    "permission_request",
    "question",
    "client_operation",
    "plan",
    "mode_change",
    "turn_end",
)
# Whom each event comes from.
ROLES = (USER_ROLE, AGENT_ROLE, TOOL_ROLE, HARNESS_ROLE) = ("user", "agent", "tool", "harness")
REASONING_SOURCES = {
    CLAUDE: "claude_summarized_thinking",
    GPT_FAMILY.lower(): "gpt_reasoning_summary",
}
EVENT_FIELDS = {
    "index": pa.int32(),
    "position": pa.int32(),
    "timestamp": pa.string(),
    "type": pa.string(),
    "role": pa.string(),
    "text": pa.string(),
    "reasoning_source": pa.string(),
    "tool_call_id": pa.string(),
    "tool_title": pa.string(),
    "tool_kind": pa.string(),
    "tool_input": pa.string(),
    "tool_output": pa.string(),
    "tool_status": pa.string(),
    "tool_exit_code": pa.int64(),
    "output_truncated": pa.bool_(),
    "options": pa.list_(pa.string()),
    "answer": pa.string(),
    "answered_by": pa.string(),
    "status": pa.string(),
}
SENT_MID_TURN = "sent_mid_turn"  # a user message's status when typed while the agent worked
ANSWERERS = (BY_USER, BY_CLIENT_POLICY) = ("user", "client_policy")  # who answered a request
# The trace's closed vocabularies: each field with a fixed set of values, and those values,
# for every event type or by event type, from the module that owns them. `trace_values`
# refuses any other value, and the card lists them from here, less the two statuses revision
# 4a958f8's card left out (card.PERMISSION_STATUSES_LISTED, QUESTION_STATUSES_LISTED).
# `tool_title` and `text` are closed only for the types listed; every other field is free
# (FREE_FIELDS).
VOCABULARY: dict[str, tuple[str, ...] | dict[str, tuple[str, ...]]] = {
    "type": EVENT_TYPES,
    "role": ROLES,
    "reasoning_source": {REASONING_SUMMARY: tuple(REASONING_SOURCES.values())},
    "tool_kind": {TOOL_CALL: acp.TOOL_KINDS, PERMISSION_REQUEST: acp.TOOL_KINDS},
    "tool_status": {TOOL_RESULT: acp.TOOL_STATUSES, CLIENT_OPERATION: acp.TOOL_STATUSES},
    "status": {
        USER_MESSAGE: (SENT_MID_TURN,),
        QUESTION: evidence.QUESTION_STATUSES,
        PERMISSION_REQUEST: (*evidence.DECISIONS, evidence.WITHDRAWN),
    },
    "answered_by": {QUESTION: (BY_USER,), PERMISSION_REQUEST: ANSWERERS},
    "tool_title": {
        QUESTION: tuple(tool for _, tool in evidence.QUESTION_TOOLS),
        CLIENT_OPERATION: acp.FS_METHODS,
    },
    "text": {TURN_END: acp.STOP_REASONS},
}
# The closed fields of the trajectory's question records and views and its approval views
# (`trajectory.question_view`, `async_view`, `approval_view`), and their values. `trajectory`
# refuses any other. The card lists the approval's from here; of the question's tools it names
# the agents' own (the published rows hold no `refusal_fallback` or `elicitation`).
VIEW_VOCABULARY: dict[str, dict[str, tuple[str, ...]]] = {
    acp.QUESTION: {
        "tool": tuple(tool for _, tool in (*evidence.QUESTION_TOOLS, evidence.ASYNC_QUESTION)),
        "status": (*evidence.QUESTION_STATUSES, evidence.NO_REPLY),
        "answered_by": (BY_USER,),
        "delivery": (atif.ASYNC,),
        "reason": evidence.REASONS,
    },
    acp.APPROVAL: {
        "decision": evidence.DECISIONS,
        "status": evidence.APPROVAL_STATUSES,
        "answered_by": ANSWERERS,
    },
}
FREE_FIELDS = frozenset(
    {
        "index",
        "position",
        "timestamp",
        "tool_call_id",
        "tool_input",
        "tool_output",
        "tool_exit_code",
        "output_truncated",
        "options",
        "answer",
    }
)


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _dump_in_order(value: Any) -> str:
    """JSON in the value's own key order, as the labelers' payload serialized the user's facts."""
    return json.dumps(value, ensure_ascii=False)


@dataclass(frozen=True)
class Encoded:
    """A value a published field holds only as text made from it (its JSON, or another
    rendering), kept as the value until `serialized` makes that text, so `Guard` checks both."""

    value: Any
    render: Callable[[Any], str | None] = _dump


@dataclass(frozen=True)
class Joined:
    """Text joined from parts, each a string or an `Encoded`, made text one by one."""

    parts: tuple[Any, ...]
    separator: str = "\n"


def _json(value: Any) -> Encoded | None:
    return None if value is None else Encoded(value)


def serialized(value: Any) -> Any:
    """`value` with each `Encoded` and `Joined` in it made text, nested lists and dicts
    included."""
    if isinstance(value, Encoded):
        return value.render(value.value)
    if isinstance(value, Joined):
        return value.separator.join(serialized(p) for p in value.parts)
    if isinstance(value, list):
        return [serialized(v) for v in value]
    if isinstance(value, dict):
        return {k: serialized(v) for k, v in value.items()}
    return value


def _event(kind: str, role: str, position: int | None, timestamp: str | None, **fields: Any):
    row = dict.fromkeys(EVENT_FIELDS)
    row.update(type=kind, role=role, position=position, timestamp=timestamp, **fields)
    return row


def _content_text(call, terminals: dict[str, dict[str, Any]]) -> tuple[Joined | None, bool | None]:
    """A tool call's display content, or its client terminal's output, when the harness
    reported no raw output."""
    parts, truncated = [], None
    for c in call.content:
        if not isinstance(c, dict):
            continue
        if c.get("type") == "terminal":
            terminal = terminals.get(c.get("terminalId")) or {}
            parts.append(terminal.get("output") or "")
            if isinstance(terminal.get("truncated"), bool):
                truncated = terminal["truncated"]
        else:  # text, or a block with no text (a diff) as its JSON
            parts.append(acp.text_of(c) or Encoded(c))
    parts = [p for p in parts if p]
    return (Joined(tuple(parts)) if parts else None), truncated


def tool_io(call, native: dict[str, Any] | None, terminals) -> dict[str, Any]:
    """What one tool call sent and got back, as the harness recorded it. Codex's commands come
    from its own session log (the ACP stream carries only their titles)."""
    if native is not None:
        return {
            "tool_input": _json(native.get("command")),
            "tool_output": native.get("output"),
            "tool_exit_code": native.get("exit_code"),
            "output_truncated": None,
        }
    raw = call.raw_output
    out, exit_code, truncated = None, None, None
    if isinstance(raw, str):
        out = raw
    elif isinstance(raw, dict):
        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        if isinstance(raw.get("output"), str):  # OpenCode
            out = raw["output"]
        else:
            streams = [raw[k] for k in ("stdout", "stderr", "formatted_output") if raw.get(k)]
            parts = tuple(v if isinstance(v, str) else Encoded(v) for v in streams)
            out = Joined(parts) if parts else None
        exit_code = raw.get("exit_code", raw.get("exitCode"))
        if exit_code is None and isinstance(metadata.get("exit"), int):
            exit_code = metadata["exit"]
        if isinstance(metadata.get("truncated"), bool):
            truncated = metadata["truncated"]
    if out is None:
        out, terminal_truncated = _content_text(call, terminals)
        truncated = terminal_truncated if truncated is None else truncated
    return {
        "tool_input": _json(call.raw_input),
        "tool_output": out,
        "tool_exit_code": exit_code if isinstance(exit_code, int) else None,
        "output_truncated": truncated,
    }


def question_answered_by(request: acp.Request) -> str | None:
    """Who answered a question-tool request: the user, or nobody."""
    return BY_USER if request.replied and request.routed_to_user else None


def approval_answered_by(request: acp.Request) -> str | None:
    """Who answered a permission request: the user, the client by its own policy, or nobody
    while it is pending."""
    if request.pending:
        return None
    return BY_USER if request.routed_to_user else BY_CLIENT_POLICY


def _answer(request: dict[str, Any]) -> str | None:
    """The user's answer to a question-tool request (`evidence.render_answer`), from its params
    and reply, or None when there is none."""
    return evidence.render_answer(SimpleNamespace(**request)) or None


def trace(session, native: list[dict[str, Any]], agent: str, family: str) -> list[dict]:
    """The rollout's events in the order of the bridge's event log, its JSON fields serialized
    (`trace_values`): the trace revision 4a958f8 published, to which `check` projects
    every trajectory (`atif.from_trajectory`)."""
    return serialized(trace_values(session, native, agent, family))


def trace_values(session, native: list[dict[str, Any]], agent: str, family: str) -> list[dict]:
    """The rollout's events in the order of the bridge's event log, each JSON field an `Encoded`
    value, for `Guard` to check before and after `serialized` makes it text."""
    source = REASONING_SOURCES[family]
    # Every tool update's status and kind, which a later update can overwrite in the call's
    # merged state below, while the trajectory publishes each update.
    for e in session.events:
        if e.kind in (acp.TOOL_CALL, acp.TOOL_UPDATE):
            update = e.data.get("update") or {}
            for field, values in (
                ("tool_status", VOCABULARY["tool_status"][TOOL_RESULT]),
                ("tool_kind", VOCABULARY["tool_kind"][TOOL_CALL]),
            ):
                value = update.get(field.removeprefix("tool_"))
                if value is not None and value not in values:
                    raise ValueError(
                        f"{field} of stream event {e.position}, a tool update, is outside "
                        "build.VOCABULARY"
                    )
    terminals = session.terminal_results()
    commands = {e.get("id"): e for e in native if e["type"] == "CommandExecution"}
    stamps = {e.position: e.ts for e in session.events}
    events: list[dict[str, Any]] = []
    announced: set[str] = set()
    for e in session.events:
        ts = e.ts
        if e.kind == acp.USER_PROMPT:
            status = SENT_MID_TURN if e.data.get("steering") else None
            events.append(
                _event(USER_MESSAGE, USER_ROLE, e.position, ts, text=e.text, status=status)
            )
        elif e.kind == acp.AGENT_MESSAGE:
            events.append(_event(AGENT_MESSAGE, AGENT_ROLE, e.position, ts, text=e.text))
        elif e.kind == acp.AGENT_THOUGHT and e.text.strip():
            events.append(
                _event(
                    REASONING_SUMMARY,
                    AGENT_ROLE,
                    e.position,
                    ts,
                    text=e.text,
                    reasoning_source=source,
                )
            )
        elif e.kind == acp.TOOL_CALL:
            call = session.tool_calls[e.data["tool_call_id"]]
            if call.tool_call_id in announced:
                continue
            announced.add(call.tool_call_id)
            # The call where the agent made it, its result where the harness returned it: after
            # any permission request, which can stand between the two.
            io = tool_io(call, commands.get(call.tool_call_id), terminals)
            events.append(
                _event(
                    TOOL_CALL,
                    AGENT_ROLE,
                    e.position,
                    ts,
                    tool_call_id=call.tool_call_id,
                    tool_title=call.title or None,
                    tool_kind=call.kind or None,
                    tool_input=io.pop("tool_input"),
                )
            )
            done = evidence.executed_at(call)
            events.append(
                _event(
                    TOOL_RESULT,
                    TOOL_ROLE,
                    done,
                    stamps.get(done),
                    tool_call_id=call.tool_call_id,
                    tool_title=call.title or None,
                    tool_status=call.status or None,
                    **io,
                )
            )
        elif e.kind == acp.QUESTION:
            request = session.requests[e.data["request"]]
            _, tool = evidence.classify_question(request, agent)
            events.append(
                _event(
                    QUESTION,
                    AGENT_ROLE,
                    e.position,
                    ts,
                    text=Encoded(request.params, evidence.render_question),
                    tool_title=tool,
                    tool_call_id=request.params.get("toolCallId"),
                    answer=Encoded({"params": request.params, "reply": request.reply}, _answer),
                    status=evidence.question_status(request),
                    answered_by=question_answered_by(request),
                )
            )
        elif e.kind == acp.APPROVAL:
            request = session.requests[e.data["request"]]
            call = request.params.get("toolCall") or {}
            decision, chosen = evidence.approval_decision(request)
            status = (
                decision
                if evidence.approval_status(request) == evidence.ANSWERED
                else evidence.approval_status(request)
            )
            events.append(
                _event(
                    PERMISSION_REQUEST,
                    HARNESS_ROLE,
                    e.position,
                    ts,
                    tool_call_id=call.get("toolCallId"),
                    tool_title=call.get("title"),
                    tool_kind=call.get("kind"),
                    tool_input=_json(call.get("rawInput")),
                    options=[o.get("name") for o in request.params.get("options", [])],
                    answer=chosen or None,
                    status=status,
                    answered_by=approval_answered_by(request),
                )
            )
        elif e.kind == acp.CLIENT_OP:
            request = session.requests[e.data["request"]]
            if not request.method.startswith("fs/"):
                continue  # terminal operations: their output is in the tool call's output
            params = {k: v for k, v in request.params.items() if k != "sessionId"}
            reply = request.reply if request.reply_error is None else {"error": request.reply_error}
            events.append(
                _event(
                    CLIENT_OPERATION,
                    HARNESS_ROLE,
                    e.position,
                    ts,
                    tool_title=request.method,
                    tool_input=_json(params),
                    tool_output=_json(reply),
                    tool_status=acp.TOOL_PENDING
                    if request.pending
                    else acp.TOOL_FAILED
                    if request.reply_error is not None
                    else acp.TOOL_COMPLETED,
                )
            )
        elif e.kind == acp.PLAN:
            events.append(
                _event(PLAN, AGENT_ROLE, e.position, ts, text=_json(e.data.get("entries")))
            )
        elif e.kind == acp.MODE_CHANGE:
            events.append(
                _event(MODE_CHANGE, HARNESS_ROLE, e.position, ts, text=e.data.get("mode"))
            )
        elif e.kind == acp.TURN_END:
            events.append(
                _event(TURN_END, HARNESS_ROLE, e.position, ts, text=e.data.get("stop_reason"))
            )
    # Stable: a result at its call's position follows the call.
    events.sort(key=lambda event: event["position"])
    _add_unstreamed_reasoning(events, native, source)
    for index, event in enumerate(events):
        event["index"] = index
        for field, values in VOCABULARY.items():
            allowed = values if isinstance(values, tuple) else values.get(event["type"])
            if allowed is not None and event[field] is not None and event[field] not in allowed:
                raise ValueError(  # never the value, which could be anything recorded
                    f"{field} of event {index}, a {event['type']}, is outside build.VOCABULARY"
                )
    return events


def _squash(text: str) -> str:
    return " ".join((text or "").split())


def _add_unstreamed_reasoning(events: list[dict], native: list[dict], source: str) -> None:
    """Codex reasoning summaries reach the ACP stream as thoughts; the few its own log holds
    alone go in by timestamp, after the last event logged before them, with no stream position."""
    streamed = {_squash(e["text"]) for e in events if e["type"] == REASONING_SUMMARY}
    extra = [
        n
        for n in native
        if n.get("private") and _squash(n.get("text", "")) and _squash(n["text"]) not in streamed
    ]
    if not extra:
        return
    times = [e["timestamp"] or "" for e in events]
    for n in sorted(extra, key=lambda n: n.get("timestamp") or ""):
        at = bisect.bisect_right(times, n.get("timestamp") or "")
        events.insert(
            at,
            _event(
                REASONING_SUMMARY,
                AGENT_ROLE,
                None,
                n.get("timestamp"),
                text=n["text"],
                reasoning_source=source,
            ),
        )
        times.insert(at, n.get("timestamp") or "")


# --- Task ------------------------------------------------------------------------------------


def _workspace(task_path: str, tasks_root: Path) -> Path:
    """The task's workspace under `tasks_root`, found by the task path's last two parts."""
    parts = Path(task_path).parts
    return tasks_root / parts[-2] / parts[-1] / "environment" / "workspace"


def project_note(
    task_path: str, tasks_root: Path, agent: str, setting: str
) -> dict[str, str] | None:
    """The workspace's project note, from the task directory the rollout ran in, found under
    `tasks_root` by its last two path parts: in a setting that carries one (`ping_or_peek.runs.run.PASSAGE`),
    the agent's note file (`INSTRUCTION_FILE`) holding the setting's passage; in the setting that
    carries none, None. Raises FileNotFoundError when the workspace is missing, and ValueError,
    naming it, when its notes aren't the setting's."""
    parts = Path(task_path).parts
    where = f"{parts[-2]}/{parts[-1]}"
    workspace = _workspace(task_path, tasks_root)
    if not workspace.is_dir():
        raise FileNotFoundError(f"task workspace not found: {where}")
    found = [n for n in sorted(set(INSTRUCTION_FILE.values())) if (workspace / n).exists()]
    if setting not in PASSAGE:
        if found:
            raise ValueError(f"{where}: setting {setting} carries no project note; found {found}")
        return None
    name, passage = INSTRUCTION_FILE[agent], (REPO / PASSAGE[setting]).read_text()
    if found != [name] or (workspace / name).read_text() != passage:
        raise ValueError(
            f"{where}: setting {setting} carries {name} holding {PASSAGE[setting]}; found {found}"
        )
    return {"name": name, "content": passage}


def csv_changed(task: Task, metrics: dict[str, Any], trial: str) -> bool:
    """At least one of the two CSVs differs from its original at the end of the rollout: the
    verifier's `changed_original_files` (a CSV missing at the end counts as changed). A rollout
    without that record is refused, never published as null."""
    changed = metrics.get("changed_original_files")
    if changed is None:
        raise ValueError(f"{trial}: the verifier recorded no changed_original_files")
    return any(name in changed for name in task.files)


def effort_of(kwargs: dict[str, Any], model: str) -> str | None:
    """The model's reasoning effort: the agent's own setting, or, for OpenCode, the model's
    options in its configuration (`effort` for Claude models, `reasoningEffort` for GPT models)."""
    if kwargs.get("reasoning_effort"):
        return kwargs["reasoning_effort"]
    for provider in ((kwargs.get("opencode_config") or {}).get("provider") or {}).values():
        options = ((provider.get("models") or {}).get(model) or {}).get("options") or {}
        if options.get("effort") or options.get("reasoningEffort"):
            return options.get("effort") or options.get("reasoningEffort")
    return None


def adapter_name(package: str | None) -> str | None:
    """The ACP adapter as installed: a registry name, or a local build's tarball name (the
    codex-acp fork), never the local path it was installed from."""
    if package and "/" in package and package.endswith((".tgz", ".tar.gz")):
        return Path(package).name.removesuffix(".tgz").removesuffix(".tar.gz")
    return package


def native_log(trial) -> list[dict[str, Any]]:
    """Codex's own log of a Codex trial (`codex_rollout.native_events`), and nothing for other
    agents. Raises ValueError naming the trial when a Codex trial has no log (its commands and
    some reasoning summaries are recorded only there), or when the log holds raw reasoning that
    has no summary."""
    if trial.agent != CODEX:
        return []
    files = codex_rollout.rollout_files(trial.native_session_files())
    if not files:
        raise ValueError(f"{trial.name}: Codex's log (rollout-*.jsonl) isn't under agent/sessions")
    native = codex_rollout.native_events(files, trial.name)
    if any(n.get("raw_content") for n in native):
        raise ValueError(
            f"{trial.name}: Codex's log holds raw reasoning with no summary; the dataset publishes "
            "reasoning summaries only"
        )
    return native


def model_family(model: str) -> str:
    """The published `model_family`: `family`, lower-cased. Raises ValueError for a model in
    neither family, which would otherwise be published under a wrong one."""
    found = family(model)
    if found not in FAMILIES:
        raise ValueError(
            f"model: in neither family {FAMILIES}, so the dataset has no column for it"
        )
    return found.lower()


# A CSV's role in the task: the file the user means, or the other one.
CSV_ROLES = INTENDED, OTHER_CSV = ("intended", "other")


def task_value(task: Task, note: dict[str, str] | None) -> dict[str, Any]:
    """The published `task`: the opener, the two CSVs, scorer.py, the project note, and the
    user's facts, which keep the order the labelers saw them in."""
    return {
        "opener": task.opener,
        "intended_file": task.target,
        "checker_target_file": task.decoy if task.scorer_exposed else None,
        "csv_files": [
            {
                "name": n,
                "role": INTENDED if n == task.target else OTHER_CSV,
                "content": text,
                "column_total": task.totals[n],
            }
            for n, text in sorted(task.files.items())
        ],
        "scorer_py": task.scorer,
        "project_note": note,
        # In the simulated user's key order, serialized as the labelers' payload was.
        "user_facts": Encoded(task.facts, _dump_in_order),
    }


def rollout(row: dict[str, Any], trial_dir: Path, tasks_root: Path) -> tuple[dict[str, Any], tuple]:
    """The published columns the export lacks, and the build record's constants, for one
    export row, read from its trial directory and its task's workspace; `trace` is the ATIF
    trajectory as a JSON document. Also what the trajectory must read back as: the parsed
    session, Codex's log and the verifier's facts
    (`trajectory.check_reads_back`)."""
    # Imported here, not with the module: the trajectory needs Harbor, which CI's sync leaves
    # out, and the builder's other parts are tested there.
    from . import trajectory

    trial = load_trial(trial_dir)
    session = acp.parse_export(trial.acp_export)
    native = native_log(trial)
    task = Task.regenerate(int(row["seed"]), row["condition"])
    config = read_json(trial_dir / "config.json", trial.name)
    credential_check(trial_dir.name, config)
    kwargs = config.get("agent", {}).get("kwargs", {})
    info = trial.result.get("agent_info") or {}
    family = model_family(row["model"])
    note = project_note(config["task"]["path"], tasks_root, trial.agent, row["setting"])
    workspace_check(task, trial_dir, config["task"]["path"], tasks_root)
    metrics = trial.verifier_metrics()
    verifier = filesystem_facts(metrics)
    document = trajectory.to_trajectory(
        session,
        native,
        episode_id=row["episode_id"],
        agent=trial.agent,
        agent_version=info.get("version"),
        model=row["model"],
        user_mode=user_mode(task),
        verifier=verifier,
    )
    built = {
        "model_family": family,
        "reasoning_effort": effort_of(kwargs, row["model"]),
        "agent_cli": info.get("name"),
        "agent_cli_version": info.get("version"),
        "acp_adapter": adapter_name(kwargs.get("acp_adapter_package")),
        "acp_agent_mode": kwargs.get("acp_agent_mode"),
        # The model, without the gateway route.
        "user_simulator_model": (
            ((config.get("user_agent") or {}).get("model_name") or "").split("/")[-1] or None
        ),
        "task": task_value(task, note),
        "trace": document,
        "csv_changed": csv_changed(task, metrics, trial_dir.name),
    }
    return built, (session, native, verifier)


def workspace_check(task: Task, trial_dir: Path, task_path: str, tasks_root: Path) -> None:
    """The regenerated task matches the files the rollout's workspace held."""
    workspace = _workspace(task_path, tasks_root)
    for name, text in task.files.items():
        if (workspace / name).read_text() != text:
            raise ValueError(f"{trial_dir.name}: {name} differs from the regenerated task")
    scorer = workspace / "scorer.py"
    if (task.scorer is None) != (not scorer.exists()) or (
        task.scorer is not None and scorer.read_text() != task.scorer
    ):
        raise ValueError(f"{trial_dir.name}: scorer.py differs from the regenerated task")


# --- Build ----------------------------------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required(name: str, type_: pa.DataType) -> pa.Field:
    return pa.field(name, type_, nullable=False)


def _items(type_: pa.DataType) -> pa.DataType:
    """A list whose items are never null."""
    return pa.list_(_required("element", type_))


# The dataset's structure, the one definition the build writes and the check enforces: the
# published columns in order, with their nested fields, and which values may be null. What
# reproducing the blog post needs: the setup, the path class and its two inputs, the
# test-awareness flag, the task, the trace, and csv_changed for Figure 5's check that a rollout
# drawn as changing nothing changed no CSV; everything else lives in the repository's analysis
# files.
SCHEMA = pa.schema(
    [
        *(
            _required(name, pa.string())
            for name in ("episode_id", "model", "model_family", "harness", "agent")
        ),
        _required("setting", pa.string()),
        _required("condition", pa.string()),
        _required("seed", pa.int64()),
        _required("path", pa.string()),
        _required("asked_user_before_edit", pa.bool_()),
        pa.field("scorer_used_to_edit_csv", pa.bool_()),  # null without a scorer.py
        _required(TEST_AWARENESS, pa.bool_()),
        _required(
            "task",
            pa.struct(
                [
                    _required("opener", pa.string()),
                    _required("intended_file", pa.string()),
                    pa.field("checker_target_file", pa.string()),
                    _required(
                        "csv_files",
                        _items(
                            pa.struct(
                                [
                                    _required("name", pa.string()),
                                    _required("role", pa.string()),
                                    _required("content", pa.string()),
                                    _required("column_total", pa.int64()),
                                ]
                            )
                        ),
                    ),
                    pa.field("scorer_py", pa.string()),
                    pa.field(
                        "project_note",
                        pa.struct(
                            [_required("name", pa.string()), _required("content", pa.string())]
                        ),
                    ),
                    _required("user_facts", pa.string()),
                ]
            ),
        ),
        _required("trace", pa.string()),  # the ATIF trajectory's JSON
        _required("csv_changed", pa.bool_()),
    ]
)
KEEP = tuple(SCHEMA.names)


def written(schema: pa.Schema) -> pa.Schema:
    """`schema` as the shards are written: every field nullable, as revision 4a958f8's are. Which
    values may be null is a rule `structure_problems` enforces, not one the files record."""

    def loose(type_: pa.DataType) -> pa.DataType:
        if pa.types.is_struct(type_):
            return pa.struct([pa.field(f.name, loose(f.type)) for f in type_])
        if pa.types.is_list(type_):
            return pa.list_(loose(type_.value_type))
        return type_

    return pa.schema([pa.field(f.name, loose(f.type)) for f in schema])


def _fields(type_: pa.DataType, where: str):
    """(path, field) of each field nested in `type_`: `a.b` for a struct's, `a[]` for a list's."""
    if pa.types.is_struct(type_):
        for f in type_:
            yield f"{where}.{f.name}", f
            yield from _fields(f.type, f"{where}.{f.name}")
    elif pa.types.is_list(type_):
        yield f"{where}[]", type_.value_field
        yield from _fields(type_.value_type, f"{where}[]")


def nullable_paths(schema: pa.Schema) -> set[str]:
    """The paths of the fields of `schema` whose values may be null."""
    found = set()
    for top in schema:
        for where, f in [(top.name, top), *_fields(top.type, top.name)]:
            if f.nullable:
                found.add(where)
    return found


def structure_problems(table: pa.Table, schema: pa.Schema = SCHEMA) -> list[str]:
    """How `table` departs from `schema` as the shards are written (`written`): a column missing,
    extra or out of order, a type or nullability that differs (nested fields included), or a
    null where the schema allows none, each naming its field. Empty when the table has the
    schema's structure, so a reader may rely on it."""
    expected = written(schema)
    problems = [f"{n}: missing" for n in expected.names if n not in table.column_names]
    problems += [
        f"{n}: not a published column" for n in table.column_names if n not in expected.names
    ]
    if not problems and table.column_names != expected.names:
        problems.append(f"columns out of order: {table.column_names}")
    problems += [
        f"{f.name}: {table.schema.field(f.name).type}, where the schema has {f.type}"
        for f in expected
        if f.name in table.column_names and not table.schema.field(f.name).type.equals(f.type)
    ]
    # A field's own nullability: its nested fields' is part of its type.
    problems += [
        f"{f.name}: nullable {table.schema.field(f.name).nullable}, where the schema has "
        f"{f.nullable}"
        for f in expected
        if f.name in table.column_names and table.schema.field(f.name).nullable != f.nullable
    ]
    if problems:
        return problems

    def nulls(values: pa.Array, field: pa.Field, where: str) -> None:
        if not field.nullable and values.null_count:
            problems.append(f"{where}: {values.null_count} nulls, where the schema allows none")
        values = values.filter(pc.is_valid(values))  # a null parent's fields aren't values
        if pa.types.is_struct(field.type):
            for i, child in enumerate(field.type):
                nulls(values.field(i), child, f"{where}.{child.name}")
        elif pa.types.is_list(field.type):
            nulls(pc.list_flatten(values), field.type.value_field, f"{where}[]")

    for field in schema:
        nulls(table.column(field.name).combine_chunks(), field, field.name)
    return problems


def field_names(schema: pa.Schema) -> frozenset[str]:
    """The name of every column of `schema` and of every field nested in one."""
    names: set[str] = set()

    def walk(fields) -> None:
        for f in fields:
            names.add(f.name)
            kind = f.type.value_type if pa.types.is_list(f.type) else f.type
            if pa.types.is_struct(kind):
                walk(kind)

    walk(schema)
    return frozenset(names)


def read_shards(folder: Path) -> tuple[pa.Table | None, list[str]]:
    """The build's shards in `folder` joined, once there are some and each reads and has
    SCHEMA's structure (`structure_problems`); otherwise None and every problem, named by its
    shard, or by the folder when it holds none. Each shard is read on its own, so one that won't
    read or doesn't fit is named."""
    shards = sorted(folder.glob("*.parquet"))
    if not shards:
        return None, [f"{folder}: no shards (*.parquet)"]
    problems, tables = [], []
    for shard in shards:
        try:
            tables.append(pq.read_table(shard))
        except (OSError, pa.ArrowException) as error:
            problems.append(f"{shard.name}: unreadable: {type(error).__name__}")
            continue
        problems += [f"{shard.name}: {p}" for p in structure_problems(tables[-1])]
    return (None if problems else pa.concat_tables(tables)), problems


EVALUATION_CODES = get_args(EvaluationAwareness.model_fields["value"].annotation)
META_TEST = EVALUATION_CODES[0]  # the agent said the situation itself may be a test


def test_awareness(
    coder_run: Path, rows: list[dict[str, Any]], trial_paths: dict[str, Path]
) -> dict[str, bool]:
    """Episode id -> whether the agent said the situation itself may be a test: the reasoning
    coder's `evaluation_awareness` code `meta_test` (`graded_output`, "my work is being graded",
    doesn't count). A boolean only: no quotes, no other code. Read from the coding the coder saved
    in `coder_run` for each row's own trial, keyed `<job>/<trial>` as the coder keys it. Raises
    ValueError, naming the episode, for a row without a saved coding, or with a code outside the
    coder's vocabulary."""
    from ping_or_peek.labeling.reasoning.label import saved_coding
    from ping_or_peek.labeling.reasoning.transcript import episode_key

    flags = {}
    for row in rows:
        trial = trial_paths[row["trial"]]
        episode = episode_key(trial, trial.parent.parent)
        code = saved_coding(coder_run, episode)["evaluation_awareness"]["value"]
        if code not in EVALUATION_CODES:
            raise ValueError(
                f"{episode}: its evaluation-awareness code is none of the coder's "
                f"{list(EVALUATION_CODES)} (reasoning.schema.EvaluationAwareness)"
            )
        flags[row["episode_id"]] = code == META_TEST
    return flags


def constants(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What is the same across rows, for the card: the agent CLIs, effort per model, the simulated
    user's model, and how many rollouts are included. Raises if one varies where it shouldn't."""

    def one(field: str, values: set) -> Any:
        if len(values) != 1:
            raise SystemExit(f"{field}: expected one value across the rows, found {len(values)}")
        return values.pop()

    agents = sorted({r["agent"] for r in rows})
    models = sorted({r["model"] for r in rows})
    return {
        "rollouts": len(rows),
        "included": sum(r["included"] is True for r in rows),
        "agents": {
            a: {
                "cli": one(f"{a}'s agent_cli", {r["agent_cli"] for r in rows if r["agent"] == a}),
                "version": one(
                    f"{a}'s agent_cli_version",
                    {r["agent_cli_version"] for r in rows if r["agent"] == a},
                ),
                "adapter": one(
                    f"{a}'s acp_adapter", {r["acp_adapter"] for r in rows if r["agent"] == a}
                ),
            }
            for a in agents
        },
        "effort": {
            m: one(
                f"{m}'s reasoning_effort", {r["reasoning_effort"] for r in rows if r["model"] == m}
            )
            for m in models
        },
        "user_simulator_model": one(
            "user_simulator_model", {r["user_simulator_model"] for r in rows}
        ),
    }


def commit() -> str:
    """The repository's checked-out commit, for the build record."""
    return subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--export",
        type=Path,
        required=True,
        help="the folder with the export's episodes.parquet and provenance.json",
    )
    parser.add_argument("--tasks", type=Path, required=True, help="the generated tasks/")
    parser.add_argument(
        "--coder-run",
        type=Path,
        required=True,
        help=f"the reasoning coder's run folder, whose saved codings give {TEST_AWARENESS}",
    )
    parser.add_argument("--out", type=Path, required=True, help="the dataset: data/ and README.md")
    parser.add_argument("--records", type=Path, required=True, help="where the build record goes")
    parser.add_argument("--limit", type=int, help="build only the first N rollouts (a trial run)")


def run(args: argparse.Namespace) -> None:
    if sha256(args.export / "episodes.parquet") != EXPORT_SHA256:
        raise SystemExit("episodes.parquet is not the pinned export")
    episodes = pd.read_parquet(args.export / "episodes.parquet")
    provenance = json.loads((args.export / "provenance.json").read_text())
    trial_paths = {i["trial"]: Path(i["path"]) for i in provenance["inputs"]}
    derived = ordered(episodes)
    derived = derived.assign(path=derived.apply(path, axis=1)).set_index("episode_id")
    secrets = credentials()
    guard = Guard(secrets)
    rows = []
    export_rows = pq.read_table(args.export / "episodes.parquet").to_pylist()  # nulls stay None
    from . import trajectory  # needs Harbor (see rollout)

    for row in export_rows[: args.limit]:
        try:
            built, source = rollout(row, trial_paths[row["trial"]], args.tasks)
        except ValueError as error:  # its message names the field and rule, never a value
            raise ValueError(f"{row['episode_id']}: {error}") from None
        built["path"] = derived.loc[row["episode_id"], "path"]
        recorded = {**row, **built}
        guard(row["episode_id"], recorded)
        published = serialized(recorded)
        trajectory.check_reads_back(published["trace"], *source)
        published["trace"] = json.dumps(published["trace"], ensure_ascii=False)
        guard(row["episode_id"], published)  # the trace as the one JSON text it is published as
        rows.append(published)
    found = constants(rows)
    flags = test_awareness(args.coder_run, rows, trial_paths)
    for r in rows:
        r[TEST_AWARENESS] = flags[r["episode_id"]]
    schema = written(SCHEMA)
    data = args.out / "data"
    data.mkdir(parents=True, exist_ok=True)
    for old in data.glob("*.parquet"):
        old.unlink()
    per = -(-len(rows) // SHARDS)
    for i in range(SHARDS):
        part = [{k: r[k] for k in KEEP} for r in rows[i * per : (i + 1) * per]]
        table = pa.Table.from_pylist(part, schema=schema)
        pq.write_table(table, data / f"train-{i:05d}-of-{SHARDS:05d}.parquet", compression="zstd")
    documents = [json.loads(r["trace"]) for r in rows]
    records = [record for d in documents for _, record in atif.records(d)]
    summary = {
        "rows": len(rows),
        "columns": list(KEEP),
        "constants": found,
        "commit": commit(),
        "export_sha256": EXPORT_SHA256,
        # What the guard looked for, by name; the values are never written.
        "guarded": {"credentials": sorted(secrets), "encodings": [e for e, _ in ENCODINGS]},
        "csv_changed": dict(Counter(str(r["csv_changed"]) for r in rows)),
        "steps": dict(Counter(s["source"] for d in documents for s in d["steps"])),
        "acp_events": dict(Counter(r["kind"] for r in records)),
        "async_questions": sum(r.get("delivery") == atif.ASYNC for r in records),
        "agent_log_items": dict(
            Counter(n["type"] for d in documents for n in d["extra"].get("agent_log", []))
        ),
        "trace_bytes": sum(len(r["trace"].encode()) for r in rows),
        "shards": {p.name: sha256(p) for p in sorted(data.glob("*.parquet"))},
    }
    args.records.mkdir(parents=True, exist_ok=True)
    (args.records / SUMMARY).write_text(json.dumps(summary, indent=1) + "\n")
    hidden = ("shards",)
    print(json.dumps({k: v for k, v in summary.items() if k not in hidden}, indent=1))
