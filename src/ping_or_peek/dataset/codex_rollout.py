"""Read Codex's own session log (its rollout file) into native evidence events.

Harbor copies the rollout to `agent/sessions/YYYY/MM/DD/rollout-*.jsonl`. Each line is a record
`{timestamp, ordinal, type, payload}`; completed turn items arrive as
`event_msg` records whose payload type is `item_completed`. Three kinds matter here:

- CommandExecution and FileChange items: what actually ran or changed, with exit codes;
- AgentMessage items with `delivery: "async"`: questions asked with request_user_input_async,
  which codex-acp 2.2.2 does not forward to ACP clients, so this log is their only record;
- Reasoning items: the model's reasoning summaries, which are private evidence.

Item shapes follow Codex 0.162.1 (`codex-rs/protocol/src/items.rs`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .trial import read_jsonl

EXECUTION_ITEMS = (
    "CommandExecution",
    "FileChange",
    "DynamicToolCall",
    "McpToolCall",
    "CollabAgentToolCall",
    "WebSearch",
)


def rollout_files(paths: list[Path]) -> list[Path]:
    return sorted(p for p in paths if p.name.startswith("rollout-") and p.suffix == ".jsonl")


def read_rollout(path: Path, trial: str) -> list[dict[str, Any]]:
    """The completed turn items of one of `trial`'s rollouts, in order, as compact native events.
    Raises RecordError, naming the trial and the file, for a line that isn't a JSON object."""
    events = []
    for record in read_jsonl(path, trial):
        payload = record.get("payload") or {}
        if record.get("type") != "event_msg" or payload.get("type") != "item_completed":
            continue
        item = payload.get("item") or {}
        kind = item.get("type")
        base = {
            "type": kind,
            "id": item.get("id"),
            "turn_id": payload.get("turn_id"),
            "ordinal": record.get("ordinal"),
            "timestamp": record.get("timestamp"),
        }
        if kind == "CommandExecution":
            events.append(
                {
                    **base,
                    "command": item.get("command"),
                    "cwd": item.get("cwd"),
                    "status": item.get("status"),
                    "exit_code": item.get("exit_code"),
                    "output": item.get("aggregated_output"),
                }
            )
        elif kind == "FileChange":
            events.append(
                {
                    **base,
                    "changes": item.get("changes"),
                    "status": item.get("status"),
                    "stdout": item.get("stdout"),
                    "stderr": item.get("stderr"),
                }
            )
        elif kind == "AgentMessage" and item.get("delivery") == "async":
            text = "\n".join(c.get("text", "") for c in item.get("content") or [])
            events.append(
                {**base, "delivery": "async", "questions": item.get("questions"), "text": text}
            )
        elif kind == "Reasoning":
            summary = "\n".join(item.get("summary_text") or [])
            text = summary or "\n".join(item.get("raw_content") or [])
            if text.strip():
                # Raw content, where there is no summary, is the reasoning itself: marked, so a
                # reader that publishes only summaries can refuse it.
                raw = {} if summary else {"raw_content": True}
                events.append({**base, "private": True, "text": text, **raw})
        elif kind in EXECUTION_ITEMS:
            events.append(
                {**base, "item": {k: v for k, v in item.items() if k not in ("type", "id")}}
            )
        elif kind in ("UserMessage", "AgentMessage"):
            text = "\n".join(c.get("text", "") for c in item.get("content") or [])
            events.append({**base, "text": text})
    return events


def native_events(paths: list[Path], trial: str) -> list[dict[str, Any]]:
    """Events from every one of `trial`'s rollouts, each turn item once (a resumed session can
    repeat items). Raises RecordError as `read_rollout` does."""
    events: list[dict[str, Any]] = []
    seen = set()
    for path in rollout_files(paths):
        for event in read_rollout(path, trial):
            key = (event["type"], event["id"], event["turn_id"])
            if key not in seen:
                seen.add(key)
                events.append(event)
    return events


def async_questions(native: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Async questions (request_user_input_async). An async message without questions
    (send_message_to_user_async) is a message to the user, not a question."""
    return [e for e in native if e.get("delivery") == "async" and e.get("questions")]


def worked_while_pending(native: list[dict[str, Any]]) -> bool | None:
    """Whether Codex ran commands or changed files while an async question was unanswered.

    Pending runs from the question to the next user message in the same turn (an answer
    steered in mid-turn), or to the end of the turn. Null when there was no async question.
    """
    asks = async_questions(native)
    if not asks:
        return None
    for ask in asks:
        same_turn = [
            e
            for e in native
            if e["turn_id"] == ask["turn_id"] and (e["ordinal"] or 0) > (ask["ordinal"] or 0)
        ]
        reply = next((e["ordinal"] for e in same_turn if e["type"] == "UserMessage"), None)
        if any(
            e["type"] in EXECUTION_ITEMS and (reply is None or e["ordinal"] < reply)
            for e in same_turn
        ):
            return True
    return False
