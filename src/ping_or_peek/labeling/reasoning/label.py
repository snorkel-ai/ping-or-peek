"""Code episodes with an LLM: strict JSON, verbatim quotes, every request and response saved.

The coder reads its rubric, `rubrics/reasoning.md`, as its instructions and one episode transcript (model and harness
hidden). Its answer must match `schema.Coding` and pass `schema.validate`; a broken rule is sent
back with the model's answer for up to two more tries. Every attempt is written to disk before the
next starts, so a run resumes where it stopped and nothing paid for is lost.

The default coder is GPT-6.1 Sol at medium effort through the Portkey gateway.
Records hold agent text, so they go under a private directory outside the repository.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ping_or_peek.config import REPO

from ..labels import RUBRICS
from .schema import RUBRIC_VERSION, Coding, CodingError, validate
from .transcript import Transcript, render

RUBRIC_PATH = RUBRICS / "reasoning.md"
ATTEMPTS = 3

OUTPUT_INSTRUCTIONS = """

## Your answer

Answer with one JSON object that follows the response schema. Every quote must be copied exactly
from the event it cites. Use empty lists where nothing applies. Code the agent's words, not your
guesses about its intentions.
"""


@dataclass(frozen=True)
class CoderConfig:
    model: str = "gpt-6.1-sol"
    reasoning_effort: str = "medium"
    max_output_tokens: int = 12000
    base_url: str = "https://api.portkey.ai/v1"
    provider: str = "@openai"

    def public(self) -> dict[str, Any]:
        return asdict(self)


def rubric_text() -> str:
    return RUBRIC_PATH.read_text()


def rubric_sha256() -> str:
    return hashlib.sha256(rubric_text().encode()).hexdigest()


def system_prompt() -> str:
    return rubric_text() + OUTPUT_INSTRUCTIONS


def response_format() -> dict[str, Any]:
    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: clean(v) for k, v in node.items() if k not in ("title", "default")}
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "Coding",
            "schema": clean(Coding.model_json_schema()),
            "strict": True,
        },
    }


def make_client(config: CoderConfig) -> Any:
    """The coder's client, keyed with PORTKEY_API_KEY from the environment, into which the
    entry point has loaded the repo's .env.

    Raises RuntimeError when it isn't set.
    """
    from openai import OpenAI

    key = os.environ.get("PORTKEY_API_KEY")
    if not key:
        raise RuntimeError("Set PORTKEY_API_KEY, in the environment or the repo's .env")
    return OpenAI(api_key=key, base_url=config.base_url, max_retries=2, timeout=900)


def provenance() -> dict[str, Any]:
    """Which code produced a record: this repo's HEAD and whether the dataset and labeling packages
    or the config had local edits."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "repo_head": git("rev-parse", "HEAD"),
        # Tracked edits only: untracked outputs written into the checkout are not code and would
        # mark every record dirty.
        "code_dirty": bool(
            git(
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                "src/ping_or_peek/dataset",
                "src/ping_or_peek/labeling",
                "src/ping_or_peek/config.py",
            )
        ),
        "rubric_version": RUBRIC_VERSION,
        "rubric_sha256": rubric_sha256(),
    }


def safe_name(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", key)


def saved_coding(run_root: Path, episode: str) -> dict[str, Any]:
    """The coding `code_episode` saved for `episode` in the run folder `run_root`. Raises
    ValueError, naming the episode, when it has no saved result or its coding failed."""
    path = run_root / safe_name(episode) / "result.json"
    if not path.exists():
        raise ValueError(f"{episode}: no saved coding in {run_root}")
    record = json.loads(path.read_text())
    if record.get("status") != "ok":
        raise ValueError(f"{episode}: its saved coding in {run_root} failed")
    return record["coding"]


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


MAX_TRANSCRIPT_CHARS = 120_000
SHORT_OUTPUT_LIMIT = 800


def clipped(transcript: Transcript, rebuild: Any = None) -> tuple[Transcript, str]:
    """The transcript the coder sends, and its text. A long episode (for example one that ran to
    the time limit) is rebuilt by `rebuild(output_limit)` with tool output clipped harder."""
    text = render(transcript)
    if len(text) > MAX_TRANSCRIPT_CHARS and rebuild is not None:
        transcript = rebuild(SHORT_OUTPUT_LIMIT)
        text = render(transcript)
    return transcript, text


def code_episode(
    transcript: Transcript,
    *,
    out_root: Path,
    config: CoderConfig,
    client: Any,
    rebuild: Any = None,
) -> dict[str, Any]:
    """Code one episode, resuming from saved attempts. Returns the result record.

    A saved coding is reused only when the request this would send (the system prompt and the
    clipped transcript) is the one its saved attempts recorded; otherwise this raises ValueError
    naming the episode, because a changed transcript needs a new run, not the old codes."""
    folder = out_root / safe_name(transcript.episode)
    result_path = folder / "result.json"
    transcript, text = clipped(transcript, rebuild)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": text},
    ]
    if result_path.exists():
        recorded = [
            json.loads(p.read_text())["request_messages"][:2]
            for p in sorted(folder.glob("attempt-*.json"))
        ]
        if not recorded:
            raise ValueError(
                f"{transcript.episode}: the saved coding has no recorded request to compare with"
            )
        if any(sent != messages for sent in recorded):
            raise ValueError(
                f"{transcript.episode}: the transcript or system prompt differs from the one the "
                "saved coding was made from; code it in a new run"
            )
        return json.loads(result_path.read_text())
    record_base = {
        "episode": transcript.episode,
        "meta": {
            "source": transcript.source,
            "model": transcript.model,
            "harness": transcript.harness,
            "setting": transcript.setting,
            "condition": transcript.condition,
            "seed": transcript.seed,
            "user_available": transcript.user_available,
            "checker_exposed": transcript.checker_exposed,
            "reasoning_kind": transcript.reasoning_kind,
            **transcript.extra,
        },
        "coder": config.public(),
        "transcript_sha256": hashlib.sha256(text.encode()).hexdigest(),
        **provenance(),
    }
    error = "no attempt made"
    for _ in range(ATTEMPTS):
        number = len(list(folder.glob("attempt-*.json"))) + 1
        attempt: dict[str, Any] = {
            **record_base,
            "request_messages": list(messages),
            "started_at": time.time(),
        }
        try:
            completion = client.chat.completions.create(
                model=f"{config.provider}/{config.model}",
                messages=messages,
                reasoning_effort=config.reasoning_effort,
                max_completion_tokens=config.max_output_tokens,
                response_format=response_format(),
            )
        except Exception as exc:  # noqa: BLE001 - every transport failure is recorded, then retried
            attempt["error"] = f"{type(exc).__name__}: {str(exc)[:1500]}"
            _write(folder / f"attempt-{number}.json", attempt)
            error = attempt["error"]
            time.sleep(2**number)
            continue
        choice = completion.choices[0]
        answer = choice.message.content or ""
        attempt.update(
            finished_at=time.time(),
            response_text=answer,
            finish_reason=choice.finish_reason,
            usage=completion.usage.model_dump() if completion.usage else None,
            response_model=getattr(completion, "model", None),
        )
        try:
            value = json.loads(answer)
            validate(value, transcript)
        except (json.JSONDecodeError, CodingError, ValueError) as exc:
            attempt["rule_violation"] = str(exc)[:2000]
            _write(folder / f"attempt-{number}.json", attempt)
            error = attempt["rule_violation"]
            messages = messages + [
                {"role": "assistant", "content": answer},
                {
                    "role": "user",
                    "content": f"Your answer broke a rule: {error}\nFix it and answer again with "
                    "the complete JSON object.",
                },
            ]
            continue
        _write(folder / f"attempt-{number}.json", attempt)
        result = {**record_base, "status": "ok", "attempts": number, "coding": value}
        _write(result_path, result)
        return result
    result = {**record_base, "status": "failed", "error": error, "coding": None}
    _write(folder / "failed.json", result)
    return result
