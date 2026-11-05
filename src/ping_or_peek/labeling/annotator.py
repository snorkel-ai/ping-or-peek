"""The LLM annotator: strict-JSON chat completions, with every request and response saved.

The annotator is any OpenAI-compatible chat-completions endpoint. Our labels come from GPT-6.1 Sol
at medium reasoning effort through a Portkey gateway (the earlier study used GPT-5.6 Sol, which stays available for
comparison). With an OpenAI key you can call the model
directly: leave `base_url` unset.

Each call is one item of work (one stage of one episode). Every attempt is written to disk before
the next one starts, so a crashed run resumes where it stopped and nothing paid for is lost.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

ATTEMPTS = 3


@dataclass(frozen=True)
class AnnotatorConfig:
    """Which model labels the episodes, and how to reach it. Holds no secrets."""

    model: str = "gpt-6.1-sol"
    reasoning_effort: str | None = "medium"  # None: the request carries no effort parameter
    max_output_tokens: int = 8000
    base_url: str | None = None  # None means the OpenAI API
    api_key_env: str = "OPENAI_API_KEY"  # the name of the variable holding the key

    @classmethod
    def portkey(
        cls, model: str = "gpt-6.1-sol", reasoning_effort: str | None = "medium"
    ) -> AnnotatorConfig:
        """Our setup: the model through the Portkey gateway. A bare OpenAI model name is routed
        as "@openai/<model>"; any other provider is given as its route, "@provider/model"."""
        return cls(
            model=model if model.startswith("@") else f"@openai/{model}",
            reasoning_effort=reasoning_effort,
            base_url=os.environ.get("PORTKEY_BASE_URL", "https://api.portkey.ai/v1"),
            api_key_env="PORTKEY_API_KEY",
        )

    @property
    def name(self) -> str:
        """The labeler, as a folder name: model (without its gateway route) and effort, e.g.
        "gpt-6.1-sol-medium". Each labeler's records in a run are kept under this name."""
        model = self.model.split("/", 1)[-1] if self.model.startswith("@") else self.model
        return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{model}-{self.reasoning_effort or 'default'}")

    def public(self) -> dict[str, Any]:
        return asdict(self)


class LabelError(ValueError):
    """The annotator's answer broke a rule of the rubric (for example, an altered quote)."""


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The JSON schema of `model` for strict structured output, without titles or defaults."""

    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: clean(v) for k, v in node.items() if k not in ("title", "default")}
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node

    return clean(model.model_json_schema())


def request_messages(system: str, payload: dict[str, Any]) -> list[dict[str, str]]:
    """The first request of one item of work: the rubric, then the payload as JSON text, whose
    key order is part of the request."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


class Annotator:
    def __init__(self, config: AnnotatorConfig, client: Any = None) -> None:
        self.config = config
        if client is None:
            from openai import OpenAI

            key = os.environ.get(config.api_key_env)
            if not key:
                raise RuntimeError(f"Set {config.api_key_env} to call the annotator")
            client = OpenAI(api_key=key, base_url=config.base_url, max_retries=2, timeout=600)
        self.client = client

    def label(
        self,
        *,
        system: str,
        payload: dict[str, Any],
        schema: type[BaseModel],
        validate: Callable[[dict[str, Any]], None],
        attempts_dir: Path,
    ) -> dict[str, Any]:
        """Ask for one strict-JSON answer and check it; feed rule violations back, up to 3 tries.

        Every attempt (request, response, error) is saved under `attempts_dir` as 1.json, 2.json...
        Raises LabelError when no attempt passes.
        """
        messages = request_messages(system, payload)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": strict_schema(schema),
                "strict": True,
            },
        }
        error = "no attempt made"
        for attempt in range(ATTEMPTS):
            number = len(list(attempts_dir.glob("*.json"))) + 1
            record: dict[str, Any] = {
                "annotator": self.config.public(),
                "request_messages": list(messages),
                "started_at": time.time(),
            }
            try:
                effort = self.config.reasoning_effort
                completion = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    max_completion_tokens=self.config.max_output_tokens,
                    response_format=response_format,
                    **({"reasoning_effort": effort} if effort else {}),
                )
            # Any transport failure is recorded, then the same request is resent after a pause.
            except Exception as exc:  # noqa: BLE001
                record["error"] = f"{type(exc).__name__}: {str(exc)[:1500]}"
                write_json(attempts_dir / f"{number}.json", record)
                error = record["error"]
                time.sleep(2 ** (attempt + 1))
                continue
            choice = completion.choices[0]
            text = choice.message.content or ""
            record.update(
                finished_at=time.time(),
                response_text=text,
                finish_reason=choice.finish_reason,
                response_id=getattr(completion, "id", None),
                usage=completion.usage.model_dump() if completion.usage else None,
            )
            try:
                if choice.finish_reason != "stop":
                    raise LabelError(f"Response ended with finish_reason={choice.finish_reason}")
                value = schema.model_validate(json.loads(text)).model_dump()
                validate(value)
            except (LabelError, ValueError) as exc:
                record["error"] = str(exc)[:1500]
                write_json(attempts_dir / f"{number}.json", record)
                error = record["error"]
                messages = [
                    *messages,
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": "Evidence validation failed: "
                        + record["error"]
                        + ". Reassess against the same evidence and rubric; do not invent "
                        "supporting text or force a positive or negative result. Use exact "
                        "original strings. Return a complete assessment.",
                    },
                ]
                continue
            record["value"] = value
            write_json(attempts_dir / f"{number}.json", record)
            return value
        raise LabelError(f"No valid answer after {ATTEMPTS} attempts: {error}")
