"""Locate and read the files of one Harbor trial directory. Read-only: never writes there.

A simulated-user trial directory holds (see harbor.models.trial.paths.TrialPaths):

    result.json                  trial metadata: task, agent, model, timing, exception
    config.json                  the trial configuration
    agent/bridge-trajectory.json the acpx session export: the agent under test's record
    agent/sessions/              native session files (Codex rollouts, Claude Code transcripts)
    user-agent/                  the simulated user's own logs and trajectory
    verifier/                    reward and the verifier's metrics JSON
"""

from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ping_or_peek.config import CONDITIONS, MODELS, REPO, harness_of

from .acp import RecordError, read_record_text

ACP_EXPORT = "agent/bridge-trajectory.json"
CONTROL = "conversational_control"  # the task with no seed: a conversation, not a measurement
CONDITION_NAMES = (*CONDITIONS, CONTROL)


def read_json(path: Path, trial: str) -> dict[str, Any]:
    """One of `trial`'s JSON record files, which each hold an object. Raises RecordError, naming
    the trial and the file, when it is absent, isn't text, isn't JSON or isn't an object."""
    text = read_record_text(path, trial)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise RecordError(f"{trial}: {path} is not JSON ({error})") from error
    if not isinstance(value, dict):
        raise RecordError(f"{trial}: {path} is not a JSON object")
    return value


def read_jsonl(path: Path, trial: str) -> list[dict[str, Any]]:
    """One of `trial`'s JSON-lines record files (a Codex rollout, a session transcript): each
    non-blank line an object. Raises RecordError, naming the trial and the file, for one that is
    absent or isn't text, and the line too for one that isn't an object."""
    records = []
    for number, line in enumerate(read_record_text(path, trial).splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RecordError(f"{trial}: line {number} of {path} is not JSON ({error})") from error
        if not isinstance(record, dict):
            raise RecordError(f"{trial}: line {number} of {path} is not a JSON object")
        records.append(record)
    return records


@dataclass
class Trial:
    path: Path
    result: dict[str, Any]
    config: dict[str, Any]
    task_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.result.get("trial_name") or self.path.name

    @property
    def job_name(self) -> str:
        """The Harbor job: `<phase>__<harness>__<setting>` in this experiment's run script."""
        return self.path.parent.name

    @property
    def setting(self) -> str | None:
        """S, A or B, from the task metadata or the job name."""
        match = re.search(r"__([SAB])$", self.job_name)
        return self.task_metadata.get("setting") or (match.group(1) if match else None)

    @property
    def phase(self) -> str:
        return self.job_name.split("__")[0]

    @property
    def task_name(self) -> str:
        if not self.result.get("task_name"):
            raise RecordError(f"{self.name}: result.json names no task")
        return self.result["task_name"]

    def _agent_config(self) -> dict[str, Any]:
        """The agent's configuration: result.json's copy, or config.json's for an older trial."""
        return (self.result.get("config") or self.config).get("agent") or {}

    @property
    def agent(self) -> str:
        """Harbor's name for the agent under test: claude-code, codex or opencode."""
        if not self._agent_config().get("name"):
            raise RecordError(f"{self.name}: neither result.json nor config.json names the agent")
        return self._agent_config()["name"]

    @property
    def harness(self) -> str:
        return harness_of(self.agent)

    @property
    def model(self) -> str:
        """The model under test: Harbor's agent_info, or the configured model name, without its
        provider prefix. Raises RecordError, naming the trial, when neither names one of the
        models under test (`config.MODELS`, the models the run script runs)."""
        info = (self.result.get("agent_info") or {}).get("model_info") or {}
        name = info.get("name") or self._agent_config().get("model_name")
        if not name:
            raise RecordError(
                f"{self.name}: neither agent_info nor the agent's config names a model"
            )
        model = name.split("/")[-1]
        if model not in MODELS:
            raise RecordError(
                f"{self.name}: its model {name!r} is none of the models under test {sorted(MODELS)}"
            )
        return model

    @property
    def exception(self) -> dict[str, Any] | None:
        """The trial's exception_info, None when it raised nothing. Raises RecordError, naming
        the trial, when it records no exception_type: every reader decides by the type."""
        info = self.result.get("exception_info")
        if info is not None and not info.get("exception_type"):
            raise RecordError(f"{self.name}: its exception_info records no exception_type")
        return info

    @property
    def acp_export(self) -> Path:
        return self.path / ACP_EXPORT

    def native_session_files(self) -> list[Path]:
        root = self.path / "agent" / "sessions"
        return sorted(root.rglob("*.jsonl")) if root.exists() else []

    def user_agent_files(self) -> list[Path]:
        root = self.path / "user-agent"
        return sorted(p for p in root.rglob("*") if p.is_file()) if root.exists() else []

    def verifier_metrics(self) -> dict[str, Any]:
        """The verifier's metrics JSON, plus Harbor's reward."""
        metrics: dict[str, Any] = {}
        root = self.path / "verifier"
        for name in ("metrics.json", "reward.json"):
            if (root / name).exists():
                metrics.update(read_json(root / name, self.name))
        if (root / "reward.txt").exists() and "reward" not in metrics:
            text = read_record_text(root / "reward.txt", self.name).strip()
            try:
                metrics["reward"] = float(text)
            except ValueError as error:
                raise RecordError(
                    f"{self.name}: {root / 'reward.txt'} is not a number ({text!r})"
                ) from error
        rewards = ((self.result.get("verifier_result") or {}).get("rewards")) or {}
        metrics.setdefault("rewards", rewards)
        return metrics

    def seconds(self, phase: str) -> float | None:
        """Wall-clock seconds of a phase: agent_execution, agent_setup, verifier, or the trial."""
        from datetime import datetime

        timing = self.result if phase == "trial" else self.result.get(phase) or {}
        start, end = timing.get("started_at"), timing.get("finished_at")
        if not start or not end:
            return None
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


class PersonaNotFound(RecordError):
    """A trial's persona can't be found in this repository: a record that breaks a rule, not a
    missing one (`RecordMissing`)."""


def _locked_persona(trial: Trial) -> dict[str, str]:
    """The persona Harbor locked for the trial: its lock's `user_persona` entry, whose `digest`
    is the persona's sha256. Raises PersonaNotFound, naming the trial, when no lock names one."""
    lock_path = trial.path / "lock.json"
    locked = (read_json(lock_path, trial.name) if lock_path.exists() else {}).get(
        "user_persona"
    ) or {}
    if not locked.get("digest"):
        raise PersonaNotFound(f"{trial.name}: no lock.json names the persona it ran with")
    return locked


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def check_persona(trial: Trial, text: str) -> None:
    """Raises PersonaNotFound, naming the trial, unless `text` (a persona given for it, such as
    `label --persona-file`) is the persona Harbor locked for the trial, by its sha256."""
    digest = _locked_persona(trial)["digest"].removeprefix("sha256:")
    if _sha256(text) != digest:
        raise PersonaNotFound(
            f"{trial.name}: the persona given has sha256 {_sha256(text)}, not the locked {digest}"
        )


def persona_text(trial: Trial) -> str:
    """The persona file this trial ran with, found in this repository by the digest Harbor locked.

    The lock names the file by its path in the checkout the run used, which may have moved, so
    the text is found by digest among this repository's files of the lock's file name: first in
    the working tree, then in the history, so a later edit to the persona never changes what an
    earlier trial is judged against. Raises PersonaNotFound, naming the trial, when it has no
    lock, no version matches, or git can't be run: a trial is never judged against no persona.
    """
    import subprocess

    locked = _locked_persona(trial)
    if not locked.get("path"):
        raise PersonaNotFound(f"{trial.name}: its lock names the persona's digest, not its file")
    name, digest = Path(locked["path"]).name, locked["digest"].removeprefix("sha256:")

    def git(*args: str) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False
            )
        except FileNotFoundError as error:
            raise PersonaNotFound(
                f"{trial.name}: git, which finds the persona in this repository, can't be run"
            ) from error

    def named(listing: str) -> list[str]:
        return sorted({p for p in listing.split("\n") if p and Path(p).name == name})

    def matches(text: str) -> bool:
        return _sha256(text) == digest

    for path in named(git("ls-files", "--", f"*{name}").stdout):
        if (REPO / path).exists() and matches((REPO / path).read_text()):
            return (REPO / path).read_text()
    for path in named(git("log", "--format=", "--name-only", "--", f"*{name}").stdout):
        for commit in git("log", "--format=%H", "--", path).stdout.split():
            shown = git("show", f"{commit}:{path}")
            if shown.returncode == 0 and matches(shown.stdout):
                return shown.stdout
    raise PersonaNotFound(
        f"{trial.name}: no version of {name} in this repository has the locked digest {digest}"
    )


class RecordMissing(Exception):
    """A record file a trial must have is absent: the trial never produced a full record. The one
    error callers such as `ping_or_peek.runs.checks` take for that, so `ping_or_peek.runs.retry` reruns the trial; any
    other error reaches them by name."""


def load_trial(path: Path) -> Trial:
    """A trial's records, read-only. Raises RecordMissing when it has no result.json, and
    RecordError, naming the trial, when its task or exception record breaks a rule."""
    path = Path(path)
    if not (path / "result.json").exists():
        raise RecordMissing(f"{path.name}: no result.json")
    result = read_json(path / "result.json", path.name)
    config = read_json(path / "config.json", path.name) if (path / "config.json").exists() else {}
    trial = Trial(path, result, config)
    trial.task_metadata = _task_metadata(trial)
    _ = trial.exception  # validated here, so a reader that stops before reading it sees it fail
    return trial


def _task_metadata(trial: Trial) -> dict[str, Any]:
    """Condition, seed and setting: from task.toml [metadata] when it is in this repository, else
    from the task's name, which the generator writes as `<condition>__seed<seed>`. Raises
    RecordError, naming the trial, when neither names the condition, or the seed of a measured
    task (the conversational control has none)."""
    import tomllib

    metadata: dict[str, Any] = {}
    task_path = ((trial.result.get("config") or trial.config).get("task") or {}).get("path")
    task_toml = REPO / task_path / "task.toml" if task_path else None  # absolute paths stay
    if task_toml and task_toml.exists():
        metadata = dict(tomllib.loads(task_toml.read_text()).get("metadata", {}))
    name = trial.task_name
    if "condition" not in metadata:
        found = [c for c in CONDITION_NAMES if c in name or c.replace("_", "-") in name]
        metadata["condition"] = found[0] if found else None
    if "seed" not in metadata:
        match = re.search(r"seed[-_]?(\d+)", name)
        metadata["seed"] = int(match.group(1)) if match else None
    if metadata["condition"] is None:
        raise RecordError(
            f"{trial.name}: neither task.toml nor the task name {name!r} names a condition"
        )
    if metadata["seed"] is None and metadata["condition"] != CONTROL:
        raise RecordError(
            f"{trial.name}: neither task.toml nor the task name {name!r} names a seed"
        )
    return metadata


def find_trials(paths: list[str] | list[Path], *, waiting: bool = False) -> list[Path]:
    """Finished trials under each path: a trial, a job, or a quoted glob of jobs such as
    ".../harbor/pilot__*". Each trial is returned once, by its resolved path, which is what keys
    and deduplicates it however the path was spelled. Globs are expanded on every call, so
    `label --watch` finds later jobs.

    Raises ValueError, naming it, for a path that selects no finished trial, unless `waiting`
    (`label --watch`), when a job may have none yet."""
    found: set[Path] = set()
    for pattern in map(str, paths):
        is_glob = any(c in pattern for c in "*?[")
        matches = sorted(p for p in map(Path, glob.glob(pattern)) if p.is_dir())
        selected: set[Path] = set()
        for p in matches if is_glob else [Path(pattern)]:
            selected.update(t.resolve() for t in ([p] if is_finished_trial(p) else trial_dirs(p)))
        if not selected and not waiting:
            raise ValueError(
                f"{pattern}: selects no finished trial (a trial, a job holding one, or a quoted "
                "glob of jobs)"
            )
        found |= selected
    return sorted(found)


def is_finished_trial(path: Path) -> bool:
    """Whether `path` is a finished trial's folder: it holds a result.json and an agent/ folder."""
    return (Path(path) / "result.json").exists() and (Path(path) / "agent").is_dir()


def trial_dirs(jobs_root: Path) -> list[Path]:
    """Every finished trial directory under a jobs root (a result.json marks a finished trial)."""
    return sorted(
        p.parent for p in Path(jobs_root).rglob("result.json") if (p.parent / "agent").is_dir()
    )
