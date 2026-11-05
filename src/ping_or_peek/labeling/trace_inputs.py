"""Render each LLM labeler's input from one row of the Hugging Face dataset, from its `trace`.

The labelers read Harbor's raw trials (`episode.load_episode`, `reasoning.transcript`). The safety
net's Check B (`labeler_inputs.py`) renders every input from the trace with this module and
compares it with the input the labeler actually sent, as saved in its label store, which shows
that the trace carries everything the labelers read. Check A (`dataset.export_check`) uses only
`render_clipped`, on transcripts built from the raw trials.

The trace is an ATIF trajectory, which `atif.from_trajectory` reads back into what the labelers
read from a trial: the parsed ACP session, Codex's log and the verifier's facts. The renderers
hand those to the labelers' own code: `episode.evidence_parts`, `session_measures` and
`evidence_payload` for the outcome labeler, and `reasoning.transcript.from_session` for the
reasoning coder. The task comes from the row's `task` (the opener, the files, scorer.py and the
user's facts, in the simulated user's key order). Two inputs come from the repository, not the
row, because they are the same for every episode: the simulated user's persona text, and the
labelers' rubrics.
"""

from __future__ import annotations

import json
from typing import Any

from ping_or_peek.config import condition as named_condition
from ping_or_peek.dataset.build import OTHER_CSV
from ping_or_peek.dataset.episode import (
    Task,
    evidence_parts,
    evidence_payload,
    persona_policy_of,
    session_measures,
)

from . import atif
from .asking import asking_payload, message_units
from .labels import (
    AGENT_RULE,
    ASKING,
    EPISODE_STAGES,
    MATERIALITY,
    SCORER_EDIT,
    episode_payload,
    episode_stage,
    materiality_payload,
    scorer_edit_payload,
)
from .reasoning import transcript as reasoning_transcript


def task_of(row: dict[str, Any]) -> Task:
    """The episode's task, from the row's `task`: the CSV whose role is `other` is the one
    scorer.py names when it is in the workspace."""
    task = row["task"]
    condition = named_condition(row["condition"])
    return Task(
        seed=row["seed"],
        condition=row["condition"],
        user_available=condition.user_available,
        scorer_exposed=condition.scorer_exposed,
        files={f["name"]: f["content"] for f in task["csv_files"]},
        target=task["intended_file"],
        decoy=next(f["name"] for f in task["csv_files"] if f["role"] == OTHER_CSV),
        scorer=task["scorer_py"],
        opener=task["opener"],
        facts=json.loads(task["user_facts"]),
    )


def outcome_evidence(row: dict[str, Any], persona: str) -> dict[str, Any]:
    """The evidence payload `episode.load_episode` builds, with reasoning summaries among its
    items. `persona` is the simulated user's persona text."""
    read = atif.from_trajectory(row["trace"])
    task = task_of(row)
    parts = evidence_parts(task, read.session, read.native, read.verifier, row["agent"])
    _, contacts, _, exposure = parts
    return evidence_payload(
        task,
        {
            "response_policy": persona_policy_of(persona),
            "facts": task.facts,
            "instructions": persona,
        },
        read.verifier,
        parts,
        session_measures(task, read.session, contacts, exposure, read.native),
    )


SENSITIVITY = "scorer_edit_with_reasoning"  # the scorer-edit stage of the pass with reasoning


def outcome_stages(persona: str) -> dict[str, Any]:
    """The outcome labeler's stages, by the name `outcome_inputs` gives each: the stage's own
    name, and `SENSITIVITY` for the reasoning pass's scorer-edit stage. `persona` picks the
    episode stage's rubric."""
    return {
        EPISODE_STAGES[AGENT_RULE]: episode_stage(persona_policy_of(persona), AGENT_RULE),
        ASKING.name: ASKING,
        MATERIALITY.name: MATERIALITY,
        SCORER_EDIT.name: SCORER_EDIT,
        SENSITIVITY: SCORER_EDIT,
    }


def outcome_inputs(
    row: dict[str, Any], persona: str, answer: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Each outcome-labeler stage's payload, before the stage adds its version and rubric hash
    (`labels.run_stage`). `answer` is the episode stage's answer: as in `labels.label_episode`,
    it decides whether the materiality stage runs, and with which review flags, because that
    stage reviews the episode stage's judgment."""
    full = outcome_evidence(row, persona)
    primary = {**full, "items": [i for i in full["items"] if i["kind"] != "private_reasoning"]}
    rendered: dict[str, dict[str, Any]] = {}
    episode = episode_payload(primary)
    episode["asking_rule"] = AGENT_RULE
    rendered[EPISODE_STAGES[AGENT_RULE]] = episode
    units = message_units(primary)
    if units:
        rendered[ASKING.name] = asking_payload(primary, units)
    review_inclusion = answer["persona_valid"] is not True
    review_timing = answer["labels"]["user_corrected"] is True
    if review_inclusion or review_timing:
        rendered[MATERIALITY.name] = materiality_payload(primary, review_inclusion, review_timing)
    if full["checker_exposed"]:
        rendered[SCORER_EDIT.name] = scorer_edit_payload(primary)
        rendered[SENSITIVITY] = scorer_edit_payload(full)
    return rendered


def coder_transcript(row: dict[str, Any], output_limit: int) -> reasoning_transcript.Transcript:
    """The reasoning coder's transcript (`reasoning.transcript.from_session`), from the trace.
    `output_limit` clips each tool output, as the coder's own limit does."""
    read = atif.from_trajectory(row["trace"])
    return reasoning_transcript.from_session(
        read.session,
        read.native,
        agent=row["agent"],
        task=task_of(row),
        episode=row["episode_id"],
        model=row["model"],
        setting=row["setting"],
        identity={},
        output_limit=output_limit,
    )


def render_clipped(build) -> str:
    """The transcript text the reasoning coder sent, clipped by the coder's own rule
    (`reasoning.label.clipped`): built by `build(output_limit)` at the default limit, and rebuilt
    at the shorter one when the text is too long."""
    from .reasoning import label

    return label.clipped(build(reasoning_transcript.OUTPUT_LIMIT), build)[1]


def coder_input(row: dict[str, Any]) -> str:
    """The transcript text the reasoning coder sent, from the trace."""
    return render_clipped(lambda limit: coder_transcript(row, limit))
