"""The dataset card (`dataset.card`) takes every closed vocabulary's terms from the module
that owns it, so no term is maintained twice."""

import ast
import re
from pathlib import Path

import pytest

from ping_or_peek import cli
from ping_or_peek import config as project_config
from ping_or_peek.dataset import acp, build, card
from ping_or_peek.labeling import atif
from ping_or_peek.runs.run import INSTRUCTION_FILE, PASSAGE

VOCABULARIES = {
    "paths (config.PATHS)": project_config.PATHS,
    "setting codes (config.SETTINGS)": project_config.SETTINGS,
    "conditions (config.CONDITIONS)": project_config.CONDITIONS,
    "harnesses (config.HARNESSES)": project_config.HARNESSES,
    "agents (config.NATIVE_AGENTS)": tuple(project_config.NATIVE_AGENTS.values()),
    "models (config.MODELS)": project_config.MODELS,
    "families (build.FAMILIES)": tuple(family.lower() for family in build.FAMILIES),
    "event types (build.EVENT_TYPES)": build.EVENT_TYPES,
    "ACP event kinds (acp.EVENT_KINDS)": acp.EVENT_KINDS,
    "CSV roles (build.CSV_ROLES)": build.CSV_ROLES,
    "reasoning sources (build.REASONING_SOURCES)": tuple(build.REASONING_SOURCES.values()),
    "note files (ping_or_peek.runs.run.INSTRUCTION_FILE)": tuple(INSTRUCTION_FILE.values()),
    "passages (ping_or_peek.runs.run.PASSAGE)": tuple(PASSAGE.values()),
    # Every closed vocabulary of the trace, from its registry.
    **{
        f"trace {field} (build.VOCABULARY)": tuple(
            values if isinstance(values, tuple) else set().union(*values.values())
        )
        for field, values in build.VOCABULARY.items()
    },
    # Every closed field of the trajectory's question and approval views, and its step sources.
    **{
        f"view {view}.{field} (build.VIEW_VOCABULARY)": values
        for view, fields in build.VIEW_VOCABULARY.items()
        for field, values in fields.items()
    },
    "ATIF step sources (atif.SOURCES)": atif.SOURCES,
}
COLUMNS = set(build.SCHEMA.names)  # a column's name, which the card writes, is not a value
# A value that is also a documented field's name (the question view's `tool`) can't be told
# apart from it in the card's text, so the rule leaves it out.
FIELD_NAMES = {field for fields in build.VIEW_VOCABULARY.values() for field in fields}


# Proper names, by their owner: each may appear only in the card assignments that define it,
# anywhere else as a whole word.
NAMES = {
    # OpenCode's agent name is its harness label, which HARNESS_LABELS defines.
    "agent names (card.AGENTS)": (
        tuple(name for name, _ in card.AGENTS.values()),
        ("AGENTS", "HARNESS_LABELS"),
    ),
    "setting names (card.SETTINGS)": (
        tuple(name for _, name, _ in card.SETTINGS),
        ("SETTINGS",),
    ),
    "model names (card.MODEL_LABELS)": (tuple(card.MODEL_LABELS.values()), ("MODEL_LABELS",)),
    "harness labels (card.HARNESS_LABELS)": (
        tuple(card.HARNESS_LABELS.values()),
        ("HARNESS_LABELS",),
    ),
    # An agent's product name (Claude Code) and a model's name (Claude Opus 5.5) hold its
    # family's label, so AGENTS and MODEL_LABELS may too.
    "family labels (build.FAMILIES)": (build.FAMILIES, ("AGENTS", "MODEL_LABELS")),
}


def literals(source: str, outside: tuple[str, ...] = ()) -> list[str]:
    """Every string literal in `source`, the literal parts of f-strings included, less those in
    the assignments to the names in `outside`."""
    tree = ast.parse(source)
    skipped = {
        id(node)
        for assignment in ast.walk(tree)
        if isinstance(assignment, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in outside for t in assignment.targets)
        for node in ast.walk(assignment)
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in skipped
    ]


def repeats(value: str, literal: str) -> bool:
    """Whether `literal` writes `value` out: equal to it, or holding it in backticks as the card
    marks a term; an identifier-shaped value (with `_`, `/`, `.` or a digit) anywhere as a whole
    identifier, not inside a longer one."""
    shaped = re.search(r"[_/.\d]", value) is not None
    whole = re.search(rf"(?<![\w]){re.escape(value)}(?![\w])", literal) is not None
    return literal == value or f"`{value}`" in literal or (shaped and whole)


def test_the_card_writes_no_vocabulary_term_out():
    source = Path(card.__file__).read_text()
    found = sorted(
        {
            (name, value)
            for name, values in VOCABULARIES.items()
            for value in values
            for literal in literals(source)
            if literal not in COLUMNS and value not in FIELD_NAMES and repeats(value, literal)
        }
    )
    assert found == []


def test_the_rule_catches_a_term_written_out():
    assert repeats("asked the user", "`asked the user`: the agent asked")
    assert repeats("GPT-6.1 Sol", "a model (GPT-6.1 Sol) from that text")
    assert not repeats("asked the user", "The agent asked the user something")
    assert not repeats("tool_call", "the same `tool_call_id`")


def test_the_card_writes_each_proper_name_only_where_its_owner_defines_it():
    source = Path(card.__file__).read_text()
    found = sorted(
        {
            (name, value)
            for name, (values, owner) in NAMES.items()
            for value in values
            for literal in literals(source, outside=owner)
            if re.search(rf"(?<![\w]){re.escape(value)}(?![\w])", literal)
        }
    )
    assert found == []


def test_the_cards_event_table_takes_each_role_from_the_builder():
    """`card.EVENTS` writes no trace role (`build.ROLES`) as a literal. The ATIF trace's table
    has no role column; the event list's roles come from the builder."""
    tree = ast.parse(Path(card.__file__).read_text())
    table = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "EVENTS" for t in node.targets)
    )
    written = {
        node.value
        for node in ast.walk(table)
        if isinstance(node, ast.Constant) and node.value in build.ROLES
    }
    assert written == set()


def test_a_build_missing_a_column_stops_the_card(tmp_path):
    """The card reads the build through the schema gate (`build.read_shards`)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    (tmp_path / "data").mkdir()
    schema = build.written(build.SCHEMA)
    kept = pa.schema([f for f in schema if f.name != "verbalized_test_awareness"])
    pq.write_table(kept.empty_table(), tmp_path / "data" / "train-0.parquet")
    with pytest.raises(SystemExit, match="verbalized_test_awareness"):
        cli.main(["dataset", "card", "--data", str(tmp_path), "--records", str(tmp_path)])


def test_the_card_lists_every_value_each_view_holds():
    """Written out by hand: each value of the question and approval views' closed fields
    appears, in backticks, in that record's description."""
    expected = {
        acp.QUESTION: {"answered", "declined", "cancelled", "pending", "withdrawn_unanswered",
                       "error", "no_reply", "user", "async", "away", "no_further_message"},
        acp.APPROVAL: {"allow", "deny", "cancelled", "pending", "answered",
                       "withdrawn_unanswered", "user", "client_policy"},
    }  # fmt: skip
    described = dict(card.EVENTS)
    for kind, values in expected.items():
        assert {v for v in values if f"`{v}`" not in described[kind]} == set(), kind


def test_an_empty_data_folder_stops_the_card_naming_it(tmp_path):
    (tmp_path / "data").mkdir()
    with pytest.raises(SystemExit, match="no shards"):
        cli.main(["dataset", "card", "--data", str(tmp_path), "--records", str(tmp_path)])


def test_the_card_describes_the_streams_extra_as_written():
    """The root `extra.acp` keeps each bookkeeping event's position and time, and whether the
    stream recorded directions and times."""
    text = " ".join(t for _, t in card.EVENTS) + Path(card.__file__).read_text()
    assert "are left out" not in text
    for key in ("bookkeeping", "has_direction", "has_timestamps"):
        assert f"`{key}`" in text, key


def test_the_card_says_what_a_records_data_is():
    """A record's `data` is what the session parser recorded of the event, not the message as
    sent (a message's is its chunk count; a request's, its request's number)."""
    source = Path(card.__file__).read_text()
    assert "`data` as sent" not in source and "parser" in source
