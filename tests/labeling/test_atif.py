"""The dataset's trace as an ATIF trajectory: written by `dataset.trajectory` with
Harbor's models, read back by `labeling.atif` into the parsed session the labelers read.

Writing needs Harbor, which CI's sync leaves out, so these tests skip there.
"""

import json

import pytest

from tests.acp_fixtures import DECOY, OPENER, TARGET, bridge_export, claude_history, text, update
from tests.acp_fixtures import codex_history as codex_fixture

pytest.importorskip("harbor", reason="Harbor is not installed; CI's sync leaves it out")

from harbor.models.trajectories import Trajectory

from ping_or_peek import cli
from ping_or_peek.dataset import acp, build, trajectory
from ping_or_peek.labeling import atif
from tests.dataset.test_dataset_check import argv as check_argv

VERIFIER = {"changed_original_files": [TARGET], "target_modified": True, "reward": 1.0}
QUESTION = "Which CSV should I append the total to?"
ASK = {"title": QUESTION, "options": [TARGET, DECOY]}


def stamp(i: int) -> str:
    return f"2026-10-13T21:00:{i:02d}.000Z"


def parsed(tmp_path, history):
    path = tmp_path / "bridge-trajectory.json"
    path.write_text(json.dumps(bridge_export(history)))
    return acp.parse_export(path)


def written(session, native=(), agent="claude-code", user_mode="available"):
    """The trajectory as a reader of the dataset gets it: serialized and parsed back."""
    document = trajectory.to_trajectory(
        session,
        list(native),
        episode_id="native-m-A-c-s100",
        agent=agent,
        agent_version="1.0",
        model="claude-opus-5-5",
        user_mode=user_mode,
        verifier=VERIFIER,
    )
    return json.loads(json.dumps(document))


def async_history():
    """Codex asks with request_user_input_async (forwarded as marked agent text) and the user
    answers mid-turn; a usage update ends the stream."""
    history = codex_fixture()[:5]  # session/new and its reply, the prompt and the first command
    marker = {"codex": {"delivery": "async", "questions": [ASK]}}
    return [
        *history,
        update("agent_message_chunk", content=text("Checking first. ")),
        update("agent_message_chunk", content=text(QUESTION), _meta=marker),
        {
            "jsonrpc": "2.0",
            "method": "_session/steering",
            "params": {"sessionId": "s1", "prompt": [text(TARGET)]},
        },
        update("agent_message_chunk", content=text("Appending to it.")),
        {"jsonrpc": "2.0", "id": 0, "result": {"stopReason": "end_turn"}},
        update("usage_update", used=1, size=2),  # bookkeeping the trace leaves out
    ]


ASYNC_NATIVE = [
    {"type": "AgentMessage", "id": "call_a", "turn_id": "t1", "ordinal": 4,
     "timestamp": stamp(4), "delivery": "async", "questions": [ASK], "text": QUESTION},
    {"type": "CommandExecution", "id": "c1", "turn_id": "t1", "ordinal": 2,
     "timestamp": stamp(2), "command": ["bash", "-lc", f"cat {TARGET}"], "cwd": "/workspace",
     "status": "completed", "exit_code": 0, "output": "value\n10\n"},
]  # fmt: skip


def same_session(read: acp.Session, raw: acp.Session) -> None:
    """Everything of the parsed session the labelers read is in the trace."""
    kept = [e for e in raw.events if e.kind != "other"]
    assert [e for e in read.events if e.kind != "other"] == kept
    assert len(read.events) == len(raw.events)
    assert read.requests == raw.requests
    assert read.tool_calls == raw.tool_calls
    assert (read.prompts, read.stop_reasons) == (raw.prompts, raw.stop_reasons)
    assert (read.cwd, read.has_timestamps) == (raw.cwd, raw.has_timestamps)


def test_a_claude_trace_is_valid_atif_and_reads_back_as_the_parsed_session(tmp_path):
    raw = parsed(tmp_path, claude_history())
    document = written(raw)
    Trajectory.model_validate(document)
    assert document["schema_version"] == "ATIF-v1.8"
    read = atif.from_trajectory(json.dumps(document))
    same_session(read.session, raw)
    assert (read.native, read.verifier) == ([], VERIFIER)
    # The user's facts are the task's, in the row's `task` column: one copy, not two.
    assert "user_facts" not in document["extra"]


def test_steps_follow_harbors_grouping(tmp_path):
    steps = written(parsed(tmp_path, claude_history()))["steps"]
    shape = [
        (
            s["source"],
            s["message"],
            s.get("reasoning_content"),
            [(c["tool_call_id"], c["function_name"]) for c in s.get("tool_calls") or []],
            [r["source_call_id"] for r in (s.get("observation") or {}).get("results", [])],
        )
        for s in steps
    ]
    assert shape == [
        ("user", OPENER.strip(), None, [], []),
        # Text and the first command, up to its completion, observed through its terminal.
        ("agent", "I'll look at the files.", "Two CSVs. Let me look.", [("t1", "execute")], ["t1"]),
        # A new tool call after a completed one starts a step; the question and the approval
        # stay with the step they came in.
        ("agent", "", None, [("t2", "read")], ["t2"]),
        ("agent", "", None, [("t4", "edit")], []),
        ("agent", "Done: I appended 12 to the file.", None, [], []),
        ("system", "", None, [], []),
    ]
    assert steps[2]["tool_calls"][0]["arguments"] == {"file_path": "/workspace/scorer.py"}
    kinds = [r["kind"] for r in steps[2]["extra"]["acp_events"]]
    assert kinds == [
        "tool_call",
        "client_op",
        "client_op_result",
        "tool_update",
        "question",
        "question_reply",
        "approval",
        "approval_reply",
    ]


def test_questions_and_approvals_carry_the_reply_and_its_place(tmp_path):
    records = [r for s in written(parsed(tmp_path, claude_history()))["steps"]
               for r in (s.get("extra") or {}).get("acp_events", [])]  # fmt: skip
    question = next(r for r in records if r["kind"] == "question")
    assert question["question"] == {
        "tool": "AskUserQuestion",
        "text": "\n".join(
            [
                "Which CSV should I update?",
                "CSV file",
                f'Options: ["{DECOY}", "{TARGET}"]',
                "Other",
                "Type your own answer, or add a note to the option you chose above (optional).",
            ]
        ),
        "answer": TARGET,
        "status": "answered",
        "answered_by": "user",
    }
    assert (question["request"]["reply_position"], question["request"]["reply_ts"]) == (
        14,
        stamp(20),
    )
    approval = next(r for r in records if r["kind"] == "approval")
    assert approval["approval"] == {
        "options": ["Allow", "Reject"],
        "answer": "Allow",
        "decision": "allow",
        "status": "answered",
        "answered_by": "user",
    }
    assert approval["request"]["reply_position"] == 16


def test_a_codex_async_question_is_one_question_at_its_message(tmp_path):
    raw = parsed(tmp_path, async_history())
    document = written(raw, ASYNC_NATIVE, agent="codex")
    Trajectory.model_validate(document)
    steps = document["steps"]
    records = [r for s in steps for r in (s.get("extra") or {}).get("acp_events", [])]
    asked = [r for r in records if r["kind"] == "question"]
    assert len(asked) == 1 and asked[0]["delivery"] == "async"
    assert asked[0]["position"] == 4
    assert asked[0]["question"] == {
        "tool": "request_user_input_async",
        "text": QUESTION,
        "options": [TARGET, DECOY],
        "answer": TARGET,
        "status": "answered",
        "reason": None,
        "answered_by": "user",
        "reply_position": 5,
        "answered_mid_turn": True,
        "rollout_id": "call_a",
    }
    # Replacing the message: its text is in no step's message.
    assert not any(QUESTION in s["message"] for s in steps)
    assert not any(r["kind"] == "agent_message" and r["position"] == 4 for r in records)
    assert document["extra"]["agent_log"] == ASYNC_NATIVE
    read = atif.from_trajectory(document)
    same_session(read.session, raw)
    assert read.native == ASYNC_NATIVE


def test_the_legacy_trace_projects_from_the_trajectory(tmp_path):
    for history, native, agent, family in (
        (claude_history(), [], "claude-code", "claude"),
        (async_history(), ASYNC_NATIVE, "codex", "gpt"),
        (codex_fixture(), [], "codex", "gpt"),
    ):
        raw = parsed(tmp_path, history)
        read = atif.from_trajectory(written(raw, native, agent=agent))
        assert build.trace(read.session, read.native, agent, family) == build.trace(
            raw, native, agent, family
        )


def test_a_trajectory_that_does_not_read_back_is_refused(tmp_path):
    raw = parsed(tmp_path, claude_history())
    document = written(raw)
    document["steps"][1]["message"] = "I'll look at the file."  # one character short
    with pytest.raises(ValueError, match="doesn't read back"):
        trajectory.check_reads_back(document, raw, [], VERIFIER)


def test_a_terminal_backed_result_carries_the_terminals_output_exit_code_and_truncation(tmp_path):
    """Claude Code runs `ls` in a client terminal: the call's observation holds the terminal's
    output, and its exit code and truncation flag, which the tool update itself doesn't carry."""
    steps = written(parsed(tmp_path, claude_history()))["steps"]
    (result,) = steps[1]["observation"]["results"]
    assert result["source_call_id"] == "t1"
    assert result["content"] == f"{DECOY}\nscorer.py\n{TARGET}\n"
    assert result["extra"] == {
        "terminals": [{"terminal_id": "term-1", "exit_code": 0, "truncated": False}]
    }


def test_bookkeeping_events_keep_their_times(tmp_path):
    """A usage update logged earlier than the event before it: the stream's clock runs backwards,
    which `episode.log_order_matches_timestamps` reads, so its time is kept, not its content."""
    from ping_or_peek.dataset import episode

    export = bridge_export(async_history())
    export["interaction_log"][-1]["at"] = stamp(0)
    path = tmp_path / "bridge-trajectory.json"
    path.write_text(json.dumps(export))
    raw = acp.parse_export(path)
    read = atif.from_trajectory(written(raw, ASYNC_NATIVE, agent="codex"))
    assert [e.ts for e in read.session.events] == [e.ts for e in raw.events]
    assert episode.log_order_matches_timestamps(read.session) is False
    assert episode.log_order_matches_timestamps(raw) is False


def test_a_trajectory_whose_request_keys_were_reordered_is_refused(tmp_path):
    """The question's options render in its schema's key order, so a reordered schema is a
    different input for the labelers."""
    raw = parsed(tmp_path, claude_history())
    document = written(raw)
    for _, record in atif.records(document):
        if record["kind"] == "question":
            schema = record["request"]["params"]["requestedSchema"]
            schema["properties"] = dict(reversed(schema["properties"].items()))
    with pytest.raises(ValueError, match="doesn't read back"):
        trajectory.check_reads_back(document, raw, [], VERIFIER)


def test_a_credential_in_a_trajectorys_keys_stops_the_build(tmp_path):
    """The guard reads the trajectory document's keys as well as its values."""
    from ping_or_peek.dataset import build

    history = claude_history()
    for message in history:
        if message.get("method") == "terminal/create":
            message["params"]["_meta"] = {"known-secret-value-123": "marker"}
    document = written(parsed(tmp_path, history))
    with pytest.raises(SystemExit) as stopped:
        build.Guard({"PORTKEY_API_KEY": "known-secret-value-123"})("e1", {"trace": document})
    assert "e1: trace." in str(stopped.value) and "(key #" in str(stopped.value)
    assert "known-secret-value-123" not in str(stopped.value)


def test_a_trace_that_does_not_read_back_stops_the_check_with_its_report_written(tmp_path):
    """Every later check reads the traces, so the check reads each back first."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    task = {"opener": "o", "intended_file": "a.csv", "checker_target_file": None,
            "csv_files": [{"name": "a.csv", "role": "intended", "content": "v", "column_total": 1}],
            "scorer_py": None, "project_note": None, "user_facts": "{}"}  # fmt: skip
    row = {"episode_id": "e", "model": "m", "model_family": "claude", "harness": "native",
           "agent": "claude-code", "setting": "S", "condition": "c", "seed": 1, "path": "p",
           "asked_user_before_edit": False, "scorer_used_to_edit_csv": None,
           "verbalized_test_awareness": False, "task": task, "trace": "[]",
           "csv_changed": False}  # fmt: skip
    (tmp_path / "data").mkdir()
    table = pa.Table.from_pylist([row], schema=build.written(build.SCHEMA))
    pq.write_table(table, tmp_path / "data" / "train-0.parquet")
    (tmp_path / "README.md").write_text("card")
    with pytest.raises(SystemExit) as stopped:
        cli.main(["dataset", "check", *check_argv(tmp_path)])
    assert stopped.value.code == 1
    report = json.loads((tmp_path / "check_report.json").read_text())
    found = report["every trace is a valid ATIF-v1.8 trajectory that reads back"]
    assert found["pass"] is False and found["detail"][0].startswith("e:")


def test_a_credential_in_a_trajectorys_views_stops_the_build(tmp_path):
    """The trajectory's views (an observation's text, which Harbor makes from a tool's raw
    output; a question's rendered options and answer) hold a credential JSON-escaped; the guard
    stops on each, as on the whole document."""
    from ping_or_peek.dataset import build

    secret = 'pa"ss\\word-123456'
    history = claude_history(answer=TARGET)
    for message in history:
        update_ = (message.get("params") or {}).get("update") or {}
        if update_.get("toolCallId") == "t2" and update_.get("status") == "completed":
            update_["rawOutput"] = {"stdout": secret, "exit_code": 0}
        if message.get("method") == "elicitation/create":
            message["params"]["requestedSchema"]["properties"]["question_0"]["oneOf"][0][
                "title"
            ] = secret
    raw = parsed(tmp_path, history)
    document = written(raw)
    content = next(
        r["content"]
        for s in document["steps"]
        for r in (s.get("observation") or {}).get("results", [])
        if r["source_call_id"] == "t2"
    )
    assert json.dumps(secret)[1:-1] in content and secret not in content  # escaped only
    guard = build.Guard({"PORTKEY_API_KEY": secret})
    for published in ({"content": content}, {"trace": document}):
        with pytest.raises(SystemExit) as stopped:
            guard("e1", published)
        assert secret not in str(stopped.value)


def test_atifs_step_sources_are_harbors():
    from typing import get_args

    from harbor.models.trajectories import Step

    assert set(atif.SOURCES) == set(get_args(Step.model_fields["source"].annotation))


def test_a_view_value_outside_its_vocabulary_stops_the_build(tmp_path):
    """An approval option of a kind ACP doesn't define would pass through as the decision."""
    history = claude_history()
    for message in history:
        for option in (message.get("params") or {}).get("options", []):
            option["kind"] = "maybe_once"
    session = parsed(tmp_path, history)
    with pytest.raises(ValueError, match="decision.*build.VIEW_VOCABULARY"):
        trajectory.approval_view(session.approvals[0])
    with pytest.raises(ValueError, match="build.VOCABULARY"):  # the build path refuses it first
        written(session)


CREDENTIAL_LIKE = "sk-synthetic-credential-0123456789"  # stands for any recorded value


def test_a_view_refusal_names_the_field_never_the_value(tmp_path):
    history = claude_history()
    for message in history:
        for option in (message.get("params") or {}).get("options", []):
            option["kind"] = CREDENTIAL_LIKE
    with pytest.raises(ValueError) as refused:
        trajectory.approval_view(parsed(tmp_path, history).approvals[0])
    assert CREDENTIAL_LIKE not in str(refused.value) and "decision" in str(refused.value)


def test_a_record_of_an_unknown_kind_is_refused_without_its_kind(tmp_path):
    document = written(parsed(tmp_path, claude_history()))
    next(record for _, record in atif.records(document))["kind"] = CREDENTIAL_LIKE
    with pytest.raises(ValueError) as refused:
        atif.from_trajectory(document)
    assert CREDENTIAL_LIKE not in str(refused.value)


def test_an_invalid_trajectory_is_refused_without_its_values(tmp_path):
    raw = parsed(tmp_path, claude_history())
    document = written(raw)
    document["steps"][0]["message"] = {"key": CREDENTIAL_LIKE}  # not a string: invalid ATIF
    with pytest.raises(ValueError) as refused:
        trajectory.check_reads_back(document, raw, [], VERIFIER)
    assert CREDENTIAL_LIKE not in str(refused.value)
    assert refused.value.__cause__ is None and refused.value.__suppress_context__


def checked(tmp_path, trace):
    """The dataset check's report on a one-episode dataset whose trace is `trace`, as text."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    task = {"opener": "o", "intended_file": "a.csv", "checker_target_file": None,
            "csv_files": [{"name": "a.csv", "role": "intended", "content": "v", "column_total": 1}],
            "scorer_py": None, "project_note": None, "user_facts": "{}"}  # fmt: skip
    row = {"episode_id": "e", "model": "m", "model_family": "claude", "harness": "native",
           "agent": "claude-code", "setting": "S", "condition": "c", "seed": 1, "path": "p",
           "asked_user_before_edit": False, "scorer_used_to_edit_csv": None,
           "verbalized_test_awareness": False, "task": task, "trace": trace,
           "csv_changed": False}  # fmt: skip
    (tmp_path / "data").mkdir()
    table = pa.Table.from_pylist([row], schema=build.written(build.SCHEMA))
    pq.write_table(table, tmp_path / "data" / "train-0.parquet")
    (tmp_path / "README.md").write_text("card")
    with pytest.raises(SystemExit):
        cli.main(["dataset", "check", *check_argv(tmp_path)])
    return (tmp_path / "check_report.json").read_text()


def test_the_checks_report_holds_no_value_of_an_invalid_trace(tmp_path):
    trace = json.dumps({"schema_version": "ATIF-v1.8", "session_id": "e", "steps": CREDENTIAL_LIKE})
    report = checked(tmp_path, trace)
    assert CREDENTIAL_LIKE[-10:] not in report  # pydantic elides a long value's middle
    found = json.loads(report)["every trace is a valid ATIF-v1.8 trajectory that reads back"]
    assert found["pass"] is False and found["detail"][0].startswith("e:")


def test_a_closed_value_outside_the_registry_stops_the_trajectory(tmp_path):
    """The ATIF build path checks every closed value against `build.VOCABULARY`."""
    history = claude_history()
    for message in history:
        if (message.get("result") or {}).get("stopReason"):
            message["result"]["stopReason"] = CREDENTIAL_LIKE
    with pytest.raises(ValueError) as refused:
        written(parsed(tmp_path, history))
    assert "build.VOCABULARY" in str(refused.value)
    assert CREDENTIAL_LIKE not in str(refused.value)


def test_harbors_validation_is_refused_by_field_path(tmp_path):
    with pytest.raises(ValueError) as refused:
        trajectory.to_trajectory(
            parsed(tmp_path, claude_history()), [], episode_id="e1", agent="claude-code",
            agent_version={"k": CREDENTIAL_LIKE}, model="claude-opus-5-5", user_mode="available",
            verifier=VERIFIER,
        )  # fmt: skip
    assert str(refused.value).startswith("e1: ")
    assert CREDENTIAL_LIKE[-10:] not in str(refused.value)
    assert refused.value.__suppress_context__


@pytest.mark.parametrize("field", ["request", "position"])
def test_a_records_recorded_reference_is_refused_without_its_value(tmp_path, field):
    document = written(parsed(tmp_path, claude_history()))
    target = next(r for _, r in atif.records(document) if r["kind"] == acp.QUESTION_REPLY)
    if field == "request":
        target["data"]["request"] = CREDENTIAL_LIKE
    else:
        target["position"] = CREDENTIAL_LIKE
    with pytest.raises(ValueError) as refused:
        atif.from_trajectory(document)
    assert CREDENTIAL_LIKE not in str(refused.value)


def test_a_malformed_record_is_refused_naming_the_rule(tmp_path):
    document = written(parsed(tmp_path, claude_history()))
    del next(r for _, r in atif.records(document))["timestamp"]
    with pytest.raises(ValueError, match="atif"):
        atif.from_trajectory(document)


def test_the_parser_and_the_reader_derive_stop_reasons_alike(tmp_path, monkeypatch):
    monkeypatch.setattr(acp, "stop_reason", lambda data: "derived")
    read = atif.from_trajectory(written(parsed(tmp_path, claude_history())))
    assert set(read.session.stop_reasons) == {"derived"}


def without_a_tool_call_id(tmp_path):
    """A Claude Code trajectory whose first tool_call_update record lost its toolCallId."""
    document = written(parsed(tmp_path, claude_history()))
    record = next(r for _, r in atif.records(document) if r["kind"] == atif.TOOL_UPDATE)
    del record["data"]["update"]["toolCallId"]
    return document


def test_a_tool_update_without_its_id_is_refused_naming_the_record_and_the_rule(tmp_path):
    # A published trace isn't a trial's record: atif refuses it with its own ValueError.
    with pytest.raises(ValueError, match=r"record \d+ of the trace.*toolCallId"):
        atif.from_trajectory(without_a_tool_call_id(tmp_path))


def test_the_check_reports_a_tool_update_without_its_id(tmp_path):
    trace = json.dumps(without_a_tool_call_id(tmp_path))
    (tmp_path / "dataset").mkdir()
    report = checked(tmp_path / "dataset", trace)
    found = json.loads(report)["every trace is a valid ATIF-v1.8 trajectory that reads back"]
    assert found["pass"] is False and "toolCallId" in found["detail"][0]
