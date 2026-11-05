"""Labeling, export, parsing and dataset fetching never substitute a default for missing or
unknown input: each case fails with a message naming the rule, or needs an explicit choice."""

import json
import random
from pathlib import Path

import pytest

from ping_or_peek import cli
from ping_or_peek.config import CONDITIONS, HARNESSES, MODELS, SETTINGS
from ping_or_peek.dataset import acp, export, trial
from ping_or_peek.dataset.user_agent import PERSONA_RULES
from ping_or_peek.labeling import label_run, labels
from tests.dataset.test_analysis_export import fake_trial

CELL = (HARNESSES[0], MODELS[0], SETTINGS[0], CONDITIONS[1], 100)


def _accepted(**changes):
    """One behavioral trial record as `label` writes it, with `changes` applied."""
    record = fake_trial(*CELL, random.Random(0))
    record.update(changes)
    return record


def test_a_trial_missing_from_the_sensitivity_pass_fails_naming_it():
    record = _accepted()
    with pytest.raises(ValueError, match=record["trial"]):
        export.build_rows([record], {record["trial"]: CELL[2]}, {}, sensitivity={})


def test_an_unknown_asking_rule_fails_naming_it():
    with pytest.raises(ValueError, match="agnt"):
        labels.episode_stage("reactive_v2", "agnt")


def test_an_unknown_reasoning_setting_fails_naming_it(tmp_path):
    with pytest.raises(ValueError, match="includ"):
        label_run.label_trial(
            tmp_path / "trial-1", root=tmp_path, annotator=None, persona={}, setting=None,
            reasoning="includ",
        )  # fmt: skip


def test_the_reasoning_cli_needs_an_explicit_superseded_choice_and_the_file(tmp_path, capsys):

    argv = ["code", str(tmp_path), "--run", "r", "--jobs-root", str(tmp_path)]
    with pytest.raises(SystemExit):
        cli.main(argv)
    assert "--superseded" in capsys.readouterr().err
    with pytest.raises(FileNotFoundError, match="absent.json"):
        cli.main([*argv, "--superseded", str(tmp_path / "absent.json")])


def test_the_reasoning_coder_needs_its_private_root(tmp_path, monkeypatch):

    monkeypatch.delenv("WS6_PRIVATE_ROOT", raising=False)
    _finished_trial(tmp_path)
    jobs = str(tmp_path / "jobs")
    argv = ["code", jobs, "--run", "r", "--jobs-root", jobs, "--no-superseded"]
    with pytest.raises(SystemExit, match="WS6_PRIVATE_ROOT"):
        cli.main(argv)


def test_the_reasoning_cli_refuses_a_trial_outside_its_jobs_root(tmp_path):

    outside = tmp_path / "elsewhere" / "job" / "trial-1"
    (outside / "agent").mkdir(parents=True)
    (outside / "result.json").write_text("{}")
    (outside / "agent" / "bridge-trajectory.json").write_text("{}")
    root = tmp_path / "jobs"
    root.mkdir()
    argv = ["code", str(outside), "--run", "r", "--jobs-root", str(root)]
    with pytest.raises(ValueError, match="trial-1"):
        cli.main([*argv, "--no-superseded"])


def test_a_session_that_records_no_working_directory_fails():
    session = acp.parse_history([])
    with pytest.raises(acp.RecordError, match="cwd"):
        _ = session.cwd


def test_a_tool_call_update_without_an_id_fails():
    update = {"sessionUpdate": "tool_call", "title": "ls", "status": "pending"}
    message = {"jsonrpc": "2.0", "method": "session/update", "params": {"update": update}}
    with pytest.raises(acp.RecordError, match="toolCallId"):
        acp.parse_history([{"message": message, "direction": "inbound", "ts": None}])


@pytest.mark.parametrize("attribute", ["agent", "model", "task_name"])
def test_a_trial_without_its_agent_model_or_task_fails_naming_it(tmp_path, attribute):
    with pytest.raises(acp.RecordError, match="trial-1"):
        getattr(trial.Trial(tmp_path / "trial-1", {}, {}), attribute)


def test_a_trial_whose_task_names_no_condition_fails_naming_it(tmp_path):
    folder = tmp_path / "job" / "trial-1"
    (folder / "agent").mkdir(parents=True)
    (folder / "result.json").write_text(json.dumps({"trial_name": "trial-1", "task_name": "x"}))
    with pytest.raises(acp.RecordError, match="trial-1"):
        trial.load_trial(folder)


def test_a_trial_without_the_simulated_users_record_is_an_infrastructure_attempt():
    record = _accepted()
    record["measures"]["user_agent_record"] = False
    assert export.infrastructure_failure(record, {}) is not None


def test_a_superseded_list_without_its_jobs_fails_naming_the_file(tmp_path):
    listing = tmp_path / "superseded.json"
    listing.write_text(json.dumps({"note": "no jobs key"}))
    with pytest.raises(ValueError, match="superseded.json"):
        export.read_superseded(listing)


def test_an_accepted_trial_without_timing_fails_naming_it():
    record = _accepted(timing=None)
    with pytest.raises(ValueError, match=record["trial"]):
        export.build_rows([record], {record["trial"]: CELL[2]}, {})


def test_a_trial_record_without_its_job_fails_naming_it():
    record = _accepted(provenance={})
    with pytest.raises(ValueError, match=record["trial"]):
        export.build_rows([record], {record["trial"]: CELL[2]}, {})


def test_a_retry_log_entry_without_its_failure_fails_naming_the_trial(tmp_path):
    log = tmp_path / "infra_retries.jsonl"
    log.write_text(json.dumps({"trial": "trial-7"}) + "\n")
    with pytest.raises(ValueError, match="trial-7"):
        export.read_retry_log(log)


def test_evidence_citing_an_item_the_episode_lacks_fails_naming_it():
    answers = {"episode": {"evidence": [{"label": "asked_user", "item_id": "e99", "quote": "q"}]}}
    with pytest.raises(ValueError, match="e99"):
        label_run.public_evidence(answers, [{"id": "e1", "kind": "agent_message"}])


def test_a_failed_trial_without_its_exception_type_fails_naming_it(tmp_path):
    folder = tmp_path / "job" / "trial-1"
    (folder / "agent").mkdir(parents=True)
    result = {
        "trial_name": "trial-1",
        "task_name": "user_available_checker_exposed__seed100",
        "exception_info": {"exception_message": "the bridge timed out"},
    }
    (folder / "result.json").write_text(json.dumps(result))
    with pytest.raises(acp.RecordError, match="trial-1"):
        label_run.infrastructure_record(folder)


def test_the_label_store_root_has_no_default(tmp_path, monkeypatch, capsys):

    monkeypatch.delenv("ASK_OR_SHORTCUT_LABEL_ROOT", raising=False)
    argv = ["label", str(tmp_path), "--run", "r", "--out", str(tmp_path / "out.json")]
    with pytest.raises(SystemExit):
        cli.main([*argv, "--no-retry-log"])
    assert "ASK_OR_SHORTCUT_LABEL_ROOT" in capsys.readouterr().err


def test_the_reasoning_cli_keys_a_relative_trial_path_under_an_absolute_jobs_root(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from ping_or_peek.labeling.reasoning import code_command as reasoning_main

    folder = tmp_path / "jobs" / "job" / "trial-1"
    (folder / "agent").mkdir(parents=True)
    (folder / "result.json").write_text("{}")
    (folder / "agent" / "bridge-trajectory.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    root = (tmp_path / "jobs").resolve()
    found = reasoning_main._trials(["jobs/job/trial-1"], None, root, set())
    assert [t.relative_to(root) for t in found] == [Path("job/trial-1")]


def _bare_trial(tmp_path, task_name):
    folder = tmp_path / "job" / "t"
    (folder / "agent").mkdir(parents=True)
    (folder / "result.json").write_text(json.dumps({"trial_name": "t", "task_name": task_name}))
    return folder


def test_check_record_reports_a_record_with_no_model_agent_or_condition(tmp_path):
    from ping_or_peek.dataset.completeness import check_record

    checks = check_record(_bare_trial(tmp_path, "user_available_checker_exposed__seed100"))[
        "checks"
    ]
    assert checks["result.json: model"]["ok"] is False
    assert checks["result.json: agent"]["ok"] is False
    other = check_record(_bare_trial(tmp_path / "other", "a task with no condition"))["checks"]
    assert other["task: condition and seed"]["ok"] is False


def test_a_failed_reasoning_coding_exits_nonzero_naming_it(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ping_or_peek.labeling.reasoning import code_command as reasoning_main

    monkeypatch.setenv("WS6_PRIVATE_ROOT", str(tmp_path / "private"))
    trial = tmp_path / "jobs" / "job" / "trial-1"
    monkeypatch.setattr(reasoning_main, "_trials", lambda *args: [trial])
    monkeypatch.setattr(
        reasoning_main.transcript,
        "from_trial",
        lambda *a, **k: SimpleNamespace(episode="job/trial-1"),
    )
    monkeypatch.setattr(reasoning_main.label, "make_client", lambda config: None)
    monkeypatch.setattr(reasoning_main.label, "code_episode", lambda *a, **k: {"status": "failed"})
    argv = [
        "code",
        str(trial),
        "--run",
        "r",
        "--jobs-root",
        str(tmp_path / "jobs"),
        "--no-superseded",
    ]
    with pytest.raises(SystemExit) as exit:
        cli.main(argv)
    assert exit.value.code not in (0, None) and "job/trial-1" in str(exit.value.code)


def test_check_record_reports_an_export_that_breaks_acp(tmp_path):
    from ping_or_peek.dataset.completeness import check_record

    folder = _bare_trial(tmp_path, "user_available_checker_exposed__seed100")
    update = {"sessionUpdate": "tool_call", "title": "ls", "status": "pending"}
    message = {"jsonrpc": "2.0", "method": "session/update", "params": {"update": update}}
    history = [{"message": message, "direction": "inbound", "ts": None}]
    (folder / "agent" / "bridge-trajectory.json").write_text(json.dumps({"history": history}))
    checks = check_record(folder)["checks"]
    assert checks["acp export: follows ACP"]["ok"] is False
    assert "toolCallId" in checks["acp export: follows ACP"]["detail"]
    assert "verifier: metrics" in checks  # the rest of the record is still checked


def test_label_trial_fails_naming_an_unknown_asking_rule(tmp_path):
    with pytest.raises(ValueError, match="agnt"):
        label_run.label_trial(
            tmp_path / "trial-1", root=tmp_path, annotator=None, persona={}, setting=None,
            asking_rule="agnt",
        )  # fmt: skip


def test_a_measured_trial_whose_task_names_no_seed_fails_naming_it(tmp_path):
    folder = _bare_trial(tmp_path, "user_available_checker_exposed")
    with pytest.raises(acp.RecordError, match="seed"):
        trial.load_trial(folder)


def test_the_conversational_control_has_no_seed(tmp_path):
    assert trial.load_trial(_bare_trial(tmp_path, trial.CONTROL)).task_metadata["seed"] is None


def _finished_trial(root):
    folder = root / "jobs" / "collection__codex__S" / "trial-1"
    (folder / "agent").mkdir(parents=True)
    (folder / "result.json").write_text("{}")
    (folder / "agent" / "bridge-trajectory.json").write_text("{}")
    return folder


def test_a_trial_reached_by_two_spellings_of_its_path_counts_once(tmp_path, monkeypatch):
    from ping_or_peek.labeling.reasoning import code_command as reasoning_main

    folder = _finished_trial(tmp_path)
    monkeypatch.chdir(tmp_path)
    spellings = ["jobs/collection__codex__S/trial-1", str(folder), "jobs"]
    assert trial.find_trials(spellings) == [folder.resolve()]
    root = (tmp_path / "jobs").resolve()
    assert reasoning_main._trials(spellings, None, root, set()) == [folder.resolve()]


def test_check_record_reports_a_trial_reached_by_two_spellings_once(tmp_path, monkeypatch, capsys):

    folder = _finished_trial(tmp_path)
    monkeypatch.chdir(tmp_path)
    cli.main(["check-record", "jobs/collection__codex__S/trial-1", str(folder)])
    assert capsys.readouterr().out.count('"trial":') == 1


@pytest.mark.parametrize("path", ["missing", "empty", "missing__*"])
def test_a_path_that_selects_no_trial_fails_naming_it(tmp_path, path):
    _finished_trial(tmp_path)
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match=path.replace("*", r"\*")):
        trial.find_trials([tmp_path / "jobs", tmp_path / path])


def test_waiting_for_trials_is_an_explicit_choice(tmp_path):
    (tmp_path / "running").mkdir()  # a job whose trials haven't finished
    assert trial.find_trials([tmp_path / "running", tmp_path / "later__*"], waiting=True) == []


def test_check_record_fails_naming_a_path_that_selects_no_trial(tmp_path):

    with pytest.raises(ValueError, match="missing"):
        cli.main(["check-record", str(tmp_path / "missing")])


def _label_argv(tmp_path, *paths):
    return [
        "label", *map(str, paths), "--run", "r", "--out", str(tmp_path / "out.json"),
        "--label-root", str(tmp_path / "labels"), "--no-retry-log",
    ]  # fmt: skip


def test_label_watch_waits_for_jobs_that_have_no_finished_trial_yet(tmp_path, monkeypatch):
    from ping_or_peek.labeling import label_command as labeling_main

    class Scanned(Exception):
        pass

    def sleep(seconds):
        raise Scanned

    monkeypatch.setenv("PORTKEY_API_KEY", "dummy")
    monkeypatch.setattr(label_run, "label_trials", lambda *a, **k: ([], []))
    monkeypatch.setattr(labeling_main.time, "sleep", sleep)
    with pytest.raises(Scanned):  # it scanned once and went on waiting
        cli.main([*_label_argv(tmp_path, tmp_path / "later__*"), "--watch", "60"])


class _Scanned(Exception):
    """Raised by a stubbed sleep: `label --watch` finished a scan and would wait."""


def _scan_once(monkeypatch):
    from ping_or_peek.labeling import label_command as labeling_main

    def sleep(seconds):
        raise _Scanned

    monkeypatch.setenv("PORTKEY_API_KEY", "dummy")
    monkeypatch.setattr(label_run, "label_trials", lambda *a, **k: ([], []))
    monkeypatch.setattr(labeling_main.time, "sleep", sleep)
    return labeling_main


def test_label_watch_0_scans_once_so_it_does_not_wait(tmp_path, monkeypatch):
    _scan_once(monkeypatch)
    with pytest.raises(ValueError, match="missing"):
        cli.main([*_label_argv(tmp_path, tmp_path / "missing"), "--watch", "0"])


def _trial_of_model(tmp_path, name):
    folder = _bare_trial(tmp_path, "user_available_checker_exposed__seed100")
    result = json.loads((folder / "result.json").read_text())
    result["agent_info"] = {"model_info": {"name": name}}
    (folder / "result.json").write_text(json.dumps(result))
    return folder


@pytest.mark.parametrize("name", ["openai/", "gpt-4o"])
def test_a_model_outside_the_vocabulary_fails_naming_the_trial(tmp_path, name):
    from ping_or_peek.dataset.completeness import check_record

    folder = _trial_of_model(tmp_path, name)
    with pytest.raises(acp.RecordError, match="t:.*model"):
        _ = trial.load_trial(folder).model
    assert check_record(folder)["checks"]["result.json: model"]["ok"] is False


def test_a_model_named_with_its_provider_is_the_model(tmp_path):
    assert trial.load_trial(_trial_of_model(tmp_path, "openai/gpt-6.1-sol")).model == "gpt-6.1-sol"


def test_a_user_agent_trajectory_without_steps_fails_naming_the_trial(tmp_path):
    from ping_or_peek.dataset.completeness import check_record
    from ping_or_peek.dataset.user_agent import persona_checks

    folder = _bare_trial(tmp_path, "user_available_checker_exposed__seed100")
    (folder / "user-agent").mkdir()
    (folder / "user-agent" / "trajectory.json").write_text("{}")
    with pytest.raises(acp.RecordError, match="t:.*steps"):
        persona_checks(folder / "user-agent")
    check = check_record(folder)["checks"]["user agent record: its steps"]
    assert check["ok"] is False and "steps" in check["detail"]


def test_check_record_reports_a_trial_without_the_simulated_users_trajectory(tmp_path):
    # The export lists such a trial as an infrastructure attempt (user_agent_record is False);
    # check-record reports it by the same definition, though other user-agent files exist.
    from ping_or_peek.dataset.completeness import check_record

    folder = _bare_trial(tmp_path, "user_available_checker_exposed__seed100")
    (folder / "user-agent").mkdir()
    (folder / "user-agent" / "stdout.txt").write_text("")
    report = check_record(folder)
    record = report["checks"]["user agent record"]
    assert record["ok"] is False and "infrastructure attempt" in record["detail"]
    assert "user agent record" in report["missing"]
    assert report["checks"]["user agent record: its steps"]["ok"] is None  # no steps to check


def _user_agent(tmp_path, trajectory):
    """A trial whose simulated user left `trajectory` (text, or None for no file)."""
    folder = _bare_trial(tmp_path, "user_available_checker_exposed__seed100")
    (folder / "user-agent").mkdir()
    if trajectory is not None:
        (folder / "user-agent" / "trajectory.json").write_text(trajectory)
    return folder


NO_COMMANDS = ['{"steps": []}', '{"steps": [{"message": "hello"}]}']


@pytest.mark.parametrize("trajectory", [*NO_COMMANDS, None])
def test_persona_rules_with_nothing_to_check_are_none_not_passed(tmp_path, trajectory):
    # True: a violation found. False: checked and clean. None: nothing to check.
    from ping_or_peek.dataset.user_agent import persona_checks

    checks = persona_checks(_user_agent(tmp_path, trajectory) / "user-agent")
    assert [checks[rule] for rule in PERSONA_RULES] == [None] * 4


@pytest.mark.parametrize("trajectory", ['{"steps": {}}', '{"steps": null}', "not json"])
def test_a_malformed_user_agent_trajectory_fails_naming_the_trial(tmp_path, trajectory):
    from ping_or_peek.dataset.user_agent import persona_checks

    with pytest.raises(acp.RecordError, match="t:"):
        persona_checks(_user_agent(tmp_path, trajectory) / "user-agent")


def test_the_export_takes_unchecked_persona_rules_for_an_infrastructure_attempt():
    record = _accepted()
    record["measures"].update(dict.fromkeys(PERSONA_RULES))
    assert "not audited" in export.infrastructure_failure(record, {})


@pytest.mark.parametrize("trajectory", [NO_COMMANDS[0], "not json"])
def test_check_record_reports_persona_rules_it_could_not_check(tmp_path, trajectory):
    from ping_or_peek.dataset.completeness import check_record

    check = check_record(_user_agent(tmp_path, trajectory))["checks"][
        "user agent record: its steps"
    ]
    assert check["ok"] is False


def test_an_explicit_empty_sensitivity_fails_naming_the_option(tmp_path, monkeypatch, capsys):
    from ping_or_peek.dataset import export_command as dataset_main

    monkeypatch.setattr(dataset_main, "_export", lambda *args: 0)
    argv = ["export", str(tmp_path / "results.json"), "--out-dir", str(tmp_path), "--no-retry-log"]
    with pytest.raises(SystemExit):
        cli.main([*argv, "--no-superseded", "--sensitivity"])
    assert "--sensitivity" in capsys.readouterr().err


ROLLOUT = "agent/sessions/2026/10/13/rollout-1.jsonl"


def _read_metrics(folder):
    return trial.load_trial(folder).verifier_metrics()


def _read_export(folder):
    return acp.parse_export(folder / "agent" / "bridge-trajectory.json")


def _read_user_agent(folder):
    from ping_or_peek.runs.checks import user_agent_commands

    return user_agent_commands(folder)


def _read_rollout(folder):
    from ping_or_peek.dataset import codex_rollout

    return codex_rollout.read_rollout(folder / ROLLOUT, folder.name)


def _read_native_model(folder):
    from ping_or_peek.runs.checks import native_model

    return native_model(folder)


def _read_retry(folder):
    from ping_or_peek.runs.retry import failed_trials

    config = folder / "config.json"
    if not config.exists():  # every trial ping_or_peek.runs.run launches has one, which retry reads
        task = {"path": "tasks/plain/user_available_checker_exposed__seed100"}
        config.write_text(json.dumps({"agent": {"model_name": "claude-opus-5-5"}, "task": task}))
    return failed_trials(folder.parent.parent, "s0")


def _read_modes(folder):
    from ping_or_peek.runs.checks import claude_modes

    return claude_modes(folder)


# Each reader of a trial's record files, and a malformed file it reads.
MALFORMED = [
    ("result.json", "not json", trial.load_trial),
    ("result.json", "[]", trial.load_trial),
    ("result.json", "not json", _read_retry),
    ("config.json", "not json", trial.load_trial),
    ("config.json", "not json", _read_retry),
    ("agent/bridge-trajectory.json", "not json", _read_export),
    ("agent/bridge-trajectory.json", '{"history": {}}', _read_export),
    ("agent/bridge-trajectory.json", '{"history": [null]}', _read_export),
    ("agent/bridge-trajectory.json", "not json", _read_modes),
    ("verifier/metrics.json", "not json", _read_metrics),
    ("verifier/metrics.json", "[]", _read_metrics),
    ("verifier/reward.txt", "", _read_metrics),
    ("verifier/reward.txt", "passed", _read_metrics),
    ("lock.json", "not json", lambda folder: trial.persona_text(trial.load_trial(folder))),
    ("user-agent/trajectory.json", "not json", _read_user_agent),
    ("user-agent/trajectory.json", '{"steps": {}}', _read_user_agent),
    ("user-agent/trajectory.json", '{"steps": [null]}', _read_user_agent),
    ("user-agent/trajectory.json", '{"steps": [{"tool_calls": [null]}]}', _read_user_agent),
    ("user-agent/trajectory.json", '{"steps": [{"tool_calls": [{"arguments": "ls"}]}]}',
     _read_user_agent),
    (ROLLOUT, "not json", _read_rollout),
    (ROLLOUT, "not json", _read_native_model),
    # Not UTF-8, as from a process killed in the middle of a character.
    ("result.json", b'{"task_name": "\xff"}', trial.load_trial),
    ("agent/bridge-trajectory.json", b"\xff", _read_export),
    ("user-agent/trajectory.json", b"\xff", _read_user_agent),
    (ROLLOUT, b"\xff", _read_rollout),
]  # fmt: skip
# Files check-record doesn't read: the persona lock (`trial.persona_text` checks it) and the
# recorded requests.
NOT_IN_CHECK_RECORD = {"lock.json"}


def _malformed(tmp_path, name, content):
    """A trial whose lock names this repository's persona, so every check reaches its readers,
    with `name` holding `content`."""
    import hashlib

    from tests.labeling.test_analysis_pipeline import make_trial, task_history

    folder = make_trial(tmp_path, task_history())
    digest = hashlib.sha256((trial.REPO / "persona" / "persona.md").read_bytes()).hexdigest()
    lock = {"user_persona": {"path": "persona/persona.md", "digest": f"sha256:{digest}"}}
    (folder / "lock.json").write_text(json.dumps(lock))
    (folder / name).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        (folder / name).write_bytes(content)
    else:
        (folder / name).write_text(content)
    if name.endswith(".status"):  # a recorded request's status sits beside the request
        (folder / name).with_suffix(".json").write_text(json.dumps({"body": {}}))
    return folder


@pytest.mark.parametrize(("name", "content", "read"), MALFORMED)
def test_each_reader_fails_naming_the_trial_and_file_it_cannot_read(tmp_path, name, content, read):
    folder = _malformed(tmp_path, name, content)
    with pytest.raises(acp.RecordError) as raised:
        read(folder)
    assert "trial-1" in str(raised.value) and Path(name).name in str(raised.value)


@pytest.mark.parametrize(("name", "content"), list(dict.fromkeys((n, c) for n, c, _ in MALFORMED)))
def test_the_checks_name_a_record_file_they_cannot_read(tmp_path, name, content):
    from ping_or_peek.dataset.completeness import check_record
    from ping_or_peek.runs.checks import check_trial

    folder = _malformed(tmp_path, name, content)
    with pytest.raises(acp.RecordError) as raised:
        check_trial(folder)
    assert "trial-1" in str(raised.value) and Path(name).name in str(raised.value)
    report = check_record(folder)  # reports, never raises
    if name not in NOT_IN_CHECK_RECORD:
        assert any(Path(name).name in str(c["detail"]) for c in report["checks"].values())


# A trial's record: its files, and the folders that hold the rest. Only the record readers
# (src/ping_or_peek/dataset/trial.py, acp.parse_export) open them; everything else goes through those.
RECORD_FILES = (
    "result.json", "config.json", "lock.json", "bridge-trajectory.json", "trajectory.json",
    "metrics.json", "reward.json", "reward.txt", ".jsonl",
)  # fmt: skip
RECORD_FOLDERS = ("agent", "user-agent", "verifier")
RECORD_PATHS = ("acp_export", "native_session_files", "user_agent_files")  # Trial's own
READS = ("read_text", "read_bytes", "open", "load")  # a file opened, not text already read
READERS = {"src/ping_or_peek/dataset/trial.py", "src/ping_or_peek/dataset/acp.py"}
NOT_A_TRIALS_RECORD = {
    "src/ping_or_peek/labeling/reasoning/label.py:code_episode": "the coder's own result.json, in its run folder",
    "src/ping_or_peek/labeling/reasoning/label.py:saved_coding": "the coder's own result.json, in its run folder",
    "src/ping_or_peek/runs/run.py:results": "counts a job's result.json files and reads the job's own config",
    "src/ping_or_peek/runs/progress.py:run": "finds trials by their result.json and reads its own cache",
}


def _reaches_a_record(node) -> bool:
    import ast

    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr in RECORD_PATHS:
            return True
        if (
            isinstance(n, ast.BinOp)
            and isinstance(n.op, ast.Div)
            and isinstance(n.right, ast.Constant)
            and n.right.value in (*RECORD_FILES, *RECORD_FOLDERS)
        ):
            return True
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) in ("glob", "rglob"):
            text = ast.unparse(n)
            if any(name in text for name in RECORD_FILES):
                return True
    return False


def _opens_a_file(node) -> bool:
    import ast

    return any(
        isinstance(n, ast.Call)
        and (getattr(n.func, "attr", None) or getattr(n.func, "id", None)) in READS
        for n in ast.walk(node)
    )


def test_only_the_record_readers_open_a_trials_record_files():
    """A function outside the readers that reaches a trial's record files and opens a file itself
    would read them without the readers' checks; each must go through the readers instead."""
    import ast

    root = trial.REPO
    found = []
    for path in sorted((root / "src" / "ping_or_peek").rglob("*.py")):
        module = str(path.relative_to(root))
        if module in READERS:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{module}:{node.name}"
                skip = name in NOT_A_TRIALS_RECORD
                if not skip and _reaches_a_record(node) and _opens_a_file(node):
                    found.append(name)
    assert found == []
