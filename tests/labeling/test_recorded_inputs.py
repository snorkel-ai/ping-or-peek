"""Labeling and export use only the input that was recorded: the label store and the reasoning
coder reuse a saved answer only for the identical request, the export's retry log and superseded
list must exist, and a trial's persona is found by the digest Harbor locked."""

import hashlib
import json
import subprocess

import pytest

from ping_or_peek.config import MODELS, PRESENT_HIDDEN, SETTINGS
from ping_or_peek.dataset import export, trial
from ping_or_peek.labeling import annotator, labels
from ping_or_peek.labeling.annotator import AnnotatorConfig, write_json
from ping_or_peek.labeling.reasoning import label as coder
from ping_or_peek.labeling.reasoning import transcript as coder_transcript

CONFIG = AnnotatorConfig(model="@openai/gpt-6.1-sol")
ANSWER = {
    "scorer_used_to_edit_csv": False,
    "scorer_edit_evidence": "none",
    "edited_paths": [],
    "evidence": [],
    "rationale": "r",
    "alternative_explanations": "a",
}
PAYLOAD = {
    "prompt": "p",
    "checker_target": "b.csv",
    "filesystem": {},
    "observations": {"write_events": []},
    "items": [],
    "persona_facts": {},
}


class NoCalls:
    """An annotator that may only reuse saved answers."""

    config = CONFIG

    def label(self, **_):
        raise AssertionError("the annotator was called")


def _saved(root, payload, *, attempts=True):
    """A labeler folder holding a saved scorer-edit answer for `payload`, recorded as sent."""
    stage = labels.SCORER_EDIT
    full = {
        **payload,
        "version": stage.version,
        "rubric_sha256": labels.rubric_sha256(stage.rubric_name),
    }
    write_json(
        root / stage.name / "records" / "trial-1.json",
        {"input_sha256": labels.fingerprint(full), "annotator": CONFIG.public(), "value": ANSWER},
    )
    if attempts:
        messages = annotator.request_messages(stage.rubric, full)
        write_json(
            root / stage.name / "attempts" / "trial-1" / "1.json",
            {"request_messages": messages, "value": ANSWER},
        )


def test_a_saved_answer_is_reused_for_the_identical_request(tmp_path):
    root = tmp_path / "run" / "gpt-6.1-sol-medium"
    _saved(root, PAYLOAD)
    assert labels.run_stage(labels.SCORER_EDIT, "trial-1", PAYLOAD, root, NoCalls()) == ANSWER


def test_a_saved_answer_is_not_reused_for_a_reordered_request(tmp_path):
    # The same values in another key order: the sorted-key fingerprint matches, the request
    # the labeler would send does not.
    root = tmp_path / "run" / "gpt-6.1-sol-medium"
    _saved(root, PAYLOAD)
    reordered = dict(reversed(list(PAYLOAD.items())))
    with pytest.raises(ValueError, match="scorer_edit.*trial-1"):
        labels.run_stage(labels.SCORER_EDIT, "trial-1", reordered, root, NoCalls())


def test_a_saved_answer_without_its_recorded_request_is_not_reused(tmp_path):
    root = tmp_path / "run" / "gpt-6.1-sol-medium"
    _saved(root, PAYLOAD, attempts=False)
    with pytest.raises(ValueError, match="scorer_edit.*trial-1"):
        labels.run_stage(labels.SCORER_EDIT, "trial-1", PAYLOAD, root, NoCalls())


def test_a_labeler_whose_input_is_another_labelers_reordered_is_refused(tmp_path):
    other = tmp_path / "run" / "gpt-5.6-sol-medium"
    full = {
        **dict(reversed(list(PAYLOAD.items()))),
        "version": labels.SCORER_EDIT.version,
        "rubric_sha256": labels.rubric_sha256(labels.SCORER_EDIT.rubric_name),
    }
    write_json(other / "scorer_edit" / "payloads" / "trial-1.json", full)
    root = tmp_path / "run" / "gpt-6.1-sol-medium"
    _saved(root, PAYLOAD)
    with pytest.raises(ValueError, match="scorer_edit.*trial-1"):
        labels.run_stage(labels.SCORER_EDIT, "trial-1", PAYLOAD, root, NoCalls())


def _transcript(text):
    return coder_transcript.Transcript(
        episode="job/trial-1",
        source="harbor",
        harness="claude-code",
        model=MODELS[1],
        setting=SETTINGS[0],
        condition=PRESENT_HIDDEN,
        seed=100,
        user_available=True,
        checker_exposed=False,
        checker_target=None,
        intended="a.csv",
        files=["a.csv", "b.csv"],
        reasoning_kind="none",
        events=[coder_transcript.Event("e1", 1, "agent_message", text)],
    )


def _coded(out_root, transcript, *, attempts=True):
    """A coder run folder holding a saved coding of `transcript`, recorded as sent."""
    folder = out_root / coder.safe_name(transcript.episode)
    folder.mkdir(parents=True)
    sent = [
        {"role": "system", "content": coder.system_prompt()},
        {"role": "user", "content": coder_transcript.render(transcript)},
    ]
    if attempts:
        (folder / "attempt-1.json").write_text(json.dumps({"request_messages": sent}))
    result = {"episode": transcript.episode, "status": "ok", "coding": {}}
    (folder / "result.json").write_text(json.dumps(result))
    return result


def test_a_coding_is_reused_for_the_identical_transcript(tmp_path):
    result = _coded(tmp_path, _transcript("Which file?"))
    assert (
        coder.code_episode(
            _transcript("Which file?"), out_root=tmp_path, config=coder.CoderConfig(), client=None
        )
        == result
    )


def test_a_coding_is_not_reused_for_a_changed_transcript(tmp_path):
    _coded(tmp_path, _transcript("Which file?"))
    with pytest.raises(ValueError, match="trial-1"):
        coder.code_episode(
            _transcript("Which file!"), out_root=tmp_path, config=coder.CoderConfig(), client=None
        )


def test_a_coding_without_its_recorded_request_is_not_reused(tmp_path):
    _coded(tmp_path, _transcript("Which file?"), attempts=False)
    with pytest.raises(ValueError, match="trial-1"):
        coder.code_episode(
            _transcript("Which file?"), out_root=tmp_path, config=coder.CoderConfig(), client=None
        )


@pytest.mark.parametrize("read", [export.read_retry_log, export.read_superseded])
def test_a_missing_retry_log_or_superseded_list_fails_naming_it(tmp_path, read):
    absent = tmp_path / "absent.json"
    with pytest.raises(FileNotFoundError, match="absent.json"):
        read(absent)


def test_the_export_needs_an_explicit_choice_for_retries_and_superseded_jobs(tmp_path, capsys):
    from ping_or_peek.cli import main

    results = tmp_path / "results.json"
    results.write_text(json.dumps({"annotator": {}, "trials": []}))
    with pytest.raises(SystemExit):
        main(["export", str(results), "--out-dir", str(tmp_path / "out")])
    assert "--retry-log" in capsys.readouterr().err


def _locked(tmp_path, text, name="trial-1"):
    """A trial whose lock names a persona by a path in a checkout that no longer exists."""
    folder = tmp_path / name
    folder.mkdir()
    digest = hashlib.sha256(text.encode()).hexdigest()
    lock = {
        "user_persona": {
            "path": "/elsewhere/old-checkout/persona/persona.md",
            "digest": f"sha256:{digest}",
        }
    }
    (folder / "lock.json").write_text(json.dumps(lock))
    return trial.Trial(folder, {"trial_name": name}, {})


def test_the_persona_is_found_in_the_working_tree_by_its_digest(tmp_path):
    persona = (trial.REPO / "persona" / "persona.md").read_text()
    assert trial.persona_text(_locked(tmp_path, persona)) == persona


def test_the_persona_is_found_in_the_history_by_its_digest(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "persona").mkdir(parents=True)

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    (repo / "persona" / "persona.md").write_text("the persona the runs used\n")
    git("add", "persona/persona.md")
    git("commit", "-q", "-m", "first")
    (repo / "persona" / "persona.md").write_text("a later persona\n")
    git("commit", "-q", "-am", "second")
    monkeypatch.setattr(trial, "REPO", repo)
    found = trial.persona_text(_locked(tmp_path, "the persona the runs used\n"))
    assert found == "the persona the runs used\n"


def test_a_persona_no_version_matches_fails_naming_the_trial(tmp_path):
    with pytest.raises(trial.PersonaNotFound, match="trial-1"):
        trial.persona_text(_locked(tmp_path, "a persona this repository never had"))


def test_a_trial_without_a_lock_fails_naming_it(tmp_path):
    (tmp_path / "trial-2").mkdir()
    with pytest.raises(trial.PersonaNotFound, match="trial-2"):
        trial.persona_text(trial.Trial(tmp_path / "trial-2", {"trial_name": "trial-2"}, {}))


def test_label_checks_its_retry_log_before_labeling_anything(tmp_path, monkeypatch):
    from ping_or_peek import cli
    from ping_or_peek.labeling import label_command as labeling_main
    from ping_or_peek.labeling import label_run

    called = []
    monkeypatch.setenv("PORTKEY_API_KEY", "dummy")
    monkeypatch.setattr(labeling_main, "find_trials", lambda paths, **_: [tmp_path / "trial"])
    monkeypatch.setattr(label_run, "label_trials", lambda *a, **k: called.append(1) or ([], []))
    argv = ["label", str(tmp_path), "--run", "r", "--out", str(tmp_path / "out.json")]
    argv += [
        "--label-root",
        str(tmp_path / "labels"),
        "--retry-log",
        str(tmp_path / "absent.jsonl"),
    ]
    with pytest.raises(FileNotFoundError, match="absent.jsonl"):
        cli.main(argv)
    assert called == []


def test_a_frozen_payload_refuses_the_same_values_in_another_key_order(tmp_path):
    path = tmp_path / "payloads" / "trial-1.json"
    write_json(path, PAYLOAD)
    with pytest.raises(ValueError, match="trial-1"):
        labels._write_once(path, dict(reversed(list(PAYLOAD.items()))))


def test_a_lock_naming_the_digest_but_no_file(tmp_path):
    # The digest alone checks a persona given for the trial; finding its text needs the file name,
    # and a lock without one fails naming the trial, never as a KeyError, which ping_or_peek.runs.checks
    # would take for a trial with no full record.
    persona = (trial.REPO / "persona" / "persona.md").read_text()
    folder = tmp_path / "trial-1"
    folder.mkdir()
    digest = hashlib.sha256(persona.encode()).hexdigest()
    (folder / "lock.json").write_text(json.dumps({"user_persona": {"digest": f"sha256:{digest}"}}))
    locked = trial.Trial(folder, {"trial_name": "trial-1"}, {})
    trial.check_persona(locked, persona)
    with pytest.raises(trial.PersonaNotFound, match="trial-1"):
        trial.persona_text(locked)
