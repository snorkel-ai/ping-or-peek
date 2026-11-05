"""The command loads the repo's .env before any subcommand reads a setting from it.

Every .env here is a temporary file with dummy values; the repository's own .env is never read.
"""

import os

from ping_or_peek import cli
from ping_or_peek import config as project_config
from ping_or_peek.labeling import label_command, label_run
from ping_or_peek.labeling.reasoning import code_command


def _dotenv(tmp_path, monkeypatch, **values):
    dotenv = tmp_path / ".env"
    dotenv.write_text("".join(f"{k}={v}\n" for k, v in values.items()))
    monkeypatch.setattr(project_config, "DOTENV", dotenv)
    for name in values:
        monkeypatch.setenv(name, "")  # recorded, so each is removed again after the test
        monkeypatch.delenv(name)


def test_label_takes_its_record_store_from_dotenv(tmp_path, monkeypatch):
    _dotenv(
        tmp_path,
        monkeypatch,
        ASK_OR_SHORTCUT_LABEL_ROOT=tmp_path / "labels",
        PORTKEY_API_KEY="dummy-portkey",
    )
    roots = []

    def label_trials(trials, *, root, **_):
        roots.append(root)
        return [], []

    monkeypatch.setattr(label_command, "find_trials", lambda paths, **_: [tmp_path / "trial"])
    monkeypatch.setattr(label_run, "label_trials", label_trials)
    out = tmp_path / "results.json"
    assert (
        cli.main(["label", str(tmp_path), "--run", "r", "--out", str(out), "--no-retry-log"]) == 0
    )
    assert len(roots) == 1 and roots[0].is_relative_to(tmp_path / "labels" / "r")


def test_reasoning_takes_its_private_root_and_key_from_dotenv(tmp_path, monkeypatch):
    _dotenv(
        tmp_path,
        monkeypatch,
        WS6_PRIVATE_ROOT=tmp_path / "private",
        PORTKEY_API_KEY="dummy-portkey",
    )
    # Where a root read at import time would write, instead of the real private store.
    monkeypatch.setattr(code_command, "PRIVATE_ROOT", tmp_path / "import-time", raising=False)
    monkeypatch.setattr(code_command, "_trials", lambda *args: [])
    argv = ["code", str(tmp_path), "--run", "r", "--jobs-root", str(tmp_path), "--no-superseded"]
    cli.main(argv)
    assert (tmp_path / "private" / "labels" / "r" / "run.json").exists()
    assert os.environ["PORTKEY_API_KEY"] == "dummy-portkey"
