"""The reasoning coder's client takes its Portkey key from the environment, which the CLI has
loaded the repo's .env into (tests/test_dotenv_entry_points.py)."""

import pytest

from ping_or_peek.labeling.reasoning import label


def test_client_key_from_environment(monkeypatch):
    monkeypatch.setenv("PORTKEY_API_KEY", "dummy-from-environment")
    client = label.make_client(label.CoderConfig())
    assert client.api_key == "dummy-from-environment"


def test_client_without_a_key_fails(monkeypatch):
    monkeypatch.delenv("PORTKEY_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PORTKEY_API_KEY"):
        label.make_client(label.CoderConfig())


@pytest.mark.parametrize("version", ["coder provenance", "labeler code version"])
def test_an_edit_to_the_shared_grid_marks_the_code_dirty(version, monkeypatch):
    """Both labelers record whether the code that built their inputs had local edits; config.py
    holds the grid that code reads, so it is among the paths checked."""
    import subprocess

    from ping_or_peek.labeling import label_command

    calls = []

    def run(argv, **_):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    {"coder provenance": label.provenance, "labeler code version": label_command._code_version}[
        version
    ]()
    (status,) = [argv for argv in calls if "status" in argv]
    assert "src/ping_or_peek/config.py" in status[status.index("--") + 1 :]
