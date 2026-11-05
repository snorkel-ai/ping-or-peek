"""Credentials come from the environment or the repository's .env, and the environment wins.

Every .env here is a temporary file with dummy values; the repository's own .env is never read.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from dotenv import dotenv_values

from ping_or_peek import config as project_config

REPO = Path(__file__).resolve().parent.parent
NAME = "OPENAI_API_KEY"


def test_dotenv_is_the_repository_root_file(monkeypatch):
    monkeypatch.undo()  # drop any test-wide redirection, to see the module's own path
    assert project_config.DOTENV == REPO / ".env"


def test_the_suite_never_loads_the_repository_dotenv(monkeypatch):
    loaded = []
    monkeypatch.setattr(project_config, "load_dotenv", lambda path, **_: loaded.append(path))
    project_config.load_env()  # what every entry point's main does inside other tests
    assert REPO / ".env" not in loaded


def test_load_sets_a_variable_that_was_unset(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"{NAME}=dummy-from-dotenv\n")
    monkeypatch.setattr(project_config, "DOTENV", dotenv)
    monkeypatch.setenv(NAME, "")  # recorded, so the variable is removed again after the test
    monkeypatch.delenv(NAME)
    project_config.load_env()
    assert os.environ[NAME] == "dummy-from-dotenv"


def test_load_leaves_a_set_variable_unchanged(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"{NAME}=dummy-from-dotenv\n")
    monkeypatch.setattr(project_config, "DOTENV", dotenv)
    monkeypatch.setenv(NAME, "dummy-from-environment")
    project_config.load_env()
    assert os.environ[NAME] == "dummy-from-environment"


def test_load_sets_a_run_setting_too(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("HARBOR=dummy-harbor\n")
    monkeypatch.setattr(project_config, "DOTENV", dotenv)
    monkeypatch.setenv("HARBOR", "")  # recorded, so the variable is removed again after the test
    monkeypatch.delenv("HARBOR")
    project_config.load_env()
    assert os.environ["HARBOR"] == "dummy-harbor"


def _variables_the_code_reads() -> set[str]:
    """Each variable named in an os.environ read in a tracked file outside tests/ and .github/,
    plus the ${VAR} credentials the runners hand to Harbor. Untracked files, such as job output
    or a virtual environment, are not the code."""
    from ping_or_peek.runs import confidentiality_test, run

    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "*.py"], capture_output=True, text=True, check=True
    ).stdout.split()
    read = re.compile(r'os\.environ(?:\.get\(|\[)\s*"([A-Z][A-Z0-9_]*)"')
    names = set()
    for name in tracked:
        if Path(name).parts[0] not in ("tests", ".github"):
            names |= set(read.findall((REPO / name).read_text()))
    templates = " ".join([*run.CREDENTIAL_ENV.values(), *confidentiality_test.CREDENTIALS])
    names |= set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)\}", templates))
    return names


def test_the_template_lists_every_variable_the_code_reads():
    template = (REPO / ".env.example").read_text()
    listed = re.findall(r"^# ([A-Z][A-Z0-9_]*)=$", template, re.MULTILINE)
    assert sorted(listed) == sorted(_variables_the_code_reads())


def test_an_unfilled_copy_of_the_template_sets_nothing():
    # A blank KEY= line would set "", which Harbor passes on as an empty credential instead of
    # failing on an unset ${KEY} (harbor.utils.env.resolve_env_vars).
    assert dotenv_values(REPO / ".env.example") == {}


def _gateway_key(tmp_path: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    """Run a copy of scripts/gateway.sh in a scratch repository and print the key it exports."""
    scripts = tmp_path / "repo" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "gateway.sh", scripts / "gateway.sh")
    base = {k: v for k, v in os.environ.items() if k not in ("PORTKEY_API_KEY", "ENV_FILE")}
    return subprocess.run(
        ["bash", str(scripts / "gateway.sh"), "printenv", "PORTKEY_API_KEY"],
        env={**base, **env},
        capture_output=True,
        text=True,
        check=False,
    )


def test_gateway_reads_the_repository_dotenv_by_default(tmp_path):
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / ".env").write_text("PORTKEY_API_KEY=dummy-from-repo-dotenv\n")
    result = _gateway_key(tmp_path, {})
    assert result.stdout.strip() == "dummy-from-repo-dotenv", result.stderr


def test_gateway_env_file_overrides_the_repository_dotenv(tmp_path):
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / ".env").write_text("PORTKEY_API_KEY=dummy-from-repo-dotenv\n")
    other = tmp_path / "other.env"
    other.write_text("PORTKEY_API_KEY=dummy-from-env-file\n")
    result = _gateway_key(tmp_path, {"ENV_FILE": str(other)})
    assert result.stdout.strip() == "dummy-from-env-file", result.stderr


def test_gateway_keeps_a_key_already_set(tmp_path):
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / ".env").write_text("PORTKEY_API_KEY=dummy-from-repo-dotenv\n")
    result = _gateway_key(tmp_path, {"PORTKEY_API_KEY": "dummy-from-environment"})
    assert result.stdout.strip() == "dummy-from-environment", result.stderr


def test_gateway_without_a_key_fails(tmp_path):
    result = _gateway_key(tmp_path, {})
    assert result.returncode != 0
    assert "PORTKEY_API_KEY" in result.stderr


def test_gateway_with_a_dotenv_lacking_the_key_fails_with_a_message(tmp_path):
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / ".env").write_text("OPENAI_API_KEY=dummy-openai\n")
    result = _gateway_key(tmp_path, {})
    assert result.returncode != 0
    assert "PORTKEY_API_KEY" in result.stderr


def test_a_package_installed_outside_the_checkout_refuses_to_load(tmp_path):
    """`REPO` is two folders above `config.py`. Installed as a wheel, that is no checkout, and the
    package would read configs/, persona/ and verifier/ from the wrong folder without an error."""
    import importlib.util

    installed = tmp_path / "site-packages" / "ping_or_peek"
    installed.mkdir(parents=True)
    shutil.copy(project_config.__file__, installed / "config.py")
    spec = importlib.util.spec_from_file_location("installed_config", installed / "config.py")
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(RuntimeError, match=r"(?s)checkout.*editable"):
        spec.loader.exec_module(module)


# Values of the grid that some module other than config.py may still spell out, and why.
OTHER_THAN_THE_GRID = {
    ("labeling/annotator.py", "gpt-6.1-sol"): "the labeler's own default model, not a grid axis",
    ("labeling/label_command.py", "gpt-6.1-sol"): "the labeler's own default model",
    ("labeling/reasoning/label.py", "gpt-6.1-sol"): "the reasoning coder's own default model",
    ("labeling/reasoning/code_command.py", "gpt-6.1-sol"): "the reasoning coder's default model",
    ("runs/run.py", "gpt-6.1-sol"): "the simulated user's model, not a grid axis",
    ("runs/confidentiality_test.py", "gpt-6.1-sol"): "the probe's simulated user",
    ("runs/confidentiality_test.py", "claude-sonnet-5-5"): "the probe's agent and user models",
    ("dataset/export_check.py", "A"): "the name of Check A, printed in its report",
    ("labeling/labeler_inputs.py", "B"): "the name of Check B, printed in its report",
}  # fmt: skip


def test_config_is_the_only_definition_of_the_grid():
    """No other module writes out a setting code, a condition, a model under test, a path of an
    episode or the native harness: each takes it from `config`. Docstrings don't count."""
    import ast

    package = REPO / "src" / "ping_or_peek"
    values = {
        *project_config.SETTINGS, *project_config.CONDITIONS, *project_config.MODELS,
        *project_config.PATHS, project_config.NATIVE,
    }  # fmt: skip
    found = set()
    for path in package.rglob("*.py"):
        if path.name == "config.py":
            continue
        tree = ast.parse(path.read_text())
        docstrings = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Expr)}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in values
                and id(node) not in docstrings
            ):
                found.add((str(path.relative_to(package)), node.value))
    assert sorted(found - OTHER_THAN_THE_GRID.keys()) == []
