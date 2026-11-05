"""The checks workflow runs the tests on every pull request, forks included, without Harbor and
with a read-only token, and gives every commit pushed to main a result."""

import re
import tomllib
from pathlib import Path

import yaml

CHECKS = Path(__file__).resolve().parents[1] / ".github/workflows/checks.yml"


def load() -> dict:
    return yaml.safe_load(CHECKS.read_text())


def test_actions_are_pinned_by_sha_on_github_hosted_runners():
    for job in load()["jobs"].values():
        assert re.fullmatch(r"ubuntu-[\d.]+", job["runs-on"])
        for step in job["steps"]:
            if "uses" in step:
                assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", step["uses"])


def test_a_read_only_token_no_secret_and_no_persisted_credentials():
    assert re.findall(r"secrets\.\w+", CHECKS.read_text()) == []
    workflow = load()
    assert workflow["permissions"] == {"contents": "read"}
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            if step.get("uses", "").startswith("actions/checkout@"):
                assert step["with"]["persist-credentials"] is False


def sync_command() -> str:
    runs = [step["run"] for step in load()["jobs"]["pytest"]["steps"] if "run" in step]
    (sync,) = [run for run in runs if run.startswith("uv sync")]
    return sync


def test_installs_the_project_locked_and_without_harbor():
    """Harbor is a fork in a repository CI's read-only token can't read, so the sync leaves it out;
    the project itself is installed, so `ping_or_peek` imports wherever a test starts Python (a
    subprocess, `python -m`), not only under pytest's own path."""
    sync = sync_command().split()
    assert "--locked" in sync
    assert sync[sync.index("--no-install-package") + 1] == "harbor"
    assert not {"--only-group", "--no-install-project", "--no-dev"} & set(sync)
    runs = [step["run"] for step in load()["jobs"]["pytest"]["steps"] if "run" in step]
    assert "uv run --no-sync pytest -rs" in runs


def test_the_sync_selects_no_group_so_every_default_group_installs():
    """`uv sync` installs the default groups, and the notebook group lets the test that executes
    the figures notebook run, not fail on import. Any option that selects groups (`--group`,
    `--no-group`, `--no-dev`, `--no-default-groups`, `--only-group`) could drop one, so the sync's
    only options are the lock and Harbor's omission."""
    uv = tomllib.loads(CHECKS.parents[2].joinpath("pyproject.toml").read_text())["tool"]["uv"]
    assert {"dev", "analysis", "notebook"} <= set(uv["default-groups"])
    assert sync_command().split() == ["uv", "sync", "--locked", "--no-install-package", "harbor"]


def test_each_push_to_main_has_its_own_concurrency_group():
    # GitHub cancels a pending run when a newer one joins its group, whatever cancel-in-progress
    # says, so pushes that share a group can leave a main commit with no result.
    group = load()["concurrency"]["group"]
    assert "github.sha" in group
