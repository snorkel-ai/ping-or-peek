"""The one command, `ping-or-peek`: its subcommands, what `run` selects, and that nothing else
starts the code."""

import json
import re
import tomllib
from pathlib import Path

import pytest

from ping_or_peek import cli
from ping_or_peek.config import AWAY, CONDITIONS, MODELS, PRESENT_VISIBLE, REPO, TASKS
from ping_or_peek.runs import run

EXPECTED = Path(__file__).parent / "expected" / "run_collection_dry_run.txt"
SUBCOMMANDS = (
    "tasks", "run", "check", "retry", "progress", "s0-report", "confidentiality-test",
    "label", "code", "export", "check-record", "dataset", "safety-net",
)  # fmt: skip
GROUPS = {"dataset": ("build", "card", "check"), "safety-net": ("export", "labeler-inputs")}


def listed(output: str) -> set[str]:
    """The names an argparse help lists as its subcommands, one per indented line."""
    return set(re.findall(r"^ {2,}([a-z][a-z0-9-]*)\s", output, re.MULTILINE))


def test_help_lists_every_subcommand(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--help"])
    assert stopped.value.code == 0
    assert set(SUBCOMMANDS) <= listed(capsys.readouterr().out)


@pytest.mark.parametrize("group", GROUPS)
def test_a_group_lists_its_subcommands(group, capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main([group, "--help"])
    assert stopped.value.code == 0
    assert set(GROUPS[group]) <= listed(capsys.readouterr().out)


COMMAND_PATHS = [
    *((name,) for name in SUBCOMMANDS if name not in GROUPS),
    *((group, name) for group, names in GROUPS.items() for name in names),
]


@pytest.mark.parametrize("path", COMMAND_PATHS, ids=" ".join)
def test_every_subcommand_loads_and_documents_its_arguments(path, capsys):
    """Each subcommand's module imports and adds its arguments, so its help prints."""
    with pytest.raises(SystemExit) as stopped:
        cli.main([*path, "--help"])
    assert stopped.value.code == 0
    assert capsys.readouterr().out.startswith(f"usage: ping-or-peek {' '.join(path)}")


def test_an_unknown_subcommand_is_refused_naming_the_choices(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["nonsense"])
    assert stopped.value.code == 2
    assert "run" in capsys.readouterr().err


def test_the_console_script_is_the_cli():
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    assert project["scripts"] == {"ping-or-peek": "ping_or_peek.cli:main"}


def test_no_other_module_starts_the_code():
    """The only entry points are the CLI and the verifier: no `__main__.py` and no
    `if __name__ == "__main__"` anywhere in the package."""
    package = REPO / "src" / "ping_or_peek"
    assert sorted(p.name for p in package.rglob("__main__.py")) == []
    guarded = [
        str(p.relative_to(REPO))
        for p in package.rglob("*.py")
        if 'if __name__ == "__main__"' in p.read_text()
    ]
    assert guarded == []


def test_nothing_tells_a_reader_to_start_the_code_with_python_dash_m():
    old = re.compile(r"python3? -m (ping_or_peek|scripts|generator|dataset|labeling)\b")
    tracked = [
        *REPO.glob("*.md"), REPO / ".env.example", REPO / ".gitignore",
        *(REPO / "analysis").glob("*.md"), *(REPO / "configs").glob("*.md"),
        *(REPO / "docs").glob("*.md"), *(REPO / "persona").glob("*.md"),
        *(REPO / "scripts").glob("*.sh"), *(REPO / "src").rglob("*.py"),
    ]  # fmt: skip
    found = [
        f"{p.relative_to(REPO)}:{n}"
        for p in tracked
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if old.search(line)
    ]
    assert found == []


# --- run ---------------------------------------------------------------------------------------


@pytest.fixture
def dry_run(tmp_path, monkeypatch, capsys):
    """`ping-or-peek run ... --dry-run` with no tasks generated: the job configs it prints."""
    monkeypatch.setattr(run, "ensure_tasks", lambda job: TASKS / job.variant)
    monkeypatch.setenv("ACPX_PACKAGE", "acpx-x")
    monkeypatch.setenv("CLAUDE_ACPX_PACKAGE", "acpx-claude-x")

    def go(*args: str) -> list[dict]:
        assert cli.main(["run", *args, "--dry-run", "--jobs-dir", str(tmp_path)]) == 0
        text, decoder, configs, at = capsys.readouterr().out, json.JSONDecoder(), [], 0
        while at < len(text.rstrip()):
            config, at = decoder.raw_decode(text, at)
            configs.append(config)
            at = len(text) - len(text[at:].lstrip())
        return configs

    return go


def test_the_collection_dry_run_prints_the_configs_the_run_script_printed(
    tmp_path, monkeypatch, capsys
):
    """The expected output is what the run script printed for the collection before the CLI, with
    the repository's and the jobs folder's paths replaced."""
    monkeypatch.setattr(run, "ensure_tasks", lambda job: TASKS / job.variant)
    monkeypatch.setenv("ACPX_PACKAGE", "acpx-x")
    monkeypatch.setenv("CLAUDE_ACPX_PACKAGE", "acpx-claude-x")
    argv = ["run", "--phase", "collection", "--dry-run", "--jobs-dir", str(tmp_path)]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out.replace(str(tmp_path), "<jobs>").replace(str(REPO), "<repo>")
    assert out == EXPECTED.read_text()


def test_run_selects_on_every_axis(dry_run):
    (job,) = dry_run(
        "--model", "claude-opus-5-5", "--harness", "native", "--setting", "B",
        "--condition", PRESENT_VISIBLE, "--seeds", "100-104", "--label", "adhoc",
    )  # fmt: skip
    assert job["job_name"] == "adhoc__claude-code__B"
    assert [a["model_name"] for a in job["agents"]] == ["claude-opus-5-5"]
    assert [Path(t["path"]).name for t in job["tasks"]] == [
        f"{PRESENT_VISIBLE}__seed{seed}" for seed in range(100, 105)
    ]


def test_native_selects_each_models_own_agent_and_opencode_all_models(dry_run):
    native = dry_run("--harness", "native", "--setting", "S", "--seeds", "100", "--label", "x")
    assert {j["job_name"] for j in native} == {"x__claude-code__S", "x__codex__S"}
    (opencode,) = dry_run(
        "--harness", "opencode", "--setting", "S", "--seeds", "100", "--label", "x"
    )
    assert opencode["job_name"] == "x__opencode__S"
    assert [a["model_name"].split("/")[-1] for a in opencode["agents"]] == list(MODELS)


def test_a_phase_is_narrowed_by_the_axes_and_keeps_its_name(dry_run):
    jobs = dry_run("--phase", "pilot", "--model", "gpt-6-luna")
    assert {j["job_name"] for j in jobs} == {
        f"pilot__{agent}__{setting}" for agent in ("codex", "opencode") for setting in "SAB"
    }
    assert all(a["model_name"].endswith("gpt-6-luna") for j in jobs for a in j["agents"])


@pytest.mark.parametrize(
    ("seeds", "expected"), [("100-102", [100, 101, 102]), ("100,102", [100, 102]), ("114", [114])]
)
def test_seeds_are_a_range_or_a_list(dry_run, seeds, expected):
    (job,) = dry_run(
        "--model", MODELS[0], "--harness", "opencode", "--setting", "S", "--condition",
        CONDITIONS[0], "--seeds", seeds, "--label", "x",
    )  # fmt: skip
    assert [int(Path(t["path"]).name.split("seed")[1]) for t in job["tasks"]] == expected


def test_a_selection_without_a_phase_needs_a_label_so_it_cannot_reuse_a_phases_jobs(
    tmp_path, capsys
):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["run", "--model", MODELS[0], "--dry-run", "--jobs-dir", str(tmp_path)])
    assert stopped.value.code == 2
    assert "--label" in capsys.readouterr().err


@pytest.mark.parametrize(
    "args",
    [
        ["--phase", "s0", "--seeds", "101"],  # s0 is seed 100
        ["--phase", "controls", "--condition", CONDITIONS[0]],  # the control has no condition
        ["--phase", "pilot", "--condition", AWAY],  # the pilot runs one condition
        ["--model", MODELS[0], "--seeds", "99", "--label", "x"],  # not a seed of the grid
    ],
)
def test_a_selection_that_picks_no_job_is_refused(args, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ACPX_PACKAGE", "acpx-x")
    with pytest.raises(SystemExit) as stopped:
        cli.main(["run", *args, "--dry-run", "--jobs-dir", str(tmp_path)])
    assert stopped.value.code != 0
    assert capsys.readouterr().err.strip()


# --- run: a selection never widens to the whole grid ---------------------------------------------


def test_a_reversed_seed_range_is_refused_not_read_as_every_seed(tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(
            [
                "run",
                "--phase",
                "collection",
                "--seeds",
                "101-100",
                "--dry-run",
                "--jobs-dir",
                str(tmp_path),
            ]
        )
    assert stopped.value.code == 2
    err = capsys.readouterr().err
    assert "101-100" in err and "low to high" in err


@pytest.mark.parametrize("axis", ["models", "harnesses", "settings", "conditions", "seeds"])
def test_an_explicitly_empty_axis_picks_nothing_rather_than_everything(axis):
    with pytest.raises(ValueError, match="picks no job"):
        run.select_jobs("collection", None, **{axis: []})


@pytest.mark.parametrize("label", ["", *run.PHASES])
def test_a_label_is_not_empty_and_not_a_phase_name_so_it_cannot_reuse_a_phases_jobs(
    label, tmp_path, capsys
):
    with pytest.raises(ValueError, match="--label"):
        run.select_jobs(None, label)
    with pytest.raises(SystemExit) as stopped:
        cli.main(
            ["run", "--phase", "pilot", "--label", label, "--dry-run", "--jobs-dir", str(tmp_path)]
        )
    assert stopped.value.code == 2
    assert "--label" in capsys.readouterr().err


def test_seeds_on_the_conversational_control_pick_no_job_by_the_rule_not_a_parse_error(
    tmp_path, capsys
):
    with pytest.raises(SystemExit) as stopped:
        cli.main(
            [
                "run",
                "--phase",
                "controls",
                "--seeds",
                "100",
                "--dry-run",
                "--jobs-dir",
                str(tmp_path),
            ]
        )
    assert stopped.value.code == 2
    err = capsys.readouterr().err
    assert "picks no job" in err and "invalid literal" not in err


def test_the_env_template_names_commands_not_modules():
    assert "ping_or_peek." not in (REPO / ".env.example").read_text()


# A label goes into job names and file paths, so it is a plain name: no `_`, `.` or `/`.
BAD_LABELS = [
    "", *run.PHASES, "collection__x", "./collection", "../x", "a/b", "a\\b", "x_y", "x.y", "-x",
    "Try", " x", "collection-retry1", "pilot-retry-x", "collection-x",
]  # fmt: skip


@pytest.mark.parametrize("label", BAD_LABELS)
def test_a_label_is_a_plain_name_so_it_cannot_alias_a_phases_jobs_or_name_a_path(
    label, tmp_path, capsys
):
    with pytest.raises(ValueError, match="--label"):
        run.select_jobs(None, label)
    with pytest.raises(SystemExit) as stopped:
        cli.main(
            [
                "run",
                "--model",
                MODELS[0],
                f"--label={label}",  # `--label -x` would read -x as an option
                "--dry-run",
                "--jobs-dir",
                str(tmp_path),
            ]
        )
    assert stopped.value.code == 2
    err = capsys.readouterr().err
    assert "--label" in err and "lowercase" in err


def test_a_plain_label_names_the_jobs():
    jobs = run.select_jobs(
        None, "try-1", [MODELS[0]], ["opencode"], ["S"], [PRESENT_VISIBLE], [100]
    )
    assert [job.name for job in jobs] == ["try-1__opencode__S"]
