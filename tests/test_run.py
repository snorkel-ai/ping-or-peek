"""The run matrix matches the plan and the image."""

import json
import os

from ping_or_peek.config import MODEL_SPECS, REPO
from ping_or_peek.runs.run import ADAPTERS, PINS, agent_config, phase_jobs

DOCKERFILE = (REPO / "src" / "ping_or_peek" / "tasks" / "Dockerfile").read_text()


def test_pins_match_the_baked_image():
    assert f"bootstrap.sh | bash -s -- {PINS['claude-code']}" in DOCKERFILE
    assert f"@openai/codex@{PINS['codex']}" in DOCKERFILE
    assert f"opencode-ai@{PINS['opencode']}" in DOCKERFILE
    for package, version in ADAPTERS.values():
        assert f"{package}@{version}" in DOCKERFILE


def test_effort_convention():
    assert {m for m, (_, e) in MODEL_SPECS.items() if e == "xhigh"} == {"claude-fable-5-1", "gpt-6-astra"}


def test_phase_sizes():
    def episodes(phase):
        return sum(len(j.models) * len(j.tasks) for j in phase_jobs(phase))

    assert episodes("s0") == 24
    assert episodes("controls") == 36
    assert episodes("pilot") == 36
    assert episodes("collection") == 1620


def test_claude_permission_mode_per_setting():
    modes = {s: agent_config("claude-code", "claude-opus-5-5", s)["kwargs"]["permission_mode"]
             for s in "SAB"}
    assert modes == {"S": "auto", "A": "default", "B": "auto"}


def test_codex_mode_and_flag_per_setting(tmp_path):
    configs = {s: agent_config("codex", "gpt-6-luna", s, config_dir=tmp_path) for s in "SAB"}
    assert {s: c["kwargs"]["acp_agent_mode"] for s, c in configs.items()} == {
        "S": "workspace-write", "A": "workspace-write", "B": "agent-full-access"}
    flags = {s: "default_mode_request_user_input = true" in open(c["kwargs"]["config"]).read()
             for s, c in configs.items()}
    assert flags == {"S": False, "A": True, "B": False}


def test_only_codex_jobs_get_the_seccomp_overlay(tmp_path, monkeypatch):
    from ping_or_peek.runs.run import job_config

    monkeypatch.setenv("ACPX_PACKAGE", "acpx.tgz")
    monkeypatch.setenv("CLAUDE_ACPX_PACKAGE", "acpx-claude.tgz")
    monkeypatch.delenv("CODEX_CONTAINER", raising=False)
    for job in phase_jobs("s0"):
        overlays = job_config(job, tmp_path, 1).get("environment", {}).get("extra_docker_compose", [])
        if job.harness == "codex":
            assert overlays and "seccomp:unconfined" in open(overlays[0]).read()
        else:
            assert overlays == []


def test_only_opencode_jobs_set_effort_to_default(tmp_path, monkeypatch):
    from ping_or_peek.runs.run import job_config

    monkeypatch.setenv("ACPX_PACKAGE", "acpx.tgz")
    monkeypatch.setenv("CLAUDE_ACPX_PACKAGE", "acpx-claude.tgz")
    for job in phase_jobs("s0"):
        bridge = job_config(job, tmp_path, 1)["user_agent"]["bridge"]["kwargs"]
        expected = {"effort": "default"} if job.harness == "opencode" else None
        assert bridge.get("session_config") == expected


def test_opencode_permissions_per_setting():
    perms = {s: agent_config("opencode", "gpt-6-luna", s)["kwargs"]["opencode_config"].get("permission")
             for s in "SAB"}
    assert perms == {"S": None, "A": {"edit": "ask", "bash": "ask"}, "B": None}


def test_opencode_does_not_fake_a_git_repo():
    assert all(agent_config("opencode", "claude-opus-5-5", s)["kwargs"]["fake_vcs"] is False
               for s in "SAB")


def test_every_job_gets_setup_headroom(tmp_path, monkeypatch):
    from ping_or_peek.runs.run import SETUP_TIMEOUT_MULTIPLIER, job_config

    monkeypatch.setenv("ACPX_PACKAGE", "acpx.tgz")
    monkeypatch.setenv("CLAUDE_ACPX_PACKAGE", "acpx-claude.tgz")
    assert all(job_config(job, tmp_path, 1)["agent_setup_timeout_multiplier"]
               == SETUP_TIMEOUT_MULTIPLIER == 2.0 for job in phase_jobs("s0"))


def _fake_harbor(tmp_path, monkeypatch, first_output):
    """A `harbor` that prints `first_output` and fails on its first run, then succeeds."""
    from ping_or_peek.runs import run

    fake = tmp_path / "harbor"
    fake.write_text(
        "#!/bin/sh\n"
        'n=$(cat "$COUNT" 2>/dev/null || echo 0); echo $((n + 1)) > "$COUNT"\n'
        f'if [ "$n" = 0 ]; then echo "{first_output}"; exit 1; fi\n'
        "echo done\n")
    fake.chmod(0o755)
    monkeypatch.setenv("HARBOR", str(fake))
    monkeypatch.setenv("COUNT", str(tmp_path / "count"))
    monkeypatch.setattr(run.time, "sleep", lambda seconds: None)
    config = tmp_path / "job.json"
    config.write_text(json.dumps({"jobs_dir": str(tmp_path / "jobs"), "job_name": "job"}))
    return run.Launch("job", config, tmp_path / "job.log")


def test_a_job_stopped_at_the_egress_check_is_relaunched_and_logged(tmp_path, monkeypatch):
    from ping_or_peek.runs.run import run_jobs

    launch = _fake_harbor(tmp_path, monkeypatch,
                          "ValueError: network_mode='allowlist' is not supported by docker")
    infra_log = tmp_path / "infra_retries.jsonl"
    run_jobs([launch], tmp_path, min_free_gb=0, infra_log=infra_log)
    assert (tmp_path / "count").read_text().strip() == "2"
    assert json.loads(infra_log.read_text())["failed_checks"] == ["egress_check"]
    log = (tmp_path / "job.log").read_text()
    assert "is not supported" in log and log.rstrip().endswith("done")  # the first attempt is kept


def stub_job_config(job, jobs_dir, *args, **kwargs):
    """A job config with only what main, provenance and Launch read from it."""
    return {"jobs_dir": str(jobs_dir), "job_name": job.name,
            "user_agent": {"bridge": {"kwargs": {"acpx_package": None}}}}


def test_the_harbor_in_dotenv_is_the_one_recorded_and_launched(tmp_path, monkeypatch):
    from ping_or_peek import cli
    from ping_or_peek import config as project_config
    from ping_or_peek.runs import run

    _fake_harbor(tmp_path, monkeypatch, "RuntimeError: something else")
    monkeypatch.delenv("HARBOR")
    harbor = str(tmp_path / "harbor")
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"HARBOR={harbor}\n")
    monkeypatch.setattr(project_config, "DOTENV", dotenv)
    recorded = []
    monkeypatch.setattr(run, "job_config", stub_job_config)
    monkeypatch.setattr(run, "provenance", lambda acpx_package: recorded.append(
        os.environ.get("HARBOR")) or {})
    argv = ["run", "--phase", "s0", "--jobs-dir", str(tmp_path / "jobs"), "--harness", "native",
            "--model", next(iter(MODEL_SPECS)), "--min-free-gb", "0"]
    assert cli.main(argv) == 0
    assert recorded == [harbor]
    assert (tmp_path / "count").read_text().strip() == "1"


def test_other_failures_are_not_relaunched(tmp_path, monkeypatch):
    from ping_or_peek.runs.run import run_jobs

    launch = _fake_harbor(tmp_path, monkeypatch, "RuntimeError: something else")
    run_jobs([launch], tmp_path, min_free_gb=0, infra_log=tmp_path / "infra_retries.jsonl")
    assert (tmp_path / "count").read_text().strip() == "1"
    assert not (tmp_path / "infra_retries.jsonl").exists()


def test_native_claude_gets_its_own_acpx_and_summarized_thinking(tmp_path, monkeypatch):
    from ping_or_peek.runs.run import CLAUDE_THINKING, configured_thinking, job_config

    monkeypatch.setenv("ACPX_PACKAGE", "acpx.tgz")
    monkeypatch.setenv("CLAUDE_ACPX_PACKAGE", "acpx-claude.tgz")
    for job in phase_jobs("collection"):
        config = job_config(job, tmp_path, 1)
        bridge = config["user_agent"]["bridge"]["kwargs"]
        native_claude = job.harness == "claude-code"
        assert bridge["acpx_package"] == ("acpx-claude.tgz" if native_claude else "acpx.tgz")
        assert configured_thinking(config) == (CLAUDE_THINKING if native_claude else None)
    assert CLAUDE_THINKING == {"type": "adaptive", "display": "summarized"}


def test_native_claude_refuses_to_run_without_its_acpx(tmp_path, monkeypatch):
    import pytest

    from ping_or_peek.runs.run import job_config

    monkeypatch.setenv("ACPX_PACKAGE", "acpx.tgz")
    monkeypatch.delenv("CLAUDE_ACPX_PACKAGE", raising=False)
    claude_job = next(j for j in phase_jobs("pilot") if j.harness == "claude-code")
    with pytest.raises(SystemExit):
        job_config(claude_job, tmp_path, 1)


def test_a_harbor_installed_from_a_git_url_records_the_commit_pip_noted():
    from ping_or_peek.runs.run import harbor_commit

    commit = "64c330fe1c1dc1914e5c4af2d86bf42de7e0b798"
    source = {
        "url": "https://github.com/snorkel-ai/ping-or-peek-harbor",
        "vcs_info": {"vcs": "git", "commit_id": commit},
    }
    assert harbor_commit(source) == commit


def commit_in(folder) -> str:
    """Make `folder` a git repository with one commit, and return that commit."""
    import subprocess

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(folder), "-c", "user.name=t", "-c", "user.email=t@t", *args],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip

    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-m", "harbor")
    return git("rev-parse", "HEAD")


def test_a_harbor_from_a_local_checkout_with_no_recorded_commit_records_its_head(tmp_path):
    from ping_or_peek.runs.run import harbor_commit

    head = commit_in(tmp_path)
    assert harbor_commit({"url": f"file://{tmp_path}"}) == head


def test_the_commit_the_install_recorded_wins_over_the_checkouts_head(tmp_path):
    """pip records the commit it installed; the checkout may have moved on since."""
    from ping_or_peek.runs.run import harbor_commit

    installed = "a" * 40
    assert commit_in(tmp_path) != installed
    source = {"url": f"file://{tmp_path}", "vcs_info": {"vcs": "git", "commit_id": installed}}
    assert harbor_commit(source) == installed


def test_a_harbor_with_no_recorded_source_records_no_commit():
    from ping_or_peek.runs.run import harbor_commit

    assert harbor_commit({}) is None


def test_a_missing_task_variant_is_generated_with_its_instruction_file(tmp_path, monkeypatch):
    """`ensure_tasks` writes the tasks a job needs when its variant doesn't exist yet, with the
    setting's passage as the project note: what the first `run` in a fresh checkout does."""
    from ping_or_peek.config import CHECK_IN, CONDITIONS, SEEDS
    from ping_or_peek.runs import run

    monkeypatch.setattr(run, "TASKS", tmp_path)
    job = run.Job("pilot", "claude-code", CHECK_IN, (), ())
    root = run.ensure_tasks(job)
    assert root == tmp_path / job.variant
    assert sorted(p.name for p in root.iterdir() if p.name != "conversational_control") == sorted(
        f"{condition}__seed{seed}" for condition in CONDITIONS for seed in SEEDS
    )
    passage = (REPO / run.PASSAGE[CHECK_IN]).read_text()
    workspace = root / f"{CONDITIONS[0]}__seed{SEEDS[0]}" / "environment" / "workspace"
    assert (workspace / run.INSTRUCTION_FILE["claude-code"]).read_text() == passage
