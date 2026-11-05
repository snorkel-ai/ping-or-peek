"""Retries run the Harbor build their provenance records, and every retry is logged."""

import json
import os
from functools import partial

from ping_or_peek import cli
from ping_or_peek import config as project_config
from ping_or_peek.config import MODEL_SPECS, SETTINGS
from ping_or_peek.runs import retry
from tests.test_run import _fake_harbor, stub_job_config


def test_the_harbor_in_dotenv_is_the_one_recorded_and_launched(tmp_path, monkeypatch):
    _fake_harbor(tmp_path, monkeypatch, "RuntimeError: something else")
    monkeypatch.delenv("HARBOR")
    harbor = str(tmp_path / "harbor")
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"HARBOR={harbor}\n")
    monkeypatch.setattr(project_config, "DOTENV", dotenv)
    model, (harness, _) = next(iter(MODEL_SPECS.items()))
    harness_setting = f"{harness}__{SETTINGS[0]}"
    failure = {
        "job": f"pilot__{harness_setting}",
        "trial": "trial-1",
        "key": (harness_setting, model, "task-1"),
        "checks": {"agent_started": False},
        "details": {},
        "passed": False,
    }
    monkeypatch.setattr(retry, "failed_trials", lambda jobs_dir, phase: [failure])
    monkeypatch.setattr(retry, "job_config", stub_job_config)
    # retry takes no disk threshold; with sleep patched out, run_jobs would hold forever on a
    # host with less than its default free.
    monkeypatch.setattr(retry, "run_jobs", partial(retry.run_jobs, min_free_gb=0))
    recorded = []
    monkeypatch.setattr(
        retry, "provenance", lambda acpx_package: recorded.append(os.environ.get("HARBOR")) or {}
    )
    log = tmp_path / "log.jsonl"
    argv = ["retry", "--jobs-dir", str(tmp_path), "--phase", "pilot", "--run", "--log", str(log)]
    assert cli.main(argv) == 0
    assert recorded == [harbor]
    assert (tmp_path / "count").read_text().strip() == "1"
    assert [json.loads(line)["trial"] for line in log.read_text().splitlines()] == ["trial-1"]
