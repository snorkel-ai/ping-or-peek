"""Retry infrastructure failures only, and log every retry separately.

    uv run ping-or-peek retry --jobs-dir JOBS --phase pilot              # list failures
    uv run ping-or-peek retry --jobs-dir JOBS --phase pilot --run --log infra_retries.jsonl

A trial is retried only when a mechanical check fails (src/ping_or_peek/runs/checks.py): the episode never ran
as designed, so there is no behavioral outcome to keep. Whatever an agent chose to do, including
running to the time limit while a request is pending, is a result and is never rerun.

Each retry is a job per harness, setting and model, named
`<phase>-retry<N>-<model>__<harness>__<setting>`, running exactly that model's failed tasks.
(A Harbor job crosses its agents with its tasks, so one job for several models would rerun pairs
that already succeeded.)
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from ping_or_peek.config import HARNESSES, harness_of
from ping_or_peek.dataset.trial import read_json
from ping_or_peek.runs.checks import check_trial
from ping_or_peek.runs.run import Job, Launch, job_config, provenance, run_jobs


def failed_trials(jobs_dir: Path, phase: str) -> list[dict]:
    """Model × task pairs whose newest attempt (original or retry) failed a mechanical check."""
    attempts = []
    for pattern in (f"{phase}__*__*/*/result.json", f"{phase}-retry*__*__*/*/result.json"):
        for result in jobs_dir.glob(pattern):
            trial_dir = result.parent
            config = read_json(trial_dir / "config.json", trial_dir.name)
            key = (
                trial_dir.parent.name.split("__", 1)[1],  # harness__setting
                config["agent"]["model_name"].split("/")[-1],
                Path(config["task"]["path"]).name,
            )
            started = read_json(result, trial_dir.name).get("started_at") or ""
            attempts.append((started, key, trial_dir))
    latest: dict[tuple[str, str, str], dict] = {}
    for _, key, trial_dir in sorted(attempts):  # oldest first, so the newest attempt wins
        latest[key] = {**check_trial(trial_dir), "job": trial_dir.parent.name, "key": key}
    return [c for c in latest.values() if not c["passed"]]


def log_entry(failure: dict, phase: str, retry_job: str) -> str:
    return json.dumps({
        "logged_at": datetime.now(UTC).isoformat(),
        "phase": phase,
        "job": failure["job"],
        "trial": failure["trial"],
        "failed_checks": [name for name, ok in failure["checks"].items() if not ok],
        "details": {k: failure["details"].get(k)
                    for k in ("exception", "infra_error_texts", "error") if k in failure["details"]},
        "retried_in_job": retry_job,
    })


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--jobs-dir", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--log", type=Path, help="append one line per retried trial (with --run)")
    parser.add_argument("--n-concurrent", type=int, default=4)
    parser.add_argument("--harness", nargs="+", choices=HARNESSES, help="only these harnesses")
    parser.add_argument("--run", action="store_true", help="run the retry jobs")


def run(args: argparse.Namespace) -> None:
    if args.run and not args.log:
        args.parser.error("--run needs --log: every retry is logged")

    failures = [f for f in failed_trials(args.jobs_dir, args.phase)
                if not args.harness or harness_of(f["key"][0].split("__")[0]) in args.harness]
    for f in failures:
        print(f"{f['job']}/{f['trial']}: {', '.join(n for n, ok in f['checks'].items() if not ok)}")
    print(f"{len(failures)} infrastructure failure(s) in phase {args.phase}")
    if not args.run or not failures:
        return

    attempt = 1 + len({p.name.split("-retry", 1)[1].split("-", 1)[0]
                       for p in args.jobs_dir.glob(f"{args.phase}-retry*__*")})
    groups: dict[tuple[str, str], list[dict]] = {}
    for f in failures:
        groups.setdefault(f["key"][:2], []).append(f)
    launches = []
    for (harness_setting, model), items in sorted(groups.items()):
        harness, setting = harness_setting.split("__")
        tasks = tuple(sorted(i["key"][2] for i in items))
        job = Job(f"{args.phase}-retry{attempt}-{model}", harness, setting, (model,), tasks)
        config = job_config(job, args.jobs_dir, args.n_concurrent)  # acpx from $ACPX_PACKAGE
        with args.log.open("a") as log:
            log.writelines(log_entry(i, args.phase, job.name) + "\n" for i in items)
        path = args.jobs_dir / "configs" / f"{job.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2) + "\n")
        record = provenance(config["user_agent"]["bridge"]["kwargs"]["acpx_package"])
        path.with_suffix(".provenance.json").write_text(json.dumps(record, indent=2) + "\n")
        launches.append(Launch(job.name, path, args.jobs_dir / f"{job.name}.log"))
    run_jobs(launches, args.jobs_dir, max_jobs=1, infra_log=args.log)
