"""Progress of a phase: episodes done per job, durations, and mechanical-check failures.

    uv run ping-or-peek progress --jobs-dir JOBS --phase collection [--cache progress.json]

Checks each finished trial once (results are cached), so it is cheap to run every few hours.
Prints a markdown summary for the status file.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

from ping_or_peek.dataset.trial import read_json
from ping_or_peek.runs.checks import check_trial
from ping_or_peek.runs.run import phase_jobs


def setup_seconds(trial: Path) -> float | None:
    """Agent, bridge and user setup: from the agent's setup to the start of the episode."""
    result = read_json(trial / "result.json", trial.name)
    start = (result.get("agent_setup") or {}).get("started_at")
    end = (result.get("agent_execution") or {}).get("started_at")
    if not (start and end):
        return None
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--jobs-dir", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--cache", type=Path, help="JSON file of earlier check results")


def run(args: argparse.Namespace) -> None:
    cache = json.loads(args.cache.read_text()) if args.cache and args.cache.exists() else {}
    lines, failures, durations, setups, done_total, planned_total = [], [], [], [], 0, 0
    for job in phase_jobs(args.phase):
        planned = len(job.models) * len(job.tasks)
        finished = sorted(p.parent for p in (args.jobs_dir / job.name).glob("*/result.json"))
        for trial in finished:
            if str(trial) not in cache:
                result = check_trial(trial)
                cache[str(trial)] = {
                    "passed": result["passed"],
                    "duration_s": result["details"].get("duration_s"),
                    "failed": [k for k, ok in result["checks"].items() if not ok],
                }
            record = cache[str(trial)]
            if not record["passed"]:
                failures.append(f"{job.name}/{trial.name}: {', '.join(record['failed'])}")
            if record.get("duration_s"):
                durations.append(record["duration_s"])
            if (setup := setup_seconds(trial)) is not None:
                setups.append(setup)
        done_total += len(finished)
        planned_total += planned
        lines.append(f"| {job.name} | {len(finished)}/{planned} |")
    if args.cache:
        args.cache.write_text(json.dumps(cache, indent=1) + "\n")
    median = f"{statistics.median(durations):.0f} s" if durations else "–"
    setup = f"median {statistics.median(setups):.0f} s, max {max(setups):.0f} s" if setups else "–"
    print(
        f"{done_total}/{planned_total} episodes done; median agent phase {median}; "
        f"setup {setup}; {len(failures)} mechanical failure(s).\n"
    )
    print("| Job | Done |\n|---|---|\n" + "\n".join(lines))
    if failures:
        print(
            "\nMechanical failures (infrastructure; retried separately):\n"
            + "\n".join(f"- {f}" for f in failures)
        )
