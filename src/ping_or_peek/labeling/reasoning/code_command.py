"""Code the agents' stated reasons for asking or taking the shortcut: `ping-or-peek code`.

    ping-or-peek code <trial-or-jobs-dir>... --run NAME [--model gpt-6.1-sol]
        [--limit N] [--workers 4] [--only SUBSTRING]
        --jobs-root DIR (--superseded JSON | --no-superseded)

Every output holds agent text, so the run folder must be outside the repository; run folders go
under $WS6_PRIVATE_ROOT, which has no default. Episodes are keyed by their path under
--jobs-root, and the jobs a superseded.json lists are not data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ping_or_peek.config import REPO
from ping_or_peek.dataset.export import read_superseded, superseded_options
from ping_or_peek.dataset.trial import find_trials

from . import label, transcript


def _private_root() -> Path:
    """$WS6_PRIVATE_ROOT, read after the CLI loads the repo's .env. Exits, naming it, when unset."""
    root = os.environ.get("WS6_PRIVATE_ROOT")
    if not root:
        raise SystemExit("set WS6_PRIVATE_ROOT: the coder's private root has no default")
    return Path(root)


def _private(path: Path) -> Path:
    path = path.resolve()
    if path.is_relative_to(REPO):
        sys.exit(f"refusing to write agent text inside the repository: {path}")
    return path


def _trials(
    paths: list[str], only: str | None, jobs_root: Path, superseded: set[str]
) -> list[Path]:
    """The finished trials under `paths` (`trial.find_trials`: each once, by its resolved path)
    with an ACP export, outside `superseded` jobs, each keyed by its path under the resolved
    `jobs_root`. Raises ValueError, naming the trial, for one outside `jobs_root`."""
    found = [t for t in find_trials(paths) if only is None or only in str(t)]
    missing = [t for t in found if not (t / "agent" / "bridge-trajectory.json").exists()]
    for t in missing:  # an infrastructure failure or a trial still being written
        print(f"skipped (no ACP export): {t}", file=sys.stderr)
    dropped = [t for t in found if t.parent.name in superseded]
    for t in dropped:  # harbor/superseded.json: replaced runs, not data
        print(f"skipped (superseded job): {t}", file=sys.stderr)
    kept = [t for t in found if t not in missing and t not in dropped]
    for t in kept:
        if not t.is_relative_to(jobs_root.resolve()):
            raise ValueError(f"{t}: outside --jobs-root {jobs_root}, which keys every episode")
    return kept


def _jobs(parser: argparse.ArgumentParser) -> None:
    """The jobs root that keys every episode, and the superseded jobs or the explicit none."""
    parser.add_argument("--jobs-root", type=Path, required=True, help="the Harbor jobs root")
    superseded_options(parser)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--run", required=True)
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only")
    _jobs(parser)


def run(args: argparse.Namespace) -> None:
    args.jobs_root = args.jobs_root.resolve()  # as `_trials` resolves the trials it keys

    superseded = read_superseded(args.superseded) if args.superseded else set()
    trials = _trials(args.paths, args.only, args.jobs_root, superseded)
    config = label.CoderConfig(model=args.model, reasoning_effort=args.effort)
    run_root = _private(_private_root() / "labels" / args.run)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "run.json").write_text(
        json.dumps({"coder": config.public(), **label.provenance()}, indent=2) + "\n"
    )
    if args.limit:
        trials = trials[: args.limit]
    client = label.make_client(config)
    done, failed = 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for t in trials:
            tr = transcript.from_trial(t, args.jobs_root)

            def rebuild(limit: int, t: Path = t, root: Path = args.jobs_root):
                return transcript.from_trial(t, root, output_limit=limit)

            futures[
                pool.submit(
                    label.code_episode,
                    tr,
                    out_root=run_root,
                    config=config,
                    client=client,
                    rebuild=rebuild,
                )
            ] = tr
        for future in as_completed(futures):
            result = future.result()
            ok = result.get("status") == "ok"
            done += ok
            if not ok:
                failed.append(futures[future].episode)
            print(f"{'ok' if ok else 'FAILED'} {futures[future].episode}", flush=True)
    print(f"coded {done}, failed {len(failed)}; records in {run_root}")
    _exit_on_failures(failed)


def _exit_on_failures(failed: list[str]) -> None:
    """Exits nonzero, naming them, when any episode failed to code: a run is not done until each
    is coded."""
    if failed:
        raise SystemExit(f"{len(failed)} episodes failed to code: {', '.join(sorted(failed))}")
