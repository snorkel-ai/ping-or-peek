"""Label finished trials with an LLM: `ping-or-peek label`.

    ping-or-peek label <jobs_or_trial_dir>... --run s0 --out <results.json> \\
        --retry-log <jsonl> | --no-retry-log

Private label records go to --label-root or $ASK_OR_SHORTCUT_LABEL_ROOT, which has no default.
`label` reads the annotator's key from the environment or the repo's .env.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from ping_or_peek.config import REPO
from ping_or_peek.dataset.export import retry_options
from ping_or_peek.dataset.trial import find_trials

from .label_run import REASONING_EXCLUDED, REASONING_INCLUDED
from .labels import AGENT_RULE


def add_arguments(parser: argparse.ArgumentParser) -> None:
    # The private record store (annotator requests and responses, which can quote agent
    # reasoning) lives outside the repository, where the user says: it has no default.
    label_root = os.environ.get("ASK_OR_SHORTCUT_LABEL_ROOT")
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Harbor job roots, trial directories, or quoted globs of job roots",
    )
    parser.add_argument(
        "--run", required=True, help="Name of this labeling run, e.g. s0, pilot, collection"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="JSON file for the per-trial results"
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        default=Path(label_root) if label_root else None,
        help="Private record store; $ASK_OR_SHORTCUT_LABEL_ROOT when not given",
    )
    parser.add_argument(
        "--persona-policy",
        default="auto",
        choices=["auto", "reactive_v2"],
        help="auto: per trial, from its persona text",
    )
    parser.add_argument(
        "--persona-file",
        type=Path,
        help="The simulated user's persona text, which must be the one each trial's lock names "
        "(by sha256); without it, each trial's is found in this repository by that digest",
    )
    parser.add_argument("--setting", help="S, A or B, when the task metadata does not say")
    parser.add_argument("--annotator", choices=["portkey", "openai"], default="portkey")
    parser.add_argument(
        "--annotator-model",
        default="gpt-6.1-sol",
        help="The labeler: an OpenAI model name, or a Portkey route such as @provider/model "
        "(default gpt-6.1-sol). Each labeler's "
        "records are kept apart under <label root>/<run>/<model>-<effort>, and every labeler in "
        "a run must see identical inputs",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        help="The labeler's reasoning effort, or none to send no effort parameter",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--reasoning",
        choices=[REASONING_EXCLUDED, REASONING_INCLUDED],
        default=REASONING_EXCLUDED,
        help="Primary labels read observable behavior only (exclude, the default); the "
        "sensitivity pass also reads the agent's reasoning (include, with --scorer-edit-only)",
    )
    parser.add_argument(
        "--scorer-edit-only",
        action="store_true",
        help="Run only the scorer-edit stage (the reasoning sensitivity pass)",
    )
    parser.add_argument(
        "--watch",
        type=int,
        metavar="SECONDS",
        help="Keep labeling new trials as they finish, rescanning every SECONDS, until stopped",
    )
    retry_options(
        parser,
        "The experiment's infra_retries.jsonl: the attempts it names were rerun, and are "
        "flagged as infrastructure failures, as the export lists them",
    )


def run(args: argparse.Namespace) -> int:
    if args.label_root is None:
        args.parser.error(
            "set --label-root or ASK_OR_SHORTCUT_LABEL_ROOT: the label store has no default"
        )

    from ping_or_peek.dataset.export import read_retry_log

    from .annotator import Annotator, AnnotatorConfig, write_json
    from .label_run import flag_logged_retries, label_trials, result_order

    effort = None if args.reasoning_effort == "none" else args.reasoning_effort
    config = (
        AnnotatorConfig.portkey(args.annotator_model, effort)
        if args.annotator == "portkey"
        else AnnotatorConfig(model=args.annotator_model, reasoning_effort=effort)
    )
    persona = {
        "policy": args.persona_policy,
        "instructions": args.persona_file.read_text() if args.persona_file else None,
    }
    annotator = Annotator(config)
    labeled: dict[str, dict] = {}  # with --watch, trials already labeled by this process
    code = _code_version()  # the code this process loaded, not whatever is checked out later
    if args.retry_log:
        read_retry_log(args.retry_log)  # a missing log fails here, before anything is labeled
    while True:
        waiting = bool(args.watch)  # a job label waits on may have no finished trial yet
        trials = find_trials(args.paths, waiting=waiting)
        results, failures = label_trials(
            [t for t in trials if str(t) not in labeled],
            root=args.label_root / args.run / config.name,
            annotator=annotator,
            persona=persona,
            setting=args.setting,
            workers=args.workers,
            reasoning=args.reasoning,
            scorer_edit_only=args.scorer_edit_only,
        )
        labeled.update({r["trial_path"]: r for r in results})
        # Read on every scan: the log grows during a run.
        logged = read_retry_log(args.retry_log) if args.retry_log else {}
        write_json(
            args.out,
            {
                "run": args.run,
                "note": "Private: annotator rationales may paraphrase agent reasoning. "
                "Do not commit or publish.",
                "persona_policy": args.persona_policy,
                "reasoning_in_evidence": args.reasoning,
                "scorer_edit_only": args.scorer_edit_only,
                "asking_rule": AGENT_RULE,
                "annotator": config.public(),
                "code": code,  # a relabel needs the same code to rebuild the inputs
                "trials": flag_logged_retries(sorted(labeled.values(), key=result_order), logged),
                "failures": [{k: v for k, v in f.items() if k != "traceback"} for f in failures],
            },
        )
        print(json.dumps({"labeled": len(labeled), "failed": len(failures), "out": str(args.out)}))
        if not args.watch:
            return 1 if failures else 0
        time.sleep(args.watch)


def _code_version() -> dict[str, Any]:
    """The code's git commit, and whether the code or rubrics that build the labels (the dataset
    and labeling packages and the config) had uncommitted changes."""
    import subprocess

    def git(*args: str) -> str:
        run = subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False
        )
        return run.stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD") or None,
        "code_dirty": bool(
            git(
                "status",
                "--porcelain",
                "--",
                "src/ping_or_peek/dataset/*.py",
                "src/ping_or_peek/labeling/*.py",
                "src/ping_or_peek/labeling/rubrics",
                "src/ping_or_peek/config.py",
            )
        ),
    }
