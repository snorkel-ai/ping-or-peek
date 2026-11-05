"""Build `episodes.parquet` from the label results: `ping-or-peek export`.

    ping-or-peek export <results.json>... --out-dir <folder> \\
        --retry-log <jsonl> | --no-retry-log  --superseded <json> | --no-superseded

The result files are the ones `ping-or-peek label` writes. The folder also gets the export's data
card, `DATA_CARD.md`, its provenance and its list of infrastructure attempts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .export import retry_options, superseded_options


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("results", nargs="+", type=Path, help="Result files written by `label`")
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="The folder for the export; it has no default"
    )
    retry_options(parser, "The experiment's infra_retries.jsonl")
    superseded_options(parser)
    parser.add_argument(
        "--sensitivity",
        nargs="+",
        type=Path,
        default=[],
        help="Result files of the reasoning sensitivity pass (label --reasoning include "
        "--scorer-edit-only)",
    )


def run(args: argparse.Namespace) -> int:
    return _export(
        args.results,
        args.out_dir,
        args.retry_log,
        args.sensitivity,
        args.superseded,
    )


def _export(
    result_files: list[Path],
    out_dir: Path,
    retry_log: Path | None,
    sensitivity_files: list[Path],
    superseded_file: Path | None = None,
) -> int:
    from ping_or_peek.labeling.annotator import write_json

    from .export import (
        build_rows,
        code_hashes,
        data_card,
        read_retry_log,
        read_superseded,
        write_parquet,
    )

    trials, annotators = [], set()
    for path in result_files:
        value = json.loads(path.read_text())
        trials += value["trials"]
        annotators.add(json.dumps(value["annotator"], sort_keys=True))
    settings = {t["trial"]: t["setting"] for t in trials}
    sensitivity = None
    if sensitivity_files:
        sensitivity = {
            t["trial"]: t["labels"]
            for path in sensitivity_files
            for t in json.loads(path.read_text())["trials"]
        }
    rows, infrastructure = build_rows(
        trials,
        settings,
        read_retry_log(retry_log) if retry_log else {},
        sensitivity,
        read_superseded(superseded_file) if superseded_file else set(),
    )
    provenance = {
        "inputs": [
            {"trial": t["trial"], "path": t["trial_path"], **t["provenance"]} for t in trials
        ],
        "label_result_files": [str(p) for p in result_files],
        "sensitivity_result_files": [str(p) for p in sensitivity_files],
        "annotators": [json.loads(a) for a in sorted(annotators)],
        "code_sha256": code_hashes(),
    }
    provenance["parquet_sha256"] = write_parquet(
        rows, out_dir / "episodes.parquet", {"rows": len(rows)}
    )
    write_json(out_dir / "infrastructure_attempts.json", infrastructure)
    write_json(out_dir / "provenance.json", provenance)
    (out_dir / "DATA_CARD.md").write_text(data_card(rows, infrastructure, provenance))
    print(json.dumps({"rows": len(rows), "infrastructure_attempts": len(infrastructure)}))
    return 0
