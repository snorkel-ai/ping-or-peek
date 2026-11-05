"""Check the built dataset before anything is uploaded, after `build` and `card`.

- 1,620 rows, one per episode, the published columns in order, and the columns taken from the
  export identical to it;
- every trace a valid ATIF-v1.8 trajectory (Harbor's `Trajectory`) that reads back;
- the build equal to the latest published revision, its card and shards byte for byte
  (`byte_differences`), fetched from the Hub when the local Hugging Face cache lacks them;
- each row's `task.user_facts` in the task's own key order, as the labelers saw them;
- `csv_changed` equal to the export's `csv_modified_at_end` row by row (1,435 of 1,620, no nulls),
  and Figure 5's guard: of the rollouts it draws as changing nothing (185), none changed a CSV;
- `verbalized_test_awareness` 32 of 1,620, by model as the reasoning codes give it;
- no vendor prompt text (every line of 50+ characters in the captured system prompts and tool
  descriptions, `--references`, less the task's own lines, and a few hallmark strings), no
  local paths, and a card with no internal references or banned wording that documents every
  column and names the three settings;
- the size.

    uv run --no-sync ping-or-peek dataset check --data <dataset folder> \\
        --records <records> --export <folder with the export's episodes.parquet> \\
        --references <the vendor captures, <case>/turn<N>.json>

The vendor captures are the requests each harness sent its model in a capture outside Harbor,
against a stand-in endpoint; they don't ship with the repository, so the check has no default
folder for them.

Writes check_report.json to --records and exits 1 if any check fails. Fetching the published
files needs a token while the dataset is private.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from pydantic import ValidationError

from ping_or_peek.config import AWAY, MODELS, PATHS
from ping_or_peek.config import SETTINGS as SETTING_CODES
from ping_or_peek.labeling import atif
from ping_or_peek.runs.run import PASSAGE

from . import build, data
from .build import KEEP, TEST_AWARENESS
from .card import SETTINGS
from .episode import Task

HALLMARKS = (
    "You are Claude Code",
    "Anthropic's official CLI",
    "You are Codex",
    "You are a coding agent",
    "You are opencode",
    "You are OpenCode",
    "<system-reminder>",
    "<environment_context>",
    "base_instructions",
    "input_schema",
)
INTERNAL = re.compile(
    rf"\bdecision \d+|\bworkstream|\bWS\d|\bHarbor\b|\bacpx\b|\bexport\b|{build.EXPORT_SHA256[:8]}"
)
BANNED = re.compile(r"reward[- ]hack|\bcheat|/private/|\bcmglaze\b|\.private\.", re.IGNORECASE)
EXPORT_COLUMNS = (
    "episode_id",
    "model",
    "harness",
    "agent",
    "setting",
    "condition",
    "seed",
    "asked_user_before_edit",
    "scorer_used_to_edit_csv",
)
CSV_CHANGED = 1435  # csv_modified_at_end true in the export, of 1,620
# The notebook's Figure 5 draws these as changing nothing: a no-change path, or asked with
# the user away. There are 185, and none may have changed a CSV.
NO_CHANGE, ASKED, CHANGED_NOTHING = PATHS[3:], PATHS[0], 185
# The post's "2% of traces": the counts the coordinator checked against the reasoning codes.
TEST_AWARENESS_BY_MODEL = dict(zip(MODELS, (3, 0, 0, 13, 10, 6), strict=True))


def strings(value: Any):
    """Every string inside a value, nested lists and dicts included, keys and values."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for v in value:
            yield from strings(v)
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from strings(k)  # a key can hold data too: a path, a credential
            yield from strings(v)


def byte_differences(folder: Path, published: dict[str, Path]) -> list[str]:
    """Each file in which the build in `folder` differs from the `published` files, byte for byte:
    its card (`README.md`) and shards (`data/*.parquet`), named by their path in the dataset."""
    built = ["README.md", *(f"data/{p.name}" for p in sorted((folder / "data").glob("*.parquet")))]
    return sorted(
        [f"{name}: only in the build" for name in built if name not in published]
        + [f"{name}: only in the published revision" for name in published if name not in built]
        + [
            f"{name}: differs"
            for name in built
            if name in published and (folder / name).read_bytes() != published[name].read_bytes()
        ]
    )


def reference_lines(root: Path) -> set[str]:
    """Lines of 50+ characters from the vendors' system prompts and tool descriptions, in the
    captures under `root` (`<case>/turn<N>.json`). Raises ValueError, naming `root`, when it holds
    no capture: the guard would then find nothing."""
    turns = sorted(root.glob("*/turn*.json"))
    if not turns:
        raise ValueError(f"--references {root}: no vendor captures (<case>/turn<N>.json) in it")
    lines: set[str] = set()
    for turn in turns:
        capture = json.loads(turn.read_text())
        texts = list(strings(capture.get("system")))
        for tool in (capture.get("tools") or {}).values():
            texts.append(tool.get("description") or "")
        for text in texts:
            lines |= {line.strip() for line in text.splitlines() if len(line.strip()) >= 50}
    return lines


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, required=True, help="the dataset folder")
    parser.add_argument("--records", type=Path, required=True, help="where the report goes")
    parser.add_argument(
        "--export", type=Path, required=True, help="the folder with the export's episodes.parquet"
    )
    parser.add_argument(
        "--references",
        type=Path,
        required=True,
        help="the vendor captures (<case>/turn<N>.json), kept outside the repository",
    )


def run(args: argparse.Namespace) -> None:
    # The CLI has loaded `.env`: HF_TOKEN, for fetching the published card, may come from it.
    vendor_lines = reference_lines(args.references)  # before anything is read or written
    report: dict[str, Any] = {}

    def record(name: str, ok: bool, detail: Any) -> None:
        report[name] = {"pass": bool(ok), "detail": detail}

    def finish() -> None:
        """Write the report, print the verdicts, and exit 1 if any check failed."""
        report["pass"] = all(
            v["pass"] for v in report.values() if isinstance(v, dict) and "pass" in v
        )
        args.records.mkdir(parents=True, exist_ok=True)
        (args.records / "check_report.json").write_text(json.dumps(report, indent=1) + "\n")
        verdicts = {
            k: v["pass"] if isinstance(v, dict) else v for k, v in report.items() if k != "size"
        }
        print(json.dumps(verdicts, indent=1))
        if "size" in report:
            size = report["size"]
            print(
                f"size: {size['bytes']:,} bytes on disk, {size['decoded text bytes']:,} bytes of text"
            )
        sys.exit(0 if report["pass"] else 1)

    # Everything below reads only what these first checks guarantee: the build's files, its
    # structure (`build.SCHEMA`: columns, nested fields, required values) and the pinned export.
    shards = sorted((args.data / "data").glob("*.parquet"))
    table, problems = build.read_shards(args.data / "data")
    record(
        "the build's shards (data/*.parquet)",
        bool(shards),
        str(args.data / "data") if shards else problems,
    )
    if shards:
        record(
            "the build's schema: columns, nested fields and required values",
            not problems,
            problems,
        )
        if not problems:
            rows = table.to_pylist()
            # Every trace is read below, so each must be a valid ATIF-v1.8 trajectory that reads
            # back (`atif.from_trajectory`) before anything reads one. Imported here, not with the
            # module: validating needs Harbor, which CI's sync leaves out.
            from harbor.models.trajectories import Trajectory

            invalid = []
            for r in rows:
                try:
                    Trajectory.model_validate_json(r["trace"])
                    atif.from_trajectory(r["trace"])
                except ValidationError as error:  # its text quotes the recorded input
                    invalid.append(
                        f"{r['episode_id']}: not ATIF-v1.8 at {atif.invalid_fields(error)}"
                    )
                except ValueError as error:  # atif's refusals name no recorded value
                    invalid.append(f"{r['episode_id']}: {str(error)[:200]}")
            record(
                "every trace is a valid ATIF-v1.8 trajectory that reads back",
                not invalid,
                invalid[:20],
            )
    card_file = args.data / "README.md"
    record("the build's card (README.md)", card_file.is_file(), str(card_file))
    export_file = args.export / "episodes.parquet"
    pinned = export_file.is_file() and build.sha256(export_file) == build.EXPORT_SHA256
    record("the export is the pinned one", pinned, str(export_file))
    if not all(v["pass"] for v in report.values()):
        finish()
    built = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    export = pq.read_table(export_file)
    record("1,620 rows", len(rows) == 1620, len(rows))
    ids = [r["episode_id"] for r in rows]
    record(
        "one row per episode",
        len(set(ids)) == len(ids) and set(ids) == set(export.column("episode_id").to_pylist()),
        {"unique": len(set(ids))},
    )
    if not report["one row per episode"]["pass"]:
        finish()  # the checks below match the build's rows to the export's, one to one
    by_id = {r["episode_id"]: r for r in rows}
    order = export.column("episode_id").to_pylist()
    differ = [
        c for c in EXPORT_COLUMNS if [by_id[e][c] for e in order] != export.column(c).to_pylist()
    ]
    record("columns taken from the export identical", not differ, differ)

    # The latest published revision, fetched when the cache lacks it: a rebuild is the same files.
    latest = data.Dataset(data.REPO, ("README.md", *data.SOURCE.files))
    published_files = dict(zip(latest.files, data.files(latest, fetch=True), strict=True))
    found = byte_differences(args.data, published_files)
    record(
        f"the build equals the latest published revision, {data.revision(latest)}, byte for byte",
        not found,
        found,
    )
    # The one intended change outside the trace: the facts in the order the labelers saw them.
    unordered = [
        r["episode_id"]
        for r in rows
        if r["task"]["user_facts"]
        != json.dumps(Task.regenerate(int(r["seed"]), r["condition"]).facts, ensure_ascii=False)
    ]
    record(
        "task.user_facts in the simulated user's key order",
        not unordered,
        {"not in the task's order": unordered[:20]},
    )
    card = card_file.read_text()

    changed = [by_id[e]["csv_changed"] for e in order]
    record(
        "csv_changed equals csv_modified_at_end",
        changed == export.column("csv_modified_at_end").to_pylist()
        and None not in changed
        and sum(changed) == CSV_CHANGED,
        {
            "true": sum(c is True for c in changed),
            "false": sum(c is False for c in changed),
            "nulls": sum(c is None for c in changed),
        },
    )
    noteless = sorted({r["setting"] for r in rows if r["task"]["project_note"] is None})
    record(
        "a project note in each setting that carries one, and none in the other",
        noteless == sorted(set(SETTING_CODES) - set(PASSAGE))
        and all(r["task"]["project_note"] is not None for r in rows if r["setting"] in PASSAGE),
        {
            "rows without a note": sum(r["task"]["project_note"] is None for r in rows),
            "their settings": noteless,
        },
    )
    nothing = [
        r for r in rows if r["path"] in NO_CHANGE or (r["path"] == ASKED and r["condition"] == AWAY)
    ]
    record(
        "Figure 5's changed-nothing rollouts changed no CSV",
        len(nothing) == CHANGED_NOTHING and not any(r["csv_changed"] is not False for r in nothing),
        {
            "changed nothing": len(nothing),
            "csv_changed true or null": sum(r["csv_changed"] is not False for r in nothing),
        },
    )

    by_model = built.groupby("model")[TEST_AWARENESS].sum().astype(int).to_dict()
    aware = {
        "true": int(built[TEST_AWARENESS].sum()),
        "of": len(built),
        "nulls": int(built[TEST_AWARENESS].isna().sum()),
        "by model": by_model,
    }
    record(
        "verbalized_test_awareness 32/1,620",
        aware["true"] == 32
        and aware["of"] == 1620
        and not aware["nulls"]
        and by_model == TEST_AWARENESS_BY_MODEL,
        aware,
    )

    # Vendor prompt text, local paths, and the card's wording.
    decoded = [{**r, "trace": json.loads(r["trace"])} for r in rows]
    own = {line.strip() for r in rows for text in strings(r["task"]) for line in text.splitlines()}
    fingerprints = vendor_lines - own
    found_lines, hallmarks, local = set(), {}, 0
    for text in [*(s for r in decoded for s in strings(r)), card]:
        found_lines |= {line.strip() for line in text.splitlines()} & fingerprints
        for mark in HALLMARKS:
            if mark in text:
                hallmarks[mark] = hallmarks.get(mark, 0) + 1
        local += text.count(build.LOCAL_PATH)
    record(
        "no vendor prompt text",
        fingerprints and not found_lines and not hallmarks,
        {
            "reference lines checked": len(fingerprints),
            "lines found": sorted(found_lines)[:20],
            "hallmarks found": hallmarks,
        },
    )
    record("no local paths", local == 0, {f"{build.LOCAL_PATH} occurrences": local})
    wording = sorted(set(INTERNAL.findall(card)) | set(BANNED.findall(card)))
    record("card free of internal references and banned wording", not wording, wording)
    undocumented = [c for c in KEEP if f"`{c}`" not in card]
    unnamed = [name for _, name, _ in SETTINGS if name not in card]
    record(
        "card documents every column and names the three settings",
        not undocumented and not unnamed,
        {"undocumented": undocumented, "settings not named": unnamed},
    )
    files = sorted(p for p in args.data.rglob("*") if p.is_file())
    report["size"] = {
        "bytes": sum(p.stat().st_size for p in files),
        "files": {str(p.relative_to(args.data)): p.stat().st_size for p in files},
        "decoded text bytes": sum(len(s.encode()) for r in decoded for s in strings(r)),
    }
    finish()
