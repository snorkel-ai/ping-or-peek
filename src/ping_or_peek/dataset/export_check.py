"""Check A of the safety net: the export rebuilds exactly from the raw runs.

It labels the trials that `provenance.json` lists again, at this checkout's code, reusing every
annotator answer saved in the label store and refusing to call a model: parsing, measuring and
labeling run on the raw Harbor trials, and an input that changed since it was labeled fails
(`labels.run_stage`). Then `ping-or-peek export` builds `episodes.parquet` from those results,
and the check compares it with the published export, value for value: the same arrow
schema, rows in the same order, columns, dtypes and values, with nulls kept apart from NaN.
Parquet records the library that wrote it, so the check compares the data, never the file's hash;
it checks the reference's hash before using it. The label store reuses a saved answer only for
the identical request, so a request that differs stops the relabel, naming the stage and trial;
the check also compares every rebuilt request's text with the one the labeler sent, which reports
an input sent but not rebuilt or rebuilt but never sent, and the reasoning
coder's transcript of each trial, built from the raw trial, with the one the coder sent. Both are built by the labelers' own code
(`annotator.request_messages`, `reasoning.label.clipped`), though not through their command
lines. A match proves that this repository's labelers send exactly the inputs they sent before.

Check B, the labelers' inputs rendered from the dataset's trace, is `labeling.labeler_inputs`.

It reads private local data, so CI can't run it; its tests run on synthetic fixtures:

    uv run --no-sync ping-or-peek safety-net export --provenance <export>/provenance.json \\
        --label-root <labels> --coder-run <reasoning labels>/<run> \\
        --retry-log <infra_retries.jsonl> --superseded <superseded.json> --reference-dir <export>

Every path is an argument, and a missing one fails naming it. The check exits 0 only when
everything matches. Nothing here writes to the label store, the trials or the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ping_or_peek import cli
from ping_or_peek.config import PERSONA_FILE
from ping_or_peek.labeling.trace_inputs import outcome_stages

from .build import EXPORT_SHA256

# The reference export: the full local export the dataset is built from.
REFERENCE_SHA256 = {"episodes.parquet": EXPORT_SHA256}
KEY = "episode_id"


# --- Comparisons ----------------------------------------------------------------------------


def first_difference(a: str, b: str) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def _same(x: Any, y: Any) -> bool:
    """Whether two cells are the same value of the same type; a null equals only a null of its
    kind (None, NaN or pandas' NA)."""
    import pandas as pd

    if type(x) is not type(y):
        return False
    if pd.api.types.is_scalar(x) and pd.isna(x):
        return bool(pd.isna(y))
    return bool(x == y)


def frame_differences(reference, rebuilt, key: str) -> list[str]:
    """How `rebuilt` differs from `reference`: columns and their order, dtypes, rows and their
    order (by `key`), and each column's values. Empty only when pandas' exact
    `assert_frame_equal` passes too."""
    import pandas as pd

    out = []
    if list(reference.columns) != list(rebuilt.columns):
        out.append(
            f"columns: only in the reference {sorted(set(reference.columns) - set(rebuilt.columns))}, "
            f"only rebuilt {sorted(set(rebuilt.columns) - set(reference.columns))}, "
            f"same order: {list(reference.columns) == list(rebuilt.columns)}"
        )
    if len(reference) != len(rebuilt):
        out.append(f"rows: {len(reference)} in the reference, {len(rebuilt)} rebuilt")
    keys_equal = len(reference) == len(rebuilt) and reference[key].tolist() == rebuilt[key].tolist()
    if not keys_equal and sorted(reference[key]) == sorted(rebuilt[key]):
        out.append(f"row order: the same {key} values in a different order")
    for column in [c for c in reference.columns if c in rebuilt.columns]:
        a, b = reference[column], rebuilt[column]
        if a.dtype != b.dtype:
            out.append(f"column {column}: dtype {a.dtype} in the reference, {b.dtype} rebuilt")
        if not keys_equal:
            continue
        unequal = [
            str(k) for k, x, y in zip(reference[key], a.tolist(), b.tolist()) if not _same(x, y)
        ]
        if unequal:
            out.append(f"column {column}: {len(unequal)} values differ, at {key} {unequal[:5]}")
    if not out:
        try:
            pd.testing.assert_frame_equal(reference, rebuilt, check_exact=True)
        except AssertionError as error:
            out.append(f"assert_frame_equal: {error}")
    return out


def parquet_differences(reference: Path, rebuilt: Path, key: str) -> list[str]:
    """How the parquet file `rebuilt` differs from `reference`: its arrow schema (column types,
    without the writer's metadata), then `frame_differences` on frames that keep arrow's types.
    pandas' default reading would turn an int64 column with nulls into float64, and a null into
    NaN, hiding a changed type."""
    import pandas as pd
    import pyarrow.parquet as pq

    out = []
    expected, found = pq.read_schema(reference), pq.read_schema(rebuilt)
    if not expected.equals(found, check_metadata=False):
        changed = [
            f"{f.name}: {f.type} in the reference, "
            f"{found.field(f.name).type if f.name in found.names else 'absent'} rebuilt"
            for f in expected
            if f.name not in found.names or found.field(f.name).type != f.type
        ]
        out.append(f"arrow schema differs: {changed or 'in column order or added columns'}")
    frames = [pd.read_parquet(p, dtype_backend="pyarrow") for p in (reference, rebuilt)]
    return out + frame_differences(*frames, key)


# --- Inputs ---------------------------------------------------------------------------------


def check_reference(directory: Path, pins: dict[str, str]) -> None:
    """Raises ValueError, naming the file, when a reference file's sha256 isn't its pin."""
    for name, pin in pins.items():
        found = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if found != pin:
            raise ValueError(f"{directory / name} has sha256 {found}, not the published {pin}")


def require(paths: dict[str, Path]) -> None:
    """Raises FileNotFoundError naming every option whose path doesn't exist."""
    missing = [f"{option} {path}" for option, path in paths.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("missing input: " + "; ".join(missing))


def labeler_folder(label_root: Path, results: dict[str, Any]) -> Path:
    """One labeling run's labeler folder in the store, `<label root>/<run>/<model>-<effort>`."""
    from ping_or_peek.labeling.annotator import AnnotatorConfig

    folder = label_root / results["run"] / AnnotatorConfig(**results["annotator"]).name
    if not folder.is_dir():
        raise FileNotFoundError(f"--label-root: no labeler folder {folder}")
    return folder


def sent_messages(folder: Path, attempts_named: str = "*.json") -> tuple[str, str]:
    """The (system, user) messages a labeler sent, from its attempts in `folder`, the files
    matching `attempts_named` (the outcome labeler's `1.json`, the coder's `attempt-1.json`):
    every attempt starts with the same two. Raises FileNotFoundError naming the folder when it
    holds none, and ValueError when an attempt doesn't start with a system and a user message,
    or its attempts disagree."""
    attempts = sorted(folder.glob(attempts_named))
    if not attempts:
        raise FileNotFoundError(f"no saved attempts in {folder}")
    first = set()
    for path in attempts:
        messages = json.loads(path.read_text())["request_messages"][:2]
        if [m.get("role") for m in messages] != ["system", "user"]:
            raise ValueError(f"{path}: a request starts with a system and a user message")
        first.add(tuple(m["content"] for m in messages))
    if len(first) != 1:
        raise ValueError(f"{folder}: its attempts sent different first requests")
    return first.pop()


# --- Check A ----------------------------------------------------------------------------------


class _SavedAnswersOnly:
    """An annotator that has only the saved answers: a stage with none fails, never calls."""

    def __init__(self, config):
        self.config = config

    def label(self, **_: Any) -> dict[str, Any]:
        raise RuntimeError("no saved annotator answer for this input; the check makes no calls")


def _persona_mismatches(trials: list[Path], persona: str) -> dict[str, str]:
    """Trials whose locked persona isn't `persona` (`trial.check_persona`), with why. The lock
    names the persona by a path in the checkout the runs used, so the text comes from the
    repository."""
    from .trial import PersonaNotFound, Trial, check_persona

    found = {}
    for path in trials:
        try:
            check_persona(Trial(path, {}, {}), persona)
        except PersonaNotFound as error:
            found[path.name] = str(error)
    return found


def request_differences(fresh: Path, saved: Path, persona: str) -> list[str]:
    """The stage inputs of one labeler folder whose rebuilt request, in `fresh`, isn't the one
    the labeler sent, as `saved` keeps it in its attempts: differing in its system or user
    message, never rebuilt, or rebuilt but never sent. The request is built by the annotator's
    own `request_messages`, with the stage's rubric; `persona` picks the episode stage's. The
    request is the JSON text, whose key order counts."""
    from ping_or_peek.labeling import annotator

    stages = {stage.name: stage for stage in outcome_stages(persona).values()}
    out = []
    sent_keys = set()
    for attempts in sorted(saved.glob("*/attempts")):
        name = attempts.parent.name
        if name not in stages:
            raise ValueError(f"{saved}: {name} isn't an outcome-labeler stage")
        for folder in sorted(p for p in attempts.iterdir() if p.is_dir()):
            sent_keys.add((name, folder.name))
            payload = fresh / name / "payloads" / f"{folder.name}.json"
            if not payload.exists():
                out.append(f"{name}/{folder.name}: sent, but not rebuilt")
                continue
            sent = sent_messages(folder)
            rebuilt = tuple(
                m["content"]
                for m in annotator.request_messages(
                    stages[name].rubric, json.loads(payload.read_text())
                )
            )
            for role, was, now in zip(("system", "user"), sent, rebuilt):
                if was != now:
                    at = first_difference(was, now)
                    out.append(
                        f"{name}/{folder.name}: the {role} message differs at offset {at} "
                        f"(lengths {len(was)} sent, {len(now)} rebuilt)"
                    )
    for payload in sorted(fresh.glob("*/payloads/*.json")):
        if (payload.parent.parent.name, payload.stem) not in sent_keys:
            out.append(f"{payload.parent.parent.name}/{payload.stem}: rebuilt, but never sent")
    return out


def coder_differences(texts: dict[str, str], coder_run: Path) -> list[str]:
    """The reasoning coder's episodes whose rebuilt transcript, in `texts` by episode folder,
    isn't the one the coder sent, as `coder_run` keeps it, or whose system prompt changed."""
    from ping_or_peek.labeling.reasoning.label import system_prompt

    out = []
    for folder, text in sorted(texts.items()):
        system, sent = sent_messages(coder_run / folder, "attempt-*.json")
        if system != system_prompt():
            out.append(f"{folder}: the system prompt differs")
        if text != sent:
            at = first_difference(sent, text)
            out.append(
                f"{folder}: the transcript differs at offset {at} "
                f"(lengths {len(sent)} sent, {len(text)} rebuilt)"
            )
    return out


def _relabel(
    saved_file: Path,
    trials: list[Path],
    label_root: Path,
    work: Path,
    persona: str,
    retry_log: Path,
    workers: int,
) -> tuple[Path, list[str]]:
    """Label `trials` again as the run that wrote `saved_file` did, from a copy of its saved
    answers and the requests they were given (`run_stage` reuses an answer only for the identical
    request); the saved inputs stay behind, so every input is built afresh. Writes the results as
    `label` writes them. Returns the results file and `request_differences`.
    Raises when any trial fails, for example because its input changed."""
    from ping_or_peek.labeling.annotator import AnnotatorConfig, write_json
    from ping_or_peek.labeling.label_run import flag_logged_retries, label_trials, result_order

    from .export import read_retry_log

    saved = json.loads(saved_file.read_text())
    folder = labeler_folder(label_root, saved)
    copy = work / "labels" / saved["run"] / AnnotatorConfig(**saved["annotator"]).name
    shutil.copytree(folder, copy, ignore=shutil.ignore_patterns("payloads"))
    results, failures = label_trials(
        trials,
        root=copy,
        annotator=_SavedAnswersOnly(AnnotatorConfig(**saved["annotator"])),
        persona={"policy": saved["persona_policy"], "instructions": persona},
        workers=workers,
        reasoning=saved["reasoning_in_evidence"],
        scorer_edit_only=saved["scorer_edit_only"],
        asking_rule=saved["asking_rule"],
    )
    if failures:
        listed = "; ".join(f"{Path(f['trial_path']).name}: {f['error']}" for f in failures[:10])
        raise RuntimeError(f"{len(failures)} trials of run {saved['run']} failed: {listed}")
    out = work / f"{saved['run']}.json"
    keep = ("run", "persona_policy", "reasoning_in_evidence", "scorer_edit_only", "asking_rule")
    write_json(
        out,
        {
            **{k: saved[k] for k in keep},
            "annotator": saved["annotator"],
            "trials": flag_logged_retries(
                sorted(results, key=result_order), read_retry_log(retry_log)
            ),
            "failures": [],
        },
    )
    return out, request_differences(copy, folder, persona)


def export_check(
    provenance: Path,
    label_root: Path,
    coder_run: Path,
    retry_log: Path,
    superseded: Path,
    reference_dir: Path,
    workers: int,
) -> dict[str, list[str]]:
    """Check A. The differences found: each reference file's from the rebuilt export, each
    outcome-labeling run's requests, and the reasoning coder's transcripts."""
    from ping_or_peek.labeling import trace_inputs
    from ping_or_peek.labeling.reasoning import transcript
    from ping_or_peek.labeling.reasoning.label import safe_name

    check_reference(reference_dir, REFERENCE_SHA256)
    record = json.loads(provenance.read_text())
    trials = [Path(i["path"]) for i in record["inputs"]]
    persona = PERSONA_FILE.read_text()
    wrong = _persona_mismatches(trials, persona)
    if wrong:
        listed = dict(list(wrong.items())[:10])
        raise ValueError(f"{len(wrong)} trials' locked persona isn't {PERSONA_FILE}: {listed}")
    found: dict[str, list[str]] = {}
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)

        def relabel(files: list[str]) -> list[str]:
            paths = []
            for f in files:
                path, requests = _relabel(
                    Path(f), trials, label_root, work, persona, retry_log, workers
                )
                found[f"requests of {json.loads(Path(f).read_text())['run']}"] = requests
                paths.append(str(path))
            return paths

        primary = relabel(record["label_result_files"])
        sensitivity = relabel(record["sensitivity_result_files"])
        out = work / "export"
        argv = ["export", *primary, "--retry-log", str(retry_log), "--superseded", str(superseded)]
        argv += ["--out-dir", str(out), "--sensitivity", *sensitivity]
        if cli.main(argv) != 0:
            raise RuntimeError("ping-or-peek export failed")
        for name in REFERENCE_SHA256:
            found[name] = parquet_differences(reference_dir / name, out / name, KEY)
    # The coder ran on the same trials; its episode folders are named by job and trial.
    texts = {
        safe_name(transcript.episode_key(trial, trial.parent.parent)): trace_inputs.render_clipped(
            lambda limit, trial=trial: transcript.from_trial(trial, trial.parent.parent, limit)
        )
        for trial in trials
    }
    found["reasoning coder transcripts"] = coder_differences(texts, coder_run)
    return found


# --- Command line ---------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provenance", type=Path, required=True, help="the export's provenance.json"
    )
    parser.add_argument("--label-root", type=Path, required=True, help="the label store")
    parser.add_argument(
        "--coder-run", type=Path, required=True, help="the reasoning coder's run folder"
    )
    parser.add_argument(
        "--retry-log", type=Path, required=True, help="the experiment's infra_retries.jsonl"
    )
    parser.add_argument(
        "--superseded", type=Path, required=True, help="the experiment's superseded.json"
    )
    parser.add_argument(
        "--reference-dir", type=Path, required=True, help="the published export's folder"
    )
    parser.add_argument("--workers", type=int, default=8)


def run(args: argparse.Namespace) -> int:
    require(
        {
            "--provenance": args.provenance,
            "--label-root": args.label_root,
            "--coder-run": args.coder_run,
            "--retry-log": args.retry_log,
            "--superseded": args.superseded,
            "--reference-dir": args.reference_dir,
        }
    )
    found = export_check(
        args.provenance,
        args.label_root,
        args.coder_run,
        args.retry_log,
        args.superseded,
        args.reference_dir,
        args.workers,
    )
    print(json.dumps({"check": "A", "differences": found}, indent=2))
    return 0 if not any(found.values()) else 1
