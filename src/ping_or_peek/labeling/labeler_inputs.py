"""Check B of the safety net, `labeler-inputs`: the labelers' inputs rebuild from the
dataset's `trace`.

For every episode of the Hugging Face dataset's latest revision, or of a built dataset folder (`--data`), it renders
each LLM labeler's input from the row (`trace_inputs`: the ATIF trace read back by
`atif.from_trajectory`, handed to the labelers' own code) and compares it with the request the labeler sent, as saved in its label store:
every stage of the outcome labeler, and the reasoning coder, whose `evaluation_awareness` codes
give the dataset's `verbalized_test_awareness`. The bar is an exact match of the request text.
Every difference is reported by episode, labeler, place, offset and lengths, as unexplained: the
trace carries everything the labelers read, so none is expected and none is normalized away. A dataset
whose episodes aren't the export's, once each, fails before any comparison. Since the harness
marked some tool outputs as truncated, it also reports which inputs held them.

Check B is the test a dataset revision must pass before the labelers read it: every labeler's
input rebuilds exactly from that revision's `trace`. A build of `dataset.build` passes it, and
so does the dataset's latest revision, `data.SOURCE`, an uploaded build. Revision 4a958f8 and
earlier fail: their trace is an event list built for the blog's numbers, not an ATIF trajectory.

Check A, the export rebuilt from the raw runs, is `dataset.export_check`. Check B reads private
local data, so CI can't run it; its tests run on synthetic fixtures:

    uv run --no-sync ping-or-peek safety-net labeler-inputs \\
        --provenance <export>/provenance.json --label-root <labels> \\
        --coder-run <reasoning labels>/<run> --reference-dir <export> [--data <dataset>] \\
        --report <private.jsonl>

Every path is an argument, and a missing one fails naming it. The check exits 0 only when
everything matches. Nothing here writes to the label store, the trials or the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ping_or_peek.config import PERSONA_FILE, REPO
from ping_or_peek.dataset.export_check import (
    KEY,
    REFERENCE_SHA256,
    check_reference,
    first_difference,
    labeler_folder,
    require,
    sent_messages,
)

from .label_run import REASONING_EXCLUDED, REASONING_INCLUDED
from .labels import AGENT_RULE, EPISODE_STAGES
from .trace_inputs import SENSITIVITY, outcome_stages

# --- Comparisons ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Mismatch:
    """One difference between a saved input and the one rendered from the dataset.

    `path` locates it: `.key`, `[index]`, `[id]` for an item matched by its id, and `{}` for JSON
    inside a text, whose keys are recorded and so named by their position, `.#n`. `detail` says what differs. For text, `offset` is the first differing
    character and the lengths are the saved and rendered texts'; the texts themselves are never
    held, so a report can't leak them.
    """

    path: str
    detail: str
    offset: int | None = None
    saved_length: int | None = None
    rendered_length: int | None = None


def _json_text(text: str) -> Any:
    """The JSON a text holds, or None when it holds none (or only a scalar)."""
    if text[:1] not in "{[":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _by_id(values: list[Any]) -> dict[str, Any] | None:
    """Items by their id, when every element is a dict with an id, and the ids are unique."""
    if not all(isinstance(v, dict) and isinstance(v.get("id"), str) for v in values):
        return None
    ids = [v["id"] for v in values]
    return {v["id"]: v for v in values} if len(set(ids)) == len(ids) else None


def json_differences(
    saved: Any, rendered: Any, path: str = "", recorded: bool = False
) -> list[Mismatch]:
    """Every difference between a saved JSON input and the rendered one, saved order first.

    The evidence's `items` are matched by id; a text holding JSON is compared inside, where keys
    are `recorded`: a path names them by position, never by their text. Dict key order counts:
    the request is the JSON text, so a reordered dict is a different request.
    """
    if type(saved) is not type(rendered):
        return [Mismatch(path, "type")]
    if isinstance(saved, dict):

        def at(keys: dict[Any, Any], k: Any) -> str:
            return f"{path}.#{list(keys).index(k)}" if recorded else f"{path}.{k}"

        out = []
        for k in saved:
            if k in rendered:
                out += json_differences(saved[k], rendered[k], at(saved, k), recorded)
            else:
                out.append(Mismatch(at(saved, k), "absent from the rendered input"))
        out += [
            Mismatch(at(rendered, k), "absent from the saved input")
            for k in rendered
            if k not in saved
        ]
        if set(saved) == set(rendered) and list(saved) != list(rendered):
            out.append(Mismatch(path, "key order"))
        return out
    if isinstance(saved, list):
        saved_ids, rendered_ids = _by_id(saved), _by_id(rendered)
        keyed = (
            not recorded
            and path.endswith(".items")
            and saved_ids is not None
            and rendered_ids is not None
        )
        if keyed and len(saved_ids) == len(saved):
            out = []
            for key, value in saved_ids.items():
                if key in rendered_ids:
                    out += json_differences(value, rendered_ids[key], f"{path}[{key}]", recorded)
                else:
                    out.append(Mismatch(f"{path}[{key}]", "absent from the rendered input"))
            for value in rendered:
                if value["id"] not in saved_ids:
                    out.append(Mismatch(f"{path}[{value['id']}]", "absent from the saved input"))
            common = [k for k in saved_ids if k in rendered_ids]
            if common != [v["id"] for v in rendered if v["id"] in saved_ids]:
                out.append(Mismatch(path, "order"))
            return out
        out = []
        if len(saved) != len(rendered):
            out.append(Mismatch(path, "length", None, len(saved), len(rendered)))
        for index, (a, b) in enumerate(zip(saved, rendered)):
            out += json_differences(a, b, f"{path}[{index}]", recorded)
        return out
    if isinstance(saved, str):
        if saved == rendered:
            return []
        inner_saved, inner_rendered = _json_text(saved), _json_text(rendered)
        if inner_saved is not None and inner_rendered is not None:
            inner = json_differences(inner_saved, inner_rendered, f"{path}{{}}", recorded=True)
            if inner:
                return inner
        return [
            Mismatch(path, "text", first_difference(saved, rendered), len(saved), len(rendered))
        ]
    return [] if saved == rendered else [Mismatch(path, "value")]


# An event's header line in a coder transcript: "[e12] TOOL_CALL:", "[e9] QUESTION (tag):".
BLOCK = re.compile(r"\n(?=\[[^\]\n]+\] [A-Z_]+(?: \([^\n]*\))?:\n)")


def _blocks(text: str) -> list[tuple[str, str]]:
    """A coder transcript's (id, text) blocks: the header lines, then one block per event."""
    found = []
    for index, block in enumerate(BLOCK.split(text)):
        match = re.match(r"\[([^\]\n]+)\] ", block)
        found.append((match.group(1) if match and index else "header", block))
    return found


def text_differences(saved: str, rendered: str) -> list[Mismatch]:
    """Every difference between a saved coder transcript and the rendered one, event by event:
    `[id]` locates an event's block, and the offset is within it."""
    saved_blocks = dict(_blocks(saved))
    rendered_blocks = _blocks(rendered)
    by_id = dict(rendered_blocks)
    out = []
    for key, block in saved_blocks.items():
        if key not in by_id:
            out.append(Mismatch(f"[{key}]", "absent from the rendered input"))
        elif by_id[key] != block:
            other = by_id[key]
            out.append(
                Mismatch(f"[{key}]", "text", first_difference(block, other), len(block), len(other))
            )
    for key, _ in rendered_blocks:
        if key not in saved_blocks:
            out.append(Mismatch(f"[{key}]", "absent from the saved input"))
    common = [k for k in saved_blocks if k in by_id]
    if common != [k for k, _ in rendered_blocks if k in saved_blocks]:
        out.append(Mismatch("", "order"))
    if not out and saved != rendered:
        out.append(
            Mismatch("", "text", first_difference(saved, rendered), len(saved), len(rendered))
        )
    return out


# --- Check B ----------------------------------------------------------------------------------

# Every difference's cause. The ATIF trace carries everything the labelers read, so no
# difference is expected and none is explained away.
UNEXPLAINED = "unexplained"


def _holds_output(item: dict[str, Any], call: str, output: str) -> bool:
    """Whether an evidence item is a result of tool call `call` that holds `output` in full."""
    from .labels import strings

    if item["kind"] != "tool_result":
        return False
    body = _json_text(item["text"]) or {}
    return body.get("tool_call_id") == call and output in strings(body)


def legacy(row: dict[str, Any]) -> list[dict[str, Any]]:
    """A row's trajectory projected to the event list revision 4a958f8 published: read back into
    the session (`atif.from_trajectory`), then laid out as that revision's builder did."""
    from ping_or_peek.dataset import build

    from . import atif

    read = atif.from_trajectory(row["trace"])
    return build.trace(read.session, read.native, row["agent"], row["model_family"])


def _dataset_rows(folder: Path | None = None) -> list[dict[str, Any]]:
    """The dataset's rows as Python values: a built dataset `folder`'s shards (`data/*.parquet`),
    or the dataset's latest revision's when `folder` is None. pyarrow keeps nullable integers, which
    a pandas frame turns into floats."""
    import pyarrow.parquet as pq

    from ping_or_peek.dataset import data

    if folder is None:
        return [r for path in data.files(data.SOURCE) for r in pq.read_table(path).to_pylist()]
    from ping_or_peek.dataset.build import read_shards

    table, problems = read_shards(folder / "data")
    if problems:
        raise SystemExit(f"{folder} lacks the dataset's structure (build.SCHEMA): {problems}")
    return table.to_pylist()


def check_coverage(found: list[str], expected: set[str]) -> None:
    """Raises ValueError, naming the episodes, unless `found` holds each expected episode once
    and nothing else: a dataset revision that drops, repeats or adds an episode fails."""
    repeated = sorted(e for e, n in Counter(found).items() if n > 1)
    missing, extra = sorted(expected - set(found)), sorted(set(found) - expected)
    if repeated or missing or extra:
        raise ValueError(
            f"the dataset's episodes aren't the export's: missing {missing[:10]} "
            f"({len(missing)}), repeated {repeated[:10]} ({len(repeated)}), "
            f"not in the export {extra[:10]} ({len(extra)})"
        )


def compare_outcome(
    row: dict[str, Any],
    persona: str,
    answer: dict[str, Any],
    saved: dict[str, dict[str, Any]],
    sent: dict[str, tuple[str, str]],
) -> dict[str, list[tuple[Mismatch, str]]]:
    """Each outcome-labeler stage that applies to the episode, rendered from the trace or saved,
    with its differences and their causes. `answer` is the episode stage's saved answer, which
    decides whether the materiality stage applies; `saved` and `sent` hold, by stage, the saved
    payload and the (system, user) messages sent. A stage only one side has is a difference."""
    from . import annotator, trace_inputs
    from .labels import rubric_sha256

    rendered = trace_inputs.outcome_inputs(row, persona, answer)
    found: dict[str, list[tuple[Mismatch, str]]] = {}
    for name, stage in outcome_stages(persona).items():
        if name not in saved and name not in rendered:
            continue
        if name not in saved or name not in rendered:
            side = "saved" if name not in saved else "rendered"
            found[name] = [(Mismatch("", f"stage absent from the {side} inputs"), UNEXPLAINED)]
            continue
        full = {
            **rendered[name],
            "version": stage.version,
            "rubric_sha256": rubric_sha256(stage.rubric_name),
        }
        system, user = sent[name]
        expected_system, expected_user = (
            m["content"] for m in annotator.request_messages(stage.rubric, full)
        )
        found[name] = []
        if system != expected_system:
            found[name].append((Mismatch("system", "text"), UNEXPLAINED))
        if user != expected_user:
            mismatches = json_differences(json.loads(user), full) or [Mismatch("", "text")]
            found[name] += [(m, UNEXPLAINED) for m in mismatches]
    return found


@dataclass
class Tally:
    """One labeler's results: episodes compared and matched, and episodes by cause."""

    compared: int = 0
    matched: int = 0
    causes: Counter = field(default_factory=Counter)


def labeler_inputs_check(
    provenance: Path,
    label_root: Path,
    coder_run: Path,
    reference_dir: Path,
    report: Path,
    dataset: Path | None = None,
) -> dict[str, Any]:
    """Check B, on a built `dataset` folder, or on the dataset's latest revision when it is None.
    Writes every difference to `report` (JSON lines: episode, labeler, place, offset, lengths, cause;
    never text) and returns the summary."""
    import pyarrow.parquet as pq

    from ping_or_peek.dataset import data

    from . import trace_inputs
    from .reasoning.label import SHORT_OUTPUT_LIMIT, safe_name, system_prompt
    from .reasoning.transcript import OUTPUT_LIMIT, _clip, episode_key

    check_reference(reference_dir, {"episodes.parquet": REFERENCE_SHA256["episodes.parquet"]})
    record = json.loads(provenance.read_text())
    trial_path = {i["trial"]: Path(i["path"]) for i in record["inputs"]}
    trial = {
        r[KEY]: r["trial"] for r in pq.read_table(reference_dir / "episodes.parquet").to_pylist()
    }
    listed = (record["label_result_files"], record["sensitivity_result_files"])
    if [len(files) for files in listed] != [1, 1]:
        raise ValueError(
            "trace_inputs renders one primary labeling run and one sensitivity pass; "
            f"{provenance} lists {listed}"
        )
    (primary_file,), (sensitivity_file,) = listed
    primary = json.loads(Path(primary_file).read_text())
    sensitivity = json.loads(Path(sensitivity_file).read_text())
    rules = (
        primary["asking_rule"],
        primary["reasoning_in_evidence"],
        sensitivity["reasoning_in_evidence"],
    )
    if (
        rules != (AGENT_RULE, REASONING_EXCLUDED, REASONING_INCLUDED)
        or not sensitivity["scorer_edit_only"]
    ):
        raise ValueError(
            "trace_inputs renders the agent asking rule, primary labels without reasoning and a "
            f"scorer-edit-only sensitivity pass with it; the runs used {rules}"
        )
    persona = PERSONA_FILE.read_text()
    stages = outcome_stages(persona)
    folders = {
        name: labeler_folder(label_root, sensitivity if name == SENSITIVITY else primary)
        / stage.name
        for name, stage in stages.items()
    }
    tallies: dict[str, Tally] = defaultdict(Tally)
    truncated: list[dict[str, Any]] = []
    report.parent.mkdir(parents=True, exist_ok=True)
    rows = _dataset_rows(dataset)
    if rows:  # a revision whose trace isn't ATIF is refused before the report exists
        from . import atif

        atif.from_trajectory(rows[0]["trace"])
    with report.open("x") as out:  # never overwrites: see _refuse_report_in_inputs

        def record_mismatches(
            episode: str, labeler: str, found: list[tuple[Mismatch, str]]
        ) -> None:
            tally = tallies[labeler]
            tally.compared += 1
            tally.matched += not found
            for cause in {c for _, c in found}:
                tally.causes[cause] += 1
            for mismatch, cause in found:
                out.write(
                    json.dumps(
                        {
                            "episode_id": episode,
                            "labeler": labeler,
                            **asdict(mismatch),
                            "cause": cause,
                        }
                    )
                    + "\n"
                )

        check_coverage([r[KEY] for r in rows], set(trial))
        for row in rows:
            key = trial[row[KEY]]
            answer_file = folders[EPISODE_STAGES[AGENT_RULE]] / "records" / f"{key}.json"
            if not answer_file.exists():
                raise FileNotFoundError(f"--label-root: no episode-stage answer {answer_file}")
            answer = json.loads(answer_file.read_text())["value"]
            saved, sent = {}, {}
            for name, folder in folders.items():
                payload = folder / "payloads" / f"{key}.json"
                if payload.exists():
                    saved[name] = json.loads(payload.read_text())
                    sent[name] = sent_messages(folder / "attempts" / key)
            for name, found in compare_outcome(row, persona, answer, saved, sent).items():
                record_mismatches(row[KEY], f"outcome labeler: {name}", found)
            coded = safe_name(episode_key(trial_path[key], trial_path[key].parent.parent))
            system, user = sent_messages(coder_run / coded, "attempt-*.json")
            text = trace_inputs.coder_input(row)
            found = [] if system == system_prompt() else [(Mismatch("system", "text"), UNEXPLAINED)]
            if user != text:
                found += [(m, UNEXPLAINED) for m in text_differences(user, text)]
            record_mismatches(row[KEY], "reasoning coder", found)
            # Outputs the harness says it shortened, as revision 4a958f8's event list marks them.
            events = legacy(row)
            for event in events:
                if not event["output_truncated"]:
                    continue
                call, output = event["tool_call_id"], event["tool_output"]
                position = next(
                    e["position"]
                    for e in events
                    if e["type"] == "tool_call" and e["tool_call_id"] == call
                )
                # Where the request each labeler sent holds the output: in full, or clipped.
                held = [
                    f"outcome labeler: {name}"
                    for name, (_, request) in sent.items()
                    if any(
                        _holds_output(i, call, output) for i in json.loads(request).get("items", [])
                    )
                ]
                block = dict(_blocks(user)).get(f"e{position}", "")
                if "\noutput:\n" + output + "\n" in block + "\n":
                    held.append("reasoning coder")
                elif any(
                    f"\noutput:\n{_clip(output, n)}" in block
                    for n in (OUTPUT_LIMIT, SHORT_OUTPUT_LIMIT)
                ):
                    held.append("reasoning coder (clipped)")
                truncated.append({"episode_id": row[KEY], "tool_call_id": call, "in": held})
    return {
        "dataset": str(dataset) if dataset else data.revision(data.SOURCE),
        "labelers": {
            name: {
                "compared": t.compared,
                "matched": t.matched,
                "causes": dict(sorted(t.causes.items(), key=lambda kv: (-kv[1], kv[0]))),
            }
            for name, t in sorted(tallies.items())
        },
        "truncated_outputs": truncated,
    }


def _refuse_report_in_inputs(args: argparse.Namespace) -> None:
    """Raises ValueError, before anything is written, when Check B's `--report` could touch
    what the check reads. The report never overwrites a file, so no input can be truncated,
    whatever links lead to it; and a new report may not go inside the repository or anything the
    check reads (its options' paths, the label-result files and trials `provenance.json` lists),
    by the path as given or as resolved."""
    if args.report.exists() or args.report.is_symlink():
        raise ValueError(f"--report {args.report}: the file exists; the check never overwrites one")
    record = json.loads(args.provenance.read_text())
    inputs = (
        [REPO, args.label_root, args.coder_run, args.reference_dir, args.provenance]
        + ([args.data] if args.data else [])
        + [Path(p) for p in record["label_result_files"]]
        + [Path(p) for p in record["sensitivity_result_files"]]
        + [Path(i["path"]) for i in record["inputs"]]
    )
    given, resolved = Path(os.path.abspath(args.report)), args.report.resolve()
    inside = [
        str(p)
        for p in inputs
        for report, root in ((given, Path(os.path.abspath(p))), (resolved, p.resolve()))
        if report.is_relative_to(root)
    ]
    if inside:
        raise ValueError(
            f"--report {args.report}: write it outside the repository and every input, not in "
            f"{sorted(set(inside))}"
        )


# --- Command line ---------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provenance", type=Path, required=True, help="the export's provenance.json"
    )
    parser.add_argument(
        "--label-root", type=Path, required=True, help="the outcome labeler's store"
    )
    parser.add_argument(
        "--coder-run", type=Path, required=True, help="the reasoning coder's run folder"
    )
    parser.add_argument(
        "--reference-dir", type=Path, required=True, help="the published export's folder"
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="a built dataset folder (data/*.parquet) to check instead of the latest revision",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="a new file, outside the repository and every input, for every difference",
    )


def run(args: argparse.Namespace) -> int:
    require(
        {
            "--provenance": args.provenance,
            "--label-root": args.label_root,
            "--coder-run": args.coder_run,
            "--reference-dir": args.reference_dir,
            **({"--data": args.data} if args.data else {}),
        }
    )
    _refuse_report_in_inputs(args)
    summary = labeler_inputs_check(
        args.provenance, args.label_root, args.coder_run, args.reference_dir, args.report, args.data
    )
    print(json.dumps({"check": "B", **summary}, indent=2))
    exact = all(t["compared"] == t["matched"] for t in summary["labelers"].values())
    return 0 if exact else 1
