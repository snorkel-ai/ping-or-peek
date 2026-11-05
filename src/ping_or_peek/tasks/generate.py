"""Write the Harbor tasks: `ping-or-peek tasks`.

    ping-or-peek tasks --out tasks/plain
    ping-or-peek tasks --out tasks/claude-asking --instruction-file CLAUDE.md=configs/passage_asking.txt

Writes 45 measured tasks (3 conditions × seeds 100–114) and the conversational control.
`--instruction-file NAME=PATH` adds a workspace instruction file (CLAUDE.md or AGENTS.md),
which is how settings A and B reach the agent through the project's own instructions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ping_or_peek.config import CONDITION_SPECS, SEEDS, SETTINGS

from .harbor_task import control_task, measured_task, write_task


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, required=True, help="directory to write tasks into")
    parser.add_argument(
        "--instruction-file", help="NAME=PATH, e.g. CLAUDE.md=configs/passage_asking.txt"
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    parser.add_argument("--setting", choices=SETTINGS, help="recorded in task.toml [metadata]")


def write_tasks(
    out: Path,
    setting: str | None = None,
    instruction_file: tuple[str, Path] | None = None,
    seeds: list[int] | tuple[int, ...] = SEEDS,
) -> int:
    """Write the measured tasks of `seeds` and the conversational control into `out`, with
    `instruction_file` (a file name and the path of its text) in each workspace, and `setting`
    recorded in each task's labels. Returns how many tasks it wrote."""
    note = (
        None if instruction_file is None else (instruction_file[0], instruction_file[1].read_text())
    )
    out.mkdir(parents=True, exist_ok=True)
    specs = [measured_task(c, s, note) for c in CONDITION_SPECS for s in seeds]
    specs.append(control_task(note))
    for spec in specs:
        if setting:
            spec.labels["setting"] = setting
        write_task(spec, out)
    print(f"wrote {len(specs)} tasks to {out}")
    return len(specs)


def run(args: argparse.Namespace) -> None:
    instruction_file = None
    if args.instruction_file:
        name, path = args.instruction_file.split("=", 1)
        instruction_file = (name, Path(path))
    write_tasks(args.out, args.setting, instruction_file, args.seeds)
