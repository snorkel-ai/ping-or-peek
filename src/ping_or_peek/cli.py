"""The experiment's one command, `ping-or-peek`.

    ping-or-peek --help
    ping-or-peek tasks --out tasks/plain
    ping-or-peek run --phase collection --dry-run
    ping-or-peek label ... | code ... | export ... | dataset build|card|check ...

Each subcommand lives in a module of its own, which defines `add_arguments(parser)` and
`run(args)`; `run` returns the exit code (None is 0) and may call `args.parser.error(...)` for a
usage error. A module is imported only when its subcommand is asked for, so `--help`, and a
subcommand that needs little, work without the packages the others need. The `.env` is loaded
before anything reads a setting.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass

from ping_or_peek.config import load_env


@dataclass(frozen=True)
class Command:
    name: str
    module: str
    help: str


COMMANDS = (
    Command("tasks", "ping_or_peek.tasks.generate", "Write the Harbor tasks"),
    Command("run", "ping_or_peek.runs.run", "Run a phase's Harbor jobs, or any part of the grid"),
    Command("check", "ping_or_peek.runs.checks", "Run the mechanical checks on finished trials"),
    Command("retry", "ping_or_peek.runs.retry", "Rerun the trials that failed a mechanical check"),
    Command("progress", "ping_or_peek.runs.progress", "Report a phase's progress"),
    Command("s0-report", "ping_or_peek.runs.s0_report", "Write the sanity-sample report"),
    Command(
        "confidentiality-test",
        "ping_or_peek.runs.confidentiality_test",
        "Test whether the agent can read the simulated user's prompt",
    ),
    Command("label", "ping_or_peek.labeling.label_command", "Label finished trials with an LLM"),
    Command(
        "code",
        "ping_or_peek.labeling.reasoning.code_command",
        "Code the agents' stated reasons for asking or taking the shortcut",
    ),
    Command(
        "export", "ping_or_peek.dataset.export_command", "Build episodes.parquet from the labels"
    ),
    Command(
        "check-record",
        "ping_or_peek.dataset.completeness",
        "Report which label inputs a trial's record holds",
    ),
)
GROUPS = {
    "dataset": (
        "Build, describe and check the Hugging Face dataset",
        (
            Command("build", "ping_or_peek.dataset.build", "Build the dataset's shards"),
            Command("card", "ping_or_peek.dataset.card", "Write the dataset card"),
            Command("check", "ping_or_peek.dataset.check", "Check a build before any upload"),
        ),
    ),
    "safety-net": (
        "The two checks every refactor must keep passing",
        (
            Command(
                "export",
                "ping_or_peek.dataset.export_check",
                "Check A: the export rebuilds exactly from the raw runs",
            ),
            Command(
                "labeler-inputs",
                "ping_or_peek.labeling.labeler_inputs",
                "Check B: the labelers' inputs rebuild from the dataset's trace",
            ),
        ),
    ),
}


def requested(argv: list[str]) -> tuple[str, ...]:
    """The subcommand `argv` asks for: its first word, and its second for a group."""
    first = argv[0] if argv else None
    if first in GROUPS and len(argv) > 1:
        return (first, argv[1])
    return (first,) if first else ()


def build_parser(asked: tuple[str, ...]) -> argparse.ArgumentParser:
    """The parser of every subcommand, with the arguments of only the one in `asked`."""
    parser = argparse.ArgumentParser(prog="ping-or-peek", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    leaves: list[tuple[tuple[str, ...], Command, argparse.ArgumentParser]] = []
    for command in COMMANDS:
        leaf = commands.add_parser(command.name, help=command.help, description=command.help)
        leaves.append(((command.name,), command, leaf))
    for group, (help_text, members) in GROUPS.items():
        branch = commands.add_parser(group, help=help_text, description=help_text)
        subcommands = branch.add_subparsers(dest="subcommand", required=True, metavar="command")
        for command in members:
            leaf = subcommands.add_parser(command.name, help=command.help, description=command.help)
            leaves.append(((group, command.name), command, leaf))
    for path, command, leaf in leaves:
        if path == asked:
            module = importlib.import_module(command.module)
            module.add_arguments(leaf)
            leaf.set_defaults(handler=module.run, parser=leaf)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    load_env()
    args = build_parser(requested(argv)).parse_args(argv)
    return args.handler(args) or 0
