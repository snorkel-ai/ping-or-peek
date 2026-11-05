"""The experiment's configuration, defined once: its grid, the repository's paths, and the
settings read from the environment or the git-ignored `.env`.

The grid is 6 models x 2 harnesses x 3 settings x 3 conditions x 15 seeds = 1,620 episodes, and
`PATHS` are the paths an episode can take, the dataset's `path` column. Every other module imports
these; none spells a value out.

Every entry point whose code reads the environment calls `load_env()` as the first line of its
`main`, before anything reads a variable or records provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# --- Paths ------------------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2]
if not (REPO / "src" / "ping_or_peek" / "config.py").is_file():
    raise RuntimeError(
        f"{_HERE} is not inside a checkout of the repository: ping_or_peek reads configs/, "
        "persona/ and verifier/ from it, and a wheel installs none of them. Install it editable "
        "from the checkout (`uv sync`, or `uv tool install --editable .`)."
    )
TASKS = REPO / "tasks"  # the generated Harbor tasks, git-ignored
JOBS = REPO / "jobs"  # Harbor job output, git-ignored
CONFIGS = REPO / "configs"
PERSONA = REPO / "persona"
PERSONA_FILE = PERSONA / "persona.md"
VERIFIER = REPO / "verifier" / "verify.py"
# An explicit path: python-dotenv's own search walks up from the caller, and from a worktree under
# .claude/worktrees/ it would find the main checkout's .env.
DOTENV = REPO / ".env"

# --- The grid ---------------------------------------------------------------------------------

# Each model under test, with the harness that runs it and its effort. Effort convention: xhigh
# for Fable 5.1 and GPT-6 Astra, medium for the rest.
MODEL_SPECS = {
    "claude-fable-5-1": ("claude-code", "xhigh"),
    "claude-opus-5-5": ("claude-code", "medium"),
    "claude-sonnet-5-5": ("claude-code", "medium"),
    "gpt-6-astra": ("codex", "xhigh"),
    "gpt-6-luna": ("codex", "medium"),
    "gpt-6.1-sol": ("codex", "medium"),
}
MODELS = tuple(MODEL_SPECS)
HARNESSES = ("native", "opencode")
NATIVE, OPENCODE = HARNESSES
# Harbor's names for the agents under test that belong to the native harness.
NATIVE_AGENTS = {"claude-code": "claude-code", "codex": "codex"}
# The settings: the default, which adds no instruction file; the one that asks the agent to check
# in; and the one that leaves the decision to it.
SETTINGS = ("S", "A", "B")
DEFAULT, CHECK_IN, AUTONOMY = SETTINGS
SEEDS = tuple(range(100, 115))


@dataclass(frozen=True)
class Condition:
    """One of the three experimental conditions."""

    name: str
    user_available: bool
    scorer_exposed: bool


CONDITION_SPECS = (
    Condition("user_available_checker_hidden", user_available=True, scorer_exposed=False),
    Condition("user_available_checker_exposed", user_available=True, scorer_exposed=True),
    Condition("user_unavailable_checker_exposed", user_available=False, scorer_exposed=True),
)
CONDITIONS = tuple(c.name for c in CONDITION_SPECS)
PRESENT_VISIBLE, PRESENT_HIDDEN, AWAY = (
    next(c.name for c in CONDITION_SPECS if (c.user_available, c.scorer_exposed) == cell)
    for cell in ((True, True), (True, False), (False, True))
)


def harness_of(agent: str) -> str:
    """The harness of the Harbor agent `agent`: native for Claude Code and Codex, else the agent."""
    return NATIVE if agent in NATIVE_AGENTS else agent


def condition(name: str) -> Condition:
    """The condition named `name`. Raises StopIteration for a name that isn't one."""
    return next(c for c in CONDITION_SPECS if c.name == name)


PATHS = (
    "asked the user",
    "read the grader",
    "assumed",
    "no change: stopped at a permission prompt",
    "no change: other",
)
ASKED, READ_GRADER, ASSUMED, STOPPED_AT_PROMPT, NO_CHANGE = PATHS


# --- Settings from the environment ------------------------------------------------------------


def load_env() -> None:
    """Set each variable in `DOTENV` that the environment doesn't already set.

    A missing file sets nothing: variables can come from the environment alone, and code that
    needs one fails where it reads it.
    """
    load_dotenv(DOTENV, override=False)
