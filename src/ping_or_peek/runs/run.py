"""Build and run the experiment's Harbor jobs: `ping-or-peek run`.

    ping-or-peek run --phase s0 --dry-run         # show the job configs, run nothing
    ping-or-peek run --phase s0 --jobs-dir DIR    # run the sanity sample
    ping-or-peek run --model claude-opus-5-5 --harness native --setting B \\
        --condition user_available_checker_exposed --seeds 100-104 --label try --dry-run

Each phase is a list of Harbor jobs. A job is one agent and setting, all its models, and the
phase's tasks. Every job is an ordinary `harbor run -c <config>`, so any cell can be rerun by hand
from the printed config. `--model`, `--harness`, `--setting`, `--condition` and `--seeds` narrow a
phase to part of its grid; without `--phase` they select from the whole collection grid, under a
`--label` that names the jobs, so they can't reuse a phase's job folders.

Phases:
    s0          sanity sample: seed 100, setting S; native × 3 conditions, OpenCode × available/exposed
    controls    the conversational control, every model × harness × setting
    pilot       available/exposed, seed 100, every model × harness × setting
    collection  every condition × seeds 100–114, every model × harness × setting
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from ping_or_peek.config import (
    AUTONOMY,
    CHECK_IN,
    CONDITIONS,
    CONFIGS,
    DEFAULT,
    HARNESSES,
    JOBS,
    MODEL_SPECS,
    MODELS,
    PERSONA,
    PERSONA_FILE,
    PRESENT_VISIBLE,
    REPO,
    SEEDS,
    SETTINGS,
    TASKS,
    harness_of,
)
from ping_or_peek.tasks.generate import write_tasks
from ping_or_peek.tasks.harbor_task import PERSONA_POLICY

# Agent versions baked into the task image (src/ping_or_peek/tasks/Dockerfile); Harbor skips reinstalling them.
PINS = {"claude-code": "2.1.296", "codex": "0.162.1", "opencode": "1.19.0"}
# ACP adapters at their releases of 13 October 2026 (Harbor 0.24 pins older ones by default).
ADAPTERS = {
    "claude-code": ("@agentclientprotocol/claude-agent-acp", "0.90.0"),
    "codex": ("@agentclientprotocol/codex-acp", "2.2.2"),
}
# Claude Code's permission mode per setting, written by Harbor as
# permissions.defaultMode in the adapter's user settings.json, the adapter's only startup route.
CLAUDE_PERMISSION_MODE = {DEFAULT: "auto", CHECK_IN: "default", AUTONOMY: "auto"}
# Native Claude Code asks for summarized thinking in every setting. acpx passes the
# option in session/new `_meta.claudeCode.options`, which needs the acpx build with
# that passthrough ($CLAUDE_ACPX_PACKAGE); the other harnesses keep $ACPX_PACKAGE. Requests then
# differ from stock only in thinking.display and one beta, and the earlier thinking sent back
# carries summary text instead of empty text, which the model does not read.
CLAUDE_THINKING = {"type": "adaptive", "display": "summarized"}
# Codex per setting: codex-acp's session mode, and A's Default-mode question flag.
# workspace-write is the terminal's trusted-folder default: approvals on request, sandboxed writes.
CODEX_MODE = {DEFAULT: "workspace-write", CHECK_IN: "workspace-write", AUTONOMY: "agent-full-access"}
CODEX_QUESTION_FLAG = "\n[features]\ndefault_mode_request_user_input = true\n"
# Codex's Linux sandbox (bubblewrap) needs user namespaces, which Docker's default seccomp
# profile blocks. Every Codex job runs with seccomp unconfined; privileged is the
# fallback for hosts where that is not enough (CODEX_CONTAINER=privileged).
CODEX_CONTAINER = {
    "seccomp-unconfined": "services:\n  main:\n    security_opt:\n      - seccomp:unconfined\n",
    "privileged": "services:\n  main:\n    privileged: true\n",
}
# Workspace instruction file per harness, used by settings A and B.
INSTRUCTION_FILE = {"claude-code": "CLAUDE.md", "codex": "AGENTS.md", "opencode": "AGENTS.md"}
PASSAGE = {CHECK_IN: "configs/passage_asking.txt", AUTONOMY: "configs/passage_autonomy.txt"}

# The target's credentials, as ${VAR} templates: Harbor resolves them at run time and stores only
# the template. The bridge starts the target during setup (start_owner_on_setup), so only the
# target gets them; the simulated user's shell does not.
CREDENTIAL_ENV = {
    "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}",
    "ANTHROPIC_BASE_URL": "${ANTHROPIC_BASE_URL}",
    "ANTHROPIC_CUSTOM_HEADERS": "${PORTKEY_ANTHROPIC_HEADERS}",
    "PORTKEY_API_KEY": "${PORTKEY_API_KEY}",
    "CODEX_API_KEY": "${OPENAI_API_KEY}",
}
# The secret each CREDENTIAL_ENV entry carries, as the variable that holds it: scripts/gateway.sh
# gives the target the Portkey key under each of these names (the headers are built from it and
# OPENAI_API_KEY is set to it). ANTHROPIC_API_KEY is "dummy" there, and the base URL is public.
CREDENTIAL_SECRETS = {
    "ANTHROPIC_CUSTOM_HEADERS": "PORTKEY_API_KEY",
    "PORTKEY_API_KEY": "PORTKEY_API_KEY",
    "CODEX_API_KEY": "PORTKEY_API_KEY",
}
GATEWAY_HOSTS = ["api.portkey.ai"]
# Headroom for each trial's setup (Harbor's default is 360 s; healthy setups take under a minute):
# the bridge npm-installs acpx and the adapter in every trial, which ran past 360 s while the
# host was swapping. Setup durations are still reported, so a slow setup shows.
SETUP_TIMEOUT_MULTIPLIER = 2.0
# Harbor checks once per process, with a 30-second `docker run`, that Docker can enforce the
# network allowlist. When Docker is slow the check times out and the job stops at its first
# trial, before any agent starts. Relaunching resumes the job: Harbor keeps every trial that has
# a result and runs the rest.
EGRESS_CHECK_FAILURE = "network_mode='allowlist' is not supported"
RELAUNCHES = 3

# The simulated user: Terminus 2 driving GPT-6.1 Sol at medium effort.
USER_MODEL, USER_EFFORT = "gpt-6.1-sol", "medium"
USER_AGENT = {
    "name": "terminus-2",
    "model_name": f"openai/@openai/{USER_MODEL}",
    "kwargs": {"api_base": "https://api.portkey.ai/v1", "reasoning_effort": USER_EFFORT},
    "user_persona_path": str(PERSONA_FILE),
    "user_prompt_template_path": str(PERSONA / "user_prompt.md.j2"),
    # caller_input routes the agent's questions and approval requests to the simulated user;
    # start_owner_on_setup launches the agent with its own credentials before the user arrives.
    "bridge": {"kind": "acp", "kwargs": {"caller_input": True, "start_owner_on_setup": True}},
}


@dataclass(frozen=True)
class Job:
    phase: str
    harness: str  # claude-code, codex or opencode
    setting: str
    models: tuple[str, ...]
    tasks: tuple[str, ...]  # task directory names within the task variant

    @property
    def name(self) -> str:
        return f"{self.phase}__{self.harness}__{self.setting}"

    @property
    def variant(self) -> str:
        """The generated task set: plain at S, or with the setting's instruction file."""
        if self.setting == DEFAULT:
            return "plain"
        return f"{INSTRUCTION_FILE[self.harness].removesuffix('.md').lower()}-{self.setting}"


def ensure_tasks(job: Job) -> Path:
    """Generate the job's task variant if it does not exist yet."""
    root = TASKS / job.variant
    if not root.exists():
        note = None if job.setting == DEFAULT else (INSTRUCTION_FILE[job.harness], REPO / PASSAGE[job.setting])
        write_tasks(root, job.setting, note)
    return root


def opencode_effort(provider: str, effort: str) -> dict:
    """The request options OpenCode 1.19 attaches when a user picks this effort in its model
    picker (the model's "variant", provider/transform.ts `variants`), set as the model's options."""
    if provider == "anthropic":  # Claude 5.x: adaptive thinking, summarized
        return {"thinking": {"type": "adaptive", "display": "summarized"}, "effort": effort}
    return {"reasoningEffort": effort, "reasoningSummary": "auto",
            "include": ["reasoning.encrypted_content"]}


def agent_config(
    harness: str, model: str, setting: str, config_dir: Path = JOBS / "configs",
) -> dict:
    effort = MODEL_SPECS[model][1]
    kwargs: dict = {"version": PINS[harness]}
    if harness in ADAPTERS:
        kwargs["acp_adapter_package"], kwargs["acp_adapter_version"] = ADAPTERS[harness]
    if harness == "claude-code":
        kwargs["permission_mode"] = CLAUDE_PERMISSION_MODE[setting]
    if harness == "codex" and os.environ.get("CODEX_ACP_PACKAGE"):
        # The display-only codex-acp build (async questions shown to the user).
        kwargs["acp_adapter_package"] = os.environ["CODEX_ACP_PACKAGE"]
        kwargs.pop("acp_adapter_version")
    if harness == "codex":
        kwargs["acp_agent_mode"] = CODEX_MODE[setting]
    model_name = model
    if harness == "codex":
        # config.toml: the gateway provider, plus A's flag.
        text = (CONFIGS / "codex" / "portkey.toml").read_text()
        if setting == CHECK_IN:
            text += CODEX_QUESTION_FLAG
        out = config_dir / f"codex-{setting}.toml"
        out.write_text(text)
        kwargs["config"] = str(out)
    if harness == "opencode":
        provider = "anthropic" if model.startswith("claude-") else "openai"
        model_name = f"{provider}/{model}"
        overlay = json.loads((CONFIGS / "opencode" / f"portkey-{provider}.json").read_text())
        overlay["provider"][provider]["models"] = {model: {"options": opencode_effort(provider, effort)}}
        if setting == CHECK_IN:  # ask before edits and shell commands
            overlay["permission"] = {"edit": "ask", "bash": "ask"}
        kwargs["opencode_config"] = overlay
        kwargs["fake_vcs"] = False  # /workspace is not a git repo; don't say it is
    else:
        kwargs["reasoning_effort"] = effort
    # Setting-specific knobs (permission mode, Codex agent mode, OpenCode permissions) are
    # added here; see configs/README.md.
    return {
        "name": harness,
        "model_name": model_name,
        "kwargs": kwargs,
        "env": CREDENTIAL_ENV,
        "extra_allowed_hosts": GATEWAY_HOSTS,
    }


def job_config(
    job: Job, jobs_dir: Path, n_concurrent: int, acpx_package: str | None = None
) -> dict:
    """One Harbor JobConfig. `acpx_package` is an acpx build with caller input (npm spec or a
    tarball made with `npm pack`); it defaults to $ACPX_PACKAGE, or to $CLAUDE_ACPX_PACKAGE for
    native Claude Code jobs."""
    if job.harness == "claude-code":
        acpx_package = acpx_package or os.environ.get("CLAUDE_ACPX_PACKAGE")
        if not acpx_package:
            raise SystemExit("Set $CLAUDE_ACPX_PACKAGE to the acpx build that passes Claude Code's "
                             "thinking option.")
    acpx_package = acpx_package or os.environ.get("ACPX_PACKAGE")
    if not acpx_package:
        raise SystemExit("Set --acpx-package or $ACPX_PACKAGE to an acpx with caller input.")
    root = ensure_tasks(job)
    user_agent = json.loads(json.dumps(USER_AGENT))
    user_agent["bridge"]["kwargs"]["acpx_package"] = acpx_package
    if job.harness == "claude-code":
        user_agent["bridge"]["kwargs"]["acpx_config"] = {"claudeCodeOptions": {"thinking": CLAUDE_THINKING}}
    if job.harness == "opencode":
        # OpenCode's ACP session starts at its lowest effort and overrides the model options;
        # "default" hands effort back to each model's options, which carry the convention.
        user_agent["bridge"]["kwargs"]["session_config"] = {"effort": "default"}
    config = {
        "job_name": job.name,
        "jobs_dir": str(jobs_dir),
        "n_concurrent_trials": n_concurrent,
        "agent_setup_timeout_multiplier": SETUP_TIMEOUT_MULTIPLIER,
        "user_agent": user_agent,
        "tasks": [{"path": str(root / t)} for t in job.tasks],
    }
    config_dir = jobs_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    overlays = []
    if job.harness == "codex":
        choice = os.environ.get("CODEX_CONTAINER", "seccomp-unconfined")
        overlay = config_dir / f"codex-{choice}.compose.yaml"
        overlay.write_text(CODEX_CONTAINER[choice])
        overlays.append(str(overlay))
    if overlays:
        config["environment"] = {"extra_docker_compose": overlays}
    config["agents"] = [
        agent_config(job.harness, m, job.setting, config_dir)
        for m in job.models
    ]
    return config


def configured_thinking(config: dict) -> dict | None:
    """The thinking option a job or trial config passes to native Claude Code, if any."""
    bridge = ((config.get("user_agent") or {}).get("bridge") or {}).get("kwargs") or {}
    return ((bridge.get("acpx_config") or {}).get("claudeCodeOptions") or {}).get("thinking")


def _package(spec: str | None) -> dict | None:
    """An install spec, with the sha256 and source commit of a local tarball."""
    if not spec:
        return None
    path = Path(spec)
    if not path.is_file():
        return {"spec": spec}
    record = {"spec": spec, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    commit = path.with_suffix(".commit")  # written next to the tarball by whoever packed it
    named = re.search(r"-([0-9a-f]{7,40})\.tgz$", path.name)  # or named in it: …-<commit>.tgz
    if commit.exists():
        record["commit"] = commit.read_text().strip()
    elif named:
        record["commit"] = named.group(1)
    return record


def _git(path: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() or None


def harbor_commit(source: dict) -> str | None:
    """The commit of the Harbor build that `source`, its install's direct_url.json, names: the
    commit pip recorded for it, else the HEAD of a local checkout (a file URL). None when the
    install names neither."""
    recorded = (source.get("vcs_info") or {}).get("commit_id")
    if recorded:
        return recorded
    url = source.get("url", "")
    return _git(Path(url.removeprefix("file://"))) if url.startswith("file://") else None


def provenance(acpx_package: str | None) -> dict:
    """What produced a job: this repo, Harbor, acpx, the adapters and the pinned agents."""
    harbor = metadata.distribution("harbor")
    source = json.loads(harbor.read_text("direct_url.json") or "{}")
    if os.environ.get("HARBOR"):  # a separate Harbor build runs the jobs; record that one
        executable = shlex.split(os.environ["HARBOR"])[0]
        code = ("import json, importlib.metadata as m; d = m.distribution('harbor');"
                "print(json.dumps([d.version, d.read_text('direct_url.json')]))")
        probe = subprocess.run([str(Path(executable).parent / "python"), "-c", code],
                               capture_output=True, text=True, check=False)
        if probe.returncode == 0:
            version, url = json.loads(probe.stdout)
            source = json.loads(url or "{}")
            harbor = type("Dist", (), {"version": version})()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                           capture_output=True, text=True, check=False).stdout.strip()
    return {
        "repo_commit": _git(REPO),
        "repo_dirty": bool(dirty),
        "harbor": {"version": harbor.version, "source": source.get("url"),
                   "commit": harbor_commit(source)},
        "acpx": _package(acpx_package),
        "codex_acp": _package(os.environ.get("CODEX_ACP_PACKAGE")) or {"spec": "@".join(ADAPTERS["codex"])},
        "claude_agent_acp": {"spec": "@".join(ADAPTERS["claude-code"])},
        "agents": PINS,
        "models": MODEL_SPECS,
        "user_agent": {"harness": USER_AGENT["name"], "model": USER_MODEL, "effort": USER_EFFORT},
        "persona_policy": PERSONA_POLICY,
        "persona_sha256": {name: hashlib.sha256((PERSONA / name).read_bytes()).hexdigest()
                           for name in ("persona.md", "user_prompt.md.j2")},
    }


@dataclass
class Launch:
    """One `harbor run` of a job config, relaunched if it stops at Harbor's egress check."""

    name: str
    config: Path
    log: Path
    process: subprocess.Popen | None = None
    log_offset: int = 0  # where this attempt's output starts in the log
    results_at_start: int = 0
    relaunches: int = 0

    def results(self) -> int:
        config = json.loads(self.config.read_text())
        return len(list((Path(config["jobs_dir"]) / config["job_name"]).glob("*/result.json")))

    def start(self) -> None:
        harbor = shlex.split(os.environ.get("HARBOR", "harbor"))  # e.g. a patched Harbor build
        self.results_at_start = self.results()
        with self.log.open("a" if self.process else "w") as log:  # a relaunch appends
            self.log_offset = log.tell()
            self.process = subprocess.Popen([*harbor, "run", "-c", str(self.config), "-y"],
                                            cwd=REPO, stdout=log, stderr=subprocess.STDOUT)

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stopped_at_egress_check(self) -> bool:
        """This attempt failed with Harbor's egress-check error and no trial got a result."""
        if self.process is None or self.process.poll() in (None, 0):
            return False
        with self.log.open(errors="replace") as log:
            log.seek(self.log_offset)
            if EGRESS_CHECK_FAILURE not in log.read():
                return False
        return self.results() == self.results_at_start


def run_jobs(launches: list[Launch], disk: Path, max_jobs: int | None = None,
             min_free_gb: float = 15.0, infra_log: Path | None = None) -> None:
    """Run the jobs in order, at most `max_jobs` at a time (None: all at once). Holds before
    starting a job while free disk is low, and relaunches a job that stopped at Harbor's egress
    check (up to RELAUNCHES times, logged to `infra_log`). Harbor resolves the jobs' ${VAR}
    credentials from this process's environment, so the caller loads the repo's .env first."""
    queue = list(launches)
    while True:
        running = 0
        for launch in launches:  # each started job is looked at once per pass, so none is missed
            if launch.running():
                running += 1
            elif launch.stopped_at_egress_check() and launch.relaunches < RELAUNCHES:
                launch.relaunches += 1
                print(f"relaunching {launch.name}: Harbor's egress check timed out "
                      f"(relaunch {launch.relaunches} of {RELAUNCHES})", flush=True)
                if infra_log:
                    with infra_log.open("a") as out:
                        out.write(json.dumps({
                            "logged_at": datetime.now(UTC).isoformat(),
                            "job": launch.name, "trial": None, "failed_checks": ["egress_check"],
                            "details": {"error": EGRESS_CHECK_FAILURE},
                            "retried_in_job": launch.name, "relaunch": launch.relaunches,
                        }) + "\n")
                time.sleep(60)  # give Docker a minute
                launch.start()
                running += 1
        if not queue and not running:
            break
        if queue and (max_jobs is None or running < max_jobs):
            free = shutil.disk_usage(disk).free / 2**30
            if free >= min_free_gb:
                queue.pop(0).start()
                continue
            # swap and images compete for the same disk; a full disk crashes Docker
            print(f"holding {queue[0].name}: {free:.1f} GB free < {min_free_gb} GB", flush=True)
        time.sleep(15)
    for launch in launches:
        if launch.stopped_at_egress_check():
            print(f"{launch.name} stopped at Harbor's egress check {RELAUNCHES + 1} times; "
                  "giving up", flush=True)


PHASES = ("s0", "controls", "pilot", "collection")
LABEL = re.compile(r"[a-z0-9][a-z0-9-]*")  # a job-name prefix: no `_`, `.` or `/`


def measured(conditions, seeds) -> tuple[str, ...]:
    return tuple(f"{c}__seed{s}" for c in conditions for s in seeds)


def phase_jobs(phase: str) -> list[Job]:
    claude = tuple(m for m, (h, _) in MODEL_SPECS.items() if h == "claude-code")
    gpt = tuple(m for m, (h, _) in MODEL_SPECS.items() if h == "codex")
    every = MODELS
    names = CONDITIONS
    exposed = (PRESENT_VISIBLE,)
    if phase == "s0":
        return [
            Job("s0", "claude-code", DEFAULT, claude, measured(names, [100])),
            Job("s0", "codex", DEFAULT, gpt, measured(names, [100])),
            Job("s0", "opencode", DEFAULT, every, measured(exposed, [100])),
        ]
    tasks = {
        "controls": ("conversational_control",),
        "pilot": measured(exposed, [100]),
        "collection": measured(names, SEEDS),
    }[phase]
    return [
        Job(phase, harness, setting, models, tasks)
        for harness, models in (("claude-code", claude), ("codex", gpt), ("opencode", every))
        for setting in SETTINGS
    ]


def seeds_from(text: str) -> list[int]:
    """Seeds written as ranges and lists, `100-104` or `100,102,104`. Raises ArgumentTypeError, naming
    the text, for anything else, a range that doesn't run from low to high, or a seed outside the
    grid."""
    seeds: list[int] = []
    for part in text.split(","):
        low, dash, high = part.partition("-")
        try:
            first, last = int(low), int(high) if dash else int(low)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"{text!r}: seeds are numbers, ranges `100-104` or lists `100,102`"
            ) from None
        if last < first:
            raise argparse.ArgumentTypeError(f"{text!r}: a range of seeds runs from low to high")
        seeds += range(first, last + 1)
    outside = sorted(set(seeds) - set(SEEDS))
    if outside:
        raise argparse.ArgumentTypeError(
            f"{text!r}: {outside} are not seeds of the grid, {SEEDS[0]}-{SEEDS[-1]}"
        )
    return seeds


def select_jobs(
    phase: str | None,
    label: str | None,
    models: list[str] | None = None,
    harnesses: list[str] | None = None,
    settings: list[str] | None = None,
    conditions: list[str] | None = None,
    seeds: list[int] | None = None,
) -> list[Job]:
    """The jobs of `phase` (the whole collection grid when None) that fall in the chosen part of
    the grid, named by `label` (the phase when None). A harness is native (each model's own agent)
    or opencode; a job keeps only the chosen models, and its measured tasks only the chosen
    conditions and seeds, which leave out the conversational control. An axis left None doesn't
    narrow the selection; an empty one picks nothing. Raises ValueError when no phase and no label
    name the jobs, when a label isn't a plain name (`LABEL`) or is a phase's name or starts with
    it and a hyphen (it would alias that phase's job names, which retry finds by the `<phase>-`
    prefix), or when the selection picks no job."""
    if phase is None and label is None:
        raise ValueError("a selection without --phase needs --label to name its jobs")
    if label is not None and (
        not LABEL.fullmatch(label) or any(label == p or label.startswith(f"{p}-") for p in PHASES)
    ):
        raise ValueError(
            f"--label {label!r}: a label is lowercase letters, digits and hyphens, starting with "
            f"a letter or digit, and neither a phase's name nor starting with one and a hyphen "
            f"({', '.join(PHASES)}): it becomes part of job names and file paths, so it can't "
            "hold `__`, `.` or `/`, and retry finds a phase's jobs by `<phase>-` prefixes"
        )
    chosen = []
    for job in phase_jobs(phase or "collection"):
        keep = tuple(m for m in job.models if models is None or m in models)
        tasks = job.tasks
        if conditions is not None or seeds is not None:
            parsed = [t.rpartition("__seed") for t in tasks]
            tasks = tuple(
                t
                for t, (condition, sep, seed) in zip(tasks, parsed, strict=True)
                if sep
                and (conditions is None or condition in conditions)
                and (seeds is None or int(seed) in seeds)
            )
        if (
            keep
            and tasks
            and (harnesses is None or harness_of(job.harness) in harnesses)
            and (settings is None or job.setting in settings)
        ):
            chosen.append(
                Job(job.phase if label is None else label, job.harness, job.setting, keep, tasks)
            )
    if not chosen:
        raise ValueError("the selection picks no job")
    return chosen


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--phase", choices=PHASES, help="the phase's jobs, narrowed by what follows"
    )
    parser.add_argument(
        "--model",
        nargs="+",
        choices=MODELS,
        metavar="MODEL",
        help=f"only these: {', '.join(MODELS)}",
    )
    parser.add_argument("--harness", nargs="+", choices=HARNESSES, help="only these harnesses")
    parser.add_argument("--setting", nargs="+", choices=SETTINGS, help="only these settings")
    parser.add_argument(
        "--condition", nargs="+", choices=CONDITIONS, metavar="CONDITION",
        help=f"only these: {', '.join(CONDITIONS)}",
    )  # fmt: skip
    parser.add_argument(
        "--seeds", type=seeds_from, help="only these seeds, e.g. 100-104 or 100,102"
    )
    parser.add_argument(
        "--label",
        help="job-name prefix instead of the phase, required without one: "
        "lowercase letters, digits and hyphens, not starting with a phase's name",
    )
    parser.add_argument("--jobs-dir", type=Path, default=JOBS)
    parser.add_argument("--configs-dir", type=Path, help="where to write job configs")
    parser.add_argument("--n-concurrent", type=int, default=4)
    parser.add_argument(
        "--acpx-package",
        help="acpx with caller input (default: $ACPX_PACKAGE, "
        "or $CLAUDE_ACPX_PACKAGE for native Claude Code)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the configs and stop")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="run the jobs at once (total trials = jobs × --n-concurrent)",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        help="with --parallel, run at most this many jobs at a time, queued in order",
    )
    parser.add_argument(
        "--agent-order",
        nargs="*",
        help="queue jobs by agent in this order, e.g. opencode claude-code codex",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=15.0,
        help="hold before starting a job while free disk is below this",
    )
    parser.add_argument(
        "--infra-log",
        type=Path,
        help="append a line here for each relaunch after Harbor's egress check",
    )


def run(args: argparse.Namespace) -> None:
    try:
        jobs = select_jobs(
            args.phase,
            args.label,
            args.model,
            args.harness,
            args.setting,
            args.condition,
            args.seeds,
        )
    except ValueError as error:
        args.parser.error(str(error))
    configs_dir = args.configs_dir or args.jobs_dir / "configs"
    if args.agent_order:
        jobs.sort(key=lambda j: args.agent_order.index(j.harness))  # stable: keeps S, A, B order
    launches = []
    for job in jobs:
        config = job_config(job, args.jobs_dir, args.n_concurrent, args.acpx_package)
        if args.dry_run:
            print(json.dumps(config, indent=2))
            continue
        configs_dir.mkdir(parents=True, exist_ok=True)
        path = configs_dir / f"{job.name}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        record = provenance(config["user_agent"]["bridge"]["kwargs"]["acpx_package"])
        (configs_dir / f"{job.name}.provenance.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        launches.append(Launch(job.name, path, args.jobs_dir / f"{job.name}.log"))
    # One job at a time unless --parallel; a slot frees when a job finishes and the next starts.
    run_jobs(
        launches,
        args.jobs_dir,
        max_jobs=args.max_jobs if args.parallel else 1,
        min_free_gb=args.min_free_gb,
        infra_log=args.infra_log,
    )
