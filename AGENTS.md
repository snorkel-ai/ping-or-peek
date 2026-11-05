# Working in this repository

These rules apply to every agent and developer working here, whatever IDE or harness runs them. `CLAUDE.md` imports this file.

Read `README.md` first: it describes the experiment, how to run it, and the layout. `analysis/README.md` describes the pipeline in `src/ping_or_peek/dataset/` and `src/ping_or_peek/labeling/`, and its measures. This file says how to work; it doesn't restate either.

## Tests

- Tests make no model calls and no network calls, and need no Docker. A test that needs any of these, such as a live run, is marked `live` and excluded from the default selection. A local server that a test starts and stops itself, such as a mock model API on loopback, doesn't count as a network call.
- A test of new behavior checks it against an expectation written independently of the code, in a failing commit before the code.
- Pinned hashes and byte-for-byte reproduction checks are different: they guard the published results, and they change only when the user agrees the result should change.
- A valid failing test is never weakened to pass. A check that didn't run is not a passing check; say so.

## Engineering rules

- **No shims and no silent fallbacks.** Unsupported input, configuration or state fails with a message that names the rule. Nothing substitutes a default for a missing credential, dependency, service or file.
- **Credentials** come from the environment or the ignored `.env` and never appear in task directories, logs, fixtures or committed files.
- **No backward compatibility** unless the user asks for it. The one exception is code that reproduces a recorded run, such as a pinned write-detection revision.
- **One definition per vocabulary.** Each closed vocabulary (conditions, settings, harnesses, models, paths) is defined once, in `src/ping_or_peek/config.py`, and imported everywhere else. So are the repository's paths and the loading of `.env`. Don't repeat its string literals. The exception is `analysis/main_analysis.ipynb`, which is self-contained and uses the dataset's own values.
- **The verifier uses the standard library only,** because it runs inside the task container.
- **Per-harness differences live in one place per concern,** such as a table keyed by harness, not in branches scattered through the code.

## Code quality

- **Read files in full** before wide-ranging changes, before editing a file you haven't inspected, and when asked to investigate or audit. Don't rely on search snippets for broad changes.
- **Extend an existing mechanism before adding a parallel one.** A reviewer can follow one episode, from its task directory to its row in the export, by reading the code.
- **Inline single-line helpers** that have one call site.
- **Ask before removing** functionality or code that appears intentional.
- **Check installed package source** under `.venv` for an external API's shape instead of guessing. Harbor is a fork pinned in `pyproject.toml`.
- **Use established libraries.** Don't hand-roll what a standard, widely used library already provides, such as statistical tests, plotting or parsing. Adding a basic dependency beats writing your own: add it to the right group in `pyproject.toml`, in its own commit. The verifier is the exception, because it runs inside the task container.
- **Docstrings and comments.**
  - Docstrings state purpose and contract: inputs, outputs, side effects, raises.
  - Comments record what the code can't say: an invariant, a constraint from another system, a reason not to touch something.
  - History, decisions and dead ends go in the pull request or the docs, not in the code.
- **Lint.** New and changed code passes `uv run ruff check` and `uv run ruff format --check`.
  - Existing code doesn't follow every rule here yet.
  - Leave code you aren't otherwise changing as it is.

## Implementing a pull request

Before the branch exists:

- Scope and success criteria are written down: in the issue for planned work, in the pull request description for small work. Every criterion names the check a reviewer opens to confirm it: a test, a command and its output, or an artifact.
- A live or paid run counts as a criterion only when it exercises the change directly, such as the pipeline completing on a new slice. The behavior that run measures is evidence, not a criterion.
- One concern per pull request.

While working:

- **Worktrees.** Work in your own worktree off `main`. Several agents may be active at once.
- **Commit order.** Tests come first, as their own commits, and refactoring is a separate commit. A pull request whose tests came after the implementation says so and why.
- **Review findings.** A finding that claims a defect is a claim until demonstrated, and you change code for it only after demonstrating it:
  - for code, a test that fails on the head, committed before the fix;
  - for a published result, the reproduction check or command whose output shows it;
  - for a contradiction, the passages quoted with their locations, and a sentence on why they can't both hold;
  - for a broken rule, the rule quoted beside the line that breaks it.

  A finding you can't demonstrate is answered with what you tried and left for the user to decide. Findings of other kinds, such as optional points, style and design questions, aren't defects: take them only within the scope, and say why when you don't.
- **Scope changes** go back to the issue, not silently into the diff. Work that grew past its scope is split.
- **Pushing and merging.**
  - Commit and push only to your own branch.
  - Never merge a pull request unless the user has given you that authority.
  - Never force-push or rewrite pushed history. Bring `main` into a pushed branch with a merge commit, and rebase only before the first push.
- **Staging.** Stage only files you changed, with `git add <path>` for each. Never `git add -A` or `git add .`. Run `git status` before every commit.
- **Destructive git commands.** Never run `git reset --hard`, `git checkout .`, `git clean` or `git stash` in the main checkout or any shared one: they destroy other agents' work. `git commit --no-verify` is never allowed.
- **Rebase conflicts.** On a conflict in a file you didn't change, abort and ask.
- **Commit messages** say what changed and why, in one or two sentences.

Finishing:

- **Open the pull request** for your work. Never create, close or edit an issue unless the user asks for that action.
- **The description** covers the current head:
  - what the pull request does;
  - each criterion, with its check and whether it is met;
  - the checks run, with the exact commands;
  - each review finding that claimed a defect, with its outcome.

  Rewrite the description as the pull request changes; don't append to it.
- **Defects outside the scope** go to the user, who decides whether each becomes an issue. Don't fix them in this pull request.
- **Comments.** Write issue and pull request comments to a file and post them with `--body-file`. Keep them concise and technical.

## Reviewing a pull request

- Review the diff against the stated scope, its criteria and this file.
- Approve when the change meets its criteria and leaves the code better than before, not when it is perfect. Google's reviewer standard applies: https://google.github.io/eng-practices/review/reviewer/standard.html
- A finding that claims a defect blocks only when demonstrated, in one of the forms under "Implementing a pull request". State the failing scenario precisely enough for the author to demonstrate it; a finding without one is returned for it. Mark optional points as optional.
