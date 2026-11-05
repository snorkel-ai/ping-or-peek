# Known issues

None of these defects affects the published dataset, the analysis, or the numbers the notebook reproduces. They matter only to someone running the experiment or the labeling pipeline again. Each entry says what happens and where, when it matters, and how to work around it.

## Checks that can report a pass that didn't happen

### 1. `ping-or-peek check` exits 0 when a check fails

`run` in `src/ping_or_peek/runs/checks.py` prints each trial that fails, but returns nothing, so the command exits 0 whatever it finds.

- **When it matters:** only in a script that uses the exit status as a gate.
- **Workaround:** read the report, or use `ping-or-peek retry`, which finds failed trials itself.

### 2. `ping-or-peek confidentiality-test` can report `reachable=False` without probing

In `src/ping_or_peek/runs/confidentiality_test.py`, `run_one` doesn't check Harbor's exit status, and `probe` ignores a failed `docker exec`. If Harbor exits before any probe runs, the result is `reachable=False probes=0`.

- **When it matters:** when you use the test to confirm that, in a new setup, the agent can't read the simulated user's prompt.
- **Workaround:** treat `probes=0` as a test that didn't run.

### 3. The effort check doesn't look at OpenCode's setting

For an OpenCode trial with no recorded requests, `effort_as_configured` in `src/ping_or_peek/runs/checks.py` reads `kwargs.reasoning_effort`. But OpenCode's effort is set under `kwargs.opencode_config.provider.<provider>.models.<model>.options`, so the check passes without looking at it.

- **When it matters:** only if OpenCode's effort were misconfigured. The run script sets it from `config.py`.
- **Workaround:** none is needed with the shipped configuration, but don't rely on this check to catch a change in OpenCode's effort.

## State that can be wrong without an error

### 4. Resuming a reasoning-coder run ignores a changed model or effort

`code_episode` in `src/ping_or_peek/labeling/reasoning/label.py` reuses an episode's saved coding without checking which model and effort produced it. `code_command.py` then overwrites the run's `run.json` with the new configuration, so one run can mix codings from two configurations.

- **Workaround:** use a new `--run` for each coder configuration.

### 5. A failed Harbor probe records the wrong Harbor in provenance

When `HARBOR` points at another Harbor installation whose Python can't report its Harbor build, `provenance()` in `src/ping_or_peek/runs/run.py` doesn't fail. It records the current interpreter's Harbor version, source and commit instead.

- **Workaround:** point `HARBOR` only at an installation whose Python has Harbor installed.

## Input that isn't checked

### 6. An unknown `--phase`

Neither `ping-or-peek progress` nor `ping-or-peek retry` restricts `--phase` to the defined phases (`src/ping_or_peek/runs/progress.py`, `src/ping_or_peek/runs/retry.py`). Given an unknown phase, `progress` raises a `KeyError` and `retry` reports no failures.

- **Workaround:** spell the phase exactly: `s0`, `controls`, `pilot` or `collection`.

### 7. A partial `--agent-order`

If `ping-or-peek run --agent-order` leaves out an agent, it raises an uncaught `ValueError` (`src/ping_or_peek/runs/run.py`).

- **Workaround:** list all three agents, for example `--agent-order opencode claude-code codex`.

## By design

### 8. Narrowing a phase keeps the phase's job names

`ping-or-peek run --phase collection --model …` writes into the collection's job folders. That's how a phase is filled in or resumed.

- **To keep a partial run separate:** leave out `--phase`. Select with `--model`, `--harness`, `--setting`, `--condition` and `--seeds`, and name the run with `--label`.
