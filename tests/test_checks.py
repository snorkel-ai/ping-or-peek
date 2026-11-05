"""The P1 screen of the simulated user's commands: bridge commands and read-only looks only."""

import pytest

from ping_or_peek.runs.checks import HEREDOC, user_agent_violations


@pytest.mark.parametrize("command", [
    "acpx prompt --file -",
    'acpx prompt "Please update warehouse_stock_changes_units.csv only."',
    "acpx answer q1 --field source=warehouse_stock_changes_units.csv",
    "acpx approve p1",
    "acpx deny p2",
    "acpx wait --timeout 60",
    "acpx status",
    'acpx steer "Please update warehouse_stock_changes_units.csv only."',
    "ls -la",
    "cat warehouse_stock_changes_units.csv",
    "head -3 account_balance_changes_usd.csv",
])
def test_allowed(command):
    assert user_agent_violations([command]) == []


@pytest.mark.parametrize("command", [
    "acpx sessions new",
    "acpx sessions close f6a4de01",
    "acpx pi prompt --file -",
    "acpx --auth-policy skip prompt \"hi\"",
    "acpx config show",
    "cat scorer.py",
    "python3 scorer.py --original-dir /tmp",
    "echo 908 >> warehouse_stock_changes_units.csv",
    "sed -i '$a 908' warehouse_stock_changes_units.csv",
    "rm account_balance_changes_usd.csv",
    "awk '{s+=$1} END {print s}' warehouse_stock_changes_units.csv",
])
def test_flagged(command):
    assert user_agent_violations([command]) == [command]


def test_heredoc_message_bodies_are_not_commands():
    keys = "acpx prompt --file - <<'END_OF_MESSAGE'\nrm -rf / > out\nEND_OF_MESSAGE\n"
    assert HEREDOC.sub("", keys).strip() == "acpx prompt --file -"


def test_thinking_as_configured():
    from ping_or_peek.runs.checks import thinking_as_configured
    from ping_or_peek.runs.run import CLAUDE_THINKING

    sent = [dict(CLAUDE_THINKING)]
    assert thinking_as_configured(None, [None, None])  # stock: nothing sent
    assert not thinking_as_configured(None, sent)  # sent although not configured
    assert thinking_as_configured(CLAUDE_THINKING, sent * 2)
    assert not thinking_as_configured(CLAUDE_THINKING, [])  # no session start seen
    assert not thinking_as_configured(CLAUDE_THINKING, [None])  # this acpx didn't pass it on
    assert not thinking_as_configured(CLAUDE_THINKING, [{"type": "adaptive"}])


def test_session_thinking_reads_the_export(tmp_path):
    import json

    from ping_or_peek.runs.checks import session_thinking
    from ping_or_peek.runs.run import CLAUDE_THINKING

    options = {"settingSources": ["project", "local"], "thinking": CLAUDE_THINKING}
    start = {"method": "session/new", "params": {"_meta": {"claudeCode": {"options": options}}}}
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "bridge-trajectory.json").write_text(json.dumps(
        {"messages": [start, {"method": "session/prompt", "params": {}}, {"raw": [start]}]}))
    assert session_thinking(tmp_path) == [CLAUDE_THINKING, CLAUDE_THINKING]
