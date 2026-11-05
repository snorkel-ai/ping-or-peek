"""The analysis notebook reproduces the blog post's figures and claims from the Hugging Face
dataset, run as a reader runs it: with its kernel in `analysis/`, offline from the local Hugging
Face cache.

Every expected number below is one the blog published: the counts, p values and intervals from the
blog ledger's rows, computed by code separate from the notebook. The test appends one cell to a copy of the
notebook that prints the notebook's tables as JSON, and checks them against these numbers.
"""

import json
import math
import re
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from ping_or_peek.dataset import data

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "analysis" / "main_analysis.ipynb"
NOISE = 1.0  # points: the most a Figure 4 interval end may differ from the published one
TABLES = (
    "fig2",
    "fig3",
    "fig4",
    "fig5",
    "claims",
    "null_claim",
    "unpaired",
    "w1",
    "descriptive",
    "awareness",
)
FIGURES = ("figure2", "figure3", "figure4", "figure5")
# The published figures' colors, by the legend entry each one marks.
POST_COLORS = {
    "asked the user": "#2a78d6",
    "read the grader": "#eb6834",
    "assumed": "#1baf7a",
    "changed nothing": "#7a786f",
}


def extract(tables, figures=()):
    """A cell that prints the notebook's `tables`, and each of its `figures`' legend colors and
    bar colors, as JSON."""
    return f"""
import json as _json
from matplotlib.colors import to_hex as _hex


def _colors(figure):
    legends = [ax.get_legend() for ax in figure.axes if ax.get_legend()]
    return {{
        "legend": {{
            text.get_text(): _hex(handle.get_facecolor())
            for legend in legends
            for text, handle in zip(legend.get_texts(), legend.legend_handles)
        }},
        "bars": sorted({{_hex(bar.get_facecolor()) for ax in figure.axes for bar in ax.patches}}),
    }}


print(_json.dumps({{
    "alpha": globals().get("alpha"),
    **{{name: globals()[name].reset_index().to_dict("records") for name in {tables!r}}},
    "colors": {{name: _colors(globals()[name]) for name in {figures!r}}},
}}, default=str))
"""


def execute(tables=TABLES, figures=FIGURES, tamper=None, allow_errors=False):
    """Run a copy of the notebook as a reader runs it, offline with its kernel in `analysis/`, and
    return the copy and what it printed: its `tables` and `figures`' colors. `tamper` is code run
    right after the cell that loads the dataset."""
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    if tamper:
        load = next(i for i, c in enumerate(notebook.cells) if "snapshot_download(" in c.source)
        notebook.cells.insert(load + 1, nbformat.v4.new_code_cell(tamper))
    notebook.cells.append(nbformat.v4.new_code_cell(extract(tables, figures)))
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HF_HUB_OFFLINE", "1")
        NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            allow_errors=allow_errors,
            resources={"metadata": {"path": str(NOTEBOOK.parent)}},
        ).execute()
    printed = "".join(
        o.get("text", "") for o in notebook.cells[-1].outputs if o.output_type == "stream"
    )
    notebook.cells.pop()
    return notebook, json.loads(printed)


@pytest.fixture(scope="module")
def cached():
    """Skips the test, saying why, when the dataset isn't in the local Hugging Face cache: tests
    make no network calls, and CI doesn't fetch it."""
    try:
        data.files(data.SOURCE)
    except data.NotFetched as error:
        pytest.skip(str(error))


@pytest.fixture(scope="module")
def run(cached):
    """The committed notebook, a fresh run of it, and what the fresh run printed."""
    fresh, printed = execute()
    return nbformat.read(NOTEBOOK, as_version=4), fresh, printed


def by(rows, *keys):
    """`rows` keyed by the values of `keys`."""
    return {tuple(row[k] for k in keys): row for row in rows}


def count(value):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else int(value)


PATHS = ("asked the user", "read the grader", "assumed")


def test_the_notebook_runs_offline_from_its_folder_without_errors(run):
    _, fresh, _ = run
    errors = [
        o for cell in fresh.cells for o in cell.get("outputs", []) if o.output_type == "error"
    ]
    assert errors == []


def test_the_notebook_is_self_contained():
    code = [c.source for c in nbformat.read(NOTEBOOK, as_version=4).cells if c.cell_type == "code"]
    for source in code:
        assert not re.search(r"^\s*(from|import)\s+analysis\b", source, re.MULTILINE), source
        assert not re.search(r"\.(svg|png|jpe?g)\b|\bSVG\(|\bImage\(", source), source


def test_the_committed_outputs_are_current(run):
    # The bootstrap is seeded, so a rerun shows the same numbers.
    committed, fresh, _ = run

    def texts(notebook):
        return [
            [o.get("data", {}).get("text/plain") for o in cell.outputs if "data" in o]
            for cell in notebook.cells
            if cell.cell_type == "code"
        ]

    assert texts(committed) == texts(fresh)


def test_figure_2_matches_the_published_counts(run):
    rows = by(run[2]["fig2"], "by", "group")
    assert {key: (r["episodes"], *(r[p] for p in PATHS)) for key, r in rows.items()} == {
        ("setting", "default"): (180, 127, 41, 12),
        ("setting", "check-in nudge"): (180, 180, 0, 0),
        ("setting", "autonomy nudge"): (180, 17, 51, 112),
        ("model", "claude-fable-5-1"): (90, 42, 36, 12),
        ("model", "claude-opus-5-5"): (90, 52, 22, 16),
        ("model", "claude-sonnet-5-5"): (90, 45, 27, 18),
        ("model", "gpt-6-astra"): (90, 63, 0, 27),
        ("model", "gpt-6.1-sol"): (90, 69, 0, 21),
        ("model", "gpt-6-luna"): (90, 53, 7, 30),
        ("harness", "native"): (270, 156, 66, 48),
        ("harness", "opencode"): (270, 168, 26, 76),
    }


def test_figure_3_matches_the_published_counts(run):
    rows = by(run[2]["fig3"], "setting", "family")
    assert {key: (r["episodes"], *(r[p] for p in PATHS)) for key, r in rows.items()} == {
        ("default", "claude"): (45, 17, 27, 1),
        ("default", "gpt"): (45, 40, 4, 1),
        ("check-in nudge", "claude"): (45, 45, 0, 0),
        ("check-in nudge", "gpt"): (45, 45, 0, 0),
        ("autonomy nudge", "claude"): (45, 0, 32, 13),
        ("autonomy nudge", "gpt"): (45, 9, 3, 33),
    }


# (by, group): (visible k/n, hidden k/n, change, published low, published high)
FIGURE_4 = {
    ("setting", "default"): ((127, 180), (151, 180), -13.3, -18.9, -7.8),
    ("setting", "check-in nudge"): ((180, 180), (180, 180), 0.0, 0.0, 0.0),
    ("setting", "autonomy nudge"): ((17, 180), (33, 180), -8.9, -15.0, -2.8),
    ("family", "claude"): ((139, 270), (166, 270), -10.0, -13.3, -6.7),
    ("family", "gpt"): ((185, 270), (198, 270), -4.8, -9.3, -0.4),
    ("harness", "native"): ((156, 270), (179, 270), -8.5, -12.2, -4.8),
    ("harness", "opencode"): ((168, 270), (185, 270), -6.3, -10.4, -2.2),
}


def test_figure_4_matches_the_published_changes_and_intervals(run):
    rows = by(run[2]["fig4"], "by", "group")
    assert list(rows) == list(FIGURE_4)
    for key, (visible, hidden, change, low, high) in FIGURE_4.items():
        row = rows[key]
        assert (row["visible k"], row["visible n"]) == visible, key
        assert (row["hidden k"], row["hidden n"]) == hidden, key
        assert round(row["change"], 1) == change, key
        assert row["low"] == pytest.approx(low, abs=NOISE), key
        assert row["high"] == pytest.approx(high, abs=NOISE), key


def test_figure_5_matches_the_published_counts(run):
    rows = by(run[2]["fig5"], "setting", "family", "user")
    columns = (*PATHS, "changed nothing")
    assert {key: (r["episodes"], *(r[c] for c in columns)) for key, r in rows.items()} == {
        ("default", "claude", "available"): (90, 49, 37, 4, 0),
        ("default", "claude", "away"): (90, 0, 61, 24, 5),
        ("default", "gpt", "available"): (90, 78, 4, 8, 0),
        ("default", "gpt", "away"): (90, 0, 9, 78, 3),
        ("check-in nudge", "claude", "available"): (90, 90, 0, 0, 0),
        ("check-in nudge", "claude", "away"): (90, 0, 0, 0, 90),
        ("check-in nudge", "gpt", "available"): (90, 90, 0, 0, 0),
        ("check-in nudge", "gpt", "away"): (90, 0, 1, 2, 87),
        ("autonomy nudge", "claude", "available"): (90, 0, 48, 42, 0),
        ("autonomy nudge", "claude", "away"): (90, 0, 79, 11, 0),
        ("autonomy nudge", "gpt", "available"): (90, 17, 3, 70, 0),
        ("autonomy nudge", "gpt", "away"): (90, 0, 8, 82, 0),
    }


# claim: (A k/n or None for ½, B k/n, discordant (A only, B only) or None, p).
CLAIMS = {
    "1": ((151, 180), (127, 180), (32, 8), 0.000182166),
    "2": ((127, 180), (180, 180), (0, 53), 2.22045e-16),
    "3": ((41, 180), (0, 180), (41, 0), 9.09495e-13),
    "4": ((85, 270), (7, 270), None, 4.54182e-21),
    "5": ((59, 135), (26, 135), (44, 11), 8.69937e-06),
    "6": ((26, 135), (0, 135), None, 7.86899e-09),
    "7": ((0, 15), (12, 15), (0, 12), 0.000488281),
    "8": ((12, 180), (112, 180), (4, 104), 3.43225e-26),
    "9": ((17, 45), (40, 45), None, 6.86466e-07),
    "10": (None, (59, 73), None, 1.01364e-07),
    "11": (None, (34, 41), None, 2.53208e-05),
    "12": ((364, 540), (324, 540), (58, 18), 4.71324e-06),
    "13": ((69, 90), (49, 90), (24, 4), 0.000179991),
    "14": ((7, 90), (0, 90), (7, 0), 0.015625),
    "15": ((82, 90), (78, 90), (8, 4), 0.387695),
    "16": ((26, 90), (17, 90), (19, 10), 0.136046),
    "17": ((41, 180), (70, 180), (8, 37), 1.53741e-05),
    "18": ((51, 180), (87, 180), (4, 40), 1.70514e-08),
    "19": ((42, 90), (11, 90), (34, 3), 1.23313e-07),
}
# The claims table reports each claim's counts and p, and no pass/fail verdict.
CLAIMS_COLUMNS = [
    "claim",
    "post",
    "test",
    "A k",
    "A n",
    "B k",
    "B n",
    "A only",
    "B only",
    "p",
    "points",
]
# The unpaired Fisher p of every paired claim, the null claim N1 included.
UNPAIRED = {
    "1": 0.00368533,
    "2": 2.45853e-18,
    "3": 6.9371e-14,
    "5": 2.3368e-05,
    "7": 1.05211e-05,
    "8": 6.76419e-31,
    "12": 0.0135395,
    "13": 0.00273345,
    "14": 0.0138434,
    "15": 0.477609,
    "16": 0.161546,
    "17": 0.0013398,
    "18": 0.000139633,
    "19": 5.34462e-07,
    "N1": 0.276772,
}


def side(row, letter):
    k, n = count(row[f"{letter} k"]), count(row[f"{letter} n"])
    return None if k is None else (k, n)


def discordant(row):
    a, b = count(row["A only"]), count(row["B only"])
    return None if a is None else (a, b)


def test_the_nineteen_claims_match_the_published_counts_and_p_with_no_verdict(run):
    found = run[2]
    assert found["alpha"] == 0.05 / 19
    assert [list(row) for row in found["claims"]] == [CLAIMS_COLUMNS] * len(CLAIMS)
    rows = by(found["claims"], "claim")
    assert [key[0] for key in rows] == list(CLAIMS)
    for number, (a, b, pairs, p) in CLAIMS.items():
        row = rows[(number,)]
        assert (side(row, "A"), side(row, "B")) == (a, b), number
        assert discordant(row) == pairs, number
        assert row["p"] == pytest.approx(p, rel=1e-5), number


def test_the_null_claim_and_the_unpaired_p_values_match_the_published_ones(run):
    (n1,) = run[2]["null_claim"]
    assert (n1["claim"], side(n1, "A"), side(n1, "B"), discordant(n1)) == (
        "N1",
        (41, 180),
        (51, 180),
        (17, 27),
    )
    assert n1["p"] == pytest.approx(0.174171, rel=1e-5)
    assert {r["claim"]: r["p"] for r in run[2]["unpaired"]} == {
        number: pytest.approx(p, rel=1e-5) for number, p in UNPAIRED.items()
    }


def test_assuming_with_the_user_visible_and_away_matches(run):
    rows = by(run[2]["w1"], "setting", "family")
    assert {
        key: ((r["visible k"], r["visible n"]), (r["away k"], r["away n"]))
        for key, r in rows.items()
    } == {
        ("default", "all"): ((12, 180), (102, 180)),
        ("default", "claude"): ((4, 90), (24, 90)),
        ("autonomy nudge", "all"): ((112, 180), (93, 180)),
        ("autonomy nudge", "claude"): ((42, 90), (11, 90)),
    }


def test_every_descriptive_rate_matches_the_blog(run):
    rows = run[2]["descriptive"]
    assert [(r["blog"], r["dataset k"], r["dataset n"]) for r in rows] == [
        ("127/180", 127, 180),
        ("41/180", 41, 180),
        ("180/180", 180, 180),
        ("17/45", 17, 45),
        ("27/45", 27, 45),
        ("0/45", 0, 45),
        ("45/45", 45, 45),
        ("151/180", 151, 180),
        ("61/90", 61, 90),
        ("79/90", 79, 90),
        ("0/15", 0, 15),
        ("14/15", 14, 15),
        ("2%", 32, 1620),
    ]
    assert all(r["matches"] is True for r in rows)


def test_test_awareness_matches_by_family(run):
    assert [(r["family"], r["k"], r["n"]) for r in run[2]["awareness"]] == [
        ("all", 32, 1620),
        ("claude", 3, 810),
        ("gpt", 29, 810),
    ]


def test_the_notebook_commands_install_only_its_groups():
    # The notebook needs only the analysis and notebook groups: the documented commands install
    # just those, and `--no-sync` keeps `uv run` from installing the rest, Harbor included.
    readme = (ROOT / "README.md").read_text()
    instructions = nbformat.read(NOTEBOOK, as_version=4).cells[0].source
    for text in (readme, instructions):
        assert "uv sync --locked --only-group analysis --only-group notebook" in text
        assert "uv run --no-sync jupyter lab analysis/main_analysis.ipynb" in text


def test_the_figures_use_the_posts_colors(run):
    colors = run[2]["colors"]
    for name, legend in (
        ("figure2", ("asked the user", "read the grader", "assumed")),
        ("figure3", ("asked the user", "read the grader", "assumed")),
        ("figure5", ("asked the user", "read the grader", "assumed", "changed nothing")),
    ):
        assert colors[name]["legend"] == {label: POST_COLORS[label] for label in legend}, name
        assert set(colors[name]["bars"]) <= set(POST_COLORS.values()), name
    assert colors["figure4"]["bars"] == [POST_COLORS["asked the user"]]


@pytest.mark.parametrize(
    ("tamper", "rule"),
    [
        # One fewer episode: a combination is missing.
        ("episodes = episodes.drop(episodes.index[0])", "occurs exactly once"),
        # One more: a combination occurs twice.
        (
            "episodes = pd.concat([episodes, episodes.iloc[[0]]], ignore_index=True)",
            "occurs exactly once",
        ),
        # The grid is whole, but a visible episode is on a path Figures 2 and 3 don't show.
        (
            """
first = episodes.index[
    (episodes["condition"] == "user_available_checker_exposed")
    & (episodes["path"] == "asked the user")
][0]
episodes.loc[first, "path"] = "no change: other"
episodes.loc[first, "csv_changed"] = False
""",
            "add up to its episodes",
        ),
    ],
    ids=["an episode missing", "an episode repeated", "a visible episode on no-change path"],
)
def test_a_dataset_not_shaped_as_the_experiment_stops_the_notebook_naming_the_rule(
    cached, tamper, rule
):
    with pytest.raises(CellExecutionError, match=rule):
        execute(("fig2",), (), tamper=tamper)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("verbalized_test_awareness", None),
        ("csv_changed", None),
        ("setting", "an unknown setting"),
        ("model_family", None),
    ],
)
def test_a_value_the_analysis_does_not_know_stops_it_naming_the_column(cached, column, value):
    unknown = f"""
episodes[{column!r}] = episodes[{column!r}].astype(object)
episodes.loc[episodes.index[0], {column!r}] = {value!r}
"""
    with pytest.raises(CellExecutionError, match=rf"{column} is one the analysis knows"):
        execute(("fig2",), (), tamper=unknown)
