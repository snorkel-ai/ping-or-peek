"""The built dataset's comparison with the published revision (`dataset.check`)."""

import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ping_or_peek import cli
from ping_or_peek import config as project_config
from ping_or_peek.dataset import build, check

# A line of 50+ characters, as the vendors' captured system prompts hold them.
VENDOR_LINE = "You are a coding assistant made by a vendor; follow these rules carefully."


def references(tmp_path):
    """A folder of vendor captures, `<case>/turn<N>.json`, as `--references` takes it."""
    folder = tmp_path / "references"
    (folder / "case").mkdir(parents=True, exist_ok=True)
    capture = {"system": [{"text": VENDOR_LINE}], "tools": {"Read": {"description": "Reads."}}}
    (folder / "case" / "turn1.json").write_text(json.dumps(capture))
    return folder


def argv(tmp_path):
    return [
        "--data", str(tmp_path), "--records", str(tmp_path), "--export", str(tmp_path),
        "--references", str(references(tmp_path)),
    ]  # fmt: skip


def test_the_check_needs_the_vendor_captures_folder(tmp_path, capsys):
    """The captures stay out of the repository, so the check has no default folder for them."""
    with pytest.raises(SystemExit):
        cli.main(["dataset", "check", *argv(tmp_path)[:-2]])
    assert "--references" in capsys.readouterr().err


@pytest.mark.parametrize("folder", ["missing", "empty"])
def test_a_references_folder_without_captures_fails_naming_it(tmp_path, folder):
    """With no captures the vendor-text guard would find nothing, so it fails before reading."""
    (tmp_path / "empty").mkdir()
    given = [*argv(tmp_path)[:-1], str(tmp_path / folder)]
    with pytest.raises(ValueError, match=folder):
        cli.main(["dataset", "check", *given])
    assert not (tmp_path / "check_report.json").exists()


def test_the_vendor_lines_come_from_the_captures(tmp_path):
    assert VENDOR_LINE in check.reference_lines(references(tmp_path))


def test_the_check_loads_dotenv_before_it_reads_the_hub(tmp_path, monkeypatch):
    """Hugging Face settings such as HF_TOKEN may come from the repository's .env only."""
    dotenv = tmp_path / ".env"
    dotenv.write_text("HF_TOKEN=dummy-token\n")
    monkeypatch.setattr(project_config, "DOTENV", dotenv)
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(SystemExit):  # no shards in the folder: the check stops there
        cli.main(["dataset", "check", *argv(tmp_path)])
    assert os.environ["HF_TOKEN"] == "dummy-token"


def test_strings_include_keys():
    assert sorted(check.strings({"/Users/x/a.csv": {"k": "v"}})) == ["/Users/x/a.csv", "k", "v"]


def test_a_build_missing_a_column_stops_the_check_with_its_report_written(tmp_path):
    (tmp_path / "data").mkdir()
    pq.write_table(pa.table({"episode_id": ["a"]}), tmp_path / "data" / "train-0.parquet")
    pq.write_table(pa.table({"episode_id": ["a"]}), tmp_path / "episodes.parquet")
    with pytest.raises(SystemExit) as stopped:
        cli.main(["dataset", "check", *argv(tmp_path)])
    assert stopped.value.code == 1
    report = json.loads((tmp_path / "check_report.json").read_text())
    found = report["the build's schema: columns, nested fields and required values"]
    assert found["pass"] is False
    assert any("model" in problem for problem in found["detail"])


def test_the_schema_is_todays(tmp_path):
    """The published columns, their nested fields and which values may be null, as revision
    4a958f8 holds them but for the trace, now an ATIF trajectory's JSON, written out
    independently of the builder."""
    csv = pa.struct([("name", pa.string()), ("role", pa.string()), ("content", pa.string()),
                     ("column_total", pa.int64())])  # fmt: skip
    task = pa.struct(
        [
            ("opener", pa.string()), ("intended_file", pa.string()),
            ("checker_target_file", pa.string()), ("csv_files", pa.list_(csv)),
            ("scorer_py", pa.string()),
            ("project_note", pa.struct([("name", pa.string()), ("content", pa.string())])),
            ("user_facts", pa.string()),
        ]
    )  # fmt: skip
    published = pa.schema(
        [
            ("episode_id", pa.string()), ("model", pa.string()), ("model_family", pa.string()),
            ("harness", pa.string()), ("agent", pa.string()), ("setting", pa.string()),
            ("condition", pa.string()), ("seed", pa.int64()), ("path", pa.string()),
            ("asked_user_before_edit", pa.bool_()), ("scorer_used_to_edit_csv", pa.bool_()),
            ("verbalized_test_awareness", pa.bool_()), ("task", task),
            ("trace", pa.string()), ("csv_changed", pa.bool_()),
        ]
    )  # fmt: skip
    assert build.written(build.SCHEMA).equals(published)
    nullable = {
        "scorer_used_to_edit_csv", "task.checker_target_file", "task.scorer_py",
        "task.project_note",
    }  # fmt: skip
    assert build.nullable_paths(build.SCHEMA) == nullable


def test_a_null_task_stops_the_check_with_its_report_written(tmp_path):
    (tmp_path / "data").mkdir()
    row = {name: None for name in build.SCHEMA.names}
    table = pa.Table.from_pylist([row], schema=build.written(build.SCHEMA))
    pq.write_table(table, tmp_path / "data" / "train-0.parquet")
    with pytest.raises(SystemExit) as stopped:
        cli.main(["dataset", "check", *argv(tmp_path)])
    assert stopped.value.code == 1
    report = json.loads((tmp_path / "check_report.json").read_text())
    found = report["the build's schema: columns, nested fields and required values"]
    assert found["pass"] is False
    assert any(problem.startswith("train-0.parquet: task:") for problem in found["detail"])


def test_an_empty_data_folder_stops_the_check_with_its_report_written(tmp_path):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["dataset", "check", *argv(tmp_path)])
    assert stopped.value.code == 1
    report = json.loads((tmp_path / "check_report.json").read_text())
    assert report["the build's shards (data/*.parquet)"]["pass"] is False


def well_formed_row() -> dict:
    """One row with every value the schema requires."""
    task = {"opener": "o", "intended_file": "a.csv", "checker_target_file": None,
            "csv_files": [{"name": "a.csv", "role": "intended", "content": "v", "column_total": 1}],
            "scorer_py": None, "project_note": None, "user_facts": "{}"}  # fmt: skip
    return {"episode_id": "e", "model": "m", "model_family": "claude", "harness": "native",
            "agent": "claude-code", "setting": "S", "condition": "c", "seed": 1, "path": "p",
            "asked_user_before_edit": False, "scorer_used_to_edit_csv": None,
            "verbalized_test_awareness": False, "task": task, "trace": "{}",
            "csv_changed": False}  # fmt: skip


def test_a_well_formed_build_passes_the_structure_check(tmp_path):
    table = pa.Table.from_pylist([well_formed_row()], schema=build.written(build.SCHEMA))
    pq.write_table(table, tmp_path / "shard.parquet")
    assert build.structure_problems(pq.read_table(tmp_path / "shard.parquet")) == []
    nulled = table.set_column(
        table.column_names.index("task"), "task", pa.nulls(1, table.schema.field("task").type)
    )
    assert build.structure_problems(nulled) == ["task: 1 nulls, where the schema allows none"]


def stopped_report(tmp_path) -> dict:
    with pytest.raises(SystemExit) as stopped:
        cli.main(["dataset", "check", *argv(tmp_path)])
    assert stopped.value.code == 1
    return json.loads((tmp_path / "check_report.json").read_text())


def test_shards_with_different_schemas_stop_the_check_with_its_report_written(tmp_path):
    (tmp_path / "data").mkdir()
    pq.write_table(build.written(build.SCHEMA).empty_table(), tmp_path / "data" / "train-0.parquet")
    pq.write_table(pa.table({"episode_id": ["a"]}), tmp_path / "data" / "train-1.parquet")
    found = stopped_report(tmp_path)[
        "the build's schema: columns, nested fields and required values"
    ]
    assert found["pass"] is False
    assert any(problem.startswith("train-1.parquet: model: missing") for problem in found["detail"])


def test_an_unreadable_shard_stops_the_check_with_its_report_written(tmp_path):
    (tmp_path / "data").mkdir()
    pq.write_table(build.written(build.SCHEMA).empty_table(), tmp_path / "data" / "train-0.parquet")
    (tmp_path / "data" / "train-1.parquet").write_bytes(b"")  # cut off by an interrupted build
    found = stopped_report(tmp_path)[
        "the build's schema: columns, nested fields and required values"
    ]
    assert found["pass"] is False
    assert any(problem.startswith("train-1.parquet: unreadable") for problem in found["detail"])


def test_a_shard_whose_nullability_differs_stops_the_check_with_its_report_written(tmp_path):
    (tmp_path / "data").mkdir()
    written = build.written(build.SCHEMA)
    strict = written.set(0, written.field("episode_id").with_nullable(False))
    table = pa.Table.from_pylist([well_formed_row()], schema=strict)
    pq.write_table(table, tmp_path / "data" / "train-0.parquet")
    found = stopped_report(tmp_path)[
        "the build's schema: columns, nested fields and required values"
    ]
    assert found["pass"] is False
    assert any(p.startswith("train-0.parquet: episode_id: nullable") for p in found["detail"])


def published_copy(tmp_path):
    """A build folder and its published files, the same bytes: a card and two shards."""
    built, published = tmp_path / "built", tmp_path / "published"
    files = {}
    for name, content in (("README.md", b"card"), ("data/train-0.parquet", b"a"),
                          ("data/train-1.parquet", b"b")):  # fmt: skip
        (built / name).parent.mkdir(parents=True, exist_ok=True)
        (built / name).write_bytes(content)
        (published / name).parent.mkdir(parents=True, exist_ok=True)
        (published / name).write_bytes(content)
        files[name] = published / name
    return built, files


def test_a_build_equal_to_the_published_files_byte_for_byte_has_no_differences(tmp_path):
    built, files = published_copy(tmp_path)
    assert check.byte_differences(built, files) == []


def test_each_file_that_differs_from_the_published_one_is_named(tmp_path):
    built, files = published_copy(tmp_path)
    (built / "README.md").write_bytes(b"card!")
    (built / "data" / "train-1.parquet").unlink()
    (built / "data" / "train-2.parquet").write_bytes(b"c")
    assert check.byte_differences(built, files) == [
        "README.md: differs",
        "data/train-1.parquet: only in the published revision",
        "data/train-2.parquet: only in the build",
    ]
