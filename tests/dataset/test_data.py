"""The published dataset's files are read from the local Hugging Face cache without a network call,
and fetched only when asked; the revision is the commit the cache holds.
"""

import re

import huggingface_hub
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from ping_or_peek.dataset import data

FILES = ("data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet")
NOT_CACHED = "isn't in the local Hugging Face cache"


def test_the_source_is_the_datasets_latest_revision():
    assert data.SOURCE == data.Dataset("snorkelai/ping-or-peek", FILES)
    assert data.SOURCE.revision is None


def test_an_unfetched_revision_fails_saying_so_with_no_network_call():
    # local_files_only: an uncached revision raises without a request to the Hub.
    absent = data.Dataset(data.SOURCE.repo, data.SOURCE.files, "0" * 40)
    with pytest.raises(data.NotFetched, match=NOT_CACHED):
        data.files(absent)


def test_a_source_that_is_not_a_dataset_fails_naming_the_rule(tmp_path):
    # Every reader of a source needs its repo and revision, so a source is a Dataset only.
    with pytest.raises(TypeError, match="Dataset"):
        data.files(tmp_path / "episodes.parquet")


def test_the_revision_is_the_commit_the_cache_holds():
    """The revision is the commit the cache's copy of the latest files is from: a full sha, the
    snapshot folder that holds them."""
    try:
        cached = data.files(data.SOURCE)
    except data.NotFetched as error:
        pytest.skip(str(error))
    revision = data.revision(data.SOURCE)
    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    assert all(revision in path.parts for path in cached)


def test_an_unfetched_latest_revision_fails_saying_so():
    elsewhere = data.Dataset("snorkelai/not-a-dataset-anyone-fetched", data.SOURCE.files)
    with pytest.raises(data.NotFetched, match=NOT_CACHED):
        data.revision(elsewhere)


def test_a_failed_fetch_names_its_cause_not_the_cache(monkeypatch):
    # Offline mode makes the Hub request fail as a lost connection would, without a network call.
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    absent = data.Dataset(data.SOURCE.repo, data.SOURCE.files, "0" * 40)
    with pytest.raises(LocalEntryNotFoundError) as error:
        data.files(absent, fetch=True)
    assert NOT_CACHED not in str(error.value)
