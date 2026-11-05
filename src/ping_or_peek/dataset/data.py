"""The published Hugging Face dataset, snorkelai/ping-or-peek: `SOURCE` names its latest revision,
`files` reads its files from the local Hugging Face cache, or fetches them, and `revision` names
the commit the cache holds. `dataset.check` compares a rebuild with these files. Fetching needs the
network, and while the dataset is private a token (the `HF_TOKEN` environment variable, or
`hf auth login`); the notebook loads the dataset on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError


@dataclass(frozen=True)
class Dataset:
    """A Hugging Face dataset, the files read from it, and the revision they are read at: a full
    commit sha, or None for the latest (its `main`)."""

    repo: str
    files: tuple[str, ...]
    revision: str | None = None


REPO = "snorkelai/ping-or-peek"
SOURCE = Dataset(REPO, ("data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"))


class NotFetched(FileNotFoundError):
    """A dataset file isn't in the local Hugging Face cache."""


def files(source: Dataset, fetch: bool = False) -> list[Path]:
    """`source`'s files in the local Hugging Face cache, downloaded first when `fetch` (a network
    call). Raises NotFetched, saying so, when one is absent and `fetch` is false; a failed fetch
    raises the Hub's own error, which names its cause (offline mode, a lost connection, no
    access). Raises TypeError when `source` isn't a Dataset."""
    if not isinstance(source, Dataset):
        raise TypeError(
            f"a source is a dataset.data.Dataset (repo, files, revision), not {source!r}: "
            "every reader of it needs the dataset's repo and files"
        )
    try:
        return [
            Path(
                hf_hub_download(
                    source.repo,
                    name,
                    repo_type="dataset",
                    revision=source.revision,
                    local_files_only=not fetch,
                )
            )
            for name in source.files
        ]
    except LocalEntryNotFoundError as error:
        if fetch:
            raise
        at = f"revision {source.revision}" if source.revision else "its latest revision"
        raise NotFetched(
            f"{source.repo} at {at} isn't in the local Hugging Face cache; fetching it needs the "
            "network (`files(source, fetch=True)`)"
        ) from error


def revision(source: Dataset) -> str:
    """The commit `source`'s files are read at: its revision, or for the latest, the one the local
    cache holds (the snapshot its files are in, as they were last fetched).
    Raises NotFetched as `files` does, and ValueError, naming the rule, when the cached files come
    from different commits."""
    commits = {
        path.parents[len(Path(name).parts) - 1].name
        for name, path in zip(source.files, files(source), strict=True)
    }
    if len(commits) != 1:
        raise ValueError(
            f"{source.repo}'s cached files come from {len(commits)} commits, not one: fetch them "
            "again"
        )
    return commits.pop()
