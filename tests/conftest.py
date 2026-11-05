import importlib.util

import pytest

from ping_or_peek import config as project_config


@pytest.fixture(autouse=True)
def no_repository_dotenv(tmp_path, monkeypatch):
    """Point `project_config.load_env()` at a .env that doesn't exist, so no test reads the real one."""
    monkeypatch.setattr(project_config, "DOTENV", tmp_path / "no-such-dir" / ".env")


@pytest.fixture
def task_config():
    """Harbor's `TaskConfig` class, for tests that validate a written `task.toml`.

    Skips the test when Harbor isn't installed: CI's sync leaves it out, because `pyproject.toml`
    takes Harbor from our fork, which CI's token can't read. A Harbor that is installed but fails
    to import raises instead of skipping.
    """
    if importlib.util.find_spec("harbor") is None:
        pytest.skip("Harbor is not installed; CI's sync leaves it out")
    from harbor.models.task.config import TaskConfig

    return TaskConfig
