"""Regenerate the golden task directory: uv run python -m tests.make_golden"""

from pathlib import Path

from ping_or_peek.config import condition
from ping_or_peek.tasks.harbor_task import measured_task, write_task

if __name__ == "__main__":
    spec = measured_task(condition("user_available_checker_exposed"), 100)
    print(write_task(spec, Path(__file__).parent / "golden"))
