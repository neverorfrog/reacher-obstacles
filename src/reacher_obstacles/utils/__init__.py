import os
from typing import Type, TypeVar

from typing import Literal


def project_root() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    max_iterations = 100  # Set a limit for the number of iterations
    for _ in range(max_iterations):
        if (
            "requirements.txt" in os.listdir(current_dir)
            or "setup.py" in os.listdir(current_dir)
            or "pyproject.toml" in os.listdir(current_dir)
        ):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    raise FileNotFoundError(
        "requirements.txt not found in any parent directories within the iteration limit"
    )


def src_dir() -> str:
    return project_root() + "/src/reacher_obstacles"