import os
from abc import ABC
from pathlib import Path
from typing import Union


class InputOutput(ABC):
    @staticmethod
    def is_exist_path(path: Union[str, Path]):
        return os.path.exists(path)

    @staticmethod
    def create_dir(path: Union[str, Path]):
        return Path(path).mkdir(parents=True, exist_ok=True)
