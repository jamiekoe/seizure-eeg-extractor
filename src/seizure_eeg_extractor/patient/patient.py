from typing import Optional
from pathlib import Path
import shutil


class Patient:
    """Base patient metadata shared by dataset-specific patient classes."""

    @staticmethod
    def create_directory(path: str | Path) -> Path:
        """Create an output directory and return it as a `Path`."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def create_clean_directory(path: str | Path) -> Path:
        """Create an empty output directory, removing stale contents first."""
        path = Path(path)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def __init__(self, pid: str, input_path: str | Path) -> None:
        self._pid = pid
        self._input_path = Path(input_path)
        self._patient_path = self._input_path / pid
        self._age: Optional[float] = None
        self._gender: Optional[str] = None
        self._fs: Optional[float] = None
        self._filenames: list[str] = []

    @property
    def pid(self) -> str:
        return self._pid

    @property
    def input_path(self) -> Path:
        return self._input_path

    @property
    def patient_path(self) -> Path:
        return self._patient_path

    @property
    def age(self) -> Optional[float]:
        return self._age

    @age.setter
    def age(self, age: float) -> None:
        self._age = age

    @property
    def gender(self) -> Optional[str]:
        return self._gender

    @gender.setter
    def gender(self, gender: str) -> None:
        self._gender = gender

    @property
    def fs(self) -> Optional[float]:
        return self._fs

    @fs.setter
    def fs(self, fs: float) -> None:
        self._fs = fs

    @property
    def filenames(self) -> list[str]:
        return self._filenames

    @filenames.setter
    def filenames(self, filenames: list[str]) -> None:
        self._filenames = filenames
