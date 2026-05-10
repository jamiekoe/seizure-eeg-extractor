from pathlib import Path
from typing import Any, NamedTuple, Optional
import numpy as np
import datetime
import pickle
from dataclasses import dataclass, asdict


@dataclass
class FileDict:
    """Serializable metadata schema written to each `info.pkl` file."""

    pid: str
    fid: str
    fs: float
    num_samples: int
    num_seizures: int
    seizure_times: Optional[list[dict[str, Any]]]
    num_channels: int
    channel_names: Optional[list[str]]
    eeg_dtype: str
    file_start_time: Optional[dict[str, Any]]
    file_end_time: Optional[dict[str, Any]] = None
    duration_in_sec: Optional[int] = None


class Timestamp(NamedTuple):
    """Date/time pair used for source file and seizure timestamps."""

    date: Optional[datetime.date] = None
    time: Optional[datetime.time] = None


class SeizureInfo(NamedTuple):
    """Seizure interval metadata.

    Sample indices are always local to the extracted record when written to
    disk. Absolute timestamps are kept when the source dataset provides them.
    """

    onset_index: int
    offset_index: int
    onset_second: Optional[int] = None
    offset_second: Optional[int] = None
    onset_timestamp: Optional[Timestamp] = None
    offset_timestamp: Optional[Timestamp] = None


class File:
    """Base class for source recording files.

    Subclasses populate EEG samples, channel metadata, sampling frequency, and
    seizure intervals. This base class handles the common serialization to
    `eeg.npy` and `info.pkl`.
    """

    def __init__(self, pid: str, fid: str, fs: Optional[float] = None) -> None:
        self._pid = pid
        self._fid = fid
        self._fs = fs
        self._num_samples: Optional[int] = None
        self._duration_in_sec: Optional[int] = None
        self._seizure_times: Optional[list[SeizureInfo]] = None
        self._channel_names: Optional[list[str]] = None
        self._num_channels: Optional[int] = None
        self._eeg: Optional[np.ndarray] = None
        self._file_start_timestamp: Optional[Timestamp] = None
        self._file_end_timestamp: Optional[Timestamp] = None

    @property
    def pid(self) -> str:
        return self._pid

    @property
    def fid(self) -> str:
        return self._fid

    @property
    def fs(self) -> Optional[float]:
        return self._fs

    @fs.setter
    def fs(self, fs: float) -> None:
        self._fs = fs

    @property
    def num_samples(self) -> Optional[int]:
        return self._num_samples

    @num_samples.setter
    def num_samples(self, num_samples: int) -> None:
        self._num_samples = num_samples

    @property
    def duration_in_sec(self) -> Optional[int]:
        return self._duration_in_sec

    @duration_in_sec.setter
    def duration_in_sec(self, duration_in_sec: int) -> None:
        self._duration_in_sec = duration_in_sec

    @property
    def file_start_timestamp(self) -> Optional[Timestamp]:
        return self._file_start_timestamp

    @file_start_timestamp.setter
    def file_start_timestamp(self, time: Timestamp) -> None:
        self._file_start_timestamp = time

    @property
    def file_end_timestamp(self) -> Optional[Timestamp]:
        return self._file_end_timestamp

    @file_end_timestamp.setter
    def file_end_timestamp(self, time: Timestamp) -> None:
        self._file_end_timestamp = time

    @property
    def channel_names(self) -> Optional[list[str]]:
        return self._channel_names

    @channel_names.setter
    def channel_names(self, names: Optional[list[str]]) -> None:
        self._channel_names = names

    @property
    def num_channels(self) -> Optional[int]:
        return self._num_channels

    @num_channels.setter
    def num_channels(self, num_channels: int) -> None:
        self._num_channels = num_channels

    @property
    def seizure_times(self) -> Optional[list[SeizureInfo]]:
        return self._seizure_times

    @seizure_times.setter
    def seizure_times(self, times: list[SeizureInfo]) -> None:
        self._seizure_times = times

    @property
    def num_seizures(self) -> int:
        return len(self.seizure_times) if self.seizure_times is not None else 0

    @property
    def eeg(self) -> Optional[np.ndarray]:
        return self._eeg

    @eeg.setter
    def eeg(self, eeg: np.ndarray) -> None:
        self._eeg = eeg

    def serialize_seizure_times(self) -> list[dict[str, Any]]:
        """Convert `SeizureInfo` objects into pickle-friendly dictionaries."""
        if self.seizure_times is None:
            return []
        seizure_times = []
        for time_ in self.seizure_times:
            dict_ = time_._asdict()
            for k, v in dict_.items():
                if isinstance(v, Timestamp):
                    dict_[k] = v._asdict()
            seizure_times.append(dict_)
        return seizure_times

    def save_record_data(self, output_path: str | Path, output_dtype: str | np.dtype = "float32") -> None:
        """Write one extracted record as `eeg.npy` and `info.pkl`.

        The saved NumPy array uses `(num_samples, num_channels)` layout. The
        metadata pickle stores enough information to validate the array shape
        and reconstruct seizure sample windows without importing this package.
        """
        fs = self.fs
        num_samples = self.num_samples
        num_channels = self.num_channels
        eeg = self.eeg

        if fs is None:
            raise ValueError(f"Sampling frequency is missing for {self.pid}/{self.fid}")
        if num_samples is None:
            raise ValueError(f"Number of samples is missing for {self.pid}/{self.fid}")
        if num_channels is None:
            raise ValueError(f"Number of channels is missing for {self.pid}/{self.fid}")
        if eeg is None:
            raise ValueError(f"EEG data is missing for {self.pid}/{self.fid}")

        output_dtype = np.dtype(output_dtype)
        if output_dtype not in {np.dtype("float32"), np.dtype("float64")}:
            raise TypeError("output_dtype must be float32 or float64")
        eeg_to_save = eeg.astype(output_dtype, copy=False)

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        file_dict = asdict(FileDict(
            pid=self.pid,
            fid=self.fid,
            fs=fs,
            num_samples=num_samples,
            num_seizures=self.num_seizures,
            seizure_times=self.serialize_seizure_times() if self.seizure_times is not None else None,
            num_channels=num_channels,
            channel_names=self.channel_names,
            eeg_dtype=str(eeg_to_save.dtype),
            file_start_time=self.file_start_timestamp._asdict() if self.file_start_timestamp is not None else None,
            file_end_time=self.file_end_timestamp._asdict() if self.file_end_timestamp is not None else None,
            duration_in_sec=self.duration_in_sec,
        ))
        with (output_path / "info.pkl").open("wb") as f:
            pickle.dump(file_dict, f)
        np.save(output_path / "eeg.npy", eeg_to_save)
