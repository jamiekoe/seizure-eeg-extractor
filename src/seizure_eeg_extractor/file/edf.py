import pyedflib  # type: ignore
import numpy as np
from datetime import time
from pathlib import Path

from .file import File, Timestamp, SeizureInfo


class EDF(File):
    """Reader for one CHB-MIT EDF file and its summary metadata."""

    @staticmethod
    def _extract_file_timestamp(key: str, line: str) -> Timestamp:
        """Parse a CHB-MIT summary time field.

        Some CHB-MIT summary files represent post-midnight times as hours above
        23. The extractor wraps those hours into Python's `time` range.
        """
        if key not in line:
            raise ValueError(f"Key {key} not in line {line}!")
        hour_s, minute_s, second_s = line.split()[-1].split(":")
        hour, minute, second = int(hour_s), int(minute_s), int(second_s)
        if hour >= 24:
            hour %= 24
        file_time = time(hour, minute, second)
        return Timestamp(time=file_time)

    @staticmethod
    def _extract_num_seizures(key: str, line: str) -> int:
        if key not in line:
            raise ValueError(f"Key {key} not in line {line}!")
        num_seizures = line.split()[-1]
        return int(num_seizures)

    @staticmethod
    def _extract_seizure_time(key: str, line: str) -> int:
        if key not in line:
            raise ValueError(f"Key {key} not in line {line}!")
        time_ = line.split()[-2]
        return int(time_)

    def __init__(self, pid: str, fid: str, fs: float, patient_path: str | Path) -> None:
        super().__init__(pid, fid, fs)
        self._patient_path = Path(patient_path)

    def _extract_seizure_info(self, lines: list[str], i: int) -> list[SeizureInfo]:
        """Extract record-local seizure intervals from summary lines."""
        if self.fs is None:
            raise ValueError(f"Sampling frequency is missing for {self.pid}/{self.fid}")
        num_seizures = self._extract_num_seizures("Number of Seizures", lines[i])
        seizure_times = []
        for j in range(num_seizures):
            idx = i + 1 + j * 2
            start_sec = self._extract_seizure_time("Start Time", lines[idx])
            end_sec = self._extract_seizure_time("End Time", lines[idx + 1])
            seizure = SeizureInfo(int(start_sec*self.fs), int(end_sec*self.fs), start_sec, end_sec)
            seizure_times.append(seizure)
        return seizure_times

    def extract_channels_and_eeg(self) -> None:
        """Read channel labels and signal arrays from the EDF file."""
        reader = pyedflib.EdfReader(str(self._patient_path / f"{self.fid}.edf"))
        try:
            self.channel_names = reader.getSignalLabels()
            num_channels = reader.signals_in_file
            eeg = np.zeros((reader.getNSamples()[0], num_channels), dtype=float)
            for i in range(num_channels):
                eeg[:, i] = reader.readSignal(i)
            self.eeg = eeg
            self.num_samples = len(eeg)
            self.num_channels = num_channels
        finally:
            reader.close()

    def process_edf(self, lines: list[str], i: int) -> None:
        """Populate metadata and EEG samples for this EDF record."""
        has_start_end_times = (
            i + 2 < len(lines)
            and "Start Time" in lines[i + 1]
            and "End Time" in lines[i + 2]
        )
        if has_start_end_times:
            self.file_start_timestamp = self._extract_file_timestamp("Start Time", lines[i + 1])
            self.file_end_timestamp = self._extract_file_timestamp("End Time", lines[i + 2])
            self.seizure_times = self._extract_seizure_info(lines, i + 3)
        else:  # for patient chb24
            self.seizure_times = self._extract_seizure_info(lines, i + 1)
        self.extract_channels_and_eeg()
