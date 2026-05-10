import numpy as np
from datetime import datetime, date, time, timedelta
from pathlib import Path
from typing import Optional, overload

from .file import File, Timestamp, SeizureInfo


class Binary(File):
    """Reader for one EU `.head` / `.data` recording pair."""

    @staticmethod
    def _extract_header_value(name: str, lines: list[str]) -> str:
        """Extract an exact `name=value` header value."""
        for line in lines:
            key, separator, value = line.partition("=")
            if separator and key.strip() == name:
                return value.strip()
        raise ValueError(f"Could not find required header parameter: {name}")

    @overload
    @staticmethod
    def _extract_parameter(name: str, lines: list[str], class_: type[int]) -> int:
        ...

    @overload
    @staticmethod
    def _extract_parameter(name: str, lines: list[str], class_: type[float]) -> float:
        ...

    @staticmethod
    def _extract_parameter(name: str, lines: list[str], class_: type[int] | type[float]) -> int | float:
        """Extract a numeric `name=value` parameter from a header file."""
        return class_(Binary._extract_header_value(name, lines))

    @staticmethod
    def _extract_channels(lines: list[str]) -> Optional[list[str]]:
        """Extract channel names from the optional `elec_names` header field."""
        try:
            value = Binary._extract_header_value("elec_names", lines)
        except ValueError:
            return None
        return [name.strip() for name in value.strip("[]").split(",") if name.strip()]

    @staticmethod
    def _extract_file_timestamp(lines: list[str]) -> Timestamp:
        """Extract the absolute file start timestamp from `start_ts`."""
        # String looks like: start_ts=2009-06-18 11:54:01.000000
        date_and_time = Binary._extract_header_value("start_ts", lines)
        date_, time_text = date_and_time.split()
        year, month, day = date_.split("-")
        hr, min_, sec_text = time_text.split(":")
        sec, _, fraction = sec_text.partition(".")
        usec = int((fraction + "000000")[:6]) if fraction else 0
        file_date = date(int(year), int(month), int(day))
        file_time = time(int(hr), int(min_), int(sec), usec)
        return Timestamp(date=file_date, time=file_time)

    @staticmethod
    def _timestamp_to_datetime(timestamp: Optional[Timestamp], label: str) -> datetime:
        """Convert a complete `Timestamp` into `datetime` for interval math."""
        if timestamp is None or timestamp.date is None or timestamp.time is None:
            raise ValueError(f"{label} timestamp is incomplete")
        return datetime.combine(timestamp.date, timestamp.time)

    def __init__(self, pid: str, fid: str, patient_path: str | Path) -> None:
        super().__init__(pid, fid)
        self._patient_path = Path(patient_path)

    def _assign_seizure_info(self, seizure_info: list[SeizureInfo]) -> list[SeizureInfo]:
        """Return seizure intervals that overlap this binary file.

        EU seizure annotations are absolute patient-level intervals. This method
        intersects each seizure with the current file's absolute time span and
        converts the overlap into record-local sample indices. Cross-boundary
        seizures are clipped to `[0, num_samples]`.
        """
        seizures_in_file = []
        if self.duration_in_sec is None:
            raise ValueError(f"Duration is missing for {self.pid}/{self.fid}")
        if self.fs is None:
            raise ValueError(f"Sampling frequency is missing for {self.pid}/{self.fid}")
        if self.num_samples is None:
            raise ValueError(f"Number of samples is missing for {self.pid}/{self.fid}")
        file_onset_datetime = self._timestamp_to_datetime(self.file_start_timestamp, "File onset")
        file_offset_datetime = file_onset_datetime + timedelta(seconds=self.duration_in_sec)
        for i, seizure in enumerate(seizure_info):
            # Exclude 12th seizure in patient FR_548 (>8 hrs long)
            if self.pid == "pat_FR_548" and i + 1 == 12:
                continue
            seizure_onset_datetime = self._timestamp_to_datetime(seizure.onset_timestamp, "Seizure onset")
            seizure_offset_datetime = self._timestamp_to_datetime(seizure.offset_timestamp, "Seizure offset")
            if seizure_onset_datetime >= file_offset_datetime or seizure_offset_datetime <= file_onset_datetime:
                continue
            onset_delta = max(0.0, (seizure_onset_datetime - file_onset_datetime).total_seconds())
            offset_delta = min(
                float(self.duration_in_sec),
                (seizure_offset_datetime - file_onset_datetime).total_seconds(),
            )
            onset_index = max(0, min(self.num_samples, round(onset_delta * self.fs)))
            offset_index = max(0, min(self.num_samples, round(offset_delta * self.fs)))
            if onset_index < offset_index:
                seizures_in_file.append(SeizureInfo(
                    onset_index=onset_index,
                    offset_index=offset_index,
                    onset_second=round(onset_delta),
                    offset_second=round(offset_delta),
                    onset_timestamp=seizure.onset_timestamp,
                    offset_timestamp=seizure.offset_timestamp,
                ))
        return seizures_in_file

    def _extract_eeg(self, basename: Path, sample_bytes: int, conversion_factor: float) -> None:
        """Load the binary sample file and apply the header conversion factor."""
        num_channels = self.num_channels
        num_samples = self.num_samples
        if num_channels is None:
            raise ValueError(f"Number of channels is missing for {self.pid}/{self.fid}")
        if num_samples is None:
            raise ValueError(f"Number of samples is missing for {self.pid}/{self.fid}")
        dtype = "int16" if sample_bytes == 2 else "int32"
        data_path = basename.with_suffix(".data")
        raw = np.fromfile(str(data_path), dtype=dtype)
        expected_values = num_samples * num_channels
        if raw.size != expected_values:
            raise ValueError(
                f"Expected {expected_values} values in {data_path}, found {raw.size}"
            )
        eeg = raw.reshape((num_samples, num_channels)) * conversion_factor
        self.eeg = eeg

    def process_binary(self, seizure_info: list[SeizureInfo]) -> None:
        """Populate metadata, EEG samples, and seizure intervals for one file."""
        head_paths = sorted(self._patient_path.rglob(f"{self.fid}.head"))
        if not head_paths:
            raise FileNotFoundError(f"Could not find header file for {self.pid}/{self.fid}")
        basename = head_paths[0].with_suffix("")
        with basename.with_suffix(".head").open(encoding="utf-8") as file:
            lines = file.read().splitlines()
        self.fs = self._extract_parameter("sample_freq", lines, float)
        self.num_samples = self._extract_parameter("num_samples", lines, int)
        self.duration_in_sec = self._extract_parameter("duration_in_sec", lines, int)
        self.num_channels = self._extract_parameter("num_channels", lines, int)
        self.channel_names = self._extract_channels(lines)
        self.file_start_timestamp = self._extract_file_timestamp(lines)
        conversion_factor = self._extract_parameter("conversion_factor", lines, float)
        sample_bytes = self._extract_parameter("sample_bytes", lines, int)
        if sample_bytes not in {2, 4}:
            raise ValueError(f"Expected sample_bytes to be 2 or 4, got {sample_bytes}")
        self._extract_eeg(basename, sample_bytes, conversion_factor)
        self.seizure_times = self._assign_seizure_info(seizure_info)
