import warnings
from pathlib import Path
import numpy as np
from tqdm import trange  # type: ignore
from datetime import date, time, datetime
from .patient import Patient
from ..file import Binary, Timestamp, SeizureInfo


class EUPatient(Patient):
    """EU patient processor.

    A patient is processed by reading the patient-level seizure list, locating
    all binary `.data` recordings, then letting `Binary` parse each matching
    `.head` file and assign overlapping seizure intervals.
    """

    @staticmethod
    def _remove_new_line_chrs(lines: list[str]) -> list[str]:
        """Strip whitespace from seizure-list lines."""
        return [line.strip() for line in lines]

    @staticmethod
    def _extract_seizure_timestamp(date_string: str, time_string: str) -> Timestamp:
        """Parse a date/time pair from `seizurelist.txt` into a timestamp."""
        year, month, day = date_string.split("-")
        seizure_date = date(int(year), int(month), int(day))
        hr, min_, sec_text = time_string.split(":")
        sec, _, fraction = sec_text.partition(".")
        usec = int((fraction + "000000")[:6]) if fraction else 0
        seizure_time = time(int(hr), int(min_), int(sec), usec)
        return Timestamp(date=seizure_date, time=seizure_time)

    @classmethod
    def _extract_seizure_info(cls, lines: list[str]) -> list[SeizureInfo]:
        """Parse seizure onset/offset rows from `seizurelist.txt`.

        The EU seizure list includes absolute timestamps and dataset-level
        sample indices. `Binary` later converts the timestamps into record-local
        sample intervals for each overlapping file.
        """
        seizure_info = []
        start_idx = None
        for i, line in enumerate(lines):
            normalized = line.lower()
            if "onset" in normalized or "offset" in normalized or "onset_sample" in normalized or "offset_sample" in normalized:
                start_idx = i + 1
                break
        if start_idx is None:
            raise ValueError("Could not find seizure table header in seizurelist.txt")
        for line in lines[start_idx:]:
            if not line:
                continue
            onset_date, onset_time, offset_date, offset_time, onset_index, offset_index = line.split()
            seizure_onset_timestamp = cls._extract_seizure_timestamp(onset_date, onset_time)
            seizure_offset_timestamp = cls._extract_seizure_timestamp(offset_date, offset_time)
            seizure_onset_index = int(onset_index)
            seizure_offset_index = int(offset_index)
            seizure = SeizureInfo(
                onset_index=seizure_onset_index,
                offset_index=seizure_offset_index,
                onset_timestamp=seizure_onset_timestamp,
                offset_timestamp=seizure_offset_timestamp
            )
            seizure_info.append(seizure)
        return sorted(seizure_info, key=cls._seizure_onset_datetime)

    @staticmethod
    def _seizure_onset_datetime(seizure: SeizureInfo) -> datetime:
        """Return a sortable absolute onset datetime for a seizure."""
        timestamp = seizure.onset_timestamp
        if timestamp is None or timestamp.date is None or timestamp.time is None:
            raise ValueError("Seizure onset timestamp is incomplete")
        return datetime.combine(timestamp.date, timestamp.time)

    def __init__(self, pid: str, input_path: str | Path) -> None:
        super().__init__(pid, input_path)

    def process_patient(self, output_path: str | Path, pos: int, output_dtype: str | np.dtype = "float32") -> None:
        """Extract all binary records for this patient."""
        seizure_list_paths = list(self.patient_path.rglob("seizurelist.txt"))
        if not seizure_list_paths:
            warnings.warn(f"Could not find seizurelist.txt -> Patient {self.pid} will be skipped!")
            return
        seizure_list_path = seizure_list_paths[0]
        try:
            f = seizure_list_path.open("r", encoding="utf-8")
        except OSError as err:
            warnings.warn(f"Could not find or open {seizure_list_path.name}"
                          f" -> Patient {self.pid} will be skipped!")
            warnings.warn(str(err))
            return
        with f:
            lines = self._remove_new_line_chrs(f.readlines())
            seizure_info = self._extract_seizure_info(lines)
            self.filenames = self._extract_filenames()
            self._process_files(Path(output_path), pos, seizure_info, output_dtype)

    def _extract_filenames(self) -> list[str]:
        """Return binary recording IDs discovered from `.data` files."""
        filenames = sorted([path.stem for path in self.patient_path.rglob("*.data")])
        if not filenames:
            raise ValueError(f"Unable to find files for patient {self.pid}!")
        return filenames

    def _process_files(self, output_path: Path, pos: int, seizure_info: list[SeizureInfo],
                       output_dtype: str | np.dtype) -> None:
        """Process each EU binary file and write `record_<n>` outputs."""
        patient_output_path = self.create_clean_directory(output_path / self.pid)
        for j in trange(len(self.filenames), desc=f"Processing EU patient {self.pid}", leave=False, position=pos):
            save_path = self.create_directory(patient_output_path / f"record_{j}")
            fid = self.filenames[j]
            binary = Binary(self.pid, fid, self.patient_path)
            binary.process_binary(seizure_info)
            binary.save_record_data(save_path, output_dtype=output_dtype)
