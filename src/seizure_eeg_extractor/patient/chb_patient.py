import os
import warnings
import re
from pathlib import Path
import numpy as np
from tqdm import trange  # type: ignore
from .patient import Patient
from ..file import EDF


class CHBPatient(Patient):
    """CHB-MIT patient processor.

    A patient is processed by reading the patient summary text file, matching
    each EDF file to its summary section, reading the EDF samples, and writing a
    record directory for each source EDF.
    """

    def __init__(self, pid: str, input_path: str | Path) -> None:
        super().__init__(pid, input_path)

    def process_patient(self, output_path: str | Path, pos: int, output_dtype: str | np.dtype = "float32") -> None:
        """Extract all EDF records for this patient."""
        summary_path = self.patient_path / f"{self.pid}-summary.txt"
        try:
            f = summary_path.open("r", encoding="utf-8")
        except OSError as err:
            warnings.warn(f"Could not open {self.pid}-summary.txt -> Patient {self.pid} will be skipped!")
            warnings.warn(str(err))
            return
        with f:
            lines = f.readlines()
            self.fs = self._extract_fs(lines)
            self.filenames = self._extract_filenames()
            self._process_files(lines, Path(output_path), pos, output_dtype)

    def _extract_fs(self, lines: list[str]) -> float:
        """Extract sampling frequency from CHB-MIT summary text."""
        found_none = f"""Unable to extract sampling frequency from text file for patient {self.pid}.
        Sampling frequency will be set to 256 Hz, which is common to all CHB patients."""
        found_too_many = f"Found more than one sampling frequency for patient {self.pid}!"
        lines = [s for s in lines if "sampling rate" in s.lower()]
        if not lines:
            warnings.warn(found_none)
            return 256.0
        if len(lines) > 1:
            warnings.warn(found_too_many)
        opts = re.findall(r"(\d*\.\d+|\d+)(?: Hz)", lines[0])
        if not opts:
            warnings.warn(found_none)
            return 256.0
        if len(opts) > 1:
            warnings.warn(found_too_many)
        fs = float(opts[0])
        return fs

    def _extract_filenames(self) -> list[str]:
        """Return EDF filenames sorted by filesystem path."""
        filenames = [path.name for path in sorted(self.patient_path.glob("*.edf"))]
        if not filenames:
            raise ValueError(f"Unable to find files for patient {self.pid}!")
        return filenames

    def _process_files(self, lines: list[str], output_path: Path, pos: int, output_dtype: str | np.dtype) -> None:
        """Process each EDF file and write `record_<n>` outputs."""
        patient_output_path = self.create_clean_directory(output_path / self.pid)
        for j in trange(len(self.filenames), desc=f"Processing CHB patient {self.pid}", leave=False, position=pos):
            fn = self.filenames[j]
            fid, _ = os.path.splitext(fn)
            if self.fs is None:
                raise ValueError(f"Sampling frequency has not been set for patient {self.pid}")
            edf = EDF(self.pid, fid, self.fs, self.patient_path)
            save_path = self.create_directory(patient_output_path / f"record_{j}")
            for i, line in enumerate(lines):
                if fn in line:
                    edf.process_edf(lines, i)
                    edf.save_record_data(save_path, output_dtype=output_dtype)
                    break
            else:  # for patient chb24
                chb24_interictal_files = {
                    "chb24_02",
                    "chb24_05",
                    "chb24_08",
                    "chb24_10",
                    "chb24_12",
                    "chb24_16",
                    "chb24_18",
                    "chb24_19",
                    "chb24_20",
                    "chb24_22",
                }
                if self.pid != "chb24" or fid not in chb24_interictal_files:
                    raise ValueError(
                        f"Could not find summary metadata for EDF file {fn} in patient {self.pid}."
                    )
                edf.seizure_times = []
                edf.extract_channels_and_eeg()
                edf.save_record_data(save_path, output_dtype=output_dtype)
