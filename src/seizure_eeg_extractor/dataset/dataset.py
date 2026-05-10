from pathlib import Path
from typing import Optional
import warnings
import concurrent.futures as cf
from abc import ABC, abstractmethod

import numpy as np

from ..patient import CHBPatient, EUPatient


class Dataset(ABC):
    """Common patient discovery and processing workflow for source datasets.

    Subclasses provide the complete list of known patient IDs, validate patient
    folder naming, and construct dataset-specific patient objects. Extraction is
    parallelized across patients, while each patient processes its own records
    sequentially for deterministic record numbering.
    """

    _PATIENTS_ALL: list[str] = []

    @staticmethod
    def _get_base_name(path: Path) -> str:
        return path.name

    @staticmethod
    def _get_patient_ids(
        input_path: Path,
        patients_wanted: Optional[list[str]],
        all_patients: list[str],
        key: str,
    ) -> list[str]:
        """Return available patient IDs in canonical dataset order.

        The extractor accepts partial local dataset copies. Missing known
        patients produce warnings, while requesting only missing patients is an
        error because there would be nothing to process.
        """
        patients_have = [
            Dataset._get_base_name(path)
            for path in sorted(input_path.glob(f"{key}*"))
            if path.is_dir() and Dataset._get_base_name(path) in all_patients
        ]
        if not patients_have:
            raise ValueError("No patient folders found!")
        patients_have_as_set = set(patients_have)
        if patients_wanted is None and patients_have_as_set != set(all_patients):
            warnings.warn(f"Only found {len(patients_have)} out of {len(all_patients)} possible patient folders!")
        if patients_wanted is None:
            return [patient for patient in all_patients if patient in patients_have_as_set]
        patients_wanted_as_set = set(patients_wanted)
        set_difference = sorted(patients_wanted_as_set.difference(patients_have_as_set))
        if set_difference:
            warnings.warn(f"The following patient folders were not found: {set_difference}")
        selected_patients = [patient for patient in all_patients if patient in patients_wanted_as_set & patients_have_as_set]
        if not selected_patients:
            raise ValueError("None of the requested patient folders were found!")
        return selected_patients

    def __init__(self, input_path: str | Path, output_path: Optional[str | Path],
                 patients_wanted: Optional[list[str]], key: str,
                 output_dtype: str | np.dtype = "float32") -> None:
        self._input_path = Path(input_path).expanduser().resolve()
        if not self._input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {self._input_path}")
        if not self._input_path.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {self._input_path}")
        if output_path is None:
            output_path = self._input_path / "extracted_data"
        self._output_path = Path(output_path).expanduser().resolve()
        self._output_path.mkdir(parents=True, exist_ok=True)
        self._output_dtype = self._validate_output_dtype(output_dtype)
        self._check_format(patients_wanted)
        if patients_wanted is not None and not patients_wanted:
            raise ValueError("patients_wanted is empty!")
        self._patient_ids = self._get_patient_ids(self._input_path, patients_wanted, self._PATIENTS_ALL, key)
        self._patients: Optional[dict[str, CHBPatient | EUPatient]] = None

    @property
    def input_path(self) -> Path:
        return self._input_path

    @property
    def output_path(self) -> Path:
        return self._output_path

    @property
    def output_dtype(self) -> np.dtype:
        return self._output_dtype

    @property
    def patient_ids(self) -> list[str]:
        return self._patient_ids

    @property
    def patients(self) -> Optional[dict[str, CHBPatient | EUPatient]]:
        return self._patients

    def _check_format(self, ids: Optional[list[str]]) -> None:
        """Validate requested patient IDs before searching the filesystem."""
        if ids is None:
            return
        if not all(isinstance(p, str) for p in ids):
            raise TypeError("All IDs in patients_wanted must be strings!")
        for p in ids:
            if p not in self._PATIENTS_ALL:
                raise TypeError(f"{p} is not a valid patient ID!")

    @staticmethod
    def _validate_output_dtype(dtype: str | np.dtype) -> np.dtype:
        """Restrict saved EEG arrays to supported floating point dtypes."""
        output_dtype = np.dtype(dtype)
        if output_dtype not in {np.dtype("float32"), np.dtype("float64")}:
            raise TypeError("output_dtype must be float32 or float64")
        return output_dtype

    @abstractmethod
    def _initialize_patients(self) -> dict[str, CHBPatient | EUPatient]:
        pass

    def process_patients(self, max_workers: Optional[int] = None) -> None:
        """Extract all selected patients into the configured output directory.

        Parameters
        ----------
        max_workers:
            Maximum number of patient-level worker threads. A value of `None`
            uses `ThreadPoolExecutor`'s default. Use `1` for deterministic
            single-patient progress output.
        """
        patients = self._initialize_patients()
        with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(patient.process_patient, self.output_path, i, self.output_dtype): pid
                for i, (pid, patient) in enumerate(patients.items())
            }
            for future in cf.as_completed(futures):
                future.result()
        self._patients = patients
        print(f"The following patients have been processed: {self._patient_ids}")
