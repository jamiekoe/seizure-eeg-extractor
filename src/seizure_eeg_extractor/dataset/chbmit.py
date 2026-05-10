import warnings
from pathlib import Path
from typing import Optional

from .dataset import Dataset
from ..patient import CHBPatient, EUPatient


class CHBMIT(Dataset):
    """Extractor for the CHB-MIT Scalp EEG Database.

    The class expects the standard CHB-MIT patient folders (`chb01` through
    `chb24`) and per-patient summary files. EDF reading and seizure interval
    extraction are delegated to `CHBPatient` and `EDF`.
    """

    _PATIENTS_ALL = [f"chb{p:02}" for p in range(1, 24 + 1)]

    def __init__(self, input_path: str | Path, output_path: Optional[str | Path] = None,
                 patients_wanted: Optional[list[str]] = None,
                 output_dtype: str = "float32") -> None:
        super().__init__(input_path, output_path, patients_wanted, key="chb", output_dtype=output_dtype)

    # Age & sex info is being extracted but currently not saved or used
    def _initialize_patients(self) -> dict[str, CHBPatient | EUPatient]:
        """Create CHB patient objects and attach optional age/sex metadata."""
        patients: dict[str, CHBPatient | EUPatient] = {
            pid: CHBPatient(pid, self.input_path) for pid in self.patient_ids
        }
        try:
            f = (self.input_path / "SUBJECT-INFO").open("r", encoding="utf-8")
        except OSError as err:
            warnings.warn("Could not open SUBJECT-INFO file -> age & gender info will not be added.")
            warnings.warn(str(err))
            return patients
        with f:
            for line in f:
                words = line.split()
                if words and words[0] in self.patient_ids:
                    pid, age, gender = words[0], float(words[2]), words[1]
                    patients[pid].age = age
                    patients[pid].gender = gender
        return patients
