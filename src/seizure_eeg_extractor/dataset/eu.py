from pathlib import Path
from typing import Optional

from .dataset import Dataset
from ..patient import CHBPatient, EUPatient


class EU(Dataset):
    """Extractor for EU Epilepsy / EPILEPSIAE-style binary folders.

    The class discovers `pat_FR_*` folders and delegates seizure list parsing,
    header parsing, binary sample loading, and seizure-to-file assignment to
    `EUPatient` and `Binary`.
    """

    _PATIENTS_ALL = [f"pat_FR_{id_}" for id_ in [
        115, 139, 253, 264, 273, 375, 384, 442, 548, 565, 583, 590, 620, 635,
        818, 862, 916, 922, 958, 970, 1073, 1077, 1084, 1096, 1125, 1146, 1150]]

    def __init__(self, input_path: str | Path, output_path: Optional[str | Path] = None,
                 patients_wanted: Optional[list[str]] = None,
                 output_dtype: str = "float32") -> None:
        super().__init__(input_path, output_path, patients_wanted, key="pat_FR_", output_dtype=output_dtype)

    def _initialize_patients(self) -> dict[str, CHBPatient | EUPatient]:
        """Create EU patient objects for the selected local patient folders."""
        return {pid: EUPatient(pid, self.input_path) for pid in self.patient_ids}
