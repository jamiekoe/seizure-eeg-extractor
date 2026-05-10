import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from seizure_eeg_extractor import CHBMIT, EU


def _load_first_record(output_path: Path, patient_id: str) -> tuple[np.ndarray, dict]:
    patient_output_path = output_path / patient_id
    record_dirs = sorted(patient_output_path.glob("record_*"))
    assert record_dirs, f"No extracted records found for {patient_id}"

    record_dir = record_dirs[0]
    eeg_path = record_dir / "eeg.npy"
    info_path = record_dir / "info.pkl"
    assert eeg_path.exists()
    assert info_path.exists()

    eeg = np.load(eeg_path, mmap_mode="r")
    with info_path.open("rb") as file:
        info = pickle.load(file)
    return eeg, info


@pytest.mark.integration
def test_chbmit_real_dataset_smoke(tmp_path: Path) -> None:
    input_path = os.environ.get("CHBMIT_PATH")
    if input_path is None:
        pytest.skip("Set CHBMIT_PATH to run the CHB-MIT integration test.")

    patient_id = os.environ.get("CHBMIT_PATIENT", "chb01")
    output_path = tmp_path / "chbmit_extracted"

    dataset = CHBMIT(input_path=input_path, output_path=output_path, patients_wanted=[patient_id])
    dataset.process_patients(max_workers=1)

    eeg, info = _load_first_record(output_path, patient_id)
    assert eeg.ndim == 2
    assert eeg.shape[0] == info["num_samples"]
    assert eeg.shape[1] == info["num_channels"]
    assert info["pid"] == patient_id
    assert info["fs"] > 0


@pytest.mark.integration
def test_eu_real_dataset_smoke(tmp_path: Path) -> None:
    input_path = os.environ.get("EU_PATH")
    if input_path is None:
        pytest.skip("Set EU_PATH to run the EU Epilepsy integration test.")

    patient_id = os.environ.get("EU_PATIENT", "pat_FR_1125")
    output_path = tmp_path / "eu_extracted"

    dataset = EU(input_path=input_path, output_path=output_path, patients_wanted=[patient_id])
    dataset.process_patients(max_workers=1)

    eeg, info = _load_first_record(output_path, patient_id)
    assert eeg.ndim == 2
    assert eeg.shape[0] == info["num_samples"]
    assert eeg.shape[1] == info["num_channels"]
    assert info["pid"] == patient_id
    assert info["fs"] > 0
