from datetime import date, time
from pathlib import Path
import pickle
import warnings

import numpy as np
import pytest

from seizure_eeg_extractor.dataset.dataset import Dataset
from seizure_eeg_extractor.file.file import File
from seizure_eeg_extractor.file import Binary, EDF, SeizureInfo, Timestamp
from seizure_eeg_extractor.patient import CHBPatient, EUPatient
from seizure_eeg_extractor.patient.patient import Patient


def test_chb_sampling_frequency_is_extracted_from_summary() -> None:
    patient = CHBPatient("chb01", Path("/tmp"))

    assert patient._extract_fs(["Data Sampling Rate: 256 Hz\n"]) == 256.0


def test_chb_file_timestamp_wraps_hours_after_midnight() -> None:
    timestamp = EDF._extract_file_timestamp("End Time", "File End Time: 26:02:15")

    assert timestamp.time == time(2, 2, 15)


def test_eu_seizure_timestamp_allows_integer_seconds() -> None:
    timestamp = EUPatient._extract_seizure_timestamp("2009-06-18", "11:54:01")

    assert timestamp.date == date(2009, 6, 18)
    assert timestamp.time == time(11, 54, 1)


def test_eu_seizure_timestamp_pads_fractional_seconds() -> None:
    timestamp = EUPatient._extract_seizure_timestamp("2009-06-18", "11:54:01.5")

    assert timestamp.time == time(11, 54, 1, 500000)


def test_dataset_patient_subset_does_not_warn_about_unrequested_folders(tmp_path: Path) -> None:
    (tmp_path / "chb01").mkdir()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        selected = Dataset._get_patient_ids(tmp_path, ["chb01"], ["chb01", "chb02"], "chb")

    assert selected == ["chb01"]
    assert not any("Only found" in str(warning.message) for warning in caught)


def test_binary_header_parameter_extraction() -> None:
    lines = [
        "sample_freq=512.0",
        "num_channels=16",
    ]

    assert Binary._extract_parameter("sample_freq", lines, float) == 512.0
    assert Binary._extract_parameter("num_channels", lines, int) == 16


def test_binary_header_parameter_requires_exact_key() -> None:
    lines = [
        "not_sample_freq=512.0",
        "sample_freq = 256.0",
    ]

    assert Binary._extract_parameter("sample_freq", lines, float) == 256.0


def test_binary_channel_names_are_extracted_from_header() -> None:
    lines = ["elec_names=[F3, F4, C3, C4]"]

    assert Binary._extract_channels(lines) == ["F3", "F4", "C3", "C4"]


def test_binary_file_timestamp_pads_fractional_seconds() -> None:
    timestamp = Binary._extract_file_timestamp(["start_ts=2009-06-18 11:54:01.5"])

    assert timestamp.date == date(2009, 6, 18)
    assert timestamp.time == time(11, 54, 1, 500000)


def test_save_record_data_casts_output_dtype(tmp_path: Path) -> None:
    file = File("pid", "fid", fs=256.0)
    file.num_samples = 2
    file.num_channels = 2
    file.seizure_times = []
    file.eeg = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    output_path = tmp_path / "new_record"

    file.save_record_data(output_path, output_dtype="float32")

    eeg = np.load(output_path / "eeg.npy")
    with (output_path / "info.pkl").open("rb") as handle:
        info = pickle.load(handle)
    assert eeg.dtype == np.float32
    assert info["eeg_dtype"] == "float32"


def test_clean_directory_removes_stale_outputs(tmp_path: Path) -> None:
    output_path = tmp_path / "patient"
    output_path.mkdir()
    (output_path / "stale.txt").write_text("old", encoding="utf-8")

    cleaned = Patient.create_clean_directory(output_path)

    assert cleaned == output_path
    assert output_path.exists()
    assert not (output_path / "stale.txt").exists()


def test_edf_helpers_raise_value_error_for_missing_keys() -> None:
    with pytest.raises(ValueError, match="Start Time"):
        EDF._extract_file_timestamp("Start Time", "File Name: chb01_01.edf")


def test_chb_process_rejects_edf_missing_from_summary(tmp_path: Path) -> None:
    patient_path = tmp_path / "chb01"
    patient_path.mkdir()
    patient = CHBPatient("chb01", patient_path)
    patient.fs = 256.0
    patient.filenames = ["chb01_01.edf"]

    with pytest.raises(ValueError, match="summary metadata"):
        patient._process_files([], tmp_path / "output", pos=0, output_dtype="float32")


def test_binary_rejects_unexpected_sample_count(tmp_path: Path) -> None:
    basename = tmp_path / "record"
    np.array([1, 2, 3], dtype=np.int16).tofile(basename.with_suffix(".data"))
    binary = Binary("pid", "record", tmp_path)
    binary.num_samples = 2
    binary.num_channels = 2

    with pytest.raises(ValueError, match="Expected 4 values"):
        binary._extract_eeg(basename, sample_bytes=2, conversion_factor=1.0)


def test_eu_seizures_are_clipped_to_file_boundaries() -> None:
    binary = Binary("pat_FR_375", "fid", Path("/tmp"))
    binary.fs = 256.0
    binary.num_samples = 256 * 60
    binary.duration_in_sec = 60
    binary.file_start_timestamp = Timestamp(date(2005, 1, 1), time(12, 0, 0))
    seizures = [
        SeizureInfo(
            onset_index=0,
            offset_index=0,
            onset_timestamp=Timestamp(date(2005, 1, 1), time(11, 59, 50)),
            offset_timestamp=Timestamp(date(2005, 1, 1), time(12, 0, 10)),
        ),
        SeizureInfo(
            onset_index=0,
            offset_index=0,
            onset_timestamp=Timestamp(date(2005, 1, 1), time(12, 0, 10)),
            offset_timestamp=Timestamp(date(2005, 1, 1), time(12, 1, 10)),
        ),
    ]

    clipped = binary._assign_seizure_info(seizures)

    assert [(s.onset_index, s.offset_index) for s in clipped] == [
        (0, 10 * 256),
        (10 * 256, 60 * 256),
    ]
