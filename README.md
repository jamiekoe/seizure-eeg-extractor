# Seizure EEG Extractor

Extract EEG signal arrays and seizure metadata from seizure EEG datasets into a
simple NumPy-based record format.

Supported source datasets:

- CHB-MIT Scalp EEG Database
- EU Epilepsy / EPILEPSIAE-style binary dataset folders

The raw datasets are not included in this repository. Download and use them
according to their own access, citation, privacy, and data-use terms.

## What This Package Does

`seizure-eeg-extractor` converts each raw recording file into:

- `eeg.npy`: a NumPy array with shape `(num_samples, num_channels)`
- `info.pkl`: a Python dictionary with patient ID, file ID, sampling frequency,
  channel metadata, seizure intervals, timestamps, duration, and output dtype

The package keeps one output folder per patient and one `record_<n>` folder per
source recording. It does not train a seizure detector or create labels beyond
the seizure intervals already provided by the source dataset metadata.
When a patient is reprocessed, that patient's previous output directory is
cleared first so stale records from an earlier run cannot be mixed with the new
extraction.

## Installation

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install seizure-eeg-extractor
```

For plotting examples:

```bash
python -m pip install "seizure-eeg-extractor[examples]"
```

## Command Line Usage

Process all available CHB-MIT patients:

```bash
eeg-extract chbmit /path/to/chbmit/1.0.0
```

Process selected CHB-MIT patients:

```bash
eeg-extract chbmit /path/to/chbmit/1.0.0 \
  --patients chb17 chb20 \
  --output-path ./extracted_data
```

Process selected EU patients:

```bash
eeg-extract eu /path/to/Epilepsiae \
  --patients pat_FR_253 pat_FR_384 \
  --output-path ./extracted_data \
  --dtype float32
```

Options:

- `--patients`, `-p`: patient IDs separated by spaces or commas.
- `--output-path`, `-o`: output directory. Defaults to
  `<input_path>/extracted_data`.
- `--workers`, `-w`: number of patient-processing threads.
- `--dtype`: `float32` or `float64` for `eeg.npy`. The default is `float32`,
  which is normally the practical choice for large datasets.

## Python Usage

```python
from seizure_eeg_extractor import CHBMIT, EU

dataset = CHBMIT(
    input_path="/path/to/chbmit/1.0.0",
    output_path="./extracted_chbmit",
    patients_wanted=["chb17", "chb20"],
    output_dtype="float32",
)
dataset.process_patients(max_workers=2)

eu = EU(
    input_path="/path/to/Epilepsiae",
    output_path="./extracted_eu",
    patients_wanted=["pat_FR_253"],
)
eu.process_patients(max_workers=1)
```

## Expected Dataset Layout

CHB-MIT input should contain patient folders named `chb01` through `chb24`.
Each patient folder should contain `.edf` files and the corresponding
`<patient>-summary.txt`; the dataset root may also contain `SUBJECT-INFO`.

EU input should contain patient folders named like `pat_FR_253`. The extractor
looks recursively inside each patient folder for:

- `.head` files containing recording metadata
- `.data` files containing binary EEG samples
- `seizurelist.txt` containing seizure onset and offset timestamps

## Output Layout

```text
extracted_data/
  <patient_id>/
    record_0/
      eeg.npy
      info.pkl
    record_1/
      eeg.npy
      info.pkl
```

`eeg.npy` is saved as `(samples, channels)`. `info.pkl` contains:

| Key | Description |
| --- | --- |
| `pid` | Patient ID, for example `chb01` or `pat_FR_253`. |
| `fid` | Source recording ID without extension. |
| `fs` | Sampling frequency in Hz. |
| `num_samples` | Number of rows in `eeg.npy`. |
| `num_channels` | Number of EEG channels. |
| `channel_names` | Channel names when present in the source file/header. |
| `eeg_dtype` | Saved NumPy dtype, usually `float32`. |
| `num_seizures` | Number of seizure intervals overlapping the record. |
| `seizure_times` | List of seizure dictionaries with sample indices and, when available, timestamps. |
| `file_start_time` | Source file start timestamp when available. |
| `file_end_time` | Source file end timestamp when available. |
| `duration_in_sec` | Recording duration when available. |

Seizure sample indices are record-local. For EU records, seizures that cross a
file boundary are clipped to the part that overlaps the current recording.

## Processing Method

CHB-MIT processing reads each patient's summary text file, extracts per-file
start/end times and seizure start/end seconds, then reads EEG signals from EDF
files with `pyEDFlib`.

EU processing reads each patient's `seizurelist.txt`, parses absolute seizure
timestamps, reads each `.head` file for sample frequency, channel count, sample
count, start timestamp, binary sample width, and conversion factor, then loads
the matching `.data` file with NumPy. Seizures are assigned to every recording
whose time interval overlaps the seizure interval.

More details are in the
[processing method notes](https://github.com/jamiekoe/seizure-eeg-extractor/blob/main/docs/processing-methods.md)
and
[output format notes](https://github.com/jamiekoe/seizure-eeg-extractor/blob/main/docs/output-format.md).

## Examples

Plot one channel from an extracted record:

```bash
python examples/plot_record.py ./extracted_data/chb17/record_0 \
  --channel 0 \
  --samples 1000
```

See the
[examples README](https://github.com/jamiekoe/seizure-eeg-extractor/blob/main/examples/README.md).

## Citation

If you use this software in academic work, please cite the repository using the
metadata in
[CITATION.cff](https://github.com/jamiekoe/seizure-eeg-extractor/blob/main/CITATION.cff).

## License

This project is distributed under the
[MIT License](https://github.com/jamiekoe/seizure-eeg-extractor/blob/main/LICENSE).
