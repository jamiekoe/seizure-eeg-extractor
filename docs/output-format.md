# Output Format

The extractor writes one directory per patient and one directory per source
recording.
Re-running extraction for a patient replaces that patient's output directory
before writing new `record_<n>` folders.

```text
extracted_data/
  pat_FR_253/
    record_0/
      eeg.npy
      info.pkl
    record_1/
      eeg.npy
      info.pkl
```

## `eeg.npy`

`eeg.npy` is a two-dimensional NumPy array:

```text
(num_samples, num_channels)
```

Rows are samples over time. Columns are channels in the order reported by the
source EDF file or EU header. The default dtype is `float32`.

## `info.pkl`

`info.pkl` is a pickled dictionary. It is intentionally simple so existing code
can load it without importing this package:

```python
import pickle
from pathlib import Path

with Path("record_0/info.pkl").open("rb") as file:
    info = pickle.load(file)
```

Schema:

| Key | Type | Description |
| --- | --- | --- |
| `pid` | `str` | Patient ID. |
| `fid` | `str` | Source file ID without extension. |
| `fs` | `float` | Sampling frequency in Hz. |
| `num_samples` | `int` | Number of rows in `eeg.npy`. |
| `num_channels` | `int` | Number of columns in `eeg.npy`. |
| `channel_names` | `list[str]` or `None` | Channel names from the source metadata. |
| `eeg_dtype` | `str` | Saved dtype for `eeg.npy`. |
| `num_seizures` | `int` | Number of seizure intervals assigned to this record. |
| `seizure_times` | `list[dict]` or `None` | Seizure interval metadata. |
| `file_start_time` | `dict` or `None` | Source file start timestamp. |
| `file_end_time` | `dict` or `None` | Source file end timestamp. |
| `duration_in_sec` | `int` or `None` | Duration from source metadata. |

## Seizure Interval Schema

Each item in `seizure_times` contains:

| Key | Type | Description |
| --- | --- | --- |
| `onset_index` | `int` | Inclusive record-local start sample. |
| `offset_index` | `int` | Exclusive record-local end sample. |
| `onset_second` | `int` or `None` | Record-local start second when available. |
| `offset_second` | `int` or `None` | Record-local end second when available. |
| `onset_timestamp` | `dict` or `None` | Absolute seizure onset timestamp when available. |
| `offset_timestamp` | `dict` or `None` | Absolute seizure offset timestamp when available. |

Intervals use Python slicing conventions: `eeg[onset_index:offset_index]`.

For EU records, crossing seizures are clipped to the file interval. This means
the same real-world seizure can appear in adjacent records when it spans a file
boundary; each record contains only the overlapping segment.

## Loading A Record

```python
import pickle
import numpy as np
from pathlib import Path

record_dir = Path("extracted_data/pat_FR_253/record_0")
eeg = np.load(record_dir / "eeg.npy", mmap_mode="r")
with (record_dir / "info.pkl").open("rb") as file:
    info = pickle.load(file)

assert eeg.shape == (info["num_samples"], info["num_channels"])
```
