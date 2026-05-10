# Processing Methods

The extractor normalizes source recordings into one `eeg.npy` array and one
`info.pkl` metadata file per recording. The raw signal values are copied into a
floating point NumPy array, and seizure annotations are stored as record-local
sample intervals.

## CHB-MIT

CHB-MIT records are EDF files accompanied by per-patient summary text files.
Processing uses this sequence:

1. Find patient folders named `chb01` through `chb24`.
2. Read `<patient>-summary.txt`.
3. Extract the sampling frequency from summary lines containing
   `sampling rate`. If it is absent, the extractor falls back to `256.0 Hz`,
   which is the expected value for this dataset.
4. List `.edf` files in the patient folder.
5. For each EDF file, find the matching summary section and extract:
   file start time, file end time, number of seizures, and seizure start/end
   seconds.
6. Read the EDF signal labels and channel data with `pyEDFlib`.
7. Convert seizure seconds to sample indices with `int(seconds * fs)`.
8. Write the extracted record directory.

CHB-MIT seizure intervals are already described relative to the EDF file, so no
cross-file timestamp matching is needed.

## EU Binary Dataset

EU records are stored as paired `.head` and `.data` files, plus patient-level
`seizurelist.txt` files. Processing uses this sequence:

1. Find patient folders named like `pat_FR_253`.
2. Locate the first `seizurelist.txt` inside the patient tree.
3. Parse seizure onset and offset dates, times, and sample indices from the
   seizure list.
4. List all `.data` files recursively and process the matching `.head` file for
   each one.
5. Extract header parameters:
   `sample_freq`, `num_samples`, `duration_in_sec`, `num_channels`,
   `conversion_factor`, `sample_bytes`, optional `elec_names`, and `start_ts`.
6. Read the binary `.data` file with NumPy as `int16` when `sample_bytes=2` or
   `int32` when `sample_bytes=4`.
7. Reshape the raw vector to `(num_samples, num_channels)` and multiply by the
   header `conversion_factor`.
8. Compute the absolute time interval covered by the file.
9. Assign every seizure whose absolute onset/offset interval overlaps the file
   interval.
10. Clip crossing seizures to the part that is inside the current file.
11. Write the extracted record directory.

EU seizure intervals in `info.pkl` are record-local. For example, if a seizure
starts ten seconds before a file and ends ten seconds after file start, the
record gets an interval from sample `0` to `10 * fs`.

One historical special case is preserved: the twelfth seizure for `pat_FR_548`
is excluded because it is more than eight hours long and does not behave like
the other annotated seizure events.

## Parallelism

Parallelism is at the patient level. `Dataset.process_patients(max_workers=N)`
submits one worker per selected patient. Inside a patient, files are processed
sequentially so record numbering is deterministic.

## Numeric Precision

The extractor loads raw samples into floating point arrays and saves them as
`float32` by default. Use `--dtype float64` or `output_dtype="float64"` when
double precision is required. The selected dtype is recorded in `info.pkl` as
`eeg_dtype`.
