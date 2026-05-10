# Contributing

## Development Setup

Install the package in editable mode with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Build and metadata checks use the standard Python packaging tools:

```bash
python -m pip install build twine
python -m build --sdist --wheel
twine check dist/*
```

## Testing

Run the lightweight test suite before opening a pull request:

```bash
pytest
```

Integration tests require local raw dataset copies and are skipped by default:

```bash
CHBMIT_PATH=/path/to/chbmit/1.0.0 CHBMIT_PATIENT=chb01 pytest -m integration
EU_PATH=/path/to/Epilepsiae EU_PATIENT=pat_FR_253 pytest -m integration
```

The integration tests write extracted records to pytest-managed temporary
directories, not into the raw dataset folders.

## Data and Generated Files

Raw EEG files and extracted outputs are intentionally ignored by Git:

- `*.edf`
- `*.data`
- `*.head`
- `*.npy`
- `*.pkl`
- `data/`, `datasets/`, and `extracted_data/`

If you work with local data inside the repository tree, inspect
`git status --ignored` before committing changes.
