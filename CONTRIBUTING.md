# Contributing

## Development Setup

Install the package in editable mode with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the lightweight test suite before opening a pull request:

```bash
pytest
```

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
