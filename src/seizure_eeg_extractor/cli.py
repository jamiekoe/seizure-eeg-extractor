"""Command line interface for extracting supported EEG datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .dataset import CHBMIT, EU


DATASETS = {
    "chbmit": CHBMIT,
    "eu": EU,
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _parse_patient_ids(values: list[str] | None) -> list[str] | None:
    """Normalize space- or comma-separated patient IDs from argparse."""
    if values is None:
        return None
    patient_ids: list[str] = []
    for value in values:
        patient_ids.extend(part.strip() for part in value.split(",") if part.strip())
    return patient_ids


def _output_dtype(value: str) -> str:
    if value not in {"float32", "float64"}:
        raise argparse.ArgumentTypeError("must be one of: float32, float64")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the `eeg-extract` command parser."""
    parser = argparse.ArgumentParser(
        description="Extract EEG signals and metadata from CHB-MIT or EU Epilepsy datasets.",
    )
    parser.add_argument(
        "dataset",
        choices=sorted(DATASETS),
        help="Dataset format to process.",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default=None,
        help="Directory for extracted records. Defaults to <input_path>/extracted_data.",
    )
    parser.add_argument(
        "-p",
        "--patients",
        nargs="+",
        default=None,
        metavar="PATIENT_ID",
        help="Optional patient IDs to process, separated by spaces or commas.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=_positive_int,
        default=None,
        help="Maximum number of patient-processing threads.",
    )
    parser.add_argument(
        "--dtype",
        type=_output_dtype,
        default="float32",
        help="Floating-point dtype for eeg.npy output. Defaults to float32.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line extractor and return a process exit code."""
    args = build_parser().parse_args(argv)
    dataset_class = DATASETS[args.dataset]
    dataset = dataset_class(
        input_path=args.input_path,
        output_path=args.output_path,
        patients_wanted=_parse_patient_ids(args.patients),
        output_dtype=args.dtype,
    )
    dataset.process_patients(max_workers=args.workers)
    print(f"Extracted records written to {dataset.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
