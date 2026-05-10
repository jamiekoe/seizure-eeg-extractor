"""Example script for plotting one channel from an extracted record."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    """Build the example command parser."""
    parser = argparse.ArgumentParser(description="Plot one channel from an extracted EEG record.")
    parser.add_argument(
        "record_dir",
        type=Path,
        help="Path to a record directory containing eeg.npy and info.pkl.",
    )
    parser.add_argument(
        "-c",
        "--channel",
        type=int,
        default=0,
        help="Zero-based channel index to plot.",
    )
    parser.add_argument(
        "-n",
        "--samples",
        type=int,
        default=1000,
        help="Number of samples to plot.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load a record directory and plot the requested channel."""
    args = build_parser().parse_args(argv)
    eeg_path = args.record_dir / "eeg.npy"
    info_path = args.record_dir / "info.pkl"

    data = np.load(eeg_path)
    with info_path.open("rb") as file:
        info = pickle.load(file)

    if args.channel < 0 or args.channel >= data.shape[1]:
        raise ValueError(f"Channel index {args.channel} is outside the valid range 0-{data.shape[1] - 1}")

    channel_names = info.get("channel_names") or []
    channel_label = channel_names[args.channel] if args.channel < len(channel_names) else f"channel {args.channel}"

    plt.plot(data[:args.samples, args.channel])
    plt.title(channel_label)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
