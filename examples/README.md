# Examples

The examples assume that you have already extracted at least one record with
`eeg-extract`.

## Plot One Channel

```bash
python examples/plot_record.py ./extracted_data/chb17/record_0 \
  --channel 0 \
  --samples 1000
```

The script loads `eeg.npy` and `info.pkl`, checks that the requested channel is
valid, and plots the requested number of samples with Matplotlib.

For large records, use NumPy memory mapping in your own analysis scripts:

```python
import numpy as np

eeg = np.load("extracted_data/pat_FR_253/record_0/eeg.npy", mmap_mode="r")
window = eeg[:10_000, 0]
```
