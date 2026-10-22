"""TinyShakespeare with a byte-level tokenizer (vocab = 256; a byte is its own token id)."""
import os
import urllib.request

import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def encode(text):
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)


def decode(ids):
    return bytes(int(i) for i in ids).decode("utf-8", errors="replace")


def load_tinyshakespeare(val_fraction=0.1):
    path = os.path.join(DATA_DIR, "tinyshakespeare.txt")
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    with open(path, encoding="utf-8") as f:
        ids = encode(f.read())
    n_val = int(len(ids) * val_fraction)
    return ids[:-n_val], ids[-n_val:]


def get_batch(data, batch_size, seq_len, rng):
    starts = rng.integers(0, len(data) - seq_len - 1, size=batch_size)
    x = np.stack([data[s:s + seq_len] for s in starts])
    y = np.stack([data[s + 1:s + seq_len + 1] for s in starts])
    return x, y
