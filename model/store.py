"""Array storage that many worker processes can share without copying.

A dataset is a directory of .npy files, one per array. load() memory-maps them,
so 30 workers reading a 2 GB pool share one copy in the OS page cache instead of
each holding its own (np.load of an .npz always makes a private copy).
"""

from pathlib import Path

import numpy as np


def save(path, **arrays):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        np.save(path / f"{k}.npy", np.asarray(v))


def load(path):
    """-> {name: read-only memory-mapped array}"""
    path = Path(path)
    out = {p.stem: np.load(p, mmap_mode="r") for p in sorted(path.glob("*.npy"))}
    if not out:
        raise FileNotFoundError(f"no arrays in {path}")
    return out


def exists(path):
    return Path(path).is_dir() and any(Path(path).glob("*.npy"))


def create(path, name, shape, dtype):
    """Preallocate a .npy file on disk; workers then fill slices of it via open_rw()."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path / f"{name}.npy", mode="w+", shape=shape, dtype=dtype)


def open_rw(path, name):
    return np.load(Path(path) / f"{name}.npy", mmap_mode="r+")
