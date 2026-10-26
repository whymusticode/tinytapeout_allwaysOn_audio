"""Shared helpers for the notebooks: paths, result loading, consistent colours."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "data" / "results"
sys.path.insert(0, str(ROOT / "model"))

ARCH_ORDER = ["gru", "mingru", "tcn", "transformer"]
ARCH_COLOR = {"gru": "#1f77b4", "mingru": "#2ca02c", "tcn": "#d62728", "transformer": "#9467bd"}
SIZE_ALPHA = [0.35, 0.5, 0.7, 0.85, 1.0]
TILE_BUDGET = 16

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})


def load_runs(sub="arch_v1"):
    """All finished runs in data/results/<sub>/ -> list of dicts (each with its curve)."""
    out = []
    for p in sorted((RES / sub).glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            pass
    return out


def table(runs):
    """One row per run, the columns that matter, sorted by architecture and size."""
    cols = ["slug", "arch", "size_index", "width", "params", "tiles", "classifier_tiles", "qspi_load",
            "best_step", "stopped_step", "train_time_s", "frr_syn@1.0", "frr_real@1.0", "fa_real_ami@1.0",
            "frr_syn@real1", "frr_real@real1"]
    df = pd.DataFrame([{c: r.get(c) for c in cols} for r in runs])
    if df.empty:
        return df
    df["arch"] = pd.Categorical(df["arch"], ARCH_ORDER, ordered=True)
    return df.sort_values(["slug", "arch", "size_index"]).reset_index(drop=True)


def progress(sub="arch_v1"):
    """Latest validation line of every running job (from the logs)."""
    for log in sorted((RES / sub).glob("*.log")):
        lines = [ln for ln in log.read_text().splitlines() if " step " in ln and "val_loss" in ln]
        print(f"{log.stem:24s} {lines[-1].strip() if lines else '(starting)'}")
