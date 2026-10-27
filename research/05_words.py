# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: tt-wakeword (.venv)
#     language: python
#     name: tt-wakeword
# ---

# %% [markdown]
# # 05 · Which word?
#
# The same detector (architecture and size chosen in notebook 04) trained on each candidate
# word's simulations, then ranked on the **simulated holdout**: held-out voices, held-out whole
# voice models, held-out text, rooms and ambience. Real-recording columns are a secondary
# check of the simulator.
#
# **What makes a good wake word here:**
# * **low `frr_syn@1.0`**: rarely missed, at 1 false alarm/h on held-out simulated conversation;
# * **low `fa_sim@frr_syn0.05`**: few false alarms reach the host at a 5% miss rate (the
#   first-stage view, research/SOA.md §3.1);
# * **usable by people**: a word you'd actually say to a device, and that isn't common in
#   conversation (`group`: control words vs rare long/short words, see model/wakewords.yaml).
#
# Produced by `ARCH=<arch> SIZE=<i> data/word_sweep.sh` into `data/results/words_<arch>_<i>/`.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from nbutil import RES, load_runs

subs = sorted(p.name for p in RES.glob("words_*"))
print("sweeps:", subs)
sub = subs[-1] if subs else None
runs = load_runs(sub) if sub else []
cols = ["slug", "group", "arch", "params", "tiles", "frr_syn@1.0", "frr_syn@5.0", "fa_sim@frr_syn0.05",
        "frr_real@1.0", "fa_real_ami@1.0", "fa_real@frr_real0.05"]
df = pd.DataFrame([{c: r.get(c) for c in cols} for r in runs])
if len(df):
    df = df.sort_values(["frr_syn@1.0", "fa_sim@frr_syn0.05"]).reset_index(drop=True)
    with pd.option_context("display.max_rows", 100, "display.width", 220, "display.precision", 3):
        display(df)
else:
    print("no word sweep results yet")

# %%
if len(df):
    fig, ax = plt.subplots(figsize=(12, max(4, len(df) * 0.18)))
    colors = {g: c for g, c in zip(sorted(df["group"].unique()), plt.cm.tab10.colors)}
    ax.barh(df["slug"], df["frr_syn@1.0"], color=[colors[g] for g in df["group"]])
    ax.invert_yaxis()
    ax.set(xlabel="missed held-out TTS keywords at 1 sim FA/h (lower is better)", xlim=(0, 1))
    for g, c in colors.items():
        ax.barh([], [], color=c, label=g)
    ax.legend()
    plt.tight_layout()

# %% [markdown]
# ## Does the simulated ranking agree with real recordings?
#
# Each dot is one word: simulated miss rate (x) against real-speaker miss rate (y) at the same
# threshold. A rising trend means the simulator ranks words the way reality does, so picking
# on simulation is trustworthy.

# %%
if len(df):
    d = df.dropna(subset=["frr_real@1.0"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(d["frr_syn@1.0"], d["frr_real@1.0"])
    for _, r in d.iterrows():
        ax.annotate(r["slug"], (r["frr_syn@1.0"], r["frr_real@1.0"]), fontsize=7)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="FRR, simulated holdout", ylabel="FRR, real speakers", xlim=(0, 1.02), ylim=(0, 1.02))
    if len(d) > 2:
        print(f"rank correlation sim vs real: {d['frr_syn@1.0'].rank().corr(d['frr_real@1.0'].rank()):.2f}")
