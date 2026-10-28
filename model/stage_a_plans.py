"""Write the Stage A step configs (research/stage_a/<step>.jsonl) for `stage_a.py add`.

  python model/stage_a_plans.py a2 --bands 12 --order 4     # after A1 picks the front end

A2: each family at two sizes, a sensible first config plus seeded random draws of its knobs from
search_space.FAMILIES (random search: with ~6 trials per family the goal is ranking families,
not tuning them). Pruned at step 3000 within the size group.

Teacher: whisper_tiny_L4 (bidirectional over the window). A0 found a 50k TCN copies it no better
than speech_embedding (word_acc 0.515 vs 0.523); A2 asks whether another family can. Control: each
family's first config at 100k also taught by speech_embedding (group "A2 control", not pruned).
"""

import argparse
import json
import random
import zlib
from pathlib import Path

import search_space as SS
import students as S

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research/stage_a"

# First trial per family: the settings we'd pick by hand (06_probes student for tcn).
FIRST = {
    "tcn": {},
    "rnn": {"cell": "gru", "layers": 2, "residual": True},
    "ssm": {"kind": "mingru", "layers": 2},
    "transformer": {"layers": 2, "heads": 2, "window": 16, "ffn_mult": 2, "stem": "conv3"},
    "crnn": {"conv_layers": 2, "conv_kernel": 3, "back": "gru", "back_layers": 1},
    "dnn": {"context": 24, "depth": 3, "act": "relu"},
    "bcresnet": {"stages": 3, "blocks_per_stage": 2, "freq_stride": 2, "subspectral_groups": 2},
}
# knobs students.py does not implement yet: drawn values are mapped to implemented ones
UNIMPLEMENTED = {"ssm.kind": ("s4d", "mamba_lite"), "transformer.pos": ("alibi", "rope")}
SKIP_KNOBS = {"rnn": ("proj",)}


class _Draw:
    def __init__(self, seed):
        self.r = random.Random(seed)

    def suggest_categorical(self, n, c):
        return self.r.choice(c)

    def suggest_int(self, n, a, b, step=1):
        return self.r.randrange(a, b + 1, step)

    def suggest_float(self, n, a, b, log=False):
        return self.r.uniform(a, b)


def draw_knobs(family, seed):
    d = _Draw(seed)
    k = {}
    for name, spec in SS.FAMILIES[family].items():
        if name in SKIP_KNOBS.get(family, ()):
            continue
        v = SS._suggest(d, name, spec)
        if v in UNIMPLEMENTED.get(f"{family}.{name}", ()):
            v = {"ssm.kind": "lru", "transformer.pos": "rel_bias"}[f"{family}.{name}"]
        k[name] = v
    return k


MIN_RF_STEPS = 25  # teacher window 0.775 s at 32 ms steps (search_space: receptive field >= 0.8 s)


def rf_ok(family, knobs, bands, size):
    m, _ = S.build(family, knobs, bands, 2, 96, size)
    return m.body.rf_steps >= MIN_RF_STEPS


def a2(bands, order, per_family=6, sizes=(25_000, 100_000), teacher="whisper_tiny_L4"):
    fams = list(SS.STAGE_A_FAMILIES) + ["dnn"]
    rows = []
    for size in sizes:
        for f in fams:
            n = 1 if f == "dnn" else per_family
            for i in range(n):
                knobs, tries = FIRST[f], 0
                while i > 0 or not rf_ok(f, knobs, bands, size):  # redraw students that can't see 0.8 s
                    knobs = draw_knobs(f, zlib.crc32(f"{f}{size}{i}{tries}".encode()))
                    tries += 1
                    if rf_ok(f, knobs, bands, size):
                        break
                rows.append({"name": f"A2 {f} {size // 1000}k #{i}", "group": f"A2 {size // 1000}k", "size": size,
                             "family": f, "knobs": knobs, "bands": bands, "order": order, "hop_ms": 16,
                             "teacher": teacher})
    for r in [r for r in rows if r["size"] == sizes[-1] and r["name"].endswith("#0")]:
        rows.append({**r, "name": r["name"] + " speech_embedding", "group": "A2 control", "teacher": "speech_embedding",
                     "prune_step": 10**9})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["a2"])
    ap.add_argument("--bands", type=int, default=12)
    ap.add_argument("--order", type=int, default=4)
    a = ap.parse_args()
    rows = a2(a.bands, a.order)
    p = OUT / f"a2_families_b{a.bands}_o{a.order}.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} configs -> {p}")
