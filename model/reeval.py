"""Re-run the test metrics of finished runs with the current gpu_train.test() (e.g. after new
operating points were added), keeping each run's training curve.

  python model/reeval.py --word telescope --out data/results/arch_v1
"""

import argparse
import json
from pathlib import Path

import torch

import archs
import gpu_train as G
from synth_tts import load_candidates, slug

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", required=True)
    ap.add_argument("--out", default="data/results/arch_v1")
    args = ap.parse_args()
    k = [slug(c["phrase"]) for c in load_candidates()].index(args.word)
    D = G.Data(k)
    for js in sorted((ROOT / args.out).glob(f"{args.word}_*.json")):
        r = json.loads(js.read_text())
        ck = torch.load(js.with_suffix(".pt"), weights_only=False)
        m = archs.build(ck["arch"], ck["width"]).to(G.DEV)
        m.load_state_dict(ck["state"])
        curve = r.pop("curve")
        r.update(G.test(m, D))
        r["curve"] = curve
        js.write_text(json.dumps(r, indent=1))
        G.log(f"{js.stem}: fa_real@frr_real0.05 = {r.get('fa_real@frr_real0.05')}")


if __name__ == "__main__":
    main()
