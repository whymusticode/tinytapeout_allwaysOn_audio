"""Render extra phrases (not wake-word candidates) with the 30-voice simulator, for probes.

  python model/probe_render.py --phrases "alexa,hey jarvis,hey mycroft,okay nabu" --per-phrase 2000

Same voices, prosody ranges and train/test voice split as synth_tts; output goes to
data/probe/synth/<slug>/{train,test} in the same store format.
"""

import argparse
import sys
from multiprocessing import Pool

import synth_tts
from synth_tts import ROOT, render, save, slug

OUT = ROOT / "data" / "probe" / "synth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrases", required=True)
    ap.add_argument("--per-phrase", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=50)
    args = ap.parse_args()
    synth_tts.OUT_DIR = OUT
    phrases = args.phrases.split(",")
    jobs = [(p, [p], min(args.chunk, args.per_phrase - k), 9_000_000 + 1000 * i + k)
            for i, p in enumerate(phrases) for k in range(0, args.per_phrase, args.chunk)]
    acc = {p: {"train": ([], []), "test": ([], [])} for p in phrases}
    left = {p: sum(j[0] == p for j in jobs) for p in phrases}
    with Pool(args.jobs) as pool:
        for p, out in pool.imap_unordered(render, jobs):
            for s in out:
                acc[p][s][0].extend(out[s][0])
                acc[p][s][1].extend(out[s][1])
            left[p] -= 1
            if left[p] == 0:
                save(p, acc.pop(p))
                print(f"done {p}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
