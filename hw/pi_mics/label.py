#!/usr/bin/env python3
"""Labelled-word session: prompts words on screen while record.py keeps recording.

  python3 label.py --speaker mike --d0 0.5 --d1 3 --n 40 --note "kitchen, dishwasher on"

For each prompt: the word is shown, you say it once, naturally, and it moves on after
--window seconds (or press Enter to go on early, or type 's' + Enter to mark it skipped).
Prompts are candidate keywords and their near-misses from words.txt, in random order.

Each prompt becomes one line in labels/<date>_<speaker>.jsonl:
  {"word", "kind" (keyword|nearmiss), "target", "t0", "t1" (wall clock, unix s),
   "d_m": [mic0 distance, mic1 distance], "speaker", "note", "skipped"}
The audio is cut out of the continuous recordings later on the workstation, so this script
needs no audio libraries and can run over ssh while you walk around the room.

These recordings are TEST data (real voices). Training stays simulated.
"""

import argparse
import json
import random
import select
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_words(path):
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            w, kind, target = line.split("\t")
            out.append({"word": w, "kind": kind, "target": target})
    return out


def wait(seconds):
    """Wait up to `seconds` for a line on stdin. -> the line, or None on timeout."""
    r, _, _ = select.select([sys.stdin], [], [], seconds)
    return sys.stdin.readline().strip() if r else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True, help="who is talking (anonymous id is fine)")
    ap.add_argument("--d0", type=float, required=True, help="your distance to mic 0, m")
    ap.add_argument("--d1", type=float, required=True, help="your distance to mic 1, m")
    ap.add_argument("--n", type=int, default=40, help="prompts this session")
    ap.add_argument("--near", type=float, default=0.3, help="fraction of near-miss prompts")
    ap.add_argument("--only", help="comma-separated targets to restrict to, e.g. telescope,camel")
    ap.add_argument("--window", type=float, default=2.5, help="seconds per prompt")
    ap.add_argument("--note", default="")
    ap.add_argument("--words", default=str(HERE / "words.txt"))
    args = ap.parse_args()

    words = load_words(args.words)
    if args.only:
        keep = set(args.only.split(","))
        words = [w for w in words if w["target"] in keep]
    keys = [w for w in words if w["kind"] == "keyword"]
    near = [w for w in words if w["kind"] == "nearmiss"]
    n_near = int(round(args.n * args.near)) if near else 0
    prompts = random.choices(keys, k=args.n - n_near) + random.choices(near, k=n_near)
    random.shuffle(prompts)

    out = HERE / "labels" / f"{datetime.now():%Y-%m-%d}_{args.speaker}.jsonl"
    out.parent.mkdir(exist_ok=True)
    print(f"{len(prompts)} prompts -> {out}\nSay each word once, at a normal volume. Enter = next, s = skip.\n")
    wait(1.0)
    with out.open("a") as f:
        for i, p in enumerate(prompts):
            print(f"\n[{i + 1}/{len(prompts)}]   >>>  {p['word'].upper()}  <<<", flush=True)
            t0 = time.time()
            ans = wait(args.window)
            t1 = time.time()
            rec = {**p, "t0": t0, "t1": t1, "d_m": [args.d0, args.d1], "speaker": args.speaker,
                   "note": args.note, "skipped": ans == "s"}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            wait(0.7)  # a pause so the next word isn't run into this one
    print("\ndone")


if __name__ == "__main__":
    main()
