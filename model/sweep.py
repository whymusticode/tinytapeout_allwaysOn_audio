"""Train/evaluate many (word, hidden size) combinations in parallel, then write a report.

  python model/sweep.py --hidden 32                     # all candidates at H=32
  python model/sweep.py --words hey_tux,marvin --hidden 8,16,24,32,48,64
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from synth_tts import load_candidates, slug

ROOT = Path(__file__).resolve().parent.parent


def run(word, hidden, out, steps, threads):
    if (ROOT / out / f"{word}.json").exists():
        return word, hidden, "cached"
    log = ROOT / out / f"{word}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        r = subprocess.run([sys.executable, str(ROOT / "model/train.py"), "--word", word, "--hidden", str(hidden),
                            "--steps", str(steps), "--threads", str(threads), "--out", out],
                           stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    return word, hidden, "ok" if r.returncode == 0 else f"FAILED (see {log})"


def _src_name(src, cands):
    src = {"-1": "speech/bg", "-2": "partial phrase", "-3": "conversation", "-5": "real word"}.get(src, src)
    if src.isdigit():
        t = int(src)
        src = (cands[t]["phrase"] if t < 1000 else f"real:{cands[t - 1000]['phrase']}" if t < 2000
               else f"near-miss of {cands[t - 2000]['phrase']}")
    return src


def report(out):
    import numpy as np

    rows = [json.loads(p.read_text()) for p in sorted((ROOT / out).glob("*.json"))]
    if not rows:
        return
    rows.sort(key=lambda r: (r["frr_syn@real1"], r["frr_syn@1.0"]))
    cands = load_candidates()
    h = rows[0]["stream_hours"]
    ami = np.array([r["fa_real_ami@1.0"] for r in rows])
    read = np.array([r["fa_real_read@1.0"] for r in rows])
    convo_ratio = np.array([r["fa_real_ami@2.0"] / max(r["fa_sim_convo@2.0"], 1e-3) for r in rows])
    real_pairs = [(r["frr_syn@1.0"], r["frr_real@1.0"]) for r in rows if "frr_real@1.0" in r]
    lines = [f"# Wake-word sweep: {out}", "",
             "Training is 100% simulated (TTS + recorded ambience + simulated rooms). Streams (hours): "
             + ", ".join(f"{k} {v}" for k, v in h.items()) + ".", "",
             "## Does the simulator predict reality?", "",
             "Threshold set on the sim streams for 1 false alarm/hour; real false alarms/hour at that threshold "
             f"(median over words): AMI meetings **{np.median(ami):.2f}**, real read speech **{np.median(read):.2f}** "
             "(1.0 = perfect calibration).",
             f"Like for like (conversation only, at the sim 2 FA/h threshold): real AMI false alarms / simulated "
             f"conversation false alarms, median over words = **{np.median(convo_ratio):.2f}** "
             "(1.0 = the simulated conversation predicts real meetings exactly; <1 = simulator is pessimistic).",
             (f"Missed wake words, sim vs real speakers (threshold for 1 real FA/h), mean over words: "
              f"TTS {np.mean([r['frr_syn@real1'] for r in rows]):.3f} vs real "
              f"{np.mean([r['frr_real@real1'] for r in rows if 'frr_real@real1' in r]):.3f}." if real_pairs else ""), "",
             "## Ranking", "",
             "Ranked by **FRR @ real 1/h**: missed held-out positives with the threshold set so the *real* streams "
             "give 1 false alarm per hour. `sim` columns use the sim-calibrated threshold.", "",
             "| rank | phrase | group | FRR TTS @real 1/h | FRR real speakers @real 1/h | sim convo FA/h @sim 2/h "
             "| AMI FA/h @sim 2/h | read FA/h @sim 2/h | top sim FA source |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        src = max(r["fa_sources"].items(), key=lambda kv: kv[1])[0] if r["fa_sources"] else "-"
        real = f"{r['frr_real@real1']:.3f}" if "frr_real@real1" in r else ""
        lines.append(f"| {i} | {r['word']} | {r['group']} | {r['frr_syn@real1']:.3f} | {real} | "
                     f"{r['fa_sim_convo@2.0']:.2f} | {r['fa_real_ami@2.0']:.2f} | {r['fa_real_read@2.0']:.2f} | "
                     f"{_src_name(src, cands)} |")
    (ROOT / out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:50]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", default="", help="comma-separated slugs (default: all candidates)")
    ap.add_argument("--hidden", default="32")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--parallel", type=int, default=14)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="data/results/sweep")
    args = ap.parse_args()
    words = args.words.split(",") if args.words else [slug(c["phrase"]) for c in load_candidates()]
    hs = [int(h) for h in args.hidden.split(",")]
    jobs = [(w, h, args.out if len(hs) == 1 else f"{args.out}/h{h}") for h in hs for w in words]
    with ThreadPoolExecutor(args.parallel) as ex:
        for w, h, status in ex.map(lambda j: run(*j, args.steps, args.threads), jobs):
            print(f"{w} H={h}: {status}", flush=True)
    for out in sorted({o for _, _, o in jobs}):
        report(out)


if __name__ == "__main__":
    main()
