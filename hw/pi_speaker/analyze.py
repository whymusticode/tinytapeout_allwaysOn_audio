"""Compare real speaker-in-a-room recordings with the simulator's "muffler" (run on the workstation).

  python hw/pi_speaker/analyze.py [--model data/results/tune3_h32/telescope.pt]

Reads hw/pi_speaker/recordings/*.wav (+ .json with the distance) and out/manifest.json.
Recordings are grouped by the "mic" field of their .json (each mic has its own coloration).
In each group the closest recording (<= 0.2 m) is the reference: it holds the same speaker, mic and clips,
with almost no room. For every farther recording we ask, clip by clip:

  real   what the room really did at distance d (recording at d)
  sim    what the simulator says it does: the reference clip through Mixer.place() at d
         (N_SIM random rooms), plus the recording's own noise floor

and compare in the chip's feature domain (16 log2 bands):
  level    dB drop vs the reference (the simulator assumes 6 dB per doubling of distance)
  tilt     per-band change in spectral shape (dullness with distance)
  tail     energy 50-250 ms after a clip ends, relative to its peak (reverberation)
  score    detector max logit on keyword clips, real vs sim (with --model)

-> recordings/report.md. If the real numbers sit inside the sim spread, the muffler is
realistic; if not, the table says which way to move it (mix.placement()).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
import mix  # noqa: E402
from features import HOP, features  # noqa: E402
from make_stimuli import sweep  # noqa: E402

HERE = Path(__file__).resolve().parent
RATE = 16000
N_SIM = 20  # simulated placements per clip
TAIL = (3, 15)  # frames after the clip end used for the reverb tail (48-240 ms)
DB = 20 * np.log10(2)  # dB per log2 unit (amplitude)


def load_rec(path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x[:, 0]
    if sr != RATE:
        g = np.gcd(sr, RATE)
        x = resample_poly(x, RATE // g, sr // g).astype(np.float32)
    return x


def find(x, ref, lo=0, hi=None):
    """Sample index in x[lo:hi] where ref starts (peak of the cross-correlation)."""
    seg = x[lo:hi]
    c = fftconvolve(seg, ref[::-1], mode="valid")
    return lo + int(np.argmax(np.abs(c)))


def align(x, man):
    """Locate both sync sweeps; resample so the recording runs on the stimulus clock.
    -> (recording with stimulus sample 0 at index 0, clock error in ppm)."""
    s = sweep()
    n = len(x)
    a = find(x, s, 0, n // 2)
    b = find(x, s, n // 2)
    span_stim = man["sync_end"] - man["sync_start"]
    ratio = (b - a) / span_stim
    t = a - man["sync_start"] * ratio + np.arange(n) * ratio  # stimulus sample i -> recording time t[i]
    y = np.interp(t, np.arange(n), x, left=0, right=0).astype(np.float32)
    return y, (ratio - 1) * 1e6


def log_bands(x):
    return features(np.clip(x, -1, 1)).astype(np.float32) / 8  # (frames, 16) log2 units


def profile(f):
    """Mean band levels over the clip's loud frames (within 20 dB of its loudest frame)."""
    e = f.max(1)
    loud = e > e.max() - 20 / DB
    return f[loud].mean(0)


def tail(f, end_frame):
    e = f.max(1)
    return (e[end_frame + TAIL[0] : end_frame + TAIL[1]].mean() - e[:end_frame].max()) * DB


def segment(x, it, extra=0.4):
    return x[it["start"] : it["start"] + it["len"] + int(extra * RATE)]


def clip_stats(seg, n_clip):
    f = log_bands(seg)
    return profile(f[: n_clip // HOP]), tail(f, n_clip // HOP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE / "recordings"))
    ap.add_argument("--model", help="trained .pt checkpoint (model/train.py) to score keyword clips")
    args = ap.parse_args()
    man = json.loads((HERE / "out/manifest.json").read_text())
    recs = []
    for w in sorted(Path(args.dir).glob("*.wav")):
        meta = json.loads(w.with_suffix(".json").read_text()) if w.with_suffix(".json").exists() else {}
        if "dist_m" not in meta:
            print(f"skip {w.name}: no distance (write {w.stem}.json with dist_m)")
            continue
        x, ppm = align(load_rec(w), man)
        recs.append({"name": w.stem, "d": float(meta["dist_m"]), "note": meta.get("note", ""), "x": x, "ppm": ppm,
                     "mic": meta.get("mic", "default")})
    lines = []
    for mic in sorted({r["mic"] for r in recs}):
        group = sorted([r for r in recs if r["mic"] == mic], key=lambda r: r["d"])
        if group[0]["d"] > 0.2:
            print(f"mic {mic}: no reference recording at <= 0.2 m, skipped")
            continue
        lines += report(group, man, args)
    out = Path(args.dir) / "report.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def report(recs, man, args):
    ref = recs[0]
    items = [it for it in man["items"] if it["kind"] in ("keyword", "nearmiss", "convo")]
    pink = next(it for it in man["items"] if it["kind"] == "pink")

    scorer = Scorer(args.model) if args.model else None
    mix.P_MUFFLED = 0.0  # compare against the normal (unmuffled) distance model
    m = mix.Mixer("train")
    rng = np.random.default_rng(0)
    lines = [f"# Mic `{ref['mic']}`: real room vs simulator", "",
             f"Reference: `{ref['name']}` at {ref['d']} m. Levels and shapes are relative to it.", "",
             "| recording | d (m) | note | clock ppm | noise floor (dB re clip) | level drop real / sim (dB)"
             " | shape diff real vs sim (dB rms) | sim shape spread (dB rms) | tail real / sim p10-p90 (dB)"
             + (" | keyword score real / sim |" if scorer else " |"),
             "|---|---|---|---|---|---|---|---|---" + ("|---|" if scorer else "|")]
    band_diff = {}
    for r in recs:
        x, x0 = r["x"], ref["x"]
        noise = x[pink["start"] - int(0.9 * RATE) : pink["start"] - int(0.1 * RATE)]  # silent gap before pink
        floor = (profile(log_bands(noise)).mean() - profile(log_bands(segment(x, pink, 0))).mean()) * DB
        lv_real, sh_real, sh_sim, spread, t_real, t_sim, s_real, s_sim = [], [], [], [], [], [], [], []
        for it in items:
            p0, _ = clip_stats(segment(x0, it), it["len"])
            p1, t1 = clip_stats(segment(x, it), it["len"])
            lv_real.append((p1.mean() - p0.mean()) * DB)
            sh_real.append((p1 - p1.mean()) - (p0 - p0.mean()))
            t_real.append(t1)
            sims = []
            seg0 = segment(x0, it)
            for _ in range(N_SIM if r is not ref else 0):
                p = m.placement(rng, r["d"])
                y = m.place(seg0, p) * (ref["d"] / r["d"]) / p["gain"]  # free-field level change from ref to d
                y = y + np.resize(noise, len(y))
                ps, ts = clip_stats(y, it["len"])
                sims.append(((ps - ps.mean()) - (p0 - p0.mean()), ts))
                if scorer and it["kind"] == "keyword" and len(sims) <= 3:
                    s_sim.append(scorer(y, it["len"]))
            if sims:
                sh = np.array([s for s, _ in sims])
                sh_sim.append(sh.mean(0))
                spread.append(np.sqrt(((sh - sh.mean(0)) ** 2).mean()) * DB)
                t_sim += [t for _, t in sims]
            if scorer and it["kind"] == "keyword":
                s_real.append(scorer(segment(x, it), it["len"]))
        lv = np.mean(lv_real)
        lv_sim = -20 * np.log10(r["d"] / ref["d"])
        if r is ref:
            row = f"| {r['name']} | {r['d']} | {r['note']} | {r['ppm']:.0f} | {floor:.0f} | (ref) | | | {np.median(t_real):.0f} / |"
        else:
            d_real, d_sim = np.mean(sh_real, 0), np.mean(sh_sim, 0)
            band_diff[r["name"]] = (d_real - d_sim) * DB
            row = (f"| {r['name']} | {r['d']} | {r['note']} | {r['ppm']:.0f} | {floor:.0f} | {lv:.1f} / {lv_sim:.1f}"
                   f" | {np.sqrt(np.mean((d_real - d_sim) ** 2)) * DB:.1f} | {np.mean(spread):.1f}"
                   f" | {np.median(t_real):.0f} / {np.percentile(t_sim, 10):.0f}..{np.percentile(t_sim, 90):.0f} |")
        if scorer:
            row += f" {np.median(s_real):.1f} / {np.median(s_sim) if s_sim else float('nan'):.1f} |"
        lines.append(row)
    lines += ["", "## Per-band shape error, real minus sim (dB; + = real has more energy in that band)", "",
              "| recording | " + " | ".join(f"{c:.0f}" for c in _centers()) + " |",
              "|---|" + "---|" * 16]
    for name, d in band_diff.items():
        lines.append(f"| {name} | " + " | ".join(f"{v:+.1f}" for v in d) + " |")
    lines += ["", "How to read it:",
              "- level: real drop much smaller than sim = the room keeps sound louder than 1/d (lower crit distance in sim is too pessimistic), and vice versa.",
              "- shape diff well above sim spread = the simulator's dulling with distance is wrong; the per-band row shows which way.",
              "- tail outside the sim p10..p90 = the reverberation model (direct/tail split, crit distance) needs moving.", ""]
    return lines


def _centers():
    from frontend import band_centers

    return band_centers()


class Scorer:
    """Detector score for a clip: max logit in the frames right after the word ends, with the
    1.6 s of audio before it as context (like a training window)."""

    def __init__(self, path):
        import torch

        from train import KWS, POS_LAST

        ck = torch.load(path, weights_only=False)
        h = ck["state"]["gru.weight_hh_l0"].shape[1]
        self.model = KWS(hidden=h)
        self.model.load_state_dict(ck["state"])
        self.mu, self.sd, self.last, self.torch = ck["mu"], ck["sd"], POS_LAST, torch

    def __call__(self, seg, n_clip):
        x = seg[: n_clip + 4 * HOP]
        x = np.pad(x, (max(0, mix.WIN - len(x)), 0))[-mix.WIN :]
        x = x * (10 ** (-30 / 20) / mix.rms(x))  # same level for real and sim
        f = (log_bands(x) - self.mu) / self.sd
        with self.torch.no_grad():
            l, _ = self.model(self.torch.tensor(f[None], dtype=self.torch.float32))
        return float(l[0, -self.last :].max())


if __name__ == "__main__":
    main()
