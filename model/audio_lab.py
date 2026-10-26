"""Audio lab: listen to what the simulator makes, and compose your own scenes.

  python model/audio_lab.py [--port 8765] [--host 127.0.0.1]   then open http://localhost:8765

Two tabs:
  Dataset   random examples made exactly the way training / test data is made (same code path:
            mix.Mixer.window), annotated from the mixer's own log: which voice said what, at what
            distance, in which room, over which ambience, at what SNR and level.
  Compose   your own text -> any Piper voice / speaker / speaking rate -> placed at a distance in a
            simulated or real recorded room (optionally muffled) -> mixed over up to 3 background
            layers (recorded ambience by category, TTS conversation, crowd babble, real speech,
            AMI meeting) -> the chip's level.
Every clip shows the chip's 16-band features (what the classifier sees) and, if a trained
checkpoint is picked, the detector's score over time.
"""

import argparse
import io
import json
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf
from flask import Flask, Response, jsonify, request
from scipy.signal import resample_poly

import mix
import store
from features import features
from synth_tts import PIPER_DIR, _voice, load_candidates, slug, trim

ROOT = Path(__file__).resolve().parent.parent
RATE = mix.RATE
app = Flask(__name__)
LOCK = threading.Lock()  # Piper and the mixers aren't thread-safe
CLIPS = OrderedDict()  # id -> wav bytes (most recent 300)
_mixers, _models, _real_idx = {}, {}, {}


def mixer(split):
    if split not in _mixers:
        _mixers[split] = mix.Mixer(split)
    return _mixers[split]


def wav_bytes(x):
    b = io.BytesIO()
    sf.write(b, np.clip(x, -1, 1).astype(np.float32), RATE, subtype="PCM_16", format="WAV")
    return b.getvalue()


def keep(x):
    cid = uuid.uuid4().hex[:10]
    CLIPS[cid] = wav_bytes(x)
    while len(CLIPS) > 300:
        CLIPS.popitem(last=False)
    return cid


def describe_voice(meta):
    """'voice:spk:length_scale:noise_scale:noise_w[:text]' -> readable."""
    p = str(meta).split(":")
    s = f"TTS voice {p[0]} speaker {p[1]}"
    if len(p) >= 5:
        s += f", rate {1 / float(p[2]):.2f}x, expressiveness {float(p[3]):.2f}"
    if len(p) >= 6:
        s += f', text "{":".join(p[5:])}"'
    return s


def describe(log):
    """The mixer's log -> human-readable lines."""
    out = []
    for e in log:
        w = e["what"]
        if w == "ambience":
            out.append(f"ambience: {e['category']} recording #{e['item']}" + (f" (second source, {e['level_db']} dB)" if "level_db" in e else ""))
        elif w == "crowd babble":
            out.append(f"crowd babble: {e['conversations']} simultaneous TTS conversations")
        elif w == "crowd level":
            out.append(f"crowd level vs ambience: {e['level_db_vs_ambience']} dB")
        elif w == "conversation":
            out.append(f"conversation: {e['talkers']} talker(s) at {e['distances_m']} m")
        elif w == "utterance":
            out.append(f'  says "{e["text"]}" ({describe_voice(e["voice"])})')
        elif w == "real talker (MUSAN speech)":
            out.append(f"real talker (MUSAN speech #{e['item']}) at {e['distance_m']} m")
        elif w == "speed":
            out.append(f"speed perturbation x{e['factor']}")
        elif w == "placement":
            out.append(f"talker at {e['distance_m']} m in a {e['room']}, low-pass {e['lowpass_hz']} Hz" + (" (MUFFLED)" if e["muffled"] else ""))
        elif w == "speech before the clip":
            out.append(f"someone talks before the clip (gap {e['gap_s']} s):")
        elif w == "background":
            out.append("background: almost silent (clean example, -50 dB)" if "snr_db" in e else
                       f"background: SNR {e['snr_db_actual']} dB at the mic ({e['snr_db_at_0_5m']} dB for a talker at 0.5 m)")
        elif w == "mic level":
            out.append(f"overall level at the mic: {e['dbfs']} dBFS")
        else:
            out.append(json.dumps(e))
    return out


# --- detector scores ---

def checkpoints():
    return sorted(str(p.relative_to(ROOT)) for p in (ROOT / "data/results").glob("**/*.pt"))


def score(x, ck):
    if not ck:
        return None
    import torch

    import archs

    if ck not in _models:
        c = torch.load(ROOT / ck, weights_only=False, map_location="cpu")
        if "arch" in c:
            m = archs.build(c["arch"], c["width"])
        else:  # older CPU checkpoints (model/train.py): GRU
            from train import KWS

            m = KWS(hidden=c["state"]["gru.weight_hh_l0"].shape[1])
        m.load_state_dict(c["state"])
        _models[ck] = (m.eval(), np.asarray(c["mu"], np.float32), np.asarray(c["sd"], np.float32))
    m, mu, sd = _models[ck]
    f = (features(np.clip(x, -1, 1)).astype(np.float32) / 8 - mu) / sd
    with torch.no_grad():
        out = m(torch.tensor(f[None]))
        out = out[0] if isinstance(out, tuple) else out
    return [round(float(v), 2) for v in out[0]]


def result(title, x, lines, ck=None):
    return {"id": keep(x), "title": title, "lines": lines, "seconds": round(len(x) / RATE, 2),
            "features": features(np.clip(x, -1, 1)).astype(int).tolist(), "score": score(x, ck)}


# --- dataset samples (same code path as build_cache / build_stream) ---

def clip_from(path, rng):
    d = store.load(path)
    i = int(rng.integers(len(d["offsets"]) - 1))
    a = np.asarray(d["audio"][d["offsets"][i] : d["offsets"][i + 1]], np.float32) / 32768
    meta = str(d["meta"][i]) if "meta" in d else ""
    return a, meta


def real_clip(word, rng):
    if word not in _real_idx:
        _real_idx[word] = []
        for name in ("sc_test", "mswc_test"):
            p = ROOT / "data/pools" / name
            if store.exists(p):
                idx = np.nonzero(np.asarray(store.load(p)["word"]) == word)[0]
                _real_idx[word] += [(p, int(i)) for i in idx]
    if not _real_idx[word]:
        return None, None
    p, i = _real_idx[word][rng.integers(len(_real_idx[word]))]
    d = store.load(p)
    a = np.asarray(d["audio"][d["offsets"][i] : d["offsets"][i + 1]], np.float32) / 32768
    return a, f"real recording: {p.name.replace('_test', '').upper()} clip #{i} of '{word}'"


@app.get("/api/sample")
def api_sample():
    kind = request.args.get("kind", "positive")
    word = request.args.get("word", "telescope")
    ck = request.args.get("checkpoint") or None
    rng = np.random.default_rng(int(request.args.get("seed") or np.random.SeedSequence().entropy % 2**32))
    sw = slug(word)
    with LOCK:
        if kind == "ami":
            d = store.load(ROOT / "data/pools/ami_test")
            off = np.asarray(d["offsets"])
            m = int(rng.integers(len(off) - 1))
            s = int(rng.integers(off[m], off[m + 1] - 10 * RATE))
            x = np.asarray(d["audio"][s : s + 10 * RATE], np.float32) / 32768
            return jsonify(result("TEST (real): AMI meeting, far-field table mic, untouched",
                                  x, [f"AMI meeting {d['meeting'][m]}, 10 s from {(s - off[m]) / RATE:.0f} s", "real level, real room"], ck))
        split, title, clip, head = {
            "positive": ("train", f"TRAINING positive: '{word}'", ROOT / "data/synth" / sw / "train", None),
            "simtest_positive": ("simtest", f"SIM-TEST positive (held-out voices/rooms/ambience): '{word}'", ROOT / "data/synth" / sw / "test", None),
            "nearmiss": ("train", f"TRAINING negative: near-miss of '{word}'", ROOT / "data/synth_near" / sw / "train", None),
            "partial": ("train", "TRAINING negative: partial phrase", None, "partial"),
            "convo": ("train", "TRAINING negative: conversation", None, "convo"),
            "noise": ("train", "TRAINING negative: ambience only", None, "noise"),
            "real_positive": ("test", f"TEST (real voice): '{word}'", None, "real"),
        }[kind]
        m = mixer(split)
        m.log = []
        lines = []
        if head == "partial":
            parts = sorted((ROOT / "data/synth_neg").glob("*/train"))
            clip, meta = clip_from(parts[rng.integers(len(parts))], rng)
            lines.append(describe_voice(meta))
            x, _ = m.window(rng, "clip", clip, audio=True)
        elif head in ("convo", "noise"):
            x, _ = m.window(rng, head, audio=True)
        elif head == "real":
            a, desc = real_clip(word, rng)
            if a is None:
                return jsonify({"error": f"no real recordings of '{word}'"}), 404
            lines.append(desc)
            x, _ = m.window(rng, "clip", a, audio=True)
        else:
            a, meta = clip_from(clip, rng)
            lines.append(describe_voice(meta))
            x, _ = m.window(rng, "clip", a, audio=True)
        lines += describe(m.log)
        m.log = None
        lines.append("1.6 s window, exactly as the detector sees it in training/testing (keyword ends in the last 160 ms)")
        return jsonify(result(title, x, lines, ck))


# --- compose ---

def tts(text, voice, speaker, rate, expr, noise_w, rng):
    from piper.config import SynthesisConfig

    v = _voice(voice)
    spk = int(rng.integers(v.config.num_speakers)) if speaker < 0 else min(speaker, v.config.num_speakers - 1)
    cfg = SynthesisConfig(speaker_id=spk, length_scale=1 / rate, noise_scale=expr, noise_w_scale=noise_w)
    a = np.concatenate([c.audio_int16_array for c in v.synthesize(text, syn_config=cfg)]).astype(np.float32)
    a = resample_poly(a, RATE, v.config.sample_rate) / 32768
    return trim(a, RATE, pad_s=0.05).astype(np.float32), spk


def layer(kind, src, n, rng):
    """One background layer, rms 1. -> (audio, description lines)"""
    split = "test" if kind == "real_speech" else ("train" if src == "train" else "simtest")
    if kind.startswith("ambience:"):
        m = mixer("train" if src == "train" else "test")
        cat = list(m.cat_names).index(kind.split(":", 1)[1])
        m.log = []
        x = m.ambience(rng, n, cat=cat)
    elif kind == "ami":
        d = store.load(ROOT / "data/pools/ami_test")
        off = np.asarray(d["offsets"])
        k = int(rng.integers(len(off) - 1))
        s = int(rng.integers(off[k], off[k + 1] - n))
        x = np.asarray(d["audio"][s : s + n], np.float32) / 32768
        return x / mix.rms(x), [f"AMI meeting {d['meeting'][k]} excerpt (real conversation, real room)"]
    else:
        m = mixer(split)
        m.log = []
        x = {"conversation": lambda: m.conversation(rng, n)[0], "babble": lambda: m.babble(rng, n)[0],
             "real_speech": lambda: m.conversation(rng, n)[0]}[kind]()
    lines = describe(m.log)
    m.log = None
    return x / mix.rms(x), lines


@app.post("/api/compose")
def api_compose():
    q = request.get_json()
    rng = np.random.default_rng(int(q.get("seed") or np.random.SeedSequence().entropy % 2**32))
    with LOCK:
        lines = []
        speech, spk = None, None
        if q.get("text", "").strip():
            speech, spk = tts(q["text"], q["voice"], int(q.get("speaker", -1)), float(q.get("rate", 1)),
                              float(q.get("expr", 0.667)), float(q.get("noise_w", 0.8)), rng)
            lines.append(f'TTS "{q["text"]}": voice {q["voice"]} speaker {spk}, rate {float(q.get("rate", 1)):.2f}x, '
                         f'expressiveness {float(q.get("expr", 0.667)):.2f}')
            sp = float(q.get("speed", 1))
            if sp != 1:
                speech = mix.speed(speech, sp)
                lines.append(f"speed/pitch perturbation x{sp:.2f}")
            speech = speech / mix.rms(speech)  # a talker at 0.5 m has rms 1
            d = float(q.get("distance", 0.5))
            room = q.get("room", "sim")
            if room == "none":
                speech = speech * (mix.REF_M / d)
                lines.append(f"dry (no room), level as at {d} m")
            else:
                m = mixer("train" if room == "sim" else "test")
                muff = {"auto": None, "yes": True, "no": False}[q.get("muffled", "auto")]
                p = m.placement(rng, d, muffled=muff)
                speech = m.place(speech, p)
                lines.append(f"placed at {d} m in a {p['room_kind']} (#{p['room']}), low-pass {p['fc']} Hz"
                             + (" (MUFFLED)" if p["muffled"] else "") + f", room critical distance {p['crit_m']} m")
        pad = int(float(q.get("pad", 1.0)) * RATE)
        n = (len(speech) if speech is not None else int(float(q.get("length", 5)) * RATE)) + 2 * pad
        x = np.zeros(n, np.float32)
        if speech is not None:
            x[pad : pad + len(speech)] = speech
        bg = np.zeros(n, np.float32)
        for L in q.get("layers", []):
            if not L.get("kind") or L["kind"] == "none":
                continue
            b, desc = layer(L["kind"], L.get("source", "train"), n, rng)
            g = 10 ** (float(L.get("level", 0)) / 20)
            bg += b * g
            lines.append(f"background layer {L['kind']} ({L.get('source', 'train')} pool) at {float(L.get('level', 0)):+.0f} dB: " + "; ".join(desc))
        if np.any(bg):
            snr = float(q.get("snr", 15))
            bg *= 10 ** (-snr / 20) / mix.rms(bg)
            lines.append(f"background total at {snr} dB below a talker at 0.5 m"
                         + (f" (= {snr - 20 * np.log10(float(q.get('distance', 0.5)) / mix.REF_M):.1f} dB SNR for this talker)" if speech is not None else ""))
        x = x + bg
        lvl = float(q.get("level", -30))
        x = x * (10 ** (lvl / 20) / mix.rms(x))
        lines.append(f"overall level {lvl} dBFS at the mic" + (" (CLIPPED)" if np.abs(x).max() > 1 else ""))
        return jsonify(result("COMPOSED", x, lines, q.get("checkpoint") or None))


@app.get("/api/options")
def api_options():
    voices = []
    for f in sorted(PIPER_DIR.glob("*.onnx")):
        cfg = json.loads(Path(str(f) + ".json").read_text())
        voices.append({"name": f.stem, "speakers": cfg.get("num_speakers", 1)})
    amb = {s: [str(c) for c in store.load(mix.POOLS / f"bg_{s}")["cats"]] for s in ("train", "test")}
    return jsonify({"voices": voices, "ambience": amb, "words": [c["phrase"] for c in load_candidates()],
                    "checkpoints": checkpoints()})


@app.get("/clip/<cid>.wav")
def clip(cid):
    if cid not in CLIPS:
        return "gone", 404
    return Response(CLIPS[cid], mimetype="audio/wav")


@app.get("/")
def index():
    return Response((Path(__file__).parent / "audio_lab.html").read_text(), mimetype="text/html")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it from another machine")
    a = ap.parse_args()
    app.run(host=a.host, port=a.port, threaded=True)
