"""Render every candidate wake word with many TTS speakers / speeds / prosody settings.

Output: data/synth/<slug>/{train,test}/ (store.py format) with
    audio   int16, all clips concatenated (16 kHz)
    offsets int64, clip start indices (len = n_clips + 1)
    meta    str,   "voice:speaker:length_scale:noise_scale:noise_w:text" per clip
Speakers are split by id (every 5th speaker of each voice -> test), so test
clips always come from voices the model never heard in training.

Usage: python model/synth_tts.py [--per-phrase 2000] [--only slug,slug] [--jobs 8]
       python model/synth_tts.py --convo 100000,15000     # conversation utterances -> data/synth_convo/
       python model/synth_tts.py --nearmiss --per-phrase 500  # near-misses per candidate -> data/synth_near/
"""

import argparse
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import json

import numpy as np
import yaml
from scipy.signal import resample_poly

import store

ROOT = Path(__file__).resolve().parent.parent
PIPER_DIR = ROOT / "data" / "piper"
OUT_DIR = ROOT / "data" / "synth"
# (model, weight). Multi-speaker models give speaker variety; single-speaker models give distinct
# styles and accents (each gets an equal share of SINGLE_WEIGHT). model/data/piper_voices.sh.
MULTI = [
    ("en_US-libritts_r-medium", 0.28),  # ~900-2300 read-speech speakers
    ("en_US-libritts-high", 0.14),  # a separately trained LibriTTS model: different timbre
    ("en_GB-vctk-medium", 0.12),  # 109 British-accent speakers
    ("en_US-l2arctic-medium", 0.06),  # 24 non-native speakers
    ("en_US-arctic-medium", 0.04),
    ("en_GB-aru-medium", 0.06),  # 12 regional British speakers
    ("en_GB-semaine-medium", 0.06),  # 4 emotional styles (happy, angry, sad, neutral)
]
SINGLE = [
    "en_GB-alan-medium", "en_GB-alba-medium", "en_GB-cori-medium", "en_GB-jenny_dioco-medium",
    "en_GB-northern_english_male-medium", "en_GB-southern_english_female-low", "en_US-amy-medium",
    "en_US-bryce-medium", "en_US-danny-low", "en_US-hfc_female-medium", "en_US-hfc_male-medium",
    "en_US-joe-medium", "en_US-john-medium", "en_US-kathleen-low", "en_US-kristin-medium",
    "en_US-kusal-medium", "en_US-lessac-medium", "en_US-ljspeech-medium", "en_US-mike-medium",
    "en_US-norman-medium", "en_US-reza_ibrahim-medium", "en_US-ryan-medium", "en_US-sam-medium",
]
SINGLE_WEIGHT = 0.24
# Held-out test voices among the single-speaker models (whole voices never heard in training);
# multi-speaker models hold out every TEST_EVERY-th speaker instead.
TEST_SINGLE = {"en_GB-alba-medium", "en_GB-northern_english_male-medium", "en_US-hfc_female-medium",
               "en_US-kristin-medium", "en_US-reza_ibrahim-medium", "en_US-ryan-medium"}
VOICES = MULTI + [(v, SINGLE_WEIGHT / len(SINGLE)) for v in SINGLE]
TEST_EVERY = 5
LENGTH_SCALE, NOISE_SCALE, NOISE_W = (0.7, 1.4), (0.3, 1.1), (0.4, 1.3)  # prosody ranges


def split_of(name, spk, n_speakers):
    """'test' for held-out speakers / voices, else 'train'."""
    if n_speakers > 1:
        return "test" if spk % TEST_EVERY == 0 else "train"
    return "test" if name in TEST_SINGLE else "train"


def slug(phrase):
    return re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")


def load_candidates():
    with open(ROOT / "model" / "wakewords.yaml") as f:
        return yaml.safe_load(f)["candidates"]


def partial_phrases(phrases):
    """Every contiguous proper sub-phrase of the multi-word candidates (e.g. "hey", "tux",
    "little penguin"), minus anything that is itself a candidate. Used as hard negatives."""
    cands = set(phrases)
    parts = set()
    for p in phrases:
        w = p.split()
        for i in range(len(w)):
            for j in range(i + 1, len(w) + 1):
                if j - i < len(w):
                    parts.add(" ".join(w[i:j]))
    return sorted(parts - cands)


def text_variants(phrase):
    words = phrase.split()
    v = [phrase, phrase + "!", phrase + ".", phrase + "?", phrase.title() + "!"]
    if len(words) > 1:
        v.append(words[0] + ", " + " ".join(words[1:]) + "!")
    return v


def trim(x, rate, pad_s=0.1, rel_db=-40):
    """Drop leading/trailing near-silence, keep pad_s on each side."""
    env = np.abs(x.astype(np.float32))
    thr = env.max() * 10 ** (rel_db / 20)
    idx = np.nonzero(env > thr)[0]
    if len(idx) == 0:
        return x
    pad = int(pad_s * rate)
    return x[max(0, idx[0] - pad) : idx[-1] + pad]


_voices = {}


def _limit_ort_threads(n=2):
    """Piper builds its own InferenceSession; cap its threads so many workers can share the CPU."""
    import onnxruntime as ort

    if getattr(ort.InferenceSession, "_capped", False):
        return
    orig = ort.InferenceSession

    def capped(*args, sess_options=None, **kw):
        so = sess_options or ort.SessionOptions()
        so.intra_op_num_threads = n
        so.inter_op_num_threads = 1
        return orig(*args, sess_options=so, **kw)

    capped._capped = True
    ort.InferenceSession = capped


def _voice(name):
    from piper import PiperVoice

    _limit_ort_threads()
    if name not in _voices:
        while len(_voices) >= 2:  # each loaded voice costs ~0.5-1 GB; keep a worker's footprint small
            _voices.pop(next(iter(_voices)))
        _voices[name] = PiperVoice.load(str(PIPER_DIR / f"{name}.onnx"))
    return _voices[name]


def _pick_voice(rng):
    names = [v for v, _ in VOICES]
    weights = np.array([w for _, w in VOICES])
    return names[rng.choice(len(names), p=weights / weights.sum())]


def render(job):
    """job: (key, phrases, n, seed): n clips, each of a random phrase from `phrases`, all from one
    voice model drawn per job (random speaker and prosody per clip), so a worker loads one model."""
    key, phrases, n, seed = job
    from piper.config import SynthesisConfig

    rng = np.random.default_rng(seed)
    name = _pick_voice(rng)
    voice = _voice(name)
    out = {"train": ([], []), "test": ([], [])}
    for _ in range(n):
        texts = text_variants(phrases[rng.integers(len(phrases))])
        spk = int(rng.integers(voice.config.num_speakers))
        split = split_of(name, spk, voice.config.num_speakers)
        ls = float(rng.uniform(*LENGTH_SCALE))
        ns = float(rng.uniform(*NOISE_SCALE))
        nw = float(rng.uniform(*NOISE_W))
        text = texts[rng.integers(len(texts))]
        cfg = SynthesisConfig(speaker_id=spk, length_scale=ls, noise_scale=ns, noise_w_scale=nw)
        audio = np.concatenate([c.audio_int16_array for c in voice.synthesize(text, syn_config=cfg)])
        audio = resample_poly(audio.astype(np.float32), 16000, voice.config.sample_rate)
        audio = trim(np.clip(audio, -32768, 32767).astype(np.int16), 16000)
        out[split][0].append(audio)
        out[split][1].append(f"{name}:{spk}:{ls:.3f}:{ns:.3f}:{nw:.3f}:{text}")
    return key, out


CONVO_DIR = ROOT / "data" / "synth_convo"


def _random_speaker(rng, split):
    """Pick a voice model that has speakers in `split`, and a speaker id from it (see split_of)."""
    while True:
        name = _pick_voice(rng)
        n = json.loads((PIPER_DIR / f"{name}.onnx.json").read_text()).get("num_speakers", 1)
        if n == 1 and split_of(name, 0, 1) != split:
            continue
        while True:
            spk = int(rng.integers(n))
            if split_of(name, spk, n) == split:
                return name, _voice(name), spk


def render_convo(job):
    """Render conversation utterances into one shard (store format) and return its path."""
    from piper.config import SynthesisConfig

    import store

    idx, seed, split, shard = job
    rng = np.random.default_rng(seed)
    texts = store.load(ROOT / "data" / "text" / "utts")["text"]
    clips, meta = [], []
    name, voice, _ = _random_speaker(rng, split)  # one voice model per shard, speakers vary within it
    for i in idx:
        while True:
            spk = int(rng.integers(voice.config.num_speakers))
            if split_of(name, spk, voice.config.num_speakers) == split:
                break
        ls, ns, nw = float(rng.uniform(*LENGTH_SCALE)), float(rng.uniform(*NOISE_SCALE)), float(rng.uniform(*NOISE_W))
        cfg = SynthesisConfig(speaker_id=spk, length_scale=ls, noise_scale=ns, noise_w_scale=nw)
        chunks = [c.audio_int16_array for c in voice.synthesize(str(texts[i]), syn_config=cfg)]
        if not chunks:
            continue
        audio = resample_poly(np.concatenate(chunks).astype(np.float32), 16000, voice.config.sample_rate)
        clips.append(trim(np.clip(audio, -32768, 32767).astype(np.int16), 16000))
        meta.append((int(i), f"{name}:{spk}:{ls:.3f}:{ns:.3f}:{nw:.3f}"))
    store.save(shard, audio=np.concatenate(clips), offsets=np.cumsum([0] + [len(c) for c in clips]),
               utt=np.array([m[0] for m in meta]), meta=np.array([m[1] for m in meta]))
    return str(shard)


def merge_shards(shards, out):
    """Concatenate shard clip sets into one store dir without holding the audio in RAM."""
    import shutil

    import store

    parts = [store.load(s) for s in shards]
    n = sum(len(p["audio"]) for p in parts)
    audio = store.create(out, "audio", (n,), np.int16)
    offsets, utt, meta, at = [0], [], [], 0
    for p in parts:
        audio[at : at + len(p["audio"])] = p["audio"]
        offsets += list(at + np.asarray(p["offsets"][1:]))
        utt.append(np.asarray(p["utt"]))
        meta.append(np.asarray(p["meta"]))
        at += len(p["audio"])
    audio.flush()
    store.save(out, offsets=np.array(offsets), utt=np.concatenate(utt), meta=np.concatenate(meta))
    for s in shards:
        shutil.rmtree(s)


def main_convo(n_train, n_test, jobs, chunk=100):
    """Render n_train / n_test utterances from the train / test text split with train / test speakers."""
    import store

    utts = store.load(ROOT / "data" / "text" / "utts")
    rng = np.random.default_rng(7)
    work = []
    for split, n in (("train", n_train), ("test", n_test)):
        if store.exists(CONVO_DIR / split):
            print(f"{split}: exists, skipping", file=sys.stderr)
            continue
        idx = rng.permutation(np.nonzero(utts["split"] == (split == "test"))[0])[:n]
        for k in range(0, len(idx), chunk):
            work.append((idx[k : k + chunk], 1_000_000 + len(work), split, CONVO_DIR / "shards" / f"{split}_{k:07d}"))
    done = {"train": [], "test": []}
    with Pool(jobs) as pool:
        for i, shard in enumerate(pool.imap_unordered(render_convo, work)):
            done["test" if "/test_" in shard else "train"].append(shard)
            if i % 50 == 0:
                print(f"convo: {i + 1}/{len(work)} shards", file=sys.stderr, flush=True)
    for split, shards in done.items():
        if shards:
            merge_shards(sorted(shards), CONVO_DIR / split)
            d = store.load(CONVO_DIR / split)
            print(f"{split}: {len(d['utt'])} utterances, {len(d['audio']) / 16000 / 3600:.1f} h", file=sys.stderr)


def save(phrase, out):
    d = OUT_DIR / slug(phrase)
    d.mkdir(parents=True, exist_ok=True)
    for split, (clips, meta) in out.items():
        offsets = np.cumsum([0] + [len(c) for c in clips])
        audio = np.concatenate(clips) if clips else np.zeros(0, np.int16)
        store.save(d / split, audio=audio, offsets=offsets, meta=np.array(meta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-phrase", type=int, default=2000)
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--negatives", action="store_true", help="render partial-phrase hard negatives")
    ap.add_argument("--convo", default="", help="TRAIN,TEST: render that many conversation utterances")
    ap.add_argument("--nearmiss", action="store_true", help="render data/text/nearmiss.json -> data/synth_near/")
    args = ap.parse_args()
    if args.convo:
        n_train, n_test = map(int, args.convo.split(","))
        return main_convo(n_train, n_test, args.jobs)
    global OUT_DIR
    if args.negatives:
        OUT_DIR = ROOT / "data" / "synth_neg"

    phrases = [c["phrase"] for c in load_candidates()]
    texts = {p: [p] for p in phrases}
    if args.negatives:
        phrases = partial_phrases(phrases)
        texts = {p: [p] for p in phrases}
    if args.nearmiss:  # one output set per candidate, holding its near-miss phrases
        import json

        OUT_DIR = ROOT / "data" / "synth_near"
        near = json.loads((ROOT / "data" / "text" / "nearmiss.json").read_text())
        texts = {p: near[slug(p)] for p in phrases if near.get(slug(p))}
        phrases = list(texts)
    if args.only:
        keep = set(args.only.split(","))
        phrases = [p for p in phrases if slug(p) in keep]
    phrases = [p for p in phrases if not store.exists(OUT_DIR / slug(p) / "test")]

    # Split each phrase into chunks so all workers stay busy.
    jobs = []
    for i, p in enumerate(phrases):
        for k in range(0, args.per_phrase, args.chunk):
            jobs.append((p, texts[p], min(args.chunk, args.per_phrase - k), 7_000_000 * args.nearmiss + 1000 * i + k))
    acc = {p: {"train": ([], []), "test": ([], [])} for p in phrases}
    left = {p: sum(1 for j in jobs if j[0] == p) for p in phrases}

    with Pool(args.jobs) as pool:
        for phrase, out in pool.imap_unordered(render, jobs):
            for split in out:
                acc[phrase][split][0].extend(out[split][0])
                acc[phrase][split][1].extend(out[split][1])
            left[phrase] -= 1
            if left[phrase] == 0:
                save(phrase, acc.pop(phrase))
                print(f"done {phrase}", file=sys.stderr, flush=True)
    print(f"rendered {len(phrases)} phrases", file=sys.stderr)


if __name__ == "__main__":
    main()
