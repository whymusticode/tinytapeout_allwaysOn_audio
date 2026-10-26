"""The acoustic simulator: put speech into rooms and ambience the way the chip will hear it.

Three modes, one class:
  train    TTS speech (train speakers + train text), recorded ambience (train files),
           simulated rooms (train rooms). Everything the model learns from.
  simtest  The same simulator with held-out TTS speakers, text, ambience and rooms.
           Its scores vs the real test show how well the simulator matches reality.
  test     Real recorded speech (MUSAN speech as babble), real recorded rooms, test ambience.
           Used to put real voices (Speech Commands) into test windows; fully real
           streams (AMI meetings) need no mixing at all.

A training example is a T-frame window (1.6 s). If it holds a keyword clip, the clip
ends within the last END_SLACK frames, so a streaming model must fire right after the
word finishes.
"""

from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

import store
from features import HOP, features
from synth_tts import load_candidates, trim

T = 100  # frames per window (1.6 s)
WIN = T * HOP
RATE = 16000
END_SLACK = 10  # clip ends in the last 10 frames (160 ms)
LEVEL_DB = (-55, -12)  # overall RMS level at the mic, dBFS (AMI far-field sits near -48)
DIST_M = (0.3, 5.0)  # talker distance from the mic, log-uniform
REF_M = 0.5  # SNRs are drawn for a talker at this distance; farther talkers lose 6 dB per doubling
P_MUFFLED = 0.15  # pocket, bag, next room
SPEED = (0.9, 1.1)  # train-mode clips are sped up / slowed down (tempo and pitch together)
ROOT = Path(__file__).resolve().parent.parent
POOLS = ROOT / "data" / "pools"
N_CONTAINS = (len(load_candidates()) + 7) // 8  # bytes of packed "contains candidate" bits per window


def rms(x):
    return np.sqrt(np.mean(x**2) + 1e-12)


def db(lo, hi, rng):
    return 10 ** (rng.uniform(lo, hi) / 20)


_utt_text = None


def utt_text(i):
    global _utt_text
    if _utt_text is None:
        _utt_text = store.load(ROOT / "data" / "text" / "utts")["text"]
    return _utt_text[i]


def speed(x, r):
    """Play x at r times the speed (linear interpolation; r > 1 = faster and higher)."""
    return np.interp(np.arange(0, len(x) - 1, r), np.arange(len(x)), x).astype(np.float32)


class Mixer:
    def __init__(self, split):
        assert split in ("train", "simtest", "test")
        self.split = split
        self.log = None  # set to a list to record what each call did (audio_lab annotations)
        # Memory-mapped: every worker process shares one copy in the page cache.
        bg = store.load(POOLS / f"bg_{'train' if split == 'train' else 'test'}")
        self.bg, self.bg_cat, self.n_cat = bg["audio"], np.array(bg["cat"]), len(bg["cats"])
        self.cat_names = np.array(bg["cats"])
        self.cat_idx = [np.nonzero(self.bg_cat == c)[0] for c in range(self.n_cat)]
        r = store.load(POOLS / f"rir_{split}")
        self.rir, self.rir_off = r["audio"], np.array(r["offsets"])
        if split == "test":
            self.real_speech = store.load(POOLS / "speech_test")["audio"]
        else:
            c = store.load(ROOT / "data" / "synth_convo" / ("train" if split == "train" else "test"))
            self.utt_audio, self.utt_off = c["audio"], np.array(c["offsets"])
            self.utt_ids, self.utt_meta = np.array(c["utt"]), c["meta"]
            contains = np.array(store.load(ROOT / "data" / "text" / "utts")["contains"])
            self.utt_contains = contains[np.array(c["utt"])]

    # --- building blocks ---

    def note(self, **kw):
        if self.log is not None:
            self.log.append(kw)

    def _crop(self, rng, arr, i, n):
        s = rng.integers(0, arr.shape[1] - n + 1) if arr.shape[1] > n else 0
        c = arr[i, s : s + n].astype(np.float32) / 32768
        return np.pad(c, (0, n - len(c)))

    def ambience(self, rng, n, cat=None):
        """Recorded noise / music / environments: 1-2 sources, equal weight per category.
        cat: force one category (index into self.cat_names), single source."""
        c = rng.integers(self.n_cat) if cat is None else cat
        i = rng.choice(self.cat_idx[c])
        x = self._crop(rng, self.bg, i, n)
        self.note(what="ambience", category=str(self.cat_names[c]), item=int(i))
        if cat is None and rng.random() < 0.3:
            j = rng.integers(len(self.bg))
            g = db(-15, 0, rng)
            x = x + self._crop(rng, self.bg, j, n) * g
            self.note(what="ambience", category=str(self.cat_names[self.bg_cat[j]]), item=int(j),
                      level_db=round(20 * np.log10(g), 1))
        return x

    def room(self, rng):
        i = rng.integers(len(self.rir_off) - 1)
        self._room_id = int(i)
        return np.asarray(self.rir[self.rir_off[i] : self.rir_off[i + 1]])

    def reverb(self, rng, x, h=None):
        h = self.room(rng) if h is None else h
        y = fftconvolve(x, h)[: len(x)]
        return y * (rms(x) / max(rms(y), 1e-9))

    def placement(self, rng, d=None, muffled=None):
        """Where a talker is: distance, room response and muffling ("the muffler").
        -> dict for place(). Farther = quieter, more reverberant, duller."""
        d = float(np.exp(rng.uniform(*np.log(DIST_M)))) if d is None else d
        h = self.room(rng).astype(np.float64)
        k = int(np.argmax(np.abs(h)))
        direct = np.zeros_like(h)
        direct[: k + 40] = h[: k + 40]  # first 2.5 ms after the direct path
        tail = h - direct
        crit = rng.uniform(0.5, 1.5)  # critical distance of this room, m
        h = direct / max(np.abs(direct).max(), 1e-9) + tail / max(np.abs(direct).max(), 1e-9) * (d / crit)
        m = rng.random() < P_MUFFLED  # drawn even when forced, so random streams stay identical
        if muffled if muffled is not None else m:
            fc = rng.uniform(500, 2000)
        else:  # air absorption and off-axis loss: duller with distance
            fc = float(np.clip(7000 * (REF_M / d) ** 0.35 * rng.uniform(0.8, 1.1), 2500, 7500))
        return {"d": d, "h": h, "sos": butter(2, fc, fs=RATE, output="sos"), "gain": REF_M / d,
                "fc": round(fc), "muffled": bool(fc <= 2000), "crit_m": round(crit, 2), "room": self._room_id,
                "room_kind": "real recorded room" if self.split == "test" else "simulated room"}

    def place(self, x, p):
        """Apply a placement to close-talk audio x. Output RMS = rms(x) * gain."""
        y = sosfilt(p["sos"], fftconvolve(x, p["h"])[: len(x)])
        return (y * (rms(x) / max(rms(y), 1e-9) * p["gain"])).astype(np.float32)

    def utterance(self, rng):
        """One TTS conversation utterance -> (float audio, packed contains bits)."""
        i = rng.integers(len(self.utt_off) - 1)
        a = self.utt_audio[self.utt_off[i] : self.utt_off[i + 1]].astype(np.float32) / 32768
        if self.log is not None:
            self.note(what="utterance", text=str(utt_text(self.utt_ids[i])), voice=str(self.utt_meta[i]))
        return a / rms(a), self.utt_contains[i]

    def conversation(self, rng, n):
        """n samples of 1-3 people talking: turns with pauses and overlaps, each speaker with
        their own distance (level) and room position. -> (audio, contains bits)."""
        contains = np.zeros(N_CONTAINS, np.uint8)
        if self.split == "test":  # a real talker (one per crop), placed at a distance like simulated ones
            i = rng.integers(len(self.real_speech))
            x = self._crop(rng, self.real_speech, i, n)
            p = self.placement(rng)
            self.note(what="real talker (MUSAN speech)", item=int(i), distance_m=round(p["d"], 1))
            return self.place(x / rms(x), p) * db(-4, 4, rng), contains
        n_spk = rng.integers(1, 4)
        where = [self.placement(rng) for _ in range(n_spk)]  # each talker has a distance and a room spot
        self.note(what="conversation", talkers=int(n_spk), distances_m=[round(w["d"], 1) for w in where])
        loud = [db(-4, 4, rng) for _ in range(n_spk)]  # some people talk louder
        x = np.zeros(n, np.float32)
        pos = -int(rng.integers(0, 2 * RATE))  # start mid-utterance
        while pos < n:
            s = rng.integers(n_spk)
            u, bits = self.utterance(rng)
            a, b = max(pos, 0), min(pos + len(u), n)
            if b > a:
                x[a:b] += loud[s] * self.place(u[a - pos : b - pos], where[s])
                contains |= bits
            pos += len(u) + int(rng.uniform(-0.3, 1.2) * RATE)  # negative gap = overlap
        return x, contains

    def babble(self, rng, n):
        """A crowd: 3-6 distant talkers at once (cafe, open office, party)."""
        x = np.zeros(n, np.float32)
        contains = np.zeros(N_CONTAINS, np.uint8)
        k = rng.integers(3, 7)
        self.note(what="crowd babble", conversations=int(k))
        for _ in range(k):
            c, bits = self.conversation(rng, n)
            x += c * db(-6, 0, rng)
            contains |= bits
        return x, contains

    def background(self, rng, n):
        """Ambience, sometimes with a crowd in it."""
        x = self.ambience(rng, n)
        contains = np.zeros(N_CONTAINS, np.uint8)
        if rng.random() < 0.3:
            b, contains = self.babble(rng, n)
            g = db(-10, 5, rng)
            self.note(what="crowd level", level_db_vs_ambience=round(20 * np.log10(g), 1))
            x = x / rms(x) + b / rms(b) * g
        return x, contains

    # --- examples ---

    def window(self, rng, kind, clip=None, snr_db=(0, 35), p_clean=0.05, audio=False):
        """kind: "noise", "convo" or "clip" (keyword / partial phrase / real word in `clip`).
        snr_db is drawn for a talker at REF_M; the talker's distance lowers it further.
        -> ((T, 16) uint8 features, packed contains bits), or float audio if audio=True."""
        bg, contains = self.background(rng, WIN)
        if kind == "noise":
            x = bg
        else:
            dist_db = 0.0
            if kind == "convo":
                sig, bits = self.conversation(rng, WIN)  # talkers already placed at distances
                contains |= bits
                level = rms(sig)
            else:
                c = trim(clip.astype(np.float32), RATE, pad_s=0.05)  # real clips have silence around the word
                if self.split == "train":
                    r = rng.uniform(*SPEED)
                    c = speed(c, r)
                    self.note(what="speed", factor=round(r, 3))
                c = c / rms(c)
                p = self.placement(rng)
                self.note(what="placement", distance_m=round(p["d"], 2), room=p["room_kind"], lowpass_hz=p["fc"],
                          muffled=p["muffled"])
                c = self.place(c, p) / p["gain"]  # level handled via SNR below
                dist_db = 20 * np.log10(p["d"] / REF_M)
                end = WIN - int(rng.integers(1, END_SLACK)) * HOP - int(rng.integers(HOP))
                c = c[-end:]
                sig = np.zeros(WIN, np.float32)
                sig[end - len(c) : end] = c
                # SNR relative to the speech's loud part, not its silence.
                act = np.abs(c) > 0.05 * np.abs(c).max()
                level = rms(c[act]) if act.any() else rms(c)
                start = end - len(c)
                gap = int(rng.uniform(0.1, 0.6) * RATE)
                if self.split != "test" and start - gap > RATE // 4 and rng.random() < 0.4:
                    # someone was talking before the keyword (same or another speaker)
                    self.note(what="speech before the clip", gap_s=round(gap / RATE, 2))
                    pre, bits = self.conversation(rng, start - gap)
                    sig[: len(pre)] += pre * level * db(-6, 3, rng)
                    contains |= bits
            if rng.random() < p_clean:
                bg = bg * 0.003
                self.note(what="background", snr_db="clean (-50 dB)")
            else:
                snr = rng.uniform(*snr_db)
                bg = bg * (level / rms(bg) * 10 ** (-(snr - dist_db) / 20))
                self.note(what="background", snr_db_at_0_5m=round(snr, 1), snr_db_actual=round(snr - dist_db, 1))
            x = sig + bg
        lv = db(*LEVEL_DB, rng)
        self.note(what="mic level", dbfs=round(20 * np.log10(lv), 1))
        x = x * (lv / rms(x))
        np.clip(x, -1, 1, out=x)
        return (x if audio else features(x)), contains

    def stream(self, rng, n, snr_db=(0, 25)):
        """Continuous conversation over ambience (for false-alarm streams)."""
        sig, _ = self.conversation(rng, n)
        bg, _ = self.background(rng, n)
        x = sig + bg * (rms(sig) / rms(bg) * 10 ** (-rng.uniform(*snr_db) / 20))
        return x * (db(*LEVEL_DB, rng) / rms(x))
