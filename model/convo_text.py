"""Build the utterance corpus for the conversation simulator.

Sources (text only; no real speech audio is used for training):
  switchboard   real telephone conversations (transcripts), weight 0.5
  opensubtitles movie/TV dialogue, weight 0.35
  dailydialog   written everyday dialogues, weight 0.15
Sampling utterances from transcribed and scripted speech makes word frequencies follow
what people actually say, unlike read text.

Each utterance carries:
  split     train / test: switchboard by conversation, the others by text hash
  contains  which candidate phrases occur in it (word boundaries); training never
            uses an utterance as a negative for a candidate it contains

Output: data/text/utts/ (store.py format): text (N,) str, source (N,) uint8,
        split (N,) uint8 (0 train, 1 test), contains (N, ceil(K/8)) uint8 packed bits

Usage: python model/convo_text.py [--n 400000]
"""

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np

import store
from synth_tts import load_candidates

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "text"
OUT = ROOT / "data" / "text" / "utts"
SOURCES = ("switchboard", "opensubtitles", "dailydialog")
WEIGHTS = (0.5, 0.35, 0.15)
MIN_WORDS, MAX_WORDS = 1, 30


def clean(s):
    s = re.sub(r"\[[^\]]*\]|\([^)]*\)|<[^>]*>|\{[^}]*\}", " ", s)  # [noise], (sighs), <i>, {tags}
    s = s.replace("_", " ").replace('"', "").replace("''", "")
    s = re.sub(r"(^|\s)-+\s*", " ", s)  # subtitle dialogue dashes
    s = re.sub(r"\s+", " ", s).strip()
    words = s.split()
    if not (MIN_WORDS <= len(words) <= MAX_WORDS) or not re.search(r"[A-Za-z]", s):
        return None
    if re.search(r"[^A-Za-z0-9 ,.?!'$%&:;-]", s):  # drop odd symbols / non-English
        return None
    return s


def switchboard():
    """-> [(text, conversation id)]; partial words like 'th[e]-' are dropped."""
    out = []
    for f in sorted((RAW / "switchboard").glob("swb_ms98_transcriptions/*/*/*-trans.text")):
        conv = f.name[2:6]
        for line in f.read_text(errors="ignore").splitlines():
            parts = line.split(maxsplit=3)
            if len(parts) < 4:
                continue
            words = [w for w in parts[3].split() if not ("[" in w and "-" in w) and not w.endswith("-")]
            out.append((" ".join(words), conv))
    return out


def opensubtitles(n, rng):
    """n lines sampled uniformly from the downloaded part of the dump."""
    lines = (RAW / "opensubtitles" / "en.txt").read_text(errors="ignore").splitlines()
    idx = rng.choice(len(lines), size=min(len(lines), n * 3), replace=False)
    return [(lines[i], None) for i in idx]


def dailydialog():
    out = []
    for f in sorted((RAW / "dailydialog").glob("*/dialogues_*.txt")):
        if "_act_" in f.name or "_emotion_" in f.name:
            continue
        for line in f.read_text(errors="ignore").splitlines():
            out += [(u.strip(), None) for u in line.split("__eou__") if u.strip()]
    return out


def is_test(text, group):
    key = group if group is not None else text
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 5 == 0


def candidate_patterns():
    return [re.compile(r"\b" + r"\W+".join(map(re.escape, c["phrase"].split())) + r"\b", re.I)
            for c in load_candidates()]


def tag_contains():
    """Recompute only the `contains` bits (after the candidate list changes); text is untouched,
    so rendered audio that indexes into it stays valid."""
    d = store.load(OUT)
    pats = candidate_patterns()
    contains = np.packbits(np.array([[bool(p.search(str(t))) for p in pats] for t in d["text"]], bool), axis=1)
    store.save(OUT, contains=contains)
    return contains, pats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400_000, help="total utterances")
    ap.add_argument("--contains-only", action="store_true", help="only re-tag candidates (keeps the text)")
    args = ap.parse_args()
    if args.contains_only:
        contains, pats = tag_contains()
        k = np.unpackbits(contains, axis=1)[:, : len(pats)].sum(0).astype(np.int64)
        cands = load_candidates()
        print(", ".join(f"{cands[i]['phrase']} {k[i]}" for i in np.argsort(-k)))
        return
    rng = np.random.default_rng(0)
    pools = {"switchboard": switchboard(), "opensubtitles": opensubtitles(args.n, rng), "dailydialog": dailydialog()}
    pats = candidate_patterns()
    text, source, split, contains = [], [], [], []
    seen = set()
    for si, (name, w) in enumerate(zip(SOURCES, WEIGHTS)):
        items = pools[name]
        want = int(args.n * w)
        kept = 0
        for i in rng.permutation(len(items)):
            t = clean(items[i][0])
            if t is None or t.lower() in seen:
                continue
            seen.add(t.lower())
            text.append(t)
            source.append(si)
            split.append(int(is_test(t, items[i][1])))
            contains.append([bool(p.search(t)) for p in pats])
            kept += 1
            if kept >= want:
                break
        print(f"{name}: {kept} utterances (of {len(items)} lines)")
    contains = np.packbits(np.array(contains, bool), axis=1)
    store.save(OUT, text=np.array(text), source=np.array(source, np.uint8),
               split=np.array(split, np.uint8), contains=contains)
    k = np.unpackbits(contains, axis=1)[:, : len(pats)].sum(0).astype(np.int64)
    cands = load_candidates()
    top = np.argsort(-k)[:12]
    print("utterances containing a candidate:", ", ".join(f"{cands[i]['phrase']} {k[i]}" for i in top))
    n_words = np.array([len(t.split()) for t in text])
    print(f"{len(text)} utterances, mean {n_words.mean():.1f} words, test {np.mean(split):.0%}")


if __name__ == "__main__":
    main()
