"""Phonetic near-misses for every candidate: phrases that sound almost, but not exactly, like it.

  "hey tux"  -> "hey ducks", "hey tax", "hey tusk", "say tux", ...
  "zorbex"   -> "zorbix", "sorbex", "zorbeg", ...   (not in CMUdict: spelling edits)

Real words: CMUdict neighbours at phoneme edit distance 1 (stress ignored). Distance 0
(homophones, e.g. "tucks") is excluded, because it would really be a positive. Neighbours
that people actually say (frequent in the conversation corpus) are preferred.
Words not in CMUdict (invented words) get consonant / vowel substitutions that change
the sound.

Output: data/text/nearmiss.json {slug: [phrase, ...]}; render with
  python model/synth_tts.py --nearmiss
"""

import json
import re
from collections import Counter
from pathlib import Path

import cmudict
import numpy as np

import store
from synth_tts import load_candidates, slug

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "text" / "nearmiss.json"
PER_CANDIDATE = 40
SIMILAR = ["kg", "td", "pb", "sz", "fv", "mn", "lr", "ck", "xs", "jd", "wv"]  # sound-changing swaps
VOWELS = "aeiou"


def phones(pron):
    return tuple(re.sub(r"\d", "", p) for p in pron)


def edit1(a, b):
    """True if phone sequences a and b differ by exactly one substitution, insertion or deletion."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return any(l[:i] + l[i + 1 :] == s for i in range(len(l)))


def spelling_edits(w, rng, n=12):
    out = set()
    for i, ch in enumerate(w):
        for pair in SIMILAR:
            if ch in pair:
                out.add(w[:i] + pair[1 - pair.index(ch)] + w[i + 1 :])
        if ch in VOWELS:
            out.update(w[:i] + v + w[i + 1 :] for v in VOWELS if v != ch)
    out.discard(w)
    out = sorted(out)
    return [out[i] for i in rng.permutation(len(out))[:n]]


def main():
    rng = np.random.default_rng(0)
    d = cmudict.dict()
    by_len = {}
    for w, prons in d.items():
        if w.isalpha() and len(w) > 1:
            for p in prons:
                by_len.setdefault(len(p), []).append((w, phones(p)))
    freq = Counter(re.findall(r"[a-z']+", " ".join(map(str, store.load(ROOT / "data/text/utts")["text"])).lower()))

    def neighbours(word):
        if word not in d:
            return spelling_edits(word, rng)
        found = dict_neighbours(word)
        if len(found) < 8:  # long words rarely have dictionary neighbours one sound away
            found += [w for w in spelling_edits(word, rng) if w not in d or w in found]
        return found

    def dict_neighbours(word):
        own = {phones(p) for p in d[word]}
        found = set()
        for p in own:
            for L in (len(p) - 1, len(p), len(p) + 1):
                for w, q in by_len.get(L, ()):
                    if w != word and q not in own and edit1(p, q):
                        found.add(w)
        # prefer words people actually say; keep a few rare ones for variety
        ranked = sorted(found, key=lambda w: -freq[w])
        return ranked[:12] + [ranked[i] for i in rng.permutation(len(ranked))[:4] if i >= 12]

    out = {}
    for c in load_candidates():
        words = c["phrase"].split()
        by_pos = []  # equal share per word position, so "hey" swaps don't crowd out the rest
        for i, w in enumerate(words):
            alts = sorted({" ".join(words[:i] + [nb] + words[i + 1 :]) for nb in neighbours(w)} - {c["phrase"]})
            by_pos.append([alts[j] for j in rng.permutation(len(alts))])
        picked, i = [], 0
        while len(picked) < PER_CANDIDATE and any(by_pos):
            if by_pos[i % len(by_pos)]:
                picked.append(by_pos[i % len(by_pos)].pop())
            i += 1
        alts = picked
        out[slug(c["phrase"])] = alts
        print(f"{c['phrase']:22s} {len(alts):4d}: {', '.join(out[slug(c['phrase'])][:6])}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
