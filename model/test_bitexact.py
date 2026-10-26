"""Fast (features.py) vs golden (frontend.py) models must agree bit-for-bit.

Run: . ./env.sh && python -m pytest model/test_bitexact.py -q
"""

import numpy as np

import features
import frontend


def _signals(rng):
    n = frontend.HOP * 12 + 37  # partial last frame is dropped by both
    t = np.arange(n) / frontend.RATE
    yield "silence", np.zeros(n, np.int16)
    yield "noise", rng.integers(-32768, 32768, n).astype(np.int16)
    yield "quiet noise", rng.integers(-40, 41, n).astype(np.int16)
    for f in (120, 800, 3000, 7900):  # full-scale tones push y and acc into saturation
        yield f"tone {f}", np.round(32767 * np.sin(2 * np.pi * f * t)).astype(np.int16)
    yield "square", np.where((np.arange(n) // 40) % 2, 32767, -32768).astype(np.int16)
    yield "speech-like", features.pcm_from_float(0.05 * rng.standard_normal(n) * np.sin(2 * np.pi * 3 * t))


def test_filterbank_fast_matches_golden():
    rng = np.random.default_rng(1)
    sigs = list(_signals(rng))
    batch = features.filterbank(np.stack([s for _, s in sigs]))
    for i, (name, s) in enumerate(sigs):
        want = frontend.filterbank(s)
        assert want.shape == (12, frontend.N_BANDS)
        assert np.array_equal(features.filterbank(s), want), name
        assert np.array_equal(batch[i], want), name


def test_saturation_is_exercised():
    """The test signals must actually hit the saturation paths, or the test above proves little."""
    t = np.arange(frontend.HOP * 4) / frontend.RATE
    tone = np.round(32767 * np.sin(2 * np.pi * 3000 * t)).astype(np.int16)
    assert frontend.filterbank(tone).max() == frontend.FEAT_MAX


def test_log2_code():
    assert frontend.log2_code(0) == 0
    assert frontend.log2_code(1) == 0
    assert frontend.log2_code(0b1000) == 3 * 8
    assert frontend.log2_code(0b1111) == 3 * 8 + 7
    assert frontend.log2_code((1 << 24) - 1) == frontend.FEAT_MAX
