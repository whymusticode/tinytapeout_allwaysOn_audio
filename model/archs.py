"""Streaming wake-word architectures + a hardware cost model (tiles, QSPI bandwidth, MACs).

Every model maps (B, T, 16) normalised features -> (B, T) per-frame logits and is causal:
the logit at frame t uses frames <= t only, exactly as the chip would compute it streaming.
Strided models (TCN, transformer) emit one output per `stride` frames; it is held for the
following frames (`hold()`), so the per-frame loss and the stream metrics treat all
architectures the same way.

Architectures (one width knob each; WIDTHS gives the 5-point ladder, ~4x parameters per step,
middle point ~16 tiles by the cost model below):
  gru          1-layer GRU(H) + linear. State: H values (+ an H-value update buffer).
  mingru       2-layer minGRU (Feng et al. 2024, "Were RNNs all we needed?"): gates depend on
               the input only, so training is a parallel scan (fast on GPU) while inference
               is still a tiny recurrence (state = H per layer, updated in place).
  tcn          stride 2, then 4 causal depthwise-separable conv blocks, kernel 3, dilation
               1/2/4/8 (receptive field 31 steps = 1 s), residual (TC-ResNet / microWakeWord
               family). State: the dilated-conv history buffers.
  transformer  stride 4 (64 ms tokens), 2 pre-LN layers, 2 heads, causal attention over the
               last 16 tokens (1 s) with a learned relative-position bias, FFN 2d. State: the
               key/value cache.

Cost model (see research/02_hardware_and_architectures): the chip streams weights from the
QSPI Pmod every output step; the state either lives in on-chip flip-flops (gru, mingru) or in
the Pmod's PSRAM (tcn history, transformer KV cache), in which case only a working set stays
on chip and the state traffic is added to the QSPI budget.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_IN = 16
FRAME_S = 0.016

# --- hardware cost model --------------------------------------------------------------
TILE_BITS = 320  # flip-flops per tile at a routable utilisation. Set on sky130; a gf180 tile (346x161 um) holds
#   about the same (gf cells ~3.2x sky130 area, tile 3.1x), so the 16-tile budget carries over to the gf180 4x4
FRONTEND_TILES = 5.0  # CIC + 16-band filterbank + VAD (filterbank alone: ~1,270 DFF + ~1,660 cells, yosys)
ENGINE_TILES = 2.0  # QSPI controller, 8x8 MAC + 24-bit accumulator, sequencer, activation LUTs (estimate)
QSPI_BYTES_PER_S = 16.384e6 / 2 * 4 / 8  # SCK = clk/2, 4 data lines: ~4.1 MB/s
MACS_PER_S = 16.384e6  # one MAC per clock
WEIGHT_BITS = 8


def hold(y, stride):
    """(B, n_steps) logits at stride -> (B, n_steps*stride) per-frame logits. Step j covers frames
    [j*s, j*s+s-1] and is known after frame j*s+s-1; it is held until the next step is ready."""
    if stride == 1:
        return y
    y = y.repeat_interleave(stride, dim=1)
    return F.pad(y[:, : y.shape[1] - (stride - 1)], (stride - 1, 0), value=-20.0)


def stack(x, stride):
    """(B, T, C) -> (B, ceil(T/s), s*C): s consecutive frames side by side (zero-padded at the end)."""
    if stride == 1:
        return x
    B, T, C = x.shape
    pad = (-T) % stride
    x = F.pad(x, (0, 0, 0, pad))
    return x.reshape(B, (T + pad) // stride, stride * C)


class GRUNet(nn.Module):
    arch, stride = "gru", 1

    def __init__(self, width):
        super().__init__()
        self.H = self.width = width
        self.rnn = nn.GRU(N_IN, width, batch_first=True)
        self.out = nn.Linear(width, 1)

    def forward(self, x):
        y, _ = self.rnn(x)
        return self.out(y).squeeze(-1)

    def state_bytes(self):
        return {"on_chip": 2 * self.H + N_IN, "external": 0, "external_rw_per_step": 0}


def _g(x):  # minGRU's positive "activation" for the candidate state (keeps the log-space scan real)
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


class MinGRULayer(nn.Module):
    def __init__(self, n_in, H):
        super().__init__()
        self.lin = nn.Linear(n_in, 2 * H)

    def forward(self, x):
        k, h = self.lin(x).float().chunk(2, dim=-1)
        # h_t = (1 - z_t) h_{t-1} + z_t g(h~_t), h_0 = 0, computed as a log-space parallel scan
        log_z, log_1mz = -F.softplus(-k), -F.softplus(k)
        log_h = torch.log(_g(h))
        a = torch.cumsum(log_1mz, dim=1)
        return torch.exp(a + torch.logcumsumexp(log_z + log_h - a, dim=1))


class MinGRUNet(nn.Module):
    """Input projection to H, then 2 residual minGRU blocks: x <- x + W_o minGRU(x).
    No per-frame normalisation: it would make the model level-invariant and amplify quiet
    background (tried: 55 false alarms/h on far-field meetings)."""

    arch, stride = "mingru", 1

    def __init__(self, width):
        super().__init__()
        self.H = self.width = width
        self.inp = nn.Linear(N_IN, width)
        self.cells = nn.ModuleList([MinGRULayer(width, width) for _ in range(2)])
        self.outs = nn.ModuleList([nn.Linear(width, width) for _ in range(2)])
        self.out = nn.Linear(width, 1)

    def forward(self, x):
        x = self.inp(x)
        for cell, o in zip(self.cells, self.outs):
            x = x + o(cell(x))
        return self.out(x).squeeze(-1)

    def state_bytes(self):
        return {"on_chip": 3 * self.H + N_IN, "external": 0, "external_rw_per_step": 0}


class TCNBlock(nn.Module):
    def __init__(self, C, dilation, k=3):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.dw = nn.Conv1d(C, C, k, dilation=dilation, groups=C)
        self.pw = nn.Conv1d(C, C, 1)
        self.bn = nn.BatchNorm1d(C)  # folds into pw at inference

    def forward(self, x):  # (B, C, T)
        return x + F.relu(self.bn(self.pw(self.dw(F.pad(x, (self.pad, 0))))))


class TCNNet(nn.Module):
    arch, stride = "tcn", 2
    DIL = (1, 2, 4, 8)

    def __init__(self, width):
        super().__init__()
        self.C = self.width = width
        self.inp = nn.Linear(N_IN * self.stride, width)
        self.blocks = nn.Sequential(*[TCNBlock(width, d) for d in self.DIL])
        self.out = nn.Linear(width, 1)

    def forward(self, x):
        T = x.shape[1]
        h = self.blocks(self.inp(stack(x, self.stride)).transpose(1, 2)).transpose(1, 2)
        return hold(self.out(h).squeeze(-1), self.stride)[:, :T]

    def state_bytes(self):
        C = self.C
        hist = sum(2 * d for d in self.DIL) * C  # (k-1)*d past steps per block
        return {"on_chip": N_IN * (self.stride - 1) + 5 * C,  # frame stack + taps/working activations
                "external": hist, "external_rw_per_step": 2 * len(self.DIL) * C + len(self.DIL) * C}


class TransformerLayer(nn.Module):
    def __init__(self, d, heads, window, ff=2):
        super().__init__()
        self.h, self.w = heads, window
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.ReLU(), nn.Linear(ff * d, d))
        self.rel = nn.Parameter(torch.zeros(heads, window))  # bias per head per look-back distance

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        dist = torch.arange(T, device=x.device)[:, None] - torch.arange(T, device=x.device)[None, :]
        ok = (dist >= 0) & (dist < self.w)
        bias = self.rel[:, dist.clamp(0, self.w - 1)].masked_fill(~ok, float("-inf"))  # (h, T, T)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype)[None])
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        return x + self.ff(self.ln2(x))


class TransformerNet(nn.Module):
    arch, stride = "transformer", 4
    LAYERS, HEADS, WINDOW = 2, 2, 16

    def __init__(self, width):
        super().__init__()
        self.d = self.width = width
        self.inp = nn.Linear(N_IN * self.stride, width)
        self.layers = nn.Sequential(*[TransformerLayer(width, self.HEADS, self.WINDOW) for _ in range(self.LAYERS)])
        self.ln = nn.LayerNorm(width)
        self.out = nn.Linear(width, 1)

    def forward(self, x):
        T = x.shape[1]
        h = self.ln(self.layers(self.inp(stack(x, self.stride))))
        return hold(self.out(h).squeeze(-1), self.stride)[:, :T]

    def state_bytes(self):
        d, L, W = self.d, self.LAYERS, self.WINDOW
        return {"on_chip": N_IN * (self.stride - 1) + 7 * d + W,  # frame stack, x, q/k/v, ffn, scores
                "external": 2 * W * d * L, "external_rw_per_step": (2 * W * d + 2 * d) * L}


class AppleDNN(nn.Module):
    """Apple's 2017 "Hey Siri" first-stage shape [research/SOA.md A1]: a 5-hidden-layer sigmoid DNN
    over the last 20 frames (0.32 s here; Apple: 20 x 10 ms), evaluated every frame. Apple
    predicts ~20 phonetic states and scores them with an HMM; here the output is the keyword
    logit directly (end-to-end, as in Apple's 2023 system). The 20-frame history sits in
    PSRAM (320 B read per frame); only the activations are on chip."""

    arch, stride, CONTEXT, LAYERS = "apple", 1, 20, 5

    def __init__(self, width):
        super().__init__()
        self.W = self.width = width
        dims = [N_IN * self.CONTEXT] + [width] * self.LAYERS
        self.hidden = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.out = nn.Linear(width, 1)

    def forward(self, x):
        T = x.shape[1]
        x = F.pad(x, (0, 0, self.CONTEXT - 1, 0))  # causal: frame t sees t-19 .. t
        h = x.unfold(1, self.CONTEXT, 1).transpose(2, 3).reshape(x.shape[0], T, -1)
        for lin in self.hidden:
            h = torch.sigmoid(lin(h))
        return self.out(h).squeeze(-1)

    def state_bytes(self):
        return {"on_chip": 2 * self.W + 2 * N_IN, "external": N_IN * self.CONTEXT,
                "external_rw_per_step": N_IN * (self.CONTEXT + 1)}


class SVDFLayer(nn.Module):
    """Rank-1 SVDF (Google, [research/SOA.md G2]): per node a feature filter (linear over the input)
    and a time filter over that node's last T feature-filter outputs, then ReLU."""

    def __init__(self, n_in, nodes, T):
        super().__init__()
        self.T = T
        self.feat = nn.Linear(n_in, nodes, bias=False)
        self.time = nn.Conv1d(nodes, nodes, T, groups=nodes)  # depthwise: one T-tap filter per node

    def forward(self, x):  # (B, T, C)
        f = self.feat(x).transpose(1, 2)
        return F.relu(self.time(F.pad(f, (self.T - 1, 0)))).transpose(1, 2)


class SVDFNet(nn.Module):
    """Google E2E streaming KWS shape [G2]: 4 SVDF layers (N nodes, T=8) and 3 SVDF layers
    (N/3 nodes, T=32), linear bottlenecks of N/3 between them, one inference per 2 frames
    (32 ms; Google: 20 ms). Google's E2E_40K is N=96. SVDF memories live in PSRAM."""

    arch, stride = "svdf", 2

    def __init__(self, width):
        super().__init__()
        self.N = self.width = width
        n2 = max(8, width // 3)
        layers, d = [], N_IN * self.stride
        for i in range(4):
            layers += [SVDFLayer(d, width, 8), nn.Linear(width, n2)]
            d = n2
        for i in range(3):
            layers += [SVDFLayer(d, n2, 32)]
            d = n2
        self.layers = nn.Sequential(*layers)
        self.n2 = n2
        self.out = nn.Linear(n2, 1)

    def forward(self, x):
        T = x.shape[1]
        h = self.layers(stack(x, self.stride))
        return hold(self.out(h).squeeze(-1), self.stride)[:, :T]

    def state_bytes(self):
        mem = 4 * 8 * self.N + 3 * 32 * self.n2
        return {"on_chip": N_IN * self.stride + 2 * self.N + self.n2, "external": mem,
                "external_rw_per_step": mem + 4 * self.N + 3 * self.n2}


ARCHS = {"gru": GRUNet, "mingru": MinGRUNet, "tcn": TCNNet, "transformer": TransformerNet,
         "apple": AppleDNN, "svdf": SVDFNet}
# 5 sizes per architecture, ~4x parameters apart, the middle one ~16 tiles (cost model below)
WIDTHS = {
    "gru": (43, 86, 172, 344, 688),
    "mingru": (30, 60, 116, 232, 464),
    "tcn": (16, 33, 66, 132, 264),
    "transformer": (10, 20, 40, 80, 160),
    "apple": (41, 82, 164, 328, 656),  # 41 ~ Apple's 5x32 first stage in weights (~14-20k)
    "svdf": (36, 72, 160, 320, 640),  # 96 = Google's E2E_40K
}


def build(arch, width):
    return ARCHS[arch](width)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def cost(m):
    """Hardware estimate for the chip: tiles, QSPI traffic and MACs per 16 ms frame."""
    s = m.state_bytes()
    p = n_params(m)
    step_s = FRAME_S * m.stride
    qspi = (p * WEIGHT_BITS / 8 + s["external_rw_per_step"]) / step_s  # bytes/s
    tiles = FRONTEND_TILES + ENGINE_TILES + s["on_chip"] * 8 / TILE_BITS
    return {
        "params": p,
        "state_on_chip_B": s["on_chip"],
        "state_psram_B": s["external"],
        "tiles": round(tiles, 1),
        "classifier_tiles": round(tiles - FRONTEND_TILES, 1),  # engine + on-chip state
        "qspi_kB_per_frame": round(qspi * FRAME_S / 1e3, 1),
        "qspi_load": round(qspi / QSPI_BYTES_PER_S, 2),  # > 1: doesn't fit at 8-bit weights
        "macs_per_frame": int(p / m.stride),
        "mac_load": round(p / step_s / MACS_PER_S, 2),
    }


if __name__ == "__main__":
    print(f"{'arch':12s} {'width':>5s} {'params':>8s} {'tiles':>6s} {'state chip B':>12s} {'psram B':>8s} {'QSPI kB/fr':>10s} {'QSPI load':>9s}")
    for a, ws in WIDTHS.items():
        for w in ws:
            m = build(a, w)
            c = cost(m)
            x = torch.randn(2, 100, N_IN)
            assert m(x).shape == (2, 100)
            print(f"{a:12s} {w:5d} {c['params']:8d} {c['tiles']:6.1f} {c['state_on_chip_B']:12d} {c['state_psram_B']:8d} "
                  f"{c['qspi_kB_per_frame']:10.1f} {c['qspi_load']:9.2f}")
