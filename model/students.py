"""Stage A student networks (search_space.FAMILIES): chip features -> teacher embedding per step.

Every student takes (B, T, C) features at the front-end hop, stacks `stack` frames into one step
(the engine's output stride), runs causally, and returns (B, T // stack, D). `build()` solves the
width so the parameter count lands just under the trial's size constant.

Implemented: tcn, rnn (gru/lstm), ssm (mingru, lru), transformer, crnn, dnn, bcresnet.
Not yet: svdf, ssm s4d/mamba_lite (raise NotImplementedError).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _dilations(depth, pattern):
    return [{"exp": 2**i, "exp_cycle4": 2 ** (i % 4), "linear": i + 1, "none": 1}[pattern] for i in range(depth)]


class _Norm(nn.Module):
    def __init__(self, kind, n):
        super().__init__()
        self.f = {"bn": lambda: nn.BatchNorm1d(n), "none": nn.Identity}[kind]()

    def forward(self, h):  # (B, C, T)
        return self.f(h)


class _DW(nn.Module):
    """Causal depthwise conv; kernel 'mix357' splits channels across kernels 3, 5, 7 (MixConv)."""

    def __init__(self, width, kernel, dil, separable):
        super().__init__()
        ks = [3, 5, 7] if kernel == "mix357" else [kernel]
        splits = [width // len(ks)] * len(ks)
        splits[0] += width - sum(splits)
        self.splits, self.ks, self.dil = splits, ks, dil
        self.convs = nn.ModuleList([nn.Conv1d(c, c, k, dilation=dil, groups=c if separable else 1) for c, k in zip(splits, ks)])

    def forward(self, h):
        parts = torch.split(h, self.splits, 1)
        return torch.cat([c(F.pad(p, ((k - 1) * self.dil, 0))) for c, k, p in zip(self.convs, self.ks, parts)], 1)


class TCN(nn.Module):
    """Dilated causal conv stack. Block: h + relu(norm(pw(dw(h)))) (the 06_probes student when
    depth 8, kernel 3, exp_cycle4, separable, bn, expand 1); expand > 1 adds a pointwise
    bottleneck (width -> width*expand -> width)."""

    def __init__(self, n_in, D, width, depth=8, kernel=3, dilation="exp_cycle4", separable=True, gated=False,
                 norm="bn", expand=1):
        super().__init__()
        self.inp = nn.Linear(n_in, width)
        self.dil = _dilations(depth, dilation)
        self.dw = nn.ModuleList([_DW(width, kernel, d, separable) for d in self.dil])
        self.pw = nn.ModuleList([nn.Conv1d(width, width * expand * (2 if gated else 1), 1) for _ in self.dil])
        self.pw2 = nn.ModuleList([nn.Conv1d(width * expand, width, 1) if expand > 1 else nn.Identity() for _ in self.dil])
        self.norm = nn.ModuleList([_Norm(norm, width) for _ in self.dil])
        self.gated = gated
        self.out = nn.Linear(width, D)
        kmax = 7 if kernel == "mix357" else kernel
        self.rf_steps = 1 + sum((kmax - 1) * d for d in self.dil)

    def forward(self, x):
        h = self.inp(x).transpose(1, 2)
        for dw, pw, pw2, nm in zip(self.dw, self.pw, self.pw2, self.norm):
            u = pw(dw(h))
            if self.gated:
                a, b = u.chunk(2, 1)
                h = h + nm(pw2(torch.tanh(a) * torch.sigmoid(b)))
            elif isinstance(pw2, nn.Identity):
                h = h + F.relu(nm(u))
            else:
                h = h + nm(pw2(F.relu(u)))
        return self.out(h.transpose(1, 2))


class RNN(nn.Module):
    def __init__(self, n_in, D, width, cell="gru", layers=1, residual=False, proj=0):
        super().__init__()
        self.inp = nn.Linear(n_in, width)
        R = nn.GRU if cell == "gru" else nn.LSTM
        self.rnns = nn.ModuleList([R(width, width, batch_first=True) for _ in range(layers)])
        self.residual = residual
        self.out = nn.Linear(width, D)
        self.rf_steps = math.inf

    def forward(self, x):
        h = self.inp(x)
        for r in self.rnns:
            y, _ = r(h)
            h = h + y if self.residual else y
        return self.out(h)


class _MinGRU(nn.Module):
    """minGRU (Feng et al. 2024): gates from the input only, so training is a linear scan."""

    def __init__(self, width, expand=1):
        super().__init__()
        self.zh = nn.Linear(width, 2 * width * expand)
        self.o = nn.Linear(width * expand, width)

    def forward(self, x):
        z, hh = self.zh(x).chunk(2, -1)
        z = torch.sigmoid(z)
        h, out = torch.zeros_like(hh[:, 0]), []
        for t in range(x.shape[1]):
            h = (1 - z[:, t]) * h + z[:, t] * hh[:, t]
            out.append(h)
        return self.o(torch.stack(out, 1))


class _LRU(nn.Module):
    """Linear Recurrent Unit (Orvieto et al. 2023), diagonal complex state, real readout."""

    def __init__(self, width, state):
        super().__init__()
        r = torch.rand(state) * (0.999**2 - 0.9**2) + 0.9**2
        self.nu = nn.Parameter(torch.log(-0.5 * torch.log(r)))
        self.th = nn.Parameter(torch.rand(state) * 2 * math.pi)
        self.B = nn.Linear(width, 2 * state, bias=False)
        self.C = nn.Linear(2 * state, width)

    def forward(self, x):
        lam = torch.exp(-torch.exp(self.nu) + 1j * self.th)
        gamma = torch.sqrt(1 - lam.abs() ** 2)
        bu = self.B(x).float()
        u = torch.complex(*bu.chunk(2, -1)) * gamma
        h, out = torch.zeros_like(u[:, 0]), []
        for t in range(x.shape[1]):
            h = lam * h + u[:, t]
            out.append(h)
        h = torch.stack(out, 1)
        return self.C(torch.cat([h.real, h.imag], -1).to(x.dtype))


class SSM(nn.Module):
    def __init__(self, n_in, D, width, kind="mingru", layers=2, state=16, expand=1, conv_k=0):
        super().__init__()
        if kind not in ("mingru", "lru"):
            raise NotImplementedError(kind)
        self.inp = nn.Linear(n_in, width)
        self.conv = nn.Conv1d(width, width, conv_k, groups=width) if conv_k else None
        self.conv_k = conv_k
        self.blocks = nn.ModuleList([_MinGRU(width, expand) if kind == "mingru" else _LRU(width, state) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.out = nn.Linear(width, D)
        self.rf_steps = math.inf

    def forward(self, x):
        h = self.inp(x)
        if self.conv is not None:
            h = h + self.conv(F.pad(h.transpose(1, 2), (self.conv_k - 1, 0))).transpose(1, 2)
        for b, n in zip(self.blocks, self.norms):
            h = h + b(n(h))
        return self.out(h)


class _RMSNorm(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.g = nn.Parameter(torch.ones(n))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.g


class _Attn(nn.Module):
    """Causal attention over the last `window` steps, relative-position bias per head."""

    def __init__(self, width, heads, window, attn="softmax", pos="rel_bias"):
        super().__init__()
        self.h, self.w, self.attn, self.pos = heads, window, attn, pos
        self.qkv = nn.Linear(width, 3 * width)
        self.o = nn.Linear(width, width)
        self.rel = nn.Parameter(torch.zeros(heads, window)) if pos == "rel_bias" else None

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        d = torch.arange(T, device=x.device)
        dist = d[:, None] - d[None, :]
        ok = (dist >= 0) & (dist < self.w)
        s = q @ k.transpose(-1, -2) / math.sqrt(C // self.h)
        if self.rel is not None:
            s = s + self.rel[:, dist.clamp(0, self.w - 1)][None]
        if self.attn == "softmax":
            a = s.masked_fill(~ok, float("-inf")).softmax(-1)
        elif self.attn == "sigmoid":
            a = torch.sigmoid(s - math.log(self.w)) * ok
        else:  # relu_linear
            a = F.relu(s) * ok / self.w
        return self.o((a @ v).transpose(1, 2).reshape(B, T, C))


class Transformer(nn.Module):
    def __init__(self, n_in, D, width, layers=2, heads=2, window=16, ffn_mult=2, pos="rel_bias", attn="softmax",
                 norm="layernorm", stem="linear", conv_module=False):
        super().__init__()
        if pos not in ("rel_bias", "none"):
            raise NotImplementedError(pos)
        width = max(heads, width // heads * heads)
        self.inp = nn.Linear(n_in, width)
        self.stem = None if stem == "linear" else nn.ModuleList(
            [nn.Conv1d(width, width, 3, groups=width) for _ in range(1 if stem == "conv3" else 2)])
        mk = {"layernorm": lambda: nn.LayerNorm(width), "rmsnorm": lambda: _RMSNorm(width), "none": nn.Identity}[norm]
        self.n1 = nn.ModuleList([mk() for _ in range(layers)])
        self.n2 = nn.ModuleList([mk() for _ in range(layers)])
        self.att = nn.ModuleList([_Attn(width, heads, window, attn, pos) for _ in range(layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(width, width * ffn_mult), nn.GELU(), nn.Linear(width * ffn_mult, width))
                                  for _ in range(layers)])
        self.cm = nn.ModuleList([nn.Conv1d(width, width, 3, groups=width) for _ in range(layers)]) if conv_module else None
        self.out = nn.Linear(width, D)
        self.rf_steps = layers * (window - 1) + 1 + (0 if self.stem is None else 2 * len(self.stem))

    def forward(self, x):
        h = self.inp(x)
        if self.stem is not None:
            for c in self.stem:
                h = h + F.relu(c(F.pad(h.transpose(1, 2), (2, 0))).transpose(1, 2))
        for i, (n1, att, n2, ffn) in enumerate(zip(self.n1, self.att, self.n2, self.ffn)):
            h = h + att(n1(h))
            if self.cm is not None:
                h = h + self.cm[i](F.pad(h.transpose(1, 2), (2, 0))).transpose(1, 2)
            h = h + ffn(n2(h))
        return self.out(h)


class CRNN(nn.Module):
    def __init__(self, n_in, D, width, conv_layers=2, conv_kernel=3, back="gru", back_layers=1):
        super().__init__()
        self.conv = TCN(n_in, width, width, depth=conv_layers, kernel=conv_kernel, dilation="none")
        if back == "gru":
            self.back = RNN(width, D, width, "gru", back_layers)
        elif back == "lru":
            self.back = SSM(width, D, width, "lru", back_layers)
        else:
            self.back = Transformer(width, D, width, back_layers, heads=2, window=16)
        self.rf_steps = math.inf if back != "attention" else self.conv.rf_steps + self.back.rf_steps

    def forward(self, x):
        return self.back(self.conv(x))


class DNN(nn.Module):
    """Apple-2017-style: an MLP over the last `context` steps."""

    def __init__(self, n_in, D, width, context=16, depth=3, act="relu", bottleneck=0):
        super().__init__()
        A = {"relu": nn.ReLU, "gelu": nn.GELU, "sigmoid": nn.Sigmoid}[act]
        first = max(8, int(width * bottleneck)) if bottleneck else width
        layers = [nn.Linear(n_in * context, first), A()]
        prev = first
        for _ in range(depth - 1):
            layers += [nn.Linear(prev, width), A()]
            prev = width
        self.net = nn.Sequential(*layers, nn.Linear(prev, D))
        self.context = context
        self.rf_steps = context

    def forward(self, x):
        x = F.pad(x, (0, 0, self.context - 1, 0))
        return self.net(x.unfold(1, self.context, 1).flatten(2))


class _SubSpectralNorm(nn.Module):
    """BatchNorm per frequency sub-band group (BC-ResNet); plain BN if groups don't divide F."""

    def __init__(self, c, groups):
        super().__init__()
        self.g = groups
        self.bn = nn.BatchNorm2d(c * groups)
        self.bn1 = nn.BatchNorm2d(c)

    def forward(self, x):  # (B, C, F, T)
        B, C, Fq, T = x.shape
        if Fq % self.g:
            return self.bn1(x)
        return self.bn(x.reshape(B, C * self.g, Fq // self.g, T)).reshape(B, C, Fq, T)


class _BCBlock(nn.Module):
    """Broadcasted residual block: frequency-depthwise conv + SSN (2D part), then mean over
    frequency, causal temporal depthwise conv + pointwise (1D part), broadcast back."""

    def __init__(self, cin, cout, fstride, dil, groups):
        super().__init__()
        self.trans = nn.Sequential(nn.Conv2d(cin, cout, 1), nn.BatchNorm2d(cout), nn.ReLU()) if cin != cout else None
        self.f2 = nn.Conv2d(cout, cout, (3, 1), stride=(fstride, 1), padding=(1, 0), groups=cout)
        self.ssn = _SubSpectralNorm(cout, groups)
        self.t1 = nn.Conv1d(cout, cout, 3, dilation=dil, groups=cout)
        self.bn = nn.BatchNorm1d(cout)
        self.pw = nn.Conv1d(cout, cout, 1)
        self.dil, self.res = dil, fstride == 1

    def forward(self, x):
        if self.trans is not None:
            x = self.trans(x)
        y = self.ssn(self.f2(x))
        t = y.mean(2)
        t = self.pw(F.silu(self.bn(self.t1(F.pad(t, (2 * self.dil, 0))))))
        out = y + t[:, :, None]
        return F.relu(out + x) if self.res else F.relu(out)


class BCResNet(nn.Module):
    """BC-ResNet (Kim et al. 2021), causal in time. Runs on frames (the stacked steps are
    unstacked), reads out at each step's last frame. n_freq = front-end bands."""

    def __init__(self, n_in, D, width, n_freq=12, stages=3, blocks_per_stage=2, freq_stride=2, subspectral_groups=2):
        super().__init__()
        self.F, self.stack = n_freq, n_in // n_freq
        self.stem = nn.Conv2d(1, width, (5, 5), stride=(2, 1), padding=(2, 0))
        chans = [int(width * m) for m in (1, 1.5, 2, 2.5)[:stages]]
        blocks, c, dil = [], width, 1
        for s, co in enumerate(chans):
            for b in range(blocks_per_stage):
                blocks.append(_BCBlock(c, co, freq_stride if (b == 0 and s > 0) else 1, dil, subspectral_groups))
                c = co
                dil = min(8, dil * 2)  # time dilation per block, so the receptive field reaches the teacher's 0.78 s
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Linear(c, D)
        self.rf_steps = math.ceil((5 + sum(2 * blk.dil for blk in blocks)) / max(1, self.stack))

    def forward(self, x):  # (B, T', stack*F)
        B, Tp, _ = x.shape
        h = x.reshape(B, Tp, self.stack, self.F).reshape(B, Tp * self.stack, self.F).transpose(1, 2)[:, None]
        h = F.relu(self.stem(F.pad(h, (4, 0))))
        for blk in self.blocks:
            h = blk(h)
        h = h.mean(2)[:, :, self.stack - 1 :: self.stack]  # each step's last frame
        return self.out(h.transpose(1, 2))


FAMILY_CLASSES = {"tcn": TCN, "rnn": RNN, "ssm": SSM, "transformer": Transformer, "crnn": CRNN, "dnn": DNN,
                  "bcresnet": BCResNet}


class Student(nn.Module):
    """Frame stacking (engine stride) + the family body."""

    def __init__(self, body, stack):
        super().__init__()
        self.body, self.stack = body, stack

    def forward(self, x):  # (B, T, C) -> (B, T // stack, D); step i ends at frame (i+1)*stack-1
        B, T, C = x.shape
        s = self.stack
        return self.body(x[:, : T - T % s].reshape(B, T // s, s * C))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def macs_per_step(m, n_in, steps=100):
    """Multiply-accumulates per output step (one 8x8 MAC per clock on the chip). The flop counter
    sees fused RNN kernels only partly; models with a GRU/LSTM run ~1 MAC per weight, so they
    are counted as their parameter count."""
    from torch.utils.flop_counter import FlopCounterMode

    if any(isinstance(mod, (nn.GRU, nn.LSTM)) for mod in m.modules()):
        return n_params(m)
    m = m.cpu().eval()
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m(torch.zeros(1, steps * m.stack, n_in))
    return int(fc.get_total_flops() / 2 / steps)


def build(family, knobs, n_in, stack, D, size):
    """Largest width whose student has <= size params -> (Student, width)."""
    cls = FAMILY_CLASSES.get(family)
    if cls is None:
        raise NotImplementedError(family)
    if family == "bcresnet":
        knobs = {**knobs, "n_freq": n_in}

    def make(w):
        return Student(cls(n_in * stack, D, w, **knobs), stack)

    lo, hi = 4, 4
    while n_params(make(hi)) <= size and hi < 4096:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if n_params(make(mid)) <= size else (lo, mid)
    return make(lo), lo
