# %% [markdown]
"""
# Session 10 — The Training Loop (proxy transformer, EC2 g5/g6, native bf16)

This is one of **two** notebooks for this session (see `notebook_src_nanogpt.py` for the other).
This one reuses **Session 9's proxy transformer** (`V ≈ 10,000`, `D = 256`, 4 layers, 4 heads,
the course's own Session 2 BPE tokenizer) so the six items below run against a real model with
real text, and — because it is meant to run on a freshly-provisioned **Ampere/Ada EC2 box**
(`g5.xlarge` A10G or `g6.xlarge` L4, sm_86/sm_89) rather than the course's own T4 — this is the
notebook that produces the session's **honest, hardware-accelerated bf16** numbers. The T4 has
no native bf16 tensor cores; this box does.

The six items, each answered by a cell that actually ran:

1. Every tensor shape in one training step, named.
2. One gradient verified by hand (central-difference nudge vs `.grad`).
3. Gradient accumulation broken on purpose (average-of-averages vs total/total), single-step and
   as two training curves.
4. The grad norm, logged from step 1, with one **engineered** bad batch (per this session's
   decision — disclosed, not discovered) that spikes the norm before the loss visibly moves.
5. This run's own MFU, measured, with the peak measured by benchmark rather than quoted from a
   spec sheet.
6. `0.1` in fp32 / bf16 / fp8 E4M3, derived by hand and cross-checked against the bit patterns
   the hardware/library actually stores.

Every number below is collected into `results_proxy.json` at the end — `tools/build_readme.py`
reads it (namespaced `proxy.*`) alongside `results_nanogpt.json` to build the single comparison
`README.md`.
"""

# %%
# Bootstrap. No-op when the repo (or S9's assets, for local dev) is already checked out.
import pathlib, subprocess, sys, urllib.request

ASSETS = pathlib.Path("assets")
RAW = "https://raw.githubusercontent.com/vpw/era-v5-s9/main/assets/"
LOCAL_S9 = pathlib.Path("../../S9/assignment/assets")

for pkg in ("torch", "tokenizers", "matplotlib", "ml_dtypes"):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

ASSETS.mkdir(exist_ok=True)
for name in ("tokenizer.json", "corpus_en.txt"):
    dst = ASSETS / name
    if dst.exists():
        continue
    src = LOCAL_S9 / name
    if src.exists():
        dst.write_bytes(src.read_bytes())
    else:
        urllib.request.urlretrieve(RAW + name, dst)
print("assets:", sorted(p.name for p in ASSETS.iterdir()))

# %%
import json, math, struct, time
from dataclasses import dataclass

import ml_dtypes
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

torch.manual_seed(1337)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BF16_NATIVE = DEVICE.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
TRAIN_DTYPE = torch.bfloat16 if BF16_NATIVE else (torch.float16 if DEVICE.type == "cuda" else torch.float32)
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu"
SM = f"{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}" if DEVICE.type == "cuda" else "n/a"

print(f"torch {torch.__version__} · device {DEVICE} ({GPU_NAME}, sm_{SM})")
print(f"bf16 native (tensor-core accelerated): {BF16_NATIVE}")
print(f"training dtype for this run: {TRAIN_DTYPE}")
if DEVICE.type == "cuda" and not BF16_NATIVE:
    print("NOTE: this GPU predates Ampere — bf16 would be emulated here, not accelerated.")
    print("      Training/MFU below are measured in fp16, this box's native accelerated format.")

# %% [markdown]
"""
## §0 Configuration

Same proxy sizing as Session 9, for continuity of what "the model" means across sessions —
`V` from the tokenizer itself, not asserted.
"""

# %%
@dataclass
class Config:
    vocab_size: int = 10_000
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 4
    seq_len: int = 128
    batch_size: int = 8

IGNORE = -100
cfg = Config()
tok = Tokenizer.from_file(str(ASSETS / "tokenizer.json"))
cfg.vocab_size = tok.get_vocab_size()

print(f"V = {cfg.vocab_size:,}   D = {cfg.d_model}   layers = {cfg.n_layer}   heads = {cfg.n_head}")
print(f"T = {cfg.seq_len}   B = {cfg.batch_size}")

# %%
STREAM = torch.tensor(tok.encode((ASSETS / "corpus_en.txt").read_text(encoding="utf-8")).ids, dtype=torch.long)
print(f"corpus: {len(STREAM):,} tokens")


def get_batch(B=None, T=None, generator=None):
    """A batch of contiguous windows. Returns [B, T] token ids on DEVICE."""
    B, T = B or cfg.batch_size, T or cfg.seq_len
    ix = torch.randint(len(STREAM) - T - 1, (B,), generator=generator)
    return torch.stack([STREAM[i:i + T] for i in ix]).to(DEVICE)


_g = torch.Generator().manual_seed(0)
print("batch shape:", get_batch(generator=_g).shape)

# %% [markdown]
"""
## §0b The model

Session 9's block, unchanged: pre-norm residual stream, RMSNorm, RoPE, SwiGLU. The output head
is a separate module (`z = h · W_vocabᵀ`, `[V, D]`) since item 1 wants it named as its own tensor.
"""

# %%
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_cache(T, head_dim, device, base=10_000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(T, device=device).float(), inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(B, T, D))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(8 / 3 * cfg.d_model / 64 + 0.5) * 64
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.n2, self.ffn = RMSNorm(cfg.d_model), SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.ffn(self.n2(x))


class Trunk(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens):
        x = self.embed(tokens)
        cos, sin = rope_cache(tokens.shape[1], self.cfg.d_model // self.cfg.n_head, tokens.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.norm(x)


class Head(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(cfg.vocab_size, cfg.d_model))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, h):
        return F.linear(h, self.weight)


class Model(nn.Module):
    """tokens [B,T] -> logits [B,T,V], bundled so item 1/2 can nudge/shape one object."""

    def __init__(self, cfg):
        super().__init__()
        self.trunk, self.head = Trunk(cfg), Head(cfg)

    def forward(self, tokens):
        return self.head(self.trunk(tokens))


def build(cfg, seed=1337):
    torch.manual_seed(seed)
    return Model(cfg).to(DEVICE)


def loss_fn(model, tokens):
    logits = model(tokens)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, cfg.vocab_size),
        tokens[:, 1:].reshape(-1),
    )


model = build(cfg)
N_PARAMS = sum(p.numel() for p in model.parameters())
print(f"model: {N_PARAMS:,} parameters")

# %% [markdown]
"""
## 1. Every tensor shape in the step

One line per tensor: what it holds, and what each dimension means. Gradients get the same
treatment as activations — a gradient has the same shape as the parameter or activation it
belongs to, which is easy to state and easy to forget to check.
"""

# %%
tokens = get_batch()
logits = model(tokens)
logits.retain_grad()  # logits is not a leaf — without this its .grad is dropped after backward
loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size), tokens[:, 1:].reshape(-1))
loss.backward()

B, T = tokens.shape
V, D = cfg.vocab_size, cfg.d_model
# (slug, display name, shape, meaning) — the slug is the results.json key, so it must stay
# dot-free (the README template's placeholder resolver splits on ".").
shape_rows = [
    ("tokens", "tokens", tuple(tokens.shape), "B=batch of independent sequences, T=position in the sequence"),
    ("hidden", "hidden (trunk output)", (B, T, D), "B, T as above, D=residual-stream width (one vector per position)"),
    ("head_weight", "head.weight  (W_vocab)", tuple(model.head.weight.shape), "V=one row per vocabulary entry, D as above; z = h @ W_vocab^T"),
    ("logits", "logits", tuple(logits.shape), "B, T as above, V=one score per vocabulary entry"),
    ("logits_grad", "logits.grad", tuple(logits.grad.shape), "same shape as logits — one gradient per (batch, position, vocab-entry) score"),
    ("head_weight_grad", "head.weight.grad", tuple(model.head.weight.grad.shape), "same shape as head.weight — every row (every vocab entry) gets an update"),
    ("embed_weight_grad", "trunk.embed.weight.grad", tuple(model.trunk.embed.weight.grad.shape), "[V, D] — the input embedding table, gradient shaped like the table"),
    ("qkv_weight_grad", "trunk.blocks[0].attn.qkv.weight.grad", tuple(model.trunk.blocks[0].attn.qkv.weight.shape), "[3D, D] — packed Q,K,V projection, gradient matches the weight"),
]
for _, name, shape, meaning in shape_rows:
    print(f"  {name:<38} {str(shape):<18} {meaning}")

model.zero_grad(set_to_none=True)

# %% [markdown]
"""
## 2. Verify one gradient by hand

Two checks: the lesson's own two-weight scalar chain (`w1=3, x=2, w2=4, t=20`), which the lesson
says agrees to eight decimals — and, because that alone would only validate a toy example, a
central-difference nudge on one real weight of the model actually being trained.
"""

# %%
# 2a. The lesson's worked scalar chain.
w1 = torch.tensor(3.0, requires_grad=True)
w2 = torch.tensor(4.0, requires_grad=True)
x, t = torch.tensor(2.0), torch.tensor(20.0)

h = w1 * x
y = h * w2
L = (y - t) ** 2
L.backward()

# Analytic chain rule, one link at a time.
dL_dy = 2 * (y - t)
dL_dw2 = dL_dy * h
dL_dh = dL_dy * w2
dL_dw1 = dL_dh * x

print("2a. toy chain  w1=3, x=2, w2=4, t=20")
print(f"    forward: h={h.item()}, y={y.item()}, loss={L.item()}")
print(f"    dL/dy={dL_dy.item()}  dL/dw2={dL_dw2.item()}  dL/dh={dL_dh.item()}  dL/dw1={dL_dw1.item()}")
print(f"    autograd w1.grad={w1.grad.item()}  w2.grad={w2.grad.item()}")
print(f"    analytic == autograd (w1): {math.isclose(dL_dw1.item(), w1.grad.item(), abs_tol=1e-8)}")

eps = 1e-3
w1p = torch.tensor(3.0 + eps)
w1m = torch.tensor(3.0 - eps)
Lp = ((w1p * x) * w2 - t) ** 2
Lm = ((w1m * x) * w2 - t) ** 2
numeric = ((Lp - Lm) / (2 * eps)).item()
print(f"    central diff (eps={eps}): {numeric:.4f}  vs autograd {w1.grad.item():.4f}  "
      f"(agree to {abs(numeric - w1.grad.item()):.2e})")

# %%
# 2b. A real weight, nudged, in the model actually being trained.
torch.manual_seed(0)
probe_tokens = get_batch()
model.zero_grad(set_to_none=True)
L0 = loss_fn(model, probe_tokens)
L0.backward()

# Pick the weight with the LARGEST-magnitude gradient in the head, not an arbitrary one: a
# central difference on a near-zero gradient asks fp32 to resolve a change far below its own
# rounding floor at this loss's magnitude (catastrophic cancellation), which would make the
# comparison noise-dominated rather than a real check.
w = model.head.weight
flat_idx = w.grad.abs().argmax().item()
row, col = divmod(flat_idx, w.shape[1])
analytic_grad = w.grad[row, col].item()

eps2 = 1e-2  # larger than 2a's toy-chain eps: at this loss magnitude, 1e-3 sits near fp32's own
             # rounding floor and the finite difference becomes noise-dominated (cancellation)
with torch.no_grad():
    w[row, col] += eps2
Lp2 = loss_fn(model, probe_tokens).item()
with torch.no_grad():
    w[row, col] -= 2 * eps2
Lm2 = loss_fn(model, probe_tokens).item()
with torch.no_grad():
    w[row, col] += eps2  # restore

numeric_grad = (Lp2 - Lm2) / (2 * eps2)
print(f"2b. real weight head.weight[{row},{col}]  (eps={eps2})")
print(f"    L(w)      = {L0.item():.6f}")
print(f"    L(w+eps)  = {Lp2:.6f}")
print(f"    L(w-eps)  = {Lm2:.6f}")
print(f"    central-difference grad = {numeric_grad:.6f}")
print(f"    autograd .grad          = {analytic_grad:.6f}")
print(f"    absolute difference     = {abs(numeric_grad - analytic_grad):.2e}")

model.zero_grad(set_to_none=True)

# %% [markdown]
"""
## 3. Break gradient accumulation on purpose

First, the lesson's own static numbers, reproduced exactly as a sanity check on the arithmetic.
Then the same bug shown as two training curves: micro-batches of **different token counts**,
combined the wrong way (mean of per-micro-batch means) and the right way (total loss over total
tokens) — plus a control where the micro-batches are kept equal-length, to show why the bug
hides in casual testing.
"""

# %%
# 3a. Static replica of the lesson's numbers: (token_count, mean_loss) per micro-batch.
micro = [(4, 2.0), (4, 2.0), (2, 5.0)]
total_loss = sum(n * l for n, l in micro)
total_tokens = sum(n for n, _ in micro)
correct = total_loss / total_tokens
wrong = sum(l for _, l in micro) / len(micro)
pct_error = abs(wrong - correct) / correct * 100

print("3a. static replica of the lesson's numbers")
print(f"    micro-batches (tokens, mean_loss): {micro}")
print(f"    correct  = total_loss/total_tokens = {total_loss}/{total_tokens} = {correct:.4f}")
print(f"    wrong    = average of the means     = {wrong:.4f}")
print(f"    error    = {pct_error:.1f}%  (lesson: 2.6000 vs 3.0000 = 15.4%)")

# %%
def run_accumulation_variant(seed, mode, steps=120, micro_steps=4, T_options=(cfg.seq_len, cfg.seq_len // 2)):
    """mode: 'correct' (total loss/total tokens) or 'wrong' (average of per-micro-batch means).
    T_options with two different lengths makes the micro-batches unequal; a single-element
    T_options makes them equal (the control that hides the bug)."""
    m = build(cfg, seed=seed)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    curve = []
    g = torch.Generator().manual_seed(seed + 1)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        sum_loss_tokens, sum_tokens, mean_losses = 0.0, 0, []
        for micro_step in range(micro_steps):
            T = T_options[micro_step % len(T_options)]
            batch = get_batch(T=T, generator=g)
            logits = m(batch)
            per_tok = F.cross_entropy(
                logits[:, :-1].reshape(-1, cfg.vocab_size), batch[:, 1:].reshape(-1),
                reduction="none",
            )
            n_tok = per_tok.numel()
            micro_mean = per_tok.mean()
            mean_losses.append(micro_mean)
            if mode == "correct":
                (per_tok.sum() / micro_steps).backward()  # accumulate sum; normalise by total below
            else:
                (micro_mean / micro_steps).backward()  # average-of-averages, accumulated
            sum_loss_tokens += per_tok.sum().item()
            sum_tokens += n_tok
        if mode == "correct":
            # Gradients above were accumulated as sum(per_tok)/micro_steps; rescale so the
            # effective normaliser is the TOTAL token count, not micro_steps.
            scale = micro_steps / sum_tokens
            for p in m.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
        # Report the same true metric (total loss / total tokens) for both curves, so the plot
        # compares training quality, not bookkeeping.
        curve.append(sum_loss_tokens / sum_tokens)
        opt.step()
    return curve


curve_wrong = run_accumulation_variant(2024, "wrong")
curve_correct = run_accumulation_variant(2024, "correct")
curve_wrong_eq = run_accumulation_variant(2024, "wrong", T_options=(cfg.seq_len,))
curve_correct_eq = run_accumulation_variant(2024, "correct", T_options=(cfg.seq_len,))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(curve_wrong, label="average of averages (wrong)")
axes[0].plot(curve_correct, label="total loss / total tokens (correct)")
axes[0].set_title("unequal micro-batch token counts — bug visible")
axes[0].set_xlabel("step"); axes[0].set_ylabel("loss (total/total)"); axes[0].legend()

axes[1].plot(curve_wrong_eq, label="average of averages")
axes[1].plot(curve_correct_eq, label="total loss / total tokens")
axes[1].set_title("equal micro-batch token counts — bug hides")
axes[1].set_xlabel("step"); axes[1].legend()
fig.tight_layout()
fig.savefig("assets/proxy_accumulation_curves.png", dpi=120)
plt.close(fig)

gap_unequal = abs(curve_wrong[-1] - curve_correct[-1])
gap_equal = abs(curve_wrong_eq[-1] - curve_correct_eq[-1])
print(f"3b. final loss gap, unequal token counts: {gap_unequal:.4f}")
print(f"    final loss gap, equal token counts:   {gap_equal:.4f}  (near zero — this is why it hides)")
print("    saved assets/proxy_accumulation_curves.png")

# %% [markdown]
"""
## 4. Grad norm logged from step 1, with one engineered bad batch

Logged from the first step, not added in retrospectively. At a chosen step, one micro-batch's
gradient is **deliberately scaled up** (disclosed here, per this session's decision) to stand in
for an outlier batch — no gradient clipping is applied in this run, so the oversized update is
actually taken, which is what lets its damage show up in the loss a step or two later rather
than in the same step's own loss (that loss was computed on the pre-update weights).

**This one demonstration cell uses plain SGD, not AdamW.** Adam's per-parameter update is
normalised by a running estimate of gradient magnitude, so a single spiked gradient mostly
pollutes that running estimate rather than producing an oversized *update* — which would mask
exactly the raw-gradient-to-update relationship this item is about. Plain SGD (`update = -lr ×
grad`) keeps that relationship direct, elsewhere in this notebook AdamW is used as normal.
"""

# %%
BAD_STEP = 30
SCALE = 50.0
NORM_STEPS = 60

norm_model = build(cfg, seed=7)
opt = torch.optim.SGD(norm_model.parameters(), lr=1.0)
g = torch.Generator().manual_seed(7)

norms, losses = [], []
for step in range(NORM_STEPS):
    batch = get_batch(generator=g)
    opt.zero_grad(set_to_none=True)
    L = loss_fn(norm_model, batch)
    L.backward()
    if step == BAD_STEP:
        for p in norm_model.parameters():
            if p.grad is not None:
                p.grad.mul_(SCALE)
    total_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in norm_model.parameters() if p.grad is not None)).item()
    norms.append(total_norm)
    losses.append(L.item())
    opt.step()

# Where did the loss visibly crack relative to the smooth trend before the spike?
baseline = sum(losses[max(0, BAD_STEP - 5):BAD_STEP]) / 5
crack_step = None
for s in range(BAD_STEP + 1, min(BAD_STEP + 6, NORM_STEPS)):
    if losses[s] > baseline * 1.15:
        crack_step = s
        break

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].plot(norms); axes[0].axvline(BAD_STEP, color="r", ls="--", label=f"engineered step {BAD_STEP}")
axes[0].set_ylabel("grad norm"); axes[0].legend()
axes[1].plot(losses); axes[1].axvline(BAD_STEP, color="r", ls="--")
if crack_step is not None:
    axes[1].axvline(crack_step, color="orange", ls="--", label=f"loss cracks step {crack_step}")
    axes[1].legend()
axes[1].set_ylabel("loss"); axes[1].set_xlabel("step")
fig.tight_layout()
fig.savefig("assets/proxy_norm_before_loss.png", dpi=120)
plt.close(fig)

print(f"4. engineered spike at step {BAD_STEP}: grad norm {norms[BAD_STEP - 1]:.3f} -> {norms[BAD_STEP]:.3f}")
print(f"   loss at step {BAD_STEP}: {losses[BAD_STEP]:.4f}  (baseline {baseline:.4f})")
print(f"   loss visibly cracks at step: {crack_step}  "
      f"(lag = {crack_step - BAD_STEP if crack_step is not None else 'n/a'} steps)")
print("   saved assets/proxy_norm_before_loss.png")

# %% [markdown]
"""
## 5. This run's own MFU

`MFU = 6 * N * tokens_per_second / peak_FLOP_s`. Tokens/s is measured by timing real training
steps in this device's native accelerated dtype (see the device banner at the top — bf16 if this
box is Ampere/Ada, fp16 if it turned out to be a T4, fp32 on CPU). The peak is **measured**, not
quoted from a spec sheet: a large matmul benchmarked in the same dtype, on the same device.
"""

# %%
def measure_peak_flops(dtype, device, size=4096, iters=20):
    a = torch.randn(size, size, device=device, dtype=dtype if device.type == "cuda" else torch.float32)
    b = torch.randn(size, size, device=device, dtype=dtype if device.type == "cuda" else torch.float32)
    for _ in range(3):
        a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    flops_per_matmul = 2 * size ** 3
    return flops_per_matmul * iters / dt


PEAK_MATMUL_SIZE = 4096
peak_flops = measure_peak_flops(TRAIN_DTYPE, DEVICE, size=PEAK_MATMUL_SIZE)
print(f"5a. measured peak: {peak_flops / 1e12:.2f} TFLOP/s  "
      f"({PEAK_MATMUL_SIZE}x{PEAK_MATMUL_SIZE} matmul, {TRAIN_DTYPE}, {DEVICE})")

# %%
mfu_model = build(cfg, seed=11)
opt = torch.optim.AdamW(mfu_model.parameters(), lr=3e-4)
g = torch.Generator().manual_seed(11)
WARMUP, TIMED = 10, 40

use_autocast = DEVICE.type == "cuda"
for step in range(WARMUP):
    batch = get_batch(generator=g)
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=DEVICE.type, dtype=TRAIN_DTYPE, enabled=use_autocast):
        L = loss_fn(mfu_model, batch)
    L.backward()
    opt.step()

if DEVICE.type == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
tokens_seen = 0
for step in range(TIMED):
    batch = get_batch(generator=g)
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=DEVICE.type, dtype=TRAIN_DTYPE, enabled=use_autocast):
        L = loss_fn(mfu_model, batch)
    L.backward()
    opt.step()
    tokens_seen += batch.numel()
if DEVICE.type == "cuda":
    torch.cuda.synchronize()
dt = time.perf_counter() - t0

tokens_per_sec = tokens_seen / dt
achieved_flops = 6 * N_PARAMS * tokens_per_sec
mfu_pct = achieved_flops / peak_flops * 100

print(f"5b. N = {N_PARAMS:,} params")
print(f"    measured tokens/s = {tokens_per_sec:,.0f}  ({TIMED} steps, B={cfg.batch_size}, T={cfg.seq_len}, dtype={TRAIN_DTYPE})")
print(f"    achieved FLOP/s   = {achieved_flops / 1e12:.2f} TFLOP/s")
print(f"    peak FLOP/s       = {peak_flops / 1e12:.2f} TFLOP/s  "
      f"(measured: {PEAK_MATMUL_SIZE}x{PEAK_MATMUL_SIZE} matmul benchmark, same device/dtype)")
print(f"    MFU               = {mfu_pct:.2f}%")
print(f"    device            = {GPU_NAME} (sm_{SM}), bf16-native={BF16_NATIVE}")

# %% [markdown]
"""
## 6. `0.1` in fp32, bf16 and fp8 E4M3 — by hand, showing the bits

`0.1` in binary is the repeating fraction `0.0(0011)` — it cannot be represented exactly in any
binary floating-point format. Below: the longhand derivation (sign / biased exponent / mantissa)
at each format's mantissa width, the value each format actually stores, and the representation
error — cross-checked against `struct.pack` (fp32) and `ml_dtypes` (bf16, fp8 E4M3).
"""

# %%
def decode_bits(sign, exponent_bits, mantissa_bits, bias):
    """Longhand IEEE-754-style decode from raw bit-fields -> (stored_value, breakdown)."""
    exponent = exponent_bits - bias
    mantissa_value = 1.0
    for i, bit in enumerate(mantissa_bits):
        if bit:
            mantissa_value += 2 ** -(i + 1)
    value = (-1) ** sign * mantissa_value * 2 ** exponent
    return value, exponent, mantissa_value


def bits_of(x: int, n: int):
    return [(x >> (n - 1 - i)) & 1 for i in range(n)]


TARGET = 0.1

print("0.1 in binary (repeating): 0.0001100110011001100110011... = 1.10011001100...(2) x 2^-4")
print()

# fp32 — cross-check against struct.pack, the ground truth for this format.
raw32 = struct.unpack(">I", struct.pack(">f", TARGET))[0]
s32 = raw32 >> 31
e32 = (raw32 >> 23) & 0xFF
m32 = bits_of(raw32 & 0x7FFFFF, 23)
val32, exp32, mant32 = decode_bits(s32, e32, m32, bias=127)
print(f"fp32  (1+8+23): sign={s32}  exponent={e32} (biased, unbiased={exp32})  "
      f"mantissa={''.join(map(str, m32))}")
print(f"      stored value = {val32:.10f}   struct.pack round-trip = {struct.unpack('>f', struct.pack('>f', TARGET))[0]:.10f}")
print(f"      representation error = {val32 - TARGET:.3e}")
print()

# bf16 — cross-check against ml_dtypes.
bf16_val = ml_dtypes.bfloat16(TARGET)
raw16 = np.array([bf16_val]).view(np.uint16)[0]
s16 = int(raw16) >> 15
e16 = (int(raw16) >> 7) & 0xFF
m16 = bits_of(int(raw16) & 0x7F, 7)
val16, exp16, mant16 = decode_bits(s16, e16, m16, bias=127)
print(f"bf16  (1+8+7):  sign={s16}  exponent={e16} (biased, unbiased={exp16})  "
      f"mantissa={''.join(map(str, m16))}")
print(f"      stored value (hand-decoded) = {val16:.10f}   ml_dtypes stores = {float(bf16_val):.10f}")
print(f"      representation error = {val16 - TARGET:.3e}   "
      f"(hand decode matches ml_dtypes: {math.isclose(val16, float(bf16_val), rel_tol=1e-9)})")
print()

# fp8 E4M3 (OCP e4m3fn: 1 sign + 4 exponent + 3 mantissa, bias 7) — cross-check against ml_dtypes.
fp8_val = ml_dtypes.float8_e4m3(TARGET) if hasattr(ml_dtypes, "float8_e4m3") else ml_dtypes.float8_e4m3fn(TARGET)
raw8 = np.array([fp8_val]).view(np.uint8)[0]
s8 = int(raw8) >> 7
e8 = (int(raw8) >> 3) & 0xF
m8 = bits_of(int(raw8) & 0x7, 3)
val8, exp8, mant8 = decode_bits(s8, e8, m8, bias=7)
print(f"fp8 E4M3 (1+4+3): sign={s8}  exponent={e8} (biased, unbiased={exp8})  "
      f"mantissa={''.join(map(str, m8))}")
print(f"      stored value (hand-decoded) = {val8:.6f}   ml_dtypes stores = {float(fp8_val):.6f}")
print(f"      representation error = {val8 - TARGET:.3e}   "
      f"(hand decode matches ml_dtypes: {math.isclose(val8, float(fp8_val), rel_tol=1e-6)})")
print()
print("6b. which one would you train in, and why")
print(f"    fp32 error {abs(val32 - TARGET):.2e} < bf16 error {abs(val16 - TARGET):.2e} < fp8 error {abs(val8 - TARGET):.2e}")
print("    bf16 keeps fp32's full 8-bit exponent range (same dynamic range, no loss-scaling knob)")
print("    and pays for it in mantissa precision only — the §10 trade this session names as the")
print("    reason bf16 replaced fp16. fp8 E4M3 is named here as the production recipe it now is,")
print("    not as what this run trains in.")

# %% [markdown]
"""
## Results summary

Written to `results_proxy.json`. `tools/build_readme.py` reads this file (namespaced `proxy.*`)
together with `results_nanogpt.json` to build the single comparison `README.md`.
"""

# %%
results = {
    "config": {
        "V": cfg.vocab_size, "D": cfg.d_model, "n_layer": cfg.n_layer, "n_head": cfg.n_head,
        "T": cfg.seq_len, "B": cfg.batch_size, "N_params": N_PARAMS,
        "tokenizer": "ERA V5 Session 2 BPE (10,000 merges, en/hi/te/mr)",
        "device": str(DEVICE), "gpu_name": GPU_NAME, "sm": SM,
        "bf16_native": BF16_NATIVE, "train_dtype": str(TRAIN_DTYPE), "torch": torch.__version__,
    },
    "1_shapes": {slug: str(shape) for slug, _, shape, _ in shape_rows},
    "2_gradient_check": {
        "toy_chain": {"w1": w1.item(), "w2": w2.item(), "loss": L.item(),
                      "dL_dw1_analytic": dL_dw1.item(), "dL_dw1_autograd": w1.grad.item(),
                      "dL_dw1_central_diff": numeric},
        "real_weight": {"row": row, "col": col, "eps": eps2,
                        "central_diff": numeric_grad, "autograd": analytic_grad,
                        "abs_diff": abs(numeric_grad - analytic_grad)},
    },
    "3_accumulation": {
        "static": {"micro_batches": micro, "correct": correct, "wrong": wrong, "pct_error": pct_error},
        "curves": {"final_gap_unequal_tokens": gap_unequal, "final_gap_equal_tokens": gap_equal,
                   "steps": len(curve_wrong)},
    },
    "4_norm_before_loss": {
        "bad_step": BAD_STEP, "scale": SCALE,
        "norm_before": norms[BAD_STEP - 1], "norm_at": norms[BAD_STEP],
        "loss_baseline": baseline, "loss_at_bad_step": losses[BAD_STEP],
        "crack_step": crack_step, "lag_steps": (crack_step - BAD_STEP) if crack_step is not None else None,
    },
    "5_mfu": {
        "N": N_PARAMS, "tokens_per_sec": tokens_per_sec, "achieved_flops": achieved_flops,
        "peak_flops": peak_flops, "peak_source": "measured: 4096x4096 matmul benchmark, same device/dtype",
        "mfu_pct": mfu_pct, "dtype": str(TRAIN_DTYPE),
    },
    "6_bits": {
        "fp32": {"sign": s32, "exponent_biased": int(e32), "mantissa": "".join(map(str, m32)),
                 "value": val32, "error": val32 - TARGET},
        "bf16": {"sign": s16, "exponent_biased": int(e16), "mantissa": "".join(map(str, m16)),
                 "value": val16, "error": val16 - TARGET},
        "fp8_e4m3": {"sign": s8, "exponent_biased": int(e8), "mantissa": "".join(map(str, m8)),
                     "value": val8, "error": val8 - TARGET},
    },
}

with open("results_proxy.json", "w") as f:
    json.dump(results, f, indent=2)

print("wrote results_proxy.json")
print(json.dumps({k: results[k] for k in ("config", "5_mfu")}, indent=2))
