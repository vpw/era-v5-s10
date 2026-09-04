# %% [markdown]
"""
# Session 10 — The Training Loop (nanoGPT, free-tier Colab)

This is one of **two** notebooks for this session (see `notebook_src_proxy.py` for the other,
which reuses Session 9's proxy transformer on a freshly-provisioned bf16-native EC2 box). This
one is the model the instructor named directly for students doing the assignment solo — Andrej
Karpathy's **nanoGPT** — trained here on nanoGPT's own char-level tiny-Shakespeare dataset, and
meant to run on **free-tier Google Colab**.

Free Colab's GPU picker only offers a **T4** (sm_75) — same as the course's own EC2 box, and
with the same limitation: **no native bf16 tensor cores**. So this track's throughput and MFU
are measured and reported in **fp16**, the T4's native accelerated format; the EC2/proxy track
carries this session's honest bf16 number. Comparing the two is the point.

Same six items, same discipline — every number below is collected into `results_nanogpt.json`:

1. Every tensor shape in one training step, named.
2. One gradient verified by hand.
3. Gradient accumulation broken on purpose, single-step and as two training curves.
4. The grad norm, logged from step 1, with one **engineered** bad batch.
5. This run's own MFU, measured — fp16 on a T4, honestly labelled as such.
6. `0.1` in fp32 / bf16 / fp8 E4M3, derived by hand and cross-checked against the bits.

**To run this on Colab:** open it from GitHub (`File > Open notebook > GitHub`, or the raw
`colab.research.google.com/github/<user>/<repo>/blob/<branch>/...ipynb` URL), Runtime > Change
runtime type > **T4 GPU**, Run all, then File > Download > **Download .ipynb** and drop it back
into this repo. `scripts/extract_results.py` (from the `era-v5-gpu-run` skill) recovers
`results_nanogpt.json` from the downloaded notebook's own printed output.
"""

# %%
## 0. Environment
import os, sys, subprocess, pathlib, urllib.request

IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")


def download_if_colab(path):
    """Colab's local disk is ephemeral — force each plot into the browser's Downloads
    folder as it's produced, rather than relying on it surviving until the run ends."""
    if IN_COLAB:
        try:
            from google.colab import files
            files.download(path)
        except Exception as e:
            print(f"(download skipped for {path}: {e})")


for pkg in ("torch", "matplotlib", "ml_dtypes", "numpy"):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

ASSETS = pathlib.Path("assets")
ASSETS.mkdir(exist_ok=True)
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = ASSETS / "tinyshakespeare.txt"
if not DATA_PATH.exists():
    urllib.request.urlretrieve(SHAKESPEARE_URL, DATA_PATH)
print("assets:", sorted(p.name for p in ASSETS.iterdir()))

# %%
import json, math, struct, time
from dataclasses import dataclass

import ml_dtypes
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BF16_NATIVE = DEVICE.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
TRAIN_DTYPE = torch.bfloat16 if BF16_NATIVE else (torch.float16 if DEVICE.type == "cuda" else torch.float32)
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu"
SM = f"{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}" if DEVICE.type == "cuda" else "n/a"

print(f"in_colab={IN_COLAB}  torch {torch.__version__}  device {DEVICE} ({GPU_NAME}, sm_{SM})")
print(f"bf16 native (tensor-core accelerated): {BF16_NATIVE}")
print(f"training dtype for this run: {TRAIN_DTYPE}")
if DEVICE.type == "cuda" and not BF16_NATIVE:
    print("NOTE: T4 (or older) — bf16 would be emulated here, not accelerated.")
    print("      Training/MFU below are measured in fp16, this GPU's native accelerated format.")

# %% [markdown]
"""
## §0 Configuration and data

Char-level tiny-Shakespeare, nanoGPT's own reference dataset for the small `shakespeare_char`
config. Vocabulary is just the distinct characters in the text — no BPE, no external tokenizer,
which is exactly why this model is the one that is quick to reason about end to end.
"""

# %%
@dataclass
class GPTConfig:
    vocab_size: int = 65   # set below from the data itself
    n_embd: int = 128
    n_layer: int = 4
    n_head: int = 4
    seq_len: int = 128     # block_size, in nanoGPT's naming
    batch_size: int = 8
    dropout: float = 0.0

text = DATA_PATH.read_text(encoding="utf-8")
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

cfg = GPTConfig(vocab_size=len(chars))
STREAM = torch.tensor([stoi[c] for c in text], dtype=torch.long)

print(f"corpus: {len(text):,} chars, vocab V={cfg.vocab_size}")
print(f"C (n_embd) = {cfg.n_embd}   layers = {cfg.n_layer}   heads = {cfg.n_head}")
print(f"T = {cfg.seq_len}   B = {cfg.batch_size}")


def get_batch(B=None, T=None, generator=None):
    """A batch of contiguous windows. Returns [B, T] char ids on DEVICE."""
    B, T = B or cfg.batch_size, T or cfg.seq_len
    ix = torch.randint(len(STREAM) - T - 1, (B,), generator=generator)
    return torch.stack([STREAM[i:i + T] for i in ix]).to(DEVICE)


_g = torch.Generator().manual_seed(0)
b0 = get_batch(generator=_g)
print("batch shape:", b0.shape, " sample decode:", "".join(itos[i] for i in b0[0, :40].tolist()))

# %% [markdown]
"""
## §0b The model — nanoGPT

The reference GPT-2-style block: **LayerNorm** (not RMSNorm), **learned positional embeddings**
(not RoPE), **GELU MLP** (not SwiGLU), causal self-attention, weights tied between the token
embedding and the output head (`lm_head.weight is wte.weight`) — nanoGPT's own choice, and a
second thing item 1 gets to show: a shared tensor, not a coincidentally-equal one.
"""

# %%
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(out.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.seq_len, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying, nanoGPT's own convention
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.wte(tokens) + self.wpe(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)


def build(cfg, seed=1337):
    torch.manual_seed(seed)
    return GPT(cfg).to(DEVICE)


def loss_fn(model, tokens):
    logits = model(tokens)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, cfg.vocab_size),
        tokens[:, 1:].reshape(-1),
    )


model = build(cfg)
N_PARAMS = sum(p.numel() for p in model.parameters())
print(f"model: {N_PARAMS:,} parameters (weight-tied: wte/lm_head share {model.wte.weight.numel():,})")

# %% [markdown]
"""
## 1. Every tensor shape in the step
"""

# %%
tokens = get_batch()
logits = model(tokens)
logits.retain_grad()  # logits is not a leaf — without this its .grad is dropped after backward
loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size), tokens[:, 1:].reshape(-1))
loss.backward()

B, T = tokens.shape
V, C = cfg.vocab_size, cfg.n_embd
# (slug, display name, shape, meaning) — the slug is the results.json key, so it must stay
# dot-free (the README template's placeholder resolver splits on ".").
shape_rows = [
    ("tokens", "tokens", tuple(tokens.shape), "B=batch of independent char windows, T=position in the window"),
    ("hidden", "wte(tokens)+wpe(pos)  hidden", (B, T, C), "B, T as above, C=embedding width (n_embd)"),
    ("lm_head_weight", "lm_head.weight  (tied to wte.weight)", tuple(model.lm_head.weight.shape), "V=one row per character, C as above; z = h @ W^T, same tensor as the input embedding"),
    ("logits", "logits", tuple(logits.shape), "B, T as above, V=one score per character"),
    ("logits_grad", "logits.grad", tuple(logits.grad.shape), "same shape as logits — one gradient per (batch, position, character) score"),
    ("lm_head_weight_grad", "lm_head.weight.grad", tuple(model.lm_head.weight.grad.shape), "same shape as lm_head.weight — because of tying, this IS wte.weight.grad too"),
    ("tied_grad_is_shared", "wte.weight.grad is lm_head.weight.grad", (model.wte.weight.grad is model.lm_head.weight.grad), "tying means one gradient tensor serves both roles, not two equal ones"),
    ("qkv_weight_grad", "blocks[0].attn.c_attn.weight.grad", tuple(model.blocks[0].attn.c_attn.weight.shape), "[3C, C] — packed Q,K,V projection, gradient matches the weight"),
]
for _, name, shape, meaning in shape_rows:
    print(f"  {name:<40} {str(shape):<12} {meaning}")

model.zero_grad(set_to_none=True)

# %% [markdown]
"""
## 2. Verify one gradient by hand
"""

# %%
w1 = torch.tensor(3.0, requires_grad=True)
w2 = torch.tensor(4.0, requires_grad=True)
x, t = torch.tensor(2.0), torch.tensor(20.0)

h = w1 * x
y = h * w2
L = (y - t) ** 2
L.backward()

dL_dy = 2 * (y - t)
dL_dw2 = dL_dy * h
dL_dh = dL_dy * w2
dL_dw1 = dL_dh * x

print("2a. toy chain  w1=3, x=2, w2=4, t=20")
print(f"    forward: h={h.item()}, y={y.item()}, loss={L.item()}")
print(f"    dL/dy={dL_dy.item()}  dL/dw2={dL_dw2.item()}  dL/dh={dL_dh.item()}  dL/dw1={dL_dw1.item()}")
print(f"    autograd w1.grad={w1.grad.item()}  w2.grad={w2.grad.item()}")

eps = 1e-3
w1p = torch.tensor(3.0 + eps)
w1m = torch.tensor(3.0 - eps)
Lp = ((w1p * x) * w2 - t) ** 2
Lm = ((w1m * x) * w2 - t) ** 2
numeric = ((Lp - Lm) / (2 * eps)).item()
print(f"    central diff (eps={eps}): {numeric:.4f}  vs autograd {w1.grad.item():.4f}  "
      f"(agree to {abs(numeric - w1.grad.item()):.2e})")

# %%
torch.manual_seed(0)
probe_tokens = get_batch()
model.zero_grad(set_to_none=True)
L0 = loss_fn(model, probe_tokens)
L0.backward()

# A non-tied weight (c_attn), so the check is unambiguous — a tied-weight nudge would perturb
# the model through two roles (input embedding and output head) at once. Within it, pick the
# LARGEST-magnitude gradient, not an arbitrary element: a central difference on a near-zero
# gradient asks fp32 to resolve a change below its own rounding floor (catastrophic
# cancellation), which would make the comparison noise-dominated rather than a real check.
w = model.blocks[0].attn.c_attn.weight
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
    w[row, col] += eps2

numeric_grad = (Lp2 - Lm2) / (2 * eps2)
print(f"2b. real weight blocks[0].attn.c_attn.weight[{row},{col}]  (eps={eps2})")
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
"""

# %%
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
    m = build(cfg, seed=seed)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    curve = []
    g = torch.Generator().manual_seed(seed + 1)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        sum_loss_tokens, sum_tokens = 0.0, 0
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
            if mode == "correct":
                (per_tok.sum() / micro_steps).backward()
            else:
                (micro_mean / micro_steps).backward()
            sum_loss_tokens += per_tok.sum().item()
            sum_tokens += n_tok
        if mode == "correct":
            scale = micro_steps / sum_tokens
            for p in m.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
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
fig.savefig("assets/nanogpt_accumulation_curves.png", dpi=120)
download_if_colab("assets/nanogpt_accumulation_curves.png")
plt.close(fig)

gap_unequal = abs(curve_wrong[-1] - curve_correct[-1])
gap_equal = abs(curve_wrong_eq[-1] - curve_correct_eq[-1])
print(f"3b. final loss gap, unequal token counts: {gap_unequal:.4f}")
print(f"    final loss gap, equal token counts:   {gap_equal:.4f}  (near zero — this is why it hides)")
print("    saved assets/nanogpt_accumulation_curves.png")

# %% [markdown]
"""
## 4. Grad norm logged from step 1, with one engineered bad batch

**This one demonstration cell uses plain SGD, not AdamW.** Adam's per-parameter update is
normalised by a running estimate of gradient magnitude, so a single spiked gradient mostly
pollutes that running estimate rather than producing an oversized *update* — which would mask
exactly the raw-gradient-to-update relationship this item is about. Plain SGD (`update = -lr ×
grad`) keeps that relationship direct; elsewhere in this notebook AdamW is used as normal.

**`lr=0.05` here, not the proxy notebook's `lr=1.0`.** The two notebooks train different
architectures (LayerNorm + learned positions + GELU here, vs RMSNorm + RoPE + SwiGLU there), and
`lr=1.0` plain SGD turned out to sit right at the edge of this model's stability — a first attempt
at this scale diverged on its own by step ~30, before the engineered spike ever fired, which would
have confounded the demonstration with an unrelated instability. `lr=0.05` keeps the pre-spike
loss flat and reproducible across seeds while still producing a clearly visible post-spike crack.
"""

# %%
BAD_STEP = 30
SCALE = 50.0
NORM_STEPS = 60

norm_model = build(cfg, seed=7)
opt = torch.optim.SGD(norm_model.parameters(), lr=0.05)
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
fig.savefig("assets/nanogpt_norm_before_loss.png", dpi=120)
download_if_colab("assets/nanogpt_norm_before_loss.png")
plt.close(fig)

print(f"4. engineered spike at step {BAD_STEP}: grad norm {norms[BAD_STEP - 1]:.3f} -> {norms[BAD_STEP]:.3f}")
print(f"   loss at step {BAD_STEP}: {losses[BAD_STEP]:.4f}  (baseline {baseline:.4f})")
print(f"   loss visibly cracks at step: {crack_step}  "
      f"(lag = {crack_step - BAD_STEP if crack_step is not None else 'n/a'} steps)")
print("   saved assets/nanogpt_norm_before_loss.png")

# %% [markdown]
"""
## 5. This run's own MFU

Measured in **fp16** if this turned out to be a T4 (the expected free-Colab case), bf16 if Colab
handed out something Ampere/Ada instead, fp32 if this is a local CPU dev run.
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

Identical derivation to the proxy notebook — this is pure format arithmetic, independent of
model or device — kept here so this notebook stands on its own.
"""

# %%
def decode_bits(sign, exponent_bits, mantissa_bits, bias):
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

raw32 = struct.unpack(">I", struct.pack(">f", TARGET))[0]
s32 = raw32 >> 31
e32 = (raw32 >> 23) & 0xFF
m32 = bits_of(raw32 & 0x7FFFFF, 23)
val32, exp32, mant32 = decode_bits(s32, e32, m32, bias=127)
print(f"fp32  (1+8+23): sign={s32}  exponent={e32} (biased, unbiased={exp32})  "
      f"mantissa={''.join(map(str, m32))}")
print(f"      stored value = {val32:.10f}")
print(f"      representation error = {val32 - TARGET:.3e}")
print()

bf16_val = ml_dtypes.bfloat16(TARGET)
raw16 = np.array([bf16_val]).view(np.uint16)[0]
s16 = int(raw16) >> 15
e16 = (int(raw16) >> 7) & 0xFF
m16 = bits_of(int(raw16) & 0x7F, 7)
val16, exp16, mant16 = decode_bits(s16, e16, m16, bias=127)
print(f"bf16  (1+8+7):  sign={s16}  exponent={e16} (biased, unbiased={exp16})  "
      f"mantissa={''.join(map(str, m16))}")
print(f"      stored value (hand-decoded) = {val16:.10f}   ml_dtypes stores = {float(bf16_val):.10f}")
print(f"      representation error = {val16 - TARGET:.3e}")
print()

fp8_val = ml_dtypes.float8_e4m3(TARGET) if hasattr(ml_dtypes, "float8_e4m3") else ml_dtypes.float8_e4m3fn(TARGET)
raw8 = np.array([fp8_val]).view(np.uint8)[0]
s8 = int(raw8) >> 7
e8 = (int(raw8) >> 3) & 0xF
m8 = bits_of(int(raw8) & 0x7, 3)
val8, exp8, mant8 = decode_bits(s8, e8, m8, bias=7)
print(f"fp8 E4M3 (1+4+3): sign={s8}  exponent={e8} (biased, unbiased={exp8})  "
      f"mantissa={''.join(map(str, m8))}")
print(f"      stored value (hand-decoded) = {val8:.6f}   ml_dtypes stores = {float(fp8_val):.6f}")
print(f"      representation error = {val8 - TARGET:.3e}")
print()
print("6b. which one would you train in, and why: bf16 — same fp32 exponent range as this GPU's")
print("    other formats, no loss-scaling knob to get wrong; fp8 E4M3 is named as the production")
print("    recipe it now is, not as what this run trains in.")

# %% [markdown]
"""
## N. Results

Printed between markers so `scripts/extract_results.py` can recover it from a downloaded
notebook whose Colab runtime is long gone by the time it reaches this repo.
"""

# %%
RESULTS = {
    "env": {"in_colab": IN_COLAB, "device": str(DEVICE), "gpu_name": GPU_NAME, "sm": SM,
            "bf16_native": BF16_NATIVE, "train_dtype": str(TRAIN_DTYPE), "torch": torch.__version__},
    "config": {
        "V": cfg.vocab_size, "C": cfg.n_embd, "n_layer": cfg.n_layer, "n_head": cfg.n_head,
        "T": cfg.seq_len, "B": cfg.batch_size, "N_params": N_PARAMS,
        "dataset": "nanoGPT tiny-Shakespeare, char-level",
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

with open("results_nanogpt.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

print("===RESULTS-JSON-BEGIN===")
print(json.dumps(RESULTS))
print("===RESULTS-JSON-END===")

if IN_COLAB:
    try:
        from google.colab import files
        files.download("results_nanogpt.json")
    except Exception as e:
        print(f"(download skipped: {e})")
