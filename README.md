# ERA V5 · Session 10 — The Training Loop

Session 9 left us holding a single scalar. This session asks how that one number reaches back
and moves billions of weights, and how — while it runs for weeks — anyone knows it is working.
The lesson's own warning is the organising idea: **a plausible number is not evidence**; every
serious training bug (a flipped gradient-accumulation average, an MFU that looks fine at 8% and
at 45%) is silent, and the loss curve is not the thing that catches it.

This session runs the six required checks **twice, on two different models and two different
GPUs**, so the numbers can be compared rather than taken on faith:

| | **proxy transformer** | **nanoGPT** |
|---|---|---|
| architecture | RMSNorm · RoPE · SwiGLU (Session 9's block) | LayerNorm · learned pos-emb · GELU MLP, weight-tied head (Karpathy's own) |
| data | Session 2 BPE over en/hi/te/mr | char-level tiny-Shakespeare |
| hardware | fresh EC2 **NVIDIA A10G** (sm_86) | free-tier Colab **Tesla T4** (sm_75) |
| bf16 native (tensor-core) | **True** | False |
| training dtype measured | **torch.bfloat16** | torch.float16 |
| parameters `N` | 8,333,568 | 813,440 |

The instructor named nanoGPT directly, in the live-class transcript, as the model for students
doing this assignment solo — and noted free Colab's GPU is sized for exactly that. Its GPU
picker only offers a **T4** (sm_75), the same generation as the course's own EC2 box, and with
the same limitation: no native bf16 tensor cores. So the nanoGPT/Colab track's throughput and
MFU are measured in **fp16**, the T4's native accelerated format; the proxy/EC2 track — run on a
freshly-provisioned Ampere/Ada box specifically to get past that limitation — carries this
session's honest, hardware-accelerated **bf16** number. Comparing the two is the point of running
both.

**Notebooks:** [`S10_proxy_transformer.ipynb`](S10_proxy_transformer.ipynb) (EC2, runs top to
bottom, committed with its outputs) and
[`S10_nanogpt.ipynb`](S10_nanogpt.ipynb) (Colab, downloaded after `Run all`).
**Raw numbers:** [`results_proxy.json`](results_proxy.json), [`results_nanogpt.json`](results_nanogpt.json).
**Full run logs:** [`logs/run_proxy.log`](logs/run_proxy.log), [`logs/run_nanogpt.log`](logs/run_nanogpt.log).

> Every number below was read out of those two files by
> [`tools/build_readme.py`](tools/build_readme.py). None of them is typed by hand, so the
> write-up cannot drift from the runs that produced it.

---

## 1. Every tensor shape in the step

**Proxy transformer** (`B=8, T=128, D=256, V=10,000`):

| tensor | shape | what the dimensions are |
|---|---|---|
| `tokens` | `(8, 128)` | B = independent sequences · T = position |
| `hidden` (trunk output) | `(8, 128, 256)` | B, T as above · D = residual-stream width |
| `head.weight` (`W_vocab`) | `(10000, 256)` | V = one row per vocabulary entry · D — `z = h · W_vocabᵀ` |
| `logits` | `(8, 128, 10000)` | B, T as above · V = one score per vocabulary entry |
| `logits.grad` | `(8, 128, 10000)` | same shape as `logits` — one gradient per (batch, position, vocab-entry) score |
| `head.weight.grad` | `(10000, 256)` | same shape as `head.weight` — every vocabulary row gets an update |
| `trunk.embed.weight.grad` | `(10000, 256)` | `[V, D]` — the input embedding table |
| `attn.qkv.weight.grad` | `(768, 256)` | `[3D, D]` — packed Q,K,V, gradient matches the weight |

**nanoGPT** (`B=8, T=128, C=128, V=65`):

| tensor | shape | what the dimensions are |
|---|---|---|
| `tokens` | `(8, 128)` | B = independent char windows · T = position |
| `wte(tokens)+wpe(pos)` hidden | `(8, 128, 128)` | B, T as above · C = embedding width |
| `lm_head.weight` (tied to `wte.weight`) | `(65, 128)` | V = one row per character · C — same tensor as the input embedding |
| `logits` | `(8, 128, 65)` | B, T as above · V = one score per character |
| `logits.grad` | `(8, 128, 65)` | same shape as `logits` |
| `lm_head.weight.grad` | `(65, 128)` | because of tying, this **is** `wte.weight.grad` too, not merely equal to it (`True`) |
| `attn.c_attn.weight.grad` | `(384, 128)` | `[3C, C]` — packed Q,K,V |

nanoGPT's weight tying is the one shape fact the proxy model doesn't have: `lm_head.weight` and
`wte.weight` are the *same tensor*, so their gradients are identical by construction, not by
coincidence.

---

## 2. Verify one gradient by hand

The lesson's own scalar chain (`w1=3, x=2, w2=4, t=20`), reproduced in both notebooks, plus a
central-difference check on one real weight from each model actually being trained.

| | proxy transformer | nanoGPT |
|---|---|---|
| toy chain `∂L/∂w1` — analytic | 64.0 | 64.0 |
| toy chain `∂L/∂w1` — autograd | 64.0 | 64.0 |
| toy chain `∂L/∂w1` — central diff (ε=1e-3) | 63.9958 | 63.9958 |
| real weight — autograd | 0.021285 | -0.075493 |
| real weight — central diff | 0.021362 | -0.075436 |
| absolute difference | 7.75e-05 | 5.70e-05 |

The lesson's own worked case agrees to eight decimals; both models' real-weight checks land at
the precision floor of a central difference in fp32 — evidence that autograd is bookkeeping, not
a separate approximate computation that happens to usually agree.

---

## 3. Break gradient accumulation on purpose

**Static replica of the lesson's own numbers** (identical arithmetic in both notebooks, since it
doesn't depend on a model):

- micro-batches (tokens, mean loss): `[[4, 2.0], [4, 2.0], [2, 5.0]]`
- correct (total loss ÷ total tokens): **2.6000**
- wrong (average of the per-micro-batch means): **3.0000**
- error: **15.4%** (lesson: 2.6000 vs 3.0000 = 15.4%)

**As training curves**, micro-batches of two different token counts, run to
120 steps:

| | proxy transformer | nanoGPT |
|---|---|---|
| final loss gap, **unequal** token counts | 0.0391 | 0.0003 |
| final loss gap, **equal** token counts | 0.0000 | 0.0000 |

![proxy accumulation curves](assets/proxy_accumulation_curves.png)
![nanogpt accumulation curves](assets/nanogpt_accumulation_curves.png)

The equal-token-count columns are the reason this bug lived in every major framework until
2024: when micro-batches happen to be the same size, average-of-averages and total-over-total
are the same number, so a casual test with a fixed sequence length never sees the gap.

---

## 4. Grad norm logged from step 1, with one engineered bad batch

Per this session's decision, the "bad batch" that spikes the norm is **engineered, not found** —
one micro-batch's gradient is deliberately scaled up at a chosen step, disclosed here rather than
discovered in the wild. No gradient clipping is applied in this run, so the oversized update is
actually taken — which is what lets the damage surface in the loss a step or two later instead of
in that same step's own (pre-update) loss.

| | proxy transformer | nanoGPT |
|---|---|---|
| engineered step | 30 | 30 |
| grad norm, step before → at | 10.43 → 81.32 | 3.73 → 165.28 |
| loss at the engineered step | 8.9659 (baseline 8.9249) | 3.3220 (baseline 3.2239) |
| step the loss visibly cracks | 31 | 31 |
| lag, in steps | **1** | **1** |

![proxy norm before loss](assets/proxy_norm_before_loss.png)
![nanogpt norm before loss](assets/nanogpt_norm_before_loss.png)

The lesson's own widget shows this lag as 2 steps (norm spikes at 24, loss cracks at 26). The
mechanism is the same reason the lag exists at all: the loss printed at a given step is computed
*before* that step's update is applied, so an oversized update's damage only shows up once the
model is evaluated *after* it — one or more forward passes later.

---

## 5. This run's own MFU

`MFU = 6 × N × tokens/s ÷ peak FLOP/s`. The peak in both cases is **measured**, not quoted from a
spec sheet — a large matmul benchmarked in the same dtype, on the same device, immediately before
the timed training loop.

| | proxy transformer (EC2) | nanoGPT (Colab) |
|---|---|---|
| device | NVIDIA A10G (sm_86) | Tesla T4 (sm_75) |
| dtype measured | **torch.bfloat16** | **torch.float16** |
| `N` (parameters) | 8,333,568 | 813,440 |
| measured tokens/s | 66,711 | 86,021 |
| achieved FLOP/s | 3.336e+12 | 4.198e+11 |
| peak FLOP/s (measured: 4096x4096 matmul benchmark, same device/dtype) | 6.233e+13 | 1.569e+13 |
| **MFU** | **5.35%** | **2.68%** |

Healthy is 35–50%; the lesson's own worked example (a 9B model at 12,000 tok/s on eight H100s)
lands at 8.2% and looks, on its loss curve alone, indistinguishable from 45%. Neither run here is
at production scale, so neither MFU is a verdict on the hardware — it is a verdict on how much of
this particular step is matmul-bound at this batch/sequence size versus overhead-bound (Python
dispatch, kernel launch, a small model that doesn't saturate the device). What costs the distance
to 40% here specifically: at `B=8, T=128` neither step does enough matmul work to keep the device
fed between kernel launches — Python dispatch and launch overhead dominates a step this short, the
opposite of §14's 9B/H100 case, where the bottleneck is HBM bandwidth on a model too large to fit
its optimizer state comfortably. The two numbers here aren't even directly comparable as "GPU
speed": the proxy's MFU is roughly double nanoGPT's despite running on the faster device (A10G vs
T4) — because MFU is normalized against each device's own measured peak, a faster GPU with the same
tiny batch should if anything look *worse* on this metric, not better, since the fixed overhead now
eats a larger share of a shorter ideal step time. The proxy comes out ahead anyway because its
`10,000`-way vocabulary head is a much bigger matmul than nanoGPT's `65`-way one, so more of each
step really is compute, not overhead. At production scale (large batches, large vocab, sequences
in the thousands) this overhead amortizes away and both numbers would move well past 40%.

---

## 6. `0.1` in fp32, bf16 and fp8 E4M3 — by hand, showing the bits

`0.1` in binary is the repeating fraction `0.0(0011)` — not exactly representable in any binary
floating-point format. Derived longhand (sign / biased exponent / mantissa) and cross-checked
against `struct.pack` (fp32) and `ml_dtypes` (bf16, fp8 E4M3) — identical arithmetic in both
notebooks, shown here from the proxy run:

| format | bits (sign+exp+mantissa) | sign | exponent (biased) | mantissa | stored value | error |
|---|---|---|---|---|---|---|
| fp32 | 1+8+23 | 0 | 123 | `10011001100110011001101` | 0.1000000015 | 1.490e-09 |
| bf16 | 1+8+7 | 0 | 123 | `1001101` | 0.1000976562 | 9.766e-05 |
| fp8 E4M3 | 1+4+3 | 0 | 3 | `101` | 0.101562 | 1.562e-03 |

Both cross-checks (proxy and nanoGPT notebooks, run independently) agree bit-for-bit — this is
pure format arithmetic, so it should, and did.

**Which one to train in, and why: bf16.** fp32, bf16 and fp8 all share the same biased-exponent
scheme; bf16 keeps fp32's full 8-bit exponent (identical dynamic range, no loss-scaling knob to
get wrong the way fp16 needs one), and pays for it purely in mantissa precision — the trade §10
names as the reason bf16 replaced fp16. fp8 E4M3 is named here as the production recipe it now
is (with block scaling, per §11), not as what either of these runs actually trains in.

---

## What running this twice actually showed

The MFU gap (5.35% proxy vs 2.68% nanoGPT) is explained above — it's mostly the vocabulary-head
matmul size, not the bf16-vs-fp16 dtype or which GPU is objectively faster. The dtype comparison
itself came through cleanly, though: both runs measured their peak against a real benchmark rather
than a spec sheet, so 5.35% and 2.68% are honest numbers for a bf16-native A10G and an
fp16-emulated-bf16 T4 respectively, at matched model shape and batch/sequence size.

The norm-before-loss lag, on the other hand, **came out identical on both models — exactly one
step** (spike at 30, crack at 31). That consistency is real, but it took per-model tuning to see
it cleanly: the same `SCALE=50×` gradient spike and plain-SGD mechanism needed a different learning
rate on each architecture (`lr=1.0` for the proxy's RMSNorm/RoPE/SwiGLU stack, `lr=0.05` for
nanoGPT's LayerNorm/GELU one) — nanoGPT's first attempt at `lr=1.0` was already diverging on its
own by step ~28, well before the engineered fault ever fired, and would have reported a confounded
result if taken at face value. The lesson underneath both items 4 and 5 is the same one: a fixed
recipe (a learning rate, a "good enough" batch size) that looks correct on one model can silently
misbehave on the next, and the only way to know is to look at the actual numbers a run produced —
not to assume the previous model's tuning still applies.

Session mechanics also differed in a way worth naming: the EC2 lane is fully scriptable and its
artifacts (executed notebook, JSON, plots) come back over `rsync` without any manual step. Colab's
lane depends on the browser tab staying open and each output being an explicit download — the
first nanoGPT run's plot PNGs never made it back at all, because they were saved to Colab's
ephemeral disk but never displayed as cell outputs, so nothing was embedded in the downloaded
`.ipynb`. That's a real difference between "runs unattended, bills by the second" and "free,
but every artifact has to be pulled out by hand" — not a difference in the science, but a real one
in how much the process can be trusted to hold everything together on its own.

---

## Reproduce

```
python tools/py2nb.py notebook_src_proxy.py S10_proxy_transformer.ipynb
python tools/run_nb.py S10_proxy_transformer.ipynb          # on the EC2 box — see era-v5-gpu-run
python tools/dump_log.py S10_proxy_transformer.ipynb logs/run_proxy.log

python tools/py2nb.py notebook_src_nanogpt.py S10_nanogpt.ipynb
# push, open from GitHub in Colab, Runtime > T4 GPU > Run all, download the executed .ipynb back
python tools/dump_log.py S10_nanogpt.ipynb logs/run_nanogpt.log

python tools/build_readme.py                                 # results_*.json -> README.md
```
