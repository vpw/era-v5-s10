# S10 TODO — The Training Loop

## ▶ STATUS (2026-09-03): both notebooks written and CPU-smoke-tested; GPU/Colab runs pending

Due **Sat, Sep 5, 2026, 7:00 AM** · 1000 pts · resubmission allowed. Deliverable is a **GitHub
README.md link** (incognito-accessible) with the `.ipynb` in the same repo. No deployment step —
same single-artifact shape as S9, not S7/S8's live-app-plus-repo.

- [x] **Session verified** (2026-08-29). The lesson page's own heading reads "Session 10: The
      Training Loop" — matches this folder. The supplied link was complete this time (no repeat of
      S9's one-character-short session id).
- [x] **Assignment captured** → `S10-assignment.md`, verbatim from the assignment page. Rubric tab
      is **empty** (no criterion rows, only the 1000-pt total), so the brief's six items are the
      whole spec. Assignment title is "Session 10 - Assignment QnA".
- [x] **Lesson captured** → `resources/s10-session.md`, all 17 sections verbatim (~20KB). Well under
      the 50,000-char `get_page_text` cap this time, but captured via the clipboard route anyway
      (strip `<style>` blocks, `navigator.clipboard.writeText()` the article node, pull with
      `xclip -selection clipboard -o`) since it is verbatim and costs no context. The page still
      renders the article **twice** (a desktop `lg:grid` copy and a mobile `lg:hidden` copy) — take
      the `content-r…` node, not `main`.
- [x] **Transcript captured** (2026-08-29) → `resources/s10-transcript.md`, 102KB, from Google Doc
      `1obvxJFP-pQT5A8DuCgq-tcviA_ZZupTHhwPvvdzrNVo` via `export?format=txt`. Verified as this
      session two ways: the doc id **matches the lesson page's own Transcript link**, and its header
      dates it **2026/08/29 06:44 IST** with the opening line *"till session 9 what we've done…"*.
      Fetched with plain `curl` (the doc is link-shared and needs no browser) rather than the
      browser-download route S9 used.
- [x] **Widget extraction judged optional** (2026-08-29). Every widget in this lesson animates a
      formula or table already given in full in the prose, and every graded number comes from the
      student's own run. The float-bit constructor is the only one that could cross-check an
      assignment item (0.1's bits), and `struct.pack` / `ml_dtypes` settles that more cheaply.
- [x] **Branch cut** — `s10-training-loop`, from `s9-loss-functions`, before the first S10 commit.
      (S8 started on S7's branch name by accident and had to be renamed at submission; S9 fixed the
      order, and this keeps it.)
- [x] **Working set written** — `CLAUDE.md`, `AGENTS.md`, this file.

## ▶ OPEN DECISIONS — settle with the user before building

- [x] **D1. Where does the loop run? → new EC2 g5/g6 instance** (decided 2026-09-03). There is
      **no GPU on this machine** (S9's `.venv` is torch cpu-only), and item 5's MFU needs a
      *measured* peak, not a spec-sheet number. The existing course box (`vardhan-gpu-1`,
      `g4dn.2xlarge`, T4, **sm_75, no native bf16 tensor cores**) would make any bf16 throughput
      number a claim about emulation, not bf16 — a real problem in a session about bf16. Options
      considered: Colab Pro/Pro+ (A100/L4 selectable, but Lane B is manual — user drives the
      browser, needs a subscription); staying on the T4 and reporting fp16 instead (zero cost, but
      item 5's headline number stops being a bf16 number). **Chosen: provision a fresh EC2
      instance with an Ampere/Ada card — `g5.xlarge` (A10G, sm_86) or `g6.xlarge` (L4, sm_89),
      both native bf16** — via the generic `aws-ec2` skill (`ec2.sh create --type g5.xlarge`),
      then reuse `era-v5-gpu-run`'s Lane A pipeline unattended against the new box (same
      start → rsync → remote venv → build+execute → pull-back → stop shape as the T4 flow, just
      pointed at a different `INSTANCE_ID`/`GPU_NAME`). Bills by the second like the T4 box;
      a short nanoGPT-scale run is expected to cost well under $1. Report device + dtype next to
      every throughput/MFU number regardless (carried convention, unaffected by this choice).
      Item 6's bit patterns are pure arithmetic and need no GPU at all.
- [x] **D2. What model? → switch to nanoGPT** (decided 2026-09-03). The live-class transcript
      (`resources/s10-transcript.md`, ~lines 336-340) has the instructor naming this model by name
      for students doing the assignment solo — *"take a small model... can be the Andrej Karpathy
      nanoGPT, it's a good model"* — and walking through items 1-5 (shapes, gradient check,
      accumulation bug, grad norm, MFU) against it directly, then noting Colab's GPU is sized for
      "something like nanoGPT." That also eases D1: nanoGPT's small reference configs (e.g.
      `shakespeare_char`, ~10M params) don't strictly need an upgraded card to produce a legitimate
      MFU, though D1's new instance still gives the honest bf16 number. **Superseded:** the plan to
      reuse S9's proxy transformer (`V=10,000, D=256`, 4 layers/heads, `T=512`, S2's BPE `mr`
      tokenizer) for cross-session continuity — dropped in favor of the instructor's explicit
      pointer. Item 2's by-hand two-weight scalar chain (`w1=3, x=2, w2=4, t=20`) still runs
      alongside as the worked-example cross-check, verified against a real nanoGPT weight nudge.
      **Revised 2026-09-03 — run BOTH models, not either/or**, since the incremental cost is
      near zero: **S9's proxy transformer on the new EC2 g5/g6 box** (native bf16, per D1) *and*
      **nanoGPT on free-tier Colab** (Lane B). Free Colab's GPU picker only offers **T4** (Pro/Pro+
      unlocks A100/L4) — same sm_75, no native bf16 tensor cores as the course's own EC2 box — so
      the nanoGPT/Colab track measures and reports throughput/MFU in **fp16** (T4's native
      accelerated format); the EC2/proxy track carries the actual bf16 number. Structure: two
      notebook sources, `notebook_src_proxy.py` (EC2 g5/g6) and `notebook_src_nanogpt.py` (Colab),
      each producing its own `.ipynb` + `results_*.json`; one `README.md` (extend
      `tools/build_readme.py` to read both result files) answering all six items for each model
      side by side, plus a comparison section (bf16-native vs fp16-on-T4, EC2 vs Colab).
      Provisioning the new EC2 instance still needs an explicit go before it's actually created —
      that's a real billed resource, unlike everything decided so far.
- [x] **D3. How is item 4's "norm moved before the loss" produced? → engineer a bad batch**
      (decided 2026-09-03). Inject one deliberately bad micro-batch (an outlier target/input) at a
      known step in both models' training runs, so the norm-spikes-then-loss-cracks lag is
      guaranteed to appear and is easy to point at. State plainly in the README that it was
      engineered, per the lesson's own example (norm spikes step 24, loss cracks step 26).

## Part 1 — the six items

Each must be printed by a cell that actually ran. A number in the README with no cell behind it is
exactly the failure this session is about: *"a plausible number is not evidence."* **Implemented in
both `notebook_src_proxy.py` and `notebook_src_nanogpt.py`, CPU-smoke-tested at reduced scale
(2026-09-03) — correct, but the CPU numbers themselves are dev-only and are not what ships.** Real
numbers still need the actual EC2 g5/g6 run (proxy) and Colab T4 run (nanoGPT). Bugs the smoke
testing caught and fixed: `logits` needed `.retain_grad()` to expose its gradient shape at all; the
item-2 real-weight check was picking an arbitrary weight whose gradient was small enough that fp32
central-differencing hit its own rounding floor (fixed by picking the **max-magnitude** gradient
element, and widening ε to 1e-2 for that check specifically) — now agrees to ~1e-4–1e-5 absolute;
item 4's original AdamW-based demo never showed a lag (Adam's per-parameter normalisation absorbs
an isolated gradient spike instead of translating it into an oversized *update*) — fixed by running
that one cell on **plain SGD** instead, which reproduced a clean 1-step lag on both models.

- [x] **1. Every tensor shape in the step**, one line each naming what the dimension means —
      batch, sequence, model dim, vocab, and the shapes of the gradients as well as the activations.
      Both models also show a weight-tying-specific fact nanoGPT has and the proxy doesn't:
      `lm_head.weight` and `wte.weight` are literally the same tensor, not a coincidentally-equal one.
- [x] **2. Verify one gradient by hand.** Toy scalar chain (`w1=3,x=2,w2=4,t=20`) matches the
      lesson's numbers exactly; the real-weight central-difference check (picking the
      largest-magnitude gradient in a non-tied weight, ε=1e-2) now agrees to ~1e-4–1e-5 absolute on
      both models — see the fp32-cancellation note above for why an arbitrary small-gradient weight
      would have made this noise-dominated instead.
- [x] **3. Break gradient accumulation on purpose.** Static replica of the lesson's own numbers
      (2.6000 vs 3.0000, 15.4%) plus two training curves per model (unequal vs equal micro-batch
      token counts) confirming the gap is real when lengths differ and vanishes when they don't.
- [x] **4. Log the grad norm at every step**, engineered spike (50× on one micro-batch's gradient,
      per D3's decision), plain-SGD demo run — clean 1-step lag confirmed on both models in
      CPU smoke tests (norm spikes at step *k*, loss unmoved at *k*, cracks at *k+1*).
- [x] **5. Compute your own MFU** — implemented with a *measured* peak (large matmul benchmark in
      the run's own dtype/device, not a spec-sheet number) and measured tokens/s from a timed
      training loop. CPU smoke-test numbers are meaningless (small-matmul benchmark badly
      underestimates peak on this crippled sandbox CPU) — this item's real numbers only mean
      something once run on the actual EC2 bf16 box and Colab T4.
- [x] **6. 0.1 in fp32, bf16 and fp8 E4M3, by hand, showing the bits.** Hand-decode cross-checked
      bit-for-bit against `struct.pack`/`ml_dtypes` in both notebooks independently (pure format
      arithmetic, so it should agree, and did). Closing take: bf16, for §10's exponent-range reason.

## Ship

- [x] **S0. Scaffold both notebooks, tooling, and template** (2026-09-03) — `tools/` copied from S9
      (`py2nb.py`/`run_nb.py`/`dump_log.py` unchanged, `build_readme.py` extended to read
      `results_proxy.json` + `results_nanogpt.json` under `proxy.*`/`nanogpt.*` namespaces),
      `notebook_src_proxy.py` + `notebook_src_nanogpt.py` written and CPU-smoke-tested (reduced
      scale), `README.tmpl.md` written and its placeholders verified against a real (if CPU-scale)
      `results_proxy.json`. Full round trip validated once end-to-end: `py2nb.py` → `run_nb.py`
      (nbclient execution) → `scripts/extract_results.py` (era-v5-gpu-run skill) correctly recovers
      `results_nanogpt.json` from an executed notebook's own printed output — the exact mechanism
      the real Colab submission depends on.
- [x] **S0b. Provision the EC2 g5/g6 instance** (D1) — done 2026-09-04. `vardhan-gpu-s10`,
      `i-0b3ea94f227069b70`, `g5.xlarge` (NVIDIA A10G, sm_86, bf16-native — confirmed via
      `nvidia-smi`), AMI `ami-0391c53a255869d46` ("Deep Learning Base OSS Nvidia Driver GPU AMI
      Ubuntu 22.04 20260902" — drivers/CUDA preinstalled, no framework baked in since
      `remote_setup.sh` builds its own venv anyway), 80GB root disk (DLAMI snapshot needs ≥75GB),
      key pair `vardhan-ed25519` (imported from this machine's actual `~/.ssh/id_ed25519.pub` —
      the skill's default `vardhan-id-rsa` key pair does not match any private key on this
      machine). Public IP `13.206.235.247` (will change on stop/start). **Remember to stop this
      instance after S1a** — it bills by the second.
- [x] **S1a. Run `notebook_src_proxy.py` on the EC2 box** — done 2026-09-04. DLAMI's Ubuntu image
      was missing `python3-venv`; installed it, then `remote_setup.sh` ran clean (`torch
      2.5.1+cu121 cuda True`, `NVIDIA A10G sm_86`). Notebook executed top to bottom (25 cells, 15
      with output) in real bf16. Key real numbers: item 3 reproduces the lesson's own figures
      exactly (correct=2.6000, wrong=3.0000, 15.38% error); item 4's engineered spike shows the
      grad norm jump 10.43→81.32 at step 30 while loss barely moves (8.925→8.966), cracking at
      step 31 (1-step lag) — the "norm before loss" effect confirmed on real hardware; item 5 MFU
      = 5.35% (measured against a real 4096×4096 bf16 matmul benchmark peak, not a spec sheet) —
      low as expected for an 8.3M-param model, worth explaining in README (kernel-launch/Python
      overhead dominates at this scale, not FLOP throughput). `S10_proxy.ipynb`,
      `results_proxy.json`, `assets/proxy_accumulation_curves.png`,
      `assets/proxy_norm_before_loss.png` pulled back. Instance stopped and verified (2026-09-04).
- [x] **S1b. Run `notebook_src_nanogpt.py` on free-tier Colab** — done 2026-09-04, after one
      retry. First attempt used `lr=1.0` (tuned against the proxy model) for item 4's SGD; on the
      real Tesla T4 the loss was already 2.4x above `ln(65)=4.17` by step ~28 and jumped 12x
      further exactly at the announced bad step — before the engineered gradient scaling could
      have had any effect, so `lr=1.0` was simply unstable for this architecture on its own.
      CPU tuning across 3 seeds found `lr=0.05` keeps the pre-spike loss flat and reproducible;
      re-ran on Colab and got a clean result: `loss_baseline=3.224`, `loss_at_bad_step=3.322`
      (+3.0%), crack at step 31 (1-step lag) — matches the CPU tuning almost exactly. Also added
      explicit `files.download()` calls for the two plot PNGs, since Colab's local disk is
      ephemeral and they weren't embedded as displayed cell outputs (missing after the first
      run). `results_nanogpt.json` recovered via `extract_results.py`; item 3 again reproduces
      the lesson's exact figures (2.6000 vs 3.0000, 15.38%); item 5 MFU = 2.68% (fp16 on T4,
      measured against a real matmul benchmark peak, not a spec sheet).
- [x] **S2. Write `README.md`** — done 2026-09-04. Built via `tools/build_readme.py` from both
      real `results_*.json` (101 values filled, no unresolved placeholders). Both synthesis
      sections in `README.tmpl.md` written from the real numbers: §5 explains the proxy's ~2x
      higher MFU as vocab-head matmul size, not raw GPU speed or dtype; the closing section notes
      the identical 1-step norm-before-loss lag on both models (achieved only after per-model SGD
      lr tuning) and the EC2-vs-Colab artifact-recovery reliability gap.
- [ ] **S3. Push** via subtree split (below), with both `.ipynb`s, both `results_*.json`, and logs.
- [ ] **S4. Verify the README link in an incognito window**, then tick the form's checkbox honestly.
- [ ] **S5. Submit** the GitHub README.md link and record it back in this file.

## Standing conventions (carried from S6-S9, don't re-decide)

**Build pipeline — the notebook is generated, not hand-edited.** S9's `tools/` copied in
(`build_readme.py` extended for two result files); **two** `# %%` sources this session, not one —
`notebook_src_proxy.py` (EC2) and `notebook_src_nanogpt.py` (Colab):

```
python tools/py2nb.py notebook_src_proxy.py S10_proxy_transformer.ipynb
python tools/run_nb.py S10_proxy_transformer.ipynb     # on the EC2 box — writes results_proxy.json
python tools/dump_log.py S10_proxy_transformer.ipynb logs/run_proxy.log

python tools/py2nb.py notebook_src_nanogpt.py S10_nanogpt.ipynb
# push, open from GitHub in Colab, Run all, download the executed .ipynb back, then:
python scripts/extract_results.py S10_nanogpt.ipynb results_nanogpt.json   # era-v5-gpu-run skill
python tools/dump_log.py S10_nanogpt.ipynb logs/run_nanogpt.log

python tools/build_readme.py     # README.tmpl.md + both results_*.json -> README.md
```

**Push destination — subtree split, not a nested `.git`** (settled 2026-08-14 after comparing S6 vs
S7). Work happens as normal commits inside this TSAI branch; at submission time:

```
git subtree split --prefix=ERA/V5/S10/assignment -b s10-standalone
git push https://github.com/vpw/era-v5-s10.git s10-standalone:main
```

Keep `CLAUDE.md` / `AGENTS.md` / `TODO.md` in the split (user-confirmed at S7). There's no `gh` CLI
or credential helper on this machine — the user pastes a PAT at the push password prompt, so hand
over the command rather than running it.
