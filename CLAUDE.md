# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

Session 10 (S10) assignment of the ERA V5 course (The School of AI). The session topic is
**The Training Loop** — Session 9 left us holding a single scalar; this session asks *how does one
number reach back and move billions of weights, and while it runs for weeks, how do we know it is
working?* The lesson runs 17 sections and every one of them answers one of two questions: **what
does one training step actually do**, and **what does it cost V5 to take that step, over and over,
for weeks**. Two things the course has been using without introduction finally get one:
**backpropagation** and the **floating point number** (bf16 has been written since Session 7 as if
self-evident).

The arc: vocabulary (a *gradient* belongs to one weight, a *step* moves all of them, a *batch* is
what one step reads, a *micro-batch* is what a batch is broken into when it will not fit); the
gradient as **the answer to a nudge**; backprop as the chain rule one link at a time; **autograd as
bookkeeping, not magic**; the five-line step (`forward → loss → backward() → step() → zero_grad()`)
and why the wipe is not optional; **gradient accumulation**, which turns that accumulate-don't-replace
behaviour into a feature; **the accumulation bug that lived in every major framework until 2024**;
how a computer holds a number (sign/exponent/mantissa, range vs detail); **why bf16 replaced fp16**;
the 2026 formats (fp8 E4M3/E5M2, fp4 E2M1, block scaling); **gradient clipping and the grad norm**;
**16 bytes per weight**; **MFU**; the failures that stay quiet; and §16's V5 decisions.

**The numbers that matter for this assignment** (all from the lesson, `resources/s10-session.md`):

- §3 nudge: weight `3.000 → loss 16.000`, weight `3.001 → loss 16.064`, so slope `0.064/0.001 =` **64**.
- §4 chain by hand: `w1=3, x=2, h=6, w2=4, y=24, t=20, loss=16`; `∂L/∂y = 8`, `∂L/∂w2 = 48`,
  `∂L/∂h = 32`, `∂L/∂w1 =` **64** — and the numeric nudge gives `64.0000`, agreeing **to eight
  decimals**. This is the shape of assignment item 2.
- §7 accumulation: micro-batch 8 × 4 accumulation steps = global batch 32. V4 itself ran a
  micro-batch of **7 per GPU** and a global batch of **56** across one eight-GPU node.
- §8 the bug: micro-batches of (4 tokens, loss 2.0), (4, 2.0), (2, 5.0). Correct = `26.0/10 =`
  **2.6000**; average-of-averages = `9.0/3 =` **3.0000**; **15.4% wrong**. It hides whenever token
  counts happen to be equal, which in casual testing they usually are.
- §10: fp32 = 1+8+23, fp16 = 1+5+10, bf16 = 1+8+7. bf16 keeps **all eight exponent bits** and pays
  in detail (7.2 → **2.4 decimal digits**). fp16 dies below ~`5.96×10⁻⁸`; bf16's floor is
  `9.18×10⁻⁴¹`. Loss scaling ×1024 rescues fp16 (`10⁻⁸ × 1024 = 1.02×10⁻⁵`) and is one more knob to
  get wrong.
- §11: fp8 E4M3 = 1.2 decimal digits, E5M2 = 0.9, fp4 E2M1 = 0.6. Block scaling makes fp4 usable:
  `(16×4 + 8)/16 =` **4.5 bits per value**. NVFP4 is ~**1.73×** faster than fp8 and needs five
  things right, including leaving attention in higher precision.
- §12 clipping: `norm 8.4, cap 1.0 → scale by 0.119`. **The grad norm moves before the loss does** —
  the lesson's widget spikes the norm at step 24 and the loss only at step 26.
- §13: **16 bytes per weight** (2 bf16 weight + 2 bf16 grad + 4 fp32 master + 8 optimiser) →
  2B = 29.8 GiB, 9B = 134.1 GiB, 20B = 298.0 GiB, 120B = 1,788 GiB. An 80 GB card holds about a
  **5.4B** model with nothing left for activations. Activation checkpointing: ~**30% more compute**.
- §14 MFU `= 6 × N × tokens/s ÷ machine peak`. The worked case: a 9B model at 12,000 tok/s on eight
  H100s = **648 TFLOP/s achieved** against **7,912 TFLOP/s paid for** = **8.2% MFU**. Healthy is
  **35–50%**. At 8% the loss curve looks exactly as it does at 45%.

**Deliverable shape is the same as S9's — a notebook, not a web app.** The assignment page's
submission block: *"You're submitting GitHub Repo and README.md where these details are saved for
our review. Your ipynb (jupyter notebook) file should also be there."* One link field (GitHub
README.md, 1000 pts) plus the incognito-accessibility checkbox. Due **Sat, Sep 5, 2026, 7:00 AM**,
resubmission allowed, and the **Rubric tab is empty** — the brief's six items are the whole spec.

**The thing this session is actually testing** is the same discipline as S9, stated more bluntly:
*"Print things and check things. Every serious training bug is silent, and the loss curve is not
going to be the one that tells you."* §8's framework bug looked plausible for years; §14's 8% MFU
produces a loss curve identical to 45%; §15's V4 failures *"pass every cheap check while staying
silent."* Every item in the brief is an instrument you build against a specific silent failure.

## Layout

- `S10-assignment.md` — the assignment statement, verbatim from the assignment page (brief +
  submission block). 1000 points, due Sat Sep 5 2026, resubmission allowed. Rubric tab is **empty**.
- `resources/s10-session.md` — full lesson writeup, all 17 sections, captured verbatim from the
  lesson page (~20KB). Inline MathJax appears twice per formula (spelled-out form then rendered
  glyph run) and tables are flattened to rows — an artifact of capturing rendered text, not an
  editing choice.
- `resources/s10-transcript.md` — full live-class transcript (~102KB), downloaded from the Google
  Doc the lesson page itself links to (id `1obvxJFP-…`, verified to match the page's own link).
  Header line dates it **2026/08/29 06:44 IST** and it opens with *"till session 9 what we've done…"*.
- Not yet created: the notebook, the write-up/README, and the GitHub repo.

## Conventions

- **Submission target is a GitHub README.md link**, with the repo containing the `.ipynb`. Same
  single-artifact shape as S9 — no deployment step, no hosting decision. The link must pass an
  **incognito-window accessibility check**; the form has a dedicated checkbox for it.
- **Every number in the write-up must come from a cell that actually ran.** Carried straight from
  S9, and this session states it in the lesson's own voice: a plausible number is not evidence.
  The lesson's figures (64, 2.6000 vs 3.0000, 15.4%, 8.2%, 16 bytes/weight) are the *expected*
  values to check the harness against — state them as predictions, then show the run agreeing or
  explain the gap.
- **Reuse S9's build pipeline rather than hand-editing a notebook.** `../../S9/assignment/` has
  `notebook_src.py` (`# %%` cells) → `tools/py2nb.py` → `tools/run_nb.py` (writes `results.json`) →
  `tools/dump_log.py` → `tools/build_readme.py` (`README.tmpl.md` + `results.json` → `README.md`,
  **exits non-zero on an unresolved placeholder**). That last property is what makes "no number
  without a cell behind it" mechanical rather than aspirational. Copy the tools, don't rewrite them.
- **Widget-data extraction is optional here, not a prerequisite** — same call as S9. The lesson's
  widgets (the step-by-step loop, the nudge, the backward walk, the accumulation buffer, the
  bug-side-by-side, the float-bit constructor, the shrinking gradient, block scaling, the
  norm-before-loss chart, the memory slider, the MFU comparison) demonstrate formulas and tables
  already stated in full in the prose, and every graded number comes from the student's own run.
  The float-bit constructor is the one worth a look *if* the by-hand 0.1 derivation needs a
  cross-check — but `struct.pack` / `ml_dtypes` settles that more cheaply. Reach for
  `extract-widget-data` only if a specific widget value becomes load-bearing.
- **The `arxiv-library` skill applies only if the write-up makes external factual claims.** The
  deliverable is measurement, not citation. If the write-up does name a source (the 2024
  gradient-accumulation bug, NVFP4/Blackwell throughput, activation checkpointing, Adam's state),
  find it via the skill's arxiv MCP layer rather than trusting recall, download the PDF into the
  local library so the source is a checkable file, and index via `rag-toolkit` when a specific claim
  needs pulling out of the PDF text with a citation.
- **Known discrepancy — §15 says "eight failures" and lists seven.** The V4 table has seven rows
  (silent checkpoint misload, flush divergence at scale, clone-family collapse, role-blind
  expansion, curriculum accounting, spectral upcycling, mixture-shift instability). Nothing in the
  assignment depends on the count; don't invent an eighth. (Unlike S9's phantom "Part 3", the lesson
  §17 and the assignment page brief are **word-for-word identical** here — there is no second
  reading of the spec to reconcile.)
- **No GPU on this machine** (carried from S9: `.venv` with torch cpu-only). This bites harder than
  it did last session, because item 5 asks for **MFU**, which is a ratio against a device's peak
  FLOP/s. Use the global **`era-v5-gpu-run`** skill rather than re-deriving a GPU workflow: Lane A
  runs the notebook on the course's own EC2 T4 (`vardhan-gpu-1`, `g4dn.2xlarge`) end to end and
  pulls the executed `.ipynb`/`results.json`/`logs` back; Lane B prepares a Colab-portable notebook
  and recovers `results.json` from the downloaded `.ipynb`. Choice is D1 in `TODO.md`. **The T4 is
  sm_75 and has no native bf16** — which is awkward in a session about bf16, so whatever runs, say
  what precision the throughput numbers were measured in.
- Ties back to prior sessions explicitly: Session 9's scalar loss is where this session starts;
  §13's 16 bytes per weight is stated as the reason **Sessions 12 and 13 exist**; §6's optimiser
  state ("the two bars beneath each weight") is deferred to **Session 11**.
