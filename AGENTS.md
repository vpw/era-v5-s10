# Task

This directory is part of the assignments for the ERA V5 course of The School of AI (TSAI).
Specifically this is for the tenth session (S10).

The `S10-assignment.md` file lists the exercise in full — please refer to that for the details.
Like S9 and unlike S7/S8, this is a **notebook deliverable**: a GitHub repo whose `README.md` is the
graded artifact, with the `.ipynb` in the same repo backing every number the README claims. The
link must be publicly accessible in an incognito window. No deployment step.

# Details

The session is **The Training Loop**: how one scalar loss reaches back and moves billions of
weights, and how you know a run is healthy while it goes for weeks. It covers the vocabulary
(gradient / step / batch / micro-batch); the gradient as the answer to a nudge; backpropagation as
the chain rule applied one link at a time; autograd as bookkeeping, not magic; the five-line step
(`forward → loss → backward() → optimizer.step() → zero_grad()`) and why gradients accumulate rather
than replace; gradient accumulation, which turns that into a feature; **the accumulation bug that
lived in every major framework until 2024** (average-of-averages over unequal-length micro-batches);
floating point (sign/exponent/mantissa, range vs detail); **why bf16 replaced fp16** despite being
less accurate; the 2026 formats (fp8 E4M3/E5M2, fp4 E2M1, block scaling); **gradient clipping and
the grad norm**, which moves before the loss does; **16 bytes per weight** of training state; **MFU**;
and V4's quiet failures.

Numbers worth having in hand: the by-hand chain gives `∂L/∂w1 =` **64**, matching a numeric nudge to
**eight decimals**; the accumulation bug is **2.6000 correct vs 3.0000 wrong = 15.4%**; bf16 keeps
all **8 exponent bits** (floor `9.18×10⁻⁴¹`) where fp16 has 5 (dies below ~`5.96×10⁻⁸`, rescued by
×1024 loss scaling); fp4 block scaling costs `(16×4+8)/16 =` **4.5 bits/value**; clipping `norm 8.4,
cap 1.0 → ×0.119`; training state is **16 bytes/weight** (2+2+4+8), so an 80 GB card holds ~**5.4B**
weights and nothing else; `MFU = 6·N·tokens_per_sec ÷ peak`, worked out as **8.2%** for a 9B model at
12,000 tok/s on eight H100s, against a healthy **35–50%**.

Full writeup: `resources/s10-session.md`. Live-class transcript: `resources/s10-transcript.md`.

**What the assignment actually asks for** — six items, all in one notebook on a small model and a
real loop, *"and make it tell you the truth about itself"*:

1. **Print every tensor shape** in the step, with one line naming what each dimension means.
2. **Verify one gradient by hand.** Nudge a weight, measure the loss change, compare against what
   `backward()` reported — they must agree to several decimals. A disagreement is a finding, not a
   failure: say what it was.
3. **Break gradient accumulation on purpose.** Average-of-averages over micro-batches of *different
   lengths*, plotted against the correct token-weighted curve, so the gap is visible rather than
   asserted.
4. **Log the grad norm at every step**, then find one step where it moved before the loss did.
5. **Compute your own MFU**, report it honestly, and say what you believe costs you the distance
   to 40%.
6. **Write out 0.1 by hand in fp32, bf16 and fp8 E4M3**, showing the bits, then say which you would
   train in and why.
7. **Submit** a GitHub README.md link (incognito-accessible) with the notebook in the repo.

**The graded skill is instrumentation, not code.** The lesson's closing line is the spec: *"Print
things and check things. Every serious training bug is silent, and the loss curve is not going to be
the one that tells you."* Items 2, 3, 4 and 5 each exist because one specific failure produces a
perfectly plausible loss curve — a wrong gradient, a wrong accumulation weighting, an impending
divergence, and a machine running at a fifth of its speed.

**Two practical constraints before planning.** (a) There is **no GPU on this machine** — item 5's
MFU is a ratio against a device peak, so decide where the loop runs and, on CPU, measure the peak
rather than quoting a spec sheet for hardware not in use. (b) Item 4 needs a step where the norm
genuinely leads the loss; decide whether to engineer a bad batch (as the lesson's widget does at
step 24 vs 26) or to find one in a natural run — and say which you did.

**If the write-up ends up citing sources** (the 2024 gradient-accumulation bug, NVFP4 throughput,
activation checkpointing), use the `arxiv-library` skill rather than recall: discover via its arxiv
MCP layer, download the PDF into the local library so the source is a checkable file, and index via
`rag-toolkit` when a claim needs pulling out of the PDF text with a citation. This is optional here —
the deliverable is measurement.

As a capable agent, plan to: (1) read `resources/s10-session.md` §§3-14 for the mechanics each item
has to demonstrate, and the transcript for the instructor's framing of the silent-bug point,
(2) settle the open decisions in `TODO.md` (where the loop runs, what model, how the MFU peak is
obtained), (3) reuse S9's generated-notebook pipeline (`../../S9/assignment/notebook_src.py` +
`tools/`) so every README number traces to a cell that ran, (4) run the notebook end-to-end top to
bottom in a fresh runtime, (5) write the README with the six items and their explanations, (6) push
to GitHub and verify the link in an incognito window. TODO.md tracks progress on these steps.

## References
Refer CLAUDE.md if it exists
