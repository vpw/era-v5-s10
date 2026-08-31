# S10 TODO — The Training Loop

## ▶ STATUS (2026-08-29): scaffolded, work not started

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

- [ ] **D1. Where does the loop run?** There is **no GPU on this machine** (S9's `.venv` is torch
      cpu-only). Item 5 asks for **MFU**, a ratio against what the machine can do per second, and a
      CPU number is a legitimate answer only if the peak is *measured* (a large `matmul` benchmark)
      rather than quoted from a spec sheet for hardware not in use. **Both GPU lanes are now built
      and reusable** — global skill `era-v5-gpu-run` (`$CLAUDE_CONFIG_DIR/skills/era-v5-gpu-run/`):
      - **Lane A — the course's own EC2 T4.** `vardhan-gpu-1` = `i-025fb7b65d7e3460e`,
        `g4dn.2xlarge`, T4 16 GB, currently **stopped**. The same box S5's 8-arm ablation (3.9
        GPU-h) and S7's 6-arm run (~3 GPU-h) used. Fully scriptable:
        `scripts/run_remote_notebook.sh <session-dir> <nb>.ipynb [--stop]` does start → rsync →
        remote venv → build+execute on the box → pull back `.ipynb`/`results.json`/`logs`. Bills by
        the second; `gpu.sh stop` verifies `stopped` rather than assuming it.
      - **Lane B — Colab.** Portable-notebook cells in the skill's `assets/colab_cells.py`, the
        notebook opened straight from GitHub, and `scripts/extract_results.py` to recover
        `results.json` out of the downloaded `.ipynb`'s own printed output (tested — a notebook
        alone is sufficient). The user drives the browser; the agent prepares and receives.
      **Recommendation: Lane A**, since it is unattended and reproducible and the box already
      exists — with Lane B as the zero-cost fallback.
      **One caveat that matters for *this* session:** the T4 is **sm_75, which has no native
      bf16** (Colab's T4 likewise). §10 and §11 are exactly about bf16, so decide what item 5's
      MFU is measured in and say so — fp16 on a T4 is an honest number, bf16 on a T4 is a number
      about emulation. Item 6's bit patterns are arithmetic and unaffected.
- [ ] **D2. What model?** Reuse S9's proxy for continuity (`V = 10,000`, `D = 256`, 4 layers, 4
      heads, `T = 512`, the course's own S2 BPE `mr` tokenizer, already in `../../S9/assignment/
      assets/tokenizer.json`) so items 1 and 5 run on a real transformer rather than a toy MLP.
      Item 2's by-hand gradient still wants a **two-weight scalar chain** (the lesson's `w1=3, x=2,
      w2=4, t=20`) alongside it — verify by nudging a single real weight in the transformer too, so
      the check is on the model being trained, not only on a worked example.
- [ ] **D3. How is item 4's "norm moved before the loss" produced?** Either engineer a bad batch
      (the lesson's widget spikes the norm at step 24 and the loss at step 26) or find one in a
      natural run. Engineering it is honest as long as the notebook says so — but a natural one
      found in the logged trace is the stronger demonstration. Decide, then say which in the README.

## Part 1 — the six items

Each must be printed by a cell that actually ran. A number in the README with no cell behind it is
exactly the failure this session is about: *"a plausible number is not evidence."*

- [ ] **1. Every tensor shape in the step**, one line each naming what the dimension means —
      batch, sequence, model dim, vocab, and the shapes of the gradients as well as the activations.
- [ ] **2. Verify one gradient by hand.** Nudge a weight by a small ε, recompute the loss, and
      compare `(L(w+ε) − L(w))/ε` against `w.grad`. The lesson's own worked case agrees to **eight
      decimals** (`∂L/∂w1 = 64`). Use a central difference and state ε; if the agreement is poor,
      the disagreement *is* the deliverable — say what caused it.
- [ ] **3. Break gradient accumulation on purpose.** Micro-batches of **different token counts**,
      combined the wrong way (average of averages) and the right way (total loss ÷ total tokens),
      **plotted together**. The lesson's static case is 2.6000 vs 3.0000 = **15.4%**; show it both
      as that single-step number and as two training curves. Note in the write-up *why it hid* —
      the error vanishes when micro-batches hold equal token counts, which is what casual tests do.
- [ ] **4. Log the grad norm at every step**, then point at one step where it moved before the loss
      did. Log it from step one (§16 makes that a V5 decision, not a nicety).
- [ ] **5. Compute your own MFU** — `6 × N × tokens_per_second ÷ peak` — report it honestly, and say
      what costs the distance to 40%. Report `N`, measured tokens/s, achieved FLOP/s, the peak used
      and **where the peak number came from**. The lesson's point is that at 8% the loss curve looks
      exactly as it does at 45%, so a low number is a finding to explain, not a result to hide.
- [ ] **6. 0.1 in fp32, bf16 and fp8 E4M3, by hand, showing the bits.** Derive sign / exponent
      (with bias) / mantissa longhand, give the exact value each format actually stores and the
      representation error, then cross-check against `struct.pack` / `ml_dtypes` in a cell. Close
      with which one you would train in and why — the expected answer is bf16, for §10's reason
      (range beat detail), with fp8 named as the production recipe it now is.

## Ship

- [ ] **S1. Run the notebook top to bottom in a fresh runtime.** Keep the executed outputs in the
      committed `.ipynb` — the notebook is the README's evidence.
- [ ] **S2. Write `README.md`** — the six items with their numbers and explanations, plus the
      configuration, hardware and tokenizer they were measured on. Generate it from
      `README.tmpl.md` + `results.json` via S9's `tools/build_readme.py`, which exits non-zero on an
      unresolved placeholder, so the write-up cannot contain a number the last run did not produce.
      Edit the template, never `README.md`.
- [ ] **S3. Push** via subtree split (below), with the `.ipynb` and logs included.
- [ ] **S4. Verify the README link in an incognito window**, then tick the form's checkbox honestly.
- [ ] **S5. Submit** the GitHub README.md link and record it back in this file.

## Standing conventions (carried from S6-S9, don't re-decide)

**Build pipeline — the notebook is generated, not hand-edited.** Copy S9's `tools/` and keep
`notebook_src.py` (`# %%` cells) as the source of truth:

```
python tools/py2nb.py notebook_src.py S10_the_training_loop.ipynb
python tools/run_nb.py S10_the_training_loop.ipynb    # allow_errors=False -> writes results.json
python tools/dump_log.py S10_the_training_loop.ipynb logs/run.log
python tools/build_readme.py                          # README.tmpl.md + results.json -> README.md
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
