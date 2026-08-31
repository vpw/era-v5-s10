# Session 10 — Assignment QnA

**Available** · Due **Sat, Sep 5, 2026, 7:00 AM** · **1000 points** · Resubmission allowed
Rubric tab: **empty** (no criterion rows, only the 1000-point total).

Source: https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/assignments/cmtdud74c173g09s136ltf9zh

## The assignment

Take a small model and a real loop, and make it tell you the truth about itself.

**Print every tensor shape** in the step, and write one line saying what each dimension means.

**Verify one gradient by hand.** Nudge a weight, measure how the loss changed, and compare against
what `backward()` reported. They should agree to several decimals, and if they do not, you have
found something worth understanding.

**Break gradient accumulation on purpose.** Use the average of the averages with micro-batches of
different lengths, and plot both curves together so you see the gap rather than take my word for it.

**Log the grad norm at every step**, then find one step where it moved before the loss did.

**Compute your own MFU**, report it honestly, and say what you believe is costing you the distance
to 40%.

**Take the number 0.1** and write out by hand what it looks like in fp32, bf16 and fp8 E4M3, showing
the bits. Then say which one you would train in, and why.

You're submitting GitHub Repo and README.md where these details are saved for our review. Your ipynb
(jupyter notebook) file should also be there

## Submission

- **GitHub README.md** — 1000 pts (single link field + short caption)
- Checkbox: *"I tested this link in an incognito window — it's publicly accessible (not private)."*
