# Team email — draft

Replace `[Teammate]` with her name. The handbook is attached as a single HTML file
(`Maize_Prediction_Handbook.html`) — opens in any browser, works offline, no links needed.

---

**Subject:** Joining remotely tomorrow — analysis attached, let's lock in a call

---

Hi all,

I can't make it to the venue tomorrow, but I'll be online all day and can keep contributing in
real time. Can we set up a **Zoom or Discord** room? **What time are you starting, and which
platform?** I'll be on from the start.

Everything I've done so far is attached as a single HTML file — open it in any browser, no
internet needed. It's about an hour end to end and assumes no plant-breeding background, with
every term defined as it comes up. **If you're short on time, sections 9–14 are the results.**

Code is on the `data-audit` branch: https://github.com/AdvayBhatt/TriPlex/tree/data-audit

### Where it stands

A working pipeline from raw archives to a ranked advancement list with uncertainty intervals.
Against the real 2008 outcomes: **r = 0.191** (Cluster 1), **0.154** (Cluster 2). The closest
published study used this same commercial dataset and reports **0.06** for exactly this
scenario, so we're roughly 3× the published benchmark.

### Three things to know before going further

1. **The 2008 yields are in the data.** The docs say 2001–2007; it's actually 2000–2008 with
   outcomes included. We can grade against real truth instead of cross-validation — most teams
   won't be.
2. **Don't use the harvest traits as inputs.** Moisture, height, test weight and lodging are
   measured in October on the same plot as the yield being predicted. A model using them scores
   beautifully and can't be run on the actual January decision. Easiest way to invalidate a
   submission.
3. **Fifteen approaches tested, one improved on the baseline.** Neural nets, transformers,
   boosting, PCA and others all lost, each with a number attached. That section is probably
   worth as many marks as the model.

### Where I can help remotely

Tell me which you want me on — **spatial adjustment of the trials** (needs someone at the
venue to check whether the archive has plot coordinates), **within-family prediction** (two
thirds of the genetic variation is there and we barely touch it), or **the write-up and
submission package**, which is genuinely easier to own from a quiet room.

### [Teammate] — your R script

Ran it; there's a section near the end of the handbook on it. Three small things stop it
executing, and separately `shorthand_x` is the population (499) not the line (78,045), and
`tester` can't be fitted alongside it since every population uses exactly one. It also blows
R's 16 GB limit with those as fixed effects — switching to random effects fixes that and is
the right call anyway, since variance components need random terms.

Working version is in `R/`: environment 62%, population × environment 11%, population 5.6%,
residual 21.5%. **Your instinct about the interaction term was right** — it just needed to be
at population level, where it's actually estimable. Happy to walk through it on the call.

---

Everything reproduces from a clean checkout; `python scripts/demo.py` runs the whole pipeline
in ~3 seconds without the large files (that's also the "judge mode" the brief asks for).

Send me the platform and time and I'll be there.

Best,
Dhanush
