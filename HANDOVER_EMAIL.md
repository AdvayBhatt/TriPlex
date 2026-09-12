# Team email — draft

Replace `[Teammate]` with her name, and confirm the report link is shared before sending
(artifacts are private by default — open the page and use the share menu).

---

**Subject:** Joining remotely tomorrow — analysis so far, and let's lock in a call

---

Hi all,

I won't be able to make it to the venue tomorrow, but I'll be fully available online
throughout and can keep contributing in real time. Happy to set up a **Zoom or Discord** room
so I can join standups, pair on whatever's blocking, and turn things around quickly — whatever
works best for the rest of you. **What time do you want to start, and which platform?** I'll
be on from the beginning and will stay reachable all day.

To make the remote thing work, here's everything I've done so far, written up properly so
nobody has to wait on me to get moving.

**Full write-up:** https://claude.ai/code/artifact/57451d72-2462-4ce7-84c5-ec3901e62ce1

**Code:** the `data-audit` branch — 10 commits, pushed and ready
(https://github.com/AdvayBhatt/TriPlex/tree/data-audit)

The write-up is about an hour's read and assumes no plant-breeding background. Every acronym
and statistical concept is defined as it comes up, with a glossary at the end. Sections 1–4
are background; if you're short on time, **sections 9–14 are the results and what they mean**.

### Where things stand

There's a working end-to-end pipeline that goes from the raw archives to a ranked advancement
list with uncertainty intervals. Against the real 2008 outcomes it reaches **r = 0.191** in
Cluster 1 and **0.154** in Cluster 2.

For context: the closest published study — which turns out to have used this same commercial
dataset — reports **0.06** for exactly this scenario. So the result is roughly three times the
published benchmark, and it also beats that study's *within-family* figure while solving a
harder problem.

### Three things worth knowing before you go further

**1. The 2008 yields are in the data.** The documentation says 2001–2007; it's actually
2000–2008 with the outcomes included. That means the model can be graded against real truth
rather than cross-validation — very few teams will be doing that.

**2. Don't use the harvest traits as model inputs.** Moisture, plant height, test weight and
lodging are all measured in October, on the same plot as the yield being predicted. A model
using them scores beautifully and cannot be run on the actual January decision. It's the
single easiest way to invalidate a submission, and it's an easy mistake to make.

**3. Fifteen approaches were tested; one improved on the baseline.** Neural networks,
transformers, gradient boosting, PCA compression, haplotype methods and several others all
lost — each with a measured number attached. That "here's what we tried and why it failed"
section is probably worth as many marks as the model itself, because it's the part most teams
can't produce.

### Where I can be most useful tomorrow

Happy to take any of these remotely — tell me which you want me on:

1. **Spatial adjustment of the unreplicated trials** — the untried lever with the clearest
   rationale. It attacks plot error directly, which is what attenuates every correlation in
   the analysis. Needs plot row/range coordinates; someone at the venue could check whether
   the archive has them.
2. **Within-family prediction** — two-thirds of the genetic variation sits there and the
   current model barely touches it (r = 0.11). The largest unexploited opportunity, and the
   hardest.
3. **The write-up and submission package** — this is genuinely easier to do remotely than in a
   noisy room, so it's a natural thing for me to own if that helps.

### [Teammate] — on your R script

I ran it, and there's a section near the end of the report devoted to it. Short version:

- Three small things stop it executing — an extra closing bracket, an `lmer()` call with no
  random-effects term, and `anova()`/`update()` referencing `c1_gxe` when the object created
  was `c1_fixedG_gxe`.
- Separately, `shorthand_x` is the **population** (499 values), not the line (78,045), and
  `tester` can't be fitted alongside it because every population uses exactly one tester.
- It also exhausts R's 16 GB memory limit with those terms as fixed effects. Switching them to
  **random** effects fixes that and is the correct choice anyway — `VarCorr()` reports variance
  components, and a term has to be random to have one.

A working version is in `R/` and gives a clean decomposition: environment 62%,
population × environment 11%, population 5.6%, residual 21.5%. **Your instinct about the
interaction term was right** — it just needed to be at population level, where there are ~169
plots per cell and it's actually estimable. At line level there's one plot per cell, so it's
mathematically confounded with error. Happy to walk through it on the call if that's easier
than reading it.

### Reproducing it

Everything runs from a clean checkout. `python scripts/demo.py` runs the entire pipeline in
about three seconds on generated data, without needing the 141 MB source files — that also
doubles as the "judge mode" the brief asks for.

Let me know the platform and start time and I'll be there. Looking forward to it.

Best,
Dhanush
