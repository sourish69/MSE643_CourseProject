# MSE643_CourseProject - Phase-Oriented Generative Design of High-Entropy Alloys

**A generative machine-learning pipeline that designs new alloy compositions to order - you specify the crystal phase you want, and the model proposes chemistries that should form it.**

---

## The idea in one paragraph

High-entropy alloys mix five or more principal elements in near-equal proportions, and their configurational entropy unlocks unusual combinations of strength and thermal stability. The problem is scale: the viable composition space runs to millions of candidates, far beyond what anyone can cast and characterize by hand. Most machine-learning work in this space runs *forward* - predict a phase from a composition - which only helps you rank alloys someone already thought of. This project runs the harder direction *backward*: given a target phase, generate brand-new compositions that should form it, then prove - using physics that no model can fake - that the generator actually learned real metallurgy rather than memorizing noise.

---

## Why this is hard, and what makes the result trustworthy

Small, noisy experimental datasets are exactly the conditions under which generative models quietly fail - either by copying the training set verbatim or by collapsing into producing the same handful of outputs regardless of what's asked for. Rather than take the generator's output on faith, this project is built around three independent checks that together make the results defensible:

- **A model-free physics check.** Two well-established metallurgical rules - the valence-electron-concentration rule for the solid-solution phases, and a thermodynamic mixing-energy signature for intermetallics - are computed directly from a proposed composition's chemistry. Neither rule involves a trained model, so they can't be gamed; they either hold or they don't.
- **A negative control.** A generator that ignores the requested phase entirely and proposes compositions at random is run through the same evaluation. If the "real" generator's phase-conditioned outputs don't look meaningfully different from this baseline, the conditioning isn't doing anything - this is the test that proves the separation is genuine signal, not an artifact of the chemistry itself.
- **A strong non-learning baseline.** A second baseline builds new compositions purely by blending real training alloys of the requested phase - no neural network involved. This sets a high bar: a learned generator only earns its place if it matches or exceeds what simple interpolation between known alloys already achieves.

---

## Workflow

### 1. Consolidating the data
Several independently compiled public datasets are combined into one training set - not by simple concatenation, but through a careful pipeline that standardizes each source's own composition notation and phase vocabulary into one common form, computes a consistent set of physics descriptors for every entry (recomputing from scratch wherever a source's own descriptors were missing, incompatible, or simply wrong), and then deduplicates aggressively across sources. The same alloy is often written differently by different authors - one dataset's shorthand equiatomic notation is another's explicit percentage listing - so before anything can be deduplicated, every composition is rewritten into one canonical, uniform form. Compositions that appear in multiple sources with disagreeing phase labels are resolved by majority vote rather than silently dropped. The result is roughly a tenfold increase in usable training data over any single source, with the most dramatic gain landing on the previously scarce dual-phase class - the exact class a small dataset would have starved.

### 2. Building a shared representation
Every alloy is reduced to two aligned views, generated from the same underlying logic so they can never drift out of sync: a fraction vector over a data-driven element palette (an element only earns a slot in the palette if it appears often enough across the merged data to be learnable), and a compact set of physics descriptors computed from that same composition. The palette itself is discovered from the data rather than assumed in advance, and the whole descriptor engine is verified - before it's trusted anywhere downstream - by checking that it reproduces known reference values with high fidelity.

### 3. The discriminative side: a physics-literate classifier
A gradient-boosted classifier is trained to predict phase from the physics descriptors, evaluated with a cross-validation scheme that deliberately keeps chemically-related alloy families together rather than scattering near-duplicates across the training and test sets - a much harder, more honest test than a naive random split. But the classifier's real job here isn't the headline accuracy number; it's interpretability. A feature-attribution analysis is used to confirm that the classifier's decisions line up with known metallurgy - that it separates the solid-solution phases along the same electronic-structure rule a materials scientist would use, and that it identifies intermetallics through the same thermodynamic signature theory predicts. A model can look accurate for the wrong reasons; this step checks that it isn't.

### 4. The generative side: a conditional generative model
A generative model conditioned on the desired phase learns to propose new compositions - the phase label is fed into the model deeply enough that it cannot simply learn to ignore the condition, and its output layer is constructed so that every generated composition is chemically valid by construction (non-negative element fractions summing to one, with no post-hoc correction needed).

Getting this model to actually work was itself a diagnostic exercise worth documenting honestly. An early version suffered a well-known generative-modeling failure mode in which the model technically trains but quietly stops using the very mechanism that lets it generate anything beyond a narrow, repetitive set of outputs - invisible in the loss curve, but glaring once diversity is measured directly. Catching it required a purpose-built check that compares the spread of generated compositions against the spread of the real training data for that class; fixing it required rebalancing exactly how strongly the model is penalized for deviating from its prior, together with a floor that guarantees a minimum amount of information is always carried through. Once corrected, the model's generative capacity recovered across essentially the entire phase space - the kind of collapse-and-recovery story that's far more informative than a report that simply never mentions it happened.

### 5. Filtering and validating what comes out
Every generated composition - regardless of which generator produced it - passes through the same evaluation stage. Compositions are checked against the physics feasibility criteria for solid-solution formation; the phase-conditioning is validated model-free via the electronic-structure rule; novelty is checked against the training set (and always read alongside feasibility, since a generator that ignores physics entirely will always score "novel" simply by producing garbage); and the discriminative classifier is asked to independently re-read each generated composition's phase, checking whether what was requested is what plausibly came out.

### 6. Comparing generators honestly
The learned generator, the interpolation-only baseline, and the random negative control are all pushed through the identical evaluation pipeline and reported side by side, rather than reporting the learned model's numbers in isolation. This is what allows an honest verdict rather than a self-congratulatory one: the results show the learned generator clearing the random floor by a wide margin, and largely matching - without decisively beating - the interpolation baseline. That is reported as the finding it is, not softened, because a defensible modest result is worth more than an inflated one that collapses under scrutiny.

---

## What the project ultimately shows

- A verified physics engine that reproduces reference descriptor values with high fidelity, serving as one consistent source of truth for both training and generation.
- A discriminative classifier whose decision logic independently recovers textbook metallurgical rules, confirmed through feature attribution rather than assumed.
- A generative model that, once a genuine training failure was diagnosed and corrected, produces valid, phase-appropriate, reasonably diverse new alloy compositions.
- A rigorous three-way comparison - random floor, learned generator, interpolation baseline - that turns "the model generates alloys" into a measured, falsifiable claim rather than an assertion.

---

## Honest scope and limitations

This is a computational proof of concept, not validated materials discovery. No generated composition has been synthesized or physically tested - every claim here is about consistency with established metallurgical theory, not experimental confirmation. The training data, even after consolidation, remains modest for the scale generative models typically expect, and merging multiple independently compiled sources inevitably introduces some label disagreement between them, which is resolved by majority vote rather than eliminated. The physics checks rest on established empirical rules rather than first-principles simulation. And the generative model's central result - that it matches but does not clearly surpass a much simpler interpolation baseline - is reported as exactly that: a real, informative finding about the limits of what a compact generative model can learn from a dataset of this size, and a natural target for improvement as more data and more computational validation become available.
