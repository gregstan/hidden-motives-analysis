# `individual_architecture_analysis.md`

# Individual Utility-Architecture Analysis

> **Purpose:** build the machinery needed to answer a new question that sits directly underneath the paper’s “shared coordinate system” claim:
>
> **Do participants differ mainly by their parameter values inside one shared utility architecture, or do different participants require different utility-function architectures?**

This file is written as instructions for the coding agent working inside the **Inferring Hidden Motives** codebase. The agent should also read:

- `ihm_starter_pack.md` — high-level conceptual and technical overview of the paper.
- `model_recovery_simulation.md` — model-recovery and **Average Model Policy Distance** plan.

This file focuses on a new individual-level extension of the full-model utility-function IC analysis.

---

Here’s a copy-pasteable front section you can place near the beginning of `individual_architecture_analysis.md`. I wrote it as exposition first, implementation implications second, with minimal math and no insider phrasing.

This is grounded in the current project context: the repo implements the UBM, IC analysis, utility-function enumeration, 625-payoff grid, `make_param_info`, and AMPD-related machinery.   

---

# Individual Utility-Architecture Analysis: Conceptual Overview

> **Core question:**
> Do participants differ mainly because they have different *parameter values* inside one shared utility function, or because different participants are better described by different *utility-function architectures*?

This analysis extends the main IHM paper by testing a key assumption behind the population-level utility-function results.

The IHM paper begins with the problem of **inferring hidden motives**. In repeated binary dictator games, participants observe others choosing between payoff allocations and try to infer the social preferences behind those choices: self-interest, altruism, envy, guilt, nonlinear payoff sensitivity, and related motives. The **Utility Bayesian Model** (UBM) formalizes this as a Bayesian learning problem. A predictor begins with uncertain beliefs about a chooser’s latent social-preference parameters. After observing each choice, the predictor updates those beliefs and uses them to predict future choices.

The UBM needs a utility function inside its likelihood term. That utility function says, given a parameter vector and a payoff structure, how attractive Option A is compared to Option B. The existing Information Criterion (IC) analysis searches over the full utility-function universe and identifies the best-fitting population-level utility form. Once that population-winning form is selected, the paper reports parameter distributions in that shared model space: how much people value their own payoffs, others’ payoffs, advantageous inequality, disadvantageous inequality, nonlinear payoffs, and so on.

That is the paper’s **coordinate-system** idea:

> Once we know the right utility architecture, every individual can be located somewhere inside that shared psychological space.

But this interpretation rests on a major assumption:

> **Participants share the same utility-function form and differ only in parameter values.**

The individual architecture analysis tests that assumption.

---

## The Teeter-Totter: One Architecture vs. Many Architectures

There are two extremes.

At one extreme, we use **one common utility architecture** for everyone. This is maximally parsimonious. It says that all participants live inside the same utility-function space, and individual differences are captured by different parameter values within that space.

At the other extreme, we allow **every participant to have their own best-fitting utility architecture**. This maximizes flexibility and fit, but risks overfitting, noise, and loss of interpretability. A result like this would be hard to understand:

> Participant 1 is best described by model 233.
> Participant 2 is best described by model 17.
> Participant 3 is best described by model 402.
> …

That list is not scientifically meaningful unless we know whether those utility functions are behaviorally similar or radically different.

The goal is not to blindly choose one extreme. The goal is to measure the tradeoff between **parsimony** and **individual variation**.

That is the purpose of the **Utility-Architecture Compression Analysis**.

---

## Utility-Architecture Compression Analysis

The compression analysis asks:

> **How many distinct utility architectures are needed to describe the participant population?**

Let `K` be the number of utility architectures we allow ourselves to use.

* `K = 1` means one shared architecture for everyone.
* `K = N` means every participant can have their own best-fitting architecture.
* Intermediate values, like `K = 2`, `K = 3`, or `K = 4`, mean we are using a small **codebook** of utility architectures.

The analysis slides `K` between those extremes and tracks predictive improvement. If allowing more architectures produces a large improvement at first but then quickly levels off, that suggests a small number of utility-architecture families may capture most meaningful individual differences.

Possible outcomes:

### **Outcome 1: One shared architecture is enough**

If the curve barely improves after `K = 1`, then the population-winning utility function is probably a genuine shared coordinate system. Participants differ mainly in parameter values, not in functional form.

### **Outcome 2: A small number of stable architecture families**

If the curve improves sharply at `K = 2` or `K = 3` and then levels off, then the data may support a small number of functional phenotypes. For example, one cluster of participants may require social-comparison terms, while another may be well captured by self-interest and altruism alone.

### **Outcome 3: Gradual improvement without a clear structure**

If fit improves gradually all the way toward `K = N`, then individual architecture may be too idiosyncratic or underidentified to summarize cleanly. This does **not** necessarily mean every person has a psychologically distinct architecture. It may mean the participant-level data are noisy, sparse, or insufficiently diagnostic.

### **Outcome 4: Scatter without predictive gain**

If participant-specific architectures look scattered but do not improve held-out prediction, then the scatter is probably not meaningful. It is likely overfitting or underidentification.

The strongest evidence for individual utility-architecture differences would be:

> **stable, predictive, interpretable clusters**
> not merely scattered top-model IDs.

---

## Why We Need a Similarity Metric Between Utility Functions

Before we can ask whether participants cluster into architecture families, we need to know how similar the utility functions are.

A model ID is just a label. Model 233 and model 17 might be close cousins, or they might be completely different behavioral theories. Symbolic differences alone are not enough. Two models can look different algebraically but behave almost identically across the payoff grid. Conversely, two models can differ by only one setting but generate very different choice policies.

So we need a behavioral similarity metric.

That metric is **Average Model Policy Distance**.

---

## AMPD: Average Model Policy Distance

**AMPD measures the average difference in behavior between two utility models in the same situation given the same parameters.**

More concretely:

1. Take two utility functions: Model A and Model B.
2. Draw one full canonical parameter vector.
3. Give that same parameter vector to both models.
4. Let each model use whichever parameters it needs and ignore the rest.
5. Run both models across the same payoff structures, usually the full set of `625` payoff structures from `{1, 2, 3, 4, 5}^4`.
6. Record each model’s choice probability for every game.
7. Measure the divergence between those choice probabilities.
8. Repeat this process many times with new random parameter draws.
9. Average the divergences.

The key rule is:

> **Within a single AMPD iteration, both models receive the same parameter vector.**

Across iterations, the parameter vector changes. But within an iteration, the two models are evaluated under matched conditions.

This matters because self-distance must be zero:

```text
AMPD(model_x, model_x) = 0
```

If Model X is compared with itself under the same payoffs and the same parameters, it should behave identically. If self-distance is nonzero, the function is not measuring pure model-policy distance. It is measuring something else: variability between two independently sampled agents from the same model family.

That alternative quantity may be useful someday, but it is not the primary AMPD needed for model-space geometry.

---

## Why Sample All Nine Parameters?

Different utility functions use different parameter subsets. Some use self-interest and altruism. Some include envy and guilt. Some include exponents. Some fix self-interest. Some use conditional welfare or min-max forms.

To compare models fairly, each AMPD iteration samples a full canonical parameter vector containing all possible parameters. Then each model takes what it needs.

For example:

* A model without altruism ignores the altruism parameter.
* A model without social comparison ignores envy/guilt parameters.
* A model with one exponent uses the relevant exponent.
* A model with three exponents uses all three.
* A model with fixed self-interest ignores the free self-interest parameter.

This creates a shared reference coordinate system for comparing model behavior.

It is not claiming that every parameter has exactly the same psychological meaning in every model family. Conditional-welfare models, for example, may use parameters somewhat differently than standard additive models. But the shared-vector approach gives us the cleanest practical definition of behavioral similarity:

> Given the same reference parameter draw and the same payoff situation, how differently do these utility architectures behave?

---

## Uniform vs. Realistic Parameter Sampling

AMPD can be computed under different parameter-sampling regimes.

### **Uniform AMPD**

Uniform AMPD samples parameter values broadly from their allowed bounds.

This asks:

> Across the broad theoretical parameter space, how similar are these utility functions?

Uniform AMPD is more theory-neutral. It does not assume that the empirically observed participant distribution is the only meaningful region of parameter space.

### **Realistic AMPD**

Realistic AMPD samples parameter values from empirical parameter distributions, using the fitted population distributions from the current results. If a model requires a parameter not present in the empirical distribution, that missing parameter is sampled from its allowed bounds.

This asks:

> In the region of parameter space humans appear to occupy, how similar are these utility functions?

Realistic AMPD may be more psychologically relevant for interpreting participant architecture. Uniform AMPD may be better for general model-space cartography. Both are worth computing.

---

## JSD, Cross-Entropy, and Why the Metric Depends on the Question

AMPD is primarily a **symmetric model-similarity metric**. Neither utility function is treated as the truth. We are asking how different their induced choice policies are.

For that purpose, normalized **Jensen–Shannon divergence** is useful because it is:

* symmetric,
* bounded,
* interpretable as a distance-like divergence between probability distributions,
* appropriate for building distance matrices,
* appropriate for MDS embeddings and clustering.

So normalized JSD is the primary metric for model-space geometry.

Cross-entropy and negative log-likelihood serve a different purpose. They are directional. They answer:

> If this model is trying to predict some target behavior, how surprised is it by the observed response?

That is the right logic when one side is treated as ground truth, such as:

* human choices,
* human predictions,
* synthetic data generated by a known model,
* held-out test responses.

So the clean division is:

> **JSD for symmetric model-to-model similarity.**
> **Cross-entropy / NLL for directional prediction.**

---

## From Utility Functions to a Model-Space Map

Once we compute AMPD between every pair of utility functions, we have a full utility-function similarity matrix.

This matrix can be used to embed the utility functions in a lower-dimensional space using MDS. In that space:

* each point is a utility function,
* nearby points are behaviorally similar,
* distant points are behaviorally different.

This model-space map lets us interpret the IC analysis more intelligently.

For example, suppose the top 25 BIC-ranked models all occupy the same small region of AMPD space. That means the population IC winner is not an isolated accident. It is the best representative of a broader behavioral family.

But suppose the top-ranked models split into two distant regions. That would mean the data support multiple behaviorally distinct explanations. That would be scientifically interesting and might motivate future experiments designed to distinguish those families.

The model-space map turns a flat ranking table into a landscape.

---

## From Model Points to Participant Clouds

For each participant, we can compute or extract a fit score for every utility function.

This gives each participant a row like:

```text
participant_i × utility_model_m → BIC_i,m
```

Lower BIC means that utility architecture fits that participant better.

But we should not represent each participant only by their top-ranked model. Individual-level model selection is noisy because each participant has far fewer observations than the full population. A participant’s best model may be unstable, while their broader region of plausible models may be meaningful.

So we convert each participant’s model-specific BIC values into weights.

Models with low ΔBIC get high weight. Models with high ΔBIC get low weight.

Now each participant becomes a **BIC-weighted cloud over utility-function space**.

This is the conceptual move:

> Participants are not points.
> Participants are clouds over utility architectures.

The cloud tells us which regions of model space are plausible for that participant.

---

## Clouds, Centroids, and Elastic Bands

Once utility functions are embedded in MDS space, each utility function is like a planet.

Each participant has elastic bands connecting them to those planets. The strength of each elastic band is the participant’s BIC weight for that model.

* If one model dominates, the participant is pulled close to that model.
* If several nearby models are plausible, the participant sits near that local region.
* If many distant models are plausible, the participant’s cloud is diffuse.

For visualization, we can summarize the participant cloud as a weighted centroid in MDS space. But the centroid is only a visual shorthand. The scientifically honest object is the full cloud of model weights.

This distinction matters.

A participant with a tight cloud near one model family is architecturally well-identified. A participant with a diffuse cloud is not necessarily psychologically strange. They may simply not have enough diagnostic data to identify a utility architecture.

---

## What `H_form` Does

`H_form` is the simplest predictive test of whether individual utility architectures matter.

It asks:

> **How much predictive improvement do we gain by allowing participants to have their own utility-function forms, compared to forcing everyone to use one common form?**

The comparison is between three reference points.

### **1. Common architecture**

Everyone uses the same utility-function form, but each participant gets their own fitted parameters.

This tests the “shared coordinate system” hypothesis.

### **2. Individual architecture**

Each participant can use their own best-fitting utility-function form and their own fitted parameters.

This tests the “different people may need different architectures” hypothesis.

### **3. Chance**

A chance model predicts `0.5` for every binary response.

This gives the denominator needed to make the improvement interpretable.

`H_form` is useful because it does not just say whether individual architectures fit better in raw NLL. It asks whether the improvement is meaningful relative to a chance baseline.

Interpretation:

* `H_form ≈ 0`: individual utility forms add little or nothing beyond one common architecture.
* `H_form > 0`: participant-specific utility forms improve prediction.
* `H_form < 0`: participant-specific forms generalize worse, likely due to overfitting or instability.
* large `H_form`: utility-function heterogeneity is doing meaningful predictive work.

Ideally, `H_form` should be computed with held-out prediction. If the same data are used to select the individual form and evaluate it, the individualized model will usually look better simply because it is more flexible. Cross-validation is what separates genuine predictive value from overfitting.

---

## Utility-Architecture Compression Analysis

`H_form` compares two extremes:

```text
K = 1          one common architecture
K = N          one architecture per participant
```

The compression analysis asks the next question:

> **If individual architectures help, how many architectures are enough?**

Instead of jumping directly from one architecture to full individualization, we allow a codebook of `K` architectures.

At each value of `K`, the analysis selects `K` actual utility functions as representative architectures. These are better thought of as **medoids** than centroids, because they are real utility functions, not abstract average points.

Each participant is assigned to the codebook model that predicts them best. Then we measure predictive performance.

As `K` increases, prediction should improve. But the scientific question is how quickly it improves and where it levels off.

Possible patterns:

* If `K = 1` performs nearly as well as larger `K`, one common utility architecture is enough.
* If `K = 2` or `K = 3` captures most of the improvement, participants may fall into a small number of functional phenotypes.
* If performance improves gradually all the way to `K = N`, then there may be idiosyncratic heterogeneity without a clean taxonomy.
* If improvement appears in-sample but disappears under cross-validation, then the extra architectures are probably overfitting.

This analysis directly quantifies the tradeoff between parsimony and individual variation.

---

## Data Adequacy: Did We Collect Enough Data?

A secondary question is whether the dataset contains enough information to answer the individual architecture question.

This can be evaluated by simulation and/or resampling.

The simulation version creates synthetic participants under known assumptions, varies the number of participants and/or games per participant, and asks when key metrics stabilize. For example, we can track whether architecture recovery, AMPD-based participant clustering, `H_form`, or the compression curve reaches a stable region as synthetic sample size increases.

The tricky part is choosing realistic synthetic parameters. There are two defensible approaches:

### **Empirical-realistic simulation**

Sample synthetic participant parameters from the empirical population parameter distributions estimated from the current data. This makes the simulation psychologically grounded, but it inherits assumptions from the fitted model.

### **Uniform / broad stress-test simulation**

Sample parameters broadly from allowed bounds. This is more theory-neutral, but may generate participants unlike real humans.

Both are useful. If the analysis stabilizes under both, that is reassuring. If it stabilizes only under one, that tells us something about the dependence of the result on parameter assumptions.

The data-adequacy question should be treated as a calibration layer, not the main analysis. The central question is still whether real participants are best described by one shared architecture or multiple architectures.

---

## The Three Validation Questions

This project now has three related but distinct validation questions.

### **1. Parameter recovery**

Given a known utility form, can the optimizer recover the true parameters used to generate synthetic data?

This validates the parameter-fitting machinery.

### **2. Utility-form recovery**

Given synthetic data generated from a known utility function, can the IC pipeline recover the true utility form, or at least a behaviorally similar / nested-equivalent form?

This validates the utility-function model-selection machinery.

### **3. Individual architecture analysis**

Given real human data, are participants best described by one shared utility architecture, a small number of architecture families, or many idiosyncratic forms?

This is not just validation. It is a substantive analysis of individual differences.

AMPD is useful across these questions because it gives us a way to say whether two utility forms are behaviorally similar, even when their symbolic equations differ.

---

## What This Analysis Can Reveal

The results should be interpreted through several possible outcomes.

### **One shared basin**

Participant clouds cluster near the population-winning utility function. `H_form` is near zero. The compression curve shows little improvement after `K = 1`.

**Interpretation:**
The population-winning utility function is a strong shared coordinate system. Individual differences are mostly parameter differences.

### **A few stable architecture clusters**

Participant clouds split into a small number of stable, interpretable regions. `H_form` is positive. The compression curve levels off at small `K`.

**Interpretation:**
Participants may differ not only in parameter values but in utility architecture. This would suggest functional phenotypes of social preference.

### **Diffuse scatter without predictive improvement**

Participant top models vary widely, but model clouds are diffuse and individualized forms do not improve held-out prediction.

**Interpretation:**
Individual-level architecture is likely underidentified or noisy. The population model remains the better-supported summary.

### **Diffuse scatter with predictive improvement**

Individualized forms improve prediction, but there are no stable clusters.

**Interpretation:**
There may be idiosyncratic architecture differences, but the dataset may not support a clean taxonomy.

---

## Practical Guidance for the Coding Agent

The implementation should preserve the conceptual distinctions above.

Most importantly:

> **Do not treat a participant’s fitted parameter vector as their architecture.**
> Architecture is about which utility functional forms fit that participant.

A participant should be represented as a BIC-weighted cloud over utility functions, not merely as their top-ranked model.

Similarly:

> **Do not compute AMPD with independently sampled parameters for the two models in a pair.**

Primary AMPD must use the same full parameter vector for both models within each iteration. Otherwise, self-distance becomes nonzero and the matrix stops being a model-similarity matrix.

Also:

> **Do not make dynamic Bayesian updating part of AMPD.**

AMPD should compare static utility-to-choice policies. If dynamic updating is included, the distance becomes partly about assumptions of the UBM, priors, learning trajectories, and observation histories. That is a different analysis. AMPD should isolate the behavioral similarity of utility architectures themselves.

Finally:

> **The main scientific target is not a pretty MDS plot.**

The MDS plot is a visualization tool. The real questions are:

1. Are the top IC models behaviorally coherent?
2. Are participant model clouds concentrated near one architecture or multiple regions?
3. Does allowing individualized architectures improve held-out prediction?
4. How many architectures are needed to capture that improvement?
5. Are any clusters stable and interpretable?

If the answer is “one shared architecture,” that strengthens the paper’s coordinate-system claim. If the answer is “a small number of architecture families,” that may reveal a deeper taxonomy of social-preference computation. Either result is scientifically valuable.

The point is to turn a possible limitation of the population-level IC analysis into a direct empirical question:

> **Is the population-winning utility function a universal coordinate system, or the first approximation to a richer map of human motivational architecture?**

---

## 0. Implementation principle

> **Reuse existing code aggressively.**

Before writing new machinery, inspect the existing codebase for functions that already do any of the following:

- generate the valid utility forms;
- evaluate a utility function on binary dictator-game payoffs;
- compute SoftMax choice probabilities;
- compute raw NLL;
- fit utility parameters with simulated annealing and/or L-BFGS-B;
- run the existing IC analysis;
- save per-model losses, parameters, AIC, BIC, and ranks;
- handle chooser and predictor roles;
- retrieve model metadata, including `k`, 14 Boolean utility settings, model IDs, equations, parent-child relations, and sibling relations;
- compute or retrieve **Average Model Policy Distance**;
- generate the exhaustive 625 payoff grid.

Do **not** reimplement any of this unless the codebase truly lacks a reusable component. This project already contains a large amount of carefully tested machinery, especially around utility forms, nesting, and IC analysis. The goal is to extend and reorganize that machinery, not duplicate it.

---

## 1. Scientific motivation

The current paper compares **the full valid utility-function universe** and identifies a best-fitting seven-parameter utility architecture. That winning form separates self-interest, altruism, envy, guilt, and nonlinear payoff sensitivity, and it supports the paper’s claim that the resulting parameter estimates form a shared psychological coordinate system.

But the current IC analysis is primarily a population-level or shared-form analysis. That leaves a major question open:

> **Is the winning seven-parameter form a genuinely shared architecture, or is it a population-level compromise over people with different utility architectures?**

This analysis is designed to answer that.

### The core theoretical contrast

There are two competing pictures of individual differences:

#### **A. Shared architecture, different parameters**

Everyone is well described by the same utility form, but people differ in their parameter values.

Example:

```text
Participant 1: same 7-parameter utility form, high self-interest, low guilt
Participant 2: same 7-parameter utility form, moderate self-interest, high guilt
Participant 3: same 7-parameter utility form, low self-interest, high altruism
```

This supports the paper’s “shared coordinate system” interpretation.

#### **B. Different architectures, different parameters**

Different participants are better described by meaningfully different utility forms.

Example:

```text
Participant 1: self-interest + altruism model
Participant 2: self-interest + social-comparison model
Participant 3: nonlinear envy/guilt model
Participant 4: reference-dependent model
```

This would suggest that people differ not only in **where** they fall within a parameter space, but in the **structure of the space itself**.

### Why this is worth doing

Before we had a behavioral distance metric between utility forms, individual-level utility-form fitting would have produced a hard-to-interpret list of winners:

```text
participant_001 -> model_037
participant_002 -> model_411
participant_003 -> model_089
...
```

That is not very meaningful by itself.

Now, with **Average Model Policy Distance**, we can ask whether those models occupy the same region of behavioral model space, form stable clusters, or scatter randomly. That makes individual-level utility-architecture analysis scientifically interpretable.

The key output is not merely “which model won for each participant.” The key output is:

> **How many distinct utility architectures are needed to describe participant-level heterogeneity without overfitting?**

---

## 2. High-level analysis plan

The analysis has five layers.

```text
Layer 1 — Determine whether per-participant model fits already exist.
Layer 2 — Fit or retrieve participant × model losses and parameters.
Layer 3 — Convert each participant’s model support into a BIC-weighted cloud over utility forms.
Layer 4 — Use model-policy distances to embed and cluster participant model clouds.
Layer 5 — Build a functional-architecture compression curve: how many utility architectures are needed?
```

The most important final outputs are:

1. **Predictive Functional-Form Heterogeneity Index** — does allowing participant-specific utility forms improve held-out prediction beyond a shared architecture?
2. **Functional-Architecture Compression Curve** — how many utility architectures are needed to capture most of the individualized-form advantage?
3. **Participant model-cloud geometry** — do participants cluster around one architecture, a few stable architecture families, or random noise?
4. **Feature-level summaries** — which Boolean utility settings distinguish participant clusters or codebook architectures?

---

## 3. First task: inspect what already exists

Before building new code, inspect the outputs and internal data structures of the existing IC analysis.

The critical question is:

> **Does the existing IC pipeline already fit parameters separately for each participant and each utility function, or does it fit one population-level parameter vector per utility function?**

There are several possible situations.

### Case 1 — Per-participant fits already exist

The existing IC analysis may already store participant-level fitted parameters or participant-level NLL contributions for every model.

Look for outputs like:

```text
participant_id
role
model_id
utility_function_id
params
nll
raw_nll
penalized_nll
aic
bic
k
rank
```

or nested objects like:

```python
results[model_id][participant_id][role]
```

If these exist, do **not** refit everything. Build the individual-architecture analysis by retrieving and reshaping the existing fits.

### Case 2 — Per-participant losses exist, but parameters do not

If the code stores participant-level NLL contributions for each model but not participant-specific optimized parameters, determine whether those NLLs come from:

- one population-level parameter vector evaluated participant by participant, or
- participant-specific fitted parameter vectors.

If the NLLs are based on population-level parameters, they are not sufficient for this analysis.

### Case 3 — Only population-level fits exist

If each utility function has only one fitted parameter vector for the whole sample, then implement participant-level fitting.

Start with a pilot version before running the full analysis.

---

## 4. Data scope

Use the **human-human experiment** data for this analysis.

The relevant experiment has:

- roughly `n = 73` participants;
- repeated binary dictator games;
- randomized payoffs drawn from `{1, 2, 3, 4, 5}`;
- both chooser and predictor roles;
- roughly `~120` responses per participant;
- the full payoff grid contains `5^4 = 625` possible binary dictator-game structures.

### Use both roles for the main analysis

The main participant-level architecture analysis should fit **both chooser and predictor responses per participant** to increase the available data per participant.

However, do **not** force chooser and predictor parameters to be numerically identical.

The preferred main specification is:

> **same utility form for both roles, separate role-specific parameter vectors.**

For participant `i` and model `m`:

$$
NLL_{i,m}
=
NLL^{chooser}_{i,m}(\theta^{chooser}_{i,m})
+
NLL^{predictor}_{i,m}(\theta^{predictor}_{i,m})
$$

where:

- $\theta^{chooser}_{i,m}$ is fitted only to chooser responses;
- $\theta^{predictor}_{i,m}$ is fitted only to predictor responses;
- both parameter vectors use the same utility-function architecture `m`.

This asks:

> Does this participant’s own choice behavior and their beliefs about others seem to require the same utility architecture?

### Sensitivity analyses

Also support, if feasible:

1. chooser-only architecture fitting;
2. predictor-only architecture fitting;
3. combined-role fitting with separate role-specific parameters.

The combined-role version is the main analysis. Chooser-only and predictor-only versions are useful checks.

---

## 5. Participant-level IC calculations

For every participant `i` and every utility function `m`, compute or retrieve:

```text
participant_id
model_id
role
k
n_valid_responses
raw_nll
penalized_nll_if_used
best_params
optimization_status
```

Then construct combined-role participant-level scores.

### 5.1 Raw NLL

Use **raw unpenalized NLL** for reported fit and held-out prediction.

If the optimizer uses an L2 penalty internally, subtract the penalty before reporting model fit, consistent with the existing IC logic.

### 5.2 BIC for separate role-specific parameters

Because the main analysis fits separate chooser and predictor parameter vectors, the default BIC should sum role-specific BIC contributions:

$$
BIC_{i,m}
=
2NLL^{chooser}_{i,m}
+
k_m \log(n^{chooser}_i)
+
2NLL^{predictor}_{i,m}
+
k_m \log(n^{predictor}_i)
$$

where:

- $k_m$ is the number of free parameters in model `m`;
- $n^{chooser}_i$ is participant `i`’s number of valid chooser responses;
- $n^{predictor}_i$ is participant `i`’s number of valid predictor responses.

If either role has too few valid responses for a participant, handle that participant carefully:

- either exclude the role for that participant;
- or exclude the participant from the combined-role analysis;
- but record the rule clearly.

### 5.3 Alternative combined-`n` BIC sensitivity

Optionally compute a stricter combined-parameter BIC:

$$
BIC^{combined}_{i,m}
=
2NLL_{i,m}
+
(2k_m)\log(n^{total}_i)
$$

This is a useful sensitivity check, but the role-summed BIC above is preferred because each role-specific parameter vector is fitted to its own role-specific data.

### 5.4 Delta-BIC

For each participant:

$$
\Delta BIC_{i,m}
=
BIC_{i,m} - \min_{m'} BIC_{i,m'}
$$

The top model for a participant is the model with $\Delta BIC = 0$.

But do not represent participants only by their top model. Individual-level data are noisier than population-level data, so top-model identity can be unstable.

---

## 6. BIC-weighted model clouds

Each participant should be represented as a **probability cloud over utility forms**, not as a single winning model.

Compute BIC weights:

$$
w_{i,m}
=
\frac{\exp(-0.5\Delta BIC_{i,m})}
{\sum_{m'}\exp(-0.5\Delta BIC_{i,m'})}
$$

Interpretation:

- $\Delta BIC = 0$ gets maximal support.
- $\Delta BIC = 2$ gets about $e^{-1} \approx .37$ of the best model’s unnormalized support.
- $\Delta BIC = 10$ gets about $e^{-5} \approx .0067$ of the best model’s unnormalized support.

### Implementation notes

Use log-sum-exp for numerical stability.

Save both:

```text
bic_weight_all_models
bic_weight_delta_le_10_only_optional
```

For most calculations, use all models with numerical stabilization. For visualization, it is often clearer to show only models with nontrivial support, such as $\Delta BIC \leq 10$ or cumulative weight <= .95.

### Effective number of plausible models

For each participant, compute:

$$
N^{eff}_i
=
\frac{1}{\sum_m w_{i,m}^2}
$$

Interpretation:

- $N^{eff} \approx 1$: one architecture dominates.
- large $N^{eff}$: many architectures remain plausible for this participant.

This helps distinguish meaningful participant-level architecture differences from underidentification.

### Feature-inclusion probabilities

Each of the valid models has 14 Boolean utility settings. For each participant and each setting `s`, compute:

$$
P_i(s = 1)
=
\sum_m w_{i,m}\,\mathbb{1}(s_m = 1)
$$

This gives a soft participant-level estimate of which utility features are supported.

Examples:

```text
P_i(altruism included)
P_i(social comparison included)
P_i(envy/guilt split)
P_i(exponents included)
P_i(single versus individual exponents)
P_i(reference dependence)
P_i(payoff ratios)
```

These are often more interpretable than a participant’s exact winning model ID.

---

## 7. Average Model Policy Distance

This analysis depends on a distance matrix between utility forms.

Use the **Average Model Policy Distance** machinery described in `model_recovery_simulation.md`. If that file has already been implemented, call those functions rather than reimplementing them here.

### 7.1 What the distance means

For two utility forms `m` and `n`, Average Model Policy Distance estimates how differently they behave as choice policies over binary dictator-game payoff structures and sampled parameter vectors.

A utility form is not just an equation. It induces a policy:

$$
p_m(A \mid x, \theta)
$$

where:

- `x` is a binary dictator-game payoff structure;
- `θ` is a parameter vector;
- `p_m(A | x, θ)` is the probability that model `m` chooses Option A.

Average Model Policy Distance compares these policies across payoff structures and parameter draws.

### 7.2 Primary symmetric distance metric: normalized JSD

For general model-space geometry, use **normalized Jensen-Shannon divergence** as the primary symmetric distance.

For two Bernoulli probabilities $p$ and $q$:

$$
H(p)
=
-[p\log(p)+(1-p)\log(1-p)]
$$

$$
r
=
\frac{p+q}{2}
$$

$$
JSD(p,q)
=
H(r)-\frac{1}{2}H(p)-\frac{1}{2}H(q)
$$

Normalize by $\log 2$:

$$
D_{JS}(p,q)=\frac{JSD(p,q)}{\log 2}
$$

Then average across games and parameter draws.

Interpretation:

- `0` = identical choice probabilities;
- `1` = maximally different deterministic policies;
- values in between = graded behavioral difference.

### 7.3 Critical rule: AMPD uses shared parameter draws

The AMPD matrix used here must be the **shared-reference-parameter** matrix described in `model_recovery_simulation.md`.

For each Monte Carlo iteration comparing models `m` and `n`:

1. Draw one full canonical reference parameter vector.
2. Give that same vector to both models.
3. Let each model use the parameters it recognizes and ignore the rest.
4. Evaluate both policies on the same payoff structures.
5. Compute normalized JSD or the requested metric.

Do **not** use a matrix computed from independent parameter draws for Model A and Model B. Independent draws make `AMPD(model, model) > 0`, which measures within-model variability across random agents rather than distance between utility architectures. That diagnostic may be interesting elsewhere, but it is not appropriate for participant model-cloud geometry.

For the primary participant-architecture analysis:

```text
AMPD(model_x, model_x) = 0
```

The diagonal of the AMPD matrix should be exactly zero. If a cached matrix has nonzero diagonal entries, treat it as the wrong matrix for this analysis and regenerate it.

### 7.4 Cache pairwise distances

The full pairwise model distance matrix has:

$$
\frac{M(M-1)}{2}
$$

unique off-diagonal pairs, where:

```python
M = len(all_utility_functions)
```

Do not hard-code `476` or `480`. The valid model universe is whatever the current registry / `generate_utility_settings(...)` produces.

This distance matrix is much cheaper than optimization, but still worth caching because it will be reused for MDS, clustering, max-min subset selection, top-model coherence, participant-cloud distances, and architecture-compression analyses.

The distance function should support:

```python
get_or_compute_model_policy_distance(
    model_a,
    model_b,
    metric="normalized_jsd",
    choice_temperature=None,
    n_games=625,
    n_iters=250,
    parameter_sampling_mode="uniform",
    parameter_pairing_mode="shared",
    use_cache=True,
    cache_path=None,
    general_settings=None,
)
```

The exact signature may differ, but it must support loading precomputed distances from a CSV/parquet file **only when settings match**.

The cache should store metadata such as:

```text
model_a_id
model_b_id
metric
choice_temperature
n_games
n_iters
parameter_sampling_mode
parameter_pairing_mode
player_roles
seed
payoff_grid_mode
utility_subset_hash
distance
created_at
code_version_or_git_hash_optional
```

If settings do not match, recompute rather than silently returning an incompatible distance.

---

## 8. Participant-cloud distances

Once every participant has BIC weights over models and every pair of models has a policy distance, compute the distance between participants as the expected distance between their model clouds.

Let:

- $w_{i,m}$ = participant `i`’s BIC weight on model `m`;
- $D(m,n)$ = Average Model Policy Distance between model `m` and model `n`.

Then participant-participant distance is:

$$
D(i,j)
=
\sum_m \sum_n w_{i,m} w_{j,n} D(m,n)
$$

Efficient matrix form:

$$
D_{participants} = W D_{models} W^T
$$

where:

- `W` is an `n_participants × n_models` weight matrix;
- `D_models` is the `n_models × n_models` model-distance matrix.

### Quality checks

Verify:

```text
D(i,i) is near 0 for participants whose model weights are concentrated on one model.
D(i,j) == D(j,i) within numerical tolerance.
All distances are finite.
All distances are nonnegative.
```

Note: if participants have diffuse model clouds, `D(i,i)` using the expected distance between two independent draws from the same cloud need not be exactly zero. That self-distance can be interpreted as within-participant model uncertainty. For embedding, it may be preferable to set the diagonal to zero after computing the matrix.

Save both:

```text
participant_cloud_distance_expected.csv
participant_cloud_distance_for_embedding.csv  # diagonal forced to zero
```

---

## 9. Model-space and participant-space embeddings

### 9.1 Model-space MDS

Use the model-model distance matrix to embed the valid utility forms into 2D or 3D.

Methods:

- classical MDS if the distance matrix behaves well;
- metric MDS if needed;
- UMAP optional as exploratory visualization;
- hierarchical clustering/dendrogram optional.

Save coordinates:

```text
model_id
mds1
mds2
mds3_optional
k
bic_rank_population
delta_bic_population
S1...S14
```

Use this to visualize where the empirical winner and top models sit in the behavioral model landscape.

### 9.2 Participant-space MDS

Use the participant-participant distance matrix to embed participants.

Save coordinates:

```text
participant_id
mds1
mds2
mds3_optional
cluster_label_optional
N_eff
best_model_id
best_model_delta_bic
```

Useful plot:

```text
participant MDS point cloud
color = cluster assignment or dominant architecture feature
size = certainty, e.g. inverse N_eff
```

### 9.3 Participant centroids in model-space

If model-space coordinates exist, each participant can also be represented as a weighted centroid in model-space:

$$
z_i
=
\sum_m w_{i,m} z_m
$$

where $z_m$ is model `m`’s MDS coordinate.

This is useful for visualization, but remember that MDS coordinates are only a low-dimensional approximation of the original distance matrix. For quantitative participant distances, use the full expected-distance formula above.

---

## 10. Predictive Functional-Form Heterogeneity Index

This is the first main inferential result.

It asks:

> **Does allowing participants to have their own utility architectures improve held-out prediction beyond a common architecture?**

Define three held-out losses.

### 10.1 Common-form loss

`NLL_common`: held-out NLL when all participants use one shared utility architecture, but each participant has their own fitted role-specific parameters.

The common architecture can be:

1. the empirical population-level BIC winner, or
2. the best single architecture selected inside the training folds.

For the fairest predictive test, use the best training-selected `K = 1` architecture as the main `NLL_common`. Also report the population-winner version as a sensitivity check.

### 10.2 Individual-form loss

`NLL_individual`: held-out NLL when each participant can use their own best utility architecture, selected from training data only.

For participant `i`:

```text
best_model_i = argmin_m training_BIC_i,m
heldout_loss_i = test_NLL_i,best_model_i
```

Then:

$$
NLL_{individual}
=
\sum_i heldout\_loss_i
$$

Also optionally compute a BIC-weighted model-average version if the codebase supports averaging predictive probabilities over models.

### 10.3 Chance loss

`NLL_chance`: held-out NLL for a random-choice model that predicts `p = .5` for every binary response.

For `n_test` binary responses:

$$
NLL_{chance}=n_{test}\log 2
$$

### 10.4 Heterogeneity index

Compute:

$$
H_{form}
=
\frac{NLL_{common}-NLL_{individual}}
{NLL_{chance}-NLL_{individual}}
$$

Interpretation:

- `0`: individualized functional forms add no predictive value beyond one shared form.
- `0.05`: individualized forms recover 5% of the predictive information missed by the common form.
- `0.50`: individualized forms recover half of the gap between common architecture and chance.
- `1`: the common-form model performs roughly like chance relative to individualized forms.
- negative values: individualized forms performed worse than the common form on held-out data.

Save both raw and clipped versions:

```text
H_form_raw
H_form_clipped_0_1
```

Do not hide negative values. A negative value is informative: it means individual form selection overfit or failed to improve prediction.

---

## 11. Functional-Architecture Compression Curve

This is the second main inferential result.

It asks:

> **How many utility architectures are needed to capture the predictive benefit of allowing participant-specific functional forms?**

Think of this as building a small **codebook** of utility architectures.

- `K = 1`: everyone must use one architecture.
- `K = N participants`: every participant can effectively have their own architecture.
- intermediate `K`: participants are assigned to the best architecture among a small set of `K` allowed forms.

Because utility forms are discrete objects, use **medoids/codebook architectures**, not literal continuous centroids. The word “centroid” is conceptually helpful, but the implemented objects should be actual utility functions.

### 11.1 Definitions

Let:

- `S_K` = a selected set of `K` utility forms;
- `L_i(m)` = held-out NLL for participant `i` using utility form `m`;
- `Score_i(m)` = training selection score, usually participant-level BIC.

Participant `i` is assigned to the best architecture in the codebook using training data:

$$
a_i(S_K)
=
\arg\min_{m \in S_K} Score_i(m)
$$

Then held-out loss for participant `i` under codebook `S_K` is:

$$
L_i(S_K)=L_i(a_i(S_K))
$$

Total held-out loss:

$$
NLL(K)=\sum_i L_i(S_K)
$$

### 11.2 Greedy codebook selection

Exact optimization over all possible subsets of `K` models is combinatorially impossible. Use a greedy approximation.

Algorithm:

```text
Initialize S_1 with the single model that minimizes total training score across participants.

For K = 2, 3, ... K_max:
    For every candidate model not already in S:
        Temporarily add it to S.
        Assign each participant to their best model within the temporary codebook using training BIC.
        Compute total training score.
    Add the model that produces the largest improvement.
    Save S_K, assignments, training score, and held-out NLL.
```

Pseudocode:

```python
def greedy_architecture_codebook(train_scores, test_nll, candidate_model_ids, k_max):
    S = []
    history = []

    for K in range(1, k_max + 1):
        best_candidate = None
        best_train_score = np.inf
        best_assignments = None

        for m in candidate_model_ids:
            if m in S:
                continue
            S_temp = S + [m]
            assignments = assign_each_participant_to_best_model(train_scores, S_temp)
            train_score = total_assigned_score(train_scores, assignments)

            if train_score < best_train_score:
                best_candidate = m
                best_train_score = train_score
                best_assignments = assignments

        S.append(best_candidate)
        test_loss = total_assigned_test_nll(test_nll, best_assignments)
        history.append({
            "K": K,
            "codebook": list(S),
            "assignments": best_assignments,
            "train_score": best_train_score,
            "test_nll": test_loss,
        })

    return history
```

### 11.3 Choose `K_max`

Default:

```python
K_max = min(n_participants, 25)
```

If computation is cheap and plots remain interpretable, allow `K_max = n_participants`.

The curve often reveals the important pattern by `K <= 10`.

### 11.4 Compression metrics

Compute two related curves.

#### A. H-form as a function of K

For each `K`:

$$
H_{form}(K)
=
\frac{NLL(1)-NLL(K)}
{NLL_{chance}-NLL(K)}
$$

Interpretation:

- `K = 1` gives `0` by definition.
- increasing values mean that allowing more architectures improves held-out prediction.
- if values stay near zero, one shared architecture is enough.

#### B. Architecture advantage captured

This normalizes against the fully individualized endpoint:

$$
A(K)
=
\frac{NLL(1)-NLL(K)}
{NLL(1)-NLL_{individual}}
$$

Interpretation:

- `A(1) = 0`.
- `A(K) = .80` means `K` architectures capture 80% of the predictive benefit of fully individualized architecture selection.
- `A(K) = 1` means the codebook performs as well as fully individualized form selection.

This curve directly answers:

> **How many architectures do we need?**

### 11.5 Marginal gains and knee detection

Compute marginal gain:

$$
\Delta A(K)=A(K)-A(K-1)
$$

Also compute raw held-out improvement:

$$
\Delta NLL(K)=NLL(K-1)-NLL(K)
$$

Use multiple knee/stopping criteria:

1. **Marginal-gain threshold:** smallest `K` where adding another architecture improves `A(K)` by less than `.01` or `.02`.
2. **One-standard-error rule:** across CV folds, choose the smallest `K` whose held-out NLL is within one standard error of the best observed `K`.
3. **Kneedle/elbow algorithm:** optional automated elbow detection.
4. **Cluster stability:** prefer `K` values whose participant assignments are stable across folds/bootstrap samples.
5. **Interpretability:** prefer smaller `K` if the added architectures do not correspond to interpretable utility-setting differences.

Do not rely on only one knee rule. Save all criteria and let the manuscript decide which to report.

---

## 12. Cross-validation and leakage prevention

Held-out prediction is essential because individual-level model selection is noisy and can overfit.

### 12.1 Default split strategy

Use participant-preserving, within-participant splits:

```text
For each participant:
    split chooser responses into train/test
    split predictor responses into train/test
```

Use the same folds across models.

Recommended default:

```python
n_folds = 5
n_repeats = 5  # if computationally feasible
```

If computation is expensive, start with:

```python
n_folds = 3
n_repeats = 1
```

### 12.2 Train/test rules

For each fold:

1. Fit participant-specific parameters for each model using training responses only.
2. Compute training NLL and training BIC.
3. Select participant-specific models and codebook architectures using training scores only.
4. Evaluate selected models/codebooks on held-out responses only.

Never use held-out loss to select models, choose `K`, or choose codebook architectures.

### 12.3 Static-model assumption

This analysis should use the same static utility-function fitting logic as the IC analysis unless explicitly extending the dynamic UBM.

The purpose is to compare utility architectures, not to re-run the full dynamic Bayesian belief-updating model for every participant and every utility form.

---

## 13. Clustering participants by architecture

Use participant-cloud distances to test whether individual utility architectures form stable clusters.

### 13.1 Inputs

Use:

```text
participant_distance_matrix
participant MDS coordinates
BIC-weighted feature probabilities
best model / top models
codebook assignments
```

### 13.2 Clustering methods

Support at least two methods:

1. hierarchical clustering using participant-cloud distances;
2. k-medoids on the participant distance matrix.

Optional:

- Gaussian mixture models on MDS coordinates;
- spectral clustering;
- HDBSCAN / density clustering.

### 13.3 Cluster evaluation

For each candidate number of clusters:

```text
silhouette score
cluster sizes
bootstrap stability
adjusted Rand index across folds/repeats
mean within-cluster distance
mean between-cluster distance
```

### 13.4 Cluster interpretation

For each cluster, summarize:

- top representative utility forms;
- BIC-weighted feature inclusion probabilities;
- average participant parameters under the population-winning model;
- behavioral summaries, if available;
- role-specific differences, if chooser-only and predictor-only variants were run.

Feature summary example:

```text
Cluster 1: high support for altruism + social comparison + envy/guilt split
Cluster 2: high support for self-interest + exponents, weak support for altruism
Cluster 3: diffuse support, no stable architecture
```

### 13.5 Interpretation warning

Random scatter is **not** strong evidence for real functional-form heterogeneity.

Strong evidence for architectural heterogeneity requires:

```text
stable clusters
held-out predictive improvement
interpretable utility-setting differences
nontrivial separation in model-policy space
```

If top models scatter but held-out prediction does not improve, the conclusion is likely:

> Individual-level form selection is underidentified or overfits; the shared architecture remains the better-supported summary.

---

## 14. Common architecture versus heterogeneous architecture: three-level comparison

Organize the results around three model levels.

### Level 1 — Population/shared form, participant-specific parameters

One utility form is used for everyone, but every participant gets their own parameters.

Primary common-form candidates:

1. empirical population-level seven-parameter winner;
2. best single architecture selected from training data.

Question:

> How well does one architecture describe everyone when parameter heterogeneity is allowed?

### Level 2 — Small architecture codebook

A small set of `K` utility forms is allowed. Each participant is assigned to the best architecture in the codebook.

Question:

> Does a small number of utility architectures capture most individual differences?

### Level 3 — Individual architecture

Each participant can use their own best-fitting utility form.

Question:

> Does every participant need their own architecture, or does that overfit?

The **functional-architecture compression curve** links these levels.

---

## 15. Weighted feature accuracy / feature summaries

Although “weighted feature accuracy” was originally discussed for utility-function recovery simulations, the same logic is useful here for participant-level architecture summaries.

For each participant, compute feature probabilities:

$$
P_i(s=1)=\sum_m w_{i,m}\,\mathbb{1}(s_m=1)
$$

For clusters or codebook groups, compute the average:

$$
P_{cluster}(s=1)=\frac{1}{|C|}\sum_{i\in C}P_i(s=1)
$$

For codebook assignments, also compute the feature vector of each codebook architecture.

### Optional behaviorally weighted feature summary

If model-policy distances are used to derive behavioral feature weights, define:

$$
weight_s
=
\text{average policy distance between model pairs differing only in setting }s
$$

Then a participant or cluster’s support for behaviorally important features can be summarized by weighting settings by `weight_s`.

This is optional. The essential feature summaries are the BIC-weighted inclusion probabilities.

---

## 16. Main outputs to save

Create a dedicated output directory, e.g.:

```text
outputs/individual_architecture_analysis/
```

### 16.1 Core tables

Save these as CSV/parquet.

#### `participant_model_fits.csv`

One row per participant × model × role × fold.

```text
participant_id
fold_id
repeat_id
role
model_id
k
n_train
n_test
train_raw_nll
test_raw_nll
train_penalized_nll_if_any
best_params_json
optimization_status
```

#### `participant_model_scores.csv`

One row per participant × model × fold, combined across roles.

```text
participant_id
fold_id
repeat_id
model_id
k
train_nll_total
test_nll_total
bic_role_summed
bic_combined_optional
delta_bic
bic_weight
rank_bic
rank_train_nll
```

#### `participant_feature_support.csv`

One row per participant × fold.

```text
participant_id
fold_id
repeat_id
S1_prob
S2_prob
...
S14_prob
N_eff
best_model_id
top5_model_ids
models_delta_bic_le_10
```

#### `model_policy_distance_matrix.csv`

If not already created elsewhere.

```text
model_a_id
model_b_id
metric
settings_hash
distance
```

#### `participant_cloud_distance_matrix.csv`

Square matrix or long form.

```text
participant_i
participant_j
distance
fold_id_optional
repeat_id_optional
```

#### `architecture_codebook_curve.csv`

One row per `K` × fold/repeat.

```text
fold_id
repeat_id
K
codebook_model_ids
train_score
test_nll
nll_common
nll_individual
nll_chance
H_form_K
A_K
delta_A_K
delta_NLL_K
assignments_json
```

#### `architecture_codebook_summary.csv`

Aggregated across folds/repeats.

```text
K
mean_test_nll
se_test_nll
mean_H_form_K
se_H_form_K
mean_A_K
se_A_K
mean_delta_A
selected_by_one_se_rule
selected_by_marginal_gain_rule
selected_by_kneedle_optional
```

#### `cluster_summary.csv`

```text
cluster_id
n_participants
mean_within_distance
mean_between_distance
silhouette
representative_model_ids
S1_prob_mean
...
S14_prob_mean
notes_optional
```

---

## 17. Main figures to generate

Generate publication-quality figures, but keep raw data behind every figure.

### Figure 1 — Functional-Architecture Compression Curve

x-axis:

```text
K = number of allowed utility architectures
```

y-axis options:

```text
A(K): proportion of individualized-form advantage captured
H_form(K): predictive functional-form heterogeneity index at K
held-out NLL
```

Preferred main panel:

```text
A(K) with error bars across folds/repeats
```

Add vertical markers for selected knees/stopping points.

### Figure 2 — Common vs Individualized Held-Out Prediction

Compare:

```text
common population winner
best K=1 shared architecture
best K from compression curve
fully individualized architecture
chance
```

Use held-out NLL or normalized improvement.

### Figure 3 — Participant-Space MDS

Plot participants embedded from participant-cloud distances.

Visual encodings:

```text
color = cluster assignment or codebook assignment
size = 1 / N_eff or certainty
shape = optional role/sensitivity indicator
```

### Figure 4 — Model-Space MDS With Participant Clouds

Plot utility forms in model-policy space.

Overlay:

```text
population winner
codebook architectures
participant weighted centroids
best models per participant
```

This visually answers whether participants are concentrated around one architectural basin or spread across several.

### Figure 5 — Feature Support by Cluster or Codebook

Heatmap:

```text
rows = clusters or codebook architectures
columns = 14 utility settings
values = BIC-weighted support / feature inclusion
```

This makes clusters interpretable.

---

## 18. Quality checks and diagnostics

### 18.1 Fitting diagnostics

For participant-level fits, report:

```text
number of failed optimizations
number of boundary solutions
distribution of train NLL
train-test gap by model k
rank stability across folds/repeats
```

### 18.2 BIC-weight diagnostics

Check:

```text
weights sum to 1 per participant
no NaN or infinite weights
reasonable N_eff distribution
participants with diffuse model clouds are flagged
```

### 18.3 Distance diagnostics

Check:

```text
model distance matrix is symmetric
model self-distances are zero
participant embedding distances are finite
model-distance cache settings match current requested settings
```

### 18.4 Compression-curve diagnostics

The training curve should be monotonic non-increasing as `K` increases.

Held-out curves may not be perfectly monotonic because of sampling noise. If held-out loss worsens at higher `K`, that is evidence of overfitting and should be preserved, not smoothed away.

### 18.5 Leakage checks

No held-out data may be used to:

```text
fit parameters
select individual participant models
choose codebook architectures
choose K
compute BIC weights used for model clouds in training-based prediction
```

Held-out data are only for evaluation.

---

## 19. Suggested staged implementation

Do not begin with the full monster analysis. Build in stages.

### Stage 1 — Inspect and retrieve

Find out whether the existing IC analysis already stores participant-level fits or losses.

Deliverable:

```text
short report: existing outputs found, what they contain, what must be recomputed
```

### Stage 2 — Minimal pilot

Run the pipeline on:

```text
10 participants
20 utility functions
1 train/test split
```

Include:

```text
participant-level fitting/retrieval
BIC weights
participant-cloud distances
small compression curve
```

Deliverable:

```text
pilot outputs + sanity-check figures
```

### Stage 3 — Full participant × model analysis

Run:

```text
73 participants × valid utility forms
combined chooser+predictor roles
```

Use reduced optimization settings if necessary, but record them.

Deliverable:

```text
participant_model_scores.csv
participant_feature_support.csv
```

### Stage 4 — Distance and geometry

Compute/load Average Model Policy Distance matrix.

Then compute:

```text
participant-cloud distance matrix
model-space MDS
participant-space MDS
cluster summaries
```

### Stage 5 — Cross-validated compression curve

Run codebook selection and held-out evaluation.

Deliverable:

```text
architecture_codebook_curve.csv
architecture_codebook_summary.csv
main figures
```

### Stage 6 — Sensitivity checks

If time permits:

```text
chooser-only
predictor-only
alternative BIC penalty
alternative model-policy distance temperature
alternative distance metric
alternative K selection rules
```

---

## 20. Interpretation guide for manuscript writing

The results should be interpreted according to this decision table.

| Result pattern | Interpretation |
|---|---|
| `H_form ≈ 0`, compression curve flat after `K=1`, participant clouds near one basin | Strong support for a shared utility architecture; people differ mainly in parameters. |
| `H_form > 0`, curve reaches knee at `K=2` or `K=3`, clusters stable and interpretable | Evidence for a small number of functional phenotypes. |
| Top models scatter, but held-out prediction does not improve | Individual-level form selection is likely noisy or underidentified; shared architecture remains preferable. |
| Held-out prediction improves only gradually up to high `K`, clusters unstable | Possible idiosyncratic architecture differences, but no clean taxonomy. |
| Participant clusters align with meaningful utility settings and improve prediction | Strong evidence that people differ in utility architecture, not merely parameter values. |

The analysis should not overclaim from scatter alone.

> **Stable, predictive, interpretable clustering is evidence for functional-form heterogeneity. Random dispersion is not.**

---

## 21. How this could appear in the paper

If the analysis works cleanly, it can become a concise main-text section, perhaps called:

> **Individual-Level Utility Architecture**

Suggested manuscript claim template:

```text
We next tested whether the winning seven-parameter utility function describes a shared psychological coordinate system or instead averages over heterogeneous individual utility architectures. For each participant, we fit all valid utility forms using chooser and predictor responses, represented model uncertainty as BIC-weighted clouds over utility-function space, and evaluated whether small codebooks of utility architectures improved held-out prediction. [Result summary.] These findings suggest that [one shared architecture is sufficient / a small number of stable utility architectures captures meaningful heterogeneity / individual architecture selection is underidentified in the present dataset].
```

The main text probably needs only:

1. one paragraph explaining the question;
2. one compression-curve figure;
3. one participant-space or model-space geometry figure;
4. one short table summarizing selected architectures/clusters.

Put implementation details, robustness, and alternative clustering methods in the supplement.

---

## 22. Final reminder

This analysis is not a decorative add-on. It directly tests the status of the paper’s central coordinate-system claim.

If one architecture is enough, the paper becomes stronger:

> **The seven-parameter model is not merely the best population compromise; it is a shared architecture within which individual differences can be located.**

If a few architectures are needed, the paper becomes more interesting:

> **Human social preferences may vary not only in parameter values, but in functional architecture, revealing multiple moral-motivational phenotypes.**

Either result is scientifically valuable.

Build the analysis so that the conclusion is earned by held-out prediction, model-space geometry, and interpretable utility-feature summaries — not by noisy top-model labels alone.
