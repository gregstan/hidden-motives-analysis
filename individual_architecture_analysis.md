# `individual_architecture_analysis.md`

# Individual Utility-Architecture Analysis

> **Purpose:** build the machinery needed to answer a new question that sits directly underneath the paper’s “shared coordinate system” claim:
>
> **Do participants differ mainly by their parameter values inside one shared utility architecture, or do different participants require different utility-function architectures?**

This file is written as instructions for the coding agent working inside the **Inferring Hidden Motives** codebase. The agent should also read:

- `ihm_starter_pack.md` — high-level conceptual and technical overview of the paper.
- `model_recovery_simulation.md` — model-recovery and **Average Model Policy Distance** plan.

This file focuses on a new individual-level extension of the 476-model utility-function IC analysis.

---

## 0. Implementation principle

> **Reuse existing code aggressively.**

Before writing new machinery, inspect the existing codebase for functions that already do any of the following:

- generate the 476 utility forms;
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

The current paper compares **476 candidate utility functions** and identifies a best-fitting seven-parameter utility architecture. That winning form separates self-interest, altruism, envy, guilt, and nonlinear payoff sensitivity, and it supports the paper’s claim that the resulting parameter estimates form a shared psychological coordinate system.

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

Each of the 476 models has 14 Boolean utility settings. For each participant and each setting `s`, compute:

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

### 7.3 Cache pairwise distances

The full pairwise model distance matrix has:

$$
\frac{476 \times 475}{2}=113{,}050
$$

unique pairs.

This is much cheaper than optimization, but still worth caching.

The distance function should support:

```python
get_or_compute_model_policy_distance(
    model_a,
    model_b,
    metric="normalized_jsd",
    choice_temperature=None,
    n_games=625,
    n_iters=250,
    parameter_sampler="reference_uniform",
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
parameter_sampler
seed
payoff_grid_mode
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

Use the model-model distance matrix to embed the 476 utility forms into 2D or 3D.

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
73 participants × 476 utility forms
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
We next tested whether the winning seven-parameter utility function describes a shared psychological coordinate system or instead averages over heterogeneous individual utility architectures. For each participant, we fit all 476 utility forms using chooser and predictor responses, represented model uncertainty as BIC-weighted clouds over utility-function space, and evaluated whether small codebooks of utility architectures improved held-out prediction. [Result summary.] These findings suggest that [one shared architecture is sufficient / a small number of stable utility architectures captures meaningful heterogeneity / individual architecture selection is underidentified in the present dataset].
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
