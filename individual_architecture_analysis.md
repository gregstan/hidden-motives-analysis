# Individual Utility-Architecture Analysis

> **Coding-agent brief for the *Inferring Hidden Motives* repository**  
> **Purpose:** determine whether participants differ mainly by **parameter values inside one shared utility architecture** or whether different participants require **different utility-function architectures**.  
> **Primary audience:** an AI coding agent with access to the repo but no prior conceptual context.  
> **Implementation priority:** first extract and reuse existing per-participant × utility-function fits. Do **not** refit all models unless extraction fails or a later train/test analysis requires it.

---

## 0. Why this analysis matters

The paper currently argues that the best-fitting utility function provides a shared psychological coordinate system: a common space in which people’s social preferences can be located. That claim is powerful, but it raises a deeper question:

> **Is the winning utility function a genuinely shared psychological architecture, or is it a population-level compromise over people who are better described by different utility architectures?**

This analysis directly tests that question.

There are two possible worlds.

### World A — shared architecture, different parameters

Everyone is described by the same functional form, but people differ in their parameter values.

```text
Participant 1: same utility architecture, high self-interest, low guilt
Participant 2: same utility architecture, moderate self-interest, high guilt
Participant 3: same utility architecture, low self-interest, high altruism
```

If this world is true, the paper’s shared-coordinate-system claim becomes much stronger.

### World B — different architectures, different parameters

Different people are better described by meaningfully different utility forms.

```text
Participant 1: self-interest + altruism architecture
Participant 2: self-interest + social-comparison architecture
Participant 3: nonlinear envy/guilt architecture
Participant 4: reference-dependent architecture
```

If this world is true, the paper discovers something even richer: **functional phenotypes**. People differ not just in where they sit in a shared space, but in what kind of space describes them.

### World C — random scatter / underidentification

Participants’ top models may scatter all over the model universe without forming stable, predictive, interpretable clusters.

This does **not** strongly support different functional architectures. It may mean the per-participant data are too noisy, that many models are behaviorally near-equivalent, or that individual-level model selection overfits.

The key distinction is:

> **Stable, predictive, interpretable clusters support functional-form heterogeneity. Random scatter does not.**

---

## 1. The conceptual object: participants as model clouds

Each participant has a fit score for every utility function.

For participant `i` and utility model `m`:

```text
BIC_i_m
```

A naive analysis would represent each participant by their single best model. That would be too brittle. Individual-level data are noisier than population-level data, so exact top-model identity may bounce around.

Instead, convert model support into weights. For each participant:

$$
w_{i,m}=
\frac{\exp(-0.5\Delta BIC_{i,m})}
{\sum_{m'}\exp(-0.5\Delta BIC_{i,m'})}
$$

where:

$$
\Delta BIC_{i,m}=BIC_{i,m}-\min_{m'} BIC_{i,m'}
$$

Interpretation:

- the best model gets the largest weight;
- models close in BIC get meaningful support;
- models far away get tiny support;
- each participant becomes a **BIC-weighted cloud over utility-function space**.

This is the key idea. Participants are not single points at first. They are clouds of attraction over utility architectures.

## 2. Elastic-band intuition

The model-space map contains utility functions as points. Each participant is pulled toward those utility-function points by BIC weights.

- strong BIC weight = strong elastic band;
- weak BIC weight = weak elastic band;
- if one model dominates, the participant sits near that model;
- if several nearby models dominate, the participant sits in that basin;
- if distant models all have support, the participant is diffuse / underidentified.

This metaphor is useful for visualization, but quantitative distances should be computed using the full model-distance matrix, not only the 2D plot.

---

# Part I — Extract participant × model × role fits

## 3. First task: inspect existing IC outputs

The current IC analysis appears to save per-player, per-role fits for every utility function. If so, this analysis may be dramatically easier than refitting all models.

Look for structures like:

```python
minimum_params_and_losses[utility_setting_key][player_uuid]["loss"]["chooser"]
minimum_params_and_losses[utility_setting_key][player_uuid]["loss"]["predictor"]
minimum_params_and_losses[utility_setting_key][player_uuid]["params"]["chooser"]
minimum_params_and_losses[utility_setting_key][player_uuid]["params"]["predictor"]
```

In the current `information_criterion_analysis(...)`, these may be saved as `minvec` inside per-model IC JSON files.

### Important

If these per-participant losses and parameter vectors already exist, **do not refit**. Extract and reorganize them.

## 4. Temporary legacy source for debugging

Large precomputed fit files may not be in the new repo. During debugging, use the old local files:

```text
C:\Users\Gregory Stanley\Desktop\U of M\Research Archive\Multiplayer\ABM_Simulation\Judgment_Game\Inputs\Iter_Binary_Dictator\player_fits
```

Rules:

1. Do not commit large fit files.
2. Check whether old fit-file format matches the current codebase.
3. If minor compatibility differences exist, write a compatibility loader.
4. Mark compatibility code clearly:

```python
"TODO: Legacy compatibility path. Delete this block after the new participant_model_role_losses.csv extraction is proven to work with the current repo outputs."
```

The old code is in:

```text
ABM_Simulation > Iter_Binary_Dictator.py
```

Use it only if needed to decode legacy file structure.

## 5. Extraction deliverable

Create:

```text
processed/participant_model_role_losses.csv
```

Suggested columns:

```text
utility_idx
utility_bitstring
utility_setting_key
player_uuid
player_role
loss_nll
n_data
params_json
k_params
AIC
BIC
source_file
source_version_optional
```

This file should be the bridge between the old IC fitting machinery and the new individual-architecture analysis.

## 6. Use both roles by default

The main analysis should use both chooser and predictor responses per participant to increase information.

But do **not** force chooser and predictor parameters to be identical.

Preferred specification:

> **same functional form, separate role-specific parameter vectors**

For participant `i` and model `m`:

$$
NLL_{i,m}=NLL^{chooser}_{i,m}+NLL^{predictor}_{i,m}
$$

with separate parameters:

```text
theta_chooser_i_m
theta_predictor_i_m
```

This asks:

> Does this participant’s choice behavior and belief/prediction behavior gravitate toward the same architecture, while allowing the numerical parameter values to differ by role?

### Sensitivity analyses

If cheap, also compute:

```text
chooser-only architecture analysis
predictor-only architecture analysis
combined-role architecture analysis
```

Primary analysis:

```text
combined chooser + predictor, separate role-specific parameters
```

---

# Part II — Participant-level BIC, weights, and clouds

## 7. Participant-level BIC

Use raw unpenalized NLL for reporting. If the optimizer used a penalty internally, subtract it before reporting fit, consistent with the IC analysis.

Because chooser and predictor parameters are separate, use role-summed BIC:

$$
BIC_{i,m}
=
2NLL^{chooser}_{i,m}+k_m\log(n^{chooser}_i)
+
2NLL^{predictor}_{i,m}+k_m\log(n^{predictor}_i)
$$

where:

- `k_m` = number of free parameters in model `m` for one role;
- `n_chooser_i` = participant `i`’s valid chooser responses;
- `n_predictor_i` = participant `i`’s valid predictor responses.

If a role has too little data, handle it explicitly and document the rule.

### Optional stricter BIC sensitivity

A stricter alternative is:

$$
BIC^{combined}_{i,m}=2NLL_{i,m}+(2k_m)\log(n^{total}_i)
$$

Compute this as a sensitivity check if easy. Prefer the role-summed BIC for the main analysis because each role-specific parameter vector is fitted to role-specific data.

## 8. Delta-BIC and weights

For each participant:

$$
\Delta BIC_{i,m}=BIC_{i,m}-\min_m BIC_{i,m}
$$

BIC weights:

$$
w_{i,m}=
\frac{\exp(-0.5\Delta BIC_{i,m})}
{\sum_{m'}\exp(-0.5\Delta BIC_{i,m'})}
$$

Use log-sum-exp for numerical stability.

Save:

```text
processed/participant_model_bic_weights.csv
```

Suggested columns:

```text
player_uuid
utility_idx
utility_bitstring
BIC
ΔBIC
bic_weight
rank
k_params
loss_nll_combined
loss_nll_chooser
loss_nll_predictor
n_data_chooser
n_data_predictor
```

## 9. Effective number of plausible models

For each participant:

$$
N^{eff}_i=\frac{1}{\sum_m w_{i,m}^2}
$$

Interpretation:

| `N_eff` | Meaning |
|---:|---|
| near 1 | one architecture dominates |
| moderate | several nearby architectures are plausible |
| large | architecture is diffuse / underidentified |

Save:

```text
processed/participant_architecture_summary.csv
```

## 10. Feature-inclusion probabilities

For each participant and each Boolean utility setting `s`:

$$
P_i(s=1)=\sum_m w_{i,m} I(s_m=1)
$$

This gives participant-level support for features such as:

```text
include_altruism_term
include_social_comparison
negativity_social_comparison
use_exponential_parameters
single_exponential_parameter
reference_dependent_utility
payoff_ratios_not_differences
min_max_rawlsian_leontief
```

These feature probabilities are often more interpretable than exact model IDs.

---

# Part III — Model-space distances and participant-cloud distances

## 11. Use Average Model Policy Distance

This analysis depends on the model-model distance matrix described in `model_recovery_simulation.md`.

Do not reimplement AMPD here. Call the shared distance functions.

AMPD says how behaviorally different two utility functions are as choice policies over payoff structures and parameter draws.

Primary distance matrix:

```text
D_models[utility_idx_a, utility_idx_b]
```

Default metric:

```text
normalized Jensen-Shannon divergence averaged over payoff structures and parameter draws
```

## 12. Participant-cloud distance

Given:

- `W`: participant × model matrix of BIC weights;
- `D_models`: model × model AMPD matrix;

compute participant distances as:

$$
D_{participants}=W D_{models} W^T
$$

Elementwise:

$$
D(i,j)=\sum_m\sum_n w_{i,m}w_{j,n}D(m,n)
$$

Interpretation:

> If we randomly draw one plausible model from participant `i`’s cloud and one plausible model from participant `j`’s cloud, how far apart are those models on average?

### Self-distance note

If a participant’s model cloud is diffuse, `D(i,i)` may not be zero, because two independent draws from that same cloud can land on different models. This self-distance is interpretable as within-participant architecture uncertainty.

For embedding algorithms, save a version with the diagonal forced to zero.

Save:

```text
processed/participant_cloud_distance_expected.csv
processed/participant_cloud_distance_for_embedding.csv
```

## 13. Model-space MDS and participant placement

Use the model-model AMPD matrix to embed utility functions into 2D or 3D.

Save:

```text
processed/model_space_mds_coordinates.csv
```

Columns:

```text
utility_idx
mds1
mds2
mds3_optional
k_params
BIC_rank_population
ΔBIC_population
utility_bitstring
setting columns
```

### Participants in model space

Each participant can be plotted in the same MDS space as a weighted centroid:

$$
z_i=\sum_m w_{i,m}z_m
$$

where `z_m` is model `m`’s MDS coordinate.

This matches the elastic-band intuition: participant points are pulled toward utility-function points by BIC weights.

### But be careful

The centroid is a visualization. The quantitative participant distances should use the full distance matrix formula, not only 2D coordinates.

## 14. Participant-space MDS

Use `participant_cloud_distance_for_embedding.csv` to embed participants directly.

Save:

```text
processed/participant_architecture_mds_coordinates.csv
```

Columns:

```text
player_uuid
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
color = cluster assignment or dominant feature
size = certainty, e.g. inverse N_eff
```

---

# Part IV — Predictive Functional-Form Heterogeneity Index

## 15. What `H_form` asks

`H_form` answers one central question:

> **Does allowing participants to have their own utility architectures improve prediction beyond one common architecture?**

This is the teeter-totter score:

- same form, different parameters;
- versus different forms, different parameters.

## 16. Held-out prediction is the strongest version

The cleanest version uses train/test splits.

Why? Because individualized utility functions will almost always improve in-sample fit. The question is whether they improve **held-out prediction** enough to justify the loss of parsimony.

### Common-form model

All participants use one shared utility architecture, but each participant has their own role-specific parameters.

Loss:

```text
NLL_common
```

The common form can be:

1. the population-level IC winner;
2. the best single form selected on training data.

Use the training-selected `K=1` model as the fairest primary test. Report the population-winner version as a sensitivity check.

### Individual-form model

Each participant gets their own best architecture selected on training data.

Loss:

```text
NLL_individual
```

### Chance model

Random-choice model predicting `p = .5` for every binary response.

For `n_test` binary responses:

$$
NLL_{chance}=n_{test}\log 2
$$

## 17. Heterogeneity index

$$
H_{form}
=
\frac{NLL_{common}-NLL_{individual}}
{NLL_{chance}-NLL_{individual}}
$$

Interpretation:

| `H_form` | Meaning |
|---:|---|
| `0` | individualized architectures add no predictive value beyond one shared architecture |
| `.05` | individualized forms recover 5% of the predictive information missed by the common form |
| `.50` | individualized forms recover half the gap between common form and chance |
| `1` | common-form model is roughly chance-like relative to individualized forms |
| negative | individualized form selection overfit or performed worse than the common architecture |

Save both:

```text
H_form_raw
H_form_clipped_0_1
```

Do not hide negative values. Negative values are informative.

## 18. First-pass in-sample version

If train/test refitting is not immediately available, compute an **exploratory in-sample** version from existing fits. Label it clearly:

```text
H_form_in_sample_exploratory
```

This is useful for deciding whether the analysis is promising, but the held-out version is required for strong claims.

---

# Part V — Functional-Architecture Compression Curve

## 19. What the compression curve asks

`H_form` asks whether individualized forms help.

The compression curve asks:

> **If individualized forms help, how many distinct utility architectures are needed to capture that benefit?**

This creates a continuum:

| K | Meaning |
|---:|---|
| 1 | everyone must use one shared architecture |
| 2 | participants can be assigned to one of two architectures |
| 3 | participants can be assigned to one of three architectures |
| N participants | each participant can effectively have their own architecture |

Because utility functions are discrete objects, call these **architecture codebooks** or **medoids**, not literal centroids. A centroid can be an abstract average point; a medoid is an actual utility function.

## 20. Definitions

Let:

```text
S_K = selected set of K utility forms
L_i(m) = held-out NLL for participant i under model m
Score_i(m) = training selection score, usually participant-level BIC
```

Participant assignment:

$$
a_i(S_K)=\arg\min_{m\in S_K} Score_i(m)
$$

Participant loss under the codebook:

$$
L_i(S_K)=L_i(a_i(S_K))
$$

Total held-out loss:

$$
NLL(K)=\sum_i L_i(S_K)
$$

## 21. Greedy codebook selection

Exact search over all possible sets of `K` models is impossible. Use a greedy approximation.

Algorithm:

```text
Initialize S as empty.

For K = 1 to K_max:
    For every candidate model not already in S:
        Temporarily add candidate to S.
        Assign each participant to their best model in temporary S using training BIC.
        Compute total training score.
    Permanently add the candidate that gives the biggest improvement.
    Compute held-out NLL for the resulting assignments.
    Save S_K, assignments, training score, held-out NLL, and interpretive metadata.
```

Pseudocode:

```python
def greedy_architecture_codebook(
    train_scores_by_participant_and_model,
    test_nll_by_participant_and_model,
    candidate_model_ids,
    k_max,
):
    selected_codebook_model_ids = []
    codebook_history_rows = []

    for current_k in range(1, k_max + 1):
        best_candidate_model_id = None
        best_candidate_training_score = np.inf
        best_candidate_assignments = None

        for candidate_model_id in candidate_model_ids:
            if candidate_model_id in selected_codebook_model_ids:
                continue

            temporary_codebook = selected_codebook_model_ids + [candidate_model_id]
            participant_assignments = assign_each_participant_to_best_model(
                train_scores_by_participant_and_model=train_scores_by_participant_and_model,
                codebook_model_ids=temporary_codebook,
            )
            total_training_score = total_assigned_score(
                train_scores_by_participant_and_model=train_scores_by_participant_and_model,
                participant_assignments=participant_assignments,
            )

            if total_training_score < best_candidate_training_score:
                best_candidate_model_id = candidate_model_id
                best_candidate_training_score = total_training_score
                best_candidate_assignments = participant_assignments

        selected_codebook_model_ids.append(best_candidate_model_id)
        heldout_nll = total_assigned_test_nll(
            test_nll_by_participant_and_model=test_nll_by_participant_and_model,
            participant_assignments=best_candidate_assignments,
        )

        codebook_history_rows.append({
            "K": current_k,
            "codebook_model_ids": list(selected_codebook_model_ids),
            "participant_assignments": best_candidate_assignments,
            "train_score": best_candidate_training_score,
            "heldout_nll": heldout_nll,
        })

    return pd.DataFrame(codebook_history_rows)
```

## 22. Compression metrics

### 22.1 H-form as a function of K

$$
H_{form}(K)
=
\frac{NLL(1)-NLL(K)}
{NLL_{chance}-NLL(K)}
$$

### 22.2 Architecture advantage captured

This is the cleaner “how many architectures do we need?” curve:

$$
A(K)=
\frac{NLL(1)-NLL(K)}
{NLL(1)-NLL_{individual}}
$$

Interpretation:

| `A(K)` | Meaning |
|---:|---|
| `0` | K architectures capture none of the individualized-form advantage |
| `.80` | K architectures capture 80% of the individualized-form advantage |
| `1` | K architectures perform as well as fully individualized architecture selection |

## 23. Complexity penalty and knee detection

First, plot the raw compression curve. It is the most transparent scientific object.

Then optionally compute a penalized score:

$$
Score(K)=A(K)-\lambda\frac{K-1}{N-1}
$$

This can produce a peak, but `lambda` is a researcher-chosen penalty. Therefore, treat it as secondary.

Preferred stopping criteria:

1. **Marginal-gain threshold**  
   Stop when `A(K)-A(K-1)` drops below `.01` or `.02`.

2. **One-standard-error rule**  
   Across CV folds, choose the smallest `K` whose held-out NLL is within one standard error of the best observed `K`.

3. **Kneedle / elbow algorithm**  
   Optional automated elbow detection.

4. **Cluster stability**  
   Prefer `K` values whose participant assignments are stable across folds or bootstraps.

5. **Interpretability**  
   Prefer smaller `K` if added architectures do not correspond to interpretable utility-setting differences.

Do not rely on only one criterion. Save them all.

## 24. What outcomes mean

### Flat after K = 1

One architecture is enough. This supports the shared-coordinate-system story.

### Knee at K = 2 or 3

A small number of utility architectures explain most functional heterogeneity. This suggests functional phenotypes.

### Gradual improvement until high K

Individualization may improve prediction, but without a clean taxonomy. This could reflect idiosyncrasy, overfitting, or insufficient data.

### No held-out improvement

Individual top models may be noisy. Stick with the shared architecture.

---

# Part VI — Clustering and visualization

## 25. Participant-space clustering

Use the participant-cloud distance matrix.

Methods:

```text
hierarchical clustering
k-medoids if available
spectral clustering optional
```

Evaluate:

```text
silhouette score
bootstrap stability
adjusted Rand index across resamples
cluster interpretability by model features
held-out predictive improvement if available
```

Do not treat random scatter as evidence for functional-form heterogeneity.

Strong evidence requires:

```text
stable clusters
predictive benefit
interpretable model-feature differences
```

## 26. Model-space + participant-space plots

### 26.1 Model-space map

Use AMPD to embed utility functions.

Plot:

```text
utility functions as points
color = population ΔBIC or major model family
highlight = population IC winner
overlay = participant weighted centroids
```

This visually shows whether participants cluster around the winner or spread across the model universe.

### 26.2 Participant-space map

Use participant-cloud distances.

Plot:

```text
participants as points
color = cluster
size = 1 / N_eff or certainty
hover = best model, ΔBIC, dominant feature probabilities
```

### 26.3 Clouds for selected participants

For a few example participants, show their top supported models in model-space.

This helps readers understand the cloud concept.

---

# Part VII — Cross-validation and leakage prevention

## 27. Why train/test matters

Individualized architecture selection can overfit. Held-out prediction is the strongest way to decide whether extra architectural flexibility is worth it.

## 28. Default split strategy

Use within-participant splits:

```text
For each participant:
    split chooser responses into train/test
    split predictor responses into train/test
    preserve role balance when possible
```

Use training data to select models and fit parameters. Use held-out data only to evaluate NLL.

If full refitting is too expensive, build first-pass geometry from existing fits and implement train/test later.

## 29. First-pass vs final-pass labels

Be explicit in output names:

```text
in_sample_exploratory
cross_validated_confirmatory
```

The in-sample version is useful for exploration. The cross-validated version is necessary for strong claims.

---

# Part VIII — Outputs

## 30. Core CSV outputs

```text
processed/participant_model_role_losses.csv
processed/participant_model_scores.csv
processed/participant_model_bic_weights.csv
processed/participant_architecture_summary.csv
processed/participant_feature_support_probabilities.csv
processed/participant_cloud_distance_expected.csv
processed/participant_cloud_distance_for_embedding.csv
processed/model_space_mds_coordinates.csv
processed/participant_architecture_mds_coordinates.csv
processed/architecture_clusters.csv
processed/H_form_summary.csv
processed/architecture_compression_curve.csv
```

## 31. Core figures

```text
visuals/individual_architecture/model_space_with_participant_centroids.html
visuals/individual_architecture/participant_architecture_mds.html
visuals/individual_architecture/participant_feature_support_heatmap.html
visuals/individual_architecture/H_form_bar_or_interval.html
visuals/individual_architecture/architecture_compression_curve.html
visuals/individual_architecture/codebook_assignments_by_K.html
```

## 32. Main-text-ready figure idea

A compact main-text figure could have two panels:

### Panel A — Architecture compression curve

x-axis:

```text
K = number of allowed utility architectures
```

y-axis:

```text
A(K) or held-out NLL improvement
```

This answers how many forms are needed.

### Panel B — Participant architecture map

Participant-space MDS using BIC-weighted cloud distances.

Color by:

```text
cluster assignment
```

or:

```text
dominant architecture feature
```

This shows whether participants cluster around one shared architecture or split into stable basins.

---

# Part IX — Implementation order

## Stage 1 — Extract existing fits

Inspect IC outputs. Build `participant_model_role_losses.csv`.

## Stage 2 — Compute participant BIC weights

Build combined-role BIC, ΔBIC, BIC weights, `N_eff`, and feature support probabilities.

## Stage 3 — Load AMPD matrix

Use the shared AMPD infrastructure from `model_recovery_simulation.md`.

## Stage 4 — Build participant distances

Compute `W D W.T` and save participant-cloud distance matrices.

## Stage 5 — Embeddings and first-pass geometry

Create model-space MDS, participant-space MDS, participant centroids, and feature-support plots.

## Stage 6 — In-sample exploratory H-form and compression curve

Use existing fits as a first diagnostic if train/test is not ready.

## Stage 7 — Cross-validated H-form and compression curve

Implement train/test splits and evaluate held-out prediction.

## Stage 8 — Cluster stability and main-text summaries

Run bootstrap/fold stability, interpret clusters, and create final plots.

---

# 33. Tests and sanity checks

1. **Extraction integrity**  
   Summing per-participant/per-role losses should reproduce population-level IC loss for each model.

2. **Role handling**  
   Combined-role loss should equal chooser loss plus predictor loss.

3. **BIC weights**  
   Weights should sum to 1 for each participant.

4. **N_eff**  
   `N_eff` should be near 1 when one model dominates.

5. **Participant distance symmetry**  
   `D(i,j)` should equal `D(j,i)`.

6. **Embedding matrix diagonal**  
   For embedding, diagonal should be forced to zero.

7. **Feature support sanity**  
   Feature probabilities should lie in `[0,1]`.

8. **Codebook monotonicity**  
   Training loss should not increase as `K` increases.

9. **Held-out caution**  
   Held-out loss can increase with `K`; this is informative and may indicate overfitting.

10. **No accidental refitting**  
   First-pass extraction should not rerun expensive optimization unless explicitly requested.

---

# 34. Final note to the coding agent

This analysis is not just a technical add-on. It addresses a deep interpretive question in the paper.

The current IC analysis identifies a strong population-level utility function. This new analysis asks whether that function is a **shared architecture** or a **population compromise**. If participants cluster tightly around one model family, the seven-parameter coordinate-system claim becomes stronger. If they split into a small number of stable, predictive, interpretable clusters, the paper may reveal functional moral phenotypes. If they scatter without stability or held-out improvement, that also matters: it tells us the current data support population-level architecture better than individual-level architecture selection.

Build this carefully. Preserve the distinction between in-sample exploration and held-out prediction. Represent participants as weighted clouds, not brittle top-model labels. Reuse existing IC, utility, nesting, and AMPD infrastructure wherever possible.
