# Addendum: Improve the Utility-Architecture Compression Curve Plan

> **Audience:** coding agent working on the IHM / UBM codebase  
> **Purpose:** update the existing Stage 9 / Population Architecture Compression Curve plan with clarified scoring, stopping criteria, AMPD interpretation, candidate filtering, and redundancy diagnostics.  
> **Tone:** this is a planning document, not final code. Please integrate these recommendations into the existing planning file before implementation.

---

## 0. Big-Picture Correction

The **Utility-Architecture Compression Curve** should be the central analysis for deciding how many utility-function architectures are needed to describe individual differences.

**AMPD should not drive the selection of `K`.**

Instead:

> **Compression chooses the library.**  
> **AMPD explains the library.**  
> **Cross-validation checks whether the library generalizes.**

The earlier plan treated AMPD/model-space geometry as if it might be the first step in deciding how many architectures exist. That was useful conceptually, but the cleaner implementation is this:

1. Use the participant × model fit matrix to choose the best library of `K` utility architectures.
2. Use stopping criteria on the compression curve to decide how many architectures are justified.
3. Use AMPD afterward to interpret whether the selected architectures are behaviorally distinct or merely near-duplicates.
4. Use cross-validation / `H_form` later as a generalization check, not as a blocker for the first implementation.

The core scientific question is:

> **Is the population-winning utility function a shared coordinate system, or do participants require multiple utility architectures?**

---

## 1. Clarify the Meaning of Maximum `K`

There are two senses of maximum `K`.

### Theoretical maximum

If `K` simply means “number of models allowed in the library,” then the theoretical maximum is the total number of candidate utility functions.

```text
K_theoretical_max = n_candidate_models
```

The code can allow this.

### Maximum useful `K`

Under the current hard-assignment rule, each participant chooses their best-fitting model from the library. Therefore, the fully individualized optimum is achieved once the library includes every participant’s personal best model.

```text
unique_individual_best_models = unique(argmin_m score[i, m] for each participant i)
K_useful_max = len(unique_individual_best_models)
```

Since there can be no more unique individual winners than participants:

```text
K_useful_max <= n_participants
```

Adding more models after all unique individual winners are included cannot improve the globally optimal score unless:

- the search algorithm is greedy/suboptimal and failed to include some individual winners earlier,
- the candidate set changes across runs,
- or participants are allowed to use mixtures/ensembles of models, which is **not** the current plan.

### Implementation recommendation

Allow `K_max` to be as large as the number of candidate models, but compute and report:

```text
n_unique_individual_best_models
K_useful_max
fully_individualized_score_reached
```

If `K > K_useful_max` improves the score in a supposedly optimal search, print a diagnostic warning because this indicates either a search approximation issue or a scoring/candidate-set inconsistency.

---

## 2. `K = 1` Must Reproduce the Population IC Winner

This is a required sanity check.

The existing IC analysis already fits each utility function at the participant/role level and then aggregates losses across participants and roles. The code stores per-model minima in `minvec`, and the IC code asserts that `total_loss_model` equals the sum over `minvec[player_uuid]['loss']['chooser'] + minvec[player_uuid]['loss']['predictor']`.

Therefore, if the compression analysis is using the same candidate models and the same scoring objective as the IC analysis:

```text
K = 1 winner == population IC winner
```

If this fails, do **not** treat it as a random alternative result. Treat it as a scoring mismatch or extraction problem until proven otherwise.

---

## 3. Be Careful: Sum of Participant-Level BIC Is Not Necessarily the Same as Population BIC

This is the most important scoring clarification.

The existing Stage 9 plan uses `BIC_individual` from `participant_model_combined_fits.csv` and sums each participant’s minimum BIC across the `K`-architecture library. That is intuitive, but it may not reproduce the original population IC winner.

Why?

For a single model `m`, the original population IC score is approximately:

```text
BIC_population[m] = 2 * sum_i NLL[i, m] + k_m * log(sum_i n_data[i, m])
```

But summing participant-level BIC gives:

```text
sum_i BIC_individual[i, m]
    = 2 * sum_i NLL[i, m] + sum_i k_m * log(n_data[i, m])
```

Those penalties are different. If every participant has around 120 responses, then:

```text
sum_i k_m * log(120)
```

is much larger than:

```text
k_m * log(73 * 120)
```

So `sum_individual_BIC` can over-penalize complex models relative to the original IC analysis. That could make `K = 1` select a different model than the published/population IC winner.

### Implementation recommendation: compute multiple score matrices

The compression function should compute and save multiple scoring bases.

#### A. Raw NLL matrix

```text
participant_nll[i, m] = chooser_nll[i, m] + predictor_nll[i, m]
```

This is the cleanest fit-only quantity.

#### B. Individual-level BIC matrix

```text
participant_BIC[i, m] = 2 * participant_nll[i, m] + k_effective_m * log(n_data_i_m)
```

This is useful as a stricter individual-level model-selection score, but it should not be assumed to match the original population IC winner.

#### C. IC-equivalent participant score matrix — recommended primary for this compression analysis

Create a per-participant score whose sum over participants exactly reproduces the original population IC BIC for each model.

For model `m`:

```text
BIC_population[m] = 2 * sum_i NLL[i, m] + k_m * log(n_data_total_m)
```

Allocate the model-level complexity penalty across participants. If each participant’s `n_data` is available, allocate by data share:

```text
complexity_share[i, m]
    = k_m * log(n_data_total_m) * (n_data[i, m] / n_data_total_m)

ic_equivalent_score[i, m]
    = 2 * NLL[i, m] + complexity_share[i, m]
```

If `n_data[i, m]` is unavailable or unreliable, allocate equally:

```text
complexity_share[i, m]
    = k_m * log(n_data_total_m) / n_participants
```

Then:

```text
sum_i ic_equivalent_score[i, m] == BIC_population[m]
```

This makes the `K = 1` compression result match the population IC winner while still allowing participant-specific architecture assignment for `K > 1`.

### Required output columns

For transparency, save at least these score columns:

```text
raw_nll_score_K
sum_individual_BIC_score_K
ic_equivalent_score_K
```

The default / primary score should be:

```text
score_basis = "ic_equivalent_participant_score"
```

But the other score bases should be available for sensitivity checks.

---

## 4. Candidate Filtering: Use Participant-Aware Filtering

The current plan’s `top_n_models_included` filter ranks models by mean BIC across participants and keeps the top `N`. This is computationally convenient, but it can accidentally remove models that are poor on average but excellent for a small subgroup.

That matters because the whole point of the analysis is to detect minority utility architectures.

### Recommended filter

Use a union of:

```text
population top N models by aggregate IC/BIC
∪ top R models for each participant
∪ all participant-level best models
∪ population IC winner
```

A simple default:

```python
population_top_n_models: int | None = 120
participant_top_r_models: int | None = 10
```

The final candidate set should include:

```text
1. The top population_top_n_models by aggregate BIC.
2. Every model that appears in the top participant_top_r_models for at least one participant.
3. Every participant's personal best model.
4. The population IC winner.
```

This preserves computational tractability while protecting against the key failure mode: excluding a model that fits a minority subgroup extremely well.

### Paper-facing recommendation

For exploratory/debugging runs, use the filtered candidate set.

For the paper-facing result, either:

- run the full model set, or
- show that the participant-aware filtered set produces the same selected `K` and essentially the same architecture library.

---

## 5. Update Stopping Criteria: Make Kneedle the Default Highlighted Criterion

The current plan computes marginal gain, cumulative gain, meta-BIC, and maximum curvature. Keep those, but update the default highlighted criterion.

### Recommended default

```python
stopping_criteria = "kneedle_elbow"
```

The Kneedle-style elbow is the closest implementation of the visual intuition: find the point where the compression curve bends away from the simple linear path between minimal and maximal individualization.

### Kneedle-style elbow procedure

Given `A(K)`:

1. Normalize `K` to `[0, 1]`.
2. Normalize `A(K)` to `[0, 1]`.
3. Draw the endpoint diagonal from `(0, 0)` to `(1, 1)`.
4. Select the `K` where the curve has the largest vertical distance above that diagonal.

For a concave increasing curve, this can be implemented as:

```text
kneedle_distance(K) = A_norm(K) - K_norm(K)
selected_K_by_kneedle = argmax_K kneedle_distance(K)
```

### Also keep finite-difference curvature

Still compute:

```text
delta_A_K  = A(K) - A(K - 1)
delta2_A_K = delta_A_K - delta_A_(K - 1)
```

The maximum-curvature criterion is useful, but it can be noisier than Kneedle because second differences are sensitive to small fluctuations.

### Marginal-gain off-by-one correction

If the first `K` where the marginal gain falls below threshold is `K = 5`, then the **fifth** architecture failed to earn its place. The selected number should usually be:

```text
selected_K_by_marginal_gain = K - 1
```

Store both:

```text
first_K_with_low_marginal_gain
selected_K_by_marginal_gain
```

Optionally support:

```python
n_consecutive_low_marginal_gains_required: int = 1
```

If set to `2`, the function waits for two consecutive low-gain increments before selecting the previous stable `K`.

### Keep all criteria in output

Always compute and store:

```text
selected_by_kneedle_elbow
kneedle_distance
selected_by_max_curvature
selected_by_marginal_gain
first_K_with_low_marginal_gain
selected_K_by_marginal_gain
selected_by_cumulative_gain
selected_by_meta_bic
```

The `stopping_criteria` argument should only determine which criterion is highlighted in printed output and plots. It should not prevent the function from computing the others.

---

## 6. Meta-BIC Should Be Demoted to Exploratory

The current plan defines:

```text
meta_BIC(K) = 2 * bic_score_K + K * mean_k_effective_in_set * log(N_participants)
```

This is not obviously valid because `bic_score_K` may already be BIC-like. Multiplying it by 2 and adding another penalty risks mixing scales or double-counting complexity.

Keep a meta-BIC-style score only as an exploratory diagnostic unless a cleaner derivation is added.

Better names:

```text
architecture_penalized_score_K
exploratory_meta_BIC_K
```

Do not make this the default criterion.

---

## 7. AMPD After Selection: Interpret the Library

Once the compression analysis selects a candidate library of `K` models, use AMPD to ask whether those selected architectures are meaningfully different.

For each selected library, compute:

```text
library_ampd_min
library_ampd_mean
library_ampd_median
library_ampd_max
nearest_selected_pair_by_ampd
ampd_to_population_winner for each selected model
```

This answers:

> Are the selected models behaviorally distinct, or are some nearly redundant variants?

AMPD should not automatically override fit. A model can be AMPD-close to another selected model but still fit a small subgroup meaningfully better. Therefore, AMPD is a **diagnostic and interpretive layer**, not the primary selection criterion.

---

## 8. Condensing the Selected Library: Still Partly Unresolved, but Implement Diagnostics Now

The question of when to condense a selected library is not fully settled. The coding task should therefore implement diagnostics and flags rather than automatically deleting models.

### Core diagnostics for each selected model

For each model `m` in a selected `K`-architecture library, compute:

```text
assigned_n_m
assigned_percent_m
nearest_selected_model_idx_m
nearest_selected_model_ampd_m
pruning_cost_m
pruning_cost_normalized_m
```

Where:

```text
pruning_cost_m = score(library_without_m) - score(library)
```

and:

```text
pruning_cost_normalized_m
    = pruning_cost_m / (score_K1 - score_fully_individualized)
```

The pruning cost is more important than the historical “marginal gain when added,” because pruning cost directly asks:

> If we remove this model now, how much worse does the final selected library become?

### Why the three dimensions matter

A selected model looks potentially redundant when:

1. **Few participants are assigned to it.**  
   A model used by zero participants is definitely unnecessary. A model used by one participant is suspicious, but may still be meaningful if the fit gain is large.

2. **Removing it barely hurts fit.**  
   This is the most important criterion. If pruning cost is tiny, the model is not doing much work.

3. **It is AMPD-close to another selected model.**  
   This suggests it may be a near-duplicate architecture rather than a meaningfully distinct family.

These criteria are related but not identical. The code should compute all three and allow us to inspect their correlations.

### Recommended flagging rule

Do **not** automatically remove models at first. Instead, create flags.

Suggested defaults:

```python
min_assigned_participants_for_clear_support = 2
small_pruning_cost_threshold = 0.01
low_ampd_percentile_threshold = 0.05
```

A model is a **strong redundancy candidate** if all three are true:

```text
assigned_n_m < min_assigned_participants_for_clear_support
AND pruning_cost_normalized_m < small_pruning_cost_threshold
AND nearest_selected_model_ampd_percentile < low_ampd_percentile_threshold
```

A model is a **moderate redundancy warning** if any two of the three are true.

Do not remove a model based on `assigned_n` alone. A one-participant architecture could represent a rare-but-real minority type if removing it substantially worsens fit. Likewise, do not remove a model based on low AMPD alone. A globally similar model may differ exactly where one subgroup’s behavior is diagnostic.

### Optional redundancy score

If a single scalar score is useful, compute an exploratory redundancy score from percentile ranks:

```text
assigned_redundancy_percentile_m
    = percentile rank of low assigned_n_m

pruning_redundancy_percentile_m
    = percentile rank of low pruning_cost_normalized_m

similarity_redundancy_percentile_m
    = percentile rank of low nearest_selected_model_ampd_m
```

Then:

```text
redundancy_score_m
    = 0.50 * pruning_redundancy_percentile_m
    + 0.30 * assigned_redundancy_percentile_m
    + 0.20 * similarity_redundancy_percentile_m
```

This weighting makes pruning cost the most important criterion. Treat this score as a diagnostic, not a final decision rule.

---

## 9. Assignment Summaries Are Essential

For every `K`, save an assignment table:

```text
K
player_uuid
assigned_utility_idx
assigned_model_rank_for_player
assigned_model_delta_score_for_player
assigned_model_delta_BIC_for_player
assigned_model_AMPD_to_population_winner
assigned_model_AMPD_to_nearest_selected_model
```

For each selected model in each library, also save:

```text
K
utility_idx
assigned_n
assigned_percent
mean_assigned_participant_delta_score
median_assigned_participant_delta_score
nearest_selected_model_idx
nearest_selected_model_ampd
pruning_cost
pruning_cost_normalized
redundancy_warning_level
```

This matters because a selected `K = 5` library is not scientifically convincing if almost nobody is assigned to one or two of the models. The compression curve should create pressure against unused or barely used architectures, but the output must make this visible.

---

## 10. Interpret Selected Architectures Psychologically

Do not report selected models only as utility indices.

For each selected architecture, include:

```text
utility_idx
utility_bitstring
equation
k_params
all utility setting columns
assigned_n
assigned_percent
BIC_rank / ΔBIC from population IC
AMPD_to_population_winner
nearest_selected_model_by_AMPD
```

Also summarize the settings that distinguish selected architectures from each other.

Example interpretive questions:

```text
Does one selected architecture include social comparison while another does not?
Does one separate envy and guilt while another collapses them?
Does one use exponents while another is linear?
Does one use conditional welfare or min-max structure?
```

The scientific result should be described in terms of functional ingredients, not only model IDs.

---

## 11. Cross-Validated `H_form` Is a Generalization Check, Not the Main Analysis

The compression curve is the central in-sample / IC-style analysis.

Cross-validation answers a different question:

> If we select individualized architectures from one part of the data, do they predict held-out data better than the common architecture?

That is a generalization check.

For the first implementation, do not let cross-validation block the compression curve. But keep it in the plan as a robustness layer.

The simple extreme comparison is:

```text
K = 1 common architecture
vs.
K = fully individualized architecture
```

with:

```text
H_form = (NLL_common_test - NLL_individual_test)
         / (NLL_chance_test - NLL_individual_test)
```

Interpretation:

```text
H_form ≈ 0
    Individual architecture adds little or nothing beyond a common architecture.

H_form > 0
    Individual architecture improves held-out prediction.

H_form < 0
    Individual architecture overfits or is unstable.
```

A full cross-validated compression curve can be added later for selected candidate `K` values.

---

## 12. Data Adequacy / Stability Checks

The data-adequacy question should be treated as a calibration layer, not as the central analysis.

Start with real-data resampling before synthetic simulations.

### Real-data stability checks

Resample/subsample:

```text
participants
trials/games within participant
```

Track:

```text
selected_K_by_kneedle
selected architecture set
A(K)
assignment stability
library AMPD diagnostics
```

This asks:

> Does the architecture-compression answer stabilize as we approach the actual sample size and actual number of games?

### Synthetic checks later

Synthetic data can test known-truth recovery, but it requires assumptions about synthetic participant parameters. Use both:

```text
empirical-realistic parameter sampling
uniform / broad stress-test sampling
```

Do not make synthetic data adequacy the first implementation target.

---

## 13. Function Signature Updates

Update the compression function signature to reflect these decisions.

Suggested additions / changes:

```python
def compute_architecture_compression_curve(
    general_settings: dict,
    file_paths: dict,
    population_top_n_models: int | None = 120,
    participant_top_r_models: int | None = 10,
    K_max: int | None = None,
    exhaustive_K_max: int = 4,
    score_basis: Literal[
        "ic_equivalent_participant_score",
        "sum_individual_BIC",
        "raw_NLL",
    ] = "ic_equivalent_participant_score",
    stopping_criteria: Literal[
        "kneedle_elbow",
        "marginal_gain",
        "cumulative_gain",
        "max_curvature",
        "meta_bic",
    ] = "kneedle_elbow",
    marginal_gain_threshold: float = 0.01,
    n_consecutive_low_marginal_gains_required: int = 1,
    cumulative_gain_threshold: float = 0.80,
    diagnose_selected_library_redundancy: bool = True,
    ampd_matrix_name_or_path: str | None = None,
    n_workers: int | None = None,
    create_new_file: bool = False,
) -> pd.DataFrame:
```

Do not feel obligated to use exactly this signature if the existing codebase suggests a cleaner variant. But the function should support these conceptual options.

---

## 14. New / Revised Output Files

### Primary curve

```text
processed/population_architecture_curve.csv
```

Add columns:

```text
K
architecture_set_idxs
search_method
score_basis
raw_nll_score_K
sum_individual_BIC_score_K
ic_equivalent_score_K
score_K1
score_fully_individualized
A_K
delta_A_K
delta2_A_K
K_useful_max
n_unique_individual_best_models
selected_by_kneedle_elbow
kneedle_distance
first_K_with_low_marginal_gain
selected_K_by_marginal_gain
selected_by_cumulative_gain
selected_by_max_curvature
selected_by_meta_bic
library_ampd_min
library_ampd_mean
library_ampd_median
library_ampd_max
nearest_selected_pair_by_ampd
```

### Participant assignments

```text
processed/population_architecture_assignments.csv
```

Required columns:

```text
K
player_uuid
assigned_utility_idx
assigned_model_rank_for_player
assigned_model_delta_score_for_player
assigned_model_delta_BIC_for_player
assigned_model_AMPD_to_population_winner
assigned_model_AMPD_to_nearest_selected_model
```

### Selected-library diagnostics

```text
processed/population_architecture_library_diagnostics.csv
```

Required columns:

```text
K
utility_idx
assigned_n
assigned_percent
nearest_selected_model_idx
nearest_selected_model_ampd
nearest_selected_model_ampd_percentile
pruning_cost
pruning_cost_normalized
redundancy_warning_level
redundancy_score_optional
```

---

## 15. Verification Checklist

Before trusting results:

```text
[ ] Candidate model count is read from data; no hard-coded 480.
[ ] K = 1 primary score reproduces the population IC winner.
[ ] If K = 1 differs, diagnostic output explains whether this is due to score_basis, candidate filtering, role inclusion, or extraction failure.
[ ] The candidate set always includes the population IC winner.
[ ] The candidate set always includes every participant-level best model.
[ ] A(K) is monotonic non-decreasing.
[ ] score_K is monotonic non-increasing.
[ ] The curve reaches the fully individualized score by K_useful_max under exhaustive/global search.
[ ] Marginal-gain off-by-one correction is implemented.
[ ] Kneedle elbow is computed and saved.
[ ] AMPD diagnostics are computed after library selection, not used as the primary K-selection objective.
[ ] Redundancy diagnostics flag models; they do not silently delete them.
```

---

## 16. Final Conceptual Summary

The improved Stage 9 should answer three nested questions:

1. **How many architectures are needed?**  
   The Utility-Architecture Compression Curve answers this by finding the best `K`-model libraries.

2. **Are the selected architectures meaningfully different?**  
   AMPD answers this by measuring behavioral similarity among selected models.

3. **Do individualized architectures generalize?**  
   Cross-validated `H_form` answers this later by testing held-out prediction.

The main analysis is the compression curve. AMPD is the interpretive geometry. Cross-validation is the generalization check.

