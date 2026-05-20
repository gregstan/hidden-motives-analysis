# Plan: Population Architecture Compression Curve (Stage 9)

---

## Scientific Goal

Participants in this experiment were modelled with hundreds of utility function architectures — mathematical specifications of how a person weighs their own payoff, others' payoffs, inequality, reciprocity, and related social preferences. The best single architecture for the whole population is already known from the information-criterion (IC) analysis. But participants may differ not only in the *values* of their parameters within one architecture, but in which *structural form* of the utility function best describes them. A person who cares about reciprocity requires a fundamentally different utility function than one who cares only about material outcomes, regardless of how the parameters are tuned.

This analysis asks: **how many distinct utility function architectures does the population actually need?** It proceeds as follows: for K = 1, 2, 3, …, find the library of K utility function architectures such that, when every participant is free to use the architecture that fits them best, the total BIC cost across the population is minimized. The result is a **compression curve**: as K grows, the curve tracks what fraction of the fully individualized advantage (giving every participant their own unique best architecture) is recovered. The knee of this curve — the point where adding another architecture stops delivering meaningful improvement — is the answer to how many functional types are needed to describe the population.

Three nested questions are answered in sequence:

1. **How many architectures are needed?** — answered by the compression curve and its stopping criteria.
2. **Are the selected architectures meaningfully behaviorally distinct?** — answered by AMPD (Average Model Policy Distance) computed on the selected library after the curve identifies the winning K.
3. **Do individualized architectures generalize to held-out data?** — answered by cross-validated H_form as a robustness check (planned separately; not a blocker for the current implementation).

The architecture-compression curve is the **central analysis**. AMPD is the **interpretive geometry layer**. Cross-validation is the **generalization check**.

---

## Terminology

- **Architecture**: a utility function structure defined by which psychological components are active (e.g., does the model include altruism? reciprocity? inequality aversion? exponents?). Two participants can share an architecture but differ in the numeric values of its parameters.
- **K-architecture set** (also: *library of K architectures*): the set of K architectures available to the population. Each participant selects the architecture that fits them best (by BIC score).
- **BIC (Bayesian Information Criterion)**: a measure of model fit that penalizes parameter complexity. Lower BIC is better. BIC = 2 × NLL + k_effective × log(n_data), where NLL is negative log-likelihood and k_effective is the number of free parameters.
- **Population BIC score at K**: the sum, across all participants, of each participant's minimum BIC given the K-architecture library. This is the quantity being minimized.
- **A(K)**: the proportion of the fully individualized advantage captured by a K-architecture library. A(1) = 0 by definition (no individualization possible with a single shared architecture). A(∞) = 1 (everyone has their own architecture). The compression curve plots A(K) vs K.
- **IC-equivalent participant score**: a per-participant score whose sum across participants exactly reproduces the population-level IC BIC for each model (see Scoring section). This is the recommended primary scoring basis.
- **AMPD**: Average Model Policy Distance — a Monte Carlo measure of how differently two utility function architectures actually behave across simulated game scenarios. AMPD near zero means the architectures are behaviorally indistinguishable despite having different symbolic forms.
- **Kneedle elbow**: the K at which the compression curve has the greatest vertical distance above the straight line connecting its two endpoints (K=1 and K=K_max). This is the formal, threshold-free definition of the "knee" of the curve.
- **Pruning cost**: the increase in population BIC score when a specific architecture is removed from the selected library. High pruning cost means the architecture is doing essential work; near-zero pruning cost means the architecture is redundant.

---

## Data Sources

### Primary: `processed/participant_model_combined_fits.csv`

One row per participant × model. Key columns:
- `player_uuid`: participant identifier
- `utility_idx`: model identifier (integer)
- `combined_loss_nll`: in-sample negative log-likelihood for this participant under this model (chooser + predictor roles combined)
- `n_combined`: number of game observations for this participant
- `k_params`, `k_effective`: parameter counts

**Note on BIC and scoring:** The `BIC_individual` column in this CSV is computed as `2 × NLL + k_effective × log(n_combined)`. This is an individual-level BIC — it penalizes model complexity using each participant's own game count (~120 games). However, the population-level IC winner was selected using a pooled BIC that penalizes with the log of the *total* game count (~8,700). These two scoring approaches can differ in which model wins at K=1 (see Scoring section below). The function uses an **IC-equivalent participant score** as its primary scoring basis to ensure consistency with the published IC result.

### Secondary: IC CSV

`file_paths['file_names']['information_criterion']` — used to identify the population IC winner at startup. The winning model (model with the lowest aggregate BIC) is extracted at runtime; it is **not hardcoded**.

### Optional: AMPD matrix

The precomputed AMPD distance matrix (loaded via `compute_ampd_distance_matrix`) is used in the post-selection library diagnostics step. If no AMPD matrix is available, AMPD-dependent columns are filled with NaN.

---

## Scoring Basis (Critical Detail)

Three scoring schemes are computed and saved for transparency. The **primary** basis is (C).

### A. Raw NLL score
`raw_participant_score[i, m] = combined_loss_nll[i, m]`
Pure fit, no complexity penalty. Used as a sensitivity check.

### B. Sum-of-individual-BIC score
`individual_BIC[i, m] = 2 × NLL[i, m] + k_effective_m × log(n_combined_i)`
Penalizes with each participant's own game count. This can over-penalize complex models relative to the IC analysis: `Σᵢ k × log(n_i)` is larger than `k × log(Σᵢ n_i)` because log is concave. May produce a different K=1 winner than the IC champion.

### C. IC-equivalent participant score (recommended primary)
Allocates the population-level complexity penalty across participants proportionally by data share, so that `Σᵢ score[i, m]` exactly equals the population IC BIC for model m:

```
n_total_m = Σᵢ n_combined[i, m]
complexity_share[i, m] = k_effective_m × log(n_total_m) × (n_combined[i, m] / n_total_m)
ic_equivalent_score[i, m] = 2 × NLL[i, m] + complexity_share[i, m]
```

**Key property:** `Σᵢ ic_equivalent_score[i, m] = 2 × Σᵢ NLL[i, m] + k_effective_m × log(n_total_m) = BIC_population[m]`.

This guarantees that K=1 selects the same winner as the IC analysis. If the K=1 winner differs despite using this basis, the discrepancy must be attributed to candidate filtering or role inclusion differences — not the scoring formula itself.

**Self-check at startup:** After identifying the K=1 winner, compare it to the IC CSV winner. If they match: print a confirmation. If they do not match: print a detailed diagnostic (which model was selected, which was expected, what their scores are) and continue — do not halt, as a mismatch can itself be scientifically informative.

---

## Candidate Filtering (Participant-Aware)

A naive filter (keep only the top N models by mean population BIC) can accidentally exclude architectures that fit a small subgroup extremely well but are poor on average. Since the whole point of this analysis is to detect minority utility architectures, naive filtering risks excluding the very models the analysis is designed to find.

**Parameters:**
```
population_top_n_models: int | None = 120   # top N by aggregate IC BIC
participant_top_r_models: int | None = 10   # top R per participant by individual BIC
```

**Candidate set = union of:**
1. Top `population_top_n_models` models by aggregate BIC across all participants.
2. Every model in the top `participant_top_r_models` for at least one participant.
3. Every participant's individual best model (their personal argmin BIC).
4. The population IC winner (always included, regardless of filtering).

`None` for either parameter includes all models. Total number of candidates is read from the data — not hardcoded.

---

## Algorithm

### Step 1 — Build score matrix

Load `participant_model_combined_fits.csv`. Build a score matrix L (N_participants × N_candidate_models) using the `ic_equivalent_participant_score` basis (configurable via `score_basis`). Apply the candidate filter described above. Record `n_candidate_models` from the data.

### Step 2 — Compute anchor scores

- **K=1 score** (`score_K1`): find the single model that minimizes `L.sum(axis=0)`. This is the best single-architecture solution. Self-check against IC CSV winner (see above).
- **Fully individualized score** (`score_fully_individualized`): `L.min(axis=1).sum()` — each participant uses their personal best model. Also compute `n_unique_individual_best_models` = number of distinct models selected across participants, and `K_useful_max` = that count. Adding more models beyond `K_useful_max` cannot improve `score_fully_individualized` under optimal search.

### Step 3 — Find optimal K-architecture set for each K

**Exhaustive search (K ≤ exhaustive_K_max, default 4):**

Enumerate all C(N_candidates, K) possible sets. For each set S: `score(S) = L[:, S].min(axis=1).sum()`. Operations are batched (100,000 combinations per batch) to control memory (~200 MB per batch at K=4). Parallelized across `n_workers` independent slices using `mp.Pool`; each worker searches its slice and returns its local best.

Feasibility with N=120: K=2 (7k combos, milliseconds), K=3 (280k, ~1s), K=4 (8.1M, ~30s parallelized). With N=240: K=4 is 135M — hours; consider lowering `exhaustive_K_max` to 3.

**Greedy + local swap (K > exhaustive_K_max):**

Starting from the winning set at K-1:
1. *Greedy extension*: add the single model from the remaining candidates that maximally reduces the total score. Efficiently implemented using the current per-participant minimum as a baseline: `score(set ∪ {m}) = min(current_min_per_participant, L[:, m]).sum()`.
2. *Local swap improvement*: attempt to replace each current member with each non-member. Accept any swap that reduces total score. Repeat until no swap improves score (typically 1–3 passes). This corrects greedy's myopia and usually recovers the globally optimal set.

### Step 4 — Compute A(K) and all stopping criteria

```
A(K) = (score_K1 - score_K) / (score_K1 - score_fully_individualized)
ΔA(K) = A(K) − A(K−1)         [first finite difference; nan at K=1]
Δ²A(K) = ΔA(K) − ΔA(K−1)     [second finite difference; nan at K=1,2]
```

Compute all four stopping criteria (see Stopping Criteria section). Flag the selected K for each criterion in output columns. The `stopping_criteria` parameter only controls which criterion is highlighted in terminal output and plots — all criteria are always computed.

### Step 5 — AMPD diagnostics on the selected library

After computing the full curve, load the AMPD matrix (if available) and compute library-level AMPD statistics for each K:

```
library_ampd_min      — distance between the closest pair of selected architectures
library_ampd_mean     — mean pairwise distance within the selected library
library_ampd_median
library_ampd_max      — distance between the most distinct pair
nearest_selected_pair_by_ampd — (idx_a, idx_b) with the smallest pairwise AMPD
```

These answer: **are the selected architectures behaviorally distinct, or are some near-duplicates?** A small `library_ampd_min` means two selected architectures produce nearly identical behavior across game scenarios — they differ symbolically but not functionally. This does not automatically disqualify them (one may fit a small subgroup much better), but it is a signal worth reporting.

### Step 6 — Condensing the selected library: redundancy diagnostics

For each architecture in each K-set, compute per-architecture diagnostics that flag potential redundancy. **These are diagnostics only — the function does not automatically remove models.**

For each model m in the selected K-architecture set:

```
assigned_n_m                   — number of participants assigned to this architecture
assigned_percent_m             — assigned_n / N_participants
pruning_cost_m                 — score(library_without_m) - score(library)
pruning_cost_normalized_m      — pruning_cost_m / (score_K1 - score_fully_individualized)
nearest_selected_model_idx_m   — the other selected model with smallest AMPD to m
nearest_selected_model_ampd_m  — that AMPD value
nearest_selected_model_ampd_percentile_m — percentile of that AMPD in the full matrix
```

**Redundancy flagging:**
- *Strong candidate for removal*: `assigned_n < 2` AND `pruning_cost_normalized < 0.01` AND `nearest_ampd_percentile < 0.05`
- *Moderate warning*: any two of the three above conditions

**Weighted redundancy score (exploratory):**
```
redundancy_score = 0.50 × pruning_redundancy_percentile
                 + 0.30 × assigned_redundancy_percentile
                 + 0.20 × similarity_redundancy_percentile
```
Pruning cost is weighted most heavily because it directly answers "does removing this model hurt fit?"

Do not remove any model automatically. Save flags to the library diagnostics CSV for human inspection.

### Step 7 — Save and print

After each K completes, immediately append the result row to a partial CSV (`population_architecture_curve_partial.csv`). On startup with `create_new_file=False`, if the partial file exists and has valid rows, skip the already-completed K values and resume. On successful completion, write the three final CSVs and delete the partial file.

Print a one-line summary per K to the terminal, including ETA using `_fmt_duration` (analysis.py ~line 6166).

---

## Stopping Criteria

Controlled by:
```python
stopping_criteria: Literal[
    "kneedle_elbow", "marginal_gain", "cumulative_gain", "max_curvature", "meta_bic"
] = "kneedle_elbow"
```

All criteria are always computed and saved. The `stopping_criteria` parameter only determines which is highlighted in printed output and chart annotations.

### Default: Kneedle Elbow

The Kneedle elbow is the closest implementation of the visual intuition: find the point where the compression curve bends away from the straight line between its two endpoints (K=1, A=0) and (K_max, A(K_max)).

**Procedure:**
1. Normalize K to [0, 1]: `K_norm = (K - 1) / (K_max - 1)`
2. Normalize A(K) to [0, 1]: `A_norm = A(K) / max(A(K))`
3. For each K, compute: `kneedle_distance(K) = A_norm(K) - K_norm(K)`
4. Select: `K* = argmax_K kneedle_distance(K)`

**Why defensible:** Directly formalizes the visual intuition of "the bend in the curve." No threshold required — the optimal K is chosen automatically by maximizing departure from linearity. Well-established in clustering literature (the "elbow method" for K-means).

**Limitation:** Requires choosing K_max to define the normalization range; results can shift if K_max changes.

### Marginal Gain (ΔA < threshold)

```
first_K_with_low_marginal_gain = min K > 1 where ΔA(K) < marginal_gain_threshold
selected_K_by_marginal_gain = first_K_with_low_marginal_gain - 1
```

Note the off-by-one: if K=5 is the first K where the marginal gain falls below threshold, the fifth architecture *failed* to earn its place, so the answer is K=4.

**Parameter:** `marginal_gain_threshold: float = 0.01` (1%)
**Optional:** `n_consecutive_low_marginal_gains_required: int = 1` — if set to 2, waits for two consecutive low-gain increments before committing.

**Why defensible:** Each new architecture must recover ≥1% of the individualized advantage to justify its inclusion. Analogous to factor analysis's practice of dropping factors explaining <1% of variance.

**Limitation:** The 1% threshold is arbitrary. Always report the full curve.

### Cumulative Gain (A(K) > threshold)

```
selected_K_by_cumulative_gain = min K where A(K) > cumulative_gain_threshold
```

**Parameter:** `cumulative_gain_threshold: float = 0.80`

**Why defensible:** "The smallest library capturing 80% of the fully individualized advantage." Analogous to retaining PCA components explaining ≥80% of variance. Intuitive for a behavioral science audience.

### Maximum Curvature (calculus-inspired)

```
selected_K_by_max_curvature = argmax_K |Δ²A(K)|    for K ≥ 2
```

The second finite difference approximates the curvature of the compression curve. Since A(K) is concave by construction, Δ²A ≤ 0 everywhere; the knee is where curvature is most negative (largest absolute value = sharpest deceleration of improvement).

**Why defensible:** The discrete analog of finding the inflection point of the cumulative gain curve. No threshold required.

**Limitation:** Sensitive to noise; less reliable when improvement is spread gradually across many K values.

### Meta-BIC (exploratory only)

```
exploratory_meta_bic(K) = 2 × bic_score_K + K × mean_k_effective_in_set × log(N_participants)
selected_K_by_meta_bic = argmin_K exploratory_meta_bic(K)
```

Treats K itself as a model complexity parameter and penalizes it with BIC logic. Named `exploratory_meta_bic` to signal that this formula is heuristic (the 2× factor may double-count complexity if `bic_score_K` already incorporates BIC penalties). Useful cross-check; not the primary criterion.

---

## Function Signature

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
        "kneedle_elbow", "marginal_gain", "cumulative_gain",
        "max_curvature", "meta_bic",
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

`K_max=None` means run until the selected stopping criterion is met, or until all candidate models are exhausted. `n_workers` is clamped to `[1, cpu_count - 1]`; `None` = `cpu_count - 1`.

Returns one DataFrame (one row per K). The old function returned a tuple; this no longer has a fold structure.

---

## Output Files

### `processed/population_architecture_curve.csv`
One row per K. Columns include: K, architecture_set_idxs (JSON), search_method, score_basis, raw_nll_score_K, sum_individual_BIC_score_K, ic_equivalent_score_K, score_K1, score_fully_individualized, A_K, delta_A_K, delta2_A_K, K_useful_max, n_unique_individual_best_models, kneedle_distance, selected_by_kneedle_elbow, first_K_with_low_marginal_gain, selected_K_by_marginal_gain, selected_by_marginal_gain, selected_by_cumulative_gain, selected_by_max_curvature, exploratory_meta_bic, selected_by_meta_bic, library_ampd_min, library_ampd_mean, library_ampd_median, library_ampd_max, nearest_selected_pair_by_ampd.

### `processed/population_architecture_assignments.csv`
One row per K × participant. Columns: K, player_uuid, assigned_utility_idx, assigned_model_rank_for_player, assigned_model_delta_score_for_player, assigned_model_AMPD_to_population_winner, assigned_model_AMPD_to_nearest_selected_model.

### `processed/population_architecture_library_diagnostics.csv`
One row per K × selected architecture. Columns: K, utility_idx, equation, k_params, utility settings columns, assigned_n, assigned_percent, pruning_cost, pruning_cost_normalized, nearest_selected_model_idx, nearest_selected_model_ampd, nearest_selected_model_ampd_percentile, redundancy_warning_level, redundancy_score_optional.

### `processed/population_architecture_curve_partial.csv`
Partial results; appended after each K completes. Deleted on successful completion. Enables resume after interruption.

### `visuals/population_architecture_curve.html`
Interactive Plotly figure — see Plot Function section below.

---

## Plot Function

### `plot_architecture_compression_curve(general_settings, file_paths, fig_lay, export_fig=True) -> go.Figure`

Replaces the old function of the same name. Reads `processed/population_architecture_curve.csv` (one row per K, no fold structure) and produces an interactive Plotly figure exported to `visuals/population_architecture_curve.html`.

**Key differences from the old function:**
- No error bars (single BIC score per K, no fold SE).
- Hover text shows: K, A(K), ΔA(K), Δ²A(K), kneedle_distance, architecture_set_idxs (JSON list of model indices).
- Vertical knee markers for **all five** stopping criteria, each in a distinct color/dash style:
  - Kneedle elbow — `royalblue`, solid
  - Marginal gain — `darkorange`, dotted
  - Cumulative gain — `darkgreen`, dash-dot
  - Max curvature — `purple`, dashed
  - Meta-BIC — `firebrick`, long-dash
  Only criteria whose selected K is non-null are drawn; if two criteria agree on the same K, a single combined annotation is used.
- Secondary y-axis (right side, lighter color) for `kneedle_distance` as a line trace, so the reader can see where the elbow peaks.
- Horizontal reference lines at A=0 (K=1 baseline) and A=1 (fully individualized).
- Reads columns from the new CSV: `K`, `A_K`, `delta_A_K`, `delta2_A_K`, `kneedle_distance`, `architecture_set_idxs`, `selected_by_kneedle_elbow`, `selected_by_marginal_gain`, `selected_by_cumulative_gain`, `selected_by_max_curvature`, `selected_by_meta_bic`.
- Output path: `visuals/population_architecture_curve.html` (not `architecture_compression_curve.html`).

**Reused from old function (nearly verbatim):**
- Overall `go.Scatter` + `go.Figure` structure.
- `fig.update_layout` (title, axis labels, template, font, hover label size, margins, legend position).
- `fig.write_html` export block.
- `add_hline` reference lines.

---

## New Module-Level Parallel Worker

```python
def _exhaustive_search_worker(args: tuple) -> tuple:
    """
    Parallel worker for exhaustive architecture set search.
    Receives a slice of combination indices and the BIC matrix L.
    Returns (best_score, best_architecture_set_indices) for its slice.
    """
```

Inserted near the other parallel workers (e.g., `_cv_architecture_losses_worker`) before the main function.

---

## Files Modified

| File | Change |
|---|---|
| `analysis.py` | Add `_exhaustive_search_worker`; replace `compute_architecture_compression_curve` (~lines 10128–10440) and `plot_architecture_compression_curve` (~lines 10443–10572) |
| `main.py` | Update Stage 9 commented block to use new signature |
| `plans/population_architecture_compression_curve.md` | Copy of this plan |

---

## Verification Checklist

- [ ] Candidate model count is read from data; no hard-coded 480
- [ ] K=1 primary score reproduces population IC winner; if not, diagnostic output explains why
- [ ] Candidate set always includes the population IC winner and every participant's personal best model
- [ ] A(K) is monotonic non-decreasing; score_K is monotonic non-increasing
- [ ] The curve reaches `score_fully_individualized` by K_useful_max under exhaustive/global search
- [ ] Marginal-gain off-by-one correction implemented: `selected_K_by_marginal_gain = first_K_with_low_marginal_gain - 1`
- [ ] Kneedle elbow computed and saved in `kneedle_distance` and `selected_by_kneedle_elbow` columns
- [ ] All four stopping criteria computed regardless of which `stopping_criteria` is selected
- [ ] AMPD diagnostics computed after library selection (not used as primary K-selection criterion)
- [ ] Redundancy diagnostics flag models; do not silently remove them
- [ ] Partial CSV enables resume after interruption
- [ ] Re-run with `create_new_file=False` loads from final CSV; no recomputation
- [ ] `plot_architecture_compression_curve` produces `visuals/population_architecture_curve.html` with vertical markers for all five stopping criteria and a secondary kneedle-distance trace

---

## Appendix: The Meaning of Maximum K

There are two distinct concepts of the maximum meaningful K.

### Theoretical maximum
The theoretical maximum K is the number of candidate models (e.g., 120 or 240 after filtering). The function allows K to grow this large.

### Useful maximum
Under the hard-assignment rule (each participant picks the best single architecture), the fully individualized optimum is achieved as soon as the library includes every participant's personal best model. The number of unique personal best models is `K_useful_max ≤ N_participants`. Beyond this K, adding more architectures cannot improve the score under any globally optimal search. The function computes and reports `K_useful_max` and `n_unique_individual_best_models` as output columns. If a greedy search improves the score beyond `K_useful_max`, this signals a search approximation issue and triggers a diagnostic warning.

This is a key conceptual difference from the previous Stage 9 implementation, which used cross-validated folds and was implicitly bounded by the fold structure.

---

## Scientific Interpretation Guide (for Paper Writing)

**"How many architectures does the population need?"**
Report the K flagged by each criterion. If all four agree, the answer is robust. If they disagree, the curve is gradual and no single K is unambiguously "the" answer — which is itself a finding worth reporting.

**"What does A(K) = 0.6 at K=2 mean?"**
Two architectures capture 60% of what you would gain by giving every participant their own unique best model. If one architecture described everyone well, this gain would be negligible.

**"If K=2 doesn't include the IC champion, is that a problem?"**
No — it is scientifically interesting. It means the population IC champion is a compromise between two clusters, each better described by a different architecture. The optimal K=2 library may describe structural heterogeneity more accurately than forcing the IC winner into the picture.

**"Are the selected architectures really distinct?"**
AMPD answers this. A library with `library_ampd_min ≈ 0` contains architectures that behave nearly identically across game scenarios despite differing mathematically. A library with uniformly large AMPD values contains architectures that make genuinely different behavioral predictions.

**"Is this analysis susceptible to overfitting?"**
BIC penalizes parameter complexity within each model. The IC-equivalent scoring ensures the K=1 result matches the published IC analysis. The stopping criteria prevent choosing more K than the data support. Cross-validated H_form (computed separately) provides a held-out generalization check.

**Psychological interpretation of selected architectures:**
For each selected architecture, report: utility_idx, the utility equation, k_params, all active utility settings (reciprocity? altruism? inequality aversion? exponents?), assigned_n (how many participants use it), and AMPD to the population IC winner. Summarize which psychological ingredients distinguish the selected architectures from each other — this is the scientifically meaningful content of the result.

---

**"Does the Kneedle elbow depend on the choice of K_max, and if so, is that a problem?"**

Yes — the Kneedle distance is sensitive to K_max by construction, and this needs to be addressed explicitly in the paper.

*The math.* The Kneedle procedure normalizes K to [0, 1] using K_max as the right endpoint: K_norm = (K − 1) / (K_max − 1). The elbow is the K that maximizes the vertical distance of A(K) above the straight line connecting (K=1, A=0) to (K=K_max, A(K_max)). If K_max were smaller, the x-scaling would compress, each K would appear further to the right in normalized space, and the elbow would shift accordingly. The Kneedle elbow is therefore a statement about the shape of the curve *within the searched range*, not an absolute quantity.

*Why K_useful_max is the principled right endpoint.* The search was run to K_useful_max — the number of distinct utility architectures that are at least one participant's personal best fit (22 in this dataset). This is not an arbitrary cutoff chosen by the researcher. It is the exact K at which A(K) = 1 is first achievable: for the first time, every participant has their preferred architecture available in the library. Beyond K_useful_max, no reduction in population BIC is possible, so the curve is completely traced. The compression curve spans the full range from A(1) = 0 (forced to use one architecture for everyone) to A(K_useful_max) = 1 (everyone gets their preferred architecture), and the Kneedle elbow identifies the point of maximum benefit-to-cost ratio within that complete range. This is the correct scientific framing.

*How to write this in the paper.* Something like: "We searched from K = 1 to K_useful_max = 22, where K_useful_max is the number of architectures that are at least one participant's best-fitting model — the first K at which the fully individualized BIC optimum is achievable. The Kneedle elbow was computed over this complete range, so it identifies the K that captures the greatest fraction of the individualized advantage per additional architecture relative to the theoretical maximum. Because K_useful_max is data-determined rather than researcher-chosen, the elbow is anchored to a principled endpoint."

*Additional robustness.* Two of the other stopping criteria — marginal gain (ΔA < 1%) and maximum curvature (largest Δ²A) — do not depend on K_max at all; they are purely local measures of the curve's shape. If those criteria agree with the Kneedle result, the elbow is robust to the K_max dependence. Report all criteria and note agreement or disagreement.

*The one residual limitation.* Even with a principled K_max, a reviewer could note that K_useful_max depends on sample size: with more participants, more unique personal bests would exist, potentially raising K_useful_max. The appropriate response is that K_useful_max is descriptive of this sample and population (it characterizes the actual diversity of best-fitting models in these data), and that the compression curve describes the tradeoff between parsimony and fit for this population — which is exactly the quantity of scientific interest.

---

## Appendix: Empirical Findings from the Compression Curve Analysis (Experiment Data Notes)

*These notes record observations made during analysis of the actual data and are intended to inform paper writing. Numbers are specific to this dataset (N = 73 participants, K_useful_max = 22).*

---

### Finding 1: The K = 2 jump is the headline result (ΔA = 0.58)

Going from one shared utility function to two captures **58% of the total individualized advantage** in a single step. This is a large and important finding. A ΔA of 0.58 at K = 2 means that the population is genuinely bimodal in utility function type — the single IC-winning model is a poor descriptor of approximately half of all participants. The compression curve argues forcefully that a split library has large scientific value over a single consensus model, even before asking which models belong in that library.

---

### Finding 2: The IC winner is a statistical compromise, not anyone's best model

At K = 1, the IC-winning model (index 443) is assigned to all 73 participants and achieves a mean individual BIC of 131.7. When K = 2 allows participants to self-select into the library that fits them best, the 36 participants assigned to model 443 have a mean individual BIC of **142.0** — *worse* than the 37 participants assigned to model 455, who average **118.3**. The participants who "stay" with the IC winner are actually worse fit by it than the defectors are by their alternative.

This finding has direct implications for how population-level results should be reported. When researchers report the distribution of fitted parameters using a single best-fit model — the IC analysis nominates model 443 — they are fitting participants with a functional form that is suboptimal for a substantial portion of them. The parameter distributions, and the substantive conclusions drawn from them (e.g., guilt is more powerful than envy; the altruism term reverses the Fehr-Schmidt inequality aversion finding), are specifically the result of model 443's functional structure. Those conclusions could shift under a different utility function, and indeed likely would for the participants who are better described by model 455. This is an important caveat to be transparent about in the paper.

---

### Finding 3: The K = 2 split is nearly 50/50, behaviourally distinct, and psychologically interpretable

At K = 2, the population splits into two nearly equal groups: 36 participants (49%) assigned to model 443 and 37 (51%) assigned to model 455. The AMPD between these two models is **0.128, at the 90th percentile** of all pairwise behavioural distances in the full model space. These two models are not near-duplicates — they make genuinely different behavioural predictions across game scenarios.

The split maps onto a meaningful psychological distinction. Model 443 uses a difference-based social comparison utility: the participant's utility depends on the difference between their payoff and the other person's payoff, with a free altruism/envy parameter and curvature exponents. Model 455 uses a reference-dependent utility: outcomes are evaluated relative to a fixed reference payoff of 3, with separate parameters for gains above and losses below the reference (a gain/loss asymmetry structure). Roughly half the population treats social outcomes in terms of *relative differences*; the other half treats them in terms of *gains and losses relative to a reference point*. This is a substantive psychological split.

---

### Finding 4: Model 443 anchors every library; heroic specialists serve small minorities

One of the clearest structural patterns in the data is that **model 443 (the IC winner) appears in every optimal library from K = 1 to K = 22 without exception**. Its pruning cost remains high throughout — even at K = 8, removing it would cost a normalised BIC increase of approximately 0.08–0.46. It is not merely a legacy presence from K = 1; it is the genuine nucleus of every library, consistently assigned to 18–37% of participants at all levels.

By contrast, some models that enter later are striking for how well they fit their small assigned groups. **Model 452** achieves a mean individual BIC of **82–87** across all K values where it appears — exceptionally low, indicating that the participants who prefer it are captured almost perfectly by its functional form. **Model 373**, which first enters at K = 5, achieves a mean individual BIC of **79–80** for its 4 assigned participants. These are boutique models: they serve tiny minorities with extraordinary precision. The flip side is that their small coalitions make them hard to justify on parsimony grounds alone. They serve real people extremely well; they just don't serve many of them.

---

### Finding 5: The nested structure of optimal libraries, and the single exception

Across all K from 1 to 22, the optimal library at K is the optimal library at K − 1 plus exactly one new model. Library(4) is library(3) plus model 473. Library(5) is library(4) plus model 373. This nested, additive structure holds without exception from K = 3 through K = 22.

**The single exception is K = 2**, where model 455 appears — and then vanishes entirely from K = 3 through K = 8, before re-emerging at K = 9. This is the most structurally unusual feature of the entire compression curve.

---

### Finding 6: Model 455 as a big-tent coalition model — and the spoiler effect

Model 455's appearance at K = 2 and subsequent disappearance is not because a behaviourally similar cousin edges it out (though 452 and 455 do have a moderate AMPD of ~0.036, at the 32nd percentile). It is a more interesting phenomenon: **455 is functioning as a big-tent coalition model at K = 2**.

At K = 2, model 455 must cover everyone who is not best described by model 443. That is a heterogeneous group — participants who prefer reference-dependent utility, participants who prefer a pure-exponents form, and others who simply deviate from the 443 functional structure in different ways. Model 455 is the single best umbrella for all of these dissimilar people because it sits in a region of functional space that is tolerable (if not optimal) for each of them.

When K = 3 becomes available, the exhaustive search finds a better arrangement: the optimal library is {443, 452, 469}. Models 452 and 469 **split 455's constituency**. Model 452 is taken by participants it fits extremely well (mean BIC ~87); model 469 absorbs the remainder. This is a divide-and-conquer displacement, and the analogy to a spoiler effect in voting is apt: at K = 2, model 455 wins because it is the strongest single alternative to the incumbent (443). But when a third option enters, the constituency that was holding 455 together fragments — each sub-group defects to the specialist that better represents them. Model 455 loses not because it is bad, but because the electorate it was unifying was always internally divided, and K = 3 gives that division room to express itself.

Model 455 re-emerges at K = 9 because, by that point, all of the niches that were competing for its constituency are already filled by other specialists. The participants who were 455's natural constituency — those genuinely best described by its reference-dependent structure — are finally returned to it, freed from being absorbed by 452 or 469.

The implication for reporting: recommending model 455 to applied researchers at K = 2 should come with a caveat. It is acting as a coalition structure at that level of granularity. The K = 2 library captures the most parsimonious acknowledgement that the population is heterogeneous, but the specific pair {443, 455} does not reflect the stable, fine-grained structure that emerges at K ≥ 3. Researchers who need a practically useful split library are better guided toward K = 3 or K = 5, where the nested structure has stabilised and each model in the library is serving a more coherent constituency.

---

### Finding 7: Model 469 as a poor-fit placeholder

Model 469 has a mean individual BIC of 180–209 throughout its presence in the library (K = 3 onwards), far above the population mean and far above its co-members. The participants assigned to it are not well described by it — they are assigned to 469 because it is the least-bad option available in the library, not because it is a good fit. These participants are genuine outliers in utility function space; no model in the candidate pool captures them well.

The natural question is why the optimal K = 3 library is {443, 452, 469} rather than {443, 452, 473}, which would seem to replace the poorly-fitting 469 with a more competitive alternative. The answer is that the exhaustive search over all C(N_candidates, 3) combinations found that 469 specifically minimises total population BIC at K = 3 — meaning there is a sub-group of participants for whom 469 is decisively better than any other available model except 443 or 452. The 23 participants assigned to 469 at K = 3 are not served badly by 469 in an absolute sense unique to 469; they are poorly served by the entire top of the candidate pool, and 469 happens to be least-bad for them. These are participants whose utility structures may be genuinely idiosyncratic and worth examining individually.

---

### Note for the paper: reconciling the AMPD MDS blob with the split-library finding

When the AMPD MDS is presented in the results, readers will likely notice that utility functions form one continuous blob with no visible clustering, and wonder how to square that with the compression curve's strong case for a split library. This should be addressed briefly but directly.

The short answer is that the MDS and the compression curve answer different questions. The MDS describes the distribution of *models* in behavioural space — and that distribution is indeed continuous, with no natural discrete categories baked into the utility function structure. The compression curve describes the distribution of *participant preferences* across that same space — and participants turn out to cluster bimodally, even though the space itself has no seams. A blob in model space is fully compatible with strong clustering in participant preference space; the two are logically independent. An analogy: colours on a colour wheel form a continuous gradient with no obvious clusters, yet if you ask 73 people to pick their favourite colour, they might divide sharply into two groups. The continuity of the option space says nothing about the bimodality of the preference distribution.

A second point worth noting is that the MDS is a 2D projection of a high-dimensional distance structure. Structure that separates participants along dimensions 3, 4, 5, … of the AMPD space can be entirely invisible in the 2D plot. The blob is not evidence of the absence of structure; it is evidence of the absence of structure in two specific directions.

The one genuine caveat to acknowledge: because the model space is a continuous gradient rather than a set of discrete natural kinds, the boundary between the two participant groups is not a sharp edge — it is a gradient. Participants near the "equator" between the 443-preferring and 455-preferring halves might plausibly be assigned to either model with similar fit. The 50/50 split is a real finding, but borderline participants should be checked for assignment stability across scoring schemes (raw NLL, individual BIC, IC-equivalent). If they are stable, the split is robust; if they flip, the split is softer than the headline ΔA number suggests.

---

### Finding 8: The stopping criteria disagree — there is no optimal fit-parsimony tradeoff

The four stopping criteria return four different K values:

- **Max curvature and meta-BIC**: K = 3 (steepest deceleration of improvement; smallest penalised library size)
- **Kneedle elbow**: K = 5 (greatest departure from linearity in the compression curve)
- **Marginal gain (ΔA < 1%)**: K = 8 (last K delivering at least 1% additional gain)
- **Cumulative gain (A ≥ 80%)**: K = 4

This disagreement is not a bug or a limitation of the analysis — it is a substantive finding. **There is no uniquely correct answer to how many utility function types the population needs**, because that answer depends on how strongly one values parsimony relative to fit. A theorist who believes that one or two functional forms should suffice on parsimony grounds will stop at K = 3. An empiricist who wants the library to capture minority preference structures will continue to K = 5 or K = 8. Neither position is wrong; they reflect different values about what a model is for.

This is why the paper should present the full compression curve and report all stopping criteria rather than nominating a single winner. The appropriate framing is to give researchers a menu: here is what you gain at K = 1, 2, 3, 4, 5 — choose the library that reflects your own fit-parsimony priorities. The lack of criterion consensus is itself evidence that the curve declines gradually enough that no single K stands out as obviously optimal, which is a scientifically honest and useful thing to communicate.
