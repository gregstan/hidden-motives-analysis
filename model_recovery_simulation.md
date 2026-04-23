# Model Recovery, Utility Registry, and Model-Space Geometry

> **Coding-agent brief for the *Inferring Hidden Motives* repository**  
> **Purpose:** build the shared infrastructure needed to treat the utility-function IC analysis as a calibrated scientific instrument, not merely a massive model contest.  
> **Primary audience:** an AI coding agent with access to the repo but no prior conceptual context.  
> **Style constraint:** follow `AGENTS.md` exactly. Reuse existing functions aggressively. Do **not** reinvent utility evaluation, SoftMax choice, IC scoring, nesting, parent-child verification, file-name suffix machinery, or optimizer routines unless the existing implementation truly cannot support the extension.

---

## 0. The big picture

This project studies how people infer hidden social motives from repeated choices in binary dictator games. The central modeling question is: **what utility function should sit inside the likelihood term of the Utility Bayesian Model?**

The paper currently answers that question with a huge Information Criterion analysis over the valid utility-function universe. The exact number of valid forms may change as rules are revised, so **never hard-code the model count**. The current code base may yield 480 valid forms, but the correct number is always:

```python
n_utility_functions = len(generate_utility_settings(...))
```

The IC analysis is one of the paper’s load-bearing claims. It does not merely say, “one model had the lowest BIC.” It supports broader scientific claims about which psychological distinctions matter: self-interest, altruism, social comparison, envy/guilt asymmetry, nonlinear payoff sensitivity, reference dependence, payoff ratios, negativity parameters, and related transformations.

The infrastructure in this file is meant to answer three linked questions:

### **Question 1 — Can every utility function be indexed by a stable identity?**

Every utility function needs one canonical row in one canonical registry. All analyses should use the same identifier, bitstring, equation, parameter count, nesting family, IC scores, and distance summaries.

### **Question 2 — How far apart are utility functions behaviorally?**

Two equations can look different yet make almost identical choices over the payoff space. Conversely, two equations can look superficially similar yet behave very differently. We need a **model policy distance**: a distance between the choice policies induced by utility forms across payoff structures and parameter draws.

### **Question 3 — How well calibrated is the utility-function search?**

When the IC analysis selects a winner, we want to know whether that winner belongs to a coherent behavioral family, whether nearby high-ranking models are nearly equivalent, whether important features were recovered, and whether a known generating model can be recovered in optional synthetic tests.

> **Scientific motivation:** this turns the IC analysis from a rank table into a map. A rank table says which model won. A map shows whether the winner is alone, surrounded by near-equivalent neighbors, or tied with distant alternative theories.

---

## 1. Implementation principles

### 1.1 Reuse existing functions before writing new ones

The codebase already has substantial tested machinery. Use it.

Relevant existing functions and systems include, but are not limited to:

```text
make_param_info(...)
parameter_keys_for_utility_settings(...)
generate_utility_settings(...)
is_valid_utility_settings(...)
convert_utility_settings(...)
count_free_parameters(...)
utility(...)
choice(...)
softmax_(...)
build_utility_equation(...)
create_file_name_suffix(...)
ensure_directory_and_join(...)
information_criterion_analysis(...)
model_nesting_adjacency_matrices(...)
classify_pair_relation(...)
child-parent equivalence verification functions
```

Do not duplicate the logic of these functions. Wrap, extend, or call them.

### 1.2 Respect the import chain

The repo’s import chain is intentionally structured to avoid circular imports. The current high-level flow is approximately:

```text
config → preprocessing / utilities / typological → model → optimization → bayesian → simulation → visualization → analysis → main
```

If a function in `utilities.py` needs `build_utility_equation(...)`, `utility(...)`, or `choice(...)`, choose one of these designs:

1. pass the needed function as a callback;
2. place the new function downstream in a module that already imports `model.py`;
3. move a function only if doing so improves the architecture and does not create circular dependencies.

If any import-flow change is made, update:

```text
README.md
AGENTS.md
```

### 1.3 Preserve backward compatibility temporarily

The new system should ultimately replace several overlapping legacy files:

```text
processed/equations_to_setting.json
processed/model_nesting_data.json
processed/redundant_utility_functions.csv
```

with one registry:

```text
processed/all_utility_functions.csv
```

However, the first implementation should be backward-compatible. Whenever compatibility code is written, mark it clearly:

```python
"TODO: Legacy compatibility path. Delete this block after all_utility_functions.csv is proven to replace equations_to_setting.json, model_nesting_data.json, and redundant_utility_functions.csv."
```

The goal is not to keep two systems forever. The goal is to migrate safely.

### 1.4 Follow project style

Follow `AGENTS.md`:

- long descriptive names;
- keyword arguments for project-function calls;
- docstrings with `Arguments:` and `Returns:` blocks;
- standalone block comments as string literals, not `#` comment blocks;
- no needless abstractions;
- clarity over brevity.

---

# Part I — Central utility-function registry

## 2. Why a central registry matters

Right now, utility-function information is scattered across files and runtime objects. That creates three problems:

1. **Identity drift:** the same utility form can be referred to by a tuple, dict, equation string, suffix, or row number.
2. **Analysis friction:** distance matrices, IC results, nesting relations, and recovery metrics all need to align on model identity.
3. **Scientific opacity:** readers and future agents need one place to inspect what each model is.

The registry solves this. Every valid utility form gets one row.

Path:

```text
processed/all_utility_functions.csv
```

## 3. Required registry columns

The registry should include at least:

```text
utility_idx
utility_bitstring
k_params
conditional_welfare_mode
reference_dependent_altruism
min_max_rawlsian_leontief
use_exponential_parameters
single_exponential_parameter
apply_exponents_to_payoffs
single_payoffs_not_differences
payoff_ratios_not_differences
reference_dependent_utility
use_negativity_parameters
negativity_social_comparison
fix_self_interest_parameter
include_social_comparison
include_altruism_term
redundant_with
differing_settings
n_data
pvar
param_norm_sd
loss_nll
AIC
BIC
ΔAIC
ΔBIC
AIC_rank
BIC_rank
parents
siblings
children
ampd_to_best_rand
ampd_to_best_real
policy_regret_norm
equation
```

Additional columns are welcome if they help debugging or reproducibility.

## 4. Stable utility identity

`utility_idx` must be stable across runs. Cached distance matrices, IC outputs, nesting relations, and participant-level fits all depend on this.

Recommended procedure:

1. Generate all valid settings using the current validity rules.
2. Compute `k_params` from the authoritative parameter-order logic, preferably `make_param_info(...)` or `parameter_keys_for_utility_settings(...)`.
3. Create `utility_bitstring` from the Boolean settings in the exact canonical order used in `config.py`.
4. Sort by:

```text
k_params ascending
utility_bitstring ascending
```

The `k_params` sort matters because children should generally appear before parents, supporting the existing warm-start logic in the IC robustness analysis. The `utility_bitstring` stabilizes ordering within the same complexity level.

If existing code has a more precise child-before-parent sort, preserve it and use the bitstring as a deterministic tie-breaker.

## 5. Refactor `generate_utility_settings(...)`

Modify `generate_utility_settings(...)` so it becomes both a generator and a retrieval interface.

Suggested signature:

```python
def generate_utility_settings(
    utility_settings: UtilitySettings,
    general_settings: GeneralSettings | None = None,
    file_paths: FilePaths | None = None,
    sort_by_k: bool = True,
    create_new_file: bool = False,
    return_df: bool = False,
    build_equation_function: Callable | None = None,
    **kwargs: Any,
) -> list[UtilitySettings] | pd.DataFrame:
```

### If `create_new_file=True`

The function should:

1. generate all valid utility settings;
2. compute `k_params`;
3. compute `utility_bitstring`;
4. assign stable `utility_idx`;
5. check for redundancies;
6. compute equations if `build_equation_function` is available;
7. compute or retrieve parent/sibling/child relations if nesting utilities are available;
8. attempt to merge existing IC results from `bic_aic/`;
9. leave IC-specific columns blank if no IC results exist;
10. write `processed/all_utility_functions.csv`.

### If `create_new_file=False`

The function should:

1. try to load `processed/all_utility_functions.csv`;
2. validate that required setting columns exist;
3. return either:
   - a dataframe, if `return_df=True`;
   - `list[UtilitySettings]`, if `return_df=False`.

### Backward-compatible fallback

If the CSV does not exist and `create_new_file=False`, either:

- call the old generation path and warn loudly, or
- call the new generation path automatically.

Use the choice that is least disruptive to the existing pipeline, but mark the fallback with TODO legacy comments.

## 6. IC results should update the registry

When `information_criterion_analysis(...)` finishes, it should update the central registry with IC columns:

```text
n_data
pvar
param_norm_sd
loss_nll
AIC
BIC
ΔAIC
ΔBIC
AIC_rank
BIC_rank
```

The existing IC function already saves per-model JSON and CSV outputs. Do not break those outputs during the first migration. Instead, add registry updating as an additional step.

The code should merge by stable model identity:

```text
utility_idx
utility_bitstring
canonical Boolean setting tuple
```

Prefer `utility_idx` once the registry is trusted. During transition, cross-check all three.

---

# Part II — Selecting utility-function subsets

## 7. Why subset selection matters

Many future analyses need subsets of the model universe:

- fast debugging;
- recovery benchmarks;
- max-min diverse model sets;
- parents / children / siblings of important models;
- random stratified samples;
- top models from the empirical IC ranking;
- canonical baseline families.

A single helper prevents every analysis from inventing its own fragile selection logic.

## 8. Implement `select_utility_settings_subset(...)`

Place this near `generate_utility_settings(...)`, likely in `utilities.py`, unless import constraints make a downstream location cleaner.

Suggested signature:

```python
def select_utility_settings_subset(
    n_models: int | None = None,
    hand_picked_subset: list[int | UtilitySettings] | None = None,
    selection_mode: Literal["random", "random_by_k", "random_by_setting", "hamming", "ampd"] = "random",
    file_paths: FilePaths | None = None,
    general_settings: GeneralSettings | None = None,
    utility_settings: UtilitySettings | None = None,
    include_model_idxs: list[int | UtilitySettings] | None = None,
    exclude_model_idxs: list[int | UtilitySettings] | None = None,
    required_settings: UtilitySettings | None = None,
    required_k_params: list[int] | None = None,
    parents_of: list[int | UtilitySettings] | None = None,
    siblings_of: list[int | UtilitySettings] | None = None,
    children_of: list[int | UtilitySettings] | None = None,
    distance_matrix_path: str | None = None,
    random_seed: int | None = None,
) -> list[UtilitySettings]:
```

### Required behavior

- If `hand_picked_subset` is provided and `n_models` is `None` or greater than the subset length, return that subset after applying inclusion/exclusion filters.
- If `n_models` is `None` or greater than the available model count, do not downsample except for explicit filters.
- `required_settings` should filter Boolean settings. Since these are Booleans, requiring `False` already forbids `True`; no separate forbidden-settings argument is needed.
- `required_k_params` should default to all valid `k` values unless specified.
- `parents_of`, `siblings_of`, and `children_of` should use the central registry’s family columns.
- `hamming` mode should maximize diversity over Boolean bitstrings.
- `ampd` mode should maximize behavioral diversity using the AMPD matrix.

### Max-min selection algorithm

For Hamming or AMPD diversity:

1. start with required / included models if present;
2. if no required seed exists, choose a seed model randomly or choose the empirical IC winner;
3. repeatedly add the model whose **minimum distance** to the selected set is largest;
4. stop when `n_models` is reached.

This creates a behaviorally or symbolically heterogeneous benchmark set.

---

# Part III — Average Model Policy Distance

## 9. Why AMPD matters

A utility function is not merely an equation. A utility function plus parameters plus temperature induces a **choice policy** over payoff structures:

```text
utility form + parameters + payoff structure → P(choose A)
```

The central object is:

```text
p_model_choose_A
```

Average Model Policy Distance, or **AMPD**, measures how differently two utility forms behave as policies over:

- the binary dictator payoff grid;
- sampled parameter vectors;
- a fixed SoftMax temperature.

This is useful because symbolic model identity is not enough. Two models can differ in syntax while behaving almost identically across the task. In that case, a “wrong” IC winner may still be behaviorally right. Conversely, a symbolically close model may be behaviorally far away.

## 10. Directional vs symmetric comparisons

Use two different metric families for two different purposes.

### Directional metrics

Use directional metrics when one model or dataset is treated as truth.

Examples:

- human responses are the truth and a model predicts them;
- a synthetic generating model is the truth and a recovered model predicts it;
- the empirical winning model is treated as a reference.

Directional loss should use cross-entropy / NLL logic.

### Symmetric metrics

Use symmetric metrics when neither model is truth.

Examples:

- mapping utility-function space;
- clustering models;
- computing a distance matrix;
- visualizing top-model coherence.

Symmetric model-space geometry should use normalized Jensen-Shannon divergence by default.

## 11. Cross-entropy, entropy, KL, and JSD

### 11.1 Cross-entropy

If model A gives probability `p` to option A and model B gives probability `q`, then the expected NLL of model B when model A is the truth is:

$$
CE(p,q) = -[p\log(q) + (1-p)\log(1-q)]
$$

This is what binary NLL becomes in expectation when the source model produces infinitely many choices.

Use cross-entropy when there is a truth/source model.

### 11.2 Entropy

The Shannon entropy of a Bernoulli probability `p` is:

$$
H(p) = -[p\log(p) + (1-p)\log(1-p)]
$$

Interpretation:

- `H(p) = 0` when the source model is deterministic;
- entropy is largest when `p = .5`;
- entropy is the irreducible uncertainty of the source distribution.

### 11.3 KL divergence

The extra loss from using `q` instead of true `p` is:

$$
KL(p \Vert q) = CE(p,q) - H(p)
$$

This is directional.

### 11.4 Jensen-Shannon divergence

For symmetric model distance, use normalized JSD.

Let:

$$
r = \frac{p+q}{2}
$$

Then:

$$
JSD(p,q)=H(r)-\frac{1}{2}H(p)-\frac{1}{2}H(q)
$$

Normalize:

$$
JSD_{norm}(p,q)=\frac{JSD(p,q)}{\log 2}
$$

Interpretation:

- `0` = identical policies;
- `1` = maximally different deterministic policies;
- finite, bounded, symmetric;
- appropriate for heatmaps, MDS, clustering, and distance matrices.

## 12. Normalized policy regret

For known-truth recovery cases, compute normalized policy regret:

$$
R_{policy}
=
\frac{CE(p^*, \hat{p}) - H(p^*)}
{CE(p^*, 0.5) - H(p^*)}
$$

where:

- `p*` is the generating truth probability;
- `p_hat` is the recovered model’s probability;
- `0.5` is the random-choice baseline.

Interpretation:

| Value | Meaning |
|---:|---|
| `0` | recovered policy is as good as the true generating policy |
| `.05` | recovered model loses only 5% of recoverable predictive information |
| `1` | no better than random guessing |
| `>1` | worse than random guessing |

This is the bounded “how bad was the miss?” metric.

## 13. AMPD default behavior

Primary AMPD should **not** run the full dynamic UBM. It should compare static utility-induced choice policies:

```text
utility(...) → choice(...) → P(choose A)
```

Do this because AMPD is supposed to measure utility-function distance, not distance between Bayesian updating assumptions, priors, particle filters, or observation histories.

### Default settings

```text
metric = "normalized_jsd"
choice_temperature = general_settings["softmax_temperature"]
n_games = 625
n_iters = 250
parameter_sampling_mode = "uniform"
player_roles = ["chooser", "predictor"] when realistic sampling is used
```

Use all 625 payoff structures by default. If `n_games >= 625`, use the exhaustive payoff grid and do not randomize payoffs.

Temperature sensitivity is optional. The current preference is **not** to vary temperature for the primary AMPD matrix unless there is a specific reason.

## 14. Parameter sampling for AMPD

AMPD needs parameter draws. Implement a reusable sampler that wraps `make_param_info(...)`.

### Sampling modes

#### `uniform`

Draw all parameters uniformly from bounds supplied by `make_param_info(...)`.

This is non-circular and useful for general model-space geometry.

#### `realistic`

Draw available parameters from empirical population distributions.

Inputs should include:

```python
player_roles: list[Literal["chooser", "predictor"]] | None = ["chooser", "predictor"]
```

There are population parameter distribution dataframes for both roles. The sampler should know which role distribution it is sampling from.

If a parameter is missing from the empirical distribution, sample it uniformly using `make_param_info(...)` bounds. Do **not** invent empirical analogues.

### Why missing parameters should be random

For parameters not present in the empirical distribution, random sampling is more defensible than imposing researcher intuition. The code should not hard-code guesses about what “realistic” negative self-interest or negative altruism should be unless such values come from the data.

## 15. AMPD function interface

Suggested interface:

```python
def average_model_policy_distance(
    utility_settings_a: UtilitySettings | int,
    utility_settings_b: UtilitySettings | int,
    general_settings: GeneralSettings,
    file_paths: FilePaths,
    param_bds: ParameterBounds,
    metric: Literal["normalized_jsd", "cross_entropy", "normalized_policy_regret", "mean_abs_diff"] = "normalized_jsd",
    choice_temperature: float | None = None,
    n_games: int = 625,
    n_iters: int = 250,
    parameter_sampling_mode: Literal["uniform", "realistic"] = "uniform",
    player_roles: list[PlayerRole] | None = None,
    use_cache: bool = True,
    create_new_file: bool = False,
    random_seed: int | None = None,
) -> float:
```

Exact signature may vary, but the function must support:

- model inputs as `utility_idx` or `UtilitySettings`;
- metric switching;
- exhaustive 625-game payoff grid;
- uniform and realistic parameter sampling;
- cache retrieval keyed by settings;
- deterministic seeding.

## 16. Cache AMPD matrices by settings

AMPD should be computed once, saved, and reused.

Suggested cache files:

```text
processed/model_distance_hamming__n_models={M}.csv
processed/model_distance_ampd__metric=normalized_jsd__sampler=uniform__tau={tau}__n_games=625__n_iters=250__seed={seed}.csv
processed/model_distance_ampd__metric=normalized_jsd__sampler=realistic__roles=chooser-predictor__tau={tau}__n_games=625__n_iters=250__seed={seed}.csv
```

File names should encode settings. This follows the repo’s convention of naming files by settings to prevent accidental overwrite or retrieval.

### Cache validation

A cached distance is valid only if all settings match:

```text
model universe / registry version
metric
choice temperature
parameter sampler
player roles
n_games
n_iters
payoff grid mode
random seed or seed policy
code version hash optional
```

Do not silently reuse a matrix with incompatible settings.

## 17. AMPD convergence / robustness check

AMPD is Monte Carlo averaged over parameter draws. Add an optional robustness diagnostic.

### Purpose

We need to know whether `n_iters=250` is enough for the distance matrix to stabilize.

### Suggested implementation

Within the AMPD matrix function, optionally compute running estimates in batches:

```text
batch_size = 25
n_iters = 250
```

After each batch, store the current mean distance for each pair or for a sampled subset of pairs.

Possible diagnostics:

1. **Running mean stability**  
   Track the absolute change in pairwise distance after each batch.

2. **Matrix correlation stability**  
   Compute a full or sampled distance matrix at `n_iters=100`, `250`, and `500`, then correlate vectorized upper triangles.

3. **Maximum / median change**  
   Report median and 95th percentile absolute changes between checkpoints.

4. **Visualization**  
   Save a small line plot:

```text
x-axis: number of iterations
y-axis: mean absolute change in sampled distances
```

Stop early only if this is safe and explicitly requested. The default should compute the requested `n_iters`.

---

# Part IV — Model-space geometry analyses

## 18. What AMPD lets us ask

Once the model-model distance matrix exists, the IC landscape becomes interpretable.

A rank table alone cannot distinguish these worlds:

### World 1 — one coherent winning family

The best model wins, and the next 20 models are behaviorally close to it.

Interpretation:

> The exact winning syntax matters less than the fact that the data identify one stable behavioral basin.

### World 2 — multiple distant top families

The best model wins, but several behaviorally distant models are close behind.

Interpretation:

> The data support several distinct explanations. Future experiments should target payoff structures that discriminate among them.

AMPD tells us which world we live in.

## 19. High-value analyses to implement

### 19.1 Top-model coherence

For the top `N` models by BIC, compute mean pairwise AMPD.

Outputs:

```text
N
mean_pairwise_ampd
median_pairwise_ampd
max_pairwise_ampd
```

Scientific question:

> Do the top models form one coherent behavioral family?

### 19.2 Top-model distance heatmap

Create a heatmap for the top 25 or top 50 models by BIC.

Rows/columns:

```text
utility_idx ordered by BIC rank
```

Cell value:

```text
AMPD(model_i, model_j)
```

Annotate with:

```text
k_params
ΔBIC
major Boolean features
```

### 19.3 Distance-to-winner vs ΔBIC

For every model:

```text
x = AMPD(model, empirical_BIC_winner)
y = ΔBIC(model)
```

Interpretation:

- positive trend: worse models are behaviorally farther from winner;
- far-but-good models: multiple distinct explanations remain plausible;
- close-but-bad models: small behavioral differences matter a lot.

### 19.4 Model-space MDS / PCA-like map

Use the AMPD matrix to embed utility functions into 2D or 3D.

Save:

```text
utility_idx
mds1
mds2
mds3_optional
k_params
ΔBIC
BIC_rank
utility_bitstring
major setting columns
```

Plot all models colored by `ΔBIC` or major settings.

### 19.5 Dendrogram / clustering

Cluster utility functions using AMPD.

Use this to see whether models organize into interpretable families, such as:

- self-interest + altruism;
- social-comparison-heavy;
- exponent-heavy;
- ratio/reference-dependent;
- min-max / Rawlsian / Leontief.

### 19.6 Sliding-window diversity over rank

For a window of size `N`, compute the mean pairwise AMPD among models ranked `r` through `r+N-1`.

$$
W_r = \text{mean pairwise AMPD among models } r \ldots r+N-1
$$

Plot:

```text
x-axis = starting rank r
y-axis = W_r
```

Interpretation:

- low `W_1`: top models are coherent;
- high `W_1`: top models are behaviorally plural;
- jumps: transitions between model families;
- valleys lower in ranking: coherent but poorly fitting alternative families.

### 19.7 Feature-level behavioral importance

For each Boolean setting, compute how much flipping that setting changes policy on average.

Then compare behavioral importance to pairwise ΔBIC contribution.

This creates a powerful interpretation grid:

| Behavioral distance | BIC improvement | Interpretation |
|---|---|---|
| high | high | feature changes behavior and humans use it |
| high | low/worse | feature changes behavior but not in the empirically supported direction |
| low | high | subtle but systematic fit gain |
| low | low | likely negligible in this payoff space |

### 19.8 Payoff structures that distinguish model clusters

If top models form multiple clusters, find payoff structures where cluster policies differ most.

For each payoff structure `x`, compute cluster-level average choice probabilities and identify:

```text
x_star = payoff structure maximizing between-cluster JSD
```

Scientific payoff:

> This turns the model-space map into future experimental design. It tells us which games would best discriminate between model families.

---

# Part V — Optional known-truth utility recovery simulation

## 20. Clarifying what “synthetic utility recovery” means

This section is optional and distinct from the real-data individual architecture analysis.

### Real-data individual architecture analysis

Uses real human data. No known ground truth exists. The goal is to ask whether participants differ in functional architecture.

### Synthetic utility recovery

Uses artificial data generated from a known utility function. Ground truth is known. The goal is to ask whether the IC pipeline can recover the generating utility form or a behaviorally equivalent form.

These analyses share infrastructure:

```text
registry
subset selection
AMPD matrices
parameter sampling
IC-result extraction
feature scoring
```

But they answer different scientific questions.

## 21. Synthetic recovery design

For a selected generating model `m_star`:

1. sample heterogeneous artificial agent parameters;
2. generate binary dictator choices from `m_star`;
3. fit candidate utility functions to the synthetic data;
4. rank candidates by IC;
5. evaluate whether the true model or a close/equivalent model was recovered.

This should be chooser-choice based by default. Do not include dynamic predictor updating unless explicitly requested. Including UBM dynamics would make the analysis partly about belief updating rather than utility-form recovery.

## 22. Recovery metrics

For each synthetic recovery run, compute:

### 22.1 Exact top-1 recovery

Did the generating model rank first?

Useful but too strict.

### 22.2 Top-k recovery

Was the true model in the top 3, 5, or 10?

### 22.3 ΔBIC of the true model

$$
\Delta BIC_{true}=BIC(m^*)-BIC(m_{winner})
$$

If `ΔBIC_true <= 10`, the true model remains in the plausible winning set.

### 22.4 Rank percentile

$$
\text{rank percentile}=1-\frac{rank(m^*)-1}{M-1}
$$

This gives a bounded rank-quality metric.

### 22.5 Normalized policy regret

Use the directional metric from Section 12.

This is the most important measure of whether a non-exact recovery matters behaviorally.

### 22.6 AMPD from winner to truth

Use symmetric AMPD to ask whether the winning model is behaviorally close to the true model.

### 22.7 Weighted feature accuracy

Utility models are defined by Boolean settings. Exact model recovery may fail, but important features may still be recovered.

Unweighted:

$$
A_{feature}=1-\frac{1}{S}\sum_s I(\hat{s}\neq s^*)
$$

Weighted:

$$
A_{weighted}=1-\frac{\sum_s w_s I(\hat{s}\neq s^*)}{\sum_s w_s}
$$

Candidate weights:

1. human-data pairwise `|ΔBIC|` weights;
2. simulation-derived policy-distance weights;
3. model-space behavioral-distance weights.

Implement all three if cheap; decide later which to report.

### 22.8 Rank-distance correlation

For all candidate models in a recovery run, correlate:

```text
AMPD(candidate, true_model)
```

with:

```text
IC rank or ΔBIC(candidate)
```

This asks:

> Does the IC ranking track behavioral closeness to the truth?

## 23. Nested-equivalent recovery

Exact syntax recovery is not the right gold standard.

A recovery can succeed at several levels:

| Tier | Success type | Meaning |
|---:|---|---|
| 1 | exact | winner is the generating model |
| 2 | BIC-equivalent | generating model is within `ΔBIC <= 10` |
| 3 | nested-equivalent | winner is a child/parent equivalent under anchor parameters |
| 4 | behaviorally equivalent | policy regret or AMPD is tiny |

Use existing nesting utilities. The project already has thoroughly tested parent-child/sibling machinery and payoff-grid equivalence checks. Do not recreate that logic.

## 24. Data-adequacy curves

For synthetic recovery and parameter recovery, vary:

```text
n_agents
n_games_per_agent
```

Track:

```text
exact recovery
top-k recovery
true model within ΔBIC <= 10
normalized policy regret
weighted feature accuracy
```

Plot performance as a function of data size.

The desired result is a curve that stabilizes near or below the actual study size.

Interpretation:

> Under the simulated assumptions, the actual dataset lies in a region where recovery metrics have mostly stabilized.

This does not prove the dataset is universally “enough,” but it calibrates what the dataset can identify under known conditions.

## 25. Self-consistency / finite-data ceiling

For a fixed generating model and parameter regime:

1. generate dataset A;
2. generate dataset B from the same truth;
3. run recovery on both;
4. compare their winners, feature vectors, ranks, and policy distances.

This estimates the finite-data ceiling. If two independent datasets from the same truth often choose different exact winners, then exact top-1 recovery should not be expected to be perfect.

---

# Part VI — Synthetic data generation infrastructure

## 26. Why modularize synthetic data generation

Existing simulation code works, but parts of it are brittle:

- parameters are embedded in bot UUID strings;
- some code depends on `global_chooser_id`;
- some functions assume a specific two-parameter utility form;
- functions such as `inequality_aversion_sanity_check`, `create_simulated_dyad`, `create_simulated_data`, and `get_simulated_dyad` overlap conceptually.

The new synthetic-data generator should be modular and reusable.

## 27. Desired synthetic generator

Create a general-purpose function conceptually like:

```python
def generate_synthetic_binary_dictator_data(
    generating_utility_settings: UtilitySettings,
    generated_agent_parameters: dict[str, dict[str, float]],
    general_settings: GeneralSettings,
    file_paths: FilePaths | None = None,
    param_bds: ParameterBounds | None = None,
    n_agents: int = 50,
    n_games_per_agent: int = 120,
    payoff_structures: list[dict[str, int]] | None = None,
    use_exhaustive_payoff_grid: bool = False,
    choice_temperature: float | None = None,
    dynamic_predictor: bool = False,
    save_to_disk: bool = False,
    random_seed: int | None = None,
) -> dict[str, Any]:
```

The exact function name can differ. The important features are:

- no dependence on encoded UUIDs for ground truth;
- ground-truth parameters stored explicitly in metadata;
- utility form configurable;
- payoff structures configurable;
- choice generation calls existing `choice(...)` / `utility(...)` logic;
- output compatible with existing fitting pipelines where possible.

## 28. Payoff grid helper

Implement or reuse a helper that returns all 625 payoff structures:

```text
As, Ao, Bs, Bo ∈ {1,2,3,4,5}
```

The helper should output dictionaries compatible with `choice(...)` and existing game structures.

---

# Part VII — Deliverables and tests

## 29. Core deliverables

### Registry

```text
processed/all_utility_functions.csv
```

### Distance matrices

```text
processed/model_distance_hamming__n_models={M}.csv
processed/model_distance_ampd__metric=normalized_jsd__sampler=uniform__tau={tau}__n_games=625__n_iters=250__seed={seed}.csv
processed/model_distance_ampd__metric=normalized_jsd__sampler=realistic__roles={roles}__tau={tau}__n_games=625__n_iters=250__seed={seed}.csv
```

### Model-space outputs

```text
processed/model_space_mds_coordinates.csv
processed/top_model_coherence.csv
processed/distance_to_winner_vs_delta_bic.csv
processed/feature_behavioral_importance.csv
visuals/model_space_mds.html
visuals/top_model_ampd_heatmap.html
visuals/distance_to_winner_vs_delta_bic.html
```

### Optional synthetic recovery outputs

```text
processed/utility_recovery_runs.csv
processed/utility_recovery_summary.csv
processed/utility_recovery_knee_curves.csv
processed/utility_recovery_self_consistency.csv
```

## 30. Unit tests / sanity checks

Implement tests or diagnostic functions for:

1. **Registry stability**  
   Re-running registry creation without changing validity rules should reproduce the same `utility_idx` and `utility_bitstring`.

2. **Model identity conversion**  
   `utility_idx → settings → bitstring → utility_idx` should be stable.

3. **AMPD self-distance**  
   A model’s distance to itself should be approximately zero.

4. **Symmetry**  
   AMPD matrix should be symmetric within numerical tolerance.

5. **Hamming distance**  
   Hamming matrix should be symmetric, diagonal zero, integer-valued.

6. **Parent-child anchor equivalence**  
   Parent at child-anchor parameters should have near-zero AMPD to child.

7. **Temperature sanity**  
   AMPD should generally shrink as temperature increases, because all policies move toward random choice.

8. **Cache safety**  
   A request with different settings should not retrieve an incompatible cached matrix.

9. **IC merge integrity**  
   IC results merged into the registry should match the old IC CSV/JSON outputs.

10. **No circular imports**  
   Running `python main.py` or `python quick_demo.py` should not introduce import errors.

---

# 31. Suggested implementation order

Do this in stages:

## Stage 1 — Registry

Build and test `processed/all_utility_functions.csv`.

## Stage 2 — Subset selector

Implement `select_utility_settings_subset(...)` and test random, hand-picked, by-`k`, Hamming, AMPD placeholder modes.

## Stage 3 — Hamming matrix

Compute and cache symbolic model distances.

## Stage 4 — AMPD engine

Compute pairwise AMPD for small subsets first, then full matrix.

## Stage 5 — AMPD diagnostics

Add convergence/stability checks and cache validation.

## Stage 6 — Model-space geometry

Build top-model coherence, distance-to-winner, heatmaps, MDS, clustering, and feature behavioral importance.

## Stage 7 — Optional synthetic recovery

Only after the shared infrastructure works, implement the known-truth recovery simulations.

---

# 32. Final note to the coding agent

This work is not just cleanup. It changes what the IC analysis can mean.

Without model-space geometry, the IC section is a table of winners and losers. With model-space geometry, the IC section becomes a calibrated map of the psychological theory space. It can tell us whether the winning utility function is an isolated champion, a representative of a coherent family, or one member of several competing behavioral basins.

Please build the infrastructure carefully, reuse existing functions, preserve backward compatibility during migration, and write outputs so the analyses can be inspected by both humans and future coding agents.
