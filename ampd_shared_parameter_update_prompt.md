# AMPD Shared-Parameter Update Prompt

> **Prompt from Greg to the coding agent**  
> Please update the AMPD implementation so that it measures distance between utility-function **architectures**, not distance between two independently sampled artificial agents.

---

## Why this change matters

The current AMPD implementation samples one parameter vector for Model A and a separate parameter vector for Model B within the same Monte Carlo iteration. That makes the distance between a model and itself nonzero:

```text
AMPD(model_x, model_x) > 0
```

That is not the metric we need for model-space geometry. The main AMPD matrix is supposed to be a behavioral distance between utility-function policies. A model compared with itself should have distance zero. Nonzero self-distance means the function is mixing two things:

1. distance between utility architectures;
2. variability between two independently sampled agents from the same architecture.

The second quantity can be interesting as a separate diagnostic, but it should not be used for MDS, clustering, top-model coherence, max-min subset selection, or participant model-cloud geometry.

The primary AMPD metric should compare two models under **matched parameter and payoff conditions**.

---

## Required conceptual update

For the main AMPD calculation, add or enforce:

```python
parameter_pairing_mode: Literal["shared", "independent"] = "shared"
```

The default must be:

```python
parameter_pairing_mode="shared"
```

### Shared mode

Within each Monte Carlo iteration:

1. Draw one full canonical reference parameter vector.
2. Give that same vector to Model A and Model B.
3. Let each model use the parameters it recognizes and ignore the rest.
4. Evaluate both models on the same payoff structures.
5. Compute the chosen distance metric between their choice probabilities.
6. Average over payoff structures and iterations.

This gives the intended metric:

```text
Average Model Policy Distance = average behavioral distance between utility architectures
```

### Independent mode

It is fine to keep an optional independent mode:

```python
parameter_pairing_mode="independent"
```

But label it clearly as a diagnostic for within-family policy variability or independently sampled agent distance. Do not use it for the main AMPD matrix.

---

## Canonical full reference parameter vector

The shared reference vector should contain the largest common set of mean parameters used across the utility-function universe. At minimum, include:

```text
Vᵢᵢ
Ʌᵢᵢ
Vᵢⱼ
Ʌᵢⱼ
Ƹᵢⱼ
Ʒᵢⱼ
γ1
γ2
γ3
```

Do **not** include `τ` / `temp` in this shared reference vector. AMPD should use the fixed `choice_temperature` / `softmax_temperature` argument for both models.

Each model should receive the same full reference vector and then project it down to its active parameter set. Examples:

- a model without altruism ignores `Vᵢⱼ` and `Ʌᵢⱼ`;
- a model with fixed self-interest ignores `Vᵢᵢ`;
- a model without social comparison ignores `Ƹᵢⱼ` and `Ʒᵢⱼ`;
- a single-exponent model uses `γ1` and lets existing utility logic collapse or ignore `γ2` and `γ3`.

Use existing project functions such as `make_param_info(...)` and `parameter_keys_for_utility_settings(...)` to determine which parameters a model actually uses. Do not duplicate parameter-selection logic.

---

## Uniform sampling mode

For:

```python
parameter_sampling_mode="uniform"
```

The AMPD implementation should:

1. sample one full canonical reference vector per iteration;
2. sample every canonical parameter from its bounds;
3. project that same vector into both models;
4. evaluate both models on identical payoff structures.

Do not call `_sample_params_uniform(...)` separately for `params_a` and `params_b` in shared mode.

---

## Realistic sampling mode

For:

```python
parameter_sampling_mode="realistic"
```

The AMPD implementation should:

1. choose one role from `player_roles` for the iteration, or use a clearly documented role-combination rule;
2. draw one empirical row from that role’s population parameter distribution dataframe;
3. use that row to populate a shared full reference parameter vector;
4. sample missing parameters uniformly from `param_bds` / `make_param_info(...)`;
5. project the same completed reference vector into both models.

Default role input should be:

```python
player_roles: list[Literal["chooser", "predictor"]] | None = ["chooser", "predictor"]
```

Chooser-only and predictor-only variants are useful sensitivity analyses, but the default can include both roles unless Greg changes this later.

---

## Matrix-level requirements

For `compute_ampd_distance_matrix(...)`:

1. Set the diagonal to exactly zero.
2. Do not compute diagonal cells through Monte Carlo sampling.
3. Ensure the matrix is symmetric.
4. Ensure normalized JSD values are finite and bounded in `[0, 1]`.
5. Use the same payoff structures for both models in each pair.
6. Use shared parameter draws by default.
7. If independent-pairing diagnostics are implemented, cache them separately and include `parameter_pairing_mode="independent"` in the filename.

The correct primary matrix satisfies:

```text
AMPD(model_x, model_x) = 0
```

If a cached matrix has nonzero diagonal entries, it is not the primary AMPD matrix and should not be used for model-space or participant-cloud analyses.

---

## Cache and filename requirements

AMPD values depend on the settings that generated them. Cache files must include enough settings to prevent silent incompatible reuse.

Include at least:

```text
metric
parameter_sampling_mode
parameter_pairing_mode
softmax_temperature / choice_temperature
n_games
n_iters
player_roles
random_seed or seed policy
utility-function subset hash
canonical-settings/version hash if available
```

Do **not** cache only by `n_models`. Two different subsets can have the same number of models. Include a hash of the actual `utility_idx` values in the matrix.

---

## Registry and Boolean-column safety

When resolving `utility_idx` values to `utility_settings`:

1. Do not infer setting columns by “everything that is not metadata.”
2. Use the canonical utility-setting column order from `config.py` / `utility_settings`.
3. Be careful with booleans loaded from CSV. `bool("False")` is `True`, so add a robust parser if values may be strings.

---

## Error handling

Do not silently replace exceptions with a distance penalty during normal AMPD computation.

Default should be:

```python
error_policy="raise"
```

An optional exploratory mode may use:

```python
error_policy="penalty"
```

but it must be explicit and should log which model pair failed.

---

## Import-flow and reuse requirements

Follow `AGENTS.md` style and import-flow rules.

- Reuse `utility(...)` and `softmax_(...)` for policy evaluation.
- Reuse `make_param_info(...)` and `parameter_keys_for_utility_settings(...)` for parameter keys and bounds.
- Reuse registry functions to resolve `utility_idx`.
- If the AMPD function lives upstream of `model.py`, pass `utility(...)`, `softmax_(...)`, and `build_utility_equation(...)` as callbacks rather than creating circular imports.
- If moving functions between modules is cleaner, update `README.md` and `AGENTS.md` accordingly.

Do not reinvent utility evaluation or parameter-selection logic.

---

## Suggested helper functions

Implement helper functions along these conceptual lines. Exact names can differ if another naming scheme better matches the repo.

```python
def sample_full_reference_parameter_vector(
    general_settings: GeneralSettings,
    param_bds: dict[str, tuple[float, float]],
    parameter_sampling_mode: Literal["uniform", "realistic"],
    population_parameter_distribution_df_by_role: dict[str, pd.DataFrame] | None,
    player_roles: list[Literal["chooser", "predictor"]] | None,
    rng: random.Random,
) -> dict[str, float]:
    ...
```

```python
def project_reference_params_to_utility_settings(
    full_reference_params: dict[str, float],
    utility_settings: UtilitySettings,
    general_settings: GeneralSettings,
    param_bds: dict[str, tuple[float, float]],
) -> dict[str, float]:
    ...
```

In shared mode, use these once per iteration:

```python
full_reference_params = sample_full_reference_parameter_vector(...)
params_a = project_reference_params_to_utility_settings(full_reference_params, settings_a, ...)
params_b = project_reference_params_to_utility_settings(full_reference_params, settings_b, ...)
```

In independent mode, draw two full reference vectors separately. Keep that path clearly labeled as diagnostic.

---

## Minimal tests to add

Please add tests or assert-style checks for these invariants:

```text
average_model_policy_distance(model_x, model_x, parameter_pairing_mode="shared") == 0 or approximately 0
compute_ampd_distance_matrix(...).diagonal() == 0
AMPD matrix is symmetric
All normalized JSD values are between 0 and 1
Shared-mode and independent-mode caches do not collide
Subset matrices with the same n_models but different utility_idx sets do not collide
```

If parent-child anchor tests are easy to reuse, also test:

```text
AMPD(parent at child anchor, child) ≈ 0
```

but do not reinvent the existing nesting-verification machinery.

---

## Desired output

After this update, the main AMPD matrix should be suitable for:

- utility-function MDS / clustering;
- top-model coherence analysis;
- distance-to-winner vs. ΔBIC analysis;
- max-min AMPD subset selection;
- participant BIC-weighted model-cloud geometry;
- functional-architecture compression analyses.

The central principle is:

> The main AMPD matrix compares utility architectures under matched conditions. It should not be contaminated by random mismatch between two independently sampled agents.
