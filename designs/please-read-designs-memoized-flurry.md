# Plan: Bootstrap Population Recovery — Implementation

**Source design doc:** `designs/bootstrap_population_recovery.md` (merged below)
**Supporting context:** `designs/convert_utility_settings_master.md` (side-quest already complete — new `convert_utility_settings` is live in `utilities.py`)

---

## 1. Why this analysis exists

The IC analysis on Experiment 3 selects a 7-parameter utility function as the BIC winner (BIC ≈ 5113). The existing parameter recovery analysis (`run_param_recovery_by_k`, output in `simulations/param_recovery_by_k/`) shows that within this winning model the social preference weights recover well but the curvature exponents γ1, γ2, γ3 recover only weakly (correlations ≈ 0.39, 0.67, 0.37).

One of the paper's headline findings is that γ2 and γ3 are super-linear at the population level (means ≈ 1.34 and 1.38), contradicting the canonical assumption of diminishing marginal utility. A reviewer at *Cognition* seeing recovery correlations of 0.39 and 0.37 on the parameters carrying that claim will ask whether the population-level means can be trusted.

Low individual-level correlation is consistent with two underlying realities:

1. **Noise around the right mean.** Individual γ_hat is noisy, but the population mean of γ_hat tracks the population mean of γ_true. Population-level claims survive. Individual interpretation should be cautious.

2. **Systematic bias.** The recovery procedure pulls γ_hat toward some attractor (the L2 penalty, which scores `(mean(γ) − 1)²`, is the most likely culprit; trade-offs between exponents and weights are another). Population mean of γ_hat is biased relative to population mean of γ_true. The super-linear claim is suspect.

Correlation alone cannot distinguish these. This analysis distinguishes them by computing bias, variance ratio, and regression slope alongside correlation.

**The function is general-purpose.** Although the γ exponents are the immediate motivation, `run_population_recovery_bootstrap` works for any utility settings and any parameter subset. `utility_settings=None` resolves to the BIC winner dynamically. `parameters_of_interest=None` reports every free parameter in the model. Future uses include: validating other utility function families, comparing recovery quality across roles, or running the analysis on a new winning model after re-fitting. Nothing in the implementation should assume k=7 or assume γ parameters specifically.

---

## 2. Method overview

**Parametric bootstrap from the empirical fitted distribution:**

1. For the target utility function and player role, pull the fitted parameter vectors from the human-data IC JSON (`bic_aic/All_Utility_Forms_IC_Analysis_Experiment3.json`). Treat each as a "true" parameter vector for one synthetic agent.
2. Generate synthetic experimental data in which each synthetic agent makes choices governed by those parameters, using the same payoff sampling regime as the real experiment — via the existing `create_simulated_experiment` function.
3. Refit the same utility function to the synthetic data using the standard production fitter (`run_analysis_bayes`).
4. Compare the recovered population to the empirical population, parameter by parameter.

This differs from a classical bootstrap (resampling existing observations with replacement). The randomness comes from synthetic choice noise and payoff sampling, not from resampling participants. Use the term *parametric bootstrap from the empirical distribution* in the writeup to avoid confusion.

---

## 3. Critical constraint: do not break the existing pipeline

This task involves feeding synthetic data to an existing pipeline. The pipeline (`run_analysis_bayes`, `create_simulated_experiment`, and their dependencies) is vital and must not be tampered with unless absolutely necessary. The key strategy is to **pass edited copies of `file_paths` to the existing pipeline** so that synthetic data and fit outputs go into `processed/population_param_bootstrap/` and nowhere near the real participant data or the k-by-k recovery data.

`run_param_recovery_by_k` already demonstrates this pattern exactly — it redirects all its output to `simulations/processed/` by building a scoped copy of `file_paths` before calling `create_simulated_experiment` and `run_analysis_bayes`. The bootstrap should do the same, routing to a different subdirectory.

---

## 4. What already exists — reuse these, do not reimplement

The hard parts are already done. Grep before writing any new logic.

- **`create_simulated_experiment`** in `simulation.py` (~line 1761): generates a full synthetic experiment JSON in the same format as real raw data. Needs one small modification (Step 1 below) to accept empirical parameter vectors, but the rest of the function is unchanged.
- **`run_analysis_bayes`** in `bayesian.py` (~line 2675): the standard fitter. Call it unchanged; just pass the bootstrap-scoped `file_paths`.
- **`convert_utility_settings`** in `utilities.py` (~line 1041): already updated master-translator. Use for all settings-related operations: matching, hashing into filenames, converting between formats.
- **`make_param_info`** in `model.py`: builds the parameter spec for a utility settings dict. Already used by `create_simulated_experiment` and the fitter.
- **`is_valid_utility_settings`** in `utilities.py`: sanity-checks a utility settings dict.
- **`_collect_role_results`** (nested in `run_param_recovery_by_k`, ~line 2348): reference implementation for loading per-player fit JSONs and extracting fitted params. The bootstrap's `_collect_iteration_fits` inner helper follows the same pattern.
- **`_fmt_duration`** in `simulation.py`: formats time durations for progress printing.
- **`pretty_path`** in `config.py`: formats paths for terminal output.
- **`_hsla`** in `visualization.py`: required for all colors in the figure.

---

## 5. Verified facts about the current codebase

- `bic_aic/All_Utility_Forms_IC_Analysis_Experiment3.json`: 476 models, **14-key** `utility_settings` dicts (legacy — before the model space expanded from 476 to 505), 73 players in `minvec` for the k=7 winner
- `minvec[player_uuid]` structure: `{"params": {"chooser": {param_key: float, ...}, "predictor": {...}}, "loss": {"chooser": float, "predictor": float}}`
- Canonical form (from `config.utility_settings`) is **16-key**; the two new keys (`include_welfare_efficiency_term`, `include_relative_income_penalty`) are always `False` for legacy entries — the model space expanded from 476 to 505 when these were added
- `convert_utility_settings` (already updated) accepts `input_settings_format` for 14→16 translation
- `run_analysis_bayes` saves per-player fit JSONs to `file_paths['player_fits']/experiment_{experiment_num}/`
- `create_simulated_experiment` currently requires `altruism_key` and `altruism_targets` (non-optional)
- `run_param_recovery_by_k` routes all synthetic output to `file_paths['simulations']/` — bootstrap must route to a separate subdirectory (`processed/population_param_bootstrap/`)
- `equation_to_settings` in `utilities.py` (line ~1000): regenerates `processed/equation_to_settings.json`; call pattern: `equation_to_settings(equation_function=build_utility_equation, utility_settings=utility_settings, file_paths=file_paths, create_new_file=True)`
- `model_nesting_adjacency_matrices` in `analysis.py` (line ~4400): regenerates `processed/model_nesting_data.json`; call with `create_new_file=True`

---

## 6. Files to modify

| File | Nature of change |
|------|-----------------|
| `utilities.py` | Add `migrate_ic_data_to_16_key()` one-time migration helper |
| `bic_aic/All_Utility_Forms_IC_Analysis_Experiment3.json` | Migrated in-place to 16-key utility settings (backup exists: `- Copy.json`) |
| `bic_aic/All_Utility_Forms_IC_Analysis_Experiment3.csv` | Add two new setting columns with `False` values |
| `bic_aic/IC_Analysis_Comparison_Table_Experiment3.csv` | Add two new setting columns with `False` values |
| `processed/equation_to_settings.json` | Regenerated from the full 16-key utility universe |
| `processed/model_nesting_data.json` | Regenerated to reflect the full 16-key model space |
| `simulation.py` | Add `empirical_chooser_parameters` arg to `create_simulated_experiment`; add `run_population_recovery_bootstrap` |
| `visualization.py` | Add `plot_population_recovery_bootstrap` (two-panel: scatter + bias bar chart) |
| `.gitignore` | Add exclusion rules for large bootstrap synthetic JSON files |

---

## Step 1 — Modify `create_simulated_experiment` in simulation.py (~lines 1761–1971)

This is a minimal surgical change. The function's caching logic, UUID scheme, JSON assembly, `_simulate_pair_games` calls, and return format are all unchanged. The only change is in the per-player parameter assignment block inside the batch loop.

### Why this change is needed

`create_simulated_experiment` currently samples each player's parameters uniformly from bounds, then overrides the altruism dimension to span a target range. For the bootstrap, we want to inject specific empirical parameter vectors instead of sampling randomly — the whole point is to ask "if the true parameters are what we actually fitted in the real experiment, do we recover them?" Without this injection, the function would generate players with random parameters rather than the empirical distribution, defeating the purpose.

### Signature changes

```python
def create_simulated_experiment(
    n_players: int,
    n_games: int,
    k_params: int,
    utility_settings_k: UtilitySettings,
    general_settings: GeneralSettings,
    param_bds: ParamBounds,
    random_gen: np.random.Generator,
    altruism_key: str | None = None,              # was required; now optional
    altruism_targets: list[float] | None = None,  # was required; now optional
    file_paths: FilePaths = ...,                  # still required keyword arg
    create_new_file: bool | None = None,
    enforce_memory_limit: bool = False,
    empirical_chooser_parameters: dict[str, dict[str, float]] | None = None,  # NEW
) -> tuple[dict, dict[str, dict]]:
```

**Validation to add** (after resolving `create_new_file`, before any computation):
```python
if empirical_chooser_parameters is None and (altruism_key is None or altruism_targets is None):
    raise ValueError(
        "Either empirical_chooser_parameters or both altruism_key and altruism_targets must be provided."
    )
```

**Override `n_players` from the empirical dict** (place before the `n_players_padded` calculation):
```python
if empirical_chooser_parameters is not None:
    n_players = len(empirical_chooser_parameters)
```

### Batch loop change (the only logic that changes)

Replace the per-player parameter sampling block inside the batch loop:

```python
"New behavior — empirical injection when provided, else existing uniform+altruism logic:"
empirical_uuid_list  = list(empirical_chooser_parameters.keys()) if empirical_chooser_parameters else []
n_empirical_players  = len(empirical_uuid_list)

for local_idx in range(4):
    global_idx  = batch_start + local_idx
    player_uuid = f"synthetic_{k_params}_{global_idx:04d}"
    if empirical_chooser_parameters is not None and global_idx < n_empirical_players:
        params = dict(empirical_chooser_parameters[empirical_uuid_list[global_idx]])
        if "τ" not in params:
            params["τ"] = float(general_settings.get('softmax_temperature', 1.0))
    else:
        "Uniform sampling for padding players (or standard run when empirical_chooser_parameters is None)."
        params = {}
        for param_idx, param_key in enumerate(list(param_info_for_ubm['keys'])):
            lower_bound, upper_bound = param_info_for_ubm['bounds'][param_idx]
            if param_key.endswith("_std"):
                lower_bound = max(float(lower_bound), 1e-3)
            params[param_key] = float(random_gen.uniform(float(lower_bound), float(upper_bound)))
        if "τ" not in params:
            params["τ"] = float(general_settings.get('softmax_temperature', 1.0))
        if altruism_key is not None and altruism_targets is not None:
            params[altruism_key] = float(altruism_targets[global_idx % len(altruism_targets)])
```

Note on padding players: if `n_empirical_players` is not a multiple of 4, there will be 1–3 padding players at the end of the last batch. Their parameters are sampled uniformly from bounds (the `else` branch above). They are not in `true_params_by_uuid` (their UUIDs exceed `n_empirical_players - 1`) and will not appear in the bootstrap metrics computation.

Add `empirical_chooser_parameters` to the docstring `Arguments:` block.

**Backward compatibility**: every existing call site (in `run_param_recovery_by_k` and elsewhere) passes both `altruism_key` and `altruism_targets` explicitly as keyword args, per the project's keyword-argument convention. Default `None` values for those arguments do not affect existing call sites.

---

## Step 1b — One-time migration: expand all IC data artifacts to 16-key utility settings

Rather than translating 14→16 keys on every run (in every function that reads the IC JSON), migrate all IC data files once and resave them. The `- Copy.json` file and git history both serve as backups.

### Why this matters

When the model space expanded from 476 to 505 (adding `include_welfare_efficiency_term` and `include_relative_income_penalty`), the IC analysis files were not re-generated — they still reflect the 476-model world. Any code that reads those files and tries to compare settings against the current 16-key canonical form will get mismatches unless it translates on the fly. Rather than adding translation overhead to every such function, we do the translation once here.

After the migration, `_load_empirical_fitted_parameters` and `_resolve_bic_winner_utility_settings` can assume 16-key form throughout — no per-call legacy translation needed.

### What needs migrating

| Artifact | Change |
|---|---|
| `bic_aic/All_Utility_Forms_IC_Analysis_Experiment3.json` | Top-level tuple keys (14 → 16 bool) + `model_entry["utility_settings"]` dicts (14 → 16 key) |
| `bic_aic/All_Utility_Forms_IC_Analysis_Experiment3.csv` | Add 2 new setting columns (`include_welfare_efficiency_term`, `include_relative_income_penalty`) with `False` values |
| `bic_aic/IC_Analysis_Comparison_Table_Experiment3.csv` | Same 2 new columns |
| `processed/equation_to_settings.json` | Regenerate via `equation_to_settings(..., create_new_file=True)` |
| `processed/model_nesting_data.json` | Regenerate via `model_nesting_adjacency_matrices(..., create_new_file=True)` |

The two new columns in the CSVs should be inserted after `min_max_rawlsian_leontief` and before `use_exponential_parameters` — matching the canonical insertion order in `config.utility_settings`.

### Migration helper: `migrate_ic_data_to_16_key(file_paths, general_settings)`

Add as a standalone module-level helper in `utilities.py` (not nested). Idempotent: if already 16-key, each block skips silently.

```python
def migrate_ic_data_to_16_key(file_paths: FilePaths, general_settings: GeneralSettings) -> None:
    """
    One-time migration that expands all IC analysis artifacts from 14-key to 16-key
    utility settings format, then regenerates derived files. Idempotent: each block
    runs silently without changes if already migrated.

    Arguments:
        • file_paths: FilePaths; standard project file-path dict.
        • general_settings: GeneralSettings; forwarded to model_nesting_adjacency_matrices.
    """
```

**JSON migration block:**
```python
experiment_num = general_settings.get('experiment_num', 3)
new_two_keys = ('include_welfare_efficiency_term', 'include_relative_income_penalty')
legacy_14_key_template = {setting_key: False for setting_key in config.utility_settings
                          if setting_key not in new_two_keys}

ic_json_path = os.path.join(file_paths["bic_aic"],
    f"All_Utility_Forms_IC_Analysis_Experiment{experiment_num}.json")
with open(ic_json_path, 'r', encoding='utf-8') as ic_json_file:
    ic_data = json.load(ic_json_file)

"Skip if already 16-key (idempotency guard)."
first_model_us = next(iter(ic_data["ic_results"].values())).get("utility_settings", {})
if len(first_model_us) < 16:
    migrated_ic_results = {}
    for old_tuple_key, model_entry in ic_data["ic_results"].items():
        old_utility_settings = model_entry.get("utility_settings", {})
        new_utility_settings = convert_utility_settings(
            utility_settings=old_utility_settings,
            into=dict,
            input_settings_format=legacy_14_key_template,
        )
        model_entry["utility_settings"] = new_utility_settings
        new_tuple_key = str(convert_utility_settings(utility_settings=new_utility_settings, into=tuple))
        migrated_ic_results[new_tuple_key] = model_entry
    ic_data["ic_results"] = migrated_ic_results
    with open(ic_json_path, 'w', encoding='utf-8') as ic_json_file:
        json.dump(ic_data, ic_json_file, ensure_ascii=False, indent=4)
    print(f"[migrate_ic_data_to_16_key] JSON migrated: {pretty_path(ic_json_path)}")
```

**CSV migration block:**
```python
new_setting_cols = ['include_welfare_efficiency_term', 'include_relative_income_penalty']
insert_after_col  = 'min_max_rawlsian_leontief'

for csv_file_name in (f"All_Utility_Forms_IC_Analysis_Experiment{experiment_num}.csv",
                      "IC_Analysis_Comparison_Table_Experiment3.csv"):
    csv_path = os.path.join(file_paths["bic_aic"], csv_file_name)
    if not os.path.exists(csv_path):
        continue
    ic_csv = pd.read_csv(csv_path, encoding='utf-8', engine='python')
    if new_setting_cols[0] in ic_csv.columns:
        continue  # already migrated
    insert_position = ic_csv.columns.tolist().index(insert_after_col) + 1
    for col_offset, new_col in enumerate(new_setting_cols):
        ic_csv.insert(loc=insert_position + col_offset, column=new_col, value=False)
    ic_csv.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"[migrate_ic_data_to_16_key] CSV migrated: {pretty_path(csv_path)}")
```

**Regenerate derived files:**
```python
"Rebuild equation_to_settings.json from the complete 16-key utility universe."
import model as _model
equation_to_settings(
    equation_function=_model.build_utility_equation,
    utility_settings=config.utility_settings,
    file_paths=file_paths,
    create_new_file=True,
)

"Rebuild model_nesting_data.json — nesting structure is now defined over the full 505-model space."
model_nesting_adjacency_matrices(
    general_settings=general_settings,
    utility_settings=config.utility_settings,
    file_paths=file_paths,
    create_new_file=True,
    print_=True,
)
```

**Call site:** invoke `migrate_ic_data_to_16_key(file_paths=file_paths, general_settings=general_settings)` near the start of `run_population_recovery_bootstrap` (before loading the IC JSON). Idempotency guards mean subsequent calls add no cost.

---

## Step 2 — Add `run_population_recovery_bootstrap` in simulation.py

Place immediately after `run_param_recovery_by_k` (~line 2503). The structure mirrors `run_param_recovery_by_k`: a setup phase, a bootstrap loop (with 3 sub-phases inside each iteration), and an output phase.

### Signature

```python
def run_population_recovery_bootstrap(
    general_settings: GeneralSettings,
    file_paths: FilePaths,
    param_bds: ParamBounds,
    figure_layout: FigLay,
    utility_settings: UtilitySettings | None = None,
    player_role: str = 'chooser',
    dynamic_predictor: bool = False,
    parameters_of_interest: list[str] | None = None,
    n_bootstrap_iterations: int = 1,
    n_games: int | None = None,
    random_seed: int | None = None,
    enforce_memory_limit: bool = False,
    create_new_file: bool | None = None,
    base_hue: int | None = None,
) -> tuple[pd.DataFrame, dict]:
```

**Argument notes:**
- `utility_settings=None` → resolved dynamically to the BIC winner for `experiment_num`. Print the resolved form. Never hard-code k=7 or any specific settings dict.
- `player_role='chooser'` — default matches the static fits that produced the paper's Section 8 statistics. Predictor-side bootstrap is supported but not the primary use case.
- `dynamic_predictor=False` — forwarded to `create_simulated_experiment` and `general_settings_for_bootstrap`. False matches the static fit; True engages the full UBM belief-updating loop.
- `n_bootstrap_iterations=1` — each iteration is expensive (~single-digit hours on Greg's machine for one pass). Default 1. Set higher only when standard errors on the bias are needed.
- `n_games=None` — cascade from `general_settings['parameter_recovery_settings']['n_games']`, then default 60 (matching the k-by-k recovery convention and the approximate real-experiment game count).
- `random_seed=None` — cascade from `general_settings['parameter_recovery_settings']['random_seed']`. `None` passed to `np.random.default_rng` gives an unseeded run.
- `create_new_file=None` — cascade from `general_settings['create_new_file']`.

### Inner helpers (all nested inside `run_population_recovery_bootstrap`)

**`_resolve_bic_winner_utility_settings()`** — ~25 lines

Reads `bic_aic/All_Utility_Forms_IC_Analysis_Experiment{experiment_num}.csv`, finds the row with the minimum BIC, builds a 16-key settings dict. After the Step 1b migration, the CSV already has the two new columns (both False for all rows), so no backfill is needed here.

```python
setting_cols_14 = [
    'conditional_welfare_mode', 'reference_dependent_altruism', 'min_max_rawlsian_leontief',
    'use_exponential_parameters', 'apply_exponents_to_payoffs', 'uniform_exponential_parameter',
    'single_payoffs_not_differences', 'payoff_ratios_not_differences', 'reference_dependent_utility',
    'use_negativity_parameters', 'negativity_social_comparison', 'fix_self_interest_parameter',
    'include_social_comparison', 'include_altruism_term',
]
```

After reading the best row, build the full 16-key dict by reading all setting columns present in the CSV (including the two new ones after migration) and defaulting any missing columns to False. Print the resolved bitstring and equation string before returning — this is non-negotiable; it is the primary defense against silently testing the wrong model.

**`_load_empirical_fitted_parameters(resolved_utility_settings)`** — ~25 lines

After migration, the IC JSON has 16-key `utility_settings` dicts. The loader:
1. Opens the JSON
2. For each model, converts `model_entry["utility_settings"]` to tuple via `convert_utility_settings(..., into=tuple)` and compares to `convert_utility_settings(resolved_utility_settings, into=tuple)`
3. When matched: for each player UUID in `model_entry["minvec"]`, extracts `model_entry["minvec"][player_uuid]["params"][player_role]`
4. Drops players where the params dict is empty or any value is `None`/`NaN`
5. Raises `RuntimeError` if no match found (with a message that prints the target bitstring)
6. Returns `dict[str, dict[str, float]]` of length N (typically 73 for Experiment 3)

**`_bootstrap_run_label(iteration_index)`** — ~10 lines

Returns a label encoding the utility settings (bitstring via `convert_utility_settings(into=str)`), the player role, the iteration index, and optionally the seed. Used to construct unique file names for every bootstrap run, so multiple configurations can coexist.

**`_collect_iteration_fits(fit_dir, k_params, true_params_by_uuid, parameters_to_report)`** — ~45 lines

Mirrors `_collect_role_results` from `run_param_recovery_by_k`. Finds all JSONs in `fit_dir` matching the `synthetic_{k_params}_` UUID prefix; for each, extracts fitted params using the same method-priority chain:

```python
for method_name in ("grid", "particle", "naive", "update", "globloc", "bayes", "general"):
    if method_name in estimates_by_method and target_uuid in estimates_by_method[method_name]:
        fitted_params = estimates_by_method[method_name][target_uuid].get(player_role, {}).get("params", {})
        break
```

Returns `{synthetic_uuid: {"true": true_params_by_uuid[synthetic_uuid], "fitted": fitted_params}}` — only includes UUIDs that are in `true_params_by_uuid` (i.e., the empirical players, not padding players).

### Main body outline

```
Setup:
  1. Cascade create_new_file from general_settings
  2. Cascade n_games (default 60)
  3. Cascade random_seed from general_settings['parameter_recovery_settings']['random_seed']
  4. Run IC data migration: migrate_ic_data_to_16_key(file_paths, general_settings) — idempotent
  5. Resolve utility_settings: if None, call _resolve_bic_winner_utility_settings()
  6. make_param_info for resolved_utility_settings → param_info_for_bootstrap
  7. is_valid_utility_settings sanity check
  8. k_params = count_free_parameters(resolved_utility_settings)
  9. empirical_params = _load_empirical_fitted_parameters(resolved_utility_settings)
  10. n_synthetic_players = len(empirical_params)
  11. Print mandatory settings stamp: bitstring, equation, N players, n_games, role, seed
  12. parameters_to_report = parameters_of_interest or [all non-std/non-cov param keys]
  13. random_gen = np.random.default_rng(random_seed)
  14. Build general_settings_for_bootstrap (deep copy + overrides):
      - fit_roles_together = False
      - warmstart_policy disabled (same overrides as in run_param_recovery_by_k)
      - fit_predictor_role = (player_role == 'predictor')
      - dynamic_predictor = dynamic_predictor
      - experiment_num = 3
      - create_new_file = True
  15. bootstrap_subdir = os.path.join(file_paths['processed'], 'population_param_bootstrap')
      os.makedirs(bootstrap_subdir, exist_ok=True)
  16. bootstrap_label = _bootstrap_run_label(iteration_index=0)  (label for output files)
  17. Print startup banner (iterations, N agents, n_games, seed, output dir)

  Before entering the loop, check for a cached metrics CSV: if create_new_file is False and the
  metrics CSV for this bootstrap_label exists, load and return it immediately.

Bootstrap loop (iteration_index in range(n_bootstrap_iterations)):
  t_iter_start = time.time()

  A. Build file_paths_for_iter (deep copy of file_paths, then redirect):
     iter_label = _bootstrap_run_label(iteration_index)
     - processed    → bootstrap_subdir
     - player_fits  → os.path.join(bootstrap_subdir, 'player_fits')
     - param_data   → os.path.join(bootstrap_subdir, 'param_data')
     - file_names['player_pairs_exper3']     = f"synthetic_histories_{iter_label}.json"
     - file_names['players_to_dyads_exper3'] = f"players_to_dyads_{iter_label}.json"

  B. Phase 1 — Generate synthetic data:
     histories_dict, true_params_by_uuid = create_simulated_experiment(
         n_players=n_synthetic_players,
         n_games=n_games,
         k_params=k_params,
         utility_settings_k=resolved_utility_settings,
         general_settings=general_settings_for_bootstrap,
         param_bds=param_bds,
         random_gen=random_gen,
         empirical_chooser_parameters=empirical_params,
         file_paths=file_paths_for_iter,
         create_new_file=True,
         enforce_memory_limit=enforce_memory_limit,
     )

  C. Phase 2 — Fit synthetic data:
     Load histories from disk (same pattern as run_param_recovery_by_k _fit_one_k):
     with open(iter_histories_file_path, 'r', encoding='utf-8') as histories_file:
         histories_loaded = json.load(histories_file)
     run_analysis_bayes(
         histories_data=histories_loaded,
         file_paths=file_paths_for_iter,
         param_info=param_info_for_bootstrap,
         utility_settings=resolved_utility_settings,
         general_settings=general_settings_for_bootstrap,
         print_=True,
     )

  D. Phase 3 — Collect fitted params and compute metrics:
     fit_dir = os.path.join(file_paths_for_iter['player_fits'], 'experiment_3')
     paired_fits = _collect_iteration_fits(fit_dir, k_params, true_params_by_uuid, parameters_to_report)
     for param_key in parameters_to_report:
         empirical_vals = np.array([player_pair["true"].get(param_key, np.nan) for player_pair in paired_fits.values()])
         recovered_vals = np.array([player_pair["fitted"].get(param_key, np.nan) for player_pair in paired_fits.values()])
         valid_mask = ~(np.isnan(empirical_vals) | np.isnan(recovered_vals))
         n_valid_players = int(valid_mask.sum())
         correlation_pearson, _ = pearsonr(empirical_vals[valid_mask], recovered_vals[valid_mask])
         bias = np.mean(recovered_vals[valid_mask]) - np.mean(empirical_vals[valid_mask])
         bias_normalized = bias / np.std(empirical_vals[valid_mask])
         variance_ratio = np.std(recovered_vals[valid_mask]) / np.std(empirical_vals[valid_mask])
         regression_slope, regression_intercept, regression_r, _, _ = scipy_stats.linregress(
             empirical_vals[valid_mask], recovered_vals[valid_mask])
         iteration_records.append({
             'iteration_index': iteration_index,
             'parameter_key': param_key,
             'n_synthetic_agents': n_valid_players,
             'mean_empirical': np.mean(empirical_vals[valid_mask]),
             'std_empirical': np.std(empirical_vals[valid_mask]),
             'mean_recovered': np.mean(recovered_vals[valid_mask]),
             'std_recovered': np.std(recovered_vals[valid_mask]),
             'correlation_pearson': correlation_pearson,
             'bias': bias,
             'bias_normalized': bias_normalized,
             'variance_ratio': variance_ratio,
             'regression_slope': regression_slope,
             'regression_intercept': regression_intercept,
             'regression_r2': regression_r ** 2,
         })

  E. Per-iteration progress print: elapsed time, ETA for remaining iterations

Output:
  metrics_df = pd.DataFrame(iteration_records)
  Save metrics CSV:
    bootstrap_subdir/population_recovery_metrics_{bootstrap_label}.csv (encoding='utf-8-sig')
  Save figure:
    from visualization import plot_population_recovery_bootstrap as _plot_bootstrap
    _plot_bootstrap(metrics_df=metrics_df, figure_layout=figure_layout,
                    base_hue=base_hue, out_fig_path=<html_path>)
  Write summary markdown:
    bootstrap_subdir/population_recovery_summary_{bootstrap_label}.md
    (See decision logic thresholds in Section 9 below)
  Print completion banner with output paths.

Return (metrics_df, detailed_results_dict)
```

`detailed_results_dict` maps `iteration_index → paired_fits` for downstream inspection.

### Required imports (add to simulation.py header if not already present)
- `from scipy.stats import pearsonr`
- `from scipy import stats as scipy_stats`

---

## Step 3 — Add `plot_population_recovery_bootstrap` to visualization.py

Two-panel Plotly figure. Follow all style guide rules (HSLA colors via `_hsla`, no CSS/hex/rgba, keyword args).

```python
def plot_population_recovery_bootstrap(
    metrics_df: pd.DataFrame,
    figure_layout: FigLay,
    base_hue: int | None = None,
    out_fig_path: str | None = None,
) -> go.Figure:
```

- Use `plotly.subplots.make_subplots(rows=1, cols=2)`
- **Panel 1 — scatter (empirical vs recovered)**:
  - One trace per unique `parameter_key` in `metrics_df`
  - Color: `_hsla(hue=(base_hue + 20 * series_index) % 360, saturation_percent=60, lightness_percent=50, alpha=1.0)`
  - x = `mean_empirical`, y = `mean_recovered` (aggregated across iterations when `n_bootstrap_iterations > 1`)
  - If `n_bootstrap_iterations > 1`: individual iteration points as semi-transparent markers, mean as filled marker
  - Dashed grey identity line (y=x) as a reference shape: `_hsla(hue=0, saturation_percent=0, lightness_percent=40)`
- **Panel 2 — bar chart (bias_normalized)**:
  - One bar per `parameter_key`, color = same HSLA scheme as Panel 1
  - y = `bias_normalized`
  - Bold black horizontal line at y=0
  - If `n_bootstrap_iterations > 1`: SE error bars (SE = std of `bias_normalized` across iterations / sqrt(n_iterations))
- Apply `figure_layout` template; save HTML via `fig.write_html(out_fig_path)`

---

## Step 4 — .gitignore additions

Add near the end of `.gitignore`:

```
# Bootstrap population recovery — large synthetic JSON files; metrics CSV and figure HTML are tracked
processed/population_param_bootstrap/synthetic_histories_*.json
processed/population_param_bootstrap/players_to_dyads_*.json
processed/population_param_bootstrap/player_fits/
processed/population_param_bootstrap/param_data/
```

---

## 7. Decision logic for the paper (from design doc Section 9)

These thresholds guide how to frame bootstrap results in Section 7.4 of the manuscript. They apply per parameter.

**Population means recover** — all of: `abs(bias_normalized) < 0.2`, `regression_slope > 0.7`, and (when `n_bootstrap_iterations > 1`) the 0 line is within `bias ± 1·SE`.

*Write:* "A parametric bootstrap from the empirical fitted distribution confirms that population-level means recover without substantial bias under the same fitting pipeline applied to real human data, even though individual-level recovery correlations are modest. The directional super-linearity finding is therefore supported despite the individual-level noise."

**Biased but directional** — `abs(bias_normalized) > 0.3` or `regression_slope < 0.5` for at least one parameter.

*Write:* Name the bias, identify the likely source (L2 penalty toward 1, exponent/weight trade-off), report bias-corrected means alongside raw estimates. Soften directional claim: exponents exceed 1 on average, but the magnitude is partly suppressed by identifiability constraints.

**Severe bias** — `abs(bias_normalized) > 0.5` or `regression_slope < 0.3` for any parameter.

*Write:* The directional claim itself is in trouble. Discussion must engage with the possibility that apparent super-linearity is partly an artifact of the optimization landscape.

These thresholds should appear in the summary markdown written by `run_population_recovery_bootstrap`, along with a sentence-level verdict for each parameter.

---

## 8. Gotchas and implementation notes (from design doc Section 10)

**The L2 penalty.** The exponent penalty `(mean(γ) − 1)²` is the most likely source of bias if the bootstrap finds it. Worth logging the penalty's contribution to total loss at the end of each run. If the result is biased, the natural follow-up experiment is to rerun with the exponent penalty disabled (`λ=0` on exponents only) and see whether the bias disappears. Do not do this in the first pass — flag as a follow-up in the summary markdown.

**Memory.** `create_simulated_experiment` already implements `enforce_memory_limit`. The bootstrap passes this through. Seven parameters with default grid bins is well within memory.

**Time.** A single fit of one utility function on 73 synthetic agents is roughly equivalent to one row of the IC analysis. Without `dynamic_predictor=True`, expect single-digit hours on Greg's machine for one iteration. On Rick's 28-core machine, under an hour. Multiply by `n_bootstrap_iterations`.

**`dynamic_predictor=False`.** Default matches the static fit used for the paper's Section 8 statistics. When True, the simulation and fitting both engage the full UBM belief-updating loop for the predictor side. This is expensive and is not needed for the immediate γ defense. The argument is plumbed through so it works in both modes.

**Settings stamp is non-negotiable.** Print the resolved utility_settings bitstring and the resolved equation string at the start of every run. The biggest silent-bug risk in this analysis is testing the wrong utility function. Any run without this stamp should be treated as suspect.

**Caching.** The bootstrap respects the standard caching pattern. `create_new_file=False` with a populated cache subdirectory loads the cached metrics CSV and returns immediately. The iteration index is encoded in cache filenames so partial multi-iteration runs can resume from the last completed iteration.

**Unicode param keys.** The IC JSON stores parameter names as Unicode (Vᵢᵢ, Ɛᵢⱼ, γ1, etc.). Pass these through as-is — do not re-encode or rename. `make_param_info` uses the same Unicode keys, so they will match.

---

## 9. Key risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Bootstrap tests wrong utility function silently | Mandatory settings stamp at startup (bitstring + equation); `is_valid_utility_settings` guard |
| File collision with real participant data or k-by-k recovery data | All I/O routed to `processed/population_param_bootstrap/` via scoped `file_paths_for_iter` |
| 14-key IC JSON mismatched to 16-key canonical form | One-time migration via `migrate_ic_data_to_16_key` in Step 1b; idempotent guard on every call |
| Padding players (N empirical not a multiple of 4) | Padding players use uniform sampling; not in `true_params_by_uuid`; excluded from metrics automatically |
| `altruism_key`/`altruism_targets` now optional → silent breaks | All existing call sites use keyword args; `None`-default + explicit validation catches omissions early |
| Unicode param keys corrupted in transit | Pass through as-is; `make_param_info` uses the same keys — no re-encoding needed |
| L2 penalty biasing γ population means | Flag in summary markdown; note as follow-up experiment to test |

---

## 10. Verification plan

1. Confirm that `migrate_ic_data_to_16_key` runs without error and that the IC JSON now has 16-element tuple keys; inspect one entry to verify key order matches `config.utility_settings`
2. Run `run_population_recovery_bootstrap(n_bootstrap_iterations=1, n_games=10, create_new_file=True)` as a smoke test — should complete in minutes and write metrics CSV + figure HTML
3. Confirm metrics CSV at `processed/population_param_bootstrap/population_recovery_metrics_*.csv` with the expected 13 columns
4. Confirm figure HTML opens with two panels (scatter + bar chart)
5. Confirm that no synthetic histories JSON appeared in `simulations/`, `player_fits/`, or `raw_data/` — only in `processed/population_param_bootstrap/`
6. Run `run_param_recovery_by_k` for a single k and confirm `simulations/param_recovery_by_k/param_recovery_by_k.csv` is unaffected
7. Run with `n_games=60` for real results; compare `correlation_pearson` for the resolved model against the value in `simulations/param_recovery_by_k/param_recovery_by_k.csv` (should be consistent within sampling variance)
8. Run with an explicitly supplied `utility_settings` dict different from the BIC winner to confirm model-agnostic behavior
9. Confirm `create_simulated_experiment` with `empirical_chooser_parameters=<dict>` (and `altruism_key=None`) works; confirm the same function with the original signature still works identically
