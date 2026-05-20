# Plan: Participant Model Fit Extraction + Realistic AMPD Sampling

## Context

The individual architecture analysis tests whether participants differ mainly by
parameter values inside one shared utility architecture, or by needing different
utility-function architectures. Two tasks are covered here:

1. **Main task — Participant fit extraction**: Extract per-participant × per-model
   loss metrics from the IC JSON's `minvec` field and reshape them into a cached
   long-format CSV that Stages 7–12 will read repeatedly.

2. **Side-quest — Realistic AMPD sampling from actual participants**: Add a new
   `parameter_sampling_mode='participant_sampled'` to `compute_ampd_distance_matrix`
   that independently samples each canonical parameter from the pool of actual
   participant-fitted values (from minvec), giving much richer AMPD diversity than
   either uniform or single-participant-vector sampling.

---

## Why the CSV is needed

The 617 MB IC JSON is the authoritative data source, but loading it on every
downstream function call is impractical. The extraction CSV is a ~34K-row
(73 players × 476 models) reshaped cache of **loss and IC metrics only** — no
parameter values. Stages 7–12 will read this CSV directly without touching the
JSON again unless re-extraction is explicitly requested.

---

## Data confirmed available in current repo

- `bic_aic/All_Utility_Forms_IC_Analysis_Experiment3.json` (617 MB) — has `minvec`
  for all 476 models, all 73 players, including k=0. Per-player losses verified:
  they sum exactly to the aggregate `loss` field in each model entry.
- Raw experiment data readable by `dyads_for_a_player` — provides exact per-player
  game counts for n_data.

---

## Task 1: `extract_participant_model_combined_fits`

### File to modify
`analysis.py` — append after line 8207. Section label should be descriptive:

```python
"=========================================================================================="
"====================== Participant Model Fit Extraction =================================="
"=========================================================================================="
```

### Function signature
```python
def extract_participant_model_combined_fits(
    general_settings: dict,
    file_paths: dict,
    create_new_file: bool = False,
) -> pd.DataFrame:
```

Docstring covers: what it does, arguments (bullet format), returns (CSV path + DataFrame).

### Algorithm

**Step 1 — Resolve paths**
- IC JSON: `file_paths['bic_aic'] / All_Utility_Forms_IC_Analysis_Experiment{N}.json`
- Output CSV: `file_paths['processed'] / participant_model_combined_fits.csv`

**Step 2 — Early return if cached**
- If `create_new_file=False` and output CSV exists, load and return it.

**Step 3 — TEMPORARY PATCH (clearly delimited, easy to delete)**
```python
# ============================ TEMPORARY PATCH ====================================
# Remove this block once experiment_3 IC data is fully up-to-date in this repo.
_OLD_REPO_IC_JSON_PATH = (
    r"C:\Users\Gregory Stanley\Desktop\U of M\Research Archive\Multiplayer"
    r"\ABM_Simulation\Judgment_Game\Inputs\Iter_Binary_Dictator"
    r"\bic_aic\All_Utility_Forms_IC_Analysis_Experiment3.json"
)
_MINIMUM_VALID_IC_JSON_BYTES = 50_000_000
if (
    not os.path.exists(ic_json_path)
    or os.path.getsize(ic_json_path) < _MINIMUM_VALID_IC_JSON_BYTES
):
    print(
        f"\n{'='*72}\n"
        f"[TEMPORARY PATCH] Current-repo IC JSON is missing or too small.\n"
        f"  Current path  : {ic_json_path}\n"
        f"  Falling back  : {_OLD_REPO_IC_JSON_PATH}\n"
        f"Delete this block once this repo's IC data is regenerated.\n"
        f"{'='*72}\n"
    )
    ic_json_path = _OLD_REPO_IC_JSON_PATH
# ========================= END TEMPORARY PATCH ===================================
```

**Step 4 — Compute n_data per player per role from raw game history**

Use preprocessing functions to get exact per-player game counts. Store all three
values (chooser, predictor, combined) because downstream stages may want per-role
BIC or per-role weighting:
```python
experiment_num = general_settings['experiment_num']
all_uuids = prep.all_player_uuids(
    file_paths=file_paths, experiment_num=experiment_num, only_humans=True,
)
n_data_by_player = {}  # maps player_uuid -> {'n_chooser': int, 'n_predictor': int, 'n_combined': int}
for player_uuid in all_uuids:
    dyads = prep.dyads_for_a_player(
        player_uuid=player_uuid, experiment_num=experiment_num,
        file_paths=file_paths, dyad_already_analyzed=False,
    )
    n_chooser = sum(
        len(games) for games in dyads.values()
        if games and games[0]['chooser'] == player_uuid
    )
    n_predictor = sum(
        len(games) for games in dyads.values()
        if games and games[0]['predictor'] == player_uuid
    )
    n_data_by_player[player_uuid] = {
        'n_chooser': n_chooser,
        'n_predictor': n_predictor,
        'n_combined': n_chooser + n_predictor,
    }
```
Print each player's (n_chooser, n_predictor, n_combined) so unusual counts are
visible. Most players have 120 total games with randomly assigned roles.
Add per-role columns to the output CSV: `n_chooser`, `n_predictor`, `n_combined`.

**Step 5 — Load IC JSON**
Print a message before loading ("Loading IC JSON from {path}..."). Load with
`encoding='utf-8-sig'`.

**Step 6 — Build long-format rows**

For each model key in `ic_results`:
- Skip models that have no `minvec` or an empty `minvec` dict.
- Extract: `utility_tuple_str = model_key`, `utility_idx = entry['idx']`,
  `k_params = entry['k_params']`, `equation = entry['U']`,
  `utility_settings_dict = entry['utility_settings']`
- For each `player_uuid` in `minvec`:
  - `chooser_loss_nll = minvec[player_uuid]['loss']['chooser']`
  - `predictor_loss_nll = minvec[player_uuid]['loss']['predictor']`
  - `combined_loss_nll = chooser_loss_nll + predictor_loss_nll`
  - `k_effective = 2 * k_params` (both roles free)
  - Look up `n_entry = n_data_by_player.get(player_uuid, None)`
  - `n_chooser = n_entry['n_chooser'] if n_entry else np.nan`
  - `n_predictor = n_entry['n_predictor'] if n_entry else np.nan`
  - `n_combined = n_entry['n_combined'] if n_entry else np.nan`
  - `AIC_individual = 2 * combined_loss_nll + 2 * k_effective`
  - `BIC_individual = 2 * combined_loss_nll + k_effective * np.log(n_combined)` if n_combined is finite

**Note on 480 vs 476 discrepancy**: The model universe has 480 forms but the old IC
JSON has only 476. This is not an error. Code will simply not produce rows for the 4
missing models. Add a print statement noting how many models were found in the JSON.

**Step 7 — Per-participant ΔBIC**
- Group by `player_uuid`; `delta_BIC_individual = BIC_individual - min(BIC_individual)`
- Use `groupby(...).transform('min')` for efficiency.

**Step 8 — BIC weights (numerically stable)**
```python
from scipy.special import logsumexp
log_w_unnorm = -0.5 * delta_BIC_individual
# Per participant: log_Z = logsumexp of log_w_unnorm
log_z_by_player = (
    combined_fits_df.groupby('player_uuid')['log_w_unnorm']
    .transform(lambda x: logsumexp(x.values))
)
combined_fits_df['BIC_weight'] = np.exp(log_w_unnorm - log_z_by_player)
```

**Step 9 — Per-participant summary columns** (computed per group, joined back)
- `effective_number_of_models = 1 / sum(w^2)`
- `model_weight_entropy = -sum(w * np.log(w + 1e-300))`
- `top_model_utility_idx` = utility_idx at argmin(ΔBIC)
- `top_model_delta_BIC` = min(ΔBIC) per player
- `n_models_with_delta_BIC_le_2` = count where ΔBIC ≤ 2
- `n_models_with_delta_BIC_le_10` = count where ΔBIC ≤ 10

**Step 10 — Save**
```python
combined_fits_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
print(f"Saved: {output_csv_path}  ({len(combined_fits_df)} rows)")
```

### Output CSV columns
`player_uuid`, `utility_tuple_str`, `utility_idx`, `k_params`, `k_effective`,
`equation`, `chooser_loss_nll`, `predictor_loss_nll`, `combined_loss_nll`,
`n_chooser`, `n_predictor`, `n_combined`, `AIC_individual`, `BIC_individual`,
`delta_BIC_individual`, `BIC_weight`, `effective_number_of_models`,
`model_weight_entropy`, `top_model_utility_idx`, `top_model_delta_BIC`,
`n_models_with_delta_BIC_le_2`, `n_models_with_delta_BIC_le_10`

Plus the 14 boolean utility settings columns (joined from each model's
`entry['utility_settings']` dict).

---

## Task 2 (Side-quest): Realistic AMPD sampling from participant values

### New parameter_sampling_mode: `'participant_sampled'`

**File to modify**: `analysis.py` — the section containing `_sample_full_reference_params`
and the AMPD computation logic.

**Pool-building logic: local function inside `compute_ampd_distance_matrix`**

Rather than a module-level `_` helper, define a local function inside
`compute_ampd_distance_matrix`:

```python
def _build_participant_parameter_pools_from_ic(ic_json_path_local):
    """Build per-parameter value pools from all participant minvec entries."""
    pools = {key: [] for key in param_bds}
    with open(ic_json_path_local, 'r', encoding='utf-8-sig') as fh:
        ic_json = json.load(fh)
    for model_entry in ic_json['ic_results'].values():
        for player_data in model_entry.get('minvec', {}).values():
            for role in ('chooser', 'predictor'):
                for param_name, param_value in player_data['params'].get(role, {}).items():
                    if param_name in pools:
                        pools[param_name].append(float(param_value))
    return {k: v for k, v in pools.items() if v}
```

Call this once at the top of `compute_ampd_distance_matrix` if
`parameter_sampling_mode == 'participant_sampled'`.

**Note on existing `_` module-level helpers before `compute_ampd_distance_matrix`**

During implementation, grep for each of these helpers outside their local usage
context. If any are referenced only within `average_model_policy_distance`, move
them inside as inner functions per the style guideline. If referenced elsewhere,
leave them as module-level.

**Extend `_sample_full_reference_params` (or its local equivalent)**

Add a new branch:
```python
elif parameter_sampling_mode == 'participant_sampled':
    for key in param_bds:
        pool = participant_parameter_pools.get(key, [])
        params[key] = float(rng.choice(pool)) if pool else rng.uniform(*param_bds[key])
```
Pass `participant_parameter_pools` as a new optional kwarg (default `None`).

**Wiring**: If mode is `'participant_sampled'`, build pools once, then pass them
into each call to the reference-param sampler. No new config keys needed.

---

## Side-quest: create `plans/` folder in project root

Add a `plans/` directory at the project root to store planning documents like this
one for future reference. Copy the current plan file from the Claude plans directory
into `plans/participant_fit_extraction_and_realistic_ampd.md`. Going forward, plans
for this project should be saved there (in addition to or instead of the Claude
internal plans directory).

---

## Wire-up in main.py

Add near line 21 (inside `main()`), right after the current Stage 5 calls and
before the `exit()`:

```python
"Extract per-participant per-model combined fit table (Stage 6 of individual architecture analysis)."
combined_fits_df = extract_participant_model_combined_fits(
    general_settings=general_settings,
    file_paths=file_paths,
)
pp.pprint(combined_fits_df.head())
pp.pprint(combined_fits_df.groupby('player_uuid')['BIC_weight'].sum())
```

The second `pprint` is a quick sanity check that BIC weights sum to 1.0 per player.

---

## Verification

1. Run `main.py` — confirm `processed/participant_model_combined_fits.csv` is created
2. Row count: 73 players × (476 models from IC JSON) = 34,748 rows expected
3. Print confirms per-player n_data — flag players with ≠ 120 games
4. BIC weights sum to 1.0 per player (pprint of groupby sum)
5. Effective number of models between 1 and 476 for all players
6. TEMPORARY PATCH is NOT triggered (current repo JSON is 617 MB > 50 MB floor)
7. Re-run with `create_new_file=False`: loads from cache, no JSON load
8. (Side-quest) Set `ampd_settings['parameter_sampling_mode'] = 'participant_sampled'`
   in config.py and verify `compute_ampd_distance_matrix` runs without error
