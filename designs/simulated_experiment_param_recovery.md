# Plan: `create_simulated_experiment()` and `run_param_recovery_by_k` Redesign

## Context and Motivation

The goal of `run_param_recovery_by_k` is to test whether the optimizer reliably recovers a
chooser's true utility parameters from behavioral data. This is a classic "parameter recovery"
simulation: sample ground-truth utility params for each agent → generate synthetic choices under
those params → fit the optimizer blind to the true params → measure correlation between true and
recovered values. Running the study separately for each k-dimensional model family reveals how
recovery quality varies with model complexity, which is important for interpreting the IC results
(if k=7 recovers poorly, high-k wins could reflect overfit rather than true fit).

The previous implementation created synthetic choosers paired with dummy (non-updating) predictors,
giving each player a single role in a single dyad. This conflicted with the pipeline's assumption
that every player appears in multiple dyads and in both chooser and predictor roles. The "Final
agent pass" in `fit_params_by_player` crashed when it tried to set up predictor grid params for
what it thought was a pure-chooser UUID with no parameter_estimates.

The redesign mirrors the actual experiment structure exactly: groups of 4 players with round-robin
role switching. Each player acts as both chooser and predictor across multiple dyads, the same way
real participants do. By constructing the synthetic data in exactly the same format as the real
raw data (same JSON schema, same player_info fields, same dyad key conventions), we:
1. Eliminate all special-case guards in `bayesian.py` — the pipeline never needs to know it is
   running on simulation data vs. real data.
2. Reduce the risk of inadvertently breaking existing vital functions — the pipeline is exercised
   on a realistic input, so any regressions would surface immediately.
3. Make the parameter recovery test more faithful — the optimizer runs under exactly the same
   conditions it faces on real human data, so recovery quality is a direct proxy for how well
   the method will work in practice.

**Primary question answered by this simulation:** Does the optimizer recover the true Vᵢⱼ (and
other params) from a chooser's behavioral choices, across model families of varying dimensionality?

---

## Files to Modify

- [simulation.py](simulation.py) — new `latin_square()` function, new `create_simulated_experiment()`, Phase 1 rewrite, `_assemble_histories_dict` update, Phase 2/3 updates
- [bayesian.py](bayesian.py) — revert all 5 guard lines added in the previous session
- [config.py](config.py) — no changes needed (the `parameter_recovery_settings` block already exists; `fit_choosers` key can be removed since it will no longer be needed)

---

## Step 1: Revert `bayesian.py` Guards

All 5 lines added in the previous session must be reverted. The key insight: the guards were
only needed because synthetic UUIDs contained 'chooser'/'predictor' as substrings. With role-
agnostic UUIDs and `experiment_num=3`, the pipeline naturally handles both roles.

**Delete these 2 lines entirely** (`simulation.py → bayesian.py → fit_params_by_player`, ~line 1628):
```python
"In simulation (experiment_num=0), skip chooser fitting unless parameter_recovery_settings explicitly enables it."
if experiment_num == 0 and 'chooser' in player_uuid:
    if not general_settings.get('parameter_recovery_settings', {}).get('fit_choosers', False):
        return
```

**Delete these 3 lines entirely** (role loop, ~line 2097):
```python
if experiment_num == 0 and 'chooser' in player_uuid and player_role == 'predictor':
    continue
if experiment_num == 0 and 'predictor' in player_uuid and player_role == 'chooser':
    continue
if player_role == 'chooser' and experiment_num != 3 and not general_settings.get('parameter_recovery_settings', {}).get('fit_choosers', False):
    continue
```

**Restore the original chooser-skip line** — the 3rd line above was a modified version of the
original. The original was:
```python
if player_role == 'chooser' and experiment_num != 3:
    continue
```

**Delete these 2 lines entirely** (CSV save loop, ~line 2113):
```python
if experiment_num == 0 and role_to_fit == 'chooser' and not general_settings.get('parameter_recovery_settings', {}).get('fit_choosers', False):
    continue
```

After reversion, chooser roles are skipped for experiment_num != 3 (original behavior). Since we
will use `analysis_experiment_num = 3` for the param recovery simulation, chooser fitting happens
automatically without any guards.

---

## Step 2: Add `latin_square()` Function in `simulation.py`

Copy the function provided by the user as-is into simulation.py (module-level, near top):

```python
def latin_square(size: int) -> list[list[int]]:
    """
    Produces a symmetrical size x size latin square.

    Author: Liam Tsimhoni (Morality Game research assistant).
    """
    latin_sq = np.zeros(shape=(size, size), dtype=object)
    def val(idx, jdx, n_players=size):
        if (jdx == 0):                              return idx
        if (idx == n_players - 1):                  return ((n_players // 2) * (jdx - 1) % (n_players - 1))
        if (jdx == (2 * idx) % (n_players - 1) + 1): return n_players - 1
        return (jdx - 1 - idx) % (n_players - 1)
    for idx in range(size):
        for jdx in range(size):
            latin_sq[idx][jdx] = val(idx, jdx, size)
    return latin_sq
```

**How it encodes pairings for n_players=4:** The output is a 4×4 array. `latin_sq[idx][0] = idx`
always (diagonal = self), so column `jdx=0` is skipped. Columns `jdx=1,2,3` each define one
rotation of opponents: `latin_sq[idx][jdx] = k` means player `idx` is paired with player `k` in
rotation `jdx`. Since each column has no repeated values and `latin_sq[idx][jdx]=k` implies
`latin_sq[k][jdx]=idx` (symmetric), each column gives exactly 2 unique pairs from 4 players.

Verified for size=4:
```
latin_sq = [[0, 3, 1, 2],
             [1, 2, 0, 3],
             [2, 1, 3, 0],
             [3, 0, 2, 1]]
```
- Rotation jdx=1: pairs (0,3) and (1,2)
- Rotation jdx=2: pairs (0,1) and (2,3)
- Rotation jdx=3: pairs (0,2) and (1,3)

Each player faces each other player exactly once across the 3 rotations. ✓

**Pair extraction helper** (used inside `create_simulated_experiment()`):
```python
def _pairs_from_ls_column(latin_sq, jdx: int) -> list[tuple[int, int]]:
    seen, pairs = set(), []
    for idx in range(len(latin_sq)):
        k = int(latin_sq[idx][jdx])
        if k != idx and (idx, k) not in seen and (k, idx) not in seen:
            pairs.append((idx, k))
            seen.add((idx, k))
    return pairs
```

---

## Step 3: New `create_simulated_experiment()` Function

Write as a module-level function in `simulation.py`, right before `run_param_recovery_by_k`.

### Purpose and Design

Each batch of 4 players produces 6 dyads — one per unique undirected pair. The number of games
per pair varies across pairs and across batches, just as it did in the real experiment. This
variability is generated by sampling 3 random proportions per batch, which — through a system of
row-sum constraints — fully determine all 6 pair game counts.

- **4-player batches.** If n_players is not a multiple of 4, pad by adding extra synthetic
  players (with randomly sampled true params) until the count is a multiple of 4.
- **6 dyads per batch** (one per unique pair). Each player participates in exactly 3 dyads.
- **Game count parametrization:** For a batch {0,1,2,3}, sample 3 values x,y,z summing to
  n_games. These give pair game counts via:
  ```
  G[0,1]=x,  G[0,2]=y,  G[0,3]=n_games-x-y
  G[1,2]=z,  G[1,3]=n_games-x-z
  G[2,3]=n_games-y-z
  ```
  Each player's total = n_games (row-sum constraint by construction). Different batches get
  different (x,y,z) so matching proportions need not be the same across groups.
- **Role assignment** (chooser/predictor): For each pair (i<j), randomly decide which player
  is chooser (Bernoulli coin flip via `rng`). Across the 3 dyads each player participates in,
  they are chooser approximately half the time.
- Player UUIDs: `synthetic_{k_params}_{global_player_idx:04d}` — role-agnostic, zero-padded to
  uniform length (e.g. `synthetic_3_0012`).
- `player_type='synthetic'` in player_info — `run_analysis_bayes` line 2739 already updated to
  accept `'synthetic'` alongside `'participant'`, so this triggers auto-selection cleanly.
- `dynamic_predictor=True` — runs full UBM belief-updating so predictor data is available for
  future predictor recovery analysis without re-generating data.
- True params in `game[0]['true_params_chooser/predictor']` AND in `player_info[uuid]['true_params']`.

### Sampling (x, y, z)

```python
raw = rng.uniform(0.0, 1.0, 3)
x, y, z = (raw / raw.sum()) * n_games   # float proportions of n_games
"Convert to integers, preserving sum = n_games using largest-remainder rounding."
floored = np.floor([x, y, z]).astype(int)
remainders = [x - floored[0], y - floored[1], z - floored[2]]
remainder_needed = n_games - floored.sum()
top_indices = np.argsort(remainders)[-remainder_needed:]
for idx in top_indices:
    floored[idx] += 1
x, y, z = int(floored[0]), int(floored[1]), int(floored[2])
"Derived game counts (guaranteed ≥ 1 if x,y,z ≥ 1; clamp with max(1,...) for safety)."
g = {
    (0,1): x,              (0,2): y,              (0,3): max(1, n_games - x - y),
    (1,2): z,              (1,3): max(1, n_games - x - z),
    (2,3): max(1, n_games - y - z),
}
```

### Signature

```python
def create_simulated_experiment(
    n_players: int,
    n_games: int,
    k_params: int,
    utility_settings_k: UtilitySettings,
    param_info_k: dict,
    general_settings: GeneralSettings,
    param_bds: dict,
    random_gen: np.random.Generator,
    altruism_key: str,
    altruism_targets: list[float],
) -> tuple[list[dict], dict[str, dict]]:
```

Returns `(dyad_list, true_params_by_uuid)` where `dyad_list` is a list of `{DyadKey: DyadGames}`.

### Internal Loop Logic

```python
"Pad n_players to the nearest multiple of 4."
n_players_padded = n_players + (4 - n_players % 4) % 4

dyad_list           = []
true_params_by_uuid = {}

for batch_start in range(0, n_players_padded, 4):
    "Assign true params to each of the 4 players in this batch."
    batch_params = {}
    for local_idx in range(4):
        global_idx = batch_start + local_idx
        uuid = f"synthetic_{k_params}_{global_idx:04d}"
        params = _sample_params_from_bounds(param_info_k)
        params[altruism_key] = float(altruism_targets[global_idx % len(altruism_targets)])
        batch_params[local_idx] = params
        true_params_by_uuid[uuid] = dict(params)

    "Sample (x,y,z) for this batch and derive all 6 pair game counts."
    g = _sample_batch_game_counts(random_gen=random_gen, n_games=n_games)  # returns dict keyed by (min,max) pair

    "One dyad per unique pair, role randomly assigned."
    for (idx, jdx), n_games_pair in g.items():
        if random_gen.random() < 0.5:
            cho_local, pred_local = idx, jdx
        else:
            cho_local, pred_local = jdx, idx

        global_cho  = batch_start + cho_local
        global_pred = batch_start + pred_local
        chooser_uuid   = f"synthetic_{k_params}_{global_cho:04d}"
        predictor_uuid = f"synthetic_{k_params}_{global_pred:04d}"

        dyad = create_simulated_dyad(
            n_games=n_games_pair,
            params_chooser=batch_params[cho_local],
            params_predictor=batch_params[pred_local],
            general_settings=general_settings,
            utility_settings=utility_settings_k,
            param_bds=param_bds,
            default_utility_settings=False,
            dynamic_predictor=True,
        )
        (_, games_list), = dyad.items()
        for game in games_list:
            game["chooser"]   = chooser_uuid
            game["predictor"] = predictor_uuid
        fixed_key = f"({predictor_uuid}, {chooser_uuid})"
        dyad_list.append({fixed_key: games_list})

return dyad_list, true_params_by_uuid
```

The `_sample_batch_game_counts(rng, n_games)` helper encapsulates the (x,y,z) sampling and
derivation of all 6 pair game counts. Returns a dict `{(idx,jdx): int}` for the 6 unique pairs.

---

## Step 3b: Extend `create_simulated_dyad()` — New Parameters

Add two new optional parameters to `create_simulated_dyad` (both backwards-compatible defaults):

### `matching_probability: float = 1.0`

Currently hardcoded to `1.0` in each game dict. Change the one line in the game dict to:
```python
"matching_probability": matching_probability,
```

In `create_simulated_experiment`, pass `matching_probability = G[i][j] / n_games` to encode the
proportion of the game budget that this particular pair accounts for.

### `role_switch: bool = False`

**Key semantic:** when `role_switch=True`, `params_chooser` means **player 0's utility parameters**
and `params_predictor` means **player 1's utility parameters** — the parameters belong to the
player, not the role label. Phase 1 has player 0 as chooser and player 1 as predictor. Phase 2
reverses the roles, so player 0's params become the predictor params and player 1's params become
the chooser params. The docstring must state this clearly.

When `True`, simulates two complete phases — one per role direction — and returns a **2-key dict**
instead of the usual single-key dict. Phase 2 re-uses `_run_phase` with params/UUIDs swapped to
follow the players into their new roles:

```python
if role_switch:
    "Phase 1: player 0 = chooser (params_chooser), player 1 = predictor (params_predictor)."
    "Phase 2: roles flip — player 0 becomes predictor, player 1 becomes chooser."
    "         params follow the player, not the role label."
    phase2_dyad = _run_phase(
        n_games=n_games,
        params_chooser=params_predictor,   # player 1's params → player 1 is now chooser
        params_predictor=params_chooser,   # player 0's params → player 0 is now predictor
        chooser_uuid=predictor_uuid,       # player 1's UUID → now acting as chooser
        predictor_uuid=chooser_uuid,       # player 0's UUID → now acting as predictor
        ...
    )
    return {
        dyad_key: dyad_games,                                          # phase 1
        f"({chooser_uuid}, {predictor_uuid})": phase2_dyad,           # phase 2
    }
return {dyad_key: dyad_games}
```

The `_run_phase` helper encapsulates the existing per-game loop + dynamic predictor `agent()` call,
so the phase-1 and phase-2 simulation paths share the same logic without code duplication.

**`create_simulated_experiment` uses `role_switch=False` (default)**. Role assignment per pair is
handled at the batch level by the random Bernoulli coin flip, giving each player ≈1.5 chooser
dyads and ≈1.5 predictor dyads across their 3 pairs. The player0/player1 distinction is only
relevant when `role_switch=True`.

**Existing callers** (`create_simulated_data`, tests, etc.) use neither new parameter → no change.

---

## Step 4: Update `_assemble_histories_dict`

**Background on `player_type` values:**
- `'robot'` = bots from the actual experiment (stand-ins when fewer than 4 humans showed up). Functions like `all_player_uuids` filter these out, and `run_analysis_bayes` does not process them through the fitting pipeline.
- `'participant'` = real human participants — processed normally.
- `'synthetic'` = our simulated players (user already updated `run_analysis_bayes` line 2739 to accept `'participant'` or `'synthetic'`). `all_player_uuids` does not filter 'synthetic', so these players will be included in the fitting pipeline.

The old hardcoded `player_type='robot'` in `_assemble_histories_dict` was one of the reasons the old simulation code never reached Phase 2 correctly. Changing to `'synthetic'` fixes this.

Add two parameters:

```python
def _assemble_histories_dict(
    dyad_list: list[dict],
    player_type: str = 'robot',
    true_params_by_uuid: dict | None = None,
) -> dict:
```

Inside, when building `player_info[player_uuid]`, use `player_type` and inject `true_params`:
```python
info = {
    'player_type': player_type,
    'avatar_shape': avatar_shapes[random_gen.randint(0, len(avatar_shapes)-1)],
    'player_color': f'hlsa(...)',
}
if true_params_by_uuid and player_uuid in true_params_by_uuid:
    info['true_params'] = true_params_by_uuid[player_uuid]
player_info[player_uuid] = info
```

`_assemble_histories_dict` is an inner function of `run_param_recovery_by_k` with no external
callers, so the default `player_type='robot'` is vestigial — we always pass `'synthetic'` explicitly.
The `create_simulated_data` / `create_simulated_dyad` path never calls this function.

---

## Step 5: Phase 1 Rewrite in `run_param_recovery_by_k`

The current Phase 1 runs two separate loops (one for choosers, one for predictors) that each
call `create_simulated_dyad` and then replace UUIDs. Both loops are deleted and replaced with
a single `create_simulated_experiment` call per k.

**Before the k-loop** (near where `rng = random.Random(random_seed)` is today), add:
```python
random_seed = general_settings.get('random_seeds', {}).get('seed', None)
random_gen  = np.random.default_rng(random_seed)
```

**Remove** the existing `rng = random.Random(random_seed)` line (Python's `random.Random` object).

**Inside the k-loop**, replace the chooser and predictor loops (~lines 1836–1894) with:
```python
"Pad n_choosers to a multiple of 4; all synthetic players fill both roles across dyads."
n_players_k = n_choosers + (4 - n_choosers % 4) % 4
if n_players_k != n_choosers:
    print(f"[k={k_params}] n_choosers={n_choosers} is not a multiple of 4 — padding to {n_players_k}.")

dyads_for_k, true_params_by_uuid_k = create_simulated_experiment(
    n_players=n_players_k,
    n_games=n_games,
    k_params=k_params,
    utility_settings_k=u_settings_k,
    param_info_k=param_info_k,
    general_settings=general_settings,
    param_bds=param_bds,
    random_gen=random_gen,
    altruism_key=altruism_key,
    altruism_targets=altruism_targets,
)
histories_k = _assemble_histories_dict(
    dyad_list=dyads_for_k,
    player_type='synthetic',
    true_params_by_uuid=true_params_by_uuid_k,
)
```

The altruism_targets list can be generated before the k-loop (reused across k values) or inside
it. In either case it must be at least as long as `n_players_k`.

**`general_settings_k` changes:**
- Set `experiment_num = 3` (was `= 0`) — the existing check
  `if player_role == 'chooser' and experiment_num != 3: continue` in `bayesian.py` then
  naturally allows chooser fitting without any guards
- Remove the `fit_choosers` flag injection (no longer needed after bayesian.py guards reverted)

**File routing change** (inside the k-loop where `file_paths_k` is assembled):
```python
"Was: file_paths_k['file_names']['player_pairs_exper0'] = ..."
k_file_name = f"Social_Preference_Prediction_Pairs_Param_Recovery_k{k_params}.json"
file_paths_k['file_names']['player_pairs_exper3'] = k_file_name
```
This writes to `processed/` in a name that cannot collide with real experiment data.

**`k_phase1[k_params]` metadata dict** — add `'true_params_by_uuid'` and remove old UUID lists:
```python
k_phase1[k_params] = {
    'histories_k':          histories_k,
    'file_paths_k':         file_paths_k,
    'param_info_k':         param_info_k,
    'u_settings_k':         u_settings_k,
    'general_settings_k':   general_settings_k,
    'true_params_by_uuid':  true_params_by_uuid_k,   # NEW — replaces chooser/predictor uuid lists
    'n_players_k':          n_players_k,              # NEW — used in Phase 3 role loop
}
```
Delete `'chooser_uuids_for_k'` and `'predictor_uuids_for_k'` — they are no longer needed.

---

## Step 6: Phase 2 Update — `_fit_one_k`

The inner helper `_fit_one_k` currently builds `player_uuids_to_fit` by concatenating the
two UUID lists and passes it explicitly to `run_analysis_bayes`. With `player_type='synthetic'`
and auto-selection now working, we drop the explicit list entirely:

```python
def _fit_one_k(meta: dict) -> None:
    run_analysis_bayes(
        histories_data=meta['histories_k'],
        file_paths=meta['file_paths_k'],
        param_info=meta['param_info_k'],
        utility_settings=meta['u_settings_k'],
        general_settings=meta['general_settings_k'],
        "player_uuids=None — auto-selection via player_type='synthetic' in player_info"
    )
```

No other changes to Phase 2. The parallel/sequential execution logic, timing prints, and
`use_existing_fits` branch are all unchanged.

---

## Step 7: Phase 3 Update — `_collect_role_results` and role loop

Three concrete changes:

### 7a. `fit_dir`
```python
"Was: fit_dir = os.path.join(file_paths['player_fits'], f'experiment_{analysis_experiment_num}')"
fit_dir = os.path.join(file_paths['player_fits'], 'experiment_3')
```

### 7b. `_collect_role_results` — UUID prefix
```python
"Was: uuid_prefix = f'synthetic_{role}_{k_params}_'"
uuid_prefix = f"synthetic_{k_params}_"
```
All synthetic players share this prefix regardless of which role they happened to fill in a
given dyad. The fit files are named by player UUID, so the prefix uniquely selects the right
set of k=n files.

`true_params_chooser` and `true_params_predictor` are still present in each game dict
(written there by `create_simulated_dyad`), so the existing lookup inside `_collect_role_results`
continues to work without change:
```python
true_pred = first_game.get("true_params_predictor", {})
true_ch   = first_game.get("true_params_chooser",   {})
```

### 7c. Role loop — use `n_choosers` for both roles
Currently the loop is:
```python
for role, n_role in (("chooser", n_choosers), ("predictor", n_predictors)):
    if n_role == 0:
        continue
```
In the new design every player fills both roles, so:
```python
n_players_k = k_phase1[k_params]['n_players_k']
for role in ("chooser", "predictor"):
    corr_only, dyad_entries = _collect_role_results(k_params, role, param_info_k)
    ...
```
Remove the `n_role == 0` guard (or keep it trivially — `n_players_k` is always ≥ 4).

The rest of Phase 3 (aggregation, CSV write, Plotly figure) is unchanged.

---

## What Does NOT Change

- `create_simulated_dyad` — unchanged; still used by `create_simulated_data` / `run_simulation_recovery_analysis`
- `create_simulated_data` — unchanged
- `stable_bot_id` — unchanged
- `run_simulation_recovery_analysis` — unchanged
- All other analysis functions — unchanged

---

## `n_games` Semantics in New Design

`n_games` = the total game budget per player. For each batch of 4, three values (x,y,z summing
to n_games) determine how many games each of the 6 pairs plays. Each player participates in
exactly 3 dyads, and the game counts for those 3 dyads sum to n_games (by the row-sum constraint).

Example with n_games=120:
- Batch 1: AB=60, AC=40, AD=20, BC=20, BD=40, CD=60
- Each player's total: A=60+40+20=120, B=60+20+40=120, C=40+20+60=120, D=20+40+60=120

Each player acts as chooser in approximately 1.5 of their 3 dyads (depending on random role assignment).

---

## Implementation Order

1. Revert all 5 guard lines in [bayesian.py](bayesian.py) (restore original chooser-skip at line 2101)
2. Add `latin_square()` and `_sample_batch_game_counts()` to [simulation.py](simulation.py)
3. Add `create_simulated_experiment()` to [simulation.py](simulation.py)
4. **STOP HERE — print a sample JSON structure and review with Greg before continuing.**
5. Update `_assemble_histories_dict` to accept `player_type` and `true_params_by_uuid`
6. Rewrite Phase 1 of `run_param_recovery_by_k`
7. Update Phase 2 (`player_uuids=None`, `experiment_num=3` in `general_settings_k`)
8. Update Phase 3 (`fit_dir`, UUID prefix, `true_params_by_uuid` lookup)
9. Syntax-check [simulation.py](simulation.py) and [bayesian.py](bayesian.py)
10. Run with `n_choosers=4, n_games=120` as smoke test (1 batch of 4, smallest possible)
11. Verify 6 dyads in JSON, `player_type='synthetic'` in player_info, 4 fit files in `experiment_3/`
12. Run full `n_choosers=20, n_games=120`

---

## Verification Checklist

- [ ] bayesian.py: all 5 guard lines deleted, original chooser-skip restored
- [ ] latin_square(4) produces correct 4×4 matrix (each value once per row/column)
- [ ] _sample_batch_game_counts returns 6 pair game counts summing to n_games per player
- [ ] create_simulated_experiment() with n_players=4: 6 dyads, 4 UUIDs with `player_type='synthetic'`
- [ ] n_players not multiple of 4: padded correctly (no ValueError)
- [ ] Each player UUID appears in at least 1 chooser and at least 1 predictor dyad (approximate balance)
- [ ] true_params_by_uuid populated for all 4 UUIDs
- [ ] player_pairs_exper3 k-file written to processed/ directory
- [ ] run_analysis_bayes completes for experiment_num=3 on synthetic data
- [ ] Fit JSONs created for all synthetic players in player_fits/experiment_3/
- [ ] Phase 3 CSV shows correlations for both chooser and predictor roles
- [ ] No role-specific guards anywhere in bayesian.py
