# Plan: Unify Simulation Functions via `_simulate_pair_games` Helper + Rename `choice` → `response`

## Context and Motivation

Three related problems motivate this plan:

1. **`choice()` is misnamed.** It lives in `model.py` but is called for both chooser and predictor
   roles throughout the codebase. It computes a softmax response probability or, when
   `select=True`, samples a binary "A"/"B" response. The name implies only chooser action. A more
   neutral name (`response()`) fits both roles without confusion.

2. **`select` is ambiguous.** The parameter name `select` in `choice()` and `agent()` doesn't
   convey what is being selected. `select_responses` is more explicit.

3. **`create_simulated_dyad` and `create_simulated_experiment` duplicate logic.** Both functions
   do the same 4-step per-pair pattern: build game dicts → generate choices → run UBM →
   serialize + relabel. The only structural difference is that `create_simulated_dyad` uses fixed
   roles (player 1 always chooser) while `create_simulated_experiment` uses a per-round coin flip.
   Factoring the shared logic into a private `_simulate_pair_games` helper eliminates the
   duplication and makes `create_simulated_dyad` extensible to the role-flip case.

   As a byproduct, `create_simulated_dyad` gets the `per_round_role_flip` capability needed for
   the `run_param_recovery_by_k` pipeline, and `dyad_id` (unused outside the function) is
   replaced with clean `player_1_uuid`/`player_2_uuid` parameters.

---

## Files to Modify

- [model.py](model.py) — rename `choice()` → `response()`, `select` → `select_responses`
- [bayesian.py](bayesian.py) — rename `select` → `select_responses` in `agent()`, add
  `select_responses` to `simulate_dyad()`; update internal `choice()` → `response()` call sites
- [simulation.py](simulation.py) — new `_simulate_pair_games` helper; refactor
  `create_simulated_dyad`; update `create_simulated_experiment`; update `create_simulated_data`
- [optimization.py](optimization.py) — rename call site only
- [visualization.py](visualization.py) — rename call site only (2 sites)
- [analysis.py](analysis.py) — rename call site only (2 sites)
- [mle.py](mle.py) — rename call site only
- [architecture.py](architecture.py) — update `dyad_id=` call to `player_1_uuid=`/`player_2_uuid=`

---

## Step 1: Rename `choice()` → `response()` and `select` → `select_responses`

### 1a. `model.py:431`
Change the function definition signature only — no body changes:
```python
"Before:"
def choice(current_game, agent_params, utility_settings,
           softmax_temperature=1.5, select: bool = False, ...):

"After:"
def response(current_game, agent_params, utility_settings,
             softmax_temperature=1.5, select_responses: bool = False, ...):
```
Inside the body, rename `select` → `select_responses` at the two places it appears (the docstring
reference and the `if select:` guard at line ~495).

### 1b. `bayesian.py` — `agent()` signature + internal call
Signature (line 927): rename `select: bool = False` → `select_responses: bool = False`.

Internal call (line ~1106–1112):
```python
"Before:"
choice_output = choice(current_game=game_dict, agent_params=..., ..., select=select)
"After:"
choice_output = response(current_game=game_dict, agent_params=..., ..., select_responses=select_responses)
```
Lines ~1120 and ~1126 reference `if select:` — rename to `if select_responses:`.

### 1c. `bayesian.py` — `simulate_dyad()` signature + internal calls
Add `select_responses: bool = False` to the signature (line 1351).
Pass it through to both `agent()` calls inside the game loop (line ~1405):
```python
dyad_games = agent(dyad_games=dyad_games, game_idx_start=meeting_idx, game_idx_stop=meeting_idx,
                   initial_params=player_params[player_uuid], param_info=param_info,
                   utility_settings=utility_settings, player_uuid=player_uuid,
                   general_settings=general_settings, select_responses=select_responses)
```

### 1d. Remaining `choice()` call sites (pure rename, no logic change)

| File | Line(s) | Change |
|------|---------|--------|
| bayesian.py | ~71, ~214, ~712, ~758, ~856 | `choice(` → `response(` (all pass `select=False` explicitly → rename to `select_responses=False`) |
| simulation.py | ~139, ~147, ~1742, ~2489 | `choice(` → `response(`, rename `select=True` → `select_responses=True` |
| optimization.py | ~1069 | `choice(` → `response(` |
| visualization.py | ~229, ~615 | `choice(` → `response(` |
| analysis.py | ~3359 | `choice(` → `response(` |
| mle.py | ~307 | `choice(` → `response(` |
| typological.py | ~601 | `choice(` → `response(` (implicit `select=False`, no kwarg to rename) |

All 16 call sites. No callers change any logic — the rename is purely mechanical.

---

## Step 2: Extract `_simulate_pair_games` Helper

Add a new **module-level private helper** in `simulation.py`, placed immediately before
`create_simulated_dyad`. This function encapsulates the shared per-pair simulation logic.

### Design

**Core approach** (avoids the ordering problem where a predictor tries to observe a choice that
hasn't been generated yet):
1. Build game dicts with payoffs and role assignments (`choice=None`, `prediction=None`).
2. Pre-generate all choices in one pass: for each game, call `response(select_responses=True)` for
   whoever is chooser that round and write `game["choice"]`.
3. Run predictor UBM for player 1: `agent(..., player_role='predictor', select_responses=True)`.
   Skips rounds where player 1 is not predictor. Writes predictions into `game["prediction"]`.
4. Run predictor UBM for player 2: same for player 2.
5. `serialize_or_drop_param_vectors(drop_grids=True)` + relabel `update_method` key → `'sim_pred'`.

Step 2 pre-generates choices before any UBM run, ensuring that when either player runs their
predictor UBM (step 3/4), every `game["choice"]` is already populated.

### Signature

```python
def _simulate_pair_games(
    n_games: int,
    params_player_1: dict[str, float],
    params_player_2: dict[str, float],
    uuid_player_1: str,
    uuid_player_2: str,
    utility_settings: UtilitySettings,
    general_settings: GeneralSettings,
    param_info: ParamInfo,
    per_round_role_flip: bool = False,
    matching_probability: float = 1.0,
    dynamic_predictor: bool = True,
    payoff_structures: list[dict] | None = None,
    random_gen: np.random.Generator | None = None,
) -> list[dict]:
```

Returns the raw `games_list` (no dyad-key wrapper). Callers wrap it.

### Payoff generation
When `random_gen` is provided, use `int(random_gen.integers(1, 6))`. Otherwise fall back to
`random.randint(1, 5)` (preserves current behavior for callers that don't pass `random_gen`).

### Role assignment
- `per_round_role_flip=False`: player_1 is chooser in ALL rounds, player_2 is predictor.
- `per_round_role_flip=True`: per-round coin flip using
  `random_gen.random() < 0.5` (or `random.random() < 0.5` fallback). Coin determines who is
  chooser vs. predictor for that round.

### `dynamic_predictor=False` path
Skip steps 3–4 (no UBM). Instead, generate predictions via `response(select_responses=True)` for
each round's predictor (same as the current `create_simulated_dyad` dynamic_predictor=False path).
No serialize/relabel needed (no grid data produced).

### True params embedding
Write `true_params_chooser` and `true_params_predictor` into `games_list[0]` based on whoever is
actually chooser/predictor in round 0.

---

## Step 3: Refactor `create_simulated_dyad`

### New signature

```python
def create_simulated_dyad(
    n_games: int,
    params_chooser: dict[str, float],          # player 1 params (always chooser when per_round_role_flip=False)
    params_predictor: dict[str, float],        # player 2 params (always predictor when per_round_role_flip=False)
    general_settings: GeneralSettings,
    utility_settings: UtilitySettings,
    param_bds: ParamBounds,
    payoff_structures: list[dict] | None = None,
    default_utility_settings: bool = True,
    dynamic_predictor: bool = True,
    player_1_uuid: str | None = None,          # replaces dyad_id
    player_2_uuid: str | None = None,          # replaces dyad_id
    matching_probability: float = 1.0,
    per_round_role_flip: bool = False,
    random_gen: np.random.Generator | None = None,
) -> dict[DyadKey, DyadGames]:
```

**`dyad_id` is removed entirely.** The old `dyad_id` logic produced
`f"{dyad_id}_chooser"` / `f"{dyad_id}_predictor"`. Callers that relied on this (only
`architecture.py`) should instead pass `player_1_uuid` / `player_2_uuid` directly.

**UUID defaults:**
```python
player_1_uuid_resolved = player_1_uuid if player_1_uuid is not None else f'synthetic_player_1_{uuid.uuid4().hex[:12]}'
player_2_uuid_resolved = player_2_uuid if player_2_uuid is not None else f'synthetic_player_2_{uuid.uuid4().hex[:12]}'
```
`stable_bot_id` is no longer called inside `create_simulated_dyad`. Callers that need
content-addressed predictor UUIDs (i.e. `create_simulated_data` and `get_simulated_dyad`) pass
`player_2_uuid=stable_bot_id(...)` explicitly.

**Body:** After resolving UUIDs and utility settings, delegate to `_simulate_pair_games(...)` and
wrap the result:
```python
games_list = _simulate_pair_games(
    n_games=n_games,
    params_player_1=params_chooser,
    params_player_2=params_predictor,
    uuid_player_1=player_1_uuid_resolved,
    uuid_player_2=player_2_uuid_resolved,
    utility_settings=utility_settings_,
    general_settings=general_settings,
    param_info=param_info_,
    per_round_role_flip=per_round_role_flip,
    matching_probability=matching_probability,
    dynamic_predictor=dynamic_predictor,
    payoff_structures=payoff_structures,
    random_gen=random_gen,
)
dyad_key = f"({games_list[0]['predictor']}, {games_list[0]['chooser']})"
return {dyad_key: games_list}
```

`param_info_` is built inside `create_simulated_dyad` via `make_param_info(...)` (same as now).

**Docstring update:** Note that `params_chooser` / `params_predictor` refer to player 1 / player 2
parameters and only correspond to fixed roles when `per_round_role_flip=False`.

---

## Step 4: Update Callers of `create_simulated_dyad`

### `create_simulated_data` (simulation.py, ~line 394)
Pass `player_2_uuid` explicitly so the predictor UUID remains content-addressed:
```python
player_dyad = create_simulated_dyad(
    ...
    player_2_uuid=stable_bot_id(params=params_predictor, player_role='predictor', n_games=n_games),
)
```
No other changes needed — `create_simulated_data` does not pass `dyad_id` today.

### `architecture.py::compute_model_recovery_simulation` (~line 1868)
Replace `dyad_id=f"synthetic_agent_{agent_idx}"` with explicit UUID arguments:
```python
player_dyad = create_simulated_dyad(
    ...
    player_1_uuid=f"synthetic_agent_{agent_idx}_chooser",
    player_2_uuid=f"synthetic_agent_{agent_idx}_predictor",
)
```
This preserves the UUID format that `_build_synthetic_histories_json` reconstructs from `agent_idx`.

### `deprecated.py::compute_recovery_fit_worker_model_recovery` (~line 1765)
Does not pass `dyad_id`; only uses `next(iter(dyad_data.values()))`. No change required.

---

## Step 5: Update `create_simulated_experiment`

Replace the inline per-pair game-generation loop (~lines 1707–1803) with a call to
`_simulate_pair_games`:

```python
for (local_idx_a, local_idx_b), n_games_pair in pair_game_counts.items():
    uuid_a   = f"synthetic_{k_params}_{(batch_start + local_idx_a):04d}"
    uuid_b   = f"synthetic_{k_params}_{(batch_start + local_idx_b):04d}"
    params_a = batch_params[local_idx_a]
    params_b = batch_params[local_idx_b]
    matching_probability = n_games_pair / n_games

    games_list = _simulate_pair_games(
        n_games=n_games_pair,
        params_player_1=params_a,
        params_player_2=params_b,
        uuid_player_1=uuid_a,
        uuid_player_2=uuid_b,
        utility_settings=utility_settings_k,
        general_settings=general_settings_for_ubm,
        param_info=param_info_for_ubm,
        per_round_role_flip=True,
        matching_probability=matching_probability,
        dynamic_predictor=True,
        random_gen=random_gen,
    )
    dyad_key = f"({games_list[0]['predictor']}, {games_list[0]['chooser']})"
    dyad_list.append({dyad_key: games_list})
```

The n_bins cap logic (`general_settings_for_ubm`) and `param_info_for_ubm` remain at the
batch-loop level as they are today (built once, reused across pairs). They are passed into
`_simulate_pair_games` as arguments.

---

## What Does NOT Change

- `create_simulated_data` overall flow — only the `player_2_uuid` addition at the call site
- `get_simulated_dyad` — still calls `stable_bot_id` directly (not via `create_simulated_dyad`)
- `load_simulated_fits_from_json`, `compute_param_recovery_correlations`, `run_simulation_recovery_analysis`
- `run_param_recovery_by_k` (Phase 1/2/3 logic unchanged — it calls `create_simulated_experiment`,
  which internally changes but retains the same external contract)
- `stable_bot_id` function — unchanged, still used by `get_simulated_dyad` and passed by callers
  that need deterministic predictor UUIDs

---

## Implementation Order

1. Part 1: Rename `choice()` → `response()` and `select` → `select_responses` everywhere (mechanical)
2. Add `select_responses` to `simulate_dyad()` (3 lines)
3. Write `_simulate_pair_games` helper in simulation.py
4. Refactor `create_simulated_dyad` to delegate to `_simulate_pair_games`
5. Update `create_simulated_data` (`player_2_uuid=stable_bot_id(...)`)
6. Update `architecture.py` (`dyad_id=` → `player_1_uuid=`/`player_2_uuid=`)
7. Update `create_simulated_experiment` inner loop to use `_simulate_pair_games`
8. Syntax-check all modified files (`python -m py_compile <file>`)
9. Smoke test: `create_simulated_dyad(n_games=10, ..., per_round_role_flip=False)` — verify dyad key, true_params embedded, predictions written
10. Smoke test: `create_simulated_experiment(n_players=4, n_games=30, ...)` — verify 6 dyads, mixed roles per round, predictions populated
11. Full test: `run_param_recovery_by_k(n_players=8, n_games=30, k_params_range=(2,3))` — verify fit files in `simulations/experiment_3/`, Phase 3 CSV produced

---

## Verification Checklist

- [ ] `python -m py_compile model.py bayesian.py simulation.py optimization.py visualization.py analysis.py mle.py architecture.py`
- [ ] `response()` with `select_responses=False` returns float in [0, 1] (unchanged behavior)
- [ ] `response()` with `select_responses=True` returns int 0 or 1 (unchanged behavior)
- [ ] `create_simulated_dyad(..., per_round_role_flip=False)`: single dyad key, `game[0]["true_params_chooser"]` set, `game[0]["prediction"]` is "A" or "B" (not None)
- [ ] `create_simulated_dyad(..., per_round_role_flip=True)`: chooser/predictor fields vary across rounds
- [ ] `create_simulated_data` still produces valid dyad histories (quick sanity run)
- [ ] `create_simulated_experiment(n_players=4, ...)`: 6 dyads, 4 UUIDs, `player_type='synthetic'`
- [ ] `run_param_recovery_by_k(n_players=8, n_games=30, k_params_range=(2,3))`: completes without crash, fit files in `simulations/`
