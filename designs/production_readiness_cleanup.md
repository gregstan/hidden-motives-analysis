# Plan: Production-Readiness Cleanup

## Context

The audit from the previous session identified 16 TODOs/FIXMEs, stale documentation from the
`simulated_bot_uuids` → `stable_bot_id` refactor, and a new feature request (global random seed
control). This plan organizes everything into a new-feature section, four cleanup tiers, and a
future-work register. Decision items are explained in full so Greg can make an informed call
before any code is changed.

---

## New Feature: Global Random Seed Control

### What this means for the paper

With a single setting flipped to `True`, every analysis — AMPD distance matrix, model recovery
simulation, IC optimization, particle filter, parameter guessing, payoff sampling, model selection
— becomes reproducible to the last decimal place. This is standard for published computational
papers and reviewers increasingly expect it.

### Current state

The codebase already has *partial* seed support in three separate nested dicts:
- `general_settings['ampd_settings']['random_seed']` — controls AMPD Monte Carlo
- `general_settings['model_recovery_settings']['random_seed']` — controls model recovery
- `general_settings['optimization_policy']['dual_annealing_seed']` — controls optimizer

Individual functions also accept `random_seed: int | None = None` parameters in several places.
But there is no master on/off switch, and most functions use uncontrolled `random.*` and
`np.random.*` global state.

### Recommended design: single master seed

Use a single master seed rather than per-component seeds. Setting one number guarantees full
reproducibility; component-level control is not needed for a paper. Add to `general_settings`
in `config.py`:

```python
'random_seeds': {
    'use_seeds': False,   # False = standard non-deterministic behavior (current default)
    'seed':      42,      # Master seed; used only when use_seeds=True
},
```

Immediately after `general_settings` is defined in `config.py`, add enforcement:

```python
"If use_seeds=False, nullify the seed so all downstream code treats it as unseeded."
if not general_settings['random_seeds']['use_seeds']:
    general_settings['random_seeds']['seed'] = None
```

### Seed propagation

**Step 1 — `main.py` startup** (before any analysis runs):
```python
_master_seed = general_settings.get('random_seeds', {}).get('seed', None)
if _master_seed is not None:
    import random as _random_module
    _random_module.seed(_master_seed)
    np.random.seed(_master_seed)
    print(f"Reproducibility mode active. Master random seed: {_master_seed}.")
```

**Step 2 — propagate to the three existing nested seed locations** in `config.py` (so they all
draw from the same master):
```python
general_settings['ampd_settings']['random_seed']             = general_settings['random_seeds']['seed']
general_settings['model_recovery_settings']['random_seed']   = general_settings['random_seeds']['seed']
general_settings['optimization_policy']['dual_annealing_seed'] = general_settings['random_seeds']['seed']
```

**Step 3 — functions that accept `random_seed=None`**: No changes needed. When the global
`np.random` and `random` state is seeded in Step 1, and the three nested seed locations are
propagated in Step 2, those functions automatically inherit deterministic behavior. The local
`random_seed` parameter remains as a per-call override.

**Step 4 — `uuid.uuid4()` calls** in `simulation.py`: These are used for chooser IDs (never
looked up by filename) and are not reproducibility-relevant. Leave them as-is.

### Key randomness locations covered by the master seed

| File | Location | What it controls |
|------|-----------|-----------------|
| `main.py` startup | global `random.seed` + `np.random.seed` | All uncontrolled `random.*` and `np.random.*` calls |
| `config.py` propagation | `dual_annealing_seed` | Optimization global/local search |
| `config.py` propagation | `ampd_settings['random_seed']` | AMPD behavioral-distance Monte Carlo |
| `config.py` propagation | `model_recovery_settings['random_seed']` | Model recovery simulation |
| `simulation.py:127–130` | payoff sampling | `random.randint(1, 5)` per game |
| `simulation.py:354–362` | parameter grid | `random.uniform(...)` per parameter range |
| `bayesian.py:324` | MCMC | Metropolis-Hastings acceptance draw |
| `bayesian.py:627, 797` | grid/prior downsampling | `random.sample(...)` |
| `analysis.py:10783–10784` | model recovery per-agent | `random.seed(master + agent_idx * 1000)` — already offset-based, will work with master seed |
| `utilities.py:1948–2032` | model selection | `_random.sample`, `_random.choice`, `_random.shuffle` |

### Files modified

- `config.py`: Add `random_seeds` nested dict inside `general_settings`; add enforcement block; propagate to three nested seed locations.
- `main.py`: Add seeding block at the top of `main()`.
- No other files need changes — global seed state covers them.

---

## Tier 1 — Trivial Documentation Fixes

### 1A. `simulation.py` — `get_simulated_dyad` docstring (lines ~516–540)

Three sentences still say "`simulated_bot_uuids(...)`". Replace all three with `stable_bot_id`.
Also correct the sentence about `params_chooser` — it is no longer used for filename
reconstruction; only `params_predictor` is needed.

**File:** `simulation.py` | **Action:** Rewrite three docstring sentences.

---

### 1B. `AGENTS.md` module table (line ~572)

The `simulation.py` row lists `simulated_bot_uuids` as a key function. Replace with `stable_bot_id`.

**File:** `AGENTS.md` | **Action:** One-word substitution.

---

## New Item: `softmax_temperature` Naming Standardization

### What the problem is

The softmax temperature parameter appears under multiple names in function signatures and local
variables across the repo: `choice_temperature`, `temp`, `temperature`, and `softmax_temperature`.
The parameter *key* in param dicts is `'τ'` (Greek letter, intentional), but the *argument name*
in function calls is inconsistent.

### Action

Grep for `choice_temperature` across all `.py` files. Rename every occurrence as a function
argument name or local variable to `softmax_temperature`. Do NOT rename:
- The dict key `'τ'` — that is the model-level parameter name and part of the IC model spec.
- The general_settings key `'softmax_temperature'` — that already uses the correct name.

Key files expected to have `choice_temperature`: `bayesian.py` (the game loop at ~line 1155 and
surrounding context). May also appear in `simulation.py` and `visualization.py`.

**Files to modify:** `bayesian.py`, `simulation.py`, `visualization.py`, `analysis.py` (anywhere
`choice_temperature` appears as a variable or argument name).

---

## Tier 2 — Trivial Code Cleanup (safe to delete, no decisions needed)

### 2A. `preprocessing.py:325` — `#HACK REMOVE AFTER DEMO`

```python
demo_mode = True  #HACK REMOVE AFTER DEMO
if demo_mode:
    histories_file_path = os.path.join(ROOT, 'demo_files', 'processed', ...)
else:
    histories_file_path = os.path.join(ROOT, 'processed', ...)   # never reached
```

Delete the `demo_mode = True` line and the `if/else`. Keep only the `else` branch path
(the non-demo path that loads from `processed/`).

---

### 2B. `utilities.py:2859` — dead `random_guesses_are_unique` comment

```python
# random_guesses_are_unique=not general_settings.get('run_in_parallel', True),  # not used TODO delete???
```

The comment says it is not used. Delete the line.

---

### 2C. `simulation.py:2696` — commented-out `segment_size = 1` override

```python
# TODO Figure out why this is here.
# segment_size = 1
# if n_games_in_dyad < 8:
```

A workaround for small dyads that was removed. Delete all three commented lines and the TODO.

---

### 2D. `model.py:693` — orphan comment with typo

```python
"if loss_av or (la_socc and term_type == 'social_comparison'):" # TODO figure out if this line should be delted
```

This is a commented-out simpler version of the condition on the line above. The current (more
complex) condition is correct: the extra clause `not (term_type == "self-interest" and fix_self)`
is necessary when `fix_self_interest_parameter=True`. Delete this string-comment and the TODO
typo. No logic change.

---

## Tier 3 — Deferred Decisions with a Clear Recommended Answer

### 3A. `utilities.py:1374` — legacy compatibility fallback path

**What it is:** An `else` branch in `generate_utility_settings` that catches the case where
`processed/all_utility_functions.csv` is missing. It warns and falls back to live generation.
The TODO says: "Delete after `all_utility_functions.csv` is proven to replace the three legacy
JSON files."

**Recommendation:** Keep permanently as defensive code. A fallback that prints a clear warning
and re-generates from scratch is harmless and useful for any future environment where the CSV
is missing. Reword the TODO to: "Intentional fallback — keep permanently."

**Action:** Replace the TODO string with a comment marking this as intentional defensive code.

---

### 3B. `bayesian.py:332` — MCMC acceptance rate: log or discard?

**What it is:** After the MCMC chain runs, `acceptance_rate = accepted_count / float(chain_length)`
is computed but never used. Optimal MCMC acceptance is ~0.23–0.44; values outside this range
indicate poor proposal calibration.

**Recommendation:** Print it conditionally under `general_settings.get('verbose', False)`.
No new parameter needed.

**Action:** Add:
```python
if general_settings.get('verbose', False):
    print(f"  MCMC acceptance rate: {acceptance_rate:.3f}")
```
Remove the TODO.

---

### 3C. `bayesian.py:576` — covariance validation order

**What it is:** PSD repair runs before shape validation. A wrongly-shaped matrix gets "repaired"
and then fails a shape check anyway.

**Recommendation:** Move shape validation before PSD repair (fail-fast on obviously malformed
input).

**Action:** Swap the two checks. Remove the TODO.

---

### 3D. `bayesian.py:2835` — multiprocessing computed settings are ignored

**What it is:** `n_workers`, `chunksize`, and `maxtasksperchild` are computed from
`general_settings` but the actual `mp.Pool` call uses `mp.cpu_count() - 1` and default chunking.

**Recommendation:** Pass the computed values. The spawn-context question (platform isolation)
is a separate concern and can be deferred.

**Action:** Update `mp.Pool(...)` and `imap_unordered(...)` to pass `processes=n_workers`,
`maxtasksperchild=maxtasksperchild`, and `chunksize=chunksize`. Replace the TODO with a
one-line comment: "Spawn context and worker initializer left as future improvement."

---

### 3E. `utilities.py:1155` — social-comparison + single-payoffs constraint: **check result first**

**Finding from the audit:** The constraint is currently commented out. The 480 valid models
are enumerated WITHOUT this constraint, meaning models with BOTH `include_social_comparison=True`
AND `single_payoffs_not_differences=True` currently pass validation and may be included.

**The semantic issue:** Social comparison computes Vᵢⱼ × (Aᵢ − Aⱼ) — the comparison term is
inherently a *difference*. If `single_payoffs_not_differences=True`, the model uses Vᵢⱼ × Aᵢ
instead, which is not a comparison at all; it is just an altruism term. The two flags are
semantically incompatible.

**Before deciding:** Run a one-off diagnostic in `generate_utility_settings` to count how many
of the 480 models have both flags True. This determines the stakes:
- If zero models have both: the constraint is redundant and the commented lines can be deleted.
- If some models have both: uncommenting the constraint reduces the valid set and could affect
  IC results (models fitted to data using this combination would become invalid).

**Action:** Add a temporary assertion and run it. Report the count to Greg before uncommenting
or deleting.

---

### 3F. `analysis.py:3327` — `conditional_welfare_mode` nesting branch (**diagnostic first**)

**What it is:** A commented-out `elif` in `map_child_to_parent_special_param_info` for the case
where a parent model gains `conditional_welfare_mode`. The logic would set Ʌᵢᵢ = Vᵢᵢ and
Ʌᵢⱼ = Vᵢⱼ to make the parent's "ahead" and "behind" branches behave identically — replicating
the child's behavior, which has no such branching.

**Greg's concern:** The nesting code was laborious to get right. Any change here risks
introducing a subtle bug that the smoketest might not catch.

**Before deciding:** Run two diagnostics and report results:

1. **Does `conditional_welfare_mode` ever appear as `changed_utility_setting`?**
   Call `model_nesting_adjacency_matrices(...)` or inspect the nesting adjacency CSV and check
   whether any parent/child edge has `conditional_welfare_mode` as the differing flag. If zero
   edges have it, the branch is structurally dead and can be deleted safely.

2. **If it does appear:** Run `run_child_parent_probability_equivalence_smoketest` with the
   branch *still commented out* to see whether there are currently failing cases. Then uncomment,
   re-run, and compare. If the smoketest was already passing without the branch, the existing
   code handles this case correctly by some other path.

**Action:** Run the diagnostics. Report the findings. Do not uncomment until the results
are reviewed.

---

### 3G. `bayesian.py:2785` — experiment 0 predictor UUID inclusion

**What it is:** When auto-generating player lists, experiment 0 additionally includes UUIDs
containing `'predictor'`. This is needed for the parameter-recovery simulation, which specifically
fits the predictor bot's parameters to check recovery.

**Recommendation:** Keep as-is. The logic is correct. Add a comment explaining why.

**Action:** Add a one-line comment: "Experiment 0 is the simulation study; predictor-bot UUIDs
are included so the optimizer can attempt to recover their known ground-truth parameters."
Remove the TODO.

---

## Tier 4 — Deferred Decisions Requiring Researcher Input

### 4A. `bayesian.py:250` — Jacobian-corrected posterior → **DELETE** (decided)

Delete the dead `log_posterior_(...)` function. The active `log_posterior_unif(...)` is used
everywhere; the Jacobian version is dead code. MCMC is not the primary fitting method so the
theoretical upside does not justify the maintenance cost.

**Action:** Delete `log_posterior_(...)` in its entirety. Remove the TODO comment.

---

### 4B. `bayesian.py:1155` — predictor vs. chooser temperature → **keep symmetric** (decided)

Both roles use the same `softmax_temperature`. Remove the TODO and add a one-line comment:
"Predictor and chooser intentionally share the same temperature — asymmetric temperatures
would add a parameter without a clear theoretical motivation."

This item is also covered by the `softmax_temperature` naming standardization above.

---

### 4C. `bayesian.py:1734` — chooser std-dev cleanup → **keep** (decided)

Keep the `_std` parameter deletion for the chooser. In the UBM, the predictor holds a
distribution of beliefs; the chooser just has parameters. Fitting std-devs for the chooser
conflates the model's internal representation with optimizer uncertainty.

**Action:** Replace the date-stamped TODO comment with a docstring explanation:
"Chooser parameters do not include prior standard deviations. In the UBM, standard deviations
describe the width of the *predictor's* belief distribution about the chooser's parameters;
the chooser has no analogous internal uncertainty representation."

---

### 4D. `visualization.py:224` — temperature dampening → **Option D** (decided)

Add a `temperature_scale: float = 0.75` parameter to the function. The current behavior
(multiply by 0.75) becomes the default, but callers can pass 1.0 for a statistically faithful
heatmap. Remove the TODO; replace with a docstring line explaining what the parameter does.

---

### 4E. `visualization.py:447` — observation-phase choice marker → **defer with TODO** (decided)

Replace the current bare TODO comment with a richer one that documents the issue for a future
implementer:

```python
# TODO (deferred): For experiments 1 and 2, games alternate between an observation phase
# (op: choice populated, prediction=None) and a response phase (rp: prediction populated,
# choice=None). The current filter includes both phases. The correct fix is to filter to
# op games only — either by checking `game.get('choice') is not None` or by adding an
# explicit `'phase'` key in preprocessing.py. Deferred because the belief-update figure
# is not currently used for experiments 1 or 2 in the paper.
```

No code changes to the filter itself.

---

## New Item: Move Visualization Functions from `analysis.py` to `visualization.py`

### Background

`analysis.py` contains plotting functions that conceptually belong in `visualization.py`. The
import chain is `visualization.py → simulation → bayesian → ...` and
`analysis.py → visualization → ...`. This means `visualization.py` cannot import from
`analysis.py` (circular dependency), but `analysis.py` can freely import from `visualization.py`.

Moving plot functions from `analysis.py` *to* `visualization.py` is safe only if the functions
being moved do not call other functions defined in `analysis.py`. Functions that compute analysis
results *and* plot them (mixed compute+plot) cannot move as-is; only pure-plot functions that
receive a pre-computed DataFrame or dict can move cleanly.

### Action (investigate before implementing)

Grep `analysis.py` for all functions whose names begin with `plot_`. For each, determine:
1. Does it call any other `analysis.py` function internally (compute dependency)?
2. Does it only take pre-computed data structures (DataFrames, dicts) as input?

Functions satisfying condition 2 are candidates to move. Functions failing condition 1 stay in
`analysis.py`. A helper that is called by both an analysis function and a plot function stays
at module level in `analysis.py` and is imported from there.

**Files modified:** `analysis.py` (remove moved functions), `visualization.py` (add them),
`main.py` (no change if functions are re-exported via `*` imports).

**Flag:** This is a reorganization, not a logic change. Scope to pure-plot functions only.
Do not move mixed compute+plot functions — that refactor requires splitting them first.

---

## Future Work Register (not for this cleanup, but worth tracking)

These items came up in the audit and are worth documenting but are out of scope for now:

### FW-1. Working covariance parameters in the UBM

Covariance between parameters (e.g., Vᵢᵢ and Vᵢⱼ jointly) has been attempted and has caused
significant difficulty. The current model treats parameters as independent in the prior grid.
A working covariance implementation would require a full multivariate prior, Cholesky
factorization for PSD enforcement, and careful handling of the covariance matrix shape as the
utility architecture changes across the 480 models. High potential benefit but significant
implementation cost.

### FW-2. Hierarchical Bayesian model

Currently each participant is fitted independently. A hierarchical model would share statistical
strength across participants by fitting a population distribution over parameters simultaneously
with individual-level estimates. This would improve estimates for participants with sparse data
and produce principled uncertainty estimates at the population level. Implementation would require
Stan, PyMC, or a custom variational inference approach — significant effort, but high scientific
value if the paper extends to a journal submission.

---

## Execution Order

Follow this order to minimize risk:

1. **`softmax_temperature` rename** — search-and-replace, no logic changes, do first so later edits use the correct name
2. **Random seeds** — touch only `config.py` and top of `main.py`
3. **Tier 1 doc fixes** — zero risk
4. **Tier 2 deletes** — zero risk
5. **Tier 3A–3D, 3G, and decided 4A–4E** — no logic changes, or safe isolated changes
6. **3E diagnostic** — run and report the count of models with both `include_social_comparison` and `single_payoffs_not_differences` True before any code change
7. **3F diagnostic** — run nesting adjacency check and smoketest; report before uncommenting anything
8. **Visualization move** — last, after the rest is stable; pure-plot functions only

## Summary of All Decisions Made

| Item | Decision |
|------|----------|
| Random seeds | Add `random_seeds` dict to `general_settings`; single master seed; propagate to three existing nested locations |
| `softmax_temperature` | Standardize name across the repo; do not rename the `'τ'` dict key |
| 1A sim.py docstring | Fix stale `simulated_bot_uuids` references |
| 1B AGENTS.md table | Replace `simulated_bot_uuids` with `stable_bot_id` |
| 2A demo_mode hack | Delete; keep only the non-demo path |
| 2B dead comment | Delete commented `random_guesses_are_unique` line |
| 2C segment_size | Delete three commented lines |
| 2D model.py comment | Delete the simplified-condition string comment |
| 3A legacy fallback | Reword TODO to "intentional — keep" |
| 3B MCMC acceptance | Print under `verbose` flag |
| 3C covariance order | Move shape validation before PSD repair |
| 3D mp.Pool settings | Use computed `n_workers`, `chunksize`, `maxtasksperchild` |
| 3E social comparison | Run diagnostic first; do not change code yet |
| 3F nesting branch | Run diagnostic + smoketest first; do not uncomment yet |
| 3G exp-0 predictor UUIDs | Keep; add explanatory comment |
| 4A Jacobian posterior | **Delete** the dead `log_posterior_` function |
| 4B predictor temperature | **Keep symmetric**; add comment; covered by naming rename |
| 4C chooser std-devs | **Keep** cleanup; replace date-stamp TODO with docstring explanation |
| 4D temperature dampening | **Option D**: add `temperature_scale: float = 0.75` parameter |
| 4E exp 1/2 choice marker | **Defer**: replace TODO with rich documentation comment |
| Visualization move | Investigate pure-plot functions; move only those with no analysis.py call dependency |

## Verification

- `python -c "import ast; ast.parse(open('simulation.py').read()); print('OK')"` after each file edit
- `python -c "import ast; ast.parse(open('bayesian.py').read()); print('OK')"` after each file edit
- Run `run_child_parent_probability_equivalence_smoketest(...)` after 3F if branch is changed
- Run `quick_demo.py` end-to-end to confirm no regressions
- Confirm experiment 0 data still loads after removing `preprocessing.py` demo_mode hack
- Toggle `random_seeds['use_seeds'] = True` and run `quick_demo.py` twice; outputs must be identical
