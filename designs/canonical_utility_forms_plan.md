# Plan: Add `include_welfare_efficiency_term` and `include_relative_income_penalty` Flags

## Context and Motivation

This paper conducts the largest model comparison to date in experimental economics for binary
Dictator Games. The central argument is that the field has relied almost unquestioningly on a
handful of hand-me-down utility models from the 1990s and early 2000s — Fehr-Schmidt (1999),
Bolton-Ockenfels (2000), Charness-Rabin (2002), Andreoni-Miller (2002), Engelmann-Strobel (2004)
— treating them as canonical despite the vast untested space of psychologically plausible utility
functional forms. The paper's scientific contribution is to survey that entire space systematically
rather than cherry-picking forms that seemed intuitive to their authors at the time.

A crucial part of this argument is that **all the canonical forms the field actually uses must
themselves be points in the model space we test**. If they are not, a reviewer can correctly
object that we omitted the very models we claim to supersede. Currently, Engelmann-Strobel (2004)
and Bolton-Ockenfels ERC (2000) cannot be reached within the existing 480-model IC pipeline
because the flag system lacks the structural knobs for their functional forms. Adding two new
flags resolves this omission and strengthens the paper's core claim: not only is our model space
vast, but it provably contains all the models researchers have historically relied on.

Ultimately this contributes to a more nuanced and comprehensive understanding of moral psychology.
Utility functions of this kind are a critical component of almost any computational model of moral
psychology, and our ability to adjudicate among them rigorously is what gives the paper its value.

**Goal:** Add two new flags. Preserve all 480 existing models exactly. Every existing model must
produce the same equation, the same parameter keys, the same IC scores, and the same nesting
relationships as before.

---

## Critical Pre-Change Backups

Before touching any code, copy the following files to `_baseline` versions so post-change
results can be compared exactly:

| File | Location |
|---|---|
| `child_parent_embedding_sanity_checks-grid-chooser-1.csv` | `demo_files/bic_aic/` |
| `child_parent_prob_equivalence.csv` | `demo_files/processed/` |
| `all_utility_functions.csv` | `demo_files/processed/` |
| `model_nesting_data.json` | `demo_files/processed/` |
| `model_distance_ampd__n_models=480*.csv` | `demo_files/processed/` (any AMPD matrix file) |

Then run:
```python
run_child_parent_embedding_sanity_checks(
    general_settings=general_settings, file_paths=file_paths,
    param_bds=param_bds, utility_settings=utility_settings,
    player_role_to_fit="chooser", fit_for_n_players=1,
    random_seed=20250406, numeric_tolerance=1e-3, verbose=True,
    csv_file_name="child_parent_embedding_sanity_checks_baseline.csv",
)
run_child_parent_probability_equivalence_smoketest(
    utility_settings=utility_settings, file_paths=file_paths,
    param_bds=param_bds, rand_payoff_idx=True, n_trials=12,
    rng_seed=None, tolerance=1e-12, verbose=True,
)
```

The `all_utility_functions.csv` sanity check is particularly important: after the code changes,
regenerating this file with `create_new_file=True` should produce a CSV that is a strict superset
of the old one — same rows in the same order for the 480 existing models, two new boolean columns
added (`include_welfare_efficiency_term`, `include_relative_income_penalty`, both `False` for all
480 original rows), and new rows for the new models appended at the end. If the 480-row portion
matches the baseline exactly, the existing model space is confirmed intact.

---

## New Flag Semantics

### Flag 1: `include_welfare_efficiency_term`

Placed at index 3 in `ordered_keys` (between `min_max_rawlsian_leontief` and
`use_exponential_parameters`). Produces forms like:

- Basic: `Uᵢ(A) = (1−Vᵢᵢ)×πᵢᴬ + Vᵢᵢ×(πᵢᴬ+πⱼᴬ)/2`
- With exponent: `Uᵢ(A) = (1−Vᵢᵢ)×(πᵢᴬ)^γ₁ + Vᵢᵢ×((πᵢᴬ+πⱼᴬ)/2)^γ₁`
- Full E&S (with `include_social_comparison=True`):
  `Uᵢ(A) = (1−Vᵢᵢ−Vᵢⱼ)×πᵢᴬ + Vᵢᵢ×(πᵢᴬ+πⱼᴬ)/2 + Vᵢⱼ×min(πᵢᴬ, πⱼᴬ)`

Vᵢᵢ is **remapped** from optimizer range [-1,1] to [0,1] inside `utility()`:
`Vᵢᵢ_used = (Vᵢᵢ + 1) / 2`. Bounds and parameter keys are unchanged.

**Toggle for the maximin component:** `include_social_comparison=True` (not
`include_altruism_term`) adds `Vᵢⱼ × min(πᵢᴬ, πⱼᴬ)`. This is the right flag because
`min(πᵢᴬ, πⱼᴬ)` is inherently a social comparison — it depends on who has less — just
as `include_social_comparison` already toggles the comparison-based term in the other
structural families. This parallels the `min_max_rawlsian_leontief` family, where
`include_social_comparison=True` selects the Rawlsian min over the Leontief max.
Within the welfare efficiency family, the social comparison parameter is `Vᵢⱼ`, not
`αᵢⱼ` — consistent with the `min_max` family.

`include_altruism_term` within the welfare efficiency family retains its standard meaning
(`Vᵢⱼ × πⱼᴬ`), but combining both `include_social_comparison=True` and
`include_altruism_term=True` within this family is blocked by validation (semantically
redundant and over-parameterized). So the welfare efficiency family produces exactly
two main variants: with and without the maximin social comparison component.

**Sibling flag: `single_payoffs_not_differences`** is NOT blocked within the welfare
efficiency family. This produces sibling forms:
- `single_payoffs_not_differences=True` (E&S original): bases are `πᵢᴬ`, `πⱼᴬ`
- `single_payoffs_not_differences=False` (difference form): bases are `πᵢᴬ−πᵢᴮ`, `(πᵢᴬ+πⱼᴬ)/2 − (πᵢᴮ+πⱼᴮ)/2`
  Maximin when `include_social_comparison=True`: `Vᵢⱼ×min(πᵢᴬ−πᵢᴮ, πⱼᴬ−πⱼᴮ)`

These are siblings because the number of free parameters is the same. `payoff_ratios_not_differences`
and `reference_dependent_utility` remain blocked (the ratio form conflicts with ERC,
and reference dependence conflicts with the welfare remapping).

### Flag 2: `include_relative_income_penalty`

Placed at index 4, right after Flag 1. Adds an ERC-style penalty on top of whatever
self-interest and altruism terms are already present:

- Linear: `−αᵢⱼ×(πᵢᴬ/(πᵢᴬ+πⱼᴬ) − ½)`
- Power: `−αᵢⱼ×(πᵢᴬ/(πᵢᴬ+πⱼᴬ) − ½)^γ₁` (or γ₃ when not `uniform_exponential_parameter`)

Uses the existing `αᵢⱼ` parameter. No new parameters. `apply_exponents_to_payoffs`
does **not** apply here (user decision).

---

## New `ordered_keys` (16 flags)

```python
ordered_keys = (
    'conditional_welfare_mode',          # 0
    'reference_dependent_altruism',      # 1
    'min_max_rawlsian_leontief',         # 2
    'include_welfare_efficiency_term',   # 3  <- NEW
    'include_relative_income_penalty',   # 4  <- NEW
    'use_exponential_parameters',        # 5
    'apply_exponents_to_payoffs',        # 6
    'uniform_exponential_parameter',     # 7
    'single_payoffs_not_differences',    # 8
    'payoff_ratios_not_differences',     # 9
    'reference_dependent_utility',       # 10
    'use_negativity_parameters',         # 11
    'negativity_social_comparison',      # 12
    'fix_self_interest_parameter',       # 13
    'include_social_comparison',         # 14
    'include_altruism_term',             # 15
)
```

Bitstring format changes from `XXXX-XXXX-XXXX-XX` (4-4-4-2) to `XXXX-XXXX-XXXX-XXXX` (4-4-4-4).

---

## Function-by-Function Impact Analysis

### Functions that REQUIRE changes

#### 1. `config.py` -- `UtilitySettings` TypedDict (line 125)
Add two new boolean keys after `min_max_rawlsian_leontief`:
```python
include_welfare_efficiency_term: bool
include_relative_income_penalty: bool
```

#### 2. `config.py` -- `utility_settings` dict (line 728)
Add two new entries with `False` defaults (preserves existing behavior):
```python
'include_welfare_efficiency_term': False,
'include_relative_income_penalty': False,
```

#### 3. `config.py` -- `parameter_keys_for_utility_settings` (line 266)
Two additions:

**For `include_relative_income_penalty`**: The function currently adds `αᵢⱼ` only
inside the `include_social_comparison` block. Add after that block:
```python
if utility_settings.get('include_relative_income_penalty'):
    if 'αᵢⱼ' not in param_keys:
        param_keys.append('αᵢⱼ')
```

**For `include_welfare_efficiency_term`**: Vᵢᵢ is already added by the
`not fix_self_interest_parameter` branch (validation blocks `fix_self_interest_parameter=True`
within this family). Vᵢⱼ must be added when `include_social_comparison=True` within
this family — but the existing `include_social_comparison` branch adds `αᵢⱼ`, not `Vᵢⱼ`.
Add a guard at the top of the `include_social_comparison` block:
```python
if utility_settings['include_social_comparison']:
    if utility_settings.get('include_welfare_efficiency_term'):
        param_keys.append('Vᵢⱼ')   # maximin uses Vᵢⱼ, not αᵢⱼ
    else:
        param_keys.append('αᵢⱼ')
        if utility_settings['use_negativity_parameters'] or utility_settings['negativity_social_comparison']:
            param_keys.append('βᵢⱼ')
```

#### 4. `utilities.py` -- `convert_utility_settings` (line 1043)
**Delete the hardcoded `ordered_keys` tuple entirely** (lines 1063-1078). Python 3.7+
guarantees dict insertion order, so `config.py`'s `utility_settings` dict is the single
source of truth for flag order. No second copy is needed.

Replacing `ordered_keys` in each conversion path:

- **dict → tuple/str/int**: use `list(utility_settings.keys())` (the input dict's own order).
- **tuple → dict**: requires the caller to pass the canonical template dict. Add a new
  `template` parameter (default `None`). When `utility_settings` is a tuple and `into` is
  `dict`, require `template` to be provided:
  ```python
  def convert_utility_settings(utility_settings, into=tuple, template=None):
      ...
      if isinstance(utility_settings, tuple) and into is dict:
          if template is None:
              raise ValueError("template (a dict[str, bool]) is required for tuple->dict conversion.")
          ordered_keys = list(template.keys())
          ...
  ```
  All existing callers of `convert_utility_settings(some_tuple, into=dict)` must be
  updated to pass `template=utility_settings` (the global from config.py). Grep the
  codebase for such calls before implementing.

The length-mismatch error message should now say "N flag keys in template" rather than
"N ordered_keys".

**Impact on callers:** Any call that converts a tuple to a dict must now pass `template`.
All other call forms (dict→tuple, dict→str, dict→int, dict→dict) are unaffected since
they can derive order from the input dict itself.

#### 5. `utilities.py` -- `is_valid_utility_settings` (line 1126)
Add validation rules enforcing clean family separation. After the existing
`min_max_rawlsian_leontief` block, add:

```python
if candidate['include_welfare_efficiency_term']:
    explanation = "If using welfare efficiency term, then "
    if candidate['conditional_welfare_mode'] or candidate['min_max_rawlsian_leontief']:
        explanation += "no other structural family flag may be active."
        return explanation if provide_explanation else False
    if candidate['include_relative_income_penalty']:
        explanation += "relative income penalty cannot be combined."
        return explanation if provide_explanation else False
    if candidate['use_negativity_parameters'] or candidate['negativity_social_comparison']:
        explanation += "negativity parameters are not supported."
        return explanation if provide_explanation else False
    if candidate['include_social_comparison'] and candidate['include_altruism_term']:
        explanation += "cannot combine both social comparison and altruism term (over-parameterized)."
        return explanation if provide_explanation else False
    if candidate['fix_self_interest_parameter']:
        explanation += "Vᵢᵢ must be free (controls the welfare weight)."
        return explanation if provide_explanation else False
    if candidate['payoff_ratios_not_differences']:
        explanation += "ratio form conflicts with the ERC flag and is not supported here."
        return explanation if provide_explanation else False
    if candidate['reference_dependent_utility']:
        explanation += "reference dependence conflicts with the welfare remapping."
        return explanation if provide_explanation else False
    "include_social_comparison=True IS allowed — it adds Vᵢⱼ×min(πᵢ,πⱼ)."
    "single_payoffs_not_differences IS allowed — produces sibling forms."

if candidate['include_relative_income_penalty']:
    explanation = "If using relative income penalty, then "
    if candidate['conditional_welfare_mode'] or candidate['min_max_rawlsian_leontief']:
        explanation += "no other structural family flag may be active."
        return explanation if provide_explanation else False
    if candidate['payoff_ratios_not_differences']:
        explanation += "ratio form is already embedded in the penalty term."
        return explanation if provide_explanation else False
    if candidate['use_negativity_parameters'] or candidate['negativity_social_comparison']:
        explanation += "negativity parameters are not supported."
        return explanation if provide_explanation else False
```

#### 6. `utilities.py` -- `_apply_minimal_dependent_fixes` (line 2479)
Add implication handlers:
- Pivot `include_welfare_efficiency_term=True`: force `conditional_welfare_mode=False`,
  `min_max_rawlsian_leontief=False`, `use_negativity_parameters=False`,
  `negativity_social_comparison=False`, `fix_self_interest_parameter=False`,
  `include_relative_income_penalty=False`, `payoff_ratios_not_differences=False`,
  `reference_dependent_utility=False`. Do **not** force `include_social_comparison=False`
  (it is allowed — activates maximin) and do **not** force `single_payoffs_not_differences`
  (either value produces a valid sibling).
- Pivot `include_relative_income_penalty=True`: force `conditional_welfare_mode=False`,
  `min_max_rawlsian_leontief=False`, `payoff_ratios_not_differences=False`,
  `use_negativity_parameters=False`, `negativity_social_comparison=False`,
  `include_welfare_efficiency_term=False`

#### 7. `utilities.py` -- `_format_utility_bitstring` (line 2561)
Update for 16-char format:
```python
def _format_utility_bitstring(raw_bitstring: str) -> str:
    """Formats a 16-character raw bitstring into XXXX-XXXX-XXXX-XXXX."""
    return f"{raw_bitstring[0:4]}-{raw_bitstring[4:8]}-{raw_bitstring[8:12]}-{raw_bitstring[12:16]}"
```

#### 8. `utilities.py` -- `compute_hamming_distance_matrix` (line 2059)
**Superseded by Side Quest 1.** If the function is dead code (no callers beyond imports),
it will be deleted entirely — no assertion update needed. If it has callers, update the
assertion to `<= 16` and update the docstring. Resolve at Side Quest 1 time.

#### 9. `utilities.py` -- `classify_pair_relation` (line 2641)
Add both new flags to `settings_when_flipped_dont_make_relatives`:
```python
settings_when_flipped_dont_make_relatives = (
    'conditional_welfare_mode',
    'min_max_rawlsian_leontief',
    'include_welfare_efficiency_term',   # <- NEW
    'include_relative_income_penalty',   # <- NEW
)
```
This ensures crossing the family boundary returns 'neither'. Within the new families,
sibling/parent-child detection via other flags (e.g., `include_altruism_term`,
`use_exponential_parameters`) still works correctly.

Also confirm the hardcoded `settings_when_flipped_make_children_parents` tuple does
NOT include the two new flags.

#### 10. `utilities.py` -- `map_child_to_parent_special_param_info`
Since both new flags are in `settings_when_flipped_dont_make_relatives`, they will
never appear as `changed_utility_setting` in parent-child pairs within the original
480-model family. **No changes required.** Within the new families, `include_altruism_term`
toggling (parent gains Vᵢⱼ) is handled by the existing mapping. Verify via smoketest.

#### 11. `model.py` -- `utility` (line 92)
Add a new structural branch after `min_max_rawlsian_leontief` and before the
standard additive section:

```python
elif utility_settings.get('include_welfare_efficiency_term'):
    Vᵢᵢ_used = (Vᵢᵢ + 1) / 2
    if utility_settings['use_exponential_parameters']:
        si_part   = (1 - Vᵢᵢ_used) * (payAi ** exp1)
        welf_part = Vᵢᵢ_used * (((payAi + payAj) / 2) ** exp1)
    else:
        si_part   = (1 - Vᵢᵢ_used) * payAi
        welf_part = Vᵢᵢ_used * ((payAi + payAj) / 2)
    if utility_settings['include_social_comparison']:
        "Maximin: Vᵢⱼ×min(πᵢᴬ, πⱼᴬ) — uses exp2 when uniform_exponential_parameter=False."
        exp_j   = exp2 if not utility_settings['uniform_exponential_parameter'] else exp1
        maximin = Vᵢⱼ * (min(payAi, payAj) ** exp_j
                          if utility_settings['use_exponential_parameters']
                          else min(payAi, payAj))
    else:
        maximin = 0.0
    if separate_terms:
        return {'self_interest': float(si_part + welf_part), 'altruism': 0.0, 'social_comp': float(maximin)}
    return si_part + welf_part + maximin
```

For `include_relative_income_penalty`, add **after** the standard
`self_interest + altruism + social_comp` total is computed:
```python
if utility_settings.get('include_relative_income_penalty'):
    denom  = payAi + payAj
    sigma  = payAi / denom if denom > 0 else 0.5
    dev    = sigma - 0.5
    penalty = -αᵢⱼ * ((dev ** exp3) if utility_settings['use_exponential_parameters'] else dev)
    if separate_terms:
        return {**result_dict, 'social_comp': float(result_dict['social_comp'] + penalty)}
    return total + penalty
```

#### 12. `model.py` -- `build_utility_equation` (line 458)
Extract new locals at the top alongside the existing flag extractions:
```python
inc_welf = utility_settings.get('include_welfare_efficiency_term', False)
inc_rip  = utility_settings.get('include_relative_income_penalty', False)
```

Add a new `elif inc_welf:` branch (parallel to `con_welf` and `min_max`) producing
strings like:
- `Uᵢ(A) = (1−((Vᵢᵢ+1)/2))×πᵢᴬ + ((Vᵢᵢ+1)/2)×(πᵢᴬ+πⱼᴬ)/2`
  (when `include_social_comparison=False`)
- `Uᵢ(A) = (1−((Vᵢᵢ+1)/2))×πᵢᴬ + ((Vᵢᵢ+1)/2)×(πᵢᴬ+πⱼᴬ)/2 + Vᵢⱼ×min(πᵢᴬ, πⱼᴬ)`
  (when `include_social_comparison=True` — the full E&S form)

For `inc_rip`, append the penalty term string before returning from the standard
additive path:
```python
exp_tag  = f"^γ{'₁' if one_exp else '₃'}" if use_exp else ""
rip_term = f" − αᵢⱼ×([{payAi}/({payAi}+{payAj}) − 1/2]){exp_tag}"
```

---

### Functions that adapt automatically (no changes needed)

| Function | Why automatic |
|---|---|
| `equation_to_settings` | Calls `generate_utility_settings` -> new models appear automatically |
| `generate_utility_settings` | Iterates `utility_settings.keys()` + filters via `is_valid_utility_settings` |
| `compute_conditional_hamming_distance_matrix` | Uses `flag_keys = list(settings_list[0].keys())` |
| `identify_redundant_utility_functions` | Calls `generate_utility_settings` |
| `parents_children_of` | Iterates `ordered_keys = list(base.keys())` dynamically |
| `model_nesting_adjacency_matrices` | Uses `classify_pair_relation` -> automatic once that's updated |
| `summarize_nesting_relationship_counts` | Consumes adjacency data |
| `test_utility_functions` | Flips named settings dynamically |
| `run_child_parent_embedding_sanity_checks` | Uses `list(utility_settings.keys())` |
| `run_child_parent_probability_equivalence_smoketest` | Uses `parameter_keys_for_utility_settings` |
| `verify_utility_vs_string_equation` | Uses `generate_utility_settings` + `list(utility_settings.keys())` |
| `select_child_params_for_parent` | No flag-specific logic |
| `best_fitting_model_parameters` | No flag-specific logic |
| `best_fitting_child_parameters_for_parent` | Delegates to nesting infrastructure |

---

## Side Quest 1: `compute_hamming_distance_matrix` Removal

Both `compute_hamming_distance_matrix` (utilities.py:2059) and
`compute_conditional_hamming_distance_matrix` (utilities.py:2146) are imported in
`behavioral_distances.py` and `architecture.py`. The conditional version is strictly
more informative.

**Action:** Grep all `.py` files for actual *calls* to `compute_hamming_distance_matrix`
(not just imports). If it is called nowhere in any logic path, remove it from
`utilities.py` and remove the import lines. If called somewhere, just update the
assertion to `<= 16`.

---

## Side Quest 2: Inner Functions in `verify_utility_vs_string_equation`

The inner functions in `verify_utility_vs_string_equation` (`_all_payoff_tuples`,
`_random_payoff_tuples`, `_sample_means_for`) are local payoff/parameter generators.
The 10 functions at the bottom of `utilities.py` (after line 3257) are equation-string
parsers (`eval_pretty_equation_rhs`, `normalize_pretty_rhs_for_eval`, etc.).
**No redundancy. No action needed.**

---

## Side Quest 3: AMPD Matrix Incremental Update

The existing 480×480 AMPD matrix took a very long time to compute (30 iterations per
pair). Computing a full (480+N)×(480+N) matrix from scratch would be prohibitive.

**Plan:** After the new models are enumerated, extend the existing matrix by computing
only the new rows and columns — i.e., the AMPD distances between each new model and
all existing 480 models, plus the distances among new models themselves. The original
480×480 block stays unchanged.

**Implementation sketch:**
1. Load the existing AMPD matrix CSV (`model_distance_ampd__n_models=480*.csv`).
2. Identify the new model indices (those not present in the loaded matrix).
3. Run `_ampd_pair_worker` only for (new, existing) and (new, new) pairs.
4. Stitch the new rows/columns into the full matrix and save.

This requires that `compute_ampd_distance_matrix` supports a `skip_existing=True`
mode that reads a partial matrix and fills only the missing cells. Implement this
before re-running the full AMPD analysis.

---

## Post-Implementation: Update "480" References

After regenerating the registry with `create_new_file=True`, count new model total N.
Update hardcoded references in:
- `config.py:709` -- `'n_candidate_models': 480`
- `analysis.py:1142, 2388`
- `utilities.py:1743, 2293`
- `architecture.py:120`
- `quick_demo.py:13, 25, 57, 189, 211, 559, 562, 570, 601`
- `behavioral_distances.py:26, 546, 592, 604`

Where possible make dynamic (derive from registry CSV rather than hardcoding).

---

## Post-Implementation: IC Scatterplot Canonical Annotation

In `plot_ic_scores_delta_bic` (analysis.py), read `canonical_utility_settings.json`,
look up each spec's `utility_idx` in the registry CSV, and add an annotation layer with
model names (Fehr-Schmidt, Bolton-Ockenfels, Charness-Rabin, Andreoni-Miller,
Engelmann-Strobel, Messick-McClintock).

---

## Implementation Order

1. **Copy baseline files** (all_utility_functions.csv, nesting JSON, AMPD matrix, sanity CSVs)
2. **Run baseline sanity checks** (embedding test + smoketest)
3. `config.py`: TypedDict, `utility_settings` dict, `parameter_keys_for_utility_settings`
4. `utilities.py`: `convert_utility_settings`, `is_valid_utility_settings`, `_apply_minimal_dependent_fixes`, `_format_utility_bitstring`, `compute_hamming_distance_matrix` assertion, `classify_pair_relation`
5. `model.py`: `utility`, `build_utility_equation`
6. Syntax-check all modified files
7. Run `generate_utility_settings(create_new_file=True)` -- print **all new equations** via
   `build_utility_equation` for every model where either new flag is `True`, plus the total N.
   **STOP HERE** for Greg to review the new forms and confirm N is manageable before continuing.
8. Verify `all_utility_functions.csv`: 480-row portion must match baseline exactly; new rows appended; two new columns present (all False for original 480)
9. Run `verify_utility_vs_string_equation` -- confirm string/code agreement for all new models
10. Run post-change nesting sanity checks; compare CSVs against baseline
11. Side quest 1: grep, remove `compute_hamming_distance_matrix` if dead
12. Update "480" references with new N
13. Update `quick_demo.py` with new flag demos
14. **Side quest 3**: Implement AMPD incremental update, extend matrix for new models
15. Run `quick_demo.py` end-to-end

---

## Verification Checklist

- [ ] Baseline files copied before any code change
- [ ] Baseline sanity check CSVs saved
- [ ] Syntax checks pass for all modified files
- [ ] All 480 existing models still pass `is_valid_utility_settings` after adding new flags
- [ ] `generate_utility_settings(create_new_file=True)` runs; new N > 480
- [ ] `all_utility_functions.csv`: 480-row portion matches baseline; new columns present
- [ ] `verify_utility_vs_string_equation` passes for all new models
- [ ] Post-change nesting CSVs match baseline for all original 480-model pairs
- [ ] E&S equation (welfare_efficiency=T, social_comparison=T): contains welfare_weight×avg_payoff + Vij×min terms
- [ ] ERC equation: `πᵢᴬ − αᵢⱼ×[πᵢᴬ/(πᵢᴬ+πⱼᴬ) − 1/2]`
- [ ] AMPD matrix extended incrementally (no recompute of 480×480 block)
- [ ] `quick_demo.py` end-to-end pass
