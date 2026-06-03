# Plan: Add `tie_self_interest_and_altruism` Flag (17th Utility Setting)

## Context

Andreoni & Miller (2002) use a CES utility function:

`U = [α·π_self^ρ + (1−α)·π_other^ρ]^(1/ρ)`

The outer `^(1/ρ)` is a monotone transform absorbed by softmax temperature — no separate handling needed. The exponents can already be tied via `uniform_exponential_parameter`. The one structural property not testable in the current 505-model space is the **weight constraint α + (1−α) = 1**, which ties the altruism weight to the complement of the self-interest weight.

The conditional welfare mode already enforces this complement structure, but always conditionally (switching weights based on who is ahead). A standalone unconditional form — `Vᵢᵢ·self + (1−Vᵢᵢ)·alt` — is not in the current candidate set. The paper's claim that it tests all canonical forms has a thorn: the specific A&M constrained form is absent. Adding a 17th flag fills this gap with a well-defined parent-child relationship (the tied form is a child of the free-weight form, with one fewer free parameter).

---

## Files to Modify

- [config.py](config.py) — TypedDict, defaults, `parameter_keys_for_utility_settings`, `create_file_name_suffix` bitstring format
- [model.py](model.py) — `utility()` numeric computation, `build_utility_equation()` string rendering
- [utilities.py](utilities.py) — `is_valid_utility_settings`, `_apply_minimal_dependent_fixes`, `classify_pair_relation`, embedding warm-start block, `convert_utility_settings` bitstring detection

---

## Step 1 — config.py: TypedDict + defaults

Already done by the user. The TypedDict entry and the `'tie_self_interest_and_altruism': False` default are in place at the user's preferred insertion position.

---

## Step 2 — config.py: `parameter_keys_for_utility_settings`

In the standard additive `else` branch (~line 334), the current block adds `Vᵢⱼ` whenever `include_altruism_term=True`. Gate it on the new flag:

```python
if utility_settings['include_altruism_term']:
    if not utility_settings.get('tie_self_interest_and_altruism', False):
        param_keys.append('Vᵢⱼ')
        if utility_settings['use_negativity_parameters']:
            param_keys.append('λᵢⱼ')
```

When `tie=True`, both `Vᵢⱼ` and `λᵢⱼ` are derived from Vᵢᵢ and λᵢᵢ respectively (`1 - Vᵢᵢ`, `1 - λᵢᵢ`), so they carry no free parameters. `count_free_parameters` calls this function and auto-inherits the correct k.

**Do not change** the conditional-welfare branch (lines 312–314) or the welfare-efficiency branch (lines 322–323) — the new flag is invalid in those modes.

---

## Step 3 — config.py: `create_file_name_suffix` bitstring format

The bitstring is built with `sorted(utility_settings.items())`. The new key `'tie_self_interest_and_altruism'` sorts alphabetically between `'single_payoffs_not_differences'` (position 14) and `'use_exponential_parameters'` (position 16), landing at **position 15** in the sorted 17-key sequence.

Update the format string from 16 to 17 bits, extending the final group from 4 to 5:

```python
"Old (16 bits):  XXXX-XXXX-XXXX-XXXX"
"New (17 bits):  XXXX-XXXX-XXXX-XXXXX"

raw_utility_bits = "".join(str(int(val)) for _, val in sorted(utility_settings.items()))
fmt_utility_bits = f"{raw_utility_bits[0:4]}-{raw_utility_bits[4:8]}-{raw_utility_bits[8:12]}-{raw_utility_bits[12:17]}"
file_name_suffix += "--" + fmt_utility_bits
```

---

## Step 4 — model.py: `utility()` numeric computation

In the standard additive branch (~lines 309–318), replace the altruism block with a tied-weight-aware version. Gate the normalization on `normalize_conditional_welfare_params` — the same flag already used for the conditional-welfare branch — so verification code gets consistent behavior at all call sites (which already pass `normalize_conditional_welfare_params=False`):

```python
"Altruism term"
if utility_settings['include_altruism_term']:
    tie_weights = utility_settings.get('tie_self_interest_and_altruism', False)
    if tie_weights:
        "Mirror the conditional-welfare normalization pattern: Vᵢᵢ maps [-1,1] → [0,1]"
        "so (1 - Vᵢᵢ_norm) stays in [0,1]. Skipped when normalize_conditional_welfare_params=False"
        "(which is how all verification call sites invoke utility(), keeping string and numeric consistent)."
        if normalize_conditional_welfare_params:
            Vᵢᵢ_norm = (Vᵢᵢ + 1) / 2
            λᵢᵢ_norm = (λᵢᵢ + 1) / 2
        else:
            Vᵢᵢ_norm = Vᵢᵢ
            λᵢᵢ_norm = λᵢᵢ
        weight_1_al = 1.0 - Vᵢᵢ_norm
        weight_2_al = (1.0 - λᵢᵢ_norm) if utility_settings['use_negativity_parameters'] else 0.0
    else:
        weight_1_al = Vᵢⱼ
        weight_2_al = λᵢⱼ if utility_settings['use_negativity_parameters'] else 0.0
    altruism = utility_term(payoff_1=pay1al, payoff_2=pay2al,
                            weight_1=weight_1_al, exponent=exp2,
                            weight_2=weight_2_al,
                            use_negativity_parameters=utility_settings['use_negativity_parameters'],
                            use_exponential_parameters=utility_settings['use_exponential_parameters'],
                            single_payoffs_not_differences=utility_settings['single_payoffs_not_differences'],
                            payoff_ratios_not_differences=utility_settings['payoff_ratios_not_differences'])
else:
    altruism = 0.0
```

**Why `normalize_conditional_welfare_params` and no verification code changes:** every call to `utility()` inside the verification functions already passes `normalize_conditional_welfare_params=False` (analysis.py lines 4261, 3492, 3412, 3663–3664). With `False`, `Vᵢᵢ_norm = Vᵢᵢ` (raw), so the string equation `(1 - Vᵢᵢ)` exactly matches the numeric path. No changes to analysis.py are needed.

**Do not change** the conditional-welfare branch (~lines 170–178) — `tie_self_interest_and_altruism` is invalid there.

---

## Step 5 — model.py: `build_utility_equation()` string rendering

At the top of the function alongside the other flag extractions (~lines 544–558), add:
```python
tie_wts  = utility_settings.get('tie_self_interest_and_altruism', False)
```

In the `term()` inner function, `"altruism"` case (~line 673):
```python
elif term_type == "altruism":
    if not alt_term:
        return ""
    weight1 = "(1 - Vᵢᵢ)" if tie_wts else "Vᵢⱼ"
    weight2 = "(1 - λᵢᵢ)" if tie_wts else "λᵢⱼ"
    operator1, operator2 = " + ", " - "
    ...
```

The rendered equation for the core A&M form (tie=True, single_payoffs=True, single_exp=True) becomes:
`Uᵢ(A) = Vᵢᵢ(πᵢᴬ)^γ₁ + (1 - Vᵢᵢ)(πⱼᴬ)^γ₁`

---

## Step 6 — utilities.py: `is_valid_utility_settings`

Add a new validation block immediately before the final `return True` (~line 1435):

```python
if candidate.get('tie_self_interest_and_altruism', False):
    explanation = "If tying self-interest and altruism weights, then "
    if not candidate['include_altruism_term']:
        explanation += "an altruism term must be present to tie."
        return explanation if provide_explanation else False
    if candidate['fix_self_interest_parameter']:
        explanation += "the self-interest parameter Vᵢᵢ must be free (altruism would collapse to 0 if Vᵢᵢ is fixed at 1)."
        return explanation if provide_explanation else False
    if candidate['conditional_welfare_mode']:
        explanation += "conditional welfare mode already defines its own complement structure."
        return explanation if provide_explanation else False
    "NOTE: The min-max/Rawlsian/Leontief prohibition below is a scope limitation, not a theoretical"
    "one. min(Vᵢᵢ·base_self, (1-Vᵢᵢ)·base_alt) is a coherent tied-weight Leontief form — excluded"
    "only to limit combinatorial explosion and avoid adding tied-weight code paths to the min-max"
    "branch of utility() and build_utility_equation(). Revisit if desired in the future."
    if candidate['min_max_rawlsian_leontief']:
        explanation += (
            "the tied-weight form is currently excluded from the min-max/Rawlsian/Leontief branch "
            "(scope limitation, not a theoretical prohibition — revisit in the future if desired)."
        )
        return explanation if provide_explanation else False
    "NOTE: include_welfare_efficiency_term=True is implicitly prohibited through prior rules —"
    "welfare efficiency already forces include_altruism_term=False (existing rule, line 1396),"
    "which conflicts with the tie requirement for include_altruism_term=True above."
    "No explicit check here needed; this comment documents why."

    artificially_limit_combinations = True
    if artificially_limit_combinations:
        "Option A: restrict tie to single-payoff forms only. Theoretical motivation: the A&M CES"
        "aggregator [α·πᵢ^ρ + (1-α)·πⱼ^ρ]^(1/ρ) is defined over individual payoffs, not differences."
        "The complement constraint α+(1-α)=1 has its economic interpretation as a CES aggregator"
        "precisely in that context. Toggle artificially_limit_combinations=False to allow the tied"
        "weight form across all payoff representations (utility() and build_utility_equation() both"
        "support this correctly — the restriction lives only here)."
        if not candidate['single_payoffs_not_differences']:
            explanation += (
                "the tied-weight form is currently restricted to single-payoff models "
                "(artificially_limit_combinations=True). Set that flag False in "
                "is_valid_utility_settings to allow tied weights with difference and ratio forms."
            )
            return explanation if provide_explanation else False
```

**Why `n_social_preference_params` does not need updating:**

The count in `is_valid_utility_settings` is:
```python
n_social_preference_params = sum([
    int(not fix_self_interest_parameter),  # 1 if Vᵢᵢ is free (a social preference WEIGHT, not just a term)
    int(include_altruism_term),            # 1 if the altruism term is structurally present
    int(include_social_comparison)         # 1 if the social comparison term is structurally present
])
```

Note the semantics are slightly mixed: `fix_self_interest_parameter` contributes based on **weight freedom** (whether Vᵢᵢ is free), while the other two contribute based on **term presence**. The purpose of the count is to drive the exponent rule: if only one social preference weight/term exists in the model (n=1), distinct exponents γ1 ≠ γ2 are nonsensical (no second term to apply γ2 to), so `uniform_exponential_parameter=True` is required.

When `tie=True`:
- `include_altruism_term=True` (required) → contributes 1 to n
- `fix_self_interest_parameter=False` (required) → contributes 1 to n
- Together these guarantee n ≥ 2 for every valid tied model

n can never drop to 1 or 0, so the "n=1 → single exponent required" restriction can never fire incorrectly on a tied model. The altruism term contributes to n even though its weight Vᵢⱼ is derived rather than free — the count tracks term structure, not parameter freedom, for the altruism and social comparison terms. No code change needed.

**Valid combinations (by exclusion):** `tie=True` is compatible with `use_negativity_parameters`, `include_social_comparison`, `include_relative_income_penalty`, `use_exponential_parameters` (tied or untied), `single_payoffs_not_differences`, `payoff_ratios_not_differences`, `reference_dependent_utility`, and `apply_exponents_to_payoffs`.

**Combinatorial estimate:** With `artificially_limit_combinations=True` (Option A — single payoffs only), roughly 30–50 new models, bringing the total from 505 to approximately 535–555. Without it (~200–250 new models, ~700–755 total). `utility()` and `build_utility_equation()` are implemented without assuming the Option A constraint — the restriction lives only in `is_valid_utility_settings` and can be toggled by flipping `artificially_limit_combinations`.

---

## Step 7 — utilities.py: `_apply_minimal_dependent_fixes`

Add a new pivot block after the existing `include_relative_income_penalty` block (~line 2840):

```python
if pivot == 'tie_self_interest_and_altruism' and utility_settings['tie_self_interest_and_altruism']:
    "Altruism must be present to tie; Vᵢᵢ must be free or altruism collapses to zero."
    utility_settings['include_altruism_term']       = True
    utility_settings['fix_self_interest_parameter'] = False
```

Structural mode conflicts (`conditional_welfare_mode`, `min_max_rawlsian_leontief`, `include_welfare_efficiency_term`) are left for `is_valid_utility_settings` to reject — forcing them off here would be over-aggressive and could cascade incorrectly.

---

## Step 8 — utilities.py: `classify_pair_relation`

The hardcoded tuple `settings_when_flipped_make_children_parents` at ~line 2985 must include the new flag:

```python
settings_when_flipped_make_children_parents = (
    'use_exponential_parameters',
    'uniform_exponential_parameter',
    'use_negativity_parameters',
    'negativity_social_comparison',
    'fix_self_interest_parameter',
    'include_social_comparison',
    'include_altruism_term',
    'tie_self_interest_and_altruism',    # ← new; tying removes Vᵢⱼ (and λᵢⱼ), reducing k
)
```

Flipping `tie=True` removes 1 parameter (or 2 if negativity is on) → the tied model is the child, the free-weight model is the parent. This integrates correctly with the existing k-comparison logic in `classify_pair_relation`.

**No special-casing required** in the sibling block — toggling `tie` always changes k, so it always lands in the parent/child branch, never the sibling branch.

---

## Step 9 — utilities.py: Embedding warm-start block

Add a new block after the conditional-welfare embedding block (~line 3377) in the `embed_child_params_in_parent_model` function:

```python
"--- SPECIAL: tied-altruism child (Vᵢⱼ = 1−Vᵢᵢ) → free-altruism parent ---"
if (not parent_utility_settings.get('tie_self_interest_and_altruism', False)
        and child_utility_settings.get('tie_self_interest_and_altruism', False)):

    "Seed parent's free Vᵢⱼ at the value implied by child's Vᵢᵢ."
    Vii = float(child_fitted_parameters.get('Vᵢᵢ', 1.0))
    Lai = float(child_fitted_parameters.get('λᵢᵢ', 0.0))
    if 'Vᵢⱼ' in parent_keys:
        embedded_parent_values['Vᵢⱼ'] = 1.0 - Vii
    if 'λᵢⱼ' in parent_keys:
        embedded_parent_values['λᵢⱼ'] = 1.0 - Lai
```

No γ2 special-casing is needed: both parent and child expose γ2 identically when `use_exponential_parameters=True` and `uniform_exponential_parameter=False`, so the generic param-matching pass handles it correctly.

---

## Step 10 — utilities.py: `convert_utility_settings` bitstring detection

The auto-detection for raw bitstrings currently checks `len(raw) == 16`. Update to `len(raw) == 17`. Also update the formatted-bitstring detection: after stripping dashes, a valid formatted 17-bit string is 17 chars. The old 16-char bitstrings remain readable via `input_settings_format` for legacy migration.

---

## Nesting Relationships Created

| Child model | Parent model | Δk |
|---|---|---|
| `tie=True, negativity=False` | same settings with `tie=False` | −1 (Vᵢⱼ removed) |
| `tie=True, negativity=True` | same settings with `tie=False` | −2 (Vᵢⱼ and λᵢⱼ removed) |

Every tied model's parent is its free-weight counterpart. No indirect or non-obvious nesting relationships are introduced — `tie` is orthogonal to all other flags that create parent-child edges (exponential parameters, negativity, social comparison, altruism inclusion, fix self-interest).

**Interaction with conditional-welfare:** The conditional-welfare child (no explicit altruism) already has the form `Vᵢᵢ·self + (1−Vᵢᵢ)·alt` within each branch. It is NOT a relative of the new tied models — the two are in different structural families (`conditional_welfare_mode=True` vs. `False`) and will classify as "neither" in `classify_pair_relation`, which is correct.

---

## Validity Edge Cases to Verify Manually

1. `tie=True, fix_self=True` → **invalid** (altruism collapses to zero)
2. `tie=True, include_altruism_term=False` → **invalid** (nothing to tie)
3. `tie=True, conditional_welfare_mode=True` → **invalid** (incompatible family)
4. `tie=True, use_negativity_parameters=True, include_social_comparison=False` → **valid** (removes Vᵢⱼ and λᵢⱼ; params are Vᵢᵢ, λᵢᵢ + exponents)
5. `tie=True, single_payoffs_not_differences=True, uniform_exponential_parameter=True` → **valid** (the core A&M form; params: Vᵢᵢ, γ₁)
6. `tie=True, include_social_comparison=True` → **valid** (Vᵢᵢ, αᵢⱼ + exponents)
7. `tie=True, include_relative_income_penalty=True` → **valid** (Vᵢᵢ, αᵢⱼ penalty + exponents)
8. `tie=True, use_exponential_parameters=True, uniform_exponential_parameter=False` → **valid** (Vᵢᵢ, γ₁, γ₂; untied exponents allowed per user intent)

---

## Implementation Order

1. `config.py` — TypedDict, defaults, `parameter_keys_for_utility_settings` gate, bitstring format (Steps 1–3)
2. `model.py` — `utility()` altruism block, `build_utility_equation()` weight rendering (Steps 4–5)
3. `utilities.py` — `is_valid_utility_settings` block (Step 6)
4. `utilities.py` — `_apply_minimal_dependent_fixes` pivot block (Step 7)
5. `utilities.py` — `classify_pair_relation` hardcoded tuple (Step 8)
6. `utilities.py` — embedding warm-start block (Step 9)
7. `utilities.py` — `convert_utility_settings` bitstring detection (Step 10)
8. Compile check: `python -m py_compile config.py model.py utilities.py`

---

## Verification

Run in order after implementation:

```python
"1. Compile check — zero errors required"
python -m py_compile config.py model.py utilities.py analysis.py

"2. Spot-check: the core A&M form exists and has the right parameter count"
from config import utility_settings
from utilities import is_valid_utility_settings, count_free_parameters
from utilities import convert_utility_settings

am_form = {**utility_settings,
    'tie_self_interest_and_altruism': True,
    'single_payoffs_not_differences': True,
    'uniform_exponential_parameter':   True,
    'use_exponential_parameters':     True,
    'include_altruism_term':          True,
    'fix_self_interest_parameter':    False,
    'conditional_welfare_mode':       False,
    'include_social_comparison':      False,
    'use_negativity_parameters':      False,
}
assert is_valid_utility_settings(am_form)
assert count_free_parameters(am_form) == 2   # Vᵢᵢ, γ₁ only

"3. The tied model is a child of its free-weight counterpart"
from utilities import classify_pair_relation
free_form = {**am_form, 'tie_self_interest_and_altruism': False}
assert is_valid_utility_settings(free_form)
assert count_free_parameters(free_form) == 3  # Vᵢᵢ, Vᵢⱼ, γ₁
rel_1_to_2, rel_2_to_1, changed = classify_pair_relation(am_form, free_form, utility_settings)
assert rel_1_to_2 == 'child' and rel_2_to_1 == 'parent'

"4. Invalid combinations are rejected"
bad1 = {**am_form, 'fix_self_interest_parameter': True}
bad2 = {**am_form, 'include_altruism_term': False}
bad3 = {**am_form, 'conditional_welfare_mode': True}
assert not is_valid_utility_settings(bad1)
assert not is_valid_utility_settings(bad2)
assert not is_valid_utility_settings(bad3)

"5. Numeric utility matches string equation for all new valid forms"
verify_utility_vs_string_equation(utility_settings=utility_settings)  # must pass for all ~700 models

"6. Children embedded in parents reproduce exact same choice probability"
verify_same_inputs_same_outputs_for_children_and_parents(utility_settings=utility_settings)
run_child_parent_probability_equivalence_smoketest(utility_settings=utility_settings)

"7. Regenerate nesting adjacency matrices"
model_nesting_adjacency_matrices(create_new_file=True, ...)
```
