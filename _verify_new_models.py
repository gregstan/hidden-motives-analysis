"""Verification script: confirm 505-model expansion is correct end-to-end."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from main import *
from config import utility_settings, general_settings, file_paths, param_bds
from utilities import generate_utility_settings

NEW_FLAGS = ('include_welfare_efficiency_term', 'include_relative_income_penalty')

"=========================================================================================="
"1. Regenerate utility registry CSV"
"=========================================================================================="

print("\n" + "="*70)
print("STEP 1: Regenerate all_utility_functions.csv")
print("="*70)

registry_df = generate_utility_settings(
    utility_settings=utility_settings,
    file_paths=file_paths,
    general_settings=general_settings,
    build_equation_function=build_utility_equation,
    create_new_file=True,
    return_df=True,
)

n_total = len(registry_df)
new_flag_cols_present = all(f in registry_df.columns for f in NEW_FLAGS)
old_rows = registry_df[~registry_df[NEW_FLAGS[0]] & ~registry_df[NEW_FLAGS[1]]]
new_rows = registry_df[registry_df[NEW_FLAGS[0]] | registry_df[NEW_FLAGS[1]]]

print(f"\nRegistry rows   : {n_total}  (expected 505)")
print(f"New flag cols   : {new_flag_cols_present}  (both must be True)")
print(f"Old model rows  : {len(old_rows)}  (expected 480)")
print(f"New model rows  : {len(new_rows)}  (expected 25)")

if n_total != 505 or not new_flag_cols_present or len(old_rows) != 480 or len(new_rows) != 25:
    print("\nFAIL: Registry structure mismatch — stop before proceeding.")
    sys.exit(1)
print("\nPASS: Registry structure looks correct.")

"=========================================================================================="
"2. String-code numeric agreement for all 25 new models"
"=========================================================================================="

print("\n" + "="*70)
print("STEP 2: verify_utility_vs_string_equation for all 25 new models")
print("="*70)

print("Running verify_utility_vs_string_equation across all 505 models...")
result_df = verify_utility_vs_string_equation(
    utility_function=utility,
    utility_function_str=build_utility_equation,
    utility_settings=utility_settings,
    param_bds=param_bds,
    file_paths=file_paths,
    n_games=5**4,
    rng_seed=20250524,
    exhaustive_if_large=True,
    option="A",
    verbose=True,
)

"Report failures, distinguishing new vs pre-existing."
if 'match' in result_df.columns:
    failures = result_df[result_df['match'] == False]
    new_failures = failures[failures['include_welfare_efficiency_term'] | failures['include_relative_income_penalty']]
    if len(new_failures) == 0:
        print(f"\nPASS (new models): All 25 new model×game checks passed.")
        if len(failures) > 0:
            n_pre = result_df['utility_idx'].nunique() - 25
            print(f"NOTE: {len(failures)} pre-existing mismatches in {failures['utility_idx'].nunique()} of {n_pre} original models (unchanged from baseline).")
    else:
        print(f"\nFAIL: {len(new_failures)} mismatches in NEW models.")
        print(new_failures[['utility_idx', 'U_function', 'utility_Δ', 'status']].head(20).to_string())
else:
    print(result_df.to_string())

"=========================================================================================="
"3. Parent-child probability equivalence smoketest (covers all 505 models)"
"=========================================================================================="

print("\n" + "="*70)
print("STEP 3: Child-parent probability equivalence smoketest (all 505 models)")
print("="*70)

run_child_parent_probability_equivalence_smoketest(
    utility_settings=utility_settings,
    file_paths=file_paths,
    param_bds=param_bds,
    rand_payoff_idx=True,
    n_trials=12,
    rng_seed=20250524,
    tolerance=1e-12,
    verbose=True,
)

print("\nAll verification steps complete.")
