"""Temporary script: print all new utility equations and total N."""
import sys
sys.path.insert(0, r"C:\Users\Gregory Stanley\Desktop\U of M\Research Archive\Multiplayer\hidden-motives-analysis")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from config import utility_settings
from utilities import generate_utility_settings
from model import build_utility_equation

all_settings = generate_utility_settings(utility_settings=utility_settings)
total_n = len(all_settings)

new_models = [s for s in all_settings if s.get('include_welfare_efficiency_term') or s.get('include_relative_income_penalty')]
old_models = [s for s in all_settings if not s.get('include_welfare_efficiency_term') and not s.get('include_relative_income_penalty')]

print(f"\nTotal models: {total_n}")
print(f"Existing models: {len(old_models)}")
print(f"New models (welfare efficiency + RIP): {len(new_models)}")
print(f"\n{'='*80}")
print(f"NEW MODEL EQUATIONS")
print(f"{'='*80}")

welf_models = [s for s in new_models if s.get('include_welfare_efficiency_term')]
rip_models  = [s for s in new_models if s.get('include_relative_income_penalty')]

print(f"\n--- WELFARE EFFICIENCY MODELS ({len(welf_models)}) ---")
for i, s in enumerate(welf_models):
    eq = build_utility_equation(utility_settings=s, option="A")
    flags = {k: v for k, v in s.items() if v and k not in ('include_welfare_efficiency_term',)}
    active = [k for k, v in flags.items() if v]
    print(f"\n  [{i+1}] {eq}")
    print(f"       Active: {active}")

print(f"\n--- RELATIVE INCOME PENALTY MODELS ({len(rip_models)}) ---")
for i, s in enumerate(rip_models):
    eq = build_utility_equation(utility_settings=s, option="A")
    flags = {k: v for k, v in s.items() if v and k not in ('include_relative_income_penalty',)}
    active = [k for k, v in flags.items() if v]
    print(f"\n  [{i+1}] {eq}")
    print(f"       Active: {active}")
