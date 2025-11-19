# Inferring Hidden Motives — Analysis Code

Analysis code for:

**Stanley, G. N., Lewis, R. L., & Zhang, J. (2025). _Inferring Hidden Motives:_ Bayesian learning of social preferences in iterated binary dictator games.**

This repository reproduces results end‑to‑end:
1. Simulations (parameter recovery; update‑speed; convergence)
2. Illustrations (2D/3D belief updates, accuracy)
3. Alternative‑model contest (non‑Bayesian baselines)
4. Typological Bayesian comparisons (discrete profiles)
5. Information‑criterion (IC) search across **476** utility forms
6. Nesting and verification checks (parent/child/sibling structure)
7. Human‑data results (parameter distributions, correlations, subpopulations)
8. Inequality‑aversion analyses

> **Data note:** by default, large data/outputs are **not** tracked in Git to keep clones light. See `data/README.md` and the **Data** section below for options.

---

## Repo layout
hidden-motives-analysis/
README.md
LICENSE
.gitignore
config.py # settings, types, library imports, and path map

main.py # orchestrates analyses in the order used in the paper
preprocessing.py # data cleaning / merges (Iter_Binary_Dictator_Preprocessing)
utilities.py # general helpers (Iter_Binary_Dictator_Generalist)
typological.py # typological Bayesian (discrete) models

docs/
data_dictionary.md # coding sheet (file→column→meaning)
architecture.md # how modules/settings/pipeline fit together

raw_data/ # raw inputs (CSV/JSON)
processed/ # cleaned/merged data products
param_data/ # (ignored) fitted parameter JSON/CSV
player_fits/ # (ignored) per-player fit artifacts
dyad_data/ # (ignored) dyad-level caches
discrete/ # (ignored) typological model outputs
bic_aic/ # (ignored) IC outputs / tables
visuals/ # Plotly HTML / PNG figures

## Quick start

```bash
# 0) Create and activate a virtual environment
python -m venv .venv
# Windows:
. .venv/Scripts/activate
# macOS/Linux:
# source .venv/bin/activate

# 1) Install dependencies
pip install -r requirements.txt

# 2) Provide data
# Option A: Put raw CSV/JSON under ./raw_data/ (see docs/data_dictionary.md)
# Option B: Put preprocessed CSVs under ./processed/
# (For full datasets, see OSF link below.)

# 3) Run the full analysis (default toggles at the bottom of main.py)
python main.py