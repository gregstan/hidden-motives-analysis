# Architecture & Module Map

This codebase follows a “single switchboard” pattern: **`config.py`** defines paths and settings; `main.py` orchestrates analysis blocks in the order used in the manuscript.

## Core modules

- **`config.py`**
  - `file_paths`: repo‑relative folders and canonical filenames (no absolutes)
  - `general_settings`: analysis mode (Bayesian/MLE), optimizer (“glob”, “loc”, “globloc”), grid/parametric updates, parallel flags, plotting, etc.
  - `utility_settings`: toggles to build the utility equation string
  - `param_info`, `param_bds`: parameter metadata/bounds

- **`main.py`**
  - Organizes functions under section headers mirroring the paper:
    - Utility/choice functions → MLE block → Bayesian block  
    - Simulations 1–3 → Belief‑update visuals → Alternative models  
    - Typological (discrete) models → IC analysis → Nesting/verification  
    - Human‑data results → Inequality‑aversion analyses
  - `run_code_settings` at bottom toggles major blocks for quick smoke tests.

- **`preprocessing.py`**
  - Cleans/merges raw inputs; writes standardized CSVs to `processed/`.

- **`utilities.py`**
  - General helpers: utility‑equation builders, nesting graphs, string↔settings conversion, verification checks, etc.

- **`typological.py`**
  - Discrete/typological Bayesian model code (population and individual fits).

## Key objects and how they flow

1. **Data** (`raw_data/` → `processed/`)  
   Preprocessing makes uniform tables for fitting and plots.

2. **Settings** (`config.py`)  
   Passed into all major functions: `general_settings`, `utility_settings`, `file_paths`, `param_bds`.

3. **Fitting** (Bayesian/MLE)  
   Writes per‑player/per‑dyad outputs to `param_data/` and unified CSVs (e.g., `All_Fitted_Data_Bayesian.csv`) into `processed/`.

4. **IC / Nesting**  
   Utility sets are enumerated (476 forms); nesting graphs/verification; writes tables to `bic_aic/`.

5. **Figures**  
   Written to `visuals/` as Plotly HTML (export parameters in `config.py`).

## Development tips

- Keep paths repo‑relative (use `config.file_paths`).
- If adding a new figure, write a function (saves to `visuals/`) and call it from `main.py`.
- For large datasets, use OSF and place files under `raw_data/` or `processed/` locally.
