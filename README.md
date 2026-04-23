# Inferring Hidden Motives — Utility Bayesian Model

This repository implements the **Utility Bayesian Model (UBM)** from *Inferring Hidden Motives:
Bayesian Models of Preference Learning in Repeated Dictator Games* (Stanley, Zhang, & Lewis, 2025; arXiv).
The UBM formalizes how an observer updates beliefs about another person's *social preferences*
(e.g., altruism vs. selfishness; envy vs. guilt) from their repeated payoff-allocation choices
in *iterated binary dictator games*.

The full paper includes computationally intensive analyses (e.g., an IC comparison across 480 utility
functions that takes weeks to run on a multi-core machine). [`quick_demo.py`](quick_demo.py) provides a
configurable entry point that can run every major analysis at reduced scale (toggled via `light_mode`)
or at full scale — making it useful for both fast debugging and producing the paper's main results.

---

## 1) Requirements

- **Python**: 3.9+
- **Packages** (see [`requirements.txt`](requirements.txt)):
  - `numpy>=1.24`
  - `pandas>=2.2`
  - `scipy>=1.11`
  - `plotly==5.11.0`
  - `statsmodels>=0.14`
  - `numexpr>=2.8`

---

## 2) Setup and run

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python quick_demo.py
```

All demo outputs are written under `demo_files/` so you can delete that folder afterward.
Sections 8–11 (requiring raw experiment data) additionally write to `demo_files/` to ensure
precomputed results (e.g., the IC analysis results) are never overwritten by a smaller demo run.

---

## 3) Quick demo configuration

At the top of [`quick_demo.py`](quick_demo.py), set the flags you want:

```python
analysis_options = {
    'light_mode': True,           # True: fast/small versions. False: full scale.

    # No external data required
    'run_model_demos':              True,  # All 480 utility equations; Bayesian core checks.
    'run_nesting_tests':            True,  # Model nesting adjacency; equivalence and embedding checks.

    # Synthetic simulation data
    'run_simulation':               True,  # Parameter recovery simulation (Figures 5–6).
    'run_particle_filter_test':     True,  # Particle filter vs full-grid fidelity check.
    'run_recovery_by_k':            True,  # Recovery accuracy across k-param model complexity.
    'run_update_speed_analysis':    True,  # Belief update speed regression.
    'visualize_belief_updates':     True,  # Interactive 3D Bayesian update plots.

    # Requires raw experiment data in raw_data/
    'run_model_comparison':         False, # Alternative model contest + typological comparison.
    'run_ic_analysis':              False, # IC utility comparison (5 forms light, 480 full).
    'run_parameter_distribution':   False, # Population parameter distributions.
    'run_inequality_aversion':      False, # Inequality aversion bot competition heatmaps.
}
```

Key rules:
- `light_mode=True` is recommended for fast debugging. It reduces particle counts, bin counts,
  game counts, and player counts across every section.
- `visualize_belief_updates` and `run_update_speed_analysis` require `run_simulation` to have
  produced data first (either in the same run or a prior run).
- `run_ic_analysis` with `light_mode=False` runs all 480 models and takes **weeks**.
  With `light_mode=True` it runs 5 representative forms (one per k level, k=1..5).
- Sections 8–11 check whether the required raw CSV files exist. If any are missing a loud
  warning is printed (missing raw data is treated as an error, not a normal skip).

---

## 4) What to look at after running

### Sections 1–2 (utility model + nesting): terminal output
The equations for all 480 utility forms print to the terminal, along with nesting relationship
counts, parent-child classification examples, and pass/fail results from the equivalence
and embedding sanity checks.

### Section 3 (parameter recovery simulation): HTML figures
Open these in a browser:

- **Violin of recovery correlations**:
  [`demo_files/player_fits/simulation_results/corr_violin_Vij_first.html`](demo_files/player_fits/simulation_results/corr_violin_Vij_first.html)

- **Correlation vs. round** (belief accuracy improving over sequential games):
  [`demo_files/player_fits/simulation_results/corr_by_round_sim_pred.html`](demo_files/player_fits/simulation_results/corr_by_round_sim_pred.html)
  — select **μ(Vij)** in the dropdown.

### Section 7 (3D belief update visualization): HTML figures
Folder: [`demo_files/visuals/bayesian_updates_3d/`](demo_files/visuals/bayesian_updates_3d/)

Open any HTML file. In the interactive plot:
- **Left surface** = the model's **prior** over the target parameters.
- **Right surface** = the **likelihood** (choice probability for the option actually chosen).
- **Use the slider** to step through rounds and watch the posterior concentrate.

### Sections 8–11 (real-data analyses): outputs in `demo_files/`
All outputs from these sections go to `demo_files/` to protect any precomputed results in the
main output directories. This includes IC results in `demo_files/bic_aic/`, fitted parameters
in `demo_files/player_fits/`, and visualizations in `demo_files/visuals/`.

---

## 5) Codebase structure

- [`quick_demo.py`](quick_demo.py) — configurable demo entry point (all 11 analysis sections).
- [`main.py`](main.py) — thin entry point for the full pipeline: `run_code_settings` flags + `main()`.
- [`config.py`](config.py) — all settings, paths, type aliases, and parameter specs.
- [`model.py`](model.py) — utility functions, softmax choice model, `build_utility_equation`.
- [`optimization.py`](optimization.py) — optimization helpers + full MLE pipeline (`run_analysis_mle`).
- [`bayesian.py`](bayesian.py) — the core UBM: `bayesian_update_grid`, `agent`, `run_analysis_bayes`.
- [`simulation.py`](simulation.py) — parameter recovery simulations and particle filter validation.
- [`visualization.py`](visualization.py) — belief-update visualizations (2D, 3D, accuracy).
- [`analysis.py`](analysis.py) — model comparison, IC analysis, nesting verification, parameter
  distributions, inequality aversion.
- [`utilities.py`](utilities.py) — utility-function enumeration, nesting classification, penalties.
- [`preprocessing.py`](preprocessing.py) — data loading, dyad construction, experiment cleaning.
- [`typological.py`](typological.py) — discrete/typological Bayesian variants.

**Import chain (no circular dependencies):**
`config → preprocessing / utilities / typological → model → optimization → bayesian → simulation → visualization → analysis → main`

Additional docs:
- [`docs/data_dictionary.md`](docs/data_dictionary.md) — data structures and file outputs.
- [`docs/architecture.md`](docs/architecture.md) — high-level architecture notes.
- [`docs/core_function_map.md`](docs/core_function_map.md) — guided map of core functions.
- [`Agents.md`](Agents.md) — AI collaborator orientation (coding style, domain vocabulary, architecture).
- [`ihm_starter_pack.md`](ihm_starter_pack.md) — compact briefing on the paper's model and results.

---

## 6) Pointers to core implementations

### Utility → choice likelihood
- `utility_term` (building blocks): [`model.py`](model.py)
- `utility` (full utility function): [`model.py`](model.py)
- `softmax_` (probabilistic choice rule): [`model.py`](model.py)
- `choice` (wraps softmax for option A/B): [`model.py`](model.py)
- `build_utility_equation` (string form): [`model.py`](model.py)

### Utility Bayesian Model (UBM)
- `prior_grid_from_params` (construct priors over latent traits): [`bayesian.py`](bayesian.py)
- `bayesian_update_grid` (grid / particle-filter posterior update): [`bayesian.py`](bayesian.py)
- `agent` (self-perpetuating UBM over sequential games): [`bayesian.py`](bayesian.py)

### Optimization and loss
- `global_local_optimization` (simulated annealing → local refinement): [`optimization.py`](optimization.py)
- `loss_function_bayes` (NLL from UBM predictions): [`bayesian.py`](bayesian.py)
- `fit_params_by_player` (fit one player across all dyads): [`bayesian.py`](bayesian.py)
- `run_analysis_bayes` (full fitting pipeline, multiprocessing): [`bayesian.py`](bayesian.py)

### Simulation and validation
- `create_simulated_data`: [`simulation.py`](simulation.py)
- `run_simulation_recovery_analysis`: [`simulation.py`](simulation.py)
- `verify_particle_filter_fidelity`: [`simulation.py`](simulation.py)
- `run_param_recovery_by_k`: [`simulation.py`](simulation.py)

### Model nesting and IC analysis
- `model_nesting_adjacency_matrices`: [`analysis.py`](analysis.py)
- `run_child_parent_embedding_sanity_checks`: [`analysis.py`](analysis.py)
- `run_child_parent_probability_equivalence_smoketest`: [`analysis.py`](analysis.py)
- `information_criterion_analysis`: [`analysis.py`](analysis.py)

---

## 7) Where to find key details in the paper (arXiv PDF)

- **UBM formulation**: Section 3.3, especially 3.3.2–3.3.5.
- **Simulation methods and results**: Section 3.3.6 (Figures 5–6; `light_mode` will differ slightly).
- **Belief-update visualization**: Figure 4 (the demo generates interactive 3D versions).
- **IC analysis and nesting**: Section 4, especially 4.4.
- **Parameter distributions**: Section 5.

---

## 8) Full IC model comparison output (precomputed)

The large-scale IC comparison across 480 utility functions is too compute-intensive to run in the
quick demo (months on the paper's settings). Results are included as:

[`bic_aic/All_Utility_Forms_IC_Analysis_Experiment3.csv`](bic_aic/All_Utility_Forms_IC_Analysis_Experiment3.csv)

Key columns: *equation*, *loss*, *k_params*, *AIC*, *BIC*, *ΔAIC*, *ΔBIC*, *AIC_rank*, *BIC_rank*.
Each row is one utility-function specification defined by its boolean `utility_settings` columns.
Sort by *BIC* ascending for the overall ranking. The generating function is
`information_criterion_analysis` in [`analysis.py`](analysis.py).
