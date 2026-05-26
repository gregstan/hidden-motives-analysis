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
  [`demo_files/simulations/corr_violin_Vij_first.html`](demo_files/simulations/corr_violin_Vij_first.html)

- **Correlation vs. round** (belief accuracy improving over sequential games):
  [`demo_files/simulations/corr_by_round_sim_pred.html`](demo_files/simulations/corr_by_round_sim_pred.html)
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

- [`quick_demo.py`](quick_demo.py) — configurable demo entry point (all 13 analysis sections).
- [`main.py`](main.py) — thin entry point for the full pipeline: `run_code_settings` flags + `main()`.
- [`config.py`](config.py) — all settings, paths, type aliases, and parameter specs.
- [`model.py`](model.py) — utility functions, softmax choice model, `build_utility_equation`.
- [`optimization.py`](optimization.py) — optimization helpers; `run_analysis_mle` is the legacy MLE pipeline (callable via `analysis_mode='mle'` but not used in the paper analyses).
- [`bayesian.py`](bayesian.py) — the core UBM: `bayesian_update_grid`, `agent`, `run_analysis_bayes`.
- [`simulation.py`](simulation.py) — parameter recovery simulations and particle filter validation.
- [`visualization.py`](visualization.py) — belief-update visualizations (2D, 3D, accuracy).
- [`analysis.py`](analysis.py) — all post-fitting analyses: model comparison, AMPD behavioral-distance
  matrix, IC analysis, nesting verification, parameter distributions, inequality aversion,
  individual architecture compression curve, model recovery simulation.
- [`utilities.py`](utilities.py) — utility-function enumeration, nesting classification, penalties.
- [`preprocessing.py`](preprocessing.py) — data loading, dyad construction, experiment cleaning.
- [`typological.py`](typological.py) — discrete/typological Bayesian variants.

**Import chain (no circular dependencies):**
`config → preprocessing / utilities / typological → model → optimization → bayesian → simulation → visualization → analysis → main`

Additional docs:
- [`docs/data_dictionary.md`](docs/data_dictionary.md) — data structures and file outputs.
- [`docs/architecture.md`](docs/architecture.md) — high-level architecture notes.
- [`docs/core_function_map.md`](docs/core_function_map.md) — guided map of core functions.
- [`AGENTS.md`](AGENTS.md) — full developer conventions: pipeline robustness patterns, coding style,
  domain vocabulary, architecture. **Read this before extending the codebase.**
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
- `verify_same_inputs_same_outputs_for_children_and_parents`: [`analysis.py`](analysis.py)
- `run_child_parent_probability_equivalence_smoketest`: [`analysis.py`](analysis.py)
- `information_criterion_analysis`: [`analysis.py`](analysis.py)

### Individual architecture compression curve
- `extract_participant_model_combined_fits`: [`analysis.py`](analysis.py)
- `compute_architecture_compression_curve`: [`analysis.py`](analysis.py)
- `plot_architecture_compression_curve`: [`analysis.py`](analysis.py)

### Model recovery simulation
- `compute_model_recovery_simulation`: [`analysis.py`](analysis.py)
- `plot_model_recovery_simulation`: [`analysis.py`](analysis.py)

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

**Dynamic updating.** By default the IC analysis uses a static (no belief-updating) version of the
UBM for computational tractability. A full dynamic version — fitting individual-level Bayesian
belief-updating parameters across all 73 participants for each of the 480 utility functions — is
available via the `dynamic_updating=True` argument, but is prohibitively slow on a typical machine
(months on 6 cores). In `main.py`, `dynamic_updating` is set automatically: it enables itself when
`general_settings['run_in_parallel']` is `True` **and** the machine has 10 or more logical CPU
cores (`multiprocessing.cpu_count() >= 10`). Do not attempt the dynamic version without a
high-core-count machine.

---

## 9) Two post-IC analyses: compression curve and model recovery

These analyses run after the IC comparison and are enabled via `run_code_settings` in
[`main.py`](main.py). Both read all their settings from nested dicts inside `general_settings`
in [`config.py`](config.py), so no parameters need to be passed at the call site in `main.py`.

### Individual architecture compression curve

**Research question:** *How many distinct utility function architectures does the population actually need?*

The IC analysis names the single best architecture for the whole population. But participants may
differ not just in preference *magnitudes* but in the *structure* of their utility function —
whether they incorporate inequality aversion, nonlinear payoff sensitivity, etc. The compression
curve asks how much is gained by allowing K distinct architectures, with each participant adopting
whichever fits them best.

A(K) measures the fraction of the fully individualized BIC advantage (every participant using their
own best architecture) recovered by a K-architecture library. The curve's knee — found by the
Kneedle elbow criterion (among four other stopping criteria, all always computed) — identifies the
minimum number of structurally distinct types needed to describe the population.

**Enable:** `run_code_settings['run_individual_architecture_analysis'] = True` in `main.py`.

**Configure** via `general_settings['individual_architecture_settings']` in `config.py`:

| Key | Default | Meaning |
|-----|---------|---------|
| `population_top_n_models` | `120` | Top-N models by aggregate IC BIC to include as candidates |
| `participant_top_r_models` | `10` | Top-R models per participant (participant-aware filter, never excludes any participant's personal best) |
| `K_max` | `None` | Hard ceiling on K; `None` runs until stopping criterion fires |
| `exhaustive_K_max` | `4` | Exhaustive combination search for K ≤ this; greedy + local-swap for larger K |
| `score_basis` | `'ic_equivalent_participant_score'` | Scoring basis; primary choice ensures K=1 matches the IC champion |
| `stopping_criteria` | `'kneedle_elbow'` | Which criterion is highlighted in output; all five are always computed and saved |

**Outputs** (in `processed/`):
- `population_architecture_curve.csv` — one row per K; A(K), ΔA(K), all stopping-criterion flags, AMPD library statistics.
- `population_architecture_assignments.csv` — one row per K × participant; which architecture each participant is assigned.
- `population_architecture_library_diagnostics.csv` — one row per K × selected architecture; pruning cost, assignment count, redundancy flags.
- `visuals/population_architecture_curve.html` — interactive Plotly figure.

**Prerequisite:** IC results must exist in `bic_aic/` (produced by `run_information_criterion_analysis`).

---

### Model recovery simulation

**Research question:** *How many games and participants are needed for the IC pipeline to reliably identify the generating utility model?*

Synthetic chooser data is generated from a known model (the IC winner by default) using realistic
parameter distributions extracted from the IC results. For a grid of (n_games × n_agents) conditions,
the full IC candidate comparison is run and eight recovery metrics are measured: whether the
generating model wins population BIC, its rank, the BIC gap to the runner-up, AMPD behavioral
distance to the actual winner, conditional Hamming distance, and more.

**Enable:** `run_code_settings['run_model_recovery_simulation'] = True` in `main.py`.

**Configure** via `general_settings['model_recovery_settings']` in `config.py`:

| Key | Default | Meaning |
|-----|---------|---------|
| `generating_model` | `443` | `utility_idx` of the model used to generate synthetic data |
| `n_agents_grid` | `[73]` | Synthetic-participant adequacy curve; max(grid) agents generated once |
| `n_games_grid` | `[20, 40, 60, 90, 120, 180, 240]` | Games-per-agent adequacy curve |
| `softmax_temperature` | `0.5` | Fixed τ for both data generation and NLL fitting |
| `candidate_model_selection_mode` | `'hamming'` | Candidate diversity method: `'hamming'` or `'ampd'` |
| `n_candidate_models` | `480` | Candidate set size for IC comparison |
| `random_seed` | `42` | Reproducibility seed |

**Outputs:**
- A parameter-encoded CSV in `processed/`, e.g. `model_recovery_gen=443_cands=480_hamming_tau=0p5_agents=73_games=20-40-60-90-120-180-240_seed=42.csv`.
- `visuals/model_recovery_simulation_443.html` — interactive Plotly figure; a dropdown selects all eight metrics (normalized to [0, 1] on a shared axis) or any individual metric.

**Prerequisite:** `bic_aic/All_Utility_Forms_IC_Analysis_Experiment3.json` must exist — it supplies the parameter distributions used for synthetic data generation.
