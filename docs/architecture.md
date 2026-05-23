# Architecture & Module Map

This codebase follows a "single switchboard" pattern: **`config.py`** defines paths and settings; **`main.py`** orchestrates analysis blocks in the order used in the manuscript.

## Core modules

- **`config.py`**
  - `file_paths`: repo-relative folders and canonical filenames (Path objects; no hardcoded absolutes)
  - `general_settings`: analysis mode (Bayesian/MLE), optimizer, grid/particle updates, parallel flags, plotting settings, and nested sub-dicts for AMPD, architecture curve, and model recovery
  - `utility_settings`: 14 boolean toggles that define the utility functional form
  - `param_info`, `param_bds`: parameter metadata and bounds
  - `RunCodeSettings`, `GeneralSettings`, `IndividualArchitectureSettings`, `ModelRecoverySettings`, `AmpdSettings`: TypedDicts for all major setting dicts
  - `ensure_directory_and_join()`: safe file path builder that creates directories on demand

- **`model.py`**
  - `utility_term`, `utility`: utility function (all 480 forms via `utility_settings`)
  - `softmax_`, `choice`: probabilistic choice rule and likelihood wrapper
  - `build_utility_equation`: human-readable string form of the active utility equation

- **`optimization.py`**
  - `global_local_optimization`: dual annealing → L-BFGS-B optimizer chain used throughout the project
  - `run_analysis_mle`: legacy MLE fitting pipeline (callable via `analysis_mode='mle'`; not used in paper analyses)

- **`bayesian.py`**
  - `prior_grid_from_params`: converts mean/std parameters to a discrete PMF grid
  - `bayesian_update_grid`: single-round posterior update (full grid or particle filter)
  - `agent`: the core UBM — sequential belief-updating over a dyad's games
  - `loss_function_bayes`: NLL + penalty; the objective minimized during fitting
  - `fit_params_by_player`, `run_analysis_bayes`: per-player and full-dataset Bayesian fitting pipeline

- **`simulation.py`**
  - `create_simulated_data`, `run_simulation_recovery_analysis`: parameter recovery simulation
  - `verify_particle_filter_fidelity`: particle filter vs. full-grid validation
  - `run_param_recovery_by_k`, `run_update_speed_simulation_regression`: supplementary simulations

- **`visualization.py`**
  - `visualize_bayesian_updates_2d`, `visualize_bayesian_updates_3d`: interactive belief-update plots
  - `belief_accuracy_analysis`, `plot_param_recovery_by_round`: accuracy and recovery figures

- **`analysis.py`**
  - Stages 5–12 post-fitting analyses: AMPD distance matrix, model-space geometry (MDS), IC extraction, individual architecture compression curve, model recovery simulation
  - `information_criterion_analysis`: large-scale IC comparison across 480 utility functions
  - `compute_architecture_compression_curve`, `plot_architecture_compression_curve`: how many structurally distinct utility types describe the population?
  - `compute_model_recovery_simulation`, `plot_model_recovery_simulation`: IC pipeline data-adequacy check
  - `model_nesting_adjacency_matrices`, `run_child_parent_embedding_sanity_checks`: nesting graph and validation

- **`utilities.py`** (imported as `gnrl`)
  - `generate_utility_settings`: enumerates all 480 valid utility forms
  - `is_valid_utility_settings`, `classify_pair_relation`, `parents_children_of`: model-space topology
  - `map_child_to_parent_special_param_info`: child→parent parameter transfer for warm-starting

- **`preprocessing.py`** (imported as `prep`)
  - Cleans and merges raw CSVs; writes standardized game-history JSONs to `processed/`

- **`typological.py`**
  - Discrete/typological Bayesian variants (parallel to the continuous UBM)

- **`mle.py`**
  - Legacy MLE fitting pipeline; still callable via `analysis_mode='mle'` in `general_settings` but not used in the paper analyses

## Import chain (no circular dependencies)

```
config → preprocessing / utilities / typological → model → optimization → bayesian → simulation → visualization → analysis → main
mle → optimization   [legacy; not imported by any active module except main.py]
```

## Key objects and how they flow

1. **Data** (`raw_data/` → `processed/`)
   Preprocessing makes uniform game-history JSONs for fitting and plots.

2. **Settings** (`config.py`)
   `general_settings`, `utility_settings`, `file_paths`, `param_bds` are passed into all major functions.

3. **Fitting** (Bayesian)
   Writes per-player parameter estimates to `player_fits/` (JSON) and unified CSVs to `processed/`.

4. **IC comparison and nesting**
   Enumerates 480 utility forms; writes IC tables to `bic_aic/`; nesting graph prevents violations.

5. **Post-IC analyses**
   Architecture compression curve and model recovery simulation read from `bic_aic/` and `processed/`; write outputs to `processed/` and `visuals/`.

6. **Figures**
   Written to `visuals/` as interactive Plotly HTML files.

## `run_code_settings` execution order

```python
run_code_settings = {
    'run_simulation_analyses':              ...,  # parameter recovery and particle filter
    'run_illustrate_belief_updates':        ...,  # 3D Bayesian belief update plots
    'run_alternative_model_contest':        ...,  # UBM vs. non-Bayesian alternatives
    'run_typological_bayesian_models':      ...,  # discrete-type Bayesian models
    'run_information_criterion_analysis':   ...,  # IC comparison across 480 utility forms
    'run_model_nesting_violation_analysis': ...,  # nesting graph + embedding sanity checks
    'run_individual_architecture_analysis': ...,  # compression curve (how many utility types?)
    'run_model_recovery_simulation':        ...,  # IC pipeline data-adequacy check
    'run_parameter_distribution_results':   ...,  # population parameter distributions
    'run_inequality_aversion_analysis':     ...,  # inequality aversion bot competition
}
```
