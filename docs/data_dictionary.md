# Data Dictionary — Inferring Hidden Motives (coding sheet)

This document maps explains core variables and data types.

> Version: 0.1 • Last updated: [12/12/2025] • Maintainer: Greg Stanley 


----------------------------------------------------------------------------------------------------

# Directory Structure

- **raw_data**: 
  * Contains all the individual raw data files for experiment 3.
- **processed**:
  * All processed data files.
- **player_fits**:
  * JSON files with fitted parameters for each player and the parameter updates across the rounds
  - experiment_0: Fitted parameters for the parameter recovery simulation
  - experiment_1: Fitted parameters for experiment 1 data (unused)
  - experiment_2: Fitted parameters for experiment 2 data
  - experiment_3: Fitted parameters for experiment 3 data
  - loss_reports: CSV files mapping parameter sets to losses per player
  - simulation_results: html figures and CSV tables of processed simulation data
- **param_data**:
  * Data produced by run_analysis_bayes or run_analysis_mle
- **dyad_data**:
  * Data produced by run_analysis_bayes or run_analysis_mle

----------------------------------------------------------------------------------------------------

## Dyad Game Structure
- `Key`: "(uuidA, uuidB)" is the dyad key (ordered tuple string).
- `Value`: A list of games between those two players, each with:
  - `chooser` / `predictor`: UUIDs for roles
  - `matching_probability`: dyad continuation probability in the next round
  - `payoff_A_*`, `payoff_B_*`: payoffs for options A/B to each player
  - `choice`, `prediction`: options chosen/predicted
  - `abdicated_chooser`, abdicated_predictor: timeouts/abstentions
  - `timestamp`: Unix epoch seconds
  - `round`: round index in the experiment

# Dyad histories (Pairs file)

> Example from Social_Preference_Prediction_Pairs_Exper3.json) for how games are stored:

```python
"histories": {
    "(uuidA, uuidB)": [
        {
            "chooser": "uuidA",
            "predictor": "uuidB",
            "matching_probability": 0.23180873180873182,
            "payoff_A_chooser": 3,
            "payoff_A_predictor": 5,
            "payoff_B_chooser": 2,
            "payoff_B_predictor": 2,
            "choice": "A",
            "prediction": "A",
            "abdicated_chooser": false,
            "abdicated_predictor": false,
            "timestamp": 1732648731,
            "round": 0
        },
        {
            "chooser": "uuidB",
            "predictor": "uuidA",
            "matching_probability": 0.23180873180873182,
            "payoff_A_chooser": 1,
            "payoff_A_predictor": 5,
            "payoff_B_chooser": 1,
            "payoff_B_predictor": 5,
            "choice": "B",
            "prediction": "B",
            "abdicated_chooser": false,
            "abdicated_predictor": false,
            "timestamp": 1732648807,
            "round": 7
        },
        ...
    ],
    "(uuidC, uuidD)": [ ... ],
},
"player_info": {
    "uuidA": {
        "player_type": "robot",
        "avatar_shape": "hour-glass",
        "avatar_color": "hsla(18, 67%, 52%, 1.0)"
    },
    "uuidB": {
        "player_type": "participant",
        "avatar_shape": "round-square",
        "avatar_color": "hsla(196, 92%, 41%, 1.0)"
    },
    "uuidC": {
        "player_type": "robot",
        "avatar_shape": "ghost",
        "avatar_color": "hsla(78, 58%, 59%, 1.0)"
    },
    ...
}
```

----------------------------------------------------------------------------------------------------

Several top-level structures get passed into most functions:

- `general_settings`: how to fit and update (analysis mode, optimizer, learning method).
- `utility_settings`: which terms appear in the utility equation.
- `param_bds` / `param_info`: parameter bounds and derived metadata.
- `fig_lay`: plotting/visual style.
- `file_paths`: all folders and canonical filenames.

Understanding these is essential for navigating the code.

# general_settings
A wide variety of crucial variables that are more organized when kept together:
```python
general_settings: GeneralSettings = {
    'update_method': update_method,
    'analysis_mode': analysis_mode,
    'analysis_unit': analysis_unit,
    'experiment_num': experiment_num,
    'loss_funct_type': loss_funct_type,
    'track_evolution': track_evolution,
    'create_new_file': create_new_file,
    'run_in_parallel': run_in_parallel,
    'include_covariance': include_covariance,
    'softmax_temperature': softmax_temperature,
    'optimization_method': optimization_method,
    'confidence_weighted': confidence_weighted,
    'use_particle_filter': use_particle_filter,
    'guess_params_randomly': guess_params_randomly,
    'temperature_is_param': temperature_is_param,
    'n_bins_per_dimension': n_bins_per_dimension,
    'optimization_policy': optimization_policy,
    'fit_roles_together': fit_roles_together,
    'use_initial_params': use_initial_params,
    'warmstart_policy': warmstart_policy,
    'penalty_weight': penalty_weight,
    'learning_rate': learning_rate,
    'sample_ratio': sample_ratio,
    'export_fig': export_fig,
    'write_mode': write_mode,
    'dark_mode': dark_mode,
}
warmstart_policy = {
    "enabled": True,
    "schedule": "binary",
    "cold_iters": 4,
    "explore_iters": 4,
    "temperature_low": 0.01,
    "temperature_high": 1000.0,
    "disable_dual_annealing_when_warm": True,
}
optimization_policy = {
    'n_random_starts': 1,
    'maxiter_global': 36, 'maxfun_global': 36,
    'maxiter_local': 24,  'maxfun_local': 24,
    'run_trust_constr': False,
    'dual_annealing_seed': None,
    'trust_maxiter': 600,
    'trust_gtol': 1e-6,
    'trust_xtol': 1e-8,
    'trust_verbose': False,
    'local_methods': ['L-BFGS-B'],
}
```
- **update_method**: 
  - `'grid'`: Full grid-based updating or particle filter-based updating if use_particle_filter, 
  - `'naive'`: Non-Bayesian. Only uses likelihood term as a posterior, no priors, 
  - `'no_learning'`: Non-Bayesian. Only uses priors as a posterior, no likelihood,
  - `'parametric'`: Assumes normally distributed hypothesis spaces, not used in paper
  - `'mcmc'`: legacy, not used in paper,
- **analysis_mode**: 
  - `'bayesian'`: Uses Utility Bayesan Model 
  - `'mle'`: Uses maximum likelihood estimation (not in the paper)
- **analysis_unit**: 
  - `'player'`: Fit parameters for each player using all dyads they appear in,
  - `'dyad'` Fit parameters per dyad, which is a series of games between two players
- **experiment_num**:
    1. : simulation (parameter recovery, convergence of posteriors, and learning rate)
    2. : 456 participant study with two games per dyad (not fully exploited in the paper)
    3. : human–bot experiment, where participants played with four avatars, each with distinct utility functions
    4. : human–human experiment, where humans play other humans and is the main focus for the Hidden Motives paper
- **loss_funct_type**: 
  - `'log'` for negative log-likelihood, 
  - `'ssr'` for sum of squared residuals.
- **track_evolution**: 
  - In MLE mode, if True, fits parameters as they evolve across dyads rather than assuming they are fixed.
- **create_new_file**: 
  - If False, reuse existing files when present; if True, regenerate and overwrite.
- **run_in_parallel**: 
  - Controls multiprocessing; True uses multiple cores where possible.
- **include_covariance**: 
  - Whether to include covariance terms between parameters (experimental).
- **softmax_temperature**: 
  - Softmax temperature value when not fitting it as a parameter.
- **optimization_method**: 
  - 'global', 'local', or 'globloc'; 'globloc' runs a global optimizer followed by a local refinement.
- **confidence_weighted**: 
  - When True, weights loss by inverse posterior variance; confident correct predictions are rewarded and confident mistakes penalized more heavily.
- **use_particle_filter**: 
  - True uses particle filtering on discrete PMFs; False uses full grid updating. Use when update_method = 'grid'.
- **guess_params_randomly**: 
  - If False, may try to reuse previous best guesses (warm starts) where implemented.
- **temperature_is_param**: 
  - True if softmax temperature is a free parameter to fit.
- **n_bins_per_dimension**: 
  - Number of grid bins per parameter dimension; controls resolution of PMFs.
- **optimization_policy**: 
  - Dictionary specifying global/local iteration and function evaluation budgets and trust-region settings (see below).
- **fit_roles_together**: 
  - Intended to fit both roles for a participant jointly (not active yet because in the current model chooser and predictor parameters do not interact).
- **use_initial_params**: 
  - Determines whether figures/reporting use prior (True) or posterior (False) parameters.
- **warmstart_policy**: 
  - Configuration for warm-start scheduling (e.g., using child model parameters as parent initializations in IC robustness analyses; see below).
- **penalty_weight**: 
  - Regularization weight multiplying parameter penalties during optimization.
- **learning_rate**: 
  - Step size for parametric updating (analogous to learning rate in gradient-based methods; in grid/particle modes, update speed is instead shaped by prior variance and temperature).
- **sample_ratio**: 
  - Proportion of grid/particle cells sampled for updates (trade-off between precision and speed).
- **export_fig**: 
  - If True, exports Plotly figures to visuals/; otherwise, can show in browser only.
- **write_mode**: 
  - 'overwrite', 'resume', or 'readonly'; used primarily in IC analysis to control file updating behavior.
- **dark_mode**: 
  - Toggles figure template and font colors (dark vs. light theme).
- **ampd_settings**: nested dict controlling AMPD (Average Model Policy Distance) computation:
  - `metric`: distance metric, default `'normalized_jsd'`
  - `n_games`: number of simulated games per comparison, default 625
  - `n_iters`: number of Monte Carlo iterations, default 30 (use 250 for full-precision)
  - `parameter_sampling_mode`: `'uniform'` samples from param_bds; `'fitted'` uses posterior estimates
  - `parameter_pairing_mode`: `'shared'` pairs same parameter draw across both models
  - `player_roles`: `None` or a list of roles to include
  - `random_seed`: reproducibility seed; `None` = unseeded
- **individual_architecture_settings**: nested dict for the architecture compression curve (how many structurally distinct utility types describe the population?):
  - `population_top_n_models`: top-N models by aggregate BIC to include as candidates (default 120; `None` = all)
  - `participant_top_r_models`: top-R models per participant added to the candidate set (default 10)
  - `K_max`: hard ceiling on K; `None` runs until stopping criterion fires
  - `exhaustive_K_max`: exhaustive search for K ≤ this value; greedy+swap for larger K (default 4)
  - `score_basis`: `'ic_equivalent_participant_score'` (recommended), `'sum_individual_BIC'`, or `'raw_NLL'`
  - `stopping_criteria`: which criterion is highlighted; all five always computed (`'kneedle_elbow'`, `'marginal_gain'`, `'cumulative_gain'`, `'max_curvature'`, `'meta_bic'`)
  - `marginal_gain_threshold`: threshold for marginal gain criterion (default 0.01)
  - `n_consecutive_low_marginal_gains_required`: consecutive low-gain increments before stopping (default 1)
  - `cumulative_gain_threshold`: cumulative gain threshold (default 0.80)
  - `diagnose_selected_library_redundancy`: whether to compute per-architecture redundancy flags (default True)
  - `n_workers`: number of parallel workers for exhaustive search; `None` = cpu_count - 1
- **model_recovery_settings**: nested dict for the model recovery simulation (IC pipeline data adequacy):
  - `generating_model`: `utility_idx` of the model used to generate synthetic data (default 443)
  - `n_agents_grid`: list of synthetic-participant counts to test (default [73])
  - `n_games_grid`: list of game counts to test (default [20, 40, 60, 90, 120, 180, 240])
  - `softmax_temperature`: fixed τ for both data generation and NLL fitting (default 0.5)
  - `candidate_model_selection_mode`: `'hamming'` or `'ampd'`; diversity method for candidate set
  - `n_candidate_models`: number of candidate models for IC comparison (default 480)
  - `ampd_matrix_name_or_path`: path to a precomputed AMPD matrix; `None` = compute on demand
  - `random_seed`: reproducibility seed (default 42)


# utility_settings
These boolean flags determine the funtional form of the utility function, including the number of free parameters:
```python
utility_settings: UtilitySettings = {
    'conditional_welfare_mode':       False,
    'reference_dependent_altruism':   False,
    'min_max_rawlsian_leontief':      False,
    'use_exponential_parameters':     True,
    'single_exponential_parameter':   False,
    'apply_exponents_to_payoffs':     False,
    'single_payoffs_not_differences': False,
    'payoff_ratios_not_differences':  False,
    'reference_dependent_utility':    False,
    'use_negativity_parameters':      False,
    'negativity_social_comparison':   True,
    'fix_self_interest_parameter':    False,
    'include_social_comparison':      True,
    'include_altruism_term':          True,
}
```
1.  **include_altruism_term**: 
    - Include an other-regarding payoff weight 𝑉𝑖𝑗.
2.  **include_social_comparison**: 
    - Include a social comparison term (envy/guilt, disparity components).
3.  **use_exponential_parameters**: 
    - Exponentiate some parameters (e.g., curvature) to ensure positivity or shape constraints.
4.  **single_exponential_parameter**: 
    - Use one shared exponent vs. separate exponents for different terms.
5.  **apply_exponents_to_payoffs**: 
    - If True, apply exponents to payoff magnitudes; otherwise to parameters.
6.  **single_payoffs_not_differences**: 
    - Base utility on raw payoff levels instead of payoff differences.
7.  **payoff_ratios_not_differences**: 
    - Use ratios rather than differences for some payoff components.
8.  **reference_dependent_utility**: 
    - Treat utility as gains/losses around a reference point.
9.  **reference_dependent_altruism**: 
    - Make altruistic utility reference-dependent.
10. **use_negativity_parameters**: 
    - Separate parameters for negative deviations (loss weighting).
11. **negativity_social_comparison**: 
    - Apply negativity parameters specifically to comparison terms.
12. **conditional_welfare_mode**: 
    - Restrict welfare computations to certain outcome subsets (e.g., conditional on signs or thresholds).
13. **min_max_rawlsian_leontief**: 
    - Use a Rawls/Leontief-style aggregation across players/terms (min/max operations).
14. **fix_self_interest_parameter**: 
    - If True, keep the self-interest weight fixed at a chosen value.

# param_info and param_bds
param_info stores the variable key names, random initial guesses, and parameter bounds. 
make_param_info() uses utility_settings to determine how many parameters will be in param_info
```python
param_bds: ParameterBounds = {
    'Vᵢᵢ': (-1, 1), 'Ʌᵢᵢ': (-1, 1), 'Vᵢⱼ': (-1, 1), 'Ʌᵢⱼ': (-1, 1), 'Ƹᵢⱼ': (-1, 1), 
    'Ʒᵢⱼ': (-1, 1), 'γ1': (1e-4, 2), 'γ2': (1e-4, 2), 'γ3': (1e-4, 2),
    'Vᵢᵢ_std': (1e-2, 4), 'Ʌᵢᵢ_std': (1e-2, 4), 'Vᵢⱼ_std': (1e-2, 4), 
    'Ʌᵢⱼ_std': (1e-2, 4), 'Ƹᵢⱼ_std': (1e-2, 4), 'Ʒᵢⱼ_std': (1e-2, 4),
    'γ1_std': (1e-2, 1), 'γ2_std': (1e-2, 1), 'γ3_std': (1e-2, 1),
}

param_info = {
    'param_keys': ['Vᵢᵢ', 'Vᵢⱼ', 'Ƹᵢⱼ'],
    'param_guesses': [0.453, -0.022, 0.398],
    'param_bds': param_bds
}
```

param_bds stores lower/upper bounds for each parameter. 
# Parameters are:
- `Vᵢᵢ`: self-interest weight
- `Vᵢⱼ`: altruism weight
- `Ʌᵢᵢ`, `Ʌᵢⱼ`: loss/negativity weights for self and other
- `Ʒᵢⱼ`: normal envy (disutility when other > self)
- `Ƹᵢⱼ`: reverse envy/guilt (disutility when self > other)
- `γ1`, `γ2`, `γ3`: exponents/sharpness parameters used in some utility forms
- `*_std`: prior standard deviations for corresponding parameters (used in Bayesian grid/particle representations)


Plot layout (fig_lay)
```python
txt_color = "white" if dark_mode else "black"
txtfam = "Calibri"

fig_lay: FigLay = {
    "template": "plotly_dark" if dark_mode else "plotly_white",
    "font": dict(family=txtfam, color=txt_color, size=24),
    "tickfont": dict(family=txtfam, color=txt_color, size=30),
    "titlefont_size": 48, "title_x": 0.5, "title_y": 0.96, "scale": ("x", 1),
    "colorscales": ['Viridis', 'Plasma', 'Inferno', 'matter', 'haline', 'thermal', 'dense', 'Magma'],
    "annotations": {"font":  dict(family=txtfam, color=txt_color, size=34), "showarrow": False},
    "xaxis" : {"title_font": dict(family=txtfam, color=txt_color, size=34),
               "tickfont": dict(size=30, family=txtfam, color=txt_color)},
    "yaxis" : {"title_font": dict(family=txtfam, color=txt_color, size=34),
               "tickfont": dict(size=30, family=txtfam, color=txt_color)},
    "hoverlabel": dict(font_size=30, font_family=txtfam),
    "markersize": 16
}
```
> This is a central style dict passed to plotting helpers; it standardizes fonts, colors, tick sizes, and color scales across all generated figures.

# File paths & canonical filenames
```python
ROOT = Path(__file__).resolve().parent

file_paths: FilePaths = {
    "raw_data":    ROOT / "raw_data",
    "processed":   ROOT / "processed",
    "param_data":  ROOT / "param_data",
    "player_fits": ROOT / "player_fits",
    "dyad_data":   ROOT / "dyad_data",
    "discrete":    ROOT / "discrete",
    "visuals":     ROOT / "visuals",
    "bic_aic":     ROOT / "bic_aic",
    "file_names": {
        "player_pairs_exper0": "Social_Preference_Prediction_Pairs_Exper0.json",
        "player_pairs_exper1": "Social_Preference_Prediction_Pairs_Exper1.json",
        "player_pairs_exper2": "Social_Preference_Prediction_Pairs_Exper2.json",
        "player_pairs_exper3": "Social_Preference_Prediction_Pairs_Exper3.json",
        "processed_data_exper0": "Social_Preference_Prediction_Processed_Results_Exper0.csv",
        "processed_data_exper1": "Social_Preference_Prediction_Processed_Results_Exper1.csv",
        "processed_data_exper2": "Social_Preference_Prediction_Processed_Results_Exper2.csv",
        "processed_data_exper3": "Social_Preference_Prediction_Processed_Results_Exper3.csv",
        "params_data_exper1_iter": "Social_Preference_Prediction_Parameters_Exper1_Iter.json",
        "params_data_exper2_iter": "Social_Preference_Prediction_Parameters_Exper2_Iter.json",
        "params_data_exper3_iter": "Social_Preference_Prediction_Parameters_Exper3_Iter.json",
        "params_data_exper1_fit1": "Social_Preference_Prediction_Parameters_Exper1_Fit1.json",
        "params_data_exper2_fit1": "Social_Preference_Prediction_Parameters_Exper2_Fit1.json",
        "params_data_exper3_fit1": "Social_Preference_Prediction_Parameters_Exper3_Fit1.json",
        "params_hist_exper1_iter": "Social_Preference_Prediction_Parameters_Exper1_Iter.html",
        "params_hist_exper2_iter": "Social_Preference_Prediction_Parameters_Exper2_Iter.html",
        "params_hist_exper3_iter": "Social_Preference_Prediction_Parameters_Exper3_Iter.html",
        "params_hist_exper1_fit1": "Social_Preference_Prediction_Parameters_Exper1_Fit1.html",
        "params_hist_exper2_fit1": "Social_Preference_Prediction_Parameters_Exper2_Fit1.html",
        "params_hist_exper3_fit1": "Social_Preference_Prediction_Parameters_Exper3_Fit1.html",
        "params_data_exper0_bayes": "Social_Preference_Prediction_Parameters_Exper0_Bayes.json",
        "params_data_exper1_bayes": "Social_Preference_Prediction_Parameters_Exper1_Bayes.json",
        "params_data_exper2_bayes": "Social_Preference_Prediction_Parameters_Exper2_Bayes.json",
        "params_data_exper3_bayes": "Social_Preference_Prediction_Parameters_Exper3_Bayes.json",
        "params_hist_exper1_bayes": "Social_Preference_Prediction_Parameters_Exper1_Bayes.html",
        "params_hist_exper2_bayes": "Social_Preference_Prediction_Parameters_Exper2_Bayes.html",
        "params_hist_exper3_bayes": "Social_Preference_Prediction_Parameters_Exper3_Bayes.html",
        "all_fitted_data_bayesian": "Social_Preference_Prediction_Parameters_Bayes.csv",
        "all_fitted_data_naive":    "Social_Preference_Prediction_Parameters_Naive.csv",
        "all_fitted_data_mle":      "Social_Preference_Prediction_Parameters_MLE.csv",
        "raw_data_exper1": "Judgment_Game_Data_Experiments_1abcd_Post-exclusions.csv",
        "raw_data_exper2": "Judgment_Game_Data_Experiment_2_Pre-exclusions.csv",
        "raw_data_exper3": "Morality_Game_Iter_bDG_Results_Combined.csv",
        "players_to_dyads_exper0": "SPPP_Exper0_Players_to_Dyad_Keys.json",
        "players_to_dyads_exper1": "SPPP_Exper1_Players_to_Dyad_Keys.json",
        "players_to_dyads_exper2": "SPPP_Exper2_Players_to_Dyad_Keys.json",
        "players_to_dyads_exper3": "SPPP_Exper3_Players_to_Dyad_Keys.json",
        "information_criterion":   "All_Utility_Forms_IC_Analysis_Experiment3.csv",
        "problematic_pairs":       "parent_worse_than_child_pairs.csv",
        "embedding_equality":      "embedding_equality_results.csv",
        "all_minimal_pairs":       "minimal_pairs_summary.csv",
    }
}
```
> All paths are relative to the repository root (ROOT).
> file_name_suffix and add_remove_file_name_suffix(...) may add suffixes based on configuration (e.g., to distinguish runs with different utility settings).

----------------------------------------------------------------------------------------------------

# Type aliases & key composite types

> These help you reason about data structures passed between functions:

`PlayerUUID`      = str
`PlayerRole`      = Literal['chooser', 'predictor']
`DyadKey`         = str | tuple[PlayerUUID, PlayerUUID] | list[PlayerUUID]
`BoolTuple`       = Tuple[bool, ...]

`ParameterBounds` = dict[str, tuple[int | float, int | float]]
`GeneralSettings` = dict[str, str | int | float | bool]
`UtilitySettings` = dict[str, bool]

`FigLay`          = dict[str, Any]
`ColumnNames `    = dict[str, list[str] | dict[str, str]]

`ParamKeys `      = list[str]
`ParamBounds `    = list[tuple[int | float]]
`ParamGuesses`    = Callable[[], Dict[str, float]]
`ParamCovar`      = list[str]
`ParamInfo `      = dict[str, ParamKeys | ParamBounds | ParamGuesses | ParamCovar]

`CovMat`          = NDArray[np.float64] | None
`CovMatDict`      = dict[str, dict[str, dict[str, CovMat]]] | None

`Output`          = dict[str, float]
`Params`          = dict[str, float]
`ParamVectors`    = dict[tuple[int] | str, float]
`MetaData`        = dict[str, int | float | dict[str, list[float]]]
`ParamEst`        = dict[str, dict[str, dict[str, dict[str, Output | Params | MetaData | ParamVectors]]]]

`DyadGame`        = dict[str, str | int | bool | ParamEst]
`DyadGames`       = list[DyadGame]

`PlayerInfo`      = dict[str, dict[str, str]]
`Histories`       = dict[str, DyadGames | PlayerInfo]
`ParamVals`       = dict[PlayerUUID, dict[PlayerRole, Params]]

`PlayerDyads`     = dict[DyadKey, DyadGames]

`WriteMode` = Literal["readonly", "resume", "overwrite"]

----------------------------------------------------------------------------------------------------

# Morality Game raw trees 

> Raw JSON files contain lists of game trees. Each tree is a nested object (root + recursive options) capturing the full strategic state and all responses recorded at that node. 

Tree schema (example, abridged)
```python
{
    "idnum": 0,
    "parent": -1,
    "label": "",
    "payoffs": [0, 0],
    "chooser": [True, False],
    "choice": [
        {
            "option": "A", 
            "keypress": "select-node", 
            "rtimedn": 21.404, 
            "rtimeup": 24.512, 
            "timestamp": 1732648720
        },
        None
    ],
    "predictor": [False, True],
    "prediction": [
        None,
        {
            "option": "A", 
            "keypress": "select-node", 
            "rtimedn": 6.312, 
            "rtimeup": 9.891, 
            "timestamp": 1732648723
        }
    ],
    "probability": [1.0, 1.0],
    "positionxy": [0.5, 1.0],
    "time": [0, 12],
    "info_set": [[0]],
    "beliefs": {},
    "options": [ ... child nodes ... ],
    "adjacency_matrix": [
        [1.0, 0.23180873180873182],
        [0.23180873180873182, 1.0]
    ],
    "timestamps": {
        "emit_tree_time": 1732648722.877636,
        "update_node_time": 1732648722.877636,
        "abdicate_node_time": 1732648734.877636,
        "round_deadline_time": 1732648736.417223,
        "round_started_time": 1732648715.417223,
        "round_ended_time": None
    },
    "players": [
        {
            "user_number": 0,
            "uuid": "6e82ab61-3e9b-44ed-94f5-d85668628ea5",
            "sid": "_CReVqgxS-e_eyayAAIx",
            "avatar": {
                "shape": "hour-glass",
                "color": "hsla(18, 67%, 52%, 1.0)",
                "texture": "swirling_space.png"
            },
            "cumulative_payoffs": 133,
            "player_type": "participant"
        },
        {
            "user_number": 2,
            "uuid": "3c48f1f3-63f8-4276-b96c-fc478c875973",
            "sid": "s8JrYx5I8rsNR6AqAAIz",
            "avatar": {
                "shape": "round-square",
                "color": "hsla(196, 100%, 50%, 1.00)",
                "texture": "oil_bubbles.png"
            },
            "cumulative_payoffs": 135,
            "player_type": "participant"
        }
    ],
    "avatar_colors": [
        [18, 67, 52, 1.0],
        [196, 92, 41, 1.0],
        [287, 50, 50, 1.0]
    ],
    "round_room_batch": [0, 0, 91, 1732648731],
    "title": "BDG-356-3522",
    "current_nodeid": 1,
    "nplayers": 2
}
```

# Field meanings
**Key**	           **Type**	              **Meaning**
`idnum`	           int	                  Node identifier (root typically 0).
`parent`	       int	                  Parent node id (-1 for root).
`label`	           str	                  Human-readable node label (often option label "A", "B" on children).
`payoffs`          list[float]	          Payoff vector at this node (per player).
`chooser`	       list[bool]	          Length = nplayers. Who has choice control at this node.
`choice`	       list[dict | None]	  Per-player choice record (if chooser): option, keypress, rtimedn, rtimeup, timestamp. Non-choosers: null.
`predictor`	       list[bool]	          Length = nplayers. Who is predicting at this node.
`prediction`	   list[dict | None]	  Per-player prediction record (if predictor). Same structure as choice.
`probability`	   list[float]	          For chance nodes, encodes transition probabilities to children.
`positionxy`	   [float, float]	      UI layout coordinate for rendering.
`time`	           [start, end]	          Node’s active time window within the round (seconds).
`info_set`	       list[list[int]]	      Information set indices (for simultaneous / imperfect information).
`beliefs`	       dict	                  Nested belief storage (often empty in this dataset).
`options`	       list[node]	          Child nodes (same schema recursively).
`adjacency_matrix` n×n float	          Pairwise matching probabilities for the next round among players in this room.
`timestamps`	   dict	                  Absolute epoch times for server events (e.g., emit, update, abdicate, deadline, start/end).
`players`	       list[dict]	          Per-player metadata: uuid, avatar metadata, cumulative payoff, player_type.
`avatar_colors`	   list[list[int, float]] HSLA color values for avatars.
`round_room_batch` [int, int, int, float] Round index, room index, batch index, and timeslot timestamp.
`title`	           str	                  Game identifier (often includes task code).
`current_nodeid`   int	                  Node that was current when the tree snapshot was emitted.
`nplayers`	       int	                  Number of players in this game.

Note: The Morality Game raw JSON files are just lists of these trees; preprocessing scripts collapse them into dyad-level and player-level records.

