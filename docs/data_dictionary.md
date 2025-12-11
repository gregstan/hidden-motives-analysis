# Data Dictionary — Inferring Hidden Motives (coding sheet)

This document maps **files → columns → meanings**, along with value ranges, units, and codes.

> Version: 0.1 • Last updated: [TODAY] • Maintainer: Greg Stanley 

---

## File map (expected locations)

- **Raw inputs** (`raw_data/`)
  - `Judgment_Game_Data_Experiments_1abcd_Post-exclusions.csv`
  - `Judgment_Game_Data_Experiment_2_Pre-exclusions.csv`
  - `Morality_Game_Iter_bDG_Results_Combined.csv`
  - `SPPP_ExperX_Players_to_Dyad_Keys.json` (X ∈ {0,1,2,3})
  - `Social_Preference_Prediction_Pairs_ExperX.json` (X ∈ {0,1,2,3})

- **Processed** (`processed/`) — created by the pipeline
  - `Social_Preference_Prediction_Processed_Results_ExperX.csv`

- **Parameters / fits** (`param_data/`, `player_fits/`, `dyad_data/`) — created by the pipeline
  - `Social_Preference_Prediction_Parameters_Exper{N}_{Bayes|MLE|Iter|Fit1}.{csv|json}`
  - `All_Fitted_Data_{Bayesian|MLE}.csv`

- **IC tables** (`bic_aic/`) — created by the pipeline
  - `All_Utility_Forms_IC_Analysis_Experiment3.csv`

> The canonical filenames are also listed in `config.py` under `file_paths["file_names"]`. Keep them unchanged for reproducibility.

---

## Common columns (processed / unified tables)

| Column name | Type | Description | Units / Codes |
|---|---:|---|---|
| `experiment_num` | int | Experiment index (1, 2, 3; 0 for simulation) | {0,1,2,3} |
| `dyad_key` | str | Unique dyad identifier | e.g., `<p1>_<p2>` |
| `player_uuid_chooser` | str | Chooser’s anon ID | opaque ID |
| `player_uuid_predictor` | str | Predictor’s anon ID | opaque ID |
| `round_idx` | int | Round within dyad | 0‑based or 1‑based (state which) |
| `payoff_self` | int/float | Chooser payoff for chosen option | points (→ cents) |
| `payoff_other` | int/float | Counterpart payoff | points |
| `choice` | {A,B,...} | Chooser option taken | categorical |
| `prediction` | {A,B,...} | Predictor’s predicted option | categorical |
| `rt_ms` | float | Reaction time | ms |
| `belief_mean_Vij` | float | Predictor’s current mean belief about altruism parameter \(V_{ij}\) | [−1, 1] |
| `belief_std_Vij` | float | Predictor’s current std for \(V_{ij}\) | ≥ 0 |
| `tau` | float | Softmax temperature | > 0 |
| `loss_nll` | float | Negative log-likelihood for the fit | scalar |
| `BIC` / `AIC` | float | Information criteria for a candidate utility form | scalar |
| `utility_id` | str | Hash/string of the active utility form | e.g., `Vii+Vij+Ineq` |

> **Note:** adapt names to your actual columns; if you use different naming (e.g., `Vᵢᵢ`/`Vᵢⱼ`), include the exact Unicode names and ASCII aliases.

---

## Stimuli / game specification fields

If your raw or processed files include tree/game descriptors, define them here:

| Column | Description | Example |
|---|---|---|
| `tree_id` | Unique game-tree ID | `t_2025_0031` |
| `node_id` | Current node identifier | `n7` |
| `option_labels` | Option set at node | `["A", "B"]` |
| `expected_payoffs` | Payoff vectors for options | `{"A":[3,5], "B":[5,1]}` |
| `probabilities` | Chance node probabilities (if any) | `{"A": 0.7, "B": 0.3}` |

If stimuli are loaded from JSON in `raw_data/`, list the JSON schema or link the file and describe keys here.

---

## Codes & missing values

- Missing numeric: `NaN`
- Missing categorical: empty string or `NA` (state which)
- Abdications/timeouts: `[TODO: code used]`
- Exclusion flags: `[TODO: name/codes]`

---

## Transformations (high level)

- **Preprocessing:** row filtering by exclusion criteria; merges player↔dyad keys; computes round indices; harmonizes column names.
- **Fitting:** per‑player/per‑dyad Bayesian or MLE routines; outputs parameter JSON/CSV in `param_data/` and unified tables in `processed/`.
- **IC search:** enumerates 476 utility forms; writes IC tables to `bic_aic/`.
- **Figures:** Plotly HTMLs to `visuals/`.

For full details see `preprocessing.py` and `main.py` (section headers match the paper).

Data Dictionary & Core Data Structures

This document explains the main data formats and core runtime objects used in the Inferring Hidden Motives analysis code.

Morality Game raw trees 

Raw JSON files contain lists of game trees. Each tree is a nested object (root + recursive options) capturing the full strategic state and all responses recorded at that node. 

Tree schema (example, abridged)
{
  "idnum": 0,
  "parent": -1,
  "label": "",
  "payoffs": [0, 0],
  "chooser": [true, false],
  "choice": [
    {"option": "A", "keypress": "select-node", "rtimedn": 21.404, "rtimeup": 21.404, "timestamp": 1732648720},
    null
  ],
  "predictor": [false, true],
  "prediction": [
    null,
    {"option": "A", "keypress": "select-node", "rtimedn": 6.312, "rtimeup": 6.312, "timestamp": 1732648723}
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
    "round_ended_time": null
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

Field meanings
Key	Type	Meaning
idnum	int	Node identifier (root typically 0).
parent	int	Parent node id (-1 for root).
label	str	Human-readable node label (often option label "A", "B" on children).
payoffs	list[float]	Payoff vector at this node (per player).
chooser	list[bool]	Length = nplayers. Who has choice control at this node.
choice	list[dict | null]	Per-player choice record (if chooser): option, keypress, rtimedn, rtimeup, timestamp. Non-choosers: null.
predictor	list[bool]	Length = nplayers. Who is predicting at this node.
prediction	list[dict | null]	Per-player prediction record (if predictor). Same structure as choice.
probability	list[float]	For chance nodes, encodes transition probabilities to children.
positionxy	[float,float]	UI layout coordinate for rendering.
time	[start,end]	Node’s active time window within the round (seconds).
info_set	list[list[int]]	Information set indices (for simultaneous / imperfect information).
beliefs	dict	Nested belief storage (often empty in this dataset).
options	list[node]	Child nodes (same schema recursively).
adjacency_matrix	n×n float	Pairwise matching probabilities for the next round among players in this room.
timestamps	dict	Absolute epoch times for server events (e.g., emit, update, abdicate, deadline, start/end).
players	list[dict]	Per-player metadata: uuid, avatar metadata, cumulative payoff, player_type.
avatar_colors	list[list[int,float]]	HSLA color values for avatars.
round_room_batch	[round, room, batch, timeslot]	Round index, room index, batch index, and timeslot timestamp.
title	str	Game identifier (often includes task code).
current_nodeid	int	Node that was current when the tree snapshot was emitted.
nplayers	int	Number of players in this game.

Note: The Morality Game raw JSON files are just lists of these trees; preprocessing scripts collapse them into dyad-level and player-level records.

Dyad histories & player↔dyad mappings

Preprocessing organizes games into dyads, where a dyad is a pair of player UUIDs and their sequence of games. This is represented in the Social_Preference_Prediction_Pairs_ExperX.json files under a top-level histories dictionary.

Dyad histories (Pairs file)

Example (Social_Preference_Prediction_Pairs_Exper3.json):

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
  "(uuidC, uuidD)": [ ... ]
}


Key: "(uuidA, uuidB)" is the dyad key (ordered tuple string).

Value: A list of games between those two players, each with:

chooser / predictor: UUIDs for roles

matching_probability: dyad continuation probability in the next round

payoff_A_*, payoff_B_*: payoffs for options A/B to each player

choice, prediction: options chosen/predicted

abdicated_chooser, abdicated_predictor: timeouts/abstentions

timestamp: Unix epoch seconds

round: round index in the experiment

Player→dyad index (Players_to_Dyad_Keys file)

For player-level fits, we need to know which dyads each player participated in. SPPP_Exper3_Players_to_Dyad_Keys.json maps:

{
  "uuidA": [
    "(uuidA, uuidB)",
    "(uuidA, uuidC)"
  ],
  "uuidB": [
    "(uuidA, uuidB)",
    "(uuidB, uuidD)"
  ]
}


Keys are player UUIDs.

Values are lists of dyad keys (strings of the form "(uuid_i, uuid_j)").

For most analyses in the paper, parameters are fit per player using all dyads they appear in (see analysis_unit in general_settings).

Core runtime objects (config.py)

Several top-level structures get passed into most functions:

general_settings: how to fit and update (analysis mode, optimizer, learning method).

utility_settings: which terms appear in the utility equation.

param_bds / param_info: parameter bounds and derived metadata.

fig_lay: plotting/visual style.

file_paths: all folders and canonical filenames.

Understanding these is essential for navigating the code.

general_settings
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


Settings (current defaults and meaning)

update_method: 'grid' (grid/particle filter belief updates), 'parametric', 'mcmc' (legacy), 'naive' (posterior-only non-Bayesian baseline), 'no_learning' (priors held fixed, no updating).

analysis_mode: 'bayesian' (paper’s default) or 'mle' (max-likelihood variant).

analysis_unit: 'player' (fit parameters for each player using all dyads they appear in) or 'dyad' (fit per dyad).

experiment_num:

0: simulation

1: large lab experiment with avatars (not fully exploited in the paper)

2: human–bot experiment

3: human–human experiment (main focus for Hidden Motives)

loss_funct_type: 'log' for negative log-likelihood, 'ssr' for sum of squared residuals.

track_evolution: in MLE mode, if True, fits parameters as they evolve across dyads rather than assuming they are fixed.

create_new_file: if False, reuse existing files when present; if True, regenerate and overwrite.

run_in_parallel: controls multiprocessing; True uses multiple cores where possible.

include_covariance: whether to include covariance terms between parameters (experimental).

softmax_temperature: softmax temperature value when not fitting it as a parameter.

optimization_method: 'glob', 'loc', or 'globloc'; 'globloc' runs a global optimizer followed by a local refinement.

confidence_weighted: when True, weights loss by posterior confidence; confident correct predictions are rewarded and confident mistakes penalized more heavily.

use_particle_filter: True uses particle filtering on discrete PMFs; False uses full grid updating.

guess_params_randomly: if False, may try to reuse previous best guesses (warm starts) where implemented.

temperature_is_param: True if softmax temperature is a free parameter to fit.

n_bins_per_dimension: number of grid bins per parameter dimension; controls resolution of PMFs.

optimization_policy: dictionary specifying global/local iteration and function evaluation budgets and trust-region settings (see below).

fit_roles_together: intended to fit both roles for a participant jointly (not active yet because the current model factorizes chooser and predictor parameters).

use_initial_params: determines whether figures/reporting use prior (True) or posterior (False) parameters.

warmstart_policy: configuration for warm-start scheduling (e.g., using child model parameters as parent initializations in IC robustness analyses).

penalty_weight: regularization weight multiplying parameter penalties (e.g., for IC search).

learning_rate: step size for parametric updating (analogous to learning rate in gradient-based methods; in grid/particle modes, update speed is instead shaped by prior variance and temperature).

sample_ratio: proportion of grid/particle cells sampled for updates (trade-off between precision and speed).

export_fig: if True, exports Plotly figures to visuals/; otherwise, can show in browser only.

write_mode: 'overwrite', 'resume', or 'readonly'; used primarily in IC analysis to control file updating behavior.

dark_mode: toggles figure template and font colors (dark vs. light theme).

Warm-start policy

warmstart_policy = {
    "enabled": True,
    "schedule": "binary",
    "cold_iters": 4,
    "explore_iters": 4,
    "temperature_low": 0.01,
    "temperature_high": 1000.0,
    "disable_dual_annealing_when_warm": True,
}


Controls how parameter warm-starts are used in IC robustness analyses (e.g., child → parent parameter passing). See paper’s IC robustness section for more context.

Optimization policy

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


Encapsulates tuning for SciPy optimizers used in both Bayesian and ML modes.

utility_settings
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


These flags determine which components appear in the utility equation under utility() / build_utility_equation():

include_altruism_term: include an other-regarding payoff weight 
𝑉
𝑖
𝑗
V
ij
	​

.

include_social_comparison: include a social comparison term (envy/guilt, disparity components).

use_exponential_parameters: exponentiate some parameters (e.g., curvature) to ensure positivity or shape constraints.

single_exponential_parameter: use one shared exponent vs. separate exponents for different terms.

apply_exponents_to_payoffs: if True, apply exponents to payoff magnitudes; otherwise to parameters.

single_payoffs_not_differences: base utility on raw payoff levels instead of payoff differences.

payoff_ratios_not_differences: use ratios rather than differences for some payoff components.

reference_dependent_utility: treat utility as gains/losses around a reference point.

reference_dependent_altruism: make altruistic utility reference-dependent.

use_negativity_parameters: separate parameters for negative deviations (loss weighting).

negativity_social_comparison: apply negativity parameters specifically to comparison terms.

conditional_welfare_mode: restrict welfare computations to certain outcome subsets (e.g., conditional on signs or thresholds).

min_max_rawlsian_leontief: use a Rawls/Leontief-style aggregation across players/terms (min/max operations).

fix_self_interest_parameter: if True, keep the self-interest weight fixed at a chosen value.

These toggles determine which entries from param_bds and param_info are actively used in the current run.

Parameter bounds & param_info
param_bds: ParameterBounds = {
    'Vᵢᵢ': (-1, 1), 'Ʌᵢᵢ': (-1, 1), 'Vᵢⱼ': (-1, 1), 'Ʌᵢⱼ': (-1, 1), 'Ƹᵢⱼ': (-1, 1), 'Ʒᵢⱼ': (-1, 1),
    'γ1': (1e-4, 2), 'γ2': (1e-4, 2), 'γ3': (1e-4, 2),
    'Vᵢᵢ_std': (1e-2, 4), 'Ʌᵢᵢ_std': (1e-2, 4), 'Vᵢⱼ_std': (1e-2, 4), 'Ʌᵢⱼ_std': (1e-2, 4),
    'Ƹᵢⱼ_std': (1e-2, 4), 'Ʒᵢⱼ_std': (1e-2, 4),
    'γ1_std': (1e-2, 1), 'γ2_std': (1e-2, 1), 'γ3_std': (1e-2, 1),
}

param_info = make_param_info(param_bds=param_bds, utility_settings=utility_settings, general_settings=general_settings)


param_bds stores lower/upper bounds for each parameter. Parameters are:

Vᵢᵢ: self-interest weight

Vᵢⱼ: altruism weight

Ʌᵢᵢ, Ʌᵢⱼ: loss/negativity weights for self and other

Ʒᵢⱼ: normal envy (disutility when other > self)

Ƹᵢⱼ: reverse envy/guilt (disutility when self > other)

γ1, γ2, γ3: exponents/sharpness parameters used in some utility forms

*_std: prior standard deviations for corresponding parameters (used in Bayesian grid/particle representations)

param_info is a derived structure that collects:

The active parameter list given utility_settings

Bounds as arrays or lists

Functions for initial guesses / prior grids

Optional covariance structures (when include_covariance=True)

param_info is passed into fitting and simulation routines to keep parameters consistent across the codebase.

Plot layout (fig_lay)
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


This is a central style dict passed to plotting helpers; it standardizes fonts, colors, tick sizes, and color scales across all generated figures.

File paths & canonical filenames
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


All paths are relative to the repository root (ROOT).

file_name_suffix and add_remove_file_name_suffix(...) may add suffixes based on configuration (e.g., to distinguish runs with different utility settings).

Type aliases & key composite types

These help you reason about data structures passed between functions:

PlayerUUID   = str
PlayerRole   = Literal['chooser', 'predictor']
DyadKey      = str | tuple[PlayerUUID, PlayerUUID] | list[PlayerUUID]
BoolTuple    = Tuple[bool, ...]

ParameterBounds = dict[str, tuple[int | float, int | float]]
GeneralSettings = dict[str, str | int | float | bool]
UtilitySettings = dict[str, bool]

FigLay      = dict[str, Any]
ColumnNames = dict[str, list[str] | dict[str, str]]

ParamKeys    = list[str]
ParamBounds  = list[tuple[int | float]]
ParamGuesses = Callable[[], Dict[str, float]]
ParamCovar   = list[str]
ParamInfo    = dict[str, ParamKeys | ParamBounds | ParamGuesses | ParamCovar]

CovMat     = NDArray[np.float64] | None
CovMatDict = dict[str, dict[str, dict[str, CovMat]]] | None

Output     = dict[str, float]
Params     = dict[str, float]
ParamVectors = dict[tuple[int] | str, float]
MetaData     = dict[str, int | float | dict[str, list[float]]]
ParamEst     = dict[str, dict[str, dict[str, dict[str, Output | Params | MetaData | ParamVectors]]]]

DyadGame  = dict[str, str | int | bool | ParamEst]
DyadGames = list[DyadGame]

PlayerInfo = dict[str, dict[str, str]]
Histories  = dict[str, DyadGames | PlayerInfo]
ParamVals  = dict[PlayerUUID, dict[PlayerRole, Params]]

PlayerDyads = dict[DyadKey, DyadGames]

WriteMode = Literal["readonly", "resume", "overwrite"]


ParamEst and ParamVals capture fitted parameter structures at different aggregation levels (player, dyad, role).

DyadGame and DyadGames are the fundamental containers for per-game/per-dyad results.

Histories corresponds closely to the histories mapping in Social_Preference_Prediction_Pairs_ExperX.json.

Valid analysis types

The analysis code uses named analysis types for organizing results:

within_player

within_dyad_symmetry

within_dyad_accuracy

across_dyad_one_chooser_many

across_dyad_many_predictors_one_chooser

across_analysis_modes

These tag different comparison levels (e.g., within player, within dyad, or across roles/modes).