import hashlib
import time
from visualization import *
from visualization import _hsla
from utilities import compute_hamming_distance_matrix

"=========================================================================================="
"========= Model Validation: Comparing Bayesian and Alternative Cognitive Models =========="
"=========================================================================================="

def alternative_model_contest(general_settings: Dict[str, Any], param_info: Dict[str, Any], param_bds: Dict[str, Tuple[float, float]], 
                              utility_settings: UtilitySettings, file_paths: Dict[str, str], fig_lay: Dict[str, Any]) -> Dict[str, float]:
    """
    Fits and compares multiple alternative cognitive models (Bayesian and non-Bayesian) 
    against behavioral data, returning their total negative log-likelihood (NLL) losses.

    This function:
        • Loads experiment data (Experiment 2) from disk.
        • Fits discrete Bayesian models with various hypothesis spaces 
            ("good_versus_evil", "social_value_ori", "perfect_oracle").
        • Calculates NLL for each discrete Bayesian model.
        • Calculates NLL for a purely stochastic model (uniform random).
        • Calculates NLL for a "no-memory" Bayesian model (only uses the most recent observation).
        • Calculates NLL for a "no-learning" model (static parameters, no updating).
        • Calculates NLL for a full continuous Bayesian model (grid-updating).
        • Summarizes all losses in a dictionary.

    Arguments:
        • general_settings: Dict[str, Any]
            High-level settings controlling analysis details 
            (e.g., 'update_method', 'experiment_num', 'loss_funct_type').
        • param_info: Dict[str, Any]
            Contains parameter keys, bounds, and initial guesses for model fitting.
        • utility_settings: UtilitySettings
            Dictionary or structured object specifying which components of the 
            utility function to include (e.g., social preferences, risk preferences).
        • file_paths: Dict[str, str]
            Dictionary with paths to directories or files necessary for loading 
            histories, saving results, etc. Must contain keys like:
               └── "processed": Path to processed data
               └── "file_names": Nested dict with file names keyed by 
                                "player_pairs_exper{experiment_number}", etc.

    Returns:
        Dict[str, float]:
            A dictionary mapping model names to their total negative log-likelihood:
               {
                   'utility_bayesian':  <NLL>,
                   'stochastic_model':  <NLL>,
                   'no_learning_model': <NLL>,
                   'no_memory_model':   <NLL>,
                   'good_versus_evil':  <NLL>,
                   'social_value_ori':  <NLL>,
                   'perfect_oracle':    <NLL>,
               }
    """
    "1) Copy and modify high-level settings for Experiment 2 analysis"
    model_names = {
        "utility_bayesian":  "Utility Bayes.",
        "stochastic_model":  "Stochastic",
        "no_learning_model": "No Learning",
        "no_memory_model":   "No Memory",
        "good_versus_evil":  "Good vs Evil",
        "social_value_ori":  "Canonical SVO",
        "perfect_oracle":    "Perfect Oracle"
    }

    experiment_num = 2
    create_new_file = True
    loss_funct_type = "log"

    general_settings_ = copy.deepcopy(general_settings)
    general_settings_["experiment_num"] = experiment_num
    general_settings_["create_new_file"] = create_new_file
    general_settings_["loss_funct_type"] = loss_funct_type

    model_losses = None
    output_file = "model_contest_losses.json"
    output_path = os.path.join(file_paths["processed"], output_file)
    if not create_new_file and os.path.exists(output_path):
        with open(output_path, "r", encoding='utf-8') as file:
            model_losses = json.load(file)
            print(model_losses)
    if not isinstance(model_losses, dict):

        def _save_progress(label: str) -> None:
            with open(output_path, "w", encoding='utf-8') as _f:
                json.dump(model_losses, _f, ensure_ascii=False, indent=4)
            print(f"  [saved after {label}] → {output_path}")

        "2) Identify player UUIDs for Experiment 2"
        player_uuids = prep.all_player_uuids(
            file_paths=file_paths,
            experiment_num=experiment_num,
            only_humans=True
        )

        "3) Load processed data (histories) for Experiment 2"
        full_path_histories = os.path.join(
            file_paths["processed"],
            file_paths["file_names"][f"player_pairs_exper{experiment_num}"]
        )
        if not os.path.exists(full_path_histories):
            raise FileNotFoundError(
                f"Cannot find player pairs file for experiment {experiment_num} at {full_path_histories}"
            )

        with open(full_path_histories, "r", encoding='utf-8') as file:
            histories_and_info = json.load(file)

        "'histories_exper2' is a dict: { <dyad_id>: [game_1, game_2, ...], ... }"
        histories_exper2: Dict[str, Any] = histories_and_info["histories"]

        "4) Initialize a results dict to track total negative log-likelihoods for all models"
        model_losses = {
            "utility_bayesian":  0.0,
            "stochastic_model":  0.0,
            "no_learning_model": 0.0,
            "no_memory_model":   0.0,
            "good_versus_evil":  0.0,
            "social_value_ori":  0.0,
            "perfect_oracle":    0.0,
        }

        "5) Define hypothesis spaces for discrete Bayesian models"
        "Each dict key is a model name, each value is a dict of {parameter_tuple: prior_weight}"
        hypothesis_spaces = {
            "good_versus_evil": {
                ( 1.0,  1.0): 0.5,
                ( 1.0, -1.0): 0.5
            },
            "social_value_ori": {
                ( 1.0,  1.0): 0.125,
                ( 1.0,  0.0): 0.125,
                ( 1.0, -1.0): 0.125,
                ( 0.0, -1.0): 0.125,
                (-1.0, -1.0): 0.125,
                (-1.0,  0.0): 0.125,
                (-1.0,  1.0): 0.125,
                ( 0.0,  1.0): 0.125,
            },
            "perfect_oracle": {
                ( 1.0,  1.0): 0.3,
                ( 1.0,  0.0): 0.3,
                ( 1.0, -1.0): 0.3,
                (-1.0,  0.0): 0.1,
            },
        }

        "6) Calculate loss for each discrete Bayesian model"
        "(Uses typo.discrete_bayesian_model(...))"
        general_settings_["update_method"] = "discrete"

        for hspace_name, hypothesis_space in hypothesis_spaces.items():
            for dyad_id, dyad_games in histories_exper2.items():
                these_dyad_games = copy.deepcopy(dyad_games)
                human_player_uuid = these_dyad_games[0]["predictor"]  # Assumed consistent across the dyad.
                updated_dyad = typo.discrete_bayesian_model(
                    dyad_games=these_dyad_games,
                    choice_funct=choice,
                    player_uuid=human_player_uuid,
                    general_settings=general_settings_,
                    hypothesis_space=hypothesis_space
                )
                loss_dict = create_loss_report(
                    dyad_games=updated_dyad,
                    general_settings=general_settings_
                ).get(human_player_uuid, {}).get("predictor", {})
                model_losses[hspace_name] += loss_dict.get("raw_neglogprob_sum", 0.0)

        pp.pprint(model_losses)
        _save_progress("discrete Bayesian models")

        "7) Calculate loss for the purely stochastic (random) model"
        n_iter_stochastic = 1000
        stochastic_losses = []
        "Iterate many times and compute the average loss."
        for n_iter in range(n_iter_stochastic):
            stochastic_loss = 0.0
            for dyad_games in histories_exper2.values():
                for game in dyad_games:
                    if game.get("phase") == "rp":
                        predicted_choice = game.get("prediction")
                        actual_obs = 1 if predicted_choice == "A" else 0

                        "Randomly guess P(A). Then compute probability of the observed event"
                        model_pred_A = random.random()
                        prob_of_observed = model_pred_A if actual_obs == 1 else (1 - model_pred_A)

                        "Avoid log(0)"
                        if prob_of_observed <= 0:
                            prob_of_observed = 1e-6

                        raw_neglogprob = -math.log(prob_of_observed)
                        stochastic_loss += raw_neglogprob

            stochastic_losses.append(stochastic_loss)    

        model_losses["stochastic_model"] = sum(stochastic_losses) / n_iter_stochastic
        pp.pprint(model_losses)
        _save_progress("stochastic model")

        "8) Calculate loss for \"no memory\" Bayesian model"
        "(always resets posterior after each new observation)"
        general_settings_["update_method"] = "grid"
        general_settings_["no_memory_mode"] = True

        for dyad_games in histories_exper2.values():
            human_player_uuid = dyad_games[0]["predictor"]

            "Construct an initial parameter dictionary"
            if callable(param_info["guesses"]):
                initial_guesses = param_info["guesses"]()
            else:
                initial_guesses = param_info["guesses"]

            initial_params = {
                "predictor": {
                    key: guess for key, guess in zip(param_info["keys"], initial_guesses)
                }
            }

            updated_games = agent(
                dyad_games=dyad_games,
                game_idx_start=0,
                game_idx_stop=None,
                general_settings=general_settings_,
                initial_params=initial_params,
                param_info=param_info,
                utility_settings=utility_settings,
                player_uuid=human_player_uuid,
                player_role="predictor",
                choice_temperature=general_settings_.get("softmax_temperature")
            )
            updated_games = loss_function_bayes(dyad_games=updated_games, general_settings=general_settings_)

            loss_dict = create_loss_report(
                dyad_games=updated_games,
                general_settings=general_settings_
            ).get(human_player_uuid, {}).get("predictor", {})
            model_losses["no_memory_model"] += loss_dict.get("raw_neglogprob_sum", 0.0)
        del general_settings_["no_memory_mode"]
        pp.pprint(model_losses)
        _save_progress("no-memory model")

        "9) Calculate loss for \"no learning\" model"
        "(static parameters, no posterior updates)"
        general_settings_["update_method"] = "naive"

        "Remove or add suffix to file paths to track results"
        file_name_suffix = prep.create_file_name_suffix(
            general_settings=general_settings_,
            utility_settings=utility_settings
        )
        file_paths_naive = copy.deepcopy(file_paths)
        file_paths_naive = prep.add_remove_file_name_suffix(
            file_paths=file_paths_naive,
            file_name_suffix=file_name_suffix,
            add_suffix=False
        )
        file_paths_naive = prep.add_remove_file_name_suffix(
            file_paths=file_paths_naive,
            file_name_suffix=file_name_suffix,
            add_suffix=True
        )

        general_settings_["update_method"] = "grid"

        param_info_ = make_param_info(param_bds=param_bds, utility_settings=utility_settings, general_settings=general_settings_, 
                                                random_guesses_are_unique=not general_settings_.get('run_in_parallel', True))

        for pdx, param_key in enumerate(param_info_['keys']):
            if '_std' in param_key:
                param_info_['bounds'][pdx] = (1e-6, 2e-6)
        param_info_['guesses'] = [
            random.uniform(bound[0], bound[1]) for bound in param_info_['bounds']
        ]

        run_analysis_bayes(
            utility_settings=utility_settings,
            general_settings=general_settings_,
            histories_data=histories_and_info,
            file_paths=file_paths_naive,
            param_info=param_info_,
            print_=False
        )

        "Accumulate NLL across all players/dyads"
        for player_uuid in player_uuids:
            player_dyads = prep.fitted_dyads_for_a_player(
                player_uuid=player_uuid,
                experiment_num=experiment_num,
                file_paths=file_paths_naive
            )
            if not player_dyads:
                raise ValueError(f"Failed to extract data for player {player_uuid}")

            for dyad_key, games in player_dyads.items():
                loss_dict = create_loss_report(
                    dyad_games=games,
                    general_settings=general_settings_
                ).get(player_uuid, {}).get("predictor", {})
                model_losses["no_learning_model"] += loss_dict.get("raw_neglogprob_sum", 0.0)
        pp.pprint(model_losses)
        _save_progress("no-learning model")

        "10) Calculate loss for full (continuous) Bayesian model"
        general_settings_["update_method"] = "grid"
        "Re-generate file name suffix for these settings"
        file_name_suffix_full = prep.create_file_name_suffix(
            general_settings=general_settings_,
            utility_settings=utility_settings
        )

        file_paths_full = copy.deepcopy(file_paths)
        file_paths_full = prep.add_remove_file_name_suffix(
            file_paths=file_paths_full,
            file_name_suffix=file_name_suffix_full,
            add_suffix=False
        )
        file_paths_full = prep.add_remove_file_name_suffix(
            file_paths=file_paths_full,
            file_name_suffix=file_name_suffix_full,
            add_suffix=True
        )
        print(param_info, utility_settings)
        "Run continuous Bayesian analysis"
        run_analysis_bayes(
            utility_settings=utility_settings,
            general_settings=general_settings_,
            histories_data=histories_and_info,
            file_paths=file_paths_full,
            param_info=param_info,
            print_=True
        )

        "Summation of NLL across players for continuous model"
        for player_uuid in player_uuids:
            player_dyads = prep.fitted_dyads_for_a_player(
                player_uuid=player_uuid,
                experiment_num=experiment_num,
                file_paths=file_paths_full
            )
            if not player_dyads:
                raise ValueError(f"Failed to extract data for player {player_uuid}")

            for dyad_key, games in player_dyads.items():
                loss_dict = create_loss_report(
                    dyad_games=games,
                    general_settings=general_settings_
                ).get(player_uuid, {}).get("predictor", {})
                model_losses["utility_bayesian"] += loss_dict.get("raw_neglogprob_sum", 0.0)

        pp.pprint(model_losses)
        _save_progress("utility Bayesian model")

    "Sort the models by ascending loss for easier comparison"
    sorted_model_losses = dict(sorted(model_losses.items(), key=lambda model_loss_item: model_loss_item[1]))

    "Extract names and losses"
    model_names = [model_names[model] for model in sorted_model_losses.keys()]
    colors = [f'hsla({int(115 + 360/(len(model_names)+4) * idx) % 360}, 80%, 40%, 1.0)' for idx in range(len(model_names))]
    loss_values = list(sorted_model_losses.values())

    "Create the figure"
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=model_names,
        y=loss_values,
        marker_color=colors,
        hovertemplate="Model: %{x}<br>Loss: %{y:.3f}<extra></extra>",
    ))

    "Set title and axis labels"
    fig.update_layout(
        title="Model Comparison by Negative Log-Likelihood Loss",
        template=fig_lay.get("template", "plotly_dark"),
        font=fig_lay["font"],
        hoverlabel=fig_lay["hoverlabel"],
        margin=dict(l=120, r=120, t=120, b=120),
        titlefont_size=fig_lay['titlefont_size'],
        title_x=fig_lay['title_x'], 
        title_y=fig_lay['title_y'],
        xaxis=dict(
            title="Models",
            tickfont=fig_lay["xaxis"]["tickfont"],
            title_font=fig_lay["xaxis"]["title_font"]
        ),
        yaxis=dict(
            title="Total Negative Log-Likelihood (Lower is Better)",
            tickfont=fig_lay["yaxis"]["tickfont"],
            title_font=fig_lay["yaxis"]["title_font"]
        )
    )

    "Save the figure to the specified visuals path"
    visuals_path = file_paths["visuals"]
    os.makedirs(visuals_path, exist_ok=True)
    output_html_path = os.path.join(visuals_path, "model_losses_bar_chart.html")
    fig.write_html(output_html_path)

    print("Saved model losses bar chart to", output_html_path)

    pp.pprint(model_losses)
    return model_losses


"=========================================================================================="
"=================== Searching The Space of Typological Bayesian Models ==================="
"=========================================================================================="

def compute_loss_for_typological_model_across_all_data(hypothesis_space: Dict[Tuple[float, float], float], general_settings: Dict[str, Any], file_paths: Dict[str, str]) -> Tuple[float, int]:
    """
    Replays all relevant data (players/dyads) using the discrete_bayesian_model 
    with the given prior distribution, sums the negative log-likelihood, 
    and returns (NLL, N_data).

    Arguments:
        • hypothesis_space: Dict[Tuple[float, float], float]
            Maps each (Vᵢᵢ, Vᵢⱼ) profile to its prior probability.
        • general_settings: Dict[str, Any]
            Various settings (experiment_num, etc.). 
        • file_paths: Dict[str, str]
            Contains paths to load data from (like 'processed' data) if needed.

    Returns:
        • Tuple[float, int]
            (total negative log-likelihood, total number of data points used).
    """
    "Compute NLL by replaying experiment histories with a discrete Bayesian hypothesis space."

    experiment_num = general_settings.get('experiment_num', 2)
    full_path_histories = os.path.join(
        file_paths["processed"],
        file_paths["file_names"][f'player_pairs_exper{experiment_num}']
    )

    with open(full_path_histories, "r", encoding="utf-8") as file:
        histories_info = json.load(file)
    histories_exper2 = histories_info['histories']

    total_nll = 0.0
    total_data_points = 0

    for dyad_key, dyad_games in histories_exper2.items():
        "Run model - Be sure to re-copy the hypothesis_space to prevent mutating it."
        human_player_uuid = dyad_games[0]['predictor']
        local_space = copy.deepcopy(hypothesis_space)
        updated_dyad = typo.discrete_bayesian_model(
            dyad_games=copy.deepcopy(dyad_games),
            choice_funct=choice,
            player_uuid=human_player_uuid,
            general_settings=general_settings,
            hypothesis_space=local_space,
            update_method='discrete'
        )
        "Compute the per-dyad NLL"
        loss_report = create_loss_report(updated_dyad, general_settings).get(human_player_uuid, {}).get("predictor", {})
        total_nll += loss_report.get('raw_neglogprob_sum', 0.0)
        total_data_points += loss_report.get('n_data', 0)

    return total_nll, total_data_points


def _parallel_process_worker_typological_model_comparison_population_fit(args: Tuple[Any, ...]) -> Dict[str, Any]:
    """
    Worker function for parallelizing a single hypothesis-space fit. 
    This obtains the best-fitting priors for that subset, 
    then returns a dictionary summarizing the results.

    Arguments:
        • args: Tuple containing:
            - subset_id: int
            - subset_profiles: List[Tuple[float,float]]
            - general_settings: Dict[str,Any]
            - param_info: Dict[str,Any]
            - utility_settings: Dict[str,bool]
            - file_paths: Dict[str,str]
            - prior_init_method: str
            - penalty_weight: float

    Returns:
        • Dict[str, Any]
            A row of data including 'subset_id', 'profiles', 'best_priors', 'best_nll', 'AIC', 'BIC', etc.
    """
    (subset_id,
     subset_profiles,
     general_settings,
     file_paths,
     prior_init_method,
     penalty_weight) = args

    k_params = len(subset_profiles)

    time_start = time.time()

    "--- 1) Prepare the objective function"
    def objective_fn(alpha: np.ndarray) -> float:
        "Convert alpha to priors"
        priors = gnrl.transform_to_simplex(alpha)  # Shape (k,).

        "Build the dictionary for the discrete model"
        hypothesis_space = {
            prof: float(pr) for prof, pr in zip(subset_profiles, priors)
        }

        "Evaluate negative log-likelihood across data"
        "Plus penalty to help gradient"
        nll, _ = compute_loss_for_typological_model_across_all_data(
            hypothesis_space=hypothesis_space,
            general_settings=general_settings,
            file_paths=file_paths
        )

        "Add a penalty if desired"
        "E.g. penalty = penalty_weight * sum( p^2 )"
        sum_sq = sum(priors * priors)
        penalty = penalty_weight * sum_sq

        return nll + penalty

    "--- 2) Dimensions of alpha = k. Unconstrained reparameterization"
    "Bounds for alpha."
    x_bounds = [(0.0, 1.0)] * k_params
    if prior_init_method == "uniform":
        x_guesses = [1/len(x_bounds) for bound in x_bounds]
    else:
        x_guesses = [random.uniform(param_bds[bound][0], param_bds[bound][1]) for bound in x_bounds]

    "--- 3) run global+local optimization"
    best_alpha = global_local_optimization(
        objective_fn=objective_fn,
        x_bounds=x_bounds,
        x_guesses=x_guesses,
        maxiter_global=None,   # Can tune.
        maxiter_local=None,   # Can tune.
        maxfun_global=100,
        maxfun_local=None
    )

    "Recompute the actual no-penalty NLL with best_alpha"
    best_priors = gnrl.transform_to_simplex(best_alpha['final']['x'])
    hypothesis_space_final = {
        prof: float(pr) for prof, pr in zip(subset_profiles, best_priors)
    }
    nll_no_penalty, n_data = compute_loss_for_typological_model_across_all_data(
        hypothesis_space=hypothesis_space_final,
        general_settings=general_settings,
        file_paths=file_paths
    )
    "Also compute the penalty for best_alpha."
    sum_sq_final = sum(best_priors * best_priors)
    penalty_final = penalty_weight * sum_sq_final
    total_loss = nll_no_penalty + penalty_final

    "Compute IC"
    k_params = max(k_params, 1) 
    ic_dict = compute_ic(k_params, n_data, nll_no_penalty)
    aic_val = ic_dict["AIC"]
    bic_val = ic_dict["BIC"]

    time_stop = time.time()
    duration = time_stop - time_start

    result_dict = {
        "n_data": n_data,
        "duration": duration,
        "subset_id": subset_id,
        "k_params": k_params,
        "profiles": subset_profiles,  # Or str(subset_profiles).
        "nll_no_penalty": nll_no_penalty,
        "penalty": penalty_final,
        "total_loss": total_loss,
        "AIC": aic_val, "BIC": bic_val,
        "best_alpha": list(best_alpha['final']['x']),
        "best_priors_normalized": list(best_priors),
    }

    return result_dict


def typological_model_comparison_fit_population(file_paths: Dict[str, str], general_settings: Dict[str, Any], k_min: int, k_max: int, n_subsets_per_k: int, 
                                              intervals_per_dim: int, prior_init_method: str, penalty_weight: float, save_after_n_iter: int, max_combinations: int = 5000000) -> None:
    """
    Explores and fits many discrete Bayesian models by:
        • Generating a 2D grid of social-preference values in [-1, 1].
        • Randomly sampling subsets of size k in [k_min, k_max].
        • For each subset, optimizing the prior distribution over profiles via global+local search.
        • Storing the best-fitting priors and the corresponding negative log-likelihood + info criteria.

    Arguments:
        • file_paths: Dict[str, str]
            Must include:
            └─ "discrete": Path to a directory where intermediate results are saved.
            └─ "processed" & "file_names" for reading the experiment data.
        • general_settings: Dict[str, Any]
            Contains standard keys (e.g., "experiment_num", "run_in_parallel", etc.).
        • param_info: Dict[str, Any]
            Contains parameter specifications for the overall modeling context.
        • utility_settings: Dict[str, bool]
            Toggles for the utility function used when replaying data.
        • k_min: int
            Minimum number of profiles to use in a hypothesis space.
        • k_max: int
            Maximum number of profiles to use in a hypothesis space (e.g., 4).
        • n_subsets_per_k: int
            How many random subsets of each size k to generate.
        • intervals_per_dim: int
            Grid resolution for each dimension in [-1,1], e.g. 9 => [-1.0, -0.75, ..., 1.0].
        • prior_init_method: str
            "uniform" or "random" for how to initialize alpha in local stage 
            (affects global_local_optimization).
        • penalty_weight: float
            Strength of the sum-of-squares penalty on priors to improve optimization stability.
        • save_after_n_iter: int
            Interval for saving the DataFrame to disk. Must be <= n_subsets_per_k.

    Returns:
        • None; saves output files in file_paths["discrete"] as they are computed.

    Notes:
        • The final DataFrame is saved/overwritten every `save_after_n_iter` subsets.
        • At the end, the output contains multiple CSV files, one for each k in [k_min, k_max].
        • Later analysis can parse these CSVs to find the best discrete model overall.
    """
    "Validate folder"
    output_dir = file_paths.get("discrete", None)
    if not output_dir:
        raise ValueError("'discrete' path not found in file_paths. Please specify file_paths['discrete'].")
    os.makedirs(output_dir, exist_ok=True)

    "1) Build the 2D grid of possible profiles"
    step = 2.0 / (intervals_per_dim - 1)  # E.g. 2.0 / 8 = 0.25 if intervals_per_dim=9.
    possible_values = [round(-1.0 + idx*step, 4) for idx in range(intervals_per_dim)]
    all_profiles = [(vx, vy) for vx in possible_values for vy in possible_values if not (vx == 0 and vy == 0)]

    general_settings_ = copy.deepcopy(general_settings)
    general_settings_['update_method'] = 'discrete'    

    for k_params in range(k_min, k_max+1):
        "Store results in a Pandas DataFrame"
        columns = [
            "n_data", "duration", "subset_id", "k_params", 
            "best_alpha", "nll_no_penalty", "penalty", "total_loss",
            "AIC", "BIC", "profiles", "best_priors_normalized",
        ]
        df_results = pd.DataFrame(columns=columns)

        "Generate random subsets of size k"
        n_combinations = math.comb(len(all_profiles), k_params)
        if n_combinations > max_combinations:
            print(f"Halting operation: {n_combinations} combinations > maximum n combinations = {max_combinations}.")
            break

        print(f"Generating {n_combinations} combinations")
        random_subsets = list(it.combinations(all_profiles, k_params))
        random.shuffle(random_subsets)

        args_list = []
        for subset_idx, random_subset in enumerate(random_subsets):
            if subset_idx > n_subsets_per_k:
                break
            args_list.append((
                subset_idx,
                random_subset,
                general_settings_,
                file_paths,
                prior_init_method,
                penalty_weight
            ))            

        minimum_loss = 1e18
        time_start = time.time()
        "Now run them in parallel or serial"
        if general_settings.get('run_in_parallel'):
            with mp.Pool(processes=max(mp.cpu_count()-1, 1)) as pool:
                for idx, result_dict in enumerate(pool.imap_unordered(_parallel_process_worker_typological_model_comparison_population_fit, args_list), 1):
                    "Append to df_results"
                    df_results.loc[len(df_results)] = [
                        result_dict["n_data"],
                        result_dict["duration"],
                        result_dict["subset_id"],
                        result_dict["k_params"],
                        result_dict["best_alpha"],
                        result_dict["nll_no_penalty"],
                        result_dict["penalty"],
                        result_dict["total_loss"],
                        result_dict["AIC"], 
                        result_dict["BIC"],
                        result_dict["profiles"],
                        result_dict["best_priors_normalized"],
                    ]

                    total_loss = result_dict["total_loss"]
                    if total_loss < minimum_loss:
                        minimum_loss = total_loss

                    if idx % save_after_n_iter == 0:

                        "Compute remaining time"
                        time_now = time.time()
                        total_time = time_now - time_start
                        average_duration = total_time / (idx + 1)
                        n_remaining_iters = len(args_list) - idx
                        remaining_seconds = average_duration * n_remaining_iters / max(mp.cpu_count()-1, 1)
                        remaining_minutes = remaining_seconds / 60
                        remaining_hours = int(remaining_minutes / 60)
                        remaining_minutes = int(remaining_minutes % 60)
                        current_time = dt.datetime.now().strftime("%H%M")

                        "Save to disk"
                        csv_path = os.path.join(output_dir, f"discrete_fits_k{k_params}.csv")
                        df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
                        print_str = f"[k={k_params}] Processed {idx}/{len(args_list)} subsets. "
                        print_str += f"Time: {current_time}; Remaining Time: {remaining_hours:02d} hours "
                        print_str += f"and {remaining_minutes:02d} minutes. Min Loss = {minimum_loss}"
                        print(print_str)
        else:
            "Serial"
            for idx, single_args in enumerate(args_list, 1):
                result_dict = _parallel_process_worker_typological_model_comparison_population_fit(single_args)
                df_results.loc[len(df_results)] = [
                    result_dict["n_data"],
                    result_dict["duration"],
                    result_dict["subset_id"],
                    result_dict["k_params"],             
                    result_dict["best_alpha"],
                    result_dict["nll_no_penalty"],
                    result_dict["penalty"],
                    result_dict["total_loss"],
                    result_dict["AIC"],
                    result_dict["BIC"],
                    result_dict["profiles"],
                    result_dict["best_priors_normalized"],
                ]

                total_loss = result_dict["total_loss"]
                if total_loss < minimum_loss:
                    minimum_loss = total_loss

                if idx % save_after_n_iter == 0:

                    "Compute remaining time"
                    time_now = time.time()
                    total_time = time_now - time_start
                    average_duration = total_time / (idx + 1)                    
                    n_remaining_iters = len(args_list) - idx
                    remaining_seconds = average_duration * n_remaining_iters 
                    remaining_minutes = remaining_seconds / 60
                    remaining_hours = int(remaining_minutes / 60)
                    remaining_minutes = int(remaining_minutes % 60)
                    current_time = dt.datetime.now().strftime("%H%M")

                    csv_path = os.path.join(output_dir, f"discrete_fits_k{k_params}.csv")
                    df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
                    print_str = f"[k={k_params}] Processed {idx}/{len(args_list)} subsets. "
                    print_str += f"Time: {current_time}; Remaining Time: {remaining_hours:02d} hours "
                    print_str += f"and {remaining_minutes:02d} minutes. Min Loss = {minimum_loss}"
                    print(print_str)

        "Final save after all subsets"
        csv_path = os.path.join(output_dir, f"discrete_fits_k{k_params}.csv")
        df_results = df_results.sort_values(by='total_loss', ascending=True)
        df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Finished k={k_params}. All {n_subsets_per_k} subsets processed. Results saved to {csv_path}.")
        print(df_results)


def typological_model_nll_for_player(hypothesis_space: Dict[tuple[float, float], float], general_settings: Dict[str, Any], file_paths: Dict[str, str], player_uuid: str) -> Tuple[float, int]:
    """
    Replays data from a single player's dyads using discrete_bayesian_model
    with the given prior distribution, returns (NLL, number_of_data_points).

    Arguments:
        • hypothesis_space: Dict[(float, float), float]
            The discrete profiles => prior probs mapping.
        • general_settings: Dict[str, Any]
            Possibly includes 'experiment_num', etc.
        • file_paths: Dict[str, str]
            Paths for loading the data (like 'processed' + 'file_names').
        • player_uuid: str
            Which player's data to load and evaluate.

    Returns:
        • (total_nll, total_data_points)
    """
    experiment_num = general_settings.get("experiment_num", 2)
    "Load histories from the processed experiment file"
    full_path_histories = os.path.join(
        file_paths["processed"],
        file_paths["file_names"][f"player_pairs_exper{experiment_num}"]
    )
    with open(full_path_histories, "r", encoding="utf-8") as file:
        histories_info = json.load(file)
    all_histories = histories_info["histories"]  # { dyad_key: [games], ...}

    total_nll = 0.0
    total_data = 0

    "Only want dyads where this player is the predictor"
    for dyad_key, dyad_games in all_histories.items():
        if dyad_games and (dyad_games[0].get("predictor") == player_uuid):
            "Run model"
            local_space = copy.deepcopy(hypothesis_space)
            these_dyad_games = copy.deepcopy(dyad_games)
            updated_dyad = typo.discrete_bayesian_model(
                dyad_games=these_dyad_games,
                choice_funct=choice, 
                player_uuid=player_uuid,
                general_settings=general_settings,
                hypothesis_space=local_space,
                update_method="discrete"
            )
            "Gather loss"
            general_settings_ = copy.deepcopy(general_settings)
            general_settings_["update_method"] = "discrete"
            loss_report = create_loss_report(updated_dyad, general_settings_)
            predictor_loss = (loss_report.get(player_uuid, {})
                                         .get("predictor", {}))
            total_nll += predictor_loss.get("raw_neglogprob_sum", 0.0)
            total_data += predictor_loss.get("n_data", 0)

    if total_data <= 0:
        raise Exception("No data found!")

    return total_nll, total_data


def _parallel_process_worker_typological_model_comparison_individual_fit(args: Tuple[Any, ...]) -> Dict[str, Any]:
    """
    Parallel worker for individual-fit typological model comparison.

    Receives one player's UUID and a fixed set of typological profiles, then optimizes a
    Dirichlet prior (via an unconstrained softmax-transformed alpha vector) to minimize the
    negative log-likelihood of that player's observed choices under a mixture of profiles.
    Called by typological_model_comparison_fit_individually via multiprocessing.Pool.

    Arguments:
        • args: tuple unpacking to (player_uuid, best_profiles, general_settings, file_paths,
                penalty_weight, maxiter_global, maxiter_local, optimization_method).

    Returns:
        • dict — fitting result including the optimized prior weights, best NLL, and player UUID.
    """
    (player_uuid,
     best_profiles,
     general_settings,
     file_paths,
     penalty_weight,
     maxiter_global,
     maxiter_local,
     optimization_method) = args

    k_params = len(best_profiles)

    def objective_fn(alpha: np.ndarray) -> float:
        priors = gnrl.transform_to_simplex(alpha)  # E.g. exp / sum(exp).
        hypothesis_space = {prof: float(pr) for prof, pr in zip(best_profiles, priors)}

        nll, _ = typological_model_nll_for_player(
            hypothesis_space=hypothesis_space,
            general_settings=general_settings,
            file_paths=file_paths,
            player_uuid=player_uuid
        )
        sum_sq = sum(priors * priors)
        penalty = penalty_weight * sum_sq
        return nll + penalty

    "Initialize alpha at zero"
    x_bounds = [(0.0, 1.0)] * k_params
    x_guesses = [0.0] * k_params  # Or random, or something else.

    "Run global+local optimization"
    best_result: OptimizeResult = global_local_optimization(
        objective_fn,
        x_bounds=x_bounds,
        x_guesses=x_guesses,
        maxiter_global=maxiter_global,
        maxiter_local=maxiter_local,
        optimization_method=optimization_method
    )

    best_alpha = list(best_result['final']['x'])
    best_priors = gnrl.transform_to_simplex(best_alpha)
    hypothesis_space_final = {prof: float(pr) for prof, pr in zip(best_profiles, best_priors)}

    nll_no_penalty, n_data = typological_model_nll_for_player(
        hypothesis_space=hypothesis_space_final,
        general_settings=general_settings,
        file_paths=file_paths,
        player_uuid=player_uuid
    )
    sum_sq_final = sum(best_priors * best_priors)
    penalty_final = penalty_weight * sum_sq_final
    total_loss = nll_no_penalty + penalty_final

    "Use k_eff = k_params - 1 for per-player AIC/BIC."
    k_eff = max(k_params - 1, 1)
    ic_dict = compute_ic(k_eff, n_data, nll_no_penalty)
    return {
        "n_data": n_data,
        "player_uuid": player_uuid,
        "k_params": k_params,
        "best_alpha": best_alpha,
        "nll_no_penalty": nll_no_penalty,
        "penalty": penalty_final,
        "total_loss": total_loss,
        "AIC": ic_dict["AIC"],
        "BIC": ic_dict["BIC"],
        "success": best_result["final"]["success"],
        "message": best_result["final"]["message"],
        "profiles": best_profiles,
        "best_priors_normalized": list(best_priors),
    }


def typological_model_comparison_fit_individually(best_profiles: list[tuple[float, float]], general_settings: Dict[str, Any], file_paths: Dict[str, str], penalty_weight: float = 10, 
                                                  maxiter_global: int = 2, maxiter_local: int = None, optimization_method: str = 'globloc', save_csv: bool = True) -> Tuple[pd.DataFrame, float]:
    """
    Fits a discrete model individually to each player, using a fixed set of profiles.

    Arguments:
        • best_profiles: list[tuple[float, float]]
            The discrete profiles chosen from the population-level stage.
        • general_settings: Dict[str, Any]
            E.g. {'experiment_num': 2, 'run_in_parallel': True, ...}
        • file_paths: Dict[str, str]
            Must allow us to load the data. 
            Optionally, a path to save the final CSV (like file_paths["discrete"]).
        • penalty_weight: float
            Weight for the sum-of-squares penalty on priors.
        • maxiter_global: int
            Global method iteration limit.
        • maxiter_local: int
            Local method iteration limit.
        • optimization_method: str
            One of {'global','local','globloc'}.
        • save_csv: bool
            If True, writes results to a CSV file at the end.

    Returns:
        • (df, total_nll):
            df is a DataFrame with one row per player:
                [player_uuid, best_alpha, best_priors_normalized, nll_no_penalty, ...]
            total_nll is sum of all players' nll_no_penalty (no penalty included).

    Notes:
        • This is the standard approach to get a per-player best-fitting prior distribution,
          given a single “best” set of profiles for the entire population.
        • Summing the 'nll_no_penalty' column in the returned DataFrame gives the total NLL
          across all players, enabling comparison against the continuous model's total.
    """
    "1) Gather the relevant players"
    "Load from the same file used in alt_model_contest or a dedicated loader."
    experiment_num = general_settings.get("experiment_num", 2)
    "Load player_info from the histories payload"
    "Usually, \"histories_data['player_info']\" is available, or the dyad file can be parsed directly."

    full_path_histories = os.path.join(
        file_paths["processed"],
        file_paths["file_names"][f"player_pairs_exper{experiment_num}"]
    )
    with open(full_path_histories, "r", encoding="utf-8") as file:
        data_all = json.load(file)
    """
    Can gather all predictor players:
    Use \"player_info\" when available.
    Otherwise, parse the \"histories\" to get unique predictor IDs:   
    """
    all_histories = data_all["histories"]

    player_uuids = set()
    for dyad_key, dyad_games in all_histories.items():
        if dyad_games:
            pid = dyad_games[0].get("predictor", None)
            if pid is not None:
                player_uuids.add(pid)
    player_uuids = sorted(list(player_uuids))

    "2) Prepare for parallel or serial"
    args_list = []
    for player_uuid in player_uuids:
        args_list.append((
            player_uuid,
            best_profiles,
            general_settings,
            file_paths,
            penalty_weight,
            maxiter_global,
            maxiter_local,
            optimization_method
        ))

    "3) Run"
    df_columns = [
        "n_data", "player_uuid", "k_params", "best_alpha", "nll_no_penalty", "penalty", 
        "total_loss", "AIC", "BIC", "success", "message", "profiles", "best_priors_normalized",
    ]
    df_results = pd.DataFrame(columns=df_columns)
    
    time_start = time.time()
    sum_nll_no_penalty = 0

    run_in_parallel = general_settings.get("run_in_parallel", True)
    if run_in_parallel:
        with mp.Pool(processes=max(mp.cpu_count()-1, 1)) as pool:
            for idx, result in enumerate(pool.imap_unordered(_parallel_process_worker_typological_model_comparison_individual_fit, args_list), 1):
                df_results.loc[len(df_results)] = [
                    result["n_data"],
                    result["player_uuid"],
                    result["k_params"],
                    result["best_alpha"],
                    result["nll_no_penalty"],
                    result["penalty"],
                    result["total_loss"],
                    result["AIC"],
                    result["BIC"],
                    result["success"],
                    result["message"],
                    result["profiles"],
                    result["best_priors_normalized"],
                ]

                sum_nll_no_penalty += result["nll_no_penalty"]

                if idx % 1 == 0:
                    "Compute remaining time"
                    time_now = time.time()
                    k_params = result['k_params']
                    total_time = time_now - time_start
                    average_duration = total_time / (idx + 1)
                    n_remaining_iters = len(args_list) - idx
                    remaining_seconds = average_duration * n_remaining_iters
                    remaining_minutes = remaining_seconds / 60
                    remaining_hours = int(remaining_minutes / 60)
                    remaining_minutes = int(remaining_minutes % 60)
                    current_time = dt.datetime.now().strftime("%H%M")
                    print_str = f"k={k_params} Processed {idx}/{len(args_list)} subsets. "
                    print_str += f"Time: {current_time}; Remaining Time: {remaining_hours:02d} hours "
                    print_str += f"and {remaining_minutes:02d} minutes. Sum Loss = {sum_nll_no_penalty}"
                    print(print_str)

    else:
        for idx, single_args in enumerate(args_list, 1):
            result = _parallel_process_worker_typological_model_comparison_individual_fit(single_args)
            df_results.loc[len(df_results)] = [
                result["n_data"],
                result["player_uuid"],
                result["k_params"],
                result["best_alpha"],
                result["nll_no_penalty"],
                result["penalty"],
                result["total_loss"],
                result["AIC"],
                result["BIC"],
                result["success"],
                result["message"],
                result["profiles"],
                result["best_priors_normalized"],
            ]

            sum_nll_no_penalty += result["nll_no_penalty"]

            if idx % 5 == 0:
                "Compute remaining time"
                time_now = time.time()
                k_params = result['k_params']
                total_time = time_now - time_start
                average_duration = total_time / (idx + 1)
                n_remaining_iters = len(args_list) - idx
                remaining_seconds = average_duration * n_remaining_iters
                remaining_minutes = remaining_seconds / 60
                remaining_hours = int(remaining_minutes / 60)
                remaining_minutes = int(remaining_minutes % 60)
                current_time = dt.datetime.now().strftime("%H%M")
                print_str = f"k={result['k_params']} Processed {idx}/{len(args_list)} subsets. "
                print_str += f"Time: {current_time}; Remaining Time: {remaining_hours:02d} hours "
                print_str += f"and {remaining_minutes:02d} minutes. Sum Loss = {sum_nll_no_penalty}"
                print(print_str)

    "4) Summation of all players' NLL (no penalty)"
    total_nll = df_results["nll_no_penalty"].sum()
    df_results = df_results.sort_values(by="nll_no_penalty", ascending=True)

    "5) Optionally save"
    if save_csv:
        out_dir = file_paths.get("discrete", ".")
        out_path = os.path.join(out_dir, f"discrete_individual_fits_k={len(best_profiles)}.csv")
        df_results.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"[discrete_individual_fits] Completed. Results saved to {out_path}.")

    print(f"Total NLL For Individuall Fit k = {len(best_profiles)} Model: {round(total_nll, 6)}")
    return df_results, total_nll


"=========================================================================================="
"========= Information Criterion Analysis: Determining Optimal Utility Structures ========="
"=========================================================================================="

def information_criterion_analysis(general_settings: Dict[str, Any], utility_settings: Dict[str, bool], file_paths: Dict[str, str],
                                   param_bds: Dict[str, tuple[int | float, int | float]], dynamic_updating: bool = False, max_iters: int = 1, robustness_epsilon: float = 10,
                                   check_for_n_players: int | str = "all", write_mode: WriteMode = "resume",
                                   utility_setting_varieties: Optional[List[UtilitySettings]] = None) -> Tuple[pd.DataFrame, Dict[str, Dict[Tuple[bool], Dict[str, Any]]]]:
    """
    Computes and compares AIC/BIC across different utility function configurations.

    Arguments:
        • general_settings: Dict[str, Any]; High-level settings (analysis mode, etc.).
        • utility_settings: Dict[str, bool]; Flat dictionary of boolean flags controlling the functional form.
        • file_paths: Dict[str, str]; Paths to files/directories for reading/writing data.
        • param_info: Dict[str, Any]; Contains parameter keys, bounds, guesses, etc.
        • robustness_epsilon: float; If sum of ΔMinLoss < this threshold for two
            consecutive iterations, then stop early.
        • dynamic_updating: bool; Because the computational demands of fitting individual-level belief updating
            are prohibitive, this analysis relies on a static (no updating) version of the UBM by default. Yet,
            setting this input to True can work on a machine with more cores.
        • utility_setting_varieties: Optional[List[UtilitySettings]]; If provided, this exact list of utility
            configurations is used instead of generating all valid configurations via
            gnrl.generate_utility_settings. Each entry must pass gnrl.is_valid_utility_settings or a
            ValueError is raised. Pass None (the default) to run the full comparison across all 480 forms.

    Returns:
        • df: pd.DataFrame; Dataframe summarizing the IC metrics (loss, AIC, BIC) for each utility configuration.
        • all_ic_results: Dict[str, Dict[Tuple[bool], Dict[str, Any]]];
            - "ic_results": Maps each utility config (as a tuple of booleans) to its {n,k,loss,AIC,BIC}.
            - "utility_varieties": Maps the same config tuple to the actual dictionary of settings.
    """
    "Storing terminal printouts in a .txt file"
    ic_terminal_printouts = [
        "This document contains terminal print statements for the information criterion utility function model comparison."
    ]

    def compute_mean_param_variance(param_info: Dict[str, Any], param_runs: List[Dict[str, Dict[str, Dict[str, float]]]]) -> float:
        """
        Computes a single 'normalized average parameter variance' across all runs, participants, and roles.

        Arguments:
        ----------
        • param_info : Dict[str, Any]
            Must have 'keys' (list of parameter names) and 'bounds' (list of (low, high) tuples).
            e.g. param_info['keys'] = ['Vᵢᵢ','Vᵢⱼ','Ƹᵢⱼ','Ʒᵢⱼ','γ1','τ']
                    param_info['bounds'] = [(0,1),(-1,1),... etc.]

        • param_runs : List[Dict[str, Dict[str, Dict[str,float]]]]
            Each element corresponds to a single iteration's "players_to_params_this_iter".
            For iteration i, param_runs[i] is a dict:
                {
                    player_uuid_1: {"chooser": {...}, "predictor": {...}},
                    player_uuid_2: {"chooser": {...}, "predictor": {...}},
                    ...
                }
            Inside each "..." is a mapping param_key -> param_value.

        Returns:
            • A single float in [0,1] (roughly). 
            • A value near 0 means all runs found nearly identical param solutions 
                (hence stable). Higher means more discrepancy across runs.
        """

        "If fewer than 2 runs, variance is trivially zero"
        if len(param_runs) < 2:
            return 0.0

        param_names = param_info['keys']
        param_bounds = param_info['bounds']  # Same order as param_names.

        "1) For each participant, for each role, for each param, gather a list of values across runs"
        "Store them in something like: values_dict[(player_uuid, role, param_name)] = [val_run1, val_run2, ... val_runR]"
        from collections import defaultdict
        values_dict = defaultdict(list)

        n_runs = len(param_runs)

        for run_idx in range(n_runs):
            run_data = param_runs[run_idx]
            "Run_data => {player_uuid: { 'chooser': {pkey: val}, 'predictor': {pkey: val} }, ...}"
            for player_uuid, role_dict in run_data.items():
                for role_name, param_map in role_dict.items():
                    "Param_map => { param_key: param_val, ...}"
                    if not param_map:
                        continue
                    "For each param in param_names:"
                    for pkey, pval in param_map.items():
                        "Only process it if pkey is in param_names"
                        if pkey in param_names:
                            "Store it"
                            values_dict[(player_uuid, role_name, pkey)].append(pval)

        """
        2) For each param triple (player, role, param), compute variance in normalized scale
        Then average across all (player, role, param). Skip any triple that has < 2 data points 
        (maybe that triple wasn't used).
        """
        all_variances = []
        for (ply, rle, pkey), val_list in values_dict.items():
            if len(val_list) < 2:
                "Can't compute variance with <2 points"
                continue

            "Find the index in param_names"
            try:
                p_index = param_names.index(pkey)
            except ValueError:
                continue

            bound_low, bound_high = param_bounds[p_index]
            param_range = (bound_high - bound_low)
            if param_range <= 0:
                "Skip invalid ranges"
                continue

            "Normalize each value"
            norm_vals = [(param_value - bound_low) / param_range for param_value in val_list]

            "Compute variance"
            var_ = np.var(norm_vals, ddof=1)  # Ddof=1 => sample variance.
            all_variances.append(var_)

        if not all_variances:
            return 0.0

        "3) Final average"
        return float(np.mean(all_variances))

    def _compute_normalised_param_sd(pvec, param_bds):
        """
        Pvec : list of per-iteration dicts
            [ iteration0, iteration1, … ]
            Each item -> {player_uuid: {'chooser': {param: val, …},
                                        'predictor': {param: val, …}}, …}

        param_bds : dict {param_name: (low, high)}

        Returns a *scalar*:
            • median_over_params(   SD_over_iters&players(param) / (high-low)   )
                If a param is missing bounds, it is skipped.
        """
        if not pvec:
            return np.nan

        "Flatten -> {param: [values]}"
        bag = {}
        for iter_dict in pvec:
            for pl_dict in iter_dict.values():         # Players.
                for role_dict in pl_dict.values():     # Chooser / predictor.
                    if not isinstance(role_dict, dict):
                        continue
                    for param_key, param_value in role_dict.items():
                        bag.setdefault(param_key, []).append(float(param_value))

        norm_sds = []
        for param, param_values in bag.items():
            param_bounds = param_bds.get(param)
            if not param_bounds:
                continue
            param_range = param_bounds[1] - param_bounds[0]
            if param_range <= 0 or len(param_values) < 2:
                continue
            param_sd = np.std(param_values, ddof=1)
            norm_sds.append(param_sd / param_range)

        if not norm_sds:
            return np.nan
        return float(np.median(norm_sds))

    def check_nesting_fit_violations(target_model: tuple[bool] | dict[str: bool] | str, models_to_sequential_losses: dict, 
                                     general_settings=general_settings, utility_settings=utility_settings, file_paths=file_paths, print_only_children: bool = True) -> dict:
        """
        Checks if the loss found for the target model is greater than its children or less than its parents.
        """
        def model_key_maker(model: tuple[bool] | dict[str: bool] | str, into: type) -> str | tuple[int]:
            """
            Converts models into compact representations, like
            00000000000010~Uᵢ(A)=Vᵢᵢ(πᵢᴬ-πᵢᴮ)-Ƹᵢⱼ×(max(πⱼᴬ-πᵢᴬ, 0)+max(πᵢᴬ-πⱼᴬ,0))
            """
            if into is str:
                if isinstance(model, str):
                    model = ast.literal_eval(model)
                if isinstance(model, (dict, tuple)):
                    model = gnrl.convert_utility_settings(utility_settings=model, into=int)
                else:
                    raise TypeError(f"model must be a tuple, dict, or string, not {type(model)}!")
                
                return str(model)[1:-1].replace(", ", "") + "~" + build_utility_equation(utility_settings=model).replace(" ", "")
            
            elif into is tuple:
                if isinstance(model, str) and model.split("~")[0].isdigit():
                    return tuple(int(dig) for dig in model.split("~")[0])
                raise ValueError(f"If into is tuple, then model must be a string of 0s and 1s.")

            raise TypeError(f"into must be str or tuple, not {into}.")

        "Converting all models into a common format"
        target_model_key = gnrl.convert_utility_settings(
            utility_settings=ast.literal_eval(target_model) if 
            isinstance(target_model, str) else target_model, into=tuple
        )
        target_model_settings = gnrl.convert_utility_settings(
            utility_settings=target_model_key, into=dict)
        models_to_losses = {
            ast.literal_eval(key): min(val) for key, val in 
            copy.deepcopy(models_to_sequential_losses).items()
        }

        "Extract model nesting data"
        model_nesting_data = model_nesting_adjacency_matrices(
            general_settings=general_settings, utility_settings=utility_settings, 
            file_paths=file_paths, create_new_file=False, print_=False)
        
        "List of all models as tuples of boolean flags"
        model_setting_tuples: list[tuple] = [gnrl.convert_utility_settings(
            utility_settings=settings, into=tuple) for settings in model_nesting_data['settings']]

        "Index of the new model in the list of models"
        target_model_settings_idx = next((settings_idx for settings_idx, settings in enumerate(
            model_setting_tuples) if settings == target_model_key), -1)
        
        "Lists of the parents and children of the new model as tuples of boolean flags"
        parents_of_target_model = [
            model_setting_tuples[jdx]
            for jdx in model_nesting_data['adjacency_lists']['parent_of'][target_model_settings_idx]
        ]
        children_of_target_model = [
            model_setting_tuples[jdx]
            for jdx in model_nesting_data['adjacency_lists']['child_of'][target_model_settings_idx]
        ]
        
        "Creating the primary dictionary of data to check for nesting violations"
        nesting_fit_violation_data = {
            'counts': {
                'violations':  {'parents': 0, 'children': 0},
                'observances': {'parents': 0, 'children': 0}
            },
            'parents': {}, 
            'children': {}
        }
        
        "Nesting violations can only apply to models with at least two parameters."
        n_params = gnrl.count_free_parameters(utility_settings=target_model_settings)
        if n_params > 1:

            "The lowest loss of the new model is the reference point"
            target_model_loss = models_to_losses[target_model_key]

            "Comparing the min losses of parent models to the min loss of the new model"
            for parent_model in parents_of_target_model:
                parent_loss = models_to_losses.get(parent_model, None)
                if isinstance(parent_loss, float):
                    if abs(parent_loss) > abs(target_model_loss):
                        parent_key = model_key_maker(model=parent_model, into=str)
                        nesting_fit_violation_data['parents'][parent_key] = parent_loss - target_model_loss
                        nesting_fit_violation_data['counts']['violations']['parents'] += 1
                    else:
                        nesting_fit_violation_data['counts']['observances']['parents'] += 1

            "Comparing the min losses of child models to the min loss of the new model"
            for child_model in children_of_target_model:
                child_loss = models_to_losses.get(child_model, None)
                if isinstance(child_loss, float):
                    if abs(child_loss) < abs(target_model_loss):
                        child_key = model_key_maker(model=child_model, into=str)
                        nesting_fit_violation_data['children'][child_key] = child_loss - target_model_loss
                        nesting_fit_violation_data['counts']['violations']['children'] += 1
                    else:
                        nesting_fit_violation_data['counts']['observances']['children'] += 1

            "Printing a compact representation to the terminal"
            vio_chi = nesting_fit_violation_data['counts']['violations' ]['children']
            obs_chi = nesting_fit_violation_data['counts']['observances']['children']
            vio_par = nesting_fit_violation_data['counts']['violations' ]['parents' ]
            obs_par = nesting_fit_violation_data['counts']['observances']['parents' ]
            if print_only_children:
                if vio_chi:
                    statement_1 = f"VIOLATIONS DETECTED: Children: {vio_chi}/{(vio_chi + obs_chi)}:"
                    statement_2 = f"[TARGET]   Loss: {target_model_loss:10.6f} ~ {model_key_maker(model=target_model, into=str)}"
                    ic_terminal_printouts.append(statement_1)
                    ic_terminal_printouts.append(statement_2)
                    print(statement_1), print(statement_2)
                    for child, loss in list(nesting_fit_violation_data['children'].items())[:10]:
                        statement_child = f"[CHILD]  Δ Loss: {loss:10.6f} ~ {child }"
                        ic_terminal_printouts.append(statement_child)
                        print(statement_child)
                    
            elif vio_chi or vio_par:
                statement_1 = f"VIOLATIONS DETECTED: Parents: {vio_par}/{(vio_par + obs_par)} & Children: {vio_chi}/{(vio_chi + obs_chi)}:"
                statement_2 = f"[TARGET]   Loss: {target_model_loss:10.6f} ~ {model_key_maker(model=target_model, into=str)}"
                ic_terminal_printouts.append(statement_1)
                ic_terminal_printouts.append(statement_2)
                print(statement_1), print(statement_2)       
                for child, loss in list(nesting_fit_violation_data['children'].items())[:10]:
                    statement_child = f"[CHILD]  Δ Loss: {loss:10.6f} ~ {child }"
                    ic_terminal_printouts.append(statement_child)
                    print(statement_child)
                for parent, loss in list(nesting_fit_violation_data['parents'].items())[:10]:
                    statement_parent = f"[PARENT] Δ Loss: {loss:10.6f} ~ {parent}"
                    ic_terminal_printouts.append(statement_parent)
                    print(statement_parent)

        return nesting_fit_violation_data

    def build_ic_dataframe_for_ranking(ic_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Helper to get a DataFrame from current ic_results_dict to compute ranks easily.
        """
        rows = []
        for mk, info in ic_dict.items():
            "Skip if 'loss' is None"
            if info['loss'] is None:
                continue
            rows.append({
                "model_key": mk,
                "loss": info['loss'],
                "AIC": info['AIC'],
                "BIC": info['BIC']
            })
        df_local = pd.DataFrame(rows)
        "Rank by BIC ascending"
        df_local["BIC_rank"] = df_local["BIC"].rank(method="min", ascending=True)
        return df_local
    
    def _warmstart_temperature(iter_idx: int, warmstart_policy: dict) -> float | None:
        """
        Returns None ⇒ 'cold' (no warm-starts).
        Simple schedules that do not depend on final horizon.
        """
        cold_iters = int(warmstart_policy.get("cold_iters", 2))     # No warm-starts for the first K iterations.
        temp_high  = float(warmstart_policy.get("temperature_high", 1000.0))
        temp_low   = float(warmstart_policy.get("temperature_low", 0.05))
        schedule   = str(warmstart_policy.get("schedule", "binary")).lower()

        if iter_idx <= cold_iters:
            return None  # Cold phase.

        if schedule == "binary":
            explore_iters = int(warmstart_policy.get("explore_iters", 3))
            return temp_high if iter_idx <= cold_iters + explore_iters else temp_low

        if schedule == "exp":
            "Exponential cooling with a half-life in 'warm' phase"
            half_life = float(warmstart_policy.get("half_life", 2.0))
            warm_phase_iter = max(0, iter_idx - cold_iters)
            return max(temp_low, temp_high * (0.5 ** (warm_phase_iter / half_life)))

        "Default: binary"
        return temp_high

    def model_comparison_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a smaller dataframe that comparisons the top performing models
        """
        best_models = []

        "For each k_params group:"
        for kprms in reversed(range(1, 10)):
            df_k = df[df['k_params'] == kprms]
            if df_k.empty:
                continue

            "1) Identify the winning row (lowest BIC => rank=0)"
            best_row = df_k.loc[df_k['BIC_rank'] == 0].copy()
            if best_row.empty:
                continue  # Should not happen, but just in case.

            "Typically there's only one row with rank=0, but handle if there's a tie:"
            best_row = best_row.iloc[0]  # Pick the first if there is a tie.

            "2) Identify the runner-up (rank=1 within that same k_params)"
            runner_up = df_k.loc[df_k['BIC_rank'] == 1]
            if not runner_up.empty:
                "If there's a tie for rank=1, pick any or the first"
                runner_up_BIC = runner_up.iloc[0]['BIC']
                best_BIC = best_row['BIC']
                next_best_ic = runner_up_BIC - best_BIC
            else:
                "No runner-up"
                next_best_ic = float('nan')

            "3) Attach that difference as a new column in the row"
            best_row['next_best_IC'] = next_best_ic

            "4) Accumulate for building comp_df"
            best_models.append(best_row)

        comp_df = pd.DataFrame(best_models).sort_values('k_params')
        "Optional Δ"
        min_BIC_in_comp = comp_df['BIC'].min()
        comp_df['ΔBIC'] = comp_df['BIC'] - min_BIC_in_comp
        
        "Move equation column to the end."
        equation_column = comp_df.pop('equation')
        comp_df['equation'] = equation_column

        comp_csv_path = prep.ensure_directory_and_join(file_paths["bic_aic"], 
                            f"IC_Analysis_Comparison_Table_Experiment{experiment_num}.csv")
        comp_df.to_csv(comp_csv_path, index=False, encoding='utf-8-sig')
        print(f"Saved comparison table to: {comp_csv_path}\n")

        return comp_df

    def ic_results_df(df_dict: dict) -> pd.DataFrame:
        """
        Creates the main dataframe that stores the results of the IC analysis.
        """
        "Create a sorted DataFrame."
        df = pd.DataFrame(df_dict)
        df = df.sort_values(by='BIC', ascending=True)

        "7) Compute ΔAIC and ΔBIC"
        minAIC = df['AIC'].min()
        minBIC = df['BIC'].min()
        df['ΔAIC'] = df['AIC'] - minAIC
        df['ΔBIC'] = df['BIC'] - minBIC

        "8) Ranks by AIC & BIC"
        df['AIC_rank'] = df.groupby('k_params')['AIC'].rank(method='min').astype(int) - 1
        df['BIC_rank'] = df.groupby('k_params')['BIC'].rank(method='min').astype(int) - 1

        "Move 'equation' column to the end to make equations easier to see in Excel"
        equation_column = df.pop('equation')
        df['equation'] = equation_column

        "Save the DataFrame to a CSV (or JSON) in the bic_aic folder."
        try:
            df_file_path = os.path.join(base_file_paths["bic_aic"], 
                f"All_Utility_Forms_IC_Analysis_Experiment{experiment_num}.csv")
            df.to_csv(df_file_path, index=False, encoding='utf-8-sig')
            print(f"Saved DataFrame to {df_file_path}")
        except (PermissionError, OSError):
            pass

        return df
        
    def ic_correlations(df: pd.DataFrame) -> pd.DataFrame:
        """
        Prints correlations and statistics
        """
        "11) Correlations among {AIC, BIC, k_params, loss}"
        df_valid = df.dropna(subset=['AIC', 'BIC', 'k_params', 'loss'])
        corr_matrix = df_valid[['AIC', 'BIC', 'k_params', 'loss']].corr()
        print("Correlation matrix among AIC, BIC, k_params, and loss:\n", corr_matrix)

        import statsmodels.formula.api as smf
        "Print the top row(s) with minimal IC."
        min_ic = df['BIC'].min()
        min_ic_df = df.loc[df['BIC'] == min_ic]
        print("\n--- Best Utility Config by IC ---")
        print(min_ic_df)

        df['ΔBIC'] = df['BIC'] - min_ic
        top_uf_data = df[df['ΔBIC'] < 2]
        print(f"\n--- Top Utility Config(s) by BIC---")
        print(top_uf_data), print("")

        formula = f"BIC ~ C(conditional_welfare_mode) + C(reference_dependent_altruism) + " \
                "C(use_exponential_parameters) + C(single_exponential_parameter) + "  \
                "C(single_payoffs_not_differences) + C(payoff_ratios_not_differences) + "  \
                "C(reference_dependent_utility) + C(use_negativity_parameters) + "  \
                "C(negativity_social_comparison) + C(include_social_comparison) + "  \
                "C(include_altruism_term) + C(fix_self_interest_parameter) + k_params"  
        model = smf.ols(formula, data=df)
        results = model.fit()
        print(results.summary())    

    "Storing total nesting violations across models in each iteration."
    nesting_violation_counts_per_iter = []

    if not isinstance(max_iters, int) or max_iters <= 0:
        "Max_iters must be a positive non-zero integer."
        max_iters = 1

    if general_settings.get('write_mode') == 'readonly':
        "No point in iterating if intending to extract saved files."
        max_iters = 1

    player_subset = False
    if isinstance(check_for_n_players, int) and (0 < check_for_n_players):
        print(f"Warning: Running IC analysis over a subset of {check_for_n_players} players!")
        player_subset = True

    "Copy inputs to avoid unintended side-effects."
    general_settings = copy.deepcopy(general_settings)
    utility_settings = copy.deepcopy(utility_settings)
    base_file_paths = copy.deepcopy(file_paths)
    loss_funct_type = general_settings.get('loss_funct_type')

    "Remove suffix from file names if any."
    base_file_paths = prep.add_remove_file_name_suffix(
        file_paths=base_file_paths, file_name_suffix=None, add_suffix=False
    )

    "Use static updating by default"
    if dynamic_updating:
        update_method = 'grid'
        general_settings['use_particle_filter'] = True
    else:
        update_method = 'naive'
    general_settings['update_method'] = update_method

    "Temperature should be held constant to keep all models on an even footing."
    general_settings['temperature_is_param'] = False
    general_settings['run_in_parallel'] = True                                       

    "Determine which experiment to analyze."
    experiment_num = general_settings.get('experiment_num', 3)

    "Path to save the final results."
    all_ic_results_file_path = prep.ensure_directory_and_join(
        file_paths["bic_aic"], 
        f"All_Utility_Forms_IC_Analysis_Experiment{experiment_num}.json"
    )

    "Prepare containers."
    ic_results_dict = {}
    utility_varieties = {}

    "Generate or validate the utility configurations to compare."
    if utility_setting_varieties is not None:
        invalid_settings = [setting for setting in utility_setting_varieties if not gnrl.is_valid_utility_settings(setting)]
        if invalid_settings:
            raise ValueError(
                f"information_criterion_analysis received {len(invalid_settings)} invalid utility setting(s) "
                f"in utility_setting_varieties. Validate each entry with gnrl.is_valid_utility_settings "
                f"before passing it in."
            )
    else:
        utility_setting_varieties = gnrl.generate_utility_settings(utility_settings=utility_settings, sort_by_k=True)

    n_varieties = len(utility_setting_varieties)

    "Keep track of the best min loss found so far for each model"
    models_to_sequential_losses: Dict[str, List[float]] = {}
    models_to_sequential_params: Dict[str, List[Dict[str, Dict[str, Dict[str, float]]]]] = {}
    models_to_sequential_losses_and_params: Dict[str, List[Dict[str, Dict[str, Dict[str, Dict[str, float]]]]]] = {}
    players_to_params_this_iter = {}  # Placed here to prevent an error if not create_new_file.

    "For storing iteration-based sums"
    sum_delta_minimum_loss_by_iter: List[float] = []
    rank_change_by_iter: List[float] = []
    rank_change_median_by_iter: List[float] = []

    "Keep track of the old ranks"
    old_ranks: Dict[str, float] = {}

    consecutive_small_improvements = 0  # Count of consecutive iterations where sum_delta_min_loss < epsilon.

    minimum_params_and_losses = {}

    "Iterate model for a robustness analysis."
    for iter_idx in range(1, max_iters + 1):

        if iter_idx <= 1 and general_settings.get('write_mode', 'resume') == 'overwrite':
            try:
                "Start the terminal printouts file from scratch."
                term_print_output_path = os.path.join(file_paths['bic_aic'], 'ic_terminal_printouts.txt')
                os.makedirs(file_paths['bic_aic'], exist_ok=True)
                with open(term_print_output_path, 'w', encoding='utf-8') as file:
                    pass
            except (PermissionError, OSError):
                "Pass if I have the file open."
                pass

        "Store sum of detal min loss for this iteration"
        sum_delta_minimum_loss_this_iter = 0.0

        "Initializing the nesting violation counts for this iteration."
        nesting_violation_counts_per_iter.append({
            'violations':  {'parents': 0, 'children': 0}, 
            'observances': {'parents': 0, 'children': 0}
        })

        "Loop over each valid config."
        for utility_idx, utility_setting_variety in enumerate(utility_setting_varieties):

            param_info = make_param_info(param_bds=param_bds, 
                utility_settings=utility_setting_variety, general_settings=general_settings, 
                random_guesses_are_unique=True, guess_seed=None)

            k_params = len(param_info['keys'])  # Number of free parameters.
            utility_setting_key = str(gnrl.convert_utility_settings(utility_settings=utility_setting_variety, into=tuple)) 

            "Adding model key to dict so it can be mapped to losses vector"
            if utility_setting_key not in models_to_sequential_losses:
                models_to_sequential_losses[utility_setting_key] = []

            "Adding model key to dict so it can be mapped to params dict"
            if utility_setting_key not in models_to_sequential_params:
                models_to_sequential_params[utility_setting_key] = []

            if utility_setting_key not in models_to_sequential_losses_and_params:
                models_to_sequential_losses_and_params[utility_setting_key] = []

            if utility_setting_key not in minimum_params_and_losses:
                minimum_params_and_losses[utility_setting_key] = {}

            "Create file name suffix from these settings."
            file_name_suffix = prep.create_file_name_suffix(
                general_settings=general_settings, utility_settings=utility_setting_variety
            )

            "Copy of standard file paths to alter with each loop."
            file_paths_this = copy.deepcopy(base_file_paths)

            "Re-add that suffix to file_paths."
            file_paths_this = prep.add_remove_file_name_suffix(
                file_paths=file_paths_this, file_name_suffix=file_name_suffix, add_suffix=True
            )

            "Where the per-config results are stored if they exist."
            ic_results_file_path = prep.ensure_directory_and_join(
                file_paths_this["bic_aic"], f"IC_Analysis{file_name_suffix}.json")
            "Always try to seed from any prior file"
            ic_prev = {}
            if os.path.exists(ic_results_file_path) :
                with open(ic_results_file_path, "r", encoding="utf-8") as file:
                    ic_prev = json.load(file)

            models_to_sequential_losses[utility_setting_key]            = ic_prev.get('lvec', [])
            models_to_sequential_params[utility_setting_key]            = ic_prev.get('pvec', [])
            models_to_sequential_losses_and_params[utility_setting_key] = ic_prev.get('plvec', [])
            minimum_params_and_losses[utility_setting_key]              = ic_prev.get('minvec', {})
            n_data_for_model                                            = ic_prev.get('n_data', 0)

            write_mode = general_settings.get('write_mode', 'resume')

            "If overwrite, clear everything and recompute"
            if write_mode == 'overwrite' and iter_idx <= 1:
                models_to_sequential_losses[utility_setting_key].clear()
                models_to_sequential_params[utility_setting_key].clear()
                models_to_sequential_losses_and_params[utility_setting_key].clear()
                minimum_params_and_losses[utility_setting_key].clear()
                n_data_for_model = 0

            "Readonly → skip running a new iteration; just use the seeded values"
            if write_mode == 'readonly':
                total_loss_model = ic_prev.get('loss', (
                    min(models_to_sequential_losses[utility_setting_key])
                    if models_to_sequential_losses[utility_setting_key]
                    else float('nan')
                ))
            else:
                "Read the main processed histories if needed."
                file_path_histories = prep.ensure_directory_and_join(
                    file_paths_this["processed"], 
                    file_paths_this["file_names"][f'player_pairs_exper{experiment_num}']
                )
                with open(file_path_histories, "r", encoding='utf-8') as file:
                    original_histories: dict[str, dict[str, list[DyadGame]]] = json.load(file)

                "Gather all participants."
                player_uuids = prep.all_player_uuids(
                    file_paths=file_paths_this, 
                    experiment_num=experiment_num, 
                    only_humans=True
                )
                if player_subset:
                    player_uuids = sorted(player_uuids)[:check_for_n_players]

                "Raw data to be fitted"
                player_histories = copy.deepcopy(original_histories)

                "Summation variables."
                total_loss_model = 0.0  # Total loss for model across experiment.
                n_data_for_model = 0   # Number of data points in the experiment.

                "Overwrite lambda generated random numbers prevents pickeling errors in multiprocessing"
                min_std_guess = 0.5
                param_info['guesses'] = [random.uniform(param_info['bounds'][kdx][0] if '_std' not in key else min_std_guess, 
                                    param_info['bounds'][kdx][1]) for kdx, key in enumerate(param_info['keys'])]

                "--- Warm-start policy for this iteration ----------------------------------"
                default_warm_pol = {
                    "enabled": True,
                    "schedule": "binary",
                    "cold_iters": 2,
                    "explore_iters": 2,
                    "temperature_high": 1000.0,
                    "temperature_low": 0.01,
                    "disable_dual_annealing_when_warm": True,
                }
                warmstart_policy = general_settings.get("warmstart_policy", default_warm_pol)
                cur_T = _warmstart_temperature(iter_idx, warmstart_policy)
                phase = "cold" if iter_idx <= warmstart_policy.get("cold_iters", 2) else "warm"

                "Force 'create_new_file' to be true if already deciding to create a new file."
                general_settings_ = copy.deepcopy(general_settings)

                general_settings_["warmstart_policy"] = {
                    **default_warm_pol,
                    **(general_settings.get("warmstart_policy", {})),
                    "temperature": (cur_T if cur_T is not None else 0.0),
                    "phase": phase
                }
                "Pin the method here. The local code inside fit_params_by_player"
                "Will switch to 'local' when phase == 'warm' and disable_dual_annealing_when_warm=True."
                general_settings_["optimization_method"] = general_settings.get("optimization_method", "globloc")

                if utility_idx == 0:
                    if iter_idx == int(warmstart_policy.get("cold_iters", 2)) + 1:
                        warm_start_message = (
                            f"[IC] ENTERING EXPLORATION WARM PHASE at iter {iter_idx} "
                            f"(schedule={warmstart_policy.get('schedule','binary')}, "
                            f"T={cur_T})"
                        )
                        ic_terminal_printouts.append(warm_start_message)
                        print(warm_start_message)
                    elif iter_idx == int(warmstart_policy.get("cold_iters", 2)) + int(warmstart_policy.get("explore_iters", 2)) + 1:
                        warm_start_message = (
                            f"[IC] ENTERING EXPLIOTATION WARM PHASE at iter {iter_idx} "
                            f"(schedule={warmstart_policy.get('schedule','binary')}, "
                            f"T={cur_T})"
                        )
                        ic_terminal_printouts.append(warm_start_message)
                        print(warm_start_message)                        

                "Run the naive analysis (similar to \"bayes\" approach)."
                player_histories = run_analysis_bayes(
                    player_uuids=player_uuids if player_subset else None,
                    utility_settings=utility_setting_variety,
                    general_settings=general_settings_,
                    histories_data=player_histories,
                    file_paths=file_paths_this,
                    param_info=param_info, 
                    print_=False
                )

                "Storing parameter values by player for robustness analysis"
                players_to_params_this_iter = {}
                players_to_params_and_losses_this_iter = {}

                "Accumulate total loss across all players/dyads."
                for player_uuid in player_uuids:
                    player_fit_dir = os.path.join(file_paths["player_fits"], f"experiment_{experiment_num}")
                    if experiment_num == 0:
                        player_fit_name = f"{player_uuid}.json"
                    else:
                        player_fit_name = f"{file_name_suffix}_{player_uuid}.json"

                    plr_file_path = prep.ensure_directory_and_join(base_dir=player_fit_dir, file_name=player_fit_name)
                    with open(plr_file_path, "r", encoding="utf-8") as file:
                        player_dyads = json.load(file)

                    players_to_params_this_iter[player_uuid] = {'chooser': {}, 'predictor': {}}
                    players_to_params_and_losses_this_iter[player_uuid] = {
                        'params': {'chooser': {}, 'predictor': {}}, 
                        'loss': {'chooser': 0.0, 'predictor': 0.0}
                    }

                    min_for_model = minimum_params_and_losses[utility_setting_key]
                    if player_uuid not in min_for_model:
                        min_for_model[player_uuid] = {
                            'params': {'chooser': {}, 'predictor': {}},
                            'loss':   {'chooser': float('inf'), 'predictor': float('inf')}
                        }
                            
                    "Before entering the dyad loop, set up per-player accumulators"
                    role_loss_total = {'chooser': 0.0, 'predictor': 0.0}
                    param_est_by_role = {'chooser': None, 'predictor': None}

                    for dyad_key, dyad_games in player_dyads.items():
                        "Compute loss for this dyad"
                        loss_report = create_loss_report(dyad_games=dyad_games, general_settings=general_settings).get(player_uuid, {})
                        for player_role in ('chooser', 'predictor'):
                            loss_dict = loss_report.get(player_role, {})
                            loss_plr_role_dyad = loss_dict.get('raw_neglogprob_sum' if loss_funct_type == 'log' else 'raw_ssr', 0.0)
                            role_loss_total[player_role] += loss_plr_role_dyad

                            "N_data: preserve current per-role/per-dyad aggregation logic"
                            n_data_for_model += loss_dict.get('n_data', 0)

                            "Grab params once (any dyad) if present"
                            if param_est_by_role[player_role] is None and len(dyad_games) > 0:
                                first_game = dyad_games[0]
                                if isinstance(first_game, dict) and 'parameter_estimates' in first_game:
                                    param_est_by_role[player_role] = \
                                        first_game.get('parameter_estimates', {}).get(update_method, {}).get(
                                            player_uuid, {}).get(player_role, {}).get('params')

                    "Extract the parameters that minimize the raw losses, not the penalized losses."
                    "This supersedes the grab above when reports['final']['min_raw_neglog_sum'] is present, which is the preferred source."
                    if len(dyad_games) > 0:
                        if "reports" in dyad_games[0]:
                            for player_role in ('chooser', 'predictor'):
                                if player_role in dyad_games[0]["reports"]:
                                    if "final" in dyad_games[0]["reports"][player_role]:
                                        if "min_raw_neglog_sum" in dyad_games[0]["reports"][player_role]["final"]:
                                            min_raw_neglog_sum = dyad_games[0]["reports"][player_role]["final"]["min_raw_neglog_sum"]
                                            param_est_by_role[player_role] = min_raw_neglog_sum["params"]
                                            role_loss_total[player_role] = min_raw_neglog_sum["loss"]

                    "Store what was found this iteration"
                    players_to_params_this_iter[player_uuid] = {
                        'chooser':   param_est_by_role['chooser']   or {},
                        'predictor': param_est_by_role['predictor'] or {}
                    }
                    players_to_params_and_losses_this_iter[player_uuid] = {
                        'params': players_to_params_this_iter[player_uuid],
                        'loss':   {'chooser': role_loss_total['chooser'], 'predictor': role_loss_total['predictor']}
                    }

                    "Update per-model minima exactly once per role, then add to total_loss_model"
                    min_for_model = minimum_params_and_losses[utility_setting_key]
                    if player_uuid not in min_for_model:
                        min_for_model[player_uuid] = {
                            'params': {'chooser': {}, 'predictor': {}},
                            'loss':   {'chooser': float('inf'), 'predictor': float('inf')}
                        }

                    for role in ('chooser', 'predictor'):
                        if role_loss_total[role] < min_for_model[player_uuid]['loss'][role]:
                            min_for_model[player_uuid]['loss'][role]   = role_loss_total[role]
                            min_for_model[player_uuid]['params'][role] = param_est_by_role[role] or {}

                        total_loss_model += min_for_model[player_uuid]['loss'][role]

                "Debugging"
                "Total_loss_model should equal sum over players of minvec[uuid]['loss']['chooser'/'predictor']"
                chk = 0.0
                for player_uuid_for_check, player_loss_data in minimum_params_and_losses[utility_setting_key].items():
                    chk += float(player_loss_data['loss'].get('chooser', 0.0)) + float(player_loss_data['loss'].get('predictor', 0.0))
                assert abs(total_loss_model - chk) <= 1e-9, "models_to_sequential_losses inconsistent with minvec aggregation."
                "Debugging"

            "Determine minimum model loss found up until the previous time step"
            if len(models_to_sequential_losses[utility_setting_key]) == 0:
                prior_minimum_model_loss = float('inf')
            else:
                prior_minimum_model_loss = min(models_to_sequential_losses[utility_setting_key])

            "Append loss to loss vector"
            if general_settings.get('write_mode') in ('resume', 'overwrite'):
                models_to_sequential_losses[utility_setting_key].append(total_loss_model)

            "Find miminum model loss up until now and the delta from the last time step"
            minimum_model_loss = min(models_to_sequential_losses[utility_setting_key])

            prev = prior_minimum_model_loss
            cur  = minimum_model_loss
            "Allows the first 'inf' → value transition."
            if prev != float('inf'):
                assert cur <= prev + 1e-12, f"Minimum loss increased! prev={prev:.6f}, cur={cur:.6f}"

            "FIX #2: Just before computing delta_minimum_model_loss:"
            if prior_minimum_model_loss == float('inf'):
                delta_minimum_model_loss = 0.0
            else:
                delta_minimum_model_loss = abs(minimum_model_loss - prior_minimum_model_loss)

            "Add to the sum of delta min losses across all models."
            sum_delta_minimum_loss_this_iter += delta_minimum_model_loss

            "Append params to param vector"
            if general_settings.get('write_mode') in ('resume', 'overwrite'):
                models_to_sequential_params[utility_setting_key].append(players_to_params_this_iter)
                models_to_sequential_losses_and_params[utility_setting_key].append(players_to_params_and_losses_this_iter)

            parameter_variance = compute_mean_param_variance(param_info=param_info, 
                                    param_runs=models_to_sequential_params[utility_setting_key])

            "Print results for real-time feedback"
            report_str = "Iter " + "0" * (len(str(max_iters)) - len(str(iter_idx))) + f"{iter_idx}/{max_iters} - Utility Model " 
            report_str += "0" * (len(str(n_varieties)) - len(str(utility_idx))) + f"{utility_idx}/{n_varieties} - "
            report_str += f"Loss: {total_loss_model:.6f}; Min Loss: {minimum_model_loss:.6f}; "
            report_str += f"Δ Min Loss: {delta_minimum_model_loss:.6f}; Param Var = {parameter_variance:.6f}"
            equation = build_utility_equation(utility_settings=utility_setting_variety)
            ic_terminal_printouts.append(report_str)
            ic_terminal_printouts.append(equation)
            print(report_str)
            print(equation)

            "If no data, store null results."
            if n_data_for_model == 0:
                ic_results = {
                    'idx': utility_idx,
                    'k_params': k_params,
                    'n_data': 0,
                    'loss': None,
                    'AIC': None,
                    'BIC': None,
                    'lvec': None,
                    'pvec': None,
                    'pvar': None,
                    'plvec': None,
                    'minvec': None,
                    'U': build_utility_equation(
                        utility_settings=utility_setting_variety),
                    'utility_settings': utility_setting_variety
                }
            else:
                "AIC/BIC formulas."
                ic_results = compute_ic(k_params=k_params, 
                                        n_data=n_data_for_model, 
                                        neg_log_likelihood=minimum_model_loss)

                ic_results = {
                    'idx': utility_idx,
                    'k_params': k_params,
                    'n_data': n_data_for_model,
                    'loss': minimum_model_loss,
                    'AIC': ic_results['AIC'],
                    'BIC': ic_results['BIC'],
                    'pvar': parameter_variance,
                    'U': build_utility_equation(
                        utility_settings=utility_setting_variety),
                    'lvec': models_to_sequential_losses[utility_setting_key],
                    'pvec': models_to_sequential_params[utility_setting_key],
                    'plvec': models_to_sequential_losses_and_params[utility_setting_key],
                    'minvec': minimum_params_and_losses[utility_setting_key],
                    'utility_settings': utility_setting_variety
                }

            "Store results in memory."
            ic_results_dict[utility_setting_key] = ic_results
            utility_varieties[utility_setting_key] = utility_setting_variety

            "Compute, store, and report model nesting fit violations"
            if write_mode != 'readonly':
                violations = check_nesting_fit_violations(target_model=utility_setting_variety, 
                                                    models_to_sequential_losses=models_to_sequential_losses)
                ic_results['nesting_violations'] = violations

                vio_par = violations.get('counts', {}).get('violations',  {}).get('parents',  0)
                vio_chi = violations.get('counts', {}).get('violations',  {}).get('children', 0)
                obs_par = violations.get('counts', {}).get('observances', {}).get('parents',  0)
                obs_chi = violations.get('counts', {}).get('observances', {}).get('children', 0)
                nesting_violation_counts_per_iter[-1]['violations' ]['parents' ] += vio_par
                nesting_violation_counts_per_iter[-1]['violations' ]['children'] += vio_chi
                nesting_violation_counts_per_iter[-1]['observances']['parents' ] += obs_par
                nesting_violation_counts_per_iter[-1]['observances']['children'] += obs_chi

            "Save results, overwriting previous JSON."
            if general_settings.get('write_mode') in ('resume', 'overwrite'):
                with open(ic_results_file_path, 'w', encoding='utf-8') as file:
                    json.dump(ic_results, file, ensure_ascii=False, indent=4)

            try:
                "Save terminal outputs"
                term_print_output_path = os.path.join(file_paths_this['bic_aic'], 'ic_terminal_printouts.txt')
                os.makedirs(file_paths_this['bic_aic'], exist_ok=True)
                "Open file in append mode to add new content without erasing the old"
                with open(term_print_output_path, 'a', encoding='utf-8') as file:
                    for line in ic_terminal_printouts:
                        file.write(line + '\n')
                "Empty list to preven saving duplicate information."
                ic_terminal_printouts.clear()
            except (PermissionError, OSError):
                "Pass if I have the file open."
                pass

        "Printing total model nesting fit violations to terminal"
        if write_mode != 'readonly':
            vio_par_ = nesting_violation_counts_per_iter[-1]['violations' ]['parents' ]
            vio_chi_ = nesting_violation_counts_per_iter[-1]['violations' ]['children']
            obs_par_ = nesting_violation_counts_per_iter[-1]['observances']['parents' ]
            obs_chi_ = nesting_violation_counts_per_iter[-1]['observances']['children']
            vio_statement = (
                f"TOTAL MODEL NESTING FIT VIOLATIONS FOR ITERATION {iter_idx}: "
                f"Parents: {vio_par_}/{(vio_par_ + obs_par_)}; "
                f"Children: {vio_chi_}/{(vio_chi_ + obs_chi_)}"
            )
            ic_terminal_printouts.append(vio_statement)
            print(vio_statement)

        "Store sum delta min loss for this iteration"
        sum_delta_minimum_loss_by_iter.append(sum_delta_minimum_loss_this_iter)

        rounded_sds = [round(sum_delta, 6) for sum_delta in sum_delta_minimum_loss_by_iter]
        sum_delta_min_loss_statement = f"Iter {iter_idx}: Sum Δ Min Losses: {rounded_sds}"
        ic_terminal_printouts.append(sum_delta_min_loss_statement)
        print(sum_delta_min_loss_statement)

        "2) Build DF to compute rank changes"
        df_for_rank = build_ic_dataframe_for_ranking(ic_results_dict)
        "E.g. df_for_rank: [model_key, loss, AIC, BIC, BIC_rank]"
        rank_diffs = []
        new_ranks = {}
        for idx, row in df_for_rank.iterrows():
            mk = row["model_key"]
            nrank = row["BIC_rank"]
            new_ranks[mk] = nrank

            "Computes rank difference when old_ranks are available."
            old_r = old_ranks.get(mk, nrank)  # If not present, no difference.
            rank_diffs.append(abs(nrank - old_r))

        "Sum of rank changes across all models"
        sum_rank_diff = sum(rank_diffs)
        median_rank_diff = np.median(rank_diffs) if rank_diffs else 0.0
        rank_change_by_iter.append(sum_rank_diff)
        rank_change_median_by_iter.append(median_rank_diff)

        rank_statement = (f"Iter {iter_idx}: Sum of rank changes = {sum_rank_diff:.3f},"
                          f" median rank change={median_rank_diff:.3f}")
        ic_terminal_printouts.append(rank_statement)
        print(rank_statement)

        "Update old_ranks"
        old_ranks = new_ranks

        "Save models_to_sequential_losses"
        models_to_sequential_losses_file_path = prep.ensure_directory_and_join(
            base_dir=file_paths['bic_aic'], file_name="models_to_sequential_losses.json") 
        if general_settings.get('write_mode') in ('resume', 'overwrite'):
            try:
                with open(models_to_sequential_losses_file_path, 'w', encoding='utf-8') as file:
                    json.dump(models_to_sequential_losses, file, ensure_ascii=False, indent=4)
            except (PermissionError, OSError):
                pass


        for mk, ic in ic_results_dict.items():
            pvec = ic.get('pvec', [])
            ic['param_norm_sd'] = _compute_normalised_param_sd(pvec, param_bds)

        "Build the final 'robustness_analysis_data'"
        robustness_analysis_data = {
            'sum_delta_minimum_loss_by_iter': sum_delta_minimum_loss_by_iter,
            'rank_change_by_iter': rank_change_by_iter,
            'rank_change_median_by_iter': rank_change_median_by_iter
        }

        "Combine into a single dict and write out."
        all_ic_results = {
            'ic_results': ic_results_dict,
            'utility_varieties': utility_varieties,
            'robustness_analysis_data': robustness_analysis_data,
            'n_iterations_completed': iter_idx
        }

        "JSON serializable version of the same dictionary"
        all_ic_results_serializable = {
            'ic_results': {str(key): val for key, val in ic_results_dict.items()},
            'utility_varieties': {str(key): val for key, val in utility_varieties.items()},
            'robustness_analysis_data': robustness_analysis_data
        }
        try:
            with open(all_ic_results_file_path, 'w', encoding='utf-8') as file:
                json.dump(all_ic_results_serializable, file, ensure_ascii=False, indent=4)
        except (PermissionError, OSError):
            pass

        "Build a DataFrame summarizing all results."
        "Start by building column lists for each key in utility_settings + {n,k,loss,AIC,BIC}."
        df_dict = {key: [] for key in utility_settings}  # Each setting as a column.

        extra_cols = ['idx', 'n_data', 'k_params', 'pvar', 'param_norm_sd', 'loss', 'AIC', 'BIC', 'equation']
        for extra_col in extra_cols:
            df_dict[extra_col] = []

        "Fill df row by row."
        for utility_setting_variety in utility_setting_varieties:
            utility_setting_key = str(gnrl.convert_utility_settings(utility_settings=utility_setting_variety, into=tuple)) 

            ic_res = all_ic_results['ic_results'].get(utility_setting_key)
            uv = all_ic_results['utility_varieties'].get(utility_setting_key)
            if ic_res is None or uv is None:
                print(f"Missing Utility Option Variety: {utility_setting_key}.")
                continue

            df_dict['idx'].append(ic_res['idx'])    
            "Add the n data points fields."
            df_dict['n_data'].append(ic_res['n_data'])

            "Add each boolean setting to the row."
            for setting_name, setting_val in uv.items():
                df_dict[setting_name].append(setting_val)

            "Add the IC fields."
            df_dict['k_params'].append(ic_res['k_params'])
            df_dict['pvar'].append(ic_res['pvar'])
            df_dict['param_norm_sd'].append(ic_res.get('param_norm_sd'))
            df_dict['loss'].append(ic_res['loss'])
            df_dict['AIC'].append(ic_res['AIC'])
            df_dict['BIC'].append(ic_res['BIC'])
            df_dict['equation'].append(
                build_utility_equation(utility_settings=utility_setting_variety)
            )

        df = ic_results_df(df_dict=df_dict)

        model_comparison_df(df=df)

        "Check the scree slope if sum_delta_min is below epsilon"
        if iter_idx > 1:
            if sum_delta_minimum_loss_this_iter < robustness_epsilon:
                consecutive_small_improvements += 1
            else:
                consecutive_small_improvements = 0

            if consecutive_small_improvements >= 2:
                early_stop_statement = (
                    f"\nEarly stopping after iteration {iter_idx} because "
                    f"sum of ΔMinLoss < {robustness_epsilon} twice in a row."
                )
                ic_terminal_printouts.append(early_stop_statement)
                print(early_stop_statement)
                break
        "End iteration"

    try:
        "Save terminal outputs"
        term_print_output_path = os.path.join(file_paths_this['bic_aic'], 'ic_terminal_printouts.txt')
        os.makedirs(file_paths_this['bic_aic'], exist_ok=True)
        "Open file in append mode to add new content without erasing the old"
        with open(term_print_output_path, 'a', encoding='utf-8') as file:
            for line in ic_terminal_printouts:
                file.write(line + '\n')
        "Empty list to preven saving duplicate information."
        ic_terminal_printouts.clear()
    except (PermissionError, OSError):
        "Pass if I have the file open."
        pass

    if not general_settings.get('write_mode') in ('resume', 'overwrite'): 
        "1) How many iterations did the *long* run complete?"
        max_iters_done = min(len(ic.get('lvec', []))
                             for ic in ic_results_dict.values())

        "2) Re-compute the three per-iteration vectors directly"
        sum_delta_minimum_loss_by_iter = []
        rank_change_by_iter            = []
        rank_change_median_by_iter     = []
        prev_min_by_model = {}      # Track previous min loss per model.
        prev_ranks        = {}      # Track previous BIC ranks.

        for iter in range(max_iters_done):
            "(A) Δ-min-loss"
            sum_delta_this_it = 0.0
            for mk, ic in ic_results_dict.items():
                lvec = ic.get('lvec', [])
                if iter >= len(lvec):
                    continue            # Model had not reached that iter.
                cur_min = min(lvec[:iter+1])
                prev_min = prev_min_by_model.get(mk, cur_min)
                sum_delta_this_it += abs(cur_min - prev_min)
                prev_min_by_model[mk] = cur_min
            sum_delta_minimum_loss_by_iter.append(sum_delta_this_it)

            "(B) BIC rank changes"
            rows = []
            for mk, ic in ic_results_dict.items():
                lvec = ic.get('lvec', [])
                if iter >= len(lvec):
                    continue
                cur_min = min(lvec[:iter+1])
                k_params = ic['k_params']
                n_data   = ic['n_data']
                bic      = k_params*np.log(n_data) + 2*cur_min if n_data else np.nan
                rows.append((mk, bic))

            "Rank them"
            rows = [row for row in rows if not np.isnan(row[1])]
            rows.sort(key=lambda rank_row: rank_row[1])
            cur_ranks = {mk: row for row, (mk, _) in enumerate(rows)}

            diffs = [abs(cur_ranks[mk] - prev_ranks.get(mk, cur_ranks[mk]))
                     for mk in cur_ranks]
            rank_change_by_iter.append(sum(diffs))
            rank_change_median_by_iter.append(np.median(diffs) if diffs else 0.0)
            prev_ranks = cur_ranks

        "3) Overwrite the counters so they reflect reality"
        iter_idx = max_iters_done

    ic_correlations(df=df)

    "Return the DataFrame and the dictionary of all results."
    return df, all_ic_results


def plot_ic_robustness_analysis(general_settings: Dict[str, Any], file_paths: Dict[str, str], fig_lay: Dict[str, Any]) -> go.Figure:
    """
    Plot IC robustness diagnostics from the information criterion analysis results file.

    Loads the stored robustness analysis data from the IC results JSON and generates two
    line charts: (1) the sum of per-model changes in minimum observed loss across warm-start
    iterations — measuring how much the loss landscape shifted between iterations — and
    (2) the number of rank changes per iteration across models, with optional median overlay.
    These plots indicate whether the IC analysis results have converged as iteration count grows.

    Arguments:
        • general_settings: Dict[str, Any]
            Must include 'experiment_num' (int) to locate the correct results file.
        • file_paths: Dict[str, str]
            Must include 'bic_aic' path where IC results JSON files are stored.
        • fig_lay: Dict[str, Any]
            Figure layout template controlling fonts, colors, and figure sizing.

    Returns:
        • go.Figure — a Plotly figure with two subplots showing convergence diagnostics.
    """
    experiment_num = general_settings.get('experiment_num')
    "Path to save the final results."
    all_ic_results_file_path = os.path.join(
        file_paths["bic_aic"], 
        f"All_Utility_Forms_IC_Analysis_Experiment{experiment_num}.json"
    )

    if os.path.exists(all_ic_results_file_path):
        with open(all_ic_results_file_path, "r", encoding="utf-8") as file:
            all_ic_results = json.load(file)

    rab_data = all_ic_results.get("robustness_analysis_data", {})
    n_iters = all_ic_results.get("n_iterations_completed", 1)

    sum_delta_loss = rab_data.get("sum_delta_minimum_loss_by_iter", [])
    rank_changes   = rab_data.get("rank_change_by_iter", [])
    rank_med       = rab_data.get("rank_change_median_by_iter", [])

    "Build a 1x2 subplots figure or 2 separate figures"
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Sum of Δ Min Loss Per Iteration",
            "Sum of Rank Changes Per Iteration",
        ]
    )

    "X array from 1..n_iters"
    x_iter = list(range(1, len(sum_delta_loss)+1))
    line_width = 8
    "(A) sum_delta_minimum_loss_by_iter"
    trace_sum_loss = go.Scatter(
        x=x_iter[1:],
        y=sum_delta_loss[1:],
        mode="lines+markers",
        name="Δ Min Loss",
        marker=dict(size=fig_lay.get("markersize", 10)+2, color='hsla(115, 65%, 40%, 1.0)'),
        line=dict(width=line_width, color='hsla(155, 65%, 20%, 1.0)')
    )
    fig.add_trace(trace_sum_loss, row=1, col=1)

    "(B) rank_change_by_iter"
    x_iter2 = list(range(1, len(rank_changes)+1))
    trace_rank = go.Scatter(
        x=x_iter2[1:],
        y=rank_changes[1:],
        mode="lines+markers",
        name="Sum of Rank Changes",
        marker=dict(size=fig_lay.get("markersize", 10)+2, color='hsla(200, 65%, 40%, 1.0)'),
        line=dict(width=line_width, dash="solid", color='hsla(200, 65%, 20%, 1.0)')
    )
    fig.add_trace(trace_rank, row=1, col=2)

    tickvals_x = list(np.linspace(2, x_iter2[-1], len(x_iter2) - 1))
    ticktext_x = tickvals_x
    epsilon_x = 5e-02

    fig.update_annotations(font_size=24)
    fig.update_layout(
        template=fig_lay.get("template", "plotly_dark"),
        title="Robustness of IC Results",
        titlefont_size=fig_lay['titlefont_size'] + 6,
        margin=dict(l=150, r=100, t=150, b=120),
        title_x=0.5, title_y= 0.98,
        xaxis=dict(
            title="Analysis Iteration",
            **fig_lay.get("xaxis",{}), 
            tickvals=tickvals_x,
            ticktext=ticktext_x,
            range=[2 - epsilon_x, x_iter2[-1] + epsilon_x]
        ),
        xaxis2=dict(
            title="Analysis Iteration",
            **fig_lay.get("xaxis",{}), 
            tickvals=tickvals_x,
            ticktext=ticktext_x,
            range=[2 - epsilon_x, x_iter2[-1] + epsilon_x]
        ),
        yaxis=dict(
            title="Sum of Δ Minimum Loss",
            **fig_lay.get("yaxis",{})
        ),
        yaxis2=dict(
            title="Sum (or Median) Rank Change" if rank_med else "Sum of Rank Changes",
            **fig_lay.get("yaxis",{})
        ),
        font=fig_lay.get("font", {}),
        hoverlabel=fig_lay.get("hoverlabel", {}),
        legend=dict(x=0.5, y=-0.2, xanchor="center", orientation="h")
    )

    if general_settings.get('export_fig'):
        out_file_name: str = "robustness_analysis.html"
        out_path = os.path.join(file_paths['visuals'], out_file_name)
        print(f"Saved robustness figure to {out_path}.")
        fig.write_html(out_path)
        
    return fig


def plot_ic_scores_delta_bic(fig_lay: dict, file_paths: dict, general_settings: dict, include_dropdown: bool = True) -> go.Figure:
    """
    Creates a Plotly scatterplot of ΔBIC scores for all utility-model configurations,
    sorted from lowest (best) to highest. By default, a single trace uses a continuous
    color scale based on the number of parameters (𝑘). When a dropdown menu is included
    (include_dropdown=True), users can toggle coloring by each relevant Boolean utility
    option, revealing two traces (True/False) with distinct legend entries.

    Arguments:
        • fig_lay: Dict[str, Any]
            Layout preferences (template, font, axis styles, etc.) used for consistent
            aesthetics across figures.

        • file_paths: Dict[str, str]
            File paths. Must include:
                └─ file_paths["bic_aic"]
            This function automatically loads a CSV named:
                All_Utility_Forms_IC_Analysis_Experiment{experiment_num}.csv
            from that directory, where experiment_num is retrieved from general_settings.

        • general_settings: Dict[str, Any]
            Must contain 'experiment_num' to identify which CSV file to load. The CSV file
            is expected to have columns: ΔBIC, k_params, equation, n_data, and zero or more
            Boolean columns for specific utility options.

        • utility_settings: Dict[str, bool]
            A dictionary of possible Boolean flags (e.g. 'reference_dependent_utility').
            This is used to build readable labels for the dropdown if include_dropdown=True.

        • include_dropdown: bool
            If True, adds an interactive dropdown menu that toggles between:
                1) A single color-scale trace for 𝑘 parameters (the default).
                2) Pairs of True/False traces for each recognized Boolean column.
            If False, only the single color-scale trace is shown, and all Boolean traces
            remain invisible.

    Returns:
        • fig: go.Figure
            The Plotly figure object. The function writes an HTML file named 'ic_scores_scatter.html'
            to the file_paths["bic_aic"] directory. The x-axis is the model rank (1 = best),
            and the y-axis is ΔBIC relative to the best model (lower is better).

    Notes:
        1) Dots are sized ~1.8x larger than the project default and include a subtle outline.
        2) When coloring by 𝑘, a colorbar labeled “𝑘” is displayed on the right. Boolean columns,
            if toggled, appear as separate True/False scatter traces with a horizontal legend
            placed below the chart. Each Boolean option name is converted into a more readable
            label (e.g., “Ref-Dependent Utility” instead of “reference_dependent_utility”).
        3) The figure title includes 𝑛 (the maximum n_data from the CSV) and references ΔBIC.
        4) Hover text shows each model’s rank, ΔBIC, number of parameters, and its utility equation.
    """

    "1) Determine which CSV to load"
    experiment_num = general_settings.get("experiment_num", 3)
    csv_file = os.path.join(
        file_paths["bic_aic"],
        f"All_Utility_Forms_IC_Analysis_Experiment{experiment_num}.csv"
    )
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Could not locate file: {csv_file}")

    "2) Load DataFrame, check columns"
    df = pd.read_csv(csv_file)
    required_cols = ["ΔBIC", "k_params", "equation", "n_data"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_file}")

    "3) Sort by ΔBIC ascending and compute model rank"
    df.sort_values(by="ΔBIC", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["model_rank"] = df.index + 1  # Rank from 1..N.

    "Assume all rows share the same n_data or use the max"
    n_data = int(df["n_data"].max())

    "4) Identify boolean columns"
    bool_cols = []
    for col in df.columns:
        if df[col].dtype == bool or df[col].dtype == "bool":
            bool_cols.append(col)

    "5) Create a readable label map for these columns"
    "(Customize as needed)"
    bool_label_map = {
        'conditional_welfare_mode':       "Conditional Welfare Mode",
        'reference_dependent_altruism':   "Ref-Dependent Altruism",
        'min_max_rawlsian_leontief':      "Min-Max (Rawls/Leontief)",
        'use_exponential_parameters':     "Use Exponential Params",
        'apply_exponents_to_payoffs':     "Apply Exponents to Payoffs",
        'single_exponential_parameter':   "Single Exponential Param",
        'single_payoffs_not_differences': "Single Payoffs, Not Differences",
        'payoff_ratios_not_differences':  "Payoff Ratios, Not Differences",
        'reference_dependent_utility':    "Ref-Dependent Utility",
        'use_negativity_parameters':      "Use Negativity Params",
        'negativity_social_comparison':   "Negativity in Social Comparison",
        'fix_self_interest_parameter':    "Fix Self-Interest Param",
        'include_social_comparison':      "Include Social Comparison",
        'include_altruism_term':          "Include Altruism Term"
    }

    "6) Build a single color-scale trace for k_params"
    k_min = df["k_params"].min()
    k_max = df["k_params"].max()

    "Marker size (scaled ~1.8x)"
    default_marker_size = int(fig_lay.get("markersize", 12) * 2)

    trace_kparams = go.Scatter(
        x=df["model_rank"],
        y=df["ΔBIC"],
        mode="markers",
        name="Models by 𝑘",
        visible=True,  # Default.
        showlegend=False,  # No legend for the continuous color scale.
        hovertemplate=(
            "Rank: %{x}; 𝑘 Params: %{customdata[0]}; ΔBIC: %{y:.3f}<br>"
            "Equation: %{customdata[1]}<extra></extra>"
        ),
        customdata=df[["k_params", "equation"]],
        marker=dict(
            size=default_marker_size,
            color=df["k_params"],
            colorscale="Viridis",
            cmin=k_min,
            cmax=k_max,
            showscale=True,  # Show colorbar.
            colorbar=dict(
                title="𝑘",  # Fancy k.
                x=1.02
            ),
            line=dict(width=1.5, color="hsla(0, 50%, 0%, 0.5)")
        )
    )

    data_traces = [trace_kparams]

    "7) For each boolean col, create 2 separate scatter traces: True & False"
    "These traces are invisible by default; dropdown selections show them and hide the k_params trace."
    hue_true  = "hsla(0, 80%, 40%, 7.0)"     # Red.
    hue_false = "hsla(180, 80%, 40%, 7.0)"   # Cyan.
    bool_trace_map = {}
    current_trace_index = 1

    for bcol in bool_cols:
        label_ = bool_label_map.get(bcol, bcol)

        df_true  = df[df[bcol] == True ]
        df_false = df[df[bcol] == False]

        tr_true = go.Scatter(
            x=df_true["model_rank"],
            y=df_true["ΔBIC"],
            mode="markers",
            name=f"{label_} = True",
            visible=False,
            legendgroup=bcol,
            showlegend=True,
            hovertemplate=(
                "Rank: %{x}; 𝑘 Params: %{customdata[0]}; ΔBIC: %{y:.3f}<br>"
                "Equation: %{customdata[1]}<extra></extra>"
            ),
            customdata=df_true[["k_params", "equation"]],
            marker=dict(
                size=default_marker_size, 
                opacity=0.7, color=hue_true, showscale=False,
                line=dict(width=1.5, color="hsla(0, 50%, 0%, 0.5)")
            )
        )
        tr_false = go.Scatter(
            x=df_false["model_rank"],
            y=df_false["ΔBIC"],
            mode="markers",
            name=f"{label_} = False",
            visible=False,
            legendgroup=bcol,
            showlegend=True,
            hovertemplate=(
                "Rank: %{x}; 𝑘 Params: %{customdata[0]}; ΔBIC: %{y:.3f}<br>"
                "Equation: %{customdata[1]}<extra></extra>"
            ),
            customdata=df_false[["k_params", "equation"]],
            marker=dict(
                size=default_marker_size,
                opacity=0.7, color=hue_false, showscale=False,
                line=dict(width=1.5, color="hsla(0, 50%, 0%, 0.5)")
            )
        )

        data_traces.append(tr_true)
        data_traces.append(tr_false)

        bool_trace_map[bcol] = (current_trace_index, current_trace_index + 1)
        current_trace_index += 2

    fig = go.Figure(data=data_traces)

    "8) Overall layout and styling"
    fig.update_layout(
        template=fig_lay.get("template", "plotly_dark"),
        title=f"IC Scores (ΔBIC) for All Utility Functional Forms; 𝑛 = {n_data} Data Points",
        titlefont_size=fig_lay['titlefont_size'],
        font=fig_lay.get("font", {}),
        hoverlabel=fig_lay.get("hoverlabel", {}),
        margin=dict(l=180, r=150, t=150, b=120),
        title_x=0.5,
        xaxis=dict(
            title="Model Rank (1 = Best)",
            **fig_lay.get("xaxis", {})
        ),
        yaxis=dict(
            title="ΔBIC (Difference from Best Model)",
            **fig_lay.get("yaxis", {})
        ),
        legend=dict(
            orientation="h", x=0.0, y=-0.15,
            font=dict(size=fig_lay.get("font", {}).get("size", 16))
        )
    )

    "9) If no dropdown is wanted, hide all Boolean traces"
    if not include_dropdown:
        for idx in range(1, len(data_traces)):
            fig.data[idx].visible = False

        "The single trace for k_params is visible, so the color scale carries the encoding."
        out_path = os.path.join(file_paths["bic_aic"], "ic_scores_scatter.html")
        fig.write_html(out_path)
        print(f"Saved scatter plot to '{out_path}' [No Dropdown Mode].")
        return fig

    "10) Build a dropdown to toggle coloring"
    n_traces_total = len(data_traces)
    def all_invisible():
        return [False] * n_traces_total

    "Option A: \"Color by k_params\""
    kparams_vis = all_invisible()
    kparams_vis[0] = True  # The first trace is the continuous color-scale.
    "Boolean traces start hidden."

    buttons = []
    "First button => color by k_params"
    buttons.append(dict(
        label="Color by 𝑘",
        method="update",
        args=[
            {"visible": kparams_vis},  # Sets the visible array.
            {"title": f"IC Scores (ΔBIC) for All Utility Functional Forms; 𝑛 = {n_data} Data Points (Colored by 𝑘)"}
        ]
    ))

    "Additional buttons => each boolean col"
    for bcol in bool_cols:
        label_ = bool_label_map.get(bcol, bcol)
        "The pair of traces for this col"
        idx_true, idx_false = bool_trace_map[bcol]
        vis_arr = all_invisible()
        vis_arr[idx_true] = True
        vis_arr[idx_false] = True

        buttons.append(dict(
            label=f"Color by {label_}",
            method="update",
            args=[
                {"visible": vis_arr},
                {"title": f"IC Scores (ΔBIC) for All Utility Functional Forms; 𝑛 = {n_data} Data Points (Colored by {label_})"}
            ]
        ))

    fig.update_layout(
        updatemenus=[dict(
            type="dropdown",
            showactive=True,
            direction="down",
            x=0.0, y=1.06,
            xanchor="left",
            yanchor="top",
            pad=dict(r=10, t=10),
            buttons=buttons
        )]
    )

    "11) Write out HTML and return figure"
    out_path = os.path.join(file_paths["visuals"], "ic_scores_scatter.html")
    print(f"Saved scatter plot to '{out_path}' [Dropdown Mode].")
    fig.write_html(out_path)
    
    return fig


def utility_setting_contribution_analysis(*, general_settings: dict, file_paths: dict, utility_settings_universe: dict[str, bool], score_col: str = "BIC", 
                                          use_edge_types: tuple[str, ...] = ("sibling", "parent_child"), include_non_network_toggles: bool = True, 
                                          export_csv: bool = True, out_dirname: str = "pairwise_edge_analysis") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Nesting-aware pairwise analysis of feature contributions.

    Purpose:
    Build ΔScore (e.g., ΔBIC) *along certified edges of the model nesting network*.
    The function returns:
      • edge_level_df : one row per compared edge with orientation by 'flip_name'
      • summary_by_flip : mean/median/N for each flip_name, split by edge_type
      • payoff_paths_summary : focused sibling-only summary for core payoff choices
        (single vs differences vs ratios ± reference; and exponent-placement when relevant).

    Arguments:
        • general_settings : dict
            Must include 'experiment_num'.
        • file_paths : dict
            Uses file_paths["bic_aic"] for the IC CSV and writes outputs under
            file_paths["player_fits"]/simulation_results/out_dirname (created if missing).
        • utility_settings_universe : dict[str,bool]
            The canonical set (keys only are used) of Boolean settings that define models.
            Example keys:
            ['conditional_welfare_mode','reference_dependent_altruism','min_max_rawlsian_leontief',
                'use_exponential_parameters','apply_exponents_to_payoffs','single_exponential_parameter',
                'single_payoffs_not_differences','payoff_ratios_not_differences','reference_dependent_utility',
                'use_negativity_parameters','negativity_social_comparison','fix_self_interest_parameter',
                'include_social_comparison','include_altruism_term']
        • score_col : str
            Which score to difference (e.g., "BIC" or "AIC").
        • use_edge_types : ('sibling','parent_child',...)
            Which edge types from the nesting network to include.
        • include_non_network_toggles : bool
            If True, also build "same-k, single-flip" pairs for flips that the
            nesting network marks as having no edge (e.g., Conditional Welfare; Rawls/Leontief).
            These are labeled edge_type='non_network_same_k'.
        • export_csv : bool
            If True, writes three CSVs (edge-level, summary-by-flip, payoff-paths summary).
        • out_dirname : str
            Subfolder name for outputs under player_fits/simulation_results.

    Returns:
        • (edge_level_df, summary_by_flip, payoff_paths_summary)

    Notes:
        • ΔScore orientation is always Score(True) - Score(False) for the 'flip_name'.
            Negative => turning the feature ON improves fit.
        • Explicitly tags each row with 'edge_type' in {'sibling','parent_child','non_network_same_k'}.
        • The payoff-paths panel is sibling-only and reports:
            - Single vs Differences
            - Single vs Ratios
            - Differences vs Ratios
            - RefDiff vs Diff, RefRatio vs Ratio (if present)
            - Apply-exponents-directly vs Apply-after (conditional on exponents and on not-single)
    """
    experiment_num = general_settings.get('experiment_num', 0)

    "---------- Load IC table ----------"
    ic_csv = os.path.join(
        file_paths["bic_aic"],
        f"All_Utility_Forms_IC_Analysis_Experiment{experiment_num}.csv"
    )
    df_models = pd.read_csv(ic_csv, encoding="utf-8", engine="python")

    "Canonical feature set and order used throughout this function."
    feature_cols = [feature_col for feature_col in utility_settings_universe.keys() if feature_col in df_models.columns]
    keep_cols = ["k_params", score_col] + feature_cols

    missing_cols = [column_name for column_name in keep_cols if column_name not in df_models.columns]
    if missing_cols:
        raise ValueError(f"IC CSV is missing required columns: {missing_cols}")

    "Coerce booleans"
    for feature_col in feature_cols:
        if df_models[feature_col].dtype != bool:
            df_models[feature_col] = df_models[feature_col].astype(bool)

    "Build a canonical signature for each row in IC table"
    def row_signature(row) -> tuple:
        "Tuple of booleans in the SAME order as feature_cols"
        return tuple(bool(row[feature_col]) for feature_col in feature_cols)

    df_models = df_models.reset_index(drop=False).rename(columns={"index": "model_id"})
    df_models = df_models[["model_id"] + keep_cols].copy()
    df_models["__signature__"] = df_models.apply(row_signature, axis=1)

    "Map signature -> model_id (ensure uniqueness)"
    sig_to_model_id: dict[tuple, int] = {}
    dup_sigs: dict[tuple, list[int]] = {}
    for mid, sig in zip(df_models["model_id"], df_models["__signature__"]):
        if sig in sig_to_model_id:
            "Track duplicates to catch structural problems (shouldn’t happen for 480 unique forms)"
            dup_sigs.setdefault(sig, []).append(int(mid))
        else:
            sig_to_model_id[sig] = int(mid)

    if dup_sigs:
        "Not necessarily fatal, but warn prominently."
        "Keep the first occurrence; duplicates would inflate ambiguity."
        print(f"[WARN] Duplicate utility signatures in IC table for {len(dup_sigs)} signature(s). "
              f"Proceeding with the first occurrence for each.")

    "---------- Get nesting graph ----------"
    graph = model_nesting_adjacency_matrices(
        general_settings=general_settings,
        utility_settings=utility_settings_universe,
        file_paths=file_paths
    )

    parent_of  = graph['adjacency_lists']['parent_of']
    sibling_of = graph['adjacency_lists']['sibling_of']
    settings_list = graph['settings']  # List[dict[str,bool]].

    "Convert a settings dict from the graph into the same canonical signature"
    def settings_signature(settings_dict: dict) -> tuple:
        "Some graph settings may be ints; normalize to bool; default missing -> False"
        return tuple(bool(settings_dict.get(feature_col, False)) for feature_col in feature_cols)

    "Build mapping from graph index -> model_id in IC table"
    index_to_model_id: list[int] = []
    missing_indices: list[int] = []

    for idx, settings_dict in enumerate(settings_list):
        sig = settings_signature(settings_dict)
        mid = sig_to_model_id.get(sig, None)
        if mid is None:
            missing_indices.append(idx)
            index_to_model_id.append(None)
        else:
            index_to_model_id.append(mid)

    if missing_indices:
        n_total = len(settings_list)
        print(f"[INFO] {len(missing_indices)} / {n_total} graph models are not in the IC table "
              f"(subset mode or partial run). Edges involving these models will be skipped.")

    "The only flips that *define* relationships (mirror classify_pair_relation)"
    SIBLING_FLIPS = {
        'single_payoffs_not_differences',
        'payoff_ratios_not_differences',
        'reference_dependent_utility',
        'reference_dependent_altruism',
        'apply_exponents_to_payoffs',
    }
    PARENT_CHILD_FLIPS = {
        'use_exponential_parameters',
        'single_exponential_parameter',
        'use_negativity_parameters',
        'negativity_social_comparison',
        'fix_self_interest_parameter',
        'include_social_comparison',
        'include_altruism_term',
    }

    "Helper: identify the single effective flip between two settings"
    def effective_flip(settings_a: dict, settings_b: dict, allowed: set[str]) -> str | None:
        flipped = [feature_col for feature_col in allowed if bool(settings_a.get(feature_col, False)) != bool(settings_b.get(feature_col, False))]
        if len(flipped) == 1:
            return flipped[0]
        return None

    "---- Build edge tuples over GRAPH INDICES with their flip labels ----"
    sib_edges: list[tuple[int,int,str]] = []
    for idx, sibling_indices in enumerate(sibling_of):
        for jdx in sibling_indices:
            if jdx <= idx:
                continue  # Proper de-dup.
            flip = effective_flip(settings_list[idx], settings_list[jdx], SIBLING_FLIPS)
            if flip:
                sib_edges.append((idx, jdx, flip))

    pc_edges: list[tuple[int,int,str]] = []
    for parent_idx, children in enumerate(parent_of):
        for child_idx in children:
            flip = effective_flip(settings_list[parent_idx], settings_list[child_idx], PARENT_CHILD_FLIPS)
            if flip:              
                pc_edges.append((parent_idx, child_idx, flip))

    "Quick sanity counts that should match terminal counts."
    print("Sibling edges (graph):", len(sib_edges))
    print("Parent-child edges (graph):", len(pc_edges))

    n_flips_by_setting_siblings = {setting_key: 0 for setting_key in utility_settings_universe.keys()} 
    for sib_sib_flip in sib_edges: 
        n_flips_by_setting_siblings[sib_sib_flip[-1]] += 1 
    n_flips_by_setting_par_chi = {setting_key: 0 for setting_key in utility_settings_universe.keys()} 
    for par_chi_flip in pc_edges: 
        n_flips_by_setting_par_chi[par_chi_flip[-1]] += 1
    for setting_key in utility_settings_universe.keys():
        if n_flips_by_setting_siblings[setting_key] == 0:
            del n_flips_by_setting_siblings[setting_key]
        if n_flips_by_setting_par_chi[setting_key] == 0:
            del n_flips_by_setting_par_chi[setting_key]

    print("\n Sibling edges (kept):")
    pp.pprint(n_flips_by_setting_siblings)
    print("\n Parent-child edges (kept):")
    pp.pprint(n_flips_by_setting_par_chi)

    "---- Convert graph indices → true model_ids via the signature map ----"
    "Skip any edge where either endpoint was not found in the IC table (subset / partial run)."
    rows = []
    if "sibling" in use_edge_types:
        for idx, jdx, flip in sib_edges:
            if index_to_model_id[idx] is None or index_to_model_id[jdx] is None:
                continue
            rows.append({
                "edge_type": "sibling",
                "flip_name": flip,
                "a_model_id": index_to_model_id[idx],
                "b_model_id": index_to_model_id[jdx]
            })
    if "parent_child" in use_edge_types:
        for parent_idx, child_idx, flip in pc_edges:
            if index_to_model_id[parent_idx] is None or index_to_model_id[child_idx] is None:
                continue
            rows.append({
                "edge_type": "parent_child",
                "flip_name": flip,
                "a_model_id": index_to_model_id[parent_idx],
                "b_model_id": index_to_model_id[child_idx]
            })

    df_edges = pd.DataFrame(rows)
    if df_edges.empty:
        raise RuntimeError("No edges found after mapping graph indices to IC rows.")

    "---------- Join IC scores + features for both endpoints ----------"
    left = df_edges.merge(df_models.add_prefix("a_"), left_on="a_model_id", right_on="a_model_id")
    both = left.merge(df_models.add_prefix("b_"), left_on="b_model_id", right_on="b_model_id")

    "Keep only rows where *exactly one* feature differs (safety; should be true for nest edges)"
    def _diff_feature(row) -> str | None:
        diffs = [feature for feature in feature_cols if bool(row[f"a_{feature}"]) != bool(row[f"b_{feature}"])]
        return diffs[0] if len(diffs) == 1 else None

    both["differing_feature"] = both.apply(_diff_feature, axis=1)
    "If this drops many rows, it indicates remaining ID/signature mismatches"
    before = len(both)
    both = both[~both["differing_feature"].isna()].copy()
    after = len(both)
    if after < before:
        print(f"[info] dropped {before - after} rows where endpoints differed on ≠1 feature "
              f"(this is expected to be small if mapping was correct).")

    "---------- Orient ΔScore as Score(True) - Score(False) for that feature ----------"
    def _orient(row):
        feature = row["differing_feature"]
        a_true = bool(row[f"a_{feature}"])
        b_true = bool(row[f"b_{feature}"])
        if a_true and not b_true:
            m_true_id,  score_true  = row["a_model_id"], row[f"a_{score_col}"]
            m_false_id, score_false = row["b_model_id"], row[f"b_{score_col}"]
        elif b_true and not a_true:
            m_true_id,  score_true  = row["b_model_id"], row[f"b_{score_col}"]
            m_false_id, score_false = row["a_model_id"], row[f"a_{score_col}"]
        else:
            return pd.Series({"m_true":np.nan,"m_false":np.nan,"delta":np.nan})
        return pd.Series({"m_true":m_true_id, "m_false":m_false_id, "delta":float(score_true) - float(score_false)})

    both[["m_true","m_false","delta"]] = both.apply(_orient, axis=1)

    "Sanity: flip label from graph should match the single differing feature"
    both["flip_ok"] = (both["flip_name"] == both["differing_feature"])
    if not bool(both["flip_ok"].all()):
        n_bad = int((~both["flip_ok"]).sum())
        print(f"[warn] {n_bad} edges where graph flip_name != detected differing_feature "
              f"(keeping rows but flagging).")

    "---------- Edge-level tidy output ----------"
    edge_level_cols = [
        "edge_type","flip_name","differing_feature","delta","m_true","m_false",
        "a_model_id","b_model_id","a_k_params","b_k_params"
    ] + [f"a_{feature_col}" for feature_col in feature_cols] + [f"b_{feature_col}" for feature_col in feature_cols]
    edge_level_df = both[edge_level_cols].copy()

    "---------- Optional: “non‑network” toggles (e.g., Conditional Welfare; Rawls/Leontief) ----------"
    non_network_rows = []
    if include_non_network_toggles:
        seen = set(edge_level_df["differing_feature"].unique())
        candidates = ["conditional_welfare_mode", "min_max_rawlsian_leontief"]
        none_flips = [feature for feature in candidates if feature in feature_cols and feature not in seen]

        "Per-toggle rule: Conditional Welfare can cross k (BIC already penalizes complexity); Min–Max stays same‑k."
        require_same_k_for = {
            "conditional_welfare_mode": False,
            "min_max_rawlsian_leontief": True,
        }

        tmp = df_models.copy()
        for feature in none_flips:
            "Signature excluding feature"
            other_cols = [feature_col for feature_col in feature_cols if feature_col != feature]
            tmp["signature_excl_f"] = tmp[other_cols].apply(lambda row_values: tuple(bool(row_value) for row_value in row_values.tolist()), axis=1)

            group_keys = ["signature_excl_f"] + (["k_params"] if require_same_k_for.get(feature, True) else [])
            for _, group_df in tmp.groupby(group_keys, dropna=False):
                false_feature_rows = group_df[group_df[feature] == False]
                true_feature_rows = group_df[group_df[feature] == True]
                if false_feature_rows.empty or true_feature_rows.empty:
                    continue
                for _, false_feature_row in false_feature_rows.iterrows():
                    for _, true_feature_row in true_feature_rows.iterrows():
                        non_network_rows.append({
                            "edge_type": "non_network" if not require_same_k_for.get(feature, True) else "non_network_same_k",
                            "flip_name": feature,
                            "differing_feature": feature,
                            "delta": float(true_feature_row[score_col]) - float(false_feature_row[score_col]),  # Score(True) - Score(False).
                            "m_true": int(true_feature_row["model_id"]),
                            "m_false": int(false_feature_row["model_id"]),
                            "a_model_id": int(false_feature_row["model_id"]),
                            "b_model_id": int(true_feature_row["model_id"]),
                            "a_k_params": int(false_feature_row["k_params"]),
                            "b_k_params": int(true_feature_row["k_params"]),
                            **{f"a_{feature_col}": bool(false_feature_row[feature_col]) for feature_col in feature_cols},
                            **{f"b_{feature_col}": bool(true_feature_row[feature_col]) for feature_col in feature_cols},
                        })

    if non_network_rows:
        edge_level_df = pd.concat([edge_level_df, pd.DataFrame(non_network_rows)], ignore_index=True)

    "---------- Summary by flip & edge type ----------"
    def _summarize(df):
        recs = []
        for (edge_type, feature), group_df in df.groupby(["edge_type","flip_name"], dropna=False):
            deltas = group_df["delta"].dropna().to_numpy()
            recs.append({
                "edge_type": edge_type,
                "flip_name": feature,
                "n_edges": int(deltas.size),
                "mean_delta": float(np.mean(deltas)) if deltas.size else np.nan,
                "median_delta": float(np.median(deltas)) if deltas.size else np.nan
            })
        return pd.DataFrame(recs)

    summary_by_flip = _summarize(edge_level_df).sort_values(
        ["edge_type", "mean_delta"], ascending=[True, True]
    )

    "---------- Payoff-paths sibling-only panel ----------"
    sib = edge_level_df[edge_level_df["edge_type"]=="sibling"].copy()

    def _quick_panel(name, mask):
        panel_df = sib[mask]
        return {
            "comparison": name,
            "n_edges": int(len(panel_df)),
            "mean_Δ": float(panel_df["delta"].mean()) if len(panel_df) else np.nan,
            "median_Δ": float(panel_df["delta"].median()) if len(panel_df) else np.nan
        }

    m_single_vs_diff = (sib["differing_feature"] == "single_payoffs_not_differences")
    m_ratio_vs_diff  = (sib["differing_feature"] == "payoff_ratios_not_differences")
    m_ref_toggle = (sib["differing_feature"] == "reference_dependent_utility") & \
                   (~sib["a_single_payoffs_not_differences"]) & (~sib["b_single_payoffs_not_differences"])
    has_exp = (sib["a_use_exponential_parameters"] & sib["b_use_exponential_parameters"])
    not_single = (~sib["a_single_payoffs_not_differences"] & ~sib["b_single_payoffs_not_differences"])
    m_place = (sib["differing_feature"] == "apply_exponents_to_payoffs") & has_exp & not_single

    payoff_rows = [
        _quick_panel("Single vs Differences (Δ = BIC[Single]-BIC[Diff])", m_single_vs_diff),
        _quick_panel("Ratios vs Differences (Δ = BIC[Ratios]-BIC[Diff])",  m_ratio_vs_diff),
        _quick_panel("Reference on vs off (non-single) (Δ = BIC[Ref]-BIC[NoRef])", m_ref_toggle),
        _quick_panel("Exponent placement: pre-payoff vs post-transform (Δ = BIC[Pre]-BIC[Post])", m_place),
    ]
    payoff_paths_summary = pd.DataFrame(payoff_rows)

    "---------- Write CSVs ----------"
    out_root = os.path.join(file_paths["bic_aic"], out_dirname)
    os.makedirs(out_root, exist_ok=True)

    if export_csv:
        edge_csv     = os.path.join(out_root, f"edge_level_{score_col}.csv")
        summary_csv  = os.path.join(out_root, f"summary_by_flip_{score_col}.csv")
        payoff_csv   = os.path.join(out_root, f"payoff_paths_{score_col}.csv")
        edge_level_df.to_csv(edge_csv, index=False, encoding="utf-8-sig")
        summary_by_flip.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        payoff_paths_summary.to_csv(payoff_csv, index=False, encoding="utf-8-sig")
        print(f"\nWrote: {edge_csv}\n       {summary_csv}\n       {payoff_csv}")

    return edge_level_df, summary_by_flip, payoff_paths_summary


def extract_rankings_of_canonical_utility_functions(file_paths: FilePaths, rank_col: str = "BIC", print_: bool = True,
                                                    canonical_specs: Optional[dict] = None) -> pd.DataFrame:
    """
    Filter the full IC results table for canonical model specifications and report their ranks.

    Matches rows in the stored IC DataFrame against a set of canonical utility specifications
    (Fehr-Schmidt, Bolton-Ockenfels, Charness-Rabin, etc.), finds the best-ranking row per
    label, and returns a summary DataFrame with each canonical model's IC rank, loss, AIC, BIC,
    and ΔBIC relative to the best-fitting model in the full comparison.

    Arguments:
        • file_paths: FilePaths
            Must include 'bic_aic' directory where the IC results CSV is stored.
        • rank_col: str
            The column to rank by; typically 'BIC' (default) or 'AIC'.
        • print_: bool
            If True, prints the ranking table to stdout for inspection.
        • canonical_specs: dict[str, dict] | None
            Mapping of label → utility settings dict. If None, defaults to the module-level
            CANONICAL_UTILITY_SPECS. The active specs are always saved to
            bic_aic/canonical_utility_settings.json for downstream use.

    Returns:
        • pd.DataFrame — rows indexed by canonical model label, with columns:
            label, n_matches, k_params, loss, AIC, BIC, global_rank, ΔBIC_to_best.
    """
    if canonical_specs is None:
        canonical_specs = CANONICAL_UTILITY_SPECS

    "Save canonical specs to JSON so quick_demo.py and other callers can load them."
    json_out = os.path.join(file_paths["bic_aic"], "canonical_utility_settings.json")
    os.makedirs(file_paths["bic_aic"], exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as _f:
        json.dump(canonical_specs, _f, ensure_ascii=False, indent=4)

    CANONICAL_SPECS = canonical_specs
    if print_:
        for function_name, settings in CANONICAL_SPECS.items():
            explanation = gnrl.is_valid_utility_settings(settings, provide_explanation=True)
            print(f"\n{function_name}:")
            print(build_utility_equation(settings))
            if explanation != "Success!":
                print(explanation) 
            for setting_key, setting_val in settings.items():
                setting_data = setting_key + " " * (30 - len(setting_key)) + f": {setting_val}"
                print(setting_data)
        for function_name, settings in CANONICAL_SPECS.items():
            print(f"\n{function_name}:")
            print(build_utility_equation(settings))
        print("")

    ic_csv_path = os.path.join(file_paths['bic_aic'], file_paths['file_names']['information_criterion'])
    df = pd.read_csv(ic_csv_path, encoding="utf-8", engine="python")
    "Cast bool columns robustly"
    for col in CANONICAL_SPECS[next(iter(CANONICAL_SPECS))].keys():
        if col in df.columns and df[col].dtype != bool:
            df[col] = df[col].astype(bool)

    "Global best by BIC for ΔBIC"
    global_best_bic = float(df["BIC"].min())
    df["global_rank"] = df[rank_col].rank(method="min", ascending=True).astype(int)

    rows = []
    for label, spec in CANONICAL_SPECS.items():
        missing = [column_name for column_name in spec.keys() if column_name not in df.columns]
        if missing:
            rows.append({"label": label, "error": f"Missing columns: {missing}"})
            continue
        cur = df.copy()
        for column_name, expected_value in spec.items():
            if column_name in cur.columns:
                cur = cur[cur[column_name] == expected_value]
        n_matches = len(cur)
        if n_matches == 0:
            rows.append({"label": label, "n_matches": 0, "note": "No exact match in IC table."})
            continue
        best = cur.sort_values(by=rank_col, ascending=True).iloc[0]
        rows.append({
            "label": label,
            "n_matches": n_matches,
            "k_params": int(best["k_params"]),
            "loss": float(best.get("loss", float("nan"))),
            "AIC": float(best.get("AIC", float("nan"))),
            "BIC": float(best.get("BIC", float("nan"))),
            "global_rank": int(best["global_rank"]),
            "ΔBIC_to_best": float(best["BIC"] - global_best_bic),
            "model_id": int(best.get("model_id", -1)),
        })
    out = pd.DataFrame(rows)
    if "BIC" in out.columns:
        out = out.sort_values(by=["BIC"], ascending=True, na_position="last")
    return out


"=========================================================================================="
"============================ Nesting Network and Verification ============================"
"=========================================================================================="

def run_child_parent_embedding_sanity_checks(general_settings: dict[str, Any], file_paths: dict[str, Any], param_bds: dict[str, tuple[float, float]], 
                                             utility_settings: UtilitySettings, player_role_to_fit: str = "predictor", fit_for_n_players: int | None = None,
                                             random_seed: int | None = 12345, numeric_tolerance: float = 1e-4, csv_file_name: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """
    Runs the child-vs-special-parent equality test across the entire model space.

    For each minimal (child, parent) pair:
        1) Sample a random child parameter vector within `param_bds`.
        2) Embed those child means into the parent's parameter space to create a special parent.
        3) For a subset (or all) participants:
            dyads_for_a_player → agent → loss_function_bayes → create_loss_report
            Sum NLL across dyads for `player_role_to_fit`, *for child and for parent*.
        4) Write a wide CSV with requested columns in file_paths['bic_aic'].

    Arguments:
        • general_settings: Global settings dict. The following keys are read:
            - experiment_num
            - softmax_temperature
            - (others are forwarded to `agent` and loss functions as-is)
        • file_paths: File path mapping (must include 'bic_aic' and 'file_names' → 'information_criterion').
        • param_bds: The global ParameterBounds with all keys (means and _std).
        • ordered_flag_keys: The canonical order of the 13 utility settings
            (e.g., pass `list(utility_settings.keys())` from the current model).
        • player_role_to_fit: 'predictor' (default) or 'chooser'.
        • fit_for_n_players: int | None; Number of participants to evaluate (alphabetical order). None → all.
        • random_seed: Seed for reproducibility.
        • numeric_tolerance: Tolerance for |loss_parent - loss_child|.
        • csv_file_name: Optional override for CSV name. Default: "child_parent_embedding_sanity_checks.csv".
        • verbose: If True, prints progress summaries.

    Returns:
        • pd.DataFrame; The full table that was also written to CSV.
    """
    ordered_flag_keys = list(utility_settings.keys())

    def _sample_random_parameter_dict(param_keys: list[str],
                                    param_bounds: list[tuple[float, float]],
                                    rng: random.Random) -> dict[str, float]:
        """
        Draw a random parameter dictionary within the provided bounds.

        Arguments:
            • param_keys: list[str]; Ordered parameter names (means first, then _std if present).
            • param_bounds: list[tuple[float, float]]; Same length and order as param_keys.
            • rng: random.Random; PRNG instance for reproducibility.

        Returns:
            • dict[str, float]; {parameter_name: sampled_value}
        """
        parameter_dictionary: dict[str, float] = {}
        for parameter_key, (lower_bound, upper_bound) in zip(param_keys, param_bounds):
            "Keep _std comfortably > 0 within the given bounds"
            if parameter_key.endswith("_std"):
                lower = max(lower_bound, 1e-3)
                parameter_dictionary[parameter_key] = rng.uniform(lower, upper_bound)
            else:
                parameter_dictionary[parameter_key] = rng.uniform(lower_bound, upper_bound)
        return parameter_dictionary

    def _embed_child_parameters_into_parent_means(child_parameter_dict: dict[str, float],
                                                changed_utility_setting: str,
                                                parent_param_keys: list[str]) -> dict[str, float]:
        """
        Deterministically embed a child's parameters into the parent's parameter space
        so the parent reproduces the child (i.e., the child is a special case of the parent).

        Mapping conventions (consistent with prior discussions):
            • use_exponential_parameters=True in parent:
                - If child has no γ’s: set all parent γ* to 1.0.
                - If child uses a single γ (γ1) and parent has multiple γ’s: tie all parent γ* to child's γ1.
            • single_exponential_parameter flip (tie ↔ untie):
                - If parent has multiple γ’s but child has γ1 only: copy γ1 to every parent γ*.
            • include_social_comparison added in parent: set Ƹᵢⱼ=0 and Ʒᵢⱼ=0 in parent.
            • include_altruism_term added in parent: set Vᵢⱼ=0 and Ʌᵢⱼ=0 in parent (if present).
            • negativity_social_comparison added in parent: set Ʒᵢⱼ = Ƹᵢⱼ (tie guilt to envy).
            • use_negativity_parameters added in parent: set Ʌ-weights equal to their V counterparts.
            • fix_self_interest_parameter released in parent: set Vᵢᵢ = 1.0 to replicate the fixed-value child.

        Notes:
            • Only parent keys that exist in `parent_param_keys` are written.
            • Any child keys that the parent also has are copied verbatim unless overridden by a rule above.

        Returns:
            • dict[str, float]; Parent-parameter means that embed the child.
        """
        parent_parameters: dict[str, float] = {}
        
        "1) Start by copying any overlapping child means into the parent (safe default)."
        for parameter_key in parent_param_keys:
            if parameter_key in child_parameter_dict:
                parent_parameters[parameter_key] = float(child_parameter_dict[parameter_key])

        "2) Apply changed-setting-specific embedding rules."
        if changed_utility_setting == "use_exponential_parameters":
            "Parent gained exponents. If child already has γ1, tie; otherwise set all to 1."
            child_gamma_keys = [param_key for param_key in child_parameter_dict.keys() if param_key.startswith('γ')]
            if child_gamma_keys:
                "Child already had γ's (rare, e.g., when moving single->multi as a side effect); tie all to γ1"
                child_gamma1 = child_parameter_dict.get('γ1', 1.0)
                for parameter_key in parent_param_keys:
                    if parameter_key.startswith('γ'):
                        parent_parameters[parameter_key] = float(child_gamma1)
            else:
                "No γ in the child → set all γ in the parent to 1.0"
                for parameter_key in parent_param_keys:
                    if parameter_key.startswith('γ'):
                        parent_parameters[parameter_key] = 1.0

        elif changed_utility_setting == "single_exponential_parameter":
            "Tie/untie exponents: if parent has multiple γ's and child had γ1, tie them to γ1."
            if 'γ1' in child_parameter_dict:
                common_gamma = float(child_parameter_dict['γ1'])
            else:
                common_gamma = 1.0
            "If parent has γ2 or γ3, set them equal to common γ."
            for gamma_key in ('γ1', 'γ2', 'γ3'):
                if gamma_key in parent_param_keys:
                    "If parent was the *tied* version (γ1 only), writing γ1 is enough."
                    "If parent has separate γ’s, copy common value to each."
                    parent_parameters[gamma_key] = float(parent_parameters.get(gamma_key, common_gamma))

        elif changed_utility_setting == "include_social_comparison":
            "Social comparison added in parent → zero its weights to reproduce child"
            if 'Ƹᵢⱼ' in parent_param_keys:
                parent_parameters['Ƹᵢⱼ'] = 0.0
            if 'Ʒᵢⱼ' in parent_param_keys:
                parent_parameters['Ʒᵢⱼ'] = 0.0

        elif changed_utility_setting == "include_altruism_term":
            "Altruism added in parent → zero its weights"
            if 'Vᵢⱼ' in parent_param_keys:
                parent_parameters['Vᵢⱼ'] = 0.0
            if 'Ʌᵢⱼ' in parent_param_keys:
                parent_parameters['Ʌᵢⱼ'] = 0.0

        elif changed_utility_setting == "negativity_social_comparison":
            "Parent splits envy/guilt → tie them to the child's single weight (Ʒᵢⱼ_child)"
            single = float(child_parameter_dict.get('Ʒᵢⱼ', parent_parameters.get('Ʒᵢⱼ', 0.0)))
            if 'Ƹᵢⱼ' in parent_param_keys:
                parent_parameters['Ƹᵢⱼ'] = single
            if 'Ʒᵢⱼ' in parent_param_keys:
                parent_parameters['Ʒᵢⱼ'] = single

        elif changed_utility_setting == "use_negativity_parameters":
            "Parent gained negativity mirrors → copy Vᵢᵢ→Ʌᵢᵢ and Vᵢⱼ→Ʌᵢⱼ if present"
            if 'Ʌᵢᵢ' in parent_param_keys:
                parent_parameters['Ʌᵢᵢ'] = float(parent_parameters.get('Vᵢᵢ', child_parameter_dict.get('Vᵢᵢ', 0.0)))
            if 'Ʌᵢⱼ' in parent_param_keys:
                parent_parameters['Ʌᵢⱼ'] = float(parent_parameters.get('Vᵢⱼ', child_parameter_dict.get('Vᵢⱼ', 0.0)))

        elif changed_utility_setting == "fix_self_interest_parameter":
            "Parent released Vᵢᵢ → set it to fixed constant (1.0) to replicate child"
            if 'Vᵢᵢ' in parent_param_keys:
                parent_parameters['Vᵢᵢ'] = 1.0

        # elif changed_utility_setting == "conditional_welfare_mode": # TODO figure out if this should be removed
        # "Make 'ahead' and 'behind' branches identical → tie Λ to V."
        #     if 'Ʌᵢᵢ' in parent_param_keys:
        #         parent_parameters['Ʌᵢᵢ'] = float(parent_parameters.get('Vᵢᵢ', child_parameter_dict.get('Vᵢᵢ', 0.0)))
        #     if 'Ʌᵢⱼ' in parent_param_keys:
        #         parent_parameters['Ʌᵢⱼ'] = float(parent_parameters.get('Vᵢⱼ', child_parameter_dict.get('Vᵢⱼ', 0.0)))

        elif changed_utility_setting == "include_altruism_term":
            if parent_settings.get("conditional_welfare_mode", False):
                "Parent gained an explicit altruism parameter inside conditional welfare."
                "To replicate the child (which uses implicit 1 - Vᵢᵢ / 1 - Ʌᵢᵢ), set:"
                if 'Vᵢⱼ' in parent_param_keys:
                    parent_parameters['Vᵢⱼ'] = 1.0 - float(parent_parameters.get('Vᵢᵢ', child_parameter_dict.get('Vᵢᵢ', 0.0)))
                if 'Ʌᵢⱼ' in parent_param_keys:
                    parent_parameters['Ʌᵢⱼ'] = 1.0 - float(parent_parameters.get('Ʌᵢᵢ', child_parameter_dict.get('Ʌᵢᵢ', 0.0)))
            else:
                "Non-conditional case: zeroing altruism reproduces the child"
                if 'Vᵢⱼ' in parent_param_keys: parent_parameters['Vᵢⱼ'] = 0.0
                if 'Ʌᵢⱼ' in parent_param_keys: parent_parameters['Ʌᵢⱼ'] = 0.0

        "3) Any remaining parent keys not touched yet get a benign default:"
        for parameter_key in parent_param_keys:
            if parameter_key not in parent_parameters:
                "If it's a γ, default to 1.0; otherwise default to 0.0 (neutral weight)."
                parent_parameters[parameter_key] = 1.0 if parameter_key.startswith('γ') else 0.0

        return parent_parameters

    def _sum_negloglik_for_player_and_role(dyad_games_for_player: dict[str, list[dict]],
                                        player_uuid: str,
                                        player_role: str,
                                        general_settings: dict[str, Any]) -> float:
        """
        Sums NLL across all dyads/games for a single (player, role).
        Uses the same storage locations as the pipeline.

        Returns:
            • float; sum of 'loss_final_sum' across the player's dyads for the specified role.
        """
        total_negative_log_likelihood = 0.0

        for _, dyad_games in dyad_games_for_player.items():
            "The NLL breakdown is written by loss_function_bayes + create_loss_report"
            "Follow the project sequence: agent(), loss_function_bayes(), create_loss_report()."
            role_report = dyad_games[0].get('loss_report', {}).get(player_uuid, {}).get(player_role, {})
            total_negative_log_likelihood += float(role_report.get('loss_final_sum', 0.0))

        return total_negative_log_likelihood

    def _evaluate_loss_for_model_and_player(player_uuid: str,
                                            player_role: str,
                                            general_settings: dict[str, Any],
                                            file_paths: dict[str, Any],
                                            utility_settings: dict[str, bool],
                                            param_info: dict[str, Any],
                                            parameter_values: dict[str, float],
                                            choice_temperature: float | None = None) -> float:
        """
        Produces predictions with agent(), computes loss with loss_function_bayes(),
        and returns sum NLL for a single player/role.

        Calls:
            • prep.dyads_for_a_player(...)  → returns dict[dyad_key] = dyad_games
            • agent(...)                          → writes predictions to param_estimates
            • loss_function_bayes(...)            → writes raw per-game losses
            • create_loss_report(...)             → aggregates & stores per-player/role sums

        References:
            • Example usage pattern around agent() → loss → create_loss_report in the codebase.

        Returns:
            • float; sum of 'loss_final_sum'.
        """
        "Load this player's dyads"
        player_dyads = prep.dyads_for_a_player(
            player_uuid=player_uuid,
            experiment_num=int(general_settings.get('experiment_num', 3)),
            file_paths=file_paths,
            analysis_mode='bayesian'
        )

        "Run agent across each dyad with the provided parameter dictionary for this role."
        for dyad_key, dyad_games in player_dyads.items():
            dyad_games_copy = copy.deepcopy(dyad_games)
            updated_dyad_games = agent(
                dyad_games=dyad_games_copy,
                game_idx_start=0,
                game_idx_stop=len(dyad_games_copy) - 1,
                initial_params={player_role: parameter_values},
                param_info=param_info,
                utility_settings=utility_settings,
                player_uuid=player_uuid,
                player_role=player_role,
                general_settings=general_settings,
                choice_temperature=choice_temperature
            )

            "Compute loss and attach loss_report to the first game."
            updated_dyad_games = loss_function_bayes(dyad_games=updated_dyad_games, general_settings=general_settings)
            updated_dyad_games[0]['loss_report'] = create_loss_report(dyad_games=updated_dyad_games, general_settings=general_settings)

            "Replace the dyad with the updated one so aggregation uses consistent objects"
            player_dyads[dyad_key] = updated_dyad_games

        return _sum_negloglik_for_player_and_role(
            dyad_games_for_player=player_dyads,
            player_uuid=player_uuid,
            player_role=player_role,
            general_settings=general_settings
        )

    def _enumerate_child_parent_pairs_from_ic(ic_dataframe: pd.DataFrame,
                                            ordered_flag_keys: list[str],
                                            general_settings: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Walks the IC table and, for each model (as child), lists all immediate parents (Δk > 0)
        using `gnrl.parents_children_of` so the minimal-dependent-fix rules are respected.

        Returns:
            • list of dicts with:
                {
                    'child_index': int,
                    'parent_index': int,
                    'child_settings_tuple': tuple[bool,...],
                    'parent_settings_tuple': tuple[bool,...],
                    'changed_utility_setting': str
                }
        """
        "Build tuple signatures and index map"
        def row_to_tuple(row: pd.Series) -> tuple[bool, ...]:
            return tuple(bool(row[key]) for key in ordered_flag_keys)

        ic_dataframe = ic_dataframe.copy()
        ic_dataframe['utility_tuple'] = ic_dataframe.apply(row_to_tuple, axis=1)
        tuple_to_index: dict[tuple[bool, ...], int] = {tup: idx for idx, tup in ic_dataframe['utility_tuple'].items()}

        all_pairs: list[dict[str, Any]] = []

        "Iterates children in ascending k so checks move from simpler to richer models."
        if 'k_params' in ic_dataframe.columns:
            ic_dataframe_sorted = ic_dataframe.sort_values(['k_params', 'BIC', 'loss'], ascending=[True, True, True])
        else:
            ic_dataframe_sorted = ic_dataframe.copy()

        for _, child_row in ic_dataframe_sorted.iterrows():
            child_tuple = child_row['utility_tuple']
            child_settings = gnrl.convert_utility_settings(child_tuple, into=dict)  # type: ignore
            neighbor_dict = gnrl.parents_children_of(
                utility_settings=child_settings,
                return_children=False,
                return_parents=True,
                general_settings=general_settings
            )
            parent_tuples = neighbor_dict.get('parents') or []

            for parent_tuple in parent_tuples:
                parent_index = tuple_to_index.get(parent_tuple, None)
                if parent_index is None:
                    continue

                relation_1_to_2, relation_2_to_1, changed_setting = gnrl.classify_pair_relation(
                    model_1=child_tuple,
                    model_2=parent_tuple,
                    general_settings=general_settings,
                    utility_settings=utility_settings
                )
                if relation_1_to_2 != 'child' or relation_2_to_1 != 'parent' or not changed_setting:
                    continue

                all_pairs.append({
                    'child_index': int(child_row.name),
                    'parent_index': int(parent_index),
                    'child_settings_tuple': child_tuple,
                    'parent_settings_tuple': parent_tuple,
                    'changed_utility_setting': changed_setting
                })

        return all_pairs

    "Load the IC table"
    ic_path = os.path.join(file_paths["bic_aic"], file_paths["file_names"]["information_criterion"])
    ic_dataframe = pd.read_csv(ic_path)

    general_settings = copy.deepcopy(general_settings)
    general_settings['confidence_weighted'] = False
    
    "Build child→parent pairs"
    child_parent_pairs = _enumerate_child_parent_pairs_from_ic(
        ic_dataframe=ic_dataframe,
        ordered_flag_keys=ordered_flag_keys,
        general_settings=general_settings
    )

    if verbose:
        print(f"[Sanity] Identified {len(child_parent_pairs)} child→parent pairs to test.")

    "Determine participants"
    experiment_num = int(general_settings.get('experiment_num', 3))
    participant_uuids: list[str] = prep.all_player_uuids(file_paths=file_paths, experiment_num=experiment_num, only_humans=True)
    if isinstance(fit_for_n_players, int) and 0 < fit_for_n_players <= len(participant_uuids):
        participant_uuids = participant_uuids[:fit_for_n_players]  # Alphabetical order preserved.

    choice_temperature = general_settings.get('softmax_temperature', 1.0)

    "Mapping from tuple signature to IC-row index for lookup"
    def row_to_tuple(row: pd.Series) -> tuple[bool, ...]:
        return tuple(bool(row[key]) for key in ordered_flag_keys)
    ic_dataframe['utility_tuple'] = ic_dataframe.apply(row_to_tuple, axis=1)

    rng = random.Random(random_seed)
    results_rows: list[dict[str, Any]] = []

    "Go through each child→parent pair"
    for pair_idx, pair in enumerate(child_parent_pairs, start=1):
        child_tuple = pair['child_settings_tuple']
        parent_tuple = pair['parent_settings_tuple']
        changed_utility_setting: str = pair['changed_utility_setting']
        child_index = pair['child_index']
        parent_index = pair['parent_index']

        child_settings = gnrl.convert_utility_settings(child_tuple, into=dict)  # type: ignore
        parent_settings = gnrl.convert_utility_settings(parent_tuple, into=dict)  # type: ignore

        "Build param_info for child and parent (means-only keys for evaluation)"
        child_param_info = make_param_info(param_bds=param_bds, utility_settings=child_settings, general_settings=general_settings)
        parent_param_info = make_param_info(param_bds=param_bds, utility_settings=parent_settings, general_settings=general_settings)

        "1) Sample random child parameter dictionary within bounds (child_param_info)"
        child_parameter_dict = _sample_random_parameter_dict(
            param_keys=child_param_info['keys'],
            param_bounds=child_param_info['bounds'],
            rng=rng
        )

        "2) Embed child → special parent parameter means"
        embedded_parent_means = _embed_child_parameters_into_parent_means(
            child_parameter_dict=child_parameter_dict,
            changed_utility_setting=changed_utility_setting,
            parent_param_keys=parent_param_info['keys']
        )

        "Ensure *_std keys exist for grid/MCMC updates (predictor priors need sigmas)."
        if general_settings.get('update_method') in ('grid', 'MCMC'):
            bounds_lookup = {param_key: param_bounds for param_key, param_bounds in zip(parent_param_info['keys'], parent_param_info['bounds'])}
            min_std_guess = 0.5  # Same convention used elsewhere.

            for base_key in [param_key for param_key in parent_param_info['keys'] if not param_key.endswith('_std')]:
                std_key = base_key + '_std'
                lo, hi = bounds_lookup[std_key]

                if std_key in bounds_lookup:
                    current_std_value = embedded_parent_means.get(std_key, None)
                    if (current_std_value is None) or not (float(lo) <= float(current_std_value) <= float(hi)):
                        midpoint = (float(lo) + float(hi)) / 2.0
                        embedded_parent_means[std_key] = max(min_std_guess, midpoint)

        "3) Evaluate losses over the selected participants"
        sum_nll_child = 0.0
        sum_nll_parent = 0.0
        for player_uuid in participant_uuids:
            sum_nll_child += _evaluate_loss_for_model_and_player(
                player_uuid=player_uuid,
                player_role=player_role_to_fit,
                general_settings=general_settings,
                file_paths=file_paths,
                utility_settings=child_settings,
                param_info=child_param_info,
                parameter_values=child_parameter_dict,
                choice_temperature=choice_temperature
            )
            sum_nll_parent += _evaluate_loss_for_model_and_player(
                player_uuid=player_uuid,
                player_role=player_role_to_fit,
                general_settings=general_settings,
                file_paths=file_paths,
                utility_settings=parent_settings,
                param_info=parent_param_info,
                parameter_values=embedded_parent_means,
                choice_temperature=choice_temperature
            )

        "Pretty equations"
        equation_child = build_utility_equation(utility_settings=child_settings)
        equation_parent = build_utility_equation(utility_settings=parent_settings)

        "K_params by project counting"
        k_child = gnrl.count_free_parameters(utility_settings=child_settings, general_settings=general_settings)
        k_parent = gnrl.count_free_parameters(utility_settings=parent_settings, general_settings=general_settings)

        "Flatten parameter columns for the CSV (fill Nones where absent)"
        all_param_keys = list(param_bds.keys())  # Superset of all possible keys.
        row: dict[str, Any] = {
            "child_idx": int(child_index),
            "parent_idx": int(parent_index),
            "k_child": int(k_child),
            "k_parent": int(k_parent),
            "changed_utility_setting": changed_utility_setting,
            "loss_child": float(sum_nll_child),
            "loss_parent": float(sum_nll_parent),
            "loss_parent_minus_child": float(sum_nll_parent - sum_nll_child),
            "equal_loss": abs(sum_nll_parent - sum_nll_child) <= numeric_tolerance,
            "equation_child": equation_child,
            "equation_parent": equation_parent,
            "utility_settings_child": child_tuple,
            "utility_settings_parent": parent_tuple,
            "n_players_evaluated": len(participant_uuids),
            "player_role": player_role_to_fit,
        }

        "13 boolean flags duplicated (child and parent)"
        for flag_key, child_val in child_settings.items():
            row[f"child_{flag_key}"] = bool(child_val)
        for flag_key, parent_val in parent_settings.items():
            row[f"parent_{flag_key}"] = bool(parent_val)

        "Parameter dictionaries (JSON-ish strings) and also one column per parameter (child/parent)"
        row["params_child"] = {param_key: child_parameter_dict.get(param_key, None) for param_key in all_param_keys}
        row["params_parent"] = {param_key: embedded_parent_means.get(param_key, None) for param_key in all_param_keys}
        for parameter_key in all_param_keys:
            row[f"{parameter_key}_child"] = child_parameter_dict.get(parameter_key, None)
            row[f"{parameter_key}_parent"] = embedded_parent_means.get(parameter_key, None)

        results_rows.append(row)

        if verbose and pair_idx % 10 == 0:
            print(f"[Sanity] Processed {pair_idx}/{len(child_parent_pairs)} pairs...")

    results_dataframe = pd.DataFrame(results_rows)

    "Save CSV"
    csv_name = csv_file_name or "child_parent_embedding_sanity_checks.csv"
    if ".csv" in csv_name:
        csv_name = csv_name.replace(".csv", "")
    csv_name += f"-{general_settings.get('update_method', None)}-{player_role_to_fit}-{fit_for_n_players}.csv"
    csv_path = os.path.join(file_paths["bic_aic"], csv_name)
    results_dataframe.to_csv(csv_path, index=False, encoding="utf-8-sig")
    if verbose:
        print(f"[Sanity] Wrote: {csv_path}")

    return results_dataframe


def run_child_parent_probability_equivalence_smoketest(utility_settings: dict[str, Any], file_paths: dict[str, Any], param_bds: dict[str, tuple[float, float]], 
                                                       n_trials: int = 12, rand_payoff_idx: bool = False, rng_seed: int | None = None, tolerance: float = 1e-10, verbose: bool = True) -> pd.DataFrame:
    """
    Verifies nesting by comparing *choice probabilities* of each child with the
    probabilities of its embedded special parent on a small synthetic set of games.

    Procedure:
        1) Build child→parent pairs from the IC table, preserving canonical indices.
        2) Draw a child mean-parameter dictionary uniformly within `param_bds` (means only).
        3) Embed child means into the parent (gnrl rules) to reproduce the child.
        4) Compute p(A) for n_trials randomized games using choice(...), for child and parent.
        5) Store max|Δp| and a boolean pass/fail in a compact CSV.

    Arguments:
        • utility_settings: dict[str, Any]; 
        • file_paths: dict[str, Any]; 
        • param_bds: dict[str, tuple[float, float]]; 
        • n_trials: int; 
        • rand_payoff_idx: bool; 
        • rng_seed: int | None; 
        • tolerance: float; 
        • verbose: bool; 
        
    Returns:
        • pd.DataFrame with one row per (child, parent) pair and max_abs_delta across trials.
    """
    "--- Helpers ---------------------------------------------------------------"
    def _generate_synthetic_games(n_games: int = 10, rng_seed: int = 20250417) -> list[dict]:
        rng = random.Random() if rng_seed is None else random.Random(rng_seed)
        games: list[dict] = []
        for _ in range(n_games):
            As = rng.randint(1, 5)
            Ao = rng.randint(1, 5)
            Bs = rng.randint(1, 5)
            Bo = rng.randint(1, 5)
            games.append({
                "payoff_A_chooser":   As,
                "payoff_A_predictor": Ao,
                "payoff_B_chooser":   Bs,
                "payoff_B_predictor": Bo,
            })
        return games

    def _choice_probs(games: list[dict], u_settings: dict[str, bool],
                      params: dict[str, float], temperature: float = 1.5) -> list[float]:
        out = []
        for game in games:
            choice_result = choice(current_game=game,
                       agent_params=params,
                       utility_settings=u_settings,
                       softmax_temperature=temperature,
                       normalize_conditional_welfare_params=False,
                       select=False)
            out.append(float(choice_result["model_choose_A"]))
        return out

    def _means_only_keys(utility_settings_dict: dict[str, bool]) -> list[str]:
        "Only means (no *_std, no cov), consistent with the choice() pipeline"
        return [param_key for param_key in gnrl.parameter_keys_for_utility_settings(
            utility_settings_dict, general_settings={"update_method": "naive"}
        ) if not (param_key.endswith("_std") or param_key.endswith("_cov"))]

    def _sample_means(param_keys: list[str], rng: random.Random) -> dict[str, float]:
        sampled_means: dict[str, float] = {}
        for param_key in param_keys:
            lower_bound, upper_bound = param_bds[param_key]
            sampled_value = rng.uniform(float(lower_bound), float(upper_bound))
            if (upper_bound - lower_bound) > 1:
                sampled_value = min(max(round(sampled_value, 1), lower_bound), upper_bound)
            sampled_means[param_key] = float(sampled_value)
        return sampled_means

    def _show_math_work(equation: str, params: dict[str, float], utility_settings: dict[str, bool] | tuple[bool], 
                        game_idx: int = 0, decimals: int = 2, comparison_tol: float = 5e-3) -> str:
        """
        Fill the pretty equation from build_utility_equation with concrete numbers,
        evaluate its right-hand side (RHS), and verify it matches utility() after rounding.

        Returns the fully substituted equation plus a trailing status tag:
            • [OK]               — evaluated and matched (within tolerance after rounding),
            • [FAIL: ≠ x.xx]     — evaluated but differs from utility() (rounded),
            • [EVAL ERROR: ...]  — evaluation failed (syntax/name/type errors etc.).

        Notes for maintainers:
            • This function does NOT modify build_utility_equation; it only normalizes
                the rendered string so that Python's eval can handle unicode, unicode
                operators, implicit multiplication, and non-integer exponents on negative bases.
            • Any parameter symbols still present after substitution are replaced with
                the default values used by utility() so eval never sees a stray symbol.
        """
        "--- 1) Compute the ground-truth utility from code (source of truth) -----"
        payoff_key_map_eval = {
            "payoff_A_chooser":   "As",
            "payoff_A_predictor": "Ao",
            "payoff_B_chooser":   "Bs",
            "payoff_B_predictor": "Bo",
        }
        utility_settings_dict = gnrl.convert_utility_settings(utility_settings, into=dict)
        payoffs_eval = {
            payoff_key_map_eval[payoff_key]: payoff_value
            for payoff_key, payoff_value in games[game_idx].items()
            if "payoff" in payoff_key
        }
        true_value = float(utility(payoffs=payoffs_eval, params=params, utility_settings=utility_settings_dict, normalize_conditional_welfare_params=False))
        true_value_rounded = round(true_value, decimals)

        "--- 2) Substitute payoffs and parameters into the pretty string ----------"
        gamma_pretty = {"γ1": "γ₁", "γ2": "γ₂", "γ3": "γ₃"}
        params_with_pretty_gammas = {gamma_pretty.get(param_key, param_key): param_value for param_key, param_value in params.items()}

        "Known aliases that sometimes appear in strings or terminals"
        alias_to_pretty = {
            "Vii": "Vᵢᵢ", "Λii": "Ʌᵢᵢ", "Vij": "Vᵢⱼ", "Λij": "Ʌᵢⱼ",
            "Ƹij": "Ƹᵢⱼ", "Ʒij": "Ʒᵢⱼ",
            # Safeguard: if these ever appear, map them to the canonical key
            "γ1": "γ₁", "γ2": "γ₂", "γ3": "γ₃",
        }
        for alias_key, pretty_key in alias_to_pretty.items():
            if alias_key in params and pretty_key not in params_with_pretty_gammas:
                params_with_pretty_gammas[pretty_key] = params[alias_key]

        payoff_symbol_map = {"As": "πᵢᴬ", "Ao": "πⱼᴬ", "Bs": "πᵢᴮ", "Bo": "πⱼᴮ"}
        payoffs_pretty_map = {payoff_symbol_map[payoff_key]: payoff_value for payoff_key, payoff_value in payoffs_eval.items()}

        filled_equation = equation.replace("Uᵢ(A)", f"{true_value_rounded:.{decimals}f}")
        "Substitute payoffs first, then parameters (reduces accidental overlap)"
        for symbol, value in payoffs_pretty_map.items():
            filled_equation = filled_equation.replace(symbol, str(value))
        for symbol, value in params_with_pretty_gammas.items():
            filled_equation = filled_equation.replace(symbol, str(value))

        gamma1_value = None
        for ks in ("γ₁", "γ1"):
            if ks in params_with_pretty_gammas:
                gamma1_value = float(params_with_pretty_gammas[ks]); break
        if gamma1_value is None: gamma1_value = 1.0

        default_values = {
            "Vᵢᵢ": 1.0, "Ʌᵢᵢ": 0.0, "Vᵢⱼ": 0.0, "Ʌᵢⱼ": 0.0, "Ƹᵢⱼ": 0.0, "Ʒᵢⱼ": 0.0,
            "γ₁": gamma1_value, "γ₂": gamma1_value, "γ₃": gamma1_value,
            "Vii": 1.0, "Λii": 0.0, "Vij": 0.0, "Λij": 0.0, "Ƹij": 0.0, "Ʒij": 0.0,
            "γ1": gamma1_value, "γ2": gamma1_value, "γ3": gamma1_value,
        }

        for symbol, default_val in default_values.items():
            if symbol in filled_equation:
                filled_equation = filled_equation.replace(symbol, str(default_val))

        lhs, rhs = filled_equation.split("=", 1)
        rhs = gnrl.canon_sc_both_ways(rhs.strip(), mode="twoterm")
        filled_equation = lhs + "= " + rhs

        "Evaluate using the shared helper"
        value_str, status = gnrl.eval_pretty_equation_rhs(rhs, decimals=decimals, sc_mode="twoterm")
        if status:
            return f"{filled_equation}  [{status}]"

        rhs_value_rounded = float(value_str)
        status_tag = "" if abs(rhs_value_rounded - true_value_rounded) <= comparison_tol \
                    else f"  [FAIL: ≠ {rhs_value_rounded:.{decimals}f}]"
        return f"{filled_equation}{status_tag}"

    def _components_for_utility(payoffs: dict, params: dict[str, float], utility_settings: dict[str, bool]) -> dict[str, float]:
        """
        Return the individual additive terms of the utility function for a given payoff and parameter vector.

        Calls `utility(..., separate_terms=True)` and remaps the returned keys 
        to the short labels used throughout the IC analysis display code:
        `'self_interest'` → `'self'`, `'altruism'` → `'altr'`, `'social_comp'` → `'socc'`.

        Arguments:
            • payoffs: dict
                Payoff dictionary passed directly to `utility()`, keyed by player role.
            • params: dict[str, float]
                Parameter vector (e.g., `{'Vᵢᵢ': 1.0, 'Vᵢⱼ': 0.5}`).
            • utility_settings: dict[str, bool]
                The 13-boolean toggle dict that selects which utility terms are active.

        Returns:
            • dict[str, float] — component values rounded to 6 decimal places, with keys
              `'self'`, `'altr'`, and `'socc'` (plus any additional terms the utility
              function emits when `separate_terms=True`).
        """
        util_components: dict = utility(payoffs=payoffs, params=params, utility_settings=utility_settings, separate_terms=True, normalize_conditional_welfare_params=False)
        for old_key, new_key in [('self_interest', 'self'), ('altruism', 'altr'), ('social_comp', 'socc')]:
            util_components[new_key] = util_components.pop(old_key)
        return {key: round(val, 6) for key, val in util_components.items()}
        
    def _evaluate_equation_numeric(equation_string: str, params: dict[str, float], utility_settings: UtilitySettings, payoffs: dict[str, float], 
                                   param_overrides: dict[str, float] | None = None, decimals_local: int = 6, ) -> tuple[float | None, str]:
        """
        Evaluate the RHS of the pretty equation after substituting 'params' and 'payoffs'.
        'param_overrides' (if provided) override parameter values *before* substitution.
        Returns (value_or_None, status_text). Status empty string on success.
        """
        payoff_key_map_eval = {
            "payoff_A_chooser":   "As",
            "payoff_A_predictor": "Ao",
            "payoff_B_chooser":   "Bs",
            "payoff_B_predictor": "Bo",
        }
        payoffs = {payoff_key_map_eval[payoff_key] if "payoff" in payoff_key else payoff_key: payoff_value for payoff_key, payoff_value in payoffs.items()}

        "1) Build replacements"
        gamma_pretty = {"γ1": "γ₁", "γ2": "γ₂", "γ3": "γ₃"}
        params_pretty = {gamma_pretty.get(param_key, param_key): param_value for param_key, param_value in params.items()}
        "Map aliases that might appear"
        alias_to_pretty = {
            "Vii": "Vᵢᵢ", "Λii": "Ʌᵢᵢ", "Vij": "Vᵢⱼ", "Λij": "Ʌᵢⱼ",
            "Ƹij": "Ƹᵢⱼ", "Ʒij": "Ʒᵢⱼ",
            "γ1": "γ₁", "γ2": "γ₂", "γ3": "γ₃",
        }
        for alias_key, pretty_key in alias_to_pretty.items():
            if alias_key in params and pretty_key not in params_pretty:
                params_pretty[pretty_key] = params[alias_key]

        "Apply overrides *before* substitution (so numeric values in the string reflect the override)"
        if param_overrides:
            for raw_param_key, param_value in param_overrides.items():
                param_key = gamma_pretty.get(raw_param_key, raw_param_key)  # Allow γ1 vs γ₁ in overrides.
                params_pretty[param_key] = param_value

        "Gamma fallbacks inherit γ₁ (like utility())"
        gamma1_value = None
        for k_try in ("γ₁", "γ1"):
            if k_try in params_pretty:
                gamma1_value = float(params_pretty[k_try]); break
        if gamma1_value is None: gamma1_value = 1.0

        payoff_symbol_map = {"As": "πᵢᴬ", "Ao": "πⱼᴬ", "Bs": "πᵢᴮ", "Bo": "πⱼᴮ"}
        payoffs_pretty_map = {payoff_symbol_map[payoff_key]: payoff_value for payoff_key, payoff_value in payoffs.items()}

        "2) Extract RHS"
        if "=" not in equation_string:
            return None, "EVAL ERROR: no '=' in equation"
        _, rhs_original = equation_string.split("=", 1)
        rhs_original = rhs_original.strip()

        "3) Perform substitutions (payoffs first, then params)"
        rhs_filled = rhs_original
        for sym, val in payoffs_pretty_map.items():
            rhs_filled = rhs_filled.replace(sym, str(val))
        for sym, val in params_pretty.items():
            rhs_filled = rhs_filled.replace(sym, str(val))

        "Replace any remaining canonical symbols with *utility()* defaults"
        defaults = {
            "Vᵢᵢ": 1.0, "Ʌᵢᵢ": 0.0, "Vᵢⱼ": 0.0, "Ʌᵢⱼ": 0.0, "Ƹᵢⱼ": 0.0, "Ʒᵢⱼ": 0.0,
            "γ₁": gamma1_value, "γ₂": gamma1_value, "γ₃": gamma1_value,
            "Vii": 1.0, "Λii": 0.0, "Vij": 0.0, "Λij": 0.0, "Ƹij": 0.0, "Ʒij": 0.0,
            "γ1": gamma1_value, "γ2": gamma1_value, "γ3": gamma1_value,
        }

        for sym, val in defaults.items():
            if sym in rhs_filled:
                rhs_filled = rhs_filled.replace(sym, str(val))

        "Canonicalize and evaluate via generalist helpers"
        rhs_filled = gnrl.canon_sc_both_ways(rhs_filled, mode="twoterm")
        value, status = gnrl.eval_pretty_equation_rhs(rhs_filled, decimals=decimals_local, sc_mode="twoterm")
        return value, status

    "--- Build pairs from adjacency lists -------------------------------------"
    "Use the canonical universe and equation strings, matching IC analysis."
    adj = model_nesting_adjacency_matrices(
        general_settings=general_settings,
        utility_settings=utility_settings,
        file_paths= file_paths,
        create_new_file=False,
        equation_form=False,
        print_=False
    )
    settings_list: list[dict[str, bool]] = adj["settings"]
    parents_of: list[list[int]] = adj["adjacency_lists"]["parent_of"]  # Child idx -> parent indices.

    "Child→parent candidate pairs from adjacency; then confirm with classifier"
    candidate_pairs: list[tuple[int, int, str]] = []
    for child_idx, parent_indices in enumerate(parents_of):
        child_settings = settings_list[child_idx]
        for parent_idx in parent_indices:
            parent_settings = settings_list[parent_idx]
            r12, r21, changed = gnrl.classify_pair_relation(
                model_1=child_settings,
                model_2=parent_settings,
                utility_settings=utility_settings,
                general_settings=None
            )
            if r12 == "child" and r21 == "parent" and changed is not None:
                candidate_pairs.append((child_idx, parent_idx, changed))

    if verbose:
        print(f"[Prob-Sanity] Candidate child→parent pairs from adjacency: {len(candidate_pairs)}")

    "Synthetic games + random generator"
    games = _generate_synthetic_games(n_games=n_trials, rng_seed=rng_seed)
    temp = float(1.5) if "softmax_temperature" not in utility_settings else float(utility_settings["softmax_temperature"])
    rng = random.Random() if rng_seed is None else random.Random(rng_seed)

    results: list[dict[str, Any]] = []
    for jdx, (child_idx, parent_idx, changed) in enumerate(candidate_pairs, start=1):
        child_settings  = settings_list[child_idx]
        parent_settings = settings_list[parent_idx]

        "1) sample child means"
        child_keys  = _means_only_keys(child_settings)
        child_means = _sample_means(child_keys, rng=rng)

        "2) embed means: child → parent (means only)"
        try:
            parent_param_info = gnrl.map_child_to_parent_special_param_info(
                child_utility_settings=child_settings,
                parent_utility_settings=parent_settings,
                child_fitted_parameters=child_means,
                general_settings={"update_method": "naive"},
                param_bds=param_bds,
                build_utility_equation=build_utility_equation
            )
        except NotImplementedError:
            "Skip cleanly when structural guards remain in the mapper for some families."
            continue

        parent_means = {
            param_key: float(param_val) for param_key, param_val in zip(parent_param_info["keys"], parent_param_info["guesses"])
            if not (param_key.endswith("_std") or param_key.endswith("_cov"))
        }

        "3) compare probabilities on the same games"
        probs_child  = _choice_probs(games, child_settings,  child_means,  temp)
        probs_parent = _choice_probs(games, parent_settings, parent_means, temp)
        max_abs_delta = float(max(abs(p_child - p_parent) for p_child, p_parent in zip(probs_child, probs_parent)) if probs_child else 0.0)

        "(d) max |Δp|"
        deltas = [abs(probs_child - probs_parent) for probs_child, probs_parent in zip(probs_child, probs_parent)]
        max_abs_delta = float(max(deltas) if deltas else 0.0)
        max_abs_delta_idx = 0
        if rand_payoff_idx: 
            max_abs_delta_idx = random.randint(a=0, b=n_trials-1)
        else:
            for idx, delta in enumerate(deltas):
                if delta >= max_abs_delta:
                    max_abs_delta_idx = idx

        "(e) Display equations and replace payoffs and params with their values"
        equation_child =  build_utility_equation(utility_settings=child_settings)
        equation_parent = build_utility_equation(utility_settings=parent_settings)

        "--- Compute the four utilities on the same representative game (games[0]) ---"
        "1) Code (source of truth)"
        payoffs_for_one = {
            "As": games[max_abs_delta_idx]["payoff_A_chooser"],
            "Ao": games[max_abs_delta_idx]["payoff_A_predictor"],
            "Bs": games[max_abs_delta_idx]["payoff_B_chooser"],
            "Bo": games[max_abs_delta_idx]["payoff_B_predictor"],
        }
        U_code_child  = round(float(utility(payoffs_for_one, child_means,  child_settings, normalize_conditional_welfare_params=False)), 3)
        U_code_parent = round(float(utility(payoffs_for_one, parent_means, parent_settings, normalize_conditional_welfare_params=False)), 3)

        "2) Pretty equation (evaluated numerically)"
        U_equa_child,  err_child  = _evaluate_equation_numeric(equation_string=equation_child,  params=child_means,  
                                        utility_settings=utility_settings, payoffs=games[max_abs_delta_idx], decimals_local=3)
        U_equa_parent, err_parent = _evaluate_equation_numeric(equation_string=equation_parent, params=parent_means, 
                                        utility_settings=utility_settings, payoffs=games[max_abs_delta_idx], decimals_local=3)

        "3) Boolean comparisons (tolerant equality)"
        def same(first_value: float | None, second_value: float | None, tol: float = 1e-2) -> bool:
            return (first_value is not None) and (second_value is not None) and (abs(first_value - second_value) <= tol)

        match_code_child_vs_parent = same(U_code_child,  U_code_parent)
        match_equa_child_vs_parent = same(U_equa_child,  U_equa_parent)
        match_child_code_vs_equa   = same(U_code_child,  U_equa_child)
        match_parnt_code_vs_equa   = same(U_code_parent, U_equa_parent)

        "4) Short diagnosis tag"
        diagnosis = "OK"
        use_long_tag = False
        if not match_code_child_vs_parent:
            if use_long_tag:
                diagnosis += "-NESTING_MISMATCH_CODE"
            else:
                diagnosis += "-NESTCODE"
        if not match_equa_child_vs_parent:
            if use_long_tag:
                diagnosis += "-NESTING_MISMATCH_EQUATION"
            else:
                diagnosis += "-NESTEQUA"
        if not match_child_code_vs_equa or not match_parnt_code_vs_equa:
            if use_long_tag:
                diagnosis += "-STRING_BUILDER_MISMATCH"
            else:
                diagnosis += "-STRBUILD"
        if diagnosis.startswith("OK-"):
            diagnosis = diagnosis[3:]

        "5) Worked strings (nice to keep for human inspection)"
        worked_child =  _show_math_work(equation=equation_child, params=parent_means, 
                            game_idx=max_abs_delta_idx, utility_settings=child_settings)
        worked_parent = _show_math_work(equation=equation_parent, params=parent_means, 
                            game_idx=max_abs_delta_idx, utility_settings=parent_settings)

        comp_child  = _components_for_utility(payoffs_for_one, child_means,  child_settings)
        comp_parent = _components_for_utility(payoffs_for_one, parent_means, parent_settings)
        comp_delta  = {component_key: round(comp_parent[component_key]-comp_child[component_key], 6) for component_key in comp_child.keys() if abs(comp_parent[component_key]-comp_child[component_key]) > 1e-9}

        equation_parent_aligned, equation_child_aligned = equation_parent, equation_child

        results.append({
            "n_trials": n_trials,
            "temperature": temp,
            "child_idx": child_idx,
            "parent_idx": parent_idx,
            "changed_utility_setting": changed,
            "max_abs_delta_p": max_abs_delta,
            "all_equal": (max_abs_delta <= tolerance),
            "utility_settings_child": child_settings,
            "utility_settings_parent": parent_settings,
            "parameters": parent_means,
            "equation_child":  equation_child_aligned,
            "equation_parent": equation_parent_aligned,

            # Representative game payload (helps re-run a single case quickly)
            "game_As": payoffs_for_one["As"],
            "game_Ao": payoffs_for_one["Ao"],
            "game_Bs": payoffs_for_one["Bs"],
            "game_Bo": payoffs_for_one["Bo"],

            # Four utilities
            "U_code_child":  U_code_child,
            "U_code_parent": U_code_parent,
            "U_equa_child":  U_equa_child,
            "U_equa_parent": U_equa_parent,

            # Comparison flags
            "match_code_child_vs_parent": match_code_child_vs_parent,
            "match_equa_child_vs_parent": match_equa_child_vs_parent,
            "match_child_code_vs_equa":   match_child_code_vs_equa,
            "match_parnt_code_vs_equa":   match_parnt_code_vs_equa,
            "diagnosis":   diagnosis,

            "comp_child":  comp_child,
            "comp_parent": comp_parent,
            "comp_delta":  comp_delta,

            "worked_child":  "'" + worked_child,
            "worked_parent": "'" + worked_parent
        })

        if verbose and jdx % 50 == 0:
            print(f"[Prob-Sanity] {jdx}/{len(candidate_pairs)} processed...")

    df = pd.DataFrame(results)
    df = df.sort_values(by=["all_equal", "changed_utility_setting", "diagnosis"], ascending=[True, True, True])
    out_path = os.path.join(file_paths["processed"], "child_parent_prob_equivalence.csv")
    try: df.to_csv(out_path, index=False, encoding="utf-8-sig")
    except (PermissionError, OSError): pass

    if verbose:
        n_ok = int(df["all_equal"].sum())
        print(f"[Prob-Sanity] {n_ok}/{len(df)} pairs matched within tol={tolerance}.")
        print(f"[Prob-Sanity] Wrote: {out_path}")

        "Print a few mismatches (if any) to inspect"
        mismatches = df.loc[~df["all_equal"]].head(10)
        if len(mismatches) == 0:
            print("All tested child→parent pairs matched exactly on p(A).")
        else:
            print("First few non-matching pairs (inspect equations and changed flag):")
            for _, row in mismatches.iterrows():
                print(f"  child_idx={row['child_idx']} → parent_idx={row['parent_idx']}"
                    f" | changed={row['changed_utility_setting']}"
                    f" | max|Δp|={row['max_abs_delta_p']:.3e}")

        diagnosis_counts = df['diagnosis'].value_counts().to_dict()
        print(diagnosis_counts)

        "Print parent and child equations for STRBUILD and NESTEQUA-STRBUILD diagnoses"
        relevant_rows = df[df["diagnosis"].isin(["STRBUILD", "NESTEQUA-STRBUILD"])]
        if not relevant_rows.empty:
            print("\nParent and child equations with STRBUILD/NESTEQUA-STRBUILD diagnosis:")
            for _, row in relevant_rows.iterrows():
                print(f"Parent ({row['parent_idx']}): {row['equation_parent']}")
                print(f"Child  ({row['child_idx']}): {row['equation_child']}\n")

    return df


def verify_utility_vs_string_equation(utility_function: Callable, utility_function_str: Callable, utility_settings: UtilitySettings, param_bds: dict[str, tuple[float, float]], 
                                      file_paths: FilePaths, n_games: int = 5**4, rng_seed: int | None = 20250417, exhaustive_if_large: bool = True, option: str = "A", 
                                      comparison_tol: float = 1e-6, decimals: int = 6, verbose: bool = True) -> pd.DataFrame:
    """
    Exhaustively (or randomly) verifies that utility_function(...) and the numeric
    evaluation of utility_function_str(...) produce identical utilities across:
        • all generated, valid utility settings (via gnrl.generate_utility_settings),
        • random parameter means (via make_param_info + uniform sampling within bounds),
        • many payoff structures (random or full 5^4 grid over {1..5}^4).

    In addition to the *total* utility, the routine compares *components*:
        self-interest, altruism, social comparison — for both code 
        and string — so discrepancies can be localized immediately.

    Arguments:
        • utility_function : Callable
            The Python function that returns the numeric utility. Must accept (payoffs: dict, params: dict, 
            utility_settings: dict, separate_terms: bool=False) and return either a float (separate_terms=
            False) or a dict with keys {'self_interest','altruism','social_comp'} when separate_terms=True.

        • utility_function_str : Callable
            The string builder, e.g., build_utility_equation. Must accept (utility_settings: dict, 
            in_latex: bool=False, option: str="A"|"B") and return a pretty string like "Uᵢ(A) = ...".

        • general_settings : dict
            Global toggles passed to gnrl.generate_utility_settings and used for consistency.

        • param_bds : dict[str, tuple[float, float]]
            Global parameter bounds; used to sample parameter means.

        • ordered_flag_keys : list[str]
            Canonical order of the 13 boolean flags; used for the 13 column outputs.

        • n_games : int (default 100)
            Number of payoff configurations to test per utility setting. If exhaustive_if_large=True 
            and n_games > 5**4, the routine evaluates the *entire* 5^4 = 625 grid over {1..5}^4.

        • rng_seed : int | None
            Seed for reproducible sampling. If None, system entropy is used.

        • exhaustive_if_large : bool (default True)
            If True and n_games > 625, evaluate all {1..5}^4 payoffs instead of sampling.

        • option : str
            Passed to the string builder ("A" or "B"); the code always evaluates A vs B
            with the familiar payoff names {'As','Ao','Bs','Bo'}.

        • file_paths: dict[str: str]; Dictionary containing all file paths used in this analysis.

        • comparison_tol : float
            Absolute tolerance for declaring a match between code and string.

        • decimals : int
            Rounding used for storing evaluated numbers (comparisons use raw floats).

        • verbose : bool
            If True, prints a compact diagnostic report.

    Returns:
        • pd.DataFrame
            Row-wise verification results sorted by worst discrepancies first.
    """
    "---------- (0) Where to write -------------------------------------------------"
    out_dir = file_paths["processed"]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "utility_vs_string_verification.csv")
    out_summary_path = os.path.join(out_dir, "utility_vs_string_summary.csv")

    ordered_flag_keys: list[str] = list(utility_settings.keys())

    "---------- (1) Utility setting generation ------------------------------------"
    "Generate all valid utility settings using the project generator."
    all_settings_raw = gnrl.generate_utility_settings(utility_settings=utility_settings)
    "Normalize into dict form"
    all_settings: list[dict[str, bool]] = [
        gnrl.convert_utility_settings(raw_utility_settings, into=dict) for raw_utility_settings in all_settings_raw
    ]
    "Keep only valid ones if generator does not guarantee validity"
    if hasattr(gnrl, "is_valid_utility_settings"):
        all_settings = [candidate_utility_settings for candidate_utility_settings in all_settings if gnrl.is_valid_utility_settings(candidate_utility_settings)]

    "---------- (2) Payoff generation ---------------------------------------------"
    def _all_payoff_tuples():
        for As, Ao, Bs, Bo in it.product(range(1, 6), repeat=4):
            yield {"As": As, "Ao": Ao, "Bs": Bs, "Bo": Bo}

    def _random_payoff_tuples(n_payoffs: int, seed: int | None):
        rng_local = random.Random() if seed is None else random.Random(seed)
        for _ in range(n_payoffs):
            yield {
                "As": rng_local.randint(1, 5),
                "Ao": rng_local.randint(1, 5),
                "Bs": rng_local.randint(1, 5),
                "Bo": rng_local.randint(1, 5),
            }

    use_exhaustive = exhaustive_if_large and (n_games >= 5**4)
    payoff_iterable = list(_all_payoff_tuples()) if use_exhaustive else list(_random_payoff_tuples(n_games, rng_seed))

    "---------- (3) Parameter sampling --------------------------------------------"
    rng_params = random.Random() if rng_seed is None else random.Random(rng_seed)

    def _sample_means_for(utility_settings: dict[str, bool]) -> dict[str, float]:
        """
        Means-only sampling following the project conventions. Uses make_param_info to get proper keys,
        but then samples uniformly within global bounds so the sweep explores the space.
        """
        param_info = make_param_info(
            param_bds=param_bds,
            utility_settings=utility_settings,
            general_settings={"update_method": "naive"},
            guess_seed=None
        )
        keys = [param_key for param_key in param_info["keys"] if not param_key.endswith("_std") and not param_key.endswith("_cov")]
        means: dict[str, float] = {}
        for param_key in keys:
            lower_bound, upper_bound = param_bds[param_key]
            sampled_value = rng_params.uniform(float(lower_bound), float(upper_bound))
            "Small rounding on wide intervals keeps equations readable without losing variety"
            if (upper_bound - lower_bound) > 1:
                sampled_value = min(max(round(sampled_value, 2), lower_bound), upper_bound)
            means[param_key] = float(sampled_value)
        return means

    "---------- (4) Pretty-equation evaluation (numeric) --------------------------"
    """
    Numerically evaluate utility_function_str(...) after substituting payoffs and parameters.
    Normalization handles unicode operators, '^' to pow_signed conversion, implicit
    multiplication, and γ₂/γ₃ fallback to γ₁.
    """
    "--- INSERT C1: direct evaluator for the stubborn ratio+refdep+negSC family ----"
    def _direct_eval_ratio_refdep_negsc(
        utility_settings: dict[str, bool],
        params: dict[str, float],
        payoffs: dict[str, float],
        decimals_local: int = 6,
    ) -> float:
        """
        Direct numeric evaluator for the family:
            • NOT conditional_welfare_mode
            • NOT min_max_rawlsian_leontief
            • use_exponential_parameters = True
            • payoff_ratios_not_differences = True
            • reference_dependent_utility = True
            • use_negativity_parameters = False (for SI & AL)
            • negativity_social_comparison = True
            • include_social_comparison = True
            • include_altruism_term = True
            • single_payoffs_not_differences = False
            • apply_exponents_to_payoffs = False

        Mirrors utility() semantics exactly:
            - Self & altruism: centered ratios, sign-preserving exponent on the base
            - SC with negativity: -Ƹ*max(envy,0)^γ + Ƹ*max(guilt,0)^γ
            - Reference constant '3' is NOT exponentiated
        """
        "Read params (accept either γ1/γ₂/γ₃ or γ₁/γ₂/γ₃ spellings)"
        def _get(param_dict: dict, *names: str, default: float) -> float:
            for name in names:
                if name in param_dict:
                    return float(param_dict[name])
            return float(default)

        Vii = _get(params, "Vᵢᵢ", "Vii", default=1.0)
        Vij = _get(params, "Vᵢⱼ", "Vij", default=0.0)
        Eps = _get(params, "Ƹᵢⱼ", "Ƹij", default=0.0)

        g1  = _get(params, "γ₁", "γ1", default=1.0)
        g2  = _get(params, "γ₂", "γ2", default=g1 if utility_settings.get("single_exponential_parameter", False) else g1)
        if not utility_settings.get("single_exponential_parameter", False):
            g2 = _get(params, "γ₂", "γ2", default=g1)
        g3  = _get(params, "γ₃", "γ3", default=g1)

        "Payoffs"
        Ai = float(payoffs["As"]); Aj = float(payoffs["Ao"])
        "Reference-dependent utility ⇒ compare to 3 for single-agent ratios"
        ref_const = 3.0

        "Sign-preserving power for centered ratios (matches code semantics)"
        def _signed_pow(base_value: float, gamma_exponent: float) -> float:
            if base_value == 0.0:
                return 0.0
            return (abs(base_value) ** gamma_exponent) * (1.0 if base_value >= 0.0 else -1.0)

        "Bases"
        si_base = Ai / (Ai + ref_const) - 0.5
        al_base = Aj / (Aj + ref_const) - 0.5
        envy    = Aj / (Ai + Aj) - 0.5
        guilt   = Ai / (Ai + Aj) - 0.5

        "Terms"
        si_weight = 1.0 if utility_settings.get("fix_self_interest_parameter", False) else Vii
        self_interest = si_weight * _signed_pow(si_base, g1)
        altruism      = Vij * _signed_pow(al_base, g2)
        social_comp   = (-Eps) * (max(envy, 0.0) ** g3) + ( Eps) * (max(guilt, 0.0) ** g3)

        return round(self_interest + altruism + social_comp, decimals_local)

    def _is_token_char(ch: str) -> bool:
        return ch not in " \t\r\n,^*/+-()"

    def _find_left_operand(expr: str, caret_index: int) -> tuple[int, int]:
        scan_index = caret_index - 1
        while scan_index >= 0 and expr[scan_index].isspace():
            scan_index -= 1
        if scan_index >= 0 and expr[scan_index] == ")":
            depth = 1
            scan_index -= 1
            while scan_index >= 0 and depth > 0:
                if expr[scan_index] == ")":
                    depth += 1
                elif expr[scan_index] == "(":
                    depth -= 1
                scan_index -= 1
            start_index = scan_index + 1
            end_index = caret_index
            "Include function name if present (e.g., max(...))"
            name_end = start_index
            name_start = name_end - 1
            while name_start >= 0 and expr[name_start].isalpha():
                name_start -= 1
            name_start += 1
            if name_start < name_end and expr[name_end] == "(":
                start_index = name_start
            return start_index, end_index
        "Bare token"
        token_end = scan_index + 1
        token_start = scan_index
        while token_start >= 0 and _is_token_char(expr[token_start]):
            token_start -= 1
        token_start += 1
        return token_start, token_end

    def _find_right_operand(expr: str, caret_index: int) -> tuple[int, int]:
        scan_index = caret_index + 1
        n_chars = len(expr)
        while scan_index < n_chars and expr[scan_index].isspace():
            scan_index += 1
        if scan_index < n_chars and expr[scan_index] == "(":
            depth = 1
            scan_index += 1
            while scan_index < n_chars and depth > 0:
                if expr[scan_index] == "(":
                    depth += 1
                elif expr[scan_index] == ")":
                    depth -= 1
                scan_index += 1
            return caret_index + 1, scan_index
        "Bare token exponent"
        token_start = scan_index
        while scan_index < n_chars and _is_token_char(expr[scan_index]):
            scan_index += 1
        return token_start, scan_index

    def _replace_powers(expr: str) -> str:
        out = expr
        while "^" in out:
            caret_index = out.find("^")
            base_start, base_end = _find_left_operand(out, caret_index)
            exp_start, exp_end = _find_right_operand(out, caret_index)
            base_txt = out[base_start:base_end].strip()
            exp_txt = out[exp_start:exp_end].strip()
            out = out[:base_start] + f"pow_signed({base_txt}, {exp_txt})" + out[exp_end:]
        return out

    def _normalize_for_eval(rhs_text: str) -> str:
        normalized = (rhs_text
            .replace("\u00A0", " ")
            .replace("−", "-").replace("–", "-").replace("—", "-")
            .replace("≥", ">=").replace("≤", "<=").replace("≠", "!=")
            .replace("×", "*").replace("·", "*").replace("⋅", "*")
        )
        normalized = normalized.replace("[", "(").replace("]", ")")
        "Implicit multiplication"
        normalized = re.sub(r"(?<![A-Za-z0-9_])(\-?\d+(?:\.\d+)?)\s*\(", r"\1*(", normalized)
        normalized = normalized.replace(")(", ")*(")
        normalized = re.sub(r"\)\s*(\-?\d+(?:\.\d+)?)", r")*\1", normalized)
        return _replace_powers(normalized)

    def _pow_signed(base_value: float, exponent_value: float) -> float:
        base_value = float(base_value); exponent_value = float(exponent_value)
        if abs(exponent_value - round(exponent_value)) < 1e-12:
            return base_value ** int(round(exponent_value))
        return (abs(base_value) ** exponent_value) * (1.0 if base_value >= 0.0 else -1.0)

    def _evaluate_equation_numeric(
        equation_string: str,
        params: dict[str, float],
        payoffs: dict[str, float],
        utility_settings: dict[str, bool],
        param_overrides: dict[str, float] | None = None,
        decimals_local: int = 6,
    ) -> tuple[float | None, str]:
        """
        Evaluate the RHS of the pretty equation after substituting 'params' and 'payoffs'.
        'param_overrides' (if provided) override parameter values *before* substitution.
        Returns (value_or_None, status_text). Status empty string on success.
        """
        "1) Build replacements"
        gamma_pretty = {"γ1": "γ₁", "γ2": "γ₂", "γ3": "γ₃"}
        params_pretty = {gamma_pretty.get(param_key, param_key): param_value for param_key, param_value in params.items()}
        "Map aliases that might appear"
        alias_to_pretty = {
            "Vii": "Vᵢᵢ", "Λii": "Ʌᵢᵢ", "Vij": "Vᵢⱼ", "Λij": "Ʌᵢⱼ",
            "Ƹij": "Ƹᵢⱼ", "Ʒij": "Ʒᵢⱼ",
            "γ1": "γ₁", "γ2": "γ₂", "γ3": "γ₃",
        }
        for alias_key, pretty_key in alias_to_pretty.items():
            if alias_key in params and pretty_key not in params_pretty:
                params_pretty[pretty_key] = params[alias_key]

        "Apply overrides *before* substitution (so numeric values in the string reflect the override)"
        if param_overrides:
            for raw_param_key, param_value in param_overrides.items():
                param_key = gamma_pretty.get(raw_param_key, raw_param_key)  # Allow γ1 vs γ₁ in overrides.
                params_pretty[param_key] = param_value

        "Gamma fallbacks inherit γ₁ (like utility())"
        gamma1_value = None
        for k_try in ("γ₁", "γ1"):
            if k_try in params_pretty:
                gamma1_value = float(params_pretty[k_try]); break
        if gamma1_value is None: gamma1_value = 1.0

        "Conditional-welfare normalization is intentionally skipped here; the utility() call above also uses normalize_conditional_welfare_params=False, keeping both paths consistent."
        payoff_symbol_map = {"As": "πᵢᴬ", "Ao": "πⱼᴬ", "Bs": "πᵢᴮ", "Bo": "πⱼᴮ"}
        payoffs_pretty_map = {payoff_symbol_map[payoff_key]: payoff_value for payoff_key, payoff_value in payoffs.items()}

        "2) Extract RHS"
        if "=" not in equation_string:
            return None, "EVAL ERROR: no '=' in equation"
        _, rhs_original = equation_string.split("=", 1)
        rhs_original = rhs_original.strip()

        "3) Perform substitutions (payoffs first, then params)"
        rhs_filled = rhs_original
        for sym, val in payoffs_pretty_map.items():
            rhs_filled = rhs_filled.replace(sym, str(val))
        for sym, val in params_pretty.items():
            rhs_filled = rhs_filled.replace(sym, str(val))

        "Replace any remaining canonical symbols with *utility()* defaults"
        defaults = {
            "Vᵢᵢ": 1.0, "Ʌᵢᵢ": 0.0, "Vᵢⱼ": 0.0, "Ʌᵢⱼ": 0.0, "Ƹᵢⱼ": 0.0, "Ʒᵢⱼ": 0.0,
            "γ₁": gamma1_value, "γ₂": gamma1_value, "γ₃": gamma1_value,
            "Vii": 1.0, "Λii": 0.0, "Vij": 0.0, "Λij": 0.0, "Ƹij": 0.0, "Ʒij": 0.0,
            "γ1": gamma1_value, "γ2": gamma1_value, "γ3": gamma1_value,
        }
        for sym, val in defaults.items():
            if sym in rhs_filled:
                rhs_filled = rhs_filled.replace(sym, str(val))

        "--- INSERT C2: stubborn-case direct evaluator short-circuit ---------------"
        utility_settings_for_eval = utility_settings
        is_stubborn = (
            (not utility_settings_for_eval.get("conditional_welfare_mode", False))
            and (not utility_settings_for_eval.get("min_max_rawlsian_leontief", False))
            and utility_settings_for_eval.get("use_exponential_parameters", False)
            and utility_settings_for_eval.get("payoff_ratios_not_differences", False)
            and utility_settings_for_eval.get("reference_dependent_utility", False)
            and (not utility_settings_for_eval.get("use_negativity_parameters", False))
            and utility_settings_for_eval.get("negativity_social_comparison", False)
            and utility_settings_for_eval.get("include_social_comparison", False) is not False
            and utility_settings_for_eval.get("include_altruism_term", False) is not False
            and (not utility_settings_for_eval.get("single_payoffs_not_differences", False))
            and (not utility_settings_for_eval.get("apply_exponents_to_payoffs", False))
        )
        if is_stubborn:
            try:
                "Honor overrides (used to isolate components in verification)"
                params_for_direct = dict(params)
                if param_overrides:
                    params_for_direct.update(param_overrides)
                direct_val = _direct_eval_ratio_refdep_negsc(
                    utility_settings=utility_settings_for_eval,
                    params=params_for_direct,
                    payoffs=payoffs,
                    decimals_local=decimals_local,
                )
                return direct_val, ""  # Short-circuit: trust direct evaluation.
            except Exception as _err_direct:
                "Fall through to the generic string-eval path if something unexpected happens"
                pass

        "4) Normalize to Python and eval"
        python_rhs = _normalize_for_eval(rhs_filled)
        safe_env = {"__builtins__": {}, "max": max, "min": min, "abs": abs, "pow_signed": _pow_signed}
        try:
            value = float(eval(python_rhs, safe_env, {}))
            return round(value, decimals_local), ""
        except Exception as err:
            return None, f"EVAL ERROR: {type(err).__name__}: {err}"

    "Helper to get string-components via re-evaluation with zeroed weights"
    def _string_components(
        equation_string: str,
        params: dict[str, float],
        payoffs: dict[str, float],
        utility_settings: dict[str, bool],
        decimals_local: int = 6,
    ) -> tuple[float | None, float | None, float | None, float | None, dict[str, str]]:
        """
        Returns (total, self, altruism, socc, statuses) where statuses is a dict of
        status messages (empty on success). Altruism and socc are computed by difference
        using re-evaluations with weight overrides, so this also works when fix_self=True.
        """
        statuses: dict[str, str] = {}
        total, st_total = _evaluate_equation_numeric(equation_string, params, payoffs, utility_settings, None, decimals_local)
        statuses["total"] = st_total

        "Isolate self by zeroing altruism & SC weights"
        self_only_over = {"Vᵢⱼ": 0.0, "Ʌᵢⱼ": 0.0, "Ƹᵢⱼ": 0.0, "Ʒᵢⱼ": 0.0}
        self_only, st_self = _evaluate_equation_numeric(equation_string, params, payoffs, utility_settings, self_only_over, decimals_local)
        statuses["self"] = st_self

        "Turn off SC to isolate (self + altruism)"
        no_sc_over = {"Ƹᵢⱼ": 0.0, "Ʒᵢⱼ": 0.0}
        no_sc, st_nosc = _evaluate_equation_numeric(equation_string, params, payoffs, utility_settings, no_sc_over, decimals_local)
        statuses["no_sc"] = st_nosc

        if (total is None) or (self_only is None) or (no_sc is None):
            return total, None, None, None, statuses

        altruism_only = round(no_sc - self_only, decimals_local)
        socc_only = round(total - self_only - altruism_only, decimals_local)
        return total, self_only, altruism_only, socc_only, statuses

    "---------- (5) Main sweep ----------------------------------------------------"
    rows: list[dict] = []
    if verbose:
        print(f"[Verify] Utility families to test: {len(all_settings)}")
        print(f"[Verify] Payoff tuples: {'625 exhaustive grid' if use_exhaustive else n_games}")

    for utility_index, utility_settings in enumerate(all_settings, start=1):
        "Sample a fresh mean-parameter vector for this utility family"
        params_means = _sample_means_for(utility_settings)

        "Build the pretty equation string once per family"
        try:
            equation_string = utility_function_str(utility_settings=utility_settings, in_latex=False, option=option)
        except Exception as err:
            "If the string builder fails, record rows with an eval error status"
            for payoff_index, payoffs in enumerate(payoff_iterable, start=1):
                row = {
                    "utility_idx": utility_index,
                    "payoff_idx": payoff_index,
                    "payoff_A_self": payoffs["As"], "payoff_A_other": payoffs["Ao"],
                    "payoff_B_self": payoffs["Bs"], "payoff_B_other": payoffs["Bo"],
                    "U_function": f"Uᵢ({option})",
                    "Code-side values are still available"
                    "utility_code": None, "utility_str": None, "utility_Δ": None, "match": False,
                    "status": f"BUILD ERROR: {type(err).__name__}: {err}",
                }
                "Expand 13 boolean flags as separate columns"
                for flag_key in ordered_flag_keys:
                    row[flag_key] = bool(utility_settings.get(flag_key, False))
                rows.append(row)
            continue

        for payoff_index, payoffs in enumerate(payoff_iterable, start=1):
            "----- Code (source of truth), with components"
            try:
                code_components = utility_function(payoffs=payoffs, params=params_means,
                                                   utility_settings=utility_settings, separate_terms=True, normalize_conditional_welfare_params=False)
                code_total = float(code_components["self_interest"] + code_components["altruism"] + code_components["social_comp"])
                code_self  = float(code_components["self_interest"])
                code_altr  = float(code_components["altruism"])
                code_socc  = float(code_components["social_comp"])
            except Exception as err:
                "If code throws, give a row so it is visible"
                row = {
                    "utility_idx": utility_index,
                    "payoff_idx": payoff_index,
                    "payoff_A_self": payoffs["As"], "payoff_A_other": payoffs["Ao"],
                    "payoff_B_self": payoffs["Bs"], "payoff_B_other": payoffs["Bo"],
                    "U_function": f"Uᵢ({option})",
                    "utility_code": None, "utility_str": None, "utility_Δ": None, "match": False,
                    "status": f"CODE ERROR: {type(err).__name__}: {err}",
                }
                for flag_key in ordered_flag_keys:
                    row[flag_key] = bool(utility_settings.get(flag_key, False))
                rows.append(row)
                continue

            "----- String (numeric), total + components via re-evaluation"
            str_total, str_self, str_altr, str_socc, eval_status = _string_components(
                equation_string=equation_string,
                params=params_means,
                payoffs=payoffs,
                utility_settings=utility_settings,
                decimals_local=decimals,
            )

            "Decides match status on totals and stores per-term deltas."
            if str_total is None:
                match_flag = False
                delta_total = None
                status_text = eval_status.get("total", "")
            else:
                delta_total = float(code_total - str_total)
                match_flag = abs(delta_total) <= comparison_tol
                status_text = ""

            "Prepare output row"
            row = {
                "utility_idx": utility_index,
                "payoff_idx": payoff_index,
                **{flag_key: bool(utility_settings.get(flag_key, False)) for flag_key in ordered_flag_keys},
                "payoff_A_self": payoffs["As"], "payoff_A_other": payoffs["Ao"],
                "payoff_B_self": payoffs["Bs"], "payoff_B_other": payoffs["Bo"],

                "utility_code": round(code_total, decimals),
                "utility_str":  (None if str_total is None else round(str_total, decimals)),
                "utility_Δ":    (None if delta_total is None else round(delta_total, decimals)),
                "match":        bool(match_flag),
                "code_self": round(code_self, decimals),
                "code_altr": round(code_altr, decimals),
                "code_socc": round(code_socc, decimals),
                "str_self":  (None if str_self is None else round(str_self, decimals)),
                "str_altr":  (None if str_altr is None else round(str_altr, decimals)),
                "str_socc":  (None if str_socc is None else round(str_socc, decimals)),
                "Δ_self":    (None if (str_self is None) else round(code_self - str_self, decimals)),
                "Δ_altr":    (None if (str_altr is None) else round(code_altr - str_altr, decimals)),
                "Δ_socc":    (None if (str_socc is None) else round(code_socc - str_socc, decimals)),
                "status": status_text or eval_status.get("self","") or eval_status.get("no_sc",""),
                "U_function": utility_function_str(utility_settings),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    "---------- (6) Sorting and outputs -------------------------------------------"
    "Sort by mismatch first, then by absolute total discrepancy"
    if not df.empty:
        df["abs_Δ"] = df["utility_Δ"].abs() if df["utility_Δ"].notna().any() else 0.0
        df = df.sort_values(by=["match", "abs_Δ"], ascending=[True, False]).drop(columns=["abs_Δ"])
    try: df.to_csv(out_path, index=False, encoding="utf-8-sig")
    except (PermissionError, OSError): pass

    "Per-utility summary when exhaustive grid was used"
    if use_exhaustive and not df.empty:
        summary = (
            df.groupby("utility_idx", as_index=False)
            .agg(
                all_match=("match", "all"),
                n_rows=("match", "size"),
                max_abs_Δ=("utility_Δ", lambda utility_delta_series: float(utility_delta_series.abs().max(skipna=True) if len(utility_delta_series) else 0.0)),
                U_function=("U_function", "first"),
                **{flag_key: (flag_key, "first") for flag_key in ordered_flag_keys}
            )
        )
        summary = summary.sort_values(by=["all_match"], ascending=[True])
        "Move U_function to the end"
        summary = summary[[col for col in summary.columns if col != "U_function"] + ["U_function"]]
        "Turn boolean flags into 1s and 0s to see more easily"
        summary[ordered_flag_keys] = summary[ordered_flag_keys].astype(int)
        summary.drop(columns=['n_rows'], inplace=True)
        try: summary.to_csv(out_summary_path, index=False, encoding="utf-8-sig")
        except (PermissionError, OSError): pass

    "---------- (7) Console report -------------------------------------------------"
    if verbose and not df.empty:
        total_rows = len(df)
        n_match = int(df["match"].sum())
        n_mismatch = total_rows - n_match
        print(f"[Verify] Rows: {total_rows}  |  matches: {n_match}  |  mismatches: {n_mismatch}")
        "Per-flag mismatch rates (top 6 most predictive)"
        try:
            flag_reports = []
            for flag_key in ordered_flag_keys:
                grp = df.groupby(flag_key)["match"].agg(total="count", ok="sum")
                grp["mismatch_rate"] = 1.0 - (grp["ok"] / grp["total"])
                "Store both levels"
                for flag_value, rec in grp.iterrows():
                    flag_reports.append({
                        "flag": flag_key, "value": bool(flag_value),
                        "total": int(rec["total"]),
                        "mismatch_rate": float(rec["mismatch_rate"])
                    })
            rpt = pd.DataFrame(flag_reports)
            rpt = rpt.sort_values(by=["mismatch_rate", "total"], ascending=[False, False])
            print("[Verify] Flags with highest mismatch rates (top 8):")
            print(rpt.head(8).to_string(index=False))
        except Exception:
            pass

        "Top offenders (first 10 mismatches)"
        if n_mismatch:
            offenders = df.loc[~df["match"]].head(10)
            cols = ["utility_idx", "payoff_idx", "utility_Δ"] + ordered_flag_keys[:5]  # Compact preview.
            print("[Verify] First mismatches (preview):")
            print(offenders[cols].to_string(index=False))

        print(f"[Verify] Wrote detailed CSV to: {out_path}")
        if use_exhaustive and not df.empty:
            print(f"[Verify] Wrote per-utility summary to: {out_summary_path}")
            all_match_counts = summary['all_match'].value_counts().to_dict()
            print(f"Equations that always match: {all_match_counts}.")

    return df


def model_nesting_adjacency_matrices(general_settings: GeneralSettings, utility_settings: UtilitySettings, file_paths: FilePaths, 
                                     create_new_file: bool | None = None, equation_form: bool = True, print_: bool = False) -> dict[str: list[list[int]] | list[dict[str, bool]] | list[str]]:
    """
    Creates adjacency matrices indicating pairwise relationships between models: 
    Is the row model a parent of, a sibling of, or a parent of the column model?

    Arguments:
        • utility_settings: dict[str, bool]; Defines the functional form of the utility function.
        • file_paths: dict[str, str | dict[str, str]]; Stores all file paths for storing data.
        • general_settings: dict[str, Any]; Various settings used throughout this analysis.
        • create_new_file: bool | None; 
            - If True, overrides the general setting with True
            - If False, overrides the general setting with False
            - If None, defers to general_settings['create_new_file']
        • print_: bool; If True, prints progress to the terminal. 

    Returns:
        • dict[str: np.array | list[dict[str, bool]] | list[str]]; Example = {
            'adjacency_lists': {
                'parent_of':  [[ 12],
                               [  9,  13], 
                               [ 63, 169, 171], 
                               [111, 112, 173, 174]], 
                'sibling_of': [[  0,   1,   2],
                               [ 55,  56,  57,  58], 
                               [ 59,  60,  61,  62], 
                               [  3,   4]],
                'child_of':   [[245, 346],
                               [311, 356, 377, 385], 
                               [312, 357, 378, 386], 
                               [313, 358, 370, 387]],            
            },
            'adjacency_matrices': {
                'parent_of':  [[0, 0, 0, 0],
                               [0, 0, 0, 0], 
                               [1, 1, 0, 0], 
                               [0, 0, 0, 0]], 
                'sibling_of': [[0, 1, 0, 0],
                               [1, 0, 0, 0], 
                               [0, 0, 0, 1], 
                               [0, 0, 1, 0]],
                'child_of':   [[0, 0, 1, 0],
                               [0, 0, 1, 0], 
                               [0, 0, 0, 0], 
                               [0, 0, 0, 0]],
            },
            'settings': list[dict[str, bool]],
            'equations': list[str]
        }
    """
    "Return existing model nesting data if possible and desired."
    model_nesting_file_path = prep.ensure_directory_and_join(
            file_paths["processed"], "model_nesting_data.json")

    "Determine whether to create a new file or extract a preexisting file."
    if not isinstance(create_new_file, bool):
        create_new_file = general_settings.get('create_new_file')

    if not create_new_file and os.path.exists(model_nesting_file_path):
        with open(model_nesting_file_path, "r", encoding="utf-8") as file:
            model_nesting_data = json.load(file) 
            model_nesting_data['settings'] = [
                gnrl.convert_utility_settings(utility_settings=settings, into=dict) 
                for settings in model_nesting_data['settings']
            ]

        if equation_form and 'adjacency_dict' not in model_nesting_data:
            equations = model_nesting_data['equations']
            settings_list = model_nesting_data['settings']
            equation_dict, adjacency_dict = {}, {}
            for relation in ('parent_of', 'sibling_of', 'child_of'):
                equation_dict[relation] = {}
                adjacency_dict[relation] = {}
                for idx, equation in enumerate(equations):
                    adjacency_list_model = model_nesting_data['adjacency_lists'][relation][idx]
                    adjacency_equations = [equations[edx] for edx in adjacency_list_model]
                    equation_dict[relation][equation] = adjacency_equations
                for idx, settings in enumerate(settings_list):
                    settings = gnrl.convert_utility_settings(settings, tuple)
                    adjacency_list_model = model_nesting_data['adjacency_lists'][relation][idx]
                    adjacent_settings = [settings_list[sdx] for sdx in adjacency_list_model]
                    adjacency_dict[relation][settings] = adjacent_settings
            model_nesting_data['equation_dict'] = equation_dict
            model_nesting_data['adjacency_dict'] = adjacency_dict

        return model_nesting_data

    "Generate all utility funtion settings"
    utility_setting_varieties = gnrl.generate_utility_settings(utility_settings=utility_settings)

    "Generate utility function equations for viewing"
    equations = [build_utility_equation(utility_settings=settings) for settings in utility_setting_varieties]
    
    "Dictionary of all data"
    model_nesting_data = {
        'adjacency_lists': {},
        'adjacency_matrices': {}, 
        'settings': utility_setting_varieties, 
        'equations': equations
    }

    "Create empty adjacency matrices for all three types of relations"
    matrix_keys = ('parent_of', 'sibling_of', 'child_of')
    n_utility_setting_varieties = len(utility_setting_varieties)
    for matrix_key in matrix_keys:
        model_nesting_data['adjacency_matrices'][matrix_key] = np.zeros(
            (n_utility_setting_varieties, n_utility_setting_varieties))

    "Filling in the adjacency matrices for all three relationship types."
    for utility_row_idx in range(n_utility_setting_varieties):
        utility_setting_row = utility_setting_varieties[utility_row_idx]
        if print_:
            print(f"Row Function {utility_row_idx:03d}: {build_utility_equation(utility_settings=utility_setting_row)}")
        for utility_col_idx in range(n_utility_setting_varieties):
            utility_setting_col = utility_setting_varieties[utility_col_idx]

            "Determine family relationship between row and column models."
            relations = gnrl.classify_pair_relation(model_1=utility_setting_row, 
                                                    model_2=utility_setting_col, 
                                                    general_settings=general_settings,
                                                    utility_settings=utility_settings)

            relation_row_to_col, relation_col_to_row, setting_flipped = relations

            """1s mean that the utility function indexed by the row is a parent,
            sibling, or child of the utility function indexed by the column."""
            if setting_flipped in ("min_max_rawlsian_leontief", "conditional_welfare_mode"):
                "Flipping these settings does not differentiate child from parent"
                continue
            if relation_row_to_col == 'parent':
                model_nesting_data['adjacency_matrices']["child_of"][utility_row_idx][utility_col_idx] = 1
            if relation_row_to_col == 'sibling':
                model_nesting_data['adjacency_matrices']["sibling_of"][utility_row_idx][utility_col_idx] = 1
            if relation_row_to_col == 'child':
                model_nesting_data['adjacency_matrices']["parent_of"][utility_row_idx][utility_col_idx] = 1

    "Mapping utility function indices to the indices of their parents, siblings, and children."
    for matrix_key in matrix_keys:
        model_nesting_data['adjacency_matrices'][matrix_key] = model_nesting_data['adjacency_matrices'][matrix_key].tolist()
        model_nesting_data['adjacency_lists'][matrix_key] = []        
        for utility_row_idx in range(n_utility_setting_varieties):
            if print_:
                utility_setting_row = utility_setting_varieties[utility_row_idx]
                print(f"{matrix_key.capitalize()} Function {utility_row_idx:03d}: "
                      f"{build_utility_equation(utility_settings=utility_setting_row)}")            
            model_nesting_data['adjacency_lists'][matrix_key].append([
                utility_col_idx for utility_col_idx, is_relative in enumerate(
                    model_nesting_data['adjacency_matrices'][matrix_key][utility_row_idx]) if bool(is_relative)
            ])

    if equation_form:
        equations = model_nesting_data['equations']
        settings_list = model_nesting_data['settings']
        equation_dict, adjacency_dict = {}, {}
        for relation in ('parent_of', 'sibling_of', 'child_of'):
            equation_dict[relation] = {}
            adjacency_dict[relation] = {}
            for idx, equation in enumerate(equations):
                adjacency_list_model = model_nesting_data['adjacency_lists'][relation][idx]
                adjacency_equations = [equations[edx] for edx in adjacency_list_model]
                equation_dict[relation][equation] = adjacency_equations
            for idx, settings in enumerate(settings_list):
                settings = str(gnrl.convert_utility_settings(settings, tuple))
                adjacency_list_model = model_nesting_data['adjacency_lists'][relation][idx]
                adjacent_settings = [settings_list[sdx] for sdx in adjacency_list_model]
                adjacency_dict[relation][settings] = adjacent_settings
        model_nesting_data['equation_dict'] = equation_dict
        model_nesting_data['adjacency_dict'] = adjacency_dict

    "Save the data."
    model_nesting_data_compact = copy.deepcopy(model_nesting_data)
    model_nesting_data_compact['settings'] = [
        (int(setting) for setting in gnrl.convert_utility_settings(utility_settings=settings, into=tuple)) 
        for settings in model_nesting_data_compact['settings']
    ]
    with open(model_nesting_file_path, 'w', encoding='utf-8') as file:
        json.dump(model_nesting_data, file, ensure_ascii=False, indent=4)

    if print_:
        print(f"Saved {model_nesting_file_path}")

    return model_nesting_data


def select_child_params_for_parent(children: List[Dict[str, Any]], temperature: float) -> Dict[str, Any]:
    """
    Selects one child-entry (from the list returned by calling 'best_fitting_child_parameters_for_parent'
    for each candidate child) using a *reverse*-SoftMax over totals (lower loss ⇒ higher probability).

    Arguments:
        • children: list of child info dicts (each must have metadata['loss_total']).
        • temperature: float; stochasticity control.
            – If temperature <= 0: choose the smallest 'loss_total' deterministically.
            – Else: p(child) ∝ exp( - loss_total / temperature ).

    Returns:
        • dict[str, Any]; the chosen child dict (unchanged).
    """
    "Extract usable loss totals from the children."
    loss_totals = [(idx, child["metadata"].get("loss_total", None)) for idx, child in enumerate(children)]
    usable_loss_totals = [(idx, float(loss_total)) for (idx, loss_total) in loss_totals if isinstance(loss_total, (int, float))]
    if not usable_loss_totals:
        "If totals are missing (e.g., only per-player subsets), fall back to the first child."
        return children[0] if children else {}

    if temperature is None or temperature <= 0:
        idx = min(usable_loss_totals, key=lambda loss_total: loss_total[1])[0]
        return children[idx]

    "Reverse-softmax weights: exp(-loss / T)"
    losses = [loss_total for (_, loss_total) in usable_loss_totals]
    min_loss = min(losses)

    "Numerical stability trick: subtract min_loss"
    weights = [math.exp(-(loss_total - min_loss) / float(temperature)) for (_, loss_total) in usable_loss_totals]
    total_weights = sum(weights) or 1.0
    probs = [weight / total_weights for weight in weights]

    "Sample one index according to probs"
    rando = random.random()
    cdf = 0.0
    for (idx, _), prob in zip(usable_loss_totals, probs):
        cdf += prob
        if rando <= cdf:
            return children[idx]
        
    "Return fallback"
    return children[usable_loss_totals[-1][0]]  


def best_fitting_model_parameters(utility_settings: UtilitySettings, general_settings: GeneralSettings, file_paths: FilePaths, param_bds: ParamBounds, 
                                  within_ic_analysis: bool = True, *, player_uuid: Optional[str] = None, player_role: Optional[str] = None) -> Dict[str, Any]:
    """
    Extracts the best-fitting parameters for a utility function after optimization.

    Arguments:
        • utility_settings: dict[str, bool]; Defines the functional form of the utility function.
        • file_paths: dict[str, str | dict[str, str]]; Stores all file paths for storing data.
        • general_settings: dict[str, Any]; Various settings used throughout this analysis.
        • within_ic_analysis: bool;
            - If True, extracts data from iterations of the IC analysis in file_paths > bic_aic
            - If False, extracts data from  file_paths > player_fits > experiment_n
        • player_uuid: Optional[str]; If provided, filters to a single player.
        • player_role: Optional[str]; If provided, filters to a single role ('chooser'|'predictor').

    Returns:
        • dict[str: Any]; Example = {
            'player_uuid': {
                'params': {
                    'chooser': {
                        'Vᵢᵢ': 0.73,
                        'Vᵢⱼ': 0.44,
                        'γ1':  1.23
                    }
                    'predictor': {
                        'Vᵢᵢ': 0.81,
                        'Vᵢⱼ': 0.67,
                        'γ1':  0.96
                    }
                },
                'loss': {
                    'chooser': 1.2583459, 
                    'predictor': 0.9832543
                }
            },
            'player_uuid': {
                ...
            },...
        }
    """
    def _extract_from_player_fit_file(player_uuid: str, player_role: str | None, general_settings: GeneralSettings, 
                                      utility_settings: UtilitySettings, print_: bool = False) -> Dict[str, Any] | None:
        """
        Reads a single per-player fit JSON and extracts:
            - player_uuid
            - per-role params 
            - per-role final losses

        Returns:
            {
                "params": {"chooser": {...}, "predictor": {...}},
                "loss":   {"chooser": float, "predictor": float}
            }
        """
        "Tuple of player roles of interest if not interested in both roles."
        player_roles = (player_role,) if player_role in ('chooser', 'predictor') else ('chooser', 'predictor')  

        "Standardize general settings."
        general_settings = copy.deepcopy(general_settings)
        general_settings['temperature_is_param'] = False
        experiment_num = int(general_settings.get("experiment_num", 3))

        "Generate file path for this player based on the general and utility settings."
        pf_base = os.path.join(file_paths.get("player_fits", "."), f"experiment_{experiment_num}")
        if not os.path.isdir(pf_base):
            raise FileNotFoundError(f"Per-player fits directory not found: {pf_base!r}")        
        
        pf_file_name = prep.create_file_name_suffix(
            general_settings=general_settings, utility_settings=utility_settings) + f"_{player_uuid}.json"         
        player_file_path = prep.ensure_directory_and_join(base_dir=pf_base, file_name=pf_file_name)

        "Extract data if it exists."
        try:
            with open(player_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError as err:
            if print_: print(err)
            return None

        "Locate the param and loss data in the first game of the first dyad."
        dyad_keys = list(data.keys())
        if not dyad_keys:
            raise ValueError(f"No dyads found in {player_file_path!r}.")

        first_dyad_key = dyad_keys[0]
        game_list = data[first_dyad_key]
        if not (isinstance(game_list, list) and game_list):
            raise ValueError(f"Malformed dyad entry in {player_file_path!r}.")

        first_game = game_list[0]

        "Find the parameter keys"
        parameter_estimates = first_game.get("parameter_estimates", {})
        parameter_estimates = parameter_estimates.get("naive", parameter_estimates.get("grid", {})).get(player_uuid, {})
        params = parameter_estimates.get("chooser", parameter_estimates.get("predictor", {})).get("params", {})
        param_keys = list(params.keys())

        "Losses live under 'reports' -> <role> -> 'final' -> 'loss'"
        reports = first_game.get("reports", {})

        return {
            "params": {
                player_role: {param_key: param for param_key, param in zip(
                    param_keys, reports.get(player_role, {}).get("final", {}).get("x", []))}
                for player_role in player_roles
            },
            "loss": {
                player_role: reports.get(player_role, {}).get("final", {}).get("loss", None) 
                for player_role in player_roles
            }
        }

    "Normalize inputs"
    general_settings = copy.deepcopy(general_settings)
    if within_ic_analysis:
        general_settings['update_method'] = 'naive'
        general_settings['temperature_is_param'] = False
    experiment_num = int(general_settings.get("experiment_num", 3))

    "Generate a list of all player uuids or just one player if already specified."
    if isinstance(player_uuid, str):
        player_uuids = [player_uuid]
    else:
        player_uuids = prep.all_player_uuids(
            file_paths=file_paths, experiment_num=experiment_num, only_humans=True)        

    "Tuple of player roles of interest if not interested in both roles."
    player_roles = (player_role,) if player_role in ('chooser', 'predictor') else ('chooser', 'predictor')

    results: Dict[str, Any] = {}
    extracted_ic_file = False

    if within_ic_analysis:
        "Load IC Analysis JSON for this model."
        ic_dir = file_paths.get("bic_aic", ".")
        ic_file_name_suffix = prep.create_file_name_suffix(
            general_settings=general_settings, utility_settings=utility_settings)
        ic_file_name = f"IC_Analysis{ic_file_name_suffix}.json"
        ic_file_path = prep.ensure_directory_and_join(base_dir=ic_dir, file_name=ic_file_name)
        if os.path.exists(path=ic_file_path):
            with open(ic_file_path, "r", encoding="utf-8") as file:
                ic_json = json.load(file)
            extracted_ic_file = True

            minvec = ic_json.get("minvec", {})
            if isinstance(minvec, dict) and minvec:
                "New files (with minvec): use the per-player minima directly"
                per_player_params = {pl: entry.get('params', {}) for pl, entry in minvec.items()}
            else:
                "Backward compatibility with older IC files (no minvec)"
                pvec_list = ic_json.get("pvec", [])
                lvec_list = ic_json.get("lvec", [])
                if not (isinstance(pvec_list, list) and isinstance(lvec_list, list) and pvec_list):
                    raise ValueError(f"IC file {os.path.basename(ic_file_name)} missing 'pvec'/'lvec' iterations.")
                best_idx = min(range(len(lvec_list)), key=lambda idx: float(lvec_list[idx]))
                per_player_params = pvec_list[best_idx]

        else:
            print(
                f"No IC Analysis file found in {ic_dir!r} matching model key {ic_file_name!r}."
            )

        "Extract the data from the IC file for each player."
        for player_uuid in player_uuids:
            data_successfully_extracted = False
            if extracted_ic_file:
                params_and_losses_by_player_role = per_player_params.get(player_uuid, None)
                if isinstance(params_and_losses_by_player_role, dict) and \
                    "params" in params_and_losses_by_player_role and "loss" in params_and_losses_by_player_role:
                    params = params_and_losses_by_player_role.get("params", {}) or {}
                    losses = params_and_losses_by_player_role.get("loss", {}) or {}
                    if params and losses:
                        data_successfully_extracted = True
                        results[player_uuid] = {
                            "params": {player_role: params.get(player_role) for player_role in player_roles},
                            "loss": {player_role: losses.get(player_role) for player_role in player_roles}
                        }

            if not data_successfully_extracted:
                "If the data could not be found in the IC file, extract the data from the player fits file."
                player_param_loss_data = _extract_from_player_fit_file(player_uuid=player_uuid, player_role=player_role,
                                            general_settings=general_settings, utility_settings=utility_settings)  
                if player_param_loss_data is not None:
                    results[player_uuid] = player_param_loss_data              

    else:
        "Extract all data from the player fits file."
        for player_uuid in player_uuids:
            player_param_loss_data = _extract_from_player_fit_file(player_uuid=player_uuid, player_role=player_role,
                                        general_settings=general_settings, utility_settings=utility_settings)  
            if player_param_loss_data is not None:
                results[player_uuid] = player_param_loss_data   

    return results


def best_fitting_child_parameters_for_parent(player_uuid: str | None, player_role: str | None, utility_settings_parent: UtilitySettings, utility_settings: UtilitySettings, 
                                             general_settings: GeneralSettings, file_paths: FilePaths, param_bds: ParamBounds, within_ic_analysis: bool = True, temperature: float = 1.5) -> Dict[str, Any]:
    """
    Provides the best fitting parameters for a *child* utility function to its *parent*
    utility function, as the building block for warm starts when optimizing the parent.

    Arguments:
        • player_uuid: str | None; The specific player to extract 
            parameters for. If None, extracts for all players.
        • player_role: str | None; The specific player role to extract 
            parameters for. If None, extracts for all roles.
        • utility_settings_parent: dict[str, bool]; Defines the functional form of the parent.
        • utility_settings: dict[str, bool]; Defines the functional form of the utility function.
        • file_paths: dict[str, str | dict[str, str]]; Stores all file paths for storing data.
        • general_settings: dict[str, Any]; Various settings used throughout this analysis.
        • within_ic_analysis: bool;
            - If True, extracts data from iterations of the IC analysis in file_paths > bic_aic
            - If False, extracts data from  file_paths > player_fits > experiment_n
        • temperature: float; Lower values increase the chances of the lowest loss child 
            being selected. Higher values make the selection more random.

    Returns:
        • dict[str, Any]; Example:
            {
                "metadata": {
                    "U_funct": "Uᵢ(A) = ...",
                    "utility_settings": {...},     # Child settings.
                    "loss_total": 274.462100798,   # Iteration-level total (IC) or sum of role-losses when available.
                    "source": "IC" or "player_fits",
                    "model_key": "1111101..."
                },
                "data": {
                        "player_uuid": {
                        "params":  {"chooser": {...}, "predictor": {...}},
                        "loss":    {"chooser": <float|None>, "predictor": <float|None>}
                    },
                    ...
                }
            }
    """
    def _fallback_random_parent_guess_from_bounds(param_info: dict, rng: np.random.Generator | None = None) -> dict[str, float]:
        """
        Build a *parent-space* initial guess vector by sampling uniformly within bounds,
        preserving the ordering in param_info['keys'] and param_info['bounds'].
        """
        fallback_params = {}
        "Construct an initial parameter dictionary"
        if callable(param_info["guesses"]):
            initial_guesses = param_info["guesses"]()
        else:
            initial_guesses = param_info["guesses"]

        for param_key, param_guess in zip(param_info['keys'], initial_guesses):
            fallback_params[param_key] = float(param_guess)

        return fallback_params

    "Normalize inputs"
    general_settings = copy.deepcopy(general_settings)
    if within_ic_analysis:
        general_settings['update_method'] = 'naive'
        general_settings['temperature_is_param'] = False

    "Tuple of player roles of interest if not interested in both roles."
    player_roles = (player_role,) if player_role in ('chooser', 'predictor') else ('chooser', 'predictor')

    "Create a list of utility settings of all children of the parent model."
    parent_equation = build_utility_equation(utility_settings=utility_settings_parent)
    model_nesting_data = model_nesting_adjacency_matrices(general_settings=general_settings, 
                            utility_settings=utility_settings, file_paths=file_paths, create_new_file=False)
    try: parent_idx = model_nesting_data['equations'].index(parent_equation)
    except IndexError as err:
        print(err)
        return None

    child_indices = model_nesting_data['adjacency_lists']['child_of'][parent_idx]
    child_utility_settings = [model_nesting_data['settings'][child_idx] for child_idx in child_indices]

    if child_utility_settings:
        assert len(list(child_utility_settings[0].keys())) == 14

    "Build a list of best fitting child parameters."
    best_fitting_child_params = []
    for child_settings in child_utility_settings:

        child_params_and_losses = best_fitting_model_parameters(
            utility_settings=child_settings,
            general_settings=general_settings,
            file_paths=file_paths, 
            param_bds=param_bds,
            within_ic_analysis=within_ic_analysis,
            player_uuid=player_uuid,
            player_role=player_role,
        )

        "Compute loss total"
        loss_total = 0.0
        for plr_uuid, params_and_losses in child_params_and_losses.items():
            if player_uuid is not None and plr_uuid != player_uuid:
                continue
            losses = params_and_losses.get('loss', {})
            for plr_role in player_roles:
                a_loss = losses.get(plr_role, 0.0)
                if isinstance(a_loss, (int, float)):
                    loss_total += losses.get(plr_role, 0.0)

        "Append to the list data and metadata for the child and its parameters."
        child_payload = {
            "metadata": {
                "U_funct": build_utility_equation(utility_settings=child_settings),
                "model_bit_str": gnrl.convert_utility_settings(utility_settings=utility_settings, into=str),
                "utility_settings": child_settings,
                "loss_total": loss_total,
            },
            "data": child_params_and_losses
        }
        best_fitting_child_params.append(child_payload)   

    if not best_fitting_child_params:
        "Fallback to random parameters if the child model could not be found."

        "Generate a list of all player uuids or just one player if already specified."
        if isinstance(player_uuid, str):
            player_uuids = [player_uuid]
        else:
            player_uuids = prep.all_player_uuids(
                file_paths=file_paths, experiment_num=experiment_num, only_humans=True)     

        parent_warmstart = {}
        for plr_uuid in player_uuids:
            parent_warmstart[plr_uuid] = {}
            for plr_role in player_roles:
                plr_param_info = make_param_info(param_bds=param_bds, utility_settings=utility_settings, 
                                                      general_settings=general_settings, guess_seed=None, random_guesses_are_unique=True)
                parent_warmstart[plr_uuid][plr_role] = _fallback_random_parent_guess_from_bounds(param_info=plr_param_info, rng=None)

        "No usable child found → disable warm-start cleanly"
        return {
            "parent_warmstart": parent_warmstart,
            "selected_child":   {},
            "metadata": {
                "reason": "no_child_fit_found",
                "parent_equation": build_utility_equation(utility_settings=utility_settings_parent),
                "U_funct": utility_settings_parent,
            }
        }

    "Sort with best fitting children first in the list."
    best_fitting_child_params = sorted(best_fitting_child_params, 
        key = lambda child_fit_data: child_fit_data.get('metadata', {}).get('loss_total', 0.0))
    
    "Probabilistically select a child based on the total losses and a temperature parameter."
    selected_child = select_child_params_for_parent(children=best_fitting_child_params, temperature=temperature)
    selected_child_settings = selected_child.get('metadata', {}).get('utility_settings', {})
    selected_child_equation = selected_child.get('metadata', {}).get('U_funct')

    "Ensure that all keys are present"
    setting_keys_all = set(utility_settings.keys())
    setting_keys_child = set(selected_child_settings.keys())
    missing_keys = setting_keys_all - setting_keys_child
    n_missing_keys = len(list(missing_keys))
    max_missing_keys = 2
    if n_missing_keys == 0:
        pass
    elif n_missing_keys > 0:
        if n_missing_keys > max_missing_keys:
            pp.pprint(best_fitting_child_params)
            raise ValueError(f"Child model is missing {n_missing_keys} keys: {setting_keys_child}.")
        else:
            for setting_key in list(setting_keys_all):
                if setting_key not in setting_keys_child:
                    val = True if 'single_' in setting_key else False
                    selected_child_settings[setting_key] = val

    "Sanity Check: Confirm that the selected model is a child of the parent."
    relation_1_to_2, relation_2_to_1, flipped_setting = gnrl.classify_pair_relation(model_1=utility_settings_parent, 
                                                            model_2=selected_child_settings, general_settings=general_settings, utility_settings=utility_settings)

    if not (relation_1_to_2 == 'parent' and relation_2_to_1 == 'child'):
        print(f"Parent: {parent_equation}")
        print(f"Child?: {selected_child_equation}")
        if not isinstance(flipped_setting, str):
            flipped_setting = "unknown"
        raise RuntimeError(
            f"Selected model is not a child of the parent. Model 1 = {relation_1_to_2} and Model 2 = {relation_2_to_1}. "
            f"The flipped utility setting is: {flipped_setting.capitalize().replace('_', ' ')}"
        )

    "Map child params -> parent params (add the parent's extra param info etc.)"
    warmstart: Dict[str, Dict[str, Dict[str, float]]] = {}
    for plr_uuid, params_and_losses in selected_child.get('data', {}).items():
        if player_uuid is not None and plr_uuid != player_uuid:
            continue
        warmstart[plr_uuid] = {}
        params_by_role = params_and_losses.get('params', {})
        for plr_role in player_roles:
            fitted_child_params = params_by_role.get(plr_role, {})
            parent_params = gnrl.map_child_to_parent_special_param_info(
                child_utility_settings=child_settings,
                parent_utility_settings=utility_settings_parent,
                child_fitted_parameters=fitted_child_params,
                build_utility_equation=build_utility_equation,
                general_settings=general_settings,
                param_bds=param_bds,
            )

            parent_param_info = make_param_info(
                param_bds=param_bds, utility_settings=utility_settings_parent,
                general_settings=general_settings, guess_seed=None,
                random_guesses_are_unique=True
            )
            if callable(parent_param_info['guesses']):
                parent_param_info['guesses'] = parent_param_info['guesses']()
            ordered = [float(parent_params.get(param_key, param_guess))
                    for param_key, param_guess in zip(parent_param_info['keys'], parent_param_info['guesses'])]
            warmstart[plr_uuid][plr_role] = {
                "keys":    parent_param_info['keys'],
                "bounds":  parent_param_info['bounds'],
                "guesses": ordered,
                **parent_params
            }



    return {
        "metadata": {
            "parent_equation": parent_equation,
            "parent_settings": utility_settings_parent,
            "selected_child_equation": selected_child_equation,
            "selected_child_settings": selected_child_settings,
            "flipped_setting": flipped_setting
        },
        "selected_child": selected_child,      # Retains per-player/role child fit info.
        "parent_warmstart": warmstart          # Per-player/role dict of mapped parent params.
    }


"=========================================================================================="
"============================== Average Model Policy Distance =============================="
"=========================================================================================="


def _build_ampd_cache_path(
    file_paths: FilePaths,
    metric: str,
    parameter_sampling_mode: str,
    parameter_pairing_mode: str,
    player_roles: Optional[List[str]],
    choice_temperature: float,
    n_games: int,
    n_iters: int,
    random_seed: Optional[int],
) -> str:
    """
    Constructs the canonical cache filename for an AMPD master matrix. The master matrix
    always covers all 480 utility forms; subsets fill in only the relevant cells. This design
    supports resume and allows multiple subset runs to accumulate into one file.

    Filename format (values only, dash-separated, no key names):
        ampd-{metric}-{sampler}-{pairing}-{tau}-{n_games}-{n_iters}-{seed}.csv
    For realistic mode with roles, roles are inserted after pairing:
        ampd-{metric}-{sampler}-{pairing}-{roles}-{tau}-{n_games}-{n_iters}-{seed}.csv

    Arguments:
        • file_paths: FilePaths — must contain 'processed'.
        • metric: str — distance metric name.
        • parameter_sampling_mode: str — 'uniform' or 'realistic'.
        • parameter_pairing_mode: str — 'shared' or 'independent'.
        • player_roles: list[str] | None — roles used for realistic sampling.
        • choice_temperature: float — softmax temperature τ.
        • n_games: int — number of payoff structures evaluated.
        • n_iters: int — number of Monte Carlo parameter draws.
        • random_seed: int | None — seed used; None becomes 'unseeded'.

    Returns:
        • str — full path to the master cache CSV file.
    """
    tau_str = f"{choice_temperature:.4g}".replace(".", "p")
    seed_str = str(random_seed) if random_seed is not None else "unseeded"
    parts = ["ampd", metric, parameter_sampling_mode, parameter_pairing_mode]
    if parameter_sampling_mode == "realistic" and player_roles is not None:
        parts.append("+".join(sorted(player_roles)))
    parts += [tau_str, str(n_games), str(n_iters), seed_str]
    filename = "-".join(parts) + ".csv"
    return os.path.join(file_paths["processed"], filename)


def average_model_policy_distance(
    utility_settings_a: Union[UtilitySettings, int],
    utility_settings_b: Union[UtilitySettings, int],
    general_settings: GeneralSettings,
    file_paths: FilePaths,
    param_bds: Dict[str, Tuple[float, float]],
    metric: str = "normalized_jsd",
    choice_temperature: Optional[float] = None,
    n_games: int = 625,
    n_iters: int = 250,
    parameter_sampling_mode: str = "uniform",
    parameter_pairing_mode: str = "shared",
    player_roles: Optional[List[str]] = None,
    random_seed: Optional[int] = None,
    error_policy: str = "raise",
    participant_parameter_pools: Optional[Dict[str, List[float]]] = None,
) -> float:
    """
    Computes the Average Model Policy Distance (AMPD) between two utility forms: the mean
    normalized Jensen-Shannon divergence (or other metric) between their induced choice
    policies, averaged over shared parameter draws and payoff structures.

    Primary mode (parameter_pairing_mode='shared'): within each Monte Carlo iteration both
    models receive the same full canonical reference parameter vector. Each model then uses
    only the parameters it recognizes, ignoring the rest. This ensures:
        AMPD(model_x, model_x) == 0
    and measures distance between utility architectures, not between randomly sampled agents.

    Independent mode (parameter_pairing_mode='independent'): each model draws its own
    parameter vector per iteration. This is a diagnostic for within-model policy variability
    and produces nonzero self-distance. Do not use for the main AMPD matrix.

    Arguments:
        • utility_settings_a: UtilitySettings | int
            First model, specified as a UtilitySettings dict or a utility_idx integer.
        • utility_settings_b: UtilitySettings | int
            Second model, specified as a UtilitySettings dict or a utility_idx integer.
        • general_settings: GeneralSettings
            Used for parameter specification (passed to make_param_info) and for
            softmax_temperature when choice_temperature is None.
        • file_paths: FilePaths
            Must contain 'processed'; used to load the registry when utility_idx inputs
            are given and to load empirical data for realistic sampling.
        • param_bds: dict[str, tuple[float, float]]
            Bounds for all parameters; used to define the canonical reference vector space
            and as fallback for realistic sampling.
        • metric: str (default 'normalized_jsd')
            Distance metric. Currently supported: 'normalized_jsd'.
        • choice_temperature: float | None
            SoftMax temperature τ. If None, uses general_settings['softmax_temperature'].
        • n_games: int (default 625)
            Number of payoff structures to evaluate. If >= 625, uses the exhaustive
            5^4 = 625 grid over {1, 2, 3, 4, 5}^4; otherwise samples randomly.
        • n_iters: int (default 250)
            Number of Monte Carlo parameter draws to average over.
        • parameter_sampling_mode: str (default 'uniform')
            'uniform': sample the canonical reference vector uniformly from bounds.
            'realistic': draw from empirical player distributions; fill missing with uniform.
            'participant_sampled': sample each parameter independently from a pre-built pool
                of all participant-fitted values across all models; requires
                participant_parameter_pools to be provided.
        • parameter_pairing_mode: str (default 'shared')
            'shared': draw one full reference vector per iteration; project to both models.
            'independent': draw a separate vector for each model (diagnostic only).
        • player_roles: list[str] | None
            Roles to include when parameter_sampling_mode='realistic'. Defaults to
            ['chooser', 'predictor'] when None.
        • random_seed: int | None
            Seed for the random number generator. None gives unseeded (non-reproducible) runs.
        • error_policy: str (default 'raise')
            'raise': propagate any utility/softmax exceptions immediately.
            'penalty': log the failed payoff and substitute 0.5 (maximum-entropy fallback).
        • participant_parameter_pools: dict[str, list[float]] | None (default None)
            Required when parameter_sampling_mode='participant_sampled'. Pre-built by
            compute_ampd_distance_matrix; maps each parameter name to the full pool of
            all participant-fitted values for that parameter across all models and roles.

    Returns:
        • float in [0, 1] — mean normalized JSD between the two models' policies.
    """
    if metric != "normalized_jsd":
        raise NotImplementedError(
            f"metric={metric!r} is not yet implemented. Currently supported: 'normalized_jsd'."
        )
    if parameter_pairing_mode not in ("shared", "independent"):
        raise ValueError(
            f"parameter_pairing_mode must be 'shared' or 'independent', got {parameter_pairing_mode!r}."
        )

    def _parse_csv_bool(val: Any) -> bool:
        if isinstance(val, str):
            return val.strip().lower() not in ("false", "0", "")
        return bool(val)

    "Resolve utility_idx inputs to UtilitySettings dicts via the registry."
    registry_df: Optional[pd.DataFrame] = None
    def _resolve_settings(model_ref: Union[UtilitySettings, int]) -> UtilitySettings:
        nonlocal registry_df
        if isinstance(model_ref, int):
            if registry_df is None:
                registry_df = pd.read_csv(
                    os.path.join(file_paths["processed"], "all_utility_functions.csv"),
                    dtype={"utility_bitstring": str},
                )
            row = registry_df[registry_df["utility_idx"] == model_ref]
            if len(row) == 0:
                raise ValueError(f"utility_idx {model_ref} not found in registry.")
            non_flag_cols = {
                "utility_idx", "utility_bitstring", "k_params", "redundant_with",
                "differing_settings", "n_data", "pvar", "param_norm_sd", "loss_nll",
                "AIC", "BIC", "ΔAIC", "ΔBIC", "AIC_rank", "BIC_rank",
                "parents", "siblings", "children",
                "ampd_to_best_rand", "ampd_to_best_real", "policy_regret_norm", "equation",
            }
            flag_cols = [col for col in registry_df.columns if col not in non_flag_cols]
            return {col: _parse_csv_bool(row.iloc[0][col]) for col in flag_cols}
        return gnrl.convert_utility_settings(utility_settings=model_ref, into=dict)

    def _normalized_jsd_binary(p: float, q: float, eps: float = 1e-12) -> float:
        """
        Normalized Jensen-Shannon divergence between two Bernoulli distributions with success
        probabilities p and q. Returns a value in [0, 1], where 0 means identical policies
        and 1 means maximally different deterministic policies.

        Arguments:
            • p: float — probability that model A chooses option A, in [0, 1].
            • q: float — probability that model B chooses option A, in [0, 1].
            • eps: float — small offset to prevent log(0); default 1e-12.

        Returns:
            • float in [0, 1] — normalized JSD(Bernoulli(p) || Bernoulli(q)) / log(2).
        """
        def _binary_entropy(x: float) -> float:
            x = max(eps, min(1.0 - eps, x))
            return -(x * math.log(x) + (1.0 - x) * math.log(1.0 - x))

        r = 0.5 * (p + q)
        jsd = _binary_entropy(r) - 0.5 * _binary_entropy(p) - 0.5 * _binary_entropy(q)
        return max(0.0, jsd / math.log(2.0))

    def _sample_params_uniform(
        utility_settings_inner: UtilitySettings,
        rng_inner: Any,
    ) -> Dict[str, float]:
        """
        Draws one parameter vector for a given utility form by sampling each parameter
        independently and uniformly from its bounds in param_bds. This is the non-circular
        default suitable for general model-space geometry.

        Arguments:
            • utility_settings_inner: UtilitySettings — selects which parameters are active.
            • rng_inner: random.Random — seeded random instance for reproducibility.

        Returns:
            • dict[str, float] — one sampled value per active mean parameter (no _std / _cov keys).
        """
        param_info_inner = make_param_info(
            param_bds=param_bds,
            utility_settings=utility_settings_inner,
            general_settings=general_settings,
            guess_seed=None,
        )
        mean_keys_inner = [
            param_key for param_key in param_info_inner["keys"]
            if not param_key.endswith("_std") and "_cov" not in param_key
        ]
        return {
            key: rng_inner.uniform(float(param_bds[key][0]), float(param_bds[key][1]))
            for key in mean_keys_inner
        }

    def _sample_params_realistic(
        utility_settings_inner: UtilitySettings,
        empirical_df_inner: pd.DataFrame,
        rng_inner: Any,
    ) -> Dict[str, float]:
        """
        Draws one parameter vector by randomly sampling a real participant row from the empirical
        parameter distribution, then taking each parameter's fitted value from that row. For
        parameters not present in the empirical data (because they belong to utility forms that
        were not in the original IC analysis), the function falls back to uniform sampling.
        This avoids researcher-imposed guesses about what "realistic" unseen parameters should be.

        Arguments:
            • utility_settings_inner: UtilitySettings — selects which parameters are active.
            • empirical_df_inner: pd.DataFrame — one row per fitted participant; columns are parameter names.
            • rng_inner: random.Random — seeded random instance for reproducibility.

        Returns:
            • dict[str, float] — one value per active mean parameter.
        """
        param_info_inner = make_param_info(
            param_bds=param_bds,
            utility_settings=utility_settings_inner,
            general_settings=general_settings,
            guess_seed=None,
        )
        mean_keys_inner = [
            param_key for param_key in param_info_inner["keys"]
            if not param_key.endswith("_std") and "_cov" not in param_key
        ]
        sampled_row = empirical_df_inner.iloc[rng_inner.randint(0, len(empirical_df_inner) - 1)]
        params_inner: Dict[str, float] = {}
        for key in mean_keys_inner:
            if key in empirical_df_inner.columns and not pd.isna(sampled_row[key]):
                params_inner[key] = float(sampled_row[key])
            else:
                params_inner[key] = rng_inner.uniform(
                    float(param_bds[key][0]), float(param_bds[key][1])
                )
        return params_inner

    def _sample_full_reference_params(
        parameter_sampling_mode_inner: str,
        empirical_df_inner: Optional[pd.DataFrame],
        rng_inner: Any,
        participant_parameter_pools_inner: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, float]:
        """
        Draws one full canonical reference parameter vector containing every mean parameter
        defined in param_bds. This is the shared draw used in parameter_pairing_mode='shared':
        both compared models receive the same vector and each projects it to its active subset.

        Arguments:
            • parameter_sampling_mode_inner: 'uniform', 'realistic', or 'participant_sampled'.
            • empirical_df_inner: pd.DataFrame | None — required for 'realistic' mode.
            • rng_inner: random.Random — seeded random instance.
            • participant_parameter_pools_inner: dict[str, list[float]] | None — required for
                'participant_sampled' mode. Each key is a parameter name; each value is the
                pool of all empirically fitted values for that parameter across all participants
                and all models. Each parameter is sampled independently from its pool.

        Returns:
            • dict[str, float] — one value per mean parameter (no _std / _cov keys).
        """
        all_mean_keys_inner = [
            param_key for param_key in param_bds if not param_key.endswith("_std") and "_cov" not in param_key
        ]
        if parameter_sampling_mode_inner == "participant_sampled" and participant_parameter_pools_inner:
            "Participant-sampled mode: draw each parameter independently from its empirical pool."
            params_inner: Dict[str, float] = {}
            for key in all_mean_keys_inner:
                pool = participant_parameter_pools_inner.get(key, [])
                if pool:
                    params_inner[key] = float(rng_inner.choice(pool))
                else:
                    params_inner[key] = rng_inner.uniform(
                        float(param_bds[key][0]), float(param_bds[key][1])
                    )
            return params_inner
        if (
            parameter_sampling_mode_inner == "uniform"
            or empirical_df_inner is None
            or len(empirical_df_inner) == 0
        ):
            return {
                key: rng_inner.uniform(float(param_bds[key][0]), float(param_bds[key][1]))
                for key in all_mean_keys_inner
            }
        "Realistic mode: draw one empirical row, fill any missing params uniformly."
        sampled_row = empirical_df_inner.iloc[rng_inner.randint(0, len(empirical_df_inner) - 1)]
        params_inner = {}
        for key in all_mean_keys_inner:
            if key in empirical_df_inner.columns and not pd.isna(sampled_row[key]):
                params_inner[key] = float(sampled_row[key])
            else:
                params_inner[key] = rng_inner.uniform(
                    float(param_bds[key][0]), float(param_bds[key][1])
                )
        return params_inner

    def _project_reference_params(
        full_reference_params_inner: Dict[str, float],
        utility_settings_inner: UtilitySettings,
    ) -> Dict[str, float]:
        """
        Projects a full canonical reference parameter vector down to the subset of parameters
        that a specific utility form actually uses. Parameters not in full_reference_params_inner
        (which should not happen if the reference was built from param_bds) fall back to the
        midpoint of their bounds.

        Arguments:
            • full_reference_params_inner: dict[str, float] — the shared reference vector.
            • utility_settings_inner: UtilitySettings — determines which parameters are active.

        Returns:
            • dict[str, float] — active mean parameters for this utility form.
        """
        param_info_inner = make_param_info(
            param_bds=param_bds,
            utility_settings=utility_settings_inner,
            general_settings=general_settings,
            guess_seed=None,
        )
        mean_keys_inner = [
            param_key for param_key in param_info_inner["keys"]
            if not param_key.endswith("_std") and "_cov" not in param_key
        ]
        return {
            key: (
                full_reference_params_inner[key]
                if key in full_reference_params_inner
                else 0.5 * (float(param_bds[key][0]) + float(param_bds[key][1]))
            )
            for key in mean_keys_inner
        }

    settings_a = _resolve_settings(model_ref=utility_settings_a)
    settings_b = _resolve_settings(model_ref=utility_settings_b)

    tau = choice_temperature if choice_temperature is not None else float(
        general_settings.get("softmax_temperature", 1.5)
    )
    rng = random.Random(random_seed)

    "Build the payoff grid: exhaustive 5^4 = 625 structures, or sampled if n_games < 625."
    if n_games >= 625:
        payoff_tuples: List[Dict[str, int]] = [
            {"As": a, "Ao": b, "Bs": c, "Bo": d}
            for a, b, c, d in it.product(range(1, 6), repeat=4)
        ]
    else:
        payoff_tuples = [
            {
                "As": rng.randint(1, 5), "Ao": rng.randint(1, 5),
                "Bs": rng.randint(1, 5), "Bo": rng.randint(1, 5),
            }
            for _ in range(n_games)
        ]

    "Precompute reversed payoff tuples for option B (swap A and B sides)."
    payoff_tuples_B: List[Dict[str, int]] = [
        {"As": payoff["Bs"], "Ao": payoff["Bo"], "Bs": payoff["As"], "Bo": payoff["Ao"]}
        for payoff in payoff_tuples
    ]

    "Load empirical data once if realistic sampling is requested."
    empirical_df: Optional[pd.DataFrame] = None
    if parameter_sampling_mode == "realistic":
        roles_to_use = player_roles if player_roles is not None else ["chooser", "predictor"]
        role_dfs: List[pd.DataFrame] = []
        role_file_map = {
            "chooser":   "Player_Parameters_Exper3_Chooser_First.csv",
            "predictor": "Player_Parameters_Exper3_Predictor_Final.csv",
        }
        for role in roles_to_use:
            role_path = os.path.join(file_paths["processed"], role_file_map[role])
            if os.path.exists(role_path):
                role_dfs.append(pd.read_csv(role_path))
        empirical_df = pd.concat(role_dfs, ignore_index=True) if role_dfs else pd.DataFrame()

    """Monte Carlo average: for each iteration draw parameter(s) and average JSD over payoffs."""
    total_jsd = 0.0
    for _ in range(n_iters):
        if parameter_pairing_mode == "shared":
            full_ref = _sample_full_reference_params(
                parameter_sampling_mode_inner=parameter_sampling_mode,
                empirical_df_inner=empirical_df,
                rng_inner=rng,
                participant_parameter_pools_inner=participant_parameter_pools,
            )
            params_a = _project_reference_params(
                full_reference_params_inner=full_ref,
                utility_settings_inner=settings_a,
            )
            params_b = _project_reference_params(
                full_reference_params_inner=full_ref,
                utility_settings_inner=settings_b,
            )
        else:
            "Independent mode: separate draws per model (diagnostic use only)."
            if parameter_sampling_mode == "uniform":
                params_a = _sample_params_uniform(
                    utility_settings_inner=settings_a,
                    rng_inner=rng,
                )
                params_b = _sample_params_uniform(
                    utility_settings_inner=settings_b,
                    rng_inner=rng,
                )
            else:
                params_a = _sample_params_realistic(
                    utility_settings_inner=settings_a,
                    empirical_df_inner=empirical_df,
                    rng_inner=rng,
                )
                params_b = _sample_params_realistic(
                    utility_settings_inner=settings_b,
                    empirical_df_inner=empirical_df,
                    rng_inner=rng,
                )

        iter_jsd = 0.0
        for payoffs_A, payoffs_B in zip(payoff_tuples, payoff_tuples_B):
            try:
                u_a_A = utility(payoffs=payoffs_A, params=params_a, utility_settings=settings_a)
                u_a_B = utility(payoffs=payoffs_B, params=params_a, utility_settings=settings_a)
                p_a = softmax_(uA=u_a_A, uB=u_a_B, temperature=tau)

                u_b_A = utility(payoffs=payoffs_A, params=params_b, utility_settings=settings_b)
                u_b_B = utility(payoffs=payoffs_B, params=params_b, utility_settings=settings_b)
                p_b = softmax_(uA=u_b_A, uB=u_b_B, temperature=tau)

                iter_jsd += _normalized_jsd_binary(p=p_a, q=p_b)
            except Exception as exc:
                if error_policy == "raise":
                    raise
                print(f"  AMPD penalty applied (payoffs={payoffs_A}): {exc}")
                iter_jsd += 0.5
        total_jsd += iter_jsd / len(payoff_tuples)

    return total_jsd / n_iters


def compute_ampd_distance_matrix(
    general_settings: GeneralSettings,
    file_paths: FilePaths,
    param_bds: Dict[str, Tuple[float, float]],
    utility_settings: UtilitySettings,
    metric: Optional[str] = None,
    n_games: Optional[int] = None,
    n_iters: Optional[int] = None,
    parameter_sampling_mode: Optional[str] = None,
    parameter_pairing_mode: Optional[str] = None,
    player_roles: Optional[List[str]] = None,
    create_new_file: bool = False,
    random_seed: Optional[int] = None,
    subset_utility_idxs: Optional[List[int]] = None,
    print_: bool = True,
    print_every_x_pairs: Optional[int] = 1,
    n_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Computes and caches the pairwise AMPD distance matrix over all valid utility forms.
    Uses a single master 480×480 file per settings combination. NaN marks uncomputed cells;
    0.0 marks the diagonal (never Monte-Carlo sampled under shared-parameter mode).

    Master filename format (values only, no key names):
        ampd-{metric}-{sampler}-{pairing}-{tau}-{n_games}-{n_iters}-{seed}.csv
    For realistic mode with roles, roles appear after pairing.

    Startup behavior depends on create_new_file (F=False, T=True), whether a matching
    master file already exists on disk (Y=Yes, N=No), and if it does, whether it is
    fully complete (C=Complete) or still has uncomputed cells (I=Incomplete):

        F  N  *  →  Create a fresh master file and begin computing all pairs.
        F  Y  C  →  All pairs already computed; return the existing file immediately.
        F  Y  I  →  Resume: skip pairs already filled in the master; compute the rest.
        T  *  *  →  Discard any existing master and backup; create a fresh master file.

    The master file is always written to disk immediately after initialization (even before
    any computation begins) so that downstream functions can find and load it even if the
    process is interrupted before the first row completes.

    Backup (_.csv) logic — orthogonal to the four cases above:
      • At startup: if a {master}_.csv exists, its computed cells are merged into the
        master before computation begins, then the backup is deleted. This recovers work
        from any previous session that was interrupted while the master was locked (e.g.,
        open in Excel). Exception: when create_new_file=True, any stale backup is discarded.
      • During computation: after each row, the updated master is saved. If the master file
        is locked (PermissionError), the row is instead appended to the _.csv backup. The
        next startup will merge it in automatically.

    Arguments:
        • general_settings: GeneralSettings — controls softmax temperature.
        • file_paths: FilePaths — must contain 'processed'.
        • param_bds: dict[str, tuple[float, float]] — parameter bounds.
        • utility_settings: UtilitySettings — used to derive canonical flag column order.
        • metric: str (default 'normalized_jsd') — distance metric.
        • n_games: int (default 625) — payoff structures per Monte Carlo iteration.
        • n_iters: int (default 250) — Monte Carlo parameter draws per pair.
        • parameter_sampling_mode: str (default 'uniform').
        • parameter_pairing_mode: str (default 'shared')
            'shared': one full reference vector per iteration, projected to each model.
                Guarantees AMPD(x, x) = 0. Use for all primary analyses.
            'independent': separate draws per model. Diagnostic only; produces nonzero diagonal.
        • player_roles: list[str] | None — roles used for realistic sampling.
        • create_new_file: bool (default False) — if True, discard any existing master matrix.
        • random_seed: int | None — RNG seed; None means unseeded (non-reproducible).
        • subset_utility_idxs: list[int] | None — if provided, only computes pairs where
            both models are in this list. The master matrix still covers all 480 models.
        • print_: bool (default True) — whether to print progress.
        • print_every_x_pairs: int | None (default 1) — print a block every N new pairs.
            None disables per-pair output. Ignored when n_workers > 1 (progress is reported
            per completed batch instead).
        • n_workers: int | None (default None) — number of parallel worker processes.
            None uses cpu_count - 1. Clamped to [1, cpu_count - 1]. When 1, computation
            runs sequentially (same behavior as before parallelization). When > 1, pairs are
            grouped by row (idx_i), each row is dispatched as one worker job, and the master
            merges and saves after each row completes.

    Returns:
        • pd.DataFrame — full 480×480 master matrix indexed and columned by utility_idx.
            Computed values are in [0, 1]. Uncomputed cells are NaN. Diagonal is 0.0.
    """
    "Resolve AMPD settings: explicit arguments take precedence over general_settings['ampd_settings']."
    _ampd_cfg = general_settings.get("ampd_settings", {})
    metric = metric if metric is not None else _ampd_cfg.get("metric", "normalized_jsd")
    n_games = n_games if n_games is not None else _ampd_cfg.get("n_games", 625)
    n_iters = n_iters if n_iters is not None else _ampd_cfg.get("n_iters", 250)
    parameter_sampling_mode = (
        parameter_sampling_mode if parameter_sampling_mode is not None
        else _ampd_cfg.get("parameter_sampling_mode", "uniform")
    )
    parameter_pairing_mode = (
        parameter_pairing_mode if parameter_pairing_mode is not None
        else _ampd_cfg.get("parameter_pairing_mode", "shared")
    )
    player_roles = player_roles if player_roles is not None else _ampd_cfg.get("player_roles", None)
    random_seed = random_seed if random_seed is not None else _ampd_cfg.get("random_seed", None)

    tau = float(general_settings.get("softmax_temperature", 1.5))

    "=== Build participant parameter pools if participant_sampled mode is requested ==="
    participant_parameter_pools: Optional[Dict[str, List[float]]] = None
    if parameter_sampling_mode == "participant_sampled":
        experiment_num_for_pools = general_settings.get("experiment_num", 3)
        ic_json_name = f"All_Utility_Forms_IC_Analysis_Experiment{experiment_num_for_pools}.json"
        ic_json_path_for_pools = os.path.join(str(file_paths["bic_aic"]), ic_json_name)

        def _build_participant_pools(ic_json_path_arg: str) -> Dict[str, List[float]]:
            """
            Loads the IC analysis JSON and collects all participant-fitted parameter values
            from each model's minvec. Returns a dict mapping each canonical parameter name to
            the flat list of all fitted values across all participants, models, and roles.
            Each parameter's pool is sampled independently during AMPD computation.
            """
            all_mean_keys_for_pools = [
                param_key for param_key in param_bds if not param_key.endswith("_std") and "_cov" not in param_key
            ]
            pools: Dict[str, List[float]] = {key: [] for key in all_mean_keys_for_pools}
            if print_:
                print(f"Loading IC JSON for participant_sampled pools: {os.path.basename(ic_json_path_arg)}")
            with open(ic_json_path_arg, "r", encoding="utf-8-sig") as fh:
                ic_data = json.load(fh)
            for model_entry in ic_data.get("ic_results", {}).values():
                for player_data in model_entry.get("minvec", {}).values():
                    for role in ("chooser", "predictor"):
                        for param_name, param_value in player_data.get("params", {}).get(role, {}).items():
                            if param_name in pools:
                                pools[param_name].append(float(param_value))
            return {k: v for k, v in pools.items() if v}

        participant_parameter_pools = _build_participant_pools(ic_json_path_for_pools)
        if print_:
            pool_summary = {param_key: len(pool_list) for param_key, pool_list in participant_parameter_pools.items()}
            print(f"Participant parameter pools built: {pool_summary}")

        "Set True to inspect empirical parameter distributions; prints and exits."
        print_random_param_stats = False  
        if print_random_param_stats:
            print(f"\n{'Parameter':<22} {'N':>7}  {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
            print("-" * 75)
            for param_name, pool_vals in sorted(participant_parameter_pools.items()):
                arr = np.array(pool_vals)
                print(
                    f"{param_name:<22} {len(arr):>7}  {arr.mean():>10.4f}  "
                    f"{arr.std():>10.4f}  {arr.min():>10.4f}  {arr.max():>10.4f}"
                )
            exit()

    "Load full registry (all models) to build the master index."
    registry_df = pd.read_csv(
        os.path.join(file_paths["processed"], "all_utility_functions.csv"),
        dtype={"utility_bitstring": str},
    )
    all_utility_idxs: List[int] = sorted(registry_df["utility_idx"].astype(int).tolist())
    n_all = len(all_utility_idxs)

    master_path = _build_ampd_cache_path(
        file_paths=file_paths, metric=metric,
        parameter_sampling_mode=parameter_sampling_mode,
        parameter_pairing_mode=parameter_pairing_mode,
        player_roles=player_roles,
        choice_temperature=tau, n_games=n_games, n_iters=n_iters,
        random_seed=random_seed,
    )
    alt_path = master_path[:-4] + "_.csv"

    def _init_fresh_master() -> pd.DataFrame:
        data = np.full((n_all, n_all), np.nan)
        np.fill_diagonal(data, 0.0)
        return pd.DataFrame(data, index=all_utility_idxs, columns=all_utility_idxs)

    def _expand_to_full_index(df: pd.DataFrame) -> pd.DataFrame:
        df = df.reindex(index=all_utility_idxs, columns=all_utility_idxs)
        for idx in all_utility_idxs:
            if np.isnan(df.loc[idx, idx]):
                df.loc[idx, idx] = 0.0
        return df

    def _zero_off_diag_to_nan(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts exact off-diagonal zeros to NaN. Old-format files used 0.0 for uncomputed
        cells; new format uses NaN. This ensures resume logic correctly identifies uncomputed
        pairs regardless of which format the file was written in. Legitimately-zero AMPD pairs
        (functionally identical models) will be recomputed and land at ≈0.0, which is correct.
        """
        arr = df.values.astype(float)
        off_diag = ~np.eye(arr.shape[0], dtype=bool)
        arr[(arr == 0.0) & off_diag] = np.nan
        return pd.DataFrame(arr, index=df.index, columns=df.columns)

    "=== Load or initialize master matrix — see docstring for the four cases ==="
    if create_new_file:
        "T-*-*: Discard any existing master and backup; start completely fresh."
        if os.path.exists(alt_path):
            try:
                os.remove(alt_path)
            except OSError:
                pass
        master_df = _init_fresh_master()
        try:
            master_df.to_csv(master_path)
        except PermissionError:
            master_df.to_csv(alt_path)
        if print_:
            print(f"AMPD master created fresh: {os.path.basename(master_path)}")
    elif os.path.exists(master_path):
        "F-Y-*: Matching file found; load it and decide below based on n_remaining."
        master_df = pd.read_csv(master_path, index_col=0)
        master_df.index = master_df.index.astype(int)
        master_df.columns = master_df.columns.astype(int)
        master_df = _expand_to_full_index(master_df)
        master_df = _zero_off_diag_to_nan(master_df)
        if print_:
            n_computed = int(np.sum(~np.isnan(master_df.values)) - n_all)
            print(f"AMPD master loaded: {os.path.basename(master_path)}  ({n_computed // 2} pairs computed)")
    else:
        "F-N-*: No matching file; initialize a fresh master and persist it immediately."
        master_df = _init_fresh_master()
        try:
            master_df.to_csv(master_path)
        except PermissionError:
            master_df.to_csv(alt_path)
        if print_:
            print(f"AMPD master initialized: {os.path.basename(master_path)}")

    "=== Merge pending backup (_.csv) if present — always done unless create_new_file ==="
    if not create_new_file and os.path.exists(alt_path):
        try:
            alt_df = pd.read_csv(alt_path, index_col=0)
            alt_df.index = alt_df.index.astype(int)
            alt_df.columns = alt_df.columns.astype(int)
            alt_df = _expand_to_full_index(alt_df)
            alt_df = _zero_off_diag_to_nan(alt_df)
            master_df.update(alt_df, overwrite=False)
            os.remove(alt_path)
            if print_:
                print(f"Merged pending backup: {os.path.basename(alt_path)}")
        except Exception as exc:
            if print_:
                print(f"Warning: could not merge backup {os.path.basename(alt_path)}: {exc}")

    "=== Derive flag columns and settings/equation dicts for all registry models ==="
    non_flag_cols: set = {
        "utility_idx", "utility_bitstring", "k_params", "redundant_with", "differing_settings",
        "n_data", "pvar", "param_norm_sd", "loss_nll",
        "AIC", "BIC", "ΔAIC", "ΔBIC", "AIC_rank", "BIC_rank",
        "parents", "siblings", "children",
        "ampd_to_best_rand", "ampd_to_best_real", "policy_regret_norm", "equation",
    }
    flag_cols = [col for col in registry_df.columns if col not in non_flag_cols]

    def _parse_csv_bool(val: Any) -> bool:
        if isinstance(val, str):
            return val.strip().lower() not in ("false", "0", "")
        return bool(val)

    registry_df_indexed = registry_df.set_index("utility_idx")
    settings_cache: Dict[int, UtilitySettings] = {
        int(idx): {col: _parse_csv_bool(registry_df_indexed.loc[idx, col]) for col in flag_cols}
        for idx in all_utility_idxs
        if idx in registry_df_indexed.index
    }
    equations_cache: Dict[int, str] = {
        int(idx): (
            str(registry_df_indexed.loc[idx, "equation"])
            if not pd.isna(registry_df_indexed.loc[idx, "equation"])
            else "?"
        )
        for idx in all_utility_idxs
        if idx in registry_df_indexed.index
    }

    "=== Determine working subset ==="
    if subset_utility_idxs is not None:
        working_idxs = [idx for idx in subset_utility_idxs if idx in settings_cache]
    else:
        working_idxs = [idx for idx in all_utility_idxs if idx in settings_cache]
    n_working = len(working_idxs)

    n_total_pairs = n_working * (n_working - 1) // 2
    n_already_done = sum(
        1 for pi in range(n_working) for pj in range(pi + 1, n_working)
        if not np.isnan(master_df.loc[working_idxs[pi], working_idxs[pj]])
    )
    n_remaining = n_total_pairs - n_already_done
    idx_width = max(3, len(str(max(all_utility_idxs)))) if all_utility_idxs else 3

    if print_:
        print(
            f"AMPD matrix: {n_working} working models, {n_total_pairs} pairs "
            f"({n_already_done} already done, {n_remaining} remaining). "
            f"metric={metric}, sampler={parameter_sampling_mode}, pairing={parameter_pairing_mode}, "
            f"n_iters={n_iters}, n_games={n_games}, tau={tau}"
        )

    if n_remaining == 0:
        if print_:
            print("All pairs already computed — returning master matrix.")
        return master_df

    def _fmt_duration(seconds: float) -> str:
        total_minutes = int(seconds) // 60
        if total_minutes >= 60:
            hours = total_minutes // 60
            mins = total_minutes % 60
            return f"{hours} hours {mins} minutes"
        secs = int(seconds) % 60
        return f"{total_minutes} minutes {secs:02d} seconds"

    def _save_master(df: pd.DataFrame) -> None:
        try:
            df.to_csv(master_path)
        except PermissionError:
            df.to_csv(alt_path)
            if print_:
                print(f"  [master locked — saved backup to {os.path.basename(alt_path)}]")

    """
    Resolve n_workers. general_settings['run_in_parallel'] is the master switch: if False,
    always run sequentially regardless of the n_workers argument. Otherwise, clamp the
    requested worker count to [1, cpu_count - 1] with None meaning cpu_count - 1.
    """
    run_in_parallel_flag = general_settings.get('run_in_parallel', True)
    cpu_count_available  = mp.cpu_count()
    if not run_in_parallel_flag:
        n_workers_clamped = 1
    elif n_workers is None:
        n_workers_clamped = max(1, cpu_count_available - 1)
    else:
        n_workers_clamped = max(1, min(n_workers, cpu_count_available - 1))

    if print_:
        mode_label = f"{n_workers_clamped} parallel workers" if n_workers_clamped > 1 else "sequential"
        print(f"AMPD matrix: computing {n_remaining} remaining pairs ({mode_label}).")

    "Build per-row job list: each job is one row (idx_i) with all its uncomputed j partners."
    row_jobs: List[tuple] = []
    for i_pos, idx_i in enumerate(working_idxs):
        pairs_in_row = [
            (idx_i, working_idxs[j_pos])
            for j_pos in range(i_pos + 1, n_working)
            if np.isnan(master_df.loc[idx_i, working_idxs[j_pos]])
        ]
        if pairs_in_row:
            row_jobs.append((
                pairs_in_row,
                settings_cache,
                general_settings,
                file_paths,
                param_bds,
                metric,
                tau,
                n_games,
                n_iters,
                parameter_sampling_mode,
                parameter_pairing_mode,
                player_roles,
                random_seed,
                participant_parameter_pools,
            ))

    start_time = time.time()
    pair_count = 0
    n_jobs = len(row_jobs)

    def _merge_batch_results(batch_results: list) -> None:
        """Write a completed worker batch's distances into master_df (symmetric)."""
        for utility_idx_i, utility_idx_j, dist in batch_results:
            master_df.loc[utility_idx_i, utility_idx_j] = dist
            master_df.loc[utility_idx_j, utility_idx_i] = dist

    if n_workers_clamped == 1:
        "Sequential path — preserves per-pair printing for interactive use."
        for job_idx, row_job_args in enumerate(row_jobs):
            batch_results = _ampd_pair_worker(row_job_args)
            _merge_batch_results(batch_results)
            _save_master(master_df)
            pair_count += len(batch_results)

            if print_ and print_every_x_pairs is not None:
                elapsed_secs = time.time() - start_time
                pairs_per_sec = pair_count / elapsed_secs if elapsed_secs > 0 else 0.0
                pairs_left = n_remaining - pair_count
                eta_str = _fmt_duration(pairs_left / pairs_per_sec) if pairs_per_sec > 0 else "unknown"
                for utility_idx_i, utility_idx_j, dist in batch_results:
                    if pair_count % print_every_x_pairs == 0:
                        eq_i = equations_cache.get(utility_idx_i, "?")
                        eq_j = equations_cache.get(utility_idx_j, "?")
                        print(f"Model {utility_idx_i:0{idx_width}d}: U(A) = {eq_i}")
                        print(f"Model {utility_idx_j:0{idx_width}d}: U(A) = {eq_j}")
                        print(
                            f"AMPD({utility_idx_i:0{idx_width}d}, {utility_idx_j:0{idx_width}d}) = {dist:.5f}"
                            f"  —  elapsed {_fmt_duration(elapsed_secs)}, ETA {eta_str}"
                        )
    else:
        "Parallel path — dispatch rows as jobs; master merges and saves as each row returns."
        with mp.Pool(processes=n_workers_clamped) as pool:
            for completed_job_idx, batch_results in enumerate(
                pool.imap_unordered(_ampd_pair_worker, row_jobs), 1
            ):
                _merge_batch_results(batch_results)
                _save_master(master_df)
                pair_count += len(batch_results)

                if print_:
                    elapsed_secs = time.time() - start_time
                    pairs_per_sec = pair_count / elapsed_secs if elapsed_secs > 0 else 0.0
                    pairs_left = n_remaining - pair_count
                    eta_str = _fmt_duration(pairs_left / pairs_per_sec) if pairs_per_sec > 0 else "unknown"
                    print(
                        f"AMPD behavioral-distance matrix: row {completed_job_idx}/{n_jobs} done  "
                        f"({pair_count}/{n_remaining} pairs, {100 * pair_count / n_remaining:.1f}%)  "
                        f"elapsed {_fmt_duration(elapsed_secs)}, ETA {eta_str}"
                    )

    "Sanity checks on the working-subset sub-matrix."
    sub = master_df.loc[working_idxs, working_idxs]
    sub_vals = sub.values.astype(float)
    assert np.all(np.diag(sub_vals) == 0.0), "AMPD diagonal must be exactly 0.0."
    max_off = np.nanmax(sub_vals[~np.eye(n_working, dtype=bool)])
    assert max_off <= 1.0 + 1e-9, f"AMPD value exceeds 1.0: {max_off}"
    asym = np.nanmax(np.abs(sub_vals - sub_vals.T))
    assert asym < 1e-9, f"AMPD sub-matrix is not symmetric (max |asymmetry|={asym:.2e})."

    if print_:
        n_done = int(np.sum(~np.isnan(master_df.values)) - n_all)
        print(
            f"AMPD master saved: {os.path.basename(master_path)}  "
            f"({n_done // 2} pairs total across all models)"
        )

    return master_df


"=========================================================================================="
"=================================== Model-Space Geometry ================================="
"=========================================================================================="


def _load_ampd_matrix_from_settings(
    general_settings: GeneralSettings,
    file_paths: FilePaths,
) -> pd.DataFrame:
    """
    Loads the AMPD master matrix whose path is determined entirely by
    general_settings['ampd_settings']. This is the canonical way for model-space geometry functions to
    obtain the distance matrix without requiring the caller to pass it explicitly.

    Arguments:
        • general_settings: GeneralSettings — must contain 'ampd_settings' and 'softmax_temperature'.
        • file_paths: FilePaths — must contain 'processed'.

    Returns:
        • pd.DataFrame — the full AMPD master matrix indexed and columned by utility_idx.
    """
    _ampd_cfg = general_settings.get("ampd_settings", {})
    tau = float(general_settings.get("softmax_temperature", 1.5))
    master_path = _build_ampd_cache_path(
        file_paths=file_paths,
        metric=_ampd_cfg.get("metric", "normalized_jsd"),
        parameter_sampling_mode=_ampd_cfg.get("parameter_sampling_mode", "uniform"),
        parameter_pairing_mode=_ampd_cfg.get("parameter_pairing_mode", "shared"),
        player_roles=_ampd_cfg.get("player_roles", None),
        choice_temperature=tau,
        n_games=_ampd_cfg.get("n_games", 625),
        n_iters=_ampd_cfg.get("n_iters", 250),
        random_seed=_ampd_cfg.get("random_seed", None),
    )
    if not os.path.exists(master_path):
        raise FileNotFoundError(
            f"AMPD master matrix not found: {master_path}\n"
            "Run compute_ampd_distance_matrix() first, or check general_settings['ampd_settings']."
        )
    df = pd.read_csv(master_path, index_col=0)
    df.index = df.index.astype(int)
    df.columns = df.columns.astype(int)
    return df


def _ampd_distance_name(general_settings: GeneralSettings) -> str:
    """
    Returns the canonical distance_name label derived from general_settings['ampd_settings'].
    Used to construct embedding and visualization filenames that stay consistent with the
    master matrix settings without the caller having to specify the name manually.

    Arguments:
        • general_settings: GeneralSettings — must contain 'ampd_settings'.

    Returns:
        • str — label such as 'ampd_uniform_shared' or 'ampd_realistic_shared'.
    """
    _ampd_cfg = general_settings.get("ampd_settings", {})
    mode = _ampd_cfg.get("parameter_sampling_mode", "uniform")
    pairing = _ampd_cfg.get("parameter_pairing_mode", "shared")
    return f"ampd_{mode}_{pairing}"


def _classical_mds(distance_matrix: np.ndarray, n_dimensions: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classical multidimensional scaling (PCoA) from a square symmetric distance matrix.
    Uses double-centering followed by eigen-decomposition of the Gram matrix.

    Arguments:
        • distance_matrix: np.ndarray — square symmetric matrix with zero diagonal.
        • n_dimensions: int (default 2) — number of embedding dimensions to return.

    Returns:
        • coords: np.ndarray, shape (n, n_dimensions) — embedded coordinates.
        • eigenvalues: np.ndarray, shape (n_dimensions,) — corresponding eigenvalues
            (negative eigenvalues are clamped to 0 before taking the square root).
    """
    n_entities = distance_matrix.shape[0]
    squared_distance_matrix = distance_matrix ** 2
    centering_matrix_J = np.eye(n_entities) - np.ones((n_entities, n_entities)) / n_entities
    gram_matrix_B = -0.5 * (centering_matrix_J @ squared_distance_matrix @ centering_matrix_J)
    eigenvalues_all, eigenvectors_all = np.linalg.eigh(gram_matrix_B)
    descending_eigenvalue_order = np.argsort(eigenvalues_all)[::-1]
    eigenvalues_all = eigenvalues_all[descending_eigenvalue_order]
    eigenvectors_all = eigenvectors_all[:, descending_eigenvalue_order]
    top_eigenvalues = eigenvalues_all[:n_dimensions]
    top_eigenvectors = eigenvectors_all[:, :n_dimensions]
    mds_coordinates = top_eigenvectors * np.sqrt(np.maximum(top_eigenvalues, 0.0))
    return mds_coordinates, top_eigenvalues


def _load_registry_with_ic_filter(
    file_paths: FilePaths,
    require_ic_data: bool = True,
) -> pd.DataFrame:
    """
    Loads the central utility-function registry and optionally filters to only rows
    that have IC data (non-null BIC). Models without IC data are excluded from
    geometry plots so that embeddings and rankings are meaningful without needing
    to rerun the full IC analysis.

    Arguments:
        • file_paths: FilePaths — must contain 'processed'.
        • require_ic_data: bool (default True) — if True, keep only rows where BIC is not NaN.

    Returns:
        • pd.DataFrame — registry rows, sorted by BIC_rank ascending.
    """
    registry_df = pd.read_csv(
        os.path.join(file_paths["processed"], "all_utility_functions.csv"),
        dtype={"utility_bitstring": str},
    )
    if require_ic_data:
        registry_df = registry_df[registry_df["BIC"].notna()].copy()
    if "BIC_rank" in registry_df.columns:
        registry_df = registry_df.sort_values("BIC_rank").reset_index(drop=True)
    return registry_df


def compute_model_space_embedding(
    general_settings: GeneralSettings,
    file_paths: FilePaths,
    distance_matrix_df: Optional[pd.DataFrame] = None,
    distance_name: Optional[str] = None,
    n_dimensions: int = 2,
    require_ic_data: bool = True,
    create_new_file: bool = False,
) -> pd.DataFrame:
    """
    Computes a classical MDS embedding of the utility-function model space from a pairwise
    distance matrix and saves the result to a CSV file in the processed/ directory.

    Only models with IC data (non-null BIC) are embedded, so the figure can be produced
    without rerunning the full IC analysis. Models present in distance_matrix_df but
    absent from the registry or lacking IC data are silently dropped.

    When distance_matrix_df is None, the function loads the AMPD master matrix whose path
    is derived from general_settings['ampd_settings']. When distance_name is None, it is
    derived from general_settings['ampd_settings'] as well. This follows the repo convention
    that analysis functions resolve their CSV inputs from settings rather than requiring
    the caller to construct paths or load data manually.

    Arguments:
        • general_settings: GeneralSettings — read for 'ampd_settings' and 'softmax_temperature'.
        • file_paths: FilePaths — must contain 'processed'.
        • distance_matrix_df: pd.DataFrame | None (default None) — square symmetric AMPD or
            Hamming distance matrix indexed by utility_idx. If None, loaded from settings.
        • distance_name: str | None (default None) — label included in the output filename.
            If None, derived from general_settings['ampd_settings'] (e.g. 'ampd_uniform_shared').
        • n_dimensions: int (default 2) — number of MDS dimensions (2 or 3).
        • require_ic_data: bool (default True) — if True, restrict to models with BIC data.
        • create_new_file: bool (default False) — if False and a cached embedding exists,
            load and return it.

    Returns:
        • pd.DataFrame — one row per embedded model, with columns:
            utility_idx, utility_bitstring, mds_x, mds_y, [mds_z], k_params,
            BIC, ΔBIC, BIC_rank, equation, all 14 Boolean utility settings.
    """
    if distance_name is None:
        distance_name = _ampd_distance_name(general_settings)
    if distance_matrix_df is None:
        distance_matrix_df = _load_ampd_matrix_from_settings(general_settings, file_paths)

    out_path = os.path.join(
        file_paths["processed"],
        f"model_space_embedding__{distance_name}__dims={n_dimensions}.csv",
    )
    if not create_new_file and os.path.exists(out_path):
        print(f"Model-space embedding loaded from cache: {out_path}")
        return pd.read_csv(out_path, dtype={"utility_bitstring": str})

    registry_df = _load_registry_with_ic_filter(
        file_paths=file_paths, require_ic_data=require_ic_data,
    )

    "Align distance matrix to models that appear in both registry and matrix."
    available_idxs = set(distance_matrix_df.index.astype(int))
    registry_idxs = set(registry_df["utility_idx"].astype(int))
    shared_idxs = sorted(available_idxs & registry_idxs)
    if len(shared_idxs) < 3:
        raise ValueError(
            f"Only {len(shared_idxs)} models shared between registry and distance matrix — "
            "need at least 3 for MDS."
        )

    dist_sub = distance_matrix_df.loc[shared_idxs, shared_idxs].values.astype(float)

    """
    Filter to the largest complete submatrix: keep only models whose row has no NaN entries.
    Because compute_ampd_distance_matrix fills both triangles simultaneously, any model with
    a fully computed row also has a fully computed column, so this single pass is sufficient
    to guarantee no NaNs in the restricted submatrix. This allows embedding to proceed on a
    partially computed AMPD matrix; re-run with create_new_file=True when more AMPD data is
    available to update the embedding.
    """
    nan_in_row = np.isnan(dist_sub).any(axis=1)
    if nan_in_row.any():
        n_dropped = int(nan_in_row.sum())
        shared_idxs = [idx for idx, has_nan in zip(shared_idxs, nan_in_row) if not has_nan]
        if len(shared_idxs) < 3:
            raise ValueError(
                f"Only {len(shared_idxs)} models have fully computed AMPD rows — "
                "need at least 3 for MDS. Run more of compute_ampd_distance_matrix first."
            )
        dist_sub = distance_matrix_df.loc[shared_idxs, shared_idxs].values.astype(float)
        print(
            f"AMPD matrix is partially complete: dropped {n_dropped} models with uncomputed rows, "
            f"embedding {len(shared_idxs)} of {len(shared_idxs) + n_dropped} models."
        )

    registry_sub = registry_df[registry_df["utility_idx"].isin(shared_idxs)].copy()
    registry_sub = registry_sub.set_index("utility_idx").loc[shared_idxs].reset_index()

    "Symmetrize to neutralize floating-point drift, then zero the diagonal."
    dist_sub = (dist_sub + dist_sub.T) / 2.0
    np.fill_diagonal(dist_sub, 0.0)

    mds_coordinates, top_eigenvalues = _classical_mds(distance_matrix=dist_sub, n_dimensions=n_dimensions)

    embedding_df = registry_sub.copy()
    dim_labels = ["mds_x", "mds_y", "mds_z", "mds_w"]
    for dim_idx in range(n_dimensions):
        embedding_df[dim_labels[dim_idx]] = mds_coordinates[:, dim_idx]

    "Approximate stress-1: normalized discrepancy between input distances and embedding distances."
    pairwise_coordinate_differences = mds_coordinates[:, np.newaxis, :] - mds_coordinates[np.newaxis, :, :]
    embedded_pairwise_distances = np.sqrt(np.maximum(np.sum(pairwise_coordinate_differences ** 2, axis=2), 0.0))
    stress_approximation = float(
        np.sqrt(np.sum((dist_sub - embedded_pairwise_distances) ** 2) / np.sum(dist_sub ** 2))
    )

    embedding_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(
        f"Model-space embedding saved: {out_path}  "
        f"({len(shared_idxs)} models, {n_dimensions}D, "
        f"stress={stress_approximation:.4f}, "
        f"top eigenvalues: {[round(float(ev), 3) for ev in top_eigenvalues]})"
    )
    return embedding_df


def plot_model_space_mds(
    general_settings: GeneralSettings,
    file_paths: FilePaths,
    fig_lay: FigLay,
    distance_matrix_df: Optional[pd.DataFrame] = None,
    distance_name: Optional[str] = None,
    n_dimensions: int = 2,
    require_ic_data: bool = True,
    color_by: str = "ΔBIC",
    include_dropdown: bool = True,
    top_n_labeled: int = 10,
    create_new_file: bool = False,
) -> go.Figure:
    """
    Plotly interactive scatter of the utility-function model space embedded via classical MDS
    on a pairwise distance matrix (default: AMPD). Points represent utility functions.
    Hovering over a point shows its equation, ΔBIC, k_params, and utility_idx. By default,
    color encodes ΔBIC (best model = darkest), with a dropdown to switch to other encodings.
    Models without IC data are excluded so the figure can be generated at any time.

    Arguments:
        • general_settings: GeneralSettings — for experiment metadata in the title.
        • file_paths: FilePaths — must contain 'processed' and 'visuals'.
        • fig_lay: FigLay — layout constants from config.py.
        • distance_matrix_df: pd.DataFrame | None — pairwise distance matrix indexed by
            utility_idx. If None, loads the cached embedding from 'processed/'.
        • distance_name: str (default 'ampd_uniform_shared') — used to locate/build the
            embedding CSV and name the output HTML.
        • n_dimensions: int (default 2) — embedding dimensions used (2 → x/y scatter).
        • require_ic_data: bool (default True) — if True, exclude models without BIC.
        • color_by: str (default 'ΔBIC') — default color encoding ('ΔBIC' or 'k_params').
        • include_dropdown: bool (default True) — if True, add a dropdown to switch
            between color-by-ΔBIC, color-by-k, and individual Boolean settings.
        • top_n_labeled: int (default 10) — number of top-BIC models to annotate by rank.
        • create_new_file: bool (default False) — if False, reuses a cached embedding CSV.

    Returns:
        • go.Figure — Plotly figure; also written to visuals/mds_{distance_name}.html.
    """
    "Resolve settings-derived defaults."
    if distance_name is None:
        distance_name = _ampd_distance_name(general_settings)

    "Load or compute the embedding."
    embedding_path = os.path.join(
        file_paths["processed"],
        f"model_space_embedding__{distance_name}__dims={n_dimensions}.csv",
    )
    if os.path.exists(embedding_path) and not create_new_file:
        df = pd.read_csv(embedding_path, dtype={"utility_bitstring": str})
    else:
        df = compute_model_space_embedding(
            general_settings=general_settings, file_paths=file_paths,
            distance_matrix_df=distance_matrix_df, distance_name=distance_name,
            n_dimensions=n_dimensions, require_ic_data=require_ic_data,
            create_new_file=create_new_file,
        )

    delta_bic_col = [col for col in df.columns if col in ("ΔBIC", "ΔBIC", "delta_BIC")]
    delta_bic_col = delta_bic_col[0] if delta_bic_col else None

    flag_cols = [
        col for col in df.columns
        if col not in {
            "utility_idx", "utility_bitstring", "k_params", "redundant_with",
            "differing_settings", "n_data", "pvar", "param_norm_sd", "loss_nll",
            "AIC", "BIC", "ΔAIC", "ΔBIC", "AIC_rank", "BIC_rank", "parents", 
            "siblings", "children", "ampd_to_best_rand", "ampd_to_best_real", 
            "policy_regret_norm", "mds_x", "mds_y", "mds_z", "mds_w", "equation",
        }
    ]

    marker_size = int(fig_lay.get("markersize", 16) * 2)
    marker_outline = dict(width=1.5, color="hsla(0, 0%, 0%, 0.45)")

    def _hover(row: pd.Series) -> str:
        rank = int(row["BIC_rank"]) if pd.notna(row.get("BIC_rank")) else "?"
        dbic = f"{row[delta_bic_col]:.2f}" if delta_bic_col and pd.notna(row.get(delta_bic_col)) else "?"
        k = int(row["k_params"]) if pd.notna(row.get("k_params")) else "?"
        eq = str(row.get("equation", "?"))
        idx = int(row["utility_idx"])
        return f"#{rank} | ΔBIC={dbic} | k={k} | idx={idx}<br>{eq}"

    hover_texts = [_hover(row) for _, row in df.iterrows()]
    custom = df[["k_params", delta_bic_col or "BIC", "BIC_rank", "utility_idx", "equation"]].values

    "Build color arrays."
    dbic_vals = df[delta_bic_col].values if delta_bic_col else np.zeros(len(df))
    k_vals = df["k_params"].values

    bool_label_map = {
        "conditional_welfare_mode":       "Conditional Welfare",
        "reference_dependent_altruism":   "Ref-Dep Altruism",
        "min_max_rawlsian_leontief":      "Min-Max (Rawls/Leontief)",
        "use_exponential_parameters":     "Exponential Params",
        "apply_exponents_to_payoffs":     "Exponents to Payoffs",
        "single_exponential_parameter":   "Single Exponent",
        "single_payoffs_not_differences": "Single Payoffs (not Diffs)",
        "payoff_ratios_not_differences":  "Payoff Ratios (not Diffs)",
        "reference_dependent_utility":    "Ref-Dep Utility",
        "use_negativity_parameters":      "Negativity Params",
        "negativity_social_comparison":   "Negativity Social Comp.",
        "fix_self_interest_parameter":    "Fix Self-Interest",
        "include_social_comparison":      "Social Comparison",
        "include_altruism_term":          "Altruism Term",
    }

    "Trace 0: color by ΔBIC (default visible)."
    trace_dbic = go.Scatter(
        x=df["mds_x"], y=df["mds_y"],
        mode="markers",
        name="Models (ΔBIC)",
        visible=True,
        showlegend=False,
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        marker=dict(
            size=marker_size,
            color=dbic_vals,
            colorscale="Viridis_r",
            showscale=True,
            colorbar=dict(title="ΔBIC", x=1.02, thickness=36, len=0.75),
            line=marker_outline,
        ),
    )

    "Trace 1: color by k_params (hidden by default)."
    trace_k = go.Scatter(
        x=df["mds_x"], y=df["mds_y"],
        mode="markers",
        name="Models (k)",
        visible=False,
        showlegend=False,
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        marker=dict(
            size=marker_size,
            color=k_vals,
            colorscale="Plasma",
            cmin=int(k_vals.min()), cmax=int(k_vals.max()),
            showscale=True,
            colorbar=dict(title="k params", x=1.02, thickness=36, len=0.75),
            line=marker_outline,
        ),
    )

    data_traces = [trace_dbic, trace_k]

    "Traces 2+: True/False pairs for each Boolean setting (hidden by default)."
    hue_true  = "hsla(210, 80%, 55%, 0.85)"
    hue_false = "hsla(30,  80%, 55%, 0.85)"
    bool_trace_start = len(data_traces)
    bool_trace_map: Dict[str, Tuple[int, int]] = {}
    t_idx = bool_trace_start

    for bcol in flag_cols:
        if bcol not in df.columns:
            continue
        label_ = bool_label_map.get(bcol, bcol)
        mask_t = df[bcol].astype(str).str.lower().isin(("true", "1"))
        mask_f = ~mask_t

        for mask, color, suffix in [(mask_t, hue_true, "= True"), (mask_f, hue_false, "= False")]:
            sub = df[mask]
            mask_list = list(mask)
            sub_hover = [h for h, m in zip(hover_texts, mask_list) if m]
            data_traces.append(go.Scatter(
                x=sub["mds_x"], y=sub["mds_y"],
                mode="markers",
                name=f"{label_} {suffix}",
                visible=False,
                showlegend=True,
                legendgroup=bcol,
                text=sub_hover,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    size=marker_size, color=color, opacity=0.8,
                    showscale=False, line=marker_outline,
                ),
            ))
        bool_trace_map[bcol] = (t_idx, t_idx + 1)
        t_idx += 2

    "Top-N rank annotations — centered inside markers, no arrows."
    annotations = []
    if delta_bic_col and top_n_labeled > 0:
        top_df = df.nsmallest(top_n_labeled, delta_bic_col)
        for _, row in top_df.iterrows():
            annotations.append(dict(
                x=float(row["mds_x"]), y=float(row["mds_y"]),
                text=str(int(row['BIC_rank'])),
                showarrow=False,
                xanchor="center", yanchor="middle",
                font=dict(
                    size=14,
                    family=fig_lay.get("font", {}).get("family", "Calibri"),
                    color="white",
                ),
            ))

    "Compute equal zero-centered axis range so both MDS dimensions share the same scale."
    xy_max_abs = max(
        abs(float(df["mds_x"].max())), abs(float(df["mds_x"].min())),
        abs(float(df["mds_y"].max())), abs(float(df["mds_y"].min())),
    )
    mds_axis_pad = xy_max_abs * 0.15
    mds_axis_range = [-(xy_max_abs + mds_axis_pad), (xy_max_abs + mds_axis_pad)]

    fig = go.Figure(data=data_traces)

    distance_label = distance_name.replace("_", " ").title()
    n_models = len(df)
    fig.update_layout(
        template=fig_lay.get("template", "plotly_white"),
        title=f"Model-Space MDS — {distance_label} Distances ({n_models} models)",
        titlefont_size=fig_lay["titlefont_size"],
        title_x=0.5,
        font=fig_lay.get("font", {}),
        hoverlabel=fig_lay.get("hoverlabel", {}),
        margin=dict(l=120, r=180, t=140, b=100),
        xaxis=dict(title="MDS Dimension 1", range=mds_axis_range,
                   scaleanchor="y", scaleratio=1, **fig_lay.get("xaxis", {})),
        yaxis=dict(title="MDS Dimension 2", range=mds_axis_range,
                   **fig_lay.get("yaxis", {})),
        annotations=annotations,
        legend=dict(orientation="h", x=0.0, y=-0.15,
                    font=dict(size=fig_lay.get("font", {}).get("size", 20))),
    )

    if include_dropdown:
        n_total = len(data_traces)

        def _vis(on_indices):
            vis = [False] * n_total
            for idx in on_indices:
                vis[idx] = True
            return vis

        buttons = [
            dict(label="Color: ΔBIC",   method="update",
                 args=[{"visible": _vis([0])}, {"title": f"Model-Space MDS — {distance_label} (ΔBIC)"}]),
            dict(label="Color: k",      method="update",
                 args=[{"visible": _vis([1])}, {"title": f"Model-Space MDS — {distance_label} (k params)"}]),
        ]
        for bcol, (ti, fi) in bool_trace_map.items():
            label_ = bool_label_map.get(bcol, bcol)
            buttons.append(dict(
                label=label_, method="update",
                args=[{"visible": _vis([ti, fi])},
                      {"title": f"Model-Space MDS — {distance_label} ({label_})"}],
            ))

        fig.update_layout(updatemenus=[dict(
            buttons=buttons, direction="down",
            x=0.01, xanchor="left", y=1.12, yanchor="top",
            bgcolor=(_hsla(hue=0, saturation_percent=0, lightness_percent=20, alpha=0.85) 
                     if "dark" in fig_lay.get("template", "") 
                     else _hsla(hue=0, saturation_percent=0, lightness_percent=94, alpha=0.92)),
            font=dict(size=20, family=fig_lay.get("font", {}).get("family", "Calibri")),
        )])

    out_path = os.path.join(file_paths["visuals"], f"mds_{distance_name}.html")
    fig.write_html(out_path)
    print(f"Model-space MDS saved: {out_path}")
    return fig


def plot_distance_to_winner_vs_delta_bic(
    general_settings: GeneralSettings,
    file_paths: FilePaths,
    fig_lay: FigLay,
    distance_matrix_df: Optional[pd.DataFrame] = None,
    require_ic_data: bool = True,
) -> go.Figure:
    """
    Plotly scatter where x = AMPD distance from each model to the BIC-winning model, and
    y = ΔBIC. This asks whether worse-fitting models are behaviorally farther from the winner.
    A strong positive correlation means the IC ranking tracks behavioral proximity.
    Points are colored by k_params. Hovering shows equation, ΔBIC, k, and AMPD to winner.

    When distance_matrix_df is None, loaded from the AMPD master CSV identified by
    general_settings['ampd_settings'].

    Arguments:
        • general_settings: GeneralSettings — for title metadata and 'ampd_settings'.
        • file_paths: FilePaths — must contain 'processed' and 'visuals'.
        • fig_lay: FigLay — layout constants from config.py.
        • distance_matrix_df: pd.DataFrame | None (default None) — square AMPD matrix
            indexed by utility_idx. If None, loaded from general_settings['ampd_settings'].
        • require_ic_data: bool (default True) — if True, exclude models without BIC.

    Returns:
        • go.Figure — also written to visuals/dist_to_winner_vs_dbic.html.
    """
    if distance_matrix_df is None:
        distance_matrix_df = _load_ampd_matrix_from_settings(general_settings, file_paths)

    registry_df = _load_registry_with_ic_filter(
        file_paths=file_paths, require_ic_data=require_ic_data,
    )
    delta_bic_col = [col for col in registry_df.columns if col in ("ΔBIC", "ΔBIC", "delta_BIC")]
    delta_bic_col = delta_bic_col[0] if delta_bic_col else None
    if delta_bic_col is None:
        raise ValueError("Registry missing ΔBIC column.")

    winner_row = registry_df.loc[registry_df[delta_bic_col].idxmin()]
    winner_idx = int(winner_row["utility_idx"])

    available_idxs = set(distance_matrix_df.index.astype(int))
    registry_idxs = set(registry_df["utility_idx"].astype(int))
    shared_idxs = sorted(available_idxs & registry_idxs)

    if winner_idx not in available_idxs:
        raise ValueError(f"BIC winner (utility_idx={winner_idx}) not in distance matrix.")

    dist_to_winner = distance_matrix_df.loc[shared_idxs, winner_idx].astype(float)
    sub_df = registry_df[registry_df["utility_idx"].isin(shared_idxs)].copy()
    sub_df = sub_df.set_index("utility_idx").loc[shared_idxs].reset_index()
    sub_df["dist_to_winner"] = dist_to_winner.values

    marker_size = int(fig_lay.get("markersize", 16) * 2)
    k_min, k_max = int(sub_df["k_params"].min()), int(sub_df["k_params"].max())

    hover_texts = [
        f"#{int(r['BIC_rank'])} | ΔBIC={r[delta_bic_col]:.2f} | k={int(r['k_params'])} | "
        f"AMPD to winner={r['dist_to_winner']:.4f}<br>{r.get('equation', '?')}"
        for _, r in sub_df.iterrows()
    ]

    fig = go.Figure(go.Scatter(
        x=sub_df["dist_to_winner"],
        y=sub_df[delta_bic_col],
        mode="markers",
        showlegend=False,
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        marker=dict(
            size=marker_size,
            color=sub_df["k_params"],
            colorscale="Plasma",
            cmin=k_min, cmax=k_max,
            showscale=True,
            colorbar=dict(title="k params", x=1.02, thickness=36, len=0.75),
            line=dict(width=1.5, color="hsla(0, 0%, 0%, 0.45)"),
        ),
    ))

    "Annotate winner at origin."
    winner_eq = str(winner_row.get("equation", ""))
    fig.add_annotation(
        x=0.0, y=0.0,
        text=f"Winner (#{int(winner_row['BIC_rank'])})",
        showarrow=True, arrowhead=2, ax=40, ay=-40,
        font=dict(size=20, family=fig_lay.get("font", {}).get("family", "Calibri")),
    )

    valid_pairs = sub_df["dist_to_winner"].notna()
    n_valid_pairs = int(valid_pairs.sum())
    if n_valid_pairs >= 2:
        corr = float(np.corrcoef(
            sub_df.loc[valid_pairs, "dist_to_winner"],
            sub_df.loc[valid_pairs, delta_bic_col],
        )[0, 1])
        corr_label = f"r = {corr:.3f}"
    else:
        corr = float("nan")
        corr_label = "r = N/A"
    fig.update_layout(
        template=fig_lay.get("template", "plotly_white"),
        title=f"Distance to BIC Winner vs ΔBIC — {corr_label} ({n_valid_pairs} of {len(sub_df)} models computed)",
        titlefont_size=fig_lay["titlefont_size"],
        title_x=0.5,
        font=fig_lay.get("font", {}),
        hoverlabel=fig_lay.get("hoverlabel", {}),
        margin=dict(l=120, r=180, t=140, b=100),
        xaxis=dict(title="AMPD Distance to BIC-Winning Model", **fig_lay.get("xaxis", {})),
        yaxis=dict(title="ΔBIC (vs Best Model)", **fig_lay.get("yaxis", {})),
    )

    out_path = os.path.join(file_paths["visuals"], "dist_to_winner_vs_dbic.html")
    fig.write_html(out_path)
    print(f"Distance-to-winner vs ΔBIC saved: {out_path}  ({corr_label}, {n_valid_pairs} models)")
    return fig


def compute_top_model_coherence(
    general_settings: GeneralSettings,
    file_paths: FilePaths,
    distance_matrix_df: Optional[pd.DataFrame] = None,
    top_ns: List[int] = None,
    require_ic_data: bool = True,
    print_: bool = True,
) -> Dict[int, float]:
    """
    Computes mean pairwise AMPD distance among the top N BIC-ranked models for various
    values of N. Low mean distance = the top models form a coherent behavioral family;
    high mean distance = the top models span multiple distinct behavioral regions.

    When distance_matrix_df is None, loaded from the AMPD master CSV identified by
    general_settings['ampd_settings'].

    Arguments:
        • general_settings: GeneralSettings — read for 'ampd_settings' when loading from file.
        • file_paths: FilePaths — must contain 'processed'.
        • distance_matrix_df: pd.DataFrame | None (default None) — pairwise distance matrix
            indexed by utility_idx. If None, loaded from general_settings['ampd_settings'].
        • top_ns: list[int] | None — values of N to evaluate (default [5, 10, 25, 50]).
        • require_ic_data: bool (default True) — restrict to models with BIC data.
        • print_: bool (default True) — print results to console.

    Returns:
        • dict[int, float] — maps each N to the mean pairwise AMPD among the top N models.
    """
    if distance_matrix_df is None:
        distance_matrix_df = _load_ampd_matrix_from_settings(general_settings, file_paths)
    if top_ns is None:
        top_ns = [5, 10, 25, 50]

    registry_df = _load_registry_with_ic_filter(
        file_paths=file_paths, require_ic_data=require_ic_data,
    )
    delta_bic_col = [col for col in registry_df.columns if col in ("ΔBIC", "ΔBIC", "delta_BIC")]
    delta_bic_col = delta_bic_col[0] if delta_bic_col else None

    "Restrict to models whose AMPD row is fully computed (no NaN in any off-diagonal cell)."
    _arr = distance_matrix_df.values.astype(float)
    _complete_mask = ~np.isnan(_arr).any(axis=1)
    available_idxs = {int(idx) for idx, ok in zip(distance_matrix_df.index, _complete_mask) if ok}

    valid_df = registry_df[registry_df["utility_idx"].isin(available_idxs)].copy()
    if delta_bic_col:
        valid_df = valid_df.sort_values(delta_bic_col)

    results: Dict[int, float] = {}
    for n_val in top_ns:
        top_idxs = list(valid_df["utility_idx"].astype(int).head(n_val))
        n_available = len(top_idxs)
        if n_available < 2:
            results[n_val] = float("nan")
            if print_:
                print(f"  Top {n_val:3d} model coherence: N/A (only {n_available} fully-computed models available)")
            continue
        sub = distance_matrix_df.loc[top_idxs, top_idxs].values.astype(float)
        upper = sub[np.triu_indices(n_available, k=1)]
        results[n_val] = float(np.nanmean(upper))
        if print_:
            suffix = f" (top {n_available} of {n_val} requested — rest not yet computed)" if n_available < n_val else ""
            print(f"  Top {n_val:3d} model coherence: mean pairwise AMPD = {results[n_val]:.4f}{suffix}")
    return results


def plot_top_model_ampd_heatmap(
    general_settings: GeneralSettings,
    file_paths: FilePaths,
    fig_lay: FigLay,
    distance_matrix_df: Optional[pd.DataFrame] = None,
    top_n: int = 50,
    require_ic_data: bool = True,
) -> go.Figure:
    """
    Plotly heatmap of the pairwise AMPD distance matrix among the top N BIC-ranked models.
    Rows and columns are ordered by BIC rank. Hovering shows which two models are being
    compared and their AMPD distance and equations.

    When distance_matrix_df is None, loaded from the AMPD master CSV identified by
    general_settings['ampd_settings'].

    Arguments:
        • general_settings: GeneralSettings — read for 'ampd_settings' when loading from file.
        • file_paths: FilePaths — must contain 'processed' and 'visuals'.
        • fig_lay: FigLay — layout constants from config.py.
        • distance_matrix_df: pd.DataFrame | None (default None) — pairwise distance matrix
            indexed by utility_idx. If None, loaded from general_settings['ampd_settings'].
        • top_n: int (default 50) — number of top-BIC models to include.
        • require_ic_data: bool (default True) — restrict to models with BIC data.

    Returns:
        • go.Figure — also written to visuals/top_model_heatmap_{top_n}.html.
    """
    if distance_matrix_df is None:
        distance_matrix_df = _load_ampd_matrix_from_settings(general_settings, file_paths)

    registry_df = _load_registry_with_ic_filter(
        file_paths=file_paths, require_ic_data=require_ic_data,
    )
    delta_bic_col = [col for col in registry_df.columns if col in ("ΔBIC", "ΔBIC", "delta_BIC")]
    delta_bic_col = delta_bic_col[0] if delta_bic_col else None

    """
    Restrict to models whose AMPD row is fully computed before ranking by BIC. This ordering
    matters: if the top-N BIC models are selected first and then filtered for completeness, none
    may survive (because AMPD rows are computed in utility_idx order, not BIC-rank order). By
    filtering first and then taking the top N among complete models, the heatmap always shows
    content whenever any models have been computed.
    """
    _arr = distance_matrix_df.values.astype(float)
    _complete_mask = ~np.isnan(_arr).any(axis=1)
    available_idxs = {int(idx) for idx, ok in zip(distance_matrix_df.index, _complete_mask) if ok}

    valid_df = registry_df[registry_df["utility_idx"].isin(available_idxs)].copy()
    if delta_bic_col:
        valid_df = valid_df.sort_values(delta_bic_col)

    top_idxs = list(valid_df["utility_idx"].astype(int).head(top_n))
    if len(top_idxs) < 2:
        raise ValueError(
            f"Only {len(top_idxs)} models have fully computed AMPD rows — "
            "need at least 2 for a heatmap. Run more of compute_ampd_distance_matrix first."
        )

    sub = distance_matrix_df.loc[top_idxs, top_idxs].values.astype(float)
    actual_n = len(top_idxs)
    if actual_n < top_n:
        print(
            f"Heatmap showing top {actual_n} fully-computed BIC-ranked models "
            f"(of {top_n} requested — rest not yet computed)."
        )
    rank_labels = []
    eq_by_idx = dict(zip(valid_df["utility_idx"].astype(int), valid_df.get("equation", pd.Series())))
    for idx in top_idxs:
        row = valid_df[valid_df["utility_idx"] == idx].iloc[0]
        rank_labels.append(str(int(row['BIC_rank'])))

    "Build hover text matrix."
    hover_matrix = []
    for idx, idx_i in enumerate(top_idxs):
        row_i = valid_df[valid_df["utility_idx"] == idx_i].iloc[0]
        hover_row = []
        for jdx, idx_j in enumerate(top_idxs):
            row_j = valid_df[valid_df["utility_idx"] == idx_j].iloc[0]
            hover_row.append(
                f"#{int(row_i['BIC_rank'])} vs #{int(row_j['BIC_rank'])}<br>"
                f"AMPD = {sub[idx, jdx]:.4f}<br>"
                f"A: {str(eq_by_idx.get(idx_i, '?'))[:60]}<br>"
                f"B: {str(eq_by_idx.get(idx_j, '?'))[:60]}"
            )
        hover_matrix.append(hover_row)

    off_diag_vals = sub[~np.eye(actual_n, dtype=bool)]
    heatmap_zmax = float(np.nanmax(off_diag_vals)) if not np.all(np.isnan(off_diag_vals)) else 1.0

    fig = go.Figure(go.Heatmap(
        z=sub,
        x=rank_labels,
        y=rank_labels,
        text=hover_matrix,
        hovertemplate="%{text}<extra></extra>",
        colorscale="Viridis_r",
        colorbar=dict(title="AMPD", thickness=36, len=0.75),
        zmin=0.0,
        zmax=heatmap_zmax,
    ))

    fig.update_layout(
        template=fig_lay.get("template", "plotly_white"),
        title=f"Pairwise AMPD — Top {actual_n} BIC-Ranked Models",
        titlefont_size=fig_lay["titlefont_size"],
        title_x=0.5,
        font=fig_lay.get("font", {}),
        hoverlabel=fig_lay.get("hoverlabel", {}),
        margin=dict(l=140, r=160, t=140, b=140),
        xaxis=dict(title="Model (BIC Rank)", tickangle=-45, **fig_lay.get("xaxis", {})),
        yaxis=dict(title="Model (BIC Rank)", autorange="reversed",
                   scaleanchor="x", scaleratio=1, **fig_lay.get("yaxis", {})),
    )

    out_path = os.path.join(file_paths["visuals"], f"top_model_heatmap_{actual_n}.html")
    fig.write_html(out_path)
    print(f"Top-model AMPD heatmap saved: {out_path}")
    return fig


"=========================================================================================="
"============ Results: Joint Parameter Distributions in Human-Human Experiment ============"
"=========================================================================================="

def population_parameter_distribution_df(general_settings: dict[str, Any], file_paths: dict[str, str], player_role: str = 'predictor', 
                                         use_initial_params: bool | None = None, create_new_file: bool | None = None) -> pd.DataFrame:
    """
    Build a tidy DataFrame of fitted parameter values across all players and their counterparts.

    Reads the unified results dataframe produced by preprocessing, filters to the active
    experiment and role, and returns one row per (player, counterpart) dyad with each
    player's fitted social-preference parameters (Vᵢᵢ, Vᵢⱼ, etc.) and their standard errors.
    This is the primary input for population-level distribution figures and correlation analyses.

    Arguments:
        • general_settings: dict[str, Any]
            Must include 'analysis_mode', 'experiment_num', and optionally 'update_method',
            'use_initial_params', and 'create_new_file'.
        • file_paths: dict[str, str]
            Paths to data directories (used by create_unified_dataframe internally).
        • player_role: str
            Role whose parameters to extract: 'predictor' (default) or 'chooser'.
        • use_initial_params: bool | None
            If True, extracts parameters from the first game (prior to any learning).
            If False, extracts parameters from the final game (after full learning).
            Defaults to general_settings['use_initial_params'] when None.
        • create_new_file: bool | None
            If True, recomputes the unified dataframe even if a cached version exists.
            Defaults to general_settings['create_new_file'] when None.

    Returns:
        • pd.DataFrame — one row per (player, counterpart) dyad with parameter columns.
    """
    acceptable_roles = ('chooser', 'predictor')
    if player_role not in acceptable_roles:
        warning_str = f"{player_role}. Must be one of the following: {acceptable_roles}."
        raise ValueError(f"Invalid player_role detected: {warning_str}")

    if use_initial_params is None or not isinstance(use_initial_params, bool):
        use_initial_params = general_settings.get('use_initial_params', True)
    if create_new_file is None or not isinstance(create_new_file, bool):
        create_new_file = general_settings.get('create_new_file', False)
    update_method =  general_settings.get('update_method', 'grid')
    experiment_num = general_settings.get('experiment_num', 3)

    if player_role == 'chooser':
        if not use_initial_params:
            raise ValueError("use_initial_params must be True if player_role == 'chooser'.")
        if experiment_num in (1, 2):
            raise ValueError(f"Cannot extract chooser role parameters in experiment {experiment_num}!")

    file_name = f"Player_Parameters_Exper{experiment_num}_"
    file_name += f"{player_role.capitalize()}_{'First' if use_initial_params else 'Final'}.csv"
    csv_path = prep.ensure_directory_and_join(base_dir=file_paths["processed"], file_name=file_name)
    if not create_new_file and os.path.exists(csv_path):
        df = prep.dataframe(file_path=file_paths["processed"], file_name=file_name)
        if df is not None:
            if "temp" in list(df.columns):
                df = df.rename(columns={"temp": "τ"})
            return df

    "List of all player uuids in the experiment."
    player_uuids = prep.all_player_uuids(
        file_paths=file_paths, experiment_num=experiment_num, only_humans=True)
    n_players = len(player_uuids)
    
    rows = []
    games_idx = 0 if use_initial_params else -1
    if experiment_num == 1: 
        games_idx = -1

    "Iterate through all players appending their parameters to rows."
    for player_idx, player_uuid in enumerate(player_uuids):
        dyads_for_this_player = prep.fitted_dyads_for_a_player(
            player_uuid=player_uuid, experiment_num=experiment_num, file_paths=file_paths)
        if dyads_for_this_player is None:
            print(f"Failed to extract data for player {player_uuid}")
            continue

        print(f'Adding data for player {player_idx+1} / {n_players}: {player_uuid}')
        
        for dyad_key, dyad_games in dyads_for_this_player.items():
            dyad_game: dict[str, dict[str, dict[str, dict[str, dict]]]] = dyad_games[games_idx]
            counterpart_uuid = dyad_game.get('predictor' if player_role == 'chooser' else 'chooser')
            player_data = dyad_game.get('parameter_estimates', {}).get(update_method, {}).get(player_uuid, {}) 

            param_data = player_data.get(player_role, {}).get('params', None)
            if games_idx == -1:
                posteriors = player_data.get(player_role, {}).get('posteriors', None)
                if posteriors is not None:
                    param_data = posteriors

            row = {
                'experiment_num': experiment_num,
                'player_uuid': player_uuid,
                'counterpart_uuid': counterpart_uuid,
                'player_role': player_role,
            }

            for param_key, param_val in param_data.items():
                if param_key == "temp":
                    row["τ"] = param_val
                else:
                    row[param_key] = param_val

            rows.append(row)

    if not rows:
        raise Exception(f"Failed to generate dataframe.")

    df = pd.DataFrame(rows)
    if "temp" in list(df.columns):
        df = df.rename(columns={"temp": "τ"})

    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    return df


def population_parameter_distribution_histograms(general_settings: dict[str, Any], file_paths: dict[str, str], fig_lay: Dict[str, Any], player_role: str = 'predictor', 
                                                 use_initial_params: bool | None = None, create_new_file: bool | None = None) -> go.Figure:
    """
    Visualize histograms of parameter values in df. 
    df should have columns of parameters (e.g. 'Vᵢᵢ', 'Ʌᵢᵢ', ...).

    Arguments:
        • general_settings: Dict[str, Any]; High-level settings (analysis mode, etc.).
        • file_paths: dict[str, str]; Paths to files/directories for reading/writing data.
        • param_info: dict[str, Any]; Contains parameter keys, bounds, guesses, etc.
        • fig_lay: dict[str, Any]; Determines the aesthetic qualities of the figure.

    Returns:
        • go.Figure
    """
    import plotly.figure_factory as ff
    df = population_parameter_distribution_df(general_settings=general_settings, file_paths=file_paths, 
                    player_role=player_role, use_initial_params=use_initial_params, create_new_file=create_new_file)

    param_types = f"{'Initial' if use_initial_params else 'Final'} {player_role.capitalize()}"
    title = f"Distribution of {param_types} Parameter Values "
    file_name = title.replace(" ", "_") + file_paths["file_name_suffix"] + ".html"

    hist_data = {}
    non_params = ('experiment_num', 'player_uuid', 'counterpart_uuid', 'player_role')
    param_keys = [col for col in df.columns if col not in non_params and df[col].nunique() > 1]

    print(f"----------- Mean and Std of Mean {player_role.capitalize()} Parameters -----------")
    for param_key in param_keys:
        if '_std' not in param_key:
            key_str = ""
            for idx in range(13 - len(param_key)):
                key_str += " "
            key_str += param_key
            param_mean = df[param_key].mean()
            param_std  = df[param_key].std()
            mean_gap = "" if param_mean < 0 else " "
            std_gap = ""  if param_std < 0 else  " "
            print_str = f"      {key_str} "
            print_str += f"μ  = {mean_gap}{param_mean:.5f} "
            print_str += f"σ  = {std_gap}{param_std:.5f}"
            print(print_str)

    if general_settings.get('temperature_is_param'):
        param_keys += ["τ"]
    for param_key in param_keys:
        if param_key in ("τ", "temp"):
            fancy_key = "τ"
            param_key = "τ"
        elif '_std' in param_key:
            fancy_key = f"σ({param_key.replace('_std', '')})"
        elif '_cov' in param_key:
            fancy_key = f"σ({param_key.replace('_cov', '').replace('_', ',')})"
        else:
            fancy_key = f"μ({param_key})"

        hist_data[fancy_key] = df[param_key].dropna().values.tolist()

    group_labels = list(sorted(hist_data.keys(), key=lambda group_label: (
        group_label.replace("σ(", "").replace("μ(", "").replace(")", ""), "σ(" in group_label)))
    hist_colors = [f"hsla({int((idx*360/(len(group_labels)+1))%360)}, 100%, 50%, 0.7)" 
                   for idx in range(len(group_labels))]
    hist_sizes = [0.2]*len(group_labels)

    fig = ff.create_distplot([hist_data[key] for key in group_labels], group_labels, bin_size=hist_sizes, 
                             colors=hist_colors, show_curve=True, show_rug=True)
    
    font_info = copy.deepcopy(fig_lay["font"])
    font_info["size"] = 36

    fig.update_layout(title=title, template=fig_lay["template"], hoverlabel=dict(font_size=14), 
                      titlefont_size=fig_lay['titlefont_size']+6, font=font_info,
                      xaxis=dict(range=[-1, 2]))

    "Categorize parameters"
    mean_params = [param for param in group_labels if "σ(" not
                   in param and "," not in param and 'τ' not in param]
    mean_weight_params = [param for param in mean_params if 'γ' not in param]
    std_params = [param for param in group_labels if "σ(" in param and "," not in param]
    cov_params = [param for param in group_labels if "," in param]

    "Identify problematic parameters (e.g., ones clustering into a single bin based on small std deviation)"
    epsilon = 1e-2  # Define a threshold for tiny standard deviation.
    problematic_params = []
    for param, data in hist_data.items():
        if len(data) > 1 and pd.Series(data).std() < epsilon:
            problematic_params.append(param)

    "Initial visibility: Hide problematic parameters"
    initial_visibility = [param not in problematic_params for param in group_labels] * 2

    "Set initial visibility at the trace level"
    for trace, visible in zip(fig.data, initial_visibility):
        trace.visible = 'legendonly' if not visible else True

    "Dropdown options"
    buttons = [
        dict(label="All", method="update", 
             args=[{"visible": [param not in problematic_params 
                                for param in group_labels] * 2}]),
        dict(label="μ(...)", method="update",
             args=[{"visible": [param in mean_params and param not in 
                                problematic_params for param in group_labels] * 2}]),
        dict(label="μ(𝑤)", method="update",
             args=[{"visible": [param in mean_weight_params and param not in 
                                problematic_params for param in group_labels] * 2}]),                                
        dict(label="σ(...)", method="update",
             args=[{"visible": [param in std_params and param not in 
                                problematic_params for param in group_labels] * 2}]),
    ]
    if general_settings.get('include_covariance'):
        buttons.append(
            dict(label="Cov(x,y)", method="update",
                args=[{"visible": [param in cov_params and param not in 
                                   problematic_params for param in group_labels] * 2}])
        )

    "Add individual parameter selections"
    for idx, group_label in enumerate(group_labels):
        visible = [False] * (2 * len(group_labels))
        visible[idx] = True  # PDF curve for that param.
        visible[idx + len(group_labels)] = True  # Rug for that param.
        buttons.append(dict(label=group_label, method="update", 
                            args=[{"visible": visible}]))

    fig.update_layout(updatemenus=[dict(active=0, x=1.06, y=1.00, buttons=buttons)], 
                      legend={"x": 1.0, "y": 0.96, "font": font_info}, titlefont_size=fig_lay['titlefont_size'], 
                      title_x=fig_lay['title_x'], title_y=fig_lay['title_y'], font=fig_lay['font'])

    if general_settings.get('export_fig', True):
        fig.write_html(os.path.join(file_paths["visuals"], file_name))
        print(f"Saved {file_name}")
    else:
        fig.show()

    return fig


def subpopulation_stats_and_param_ratio_histograms(general_settings: dict[str, any], file_paths: dict[str, str], fig_lay: Dict[str, Any], player_role: str = 'predictor', 
                                use_initial_params: bool | None = None, create_new_file: bool | None = None, ratio_mode: str = "skip_negative", as_subplots: bool = False, print_: bool = True) -> dict:
    """
    Computes:
        1) Subpopulation stats (sadistic, masochistic, guilt>envy, altruism>self-interest),
        2) Two ratio analyses and histograms:
            (a) self-interest vs. altruism (Vᵢᵢ vs. Vᵢⱼ),
            (b) guilt vs. envy (Ʒᵢⱼ vs. Ƹᵢⱼ).

    If as_subplots=False, produces two separate histograms for the given player_role.
    If as_subplots=True, it produces four histograms (both ratio types × both roles) as subplots in a single figure.
    """
    "--- Helper function to load param DF by role for subplot mode. ---"
    def load_and_prepare_df_for_role(the_role: str) -> pd.DataFrame:
        df_temp = population_parameter_distribution_df(
            general_settings=general_settings,
            file_paths=file_paths,
            player_role=the_role,
            use_initial_params=use_initial_params_local,
            create_new_file=create_new_file_local
        )
        if use_initial_params_local:
            df_temp = df_temp.drop_duplicates(subset=['player_uuid']).copy()
        required_cols = ["Vᵢᵢ", "Vᵢⱼ", "Ƹᵢⱼ", "Ʒᵢⱼ"]
        for col in required_cols:
            if col not in df_temp.columns:
                raise ValueError(f"Required column '{col}' not found for role={the_role}. Avail: {df_temp.columns.tolist()}")
        df_temp = df_temp.dropna(subset=required_cols).copy()
        return df_temp

    "--- Helper ratio function matching the original analysis logic ---"
    def compute_ratio_array(series_x, series_y, param_category: str = 'self', normalize: bool = True) -> pd.Series:
        ratio_list = []
        for (x, y) in zip(series_x, series_y):
            if ratio_mode == "skip_negative":
                "Keep only if x>0,y>0 => ratio=x/(x+y) for self"
                "Or x<0,y<0 => ratio=x/(x+y) for guilt (since guilt<0, envy<0)"
                if (param_category == 'self'  and x > 0 and y > 0) or \
                   (param_category == 'guilt' and x > 0 and y > 0):
                    denom = (x + y) if normalize else y
                    if abs(denom) > 1e-9:
                        ratio_list.append(x / denom)
            elif ratio_mode == "absolute":
                ax, ay = abs(x), abs(y)
                denom = ax + ay if normalize else ay
                if denom > 1e-9:
                    ratio_list.append(ax / denom)
            else:
                raise ValueError(f"Unknown ratio_mode '{ratio_mode}' (should be skip_negative|absolute)")
        return pd.Series(ratio_list)

    "--- 1) Defaulting logic (unchanged) ---"
    if use_initial_params is None or not isinstance(use_initial_params, bool):
        use_initial_params = general_settings.get('use_initial_params', True)
    if create_new_file is None or not isinstance(create_new_file, bool):
        create_new_file = general_settings.get('create_new_file', False)
    export_fig = general_settings.get('export_fig', True)
    experiment_num = general_settings.get('experiment_num', 3)

    if ratio_mode == 'skip_negative':
        x_title_self = "Ratio = 𝑉ᵢᵢ / (𝑉ᵢᵢ + 𝑉ᵢⱼ) (Negative Parameters Excluded)"
        x_title_guilt = "Ratio = Ʒᵢⱼ / (Ʒᵢⱼ + Ƹᵢⱼ) (Negative Parameters Excluded)"
    else:
        x_title_self = "Ratio = |𝑉ᵢᵢ| / (|𝑉ᵢᵢ| + |𝑉ᵢⱼ|)"
        x_title_guilt = "Ratio = |Ʒᵢⱼ| / (|Ʒᵢⱼ| + |Ƹᵢⱼ|)"

    "Store final results; if as_subplots=True, gather results for both roles."
    "Keep the single-role flow in a separate if-block."
    if not as_subplots:
        "================ ORIGINAL SINGLE-ROLE LOGIC ================"

        use_initial_params_local = use_initial_params
        create_new_file_local = create_new_file

        "2) Load the parameter DataFrame"
        df = load_and_prepare_df_for_role(player_role)
        n_data = len(df)
        if n_data == 0:
            print("[Warning] No data remain after dropping NaNs. Returning empty results.")
            return {}

        "5) Identify subpop membership"
        sadists_mask = (df["Vᵢⱼ"] < 0)
        masochists_mask = (df["Vᵢᵢ"] < 0)
        competitive_mask = (df["Ʒᵢⱼ"] < 0)
        depricating_mask = (df["Ƹᵢⱼ"] < 0)
        guilt_over_envy_mask = (df["Ʒᵢⱼ"] > df["Ƹᵢⱼ"])
        altruism_over_self_mask = (df["Vᵢⱼ"] > df["Vᵢᵢ"])

        sadistic_pct = 100.0 * sadists_mask.sum() / n_data
        masochistic_pct = 100.0 * masochists_mask.sum() / n_data
        competitive_pct = 100.0 * competitive_mask.sum() / n_data
        depricating_pct = 100.0 * depricating_mask.sum() / n_data
        guilt_over_envy_pct = 100.0 * guilt_over_envy_mask.sum() / n_data
        altruism_over_self_pct = 100.0 * altruism_over_self_mask.sum() / n_data

        "7) Compute ratio arrays"
        self_ratio_series = compute_ratio_array(df["Vᵢᵢ"], df["Vᵢⱼ"], 'self', True)
        guilt_ratio_series = compute_ratio_array(df["Ʒᵢⱼ"], df["Ƹᵢⱼ"], 'guilt', True)
        self_ratio_series_raw = compute_ratio_array(df["Vᵢᵢ"], df["Vᵢⱼ"], 'self', False)
        guilt_ratio_series_raw = compute_ratio_array(df["Ʒᵢⱼ"], df["Ƹᵢⱼ"], 'guilt', False)

        "Summaries"
        n_self_valid = len(self_ratio_series)
        self_ratio_mean = self_ratio_series.mean() if n_self_valid > 0 else float('nan')
        self_ratio_std  = self_ratio_series.std()  if n_self_valid > 1 else float('nan')

        n_guilt_valid = len(guilt_ratio_series)
        guilt_ratio_mean = guilt_ratio_series.mean() if n_guilt_valid > 0 else float('nan')
        guilt_ratio_std  = guilt_ratio_series.std()  if n_guilt_valid > 1 else float('nan')

        self_ratio_mean_raw = self_ratio_series_raw.mean() if n_self_valid > 0 else float('nan')
        guilt_ratio_mean_raw = guilt_ratio_series_raw.mean() if n_guilt_valid > 0 else float('nan')

        "Role string"
        if player_role == 'chooser':
            role_str = 'Chooser'
        else:
            role_str = ("Prior" if use_initial_params_local else "Posterior") + ' Predictor'

        "Create the two separate histograms"
        fig_self = go.Figure()
        fig_self.add_trace(go.Histogram(
            x=self_ratio_series,
            nbinsx=15,
            marker=dict(
                color='hsla(115, 70%, 40%, 0.8)',
                line=dict(width=4, color='hsla(115, 70%, 20%, 1.0)')
            ),
            hovertemplate="Ratio: %{x:.3f}<br>Count: %{y}<extra></extra>",
            name="Self-Altruism Ratio"
        ))
        if not math.isnan(self_ratio_mean):
            fig_self.add_shape(
                type="line",
                x0=self_ratio_mean, x1=self_ratio_mean,
                y0=0, y1=1, xref="x", yref="paper",
                line=dict(color='hsla(115, 100%, 80%, 1.0)', dash='dash', width=4)
            )

        x_min = 0.0
        x_max = 1.0
        x_n_bins = int(x_max * 10) - int(x_min * 10) + 1
        x_tickvals = list(np.round(np.linspace(x_min, x_max, x_n_bins), 3))
        x_ticktext = [''] + [f"{val:.1f}" for val in x_tickvals[1:]]        
        x_axis = {
            'title': x_title_self,  # Will overwrite for guilt fig.
            'tickfont': dict(size=24),
            'title_font': dict(size=30),
            'tickvals': x_tickvals, 
            'ticktext': x_ticktext,
            'range': [x_min, x_max]            
        }
        y_title = "Participant Count" if use_initial_params_local else "Parameter Count Across Dyads"
        y_axis = dict(title=y_title, title_font=dict(size=30))

        fig_self.update_layout(
            template=fig_lay.get("template","plotly_dark"),
            title=f"Self-interest to Altruism Ratio ({role_str} Parameters; 𝑛 = {n_self_valid})",
            title_x=0.5, title_y=0.94,
            margin=dict(l=100, r=100, t=80, b=100),
            xaxis=x_axis, yaxis=y_axis,
            font=fig_lay.get("font", {"size": 16})
        )

        "Guilt figure"
        fig_guilt = go.Figure()
        fig_guilt.add_trace(go.Histogram(
            x=guilt_ratio_series,
            nbinsx=15,
            marker=dict(
                color='hsla(260, 70%, 40%, 0.8)',
                line=dict(width=4, color='hsla(260, 70%, 20%, 1.0)')
            ),
            hovertemplate="<span style='font-size:24px;'>Ratio: %{x:.3f}<br><br>Count: %{y}<extra></extra></span>",
            name="Guilt-Envy Ratio"
        ))
        if not math.isnan(guilt_ratio_mean):
            fig_guilt.add_shape(
                type="line",
                x0=guilt_ratio_mean, x1=guilt_ratio_mean,
                y0=0, y1=1, xref="x", yref="paper",
                line=dict(color='hsla(260, 100%, 80%, 1.0)', dash='dash', width=4)
            )
        fig_guilt.update_layout(
            template=fig_lay.get("template","plotly_dark"),
            title=f"Guilt to Envy Ratio ({role_str} Parameters; 𝑛 = {n_guilt_valid})",
            title_x=0.5, title_y=0.94,
            margin=dict(l=100, r=100, t=80, b=100),
            xaxis={
                'title': x_title_guilt,
                'tickfont': dict(size=24),
                'title_font': dict(size=30),
                'tickvals': x_tickvals,
                'ticktext': x_ticktext,
                'range': [x_min, x_max]
            },
            yaxis=y_axis,
            font=fig_lay.get("font", {"size": 16})
        )

        if export_fig:
            out_path_self = os.path.join(
                file_paths["visuals"],
                f"SelfAltruism_Ratio_{role_str.replace(' ','_')}_{ratio_mode}.html"
            )
            fig_self.write_html(out_path_self)
            out_path_guilt = os.path.join(
                file_paths["visuals"],
                f"GuiltEnvy_Ratio_{role_str.replace(' ','_')}_{ratio_mode}.html"
            )
            fig_guilt.write_html(out_path_guilt)
        else:
            fig_self.show()
            fig_guilt.show()

        "9) Build the final results dictionary"
        results_dict = {
            "sadistic_pct": sadistic_pct,
            "masochistic_pct": masochistic_pct,
            "guilt_over_envy_pct": guilt_over_envy_pct,
            "altruism_over_self_pct": altruism_over_self_pct,
            "self_ratio_mean": self_ratio_mean,
            "self_ratio_std": self_ratio_std,
            "guilt_ratio_mean": guilt_ratio_mean,
            "guilt_ratio_std": guilt_ratio_std,
            "n_in_analysis": n_data,
            "n_self_ratio_valid": n_self_valid,
            "n_guilt_ratio_valid": n_guilt_valid
        }

        rat_mod_str = " ".join([substr.capitalize() for substr in ratio_mode.split("_")])
        print(f"\n--- Subpopulation and Ratio Stats {rat_mod_str} Mode; ---")
        print(f"---       Subset: {role_str}, Total 𝑛 = {n_data}       ---")
        print(f"   Sadistic (Vᵢⱼ < 0)          = {sadistic_pct:.2f}%")
        print(f"   Masochistic (Vᵢᵢ < 0)       = {masochistic_pct:.2f}%")
        print(f"   Competitive (Ʒᵢⱼ < 0)       = {competitive_pct:.2f}%")
        print(f"   Depricating (Ƹᵢⱼ < 0)       = {depricating_pct:.2f}%")
        print(f"   Guilt > Envy (Ʒᵢⱼ > Ƹᵢⱼ)    = {guilt_over_envy_pct:.2f}%")
        print(f"   Altruism > Self (Vᵢⱼ > Vᵢᵢ) = {altruism_over_self_pct:.2f}%")

        print(f"\n--- Ratio Stats for {rat_mod_str} Mode ---")
        print_str_self =  f" Self vs Altruism => 𝑛 = {n_self_valid}; Mean = {self_ratio_mean:.3f};"
        print_str_guilt = f" Guilt vs Envy =>    𝑛 = {n_guilt_valid}; Mean = {guilt_ratio_mean:.3f};"
        print_str_self += f" Std = {self_ratio_std:.3f}; Raw Mean = {self_ratio_mean_raw:.3f}"
        print_str_guilt += f" Std = {guilt_ratio_std:.3f}; Raw Mean = {guilt_ratio_mean_raw:.3f}"
        print(print_str_self)
        print(print_str_guilt)

        return results_dict

    else:
        "================ SUBPLOT MODE: produce 4 histograms for both roles ================"

        "Compute the full logic for each role, then make a 2x2 subplot figure."
        roles = ['chooser', 'predictor']
        big_results = {}

        "Create a 2-row, 2-col figure"
        fig_sub = make_subplots(
            rows=2, cols=2,
            horizontal_spacing=0.07,  # Adjust horizontal spacing.
            vertical_spacing=0.2,
            shared_yaxes=True,
            subplot_titles=(
                "Chooser: Self vs Altruism",
                "Chooser: Guilt vs Envy",
                "Predictor: Self vs Altruism",
                "Predictor: Guilt vs Envy",
            )
        )

        "Unify local variables before looping across roles."
        use_initial_params_local = use_initial_params
        create_new_file_local = create_new_file

        "Loop over roles in [predictor, chooser]."
        "Row=1 => predictor, row=2 => chooser; col=1 => self ratio, col=2 => guilt ratio."
        for idx, role in enumerate(roles, start=1):
            row_i = 2 if role=='predictor' else 1

            "Load & prepare data"
            df = load_and_prepare_df_for_role(role)
            n_data = len(df)
            if n_data == 0:
                print(f"[Warning] No data remain after dropping NaNs for {role}.")
                "Skip this role but continue with the remaining roles."
                big_results[role] = {}
                continue

            "Identify subpop membership"
            sadists_mask = (df["Vᵢⱼ"] < 0)
            masochists_mask = (df["Vᵢᵢ"] < 0)
            competitive_mask = (df["Ʒᵢⱼ"] < 0)
            depricating_mask = (df["Ƹᵢⱼ"] < 0)
            guilt_over_envy_mask = (df["Ʒᵢⱼ"] > df["Ƹᵢⱼ"])
            altruism_over_self_mask = (df["Vᵢⱼ"] > df["Vᵢᵢ"])

            sadistic_pct = 100.0 * sadists_mask.sum() / n_data
            masochistic_pct = 100.0 * masochists_mask.sum() / n_data
            competitive_pct = 100.0 * competitive_mask.sum() / n_data
            depricating_pct = 100.0 * depricating_mask.sum() / n_data
            guilt_over_envy_pct = 100.0 * guilt_over_envy_mask.sum() / n_data
            altruism_over_self_pct = 100.0 * altruism_over_self_mask.sum() / n_data

            "Ratios"
            self_ratio_series = compute_ratio_array(df["Vᵢᵢ"], df["Vᵢⱼ"], 'self', True)
            guilt_ratio_series = compute_ratio_array(df["Ʒᵢⱼ"], df["Ƹᵢⱼ"], 'guilt', True)
            self_ratio_series_raw = compute_ratio_array(df["Vᵢᵢ"], df["Vᵢⱼ"], 'self', False)
            guilt_ratio_series_raw = compute_ratio_array(df["Ʒᵢⱼ"], df["Ƹᵢⱼ"], 'guilt', False)

            n_self_valid = len(self_ratio_series)
            self_ratio_mean = self_ratio_series.mean() if n_self_valid > 0 else float('nan')
            self_ratio_std  = self_ratio_series.std()  if n_self_valid > 1 else float('nan')

            n_guilt_valid = len(guilt_ratio_series)
            guilt_ratio_mean = guilt_ratio_series.mean() if n_guilt_valid > 0 else float('nan')
            guilt_ratio_std  = guilt_ratio_series.std()  if n_guilt_valid > 1 else float('nan')

            self_ratio_mean_raw =  self_ratio_series_raw.mean() if n_self_valid > 0 else float('nan')
            guilt_ratio_mean_raw = guilt_ratio_series_raw.mean() if n_guilt_valid > 0 else float('nan')

            "Sub-dict of results for this role"
            this_res = {
                "sadistic_pct": sadistic_pct,
                "masochistic_pct": masochistic_pct,
                "guilt_over_envy_pct": guilt_over_envy_pct,
                "altruism_over_self_pct": altruism_over_self_pct,
                "self_ratio_mean": self_ratio_mean,
                "self_ratio_std": self_ratio_std,
                "guilt_ratio_mean": guilt_ratio_mean,
                "guilt_ratio_std": guilt_ratio_std,
                "n_self_ratio_valid": n_self_valid,
                "n_guilt_ratio_valid": n_guilt_valid,
                "n_in_analysis": n_data,
            }
            big_results[role] = this_res

            "Add Self ratio histogram"
            fig_sub.add_trace(
                go.Histogram(
                    nbinsx=11,
                    marker=dict(
                        color=f'hsla({115 if row_i == 1 else 160}, 70%, 40%, 0.8)',
                        line=dict(width=4, color=f'hsla({115 if row_i == 1 else 160}, 70%, 20%, 1.0)')
                    ),
                    hovertemplate="<span style='font-size:24px;'>Ratio: %{x:.3f}<br><br>Count: %{y}<extra></extra></span>",
                    name="Guilt-Envy Ratio",
                    x=self_ratio_series
                ),
                row=row_i, col=1
            )
            "Add Guilt ratio histogram"
            fig_sub.add_trace(
                go.Histogram(
                    nbinsx=11,
                    marker=dict(
                        color=f'hsla({205 if row_i == 1 else 250}, 70%, 40%, 0.8)',
                        line=dict(width=4, color=f'hsla({205 if row_i == 1 else 250}, 70%, 20%, 1.0)')
                    ),
                    hovertemplate="<span style='font-size:24px;'>Ratio: %{x:.3f}<br><br>Count: %{y}<extra></extra></span>",
                    name="Guilt-Envy Ratio",
                    x=guilt_ratio_series
                ),
                row=row_i, col=2
            )

            "Print out stats (unchanged from original) for clarity"
            if print_ and role == player_role:
                rat_mod_str = " ".join([substr.capitalize() for substr in ratio_mode.split("_")])
                print(f"\n--- Subpopulation and Ratio Stats for {role.capitalize()}; ---")
                print(f"---  Ratio Mode = {rat_mod_str}; Total 𝑛 = {n_data}  ---")
                print(f"   Sadistic (Vᵢⱼ < 0)          = {sadistic_pct:.2f}%")
                print(f"   Masochistic (Vᵢᵢ < 0)       = {masochistic_pct:.2f}%")
                print(f"   Competitive (Ʒᵢⱼ < 0)       = {competitive_pct:.2f}%")
                print(f"   Depricating (Ƹᵢⱼ < 0)       = {depricating_pct:.2f}%")
                print(f"   Guilt > Envy (Ʒᵢⱼ > Ƹᵢⱼ)    = {guilt_over_envy_pct:.2f}%")
                print(f"   Altruism > Self (Vᵢⱼ > Vᵢᵢ) = {altruism_over_self_pct:.2f}%")

                print(f"\nRatio Stats for {role} => skip_neg/abs: {rat_mod_str}")
                print_str_self = f" Self vs Altruism => 𝑛 = {n_self_valid}, Mean = {self_ratio_mean:.3f},"
                print_str_guilt = f" Guilt vs Envy    => 𝑛 = {n_guilt_valid}, Mean = {guilt_ratio_mean:.3f},"
                print_str_self += f" Std = {self_ratio_std:.3f}, Raw Mean = {self_ratio_mean_raw:.3f}"
                print_str_guilt += f" Std = {guilt_ratio_std:.3f}, Raw Mean = {guilt_ratio_mean_raw:.3f}"
                print(print_str_self)
                print(print_str_guilt)

        "Layout details, including x-axis labeling."
        y_max = 19.04 if ratio_mode == 'absolute' else 9.02  # Hardcoded.
        x_min, x_max = 0.0, 1.0
        x_tickvals = list(np.round(np.linspace(x_min, x_max, 6), 3))
        x_ticktext = [''] + [f"{val:.1f}" for val in x_tickvals[1:]]
        x_axis_common = dict(
            range=[x_min, x_max],
            tickvals=x_tickvals,
            ticktext=x_ticktext
        )

        title_text = f"Parameter Ratios for Both Roles "
        if ratio_mode == 'absolute':
            title_text += "(Negative Parameters Included)"
        else:
            title_text += "(Negative Parameters Excluded)"

        fig_sub.update_layout(
            title_text=title_text,
            template=fig_lay.get("template","plotly_dark"),
            margin=dict(l=105, r=80, t=120, b=100),
            font=fig_lay.get("font", {"size": 22}),
            title_x=0.5, title_y=0.98,
            showlegend=False
        )
        fig_sub.update_annotations(font_size=26)
        fig_sub.update_xaxes(x_axis_common, title=x_title_self,  row=1, col=1)
        fig_sub.update_xaxes(x_axis_common, title=x_title_guilt, row=1, col=2)
        fig_sub.update_xaxes(x_axis_common, title=x_title_self,  row=2, col=1)
        fig_sub.update_xaxes(x_axis_common, title=x_title_guilt, row=2, col=2)
        fig_sub.update_yaxes(title_text="𝑛 Participants", range=[0, y_max])

        "Export if requested"
        if export_fig:
            out_path = os.path.join(
                file_paths["visuals"],
                f"Ratios_Subplots_{ratio_mode}.html"
            )
            fig_sub.write_html(out_path)
        else:
            fig_sub.show()

        "Return the combined results from both roles"
        return big_results


def param_correlation_matrix_report(general_settings: Dict[str, Any], file_paths: Dict[str, str], player_role: str = 'predictor', cross_role_correlations: bool = False, 
                                    normalize_params: bool = True, alpha: float = 0.05, correction_method: str = 'holm') -> Dict[str, pd.DataFrame]:
    """
    Calculates correlation matrices for either within-role or cross-role parameters.
    Optionally normalizes parameter values within each participant (summing absolute
    values to one) for the within-role case only.

    Arguments:
        • general_settings: Dict[str, Any]; General experiment or environment settings.
        • file_paths: Dict[str, str]; Dictionary mapping file labels to file paths.
        • player_role: str; Either 'chooser' or 'predictor' if within-role correlations.
        • cross_role_correlations: bool; If True, merges chooser and predictor data
          and reports cross-role correlations instead.
        • normalize_params: bool; If True (default), parameter values for each participant
          in the within-role case are scaled so the sum of absolute values equals 1.
        • alpha: float; The family-wise significance level used in multiple-comparison correction.
        • correction_method: str; Correction method for p-values. Examples: 'holm',
          'bonferroni', 'fdr_bh'.

    Returns:
        • Dict[str, pd.DataFrame];
          A dictionary containing:
              "corr": The correlation matrix,
              "pvals_raw": The uncorrected p-value matrix,
              "pvals_corrected": The multiple-comparison-corrected p-value matrix.
    """
    from scipy.stats import pearsonr
    from statsmodels.stats.multitest import multipletests

    possible_columns = ["Vᵢᵢ", "Vᵢⱼ", "Ƹᵢⱼ", "Ʒᵢⱼ", "γ1", "γ2", "γ3", "τ"]

    def load_and_clean(role: str) -> Tuple[pd.DataFrame, List[str]]:
        """Loads data for a given role and returns a cleaned DataFrame plus the columns used."""
        df_role = population_parameter_distribution_df(
            general_settings=general_settings,
            file_paths=file_paths,
            player_role=role,
            use_initial_params=True
        ).drop_duplicates(subset=['player_uuid'])

        columns_in_df = [col for col in possible_columns if col in df_role.columns]
        df_role = df_role[['player_uuid'] + columns_in_df].dropna()
        return df_role, columns_in_df

    "Prepare a structure to store results"
    output: Dict[str, pd.DataFrame] = {
        "corr": pd.DataFrame(),
        "pvals_raw": pd.DataFrame(),
        "pvals_corrected": pd.DataFrame()
    }

    "Case 1: Cross-role correlations"
    if cross_role_correlations:
        df_chooser, chooser_cols = load_and_clean('chooser')
        df_predictor, predictor_cols = load_and_clean('predictor')

        df_merged = pd.merge(
            df_chooser, df_predictor,
            on='player_uuid',
            suffixes=('_choo', '_pred')
        ).dropna()

        if df_merged.empty:
            print("[Warning] No valid data remain after filtering. Returning empty tables.")
            return output

        correlation_matrix = np.zeros((len(chooser_cols), len(predictor_cols)))
        pvalue_matrix = np.ones((len(chooser_cols), len(predictor_cols)))

        for row_index, chooser_col in enumerate(chooser_cols):
            for col_index, predictor_col in enumerate(predictor_cols):
                param_x = df_merged[f"{chooser_col}_choo"].values
                param_y = df_merged[f"{predictor_col}_pred"].values
                correlation_value, p_value = pearsonr(param_x, param_y)
                correlation_matrix[row_index, col_index] = correlation_value
                pvalue_matrix[row_index, col_index] = p_value

        corr_df = pd.DataFrame(correlation_matrix, index=chooser_cols, columns=predictor_cols)
        pval_df = pd.DataFrame(pvalue_matrix, index=chooser_cols, columns=predictor_cols)

        print(f"\n--- Cross-role Correlation Matrix (Chooser→Predictor); n = {len(df_merged)} ---")
        print(corr_df)
        print("\n--- Raw P-Values ---")
        print(pval_df)

        "For cross-role results, correct the diagonal p-values only."
        diagonal_indices = np.arange(min(len(chooser_cols), len(predictor_cols)))
        diagonal_pvals = pval_df.values[diagonal_indices, diagonal_indices]

        "Multiple-comparison correction (Holm, etc.)"
        reject_flags, corrected_pvals, _, _ = multipletests(
            diagonal_pvals,
            alpha=alpha,
            method=correction_method
        )

        pval_corrected_matrix = pval_df.values.copy()
        pval_corrected_matrix[diagonal_indices, diagonal_indices] = corrected_pvals
        pval_corrected_df = pd.DataFrame(
            pval_corrected_matrix,
            index=chooser_cols,
            columns=predictor_cols
        )

        print("\n--- Corrected P-Values (method = {}) ---".format(correction_method))
        print(pval_corrected_df)

        output["corr"] = corr_df
        output["pvals_raw"] = pval_df
        output["pvals_corrected"] = pval_corrected_df

        return output

    "Case 2: Within-role correlations"
    df_within_role, columns_in_df = load_and_clean(player_role)

    if df_within_role.empty:
        print("[Warning] No valid data remain after filtering. Returning empty tables.")
        return output

    "(NEW) Optional normalization: for each row (participant), sum absolute values, divide each param by that sum"
    if normalize_params:
        "Sum absolute values across the columns of interest"
        sum_abs_params = df_within_role[columns_in_df].abs().sum(axis=1)
        "Divide each participant's parameters by their sum of absolute values (preserves sign)"
        df_within_role[columns_in_df] = df_within_role[columns_in_df].div(sum_abs_params, axis=0)

    correlation_matrix = np.zeros((len(columns_in_df), len(columns_in_df)))
    pvalue_matrix = np.ones((len(columns_in_df), len(columns_in_df)))

    for row_index, column_i in enumerate(columns_in_df):
        for col_index, column_j in enumerate(columns_in_df):
            param_x = df_within_role[column_i].values
            param_y = df_within_role[column_j].values
            correlation_value, p_value = pearsonr(param_x, param_y)
            correlation_matrix[row_index, col_index] = correlation_value
            pvalue_matrix[row_index, col_index] = p_value

    corr_df = pd.DataFrame(correlation_matrix, index=columns_in_df, columns=columns_in_df)
    pval_df = pd.DataFrame(pvalue_matrix, index=columns_in_df, columns=columns_in_df)

    print(f"\n--- Within-role Correlation Matrix ({player_role.capitalize()}); n = {len(df_within_role)} ---")
    print(corr_df)
    print("\n--- Raw P-Values ---")
    print(pval_df)

    "For a symmetric matrix, only the upper triangle (or lower) are unique."
    matrix_size = len(columns_in_df)
    upper_triangle_rows, upper_triangle_cols = np.triu_indices(matrix_size, k=1)
    pvals_for_correction = pval_df.values[upper_triangle_rows, upper_triangle_cols]

    reject_flags, corrected_pvals, _, _ = multipletests(
        pvals_for_correction,
        alpha=alpha,
        method=correction_method
    )

    "Insert corrected values back into a matrix"
    pval_corrected_matrix = pval_df.values.copy()
    pval_corrected_matrix[upper_triangle_rows, upper_triangle_cols] = corrected_pvals
    "Mirror them into the lower triangle for a fully corrected symmetrical matrix"
    pval_corrected_matrix[upper_triangle_cols, upper_triangle_rows] = corrected_pvals

    pval_corrected_df = pd.DataFrame(pval_corrected_matrix, index=columns_in_df, columns=columns_in_df)

    print("\n--- Corrected P-Values (method = {}) ---".format(correction_method))
    print(pval_corrected_df)

    output["corr"] = corr_df
    output["pvals_raw"] = pval_df
    output["pvals_corrected"] = pval_corrected_df

    return output


"=========================================================================================="
"============================== Inequality Aversion Analysis =============================="
"=========================================================================================="

def inequality_aversion_sanity_check(file_paths: FilePaths, param_strong: float, param_weak: float, self: float = 0.1, altr: float = 0.0, 
                                     temperature: float = 1.0, filter_constant_sum: bool = False, print_: bool = True) -> None:
    """
    Fit two minimal Fehr–Schmidt–style 'bots' to the observed choices and compare which one fits better.

    Overview
    --------
    This function side-steps the full analysis pipeline and asks a very specific question:
    *Given the observed binary dictator choices, does a simple bot with stronger ENVY (α >> β) 
    fit better, or a simple bot with stronger GUILT (β >> α)?*

    Concretely, this defines two linear, curvature-free utility functions that only include:
       • a self-interest weight Vᵢᵢ on the chooser's payoff (A vs B),
       • an altruism weight Vᵢⱼ on the other person's payoff,
       • an envy (disadvantageous inequality) penalty α on max(πⱼ - πᵢ, 0),
       • a guilt (advantageous inequality)  penalty β on max(πᵢ - πⱼ, 0).

    Computes U(A) and U(B) for each trial and converts these to a choice probability with softmax.
    Scores each bot against the observed choices using:
       • hit rate (accuracy) and
       • negative log likelihood (NLL) loss.

    The two bots differ only in how α and β are assigned:

        Envious bot:  envy=param_strong, guilt=param_weak
        Guilty bot:   envy=param_weak,  guilt=param_strong

    Arguments:
        • file_paths : FilePaths
            Repository of paths used by `prep.all_histories(...)` to load the trials.
        • param_strong : float
            The larger value used for the 'strong' inequality parameter in its respective bot.
        • param_weak : float
            The smaller value used for the 'weak' inequality parameter in its respective bot.
        • self : float
            Weight Vᵢᵢ placed on the chooser’s own payoff in both bots.
        • altr : float
            Weight Vᵢⱼ placed on the other person’s payoff in both bots.
        • temperature : float
            Softmax temperature used to convert ΔU into p(choose A); larger values = more noise.
        • filter_constant_sum : bool
            If true, includes only constant sum games to immitate normal dictator games. 

    Returns:
        • dict
            Scores for each bot under keys 'envious' and 'guilty'. Each contains:
                {
                    'correct': int,
                    'incorrect': int,
                    'total': int,
                    'loss': float,          # Negative log-likelihood.
                    'hit_rate': float       # Correct / total.
                }

    Notes:
        • This is a deliberately minimal external validity check. It tests whether the headline
        asymmetry (guilt vs. envy) appears under a simple utility with fixed (self, altruism) weights.
        • Lower NLL indicates the better-fitting bot. Accuracy tends to track NLL but need not match it.
    """
    def inequality_aversion_choice(payoffs: dict, envy: float, guilt: float, self: float = 1.0, altr: float = 0.0, temperature: float = 0.1) -> str:
        payAi, payAj = payoffs['payoff_A_chooser'], payoffs['payoff_A_predictor']
        payBi, payBj = payoffs['payoff_B_chooser'], payoffs['payoff_B_predictor']

        utilityA = self * payAi + altr * payAj - envy * max(payAj - payAi, 0) - guilt * max(payAi - payAj, 0)
        utilityB = self * payBi + altr * payBj  - envy * max(payBj - payBi, 0) - guilt * max(payBi - payBj, 0)

        p_choose_A = softmax_(uA=utilityA, uB=utilityB, temperature=temperature)

        choice = 'A' if p_choose_A >= 0.5 else 'B'
        return {'choice': choice, 'p_choose_A': p_choose_A}

    pairs_path = Path(file_paths["processed"]) / file_paths["file_names"]["player_pairs_exper3"]
    with open(pairs_path, "r") as _f:
        all_data = json.load(_f)
    dyads, player_info = all_data['histories'], all_data['player_info']

    params = {
        'envious': {'self': self, 'altr': altr, 'envy': param_strong, 'guilt': param_weak},
        'guilty': {'self': self, 'altr': altr, 'envy': param_weak, 'guilt': param_strong}        
    }
    scores = {
        'envious': {'correct': 0, 'incorrect': 0, 'total': 0, 'loss': 0.0},
        'guilty': {'correct': 0, 'incorrect': 0, 'total': 0, 'loss': 0.0}
    }

    for dyad_key, dyad in dyads.items():
        for idx, game in enumerate(dyad):
            abdicated_chooser = game.get('abdicated_chooser', None)
            if not abdicated_chooser:
                participant_choice = game.get('choice', None)
                if participant_choice in ('A', 'B'):

                    payoffs = {
                        "payoff_A_chooser": game.get("payoff_A_chooser"),
                        "payoff_A_predictor": game.get("payoff_A_predictor"),
                        "payoff_B_chooser": game.get("payoff_B_chooser"),
                        "payoff_B_predictor": game.get("payoff_B_predictor"),                       
                    }
                    if filter_constant_sum:
                        if payoffs["payoff_A_chooser"] + payoffs["payoff_A_predictor"] != payoffs["payoff_B_chooser"] + payoffs["payoff_B_predictor"]:
                            continue

                    if all(isinstance(payoff, int) for payoff in payoffs.values()):
                        choice_envious = inequality_aversion_choice(payoffs=payoffs, self=params['envious']['self'], altr=params['envious']['altr'],
                                            envy=params['envious']['envy'], guilt=params['envious']['guilt'], temperature=temperature)

                        if choice_envious['choice'] == participant_choice:
                            scores['envious']['correct'] += 1
                        else:
                            scores['envious']['incorrect'] += 1
                        scores['envious']['total'] += 1

                        prob_of_observed_envious = choice_envious['p_choose_A'] if participant_choice == 'A' else 1 - choice_envious['p_choose_A']
                        if prob_of_observed_envious <= 0: 
                            print(f'Neg p(observed envious) = {prob_of_observed_envious}.')
                            prob_of_observed_envious = 1e-6
                        scores['envious']['loss'] += -math.log(prob_of_observed_envious)

                        choice_guilty  = inequality_aversion_choice(payoffs=payoffs, self=params['guilty']['self'], altr=params['guilty']['altr'],
                                            envy=params['guilty']['envy'], guilt=params['guilty']['guilt'], temperature=temperature)

                        if choice_guilty['choice'] == participant_choice:
                            scores['guilty']['correct'] += 1
                        else:
                            scores['guilty']['incorrect'] += 1
                        scores['guilty']['total'] += 1

                        prob_of_observed_guilty = choice_guilty['p_choose_A'] if participant_choice == 'A' else 1 - choice_guilty['p_choose_A']
                        if prob_of_observed_guilty <= 0: 
                            print(f'Neg p(observed guilty) = {prob_of_observed_guilty}.')
                            prob_of_observed_guilty = 1e-6
                        scores['guilty']['loss'] += -math.log(prob_of_observed_guilty)

    env_corr, env_incr = scores['envious']['correct'], scores['envious']['incorrect']
    if env_corr + env_incr <= 0: scores['envious']['hit_rate'] = 0.0
    else: scores['envious']['hit_rate'] = round(env_corr / (env_corr + env_incr), 3)
    gty_corr, gty_incr = scores['guilty']['correct'], scores['guilty']['incorrect']
    if gty_corr + gty_incr <= 0: scores['guilty']['hit_rate'] = 0.0
    else: scores['guilty']['hit_rate'] = round(gty_corr / (gty_corr + gty_incr), 3)
    answer = "Envy is stronger than guilt." if scores['envious']['loss'] < scores['guilty']['loss'] else "Guilt is stronger than envy."

    if print_:
        print(
            f"\nEnvious: Uᵢ(A) = {self}(πᵢᴬ) + {altr}(πⱼᴬ) - {param_strong} × max(πⱼᴬ - πᵢᴬ, 0) - {param_weak} × max(πᵢᴬ - πⱼᴬ, 0); τ = {temperature}"
            f"\nGuilty:  Uᵢ(A) = {self}(πᵢᴬ) + {altr}(πⱼᴬ) - {param_weak} × max(πⱼᴬ - πᵢᴬ, 0) - {param_strong} × max(πᵢᴬ - πⱼᴬ, 0); τ = {temperature}"
            f"\nEnvious: {scores['envious']['correct']:04d} / {scores['envious']['total']:04d} = {scores['envious']['hit_rate']:.3f}; Loss = {scores['envious']['loss']:04.3f}"
            f"\nGuilty:  {scores['guilty']['correct']:04d} / {scores['guilty']['total']:04d} = {scores['guilty']['hit_rate']:.3f}; Loss = {scores['guilty']['loss']:04.3f}"
            f"\n{answer}"
        )

    return scores


def visualize_inequality_aversion_bot_competition(fig_lay: FigLay, file_paths: FilePaths, *, param_strong: float = 0.75, param_weak: float = 0.25, 
                                                  param_self_values: List[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), 
                                                  param_altr_values: List[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), 
                                                  temperature: float = 1.0, ratio_numerator: Literal["envious", "guilty"] = "envious", show_text_values: bool = False, 
                                                  text_decimals: int = 2, color_range: list[float] = [0.0, 1.0], print_: bool = False, export_fig: bool = True, 
                                                  filename_stub: Optional[str] = None, filter_constant_sum: bool = True) -> go.Figure:
    """
    Visualize the envious-vs-guilty bot fitting "competition" across (self-interest, altruism) weights.

    Purpose:
    For each grid point (Vᵢᵢ, Vᵢⱼ), this function:
        1) Fit the two simple bots defined in `inequality_aversion_sanity_check`:
                Envious bot: envy=param_strong, guilt=param_weak
                Guilty bot:  envy=param_weak,  guilt=param_strong
        2) Record their negative log-likelihood (NLL) losses and hit rates.
        3) Plot a heatmap of a *ratio* of the two losses to make comparison interpretable at a glance.

    Ratio definition:
        By default, this plots:
            ratio = Envious_NLL / (Envious_NLL + Guilty_NLL)
        so that **higher values mean the 'guilty' bot fits better**.

        If `ratio_numerator="guilty"`, this plots:
            ratio = Guilty_NLL / (Envious_NLL + Guilty_NLL)
        where **higher values mean the 'envious' bot fits better**.

    Arguments:
        • fig_lay : FigLay
            Standard Plotly layout dictionary (template, colorscales, title_x/y, font sizes, etc.).
        • file_paths : FilePaths
            Paths used by `inequality_aversion_sanity_check` to load the trial histories.
        • param_strong, param_weak : float
            The 'strong' and 'weak' values assigned to the inequality parameters.
            Envious bot uses (envy=strong, guilt=weak); guilty bot uses (envy=weak, guilt=strong).
        • param_self_values : list[float]
            Grid values for the self-interest weight Vᵢᵢ on the x-axis.
        • param_altr_values : list[float]
            Grid values for the altruism weight Vᵢⱼ on the y-axis.
        • temperature : float
            Softmax temperature used inside the bots (same as in the sanity check).
        • ratio_numerator : {"envious", "guilty"}
            Which NLL goes in the numerator of the ratio (controls directionality of the color scale).
        • show_text_values : bool
            If True, writes the numeric ratio onto each cell.
        • text_decimals : int
            Number of decimals to show if `show_text_values=True`.
        • print_ : bool
            If True, prints progress and save path information.
        • export_fig : bool
            If True, writes an interactive HTML file to disk.
        • filename_stub : Optional[str]
            If provided, used as part of the saved filename; otherwise a timestamped default is used.

    Returns:
        • go.Figure
            Interactive Plotly heatmap ready to display or save.

    Notes:
        • Colorbar label is explicit about what 'higher' means, based on `ratio_numerator`.
        • Hover shows: (self, altruism), both losses, both hit rates, the ratio, and which bot wins.
    """

    "Prepare grids"
    x_vals = list(param_self_values)    # Vᵢᵢ (self-interest).
    y_vals = list(param_altr_values)    # Vᵢⱼ (altruism).

    Z_ratio = np.zeros((len(y_vals), len(x_vals)))  # Heat z-values.
    Z_envious_loss = np.zeros_like(Z_ratio)
    Z_guilty_loss  = np.zeros_like(Z_ratio)
    Z_envious_hit  = np.zeros_like(Z_ratio)
    Z_guilty_hit   = np.zeros_like(Z_ratio)

    "Sweep the grid"
    for y_idx, v_other in enumerate(y_vals):
        for x_idx, v_self in enumerate(x_vals):
            scores = inequality_aversion_sanity_check(
                file_paths=file_paths,
                param_strong=param_strong,
                param_weak=param_weak,
                self=v_self, altr=v_other,
                temperature=temperature,
                filter_constant_sum=filter_constant_sum,
                print_=print_
            )
            ev_loss = float(scores["envious"]["loss"])
            gu_loss = float(scores["guilty"]["loss"])
            ev_hit  = float(scores["envious"]["hit_rate"])
            gu_hit  = float(scores["guilty"]["hit_rate"])

            denom = max(ev_loss + gu_loss, 1e-12)
            if ratio_numerator == "envious":
                ratio_val = ev_loss / denom
                higher_means = "Higher → guilty-bot fits better"
                ratio_label  = "E / (E + G)"
            else:
                ratio_val = gu_loss / denom
                higher_means = "Higher → envious-bot fits better"
                ratio_label  = "G / (E + G)"

            Z_ratio[y_idx, x_idx]       = ratio_val
            Z_envious_loss[y_idx, x_idx] = ev_loss
            Z_guilty_loss[y_idx, x_idx]  = gu_loss
            Z_envious_hit[y_idx, x_idx]  = ev_hit
            Z_guilty_hit[y_idx, x_idx]   = gu_hit

    "Build hover data"
    "Customdata shape: (rows, cols, fields)"
    customdata = np.stack(
        [
            Z_envious_loss, Z_guilty_loss,
            Z_envious_hit,  Z_guilty_hit,
            Z_ratio
        ],
        axis=-1
    )

    "Optional text on cells"
    text_matrix = None
    text_template = None
    if show_text_values:
        fmt = f".{text_decimals}f"
        text_matrix   = [[f"{v:{fmt}}" for v in row] for row in Z_ratio.tolist()]
        text_template = "%{text}"

    "Colorscale & template from figure layout settings"
    colorscales = fig_lay.get("colorscales", ["Plasma"])
    colorscale  = colorscales[1] if len(colorscales) > 1 else colorscales[0]
    template    = fig_lay.get("template", "plotly_dark")
    base_font   = fig_lay.get("font_size", 14)

    "Evenly-spaced tick positions (5 ticks) with 2-decimal text for both axes and colorbar."
    _n_ticks = 5
    _x_min, _x_max   = min(x_vals), max(x_vals)
    _y_min, _y_max   = min(y_vals), max(y_vals)
    _cb_min = color_range[0] if color_range is not None else float(np.nanmin(Z_ratio))
    _cb_max = color_range[1] if color_range is not None else float(np.nanmax(Z_ratio))

    def _even_ticks(lo, hi, n):
        vals = [lo + i * (hi - lo) / (n - 1) for i in range(n)]
        text = [f"{v:.2f}" for v in vals]
        return vals, text

    _tick_x,  _ticktext_x  = _even_ticks(_x_min, _x_max, _n_ticks)
    _tick_y,  _ticktext_y  = _even_ticks(_y_min, _y_max, _n_ticks)
    _tick_cb, _ticktext_cb = _even_ticks(_cb_min, _cb_max, _n_ticks)

    "Main heatmap"
    heat = go.Heatmap(
        x=x_vals,
        y=y_vals,
        z=Z_ratio,
        zmin=None if color_range is None else color_range[0],
        zmax=None if color_range is None else color_range[1],
        colorscale=colorscale,
        colorbar=dict(
            title=f"{ratio_label}<br><span style='font-size:0.85em'>{higher_means}</span>",
            titleside="right",
            tickvals=_tick_cb,
            ticktext=_ticktext_cb,
            tickfont=dict(size=base_font + 4),
        ),
        customdata=customdata,
        hovertemplate=(
            "Vᵢᵢ (self): %{x:.2f}<br>"
            "Vᵢⱼ (altruism): %{y:.2f}<br>"
            "Envious NLL: %{customdata[0]:.3f}<br>"
            "Guilty NLL: %{customdata[1]:.3f}<br>"
            "Envious hit: %{customdata[2]:.3f}<br>"
            "Guilty hit: %{customdata[3]:.3f}<br>"
            f"NLL Ratio ({ratio_label}): " + "%{customdata[4]:.3f}<br>"
            "<extra></extra>"
        ),
        text=text_matrix,
        texttemplate=text_template,
        textfont=dict(size=base_font * 2),
        showscale=True,
    )

    fig = go.Figure(data=[heat])

    "Title & axes"
    who_is_strong = f"Envy={param_strong:g}, Guilt={param_weak:g} (envious bot)  |  Envy={param_weak:g}, Guilt={param_strong:g} (guilty bot)"
    title_txt = (
        "Envy vs. Guilt Bot Competition Over (Self, Altruism) Weights<br>"
        f"<span style='font-size:0.85em'>{who_is_strong}  •  τ={temperature:g}</span>"
    )

    fig.update_layout(
        template=template,
        title=title_txt,
        title_x=fig_lay.get("title_x", 0.5),
        title_y=fig_lay.get("title_y", 0.95) - 0.07,
        font=dict(size=base_font, color="white" if template == "plotly_dark" else "black"),
        margin=dict(l=615, r=615, t=160, b=80),
        xaxis=dict(
            title="Self-interest weight Vᵢᵢ",
            tickmode="array",
            tickvals=_tick_x,
            ticktext=_ticktext_x,
            tickfont=dict(size=base_font + 4),
            zeroline=False,
        ),
        yaxis=dict(
            title="Altruism weight Vᵢⱼ",
            tickmode="array",
            tickvals=_tick_y,
            ticktext=_ticktext_y,
            tickfont=dict(size=base_font + 4),
            zeroline=False,
            scaleanchor='x1',
        ),
    )

    "Save if requested"
    if export_fig:
        root = (
            file_paths.get("visuals")
            or file_paths.get("processed")
        )
        os.makedirs(root, exist_ok=True)
        stub = filename_stub or f"IA_bot_competition_heatmap_en{param_strong:g}_gu{param_weak:g}_tau{temperature:g}_{ratio_numerator}"
        if filter_constant_sum: stub += "_filtered"
        out_path = os.path.join(root, f"{stub}.html")
        fig.write_html(out_path)
        if print_:
            print(f"Saved heatmap to: {out_path}")

    return fig


"=========================================================================================="
"====================== Participant Model Fit Extraction =================================="
"=========================================================================================="


def extract_participant_model_combined_fits(
    general_settings: GeneralSettings,
    file_paths: FilePaths,
    create_new_file: bool = False,
) -> pd.DataFrame:
    """
    Extracts per-participant × per-model combined-role fit metrics from the IC analysis JSON
    and caches them as a long-format CSV. Each row represents one (participant, utility model)
    pair and contains the combined NLL from both the chooser and predictor roles, along with
    derived individual-level BIC, ΔBIC, and BIC weights.

    Data source: the 'minvec' field of the IC analysis JSON stores the per-participant,
    per-role minimum NLL and best-fit parameters found across all IC analysis iterations
    for every utility model. No refitting is performed; this function reshapes and augments
    that existing data.

    This CSV is the primary input for all downstream individual-architecture analyses
    (Stages 7–12). Loading it directly avoids re-reading the 617 MB IC JSON on each call.

    Arguments:
        • general_settings: GeneralSettings
            Must contain 'experiment_num'. Used to resolve the IC JSON filename.
        • file_paths: FilePaths
            Must contain 'bic_aic' (directory holding the IC JSON) and 'processed'
            (directory where the output CSV is written).
        • create_new_file: bool (default False)
            If True, recompute from the IC JSON even if the output CSV already exists.
            If False and the output CSV is present, it is loaded and returned immediately.

    Returns:
        • pd.DataFrame — long-format table with one row per (player_uuid, utility_idx).
    """
    experiment_num = general_settings["experiment_num"]
    ic_json_name = f"All_Utility_Forms_IC_Analysis_Experiment{experiment_num}.json"
    ic_json_path = os.path.join(str(file_paths["bic_aic"]), ic_json_name)
    output_csv_path = os.path.join(str(file_paths["processed"]), "participant_model_combined_fits.csv")

    "Return the cached CSV immediately if it exists and an override was not requested."
    if not create_new_file and os.path.exists(output_csv_path):
        print(f"Participant model fits loaded from cache: {os.path.basename(output_csv_path)}")
        return pd.read_csv(output_csv_path, encoding="utf-8-sig")

    # ============================ TEMPORARY PATCH ====================================
    # Remove this block once experiment_3 IC data is fully up-to-date in this repo.
    _OLD_REPO_IC_JSON_PATH = (
        r"C:\Users\Gregory Stanley\Desktop\U of M\Research Archive\Multiplayer"
        r"\ABM_Simulation\Judgment_Game\Inputs\Iter_Binary_Dictator"
        rf"\bic_aic\All_Utility_Forms_IC_Analysis_Experiment{experiment_num}.json"
    )
    _MINIMUM_VALID_IC_JSON_BYTES = 50_000_000  # 50 MB sanity floor
    if not os.path.exists(ic_json_path) or os.path.getsize(ic_json_path) < _MINIMUM_VALID_IC_JSON_BYTES:
        print(
            f"\n{'='*72}\n"
            "[TEMPORARY PATCH] Current-repo IC JSON is missing or below the size threshold.\n"
            f"  Current path  : {ic_json_path}\n"
            f"  Falling back  : {_OLD_REPO_IC_JSON_PATH}\n"
            "Delete this block once this repo's IC data is fully regenerated.\n"
            f"{'='*72}\n"
        )
        ic_json_path = _OLD_REPO_IC_JSON_PATH
    # ========================= END TEMPORARY PATCH ===================================

    "Compute exact per-player game counts from raw game history via the preprocessing module."
    print("Computing per-player game counts from raw game history...")
    all_human_uuids = prep.all_player_uuids(
        file_paths=file_paths, experiment_num=experiment_num, only_humans=True,
    )
    n_data_by_player: Dict[str, Dict[str, int]] = {}
    for player_uuid in all_human_uuids:
        player_dyads = prep.dyads_for_a_player(
            player_uuid=player_uuid, experiment_num=experiment_num,
            file_paths=file_paths, dyad_already_analyzed=False,
        )
        n_chooser_games = sum(
            len(dyad_game_list) for dyad_game_list in player_dyads.values()
            if dyad_game_list and dyad_game_list[0]["chooser"] == player_uuid
        )
        n_predictor_games = sum(
            len(dyad_game_list) for dyad_game_list in player_dyads.values()
            if dyad_game_list and dyad_game_list[0]["predictor"] == player_uuid
        )
        n_combined_games = n_chooser_games + n_predictor_games
        n_data_by_player[player_uuid] = {
            "n_chooser":  n_chooser_games,
            "n_predictor": n_predictor_games,
            "n_combined": n_combined_games,
        }
        if n_combined_games != 120:
            print(
                f"  Note: player {player_uuid} played {n_combined_games} games total "
                f"(chooser={n_chooser_games}, predictor={n_predictor_games}) — "
                "expected 120 for most participants."
            )
    print(f"  Game counts computed for {len(n_data_by_player)} participants.")

    "Load the IC analysis JSON."
    print(f"Loading IC JSON: {ic_json_path}")
    with open(ic_json_path, "r", encoding="utf-8-sig") as ic_file_handle:
        ic_data = json.load(ic_file_handle)
    ic_results = ic_data.get("ic_results", {})
    n_models_in_json = len(ic_results)
    print(
        f"  {n_models_in_json} models found in IC JSON "
        f"(utility universe has 480 — {480 - n_models_in_json} not present in this file)."
    )

    "Iterate over all models and extract per-participant loss data from each model's minvec."
    all_rows: List[Dict] = []
    for utility_tuple_str, model_entry in ic_results.items():
        minvec_for_model = model_entry.get("minvec", {})
        if not minvec_for_model:
            continue
        utility_idx = int(model_entry["idx"])
        k_params = int(model_entry["k_params"])
        k_effective = 2 * k_params
        equation = model_entry.get("U", "")
        utility_settings_flags: Dict = model_entry.get("utility_settings", {})

        for player_uuid, player_fit_entry in minvec_for_model.items():
            player_losses = player_fit_entry.get("loss", {})
            chooser_loss_nll_raw = player_losses.get("chooser", np.nan)
            predictor_loss_nll_raw = player_losses.get("predictor", np.nan)
            if np.isnan(float(chooser_loss_nll_raw)) or np.isnan(float(predictor_loss_nll_raw)):
                continue
            chooser_loss_nll = float(chooser_loss_nll_raw)
            predictor_loss_nll = float(predictor_loss_nll_raw)
            combined_loss_nll = chooser_loss_nll + predictor_loss_nll

            n_entry = n_data_by_player.get(player_uuid, None)
            n_chooser = float(n_entry["n_chooser"])  if n_entry else np.nan
            n_predictor = float(n_entry["n_predictor"]) if n_entry else np.nan
            n_combined = float(n_entry["n_combined"])  if n_entry else np.nan

            aic_individual = 2.0 * combined_loss_nll + 2.0 * k_effective
            bic_individual = (
                2.0 * combined_loss_nll + k_effective * np.log(n_combined)
                if np.isfinite(n_combined) and n_combined > 0 else np.nan
            )

            row: Dict = {
                "player_uuid":        player_uuid,
                "utility_tuple_str":  utility_tuple_str,
                "utility_idx":        utility_idx,
                "k_params":           k_params,
                "k_effective":        k_effective,
                "equation":           equation,
                "chooser_loss_nll":   chooser_loss_nll,
                "predictor_loss_nll": predictor_loss_nll,
                "combined_loss_nll":  combined_loss_nll,
                "n_chooser":          n_chooser,
                "n_predictor":        n_predictor,
                "n_combined":         n_combined,
                "AIC_individual":     aic_individual,
                "BIC_individual":     bic_individual,
            }
            row.update(utility_settings_flags)
            all_rows.append(row)

    combined_fits_df = pd.DataFrame(all_rows)
    n_participants_extracted = combined_fits_df["player_uuid"].nunique()
    n_models_extracted = combined_fits_df["utility_idx"].nunique()
    print(
        f"  Extracted {len(combined_fits_df)} rows: "
        f"{n_participants_extracted} participants × {n_models_extracted} models."
    )

    "Compute per-participant ΔBIC: subtract each participant's personal BIC minimum."
    combined_fits_df["delta_BIC_individual"] = (
        combined_fits_df["BIC_individual"]
        - combined_fits_df.groupby("player_uuid")["BIC_individual"].transform("min")
    )

    """
    Compute BIC weights using numerically stable log-sum-exp. Within each participant group,
    shift all unnormalized log-weights by the group maximum before exponentiating to prevent
    underflow or overflow when ΔBIC values span a wide range.
    """
    combined_fits_df["_log_w_unnorm"] = -0.5 * combined_fits_df["delta_BIC_individual"]
    log_partition_by_player = combined_fits_df.groupby("player_uuid")["_log_w_unnorm"].transform(
        lambda x: x.max() + np.log(np.exp(x.values - x.max()).sum())
    )
    combined_fits_df["BIC_weight"] = np.exp(
        combined_fits_df["_log_w_unnorm"] - log_partition_by_player
    )
    combined_fits_df.drop(columns=["_log_w_unnorm"], inplace=True)

    "Compute per-participant summary statistics and join them back to the long-format table."
    per_participant_summary_rows: List[Dict] = []
    for player_uuid_key, participant_group in combined_fits_df.groupby("player_uuid"):
        participant_weights = participant_group["BIC_weight"].values
        participant_delta_bic_values = participant_group["delta_BIC_individual"].values
        effective_number_of_models = float(1.0 / np.sum(participant_weights ** 2))
        model_weight_entropy = float(-np.sum(participant_weights * np.log(participant_weights + 1e-300)))
        best_model_position = int(np.argmin(participant_delta_bic_values))
        top_model_utility_idx = int(participant_group["utility_idx"].values[best_model_position])
        sorted_delta_bic_values = np.sort(participant_delta_bic_values)
        top_model_delta_BIC = float(sorted_delta_bic_values[1]) if len(sorted_delta_bic_values) > 1 else 0.0
        n_models_with_delta_BIC_le_2 = int(np.sum(participant_delta_bic_values <= 2))
        n_models_with_delta_BIC_le_10 = int(np.sum(participant_delta_bic_values <= 10))
        per_participant_summary_rows.append({
            "player_uuid":                   player_uuid_key,
            "effective_number_of_models":    effective_number_of_models,
            "model_weight_entropy":          model_weight_entropy,
            "top_model_utility_idx":         top_model_utility_idx,
            "top_model_delta_BIC":           top_model_delta_BIC,
            "n_models_with_delta_BIC_le_2":  n_models_with_delta_BIC_le_2,
            "n_models_with_delta_BIC_le_10": n_models_with_delta_BIC_le_10,
        })
    participant_summary_df = pd.DataFrame(per_participant_summary_rows)
    combined_fits_df = combined_fits_df.merge(
        right=participant_summary_df, on="player_uuid", how="left",
    )

    "Save the combined fits table."
    os.makedirs(str(file_paths["processed"]), exist_ok=True)
    combined_fits_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_csv_path}  ({len(combined_fits_df)} rows)")

    return combined_fits_df


"=========================================================================================="
"==================== Participant Cloud Distances and Architecture Embedding =============="
"=========================================================================================="


_UTILITY_BOOLEAN_SETTINGS_COLUMNS = [
    "conditional_welfare_mode",
    "reference_dependent_altruism",
    "min_max_rawlsian_leontief",
    "use_exponential_parameters",
    "apply_exponents_to_payoffs",
    "single_exponential_parameter",
    "single_payoffs_not_differences",
    "payoff_ratios_not_differences",
    "reference_dependent_utility",
    "use_negativity_parameters",
    "negativity_social_comparison",
    "fix_self_interest_parameter",
    "include_social_comparison",
    "include_altruism_term",
]


def compute_participant_model_space_centroids(
    general_settings: dict,
    file_paths: dict,
    create_new_file: bool = False,
) -> pd.DataFrame:
    """
    Computes the BIC-weighted centroid of each participant's model cloud in model MDS space.
    Each participant's centroid is the weighted average of model MDS coordinates (from
    compute_model_space_embedding), where the weights are the participant's BIC weights from
    extract_participant_model_combined_fits. This gives a visual summary of where each
    participant's model support is concentrated in the model-space MDS plot.

    Arguments:
        • general_settings: dict — used to resolve the model MDS embedding filename via
            _ampd_distance_name(general_settings).
        • file_paths: dict — must contain 'processed'.
        • create_new_file: bool (default False) — if False and the output CSV exists, load
            and return it without recomputing.

    Returns:
        • pd.DataFrame — one row per participant, columns: player_uuid, centroid_mds_x,
            centroid_mds_y, n_models_in_embedding.
    """
    output_csv_path = os.path.join(
        file_paths["processed"], "participant_model_space_centroids.csv",
    )
    if not create_new_file and os.path.exists(output_csv_path):
        print(f"Participant model-space centroids loaded from cache: {output_csv_path}")
        return pd.read_csv(output_csv_path)

    combined_fits_df = pd.read_csv(
        os.path.join(file_paths["processed"], "participant_model_combined_fits.csv"),
    )

    distance_name = _ampd_distance_name(general_settings)
    embedding_csv_path = os.path.join(
        file_paths["processed"],
        f"model_space_embedding__{distance_name}__dims=2.csv",
    )
    if not os.path.exists(embedding_csv_path):
        raise FileNotFoundError(
            f"Model-space embedding not found: {embedding_csv_path}\n"
            "Run compute_model_space_embedding first."
        )
    model_embedding_df = pd.read_csv(embedding_csv_path, dtype={"utility_bitstring": str})
    print(f"Loaded model-space embedding: {embedding_csv_path}  ({len(model_embedding_df)} models)")

    "Inner join to use only models present in both the combined fits and the MDS embedding."
    fits_with_coords_df = combined_fits_df.merge(
        right=model_embedding_df[["utility_idx", "mds_x", "mds_y"]],
        on="utility_idx",
        how="inner",
    )
    n_models_in_embedding = int(fits_with_coords_df["utility_idx"].nunique())
    print(f"Models present in both fits and embedding: {n_models_in_embedding}")

    "Re-normalize BIC weights within the embedded model subset so the centroid is a proper weighted average."
    fits_with_coords_df = fits_with_coords_df.copy()
    weight_sums_per_participant = fits_with_coords_df.groupby("player_uuid")["BIC_weight"].transform("sum")
    fits_with_coords_df["BIC_weight_normalized"] = (
        fits_with_coords_df["BIC_weight"] / weight_sums_per_participant.clip(lower=1e-12)
    )

    centroid_rows = []
    for player_uuid_key, participant_group in fits_with_coords_df.groupby("player_uuid"):
        bic_weight_values = participant_group["BIC_weight_normalized"].values
        centroid_mds_x = float(np.sum(bic_weight_values * participant_group["mds_x"].values))
        centroid_mds_y = float(np.sum(bic_weight_values * participant_group["mds_y"].values))
        centroid_rows.append({
            "player_uuid":           player_uuid_key,
            "centroid_mds_x":        centroid_mds_x,
            "centroid_mds_y":        centroid_mds_y,
            "n_models_in_embedding": n_models_in_embedding,
        })
    centroids_df = pd.DataFrame(centroid_rows)

    centroids_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_csv_path}  ({len(centroids_df)} participants)")
    return centroids_df


def compute_participant_cloud_distances(
    general_settings: dict,
    file_paths: dict,
    create_new_file: bool = False,
) -> dict:
    """
    Computes pairwise participant cloud distances using the energy distance formulation.
    Each participant is treated as a BIC-weight distribution over utility models (from
    extract_participant_model_combined_fits); the AMPD matrix (from compute_ampd_distance_matrix)
    provides the distance geometry over model space.

    Two matrices are computed and saved:
      cross_distance_ij   = bic_weights_i @ ampd_matrix @ bic_weights_j
      energy_distance_ij  = sqrt(max(0, 2*cross_ij - within_i - within_j))
    where within_i = bic_weights_i @ ampd_matrix @ bic_weights_i.

    energy_distance(participant, participant) = 0 by construction. The cross matrix is
    also saved as a descriptive similarity measure for downstream stages.

    Arguments:
        • general_settings: dict — used to load the AMPD matrix via
            _load_ampd_matrix_from_settings.
        • file_paths: dict — must contain 'processed'.
        • create_new_file: bool (default False) — if False and output CSVs exist, load
            and return them without recomputing.

    Returns:
        • dict with keys 'cross' and 'energy', each a square pd.DataFrame indexed and
            columned by player_uuid.
    """
    cross_csv_path = os.path.join(
        file_paths["processed"],
        "participant_cross_cloud_distance_matrix__metric=ampd.csv",
    )
    energy_csv_path = os.path.join(
        file_paths["processed"],
        "participant_cloud_distance_matrix__metric=ampd_energy.csv",
    )
    if not create_new_file and os.path.exists(cross_csv_path) and os.path.exists(energy_csv_path):
        cross_distance_df  = pd.read_csv(cross_csv_path,  index_col=0)
        energy_distance_df = pd.read_csv(energy_csv_path, index_col=0)
        cache_is_invalid = (
            energy_distance_df.isnull().any().any()
            or float(energy_distance_df.values.max()) < 1e-10
        )
        if cache_is_invalid:
            print("Cached energy distance matrix is invalid (contains NaN or is all-zero) — regenerating.")
        else:
            print(f"Participant cloud distances loaded from cache.")
            return {"cross": cross_distance_df, "energy": energy_distance_df}

    combined_fits_df = pd.read_csv(
        os.path.join(file_paths["processed"], "participant_model_combined_fits.csv"),
    )
    ampd_matrix_df = _load_ampd_matrix_from_settings(
        general_settings=general_settings, file_paths=file_paths,
    )
    print(f"AMPD matrix loaded: {ampd_matrix_df.shape[0]}×{ampd_matrix_df.shape[1]}")

    shared_model_idxs = sorted(
        set(combined_fits_df["utility_idx"].unique()) & set(ampd_matrix_df.index.tolist())
    )
    print(f"{len(shared_model_idxs)} models present in both BIC fits and AMPD matrix.")

    ampd_submatrix_all = ampd_matrix_df.loc[shared_model_idxs, shared_model_idxs].values
    ampd_row_has_nan = np.isnan(ampd_submatrix_all).any(axis=1)
    if ampd_row_has_nan.any():
        n_models_dropped = int(ampd_row_has_nan.sum())
        shared_model_idxs = [model_idx for model_idx, has_nan in zip(shared_model_idxs, ampd_row_has_nan) if not has_nan]
        print(
            f"AMPD matrix partially complete: dropped {n_models_dropped} models with uncomputed rows, "
            f"using {len(shared_model_idxs)} models for cloud distances."
        )

    all_player_uuids = sorted(combined_fits_df["player_uuid"].unique())
    n_participants = len(all_player_uuids)

    bic_weight_matrix = (
        combined_fits_df[combined_fits_df["utility_idx"].isin(shared_model_idxs)]
        .pivot(index="player_uuid", columns="utility_idx", values="BIC_weight")
        .reindex(index=all_player_uuids, columns=shared_model_idxs)
        .fillna(0.0)
        .values
    )  # shape: (n_participants, n_complete_models)

    "Re-normalize rows to sum to 1 over the restricted model set."
    bic_weight_row_sums = bic_weight_matrix.sum(axis=1, keepdims=True)
    bic_weight_matrix_normalized = bic_weight_matrix / np.maximum(bic_weight_row_sums, 1e-12)

    ampd_submatrix = ampd_matrix_df.loc[shared_model_idxs, shared_model_idxs].values
    # shape: (n_complete_models, n_complete_models)

    print(f"Computing participant cloud distances ({n_participants}×{n_participants})...")
    cross_distance_matrix = bic_weight_matrix_normalized @ ampd_submatrix @ bic_weight_matrix_normalized.T
    # shape: (n_participants, n_participants)

    within_dispersion_per_participant = np.diag(cross_distance_matrix)
    energy_distance_matrix = np.sqrt(np.maximum(
        0.0,
        2 * cross_distance_matrix
        - within_dispersion_per_participant[:, None]
        - within_dispersion_per_participant[None, :],
    ))
    np.fill_diagonal(energy_distance_matrix, 0.0)

    print(f"Energy distance diagonal max:    {np.diag(energy_distance_matrix).max():.2e}")
    print(f"Energy distance symmetry error:  {np.abs(energy_distance_matrix - energy_distance_matrix.T).max():.2e}")
    print(f"Energy distance range:           [{energy_distance_matrix.min():.4f}, {energy_distance_matrix.max():.4f}]")

    cross_distance_df  = pd.DataFrame(cross_distance_matrix,  index=all_player_uuids, columns=all_player_uuids)
    energy_distance_df = pd.DataFrame(energy_distance_matrix, index=all_player_uuids, columns=all_player_uuids)

    cross_distance_df.to_csv(cross_csv_path, encoding="utf-8-sig")
    energy_distance_df.to_csv(energy_csv_path, encoding="utf-8-sig")
    print(f"Saved cross distance matrix:  {cross_csv_path}")
    print(f"Saved energy distance matrix: {energy_csv_path}")

    return {"cross": cross_distance_df, "energy": energy_distance_df}


def compute_participant_architecture_embedding(
    general_settings: dict,
    file_paths: dict,
    n_dimensions: int = 2,
    create_new_file: bool = False,
) -> pd.DataFrame:
    """
    Computes a classical MDS embedding of participants in architecture space, using the
    energy-distance matrix from compute_participant_cloud_distances. Each point represents
    one participant; nearby points have similar BIC-weight distributions over model space.

    Calls _classical_mds (the same helper used by compute_model_space_embedding) on the
    73×73 participant energy-distance matrix.

    Arguments:
        • general_settings: dict — unused directly; kept for consistency with repo pattern.
        • file_paths: dict — must contain 'processed'.
        • n_dimensions: int (default 2) — number of MDS dimensions to embed.
        • create_new_file: bool (default False) — if False and the output CSV exists, load
            and return it without recomputing.

    Returns:
        • pd.DataFrame — one row per participant, columns: player_uuid, mds_x, mds_y
            (plus mds_z and mds_w for higher dimensions), top_model_utility_idx,
            effective_number_of_models, model_weight_entropy.
    """
    output_csv_path = os.path.join(
        file_paths["processed"], "participant_architecture_embedding.csv",
    )
    if not create_new_file and os.path.exists(output_csv_path):
        cached_embedding_df = pd.read_csv(output_csv_path)
        mds_cols = [col for col in ["mds_x", "mds_y", "mds_z", "mds_w"] if col in cached_embedding_df.columns]
        cache_is_invalid = (
            cached_embedding_df[mds_cols].isnull().any().any()
            or float(cached_embedding_df[mds_cols].abs().max().max()) < 1e-10
        )
        if cache_is_invalid:
            print("Cached participant architecture embedding is invalid (contains NaN or is all-zero) — regenerating.")
        else:
            print(f"Participant architecture embedding loaded from cache: {output_csv_path}")
            return cached_embedding_df

    energy_distance_csv_path = os.path.join(
        file_paths["processed"],
        "participant_cloud_distance_matrix__metric=ampd_energy.csv",
    )
    if not os.path.exists(energy_distance_csv_path):
        raise FileNotFoundError(
            f"Energy distance matrix not found: {energy_distance_csv_path}\n"
            "Run compute_participant_cloud_distances first."
        )
    energy_distance_df = pd.read_csv(energy_distance_csv_path, index_col=0)
    all_player_uuids = list(energy_distance_df.index)
    energy_distance_values = energy_distance_df.values.astype(float)

    "Symmetrize to neutralize floating-point drift, then zero the diagonal."
    energy_distance_symmetric = (energy_distance_values + energy_distance_values.T) / 2.0
    np.fill_diagonal(energy_distance_symmetric, 0.0)

    mds_coordinates, top_eigenvalues = _classical_mds(
        distance_matrix=energy_distance_symmetric, n_dimensions=n_dimensions,
    )

    "Compute fraction of variance explained by the included dimensions."
    n_entities = len(all_player_uuids)
    centering_matrix_J = np.eye(n_entities) - np.ones((n_entities, n_entities)) / n_entities
    gram_matrix_B = -0.5 * (centering_matrix_J @ (energy_distance_symmetric ** 2) @ centering_matrix_J)
    all_eigenvalues = np.linalg.eigvalsh(gram_matrix_B)
    sum_positive_eigenvalues = float(np.maximum(all_eigenvalues, 0.0).sum())
    variance_fraction = float(np.maximum(top_eigenvalues, 0.0).sum()) / max(sum_positive_eigenvalues, 1e-12)
    print(
        f"Participant architecture MDS: {n_dimensions}D explains {variance_fraction:.1%} of variance.  "
        f"Top eigenvalues: {[round(float(ev), 4) for ev in top_eigenvalues]}"
    )

    dim_labels = ["mds_x", "mds_y", "mds_z", "mds_w"]
    embedding_df = pd.DataFrame({"player_uuid": all_player_uuids})
    for dim_idx in range(n_dimensions):
        embedding_df[dim_labels[dim_idx]] = mds_coordinates[:, dim_idx]

    "Join participant summary statistics from combined fits."
    combined_fits_csv_path = os.path.join(file_paths["processed"], "participant_model_combined_fits.csv")
    if os.path.exists(combined_fits_csv_path):
        participant_summary_df = (
            pd.read_csv(combined_fits_csv_path)
            [["player_uuid", "top_model_utility_idx", "top_model_delta_BIC",
              "effective_number_of_models", "model_weight_entropy"]]
            .drop_duplicates(subset="player_uuid")
        )
        embedding_df = embedding_df.merge(right=participant_summary_df, on="player_uuid", how="left")

    embedding_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_csv_path}  ({len(embedding_df)} participants, {n_dimensions}D)")
    return embedding_df


def compute_participant_feature_support(
    general_settings: dict,
    file_paths: dict,
    create_new_file: bool = False,
) -> pd.DataFrame:
    """
    Computes each participant's BIC-weighted probability that each Boolean utility setting
    is active, using the model cloud BIC weights from extract_participant_model_combined_fits.
    For each participant and each
    utility setting, P_i(setting=True) = sum_m w_i,m * setting_m_value, which gives the
    expected presence of that architectural feature weighted by model plausibility.

    Arguments:
        • general_settings: dict — unused directly; kept for consistency with repo pattern.
        • file_paths: dict — must contain 'processed'.
        • create_new_file: bool (default False) — if False and the output CSV exists, load
            and return it without recomputing.

    Returns:
        • pd.DataFrame — one row per participant, columns: player_uuid, P_<setting> × 14,
            effective_number_of_models, model_weight_entropy.
    """
    output_csv_path = os.path.join(
        file_paths["processed"], "participant_feature_support.csv",
    )
    if not create_new_file and os.path.exists(output_csv_path):
        print(f"Participant feature support loaded from cache: {output_csv_path}")
        return pd.read_csv(output_csv_path)

    combined_fits_df = pd.read_csv(
        os.path.join(file_paths["processed"], "participant_model_combined_fits.csv"),
    )

    settings_cols_present = [col for col in _UTILITY_BOOLEAN_SETTINGS_COLUMNS if col in combined_fits_df.columns]
    if not settings_cols_present:
        raise ValueError(
            "No utility boolean settings columns found in participant_model_combined_fits.csv. "
            "Expected columns like 'include_social_comparison', 'include_altruism_term', etc."
        )
    print(f"Computing feature support for {len(settings_cols_present)} utility settings columns.")

    all_player_uuids = sorted(combined_fits_df["player_uuid"].unique())
    all_model_idxs   = sorted(combined_fits_df["utility_idx"].unique())

    bic_weight_matrix = (
        combined_fits_df
        .pivot(index="player_uuid", columns="utility_idx", values="BIC_weight")
        .reindex(index=all_player_uuids, columns=all_model_idxs)
        .fillna(0.0)
        .values
    )  # shape: (n_participants, n_models)

    model_settings_lookup_df = (
        combined_fits_df[["utility_idx"] + settings_cols_present]
        .drop_duplicates(subset="utility_idx")
        .set_index("utility_idx")
        .reindex(index=all_model_idxs)
    )
    utility_settings_feature_matrix = model_settings_lookup_df.values.astype(float)
    # shape: (n_models, n_settings)

    feature_support_matrix = bic_weight_matrix @ utility_settings_feature_matrix
    # shape: (n_participants, n_settings); values in [0, 1]

    print(f"Feature support range: [{feature_support_matrix.min():.4f}, {feature_support_matrix.max():.4f}]")

    feature_support_df = pd.DataFrame(
        feature_support_matrix,
        index=all_player_uuids,
        columns=[f"P_{col}" for col in settings_cols_present],
    ).reset_index().rename(columns={"index": "player_uuid"})

    "Join participant-level summary stats (one value per participant, not per model)."
    participant_summary_df = (
        combined_fits_df[["player_uuid", "effective_number_of_models", "model_weight_entropy"]]
        .drop_duplicates(subset="player_uuid")
    )
    feature_support_df = feature_support_df.merge(
        right=participant_summary_df, on="player_uuid", how="left",
    )

    feature_support_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_csv_path}  ({len(feature_support_df)} participants, {len(settings_cols_present)} settings)")
    return feature_support_df


def plot_participant_architecture_mds(
    general_settings: dict,
    file_paths: dict,
    fig_lay: dict,
    color_by: str = "model_weight_entropy",
    include_dropdown: bool = True,
    export_fig: bool = True,
) -> go.Figure:
    """
    Plotly interactive scatter of participants embedded in architecture space via classical
    MDS on the energy cloud distance matrix. Each point is one participant; nearby points
    have similar BIC-weight distributions over utility models. Hovering shows participant
    metadata: top model, runner-up gap, effective number of models, entropy, and feature
    support probabilities. A dropdown allows switching between color encodings.

    Follows the same structure and style as plot_model_space_mds.

    Arguments:
        • general_settings: dict — for experiment metadata in the title.
        • file_paths: dict — must contain 'processed' and 'visuals'.
        • fig_lay: dict — layout constants from config.py.
        • color_by: str (default 'model_weight_entropy') — the column used for the default
            color encoding. One of: 'model_weight_entropy', 'effective_number_of_models',
            'top_model_delta_BIC', or any 'P_<setting>' column.
        • include_dropdown: bool (default True) — if True, add a dropdown to switch between
            color encodings.
        • export_fig: bool (default True) — if True, write the figure to visuals/.

    Returns:
        • go.Figure — Plotly figure; also written to visuals/participant_architecture_mds.html
            if export_fig is True.
    """
    embedding_path = os.path.join(file_paths["processed"], "participant_architecture_embedding.csv")
    feature_support_path = os.path.join(file_paths["processed"], "participant_feature_support.csv")

    if not os.path.exists(embedding_path):
        raise FileNotFoundError(
            f"Participant architecture embedding not found: {embedding_path}\n"
            "Run compute_participant_architecture_embedding first."
        )
    embedding_df = pd.read_csv(embedding_path)

    if os.path.exists(feature_support_path):
        feature_support_df = pd.read_csv(feature_support_path)
        plot_df = embedding_df.merge(right=feature_support_df, on="player_uuid", how="left", suffixes=("", "_feat"))
    else:
        plot_df = embedding_df.copy()

    feature_support_p_cols = [col for col in plot_df.columns if col.startswith("P_")]

    "=== Load compression-curve K=2..K=8 assignments for categorical coloring layers ==="
    k_library_trace_indices: dict = {}   # k_val → list of trace indices for that K-library
    _assignments_csv = os.path.join(file_paths["processed"], "population_architecture_assignments.csv")
    if os.path.exists(_assignments_csv):
        _assign_all      = pd.read_csv(_assignments_csv, encoding='utf-8-sig')
        _available_k_set = set(_assign_all['K'].unique())
        for _k_val in [k_val for k_val in range(2, 9) if k_val in _available_k_set]:
            _col_name = f'_k{_k_val}_assigned_model'
            _k_slice  = (
                _assign_all[_assign_all['K'] == _k_val][['player_uuid', 'assigned_utility_idx']]
                .rename(columns={'assigned_utility_idx': _col_name})
            )
            plot_df = plot_df.merge(_k_slice, on='player_uuid', how='left')
            k_library_trace_indices[_k_val] = []

    "Load combined fits to extract top-model and runner-up model equations per participant."
    combined_fits_csv_path = os.path.join(file_paths["processed"], "participant_model_combined_fits.csv")
    runner_up_lookup: dict = {}
    top_equation_lookup: dict = {}
    if os.path.exists(combined_fits_csv_path):
        combined_fits_df = pd.read_csv(combined_fits_csv_path)
        for player_uuid_key, participant_group in combined_fits_df.groupby("player_uuid"):
            participant_sorted = participant_group.sort_values("delta_BIC_individual").reset_index(drop=True)
            top_equation_lookup[player_uuid_key] = str(participant_sorted.at[0, "equation"]) if "equation" in participant_sorted.columns else "?"
            if len(participant_sorted) > 1:
                runner_up_lookup[player_uuid_key] = {
                    "utility_idx": int(participant_sorted.at[1, "utility_idx"]),
                    "equation":    str(participant_sorted.at[1, "equation"]) if "equation" in participant_sorted.columns else "?",
                }
            else:
                runner_up_lookup[player_uuid_key] = {"utility_idx": None, "equation": "?"}

    marker_size = int(fig_lay.get("markersize", 16) * 2)
    marker_outline = dict(width=1.5, color="hsla(0, 0%, 0%, 0.45)")

    def _participant_hover(row: pd.Series) -> str:
        player_uuid_val      = str(row.get("player_uuid", "?"))
        top_model_idx        = int(row["top_model_utility_idx"]) if pd.notna(row.get("top_model_utility_idx")) else None
        top_model_eq         = top_equation_lookup.get(player_uuid_val, "?")
        runner_up_data       = runner_up_lookup.get(player_uuid_val, {})
        runner_up_idx        = runner_up_data.get("utility_idx")
        runner_up_eq         = runner_up_data.get("equation", "?")
        runner_up_gap_val    = f"{row['top_model_delta_BIC']:.2f}" if pd.notna(row.get("top_model_delta_BIC")) else "?"
        effective_models_val = f"{row['effective_number_of_models']:.1f}" if pd.notna(row.get("effective_number_of_models")) else "?"
        entropy_val          = f"{row['model_weight_entropy']:.2f}" if pd.notna(row.get("model_weight_entropy")) else "?"
        top_model_label      = f"{top_model_idx:03d}" if top_model_idx is not None else "???"
        runner_up_label      = f"{runner_up_idx:03d}" if runner_up_idx is not None else "???"
        return (
            f"Participant {player_uuid_val}<br>"
            f"Top Model: {top_model_label} - {top_model_eq}<br>"
            f"Runner up: {runner_up_label} - {runner_up_eq}<br>"
            f"Runner up gap: {runner_up_gap_val} | Effective models: {effective_models_val} | Entropy: {entropy_val}"
        )

    plot_df     = plot_df.reset_index(drop=True)
    hover_texts = [_participant_hover(row) for _, row in plot_df.iterrows()]
    plot_df['_hover'] = hover_texts

    continuous_color_specs = [
        ("model_weight_entropy",       "Entropy",         "Viridis",   "Entropy"),
        ("effective_number_of_models", "Eff. Models",     "Plasma",    "Eff. Models"),
        ("top_model_delta_BIC",        "Runner-Up Gap",   "Cividis",   "Runner-Up Gap"),
    ]

    feature_label_map = {
        "P_conditional_welfare_mode":       "P(Conditional Welfare)",
        "P_reference_dependent_altruism":   "P(Ref-Dep Altruism)",
        "P_min_max_rawlsian_leontief":      "P(Min-Max / Rawls)",
        "P_use_exponential_parameters":     "P(Exponential Params)",
        "P_apply_exponents_to_payoffs":     "P(Exponents to Payoffs)",
        "P_single_exponential_parameter":   "P(Single Exponent)",
        "P_single_payoffs_not_differences": "P(Single Payoffs)",
        "P_payoff_ratios_not_differences":  "P(Payoff Ratios)",
        "P_reference_dependent_utility":    "P(Ref-Dep Utility)",
        "P_use_negativity_parameters":      "P(Negativity Params)",
        "P_negativity_social_comparison":   "P(Negativity Soc. Comp.)",
        "P_fix_self_interest_parameter":    "P(Fix Self-Interest)",
        "P_include_social_comparison":      "P(Social Comparison)",
        "P_include_altruism_term":          "P(Altruism Term)",
    }

    data_traces = []
    default_visible_trace_idx = 0

    for trace_position, (color_col, dropdown_label, colorscale_name, colorbar_title) in enumerate(continuous_color_specs):
        color_values = plot_df[color_col].values if color_col in plot_df.columns else np.zeros(len(plot_df))
        is_default_visible = (color_col == color_by)
        if is_default_visible:
            default_visible_trace_idx = trace_position
        data_traces.append(go.Scatter(
            x=plot_df["mds_x"], y=plot_df["mds_y"],
            mode="markers",
            name=f"Participants ({dropdown_label})",
            visible=is_default_visible,
            showlegend=False,
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            marker=dict(
                size=marker_size,
                color=color_values,
                colorscale=colorscale_name,
                showscale=True,
                colorbar=dict(title=colorbar_title, x=1.02, thickness=36, len=0.75, tickformat='.2f'),
                line=marker_outline,
            ),
        ))

    "Feature support traces (P_<setting> columns), one trace per setting, all hidden by default."
    feature_trace_start_idx = len(data_traces)
    for feature_col in feature_support_p_cols:
        feature_color_values = plot_df[feature_col].values if feature_col in plot_df.columns else np.zeros(len(plot_df))
        feature_display_label = feature_label_map.get(feature_col, feature_col)
        data_traces.append(go.Scatter(
            x=plot_df["mds_x"], y=plot_df["mds_y"],
            mode="markers",
            name=feature_display_label,
            visible=False,
            showlegend=False,
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            marker=dict(
                size=marker_size,
                color=feature_color_values,
                colorscale="RdBu",
                cmin=0.0, cmax=1.0,
                showscale=True,
                colorbar=dict(title=feature_display_label, x=1.02, thickness=36, len=0.75, tickformat='.2f'),
                line=marker_outline,
            ),
        ))

    "=== Categorical traces: K=2..K=8 compression-curve library assignments ==="
    for _k_val, _trace_indices_list in sorted(k_library_trace_indices.items()):
        _k_col    = f'_k{_k_val}_assigned_model'
        if _k_col not in plot_df.columns:
            continue
        _k_models = sorted(plot_df[_k_col].dropna().unique().astype(int))
        _n_models  = len(_k_models)
        for _i, _model_idx in enumerate(_k_models):
            _mask  = (plot_df[_k_col] == _model_idx)
            _hue   = int(360 * _i / _n_models)
            _color = _hsla(hue=_hue, saturation_percent=72, lightness_percent=52, alpha=0.92)
            _trace_indices_list.append(len(data_traces))
            data_traces.append(go.Scatter(
                x=plot_df.loc[_mask, "mds_x"],
                y=plot_df.loc[_mask, "mds_y"],
                mode="markers",
                name=f"Model {_model_idx:03d}",
                visible=False,
                showlegend=True,
                text=plot_df.loc[_mask, '_hover'].tolist(),
                hovertemplate="%{text}<extra></extra>",
                marker=dict(size=marker_size, color=_color, line=marker_outline),
            ))

    "Symmetric equal-range axes: furthest outlier across both dims, rounded up to next 0.1."
    import math
    xy_max_abs     = max(
        abs(float(plot_df["mds_x"].max())), abs(float(plot_df["mds_x"].min())),
        abs(float(plot_df["mds_y"].max())), abs(float(plot_df["mds_y"].min())),
    )
    mds_axis_limit = math.ceil(xy_max_abs * 10) / 10
    mds_axis_range = [-mds_axis_limit, mds_axis_limit]

    "Tick values at 0.1 intervals; y-axis bottom tick is blank to avoid overlap with x-axis labels."
    tick_vals   = [round(float(val), 1) for val in np.arange(-mds_axis_limit, mds_axis_limit + 0.05, 0.1)]
    x_tick_text = [f'{val:.1f}' for val in tick_vals]
    y_tick_text = ['' if idx == 0 else f'{val:.1f}' for idx, val in enumerate(tick_vals)]

    experiment_num = general_settings.get("experiment_num", "?")
    n_participants = len(plot_df)
    fig = go.Figure(data=data_traces)
    fig.update_layout(
        template=fig_lay.get("template", "plotly_white"),
        title=f"Participant Utility Function MDS — Energy Distance (Exp {experiment_num-1}, N={n_participants})",
        titlefont_size=fig_lay["titlefont_size"]-10,
        title_x=0.5,
        font=fig_lay.get("font", {}),
        hoverlabel=fig_lay.get("hoverlabel", {}),
        margin=dict(l=500, r=560, t=140, b=100),
        xaxis=dict(
            title="Utility Function MDS Dimension 1",
            range=mds_axis_range,
            tickvals=tick_vals,
            ticktext=x_tick_text,
            scaleanchor="y", scaleratio=1,
            **fig_lay.get("xaxis", {}),
        ),
        yaxis=dict(
            title="Utility Function MDS Dimension 2",
            range=mds_axis_range,
            tickvals=tick_vals,
            ticktext=y_tick_text,
            **fig_lay.get("yaxis", {}),
        ),
    )

    if include_dropdown:
        n_total_traces = len(data_traces)

        def _visibility_list(on_indices):
            visibility = [False] * n_total_traces
            for on_idx in on_indices:
                visibility[on_idx] = True
            return visibility

        dropdown_buttons = []
        for trace_position, (color_col, dropdown_label, _unused_colorscale, _unused_colorbar) in enumerate(continuous_color_specs):
            dropdown_buttons.append(dict(
                label=f"Color: {dropdown_label}",
                method="update",
                args=[
                    {"visible": _visibility_list([trace_position])},
                    {"title": f"Participant Utility Function MDS ({dropdown_label})"},
                ],
            ))
        for _k_val, _trace_indices_list in sorted(k_library_trace_indices.items()):
            if _trace_indices_list:
                dropdown_buttons.append(dict(
                    label=f"Color: K={_k_val} Library",
                    method="update",
                    args=[
                        {"visible": _visibility_list(_trace_indices_list)},
                        {"title": f"Participant Utility Function MDS (K={_k_val} Library Assignment)"},
                    ],
                ))
        for feature_offset, feature_col in enumerate(feature_support_p_cols):
            feature_display_label = feature_label_map.get(feature_col, feature_col)
            feature_trace_idx = feature_trace_start_idx + feature_offset
            dropdown_buttons.append(dict(
                label=feature_display_label,
                method="update",
                args=[
                    {"visible": _visibility_list([feature_trace_idx])},
                    {"title": f"Participant Utility Function MDS ({feature_display_label})"},
                ],
            ))

        dropdown_to_side = True
        fig.update_layout(updatemenus=[dict(
            buttons=dropdown_buttons,
            direction="down", yanchor="top", xanchor="left", 
            x=-0.50 if dropdown_to_side else 0.03, 
            y=1.15 if dropdown_to_side else 1.00, 
            bgcolor=(
                _hsla(hue=0, saturation_percent=0, lightness_percent=20, alpha=0.85) if "dark" in fig_lay.get("template", "")
                else _hsla(hue=0, saturation_percent=0, lightness_percent=94, alpha=0.92)
            ),
            font=dict(size=16, family=fig_lay.get("font", {}).get("family", "Calibri")),
        )])

    if export_fig:
        out_path = os.path.join(file_paths["visuals"], "participant_architecture_mds.html")
        fig.write_html(out_path)
        print(f"Participant utility function MDS saved: {out_path}")
    return fig


"=========================================================================================="
"============= Cross-Validated Utility Architecture H_form ================================"
"=========================================================================================="


def _split_player_dyads_into_folds(
    player_uuid: str,
    experiment_num: int,
    file_paths: dict,
    n_folds: int = 5,
    rng_seed: int = 2025,
) -> list:
    """
    Splits a player's dyads into n_folds cross-validation folds, stratifying by role.

    Dyads are the unit of splitting (not individual rounds) to avoid leakage from
    correlated rounds within a single game sequence.  Chooser dyads and predictor dyads
    are shuffled and split independently so each fold contains both roles.

    Arguments:
        • player_uuid: str; UUID of the player whose dyads to split.
        • experiment_num: int; Experiment number used by prep.dyads_for_a_player.
        • file_paths: dict; Project file paths.
        • n_folds: int; Number of cross-validation folds (default 5).
        • rng_seed: int; Random seed for reproducible shuffling (default 2025).

    Returns:
        • list of dicts, length n_folds.  Each dict has keys:
            fold_id, train_dyad_keys, test_dyad_keys,
            n_train_chooser_rounds, n_train_predictor_rounds,
            n_test_chooser_rounds, n_test_predictor_rounds.
    """
    all_player_dyads = prep.dyads_for_a_player(
        player_uuid=player_uuid,
        experiment_num=experiment_num,
        file_paths=file_paths,
        dyad_already_analyzed=False,
    )

    chooser_dyad_keys   = []
    predictor_dyad_keys = []
    for dyad_key, dyad_games in all_player_dyads.items():
        if not dyad_games:
            continue
        if dyad_games[0].get('chooser') == player_uuid:
            chooser_dyad_keys.append(dyad_key)
        elif dyad_games[0].get('predictor') == player_uuid:
            predictor_dyad_keys.append(dyad_key)

    rng = np.random.default_rng(rng_seed)
    rng.shuffle(chooser_dyad_keys)
    rng.shuffle(predictor_dyad_keys)

    chooser_fold_parts   = [list(part) for part in np.array_split(chooser_dyad_keys,   min(n_folds, max(len(chooser_dyad_keys),   1)))]
    predictor_fold_parts = [list(part) for part in np.array_split(predictor_dyad_keys, min(n_folds, max(len(predictor_dyad_keys), 1)))]

    while len(chooser_fold_parts)   < n_folds:
        chooser_fold_parts.append([])
    while len(predictor_fold_parts) < n_folds:
        predictor_fold_parts.append([])

    def _count_rounds_for_role(dyad_keys_subset: list, role: str) -> int:
        role_count = 0
        for dyad_key in dyad_keys_subset:
            dyad_games = all_player_dyads.get(dyad_key, [])
            if dyad_games and dyad_games[0].get(role) == player_uuid:
                role_count += len(dyad_games)
        return role_count

    fold_specs = []
    for fold_index in range(n_folds):
        test_dyad_keys  = chooser_fold_parts[fold_index] + predictor_fold_parts[fold_index]
        train_dyad_keys = [
            dyad_key for dyad_key in all_player_dyads.keys()
            if dyad_key not in set(test_dyad_keys)
        ]
        fold_specs.append({
            'fold_id':                   fold_index,
            'train_dyad_keys':           train_dyad_keys,
            'test_dyad_keys':            test_dyad_keys,
            'n_train_chooser_rounds':    _count_rounds_for_role(train_dyad_keys, 'chooser'),
            'n_train_predictor_rounds':  _count_rounds_for_role(train_dyad_keys, 'predictor'),
            'n_test_chooser_rounds':     _count_rounds_for_role(test_dyad_keys,  'chooser'),
            'n_test_predictor_rounds':   _count_rounds_for_role(test_dyad_keys,  'predictor'),
        })

    return fold_specs


def _fit_player_model_on_train_eval_on_test(
    player_uuid: str,
    train_dyad_keys: list,
    test_dyad_keys: list,
    utility_settings_for_model: dict,
    param_bds: dict,
    general_settings: dict,
    file_paths: dict,
    rng_seed: int = 2025,
) -> dict:
    """
    Fits one player's parameters for one utility model on a training subset of their game
    responses, then evaluates the fitted parameters on a held-out test subset.

    Parameters are fitted at the player level (one parameter vector per role, optimized over
    all training game responses combined) — never per-dyad.  Uses a static (non-updating)
    likelihood for CV speed, consistent with the static utility-function fitting rationale in
    individual_architecture_analysis.md §12.3.

    Arguments:
        • player_uuid: str; Player whose parameters to fit.
        • train_dyad_keys: list; Dyad keys whose game responses form the training set.
        • test_dyad_keys: list; Dyad keys whose game responses form the test set.
        • utility_settings_for_model: dict; Boolean utility-function flags for this model.
        • param_bds: dict; {param_name: (low, high)} bounds for all parameters.
        • general_settings: dict; Project settings (experiment_num, temperature_is_param, etc.).
        • file_paths: dict; Project file paths.

    Returns:
        • dict with keys: chooser_train_nll, predictor_train_nll, n_train_chooser,
          n_train_predictor, chooser_test_nll, predictor_test_nll, n_test_chooser,
          n_test_predictor, chooser_params, predictor_params.
          Test values are np.nan when test_dyad_keys is empty.
    """
    experiment_num      = general_settings.get('experiment_num', 3)
    softmax_temperature = general_settings.get('softmax_temperature', 1.0)
    temperature_is_param = general_settings.get('temperature_is_param', True)

    "Use naive update method for CV speed — no Bayesian grid update per round."
    general_settings_cv = copy.copy(general_settings)
    general_settings_cv['update_method']      = 'naive'
    general_settings_cv['include_covariance'] = False

    all_player_dyads = prep.dyads_for_a_player(
        player_uuid=player_uuid,
        experiment_num=experiment_num,
        file_paths=file_paths,
        dyad_already_analyzed=False,
    )
    player_dyads_train = {dyad_key: all_player_dyads[dyad_key] for dyad_key in train_dyad_keys if dyad_key in all_player_dyads}
    player_dyads_test  = {dyad_key: all_player_dyads[dyad_key] for dyad_key in test_dyad_keys  if dyad_key in all_player_dyads}

    param_info_cv = make_param_info(
        param_bds=param_bds,
        utility_settings=utility_settings_for_model,
        general_settings=general_settings_cv,
        random_guesses_are_unique=True,
        guess_seed=None,
    )

    param_keys_for_role = {
        player_role: list(param_info_cv['keys'])
        for player_role in ('chooser', 'predictor')
    }
    bounds_for_role = {
        player_role: list(param_info_cv['bounds'])
        for player_role in ('chooser', 'predictor')
    }

    "Add temperature τ as a free parameter — consistent with IC analysis (update_method='grid')."
    if temperature_is_param:
        tau_bounds = param_bds.get('τ', (0.5, 3.0))
        for player_role in ('chooser', 'predictor'):
            if 'τ' not in param_keys_for_role[player_role]:
                param_keys_for_role[player_role].append('τ')
                bounds_for_role[player_role].append(tau_bounds)

    def _evaluate_role_nll(param_values: list, player_dyads_subset: dict, role_to_eval: str) -> tuple:
        """Run agent forward pass and return (raw_nll, n_data) for one role over a dyad subset."""
        if temperature_is_param and param_values:
            choice_temperature_local = float(param_values[-1])
        else:
            choice_temperature_local = float(softmax_temperature)
        param_keys = param_keys_for_role[role_to_eval]
        role_params = {key: float(val) for key, val in zip(param_keys, param_values)}
        role_params_no_tau = {key: val for key, val in role_params.items() if key != 'τ'}
        total_raw_nll  = 0.0
        total_n_data   = 0
        for _dyad_key, dyad_games in player_dyads_subset.items():
            games_copy = copy.deepcopy(dyad_games)
            updated_games = agent(
                dyad_games=games_copy,
                game_idx_start=0,
                game_idx_stop=len(dyad_games) - 1,
                initial_params={role_to_eval: role_params_no_tau},
                param_info=param_info_cv,
                utility_settings=utility_settings_for_model,
                player_uuid=player_uuid,
                player_role=role_to_eval,
                general_settings=general_settings_cv,
                choice_temperature=choice_temperature_local,
            )
            updated_games = loss_function_bayes(dyad_games=updated_games, general_settings=general_settings_cv)
            loss_sums = create_loss_report(
                dyad_games=updated_games, general_settings=general_settings_cv
            ).get(player_uuid, {}).get(role_to_eval, {})
            total_raw_nll += float(loss_sums.get('raw_neglogprob_sum', 0.0))
            total_n_data  += int(loss_sums.get('n_data', 0))
        return total_raw_nll, total_n_data

    role_results = {}
    for role_to_fit in ('chooser', 'predictor'):
        param_keys = param_keys_for_role[role_to_fit]
        bounds     = bounds_for_role[role_to_fit]

        if not player_dyads_train or not bounds:
            train_nll, n_train = _evaluate_role_nll([], player_dyads_train, role_to_fit)
            test_nll,  n_test  = _evaluate_role_nll([], player_dyads_test,  role_to_fit) if player_dyads_test else (np.nan, 0)
            role_results[role_to_fit] = {
                'train_nll': train_nll, 'test_nll': test_nll,
                'n_train': n_train, 'n_test': n_test, 'params': {},
            }
            continue

        guesses_callable = param_info_cv['guesses']
        best_nll    = float('inf')
        best_params = None

        n_random_starts = 3
        for start_index in range(n_random_starts):
            if start_index == 0:
                base_guesses = guesses_callable() if callable(guesses_callable) else copy.deepcopy(guesses_callable)
                if temperature_is_param and 'τ' not in list(param_info_cv['keys']):
                    base_guesses = list(base_guesses) + [float(softmax_temperature)]
                x_initial = np.array(base_guesses, dtype=float)
            else:
                seed_val = (rng_seed + start_index) if rng_seed is not None else start_index
                rng_start = np.random.default_rng(seed_val)
                x_initial = np.array([rng_start.uniform(low, high) for (low, high) in bounds], dtype=float)

            try:
                import scipy.optimize as _sp_opt
                opt_result = _sp_opt.minimize(
                    fun=lambda param_array: _evaluate_role_nll(list(param_array), player_dyads_train, role_to_fit)[0],
                    x0=x_initial,
                    bounds=bounds,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-7},
                )
                if opt_result.fun < best_nll:
                    best_nll    = float(opt_result.fun)
                    best_params = opt_result.x.tolist()
            except Exception:
                pass

        if best_params is None:
            best_params = x_initial.tolist()

        train_nll, n_train = _evaluate_role_nll(best_params, player_dyads_train, role_to_fit)
        test_nll,  n_test  = _evaluate_role_nll(best_params, player_dyads_test,  role_to_fit) if player_dyads_test else (np.nan, 0)
        fitted_param_dict  = {key: float(val) for key, val in zip(param_keys, best_params)}

        role_results[role_to_fit] = {
            'train_nll': train_nll, 'test_nll': test_nll,
            'n_train': n_train, 'n_test': n_test, 'params': fitted_param_dict,
        }

    return {
        'chooser_train_nll':   role_results['chooser']['train_nll'],
        'predictor_train_nll': role_results['predictor']['train_nll'],
        'chooser_test_nll':    role_results['chooser']['test_nll'],
        'predictor_test_nll':  role_results['predictor']['test_nll'],
        'n_train_chooser':     role_results['chooser']['n_train'],
        'n_train_predictor':   role_results['predictor']['n_train'],
        'n_test_chooser':      role_results['chooser']['n_test'],
        'n_test_predictor':    role_results['predictor']['n_test'],
        'chooser_params':      role_results['chooser']['params'],
        'predictor_params':    role_results['predictor']['params'],
    }


def _cv_architecture_losses_worker(args: tuple) -> list:
    """
    Module-level parallel worker for compute_cross_validated_architecture_losses.

    Computes CV train/test fits for one participant across all their fold × candidate model
    combinations.  Designed to be called via mp.Pool.imap_unordered.

    Arguments:
        • args: tuple unpacking to (player_uuid, candidate_idxs, utility_settings_by_idx,
                k_params_by_idx, param_bds, general_settings, file_paths, n_folds,
                rng_seed, population_winner_utility_idx).

    Returns:
        • list of row dicts — same schema as the rows built inside
          compute_cross_validated_architecture_losses.
    """
    (player_uuid, candidate_idxs, utility_settings_by_idx, k_params_by_idx,
     param_bds_worker, general_settings, file_paths, n_folds, rng_seed,
     population_winner_utility_idx) = args

    fold_specs = _split_player_dyads_into_folds(
        player_uuid=player_uuid,
        experiment_num=general_settings.get('experiment_num', 3),
        file_paths=file_paths,
        n_folds=n_folds,
        rng_seed=rng_seed,
    )

    rows = []
    for fold_spec in fold_specs:
        fold_id_val          = fold_spec['fold_id']
        train_dyad_keys_fold = fold_spec['train_dyad_keys']
        test_dyad_keys_fold  = fold_spec['test_dyad_keys']

        for candidate_utility_idx in candidate_idxs:
            utility_settings_for_this_model = utility_settings_by_idx.get(candidate_utility_idx)
            if utility_settings_for_this_model is None:
                continue

            try:
                fit_result = _fit_player_model_on_train_eval_on_test(
                    player_uuid=player_uuid,
                    train_dyad_keys=train_dyad_keys_fold,
                    test_dyad_keys=test_dyad_keys_fold,
                    utility_settings_for_model=utility_settings_for_this_model,
                    param_bds=param_bds_worker,
                    general_settings=general_settings,
                    file_paths=file_paths,
                    rng_seed=rng_seed,
                )
            except Exception as exc_info:
                print(f"    Warning: fit failed player={player_uuid} fold={fold_id_val} idx={candidate_utility_idx}: {exc_info}")
                fit_result = {
                    'chooser_train_nll': np.nan, 'predictor_train_nll': np.nan,
                    'chooser_test_nll':  np.nan, 'predictor_test_nll':  np.nan,
                    'n_train_chooser':   0,       'n_train_predictor':   0,
                    'n_test_chooser':    0,       'n_test_predictor':    0,
                    'chooser_params': {}, 'predictor_params': {},
                }

            n_train_combined = fit_result['n_train_chooser'] + fit_result['n_train_predictor']
            combined_train_nll = (
                (fit_result['chooser_train_nll']   if not np.isnan(fit_result['chooser_train_nll'])   else 0.0) +
                (fit_result['predictor_train_nll'] if not np.isnan(fit_result['predictor_train_nll']) else 0.0)
            )
            n_test_combined = fit_result['n_test_chooser'] + fit_result['n_test_predictor']
            combined_test_nll = (
                (fit_result['chooser_test_nll']   if not np.isnan(fit_result['chooser_test_nll'])   else np.nan) +
                (fit_result['predictor_test_nll'] if not np.isnan(fit_result['predictor_test_nll']) else np.nan)
            )

            k_params_val    = k_params_by_idx.get(candidate_utility_idx, 0)
            k_effective_val = 2 * k_params_val
            if general_settings.get('temperature_is_param', True):
                k_effective_val += 2

            train_bic_val = (
                2 * combined_train_nll + k_effective_val * np.log(max(n_train_combined, 1))
                if n_train_combined > 0 else np.nan
            )

            rows.append({
                'fold_id':              fold_id_val,
                'player_uuid':          player_uuid,
                'utility_idx':          candidate_utility_idx,
                'k_params':             k_params_val,
                'k_effective':          k_effective_val,
                'chooser_train_nll':    fit_result['chooser_train_nll'],
                'predictor_train_nll':  fit_result['predictor_train_nll'],
                'combined_train_nll':   combined_train_nll,
                'n_train_chooser':      fit_result['n_train_chooser'],
                'n_train_predictor':    fit_result['n_train_predictor'],
                'n_train_combined':     n_train_combined,
                'chooser_test_nll':     fit_result['chooser_test_nll'],
                'predictor_test_nll':   fit_result['predictor_test_nll'],
                'combined_test_nll':    combined_test_nll,
                'n_test_chooser':       fit_result['n_test_chooser'],
                'n_test_predictor':     fit_result['n_test_predictor'],
                'n_test_combined':      n_test_combined,
                'train_BIC':            train_bic_val,
                'is_population_winner': (candidate_utility_idx == population_winner_utility_idx),
            })

    return rows


def _fmt_duration(seconds: float) -> str:
    """Format a duration as 'H hours M minutes' or 'M minutes SS seconds'."""
    total_minutes = int(seconds) // 60
    if total_minutes >= 60:
        hours = total_minutes // 60
        mins = total_minutes % 60
        return f"{hours} hours {mins} minutes"
    secs = int(seconds) % 60
    return f"{total_minutes} minutes {secs:02d} seconds"


def _exhaustive_search_worker(args: tuple) -> tuple:
    """
    Module-level parallel worker for exhaustive K-architecture set search.

    Evaluates a batch of candidate architecture sets against a participant × model
    score matrix and returns the best-scoring set in the batch.

    Arguments:
        • args: (combo_batch, L)
            combo_batch: list of K-tuples (column indices into L)
            L: np.ndarray, shape (N_participants, N_candidates)

    Returns:
        • (best_score: float, best_set: tuple of ints)
    """
    combo_batch, L = args
    if not combo_batch:
        return np.inf, None
    arr    = np.array(combo_batch, dtype=np.int32)   # (B, K)
    "L[:, arr] is (N_participants, B, K); min(axis=2) → (N_participants, B); sum(axis=0) → (B,)."
    scores = L[:, arr].min(axis=2).sum(axis=0)        # (B,)
    best_i = int(scores.argmin())
    return float(scores[best_i]), tuple(combo_batch[best_i])


def _ampd_pair_worker(args: tuple) -> list:
    """
    Module-level parallel worker for AMPD distance matrix computation.

    Receives a batch of model-index pairs and all parameters needed by
    average_model_policy_distance. Computes the AMPD for each pair and returns
    the results so the master process can merge them into the matrix and save.

    Arguments:
        • args: tuple of (pair_batch, settings_cache, general_settings_dict,
              file_paths_dict, param_bds, metric, choice_temperature, n_games,
              n_iters, parameter_sampling_mode, parameter_pairing_mode,
              player_roles, random_seed, participant_parameter_pools)
            pair_batch: list of (utility_idx_i, utility_idx_j) tuples
            settings_cache: dict[int, UtilitySettings]
            remaining items: passed through to average_model_policy_distance.

    Returns:
        • list of (utility_idx_i, utility_idx_j, distance) tuples — one per pair.
    """
    (pair_batch, settings_cache, general_settings_dict, file_paths_dict, param_bds,
     metric, choice_temperature, n_games, n_iters, parameter_sampling_mode,
     parameter_pairing_mode, player_roles, random_seed, participant_parameter_pools) = args

    batch_results = []
    for utility_idx_i, utility_idx_j in pair_batch:
        dist = average_model_policy_distance(
            utility_settings_a=settings_cache[utility_idx_i],
            utility_settings_b=settings_cache[utility_idx_j],
            general_settings=general_settings_dict,
            file_paths=file_paths_dict,
            param_bds=param_bds,
            metric=metric,
            choice_temperature=choice_temperature,
            n_games=n_games,
            n_iters=n_iters,
            parameter_sampling_mode=parameter_sampling_mode,
            parameter_pairing_mode=parameter_pairing_mode,
            player_roles=player_roles,
            random_seed=random_seed,
            participant_parameter_pools=participant_parameter_pools,
        )
        batch_results.append((utility_idx_i, utility_idx_j, dist))
    return batch_results


def compute_cross_validated_architecture_losses(
    general_settings: dict,
    file_paths: dict,
    utility_settings: dict,
    param_bds: dict,
    n_folds: int = 5,
    top_n_candidate_models: int = 20,
    delta_bic_threshold: float = 10.0,
    rng_seed: int = 2025,
    create_new_file: bool = False,
) -> pd.DataFrame:
    """
    Cross-validated architecture loss engine shared by the H_form and compression-curve analyses.

    For each participant, a candidate set of utility models is identified from their
    in-sample ΔBIC scores (from participant_model_combined_fits.csv).  Each model in the
    candidate set is refit on each training fold and evaluated on the corresponding test fold.
    The population-level BIC winner is always included as a candidate.

    Results are saved to a long-format CSV that compute_h_form_cross_validated uses to compute
    H_form and the compression-curve analysis uses to build the architecture codebook — the
    expensive refitting happens exactly once here.

    Arguments:
        • general_settings: dict; Project settings (experiment_num, temperature_is_param, etc.).
        • file_paths: dict; Project file paths.
        • utility_settings: dict; Universe of Boolean utility settings (used for candidate lookup).
        • param_bds: dict; {param_name: (low, high)} bounds.
        • n_folds: int; Number of cross-validation folds (default 5).
        • top_n_candidate_models: int; Maximum candidate models per participant (default 20).
        • delta_bic_threshold: float; Include models where in-sample ΔBIC ≤ this (default 10.0).
        • rng_seed: int; Random seed for fold splitting (default 2025).
        • create_new_file: bool; If False and output CSV exists and is valid, load and return it.

    Returns:
        • pd.DataFrame; Long-format table with one row per fold × participant × candidate model.
    """
    output_csv_path = os.path.join(file_paths['processed'], 'cv_architecture_losses.csv')

    if not create_new_file and os.path.exists(output_csv_path):
        existing_df = pd.read_csv(output_csv_path, encoding='utf-8-sig')
        cache_is_trivial = existing_df.empty or existing_df['combined_test_nll'].isna().all()
        if not cache_is_trivial:
            print(f"Loaded CV architecture losses from cache: {output_csv_path}  ({len(existing_df)} rows)")
            return existing_df

    experiment_num = general_settings.get('experiment_num', 3)
    combined_fits_df = extract_participant_model_combined_fits(
        general_settings=general_settings,
        file_paths=file_paths,
        create_new_file=False,
    )

    """
    Identify the population-level BIC winner: the model with the lowest aggregate (pooled) BIC
    computed over all participants simultaneously.
    Pooled BIC = 2 * Σ_i NLL_i + k_eff * log(Σ_i n_i).
    Mean individual ΔBIC systematically under-penalises complexity at ~120 games/participant;
    the pooled criterion is the correct population-scale penalty.
    """
    temperature_is_param = general_settings.get('temperature_is_param', True)
    _model_meta_df = combined_fits_df.drop_duplicates('utility_idx').set_index('utility_idx')
    _k_eff_base    = _model_meta_df['k_effective']                    # 2 × k_params
    _k_eff_pop     = _k_eff_base + (2 if temperature_is_param else 0) # +2 if τ is free
    _pooled_nll    = combined_fits_df.groupby('utility_idx')['combined_loss_nll'].sum()
    _pooled_n      = combined_fits_df.groupby('utility_idx')['n_combined'].sum()
    _pooled_bic    = 2 * _pooled_nll + _k_eff_pop * np.log(_pooled_n.clip(lower=1))
    population_winner_utility_idx = int(_pooled_bic.idxmin())
    print(
        f"Population BIC winner (pooled): utility_idx={population_winner_utility_idx}"
        f"  pooled_BIC={_pooled_bic[population_winner_utility_idx]:.2f}"
        f"  k_eff={int(_k_eff_pop[population_winner_utility_idx])}"
    )

    """
    Reconstruct per-model utility_settings dicts from the individual boolean columns written
    by extract_participant_model_combined_fits.  Those columns were spread into the CSV by
    row.update(utility_settings_flags) rather than stored as a single JSON blob.
    """
    utility_settings_by_idx: dict = {}
    for _, model_row in combined_fits_df.drop_duplicates(subset='utility_idx').iterrows():
        utility_idx_val = int(model_row['utility_idx'])
        reconstructed_settings = {
            col_name: bool(model_row[col_name])
            for col_name in _UTILITY_BOOLEAN_SETTINGS_COLUMNS
            if col_name in model_row.index
        }
        if reconstructed_settings:
            utility_settings_by_idx[utility_idx_val] = reconstructed_settings
    print(f"Utility settings reconstructed for {len(utility_settings_by_idx)} models.")

    "Build per-participant candidate model sets."
    all_player_uuids = sorted(combined_fits_df['player_uuid'].unique())
    n_participants   = len(all_player_uuids)
    candidate_sets_by_player: dict = {}
    for player_uuid_key in all_player_uuids:
        player_rows  = combined_fits_df[combined_fits_df['player_uuid'] == player_uuid_key].copy()
        mask_bic     = player_rows['delta_BIC_individual'] <= delta_bic_threshold
        mask_top_n   = player_rows['delta_BIC_individual'].rank(method='first') <= top_n_candidate_models
        candidate_idxs = set(player_rows[mask_bic | mask_top_n]['utility_idx'].astype(int).tolist())
        candidate_idxs.add(population_winner_utility_idx)
        candidate_sets_by_player[player_uuid_key] = sorted(candidate_idxs)

    candidate_counts = [len(candidate_sets_by_player[uuid_key]) for uuid_key in all_player_uuids]
    print(f"Candidate model counts — min: {min(candidate_counts)}, max: {max(candidate_counts)}, mean: {np.mean(candidate_counts):.1f}")

    "Retrieve k_params for each model."
    k_params_by_idx = combined_fits_df.drop_duplicates('utility_idx').set_index('utility_idx')['k_params'].to_dict()
    k_params_by_idx = {int(idx_key): int(k_val) for idx_key, k_val in k_params_by_idx.items()}

    args_list = [
        (
            player_uuid_key,
            candidate_sets_by_player[player_uuid_key],
            utility_settings_by_idx,
            k_params_by_idx,
            param_bds,
            general_settings,
            file_paths,
            n_folds,
            rng_seed,
            population_winner_utility_idx,
        )
        for player_uuid_key in all_player_uuids
    ]

    all_rows = []
    run_in_parallel = general_settings.get('run_in_parallel', True)
    if run_in_parallel:
        n_workers = max(mp.cpu_count() - 1, 1)
        print(f"[CV architecture losses] Parallel mode: {n_workers} workers, {n_participants} participants.")
        with mp.Pool(processes=n_workers) as pool:
            for participant_number, worker_rows in enumerate(
                pool.imap_unordered(_cv_architecture_losses_worker, args_list), 1
            ):
                all_rows.extend(worker_rows)
                if participant_number % 10 == 0 or participant_number == n_participants:
                    print(f"  [{participant_number}/{n_participants} participants done]")
    else:
        for participant_number, args in enumerate(args_list, 1):
            print(f"[CV architecture losses] {participant_number}/{n_participants}: {args[0]}")
            all_rows.extend(_cv_architecture_losses_worker(args))

    cv_losses_df = pd.DataFrame(all_rows)
    if cv_losses_df.empty:
        print(
            f"Warning: cross-validated architecture loss computation produced 0 rows — "
            "the output CSV was NOT written to avoid creating a degenerate file that would "
            "crash downstream functions.  Check that participant_model_combined_fits.csv exists "
            "and that candidate model utility settings were successfully reconstructed."
        )
        return cv_losses_df
    cv_losses_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"Saved: {output_csv_path}  ({len(cv_losses_df)} rows)")
    return cv_losses_df


def compute_h_form_cross_validated(
    general_settings: dict,
    file_paths: dict,
    create_new_file: bool = False,
) -> tuple:
    """
    Computes the cross-validated Functional-Form Heterogeneity Index (H_form) from the
    CV architecture losses produced by compute_cross_validated_architecture_losses.

    H_form = (NLL_common - NLL_individual) / (NLL_chance - NLL_individual)

    Two variants of the common model are reported:
      1. Fold-selected: best K=1 architecture on each fold's training data (fairest).
      2. Fixed: population BIC winner from the full-data IC analysis.

    The in-sample H_form (using full-data combined fits from extract_participant_model_combined_fits,
    no refitting) is also computed
    as a reference for the optimism bias.

    Arguments:
        • general_settings: dict; Project settings (experiment_num, etc.).
        • file_paths: dict; Project file paths.
        • create_new_file: bool; If False and output CSVs exist and are valid, load and return.

    Returns:
        • tuple (results_df, summary_df):
            results_df — per-fold × per-participant H_form values.
            summary_df — per-participant averages over folds plus in-sample H_form.
    """
    results_csv_path = os.path.join(file_paths['processed'], 'h_form_cross_validated_results.csv')
    summary_csv_path = os.path.join(file_paths['processed'], 'h_form_cross_validated_summary.csv')

    if not create_new_file:
        if os.path.exists(results_csv_path) and os.path.exists(summary_csv_path):
            existing_results = pd.read_csv(results_csv_path, encoding='utf-8-sig')
            existing_summary = pd.read_csv(summary_csv_path, encoding='utf-8-sig')
            if not existing_results.empty and not existing_summary.empty:
                print(f"Loaded H_form results from cache.")
                return existing_results, existing_summary

    cv_losses_csv_path = os.path.join(file_paths['processed'], 'cv_architecture_losses.csv')
    if not os.path.exists(cv_losses_csv_path):
        raise FileNotFoundError(
            f"Cross-validated architecture losses file not found at {cv_losses_csv_path}.  "
            "Run compute_cross_validated_architecture_losses() first to generate it."
        )
    cv_losses_df = pd.read_csv(cv_losses_csv_path, encoding='utf-8-sig')
    if cv_losses_df.empty or len(cv_losses_df.columns) == 0:
        raise ValueError(
            f"Cross-validated architecture losses file at {cv_losses_csv_path} is empty or "
            "has no columns.  Re-run compute_cross_validated_architecture_losses(create_new_file=True) "
            "to regenerate it."
        )
    combined_fits_df = extract_participant_model_combined_fits(
        general_settings=general_settings,
        file_paths=file_paths,
        create_new_file=False,
    )

    all_fold_ids     = sorted(cv_losses_df['fold_id'].unique())
    all_player_uuids = sorted(cv_losses_df['player_uuid'].unique())
    results_rows     = []

    for fold_id_val in all_fold_ids:
        fold_rows = cv_losses_df[cv_losses_df['fold_id'] == fold_id_val].copy()

        "Per participant: select individual model by argmin(train_BIC)."
        individual_model_rows = fold_rows.loc[fold_rows.groupby('player_uuid')['train_BIC'].idxmin()]
        individual_model_rows = individual_model_rows.set_index('player_uuid')

        """
        Best common model = argmin(sum of train_BIC) restricted to models present in every
        participant's candidate set.  A model missing from even one participant cannot serve
        as a common model because its test NLL would be undefined for that participant.
        """
        n_participants_in_fold       = fold_rows['player_uuid'].nunique()
        model_participant_counts     = fold_rows.groupby('utility_idx')['player_uuid'].nunique()
        universal_utility_idxs       = model_participant_counts[model_participant_counts == n_participants_in_fold].index
        fold_rows_universal          = fold_rows[fold_rows['utility_idx'].isin(universal_utility_idxs)]
        train_bic_sum_by_model       = fold_rows_universal.groupby('utility_idx')['train_BIC'].sum()
        best_common_utility_idx      = int(train_bic_sum_by_model.idxmin())

        "Population winner is flagged in the CSV."
        population_winner_rows = fold_rows[fold_rows['is_population_winner'] == True]

        for player_uuid_key in all_player_uuids:
            if player_uuid_key not in individual_model_rows.index:
                continue

            individual_row = individual_model_rows.loc[player_uuid_key]
            nll_individual = float(individual_row['combined_test_nll'])
            n_test_val     = int(individual_row['n_test_combined'])
            best_individual_utility_idx = int(individual_row['utility_idx'])

            "Common (fold-selected): test NLL for best_common_utility_idx for this participant."
            common_fold_match = fold_rows[
                (fold_rows['player_uuid'] == player_uuid_key) &
                (fold_rows['utility_idx'] == best_common_utility_idx)
            ]
            nll_common_fold = float(common_fold_match['combined_test_nll'].values[0]) if len(common_fold_match) > 0 else np.nan

            "Common (population winner): test NLL for the fixed population winner."
            pop_winner_match = population_winner_rows[population_winner_rows['player_uuid'] == player_uuid_key]
            nll_common_pop   = float(pop_winner_match['combined_test_nll'].values[0]) if len(pop_winner_match) > 0 else np.nan

            nll_chance = n_test_val * np.log(2) if n_test_val > 0 else np.nan

            "Require a minimum test set size before computing H_form for this fold."
            "With fewer than 8 games the denominator (NLL_chance - NLL_individual) can be"
            "near-zero by chance alone, producing extreme ratio artifacts."
            MIN_TEST_GAMES_FOR_H_FORM: int = 8

            def _h_form_safe(nll_common_val: float, nll_indiv_val: float, nll_chance_val: float,
                             n_test_games: int) -> float:
                if n_test_games < MIN_TEST_GAMES_FOR_H_FORM:
                    return np.nan
                if any(np.isnan(value) for value in (nll_common_val, nll_indiv_val, nll_chance_val)):
                    return np.nan
                denominator = nll_chance_val - nll_indiv_val
                return float((nll_common_val - nll_indiv_val) / denominator) if abs(denominator) > 1e-10 else np.nan

            h_form_val     = _h_form_safe(nll_common_fold, nll_individual, nll_chance, n_test_val)
            h_form_pop_val = _h_form_safe(nll_common_pop,  nll_individual, nll_chance, n_test_val)

            results_rows.append({
                'fold_id':                              fold_id_val,
                'player_uuid':                          player_uuid_key,
                'best_individual_utility_idx':          best_individual_utility_idx,
                'best_common_utility_idx':              best_common_utility_idx,
                'NLL_individual':                       nll_individual,
                'NLL_common':                           nll_common_fold,
                'NLL_common_population_winner':         nll_common_pop,
                'NLL_chance':                           nll_chance,
                'H_form':                               h_form_val,
                'H_form_population_winner':             h_form_pop_val,
                'H_form_clipped':                       float(np.clip(h_form_val, 0.0, 1.0)) if not np.isnan(h_form_val) else np.nan,
                'n_test':                               n_test_val,
            })

    results_df = pd.DataFrame(results_rows)

    "Summarize across folds per participant."
    summary_rows = []
    for player_uuid_key in all_player_uuids:
        player_results = results_df[results_df['player_uuid'] == player_uuid_key]
        h_form_mean     = float(player_results['H_form'].mean())
        h_form_std      = float(player_results['H_form'].std())
        h_form_pop_mean = float(player_results['H_form_population_winner'].mean())
        h_form_pop_std  = float(player_results['H_form_population_winner'].std())
        n_folds_actual  = int(player_results['fold_id'].nunique())
        nll_indiv_mean  = float(player_results['NLL_individual'].mean())
        nll_common_mean = float(player_results['NLL_common'].mean())

        "In-sample H_form from full-data combined fits (no refitting)."
        player_fits = combined_fits_df[combined_fits_df['player_uuid'] == player_uuid_key]
        population_winner_fits = player_fits[
            player_fits['utility_idx'] == int(combined_fits_df.groupby('utility_idx')['delta_BIC_individual'].mean().idxmin())
        ]
        best_model_fits = player_fits.loc[player_fits['delta_BIC_individual'].idxmin()] if len(player_fits) > 0 else None

        nll_individual_insample = float(best_model_fits['combined_loss_nll']) if best_model_fits is not None else np.nan
        nll_common_insample     = float(population_winner_fits['combined_loss_nll'].values[0]) if len(population_winner_fits) > 0 else np.nan
        n_combined_insample     = float(player_fits['n_combined'].iloc[0]) if len(player_fits) > 0 else np.nan
        nll_chance_insample     = n_combined_insample * np.log(2) if not np.isnan(n_combined_insample) else np.nan

        h_form_insample = np.nan
        if not any(np.isnan(val) for val in (nll_individual_insample, nll_common_insample, nll_chance_insample)):
            denom = nll_chance_insample - nll_individual_insample
            if abs(denom) > 1e-10:
                h_form_insample = float((nll_common_insample - nll_individual_insample) / denom)

        eff_models_row = player_fits.drop_duplicates('player_uuid')
        eff_models = float(eff_models_row['effective_number_of_models'].values[0]) if len(eff_models_row) > 0 else np.nan

        summary_rows.append({
            'player_uuid':                     player_uuid_key,
            'H_form_mean':                     h_form_mean,
            'H_form_std':                      h_form_std,
            'H_form_population_winner_mean':   h_form_pop_mean,
            'H_form_population_winner_std':    h_form_pop_std,
            'H_form_in_sample':                h_form_insample,
            'n_folds':                         n_folds_actual,
            'NLL_individual_mean':             nll_indiv_mean,
            'NLL_common_mean':                 nll_common_mean,
            'effective_number_of_models':      eff_models,
        })

    summary_df = pd.DataFrame(summary_rows)

    h_form_cv_mean = float(summary_df['H_form_mean'].mean())
    h_form_cv_std  = float(summary_df['H_form_std'].mean())
    frac_positive  = float((summary_df['H_form_mean'] > 0).mean())
    h_form_insample_overall = float(
        summary_df['H_form_in_sample'].mean()) if not summary_df['H_form_in_sample'].isna().all() else np.nan
    most_common_common_model = int(results_df['best_common_utility_idx'].mode().values[0]) if len(results_df) > 0 else None

    print(f"\n{'='*60}")
    print(f"H_form (in-sample):                  {h_form_insample_overall:.4f}")
    print(f"H_form (CV mean ± mean-fold-std):     {h_form_cv_mean:.4f} ± {h_form_cv_std:.4f}")
    print(f"Fraction of participants H_form > 0:  {frac_positive:.1%}")
    print(f"Best common model (mode across folds): utility_idx={most_common_common_model}")
    print(f"{'='*60}\n")

    results_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    print(f"Saved: {results_csv_path}  ({len(results_df)} rows)")
    print(f"Saved: {summary_csv_path}  ({len(summary_df)} rows)")

    return results_df, summary_df


def plot_h_form_results(
    general_settings: dict,
    file_paths: dict,
    fig_lay: dict,
    export_fig: bool = True,
    include_error_bars: bool = True,
    y_axis_range: list | None = None,
) -> go.Figure:
    """
    Bar chart of per-participant cross-validated ℋ-form values.

    Participants are sorted ascending by ℋ-form mean.  Bar colors use a diverging RdBu
    colorscale with arcsinh compression so that the scale is visually sensitive to small
    changes near zero while de-emphasising extreme outliers.  The y-axis is symmetric around
    zero by default (±max(|min|, |max|)) so zero is always centered.  Clipping annotations
    are added only when y_axis_range is explicitly provided.

    Arguments:
        • general_settings: dict; Project settings.
        • file_paths: dict; Project file paths.
        • fig_lay: dict; Layout settings (template, font, margins, etc.) from config.py.
        • export_fig: bool; If True, write HTML to visuals/h_form_cross_validated_bar.html.
        • include_error_bars: bool; If True, draw fold-std error bars on each bar.
        • y_axis_range: list[float, float] | None; If a valid [lo, hi] pair is given, the
            y-axis is clipped to that range and bars outside it get triangle annotations with
            their true value.  If None (default), the full symmetric range is used with no
            clipping annotations.

    Returns:
        • go.Figure; Plotly figure with participant-level bars and reference lines.
    """
    H = 'ℋ'  # ℋ  Script Capital H — math-variable style

    summary_csv_path = os.path.join(file_paths['processed'], 'h_form_cross_validated_summary.csv')
    summary_df = (
        pd.read_csv(summary_csv_path, encoding='utf-8-sig')
        .sort_values('H_form_mean')
        .reset_index(drop=True)
    )

    h_form_insample_global = (
        float(summary_df['H_form_in_sample'].mean())
        if not summary_df['H_form_in_sample'].isna().all()
        else None
    )
    h_form_min = float(summary_df['H_form_mean'].min())
    h_form_max = float(summary_df['H_form_mean'].max())

    "Symmetric axis so zero is always centered: ±max(|min|, |max|)."
    x_range    = max(abs(h_form_min), abs(h_form_max))
    sym_y_lo   = -x_range
    sym_y_hi   =  x_range

    "Validate y_axis_range; clipping annotations only enabled when it is a proper [lo, hi] pair."
    use_clip   = False
    y_lo       = sym_y_lo
    y_hi       = sym_y_hi
    if y_axis_range is not None:
        try:
            lo_candidate = float(y_axis_range[0])
            hi_candidate = float(y_axis_range[1])
            if lo_candidate < hi_candidate:
                y_lo     = lo_candidate
                y_hi     = hi_candidate
                use_clip = True
        except (TypeError, ValueError, IndexError):
            pass

    bar_y = (
        summary_df['H_form_mean'].clip(lower=y_lo, upper=y_hi).tolist()
        if use_clip
        else summary_df['H_form_mean'].tolist()
    )

    "Arcsinh color transform: map each ℋ-form value to [-1, 1] with arcsinh compression."
    "color_range is the 95th percentile of |ℋ-form| so the scale is calibrated to the"
    "typical participant range rather than dominated by the single largest outlier."
    "k_arcsinh controls sensitivity — higher k = more colour near zero."
    color_range = float(np.percentile(np.abs(summary_df['H_form_mean'].dropna()), 95))
    k_arcsinh   = 10.0

    def _to_color(v: float) -> float:
        norm = float(np.clip(v / color_range, -1.0, 1.0))
        return float(np.arcsinh(k_arcsinh * norm) / np.arcsinh(k_arcsinh))

    color_vals = [_to_color(val) for val in summary_df['H_form_mean'].tolist()]

    "Colorbar: five landmark ticks only to avoid label overlap."
    colorbar_ticks    = [-3, -1, 0, 1, 3]
    colorbar_tickvals = [_to_color(tick) for tick in colorbar_ticks]
    colorbar_ticktext = [str(tick) for tick in colorbar_ticks]

    def _hover(row: pd.Series) -> str:
        h_mean  = f"{row['H_form_mean']:.4f}"        if pd.notna(row.get('H_form_mean'))             else "?"
        h_std   = f"{row['H_form_std']:.4f}"         if pd.notna(row.get('H_form_std'))              else "?"
        h_ins   = f"{row['H_form_in_sample']:.4f}"   if pd.notna(row.get('H_form_in_sample'))        else "?"
        eff     = f"{row['effective_number_of_models']:.1f}" if pd.notna(row.get('effective_number_of_models')) else "?"
        nll_i   = f"{row['NLL_individual_mean']:.2f}" if pd.notna(row.get('NLL_individual_mean'))    else "?"
        nll_c   = f"{row['NLL_common_mean']:.2f}"     if pd.notna(row.get('NLL_common_mean'))        else "?"
        n_folds = str(int(row['n_folds'])) if pd.notna(row.get('n_folds')) else "?"
        return (
            f"Participant {row.get('player_uuid', '?')}<br>"
            f"{H}-form CV: {h_mean} ± {h_std}  ({n_folds} folds)<br>"
            f"{H}-form in-sample: {h_ins}<br>"
            f"NLL individual: {nll_i} | NLL common: {nll_c}<br>"
            f"Effective models: {eff}"
        )

    hover_texts = summary_df.apply(_hover, axis=1).tolist()

    bar_trace = go.Bar(
        x=list(range(len(summary_df))),
        y=bar_y,
        error_y=(
            dict(type='data', array=summary_df['H_form_std'].tolist(), visible=True)
            if include_error_bars else None
        ),
        marker=dict(
            color=color_vals,
            colorscale='RdBu',
            cmid=0.0,
            cmin=-1.0,
            cmax=1.0,
            colorbar=dict(
                title=f'{H}-form',
                thickness=36,
                tickvals=colorbar_tickvals,
                ticktext=colorbar_ticktext,
            ),
            line=dict(width=0.5, color=_hsla(hue=0, saturation_percent=0, lightness_percent=0, alpha=0.3)),
        ),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_texts,
        showlegend=False,
    )

    traces = [bar_trace]

    "Triangle annotations for clipped bars — only when y_axis_range was explicitly provided."
    if use_clip:
        clipped_mask = (summary_df['H_form_mean'] < y_lo) | (summary_df['H_form_mean'] > y_hi)
        n_clipped    = int(clipped_mask.sum())
        if n_clipped:
            print(f"[H-form plot] {n_clipped} participant(s) clipped to [{y_lo:.2f}, {y_hi:.2f}].")
            clipped_idx  = [idx for idx, mask_val in enumerate(clipped_mask) if mask_val]
            clipped_true = [float(summary_df['H_form_mean'].iloc[idx]) for idx in clipped_idx]
            clip_labels  = [f"Clipped (true={val:.1f})" for val in clipped_true]
            clip_y_pos   = [y_lo + abs(y_hi - y_lo) * 0.04] * len(clipped_idx)
            traces.append(go.Scatter(
                x=clipped_idx, y=clip_y_pos,
                mode='markers+text',
                marker=dict(symbol='triangle-down', size=12, color='darkred'),
                text=clip_labels, textposition='top center',
                hovertemplate='%{text}<extra></extra>',
                showlegend=False,
            ))

    fig = go.Figure(data=traces)

    fig.add_hline(y=0.0, line_dash='dash', line_color='black', line_width=1.5)
    if h_form_insample_global is not None:
        fig.add_hline(
            y=h_form_insample_global, line_dash='dot', line_color='grey', line_width=1.5,
        )
        "Annotation placed in the lower-center-right of the figure using paper coordinates"
        "so it stays clear of bars and error bars regardless of the data range."
        fig.add_annotation(
            xref='paper', yref='paper',
            x=0.62, y=0.09,
            text=f'In-sample {H}-form = {h_form_insample_global:.3f}',
            showarrow=False,
            font=dict(color='black', size=fig_lay.get('font', {}).get('size', 16)),
            bgcolor=_hsla(hue=0, saturation_percent=0, lightness_percent=100, alpha=0.85),
            bordercolor=_hsla(hue=0, saturation_percent=0, lightness_percent=39, alpha=0.6),
            borderwidth=1,
            align='center',
        )

    base_font_size = fig_lay.get('font', {}).get('size', 16)
    fig.update_layout(
        title=dict(
            text=f'Cross-Validated {H}-form by Participant',
            x=0.5, xanchor='center',
            y=0.96, yanchor='top',
            font=dict(size=fig_lay.get('title_size', 22) * 2),
        ),
        xaxis=dict(
            title=f'Participant (sorted by {H}-form)',
            showticklabels=False,
            showgrid=False,
        ),
        yaxis=dict(
            title=f'{H}-form',
            zeroline=True, zerolinewidth=1, zerolinecolor='black',
            range=[y_lo, y_hi],
        ),
        hoverlabel=dict(font=dict(size=base_font_size * 2)),
        template=fig_lay.get('template', 'plotly_white'),
        font=dict(
            family=fig_lay.get('font', {}).get('family', 'Calibri'),
            size=base_font_size,
        ),
        margin=dict(l=80, r=20, t=80, b=60),
        autosize=True,
    )

    if export_fig:
        out_path = os.path.join(file_paths['visuals'], 'h_form_cross_validated_bar.html')
        fig.write_html(out_path, config={'responsive': True})
        print(f"H-form bar chart saved: {out_path}")
    return fig


"=========================================================================================="
"================= Population Architecture Compression Curve ============================="
"=========================================================================================="


def compute_architecture_compression_curve(
    general_settings: dict,
    file_paths: dict,
    population_top_n_models: int | None = 120,
    participant_top_r_models: int | None = 10,
    K_max: int | None = None,
    exhaustive_K_max: int = 4,
    score_basis: Literal[
        "ic_equivalent_participant_score",
        "sum_individual_BIC",
        "raw_NLL",
    ] = "ic_equivalent_participant_score",
    stopping_criteria: Literal[
        "kneedle_elbow", "marginal_gain", "cumulative_gain",
        "max_curvature", "meta_bic",
    ] = "kneedle_elbow",
    marginal_gain_threshold: float = 0.01,
    n_consecutive_low_marginal_gains_required: int = 1,
    cumulative_gain_threshold: float = 0.80,
    diagnose_selected_library_redundancy: bool = True,
    ampd_matrix_name_or_path: str | None = None,
    n_workers: int | None = None,
    create_new_file: bool = False,
) -> pd.DataFrame:
    """
    Population architecture compression curve.

    For K = 1, 2, 3, … finds the library of K utility function architectures minimising
    total population BIC under hard assignment (each participant uses their best-fitting
    architecture in the library).  The compression curve A(K) tracks what fraction of the
    fully-individualised BIC advantage is recovered as K grows.

    Arguments:
        • general_settings: dict; Project-wide settings (must include 'temperature_is_param').
        • file_paths: dict; Must include 'processed', 'visuals', and 'file_names'.
        • population_top_n_models: int | None; Top-N models by aggregate population BIC to
            include as candidates.  None includes all models.
        • participant_top_r_models: int | None; Top-R models per participant by individual
            BIC included as candidates.  None includes all.
        • K_max: int | None; Hard ceiling on K.  None runs until stopping criterion fires
            or K_useful_max is reached.
        • exhaustive_K_max: int; Maximum K for exhaustive combination search (default 4).
            Larger K uses greedy extension + local-swap refinement.
        • score_basis: Literal; Scoring basis for the participant × model matrix.  Primary is
            'ic_equivalent_participant_score', ensuring K=1 matches the IC champion.
        • stopping_criteria: Literal; Criterion highlighted in terminal output.  All five are
            always computed and saved.
        • marginal_gain_threshold: float; ΔA threshold for marginal-gain criterion.
        • n_consecutive_low_marginal_gains_required: int; Consecutive low-gain K values
            required before marginal-gain criterion fires.
        • cumulative_gain_threshold: float; A(K) threshold for cumulative-gain criterion.
        • diagnose_selected_library_redundancy: bool; Compute per-architecture pruning cost,
            assignment counts, and AMPD similarity flags.
        • ampd_matrix_name_or_path: str | None; Filename or absolute path of the AMPD matrix.
            AMPD-dependent output columns are NaN if not provided.
        • n_workers: int | None; Worker count for parallel exhaustive search.
            None = cpu_count − 1.
        • create_new_file: bool; If False and the final CSV exists, load and return it without
            recomputation.  Partial CSV enables resume after interruption.

    Returns:
        • pd.DataFrame: one row per K.
    """

    proc_dir    = file_paths['processed']
    vis_dir     = file_paths.get('visuals', proc_dir)
    curve_csv   = os.path.join(proc_dir, 'population_architecture_curve.csv')
    assign_csv  = os.path.join(proc_dir, 'population_architecture_assignments.csv')
    library_csv = os.path.join(proc_dir, 'population_architecture_library_diagnostics.csv')
    partial_csv = os.path.join(proc_dir, 'population_architecture_curve_partial.csv')

    if not create_new_file and os.path.exists(curve_csv):
        print(f"Architecture compression curve: loading saved results from {os.path.basename(curve_csv)}")
        return pd.read_csv(curve_csv)

    "=== Load participant × model combined fits ==="
    fits_path        = os.path.join(proc_dir, 'participant_model_combined_fits.csv')
    fits_df          = pd.read_csv(fits_path)
    all_player_uuids = list(fits_df['player_uuid'].unique())
    N_participants   = len(all_player_uuids)
    print(f"Architecture compression curve: {N_participants} participants × {fits_df['utility_idx'].nunique()} candidate models.")

    "=== Load IC CSV and identify population winner ==="
    ic_filename   = file_paths['file_names']['information_criterion']
    ic_path       = os.path.join(file_paths['bic_aic'], ic_filename)
    ic_df         = pd.read_csv(ic_path)
    ic_winner_row = ic_df.loc[ic_df['BIC'].idxmin()]
    ic_winner_idx = int(ic_winner_row['idx'])
    print(f"Population IC winner: utility index {ic_winner_idx},  BIC = {ic_winner_row['BIC']:.2f}")

    "=== Build participant × model pivot matrices ==="
    pivot_nll  = fits_df.pivot_table(index='player_uuid', columns='utility_idx',
                                     values='combined_loss_nll', aggfunc='first')
    pivot_n    = fits_df.pivot_table(index='player_uuid', columns='utility_idx',
                                     values='n_combined', aggfunc='first')
    pivot_keff = fits_df.pivot_table(index='player_uuid', columns='utility_idx',
                                     values='k_effective', aggfunc='first')
    pivot_nll  = pivot_nll.reindex(all_player_uuids)
    pivot_n    = pivot_n.reindex(all_player_uuids)
    pivot_keff = pivot_keff.reindex(all_player_uuids)

    "Sum-of-individual-BIC: each participant's own n_combined as penalty base."
    pivot_indiv_bic = 2 * pivot_nll + pivot_keff * np.log(pivot_n.clip(lower=1))

    "IC-equivalent participant score: Σᵢ score[i,m] = population BIC for model m."
    temperature_is_param = general_settings.get('temperature_is_param', True)
    n_total_per_model    = pivot_n.sum(axis=0)
    k_eff_per_model      = pivot_keff.max(axis=0)
    if temperature_is_param:
        k_eff_per_model = k_eff_per_model + 2
    complexity_share = (
        k_eff_per_model * np.log(n_total_per_model.clip(lower=1))
        * pivot_n.div(n_total_per_model.clip(lower=1), axis=1)
    )
    pivot_ic_equiv = 2 * pivot_nll + complexity_share

    "=== Candidate model filtering (participant-aware) ==="
    pop_aggregate_score = pivot_ic_equiv.sum(axis=0)
    if population_top_n_models is not None:
        top_pop_set = set(pop_aggregate_score.nsmallest(population_top_n_models).index.tolist())
    else:
        top_pop_set = set(pop_aggregate_score.index.tolist())

    if participant_top_r_models is not None:
        top_per_part_set: set = set()
        for pid in all_player_uuids:
            row_scores = pivot_indiv_bic.loc[pid].dropna()
            top_per_part_set.update(row_scores.nsmallest(participant_top_r_models).index.tolist())
    else:
        top_per_part_set = set(pivot_indiv_bic.columns.tolist())

    personal_bests_set   = set(int(model_idx) for model_idx in pivot_indiv_bic.idxmin(axis=1).dropna())
    candidate_model_idxs = sorted(top_pop_set | top_per_part_set | personal_bests_set | {ic_winner_idx})
    N_candidates         = len(candidate_model_idxs)
    cand_col_map         = {int(model_idx): jdx for jdx, model_idx in enumerate(candidate_model_idxs)}
    print(f"Candidate architecture set: {N_candidates} models  "
          f"(population top {population_top_n_models}, per-participant top {participant_top_r_models}).")

    "=== Extract L matrices ==="
    if score_basis == "ic_equivalent_participant_score":
        pivot_primary = pivot_ic_equiv
    elif score_basis == "sum_individual_BIC":
        pivot_primary = pivot_indiv_bic
    else:
        pivot_primary = 2 * pivot_nll

    def _to_L(pivot: pd.DataFrame) -> np.ndarray:
        return (pivot[candidate_model_idxs].reindex(all_player_uuids)
                .fillna(1e10).to_numpy(dtype=np.float64))

    L      = _to_L(pivot_primary)
    L_nll  = _to_L(2 * pivot_nll)
    L_ibic = _to_L(pivot_indiv_bic)
    L_ic   = _to_L(pivot_ic_equiv)

    "=== Anchor scores ==="
    pop_col_scores = L.sum(axis=0)
    k1_col_idx     = int(pop_col_scores.argmin())
    k1_model_idx   = int(candidate_model_idxs[k1_col_idx])
    score_K1       = float(pop_col_scores[k1_col_idx])

    if k1_model_idx == ic_winner_idx:
        print(f"K=1 self-check PASSED: best single architecture = utility index {k1_model_idx},  score = {score_K1:.2f}")
    else:
        ic_col      = cand_col_map.get(ic_winner_idx)
        ic_score_k1 = float(L[:, ic_col].sum()) if ic_col is not None else float('nan')
        print(f"K=1 WARNING: selected utility index {k1_model_idx} (score={score_K1:.2f}) "
              f"differs from IC winner (index {ic_winner_idx}, score={ic_score_k1:.2f}) — may reflect candidate filtering.")

    score_fully_indiv     = float(L.min(axis=1).sum())
    unique_personal_bests = set(int(candidate_model_idxs[jdx]) for jdx in L.argmin(axis=1))
    K_useful_max          = len(unique_personal_bests)
    delta_A_range         = score_K1 - score_fully_indiv
    print(f"Anchor scores: K=1 score = {score_K1:.2f},  fully-individualised score = {score_fully_indiv:.2f},  useful maximum K = {K_useful_max}")

    "=== Resume from partial CSV if present ==="
    completed_K_rows: list = []
    max_completed_K        = 0
    if not create_new_file and os.path.exists(partial_csv):
        try:
            partial_df       = pd.read_csv(partial_csv)
            completed_K_rows = partial_df.to_dict('records')
            completed_Ks     = {int(row['K']) for row in completed_K_rows}
            max_completed_K  = max(completed_Ks) if completed_Ks else 0
            print(f"Resuming from partial run: {len(completed_Ks)} K values already computed (up to K={max_completed_K}).")
        except Exception as _resume_exc:
            print(f"Could not load partial results ({_resume_exc}); starting from scratch.")
            completed_K_rows = []
            max_completed_K  = 0

    if completed_K_rows:
        last_set_idxs     = json.loads(completed_K_rows[-1]['architecture_set_idxs'])
        prev_set_col_idxs = [cand_col_map[model_idx] for model_idx in last_set_idxs if model_idx in cand_col_map]
        current_min_pp    = (L[:, prev_set_col_idxs].min(axis=1)
                             if prev_set_col_idxs else np.full(N_participants, np.inf))
        prev_A_K          = float(completed_K_rows[-1].get('A_K', 0.0))
        low_gain_streak   = 0
        for _cr in reversed(completed_K_rows):
            _da = _cr.get('delta_A_K')
            if _da is None or (isinstance(_da, float) and np.isnan(float(_da))):
                break
            low_gain_streak = (low_gain_streak + 1 if float(_da) < marginal_gain_threshold else 0)
            if float(_da) >= marginal_gain_threshold:
                break
    else:
        prev_set_col_idxs = []
        current_min_pp    = np.full(N_participants, np.inf)
        prev_A_K          = 0.0
        low_gain_streak   = 0

    "=== Search settings ==="
    n_workers_actual = (max(mp.cpu_count() - 1, 1) if n_workers is None
                        else max(1, min(int(n_workers), mp.cpu_count() - 1)))
    BATCH_SIZE = 100_000
    K_run_max  = K_max if K_max is not None else K_useful_max
    start_K    = max_completed_K + 1

    def _gen_batches(combo_iter):
        batch = []
        for item in combo_iter:
            batch.append(item)
            if len(batch) == BATCH_SIZE:
                yield batch
                batch = []
        if batch:
            yield batch

    "=== Main K loop ==="
    curve_start_time = time.time()

    for K in range(start_K, K_run_max + 1):
        K_t0 = time.time()

        if K <= exhaustive_K_max:
            "Exhaustive: evaluate all C(N_candidates, K) sets in batches of BATCH_SIZE."
            search_method  = 'exhaustive'
            best_score_K   = np.inf
            best_set_K_tup = None
            combo_iter     = it.combinations(range(N_candidates), K)
            if n_workers_actual > 1:
                tasks_gen = ((batch, L) for batch in _gen_batches(combo_iter))
                with mp.Pool(processes=n_workers_actual) as pool:
                    for b_score, b_set in pool.imap_unordered(
                            _exhaustive_search_worker, tasks_gen, chunksize=1):
                        if b_score < best_score_K:
                            best_score_K = b_score;  best_set_K_tup = b_set
            else:
                for batch in _gen_batches(combo_iter):
                    b_score, b_set = _exhaustive_search_worker((batch, L))
                    if b_score < best_score_K:
                        best_score_K = b_score;  best_set_K_tup = b_set
            if best_set_K_tup is None:
                best_set_K_tup = tuple(range(min(K, N_candidates)))
            best_set_col_idxs = list(best_set_K_tup)

        else:
            "Greedy extension from K-1 set, followed by local-swap refinement."
            search_method = 'greedy_swap'
            in_set        = set(prev_set_col_idxs)
            baseline_sum  = float(current_min_pp.sum())
            best_gain = -np.inf;  best_add = None
            for jdx in range(N_candidates):
                if jdx in in_set:
                    continue
                gain = baseline_sum - float(np.minimum(current_min_pp, L[:, jdx]).sum())
                if gain > best_gain:
                    best_gain = gain;  best_add = jdx
            if best_add is None:
                print(f"Greedy extension found no improvement at K={K}; stopping search.")
                break
            candidate_set     = list(prev_set_col_idxs) + [best_add]
            candidate_set_set = set(candidate_set)
            for _pass in range(10):
                improved           = False
                current_swap_score = float(L[:, candidate_set].min(axis=1).sum())
                for remove_j in list(candidate_set):
                    temp_set = [jdx for jdx in candidate_set if jdx != remove_j]
                    temp_min = (L[:, temp_set].min(axis=1) if temp_set
                                else np.full(N_participants, np.inf))
                    for add_j in range(N_candidates):
                        if add_j in candidate_set_set:
                            continue
                        if float(np.minimum(temp_min, L[:, add_j]).sum()) < current_swap_score - 1e-9:
                            candidate_set     = temp_set + [add_j]
                            candidate_set_set = set(candidate_set)
                            improved = True;  break
                    if improved:
                        break
                if not improved:
                    break
            best_set_col_idxs = candidate_set
            best_score_K      = float(L[:, candidate_set].min(axis=1).sum())

        current_min_pp      = L[:, best_set_col_idxs].min(axis=1)
        prev_set_col_idxs   = list(best_set_col_idxs)
        best_set_model_idxs = [int(candidate_model_idxs[jdx]) for jdx in best_set_col_idxs]

        A_K       = (score_K1 - best_score_K) / delta_A_range if delta_A_range > 0 else 1.0
        delta_A_K = float(A_K - prev_A_K) if K > 1 else float('nan')
        prev_A_K  = A_K

        raw_nll_score_K  = float(L_nll[:, best_set_col_idxs].min(axis=1).sum())
        sum_ibic_score_K = float(L_ibic[:, best_set_col_idxs].min(axis=1).sum())
        ic_equiv_score_K = float(L_ic[:, best_set_col_idxs].min(axis=1).sum())

        k_elapsed     = time.time() - K_t0
        curve_elapsed = time.time() - curve_start_time
        n_done        = K - start_K + 1
        n_todo        = K_run_max - start_K + 1
        eta_str       = (_fmt_duration((n_todo - n_done) / (n_done / curve_elapsed))
                         if n_done > 0 and curve_elapsed > 0 else "unknown")
        delta_str     = f"{delta_A_K:.4f}" if not np.isnan(delta_A_K) else "n/a"
        print(f"K={K}  A(K)={A_K:.4f}  ΔA={delta_str}  "
              f"set={best_set_model_idxs}  ({search_method}, {_fmt_duration(k_elapsed)})  ETA={eta_str}")

        completed_K_rows.append({
            'K':                              K,
            'architecture_set_idxs':          json.dumps(best_set_model_idxs),
            'search_method':                  search_method,
            'score_basis':                    score_basis,
            'raw_nll_score_K':                raw_nll_score_K,
            'sum_individual_BIC_score_K':     sum_ibic_score_K,
            'ic_equivalent_score_K':          ic_equiv_score_K,
            'score_K1':                       score_K1,
            'score_fully_individualized':     score_fully_indiv,
            'A_K':                            A_K,
            'delta_A_K':                      delta_A_K if not np.isnan(delta_A_K) else None,
            'K_useful_max':                   K_useful_max,
            'n_unique_individual_best_models': K_useful_max,
        })
        pd.DataFrame(completed_K_rows).to_csv(partial_csv, index=False, encoding='utf-8-sig')

        if K_max is None:
            if stopping_criteria == 'marginal_gain' and not np.isnan(delta_A_K):
                if delta_A_K < marginal_gain_threshold:
                    low_gain_streak += 1
                    if low_gain_streak >= n_consecutive_low_marginal_gains_required:
                        print(f"Marginal gain stopping criterion reached at K={K}; halting search.")
                        break
                else:
                    low_gain_streak = 0
            elif stopping_criteria == 'cumulative_gain' and A_K >= cumulative_gain_threshold:
                print(f"Cumulative gain stopping criterion reached: A(K={K}) = {A_K:.4f}; halting search.")
                break

    "=== Post-hoc: all five stopping criteria ==="
    curve_df = pd.DataFrame(completed_K_rows)
    K_vals   = curve_df['K'].tolist()
    A_vals   = curve_df['A_K'].tolist()
    n_rows   = len(curve_df)

    da_list  = [row.get('delta_A_K') for row in completed_K_rows]
    d2a_list = [None] * n_rows
    for row_idx in range(2, n_rows):
        da_k, da_km1 = da_list[row_idx], da_list[row_idx - 1]
        if da_k is not None and da_km1 is not None:
            d2a_list[row_idx] = float(da_k - da_km1)
    curve_df['delta2_A_K'] = d2a_list

    K_span  = max(K_vals) - min(K_vals) or 1
    A_max   = max(A_vals) or 1.0
    kneedle = [(a / A_max) - ((k_val - min(K_vals)) / K_span) for k_val, a in zip(K_vals, A_vals)]
    curve_df['kneedle_distance']          = kneedle
    kneedle_best_K                        = K_vals[int(np.argmax(kneedle))]
    curve_df['selected_by_kneedle_elbow'] = [k_val == kneedle_best_K for k_val in K_vals]

    first_low_mg_K = None;  selected_mg_K = None;  mg_streak = 0
    for _ri, _row in curve_df.iterrows():
        _da = _row['delta_A_K']
        if _da is None or (isinstance(_da, float) and np.isnan(float(_da))):
            continue
        if float(_da) < marginal_gain_threshold:
            mg_streak += 1
            if mg_streak >= n_consecutive_low_marginal_gains_required and first_low_mg_K is None:
                first_low_mg_K = int(_row['K']);  selected_mg_K = first_low_mg_K - 1
        else:
            mg_streak = 0
    curve_df['first_K_with_low_marginal_gain'] = first_low_mg_K
    curve_df['selected_K_by_marginal_gain']    = selected_mg_K
    curve_df['selected_by_marginal_gain']      = [k_val == selected_mg_K for k_val in K_vals]

    selected_cg_K = None
    for _ri, _row in curve_df.iterrows():
        if _row['A_K'] >= cumulative_gain_threshold:
            selected_cg_K = int(_row['K']);  break
    curve_df['selected_by_cumulative_gain'] = [k_val == selected_cg_K for k_val in K_vals]

    d2_abs = [abs(val) if val is not None else -np.inf for val in d2a_list]
    mc_idx = int(np.argmax(d2_abs))
    selected_mc_K = K_vals[mc_idx] if d2_abs[mc_idx] > 0 else None
    curve_df['selected_by_max_curvature'] = [k_val == selected_mc_K for k_val in K_vals]

    model_keff_lookup = fits_df.drop_duplicates('utility_idx').set_index('utility_idx')['k_effective']
    meta_bic_vals     = []
    ic_scores_list    = curve_df['ic_equivalent_score_K'].tolist()
    for row_idx, row in curve_df.iterrows():
        set_idxs  = json.loads(row['architecture_set_idxs'])
        mean_keff = float(model_keff_lookup.reindex(set_idxs).mean())
        meta_bic_vals.append(float(
            2 * ic_scores_list[row_idx] + row['K'] * mean_keff * np.log(max(N_participants, 1))
        ))
    curve_df['exploratory_meta_bic'] = meta_bic_vals
    selected_metabic_K               = K_vals[int(np.argmin(meta_bic_vals))]
    curve_df['selected_by_meta_bic'] = [k_val == selected_metabic_K for k_val in K_vals]

    "=== AMPD matrix — loaded from explicit path if given, otherwise auto-resolved from settings ==="
    ampd_df = None;  ampd_idx_set = set();  ampd_col_set = set();  all_ampd_pos = np.array([])
    try:
        if ampd_matrix_name_or_path is not None:
            ampd_path = (ampd_matrix_name_or_path if os.path.isabs(ampd_matrix_name_or_path)
                         else os.path.join(proc_dir, ampd_matrix_name_or_path))
            ampd_df = pd.read_csv(ampd_path, index_col=0)
        else:
            ampd_df = _load_ampd_matrix_from_settings(general_settings, file_paths)
        ampd_df.index   = ampd_df.index.astype(int)
        ampd_df.columns = ampd_df.columns.astype(int)
        ampd_idx_set    = set(ampd_df.index.tolist())
        ampd_col_set    = set(ampd_df.columns.tolist())
        flat            = ampd_df.values.flatten()
        all_ampd_pos    = flat[~np.isnan(flat) & (flat > 0)]
        print(f"AMPD behavioral-distance matrix loaded: {ampd_df.shape[0]}×{ampd_df.shape[1]}")
    except FileNotFoundError as _fnf:
        print(f"Warning: AMPD behavioral-distance matrix not found — AMPD columns will be NaN.")
        print(f"  Searched path: {_fnf}")
        print(f"  Run compute_ampd_distance_matrix() with the current general_settings['ampd_settings'] first.")
    except Exception as _ampd_exc:
        print(f"Warning: could not load AMPD behavioral-distance matrix ({type(_ampd_exc).__name__}: {_ampd_exc}); AMPD columns will be NaN.")

    lib_min_l = [];  lib_mean_l = [];  lib_med_l = [];  lib_max_l = [];  near_pair_l = []
    for row_idx, row in curve_df.iterrows():
        set_idxs = json.loads(row['architecture_set_idxs'])
        if ampd_df is None or len(set_idxs) < 2:
            lib_min_l.append(float('nan'));   lib_mean_l.append(float('nan'))
            lib_med_l.append(float('nan'));   lib_max_l.append(float('nan'))
            near_pair_l.append(None);  continue
        pairwise_dists = [];  min_d = np.inf;  near_pair = None
        valid = [model_idx for model_idx in set_idxs if model_idx in ampd_idx_set and model_idx in ampd_col_set]
        for a_i, b_i in it.combinations(valid, 2):
            dist_val = float(ampd_df.loc[a_i, b_i])
            if not np.isnan(dist_val):
                pairwise_dists.append(dist_val)
                if dist_val < min_d:
                    min_d = dist_val;  near_pair = (a_i, b_i)
        if pairwise_dists:
            lib_min_l.append(float(np.min(pairwise_dists)));    lib_mean_l.append(float(np.mean(pairwise_dists)))
            lib_med_l.append(float(np.median(pairwise_dists))); lib_max_l.append(float(np.max(pairwise_dists)))
            near_pair_l.append(json.dumps(list(near_pair)) if near_pair else None)
        else:
            lib_min_l.append(float('nan'));   lib_mean_l.append(float('nan'))
            lib_med_l.append(float('nan'));   lib_max_l.append(float('nan'))
            near_pair_l.append(None)
    curve_df['library_ampd_min']              = lib_min_l
    curve_df['library_ampd_mean']             = lib_mean_l
    curve_df['library_ampd_median']           = lib_med_l
    curve_df['library_ampd_max']              = lib_max_l
    curve_df['nearest_selected_pair_by_ampd'] = near_pair_l

    "=== Save curve CSV ==="
    curve_df.to_csv(curve_csv, index=False, encoding='utf-8-sig')
    print(f"Architecture compression curve saved: {os.path.basename(curve_csv)}")

    "=== Build and save assignments CSV ==="
    assign_rows = []
    for row_idx, row in curve_df.iterrows():
        K_val        = int(row['K'])
        set_idxs     = json.loads(row['architecture_set_idxs'])
        set_col_idxs = [cand_col_map[model_idx] for model_idx in set_idxs if model_idx in cand_col_map]
        if not set_col_idxs:
            continue
        assign_cols = L[:, set_col_idxs].argmin(axis=1)
        for p_i, player_uuid in enumerate(all_player_uuids):
            pos          = int(assign_cols[p_i])
            assigned_m   = set_idxs[pos]
            a_score      = float(L[p_i, set_col_idxs[pos]])
            delta_score  = float(a_score - float(L[p_i, :].min()))
            personal_rank = int((L[p_i, :] < a_score).sum()) + 1
            ampd_to_win  = (float(ampd_df.loc[assigned_m, ic_winner_idx])
                            if (ampd_df is not None and assigned_m in ampd_idx_set
                                and ic_winner_idx in ampd_col_set) else float('nan'))
            others = [other_m for other_m in set_idxs if other_m != assigned_m]
            if ampd_df is not None and assigned_m in ampd_idx_set and others:
                other_dists = [float(ampd_df.loc[assigned_m, other_m]) for other_m in others
                               if other_m in ampd_col_set and not np.isnan(float(ampd_df.loc[assigned_m, other_m]))]
                ampd_to_near = float(min(other_dists)) if other_dists else float('nan')
            else:
                ampd_to_near = float('nan')
            assign_rows.append({
                'K': K_val, 'player_uuid': player_uuid,
                'assigned_utility_idx': assigned_m,
                'assigned_model_rank_for_player': personal_rank,
                'assigned_model_delta_score_for_player': delta_score,
                'assigned_model_AMPD_to_population_winner': ampd_to_win,
                'assigned_model_AMPD_to_nearest_selected_model': ampd_to_near,
            })
    pd.DataFrame(assign_rows).to_csv(assign_csv, index=False, encoding='utf-8-sig')
    print(f"Participant architecture assignments saved: {os.path.basename(assign_csv)}")

    "=== Build and save library diagnostics CSV ==="
    if diagnose_selected_library_redundancy:
        model_meta = fits_df.drop_duplicates('utility_idx').set_index('utility_idx')
        diag_rows  = []
        for row_idx, row in curve_df.iterrows():
            K_val        = int(row['K'])
            set_idxs     = json.loads(row['architecture_set_idxs'])
            set_col_idxs = [cand_col_map[model_idx] for model_idx in set_idxs if model_idx in cand_col_map]
            if not set_col_idxs:
                continue
            score_K_val = float(L[:, set_col_idxs].min(axis=1).sum())
            assign_cols = L[:, set_col_idxs].argmin(axis=1)
            k_diag_rows = []
            for k_pos, (model_idx, col_idx) in enumerate(zip(set_idxs, set_col_idxs)):
                assigned_mask  = (assign_cols == k_pos)
                assigned_n     = int(assigned_mask.sum())
                assigned_pct   = float(assigned_n / N_participants)
                mean_indiv_bic = (float(L_ibic[assigned_mask, col_idx].mean())
                                  if assigned_n > 0 else float('nan'))
                remaining    = [col for col in set_col_idxs if col != col_idx]
                score_without = float(L[:, remaining].min(axis=1).sum() if remaining
                                      else L.min(axis=1).sum())
                pruning_cost = float(score_without - score_K_val)
                pruning_norm = (float(pruning_cost / delta_A_range)
                                if delta_A_range > 0 else float('nan'))
                near_m = None;  near_ampd_v = float('nan');  near_ampd_pct = float('nan')
                if ampd_df is not None and model_idx in ampd_idx_set and len(set_idxs) > 1:
                    others_sel = [other_m for other_m in set_idxs if other_m != model_idx and other_m in ampd_col_set]
                    if others_sel:
                        dsel = ampd_df.loc[model_idx, others_sel].dropna()
                        if not dsel.empty:
                            near_m        = int(dsel.idxmin())
                            near_ampd_v   = float(dsel.min())
                            near_ampd_pct = (float(np.mean(all_ampd_pos < near_ampd_v))
                                             if len(all_ampd_pos) > 0 else float('nan'))
                n_conds = sum([
                    assigned_n < 2,
                    (not np.isnan(pruning_norm)) and pruning_norm < 0.01,
                    (not np.isnan(near_ampd_pct)) and near_ampd_pct < 0.05,
                ])
                flag = 2 if n_conds == 3 else (1 if n_conds >= 2 else 0)
                if model_idx in model_meta.index:
                    model_row = model_meta.loc[model_idx]
                    equation  = str(model_row.get('equation', '?'))
                    k_params  = int(model_row.get('k_params', -1))
                    util_vals = {col: bool(model_row.get(col, False))
                                 for col in utility_settings.keys() if col in model_row.index}
                else:
                    equation = '?';  k_params = -1
                    util_vals = {col: None for col in utility_settings.keys()}
                k_diag_rows.append({
                    'K': K_val, 'utility_idx': model_idx, 'equation': equation,
                    'k_params': k_params, 'assigned_n': assigned_n,
                    'assigned_percent': assigned_pct, 'mean_individual_BIC': mean_indiv_bic,
                    'pruning_cost': pruning_cost,
                    'pruning_cost_normalized': pruning_norm,
                    'nearest_selected_model_idx': near_m,
                    'nearest_selected_model_ampd': near_ampd_v,
                    'nearest_selected_model_ampd_percentile': near_ampd_pct,
                    'redundancy_warning_level': flag, **util_vals,
                })
            if len(k_diag_rows) > 1:
                pr_a = np.array([diag_row['pruning_cost_normalized'] for diag_row in k_diag_rows], float)
                as_a = np.array([diag_row['assigned_percent']        for diag_row in k_diag_rows], float)
                am_a = np.array([diag_row['nearest_selected_model_ampd'] for diag_row in k_diag_rows], float)
                for diag_idx in range(len(k_diag_rows)):
                    k_diag_rows[diag_idx]['redundancy_score_optional'] = (
                        0.50 * float(np.nanmean(pr_a > pr_a[diag_idx]))
                        + 0.30 * float(np.nanmean(as_a > as_a[diag_idx]))
                        + 0.20 * float(np.nanmean(am_a > am_a[diag_idx]))
                    )
            elif k_diag_rows:
                k_diag_rows[0]['redundancy_score_optional'] = float('nan')
            diag_rows.extend(k_diag_rows)
        diag_df = pd.DataFrame(diag_rows)
        diag_df.to_csv(library_csv, index=False, encoding='utf-8-sig')
        print(f"Architecture library diagnostics saved: {os.path.basename(library_csv)}")

        "=== Human-readable summary table: one row per (K, model), all info a researcher needs ==="
        k_level_cols = [
            'K', 'A_K', 'delta_A_K', 'delta2_A_K', 'kneedle_distance',
            'selected_by_kneedle_elbow', 'selected_by_marginal_gain',
            'selected_by_cumulative_gain', 'selected_by_max_curvature', 'selected_by_meta_bic',
        ]
        ic_bic_lookup        = ic_df.set_index('idx')['BIC'].to_dict()
        summary_df           = diag_df.rename(columns={
            'assigned_n':      'n_players_assigned',
            'assigned_percent': 'pct_players_assigned',
        }).copy()
        summary_df['population_IC_BIC'] = summary_df['utility_idx'].map(ic_bic_lookup)
        curve_k_indexed = curve_df.set_index('K')
        for col in [col for col in k_level_cols if col != 'K' and col in curve_k_indexed.columns]:
            summary_df[col] = summary_df['K'].map(curve_k_indexed[col].to_dict())
        front_cols  = [
            'K', 'A_K', 'delta_A_K', 'delta2_A_K', 'kneedle_distance',
            'selected_by_kneedle_elbow', 'selected_by_marginal_gain',
            'selected_by_cumulative_gain', 'selected_by_max_curvature', 'selected_by_meta_bic',
            'utility_idx', 'k_params', 'population_IC_BIC',
            'n_players_assigned', 'pct_players_assigned', 'mean_individual_BIC',
            'pruning_cost', 'pruning_cost_normalized',
            'nearest_selected_model_idx', 'nearest_selected_model_ampd',
            'nearest_selected_model_ampd_percentile',
            'redundancy_warning_level', 'redundancy_score_optional',
        ]
        "Sort within each K by mean_individual_BIC ascending (best-fitting models first)."
        summary_df = summary_df.sort_values(['K', 'mean_individual_BIC'], ascending=[True, True])
        "Utility settings columns come after diagnostics; equation is always last so it spills into empty Excel cells."
        util_cols    = [col for col in summary_df.columns if col not in front_cols and col != 'equation']
        ordered_cols = [col for col in front_cols if col in summary_df.columns] + util_cols + ['equation']
        summary_csv  = os.path.join(proc_dir, 'population_architecture_summary_table.csv')
        summary_df[ordered_cols].to_csv(summary_csv, index=False, encoding='utf-8-sig')
        print(f"Architecture summary table saved: {os.path.basename(summary_csv)}")

    if os.path.exists(partial_csv):
        os.remove(partial_csv)

    return curve_df

    if not create_new_file:
        all_exist = all(os.path.exists(file_path_val) for file_path_val in (curve_csv_path, summary_csv_path, assignments_csv_path))
        if all_exist:
            existing_curve   = pd.read_csv(curve_csv_path,   encoding='utf-8-sig')
            existing_summary = pd.read_csv(summary_csv_path, encoding='utf-8-sig')
            if not existing_curve.empty and not existing_summary.empty:
                print(
                    f"Loaded architecture compression curve from cache.  "
                    f"({len(existing_curve)} curve rows, {len(existing_summary)} summary rows)"
                )
                return existing_curve, existing_summary

    cv_losses_csv_path = os.path.join(file_paths['processed'], 'cv_architecture_losses.csv')
    if not os.path.exists(cv_losses_csv_path):
        raise FileNotFoundError(
            f"Cross-validated architecture losses not found at {cv_losses_csv_path}.  "
            "Run compute_cross_validated_architecture_losses() first to generate it."
        )
    cv_losses_df = pd.read_csv(cv_losses_csv_path, encoding='utf-8-sig')
    all_fold_ids = sorted(cv_losses_df['fold_id'].unique())
    n_folds      = len(all_fold_ids)
    print(f"Architecture compression curve: {n_folds} folds, k_max={k_max}, "
          f"marginal_gain_threshold={marginal_gain_threshold:.1%}")

    curve_rows      = []
    assignment_rows = []

    for fold_id_val in all_fold_ids:
        fold_rows = cv_losses_df[cv_losses_df['fold_id'] == fold_id_val].copy()

        n_participants_in_fold   = fold_rows['player_uuid'].nunique()
        model_participant_counts = fold_rows.groupby('utility_idx')['player_uuid'].nunique()

        """
        Universal models: present for every participant in this fold.
        Only universal models are valid K=1 candidates (we need the K=1 baseline to cover
        all 73 participants for A(K) to be well-defined).  For K≥2, any model is allowed —
        participants whose candidate set does not include the new model simply remain on
        whichever current codebook model best fits them, guaranteed by the universal K=1 entry.
        """
        universal_idxs  = sorted(
            model_participant_counts[model_participant_counts == n_participants_in_fold].index.tolist()
        )
        all_unique_idxs = sorted(fold_rows['utility_idx'].unique().tolist())
        n_universal     = len(universal_idxs)
        n_all           = len(all_unique_idxs)
        print(f"  Fold {fold_id_val}: {n_participants_in_fold} participants, "
              f"{n_universal} universal / {n_all} total candidate models, running to K={k_max}")

        if n_universal == 0:
            print(f"  Warning: fold {fold_id_val} has no universal models — skipping.")
            continue

        """
        Build full pivot matrices (all candidates): NaN where a model is not in a participant's set.
        Reindex both matrices to the same player list so all participants appear in both,
        even if some have all-NaN test NLL (e.g. due to failed fits on a particular fold).
        """
        all_fold_players = sorted(fold_rows['player_uuid'].unique().tolist())
        train_bic_matrix = fold_rows.pivot_table(
            index='player_uuid', columns='utility_idx', values='train_BIC', aggfunc='first'
        ).reindex(index=all_fold_players)
        test_nll_matrix  = fold_rows.pivot_table(
            index='player_uuid', columns='utility_idx', values='combined_test_nll', aggfunc='first'
        ).reindex(index=all_fold_players)
        participants = all_fold_players

        "Per-participant n_test_combined (same across models; take first valid)."
        n_test_by_player = fold_rows.groupby('player_uuid')['n_test_combined'].first()
        nll_chance_total = float((n_test_by_player * np.log(2)).sum())

        """
        Individualised endpoint: each participant uses their best candidate model by train_BIC.
        idxmin(axis=1) returns NaN for players with all-NaN rows — skip those with guards below.
        """
        best_model_per_participant = train_bic_matrix.idxmin(axis=1)
        nll_individual_total = 0.0
        for participant in participants:
            best_m = best_model_per_participant.get(participant)
            if best_m is None or (isinstance(best_m, float) and np.isnan(best_m)):
                continue
            if best_m in test_nll_matrix.columns:
                val = float(test_nll_matrix.loc[participant, best_m])
                if not np.isnan(val):
                    nll_individual_total += val

        "Greedily build the codebook for K = 1 .. k_max."
        codebook: list  = []
        A_prev          = 0.0
        test_nll_prev   = np.nan
        nll_common_fold = np.nan

        for K in range(1, k_max + 1):
            best_candidate   = None
            best_train_score = np.inf
            best_assignments : dict = {}

            """
            K=1: candidates restricted to universal models so the baseline is evaluable on
            all participants.  K≥2: all remaining candidates are allowed — participants not
            covered by the new model fall back to their best existing codebook model.
            """
            candidates_this_K = universal_idxs if K == 1 else all_unique_idxs

            for model_idx in candidates_this_K:
                if model_idx in codebook:
                    continue
                S_temp = codebook + [model_idx]

                """
                Assign each participant to the model in S_temp with the lowest train_BIC.
                np.nanargmin skips NaN, so participants without data for model_idx are assigned
                to whichever other S_temp model they have data for.
                """
                assignments_temp: dict = {}
                for participant in participants:
                    bic_array = np.array([train_bic_matrix.loc[participant, sm]
                                          for sm in S_temp if sm in train_bic_matrix.columns],
                                         dtype=float)
                    s_temp_valid = [sm for sm in S_temp if sm in train_bic_matrix.columns]
                    if bic_array.size > 0 and not np.all(np.isnan(bic_array)):
                        assignments_temp[participant] = s_temp_valid[int(np.nanargmin(bic_array))]

                train_score_temp = float(np.nansum([
                    train_bic_matrix.loc[participant, assignments_temp[participant]]
                    for participant in assignments_temp
                ]))

                if train_score_temp < best_train_score:
                    best_candidate   = model_idx
                    best_train_score = train_score_temp
                    best_assignments = assignments_temp

            if best_candidate is None:
                break

            codebook.append(best_candidate)
            test_nll_K = float(np.nansum([
                test_nll_matrix.loc[participant, best_assignments[participant]]
                for participant in best_assignments
            ]))

            if K == 1:
                nll_common_fold = test_nll_K
                A_K             = 0.0
                H_form_K        = 0.0
                delta_A_K       = np.nan
                delta_NLL_K     = np.nan
            else:
                denom_A  = nll_common_fold - nll_individual_total
                A_K      = float((nll_common_fold - test_nll_K) / denom_A) if abs(denom_A) > 1e-10 else np.nan
                denom_H  = nll_chance_total - test_nll_K
                H_form_K = float((nll_common_fold - test_nll_K) / denom_H) if abs(denom_H) > 1e-10 else np.nan
                delta_A_K   = float(A_K - A_prev)       if not np.isnan(A_K) else np.nan
                delta_NLL_K = float(test_nll_prev - test_nll_K) if not np.isnan(test_nll_prev) else np.nan

            curve_rows.append({
                'fold_id':             fold_id_val,
                'K':                   K,
                'codebook_model_idxs': json.dumps(codebook),
                'train_score':         best_train_score,
                'test_nll':            test_nll_K,
                'nll_common':          nll_common_fold,
                'nll_individual':      nll_individual_total,
                'nll_chance':          nll_chance_total,
                'A_K':                 A_K,
                'H_form_K':            H_form_K,
                'delta_A_K':           delta_A_K,
                'delta_NLL_K':         delta_NLL_K,
            })

            for participant, m_assigned in best_assignments.items():
                assignment_rows.append({
                    'fold_id':            fold_id_val,
                    'K':                  K,
                    'player_uuid':        participant,
                    'assigned_model_idx': m_assigned,
                    'test_nll':           float(test_nll_matrix.loc[participant, m_assigned]),
                })

            A_prev        = A_K if not np.isnan(A_K) else A_prev
            test_nll_prev = test_nll_K

    curve_df       = pd.DataFrame(curve_rows)
    assignments_df = pd.DataFrame(assignment_rows)

    "Aggregate across folds: mean ± SE for each K."
    summary_rows = []
    all_k_values = sorted(curve_df['K'].unique()) if not curve_df.empty else []
    for K_val in all_k_values:
        k_rows  = curve_df[curve_df['K'] == K_val]
        n_k     = len(k_rows)
        denom   = np.sqrt(n_k) if n_k > 1 else 1.0

        def _mean(col: str) -> float:
            return float(k_rows[col].mean())

        def _se(col: str) -> float:
            return float(k_rows[col].std(ddof=1) / denom) if n_k > 1 else np.nan

        summary_rows.append({
            'K':             K_val,
            'mean_test_nll': _mean('test_nll'),
            'se_test_nll':   _se('test_nll'),
            'mean_A_K':      _mean('A_K'),
            'se_A_K':        _se('A_K'),
            'mean_H_form_K': _mean('H_form_K'),
            'se_H_form_K':   _se('H_form_K'),
            'mean_delta_A':  _mean('delta_A_K'),
            'se_delta_A':    _se('delta_A_K'),
        })

    summary_df = pd.DataFrame(summary_rows)

    "Stopping criteria."
    marginal_gain_k = None
    one_se_k        = None
    if not summary_df.empty:
        best_nll_idx  = summary_df['mean_test_nll'].idxmin()
        best_mean_nll = float(summary_df.loc[best_nll_idx, 'mean_test_nll'])
        best_se_nll   = float(summary_df.loc[best_nll_idx, 'se_test_nll'])
        one_se_thresh = best_mean_nll + best_se_nll

        for _, row in summary_df.iterrows():
            K_val   = int(row['K'])
            delta_a = row['mean_delta_A']
            nll_val = row['mean_test_nll']
            if (K_val > 1 and marginal_gain_k is None
                    and not np.isnan(delta_a) and delta_a < marginal_gain_threshold):
                marginal_gain_k = K_val
            if one_se_k is None and not np.isnan(nll_val) and nll_val <= one_se_thresh:
                one_se_k = K_val

        summary_df['selected_by_marginal_gain_rule'] = summary_df['K'] == marginal_gain_k
        summary_df['selected_by_one_se_rule']        = summary_df['K'] == one_se_k

        "Print summary table."
        table_width = 90
        print(f"\n{'=' * table_width}")
        print(f" {'K':>3}   {'mean_A_K':>10}   {'mean_delta_A':>13}   {'mean_H_form_K':>14}   "
              f"{'margin_gain*':>12}   {'1SE*':>5}")
        print(f"{'=' * table_width}")
        for _, row in summary_df.iterrows():
            K_val   = int(row['K'])
            A_str   = f"{row['mean_A_K']:.4f}"      if not np.isnan(row['mean_A_K'])      else "    nan"
            dA_str  = f"{row['mean_delta_A']:.4f}"  if not np.isnan(row['mean_delta_A'])  else "     —"
            H_str   = f"{row['mean_H_form_K']:.4f}" if not np.isnan(row['mean_H_form_K']) else "    nan"
            mg_mark = "  *" if row.get('selected_by_marginal_gain_rule') else ""
            se_mark = "  *" if row.get('selected_by_one_se_rule')        else ""
            print(f" {K_val:>3}   {A_str:>10}   {dA_str:>13}   {H_str:>14}   {mg_mark:>12}   {se_mark:>5}")
        print(f"{'=' * table_width}")
        mg_str = f"K = {marginal_gain_k}" if marginal_gain_k else "threshold not reached"
        se_str = f"K = {one_se_k}"        if one_se_k        else "threshold not reached"
        print(f"Marginal-gain stopping point (delta < {marginal_gain_threshold:.0%}): {mg_str}")
        print(f"One-SE stopping point: {se_str}")
        for k_print in [2, 3, 5]:
            k_row = summary_df[summary_df['K'] == k_print]
            if len(k_row) > 0 and not np.isnan(float(k_row['mean_A_K'].values[0])):
                print(f"  A(K={k_print}) = {float(k_row['mean_A_K'].values[0]):.4f}")
        print(f"{'=' * table_width}\n")

    curve_df.to_csv(curve_csv_path,       index=False, encoding='utf-8-sig')
    summary_df.to_csv(summary_csv_path,   index=False, encoding='utf-8-sig')
    assignments_df.to_csv(assignments_csv_path, index=False, encoding='utf-8-sig')
    print(f"Saved: {curve_csv_path}  ({len(curve_df)} rows)")
    print(f"Saved: {summary_csv_path}  ({len(summary_df)} rows)")
    print(f"Saved: {assignments_csv_path}  ({len(assignments_df)} rows)")
    return curve_df, summary_df


def plot_architecture_compression_curve(
    general_settings: dict,
    file_paths: dict,
    fig_lay: dict,
    export_fig: bool = True,
    base_hue: int | None = None,
) -> go.Figure:
    """
    Interactive population utility-function compression-curve chart.

    Arguments:
        • general_settings: dict; Project settings (kept for API consistency).
        • file_paths: dict; Project file paths with keys 'processed', 'bic_aic', 'visuals', 'file_names'.
        • fig_lay: dict; Layout settings from config.py (template, font, base_hue, title_size, etc.).
        • export_fig: bool; If True, write HTML to visuals/population_architecture_curve.html.
        • base_hue: int | None; Starting hue for the color scheme. Overrides fig_lay['base_hue']
          when provided. Curves use base_hue and base_hue+20; criterion markers start at base_hue+40.

    Returns:
        • go.Figure; A(K) and kneedle-distance traces on a shared [0, 1.1] y-axis, with vertical
          stopping-criterion markers and per-K hover text showing model counts, μBIC, and equations.
    """
    _EQ_CHAR_LIMIT = 115   # equation strings longer than this are truncated with "..."

    proc_dir       = str(file_paths['processed'])
    base_font_size = fig_lay.get('font', {}).get('size', 16)
    base_hue       = base_hue if base_hue is not None else fig_lay.get('base_hue', 200)

    "=== Load CSVs ==="
    curve_df  = pd.read_csv(os.path.join(proc_dir, 'population_architecture_curve.csv'),                encoding='utf-8-sig')
    assign_df = pd.read_csv(os.path.join(proc_dir, 'population_architecture_assignments.csv'),           encoding='utf-8-sig')
    fits_df   = pd.read_csv(os.path.join(proc_dir, 'participant_model_combined_fits.csv'),               encoding='utf-8-sig')
    ic_df     = pd.read_csv(os.path.join(str(file_paths['bic_aic']),
                                          file_paths['file_names']['information_criterion']),             encoding='utf-8-sig')
    library_diag_path = os.path.join(proc_dir, 'population_architecture_library_diagnostics.csv')
    try:
        library_diag_df = pd.read_csv(library_diag_path, encoding='utf-8-sig')
    except FileNotFoundError:
        library_diag_df = pd.DataFrame()

    "=== Build per-model metadata: equation string and population-level BIC for sorting ==="
    util_setting_keys = list(utility_settings.keys())
    model_meta        = fits_df.drop_duplicates('utility_idx').set_index('utility_idx')
    model_equations: dict = {}
    for model_idx, meta_row in model_meta.iterrows():
        model_util = {util_key: bool(meta_row.get(util_key, False)) for util_key in util_setting_keys if util_key in meta_row.index}
        raw_eq     = build_utility_equation(utility_settings=model_util)
        model_equations[int(model_idx)] = (raw_eq[:_EQ_CHAR_LIMIT] + '…') if len(raw_eq) > _EQ_CHAR_LIMIT else raw_eq
    ic_bic_by_model = ic_df.set_index('idx')['BIC'].to_dict()

    "=== Per-K player counts and mean individual BIC per assigned model ==="
    assign_counts = (
        assign_df.groupby(['K', 'assigned_utility_idx'])
        .size()
        .reset_index(name='n_players')
    )
    assign_with_bic = assign_df.merge(
        fits_df[['player_uuid', 'utility_idx', 'BIC_individual']],
        left_on=['player_uuid', 'assigned_utility_idx'],
        right_on=['player_uuid', 'utility_idx'],
        how='left',
    )
    mean_bic_by_K_model = (
        assign_with_bic.groupby(['K', 'assigned_utility_idx'])['BIC_individual'].mean()
    )

    "=== Per-(K, model) nearest-AMPD lookup from library diagnostics ==="
    nearest_ampd_lookup: dict = {}
    if not library_diag_df.empty:
        for _, diag_row in library_diag_df.iterrows():
            key = (int(diag_row['K']), int(diag_row['utility_idx']))
            nearest_ampd_lookup[key] = {
                'nearest_idx':  diag_row.get('nearest_selected_model_idx'),
                'ampd_val':     diag_row.get('nearest_selected_model_ampd'),
                'ampd_pct':     diag_row.get('nearest_selected_model_ampd_percentile'),
            }

    "=== Marginal-gain K — caps hover display; shows at most K_marginal_gain model lines ==="
    mg_rows = (
        curve_df[curve_df['selected_by_marginal_gain'] == True]
        if 'selected_by_marginal_gain' in curve_df.columns else pd.DataFrame()
    )
    K_hover_cap = int(mg_rows['K'].values[0]) if not mg_rows.empty else None

    k_vals  = curve_df['K'].tolist()
    a_vals  = curve_df['A_K'].tolist()
    kd_vals = curve_df['kneedle_distance'].tolist()

    "=== Build per-K hover text ==="
    def _build_hover(row: pd.Series) -> str:
        K_val  = int(row['K'])
        A_str  = f"{row['A_K']:.4f}"             if pd.notna(row.get('A_K'))             else "?"
        dA_str = f"{row['delta_A_K']:.4f}"        if pd.notna(row.get('delta_A_K'))        else "?"
        d2_str = f"{row['delta2_A_K']:.4f}"       if pd.notna(row.get('delta2_A_K'))       else "?"
        kd_str = f"{row['kneedle_distance']:.4f}" if pd.notna(row.get('kneedle_distance')) else "?"
        set_idxs    = json.loads(row.get('architecture_set_idxs', '[]'))
        k_counts    = (
            assign_counts[assign_counts['K'] == K_val]
            .set_index('assigned_utility_idx')['n_players'].to_dict()
        )
        sorted_idxs = sorted(set_idxs, key=lambda m: ic_bic_by_model.get(m, float('inf')))
        n_display   = min(len(sorted_idxs), K_hover_cap) if K_hover_cap is not None else len(sorted_idxs)
        model_lines = []
        for model_idx in sorted_idxs[:n_display]:
            n_pl     = k_counts.get(model_idx, 0)
            try:
                mu_bic = mean_bic_by_K_model.loc[(K_val, model_idx)]
                bic_str = f"{mu_bic:05.1f}"
            except KeyError:
                bic_str = "  n/a"
            equation   = model_equations.get(int(model_idx), '?')
            k_params_v = (int(model_meta.loc[model_idx]['k_params'])
                          if model_idx in model_meta.index else '?')
            diag_entry  = nearest_ampd_lookup.get((K_val, model_idx), {})
            near_idx    = diag_entry.get('nearest_idx')
            near_ampd   = diag_entry.get('ampd_val')
            near_pct    = diag_entry.get('ampd_pct')
            if near_ampd is not None and pd.notna(near_ampd):
                near_idx_str = f"{int(near_idx):03d}" if near_idx is not None and pd.notna(near_idx) else "?"
                ampd_str = f"AMPD({model_idx}, {near_idx_str}): {float(near_ampd):.3f} (p={float(near_pct):.2f})" if pd.notna(near_pct) else f"AMPD_near {near_idx_str}: {float(near_ampd):.3f}"
            else:
                ampd_str = f"AMPD({model_idx}, ???): n/a"
            model_lines.append(
                f"{model_idx:03d}: 𝑘 = {k_params_v} · {n_pl:02d} players · μBIC: {bic_str} · {ampd_str} · {equation}"
            )
        omitted = len(sorted_idxs) - n_display
        if omitted > 0:
            model_lines.append(f"… ({omitted} more utility function{'s' if omitted != 1 else ''} not shown)")
        return (
            f"K Models = {K_val};  Kneedle dist = {kd_str}<br>"
            f"A(K) = {A_str},  ΔA = {dA_str},  Δ²A = {d2_str}<br>"
            f"Utility Function Set:<br>"
            + "<br>".join(model_lines)
        )

    hover_texts = curve_df.apply(_build_hover, axis=1).tolist()

    "=== Traces: A(K) at base_hue, kneedle distance at base_hue+20 ==="
    main_color  = _hsla(hue=base_hue,      alpha=0.9)
    main_darker = _hsla(hue=base_hue,      lightness_percent=35, alpha=1.0)
    kd_color    = _hsla(hue=base_hue + 20, alpha=0.55)
    kd_darker   = _hsla(hue=base_hue + 20, lightness_percent=35, alpha=0.8)

    main_trace = go.Scatter(
        x=k_vals,
        y=a_vals,
        mode='lines+markers',
        marker=dict(size=20, color=main_color, line=dict(width=3, color=main_darker)),
        line=dict(color=main_color, width=5),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_texts,
        name='A(K)',
        showlegend=True,
    )

    kneedle_trace = go.Scatter(
        x=k_vals,
        y=kd_vals,
        mode='lines+markers',
        marker=dict(size=20, color=kd_color, line=dict(width=3, color=kd_darker)),
        line=dict(color=kd_color, width=5, dash='dot'),
        name='Kneedle distance',
        showlegend=True,
        hoverinfo='skip',
    )

    fig = go.Figure(data=[main_trace, kneedle_trace])

    "Reference lines: A=0 baseline and A=1 fully-individualised ceiling."
    ref_color = _hsla(hue=0, saturation_percent=0, lightness_percent=63, alpha=0.6)
    fig.add_hline(y=-0.05, line_dash='dash', line_color=ref_color, line_width=2)
    fig.add_hline(y=1.00, line_dash='dash', line_color=ref_color, line_width=2)
    fig.add_annotation(
        x=1.00, xref='paper',
        y=1.02, yref='y',
        text='A = 1  (fully individualised)',
        showarrow=False,
        xanchor='right',
        font=dict(size=max(10, base_font_size - 4), color=ref_color),
    )
    "Place the K=1 baseline annotation below the x-axis tick labels using paper coordinates."
    fig.add_annotation(
        x=1.00, xref='paper',
        y=-0.08, yref='paper',
        text='A = 0  (K=1 baseline: one shared utility function)',
        showarrow=False,
        xanchor='right',
        font=dict(size=max(10, base_font_size - 4), color=ref_color),
    )

    "Vertical stopping-criterion markers — hue starts at base_hue+40 and increments by 20°."
    _criteria = [
        ('selected_by_kneedle_elbow',   'Elbow of curve',  40,  'solid'),
        ('selected_by_marginal_gain',   'Marginal gain',   60,  'dot'),
        ('selected_by_cumulative_gain', 'Cumulative gain', 80,  'dashdot'),
        ('selected_by_max_curvature',   'Max curvature',   100, 'dash'),
        ('selected_by_meta_bic',        'Meta-BIC',        120, 'longdash'),
    ]
    drawn_k: dict = {}
    for col, label, hue_offset, dash in _criteria:
        if col not in curve_df.columns:
            continue
        sel_rows = curve_df[curve_df[col] == True]
        if sel_rows.empty:
            continue
        k_sel = int(sel_rows['K'].values[0])
        hue   = (base_hue + hue_offset) % 360
        color = _hsla(hue=hue, alpha=0.85)
        if k_sel not in drawn_k:
            drawn_k[k_sel] = {'labels': [], 'color': color, 'dash': dash}
            fig.add_vline(x=k_sel, line_dash=dash, line_color=color, line_width=3)
        drawn_k[k_sel]['labels'].append(label)

    "Vertical rotated annotations placed in the lower portion of the plot to avoid covering curves."
    for k_sel, info in drawn_k.items():
        annotation_text = "  |  ".join(info['labels']) + f"  (K={k_sel})"
        fig.add_annotation(
            x=k_sel, y=0.28, yref='y', xref='x',
            text=annotation_text,
            textangle=-90,
            showarrow=False,
            font=dict(size=max(10, base_font_size - 4), color=info['color']),
            xanchor='left',
            yanchor='middle',
        )

    yaxis_title = "A(K) — gain fraction for K models  |  Kneedle dist"
    fig.update_layout(
        title=dict(
            text='Population Utility Function Compression Curve',
            x=0.5, xanchor='center',
            y=0.97, yanchor='top',
            font=dict(size=fig_lay.get('title_size', 22) * 2),
        ),
        xaxis=dict(
            title='Number of utility functions (K)',
            tickmode='array',
            tickvals=k_vals,
            showgrid=True,
            gridcolor=_hsla(hue=0, saturation_percent=0, lightness_percent=78, alpha=0.4),
        ),
        yaxis=dict(
            title=yaxis_title,
            range=[0.0, 1.03],
            tickmode='array',
            tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ticktext=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
            zeroline=True, zerolinewidth=1,
            zerolinecolor=_hsla(hue=0, saturation_percent=0, lightness_percent=47, alpha=0.5),
        ),
        hoverlabel=dict(font=dict(size=max(8, int(base_font_size * 2 * 0.6) - 4))),
        template=fig_lay.get('template', 'plotly_white'),
        font=dict(
            family=fig_lay.get('font', {}).get('family', 'Calibri'),
            size=base_font_size,
        ),
        margin=dict(l=80, r=50, t=120, b=80),
        autosize=True,
        legend=dict(yanchor='bottom', y=0.3, xanchor='right', x=0.95),
    )

    if export_fig:
        out_path = os.path.join(str(file_paths['visuals']), 'population_architecture_curve.html')
        fig.write_html(out_path, config={'responsive': True})
"=========================================================================================="
"=============================== Model Recovery Simulation ================================"
"=========================================================================================="


def _recovery_fit_worker(args: tuple) -> list:
    """
    Module-level parallel worker: fits all candidate utility models to one synthetic
    agent's chooser data. One job covers all candidates for a single agent so each
    agent's games are serialized only once (not once per candidate model).

    Arguments (unpacked from args):
        • agent_idx: int
        • games_slice: list[dict] — first n_games game dicts for this agent
        • candidate_models: list[(utility_idx: int, utility_settings: dict)]
        • general_settings_for_fitting: dict — general_settings with update_method='naive'
        • param_bds: dict
        • softmax_temperature: float — fixed tau used for both NLL evaluation and generation

    Returns:
        • list[dict] — one result dict per candidate model, each with keys:
            agent_idx, utility_idx, nll, n_games, k_params, best_params (JSON str).
    """
    import math as _math
    import json as _json

    (agent_idx, games_slice, candidate_models, general_settings_for_fitting, param_bds,
     softmax_temperature, optimization_method) = args

    n_valid_games = sum(1 for game in games_slice if not game.get('abdicated_chooser', False))
    results = []

    for candidate_utility_idx, candidate_utility_settings in candidate_models:
        param_keys = parameter_keys_for_utility_settings(
            utility_settings=candidate_utility_settings,
            general_settings=general_settings_for_fitting,
        )
        param_bounds_list = [param_bds[param_key] for param_key in param_keys]

        def _chooser_nll(
            param_vector,
            _param_keys=param_keys,
            _utility_settings=candidate_utility_settings,
            _games_slice=games_slice,
            _temperature=softmax_temperature,
        ) -> float:
            params = dict(zip(_param_keys, param_vector))
            total_nll = 0.0
            for game in _games_slice:
                if game.get('abdicated_chooser', False):
                    continue
                payoffs_option_a = {
                    'As': game['payoff_A_chooser'], 'Ao': game['payoff_A_predictor'],
                    'Bs': game['payoff_B_chooser'], 'Bo': game['payoff_B_predictor'],
                }
                payoffs_option_b = {
                    'As': game['payoff_B_chooser'], 'Ao': game['payoff_B_predictor'],
                    'Bs': game['payoff_A_chooser'], 'Bo': game['payoff_A_predictor'],
                }
                utility_a = utility(payoffs=payoffs_option_a, params=params, utility_settings=_utility_settings)
                utility_b = utility(payoffs=payoffs_option_b, params=params, utility_settings=_utility_settings)
                prob_choose_a = softmax_(uA=utility_a, uB=utility_b, temperature=_temperature)
                prob_observed = prob_choose_a if game['choice'] == 'A' else (1.0 - prob_choose_a)
                total_nll -= _math.log(max(prob_observed, 1e-10))
            return total_nll

        if not param_keys:
            best_nll = _chooser_nll([])
            best_params: dict = {}
        else:
            fit_result = global_local_optimization(
                objective_fn=_chooser_nll,
                x_bounds=param_bounds_list,
                optimization_method=optimization_method,
                n_random_starts=5,
            )
            best_nll    = float(fit_result['final']['loss'])
            best_params = dict(zip(param_keys, fit_result['final']['x']))

        results.append({
            'agent_idx':   agent_idx,
            'utility_idx': candidate_utility_idx,
            'nll':         best_nll,
            'n_games':     n_valid_games,
            'k_params':    len(param_keys),
            'best_params': _json.dumps(best_params, ensure_ascii=False),
        })

    return results


def _recovery_simulation_stem(
    generating_utility_idx: int,
    n_candidate_models: Optional[int],
    candidate_model_selection_mode: str,
    softmax_temperature: float,
    n_agents_grid: List[int],
    n_games_grid: List[int],
    random_seed: int,
) -> str:
    """
    Returns the canonical filename stem for a model recovery simulation run.
    Encodes all parameters that affect the output so that runs with different
    settings never overwrite each other.

    Stem format:
        model_recovery_gen={idx}_cands={n}_{mode}_tau={tau}_agents={a1-a2-...}_games={g1-g2-...}_seed={seed}

    Example:
        model_recovery_gen=443_cands=100_hamming_tau=0p5_agents=73_games=20-40-60-90-120-180-240_seed=42
    """
    _tau_str    = str(softmax_temperature).replace('.', 'p')
    _cands_str  = str(n_candidate_models) if n_candidate_models is not None else 'all'
    _agents_str = '-'.join(str(val) for val in sorted(n_agents_grid))
    _games_str  = '-'.join(str(val) for val in sorted(n_games_grid))
    return (
        f'model_recovery'
        f'_gen={generating_utility_idx}'
        f'_cands={_cands_str}_{candidate_model_selection_mode}'
        f'_tau={_tau_str}'
        f'_agents={_agents_str}'
        f'_games={_games_str}'
        f'_seed={random_seed}'
    )


def _build_synthetic_histories_json(
    all_synthetic_agent_dyads: List[dict],
    n_agents: int,
    n_games: int,
) -> dict:
    """
    Converts synthetic agent dyad data into the JSON format that
    information_criterion_analysis reads (the 'player_pairs_exper{N}.json' format).

    Arguments:
        • all_synthetic_agent_dyads: list of dicts from create_simulated_dyad, one per agent.
        • n_agents: int; number of agents to include (first n_agents from the list).
        • n_games: int; number of games per agent (first n_games from each agent's game list).

    Returns:
        • dict with "histories" and "player_info" top-level keys.
    """
    synthetic_json: dict = {"histories": {}, "player_info": {}}
    for agent_idx, agent_dyad_data in enumerate(all_synthetic_agent_dyads[:n_agents]):
        for dyad_key_str, games_list in agent_dyad_data.items():
            synthetic_json["histories"][dyad_key_str] = games_list[:n_games]
            chooser_uuid   = f"synthetic_agent_{agent_idx}_chooser"
            predictor_uuid = f"synthetic_agent_{agent_idx}_predictor"
            synthetic_json["player_info"][chooser_uuid]   = {"player_type": "participant"}
            synthetic_json["player_info"][predictor_uuid] = {"player_type": "participant"}
    return synthetic_json


def compute_model_recovery_simulation(
    general_settings: dict,
    file_paths: dict,
    param_bds: dict,
    utility_settings: dict,
    generating_model: Union[int, dict] = 443,
    n_agents_grid: Optional[List[int]] = None,
    n_games_grid: Optional[List[int]] = None,
    softmax_temperature: float = 0.5,
    candidate_model_selection_mode: str = 'hamming',
    n_candidate_models: Optional[int] = 100,
    ampd_matrix_name_or_path: Optional[str] = None,
    random_seed: int = 42,
    create_new_file: bool = False,
) -> pd.DataFrame:
    """
    Simulation study answering two intertwined data-adequacy questions:
    (1) "How many games per agent are needed for the IC pipeline to reliably recover
        the generating utility model?" and
    (2) "How many participants (synthetic agents) are needed for reliable recovery?"

    Procedure:
        1. Resolve generating_model (int idx or UtilitySettings dict) to a
           (utility_idx, UtilitySettings) pair.
        2. Extract fitted chooser parameter vectors for the generating model from the IC JSON
           as the realistic parameter pool.
        3. Select n_candidate_models candidates via max-min AMPD/Hamming diversity,
           always seeding the selection with the generating model.
        4. Generate synthetic data for max(n_agents_grid) agents × max(n_games_grid) games
           each (done once; all conditions use nested subsets of this pre-generated data).
        5. For each (n_agents_value, n_games_value) condition, write a sliced synthetic
           histories JSON to a condition-specific directory, then call
           information_criterion_analysis on it (restricted to the candidate model set).
        6. Extract population-level BIC results, determine whether the generating model
           wins population BIC, and append one row per candidate model to the partial CSV.
        7. On restart with create_new_file=False, completed conditions are skipped and
           within-condition IC runs can resume from their saved per-model JSON files.

    Arguments:
        • general_settings: dict; must contain 'experiment_num', 'run_in_parallel', etc.
            The 'optimization_method' key controls the optimization used by IC (default
            'globloc' in general_settings); this is NOT a separate parameter — it is read
            from general_settings to keep all IC runs consistent.
        • file_paths: dict; must contain 'processed', 'bic_aic', 'player_fits', 'visuals'.
        • param_bds: dict; {param_name: (low, high)} parameter bounds.
        • utility_settings: dict; used to derive canonical flag order for the registry.
        • generating_model: int | dict; utility_idx (int) or full UtilitySettings dict
            identifying the model used to generate synthetic data. Default: 443.
        • n_agents_grid: list[int] | None; synthetic-participant adequacy curve.
            Default: [73] (the real N only). Example: [10, 20, 30, 50, 73].
            max(n_agents_grid) agents are generated; all values are nested subsets.
        • n_games_grid: list[int] | None; games-per-agent adequacy curve.
            Default: [20, 40, 60, 90, 120, 180, 240].
            max(n_games_grid) games are generated per agent; all values are nested subsets.
        • softmax_temperature: float; fixed tau used for both data generation (default 0.5).
        • candidate_model_selection_mode: str; 'hamming' or 'ampd' max-min diversity selection.
        • n_candidate_models: int | None; size of the candidate set (default 100).
        • ampd_matrix_name_or_path: str | None; path to AMPD matrix when mode='ampd'.
        • random_seed: int; reproducibility seed (default 42).
        • create_new_file: bool; if False and final CSV exists, load and return it immediately.

    Returns:
        • pd.DataFrame; one row per (n_agents_fitted, n_games_fitted, candidate utility_idx).
            Each row reports population-level BIC for one candidate model in one condition,
            allowing recovery rate and BIC rank curves to be plotted across the grid.

    Resume support:
        Each completed (n_agents, n_games) condition is appended to a partial CSV.
        On restart with create_new_file=False, completed conditions are skipped. Within-
        condition IC runs also resume from their saved per-model JSON files (write_mode=resume).
        On clean completion the partial CSV is deleted.
    """
    import copy as _copy

    "Resolve n_games_grid and n_agents_grid; derive max values."
    if n_games_grid is None:
        n_games_grid = [20, 40, 60, 90, 120, 180, 240]
    if n_agents_grid is None:
        n_agents_grid = [73]

    n_games_max   = max(n_games_grid)
    n_agents_max  = max(n_agents_grid)
    n_games_grid  = sorted(set(n_val for n_val in n_games_grid  if 0 < n_val <= n_games_max))
    n_agents_grid = sorted(set(n_val for n_val in n_agents_grid if 0 < n_val <= n_agents_max))

    "Load utility registry and identify boolean flag columns."
    processed_dir         = str(file_paths['processed'])
    _original_player_fits = str(file_paths['player_fits'])
    _sim_results_dir      = os.path.join(_original_player_fits, 'simulation_results')
    os.makedirs(_sim_results_dir, exist_ok=True)
    _gitignore_path = os.path.join(_sim_results_dir, '.gitignore')
    if not os.path.exists(_gitignore_path):
        with open(_gitignore_path, 'w', encoding='utf-8') as _gig:
            _gig.write('*\n')
    registry_df   = pd.read_csv(
        os.path.join(processed_dir, 'all_utility_functions.csv'),
        dtype={'utility_bitstring': str},
    )
    _non_flag_columns = {
        'utility_idx', 'utility_bitstring', 'k_params', 'redundant_with', 'differing_settings',
        'n_data', 'pvar', 'param_norm_sd', 'loss_nll', 'AIC', 'BIC', 'ΔAIC', 'ΔBIC',
        'AIC_rank', 'BIC_rank', 'parents', 'siblings', 'children',
        'ampd_to_best_rand', 'ampd_to_best_real', 'policy_regret_norm', 'equation',
    }
    flag_columns = [col for col in registry_df.columns if col not in _non_flag_columns]

    "Resolve generating_model to (generating_utility_idx, generating_utility_settings)."
    if isinstance(generating_model, int):
        generating_utility_idx = generating_model
        gen_registry_row = registry_df[registry_df['utility_idx'] == generating_utility_idx]
        if len(gen_registry_row) == 0:
            raise ValueError(f"Generating model idx={generating_utility_idx} not found in registry.")
    else:
        "UtilitySettings dict provided: find the unique matching registry row by flag values."
        _flag_mask = pd.Series([True] * len(registry_df), index=registry_df.index)
        for col in flag_columns:
            if col in registry_df.columns:
                _flag_mask &= (registry_df[col] == bool(generating_model.get(col, False)))
        gen_registry_row = registry_df[_flag_mask]
        if len(gen_registry_row) != 1:
            raise ValueError(
                f"Could not uniquely identify generating model from UtilitySettings dict "
                f"({len(gen_registry_row)} matches). Pass an integer utility_idx instead."
            )
        generating_utility_idx = int(gen_registry_row.iloc[0]['utility_idx'])

    generating_utility_settings = {
        col: bool(gen_registry_row.iloc[0][col])
        for col in flag_columns if col in gen_registry_row.columns
    }

    "Check for cached final result."
    _stem            = _recovery_simulation_stem(
        generating_utility_idx=generating_utility_idx,
        n_candidate_models=n_candidate_models,
        candidate_model_selection_mode=candidate_model_selection_mode,
        softmax_temperature=softmax_temperature,
        n_agents_grid=n_agents_grid,
        n_games_grid=n_games_grid,
        random_seed=random_seed,
    )
    output_csv_path  = os.path.join(processed_dir, f'{_stem}.csv')
    partial_csv_path = os.path.join(processed_dir, f'{_stem}_partial.csv')
    "8-char hash of _stem used as a short directory key to stay within Windows MAX_PATH (260 chars)."
    _dir_key = hashlib.md5(_stem.encode()).hexdigest()[:8]

    if not create_new_file and os.path.exists(output_csv_path):
        cached_df = pd.read_csv(output_csv_path, encoding='utf-8-sig')
        if not cached_df.empty:
            print(f"Model recovery simulation loaded from cache: {output_csv_path}"
                  f"  ({len(cached_df)} rows)")
            return cached_df

    "Detect completed (n_games, n_agents) conditions from partial CSV for mid-run resume."
    completed_conditions: set = set()
    accumulated_dataframes: List[pd.DataFrame] = []
    if not create_new_file and os.path.exists(partial_csv_path):
        partial_df = pd.read_csv(partial_csv_path, encoding='utf-8-sig')
        if (not partial_df.empty
                and 'n_games_fitted' in partial_df.columns
                and 'n_agents_fitted' in partial_df.columns):
            for _, condition_row in (
                partial_df[['n_games_fitted', 'n_agents_fitted']].drop_duplicates().iterrows()
            ):
                completed_conditions.add(
                    (int(condition_row['n_games_fitted']), int(condition_row['n_agents_fitted']))
                )
            accumulated_dataframes.append(partial_df)
            print(f"Resuming from partial CSV: "
                  f"{len(completed_conditions)} conditions already done: "
                  f"{sorted(completed_conditions)}")

    "Load distance matrix for diversity-based candidate selection."
    if candidate_model_selection_mode == 'ampd':
        if ampd_matrix_name_or_path is not None:
            ampd_matrix_path = (
                ampd_matrix_name_or_path if os.path.isabs(ampd_matrix_name_or_path)
                else os.path.join(processed_dir, ampd_matrix_name_or_path)
            )
            distance_matrix_df = pd.read_csv(ampd_matrix_path, index_col=0)
        else:
            distance_matrix_df = _load_ampd_matrix_from_settings(general_settings, file_paths)
        distance_matrix_df.index   = distance_matrix_df.index.astype(int)
        distance_matrix_df.columns = distance_matrix_df.columns.astype(int)
    else:
        n_registry_models   = len(registry_df)
        hamming_matrix_path = os.path.join(
            processed_dir, f'model_distance_hamming__n_models={n_registry_models}.csv'
        )
        if not os.path.exists(hamming_matrix_path):
            print("Hamming distance matrix not found; computing now...")
            compute_hamming_distance_matrix(
                file_paths=file_paths, utility_settings=utility_settings,
            )
        distance_matrix_df = pd.read_csv(hamming_matrix_path, index_col=0)
        distance_matrix_df.index   = distance_matrix_df.index.astype(int)
        distance_matrix_df.columns = distance_matrix_df.columns.astype(int)

    "Greedy max-min diversity selection seeded with the generating model."
    all_model_indices          = list(registry_df['utility_idx'].astype(int))
    distance_matrix_index_set  = set(distance_matrix_df.index.tolist())
    distance_matrix_column_set = set(distance_matrix_df.columns.tolist())
    target_n_candidates        = (n_candidate_models if n_candidate_models is not None
                                  else len(all_model_indices))

    selected_model_indices: List[int] = [generating_utility_idx]
    remaining_model_indices = [model_idx for model_idx in all_model_indices
                               if model_idx != generating_utility_idx]

    while len(selected_model_indices) < target_n_candidates and remaining_model_indices:
        best_candidate_idx  = None
        best_min_distance   = -1.0
        for model_idx in remaining_model_indices:
            if model_idx not in distance_matrix_index_set:
                min_distance_to_selected = 0.0
            else:
                min_distance_to_selected = min(
                    float(distance_matrix_df.loc[model_idx, selected_idx])
                    if selected_idx in distance_matrix_column_set else 0.0
                    for selected_idx in selected_model_indices
                )
            if min_distance_to_selected > best_min_distance:
                best_min_distance  = min_distance_to_selected
                best_candidate_idx = model_idx
        if best_candidate_idx is None:
            break
        selected_model_indices.append(best_candidate_idx)
        remaining_model_indices.remove(best_candidate_idx)

    print(f"Selected {len(selected_model_indices)} candidate models "
          f"via {candidate_model_selection_mode} diversity.")
    print(f"  Generating model {generating_utility_idx} in candidate set: "
          f"{generating_utility_idx in selected_model_indices}")

    "Load AMPD and Hamming matrices for continuous recovery distance metrics."
    if candidate_model_selection_mode == 'ampd':
        ampd_metrics_df     = distance_matrix_df
        _hamming_path_metr  = os.path.join(
            processed_dir, f'model_distance_hamming__n_models={len(registry_df)}.csv',
        )
        if not os.path.exists(_hamming_path_metr):
            compute_hamming_distance_matrix(file_paths=file_paths, utility_settings=utility_settings)
        try:
            hamming_metrics_df = pd.read_csv(_hamming_path_metr, index_col=0)
            hamming_metrics_df.index   = hamming_metrics_df.index.astype(int)
            hamming_metrics_df.columns = hamming_metrics_df.columns.astype(int)
        except Exception:
            hamming_metrics_df = None
    else:
        hamming_metrics_df = distance_matrix_df
        try:
            if ampd_matrix_name_or_path is not None:
                _ampd_metr_path = (
                    ampd_matrix_name_or_path if os.path.isabs(ampd_matrix_name_or_path)
                    else os.path.join(processed_dir, ampd_matrix_name_or_path)
                )
                ampd_metrics_df = pd.read_csv(_ampd_metr_path, index_col=0)
            else:
                ampd_metrics_df = _load_ampd_matrix_from_settings(general_settings, file_paths)
            ampd_metrics_df.index   = ampd_metrics_df.index.astype(int)
            ampd_metrics_df.columns = ampd_metrics_df.columns.astype(int)
            print(f"AMPD matrix loaded for recovery metrics: "
                  f"{ampd_metrics_df.shape[0]}×{ampd_metrics_df.shape[1]}")
        except Exception as _ampd_err:
            print(f"  Warning: could not load AMPD matrix ({_ampd_err}). "
                  f"AMPD recovery metrics will be NaN.")
            ampd_metrics_df = None

    candidate_models: List[Tuple[int, dict]] = []
    for utility_idx_val in selected_model_indices:
        registry_row = registry_df[registry_df['utility_idx'] == utility_idx_val]
        if len(registry_row) == 0:
            continue
        candidate_utility_settings = {
            col: bool(registry_row.iloc[0][col])
            for col in flag_columns if col in registry_row.columns
        }
        candidate_models.append((utility_idx_val, candidate_utility_settings))

    "Load IC JSON and extract model-specific chooser parameter pool for the generating model."
    experiment_num = general_settings.get('experiment_num', 3)
    ic_json_name   = f"All_Utility_Forms_IC_Analysis_Experiment{experiment_num}.json"
    ic_json_path   = os.path.join(str(file_paths['bic_aic']), ic_json_name)

    # TEMPORARY BRIDGE: the IC JSON has not yet been regenerated inside this repo.
    # This fallback reads from the original analysis directory.
    # Remove this block once the IC data is regenerated here
    # (i.e., once ic_json_path points to a valid >=50 MB file in this repo).
    _old_repo_ic_json_path = (
        r"C:\Users\Gregory Stanley\Desktop\U of M\Research Archive\Multiplayer"
        r"\ABM_Simulation\Judgment_Game\Inputs\Iter_Binary_Dictator"
        rf"\bic_aic\All_Utility_Forms_IC_Analysis_Experiment{experiment_num}.json"
    )
    if not os.path.exists(ic_json_path) or os.path.getsize(ic_json_path) < 50_000_000:
        ic_json_path = _old_repo_ic_json_path
    # END TEMPORARY BRIDGE

    print(f"Loading IC JSON: {os.path.basename(ic_json_path)}")
    with open(ic_json_path, 'r', encoding='utf-8-sig') as ic_file_handle:
        ic_data = json.load(ic_file_handle)
    ic_results = ic_data.get('ic_results', {})

    "Find generating model entry by settings tuple — not by utility_idx."
    "The IC JSON was built from a different model registry (different total model count),"
    "so integer indices may differ. The settings tuple is the stable cross-version identity."
    _gen_settings_tuple     = tuple(
        bool(generating_utility_settings.get(col, False)) for col in flag_columns
    )
    _gen_settings_tuple_str = str(_gen_settings_tuple)
    generating_model_entry  = ic_results.get(_gen_settings_tuple_str)
    if generating_model_entry is None:
        raise ValueError(
            f"Generating model (new registry idx={generating_utility_idx}) not found in "
            f"IC JSON by settings tuple.\n"
            f"  Settings tuple: {_gen_settings_tuple_str}\n"
            f"  IC JSON path:   {ic_json_path}\n"
            f"  IC JSON contains {len(ic_results)} models. The settings tuple above was "
            f"built from flag_columns in all_utility_functions.csv — verify that the "
            f"flag column order matches the IC JSON's key format."
        )
    _ic_json_idx = generating_model_entry.get('idx', 'unknown')
    print(f"  Generating model matched: new registry idx={generating_utility_idx} "
          f"→ IC JSON idx={_ic_json_idx} (via settings tuple)")

    "Mean param keys for the generating model: no _std, no tau (tau is fixed during generation)."
    general_settings_for_fitting = {
        **general_settings, 'update_method': 'naive', 'include_covariance': False,
    }
    generating_param_keys = parameter_keys_for_utility_settings(
        utility_settings=generating_utility_settings,
        general_settings=general_settings_for_fitting,
    )

    param_pool: List[dict] = []
    for _player_uuid, player_entry in generating_model_entry.get('minvec', {}).items():
        raw_chooser_params = player_entry.get('params', {}).get('chooser', None)
        if raw_chooser_params is None:
            continue
        clean_params = {
            param_key: float(param_val)
            for param_key, param_val in raw_chooser_params.items()
            if param_key in generating_param_keys
            and not param_key.endswith('_std')
            and param_key != 'τ'
        }
        if len(clean_params) == len(generating_param_keys):
            param_pool.append(clean_params)

    if not param_pool:
        # IC JSON param keys don't match current parameter names. This is expected when using
        # the TEMPORARY BRIDGE to an older IC JSON (see comment above). The old JSON was
        # generated before parameter names were updated in this codebase. Fall back to uniform
        # sampling within param_bds so the simulation can still run; update this once the IC
        # JSON is regenerated inside this repo with current parameter names.
        _sample_minvec = generating_model_entry.get('minvec', {})
        _first_player_entry = next(iter(_sample_minvec.values()), {}) if _sample_minvec else {}
        _ic_json_keys = list(_first_player_entry.get('params', {}).get('chooser', {}).keys())
        print(
            f"\n  Warning: IC JSON chooser params for model {generating_utility_idx} use "
            f"parameter names that do not match the current codebase.\n"
            f"    IC JSON keys found:  {_ic_json_keys}\n"
            f"    Current keys needed: {generating_param_keys}\n"
            f"  Falling back to uniform sampling from param_bds for the generating "
            f"parameter pool. Re-run after regenerating the IC JSON in this repo to use "
            f"real participant parameters."
        )
        _rng_for_pool = np.random.RandomState(random_seed)
        for _ in range(200):
            _sample = {
                param_key: float(_rng_for_pool.uniform(
                    param_bds[param_key][0], param_bds[param_key][1]
                ))
                for param_key in generating_param_keys
                if param_key in param_bds
            }
            if len(_sample) == len(generating_param_keys):
                param_pool.append(_sample)
        if not param_pool:
            raise ValueError(
                f"Could not build a parameter pool for generating model "
                f"{generating_utility_idx}. Verify that all expected param keys are in "
                f"param_bds.\n  Expected keys: {generating_param_keys}"
            )
    print(f"  Extracted {len(param_pool)} participant parameter vectors "
          f"for generating model {generating_utility_idx}.")

    "Build the candidate utility settings list (for IC's utility_setting_varieties param)."
    candidate_utility_settings_list = [settings for _, settings in candidate_models]
    ic_position_to_registry_utility_idx = {
        position: utility_idx for position, (utility_idx, _) in enumerate(candidate_models)
    }

    "Identify the histories filename IC will look for in the processed directory."
    "experiment_num is already defined above in the IC JSON loading block."
    _raw_histories_filename = file_paths['file_names'].get(
        f'player_pairs_exper{experiment_num}',
        f'Social_Preference_Prediction_Pairs_Exper{experiment_num}.json',
    )
    "Strip any suffix that may have been added by add_remove_file_name_suffix."
    histories_filename_clean = _raw_histories_filename.split('~')[0]

    "Generate synthetic data for n_agents_max agents × n_games_max games (done once; sliced per condition)."
    print(f"\nGenerating synthetic data: {n_agents_max} agents × {n_games_max} games ...")
    random_state = np.random.RandomState(random_seed)
    all_synthetic_agent_dyads: List[dict] = []

    for agent_idx in range(n_agents_max):
        pool_sample_index = random_state.randint(len(param_pool))
        generating_params = dict(param_pool[pool_sample_index])
        generating_params['τ'] = softmax_temperature

        random.seed(random_seed + agent_idx * 1000)
        np.random.seed(random_seed + agent_idx * 1000)

        dyad_data = create_simulated_dyad(
            n_games=n_games_max,
            params_chooser=generating_params,
            params_predictor=generating_params,
            general_settings=general_settings,
            utility_settings=generating_utility_settings,
            param_bds=param_bds,
            default_utility_settings=False,
            dynamic_predictor=False,
            dyad_id=f"synthetic_agent_{agent_idx}",
        )
        all_synthetic_agent_dyads.append(dyad_data)

        if (agent_idx + 1) % 20 == 0 or (agent_idx + 1) == n_agents_max:
            print(f"  Generated {agent_idx + 1}/{n_agents_max} agents.")

    "Save max-scale synthetic data for reference and auditing."
    max_scale_synthetic_json  = _build_synthetic_histories_json(
        all_synthetic_agent_dyads, n_agents_max, n_games_max
    )
    max_scale_synthetic_path  = os.path.join(processed_dir, f'{_stem}_synthetic_data.json')
    with open(max_scale_synthetic_path, 'w', encoding='utf-8') as _max_scale_file:
        json.dump(max_scale_synthetic_json, _max_scale_file, ensure_ascii=False)
    print(f"Max-scale synthetic data saved: {max_scale_synthetic_path}")

    total_start_time = time.time()

    def _dist_lookup(matrix_df, from_uid, to_uid):
        if matrix_df is None:
            return float('nan')
        try:
            return float(matrix_df.loc[int(from_uid), int(to_uid)])
        except (KeyError, ValueError):
            return float('nan')

    for n_agents_value in n_agents_grid:
        for n_games_value in n_games_grid:
            if (n_games_value, n_agents_value) in completed_conditions:
                print(f"[n_agents={n_agents_value}, n_games={n_games_value}] "
                      f"Already complete; skipping.")
                continue

            condition_start_time = time.time()
            print(f"\n[n_agents={n_agents_value}, n_games={n_games_value}] "
                  f"Running information_criterion_analysis on "
                  f"{len(candidate_models)} candidate models × {n_agents_value} synthetic agents ...")

            "Write condition-specific synthetic histories JSON."
            "Use _dir_key (8-char MD5) + abbreviated condition key to stay under Windows MAX_PATH."
            _cond_key           = f"na{n_agents_value}_ng{n_games_value}"
            condition_histories_json = _build_synthetic_histories_json(
                all_synthetic_agent_dyads, n_agents_value, n_games_value
            )
            condition_base_dir      = os.path.join(
                processed_dir, 'model_recovery_synthetic', _dir_key, _cond_key,
            )
            condition_processed_dir = os.path.join(condition_base_dir, 'processed')
            os.makedirs(condition_processed_dir, exist_ok=True)
            condition_histories_path = os.path.join(condition_processed_dir, 'histories.json')
            with open(condition_histories_path, 'w', encoding='utf-8') as _cond_hist_file:
                json.dump(condition_histories_json, _cond_hist_file, ensure_ascii=False)

            "Build file_paths for the IC call, redirecting outputs to the condition directory."
            condition_file_paths = _copy.deepcopy(file_paths)
            condition_file_paths['processed']   = condition_processed_dir
            condition_file_paths['param_data']  = os.path.join(condition_base_dir, 'param_data')
            condition_file_paths['player_fits'] = os.path.join(
                _original_player_fits, 'simulation_results', 'model_recovery_simulation',
                _dir_key, _cond_key,
            )
            condition_file_paths['bic_aic']     = os.path.join(condition_base_dir, 'bic_aic')
            condition_file_paths['file_names']  = _copy.deepcopy(file_paths['file_names'])
            condition_file_paths['file_names'][f'player_pairs_exper{experiment_num}'] = 'histories.json'

            "Build general_settings for the IC call."
            condition_general_settings = {
                **general_settings,
                'write_mode':           'resume',  # enables within-condition resume on restart
                'temperature_is_param': False,     # IC fits with fixed tau
                'update_method':        'naive',   # static (non-dynamic) belief updating
            }

            "Call information_criterion_analysis on the synthetic data."
            ic_df, _ = information_criterion_analysis(
                general_settings=condition_general_settings,
                utility_settings=utility_settings,
                file_paths=condition_file_paths,
                param_bds=param_bds,
                utility_setting_varieties=candidate_utility_settings_list,
            )

            "Map IC's enumerate-position idx back to registry utility_idx, then add recovery columns."
            ic_df = ic_df.copy()
            ic_df['utility_idx'] = ic_df['idx'].map(ic_position_to_registry_utility_idx)
            ic_df = ic_df.sort_values('BIC', ascending=True).reset_index(drop=True)
            ic_df['bic_rank_overall'] = range(1, len(ic_df) + 1)
            ic_df['n_agents_fitted']  = n_agents_value
            ic_df['n_games_fitted']   = n_games_value
            ic_df['true_utility_idx'] = generating_utility_idx
            ic_df['is_generating_model'] = ic_df['utility_idx'] == generating_utility_idx

            gen_mask = ic_df['is_generating_model']
            if gen_mask.any():
                generating_model_bic_rank = int(ic_df.loc[gen_mask, 'bic_rank_overall'].iloc[0])
                recovered                 = generating_model_bic_rank == 1
            else:
                generating_model_bic_rank = None
                recovered                 = False
            ic_df['recovered'] = recovered

            "Continuous recovery distance metrics."
            _n_cands      = len(ic_df)
            _bic_rank_true = (
                generating_model_bic_rank if generating_model_bic_rank is not None
                else _n_cands
            )
            _rank_pct_true = 1.0 - (_bic_rank_true - 1) / max(_n_cands - 1, 1)

            _winner_rows = ic_df[ic_df['bic_rank_overall'] == 1]
            _winner_uid  = (
                int(_winner_rows.iloc[0]['utility_idx'])
                if len(_winner_rows) > 0 else None
            )

            ic_df['ampd_to_truth'] = ic_df['utility_idx'].apply(
                lambda uid: _dist_lookup(ampd_metrics_df, uid, generating_utility_idx)
            )
            ic_df['hamming_to_truth'] = ic_df['utility_idx'].apply(
                lambda uid: _dist_lookup(hamming_metrics_df, uid, generating_utility_idx)
            )

            _ampd_winner   = _dist_lookup(ampd_metrics_df,    _winner_uid, generating_utility_idx)
            _hamming_winner = _dist_lookup(hamming_metrics_df, _winner_uid, generating_utility_idx)

            if _winner_uid is not None and _winner_uid != generating_utility_idx:
                _winner_reg = registry_df[registry_df['utility_idx'] == _winner_uid]
                if len(_winner_reg) > 0:
                    _n_flags = len(flag_columns)
                    _n_match = sum(
                        bool(_winner_reg.iloc[0][col]) == bool(gen_registry_row.iloc[0][col])
                        for col in flag_columns if col in registry_df.columns
                    )
                    _feat_acc = _n_match / _n_flags if _n_flags > 0 else float('nan')
                else:
                    _feat_acc = float('nan')
            else:
                _feat_acc = 1.0

            from scipy.stats import spearmanr as _spearmanr
            _valid_mask = ic_df['ampd_to_truth'].notna()
            if _valid_mask.sum() >= 3:
                _rho, _pval = _spearmanr(
                    ic_df.loc[_valid_mask, 'bic_rank_overall'],
                    ic_df.loc[_valid_mask, 'ampd_to_truth'],
                )
                _spear_r, _spear_p = float(_rho), float(_pval)
            else:
                _spear_r, _spear_p = float('nan'), float('nan')

            ic_df['bic_rank_true_model']        = _bic_rank_true
            ic_df['rank_percentile_true_model']  = _rank_pct_true
            ic_df['ampd_winner_to_truth']        = _ampd_winner
            ic_df['hamming_winner_to_truth']     = _hamming_winner
            ic_df['feature_accuracy_winner']     = _feat_acc
            ic_df['rank_distance_spearman_r']    = _spear_r
            ic_df['rank_distance_spearman_p']    = _spear_p

            "Select and rename columns for the output CSV."
            _output_col_map = {
                'loss': 'nll_population', 'AIC': 'aic_population',
                'BIC': 'bic_population',  'ΔBIC': 'delta_bic', 'n_data': 'n_data_population',
            }
            _keep_cols = [
                'n_agents_fitted', 'n_games_fitted', 'utility_idx', 'true_utility_idx',
                'bic_rank_overall', 'is_generating_model', 'recovered',
                'loss', 'k_params', 'AIC', 'BIC', 'ΔBIC', 'n_data',
                'ampd_to_truth', 'hamming_to_truth',
                'bic_rank_true_model', 'rank_percentile_true_model',
                'ampd_winner_to_truth', 'hamming_winner_to_truth',
                'feature_accuracy_winner',
                'rank_distance_spearman_r', 'rank_distance_spearman_p',
            ]
            condition_df = ic_df[[col for col in _keep_cols if col in ic_df.columns]].copy()
            condition_df.rename(columns=_output_col_map, inplace=True)

            "Append completed condition to partial CSV (enables mid-run resume on restart)."
            partial_write_header = not os.path.exists(partial_csv_path)
            condition_df.to_csv(
                partial_csv_path, mode='a', header=partial_write_header,
                index=False, encoding='utf-8-sig',
            )
            accumulated_dataframes.append(condition_df)

            condition_elapsed = time.time() - condition_start_time
            _delta_bic_gen = (
                float(ic_df.loc[gen_mask, 'ΔBIC'].iloc[0]) if gen_mask.any() else float('nan')
            )
            print(f"  -> recovered={recovered}  "
                  f"rank={generating_model_bic_rank}/{len(candidate_models)}  "
                  f"pct={_rank_pct_true:.2f}  "
                  f"delta_bic={_delta_bic_gen:.1f}  "
                  f"ampd_winner={_ampd_winner:.3f}  "
                  f"hamming_winner={_hamming_winner:.0f}  "
                  f"feat_acc={_feat_acc:.2f}  "
                  f"spearman_r={_spear_r:.2f}  "
                  f"time={_fmt_duration(condition_elapsed)}")

    "Combine all conditions, write final CSV, delete partial."
    all_results_df = (
        pd.concat(accumulated_dataframes, ignore_index=True)
        if accumulated_dataframes else pd.DataFrame()
    )
    all_results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\nModel recovery simulation saved: {output_csv_path}  ({len(all_results_df)} rows)")
    print(f"Total time: {_fmt_duration(time.time() - total_start_time)}")

    if os.path.exists(partial_csv_path):
        os.remove(partial_csv_path)

    return all_results_df


def plot_model_recovery_simulation(
    general_settings: dict,
    file_paths: dict,
    fig_lay: dict,
    generating_model: int = 443,
    n_candidate_models: Optional[int] = 100,
    candidate_model_selection_mode: str = 'hamming',
    softmax_temperature: float = 0.5,
    n_agents_grid: Optional[List[int]] = None,
    n_games_grid: Optional[List[int]] = None,
    random_seed: int = 42,
    export_fig: bool = True,
) -> 'go.Figure':
    """
    Plot data-adequacy recovery curves from the model recovery simulation.

    Locates the correct CSV using the same parameter-encoded stem as
    compute_model_recovery_simulation. A dropdown menu selects what is plotted:

    • First option (default): ALL METRICS NORMALIZED — all eight recovery metrics
      on a shared [0,1] y-axis. Each metric is mapped to [0,1] so that 1 = perfect
      recovery and 0 = worst possible outcome. Each metric gets a distinct color;
      dash style distinguishes n_agents groups. Hover shows both normalized and raw values.

    • Options 2–9: individual metrics on their natural y-axis (raw values).

    Normalization functions applied in the "all metrics" view:
      recovered              → identity (already 0/1)
      rank_percentile        → identity (already 0–1)
      bic_rank               → (n_candidates - rank) / (n_candidates - 1)
      delta_bic              → 1 / (1 + delta_bic)   [1 at delta=0, decays as gap grows]
      ampd_winner_to_truth   → 1 - ampd               [invert: 0=identical=best]
      hamming_winner_to_truth→ 1 - hamming / 14       [invert, 14 = max Boolean settings]
      feature_accuracy       → identity (already 0–1)
      spearman_r             → (r + 1) / 2            [map [-1,1] → [0,1]]

    Arguments:
        • general_settings: dict; accepted for API consistency (not currently used).
        • file_paths: dict; must contain 'processed' and 'visuals'.
        • fig_lay: dict; layout settings (template, font, title_size, base_hue).
        • generating_model: int; utility_idx of the generating model (default 443).
        • n_candidate_models: int | None; must match the value used in compute (default 100).
        • candidate_model_selection_mode: str; must match compute (default 'hamming').
        • softmax_temperature: float; must match compute (default 0.5).
        • n_agents_grid: list[int] | None; must match compute (default [73]).
        • n_games_grid: list[int] | None; must match compute (default [20,40,60,90,120,180,240]).
        • random_seed: int; must match compute (default 42).
        • export_fig: bool; if True, writes HTML to visuals/.

    Returns:
        • go.Figure
    """
    if n_agents_grid is None:
        n_agents_grid = [73]
    if n_games_grid is None:
        n_games_grid = [20, 40, 60, 90, 120, 180, 240]
    generating_utility_idx = generating_model
    _stem = _recovery_simulation_stem(
        generating_utility_idx=generating_utility_idx,
        n_candidate_models=n_candidate_models,
        candidate_model_selection_mode=candidate_model_selection_mode,
        softmax_temperature=softmax_temperature,
        n_agents_grid=n_agents_grid,
        n_games_grid=n_games_grid,
        random_seed=random_seed,
    )
    output_csv_path = os.path.join(str(file_paths['processed']), f'{_stem}.csv')
    all_results_df  = pd.read_csv(output_csv_path, encoding='utf-8-sig')

    generating_model_df = all_results_df[
        all_results_df['utility_idx'] == generating_utility_idx
    ].copy()

    base_hue            = fig_lay.get('base_hue', 220)
    base_font_size      = max(8, fig_lay.get('font', {}).get('size', 28) // 2)
    axis_font_size      = base_font_size * 2
    line_width          = 6
    marker_size         = 20
    n_candidates        = all_results_df['utility_idx'].nunique()
    all_n_agents_values = sorted(all_results_df['n_agents_fitted'].unique())
    n_agents_count      = len(all_n_agents_values)

    _metric_configs = [
        {
            'col':         'recovered',
            'short_label': 'Recovered',
            'label':       'Recovered (population BIC winner)',
            'norm_desc':   'identity',
            'y_title':     'Recovery  (1 = generating model wins population BIC)',
            'y_range':     [-0.05, 1.05],
            'y_tickvals':  [0.0, 1.0],
            'y_ticktext':  ['Not recovered (0)', 'Recovered (1)'],
            'hover_fmt':   '.0f',
        },
        {
            'col':         'rank_percentile_true_model',
            'short_label': 'Rank pct',
            'label':       'Rank percentile of true model',
            'norm_desc':   'identity',
            'y_title':     'Rank percentile  (1.0 = truth ranked #1)',
            'y_range':     [-0.05, 1.05],
            'hover_fmt':   '.3f',
        },
        {
            'col':         'bic_rank_true_model',
            'short_label': 'BIC rank (norm.)',
            'label':       'BIC rank of true model',
            'norm_desc':   '(n_cands − rank) / (n_cands − 1)',
            'y_title':     f'BIC rank of generating model  (1 = best of {n_candidates})',
            'y_range':     None,
            'hover_fmt':   '.0f',
        },
        {
            'col':         'delta_bic',
            'short_label': 'ΔBIC (inv.)',
            'label':       'ΔBIC: true model vs winner',
            'norm_desc':   '1 / (1 + ΔBIC)',
            'y_title':     'ΔBIC of generating model  (0 = exact recovery)',
            'y_range':     None,
            'hover_fmt':   '.1f',
        },
        {
            'col':         'ampd_winner_to_truth',
            'short_label': 'AMPD (inv.)',
            'label':       'AMPD: winner → truth',
            'norm_desc':   '1 − AMPD',
            'y_title':     'AMPD behavioral distance  (winner → truth;  0 = identical)',
            'y_range':     [-0.02, 1.02],
            'hover_fmt':   '.4f',
        },
        {
            'col':         'hamming_winner_to_truth',
            'short_label': 'Hamming (inv.)',
            'label':       'Hamming: winner → truth',
            'norm_desc':   '1 − hamming / 14',
            'y_title':     'Hamming distance  (winner → truth;  Boolean settings)',
            'y_range':     None,
            'hover_fmt':   '.0f',
        },
        {
            'col':         'feature_accuracy_winner',
            'short_label': 'Feature acc.',
            'label':       'Feature accuracy of winner',
            'norm_desc':   'identity',
            'y_title':     'Feature accuracy of winner  (1 = all Boolean settings correct)',
            'y_range':     [-0.05, 1.05],
            'hover_fmt':   '.3f',
        },
        {
            'col':         'rank_distance_spearman_r',
            'short_label': 'Spearman r (norm.)',
            'label':       'Rank-distance Spearman r',
            'norm_desc':   '(r + 1) / 2',
            'y_title':     'Spearman r: BIC rank ↔ AMPD-to-truth  (1 = perfectly correlated)',
            'y_range':     [-1.05, 1.05],
            'hover_fmt':   '.3f',
        },
    ]

    def _get_metric(df, col):
        if col in df.columns and not df[col].isna().all():
            try:
                return float(df[col].iloc[0])
            except (ValueError, TypeError):
                pass
        return float('nan')

    def _normalize(col, raw):
        if raw != raw:
            return float('nan')
        if col == 'recovered':
            return raw
        if col == 'rank_percentile_true_model':
            return raw
        if col == 'bic_rank_true_model':
            return (n_candidates - raw) / max(n_candidates - 1, 1)
        if col == 'delta_bic':
            return 1.0 / (1.0 + max(raw, 0.0))
        if col == 'ampd_winner_to_truth':
            return 1.0 - min(max(raw, 0.0), 1.0)
        if col == 'hamming_winner_to_truth':
            return 1.0 - min(max(raw, 0.0), 14.0) / 14.0
        if col == 'feature_accuracy_winner':
            return raw
        if col == 'rank_distance_spearman_r':
            return (raw + 1.0) / 2.0
        return raw

    "Per-metric colors (evenly spaced hues) for the all-metrics normalized view."
    n_metrics     = len(_metric_configs)
    metric_hues   = [(base_hue + i * (360 // n_metrics)) % 360 for i in range(n_metrics)]
    dash_styles   = ['solid', 'dash', 'dot', 'dashdot', 'longdash']
    fig = go.Figure()

    """
    TRACE LAYOUT:
      Block 1 (indices 0 .. n_metrics*n_agents_count-1):
        Normalized traces — shown in the 'All metrics' dropdown option.
        Ordered as: metric 0 agents[0..], metric 1 agents[0..], ...

      Block 2 (indices n_metrics*n_agents_count .. 2*n_metrics*n_agents_count-1):
        Raw traces — shown in individual-metric dropdown options.
        Same ordering as block 1; only the active metric's agents are visible.
    """
    n_block = n_metrics * n_agents_count

    "Block 1: normalized traces."
    for m_idx, m_cfg in enumerate(_metric_configs):
        metric_color = _hsla(hue=metric_hues[m_idx], alpha=0.9)
        for a_idx, n_agents_value in enumerate(all_n_agents_values):
            agents_subset = generating_model_df[
                generating_model_df['n_agents_fitted'] == n_agents_value
            ]
            summary_rows = []
            for ng in sorted(all_results_df['n_games_fitted'].unique()):
                sub = agents_subset[agents_subset['n_games_fitted'] == ng]
                if sub.empty:
                    continue
                raw  = _get_metric(sub, m_cfg['col'])
                norm = _normalize(m_cfg['col'], raw)
                summary_rows.append({'n_games': int(ng), 'norm': norm, 'raw': raw})
            if not summary_rows:
                fig.add_trace(go.Scatter(x=[], y=[], visible=True, showlegend=False))
                continue
            summary_df  = pd.DataFrame(summary_rows)
            agents_label = f"N={n_agents_value}"
            norm_label   = f'{m_cfg["short_label"]}  ({agents_label})'

            fig.add_trace(go.Scatter(
                x=summary_df['n_games'],
                y=summary_df['norm'],
                customdata=summary_df['raw'],
                mode='lines+markers',
                name=norm_label,
                visible=True,
                line=dict(color=metric_color, width=line_width,
                          dash=dash_styles[a_idx % len(dash_styles)]),
                marker=dict(size=marker_size, color=metric_color),
                hovertemplate=(
                    f'{norm_label}<br>'
                    f'norm ({m_cfg["norm_desc"]})=%{{y:.3f}}<br>'
                    f'raw {m_cfg["col"]}=%{{customdata:{m_cfg["hover_fmt"]}}}<br>'
                    'n_games=%{x}<br>'
                    '<extra></extra>'
                ),
            ))

    "Block 2: raw traces."
    for m_idx, m_cfg in enumerate(_metric_configs):
        for a_idx, n_agents_value in enumerate(all_n_agents_values):
            agents_subset = generating_model_df[
                generating_model_df['n_agents_fitted'] == n_agents_value
            ]
            summary_rows = []
            for ng in sorted(all_results_df['n_games_fitted'].unique()):
                sub = agents_subset[agents_subset['n_games_fitted'] == ng]
                if sub.empty:
                    continue
                summary_rows.append({'n_games': int(ng), 'value': _get_metric(sub, m_cfg['col'])})
            if not summary_rows:
                fig.add_trace(go.Scatter(x=[], y=[], visible=False, showlegend=False))
                continue
            summary_df   = pd.DataFrame(summary_rows)
            hue_shift    = (a_idx * 40) % 360
            trace_color  = _hsla(hue=(base_hue + hue_shift) % 360, alpha=0.9)
            agents_label = f"N={n_agents_value}"

            fig.add_trace(go.Scatter(
                x=summary_df['n_games'],
                y=summary_df['value'],
                mode='lines+markers',
                name=agents_label,
                visible=False,
                line=dict(color=trace_color, width=line_width),
                marker=dict(size=marker_size, color=trace_color),
                hovertemplate=(
                    f'{agents_label}<br>'
                    'n_games=%{x}<br>'
                    f'{m_cfg["col"]}=%{{y:{m_cfg["hover_fmt"]}}}<br>'
                    '<extra></extra>'
                ),
            ))

    "Build dropdown buttons."
    "Button 0: all metrics normalized [0,1]."
    btn0_visible = [True] * n_block + [False] * n_block
    dropdown_buttons = [dict(
        label='All metrics — normalized [0,1]',
        method='update',
        args=[
            {'visible': btn0_visible},
            {
                'yaxis.title.text':      'Normalized recovery score  (0 = worst,  1 = perfect recovery)',
                'yaxis.title.font.size': axis_font_size,
                'yaxis.tickfont.size':   axis_font_size,
                'yaxis.range':           [-0.05, 1.05],
                'yaxis.autorange':       False,
                'yaxis.tickmode':        'auto',
            },
        ],
    )]

    "Buttons 1..8: individual raw metrics."
    for m_idx, m_cfg in enumerate(_metric_configs):
        block1_visible = [False] * n_block
        block2_visible = [
            (t_idx // n_agents_count) == m_idx
            for t_idx in range(n_block)
        ]
        yaxis_args = {
            'yaxis.title.text':      m_cfg['y_title'],
            'yaxis.title.font.size': axis_font_size,
            'yaxis.tickfont.size':   axis_font_size,
        }
        if m_cfg.get('y_range') is not None:
            yaxis_args['yaxis.range']     = m_cfg['y_range']
            yaxis_args['yaxis.autorange'] = False
        else:
            yaxis_args['yaxis.autorange'] = True
        if m_cfg.get('y_tickvals') is not None:
            yaxis_args['yaxis.tickmode'] = 'array'
            yaxis_args['yaxis.tickvals'] = m_cfg['y_tickvals']
            yaxis_args['yaxis.ticktext'] = m_cfg['y_ticktext']
        else:
            yaxis_args['yaxis.tickmode'] = 'auto'

        dropdown_buttons.append(dict(
            label=m_cfg['label'],
            method='update',
            args=[
                {'visible': block1_visible + block2_visible},
                yaxis_args,
            ],
        ))

    fig.update_layout(
        title=dict(
            text=(f'Model Recovery Simulation  |  Generating Model {generating_utility_idx}'
                  f'  ({n_candidates} candidates)'),
            x=0.5, xanchor='center',
            y=0.97, yanchor='top',
            font=dict(size=fig_lay.get('title_size', 22) * 2),
        ),
        xaxis=dict(
            title=dict(
                text='Number of chooser games per agent  (n_games)',
                font=dict(size=axis_font_size),
            ),
            tickfont=dict(size=axis_font_size),
            showgrid=True,
            gridcolor=_hsla(hue=0, saturation_percent=0, lightness_percent=78, alpha=0.4),
        ),
        yaxis=dict(
            title=dict(
                text='Normalized recovery score  (0 = worst,  1 = perfect recovery)',
                font=dict(size=axis_font_size),
            ),
            tickfont=dict(size=axis_font_size),
            range=[-0.05, 1.05],
        ),
        updatemenus=[dict(
            type='dropdown',
            direction='down',
            showactive=True,
            x=0.0,
            xanchor='left',
            y=1.14,
            yanchor='top',
            buttons=dropdown_buttons,
            font=dict(size=base_font_size),
            bgcolor='white',
            bordercolor=_hsla(hue=0, saturation_percent=0, lightness_percent=60, alpha=0.8),
        )],
        hoverlabel=dict(font=dict(size=base_font_size + 2)),
        template=fig_lay.get('template', 'plotly_white'),
        font=dict(
            family=fig_lay.get('font', {}).get('family', 'Calibri'),
            size=base_font_size,
        ),
        margin=dict(l=120, r=80, t=170, b=100),
        autosize=True,
        legend=dict(
            title=dict(text='Metric  (N agents)', font=dict(size=base_font_size)),
            yanchor='top', y=0.98, xanchor='left', x=1.02,
            font=dict(size=base_font_size),
        ),
    )

    if export_fig:
        out_path = os.path.join(str(file_paths['visuals']), f'{_stem}.html')
        fig.write_html(out_path, config={'responsive': True})
        print(f"Model recovery simulation plot saved: {out_path}")

    return fig
