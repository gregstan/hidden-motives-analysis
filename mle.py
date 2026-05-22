from optimization import *
import preprocessing as prep

"""
DEPRECATED — Maximum Likelihood Estimation (MLE) fitting pipeline.

This module contains the non-Bayesian MLE fitting code that was used in an earlier
version of the analysis pipeline. It has been superseded by the Bayesian analysis
(run_analysis_bayes, loss_function_bayes, agent) and is preserved here for reference
only. None of these functions are called by the active analysis pipeline.

All active fitting uses the Bayesian UBM in bayesian.py. Do not add new analyses
here; do not extend this module.
"""

"=========================================================================================="
"======================================== MLE Code ========================================"
"=========================================================================================="


def compute_std_errors_mle(best_x: NDArray[np.float64], data_rows: List[Dict[str, Any]], param_info: ParamInfo,
                            utility_settings: UtilitySettings, penalty_weight: float) -> Dict[str, float]:
    """
    Estimate parameter standard errors from the curvature of the MLE loss surface at the optimum.

    Computes a finite-difference numerical Hessian of loss_function_mle evaluated at best_x,
    inverts it (falling back to the Moore-Penrose pseudo-inverse if singular), and takes the
    square root of the absolute diagonal entries as asymptotic standard error estimates.
    These are the standard errors used for reporting uncertainty around MLE parameter estimates.

    Arguments:
        • best_x: NDArray[np.float64]
            The optimized parameter vector at which to evaluate the Hessian.
        • data_rows: List[dict]
            The data subset used during fitting (same format as loss_function_mle expects).
        • param_info: ParamInfo
            Parameter configuration dict; param_info['keys'] provides the ordered parameter names.
        • utility_settings: UtilitySettings
            Utility functional form toggles (passed through to loss_function_mle).
        • penalty_weight: float
            Regularization weight used during fitting (must match what was used in optimization).

    Returns:
        • dict[str, float] — mapping from parameter name to estimated standard error.
    """
    def func_wrapper(x: NDArray[np.float64]) -> float:
        return loss_function_mle(x, data_rows, param_info, utility_settings, penalty_weight)

    "Numeric Hessian"
    hess = gnrl.numerical_hessian(func_wrapper, best_x)
    try:
        inv_hess = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        inv_hess = np.linalg.pinv(hess)

    std_err_dict = {}
    for idx, key in enumerate(param_info["keys"]):
        val = abs(inv_hess[idx, idx])
        std_err = math.sqrt(val)
        std_err_dict[key] = std_err
    return std_err_dict


def loss_function_mle(params_arr: NDArray[np.float64], data_rows: List[Dict[str, Any]], param_info: ParamInfo,
                       utility_settings: UtilitySettings, penalty_weight: float = 0.1, loss_funct_type: str = 'ssr') -> float:
    """
    Compute the MLE loss over a set of binary-choice data rows given a parameter vector.

    Evaluates the utility model for options A and B in each row, converts utilities to
    probabilities via softmax, and accumulates either squared residuals (SSR) or negative
    log-likelihood (log) loss. A parameter penalty is added per row to regularize the search
    and discourage degenerate solutions. The mean loss across rows is returned so that the
    value is comparable across subsets of different sizes.

    Arguments:
        • params_arr: NDArray[np.float64]
            Flat parameter vector of length len(param_info['keys']), ordered to match param_info.
        • data_rows: List[dict]
            Each dict must contain keys 'As', 'Ao', 'Bs', 'Bo' (payoffs) and 'selection'
            (0.0 for choice B, 1.0 for choice A; or 'A'/'B' for the log loss branch).
        • param_info: ParamInfo
            Parameter configuration dict; param_info['keys'] provides ordered parameter names.
        • utility_settings: UtilitySettings
            Boolean toggles for the active utility functional form.
        • penalty_weight: float
            Regularization strength. Larger values push parameters closer to their default values
            and improve numerical stability during global search.
        • loss_funct_type: str
            'ssr' for sum of squared residuals (pA − selection)²; 'log' for negative log-likelihood
            −log(pA) when selection='A', −log(1−pA) when selection='B'.

    Returns:
        • float — mean loss across all data rows (penalty included per row).
    """
    "Parse param_array => param_dict."
    param_dict = {key: val for (key, val) in zip(param_info["keys"], params_arr)}
    total_loss = 0.0

    for row in data_rows:
        payA = {
            "As": row["As"],
            "Ao": row["Ao"],
            "Bs": row["Bs"],
            "Bo": row["Bo"]
        }
        payB = {
            "As": row["Bs"],
            "Ao": row["Bo"],
            "Bs": row["As"],
            "Bo": row["Ao"]
        }
        uA = utility(payA, param_dict, utility_settings)
        uB = utility(payB, param_dict, utility_settings)
        pA = softmax_(uA, uB)

        selection = row["selection"]
        if loss_funct_type == "ssr":
            residual = (pA - selection) ** 2
        elif loss_funct_type == "log":
            residual = -math.log(pA if selection == 'A' else 1 - pA)

        total_loss += residual
        total_loss += gnrl.parameter_penalty(params=param_dict, penalty_weight=penalty_weight)

    mean_loss = total_loss / len(data_rows)
    return mean_loss


def extract_one_role_data_mle(dyad_games: DyadGames, player_uuid: PlayerUUID, player_role: PlayerRole) -> List[Dict[str, Any]]:
    """
    Collect per-round data needed for MLE from the dyad's meeting list for a single player+role.

    Each returned item is a dict with keys 'As', 'Ao', 'Bs', 'Bo', 'selection', 'meeting_idx', 'round'.

    Arguments:
        • dyad_games: DyadGames — list of meeting dictionaries for one dyad.
        • player_uuid: PlayerUUID — the player's UUID.
        • player_role: PlayerRole — 'chooser' or 'predictor'.

    Returns:
        • List[dict] — one row per relevant round for that player-role combination.
    """
    extracted_rows = []
    for meet_idx, meeting in enumerate(dyad_games):
        if meeting.get(player_role) != player_uuid:
            continue

        "Skip abdications."
        if player_role == 'chooser' and meeting.get('abdicated_chooser', False):
            continue
        if player_role == 'predictor' and meeting.get('abdicated_predictor', False):
            continue

        label_str = 'choice' if player_role == 'chooser' else 'prediction'
        label_val = meeting.get(label_str)
        if label_val is None:
            continue

        "Convert 'A' => 1.0, 'B' => 0.0."
        selection = 1.0 if label_val == 'A' else 0.0

        As = meeting.get('payoff_A_chooser', 0.0)
        Ao = meeting.get('payoff_A_predictor', 0.0)
        Bs = meeting.get('payoff_B_chooser', 0.0)
        Bo = meeting.get('payoff_B_predictor', 0.0)
        round_num = meeting.get('round', meet_idx)

        extracted_rows.append({
            "As": As, "Ao": Ao,
            "Bs": Bs, "Bo": Bo,
            "selection": selection,
            "meeting_idx": meet_idx,
            "round": round_num
        })

    extracted_rows.sort(key=lambda row_item: row_item["round"])
    return extracted_rows


def fit_one_player_one_role_mle(role_data: List[Dict[str, Any]], param_info: ParamInfo,
                                 utility_settings: UtilitySettings, track_evolution: bool) -> List[Dict[str, Any]]:
    """
    Fit parameters for a single player's single role across the entire role_data.

    If track_evolution=True, performs iterative fits (1..n games), storing partial results.
    Otherwise performs one final fit over all data.

    Arguments:
        • role_data: List[dict] — rows from extract_one_role_data_mle.
        • param_info: ParamInfo — parameter specification.
        • utility_settings: UtilitySettings — active utility form toggles.
        • track_evolution: bool — whether to fit iteratively or in one shot.

    Returns:
        • List[dict] — each item has 'meeting_idx', 'round', 'params', 'std_errors', 'loss'.
            Single item when track_evolution=False; one item per game when True.
    """
    if not role_data:
        return []

    results_list = []
    n_rounds = len(role_data)
    stage_indices = range(1, n_rounds + 1) if track_evolution else [n_rounds]

    for stage_count in stage_indices:
        subset = role_data[:stage_count]
        best_params, std_errs, final_loss = fit_subset_params_mle(
            subset_data=subset, param_info=param_info, utility_settings=utility_settings,
        )
        last_item = subset[-1]
        results_list.append({
            "meeting_idx": last_item["meeting_idx"],
            "round": last_item["round"],
            "params": best_params,
            "std_errors": std_errs,
            "loss": final_loss
        })

    return results_list


def fit_subset_params_mle(subset_data: List[Dict[str, Any]], param_info: ParamInfo, utility_settings: UtilitySettings,
                           penalty_weight: float = 0.1) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """
    Fit MLE parameters on a subset of binary-choice data using L-BFGS-B optimization.

    This is the innermost fitting call: given a set of data rows and the active utility
    functional form, it minimizes loss_function_mle from the param_info initial guess,
    then estimates standard errors from the Hessian curvature at the optimum.

    Arguments:
        • subset_data: List[dict]
            Data rows to fit; each must contain 'As', 'Ao', 'Bs', 'Bo', 'selection', etc.
        • param_info: ParamInfo
            Parameter configuration dict with 'keys', 'bounds', and 'guesses'.
        • utility_settings: UtilitySettings
            Boolean toggles selecting the active utility functional form.
        • penalty_weight: float
            Regularization strength passed to loss_function_mle.

    Returns:
        • tuple (best_params_dict, std_errs_dict, final_loss) where:
            - best_params_dict: dict[str, float] — optimized parameter values.
            - std_errs_dict: dict[str, float] — asymptotic standard errors per parameter.
            - final_loss: float — loss at the optimum.
    """
    if not subset_data:
        best_dict = {key: 0.0 for key in param_info["keys"]}
        err_dict = {key: float('inf') for key in param_info["keys"]}
        return best_dict, err_dict, float('inf')

    if callable(param_info["guesses"]):
        x0 = param_info["guesses"]()
    else:
        x0 = param_info["guesses"]
    bnds = param_info["bounds"]

    def objective_func(x: NDArray[np.float64]) -> float:
        return loss_function_mle(x, subset_data, param_info, utility_settings, penalty_weight)

    result = minimize(objective_func, x0, bounds=bnds, method='L-BFGS-B')
    best_x = result.x
    final_loss = result.fun

    std_errs = compute_std_errors_mle(
        best_x=best_x, data_rows=subset_data, param_info=param_info,
        utility_settings=utility_settings, penalty_weight=penalty_weight,
    )
    best_params_dict = {key: val for (key, val) in zip(param_info["keys"], best_x)}
    return best_params_dict, std_errs, final_loss


def store_params_in_dyad_mle(dyad_games: DyadGames, player_uuid: PlayerUUID, player_role: PlayerRole,
                              fit_results: List[Dict[str, Any]], utility_settings: UtilitySettings,
                              general_settings: GeneralSettings) -> List[Dict[str, Any]]:
    """
    Store MLE fit results in the dyad meeting dictionaries.

    Arguments:
        • dyad_games: DyadGames — the meeting list to update in place.
        • player_uuid: PlayerUUID — player whose results to store.
        • player_role: PlayerRole — 'chooser' or 'predictor'.
        • fit_results: List[dict] — output of fit_one_player_one_role_mle.
        • utility_settings: UtilitySettings — active utility form toggles.
        • general_settings: GeneralSettings — used for softmax_temperature.

    Returns:
        • DyadGames — the updated meeting list (modified in place).
    """
    for item in fit_results:
        meet_idx = item["meeting_idx"]
        if meet_idx < 0 or meet_idx >= len(dyad_games):
            continue

        meeting = dyad_games[meet_idx]
        param_est = meeting.setdefault("parameter_estimates", {})
        mle_dict = param_est.setdefault("mle", {})
        player_dict = mle_dict.setdefault(player_uuid, {})
        role_dict = player_dict.setdefault(player_role, {})

        role_dict["params"] = item["params"]
        role_dict["std_errors"] = item["std_errors"]

        model_select_key = "model_choose_A" if player_role == "chooser" else "model_predict_A"
        role_dict["output"] = {
            "loss": item["loss"],
            model_select_key: choice(
                current_game=meeting,
                agent_params=role_dict["params"],
                softmax_temperature=general_settings.get('softmax_temperature', 1.5),
                utility_settings=utility_settings,
                select=False,
            )["model_choose_A"]
        }

    return dyad_games


def fit_dyad_parameters_mle(dyad_games: List[Dict[str, Any]], param_info: ParamInfo, utility_settings: UtilitySettings,
                             file_paths: FilePaths, general_settings: GeneralSettings) -> Dict[str, Any]:
    """
    Fit MLE-based social preference parameters for both players in a single dyad.

    Arguments:
        • dyad_games: List[dict] — meeting/game dictionaries for this dyad.
        • param_info: ParamInfo — parameter fitting configuration.
        • utility_settings: UtilitySettings — active utility form toggles.
        • file_paths: FilePaths — used to construct per-dyad output file paths.
        • general_settings: GeneralSettings — controls track_evolution, experiment_num, etc.

    Returns:
        • List[dict] — the updated dyad_games with MLE results embedded under
            meeting["parameter_estimates"]["mle"][player_uuid][player_role].
    """
    if not dyad_games:
        return dyad_games

    first_game = dyad_games[0]
    first_choo = first_game.get('chooser')
    first_pred = first_game.get('predictor')
    if not isinstance(first_choo, str) or not isinstance(first_pred, str):
        raise ValueError("Failed to extract player UUIDs from dyad games.")
    player_uuids = sorted([first_choo, first_pred])

    dyad_file_path = prep._dyad_file_path(
        dyad_key=tuple(player_uuids), file_paths=file_paths,
        experiment_num=general_settings.get('experiment_num', 3), analysis_mode='mle',
    )
    try:
        if not general_settings.get('create_new_file', False):
            if os.path.exists(dyad_file_path):
                with open(dyad_file_path, "r", encoding='utf-8') as file_handle:
                    dyad_history = json.load(file_handle)
                if dyad_history:
                    return dyad_history
    except json.decoder.JSONDecodeError as json_error:
        print(json_error)

    for player_uuid in player_uuids:
        for role in ['chooser', 'predictor']:
            role_data = extract_one_role_data_mle(
                dyad_games=dyad_games, player_uuid=player_uuid, player_role=role,
            )
            if not role_data:
                continue
            fit_results = fit_one_player_one_role_mle(
                role_data=role_data, param_info=param_info,
                utility_settings=utility_settings,
                track_evolution=general_settings.get('track_evolution', True),
            )
            store_params_in_dyad_mle(
                dyad_games=dyad_games, player_uuid=player_uuid, player_role=role,
                fit_results=fit_results, utility_settings=utility_settings,
                general_settings=general_settings,
            )

    with open(dyad_file_path, 'w', encoding='utf-8') as file_handle:
        json.dump(dyad_games, file_handle, ensure_ascii=False, indent=4)

    return dyad_games


def run_analysis_mle(histories_data: Histories, file_paths: FilePaths, param_info: ParamInfo,
                     utility_settings: UtilitySettings, general_settings: GeneralSettings) -> Dict[str, Any]:
    """
    Run the non-cognitive MLE analysis over all dyads in histories_data.

    For each dyad, calls fit_dyad_parameters_mle to store MLE parameters in each meeting.
    Optionally tracks parameter evolution across rounds.

    Arguments:
        • histories_data: Histories — must contain 'histories': {dyad_key: [meeting, ...], ...}.
        • file_paths: FilePaths — output directory and file name configuration.
        • param_info: ParamInfo — parameter fitting configuration.
        • utility_settings: UtilitySettings — active utility form toggles.
        • general_settings: GeneralSettings — controls experiment_num, track_evolution,
            run_in_parallel, create_new_file.

    Returns:
        • Histories — the updated histories_data with MLE results in each dyad's meetings.
    """
    experiment_num  = general_settings.get('experiment_num',  3)
    track_evolution = general_settings.get('track_evolution', True)
    run_in_parallel = general_settings.get('run_in_parallel', True)
    create_new_file = general_settings.get('create_new_file', True)

    output_file    = file_paths["file_names"][f"params_data_exper{experiment_num}_{'iter' if track_evolution else 'fit1'}"]
    aggregate_path = os.path.join(file_paths["param_data"], output_file)
    dyad_output_dir = file_paths["dyad_data"]

    if not create_new_file and os.path.exists(aggregate_path):
        with open(aggregate_path, "r", encoding='utf-8') as file_handle:
            histories_data_fitted = json.load(file_handle)
        if histories_data_fitted:
            print(f"Aggregate MLE data loaded from {aggregate_path}.")
            return histories_data_fitted

    dyads_dict = histories_data.get('histories', None)
    if not dyads_dict:
        raise Exception("No 'histories' found in histories_data.")

    dyad_items = list(dyads_dict.items())
    n_dyads = len(dyad_items)
    os.makedirs(dyad_output_dir, exist_ok=True)

    args_list = [
        (dkey, meeting_list, file_paths, param_info, utility_settings, general_settings)
        for (dkey, meeting_list) in dyad_items
    ]

    from bayesian import _worker_fit_one
    if run_in_parallel:
        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            for fit_idx, dkey_returned in enumerate(pool.imap_unordered(_worker_fit_one, args_list), 1):
                print(f"MLE fitting: processed {fit_idx} / {n_dyads} dyads — {dkey_returned}.")
    else:
        for fit_idx, args in enumerate(args_list, 1):
            dkey_returned = _worker_fit_one(args)
            print(f"MLE fitting: processed {fit_idx} / {n_dyads} dyads — {dkey_returned}.")

    "Reload all individual dyad files and combine into histories_data."
    for reload_idx, dkey in enumerate(dyads_dict.keys(), 1):
        dyad_file_path = prep._dyad_file_path(
            dyad_key=dkey, file_paths=file_paths,
            experiment_num=experiment_num, analysis_mode='mle',
        )
        try:
            if os.path.exists(dyad_file_path):
                with open(dyad_file_path, "r", encoding='utf-8') as file_handle:
                    fitted_meeting = json.load(file_handle)
                histories_data['histories'][dkey] = fitted_meeting
                print(f"MLE fitting: retrieved {reload_idx} / {n_dyads} dyads — {dkey}.")
        except json.decoder.JSONDecodeError as json_error:
            print(json_error)

    with open(aggregate_path, "w", encoding='utf-8') as file_handle:
        json.dump(histories_data, file_handle, ensure_ascii=False, indent=4)
    print(f"MLE fitting complete. Aggregate data saved to {aggregate_path}.")

    return histories_data
