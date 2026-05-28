from bayesian import *
import uuid

"=========================================================================================="
"========== Simulation 1) The Optimizer Accurately Recovers Predictor Parameters =========="
"=========================================================================================="

def stable_bot_id(params: dict, player_role: str, n_games: int) -> str:
    """
    Content-addressed hash identifier for a synthetic simulation bot.

    Works for any utility model and never exceeds OS filename limits. The same
    (params, role, n_games) triple always produces the same ID, so predictor filenames
    can be reconstructed from parameters without tracking state.

    Arguments:
        • params: dict[str, float]; parameter dict that defines this bot's behavior.
        • player_role: str; 'predictor' or 'chooser'.
        • n_games: int; number of games the dyad will play.

    Returns:
        • str; e.g. 'synthetic_predictor_a3f9b1d2e5c7'.
    """
    payload = json.dumps({'params': sorted(params.items()), 'n': n_games}, sort_keys=True)
    return f'synthetic_{player_role}_{hashlib.sha256(payload.encode()).hexdigest()[:12]}'


def _simulate_pair_games(
    n_games: int,
    params_player_1: dict[str, float],
    params_player_2: dict[str, float],
    uuid_player_1: str,
    uuid_player_2: str,
    utility_settings: UtilitySettings,
    general_settings: GeneralSettings,
    param_info: ParamInfo,
    per_round_role_flip: bool = False,
    matching_probability: float = 1.0,
    dynamic_predictor: bool = True,
    payoff_structures: list[dict] | None = None,
    random_gen: np.random.Generator | None = None,
) -> list[dict]:
    """
    Core per-pair simulation loop shared by create_simulated_dyad and create_simulated_experiment.

    Generates game dicts with payoffs and role assignments, pre-generates all chooser responses,
    then runs the UBM for each player's predictor rounds. True params are embedded in round 0.

    Three-step ordering that avoids the predictor-observes-unwritten-choice problem:
        1. Build all game dicts with payoffs and role assignments (choice=None, prediction=None).
        2. Pre-generate all chooser responses via response(select_responses=True) so every
           game['choice'] is populated before any UBM pass reads it.
        3. Run agent() with player_role='predictor' for each player — skips rounds where that
           player is chooser, writes predictions, and updates beliefs from observed choices.

    Arguments:
        • n_games: int; number of games to simulate.
        • params_player_1: dict[str, float]; ground-truth utility parameters for player 1.
        • params_player_2: dict[str, float]; ground-truth utility parameters for player 2.
        • uuid_player_1: str; UUID string for player 1.
        • uuid_player_2: str; UUID string for player 2.
        • utility_settings: UtilitySettings; utility family for this simulation.
        • general_settings: GeneralSettings; passed through to agent() and serialize calls.
        • param_info: ParamInfo; parameter configuration for the UBM prior.
        • per_round_role_flip: bool;
            If False (default), player 1 is chooser and player 2 is predictor for all rounds.
            If True, chooser/predictor assignment is determined by a fair coin flip each round.
        • matching_probability: float; stored in each game dict (default 1.0).
        • dynamic_predictor: bool;
            If True (default), runs full UBM belief-updating for each player's predictor rounds.
            If False, generates predictions via response(select_responses=True) without UBM.
        • payoff_structures: list[dict] | None;
            Optional explicit per-round payoffs. If None, payoffs are drawn uniformly from {1..5}.
        • random_gen: np.random.Generator | None;
            If provided, used for payoff and role-flip sampling. Falls back to random.randint
            and random.random when None (preserves backward compatibility).

    Returns:
        • list[dict]; games_list with all choice, prediction, and parameter_estimate fields
          populated. Callers wrap this in a {dyad_key: games_list} dict.
    """
    τ_player_1 = params_player_1.get("τ")
    τ_player_2 = params_player_2.get("τ")

    "=== Step 1: Build game dicts with payoffs and role assignments ==="
    games_list: list[dict] = []
    for round_idx in range(n_games):
        if payoff_structures is not None and round_idx < len(payoff_structures):
            payoff_A_chooser   = payoff_structures[round_idx].get('payoff_A_chooser')
            payoff_A_predictor = payoff_structures[round_idx].get('payoff_A_predictor')
            payoff_B_chooser   = payoff_structures[round_idx].get('payoff_B_chooser')
            payoff_B_predictor = payoff_structures[round_idx].get('payoff_B_predictor')
            if any(payoff is None for payoff in (payoff_A_chooser, payoff_A_predictor, payoff_B_chooser, payoff_B_predictor)):
                raise ValueError(f"Payoff structure improperly formatted at round {round_idx}: {payoff_structures[round_idx]}.")
        elif random_gen is not None:
            payoff_A_chooser   = int(random_gen.integers(1, 6))
            payoff_A_predictor = int(random_gen.integers(1, 6))
            payoff_B_chooser   = int(random_gen.integers(1, 6))
            payoff_B_predictor = int(random_gen.integers(1, 6))
        else:
            payoff_A_chooser   = random.randint(1, 5)
            payoff_A_predictor = random.randint(1, 5)
            payoff_B_chooser   = random.randint(1, 5)
            payoff_B_predictor = random.randint(1, 5)

        if per_round_role_flip:
            coin_flip = random_gen.random() if random_gen is not None else random.random()
            if coin_flip < 0.5:
                chooser_uuid_this_round   = uuid_player_1
                chooser_params_this_round = params_player_1
                predictor_uuid_this_round = uuid_player_2
            else:
                chooser_uuid_this_round   = uuid_player_2
                chooser_params_this_round = params_player_2
                predictor_uuid_this_round = uuid_player_1
        else:
            chooser_uuid_this_round   = uuid_player_1
            chooser_params_this_round = params_player_1
            predictor_uuid_this_round = uuid_player_2

        game_dict = {
            "chooser":              chooser_uuid_this_round,
            "predictor":            predictor_uuid_this_round,
            "matching_probability": matching_probability,
            "payoff_A_chooser":     payoff_A_chooser,
            "payoff_A_predictor":   payoff_A_predictor,
            "payoff_B_chooser":     payoff_B_chooser,
            "payoff_B_predictor":   payoff_B_predictor,
            "choice":               None,
            "prediction":           None,
            "abdicated_chooser":    False,
            "abdicated_predictor":  False,
            "timestamp":            time.time(),
            "round":                round_idx,
        }
        if round_idx == 0:
            "Embed ground-truth params in round 0 for both players' active roles in that round."
            game_dict["true_params_chooser"]   = dict(chooser_params_this_round)
            game_dict["true_params_predictor"] = dict(params_player_2 if chooser_uuid_this_round == uuid_player_1 else params_player_1)
        games_list.append(game_dict)

    "=== Step 2: Pre-generate all chooser responses ==="
    for game_dict in games_list:
        chooser_uuid_this_game   = game_dict["chooser"]
        chooser_params_this_game = params_player_1 if chooser_uuid_this_game == uuid_player_1 else params_player_2
        τ_chooser_this_game      = chooser_params_this_game.get("τ")
        choice_bit = response(
            current_game=game_dict,
            agent_params=chooser_params_this_game,
            utility_settings=utility_settings,
            softmax_temperature=τ_chooser_this_game,
            select_responses=True,
        )["model_choose_A"]
        game_dict["choice"] = "A" if choice_bit == 1 else "B"

    "=== Step 3: Run UBM or static predictions for each player's predictor rounds ==="
    if dynamic_predictor:
        update_method           = general_settings.get('update_method', 'grid')
        initial_params_player_1 = {param_key: param_val for param_key, param_val in params_player_1.items() if param_key not in ('τ', 'temp')}
        initial_params_player_2 = {param_key: param_val for param_key, param_val in params_player_2.items() if param_key not in ('τ', 'temp')}
        games_list = agent(
            dyad_games=games_list, game_idx_start=0, game_idx_stop=n_games - 1,
            general_settings=general_settings, utility_settings=utility_settings,
            param_info=param_info, initial_params={'predictor': initial_params_player_1},
            player_uuid=uuid_player_1, player_role='predictor',
            select_responses=True, softmax_temperature=τ_player_1,
        )
        games_list = agent(
            dyad_games=games_list, game_idx_start=0, game_idx_stop=n_games - 1,
            general_settings=general_settings, utility_settings=utility_settings,
            param_info=param_info, initial_params={'predictor': initial_params_player_2},
            player_uuid=uuid_player_2, player_role='predictor',
            select_responses=True, softmax_temperature=τ_player_2,
        )
        "Drop grid arrays — retain only params (MAP estimates) and predictions for downstream recovery analysis."
        games_list = prep.serialize_or_drop_param_vectors(dyad_games=games_list, general_settings=general_settings, drop_grids=True)
        for game_dict in games_list:
            param_est = game_dict.get('parameter_estimates')
            if param_est and update_method in param_est:
                param_est['sim_pred'] = param_est.pop(update_method)
    else:
        "Static predictions — no UBM, generate prediction responses via response() directly."
        for game_dict in games_list:
            predictor_uuid_this_game   = game_dict["predictor"]
            predictor_params_this_game = params_player_1 if predictor_uuid_this_game == uuid_player_1 else params_player_2
            τ_predictor_this_game      = predictor_params_this_game.get("τ")
            pred_bit = response(
                current_game=game_dict,
                agent_params=predictor_params_this_game,
                utility_settings=utility_settings,
                softmax_temperature=τ_predictor_this_game,
                select_responses=True,
            )["model_choose_A"]
            game_dict["prediction"] = "A" if pred_bit == 1 else "B"

    return games_list


def create_simulated_dyad(
    n_games: int,
    params_chooser: dict[str, float],
    params_predictor: dict[str, float],
    general_settings: GeneralSettings,
    utility_settings: UtilitySettings,
    param_bds: ParamBounds,
    payoff_structures: list[dict[str, int]] | None = None,
    default_utility_settings: bool = True,
    dynamic_predictor: bool = True,
    player_1_uuid: str | None = None,
    player_2_uuid: str | None = None,
    matching_probability: float = 1.0,
    per_round_role_flip: bool = False,
    random_gen: np.random.Generator | None = None,
) -> dict[DyadKey, DyadGames]:
    """
    Create a single synthetic player-pair dyad with recorded choices and predictions.

    Delegates to _simulate_pair_games for the actual simulation logic. When
    per_round_role_flip=False (default), params_chooser / params_predictor refer to fixed
    player-1 / player-2 roles respectively. When per_round_role_flip=True, roles are
    reassigned by a fair coin flip each round, but params_chooser still belongs to player 1
    and params_predictor to player 2 regardless of which role they hold in any given round.

    Arguments:
        • n_games: int; number of games to simulate.
        • params_chooser: dict[str, float]; ground-truth utility parameters for player 1.
        • params_predictor: dict[str, float]; ground-truth utility parameters for player 2.
        • general_settings: GeneralSettings; passed through to _simulate_pair_games.
        • utility_settings: UtilitySettings; used when default_utility_settings=False.
        • param_bds: ParamBounds; used to build param_info for the UBM prior.
        • payoff_structures: list[dict] | None; optional explicit per-round payoffs.
        • default_utility_settings: bool;
            If True (default), use the built-in baseline utility settings (altruism only).
            If False, use the caller-provided utility_settings.
        • dynamic_predictor: bool;
            If True (default), runs full UBM belief-updating for predictor rounds.
            If False, generates static predictions via response() without UBM.
        • player_1_uuid: str | None;
            UUID for player 1 (always chooser when per_round_role_flip=False). If None,
            auto-generates f'synthetic_player_1_{uuid.uuid4().hex[:12]}'.
        • player_2_uuid: str | None;
            UUID for player 2 (always predictor when per_round_role_flip=False). If None,
            auto-generates f'synthetic_player_2_{uuid.uuid4().hex[:12]}'. Callers that need
            content-addressed IDs (e.g. create_simulated_data) should pass
            player_2_uuid=stable_bot_id(...) explicitly.
        • matching_probability: float; stored in each game dict (default 1.0).
        • per_round_role_flip: bool; if True, chooser/predictor roles are reassigned each round.
        • random_gen: np.random.Generator | None;
            If provided, used for payoff sampling and role-flip coin flips.
            Falls back to random.randint / random.random when None.

    Returns:
        • dict[DyadKey, DyadGames]; single-key dict '(predictor_uuid, chooser_uuid)' whose value
          is the games list. Each game contains 'chooser', 'predictor', payoffs, 'choice',
          'prediction', 'round', and (in round 0) 'true_params_chooser'/'true_params_predictor'.
    """
    if not isinstance(n_games, int):
        raise TypeError(f"n_games must be an integer not {type(n_games)} - {n_games}.")
    if not n_games > 0:
        raise ValueError(f"n_games must be greater than 0, not {n_games}.")

    utility_settings_: UtilitySettings = {
        'conditional_welfare_mode':       False,
        'reference_dependent_altruism':   False,
        'min_max_rawlsian_leontief':      False,
        'use_exponential_parameters':     False,
        'apply_exponents_to_payoffs':     False,
        'single_exponential_parameter':   False,
        'single_payoffs_not_differences': False,
        'payoff_ratios_not_differences':  False,
        'reference_dependent_utility':    False,
        'use_negativity_parameters':      False,
        'negativity_social_comparison':   False,
        'fix_self_interest_parameter':    False,
        'include_social_comparison':      False,
        'include_altruism_term':          True,
    }
    if not default_utility_settings:
        utility_settings_ = copy.deepcopy(utility_settings)

    player_1_uuid_resolved = player_1_uuid if player_1_uuid is not None else f'synthetic_player_1_{uuid.uuid4().hex[:12]}'
    player_2_uuid_resolved = player_2_uuid if player_2_uuid is not None else f'synthetic_player_2_{uuid.uuid4().hex[:12]}'

    param_info_ = make_param_info(param_bds=param_bds, utility_settings=utility_settings_,
                                  general_settings=general_settings, guess_seed=None, random_guesses_are_unique=True)

    games_list = _simulate_pair_games(
        n_games=n_games,
        params_player_1=params_chooser,
        params_player_2=params_predictor,
        uuid_player_1=player_1_uuid_resolved,
        uuid_player_2=player_2_uuid_resolved,
        utility_settings=utility_settings_,
        general_settings=general_settings,
        param_info=param_info_,
        per_round_role_flip=per_round_role_flip,
        matching_probability=matching_probability,
        dynamic_predictor=dynamic_predictor,
        payoff_structures=payoff_structures,
        random_gen=random_gen,
    )
    dyad_key = f"({games_list[0]['predictor']}, {games_list[0]['chooser']})"
    return {dyad_key: games_list}


def create_simulated_data(n_games: int, params_chooser_range: dict[str, float], params_predictor_range: dict[str, float], utility_settings: UtilitySettings,
                          param_bds: Dict[str, tuple[int | float, int | float]] | None = None, file_paths: FilePaths | None = None,
                          payoff_structures: list[dict[str, int]] | None = None, run_analysis: bool = True, dynamic_predictor: bool = True,
                          randomize_parameters: bool = True, max_iter: int = 1000) -> dict[DyadKey, DyadGames]:
    """
    Generate a grid (or randomized grid) of artificial chooser–predictor dyads and optionally
    run the full Bayesian analysis pipeline on the resulting synthetic dataset.

    Conceptually, this function:
        1. Expands parameter ranges for choosers and predictors into discrete values
           (either evenly spaced or random within each interval).
        2. Creates one dyad for every combination of chooser and predictor parameters.
        3. Simulates `n_games` binary dictator games for each dyad via `create_simulated_dyad`.
        4. Aggregates all dyads into a `player_histories` structure compatible with the
           standard analysis functions.
        5. Optionally writes the histories to disk and calls `run_analysis_bayes(...)` to
           verify that the optimizer recovers the known parameters.

    Arguments:
        • n_games: int;
            Number of games simulated per dyad (per chooser–predictor pair).
        • params_chooser_range: dict[str, float];
            Parameter grid specification for choosers. Each entry must be a length-3 tuple:
                (lower_bound, upper_bound, n_points)
            for keys:
                - 'Vᵢᵢ': Range for chooser self-interest means.
                - 'Vᵢⱼ': Range for chooser altruism means.
                - 'std':  Range for chooser standard deviations (applied to both Vᵢᵢ and Vᵢⱼ).
                - 'τ':    Range for chooser SoftMax temperatures.
        • params_predictor_range: dict[str, float];
            Parameter grid specification for predictors, same structure as `params_chooser_range`.
            Each combination of predictor parameters is paired with each combination of
            chooser parameters to form a dyad.
        • utility_settings: UtilitySettings;
            Base utility settings to be used for the simulated choice and prediction calls.
            A copy is internally created with all booleans turned off except
            `'include_altruism_term': True`, unless `run_analysis` modifies this downstream.
        • param_bds: dict[str, tuple[int | float, int | float]] | None;
            Bounds for the optimizer parameters, passed through to `make_param_info(...)`
            when `run_analysis=True`. Required if `run_analysis` is True.
        • file_paths: FilePaths | None;
            File path configuration dictionary used for saving simulated histories and directing
            the analysis results. Required if `run_analysis` is True.
        • payoff_structures: list[dict[str, int]] | None;
            Optional externally specified payoff sequence for all dyads. If provided, the same
            payoff list is reused for each dyad and `create_simulated_dyad` is called with
            `payoff_structures=payoff_structures`. If None, payoffs are sampled uniformly
            from {1,…,5} per game.
        • run_analysis: bool;
            If True, writes the simulated histories to disk and calls `run_analysis_bayes(...)`
            using the provided `param_bds`, `file_paths`, and `utility_settings`. If False,
            only returns the in-memory `player_histories` dict.
        • dynamic_predictor: bool;
            If True, runs the full UBM via agent() for predictors, meaning belief updating.
            If False, runs response() for predictors, meaning no belief updating.
        • randomize_parameters: bool;
            If True (default), parameter grids are populated by sampling uniform random values
            within each (min, max) range. If False, use evenly spaced linspace grids across
            each interval.
        • max_iter: int;
            Safety cap on the total number of dyads (i.e., the product of all grid resolutions).
            If the implied number of dyads exceeds `max_iter`, an Exception is raised to avoid
            accidentally launching enormous simulations.

    Returns:
        • dict[DyadKey, DyadGames];
            If `run_analysis` is False:
                - Returns a dictionary with keys:
                    'histories':   dict[DyadKey, DyadGames];
                        Mapping from dyad keys "(predictor_uuid, chooser_uuid)" to their list
                        of game dictionaries.
                    'player_info': dict[str, dict];
                        Mapping from player_uuid → avatar metadata for plotting/inspection.
            If `run_analysis` is True:
                - Returns the same structure after writing it to disk and running the Bayesian
                  analysis pipeline. The analysis results are saved to the locations specified
                  by `file_paths`.

    Raises:
        • Exception:
            - If `run_analysis=True` and either `param_bds` or `file_paths` is None.
            - If the implied number of dyads `n_iters` exceeds `max_iter`.
        • Any downstream exceptions raised by `create_simulated_dyad`, file I/O, or
          `run_analysis_bayes(...)`.

    Notes:
        • Parameter ranges are expanded in the following order:
            Vᵢᵢ_predictor × Vᵢⱼ_predictor × std_predictor × τ_predictor ×
            Vᵢᵢ_chooser × Vᵢⱼ_chooser × std_chooser × τ_chooser
          so the outer loops vary predictor parameters first, then chooser parameters.
        • Each unique player UUID is assigned a random avatar shape and color sample (HLSA)
          to make simulated players visually distinct in the UI and diagnostic plots.
    """
    if run_analysis:
        if param_bds is None:
            raise Exception("param_bds cannot be None if run_analysis.")
        if file_paths is None:
            raise Exception("file_paths cannot be None if run_analysis.")

    if dynamic_predictor and param_bds is None:
        raise ValueError(f"param_bds cannot be None if dynamic_predictor is True.")

    general_settings_ = {
        'experiment_num': 0,
        'run_in_parallel': True,
        'track_evolution': False,
        'create_new_file': True,
        'update_method': 'grid',
        'analysis_mode': 'bayesian',
        'analysis_unit': 'player',
        'n_bins_per_dimension': 9,
        'include_covariance': False,
        'softmax_temperature': 1.0,
        'temperature_is_param': True,
        'guess_params_randomly': False,
        'optimization_method': 'globloc',
        'confidence_weighted': True,
        'use_particle_filter': True,
        'fit_roles_together': False,
        'use_initial_params': True,
        'loss_funct_type': 'log',
        'penalty_weight': 0.05,
        'learning_rate': 0.8,
        'sample_ratio': 1.0,
        'export_fig': True,
        'dark_mode': True
    }

    utility_settings_: UtilitySettings = {setting_key: False for setting_key in utility_settings.keys()}
    utility_settings_['include_altruism_term'] = True

    Vᵢᵢ_chooser_range =   params_chooser_range.get("Vᵢᵢ")
    Vᵢⱼ_chooser_range =   params_chooser_range.get("Vᵢⱼ")
    std_chooser_range =   params_chooser_range.get("std")
    τ_chooser_range =     params_chooser_range.get("τ")

    Vᵢᵢ_predictor_range = params_predictor_range.get("Vᵢᵢ")
    Vᵢⱼ_predictor_range = params_predictor_range.get("Vᵢⱼ")
    std_predictor_range = params_predictor_range.get("std")
    τ_predictor_range =   params_predictor_range.get("τ")

    n_iters = 1
    for range_ in (
            Vᵢᵢ_chooser_range,   Vᵢⱼ_chooser_range,   std_chooser_range,   τ_chooser_range, 
            Vᵢᵢ_predictor_range, Vᵢⱼ_predictor_range, std_predictor_range, τ_predictor_range
        ):
        n_iters *= range_[2]
    
    if n_iters > max_iter:
        raise Exception(f"Runtime Warning! Was about to generate {n_iters} dyads!")
    print(f"Generating simulation with {n_iters} artificial agents.")

    if randomize_parameters:
        Vᵢᵢ_chooser_intervals =   [round(random.uniform(Vᵢᵢ_chooser_range[0], Vᵢᵢ_chooser_range[1]), 4) for num in range(Vᵢᵢ_chooser_range[2])]
        Vᵢⱼ_chooser_intervals =   [round(random.uniform(Vᵢⱼ_chooser_range[0], Vᵢⱼ_chooser_range[1]), 4) for num in range(Vᵢⱼ_chooser_range[2])]
        τ_chooser_intervals =     [round(random.uniform(τ_chooser_range[0],   τ_chooser_range[1]), 4)   for num in range(τ_chooser_range[2])]
        std_chooser_intervals =   [round(random.uniform(std_chooser_range[0], std_chooser_range[1]), 4) for num in range(std_chooser_range[2])]
 
        Vᵢᵢ_predictor_intervals = [round(random.uniform(Vᵢᵢ_predictor_range[0], Vᵢᵢ_predictor_range[1]), 4) for num in range(Vᵢᵢ_predictor_range[2])]
        Vᵢⱼ_predictor_intervals = [round(random.uniform(Vᵢⱼ_predictor_range[0], Vᵢⱼ_predictor_range[1]), 4) for num in range(Vᵢⱼ_predictor_range[2])]
        τ_predictor_intervals =   [round(random.uniform(τ_predictor_range[0],   τ_predictor_range[1]), 4)   for num in range(τ_predictor_range[2])]
        std_predictor_intervals = [round(random.uniform(std_predictor_range[0], std_predictor_range[1]), 4) for num in range(std_predictor_range[2])]
        
    else:
        Vᵢᵢ_chooser_intervals =   list(np.round(np.linspace(start=Vᵢᵢ_chooser_range[0], stop=Vᵢᵢ_chooser_range[1], num=Vᵢᵢ_chooser_range[2]), decimals=4))
        Vᵢⱼ_chooser_intervals =   list(np.round(np.linspace(start=Vᵢⱼ_chooser_range[0], stop=Vᵢⱼ_chooser_range[1], num=Vᵢⱼ_chooser_range[2]), decimals=4))
        τ_chooser_intervals =     list(np.round(np.linspace(start=τ_chooser_range[0],   stop=τ_chooser_range[1],   num=τ_chooser_range[2]),   decimals=4))
        std_chooser_intervals =   list(np.round(np.linspace(start=std_chooser_range[0], stop=std_chooser_range[1], num=std_chooser_range[2]), decimals=4))
        
        Vᵢᵢ_predictor_intervals = list(np.round(np.linspace(start=Vᵢᵢ_predictor_range[0], stop=Vᵢᵢ_predictor_range[1], num=Vᵢᵢ_predictor_range[2]), decimals=4))
        Vᵢⱼ_predictor_intervals = list(np.round(np.linspace(start=Vᵢⱼ_predictor_range[0], stop=Vᵢⱼ_predictor_range[1], num=Vᵢⱼ_predictor_range[2]), decimals=4))
        τ_predictor_intervals =   list(np.round(np.linspace(start=τ_predictor_range[0],   stop=τ_predictor_range[1],   num=τ_predictor_range[2]),   decimals=4))
        std_predictor_intervals = list(np.round(np.linspace(start=std_predictor_range[0], stop=std_predictor_range[1], num=std_predictor_range[2]), decimals=4))

    player_histories = {}
    "Iterate over predictor parameters"
    for Vᵢᵢ_predictor in Vᵢᵢ_predictor_intervals:
        for Vᵢⱼ_predictor in Vᵢⱼ_predictor_intervals:
            for std_predictor in std_predictor_intervals:
                for τ_predictor in τ_predictor_intervals:    
                    "Iterate over chooser parameters"
                    for Vᵢᵢ_chooser in Vᵢᵢ_chooser_intervals:
                        for Vᵢⱼ_chooser in Vᵢⱼ_chooser_intervals:
                            for std_chooser in std_chooser_intervals:
                                for τ_chooser in τ_chooser_intervals:

                                    "Create 'params' variables for both player roles."
                                    params_chooser =   {'Vᵢᵢ': Vᵢᵢ_chooser,   'Vᵢⱼ': Vᵢⱼ_chooser,   'Vᵢᵢ_std': std_chooser,   'Vᵢⱼ_std': std_chooser,   'τ': τ_chooser}
                                    params_predictor = {'Vᵢᵢ': Vᵢᵢ_predictor, 'Vᵢⱼ': Vᵢⱼ_predictor, 'Vᵢᵢ_std': std_predictor, 'Vᵢⱼ_std': std_predictor, 'τ': τ_predictor}

                                    "Generate the series of games played between these artificial agents."
                                    player_dyad = create_simulated_dyad(n_games=n_games, params_chooser=params_chooser, params_predictor=params_predictor,
                                                                        utility_settings=utility_settings_, general_settings=general_settings_,
                                                                        payoff_structures=payoff_structures, param_bds=param_bds, dynamic_predictor=dynamic_predictor,
                                                                        player_2_uuid=stable_bot_id(params=params_predictor, player_role='predictor', n_games=n_games))
                                    
                                    "Update player_histories with {DyadKey: DyadGames} dictionary."
                                    player_histories.update(player_dyad)

    avatar_shapes = [
        "arrow-head",
        "bowtie",
        "circle",
        "cross",
        "curvy-x",
        "dent-square",
        "dodecagon",
        "flame",
        "flower",
        "ghost",
        "hexagon",
        "hour-glass",
        "jagged-sun",
        "lemon",
        "moon",
        "pentagon",
        "round-square",
        "squash",
        "teardrop",
        "two-triangle",
        "star-six",
    ]

    player_info = {}
    for dyad_key in player_histories.keys():
        plr_uuid_1, plr_uuid_2 = dyad_key[1:-1].split(", ")
        for plr_uuid in (plr_uuid_1, plr_uuid_2):
            player_info[plr_uuid] = {
                'player_type': 'robot', 'avatar_shape': avatar_shapes[random.randint(0, len(avatar_shapes) - 1)],
                'player_color': f'hlsa({random.randint(0, 359)}, {random.randint(35, 65)}%, {random.randint(35, 65)}%, 1.0)'
            }

    player_histories = {'histories': player_histories, 'player_info': player_info}

    if run_analysis:
        param_info_ = make_param_info(param_bds=param_bds, utility_settings=utility_settings_, general_settings=general_settings_, 
                                                 random_guesses_are_unique=not general_settings_['run_in_parallel'])

        "Create file name suffix from these settings."
        file_name_suffix = prep.create_file_name_suffix(
            general_settings=general_settings_, utility_settings=utility_settings_
        )

        "Copy of standard file paths to alter with each loop."
        file_paths_ = copy.deepcopy(file_paths)

        "Remove suffix from file names if any."
        file_paths_ = prep.add_remove_file_name_suffix(
            file_paths=file_paths_, file_name_suffix=None, add_suffix=False
        )

        "Re-add that suffix to file_paths."
        file_paths_ = prep.add_remove_file_name_suffix(
            file_paths=file_paths_, file_name_suffix=file_name_suffix, add_suffix=True
        )

        histories_file_path = os.path.join(file_paths['processed'], file_paths['file_names'][f'player_pairs_exper0'])

        with open(histories_file_path, 'w', encoding='utf-8') as file:
            json.dump(player_histories, file, ensure_ascii=False, indent=4)
            print(f"Saved simulated histories to {pretty_path(histories_file_path)}.")
 
        histories_info = player_histories['player_info']

        run_analysis_bayes(histories_data=player_histories, file_paths=file_paths_, param_info=param_info_,
                           utility_settings=utility_settings_, general_settings=general_settings_)
    return player_histories


def parse_robot_string(robot_str: str) -> dict:
    """
    Extract "true" parameters from a player UUID string like:
        'robot_predictor_Vii=(1.0,1.0)_Vij=(-1.0,1.0)_t=1.5_n=9'
    Returns a dict of parameter values, e.g.:
        {
            "Vᵢᵢ": 1.0,
            "Vᵢᵢ_std": 1.0,
            "Vᵢⱼ": -1.0,
            "Vᵢⱼ_std": 1.0,
            "τ": 1.5,
            "n_games": 9
        }
    """
    pattern_vii = r"Vii=\((-?\d+\.?\d*),(-?\d+\.?\d*)\)"
    pattern_vij = r"Vij=\((-?\d+\.?\d*),(-?\d+\.?\d*)\)"
    pattern_t   = r"_t=(-?\d+\.?\d*)"
    pattern_n   = r"_n=(\d+)"

    match_vii = re.search(pattern_vii, robot_str)
    match_vij = re.search(pattern_vij, robot_str)
    match_t   = re.search(pattern_t,   robot_str)
    match_n   = re.search(pattern_n,   robot_str)

    parsed = {}
    if match_vii:
        parsed["Vᵢᵢ"]     = float(match_vii.group(1))
        parsed["Vᵢᵢ_std"] = float(match_vii.group(2))
    if match_vij:
        parsed["Vᵢⱼ"]     = float(match_vij.group(1))
        parsed["Vᵢⱼ_std"] = float(match_vij.group(2))
    if match_t:
        parsed["τ"]       = float(match_t.group(1))
    if match_n:
        parsed["n_games"] = int(match_n.group(1))

    return parsed


def get_simulated_dyad(file_paths: FilePaths, dyad_idx: int | None, n_games: int, params_predictor: Optional[Dict[str, float]] = None, 
                       params_chooser: Optional[Dict[str, float]] = None) -> Dict[DyadKey, DyadGames]:
    """
    Load a single simulated dyad (chooser–predictor game history) from disk.

    This is a convenience loader used when inspecting the optimizer's recovery performance
    on a particular artificial dyad. It supports two ways of selecting which dyad to load:

        1. Direct filename reconstruction:
            If `params_predictor` is provided, the function calls `stable_bot_id(...)` to
            reconstruct the predictor UUID and loads the corresponding `<predictor_uuid>.json`
            file from `json_path`. Only `params_predictor` is needed; `params_chooser` is
            not used for filename reconstruction.

        2. Index-based selection:
            If `params_predictor` is not provided, the function defers to
            `prep.get_file_by_index_or_name(...)` using `dyad_idx` to choose
            a file from `json_path` (e.g., “the k-th JSON file in that directory”).

    Arguments:
        • file_paths: dict[str: str | dict[str: str]];
            Dictionary of all files paths in this project.
        • dyad_idx: int | None;
            Index of the dyad JSON file to load when `params_predictor` is not supplied.
            Passed through to `prep.get_file_by_index_or_name` as the `file_name_idx`
            argument. Ignored if `file_name` is determined by `stable_bot_id(...)`.
        • n_games: int;
            Number of games that were simulated for this dyad. Used only when reconstructing
            the UUID via `stable_bot_id(...)` (so that the filename matches exactly).
        • params_predictor: dict[str, float] | None;
            Parameter dictionary for the predictor used during simulation. When provided,
            this is passed to `stable_bot_id(...)` to recreate the predictor UUID and thus
            the JSON filename. If None, dyad selection falls back to `dyad_idx`.
        • params_chooser: dict[str, float] | None;
            Not used for filename reconstruction (only `params_predictor` determines the
            filename). Retained as a parameter for callers that pass it; has no effect on
            which file is loaded. If None, dyad selection falls back to `dyad_idx`.

    Returns:
        • dict[DyadKey, DyadGames];
            A dictionary mapping a single DyadKey "(predictor_uuid, chooser_uuid)" to the
            list of dyad games recovered from disk. This is the same format returned by
            `create_simulated_dyad` and expected by downstream analysis code.

    Raises:
        • Exception:
            - If a filename cannot be determined from the provided arguments.
            - If the resolved file does not exist at `json_path`.
            - If `prep.get_file_by_index_or_name` fails to return a valid filename.

    Notes:
        • When `params_predictor` is supplied, it must match the dict used during simulation;
          `stable_bot_id` produces the same hash → filename deterministically. `params_chooser`
          is no longer used for filename reconstruction (chooser UUIDs are random uuid4).
        • This loader is read-only: it does not modify or re-simulate any dyads, it simply
          deserializes a previously saved JSON file.
    """
    file_name = None
    if params_predictor is not None:
        file_name = stable_bot_id(params=params_predictor, player_role='predictor', n_games=n_games) + ".json"

    if file_name is None:
        file_name = prep.get_file_by_index_or_name(directory_path=os.path.join(file_paths['player_fits'], 'experiment_0'), file_name_idx=dyad_idx, file_name=file_name)

    if file_name is None:
        raise Exception("Failed to retrieve file name.")
    
    full_path = os.path.join(file_paths['player_fits'], 'experiment_0', file_name)
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as file:
            dyad_dict = json.load(file)    
            return dyad_dict

    raise Exception("Failed to retrieve file name.")


def load_simulated_fits_from_json(json_path: str) -> pd.DataFrame:
    """
    Load one simulated dyad JSON file and reshape it into a long, round-level DataFrame.

    Each JSON file contains one or more dyads, where each dyad key maps to a list of
    game dictionaries with:
        - player UUIDs (chooser, predictor),
        - true parameters (either embedded or parsed from the UUID),
        - fitted parameters nested under "parameter_estimates".

    This function produces one row per (dyad_key, round), with columns for:
        - identifiers: dyad_key, player_uuid_chooser, player_uuid_predictor, round
        - true parameters for each role: <param>_true_chooser, <param>_true_predictor
        - fitted predictor parameters: <param>_fitted_predictor
        - optionally fitted chooser parameters: <param>_fitted_chooser

    Arguments:
        • json_path: str;
            Full path to a single JSON file containing simulated dyad histories and fits.

    Returns:
        • pd.DataFrame;
            Long-format table with one row per (dyad_key, round) and dynamic columns
            for all discovered true and fitted parameters.
    """
    with open(json_path, "r", encoding="utf-8") as file:
        dyad_dict = json.load(file)

    rows = []
    for dyad_key, games_list in dyad_dict.items():
        for game in games_list:

            "Basic info"
            round_idx = game.get("round", None)
            chooser_str   = game["chooser"]
            predictor_str = game["predictor"]

            "Parse their \"true\" parameter values"
            tp_pred = game.get("true_params_predictor", None)
            tp_ch   = game.get("true_params_chooser",   None)
            if tp_pred is None or tp_ch is None:
                chooser_true   = parse_robot_string(chooser_str)
                predictor_true = parse_robot_string(predictor_str)
            else:
                chooser_true   = dict(tp_ch)
                predictor_true = dict(tp_pred)

            "2) FITTED predictor params "
            param_est: dict = game.get("parameter_estimates", {})
            "Pick any available method block (grid/naive/particle) to tolerate general_settings changes."
            for _method_key in ("grid", "particle", "naive", "update", "globloc", "bayes", "general"):
                if _method_key in param_est:
                    fitted_block = param_est[_method_key]
                    break
            else:
                fitted_block = {}

            predictor_fitted_params = {}
            if predictor_str in fitted_block:
                pred_role_block = fitted_block[predictor_str].get("predictor", {})
                predictor_fitted_params = pred_role_block.get("params", {})

            chooser_fitted_params = {}
            if chooser_str in fitted_block:
                ch_role_block = fitted_block[chooser_str].get("chooser", {})
                chooser_fitted_params = ch_role_block.get("params", {})

            "3) One row; will add dynamic columns"
            row = {
                "dyad_key": dyad_key,
                "player_uuid_predictor": predictor_str,
                "player_uuid_chooser": chooser_str,
                "round": round_idx,
            }

            "Add TRUE columns (predictor & chooser)"
            for param_key, param_value in predictor_true.items():
                row[f"{param_key}_true_predictor"] = param_value
            for param_key, param_value in chooser_true.items():
                row[f"{param_key}_true_chooser"] = param_value

            "Add FITTED predictor columns"
            for param_key, param_value in predictor_fitted_params.items():
                row[f"{param_key}_fitted_predictor"] = param_value

            "Optional: add fitted chooser (not used here)"
            if chooser_fitted_params:
                for param_key, param_value in chooser_fitted_params.items():
                    row[f"{param_key}_fitted_chooser"] = param_value

            if "temp" in chooser_true and "τ" not in chooser_true:       
                chooser_true["τ"] = chooser_true.pop("temp")
            if "temp" in predictor_true and "τ" not in predictor_true:   
                predictor_true["τ"] = predictor_true.pop("temp")

            "Extracting the posteriors originating from the simulated predictor's assigned priors, not from fitted priors."
            sim_pred = param_est.get('sim_pred')
            if sim_pred is not None:
                sim_pred_predictor_params = sim_pred.get(predictor_str, {}).get('predictor', {}).get('params', {})
                for param_key, param_val in sim_pred_predictor_params.items():
                    row[f"{param_key}_sim_pred_predictor"] = param_val
 
            rows.append(row)

    return pd.DataFrame(rows)


def compute_param_recovery_correlations(df: pd.DataFrame, dir_path: str, out_csv_name: str, *, true_role: str = "predictor", fitted_suffix: str = "_fitted_predictor",
                                        round_mode: str = "first", params: list[str] | None = None, create_new_file: bool = False) -> pd.DataFrame:
    """
    Compute correlations between true and fitted parameters in the simulation.

    For each parameter p in `params`, this function correlates:
        p_true_<true_role>  vs  p_fitted_predictor
    either at:
        • the first round per dyad,
        • the final round per dyad, or
        • every round (round_mode="all").

    Results are saved as a tidy CSV with columns:
        ["round", "n_data", "param", "corr", "ci_lower", "ci_upper"].

    Arguments:
        • df: pd.DataFrame;
            Long-format simulation DataFrame from `load_simulated_fits_from_json`
            (or its concatenation across files).
        • dir_path: str;
            Directory where the correlation CSV should be stored.
        • out_csv_name: str;
            File name for the CSV (e.g., "correlation_results.csv").
        • true_role: str;
            Which true parameters to correlate against:
                - "predictor": parameter recovery of the optimizer.
                - "chooser":   convergence toward the chooser's true parameters.
        • round_mode: str;
            How to slice rounds:
                - "first": one row per dyad at its earliest round.
                - "final": one row per dyad at its latest round.
                - "all":   one row per (round, param).
        • params: list[str] | None;
            Base parameter names like ["Vii","Vij","τ"]. If None, auto-detect all
            parameters that have both <param>_true_<true_role> and <param>_fitted_predictor.
        • create_new_file: bool;
            If False and the CSV already exists, load and return it. If True, recompute
            and overwrite the CSV.

    Returns:
        • pd.DataFrame;
            Tidy correlation table with one row per (round_label, param).
    """
    def fisher_z_confidence_interval(pearson_correlation, sample_size, significance_level=0.05):
        """
        Return a confidence interval for a Pearson correlation using the Fisher Z transformation.

        Arguments:
            • pearson_correlation: float
                Observed Pearson r; must satisfy |r| < 1.0.
            • sample_size: int
                Number of observation pairs used to compute the correlation.
            • significance_level: float
                Two-tailed significance level; default 0.05 gives a 95% CI.

        Returns:
            • tuple[float, float] — (lower_bound, upper_bound) on the correlation scale,
              or (NaN, NaN) if `sample_size < 4` or `|pearson_correlation| >= 1.0`.
        """
        if sample_size < 4 or abs(pearson_correlation) >= 1.0:
            return (np.nan, np.nan)

        "Fisher z"
        fisher_z_transform = 0.5 * np.log((1 + pearson_correlation) / (1 - pearson_correlation))
        "Standard error"
        fisher_z_standard_error = 1.0 / math.sqrt(sample_size - 3)
        z_critical_value = 1.96  # Approximate 95% CI.

        z_lower_bound = fisher_z_transform - z_critical_value * fisher_z_standard_error
        z_upper_bound = fisher_z_transform + z_critical_value * fisher_z_standard_error
        "Transform back"
        correlation_lower_bound = (math.exp(2*z_lower_bound) - 1) / (math.exp(2*z_lower_bound) + 1)
        correlation_upper_bound = (math.exp(2*z_upper_bound) - 1) / (math.exp(2*z_upper_bound) + 1)
        return (correlation_lower_bound, correlation_upper_bound)

    out_path = os.path.join(dir_path, out_csv_name)
    if (not create_new_file) and os.path.exists(out_path):
        corr_df = pd.read_csv(out_path, encoding="utf-8", engine="python")
        if "Unnamed: 0" in corr_df.columns:
            del corr_df["Unnamed: 0"]
        return corr_df

    "Pick rows for first/final"
    if round_mode == "first":
        idx = df.groupby("dyad_key")["round"].idxmin()
        df_use = df.loc[idx]
        rounds = ["first"]
    elif round_mode == "final":
        idx = df.groupby("dyad_key")["round"].idxmax()
        df_use = df.loc[idx]
        rounds = ["final"]
    else:
        df_use = df.copy()
        rounds = sorted(df["round"].dropna().unique().tolist())

    "Auto-detect parameters if needed"
    if params is None:
        params = []
        suffix_true = f"_true_{true_role}"
        for col in df_use.columns:
            if col.endswith(suffix_true):
                base = col[: -len(suffix_true)]
                paired = f"{base}{fitted_suffix}"
                if paired in df_use.columns:
                    params.append(base)
        params = sorted(set(params))

    def _one_round(round_subset_dataframe: pd.DataFrame, label: str) -> pd.DataFrame:
        """
        Compute recovery correlations for one round subset.

        Returns one row per parameter, including the number of valid dyads,
        Pearson correlation, and Fisher-z confidence interval.
        """
        correlation_records = []
        for parameter_name in params:
            true_value_column_name = f"{parameter_name}_true_{true_role}"
            fitted_value_column_name = f"{parameter_name}{fitted_suffix}"
            valid_rows = round_subset_dataframe.dropna(subset=[true_value_column_name, fitted_value_column_name])
            n_valid_pairs = len(valid_rows)
            if n_valid_pairs < 3:
                pearson_correlation = np.nan
                confidence_interval_lower = confidence_interval_upper = np.nan
            else:
                pearson_correlation = valid_rows[[true_value_column_name, fitted_value_column_name]].corr().iloc[0, 1]
                confidence_interval_lower, confidence_interval_upper = fisher_z_confidence_interval(pearson_correlation, n_valid_pairs)
            correlation_records.append({
                "round": label,
                "n_data": n_valid_pairs,
                "param": parameter_name,
                "corr": pearson_correlation,
                "ci_lower": confidence_interval_lower,
                "ci_upper": confidence_interval_upper
            })
        return pd.DataFrame(correlation_records)

    frames = []
    if round_mode in ("first", "final"):
        frames.append(_one_round(df_use, rounds[0]))
    else:
        for round_idx in rounds:
            frames.append(_one_round(df_use[df_use["round"] == round_idx], round_idx))

    corr_df = pd.concat(frames, ignore_index=True)
    os.makedirs(dir_path, exist_ok=True)
    corr_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("Saved correlation results to:", out_path)
    return corr_df


def run_simulation_recovery_analysis(figure_layout: dict, general_settings: GeneralSettings, file_paths: FilePaths, export_fig: bool = True, create_new_file: bool = False, 
                                     produce_figures: bool = True, include_dropdown: bool = True, correlation_csv_name: str = "correlation_results.csv", use_dynamic_predictor: bool = False) -> pd.DataFrame:
    """
    End-to-end analysis of simulation-based parameter recovery.

    This orchestration function:
        1) Reads all simulated dyad JSON files from `dir_path`.
        2) Converts each to a long DataFrame via `load_simulated_fits_from_json`.
        3) Concatenates and saves the merged DataFrame to:
               ./simulations/simulated_fits.csv
        4) Computes param-recovery correlations (first/final/all) and writes
           `correlation_csv_name` in the same results folder.
        5) Optionally generates:
               - violin/boxplots of recovery by param and round,
               - scatterplots of true vs fitted parameters,
               - line plots of correlation vs round.

    Arguments:
        • figure_layout: dict;
            Plotly layout configuration used for all generated figures.
        • general_settings: GeneralSettings;
            Settings for the Bayesian analysis and plotting (e.g., dark_mode).
        • file_paths: dict[str: str | dict[str: str]];
            Dictionary of all files paths in this project.              
        • export_fig: bool;
            If True, write .html figures to disk. If False, display interactively.
        • create_new_file: bool;
            If False and merged CSV / correlation CSV already exist, reuse them.
            If True, re-read JSON, rebuild the merged DataFrame, and recompute correlations.
        • produce_figures: bool;
            If True, generate scatter, violin/box, and correlation-by-round figures.
        • include_dropdown: bool;
            Whether to include a violin/box dropdown in the univariate plots.
        • dir_path: str;
            Directory containing the per-dyad JSON files produced by the simulation fits.
        • correlation_csv_name: str;
            File name for the main correlation CSV (e.g., "correlation_results.csv").

    Returns:
        • pd.DataFrame;
            The merged, long-format simulation DataFrame (all dyads × rounds).
    """
    def plot_correlation(df: pd.DataFrame, file_paths: FilePaths, round_selection: str = "first", as_scatterplot: bool = True, params: list = None, boxplot_param: str = "Vij", 
                        include_dropdown: bool = True, figure_layout: dict = None, export_fig: bool = True, out_path: str = "corr_plot.html", fitted_suffix: str = "_fitted_predictor"):
        """
        Visualize true vs fitted parameters for a chosen round.

        Modes:
            • Scatter mode (as_scatterplot=True):
                - Plots true_predictor vs fitted_predictor for each param in `params`.
                - Adds a dropdown to toggle which parameter is shown.
                - Overlays a best-fit line and annotates correlation, R², and n.

            • Violin/box mode (as_scatterplot=False):
                - For a single `boxplot_param`, groups by the true predictor value,
                  and plots the distribution of fitted predictor estimates.
                - Annotates the overall correlation between true and fitted values.

        `round_selection` can be "first", "final", or a specific integer round.
        """
        if figure_layout is None:
            figure_layout = {}
        if as_scatterplot and (params is None or len(params) == 0):
            params = ["Vii", "Vij", "Vii_std", "Vij_std", "τ"]

        "1) Subset data to the specified round"
        if round_selection == "first":
            idxmin_ = df.groupby("dyad_key")["round"].idxmin()
            df_sub = df.loc[idxmin_]
        elif round_selection == "final":
            idxmax_ = df.groupby("dyad_key")["round"].idxmax()
            df_sub = df.loc[idxmax_]
        else:
            try:
                r_sel = int(round_selection)
                df_sub = df[df["round"] == r_sel]
            except:
                df_sub = df

        param_titles = {
            "Vii":     "Mean Self-interest μ(𝑉𝑖𝑖)", 
            "Vij":     "Mean Altruism μ(𝑉𝑖𝑗)", 
            "Vii_std": "Self-interest Standard Deviation σ(𝑉𝑖𝑖)", 
            "Vij_std": "Altruism Standard Deviation σ(𝑉𝑖𝑗)", 
            "τ":       "SoftMax Temperature (τ)"
        }

        def axis_title(param: str, role: str, type_: str) -> str:
            return f"{type_.capitalize()} {role.capitalize()} Parameter: {param_titles[param]}"

        "2) If as_scatterplot => multi param dropdown"
        if as_scatterplot:
            fig = go.Figure()
            """
            One param => 2 traces: (1) scatter, (2) best-fit line but with 
            multi param dropdown, there are effectively 2*N traces total.
            """

            "Store an annotation template for correlation"
            annotation_base = dict(
                x=0.02, y=0.95, xref='paper', yref='paper',
                showarrow=False, align="left",
                font=dict(size=18)
            )

            """
            Store all \"visible\" arrays for each param, each param has 2 traces => total = 2 * len(params).
            Better to have exactly 2 visible for the chosen param, else 2 invisible for others.
            """
            param_buttons = []
            param_traces_startidx = {}  # Param -> first trace index.

            i_trace = 0
            for param_idx, param in enumerate(params):
                param_traces_startidx[param] = i_trace

                try:
                    "Gather data"
                    xcol = f"{param}_true_predictor"
                    ycol = f"{param}{fitted_suffix}"
                    subp = df_sub.dropna(subset=[xcol, ycol])
                except KeyError:
                    param = param.replace('Vii', 'Vᵢᵢ')
                    param = param.replace('Vij', 'Vᵢⱼ')
                    "Gather data"
                    xcol = f"{param}_true_predictor"
                    ycol = f"{param}{fitted_suffix}"
                    subp = df_sub.dropna(subset=[xcol, ycol])
                "Correlation"
                corr = np.nan
                r_squared = np.nan
                n_valid_points = len(subp)
                if n_valid_points >= 2:
                    corr = subp[[xcol,ycol]].corr().iloc[0,1]
                    r_squared = corr**2

                "Scatter"
                scatter_trace = go.Scatter(
                    x=subp[xcol],
                    y=subp[ycol],
                    mode='markers',
                    name=f"{param}_scatter",
                    visible=(param_idx==0),  # Show only the first param by default.
                    hovertemplate=(
                        f"{param}_true_predictor=%{{x:.3f}}<br>"
                        f"{param}{fitted_suffix}=%{{y:.3f}}<extra></extra>"
                    ),
                    marker=dict(size=figure_layout.get("markersize", 12))
                )
                fig.add_trace(scatter_trace)
                i_trace += 1

                "Best-fit line"
                line_trace = None
                if n_valid_points >= 2:
                    "Do a linear fit"
                    xvals = subp[xcol].values
                    yvals = subp[ycol].values
                    if xvals.min() == xvals.max():
                        slope, intercept = 0.0, float(yvals.mean())
                    else:
                        slope, intercept = np.polyfit(xvals, yvals, 1)

                    "For plotting, covers the range of xvals."
                    x_min, x_max = xvals.min(), xvals.max()
                    x_line = np.linspace(x_min, x_max, 50)
                    y_line = slope*x_line + intercept
                    line_trace = go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode='lines',
                        name=f"Best Fit: {param}",
                        visible=(param_idx==0),
                        line=dict(dash='dot', width=3),
                        hoverinfo='skip'
                    )
                else:
                    "No data or not enough"
                    line_trace = go.Scatter(
                        x=[],
                        y=[],
                        mode='lines',
                        visible=(param_idx==0),
                        name=f"Best Fit: {param}"
                    )
                fig.add_trace(line_trace)
                i_trace += 1

                "Annotation text for correlation"
                if not math.isnan(corr) and not math.isnan(r_squared):
                    cor_txt = f"r = {corr:.3f}, R² = {r_squared:.3f}, n={n_valid_points}"
                else:
                    cor_txt = f"r = n/a, R²=n/a, n={n_valid_points}"

                "Store that text in the layout for param0. Override it via update menus for param>0"
                if param_idx == 0:
                    "Put it in layout"
                    fig.update_layout(
                        annotations=[dict(
                            text=cor_txt,
                            **annotation_base
                        )]
                    )

                """
                Create a param_buttons entry that sets the 2 traces for param visible
                And sets all others invisible, plus updates the annotation text, plus updates title
                Fill that after gathering them all.              
                """

            """
            Create update menu
            There are 2 traces per param => total 2*len(params).
            For param i => indices 2i, 2i+1           
            """
            n_traces = 2*len(params)
            for param_idx, param in enumerate(params):
                try:
                    "Figure out correlation for annotation"
                    xcol = f"{param}_true_predictor"
                    ycol = f"{param}{fitted_suffix}"
                    subp = df_sub.dropna(subset=[xcol,ycol])
                except KeyError:
                    param = param.replace('Vii', 'Vᵢᵢ')
                    param = param.replace('Vij', 'Vᵢⱼ')
                    xcol = f"{param}_true_predictor"
                    ycol = f"{param}{fitted_suffix}"
                    subp = df_sub.dropna(subset=[xcol,ycol])

                n_valid_points = len(subp)
                if n_valid_points > 1:
                    correlation_value = subp[[xcol,ycol]].corr().iloc[0,1]
                    r_squared = correlation_value**2
                    c_text = f"r={correlation_value:.3f}, R²={r_squared:.3f}, n={n_valid_points}"
                else:
                    c_text = f"r=n/a, R²=n/a, n={n_valid_points}"

                "Build a \"visible\" array"
                vis = [False]*n_traces
                vis[2*param_idx] = True
                vis[2*param_idx+1] = True

                param_buttons.append(
                    dict(
                        label=param,
                        method='update',
                        args=[
                            {'visible': vis},
                            {
                            'title': f"Scatter round={round_selection}, param={param}",
                            'annotations': [dict(text=c_text, **annotation_base)]
                            }
                        ]
                    )
                )

            fig.update_layout(
                template=figure_layout.get("template", "plotly_dark"),
                title=f"Scatter: round={round_selection}, param={params[0]}",
                xaxis=dict(title="Chooser True Param", **figure_layout.get("xaxis", {}), scaleanchor="y", scaleratio=1),
                yaxis=dict(title="Predictor Fitted Param", **figure_layout.get("yaxis", {})),
                updatemenus=[dict(type='dropdown', showactive=True, buttons=param_buttons, x=1.3, y=0.9)],
                hoverlabel=figure_layout.get("hoverlabel", {}),
                font=figure_layout.get("font", {})
            )

            if export_fig:
                fig.write_html(out_path)
                print("Saved scatter figure to", out_path)
            else:
                fig.show()

            return fig

        else:
            # As_scatterplot=False => use a violin-boxplot with a single param.
            param = boxplot_param
            xcol_true  = f"{param}_true_predictor"
            ycol_fitted= f"{param}_fitted_predictor"

            try:
                sub = df_sub.dropna(subset=[xcol_true, ycol_fitted]).copy()
            except KeyError:
                param = param.replace('Vii', 'Vᵢᵢ')
                param = param.replace('Vij', 'Vᵢⱼ')
                xcol_true  = f"{param}_true_predictor"
                ycol_fitted= f"{param}_fitted_predictor"
                sub = df_sub.dropna(subset=[xcol_true, ycol_fitted]).copy()        
            n_data = len(sub)
            corr = np.nan
            if n_data >= 2:
                corr = sub[[xcol_true,ycol_fitted]].corr().iloc[0,1]

            try: param_title = param_titles[param]
            except KeyError:
                param = param.replace('Vᵢᵢ', 'Vii')
                param = param.replace('Vᵢⱼ', 'Vij')
                param_title = param_titles[param]

            fig = go.Figure()
            fig.add_trace(go.Violin(
                x=sub[xcol_true],
                y=sub[ycol_fitted],
                box=dict(visible=True),
                meanline=dict(visible=True),
                line_color='hsla(115, 70%, 40%, 1.0)',
                points='all', pointpos=-0.7, jitter=0.45, 
                scalemode='count', width=0.3, name=param,
                hovertemplate=(
                    f"True {param_title} = %{{x}}<br>Fitted "
                    f"{param_title} = %{{y:.3f}}<extra></extra>"
                )
            ))

            "Title text and correlation annotation"
            title_text = f"True {round_selection.capitalize()} Round Parameter by Fitted "
            title_text += f"{round_selection.capitalize()} Round Parameter for {param_title}"
            cor_text = f"Correlation = {corr:.3f}, n = {n_data}" if not math.isnan(corr) else ""
            cor_text += " (Simulated Data)" 
            fig.update_yaxes(range=[-1.2, 1.2])
            fig.update_layout(
                title=title_text, 
                titlefont_size=figure_layout['titlefont_size']-2,
                template=figure_layout.get("template", "plotly_dark"),
                title_x=figure_layout['title_x'], title_y=figure_layout['title_y'], 
                xaxis=dict(title=axis_title(param, 'predictor', 'true'), **figure_layout.get("xaxis", {})),
                yaxis=dict(title=axis_title(param, 'predictor', 'fitted'), **figure_layout.get("yaxis", {})),
                hoverlabel=figure_layout.get("hoverlabel", {}),
                margin=dict(l=150, r=120, t=120, b=120),
                font=figure_layout.get("font", {}),
                annotations=[dict(
                    text=cor_text,
                    x=0.02, y=0.85, xref='paper', yref='paper',
                    showarrow=False, align="left",
                    font=dict(size=30)
                )]
            )

            if boxplot_param == "Vij":
                tickvals = [-1.000, -0.667, -0.333, 0.000, 0.333, 0.667, 1.000]
                ticktext = ["-1", "-⅔", "-⅓", "0", "⅓", "⅔", "1"]
                fig.update_xaxes(tickvals=tickvals, ticktext=ticktext)

            if include_dropdown: 
                """Dropdown menu to switch between violin and boxplot:"""
                fig.update_layout(updatemenus=[dict(buttons=list([
                    dict(args=["type", "violin"], label="Violin", method="restyle"),
                    dict(args=["type", "box"], label="Boxplot", method="restyle")]),
                    direction="down", pad={"r": 10, "t": 10}, showactive=True, 
                    x=0.88, xanchor="left", y=0.1, yanchor="top")])

            if export_fig:
                fig.write_html(out_path)
                print("Saved violin-box figure to", out_path)
            else:
                fig.show()

            return fig

    dir_path = ensure_directory_and_join(file_paths['player_fits'], 'experiment_0')
    sim_dir = str(file_paths['simulations'])
    os.makedirs(sim_dir, exist_ok=True)

    merged_csv_path = os.path.join(sim_dir, "simulated_fits.csv")

    df_combined = None
    if not create_new_file and os.path.exists(merged_csv_path):
        df_combined = pd.read_csv(merged_csv_path, encoding="utf-8", engine='python')    
        if 'Unnamed: 0' in df_combined.columns:
            del df_combined['Unnamed: 0']

    if df_combined is None:
        "1) Read all JSON files."
        all_dfs = []
        for file_name in os.listdir(dir_path):
            if file_name.endswith(".json"):
                path = os.path.join(dir_path, file_name)
                df_game = load_simulated_fits_from_json(path)
                all_dfs.append(df_game)
        if not all_dfs:
            print(f"No JSON found in {pretty_path(dir_path)}. Returning empty.")
            return pd.DataFrame()

        df_combined = pd.concat(all_dfs, ignore_index=True)
        df_combined["round"] = pd.to_numeric(df_combined["round"], errors="coerce")

        "2) Save"
        merged_csv_path = os.path.join(sim_dir, "simulated_fits.csv")
        df_combined.to_csv(merged_csv_path, index=False, encoding='utf-8-sig')
        print("Saved combined DataFrame to", merged_csv_path)

    "3) Correlation and plots follow the existing parameter-recovery approach."
    "Example: correlation between \"Vij_true_predictor\" and \"Vij_fitted_predictor\"."

    fitted_suffix = "_sim_pred_predictor" if use_dynamic_predictor else "_fitted_predictor"

    "Here's the function that does round-based correlation"
    corr_df_out = compute_param_recovery_correlations(df=df_combined, dir_path=sim_dir, 
                    out_csv_name=correlation_csv_name, create_new_file=create_new_file,
                    true_role="chooser", round_mode="all", fitted_suffix=fitted_suffix)

    if produce_figures:
        for round_selection in ('first', 'final'):
            for param in ("Vii", "Vij", "Vii_std", "Vij_std", "τ"):
                "Boxplot/violin"
                plot_correlation(
                    df=df_combined,
                    round_selection=round_selection,
                    as_scatterplot=False,
                    boxplot_param=param,
                    file_paths=file_paths,
                    figure_layout=figure_layout,
                    export_fig=export_fig,
                    include_dropdown=include_dropdown,
                    out_path=os.path.join(sim_dir, f"corr_violin_{param}_{round_selection}.html"),
                    fitted_suffix=fitted_suffix,
                )
            "Scatterplot"
            plot_correlation(
                df=df_combined,
                figure_layout=figure_layout,
                export_fig=export_fig,
                round_selection=round_selection,
                as_scatterplot=True,
                params=["Vii","Vij","Vii_std","Vij_std","τ"],
                file_paths=file_paths,
                out_path=os.path.join(sim_dir, f"corr_scatter_{round_selection}.html"),
                fitted_suffix=fitted_suffix,
            )

        "Correlation by round => line"
        plot_param_recovery_by_round(
            general_settings=general_settings,
            df_merged=df_combined, params=["Vii", "Vij", "Vii_std", "Vij_std", "τ"], figure_layout=figure_layout, 
            export_fig=export_fig, create_new_file=create_new_file, file_paths=file_paths, 
            file_name=("corr_by_round_sim_pred.html" if use_dynamic_predictor else "corr_by_round.html"),
            corr_csv_name=("correlation_results_by_round_sim_pred.csv" if use_dynamic_predictor else "correlation_results_by_round.csv"),
            fitted_suffix=fitted_suffix, fit_mode='poly', poly_degree=3
        )

    return df_combined


def compute_recovery_by_prior_bins(df: pd.DataFrame, var_col="Vᵢⱼ_std_fitted_predictor", temp_col="τ_fitted_predictor", param_true_chooser="Vᵢⱼ_true_chooser", 
                                      param_fitted_predictor="Vᵢⱼ_fitted_predictor", player_id_col="player_uuid_predictor", var_edges: list[float] = None, 
                                      temp_edges: list[float] = None, last_rounds: list[int] = [18,19,20], print_: bool = True) -> dict:
    """
    Quantify parameter recovery as a function of prior variance and temperature.

    For each predictor:
        1) Look at their round-0 row to read:
            var_col   = prior variance (e.g., σ(Vij))
            temp_col  = prior SoftMax temperature.
        2) Bin each into 3 levels (low/med/high) using quantile-based edges
            or user-provided edges → var_bin ∈ {1,2,3}, temp_bin ∈ {1,2,3}.
        3) For all rows in `last_rounds`, compute:
            Corr(param_true_chooser, param_fitted_predictor)
            within each (var_bin, temp_bin) combination.

    Returns 3×3 tables of correlations and bin counts plus the bin edges.

    Arguments:
        • df: pd.DataFrame;
            Long simulation DataFrame (all rounds, all dyads).
        • var_col: str;
            Column representing the prior variance at round 0
            (typically something like "<param>_std_fitted_predictor").
        • temp_col: str;
            Column representing the prior temperature at round 0
            (e.g., "τ_fitted_predictor").
        • param_true_chooser: str;
            Column name for the chooser's true parameter (e.g., "Vij_true_chooser").
        • param_fitted_predictor: str;
            Column name for the predictor's fitted parameter (e.g., "Vij_fitted_predictor").
        • player_id_col: str;
            Column identifying predictors (e.g., "player_uuid_predictor").
        • var_edges: list[float] | None;
            Optional bin edges for variance. If None, computed as tertiles.
        • temp_edges: list[float] | None;
            Optional bin edges for temperature. If None, computed as tertiles.
        • last_rounds: list[int];
            Rounds over which to compute final recovery correlations.

    Returns:
        • dict;
            {
                "corr_table":  3×3 DataFrame of recovery correlations by (var_bin, temp_bin),
                "count_table": 3×3 DataFrame of participant counts per bin,
                "var_edges":   list[float] bin edges used for variance,
                "temp_edges":  list[float] bin edges used for temperature,
            }
    """
    def bin_index(value: float, edges: list[float]) -> int:
        """
        Map value into a 1-based bin index given ordered edges.
        """
        if pd.isna(value):
            return math.nan
        idx = np.searchsorted(edges, value, side="right") - 1
        if idx<0:
            idx=0
        if idx>=len(edges)-1:
            idx=len(edges)-2
        return idx+1

    def assign_bin_edges(series: pd.Series, nbins=3) -> list[float] | None:
        """
        Compute bin edges from quantiles [0, 1/nbins, ..., 1].
        Returns None if data are constant (degenerate — no binning possible).
        """
        non_null_series = series.dropna()
        if len(non_null_series) < 3:
            return None
        qvals = non_null_series.quantile([idx/nbins for idx in range(nbins+1)]).values
        "That yields 4 values for nbins=3, i.e. 0.0, 0.33...,0.66...,1.0 quantiles"
        if qvals.min() == qvals.max():
            return None
        return qvals.tolist()

    "0) Make a copy"
    df_ = df.copy()

    df0 = df_[df_["round"]==0].copy()
    if var_col not in df0.columns:
        var_col = (var_col
                .replace("Vii", "Vᵢᵢ")
                .replace("Vij", "Vᵢⱼ"))

    if temp_col not in df0.columns:
        fallback = temp_col.replace("τ", "temp") if "τ" in temp_col else temp_col.replace("temp", "τ")
        if fallback in df0.columns:
            temp_col = fallback

    if print_:
        print("Unique participants with round=0:", df0[player_id_col].nunique())
        print("Total rows with round=0:", len(df0))

    # If var_edges or temp_edges is None => compute from quantiles.
    if var_edges is None:
        var_edges = assign_bin_edges(df0[var_col], nbins=3)
    if temp_edges is None:
        temp_edges = assign_bin_edges(df0[temp_col], nbins=3)

    if var_edges is None or temp_edges is None:
        degenerate = []
        if var_edges is None:
            degenerate.append(f"{var_col} (all values = {df0[var_col].iloc[0]:.3g})")
        if temp_edges is None:
            degenerate.append(f"{temp_col} (all values = {df0[temp_col].iloc[0]:.3g})")
        print(f"[compute_recovery_by_prior_bins] Skipping: constant data in {', '.join(degenerate)} — no binning possible.")
        return {}

    "Build a small DataFrame: [player_id, prior_var, prior_temp, var_bin, temp_bin]"
    bin_rows = []
    for pid, rowsub in df0.groupby(player_id_col):
        row = rowsub.iloc[0]  # First row if multiple.
        var_val  = row[var_col]
        temp_val = row[temp_col]
        variance_bin_index = bin_index(var_val, var_edges)
        temperature_bin_index = bin_index(temp_val, temp_edges)
        bin_rows.append({
            player_id_col: pid,
            "prior_var": var_val,
            "prior_temp": temp_val,
            "var_bin": variance_bin_index,
            "temp_bin": temperature_bin_index
        })
    df_bininfo = pd.DataFrame(bin_rows)

    "2) Merge bininfo onto all rows => so every row now has (var_bin,temp_bin, prior_var, prior_temp)"
    df_merged = pd.merge(df_, df_bininfo, on=player_id_col, how="left")

    "3) Filter to last_rounds => gather all rows from these final rounds"
    sub_last = df_merged[df_merged["round"].isin(last_rounds)].copy()
    sub_last = sub_last.dropna(
        subset=["var_bin","temp_bin", param_true_chooser, param_fitted_predictor]
    )

    "=========== (A) Build 3x3 correlation table by bins ============="
    group_cols = ["var_bin","temp_bin"]

    def group_corr_bin(gdf_sub):
        if len(gdf_sub) < 3:
            return np.nan
        xvals = gdf_sub[param_true_chooser].values
        yvals = gdf_sub[param_fitted_predictor].values
        return np.corrcoef(xvals,yvals)[0,1]

    corr_ser = (
        sub_last
        .groupby(group_cols, group_keys=False)[[param_true_chooser, param_fitted_predictor]]
        .apply(group_corr_bin)
    )

    "Corr_ser should now be a Series indexed by (var_bin, temp_bin)"
    corr_df = corr_ser.reset_index()
    "Last column is the correlation value; rename it"
    last_col = corr_df.columns[-1]
    corr_df = corr_df.rename(columns={last_col: "corr_value"})


    "Build a 3x3 correlation matrix"
    cmat = np.full((3, 3), np.nan, dtype=float)
    for _, correlation_row in corr_df.iterrows():
        variance_bin_index = int(correlation_row["var_bin"])
        temperature_bin_index = int(correlation_row["temp_bin"])
        correlation_value = correlation_row["corr_value"]
        if 1 <= variance_bin_index <= 3 and 1 <= temperature_bin_index <= 3:
            cmat[variance_bin_index - 1, temperature_bin_index - 1] = correlation_value

    var_labels = ["LowVar","MedVar","HighVar"]
    temp_labels= ["LowTemp","MedTemp","HighTemp"]
    corr_df = pd.DataFrame(cmat, index=var_labels, columns=temp_labels)
    
    "Build a 3x3 count table => how many participants are in each bin (round=0)"
    count_mat = np.zeros((3,3), dtype=float)
    bin_df = df_bininfo.dropna(subset=["var_bin","temp_bin"])
    bin_counts = bin_df.groupby(["var_bin","temp_bin"]).size()
    for (variance_bin_index, temperature_bin_index), participant_count in bin_counts.items():
        if 1 <= variance_bin_index <= 3 and 1 <= temperature_bin_index <= 3:
            count_mat[variance_bin_index - 1, temperature_bin_index - 1] = participant_count
    count_df = pd.DataFrame(count_mat, index=var_labels, columns=temp_labels)

    if print_:
        print("Correlation table (last rounds):\n", corr_df)
        print("Count table (# participants in each bin at round=0):\n", count_df)
        print("Used var edges:", var_edges)
        print("Used temp edges:", temp_edges)

    return {
        "corr_table": corr_df,     # 3x3 table of correlation by bins
        "count_table": count_df,   # 3x3 table of bin counts
        "var_edges": var_edges,
        "temp_edges": temp_edges,
    }


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    total_minutes = int(seconds) // 60
    if total_minutes >= 60:
        hours = total_minutes // 60
        mins  = total_minutes % 60
        return f"{hours} hours {mins} minutes"
    secs = int(seconds) % 60
    return f"{total_minutes} minutes {secs:02d} seconds"


def latin_square(size: int) -> list[list[int]]:
    """
    Produces a symmetrical size x size latin square.

    Author: Liam Tsimhoni (Morality Game research assistant).
    """
    latin_sq = np.zeros(shape=(size, size), dtype=object)
    def val(idx, jdx, n_players=size):
        if (jdx == 0): return idx
        if (idx == n_players - 1): return ((n_players // 2) * (jdx - 1) % (n_players - 1))
        if (jdx == (2 * idx) % (n_players - 1) + 1): return n_players - 1
        return (jdx - 1 - idx) % (n_players - 1)
    for idx in range(size):
        for jdx in range(size):
            latin_sq[idx][jdx] = val(idx, jdx, size)
    return latin_sq


def _pairs_from_ls_column(latin_sq, jdx: int) -> list[tuple[int, int]]:
    """Return the unique undirected pairs encoded by column jdx of a latin square."""
    seen, pairs = set(), []
    for idx in range(len(latin_sq)):
        opponent_idx = int(latin_sq[idx][jdx])
        if opponent_idx != idx and (idx, opponent_idx) not in seen and (opponent_idx, idx) not in seen:
            pairs.append((idx, opponent_idx))
            seen.add((idx, opponent_idx))
    return pairs


def _sample_batch_game_counts(random_gen: np.random.Generator, n_games: int) -> dict[tuple[int, int], int]:
    """
    Sample game counts for all 6 pairs in a 4-player batch.

    Draws three integers (x, y, z) each ≥ 1 with x+y+z = n_games, then derives the remaining
    three pair counts via the row-sum constraint so every player's total = n_games:

        G[0,1]=x,  G[0,2]=y,  G[0,3]=n_games-x-y
        G[1,2]=z,  G[1,3]=n_games-x-z
        G[2,3]=n_games-y-z

    Since x,y,z ≥ 1, all six counts are guaranteed ≥ 1.

    Arguments:
        • random_gen: np.random.Generator; stateful generator shared across all batches.
        • n_games: int; total game budget per player (must be ≥ 3).

    Returns:
        • dict[tuple[int,int], int]; maps each of the 6 unique (local) player-index pairs to
          the number of games that pair will play.
    """
    if n_games < 3:
        raise ValueError(f"n_games must be ≥ 3 (got {n_games}) so every pair plays ≥ 1 game.")

    "Sample from the interior of the integer simplex so every count ≥ 1 and sums = n_games."
    "The three directly sampled values are G[0,1], G[0,2], G[1,2]; the remaining three are derived via row-sum constraints."
    raw_proportions = random_gen.uniform(0.0, 1.0, 3)
    floats_shifted  = raw_proportions / raw_proportions.sum() * (n_games - 3)
    floored         = np.floor(floats_shifted).astype(int)
    remainder_needed = (n_games - 3) - int(floored.sum())
    if remainder_needed > 0:
        remainders  = floats_shifted - floored
        top_indices = np.argsort(remainders)[-remainder_needed:]
        for remainder_idx in top_indices:
            floored[remainder_idx] += 1

    sampled_counts = floored + 1   # each entry ≥ 1; [G[0,1], G[0,2], G[1,2]]

    return {
        (0, 1): int(sampled_counts[0]),
        (0, 2): int(sampled_counts[1]),
        (0, 3): n_games - int(sampled_counts[0]) - int(sampled_counts[1]),
        (1, 2): int(sampled_counts[2]),
        (1, 3): n_games - int(sampled_counts[0]) - int(sampled_counts[2]),
        (2, 3): n_games - int(sampled_counts[1]) - int(sampled_counts[2]),
    }


def create_simulated_experiment(
    n_players: int,
    n_games: int,
    k_params: int,
    utility_settings_k: UtilitySettings,
    general_settings: GeneralSettings,
    param_bds: ParamBounds,
    random_gen: np.random.Generator,
    altruism_key: str,
    altruism_targets: list[float],
    file_paths: FilePaths,
    create_new_file: bool | None = None,
    enforce_memory_limit: bool = False,
) -> tuple[dict, dict[str, dict]]:
    """
    Generate (or retrieve) a full synthetic experiment in the same JSON format as the real raw data.

    Follows the standard caching convention: if the output file already exists and
    `create_new_file=False`, loads and returns it immediately without regeneration.
    Otherwise generates, assembles, writes to disk, and returns the fresh data.

    Players are organized into batches of 4. Within each batch, all 6 unique undirected pairs
    are formed (one dyad per pair). Game counts per pair are drawn probabilistically so that
    each player's total across their 3 dyads equals n_games. Within each dyad, chooser and
    predictor roles are assigned independently by a fair coin flip on every round, mirroring
    the real experiment — a player may be chooser in one round and predictor in the next.

    After generating all rounds for a pair, the UBM (agent()) is run separately for each
    player's predictor sub-sequence within that dyad. Because agent() checks game["predictor"]
    against player_uuid per round and skips non-matching games, no special sub-sequence
    extraction is needed.

    Producing synthetic data in exactly the same format as the human raw data reduces the risk
    of inadvertently breaking existing pipeline functions and makes the parameter recovery test
    maximally faithful: the optimizer runs under the same conditions it faces on real data.

    Arguments:
        • n_players: int; total number of synthetic players (padded to the next multiple of 4).
        • n_games: int; game budget per player (each player's 3 dyads sum to n_games; must be ≥ 3).
        • k_params: int; number of free parameters in this model family (used in UUIDs).
        • utility_settings_k: UtilitySettings; utility family for this k.
        • general_settings: GeneralSettings; passed through to agent() and make_param_info().
        • param_bds: ParamBounds; passed through to make_param_info().
        • random_gen: np.random.Generator; stateful generator (shared across all batches).
        • altruism_key: str; parameter key for the altruism dimension (e.g. 'Vᵢⱼ').
        • altruism_targets: list[float]; target altruism values to assign across players.
        • file_paths: FilePaths; routing dict — must have 'processed' and 'file_names' set.
        • create_new_file: bool | None; if False, loads from disk when the output file
          exists; if True, always regenerates and overwrites; if None (default), defers to
          general_settings.get('create_new_file', False).

    Returns:
        • (histories_dict, true_params_by_uuid) where:
            - histories_dict: dict; {'histories': ..., 'player_info': ...} in the same
              format as the raw experiment JSON — ready for run_analysis_bayes.
            - true_params_by_uuid: dict[str, dict]; maps each player UUID to their
              ground-truth utility parameter vector.
    """
    "Defer to general_settings when create_new_file not explicitly specified by caller."
    if create_new_file is None:
        create_new_file = general_settings.get('create_new_file', False)

    "Retrieve cached result if available and permitted."
    histories_file_path = os.path.join(
        file_paths['processed'],
        file_paths['file_names']['player_pairs_exper3']
    )
    if not create_new_file and os.path.exists(histories_file_path):
        with open(histories_file_path, 'r', encoding='utf-8') as cached_file:
            histories_dict = json.load(cached_file)
        true_params_by_uuid = {
            player_uuid: dict(player_data.get('true_params', {}).get('chooser', {}))
            for player_uuid, player_data in histories_dict.get('player_info', {}).items()
        }
        print(f"[create_simulated_experiment k={k_params}] Loaded existing file "
              f"({len(true_params_by_uuid)} players): {pretty_path(histories_file_path)}")
        return histories_dict, true_params_by_uuid

    n_players_padded = n_players + (4 - n_players % 4) % 4

    "Build param_info once for this utility family; reused across all pairs for agent() UBM calls."
    param_info_for_ubm = make_param_info(
        param_bds=param_bds, utility_settings=utility_settings_k,
        general_settings=general_settings, guess_seed=None, random_guesses_are_unique=True,
    )

    "Cap n_bins_per_dimension for the simulation UBM so the grid fits in memory."
    if enforce_memory_limit:
        n_param_dims   = len(param_info_for_ubm['keys'])
        n_bins_default = int(general_settings.get('n_bins_per_dimension', 7))
        max_grid_bytes = 400 * 1024 ** 2
        n_bins_for_ubm = n_bins_default
        while n_bins_for_ubm > 3 and (n_bins_for_ubm ** n_param_dims) * n_param_dims * 8 > max_grid_bytes:
            n_bins_for_ubm -= 1
        skip_ubm = (n_bins_for_ubm ** n_param_dims) * n_param_dims * 8 > max_grid_bytes
        if skip_ubm:
            general_settings_for_ubm = None
            print(f"[create_simulated_experiment] k={k_params}: grid too large even at n_bins=3 "
                  f"({(3 ** n_param_dims) * n_param_dims * 8 // 1024**2} MB) — skipping UBM.")
        elif n_bins_for_ubm != n_bins_default:
            general_settings_for_ubm = copy.deepcopy(general_settings)
            general_settings_for_ubm['n_bins_per_dimension'] = n_bins_for_ubm
            print(f"[create_simulated_experiment] k={k_params}: reduced n_bins_per_dimension "
                  f"{n_bins_default} → {n_bins_for_ubm} to keep simulation UBM grid under 400 MB.")
        else:
            general_settings_for_ubm = general_settings
    else:
        general_settings_for_ubm = general_settings

    dyad_list:           list[dict]      = []
    true_params_by_uuid: dict[str, dict] = {}

    for batch_start in range(0, n_players_padded, 4):

        "Assign ground-truth parameters to each of the 4 players in this batch."
        batch_params: dict[int, dict] = {}
        for local_idx in range(4):
            global_idx  = batch_start + local_idx
            player_uuid = f"synthetic_{k_params}_{global_idx:04d}"

            "Sample params uniformly from bounds; keep std away from zero."
            params: dict[str, float] = {}
            for param_idx, param_key in enumerate(list(param_info_for_ubm['keys'])):
                lower_bound, upper_bound = param_info_for_ubm['bounds'][param_idx]
                if param_key.endswith("_std"):
                    lower_bound = max(float(lower_bound), 1e-3)
                params[param_key] = float(random_gen.uniform(float(lower_bound), float(upper_bound)))
            if "τ" not in params:
                params["τ"] = float(general_settings.get('softmax_temperature', 1.0))

            "Override the altruism dimension with a target value so coverage spans the full range."
            params[altruism_key] = float(altruism_targets[global_idx % len(altruism_targets)])

            batch_params[local_idx]          = params
            true_params_by_uuid[player_uuid] = dict(params)

        "Sample game counts for all 6 pairs in this batch."
        pair_game_counts = _sample_batch_game_counts(random_gen=random_gen, n_games=n_games)

        "One dyad per unique pair; roles are assigned per round by a fair coin flip."
        for (local_idx_a, local_idx_b), n_games_pair in pair_game_counts.items():
            uuid_a   = f"synthetic_{k_params}_{(batch_start + local_idx_a):04d}"
            uuid_b   = f"synthetic_{k_params}_{(batch_start + local_idx_b):04d}"
            params_a = batch_params[local_idx_a]
            params_b = batch_params[local_idx_b]
            matching_probability = n_games_pair / n_games

            games_list = _simulate_pair_games(
                n_games=n_games_pair,
                params_player_1=params_a,
                params_player_2=params_b,
                uuid_player_1=uuid_a,
                uuid_player_2=uuid_b,
                utility_settings=utility_settings_k,
                general_settings=general_settings_for_ubm,
                param_info=param_info_for_ubm,
                per_round_role_flip=True,
                matching_probability=matching_probability,
                dynamic_predictor=general_settings_for_ubm is not None,
                random_gen=random_gen,
            )
            dyad_key = f"({games_list[0]['predictor']}, {games_list[0]['chooser']})"
            dyad_list.append({dyad_key: games_list})

    "Assemble histories JSON structure from the dyad list."
    histories: dict = {}
    for dyad_dict in dyad_list:
        histories.update(dyad_dict)

    avatar_shapes = [
        "arrow-head", "bowtie", "circle", "cross", "curvy-x", "dent-square", "dodecagon",
        "flame", "flower", "ghost", "hexagon", "hour-glass", "jagged-sun", "lemon", "moon",
        "pentagon", "round-square", "squash", "teardrop", "two-triangle", "star-six", "stop-sign",
    ]
    player_info: dict = {}
    for dyad_key_str in histories.keys():
        player_uuid_1, player_uuid_2 = dyad_key_str[1:-1].split(", ")
        for player_uuid in (player_uuid_1, player_uuid_2):
            if player_uuid not in player_info:
                info: dict = {
                    'player_type':  'synthetic',
                    'avatar_shape': avatar_shapes[int(random_gen.integers(0, len(avatar_shapes)))],
                    'player_color': (
                        f'hlsa({int(random_gen.integers(0, 360))}, '
                        f'{int(random_gen.integers(35, 66))}%, '
                        f'{int(random_gen.integers(35, 66))}%, 1.0)'
                    ),
                }
                if player_uuid in true_params_by_uuid:
                    player_true_params = true_params_by_uuid[player_uuid]
                    info['true_params'] = {
                        'chooser':   dict(player_true_params),
                        'predictor': dict(player_true_params),
                    }
                player_info[player_uuid] = info

    histories_dict = {'histories': histories, 'player_info': player_info}

    os.makedirs(os.path.dirname(histories_file_path), exist_ok=True)
    with open(histories_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(histories_dict, output_file, ensure_ascii=False, indent=4)

    "Invalidate the players_to_dyads cache — stale cache causes dyad lookup failures on re-run."
    players_to_dyads_cache_path = os.path.join(
        file_paths['processed'],
        file_paths['file_names']['players_to_dyads_exper3']
    )
    if os.path.exists(players_to_dyads_cache_path):
        os.remove(players_to_dyads_cache_path)

    return histories_dict, true_params_by_uuid


_SENTINEL = object()  # used as a sentinel for random_seed to distinguish "not provided" from None (unseeded)


def run_param_recovery_by_k(general_settings: GeneralSettings, file_paths: FilePaths, figure_layout: FigLay, param_bds: ParamBounds,
                            n_players: int | None = None, fit_predictor_role: bool = False, n_games: int | None = None,
                            k_params_range: tuple[int, int] | None = None, n_altruism_steps: int | None = None,
                            evenly_space_altruism: bool | None = None, utility_settings_by_k: dict[int, dict[str, bool]] | None = None,
                            correlate_all_params: bool | None = None,
                            random_seed=_SENTINEL, use_existing_fits: bool = False,
                            base_hue: int | None = None,
                            temperature_is_param: bool | None = None,
                            softmax_temperature: float | None = None,
                            enforce_memory_limit: bool = False) -> tuple[pd.DataFrame, dict[int, Any]]:
    """
    Run parameter-recovery simulations across utility dimensionalities (k) and summarize accuracy.

    For each k in `k_params_range`:
        1. Resolve a utility form from `utility_settings_by_k` if provided, otherwise reads
           the IC comparison CSV to find the best-fitting model at that k.
        2. Sample `n_players` synthetic player parameter vectors within `param_bds` for the
           utility terms active at k; each player fills both roles across dyads.
        3. Build a simulation histories JSON with embedded true parameters and run
           `run_analysis_bayes(...)` to recover fitted parameters.
        4. Read the per-player fit JSONs, construct a long DataFrame, and compute first-round
           correlations: corr( <param>_true_chooser, <param>_fitted_chooser ) (or predictor).
        5. Aggregate per-parameter correlations into a macro-average across parameters at k.
        6. Store the detailed dyad list and summary under `simulated_param_recovery_by_k[k]`.
        7. Save a tidy CSV and a Plotly figure (correlation vs. k).

    All keyword arguments that are `None` (or `_SENTINEL` for `random_seed`) cascade from
    `general_settings['parameter_recovery_settings']`, then fall back to the listed defaults.

    Arguments:
        • general_settings: GeneralSettings
            Master settings dict. `'fit_roles_together'` and `'warmstart_policy'` are
            overridden internally. `'parameter_recovery_settings'` provides defaults for all
            cascadable arguments below.
        • file_paths: FilePaths
            Project file-path dict; used for input CSVs and output directories.
        • figure_layout: FigLay
            Plotly figure layout template for the output figure.
        • param_bds: ParamBounds
            Parameter bounds dict used to sample random true parameter vectors.
        • n_players: int | None
            Number of synthetic players per k. Each player fills both chooser and predictor
            roles across dyads (round-robin design). None → cascade from
            parameter_recovery_settings['n_players'], then auto-count from
            len(all_player_uuids(file_paths, experiment_num=3, only_humans=True)).
        • fit_predictor_role: bool
            If True, also run the (expensive) Bayesian predictor fitting for each player.
            Default False — only chooser fitting is performed.
        • n_games: int | None
            Games per synthetic dyad. None → cascade, then default 60.
        • k_params_range: tuple[int, int] | None
            Inclusive (k_min, k_max) range. None → cascade, then (1, 9).
        • n_altruism_steps: int | None
            Grid points across the altruism range. None → cascade, then 7.
        • evenly_space_altruism: bool | None
            True → grid; False → uniform random. None → cascade, then True.
        • utility_settings_by_k: dict[int, dict[str, bool]] | None
            Explicit k → utility settings mapping; if None the IC CSV is used.
        • correlate_all_params: bool | None
            True → correlate every free mean parameter; False → Vᵢⱼ only.
            None → cascade, then False.
        • random_seed: int | None | _SENTINEL
            RNG seed. _SENTINEL (default) → cascade from settings, then None (unseeded).
            Pass None explicitly for an explicit unseeded run.
        • use_existing_fits: bool
            If True, skip re-running the analysis and load pre-existing fit JSONs.

    Returns:
        • tuple[pd.DataFrame, dict[int, Any]]
            - corr_by_k_df: tidy DataFrame with columns [k, role, param, corr, n_data, agg_corr].
            - simulated_param_recovery_by_k: nested dict keyed by k containing per-dyad
              details and aggregated summary statistics, split by role.
    """
    "---------- Cascade from parameter_recovery_settings ----------"
    prs = general_settings.get('parameter_recovery_settings', {})

    if n_players is None:
        n_players_from_settings = prs.get('n_players', None)
        if n_players_from_settings is None:
            n_players = len(prep.all_player_uuids(file_paths=file_paths, experiment_num=3, only_humans=True))
        else:
            n_players = int(n_players_from_settings)
    if not isinstance(fit_predictor_role, bool):
        fit_predictor_role = bool(prs.get('fit_predictor_role', False))
    if n_games is None:
        n_games_setting = prs.get('n_games', None)
        n_games = int(n_games_setting) if n_games_setting is not None else 60
    if k_params_range is None:
        k_params_range = prs.get('k_params_range', (1, 9))
    if n_altruism_steps is None:
        n_altruism_steps = int(prs.get('n_altruism_steps', 7))
    if evenly_space_altruism is None:
        evenly_space_altruism = bool(prs.get('evenly_space_altruism', True))
    if correlate_all_params is None:
        correlate_all_params = bool(prs.get('correlate_all_params', False))
    if temperature_is_param is None:
        temperature_is_param = bool(general_settings.get('temperature_is_param', False))
    if softmax_temperature is None:
        softmax_temperature = float(general_settings.get('softmax_temperature', 1.0))
    if random_seed is _SENTINEL:
        random_seed = prs.get('random_seed', None)

    if n_players == 0:
        raise ValueError("n_players must be > 0.")

    random_gen = np.random.default_rng(random_seed)
    k_min, k_max = int(k_params_range[0]), int(k_params_range[1])
    k_param_values = list(range(k_min, k_max + 1))
    time_start_total = time.time()

    general_settings = copy.deepcopy(general_settings)
    general_settings['fit_roles_together'] = False
    general_settings['warmstart_policy'] = {
        "enabled": False,
        "schedule": "binary",
        "cold_iters": 1e6,
        "explore_iters": 1e6,
        "temperature_high": 1000.0,
        "temperature_low": 0.01,
        "disable_dual_annealing_when_warm": True,
    }

    "---------- I/O setup ----------"
    out_dir = os.path.join(file_paths['simulations'], "param_recovery_by_k")
    os.makedirs(out_dir, exist_ok=True)
    out_csv_path  = os.path.join(out_dir, "param_recovery_by_k.csv")
    out_fig_path  = os.path.join(out_dir, "param_recovery_by_k.html")

    n_players_per_k = n_players + (4 - n_players % 4) % 4
    print(f"\n{'='*70}")
    print(f"[param_recovery_by_k] Starting.")
    print(f"  k range:       {k_min}–{k_max}  ({len(k_param_values)} values)")
    print(f"  players/k:     {n_players_per_k} (n_players={n_players} padded to multiple of 4)")
    print(f"  games/player:  {n_games}   fit predictor role: {fit_predictor_role}")
    print(f"  total fits:    {n_players_per_k * len(k_param_values)}")
    print(f"  random seed:   {random_seed}")
    print(f"{'='*70}\n")

    "---------- (A) Resolve utility_settings_by_k ----------"
    def _load_best_per_k_from_csv() -> dict[int, dict[str, bool]]:
        """
        Read the IC comparison CSV and return the best-fitting utility settings for each k.

        For each value of k in `k_param_values`, selects the row with `BIC_rank == 0` (or the row with
        the minimum BIC if `BIC_rank` is absent) and converts the canonical boolean setting
        columns into a `utility_settings` dict.

        Returns:
            • dict[int, dict[str, bool]] — mapping from k to the winning utility settings for
              that dimensionality, as read from the IC comparison CSV.
        """
        ic_comparison_csv_path = os.path.join(
            file_paths["bic_aic"],
            f"IC_Analysis_Comparison_Table_Experiment3.csv"
        )
        if not os.path.exists(ic_comparison_csv_path):
            raise FileNotFoundError(
                f"Could not find comparison CSV at {pretty_path(ic_comparison_csv_path)}. "
                f"Provide `utility_settings_by_k` explicitly or ensure the IC section wrote this file."
            )
        ic_comparison_dataframe = pd.read_csv(ic_comparison_csv_path, encoding="utf-8", engine="python")
        "Identify boolean setting columns: those that exist in utility_settings universe"
        "Assumes repo exposes the canonical set; else infer as non-numeric/non-meta:"
        "Keep the same feature list used by the IC analysis."
        setting_cols = [
            'conditional_welfare_mode','reference_dependent_altruism','min_max_rawlsian_leontief',
            'use_exponential_parameters','apply_exponents_to_payoffs','single_exponential_parameter',
            'single_payoffs_not_differences','payoff_ratios_not_differences','reference_dependent_utility',
            'use_negativity_parameters','negativity_social_comparison','fix_self_interest_parameter',
            'include_social_comparison','include_altruism_term'
        ]
        utility_settings_by_k = {}
        for k_params in k_param_values:
            k_subset_dataframe = ic_comparison_dataframe[ic_comparison_dataframe["k_params"] == k_params]
            if k_subset_dataframe.empty:
                continue
            "Winner inside this k (BIC_rank == 0). If missing, pick min BIC."
            if "BIC_rank" in k_subset_dataframe.columns and (k_subset_dataframe["BIC_rank"] == 0).any():
                best_row = k_subset_dataframe.loc[k_subset_dataframe["BIC_rank"] == 0].iloc[0]
            else:
                best_row = k_subset_dataframe.iloc[k_subset_dataframe["BIC"].argmin()]
            "Build settings dict"
            utility_settings_for_k = {col: bool(best_row[col]) for col in setting_cols if col in k_subset_dataframe.columns}

            utility_settings_by_k[k_params] = utility_settings_for_k
        if not utility_settings_by_k:
            raise RuntimeError("Failed to resolve any utility settings from the comparison CSV.")
        return utility_settings_by_k

    def _load_best_with_altruism_per_k() -> dict[int, dict[str, bool]]:
        """
        Read the IC comparison CSV and return the best altruism-containing utility settings for each k.

        For each value of k in `k_param_values`, filters to rows where `include_altruism_term == True` and
        selects the row with the lowest BIC as the winner.  Raises a RuntimeError if no
        altruism-containing model exists for any requested k.

        Returns:
            • dict[int, dict[str, bool]] — mapping from k to the best altruism-
                containing utility settings dict for that dimensionality.
        """
        ic_comparison_csv_path = os.path.join(file_paths["bic_aic"], file_paths["file_names"]["information_criterion"])
        if not os.path.exists(ic_comparison_csv_path):
            raise FileNotFoundError(
                f"Missing {pretty_path(ic_comparison_csv_path)}. Provide utility_settings_by_k or write the IC comparison CSV first."
            )
        ic_comparison_dataframe = pd.read_csv(ic_comparison_csv_path, encoding="utf-8", engine="python")
        if "include_altruism_term" not in ic_comparison_dataframe.columns:
            raise RuntimeError("The comparison CSV lacks 'include_altruism_term' column.")
        setting_cols = [
            'conditional_welfare_mode','reference_dependent_altruism','min_max_rawlsian_leontief',
            'use_exponential_parameters','apply_exponents_to_payoffs','single_exponential_parameter',
            'single_payoffs_not_differences','payoff_ratios_not_differences','reference_dependent_utility',
            'use_negativity_parameters','negativity_social_comparison','fix_self_interest_parameter',
            'include_social_comparison','include_altruism_term'
        ]
        utility_settings_by_k = {}
        for k_params in k_param_values:
            k_subset_with_altruism = ic_comparison_dataframe[
                (ic_comparison_dataframe["k_params"] == k_params) & (ic_comparison_dataframe["include_altruism_term"] == True)
            ]
            if k_subset_with_altruism.empty:
                raise RuntimeError(f"No altruism‑containing model found for k={k_params} in the comparison CSV.")

            best_row = k_subset_with_altruism.loc[k_subset_with_altruism["BIC"].idxmin()]
            utility_settings_by_k[k_params] = {col: bool(best_row[col]) for col in setting_cols if col in k_subset_with_altruism.columns}
        return utility_settings_by_k

    if utility_settings_by_k is None:
        if evenly_space_altruism:
            utility_settings_by_k = _load_best_with_altruism_per_k()
        else:
            utility_settings_by_k = _load_best_per_k_from_csv()

    "Validate each k's k_params matches requested k"
    for k_params in k_param_values:
        utility_settings_for_k = utility_settings_by_k.get(k_params)
        if not isinstance(utility_settings_for_k, dict):
            raise ValueError(f"Missing utility_settings for k={k_params}.")
        k_est = gnrl.count_free_parameters(utility_settings=utility_settings_for_k)
        if k_est != k_params:
            raise ValueError(f"Utility settings for k={k_params} imply k={k_est}. Please correct or override.")

    "---------- Helpers ----------"
    def _find_key(candidates: list[str], keys: list[str]) -> str | None:
        for name in candidates:
            if name in keys:
                return name
        return None

    "---------- Main loop over k: 3 phases for clean parallelism ----------"
    aggregate_records = []
    simulated_param_recovery_by_k: dict[int, Any] = {}
    k_phase1: dict[int, dict] = {}

    "Phase 1 (sequential): generate synthetic data for every k — all random_gen use happens here."
    for k_phase1_idx, k_params in enumerate(k_param_values, 1):
        t_phase1_k = time.time()
        u_settings_k = utility_settings_by_k[k_params]

        "Build param_info for this utility (also used in Phase 2 and Phase 3)."
        param_info_k = make_param_info(
            param_bds=param_bds, utility_settings=u_settings_k, guess_seed=None,
            general_settings=general_settings, random_guesses_are_unique=True,
        )

        keys_k   = list(param_info_k['keys'])
        bounds_k = dict(zip(keys_k, param_info_k['bounds']))

        "Resolve altruism key & bounds."
        altruism_key = _find_key(["Vᵢⱼ", "Vij"], keys_k)
        if altruism_key is None:
            raise RuntimeError(f"[k={k_params}] The chosen utility form lacks an identifiable altruism weight (Vᵢⱼ/Vij).")
        if altruism_key not in bounds_k:
            raise RuntimeError(f"[k={k_params}] Missing bounds for altruism key: {altruism_key}.")
        altruism_lower_bound, altruism_upper_bound = map(float, bounds_k[altruism_key])

        "Build file_paths_k early so we can check for an existing Phase 1 file."
        file_paths_k = copy.deepcopy(file_paths)
        file_paths_k = prep.add_remove_file_name_suffix(
            file_paths=file_paths_k, file_name_suffix=None, add_suffix=False
        )
        k_file_name = f"Social_Preference_Prediction_Pairs_Param_Recovery_k{k_params}_n{n_players}_g{n_games}.json"
        file_paths_k['file_names']['player_pairs_exper3'] = k_file_name
        "Override the players_to_dyads cache key so it regenerates from our custom histories instead of the real exper3 cache."
        file_paths_k['file_names']['players_to_dyads_exper3'] = f"players_to_dyads_param_recovery_k{k_params}_n{n_players}_g{n_games}.json"
        "Route all synthetic output to simulations/ — keeps fit JSONs, loss reports, param aggregates, and"
        "cached histories completely segregated from real participant data without touching any vital function."
        _sim_root = str(file_paths['simulations'])
        file_paths_k['player_fits'] = _sim_root
        file_paths_k['processed']   = os.path.join(_sim_root, 'processed')
        file_paths_k['param_data']  = os.path.join(_sim_root, 'param_data')
        histories_file_path = os.path.join(file_paths_k['processed'], k_file_name)

        "Build the altruism target list — always generated to keep random_gen state consistent."
        if evenly_space_altruism:
            steps = max(2, int(n_altruism_steps))
            altruism_grid = list(np.linspace(altruism_lower_bound, altruism_upper_bound, steps))
            reps = math.ceil(n_players_per_k / steps)
            altruism_targets = (altruism_grid * reps)[:n_players_per_k]
            random_gen.shuffle(altruism_targets)
        else:
            altruism_targets = [float(random_gen.uniform(altruism_lower_bound, altruism_upper_bound))
                                for _ in range(n_players_per_k)]

        histories_k, true_params_by_uuid_k = create_simulated_experiment(
            n_players=n_players_per_k,
            n_games=n_games,
            k_params=k_params,
            utility_settings_k=u_settings_k,
            general_settings=general_settings,
            param_bds=param_bds,
            random_gen=random_gen,
            altruism_key=altruism_key,
            altruism_targets=altruism_targets,
            file_paths=file_paths_k,
            create_new_file=False,
            enforce_memory_limit=enforce_memory_limit,
        )
        n_players_k = len(true_params_by_uuid_k)
        print(f"[k={k_params}  {k_phase1_idx}/{len(k_param_values)}] Phase 1 done in {_fmt_duration(time.time() - t_phase1_k)}.")

        "Build general_settings for this k's fit run."
        general_settings_k = copy.deepcopy(general_settings)
        general_settings_k['experiment_num'] = 3
        general_settings_k['write_mode'] = 'overwrite'
        general_settings_k['fit_predictor_role'] = fit_predictor_role
        k_phase1[k_params] = {
            'histories_file_path':  histories_file_path,
            'file_paths_k':         file_paths_k,
            'param_info_k':         param_info_k,
            'u_settings_k':         u_settings_k,
            'general_settings_k':   general_settings_k,
            'true_params_by_uuid':  true_params_by_uuid_k,
            'n_players_k':          n_players_k,
        }

    "Phase 2 (sequential): fit each k in turn."
    def _fit_one_k(meta: dict) -> None:
        "Load histories from disk rather than memory so only one k's data is live at a time."
        with open(meta['histories_file_path'], 'r', encoding='utf-8') as _f:
            histories_k = json.load(_f)
        run_analysis_bayes(
            histories_data=histories_k,
            file_paths=meta['file_paths_k'],
            param_info=meta['param_info_k'],
            utility_settings=meta['u_settings_k'],
            general_settings=meta['general_settings_k'],
            print_=True
        )

    if not use_existing_fits:
        k_fit_times: list[float] = []
        for k_phase2_idx, k_params in enumerate(k_param_values, 1):
            n_players_k_fitted = k_phase1[k_params]['n_players_k']
            print(f"\n[k={k_params}  {k_phase2_idx}/{len(k_param_values)}] Phase 2: Fitting {n_players_k_fitted} players...")
            t_k_fit = time.time()
            _fit_one_k(k_phase1[k_params])
            k_elapsed = time.time() - t_k_fit
            k_fit_times.append(k_elapsed)
            k_remaining = len(k_param_values) - k_phase2_idx
            eta = (sum(k_fit_times) / len(k_fit_times)) * k_remaining
            eta_str = f" — ETA: {_fmt_duration(eta)}" if k_remaining > 0 else ""
            print(f"[k={k_params}  {k_phase2_idx}/{len(k_param_values)}] Phase 2 done in {_fmt_duration(k_elapsed)}.{eta_str}")
    else:
        print("\n[Phase 2] Skipped — use_existing_fits=True.")

    "Phase 3 (sequential): collect results for every k."
    print(f"\n[Phase 3] Collecting results for {len(k_param_values)} k values...")
    fit_dir = os.path.join(str(file_paths['simulations']), "experiment_3")

    def _collect_role_results(k_params: int, role: str, param_info_k: dict) -> tuple[pd.DataFrame, list]:
        """
        Load fit JSONs for one role at one k, compute correlations, and return the tidy rows.

        Returns:
            • corr_only: DataFrame with columns [param, corr, n_data] for round="first".
            • dyad_entries: list of dicts summarising each fitted dyad.
        """
        uuid_prefix = f"synthetic_{k_params}_"
        json_files = [
            file_name for file_name in os.listdir(fit_dir)
            if file_name.endswith(".json") and uuid_prefix in file_name
        ]
        if not json_files:
            print(f"[k={k_params}][{role}] Warning: no fit files found matching '{uuid_prefix}' in {fit_dir}.")

        dfs = [load_simulated_fits_from_json(os.path.join(fit_dir, fn)) for fn in json_files]
        df_k = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        if correlate_all_params:
            temperature_keys = set() if temperature_is_param else {'τ', 'temp'}
            params_to_correlate = [
                param_key for param_key in param_info_k['keys']
                if '_std' not in param_key and '_cov' not in param_key and param_key not in temperature_keys
            ]
        else:
            params_to_correlate = ['Vᵢⱼ']

        fitted_suffix = f"_fitted_{role}"
        corr_df_k = compute_param_recovery_correlations(
            df=df_k,
            dir_path=out_dir,
            out_csv_name=f"correlations_k{k_params}_{role}.csv",
            true_role=role,
            fitted_suffix=fitted_suffix,
            round_mode="first",
            params=params_to_correlate,
            create_new_file=True,
        )
        corr_only = corr_df_k[corr_df_k["round"] == "first"][["param", "corr", "n_data"]].copy()

        "Build per-dyad summary entries."
        dyad_entries = []
        for file_name in json_files:
            path = os.path.join(fit_dir, file_name)
            with open(path, "r", encoding="utf-8") as file:
                dyad_dict = json.load(file)
            for dkey, games in dyad_dict.items():
                if not games:
                    continue
                first_game = games[0]
                true_pred = first_game.get("true_params_predictor", {})
                true_ch   = first_game.get("true_params_chooser",   {})
                estimates_by_method = first_game.get("parameter_estimates", {})
                fitted_params_for_role = {}
                target_uuid = first_game[role]
                for method_name in ("grid", "particle", "naive", "update", "globloc", "bayes", "general"):
                    if method_name in estimates_by_method and target_uuid in estimates_by_method[method_name]:
                        fitted_params_for_role = estimates_by_method[method_name][target_uuid].get(role, {}).get("params", {})
                        break
                dyad_entries.append({
                    "games": games,
                    "synthetic_params": {"chooser": true_ch, "predictor": true_pred},
                    "fitted_params": {role: fitted_params_for_role},
                })
        return corr_only, dyad_entries

    for k_phase3_idx, k_params in enumerate(k_param_values, 1):
        meta         = k_phase1[k_params]
        param_info_k = meta['param_info_k']
        role_results = {}

        roles_to_collect = ("chooser", "predictor") if fit_predictor_role else ("chooser",)
        for role in roles_to_collect:
            corr_only, dyad_entries = _collect_role_results(k_params, role, param_info_k)
            agg_corr = float(np.nanmean(corr_only["corr"])) if not corr_only.empty else np.nan

            for _, correlation_row in corr_only.iterrows():
                aggregate_records.append({
                    "k":        k_params,
                    "role":     role,
                    "param":    correlation_row["param"],
                    "corr":     float(correlation_row["corr"]),
                    "n_data":   int(correlation_row["n_data"]),
                    "agg_corr": float(agg_corr),
                })
            role_results[role] = {
                "dyads":                 dyad_entries,
                "correlation_by_param":  {row["param"]: float(row["corr"]) for _, row in corr_only.iterrows()},
                "aggregate_correlation": agg_corr,
            }

        simulated_param_recovery_by_k[k_params] = role_results

        corr_summary = "  ".join(
            f"{role} r={results['aggregate_correlation']:.3f}" for role, results in role_results.items()
        )
        print(f"[k={k_params}  {k_phase3_idx}/{len(k_param_values)}] Phase 3: {corr_summary or 'no results'}")

    "---------- Load BIC and equation per k from IC comparison CSV ----------"
    k_bic_map:      dict[int, float] = {}
    k_equation_map: dict[int, str]   = {}
    try:
        ic_csv_path = os.path.join(file_paths["bic_aic"], "IC_Analysis_Comparison_Table_Experiment3.csv")
        if os.path.exists(ic_csv_path):
            ic_meta_df = pd.read_csv(ic_csv_path, encoding="utf-8", engine="python")
            for k_lookup in k_param_values:
                k_rows = ic_meta_df[ic_meta_df["k_params"] == k_lookup]
                if k_rows.empty:
                    continue
                if "BIC_rank" in k_rows.columns and (k_rows["BIC_rank"] == 0).any():
                    winner = k_rows.loc[k_rows["BIC_rank"] == 0].iloc[0]
                else:
                    winner = k_rows.iloc[k_rows["BIC"].argmin()]
                k_bic_map[k_lookup]      = float(winner["BIC"])    if "BIC"      in winner.index else float("nan")
                k_equation_map[k_lookup] = str(winner["equation"]) if "equation" in winner.index else ""
    except Exception:
        pass

    "---------- Build & save tidy CSV ----------"
    corr_by_k_df = pd.DataFrame(aggregate_records)
    if not corr_by_k_df.empty:
        corr_by_k_df = corr_by_k_df.rename(columns={"role": "player_role"})
        corr_by_k_df["bic"]      = corr_by_k_df["k"].map(k_bic_map)
        corr_by_k_df["equation"] = corr_by_k_df["k"].map(k_equation_map)
        col_order    = ["k", "player_role", "n_data", "param", "corr", "agg_corr", "bic", "equation"]
        corr_by_k_df = corr_by_k_df[[c for c in col_order if c in corr_by_k_df.columns]]
        corr_by_k_df = corr_by_k_df.sort_values(["k", "player_role", "param"])
    try:
        corr_by_k_df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
        print(f"Saved summary CSV to: {pretty_path(out_csv_path)}")
    except (PermissionError, OSError):
        "Pass if I have the file open."
        pass

    "---------- Plotly figure: corr vs k, delegate to plot_param_recovery_by_k ----------"
    if not corr_by_k_df.empty:
        from visualization import plot_param_recovery_by_k as _plot_recovery_figure
        _plot_recovery_figure(
            corr_df=corr_by_k_df,
            figure_layout=figure_layout,
            k_equation_map=k_equation_map,
            evenly_space_altruism=evenly_space_altruism,
            base_hue=base_hue,
            out_fig_path=out_fig_path,
        )
        print(f"Saved Plotly figure to: {pretty_path(out_fig_path)}")

    print(f"\n{'='*70}")
    print(f"[param_recovery_by_k] Complete. Total time: {_fmt_duration(time.time() - time_start_total)}.")
    print(f"  CSV  → {pretty_path(out_csv_path)}")
    print(f"  Plot → {pretty_path(out_fig_path)}")
    print(f"{'='*70}\n")

    return corr_by_k_df, simulated_param_recovery_by_k


def verify_particle_filter_fidelity(general_settings: GeneralSettings, utility_settings: UtilitySettings, 
                                    param_info: ParamInfo, file_paths: FilePaths, figure_layout: FigLay, sample_ratios: int | list[float] = 5, 
                                    random_seed: int | None = None, n_predictors: int = 10, n_games_per_dyad: int = 10) -> pd.DataFrame:
    """
    Verifies that the particle filter (PF) reproduces the full grid-based posterior update (which occurs when the sample_ratio = 1.0).
    Runs a grid-based predictor over a dyad with a set of priors and provides those same priors to a PF-based predictor and compares 
    the posteriors of the grid-based and PF-based agent --> repeats for other sets of priors --> computes a correlation coefficient
    --> does this for variable sample ratios --> plots the correlation coefficient as a function of the sample ratio. 

    Arguments:
        • param_info: dict[str, list[Any]]; Stores parameter keys, boundaries, and initial guesses.
        • utility_settings: dict[str, bool]; Defines the functional form of the utility function.
        • file_paths: dict[str, str | dict[str, str]]; Stores all file paths for storing data.
        • general_settings: dict[str, Any]; Various settings used throughout this analysis.
        • figure_layout: dict[str: Any]; Establishes the settings for the Plotly figure layout.
        • sample_ratios: int | list[float]; n uniformly spaced sample ratios between 0 and 1 or an array of sample ratios. 
            - sample_ratio is a general setting that determins the number of probabilities that computed per Bayesian update. 
        • random_seed: int | None; Seed for reproducibility across dyads, parameter draws, and the single dyad structure.
        • n_predictors: int; Number of synthetic predictors (i.e., random priors) to evaluate per sample ratio.
        • n_games_per_dyad: int; Number of games per dyad.

    Returns:
        • pd.DataFrame: Simulation data   
    """
    "Generate an array of sample_ratios"
    if isinstance(sample_ratios, int) and sample_ratios > 1:
        sample_ratios = np.round(np.linspace(start=0, stop=1, num=sample_ratios), decimals=3).tolist()[:-1]
    elif isinstance(sample_ratios, list) and all((0 <= sratio <= 1) for sratio in sample_ratios):
        sample_ratios = sorted(sample_ratios)
        if sample_ratios[-1] == 1:
            sample_ratios = sample_ratios[:-1]
    else:
        raise ValueError(f"sample_ratios must be an integer or an array like"
                         f" [0.0, 0.25, 0.5, 0.75, 1.0], not {sample_ratios}!")

    if sample_ratios[0] == 0:
        sample_ratios = sample_ratios[1:]

    "Forcing general settings necessary for this simulation."
    general_settings_grid = copy.deepcopy(general_settings)
    general_settings_grid['include_covariance'] = False
    general_settings_grid['update_method'] = 'grid'
    general_settings_grid['sample_ratio'] = 1.0

    "Prepare containers for runtime tracking and progress feedback."
    baseline_durations: list[float] = []
    pf_durations_map: dict[str, list[float]] = {f"{sample_ratio:.3f}": [] for sample_ratio in sample_ratios}
    print_every = max(1, n_predictors // 10)  # Print every ~10% of predictors.

    "Generate prior parameters for choosers and predictors."
    params_choosers = [
        {
            param_key: param_val
            for param_key, param_val
            in zip(
                param_info['keys'],
                UniformGuesser(bounds=param_info['bounds'], seed=random_seed)()  
            )
        }
        for chooser in range(n_predictors)
    ]

    params_predictors = [
        {
            param_key: param_val
            for param_key, param_val
            in zip(
                param_info['keys'],
                UniformGuesser(bounds=param_info['bounds'], seed=random_seed)()
            )
        }
        for predictor in range(n_predictors)
    ]

    "Generate dyads"
    dyads = {}
    for pred_idx, (params_choo, params_pred) in enumerate(zip(params_choosers, params_predictors)):
        dyad_games = []
        for game_idx in range(n_games_per_dyad):

            payoff_A_chooser =   random.randint(1, 5)
            payoff_A_predictor = random.randint(1, 5)
            payoff_B_chooser =   random.randint(1, 5)
            payoff_B_predictor = random.randint(1, 5)
            payoffs = {
                'payoff_A_chooser': payoff_A_chooser, 'payoff_A_predictor': payoff_A_predictor,
                'payoff_B_chooser': payoff_B_chooser, 'payoff_B_predictor': payoff_B_predictor,
            }

            "Store choices based on the chooser parameters"
            choice_response = response(current_game=payoffs, agent_params=params_choo, utility_settings=utility_settings,
                                      softmax_temperature=general_settings.get('softmax_temperature'), select_responses=True)["model_choose_A"]
            choice_response = "A" if choice_response == 1 else "B"

            dyad_game = {
                "chooser": f"C{pred_idx}",
                "predictor": f"P{pred_idx}",
                "matching_probability": 1.0,
                "payoff_A_chooser": payoff_A_chooser,
                "payoff_A_predictor": payoff_A_predictor,
                "payoff_B_chooser": payoff_B_chooser,
                "payoff_B_predictor": payoff_B_predictor,
                "choice": choice_response,
                "prediction": None,
                "abdicated_chooser": False,
                "abdicated_predictor": False,
                "timestamp": time.time(),
                "round": game_idx,            
            }
            dyad_games.append(dyad_game)

        "Generate and store parameter estimates for the full grid-based predictor"
        t_grid_start = time.perf_counter()
        dyad_games = agent(dyad_games=dyad_games, game_idx_start=0, game_idx_stop=n_games_per_dyad, player_role='predictor', 
                           initial_params={'predictor': params_pred}, param_info=param_info, utility_settings=utility_settings,
                           player_uuid=f"P{pred_idx}", general_settings=general_settings_grid)
        t_grid_stop = time.perf_counter()
        baseline_durations.append(t_grid_stop - t_grid_start)

        for game in dyad_games: 
            "Submitting predictions for aestetic reasons. Not necessary for analysis."
            model_predict_A = game.get('parameter_estimates', {}).get('grid', {}).get(
                f'P{pred_idx}', {}).get('predictor', {}).get('output', {}).get('model_predict_A')
            if isinstance(model_predict_A, float):
                game['prediction'] = "A" if random.random() > 0.5 else "B"

            "Store parameter estimates for the full grid in a different spot so that the PF estimates can be stored in the same game."
            game['parameter_estimates_full_grid'] = game.pop('parameter_estimates')

        "Generate and store parameter estimates for PF-based predictors at varying sample ratios"
        for sample_ratio in sample_ratios:
            sample_ratio_key = f'{sample_ratio:.3f}'
            general_settings_pf = copy.deepcopy(general_settings_grid)
            general_settings_pf['sample_ratio'] = sample_ratio

            t_pf_start = time.perf_counter()
            dyad_games = agent(dyad_games=dyad_games, game_idx_start=0, game_idx_stop=n_games_per_dyad, player_role='predictor', 
                            initial_params={'predictor': params_pred}, param_info=param_info, utility_settings=utility_settings,
                            player_uuid=f"P{pred_idx}", general_settings=general_settings_pf)            
            t_pf_stop = time.perf_counter()
            pf_durations_map[sample_ratio_key].append(t_pf_stop - t_pf_start)

            for game in dyad_games:
                "Rename parameter estimates based on the sample ratio value"
                game[f'parameter_estimates_pf_{sample_ratio:.3f}'] = game.pop('parameter_estimates')

        "Progress: print a heartbeat every few predictors"
        if (pred_idx + 1) % print_every == 0:
            print(f"[verify_pf] finished predictor {pred_idx + 1}/{n_predictors}")

        dyads[pred_idx] = dyad_games

    "Condense information into a dictionary mapping sample ratios to grid versus pf posteriors per player."
    "{sample_ratio: {param_key: 'grid': [posterior_P0, posterior_P1,...], 'pf': [posterior_P0, posterior_P1,...]}}"
    sample_ratios_to_posterior_pairs: dict[str: dict[str: dict[str: list[float]]]] = {}

    for sample_ratio in sample_ratios:
        sample_ratio_key = f'{sample_ratio:.3f}'
        sample_ratios_to_posterior_pairs[sample_ratio_key] = {}
        for param_key in param_info['keys']:
            sample_ratios_to_posterior_pairs[sample_ratio_key][param_key] = {'grid': [], 'pf': []}
            for pred_idx, dyad_games in dyads.items():
                final_game = dyad_games[-1]
                posteriors_grid = final_game.get(f'parameter_estimates_full_grid', {}).get('grid', {}).get(
                    f'P{pred_idx}', {}).get('predictor', {}).get('posteriors', {}).get(param_key)
                posteriors_pf = final_game.get(f'parameter_estimates_pf_{sample_ratio_key}', {}).get('grid', {}).get(
                    f'P{pred_idx}', {}).get('predictor', {}).get('posteriors', {}).get(param_key)
                sample_ratios_to_posterior_pairs[sample_ratio_key][param_key]['grid'].append(posteriors_grid)
                sample_ratios_to_posterior_pairs[sample_ratio_key][param_key]['pf'].append(posteriors_pf)

    def _safe_corr(x_values: list[float], y_values: list[float]) -> float:
        """
        Compute a robust correlation for two posterior-summary series.

        Degenerate constant-vs-constant inputs are treated as a perfect match,
        while one-sided constants return zero association instead of NaN.
        """
        x_array = np.asarray(x_values, dtype=float)
        y_array = np.asarray(y_values, dtype=float)
        if len(x_array) < 2 or len(y_array) < 2:
            return float('nan')
        x_std = float(np.std(x_array))
        y_std = float(np.std(y_array))
        if x_std == 0.0 and y_std == 0.0:
            "Identical constants across predictors ⇒ treat as perfect match."
            return 1.0
        if x_std == 0.0 or y_std == 0.0:
            "One side constant, the other not ⇒ no linear association."
            return 0.0
        corr = float(np.corrcoef(x_array, y_array)[0, 1])
        "Clip to [-1, 1] and report negative correlations."
        if corr < 0:
            print(f"[verify_pf] WARNING: negative correlation ({corr:.3f})")
        return float(np.clip(corr, -1.0, 1.0))

    "Recompute correlations with the robust helper (per param, then mean excluding _std)."
    sample_ratios_to_param_correlations: dict[str, dict[str, float]] = {}
    sample_ratios_to_mean_correlations: dict[str, float] = {}

    for sample_ratio_key, param_map in sample_ratios_to_posterior_pairs.items():
        sample_ratios_to_param_correlations[sample_ratio_key] = {}
        per_param_corrs: list[float] = []
        for param_key, grid_pf_lists in param_map.items():
            corr_val = _safe_corr(grid_pf_lists['grid'], grid_pf_lists['pf'])
            sample_ratios_to_param_correlations[sample_ratio_key][param_key] = corr_val
            if '_std' not in param_key and (corr_val == corr_val):  # Skip NaNs.
                per_param_corrs.append(corr_val)
        sample_ratios_to_mean_correlations[sample_ratio_key] = float(np.mean(per_param_corrs)) if per_param_corrs else float('nan')

    "Build and write the summary CSV (one row per sample_ratio)."
    summary_rows: list[dict[str, Any]] = []
    for sr_key in sorted(sample_ratios_to_mean_correlations.keys(), key=lambda sample_ratio_key: float(sample_ratio_key)):
        pf_times = pf_durations_map.get(sr_key, [])
        row = {
            "sample_ratio": float(sr_key),
            "corr_overall_excluding_std": float(sample_ratios_to_mean_correlations[sr_key]),
            "pf_time_mean_s": float(np.mean(pf_times)) if pf_times else float('nan'),
            "pf_time_std_s": float(np.std(pf_times, ddof=1)) if len(pf_times) > 1 else float('nan'),
            "grid_time_mean_s": float(np.mean(baseline_durations)) if baseline_durations else float('nan'),
            "grid_time_std_s": float(np.std(baseline_durations, ddof=1)) if len(baseline_durations) > 1 else float('nan'),
            "n_predictors": int(n_predictors),
            "n_games_per_dyad": int(n_games_per_dyad),
        }
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values("sample_ratio")

    k_params = len([
        param_key for param_key in param_info['keys'] 
        if '_std' not in param_key and '_cov' not in param_key
    ])

    "Build and write the per-parameter correlation CSV."
    param_rows: list[dict[str, Any]] = []
    for sr_key, param_corrs in sample_ratios_to_param_correlations.items():
        for param_key, corr_val in param_corrs.items():
            param_rows.append({
                "sample_ratio": float(sr_key),
                "param_key": param_key,
                "correlation": float(corr_val),
                "n_predictors": int(n_predictors),
                "n_games_per_dyad": int(n_games_per_dyad),
                "k_params": k_params
            })
    correlations_df = pd.DataFrame(param_rows).sort_values(["sample_ratio", "param_key"])

    "Compose file names based on inputs."
    file_stub = f"verify_particle_filter_fidelity_{len(sample_ratios):03d}-{n_predictors}-{n_games_per_dyad}-{k_params}"
    file_stub += prep.create_file_name_suffix(general_settings=general_settings, utility_settings=utility_settings)
    out_dir = file_paths["visuals"]
    os.makedirs(out_dir, exist_ok=True)
    summary_csv_path = os.path.join(out_dir, f"{file_stub}.csv")
    per_param_csv_path = os.path.join(out_dir, f"{file_stub}_by_param.csv")

    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    correlations_df.to_csv(per_param_csv_path, index=False, encoding="utf-8-sig")

    "----- Build the Plotly figure (square, anchored axes, [0,1] ranges) -----"
    "X,y data with explicit anchors at (0,0) and (1,1)"
    sample_ratios_sorted = [float(sample_ratio_key) for sample_ratio_key in sorted(sample_ratios_to_mean_correlations.keys(), key=lambda sample_ratio_key: float(sample_ratio_key))]
    corr_sorted = [float(sample_ratios_to_mean_correlations[f"{sample_ratio:.3f}"]) for sample_ratio in sample_ratios_sorted]

    fig = go.Figure()

    "Main series"
    fig.add_trace(go.Scatter(
        x=sample_ratios_sorted,
        y=corr_sorted,
        mode="markers+lines",
        name="PF vs Grid: correlation",
        hovertemplate="sample_ratio=%{x:.3f}<br>corr=%{y:.3f}<extra></extra>",
        marker=dict(size=figure_layout.get("markersize", 12), opacity=0.9)
    ))

    "Invisible anchors to force [0,1]×[0,1] domain"
    fig.add_trace(go.Scatter(
        x=[0.0, 1.0], y=[0.0, 1.0],
        mode="markers",
        marker=dict(size=1, opacity=0.0),
        showlegend=False,
        hoverinfo="skip",
        name="anchors"
    ))

    title = "Particle Filter Fidelity vs. Grid"

    "Annotation with fancy 'n' and spaces around equals"
    "Pull n_bins_per_dimension from general settings."
    n_bins = int(general_settings.get("n_bins_per_dimension", 0))
    annot_text = f"𝑛 bins = {n_bins} • 𝑛 games = {n_games_per_dyad} • 𝑛 players = {n_predictors}"

    font_base = figure_layout["annotations"]["font"].copy()
    font_base.pop("size", None)
    font_small = dict(**font_base, size=figure_layout["annotations"]["font"]["size"] - 4)

    fig.update_layout(
        title=title,
        titlefont_size=figure_layout['titlefont_size'],
        title_x=figure_layout['title_x'],
        title_y=figure_layout['title_y'],
        template=figure_layout['template'],
        hoverlabel=figure_layout['hoverlabel'],
        margin=dict(l=560, r=560, t=120, b=200),
        font=figure_layout['font'], showlegend=False,
        annotations=[
            dict(
                text=annot_text, font=font_small, xref="paper", yref="paper",
                showarrow=figure_layout["annotations"]["showarrow"], x=0.5, y=-0.05, 
            ),
            dict(
                text=build_utility_equation(utility_settings=utility_settings), font=font_small,
                showarrow=figure_layout["annotations"]["showarrow"], x=0.5, y=-0.15, xref="paper", yref="paper"
            )            
        ]
    )

    "Axes: pass axis styling dicts into update_xaxes / update_yaxes"
    fig.update_xaxes(
        title="Sample Ratio (fraction of grid evaluated)",
        range=[0, 1],
        **figure_layout.get("xaxis", {})
    )
    fig.update_yaxes(
        title="Correlation (PF posterior vs Grid posterior)",
        range=[0, 1],
        scaleanchor="x",  # Square plot: y anchored to x.
        scaleratio=1,
        **figure_layout.get("yaxis", {})
    )

    "Save the HTML"
    html_path = os.path.join(out_dir, f"{file_stub}.html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"[verify_pf] Wrote Plotly HTML:     {pretty_path(html_path)}")

    "Console report: quick glance at the summary."
    print("\n[verify_pf] Correlation & runtime summary (by sample_ratio):")
    with pd.option_context('display.float_format', lambda display_value: f"{display_value:.3f}"):
        print(summary_df.to_string(index=False))

    print(f"\n[verify_pf] Wrote summary CSV:     {pretty_path(summary_csv_path)}")
    print(f"[verify_pf] Wrote per-param CSV:   {pretty_path(per_param_csv_path)}")
    print(f"[verify_pf] Wrote Plotly HTML:     {pretty_path(html_path)}")

    return summary_df


"=========================================================================================="
"======= Simulation 2) Predictor Estimates Converge to the Chooser's True Altruism ========"
"=========================================================================================="

def plot_param_recovery_by_round(
        df_merged: pd.DataFrame,
        general_settings: GeneralSettings,
        file_paths: FilePaths,
        params=None,
        figure_layout: dict = None,
        export_fig: bool = True,
        create_new_file: bool = True,
        file_name: str = "corr_by_round.html",
        corr_csv_name: str = "correlation_results_by_round.csv",
        fitted_suffix: str = "_fitted_predictor",
        fit_mode: str = "poly",      # "poly" or "line"
        poly_degree: int = 3
    ) -> None:
    """
    Plot how parameter-recovery correlations evolve across rounds.

    Used for figure: Incremental alignment between inferred and true altruism across rounds

    This function:
        1. Calls `compute_param_recovery_correlations(..., round_mode="all")`.
        2. For each parameter, plots:
             • correlation vs. round, and
             • a best-fit line with slope and R² annotation.
        3. Uses a dropdown to toggle which parameter's traces are visible.

    Arguments:
        • df_merged: pd.DataFrame;
            Long-format DataFrame with true_* and fitted_predictor columns.
        • general_settings: GeneralSettings;
            Used mainly for display options (e.g., dark_mode).
        • file_paths: dict[str: str | dict[str: str]];
            Dictionary of all file paths in this project.
        • params: list[str] | None;
            Parameters to plot (e.g., ["Vii","Vij","Vii_std","Vij_std","τ"]).
            If None, defaults to that list.
        • figure_layout: dict | None;
            Layout settings (font, template, axis options) for Plotly.
        • export_fig: bool;
            If True, write the Plotly HTML to `dir_path/file_name`. If False, show it.
        • create_new_file: bool;
            If True, force recomputation of the correlation CSV.
        • file_name: str;
            Name of the .html file (".html" added if missing).

    Returns:
        • None;
            Writes or shows an interactive Plotly figure with correlation trajectories.
    """

    def canonicalize_param_name(param: str, available: list[str]) -> str:
        """
        Return the available column name matching either unicode or ASCII parameter spelling.

        Falls back to the original parameter name when neither spelling appears in
        `available`, allowing the caller to surface the missing-column error.
        """
        if param in available:
            return param

        candidates = [
            param.replace("Vii", "Vᵢᵢ").replace("Vij", "Vᵢⱼ"),
            param.replace("Vᵢᵢ", "Vii").replace("Vᵢⱼ", "Vij"),
        ]
        for cand in candidates:
            if cand in available:
                return cand
        return param

    def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Generic R²: 1 - SS_res / SS_tot, works for any curve (not just lines).
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.size != y_pred.size or y_true.size < 2:
            return math.nan
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        if abs(ss_tot) < 1e-12:
            return math.nan
        ss_res = np.sum((y_true - y_pred) ** 2)
        return 1.0 - (ss_res / ss_tot)

    if params is None:
        params = ["Vij", "Vii", "Vij_std", "Vii_std", "τ"]
    if figure_layout is None:
        figure_layout = {}

    dir_path = str(file_paths['simulations'])
    os.makedirs(dir_path, exist_ok=True)

    param_titles = {
        "Vii":     "Mean Self-interest μ(𝑉𝑖𝑖)", 
        "Vij":     "Mean Altruism μ(𝑉𝑖𝑗)", 
        "Vii_std": "Self-interest Standard Deviation σ(𝑉𝑖𝑖)", 
        "Vij_std": "Altruism Standard Deviation σ(𝑉𝑖𝑗)", 
        "τ":       "SoftMax Temperature (τ)"
    }

    param_containers = {
        "Vii":     "μ(𝑉𝑖𝑖)",
        "Vij":     "μ(𝑉𝑖𝑗)",
        "Vii_std": "σ(𝑉𝑖𝑖)",
        "Vij_std": "σ(𝑉𝑖𝑗)",
        "τ":       "(τ)"
    }

    "1) Correlations by round"
    corr_df = compute_param_recovery_correlations(
        df=df_merged,
        dir_path=dir_path,
        out_csv_name=corr_csv_name,
        fitted_suffix=fitted_suffix,
        create_new_file=create_new_file,
        true_role="chooser",
        round_mode="all"
    )

    corr_df = corr_df.copy()
    corr_df["round"] = pd.to_numeric(corr_df["round"], errors="coerce")
    corr_df = corr_df.dropna(subset=["round", "corr"])

    available_params = corr_df["param"].unique().tolist()
    param_mapping = {param_name: canonicalize_param_name(param_name, available_params) for param_name in params}
    corr_df = corr_df[corr_df["param"].isin(param_mapping.values())]

    fig = go.Figure()
    annotation_base = dict(
        x=0.05, y=0.95, xref='paper', yref='paper',
        showarrow=False, align="left",
        font=dict(size=28)
    )
    param_buttons = []
    n_params = len(params)

    "Stores R² and final correlation per param for the initial annotation."
    fit_stats = {}

    for idx, param_name in enumerate(params):
        param_use = param_mapping[param_name]
        df_sub = corr_df[corr_df["param"] == param_use].copy()
        df_sub = df_sub.sort_values("round")

        xvals = df_sub["round"].values
        yvals = df_sub["corr"].values
        n_data_vals = df_sub.get("n_data", pd.Series(index=df_sub.index, dtype=str)).fillna("").astype(str).values

        "Data trace: empirical correlations"
        data_trace = go.Scatter(
            name=f"Correlation: {param_containers[param_name]}",
            x=xvals, y=yvals,
            mode="lines+markers",
            line=dict(color='hsla(115, 70%, 30%, 1.0)', width=10),
            marker=dict(size=18, color='hsla(115, 70%, 40%, 1.0)', opacity=1.0),
            visible=True if idx == 0 else False,
            hovertemplate=(
                "Round = %{x}<br>"
                f"Corr = %{{y:.3f}}<br>n_data = %{{text}}<extra>{param_name}</extra>"
            ),
            text=n_data_vals
        )
        fig.add_trace(data_trace)

        "=== Best-fit curve (poly or line) ==="
        if len(xvals) >= 2:
            if fit_mode == "line":
                deg = 1
            else:  # "poly"
                deg = max(2, int(poly_degree))

            "Polyfit needs at least deg+1 points; if not, fall back to lower degree"
            deg = min(deg, len(xvals) - 1)
            coef = np.polyfit(xvals, yvals, deg)

            x_fit = np.linspace(xvals.min(), xvals.max(), 200)
            y_fit = np.polyval(coef, x_fit)
            y_pred = np.polyval(coef, xvals)
            r2_val = compute_r2(yvals, y_pred)
        else:
            x_fit = np.array([])
            y_fit = np.array([])
            r2_val = math.nan

        fit_stats[param_name] = (r2_val, yvals[-1] if len(yvals) else math.nan)

        fit_label = (
            f"Linear fit: {param_containers.get(param_name, param_name)}"
            if fit_mode == "line"
            else f"Poly (deg {poly_degree}) fit: {param_containers.get(param_name, param_name)}"
        )

        fit_trace = go.Scatter(
            name=fit_label,
            x=x_fit, y=y_fit,
            mode="lines",
            hoverinfo="skip",
            line=dict(dash='dot', width=10, color='hsla(160, 70%, 40%, 1.0)'),
            visible=True if idx == 0 else False,
        )
        fig.add_trace(fit_trace)

        "Dropdown visibility + annotation text for this param"
        visible_list = [False] * (2 * n_params)
        visible_list[2 * idx] = True      # Data.
        visible_list[2 * idx + 1] = True  # Fit.

        if len(xvals) >= 2 and not math.isnan(r2_val):
            annotation_text = (
                f"Final corr = {yvals[-1]:.3f}, "
                f"R² ({fit_mode}) = {r2_val:.3f}"
            )
        else:
            annotation_text = "Insufficient data"

        param_buttons.append(dict(
            label=param_containers[param_name],
            method="update",
            args=[
                {"visible": visible_list},
                {
                    "title": (
                        "Correlation Between Fitted Predictor Parameters and "
                        f"True Chooser Parameters by Round for {param_titles[param_name]}"
                    ),
                    "annotations": [dict(text=annotation_text, **annotation_base)]
                }
            ]
        ))

    "Default annotation for the first param"
    first_param = params[0]
    r2_first, corr_first = fit_stats.get(first_param, (math.nan, math.nan))
    if not math.isnan(r2_first) and not math.isnan(corr_first):
        ann_text = f"Final corr = {corr_first:.3f}, R² ({fit_mode}) = {r2_first:.3f}"
    else:
        ann_text = "Insufficient data"

    dark_mode = general_settings.get("dark_mode", True)
    max_round = corr_df["round"].max() if not corr_df.empty else 0

    fig.update_layout(
        title="Correlation Between Fitted Predictor Parameters And True Chooser Parameters by Round",
        title_x=figure_layout['title_x'], title_y=figure_layout['title_y'],
        titlefont_size=figure_layout['titlefont_size'] - 15,
        xaxis=dict(
            title="Round",
            range=[-0.5, max_round + 0.5],
            **figure_layout.get("xaxis", {})
        ),
        yaxis=dict(
            title="Correlation",
            range=[-1, 1],
            **figure_layout.get("yaxis", {})
        ),
        template=figure_layout.get("template", "plotly_dark"),
        hoverlabel=figure_layout.get("hoverlabel", {}),
        margin=dict(l=150, r=120, t=120, b=120),
        font=dict(
            family="Calibri",
            color="white" if dark_mode else "black",
            size=32
        ),
        updatemenus=[dict(
            type="dropdown", showactive=True,
            buttons=param_buttons, x=0.88, y=0.2
        )],
        annotations=[dict(
            text=ann_text,
            **annotation_base
        )],
        legend={"x": 0.81, "y": 0.35},
    )

    if not file_name.endswith(".html"):
        file_name += ".html"
    out_path = os.path.join(dir_path, file_name)

    if export_fig:
        fig.write_html(out_path)
        print(f"Correlation-by-round figure saved to {pretty_path(out_path)}")
    else:
        fig.show()

    return fig


def compute_prediction_accuracy_by_segment(file_paths: Dict[str, Dict[str, str] | str], general_settings: Dict[str, Any], utility_settings: Dict[str, bool], n_segments: int = 2) -> Dict[str, Any]:
    """
    Compute how participants' prediction accuracy changes across repeated meetings.

    This function is for the human–bot experiment (Experiment 1). For each predictor:
        1) Reconstructs their dyads and game histories.
        2) Segments repeated encounters with each chooser/avatar into `n_segments`
           (early / middle / late).
        3) Computes accuracy of predictions within each segment:
               accuracy = proportion(choice == prediction).

    In Experiment 2, where choosers are avatars with known preferences, it uses the
    known avatar utility function to reconstruct the avatar's “true” choice on each
    response-phase round before comparing with human predictions.

    Arguments:
        • file_paths: dict[str, dict[str,str] | str];
            Standard file_paths structure for the Iter_Binary_Dictator pipeline.
        • general_settings: dict[str, Any];
            Must contain 'experiment_num' to choose the appropriate loading logic.
        • utility_settings: dict[str, bool];
            Utility settings used to compute avatar choices in Experiment 2.
        • n_segments: int;
            Number of segments to divide each repeated interaction into
            (e.g., 2 = early vs late; 3 = early / middle / late).

    Returns:
        • dict[str, Any];
            Contains per-segment accuracy summaries. Currently prints and returns
            `accuracy_by_segment` and `accuracy_by_segment_player`.
    """
    avatar_params = {
        'utilitarian': {'Vᵢᵢ':  1.0, 'Vᵢⱼ':  1.0}, 
        'selfish':     {'Vᵢᵢ':  1.0, 'Vᵢⱼ':  0.0}, 
        'competitive': {'Vᵢᵢ':  1.0, 'Vᵢⱼ': -1.0}, 
        'masochistic': {'Vᵢᵢ': -1.0, 'Vᵢⱼ':  1.0}
    }

    experiment_num = general_settings.get('experiment_num')
    plrs_to_dyads = prep.players_to_dyads(
        experiment_num=experiment_num, file_paths=file_paths, create_new_file=False)
    player_uuids = sorted(list(plrs_to_dyads.keys()))

    dir_path = file_paths['processed']
    file_path = f"Social_Preference_Prediction_Pairs_Exper{experiment_num}.json"
    full_path = os.path.join(dir_path, file_path)

    player_histories = None
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as file:
            player_histories = json.load(file)
        if isinstance(player_histories, dict):
            player_histories = player_histories.get('histories')

    if player_histories is None:
        raise Exception(f"Failed to extract player histories")

    players_to_dyads = {}
    for player_uuid, dyads in plrs_to_dyads.items():
        if player_uuid not in players_to_dyads:
            players_to_dyads[player_uuid] = {}
        for dyad_key in dyads:
            if dyad_key in player_histories:
                players_to_dyads[player_uuid][dyad_key] = player_histories[dyad_key]

    prediction_accuracy_data = {}
    for player_uuid, player_dyads in players_to_dyads.items():
        print(player_uuid)
        meeting_indices = {}
        prediction_accuracy_data[player_uuid] = []
        for dyad_games in player_dyads.values():
            n_games_in_dyad = int(len(dyad_games) / 2) if experiment_num == 2 else len(dyad_games)
            segment_size = n_games_in_dyad // n_segments
            for dyad_game in dyad_games:
                choice = dyad_game.get('choice')
                chooser = dyad_game.get('chooser')
                prediction = dyad_game.get('prediction')
                predictor = dyad_game.get('predictor')
                round_num = dyad_game.get('round')
                if player_uuid != predictor:
                    continue
                if experiment_num == 2:
                    "In experiment 2, use what the avatar would have chosen in the response phase."
                    if dyad_game['phase'] == 'op':
                        continue
                    chooser = chooser.split("-")[1]
                    payoff_A_chooser = dyad_game.get('payoff_A_chooser', 0)
                    payoff_A_predictor = dyad_game.get('payoff_A_predictor', 0)
                    payoff_B_chooser = dyad_game.get('payoff_B_chooser', 0)
                    payoff_B_predictor = dyad_game.get('payoff_B_predictor', 0)
                    Vᵢᵢ = avatar_params[chooser]['Vᵢᵢ']
                    Vᵢⱼ = avatar_params[chooser]['Vᵢⱼ']
                    payoffsA = {'As': payoff_A_chooser, 'Ao': payoff_A_predictor, 'Bs': payoff_B_chooser, 'Bo': payoff_B_predictor}
                    payoffsB = {'As': payoff_B_chooser, 'Ao': payoff_B_predictor, 'Bs': payoff_A_chooser, 'Bo': payoff_A_predictor}
                    params = {'Vᵢᵢ': Vᵢᵢ, 'Vᵢⱼ': Vᵢⱼ}
                    utilityA = utility(payoffs=payoffsA, params=params, utility_settings=utility_settings)
                    utilityB = utility(payoffs=payoffsB, params=params, utility_settings=utility_settings)
                    if utilityA > utilityB:
                        choice = 'A'
                    elif utilityA < utilityB:
                        choice = 'B'
                    else:
                        choice = random.choice(seq=('A', 'B'))

                if chooser not in meeting_indices:
                    meeting_indices[chooser] = 0
                meeting_index = int(meeting_indices[chooser])
                meeting_indices[chooser] += 1

                if segment_size == 0:
                    meeting_segment = 0
                else:
                    meeting_segment = meeting_index // segment_size

                prediction_is_accurate = int(choice == prediction)
                prediction_accuracy_data[player_uuid].append(
                    (predictor, chooser, round_num, meeting_index, meeting_segment, prediction_is_accurate)
                )

    accuracy_by_segment = {}
    accuracy_by_segment_player = {}
    for player_uuid, accuracy_data in prediction_accuracy_data.items():
        accuracy_by_segment_player[player_uuid] = {}
        for accuracy_tuple in accuracy_data:
            player_uuid, counterpart_uuid, round_num, meeting_index, meeting_segment, accurate = accuracy_tuple
            if meeting_segment not in accuracy_by_segment:
                accuracy_by_segment[meeting_segment] = {'accurate': 0, 'inaccurate': 0, 'total': 0}
            if meeting_segment not in accuracy_by_segment_player[player_uuid]:
                accuracy_by_segment_player[player_uuid][meeting_segment] = {'accurate': 0, 'inaccurate': 0, 'total': 0}
            if accurate:
                accuracy_by_segment_player[player_uuid][meeting_segment]['accurate'] += 1
                accuracy_by_segment[meeting_segment]['accurate'] += 1
            else:
                accuracy_by_segment_player[player_uuid][meeting_segment]['inaccurate'] += 1
                accuracy_by_segment[meeting_segment]['inaccurate'] += 1
            accuracy_by_segment_player[player_uuid][meeting_segment]['total'] += 1
            accuracy_by_segment[meeting_segment]['total'] += 1

    for segment_num, scores in accuracy_by_segment.items():
        accurate, total = scores['accurate'], scores['total']
        scores['ratio'] = round(accurate / total, 6)

    pp.pprint(accuracy_by_segment)
    return {
        "accuracy_by_segment": accuracy_by_segment,
        "accuracy_by_segment_player": accuracy_by_segment_player
    } 


"=========================================================================================="
"======== Simulation 3) Prior Variance and Temperature Affect Belief Update Speed ========="
"=========================================================================================="

def compute_belief_update_speed(dyad_games: List[Dict[str, Any]], player_uuid: str, general_settings: Dict[str, Any], 
                                true_parameters: Optional[Dict[str, float]] = None, params_of_interest: Optional[list[str]] = None, fraction: float = 0.5) -> float:
    """
    Quantify how quickly a predictor's fitted parameters move toward their target.

    This implements the “update speed” measure described in the paper's simulation
    section (Prior Variance and Temperature Affect Belief Update Speed). It returns
    a scalar in [0, 1] indicating the earliest round at which the fitted parameters
    cross a given fraction of the total distance between start and target.

    Two modes:
        • Absolute mode (model-to-truth):
            If `true_parameters` is provided, tracks the Euclidean distance between
            the predictor's fitted vector and the ground-truth vector each round.
            Computes:
                d(0)  = initial distance
                d(F)  = final distance
                d*    = d(0) + fraction · (d(F) – d(0))
            and return the earliest round index t where the distance crosses d*,
            normalized by the total number of intervals (t / T).

        • Relative mode (prior-to-posterior):
            If `true_parameters` is None, tracks the distance between param(t)
            and the initial param(0), and find the earliest t where the distance
            reaches `fraction` of the total prior→final shift.

    Arguments:
        • dyad_games: list[dict[str, Any]];
            Full game history for a single dyad (all rounds, all roles).
        • player_uuid: str;
            UUID of the predictor whose update speed is being measured.
        • general_settings: dict[str, Any];
            Settings that specify the update method (e.g., "grid") and experiment_num.
        • true_parameters: dict[str, float] | None;
            Ground-truth parameter vector to move toward (e.g., chooser parameters
            in simulations). If None, use the predictor's own final parameters as
            the “target”.
        • params_of_interest: list[str] | None;
            Subset of parameter keys to include (e.g., ["Vᵢᵢ","Vᵢⱼ"]). If None,
            use all non-std / non-cov / non-temp entries in the fitted vector.
        • fraction: float;
            Fraction of the total distance toward the target that defines “arrival”.
            In the paper, 0.5 corresponds to “halfway update speed”.

    Returns:
        • float in [0, 1];
            Normalized crossing time:
                0   → very fast update (threshold crossed immediately),
                1   → slowest (threshold never crossed before the final round).
    """
    update_method = general_settings.get('update_method', 'grid')
    experiment_num= general_settings.get('experiment_num', None)

    "1) gather param vectors or scalars per round"
    round_paramvals = []  # List of (round_number, param_vector).
    for game in dyad_games:
        "Must match the predictor"
        if game.get('predictor') != player_uuid:
            continue
        if experiment_num == 2 and game.get('phase') == 'rp':
            continue

        param_est = (
            game
            .get('parameter_estimates', {})
            .get(update_method, {})
            .get(player_uuid, {})
            .get('predictor', {})
            .get('params', {})
        )
        if not param_est:
            continue
        "Build a param vector ignoring _std, etc. if params_of_interest is None => keep all"
        relevant_vals = []
        for pkey, pval in param_est.items():
            if any(excluded_token in pkey for excluded_token in ['_std', '_cov']) or pkey in ('τ', 'temp'):
                continue
            if params_of_interest is not None:
                if pkey not in params_of_interest:
                    continue
            relevant_vals.append(pval)

        rnd = game.get('round', None)
        if relevant_vals and rnd is not None:
            round_paramvals.append((rnd, tuple(relevant_vals)))

    if len(round_paramvals) < 2:
        raise ValueError("Not enough param snapshots to measure an update speed.")
    
    "2) Sort by round"
    round_paramvals.sort(key=lambda round_paramval: round_paramval[0])
    "Just store them in arrays for convenience"
    rounds = [round_param_values[0] for round_param_values in round_paramvals]
    param_series = [round_param_values[1] for round_param_values in round_paramvals]  # Each is a tuple of length k.

    "Define a function to get Euclidian distance between param vectors"
    def dist_vec(first_param_vector, second_param_vector):
        return math.sqrt(sum((first_param_value-second_param_value)**2 for (first_param_value, second_param_value) in zip(first_param_vector, second_param_vector)))

    if true_parameters is not None:
        "\"absolute\" => compare param(t) to the ground truth each round"
        "Build ground_truth vector, ignoring _std, etc., same dimension"
        ground_vals = []
        for pkey in sorted(true_parameters.keys()):
            if any(excluded_token in pkey for excluded_token in ['_std', '_cov']) or pkey in ('τ', 'temp'):
                continue
            if params_of_interest is not None and pkey not in params_of_interest:
                continue
            ground_vals.append(true_parameters[pkey])
        ground_vec = tuple(ground_vals)
        if len(ground_vec) != len(param_series[0]):
            raise ValueError("Mismatch in dimension between param_of_interest and ground truth.")
        
        "Now param_series[t] => fitted param vector"
        # Distance(t) => dist_vec(param_series[t], ground_vec).
        dist_series = [dist_vec(param_vector, ground_vec) for param_vector in param_series]
        prior_dist = dist_series[0]
        final_dist = dist_series[-1]
        "Threshold"
        threshold = prior_dist + fraction*(final_dist - prior_dist)
        """
        Find earliest t crossing
        If final_dist < prior_dist, look for dist_series[t] <= threshold.
        If final_dist > prior_dist, look for dist_series[t] >= threshold.
        Handles reversed direction when final < prior.   
        """
        direction = 1 if final_dist > prior_dist else -1

        total_time_intervals = len(dist_series)-1  # Total intervals.
        crossing_round = total_time_intervals  # Default to last => speed=1 if never cross.
        for idx, distance_at_round in enumerate(dist_series):
            # Index spans 0..len-1.
            if direction>0:
                if distance_at_round >= threshold:
                    crossing_round = idx
                    break
            else:
                if distance_at_round <= threshold:
                    crossing_round = idx
                    break
        if total_time_intervals <= 0:
            return 0.0
        normalized_crossing_time = crossing_round / float(total_time_intervals)
        if normalized_crossing_time > 1:
            normalized_crossing_time = 1
        return normalized_crossing_time

    else:
        "\"relative\" => time to cross half difference from param(0) to param(final)"
        prior_vec = param_series[0]
        final_vec = param_series[-1]
        total_time_intervals = len(param_series)-1

        "Define threshold vector = prior_vec + fraction*(final_vec - prior_vec)"
        thresh_vec = tuple(
            prior_param_value + fraction*(final_param_value - prior_param_value)
            for (prior_param_value, final_param_value) in zip(prior_vec, final_vec)
        )

        "Do Euclidian distance approach:"
        prior_dist = dist_vec(prior_vec, prior_vec)  # Equals 0.
        final_dist = dist_vec(final_vec, prior_vec)  # Total shift from prior to final.
        target_dist = fraction * final_dist

        "Each round => measure dist from prior"
        dist_from_prior = [dist_vec(param_vector, prior_vec) for param_vector in param_series]

        crossing_round = total_time_intervals
        for idx, distance_at_round in enumerate(dist_from_prior):
            "Once distance_at_round >= target_dist => crossed fraction"
            if distance_at_round >= target_dist:
                crossing_round = idx
                break

        if total_time_intervals <= 0:
            return 0.0
        normalized_crossing_time = crossing_round / float(total_time_intervals)
        if normalized_crossing_time > 1:
            normalized_crossing_time = 1
        return normalized_crossing_time


def run_update_speed_simulation_regression(general_settings: GeneralSettings, file_paths: FilePaths, 
                                           params_of_interest: Optional[list[str]] = ['Vᵢⱼ'], 
                                           use_true_params: bool = False, n_dyads: int | None = 729) -> None:
    """
    Estimate how simulated belief update speed depends on prior variance and temperature.

    This function:
        1) Loads simulated dyads from JSON (created by the bot–bot simulation).
        2) For each dyad, computes an update speed for the predictor via
           `compute_belief_update_speed`.
        3) Extracts each predictor's initial fitted variance and temperature.
        4) Runs a linear regression:
               update_speed ~ τ_fitted_predictor + Vᵢⱼ_std_fitted_predictor
           mirroring the regression in the paper's simulation section.

    Arguments:
        • general_settings: GeneralSettings;
            Should match the settings used when fitting the simulated dyads
            (e.g., update_method = "grid").
        • params_of_interest: list[str] | None;
            Parameters used when constructing the belief vector for speed measurement
            (e.g., ["Vᵢⱼ"] or ["Vᵢᵢ","Vᵢⱼ"]).
        • json_path: str;
            Directory containing the per-dyad simulation fit JSON files.
        • use_true_params: bool;
            If True, compute “absolute” speed toward the chooser's true parameters.
            If False, compute “relative” speed from prior to final fitted values.
        • n_dyads: int | None;
            Number of dyad files to process. If None, use all JSON files in `json_path`.

    Returns:
        • None;
            Prints statsmodels OLS summary for the speed ~ variance + temperature regression.
    """
    def run_regression_on_speed(df_speed: pd.DataFrame, speed_col: str = "speed_value", predictors: list[str] = ["τ", "var"], add_constant: bool = True):
        """
        Fit an OLS regression predicting belief-update speed from prior variance and temperature.

        Uses `statsmodels.OLS` to regress `speed_col` onto `predictors`, optionally adding an
        intercept column.  After fitting, prints the full regression summary to stdout.

        Arguments:
            • df_speed: pd.DataFrame
                DataFrame containing one row per simulated dyad; must include `speed_col`
                and all columns listed in `predictors`.
            • speed_col: str
                Name of the column holding the normalized crossing-time speed values (the
                dependent variable).
            • predictors: list[str]
                Names of the independent variable columns (e.g., softmax temperature and
                prior variance).
            • add_constant: bool
                If True, prepends a constant column to the predictor matrix before fitting.

        Returns:
            • statsmodels RegressionResultsWrapper — the fitted OLS model, including
              coefficients, standard errors, p-values, and R².
        """
        import statsmodels.api as sm
        "Drop na"
        regression_input_dataframe = df_speed.dropna(subset=[speed_col]+predictors).copy()
        response_variable = regression_input_dataframe[speed_col]
        predictor_matrix = regression_input_dataframe[predictors]
        if add_constant:
            predictor_matrix = sm.add_constant(predictor_matrix, prepend=True)
        model = sm.OLS(response_variable, predictor_matrix)
        results = model.fit()
        print(results.summary())
        return results

    json_path = ensure_directory_and_join(base_dir=file_paths['player_fits'], file_name="experiment_0")

    update_method = 'grid'
    param_key_map = {
        'Vii':  'Vᵢᵢ', 'Vii_std': 'Vᵢᵢ_std', 
        'Vij':  'Vᵢⱼ', 'Vij_std': 'Vᵢⱼ_std',
        'temp': 'τ', 't': 'τ'
    }
    params_of_interest = [
        param_key_map.get(param, param) for param in params_of_interest
    ]

    if not isinstance(n_dyads, int):
        files = prep.get_files_in_directory(directory_path=json_path)
        n_dyads = len(files)

    params_to_us = {}
    for dyad_idx in range(n_dyads):
        simulated_dyad = get_simulated_dyad(file_paths=file_paths, dyad_idx=dyad_idx, n_games=21)

        dyad_key = list(simulated_dyad.keys())[0]
        dyad_games = simulated_dyad[dyad_key]
        first_game: dict = dyad_games[0]
        chooser_uuid, predictor_uuid = first_game['chooser'], first_game['predictor']
        param_est = first_game.get('parameter_estimates', {}).get(update_method, {}
                        ).get(predictor_uuid, {}).get('predictor', {}).get('params', {})
        fitted_params_predictor = {
            param_key_map.get(param_key, param_key): param_val 
            for param_key, param_val in param_est.items() 
        }
        true_params_chooser = first_game.get("true_params_chooser") or parse_robot_string(robot_str=chooser_uuid)
        true_params_chooser = {
            param_key_map.get(param_key, param_key): param_val
            for param_key, param_val in true_params_chooser.items()
        }
        true_params_predictor = first_game.get("true_params_predictor") or parse_robot_string(robot_str=predictor_uuid)
        true_params_predictor = {
            param_key_map.get(param_key, param_key): param_val
            for param_key, param_val in true_params_predictor.items()
        }

        update_speed = compute_belief_update_speed(dyad_games=dyad_games, player_uuid=predictor_uuid, fraction=0.5, 
                                        general_settings=general_settings, params_of_interest=params_of_interest, 
                                        true_parameters={
                                            param_key: param_val for param_key, param_val in true_params_chooser.items() 
                                            if param_key in params_of_interest
                                        } if use_true_params else None)

        params_to_us[dyad_key] = {
            'dyad_idx': dyad_idx,
            'dyad_key': dyad_key,
            'predictor_uuid': predictor_uuid,
            'chooser_uuid': chooser_uuid,
            **{f'{param_key}_true_predictor': param_val for param_key, param_val in true_params_predictor.items()},
            **{f'{param_key}_fitted_predictor': param_val for param_key, param_val in fitted_params_predictor.items()},
            **{f'{param_key}_true_chooser': param_val for param_key, param_val in true_params_chooser.items()},
            'update_speed': update_speed
        }

        if dyad_idx % 40 == 0: print(dyad_idx)

    update_speed_df = pd.DataFrame.from_dict(params_to_us, orient='index')
    res = run_regression_on_speed(
        df_speed=update_speed_df,
        speed_col="update_speed",
        predictors=["τ_fitted_predictor","Vᵢⱼ_std_fitted_predictor"]
    )


def analyze_update_speed_in_human_bot(file_paths: Dict[str, Dict[str, str] | str], general_settings: Dict[str, Any]) -> None:
    """
    Summarize belief update speed in the human–bot experiment for each participant and avatar type.

    This function:
        1) Loads player-level dyad fits from disk.
        2) For each dyad, computes an update speed for the predictor using
           `compute_belief_update_speed`, typically toward the avatar's true (Vᵢᵢ, Vᵢⱼ).
        3) Aggregates update speeds:
               • per predictor (mean, std),
               • per counterpart (avatar type; mean, std).
        4) Returns these summaries for further plotting or regression.

    This corresponds to the empirical update-speed analysis that complements the
    simulation results in the paper.

    Arguments:
        • file_paths: dict[str, dict[str,str] | str];
            Standard Iter_Binary_Dictator file paths, including 'player_fits'.
        • general_settings: dict[str, Any];
            Must contain 'experiment_num' to know whether counterparts are human or avatars.
        • utility_settings: dict[str, bool];
            Currently unused here but included for symmetry with other analysis functions.
        • n_segments: int;
            Retained for compatibility; not currently used in this function.

    Returns:
        • dict[str, Any];
            {
                'update_speeds': dict[predictor_uuid → dict[counterpart_id → speed]],
                'update_speeds_per_predictor': dict[predictor_uuid → list[speed]],
                'update_speeds_per_counterpart': dict[counterpart_id → list[speed]],
                'update_speeds_per_predictor_mean_std': dict[predictor_uuid → {'mean','std'}],
                'update_speeds_per_counterpart_mean_std': dict[counterpart_id → {'mean','std'}],
            }
    """
    avatar_params = {
        'utilitarian': {'Vᵢᵢ':  1.0, 'Vᵢⱼ':  1.0}, 
        'selfish':     {'Vᵢᵢ':  1.0, 'Vᵢⱼ':  0.0}, 
        'competitive': {'Vᵢᵢ':  1.0, 'Vᵢⱼ': -1.0}, 
        'masochistic': {'Vᵢᵢ': -1.0, 'Vᵢⱼ':  1.0}
    }

    experiment_num = general_settings.get('experiment_num')
    plrs_to_dyads = prep.players_to_dyads(experiment_num=2, 
                    file_paths=file_paths, create_new_file=False)
    player_uuids = sorted(list(plrs_to_dyads.keys()))

    update_speeds = {}
    prediction_accuracy_data = {}
    for player_uuid in player_uuids:
        player_dyads = None
        file_name_suffix = file_paths["file_name_suffix"]
        plr_file_path = os.path.join(file_paths["player_fits"], f"experiment_{experiment_num}", 
                                f'{file_name_suffix}_' + player_uuid + ".json")

        if os.path.exists(plr_file_path):
            with open(plr_file_path, "r", encoding='utf-8') as file:
                player_dyads: dict = json.load(file)

        if player_dyads is None:
            raise Exception(f"Failed to extract dyads for player {player_uuid}.")        

        update_speeds_per_counterpart = {}
        for dyad_key, dyad_games in player_dyads.items():
            true_params = None
            counterpart_uuid = None
            if experiment_num == 2:
                avatar_uuid = dyad_games[0]['chooser']
                counterpart_type = None
                for avatar_type in ('utilitarian', 'selfish', 'competitive', 'masochistic'):
                    if avatar_type in avatar_uuid:
                        counterpart_type = avatar_type
                if counterpart_type is None:
                    raise Exception("Avatar type not found.")
                counterpart_uuid = counterpart_type
                true_params = avatar_params[counterpart_type]
            else:
                counterpart_uuid = prep._dyad_key(dyad_key=dyad_key, return_tuple=True)
                if counterpart_uuid == player_uuid:
                    counterpart_uuid = prep._dyad_key(dyad_key=dyad_key, return_tuple=True, reverse=True)

            update_speed = compute_belief_update_speed(dyad_games=dyad_games, player_uuid=player_uuid, general_settings=general_settings, 
                                                true_parameters=true_params, params_of_interest=["Vᵢᵢ", "Vᵢⱼ"], fraction=0.75)

            update_speeds_per_counterpart[counterpart_uuid] = update_speed
        update_speeds[player_uuid] = update_speeds_per_counterpart   

    update_speeds_per_predictor = {}
    update_speeds_per_counterpart = {}
    for player_uuid, stabilization_per_cnterprt in update_speeds.items():
        if player_uuid not in update_speeds_per_predictor:
            update_speeds_per_predictor[player_uuid] = []
        for counterpart_uuid, update_speed in stabilization_per_cnterprt.items():
            if counterpart_uuid not in update_speeds_per_counterpart:
                update_speeds_per_counterpart[counterpart_uuid] = []
            update_speeds_per_counterpart[counterpart_uuid].append(update_speed)
            update_speeds_per_predictor[player_uuid].append(update_speed)

    update_speeds_per_predictor_mean_std = {
        player_uuid: {
            'mean': np.mean(np.array(rates)), 
            'std': np.std(np.array(rates))
        } for player_uuid, rates in update_speeds_per_predictor.items()
    }
    update_speeds_per_counterpart_mean_std = {
        counterpart: {
            'mean': np.mean(np.array(rates)), 
            'std': np.std(np.array(rates))
        } for counterpart, rates in update_speeds_per_counterpart.items()       
    }

    return {
        'update_speeds': update_speeds,
        'update_speeds_per_predictor': update_speeds_per_predictor,
        'update_speeds_per_counterpart': update_speeds_per_counterpart,
        'update_speeds_per_predictor_mean_std': update_speeds_per_predictor_mean_std,
        'update_speeds_per_counterpart_mean_std': update_speeds_per_counterpart_mean_std,
    }


def plot_update_speed_by_counterpart(update_speeds_per_counterpart: Dict[str, List[float]], figure_layout: Dict[str, Any], export_fig: bool = True, 
                                     file_name: str = "update_speed_violin_boxplot.html", as_bar_chart: bool = True) -> go.Figure:
    """
    Visualize belief-update speed across avatar types / counterparts.

    This is primarily for Experiment 1/2, showing how quickly participants
    learn different social preference profiles (utilitarian, selfish, etc.).

    In bar mode (default), the figure shows:
        • one bar per counterpart with mean update speed
        • 95% confidence intervals as error bars.

    In the alternative mode, this can be reused as a violin/box plot
    (see inline comments) to show the full distribution per counterpart.

    Arguments:
        • update_speeds_per_counterpart: dict[str, list[float]];
            Mapping from counterpart label (e.g., "utilitarian") to a list of
            normalized update speeds across participants.
        • figure_layout: dict[str, Any];
            Plotly layout settings (template, fonts, axes, etc.).
        • export_fig: bool;
            If True, save the figure as an HTML file at `out_path`.
        • out_path: str;
            Path to the output HTML file if `export_fig` is True.
        • as_bar_chart: bool;
            If True (default), plot means with CIs as bars. If False, use
            the alternative (currently bar-structured but ready to be adapted
            to violin/box if desired).

    Returns:
        • go.Figure;
            The constructed Plotly figure.
    """
    def mean_confidence_interval(data, confidence=0.95, rnd=4):
        """Compute a confidence interval for continuous data."""
        import scipy.stats
        data_array = 1.0 * np.array(data)
        n_data, mean_value, standard_error = len(data_array), np.mean(data_array), scipy.stats.sem(data_array)
        half_width = standard_error * scipy.stats.t.ppf((1 + confidence) / 2., n_data-1)
        return (round(mean_value-half_width, rnd), round(mean_value, rnd), round(mean_value+half_width, rnd))

    out_path = ensure_directory_and_join(file_paths['visuals'], file_name)

    "Build the figure"
    fig = go.Figure()

    if as_bar_chart:
        counterparts = [counterpart.capitalize() for counterpart in update_speeds_per_counterpart.keys()]
        colors = [f'hsla({int(115 + 360/(len(counterparts)+4) * idx) % 360}, 80%, 40%, 1.0)' for idx in range(len(counterparts))]
        update_speeds_per_counterpart_mean_std = {
            counterpart: {
                'mean': np.mean(np.array(rates)), 
                'std': np.std(np.array(rates)),
                'ci': gnrl.mean_confidence_interval(data=rates, confidence=0.95, rnd=6)
            } for counterpart, rates in update_speeds_per_counterpart.items()       
        }
        update_speeds_means = [val['mean'] for val in update_speeds_per_counterpart_mean_std.values()]
        update_speeds_stds = [val['std'] for val in update_speeds_per_counterpart_mean_std.values()]
        update_speeds_cis = [summary_values['ci'] for summary_values in update_speeds_per_counterpart_mean_std.values()]
        update_speeds_cis_upper = [round(confidence_interval[2] - confidence_interval[1], 8) for confidence_interval in update_speeds_cis]
        update_speeds_cis_lower = [round(confidence_interval[1] - confidence_interval[0], 8) for confidence_interval in update_speeds_cis]
        update_speeds_means = [round(mean, 6) for mean in update_speeds_means]
        
        all_update_speeds = []
        for rates in update_speeds_per_counterpart.values():
            all_update_speeds += rates
        mean_update_speed = np.mean(np.array(all_update_speeds))
        std_update_speed = np.std(np.array(all_update_speeds))
        ci_update_speed = gnrl.mean_confidence_interval(data=all_update_speeds, confidence=0.95, rnd=6)
        print(f"Mean Update Speed: {mean_update_speed}")
        print(f"Std Update Speed:  {std_update_speed}")
        print(f"CI Update Speed:   {ci_update_speed}")

        data= {
            'counterpart': counterparts,
            'means': update_speeds_means,
            'ci_low': update_speeds_cis_lower,
            'ci_high': update_speeds_cis_upper,
            'std': update_speeds_stds
        }
        update_speed_summary_df = pd.DataFrame(data=data)
        print(update_speed_summary_df)

        fig = go.Figure([go.Bar(
            x=counterparts, 
            y=update_speeds_means,
            marker_color=colors,
            error_y=dict(
                type='data', 
                array=update_speeds_cis_upper, 
                arrayminus=update_speeds_cis_lower
            ),
            hovertemplate="Counterpart: %{x}<br>Update Speed: %{y:.3f}<extra></extra>"
        )])
    else:
        "Create one Violin trace (which can also be displayed as a Box) per counterpart."
        for counterpart, speeds in update_speeds_per_counterpart.items():
            fig.add_trace(
                go.Bar(
                    x=[counterpart]*len(speeds),  # Category on x-axis.
                    y=speeds,                     # Data points on y-axis.
                    box=dict(visible=True),
                    meanline=dict(visible=True),
                    points='all',                 # Show individual data points.
                    pointpos=-0.7,
                    jitter=0.45,
                    scalemode='count',
                    width=0.4,
                    name=counterpart,             # Legend / name for this trace.
                    line_color='hsla(115, 70%, 40%, 1.0)',
                    hovertemplate="Counterpart: %{x}<br>Update Speed: %{y:.3f}<extra></extra>"
                )
            )

    "Title and layout"
    "Uses a fixed title for the update-speed summary."
    fig.update_layout(
        title="Belief Update Speed by Counterpart",
        titlefont_size=figure_layout.get("titlefont_size", 18) - 4,
        template=figure_layout.get("template", "plotly_dark"),
        title_x=figure_layout.get('title_x', 0.5),
        title_y=figure_layout.get('title_y', 0.95),

        xaxis=dict(
            title="Counterpart Preference Profile",
            **figure_layout.get("xaxis", {})
        ),
        yaxis=dict(
            range=[0.0, 0.5] if as_bar_chart else [0.0, 0.5],
            title="Mean Update Speed (fraction of rounds)",
            **figure_layout.get("yaxis", {})
        ),
        hoverlabel=figure_layout.get("hoverlabel", {}),
        font=figure_layout.get("font", {}),
        margin=dict(l=600, r=600, t=120, b=120) if as_bar_chart \
            else dict(l=150, r=120, t=120, b=120)
    )

    if not as_bar_chart:
        "Dropdown menu to switch between violin and box"
        fig.update_layout(
            updatemenus=[dict(
                buttons=list([
                    dict(
                        args=["type", "violin"], 
                        label="Violin", 
                        method="restyle"
                    ),
                    dict(
                        args=["type", "box"], 
                        label="Boxplot", 
                        method="restyle"
                    )
                ]),
                direction="down", 
                x=1.03, 
                xanchor="left",
                y=0.7, 
                yanchor="top",
                pad={"r": 10, "t": 10},
                showactive=True
            )]
        )

    "Export or show"
    if export_fig:
        fig.write_html(out_path)
        print("Saved violin-box figure to", out_path)
    else:
        fig.show()

    return fig


