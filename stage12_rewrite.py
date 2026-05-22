"""Rewrites the Stage 12 section of analysis.py."""
import os

analysis_path = r"c:\Users\Gregory Stanley\Desktop\U of M\Research Archive\Multiplayer\hidden-motives-analysis\analysis.py"

content = open(analysis_path, encoding='utf-8-sig').read()
lines   = content.split('\n')

# Keep everything up to (but not including) line 11482 (the first Stage 12 comment line).
# Line 11482 is index 11481 in 0-based indexing.
prefix = '\n'.join(lines[:11481])

new_stage12 = '''
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
     softmax_temperature) = args

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
                optimization_method='local',
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
            'best_params': _json.dumps(best_params),
        })

    return results


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
    n_workers: Optional[int] = None,
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
        4. Generate synthetic chooser-only data for max(n_agents_grid) synthetic agents
           x max(n_games_grid) games each (done once; all n_agents/n_games conditions
           use nested subsets of this pre-generated data).
        5. For each (n_agents_value, n_games_value) condition, slice the pre-generated
           data and fit all candidates to each agent's chooser NLL (tau fixed).
        6. Compute BIC, BIC rank, delta-BIC, and recovery metrics per row.
        7. Append each completed condition to a partial CSV and resume on restart.

    Arguments:
        • general_settings: dict; must contain 'experiment_num', 'run_in_parallel', etc.
        • file_paths: dict; must contain 'processed', 'bic_aic', 'visuals'.
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
        • softmax_temperature: float; fixed tau for both data generation and NLL fitting (default 0.5).
        • candidate_model_selection_mode: str; 'hamming' or 'ampd' max-min diversity selection.
        • n_candidate_models: int | None; size of the candidate set (default 100).
        • ampd_matrix_name_or_path: str | None; path to AMPD matrix when mode='ampd'.
        • n_workers: int | None; None = cpu_count - 1; 1 = sequential (for debugging).
        • random_seed: int; reproducibility seed (default 42).
        • create_new_file: bool; if False and final CSV exists, load and return it immediately.

    Returns:
        • pd.DataFrame; one row per (n_agents_fitted, n_games_fitted, agent_idx, candidate utility_idx).

    Resume support:
        The function appends each completed (n_agents, n_games) condition to
        processed/model_recovery_simulation_{generating_utility_idx}_partial.csv.
        On restart with create_new_file=False, completed conditions are detected
        from the partial file and skipped. On clean completion the partial file is deleted.
    """
    import math as _math

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
    processed_dir = str(file_paths['processed'])
    registry_df   = pd.read_csv(
        os.path.join(processed_dir, 'all_utility_functions.csv'),
        dtype={'utility_bitstring': str},
    )
    _non_flag_columns = {
        'utility_idx', 'utility_bitstring', 'k_params', 'redundant_with', 'differing_settings',
        'n_data', 'pvar', 'param_norm_sd', 'loss_nll', 'AIC', 'BIC',
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
    output_csv_path  = os.path.join(processed_dir,
                                    f'model_recovery_simulation_{generating_utility_idx}.csv')
    partial_csv_path = os.path.join(processed_dir,
                                    f'model_recovery_simulation_{generating_utility_idx}_partial.csv')

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
        r"C:\\Users\\Gregory Stanley\\Desktop\\U of M\\Research Archive\\Multiplayer"
        r"\\ABM_Simulation\\Judgment_Game\\Inputs\\Iter_Binary_Dictator"
        rf"\\bic_aic\\All_Utility_Forms_IC_Analysis_Experiment{experiment_num}.json"
    )
    if not os.path.exists(ic_json_path) or os.path.getsize(ic_json_path) < 50_000_000:
        ic_json_path = _old_repo_ic_json_path
    # END TEMPORARY BRIDGE

    print(f"Loading IC JSON: {os.path.basename(ic_json_path)}")
    with open(ic_json_path, 'r', encoding='utf-8-sig') as ic_file_handle:
        ic_data = json.load(ic_file_handle)
    ic_results = ic_data.get('ic_results', {})

    "Find generating model entry (ic_results keyed by utility_tuple_str, not utility_idx)."
    generating_model_entry = None
    for _model_tuple_str, model_entry in ic_results.items():
        if int(model_entry.get('idx', -1)) == generating_utility_idx:
            generating_model_entry = model_entry
            break
    if generating_model_entry is None:
        raise ValueError(
            f"Generating model idx={generating_utility_idx} not found in IC JSON "
            f"({ic_json_path})."
        )

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
        raise ValueError(
            f"No complete chooser parameter vectors found for generating model "
            f"{generating_utility_idx} in IC JSON.\\n"
            f"  Expected param keys: {generating_param_keys}"
        )
    print(f"  Extracted {len(param_pool)} participant parameter vectors "
          f"for generating model {generating_utility_idx}.")

    "Generate synthetic data for n_agents_max agents x n_games_max games (done once; then sliced)."
    print(f"\\nGenerating synthetic data: "
          f"{n_agents_max} agents x {n_games_max} games ...")
    random_state = np.random.RandomState(random_seed)
    all_synthetic_agent_games: List[List[dict]] = []

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
        )
        agent_games_list = next(iter(dyad_data.values()))
        all_synthetic_agent_games.append(agent_games_list)

        if (agent_idx + 1) % 20 == 0 or (agent_idx + 1) == n_agents_max:
            print(f"  Generated {agent_idx + 1}/{n_agents_max} agents.")

    "Resolve parallelism (respects general_settings['run_in_parallel'])."
    run_in_parallel_flag = general_settings.get('run_in_parallel', True)
    cpu_count_available  = mp.cpu_count()
    if not run_in_parallel_flag:
        n_workers_clamped = 1
    elif n_workers is None:
        n_workers_clamped = max(1, cpu_count_available - 1)
    else:
        n_workers_clamped = max(1, min(n_workers, cpu_count_available - 1))

    total_start_time = time.time()

    for n_agents_value in n_agents_grid:
        for n_games_value in n_games_grid:
            if (n_games_value, n_agents_value) in completed_conditions:
                print(f"[n_agents={n_agents_value}, n_games={n_games_value}] "
                      f"Already complete; skipping.")
                continue

            n_games_start_time = time.time()
            print(f"\\n[n_agents={n_agents_value}, n_games={n_games_value}] "
                  f"Fitting {len(candidate_models)} candidates x {n_agents_value} agents ...")

            worker_args_list = [
                (agent_idx,
                 all_synthetic_agent_games[agent_idx][:n_games_value],
                 candidate_models,
                 general_settings_for_fitting,
                 param_bds,
                 softmax_temperature)
                for agent_idx in range(n_agents_value)
            ]

            agent_fit_results: List[dict] = []
            if n_workers_clamped == 1:
                for worker_idx, worker_args_item in enumerate(worker_args_list):
                    for result_row in _recovery_fit_worker(worker_args_item):
                        agent_fit_results.append(result_row)
                    if (worker_idx + 1) % 20 == 0 or (worker_idx + 1) == n_agents_value:
                        elapsed = time.time() - n_games_start_time
                        eta     = elapsed / (worker_idx + 1) * (n_agents_value - worker_idx - 1)
                        print(f"  Agent {worker_idx + 1}/{n_agents_value}  "
                              f"elapsed: {_fmt_duration(elapsed)}  ETA: {_fmt_duration(eta)}")
            else:
                with mp.Pool(n_workers_clamped) as pool:
                    for worker_idx, agent_results in enumerate(
                        pool.imap_unordered(_recovery_fit_worker, worker_args_list)
                    ):
                        agent_fit_results.extend(agent_results)
                        if (worker_idx + 1) % 20 == 0 or (worker_idx + 1) == n_agents_value:
                            elapsed = time.time() - n_games_start_time
                            print(f"  {worker_idx + 1}/{n_agents_value} agents returned  "
                                  f"elapsed: {_fmt_duration(elapsed)}")

            "Assemble result rows and compute per-agent BIC ranks."
            n_games_result_rows = []
            for fit_row in agent_fit_results:
                n_valid_games = fit_row['n_games']
                k_params      = fit_row['k_params']
                nll           = fit_row['nll']
                bic_value     = (2.0 * nll + k_params * _math.log(max(n_valid_games, 1))
                                 if n_valid_games > 0 else float('inf'))
                n_games_result_rows.append({
                    'n_games_fitted':   n_games_value,
                    'n_agents_fitted':  n_agents_value,
                    'agent_idx':        fit_row['agent_idx'],
                    'true_utility_idx': generating_utility_idx,
                    'utility_idx':      fit_row['utility_idx'],
                    'nll':              nll,
                    'k_params':         k_params,
                    'bic':              bic_value,
                    'n_valid_games':    n_valid_games,
                    'best_params':      fit_row['best_params'],
                })

            n_games_results_df = pd.DataFrame(n_games_result_rows)

            def _add_bic_rank(agent_group: pd.DataFrame) -> pd.DataFrame:
                agent_group = agent_group.copy()
                agent_group['bic_rank']  = agent_group['bic'].rank(method='min').astype(int)
                agent_group['delta_bic'] = agent_group['bic'] - agent_group['bic'].min()
                return agent_group

            n_games_results_df = n_games_results_df.groupby(
                'agent_idx', group_keys=False,
            ).apply(_add_bic_rank)
            n_games_results_df['is_generating_model'] = (
                n_games_results_df['utility_idx'] == generating_utility_idx
            )

            "Append completed condition to partial CSV (enables mid-run resume on restart)."
            partial_write_header = not os.path.exists(partial_csv_path)
            n_games_results_df.to_csv(
                partial_csv_path, mode='a', header=partial_write_header,
                index=False, encoding='utf-8-sig',
            )
            accumulated_dataframes.append(n_games_results_df)

            generating_model_mask  = n_games_results_df['is_generating_model']
            recovery_rate          = float((n_games_results_df.loc[generating_model_mask, 'bic_rank'] == 1).mean())
            mean_generating_rank   = float(n_games_results_df.loc[generating_model_mask, 'bic_rank'].mean())
            n_games_elapsed        = time.time() - n_games_start_time
            print(f"  -> recovery_rate={recovery_rate:.3f}  "
                  f"mean_rank={mean_generating_rank:.1f}  "
                  f"time={_fmt_duration(n_games_elapsed)}")

    "Combine all conditions, write final CSV, delete partial."
    all_results_df = (
        pd.concat(accumulated_dataframes, ignore_index=True)
        if accumulated_dataframes else pd.DataFrame()
    )
    all_results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\\nModel recovery simulation saved: {output_csv_path}  ({len(all_results_df)} rows)")
    print(f"Total time: {_fmt_duration(time.time() - total_start_time)}")

    if os.path.exists(partial_csv_path):
        os.remove(partial_csv_path)

    return all_results_df


def plot_model_recovery_simulation(
    general_settings: dict,
    file_paths: dict,
    fig_lay: dict,
    generating_utility_idx: int = 443,
    export_fig: bool = True,
) -> 'go.Figure':
    """
    Plot data-adequacy recovery curves from the model recovery simulation.

    Reads processed/model_recovery_simulation_{generating_utility_idx}.csv and
    produces one trace pair per n_agents_fitted value (if multiple exist in the CSV).
    Each pair shows: (1) recovery rate (fraction of agents where the generating model
    has BIC rank 1) vs n_games, and (2) mean BIC rank of the generating model vs
    n_games on a secondary y-axis.

    Arguments:
        • general_settings: dict; accepted for API consistency (not currently used).
        • file_paths: dict; must contain 'processed' and 'visuals'.
        • fig_lay: dict; layout settings (template, font, title_size, base_hue).
        • generating_utility_idx: int; must match the CSV filename (default 443).
        • export_fig: bool; if True, writes HTML to visuals/.

    Returns:
        • go.Figure
    """
    output_csv_path = os.path.join(
        str(file_paths['processed']),
        f'model_recovery_simulation_{generating_utility_idx}.csv',
    )
    all_results_df = pd.read_csv(output_csv_path, encoding='utf-8-sig')

    generating_model_df = all_results_df[
        all_results_df['utility_idx'] == generating_utility_idx
    ].copy()

    base_hue            = fig_lay.get('base_hue', 220)
    base_font_size      = max(8, fig_lay.get('font', {}).get('size', 28) // 2)
    n_candidates        = all_results_df['utility_idx'].nunique()
    all_n_agents_values = sorted(all_results_df['n_agents_fitted'].unique())

    fig = go.Figure()

    "One color-shifted trace pair per n_agents value."
    for agents_loop_idx, n_agents_value in enumerate(all_n_agents_values):
        agents_subset_df = generating_model_df[
            generating_model_df['n_agents_fitted'] == n_agents_value
        ]
        summary_rows = []
        for n_games_value in sorted(all_results_df['n_games_fitted'].unique()):
            n_games_subset_df = agents_subset_df[agents_subset_df['n_games_fitted'] == n_games_value]
            if n_games_subset_df.empty:
                continue
            summary_rows.append({
                'n_games':        int(n_games_value),
                'recovery_rate':  float((n_games_subset_df['bic_rank'] == 1).mean()),
                'mean_rank':      float(n_games_subset_df['bic_rank'].mean()),
                'mean_delta_bic': float(n_games_subset_df['delta_bic'].mean()),
            })
        if not summary_rows:
            continue
        summary_df      = pd.DataFrame(summary_rows)
        hue_shift       = (agents_loop_idx * 40) % 360
        primary_color   = _hsla(hue=(base_hue + hue_shift) % 360, alpha=0.9)
        secondary_color = _hsla(hue=(base_hue + hue_shift + 120) % 360, alpha=0.65)
        agents_label    = f"N={n_agents_value}"

        fig.add_trace(go.Scatter(
            x=summary_df['n_games'],
            y=summary_df['recovery_rate'],
            mode='lines+markers',
            name=f'Recovery rate ({agents_label})',
            line=dict(color=primary_color, width=3),
            marker=dict(size=10),
            hovertemplate=(
                f'{agents_label}<br>'
                'n_games=%{x}<br>'
                'recovery_rate=%{y:.3f}<br>'
                '<extra></extra>'
            ),
        ))

        fig.add_trace(go.Scatter(
            x=summary_df['n_games'],
            y=summary_df['mean_rank'],
            mode='lines+markers',
            name=f'Mean BIC rank ({agents_label})',
            yaxis='y2',
            line=dict(color=secondary_color, width=2, dash='dash'),
            marker=dict(size=8),
            hovertemplate=(
                f'{agents_label}<br>'
                'n_games=%{x}<br>'
                'mean_rank=%{y:.1f}<br>'
                '<extra></extra>'
            ),
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
            title='Number of chooser games per agent (n_games)',
            showgrid=True,
            gridcolor=_hsla(hue=0, saturation_percent=0, lightness_percent=78, alpha=0.4),
        ),
        yaxis=dict(
            title='Recovery rate (fraction of agents with rank-1 BIC)',
            range=[0.0, 1.05],
            tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ),
        yaxis2=dict(
            title=f'Mean BIC rank of generating model (out of {n_candidates})',
            overlaying='y',
            side='right',
            showgrid=False,
            range=[0, n_candidates + 1],
        ),
        hoverlabel=dict(font=dict(size=max(8, base_font_size * 2 - 4))),
        template=fig_lay.get('template', 'plotly_white'),
        font=dict(
            family=fig_lay.get('font', {}).get('family', 'Calibri'),
            size=base_font_size,
        ),
        margin=dict(l=80, r=100, t=120, b=80),
        autosize=True,
        legend=dict(yanchor='top', y=0.95, xanchor='left', x=0.05),
    )

    if export_fig:
        out_path = os.path.join(
            str(file_paths['visuals']),
            f'model_recovery_simulation_{generating_utility_idx}.html',
        )
        fig.write_html(out_path, config={'responsive': True})
        print(f"Model recovery simulation plot saved: {out_path}")

    return fig
'''

new_content = prefix + new_stage12
open(analysis_path, 'w', encoding='utf-8-sig').write(new_content)
print(f"Done. Total lines: {new_content.count(chr(10))}")
