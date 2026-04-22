from config import *

"=========================================================================================="
"============================ Basic Utilities for Loading Data ============================"
"=========================================================================================="

def combine_csv_files(file_paths: FilePaths, output_filename: str = "combined_data.csv") -> pd.DataFrame | None:
    """
    Combines multiple CSV files from a given directory into a single DataFrame.

    Arguments:
        • file_path_raw_data: str; Path to the directory containing CSV files.
        • file_path_outputs: str; Path to the directory where the combined CSV file will be saved.
        • output_filename: str; Name of the output CSV file.

    Returns:
        • combined_df: pandas.DataFrame; The combined DataFrame.
    """
    file_path_outputs = file_paths["processed"]

    all_dataframes = []
    for filename in os.listdir(file_paths["raw_data"]):
        filepath = os.path.join(file_paths["raw_data"], filename)
        if filename.endswith(".csv"):  # Check if the file is a CSV file
            try:
                df = pd.read_csv(filepath)
                all_dataframes.append(df)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    if not all_dataframes:
        print("No CSV files found in the input directory.")
        return None

    combined_df = pd.concat(all_dataframes, ignore_index=True)
    combined_df.to_csv(os.path.join(file_path_outputs, output_filename), index=False)

    return combined_df


def get_file_by_index_or_name(directory_path: str, file_name_idx: int | None = None, file_name: str | None = None) -> str | None:
    """
    Retrieve a filename from a directory either by integer position or by exact name.

    Exactly one of `file_name_idx` or `file_name` should be provided.  If neither is
    provided, or if the index is out of range, or the name is not found, the function
    prints a diagnostic message and returns None.

    Arguments:
        • directory_path: str
            Path to the directory to search.
        • file_name_idx: int | None
            Zero-based index into the list of files returned by `get_files_in_directory`.
            Negative indices are not supported and will be treated as out-of-range.
        • file_name: str | None
            Exact filename (not full path) to look up in the directory listing.

    Returns:
        • str | None — the matching filename, or None if no match is found.
    """
    files = get_files_in_directory(directory_path)

    if file_name_idx is not None:
        if 0 <= file_name_idx < len(files):
            return files[file_name_idx]
        else:
            print(f"Index {file_name_idx} out of range (0–{len(files)-1})")
            return None

    if file_name is not None:
        if file_name in files:
            return file_name
        else:
            print(f"File '{file_name}' not found in directory.")
            return None

    print("Please provide either a file index or a file name.")
    return None


def dataframe(file_path: str, file_name: str) -> pd.DataFrame | None:
    """
    Load a CSV file from disk and return it as a DataFrame.

    Arguments:
        • file_path: str
            Directory containing the CSV file.
        • file_name: str
            Name of the CSV file (including extension).

    Returns:
        • pd.DataFrame | None — the loaded DataFrame, or None if the file does not exist.
    """

    full_path = os.path.join(file_path, file_name)

    if os.path.exists(full_path):
        results_df = pd.read_csv(full_path, encoding="utf-8", engine='python')    
        if 'Unnamed: 0' in results_df.columns:
            del results_df['Unnamed: 0']

        return results_df

    print(f"File {file_name} not found at {file_path}. Please check the path.")
    return None


def get_files_in_directory(directory_path: str) -> list[str]:
    """Return a list of file names (not directories) in the given directory."""
    try:
        return [file for file in os.listdir(directory_path) 
                if os.path.isfile(os.path.join(directory_path, file))]
    except FileNotFoundError:
        print(f"Directory not found: {directory_path}")
        return []
    except Exception as err:
        print(f"Error: {err}")
        return []


def _dyad_key(dyad_key: DyadKey, return_tuple: bool = False, reverse: bool = False) -> DyadKey:
    """
    Standardize a dyad key into a canonical sorted string or tuple.

    Accepts either a `"(uuid1, uuid2)"` string or a list/tuple of two UUIDs, sorts the
    pair alphabetically (or reverse-alphabetically if `reverse=True`), and returns either
    a formatted `"(uuid1, uuid2)"` string or a raw two-element tuple.

    Arguments:
        • dyad_key: DyadKey
            The raw dyad identifier — either a parenthesized string or a list/tuple.
        • return_tuple: bool
            If True, return a `(uuid1, uuid2)` tuple instead of a formatted string.
        • reverse: bool
            If True, sort UUIDs in descending order.

    Returns:
        • DyadKey — canonicalized dyad key as a string (default) or tuple.
    """
    if isinstance(dyad_key, (tuple, list)):
        dyad_key = list(dyad_key)
    else:
        dyad_key = dyad_key[1:-1].split(", ")
    player_uuid_1, player_uuid_2 = sorted(dyad_key, reverse=reverse)    
    if return_tuple:
        return (player_uuid_1, player_uuid_2)
    return f"({player_uuid_1}, {player_uuid_2})"


def _dyad_file_path(dyad_key: DyadKey, file_paths: FilePaths, experiment_num: int = 3, analysis_mode: str = 'bayesian', analysis_unit: str = 'player') -> str:
    """
    Build the full file path for a dyad's stored output JSON.

    Extracts the first 8 characters of each player UUID to form a compact suffix, then
    constructs the filename from the standard naming template in `file_paths['file_names']`
    and appends the dyad-specific suffix.

    Arguments:
        • dyad_key: DyadKey
            Dyad identifier in any format accepted by `_dyad_key`.
        • file_paths: FilePaths
            Project file-path dict; must contain `'dyad_data'` and `'file_names'` entries.
        • experiment_num: int
            Experiment number (1, 2, or 3); used to select the correct filename template.
        • analysis_mode: str
            `'bayesian'` or `'mle'`; determines which filename template key is used.
        • analysis_unit: str
            Reserved for future use; currently unused in path construction.

    Returns:
        • str — absolute path to the dyad JSON file.
    """
    player_uuid_1, player_uuid_2 = _dyad_key(dyad_key=dyad_key, return_tuple=True)
    short_dkey = f"{player_uuid_1[:8]}-{player_uuid_2[:8]}"
    dyad_file_name = file_paths["file_names"][
        f"params_data_exper{experiment_num}_{'bayes' if analysis_mode == 'bayesian' else 'iter'}"][:-5] + f"-dyad_{short_dkey}.json"
    
    return os.path.join(file_paths["dyad_data"], dyad_file_name)


def get_dyad_data(dyad_key: int | DyadKey, file_paths: FilePaths, dyad_already_analyzed: bool = True, experiment_num: int = 3, analysis_mode: str = 'bayesian') -> list[dict[str, Any]]:
    """
    Retrieve and process dyad data from JSON files for a specific dyad key.
    
    This function reads a dyad file from the specified directory, processes its contents, 
    and handles both grid-based and non-grid-based parameter estimates. For grid-based estimates, 
    JSON-encoded keys and arrays are converted back to their original non-JSON serializable formats.

    Arguments:
        • dyad_key (int): The key identifying the dyad to process. Can be an integer index or a specific identifier.
        • file_paths (Dict[str, str]): Dictionary containing paths, including 'dyad_data' (path to dyad data directory)
                                     and 'file_name_suffix' (identifier for the type of utility function used).
        • experiment_num (int, optional): Experiment identifier. Defaults to 3.
        • analysis_mode (str, optional): Analysis mode, either 'bayesian' or 'iterative'. Defaults to 'bayesian'.

    Returns:
        • Dict: The processed dyad data.

    Raises:
        • Exception: If the dyad data file cannot be loaded or is empty.
    """
    import ast

    if dyad_already_analyzed:
        "Determine the file path based on the dyad key"
        if isinstance(dyad_key, int):
            directory = os.listdir(file_paths["dyad_data"])
            analysis_mode_key = "Bayes" if analysis_mode == "bayesian" else "Iter"
            file_names = [
                file_name for file_name in directory
                if all(fstr in file_name for fstr in [f"Exper{experiment_num}", analysis_mode_key, file_paths["file_name_suffix"]])
            ]
            dyad_file_path = os.path.join(file_paths["dyad_data"], file_names[dyad_key % len(file_names)])
        else:
            dyad_file_path = _dyad_file_path(dyad_key=dyad_key, file_paths=file_paths, 
                                             experiment_num=experiment_num, analysis_mode=analysis_mode)

        "Load the dyad data from the file"
        with open(dyad_file_path, "r", encoding='utf-8') as file:
            dyad_data = json.load(file)
            if not dyad_data:
                raise Exception(f"Failed to extract dyad data for dyad {dyad_key}")

    else:
        full_path = os.path.join(file_paths["processed"], file_paths[
            "file_names"][f'player_pairs_exper{experiment_num}'])
        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8") as file:
                histories_: Histories = json.load(file)
            histories: DyadGames | PlayerInfo | None = histories_.get('histories', None)
            if histories is None:
                raise Exception(f"Failed to extract player histories for {dyad_key}.")
            
            if isinstance(dyad_key, int):
                dyad_keys = list(histories.keys())
                dyad_key = dyad_keys[dyad_key % len(dyad_keys)]
                dyad_data = histories[dyad_key]
            else:
                dyad_key = _dyad_key(dyad_key=dyad_key)
                dyad_data = histories.get(dyad_key, None)
                if dyad_data is None:
                    dyad_key = _dyad_key(dyad_key=dyad_key, reverse=True)
                    dyad_data = histories.get(dyad_key, None)
                    if dyad_data is None:
                        raise Exception(f"Failed to extract dyad data for {dyad_key}.")         
        else:
            raise Exception(f"File path does not exist: {full_path}.")

    "Process grid-based parameter estimates if they exist"
    for dyad_game in dyad_data:
        param_est = dyad_game.get('parameter_estimates', {}).get('grid', {})
        for player_uuid, role_dict in param_est.items():
            for player_role, role_data in role_dict.items():
                "Process param_vectors"
                if isinstance(role_data, dict):
                    if 'param_vectors' in role_data:
                        role_data['param_vectors'] = {
                            ast.literal_eval(key): val for key, val in role_data['param_vectors'].items()
                        }

                        "Process tickvals"
                        if 'meta_data' in role_data and 'tickvals' in role_data['meta_data']:
                            role_data['meta_data']['tickvals'] = {
                                param_key: np.array([round(val, 9) for val in ticksarray])
                                for param_key, ticksarray in role_data['meta_data']['tickvals'].items()
                            }

    return dyad_data


def all_player_uuids(file_paths: FilePaths, experiment_num: int | None, only_humans: bool = True) -> list[str]:
    """
    Extracts all player uuids for an experiment.

    Arguments:
        • file_paths: FilePaths; Dictionary of file paths.
        • experiment_num: int; Experiment number 1, 2, or 3.
        • only_humans: bool; If True, excludes bots

    Returns:
        • list[str]; List of string player uuids
    """
    if experiment_num in (0, 1, 2, 3):
        experiment_numbers = [experiment_num]
    else:
        experiment_numbers = [0, 1, 2, 3]

    player_uuids = []
    for exper_num in experiment_numbers:
        with open(os.path.join(file_paths['processed'], file_paths['file_names'][
            f'player_pairs_exper{exper_num}']), "r", encoding='utf-8') as file:
            raw_data: dict = json.load(file)     
            player_info: dict = raw_data.get('player_info', {})
            for player_uuid, info in player_info.items():
                if not only_humans or only_humans and info.get('player_type') == 'participant':
                    player_uuids.append(player_uuid)

    return sorted(player_uuids)


def dyads_for_a_player(player_uuid: str | int, experiment_num: int, file_paths: FilePaths, dyad_already_analyzed: bool = False, analysis_mode: str = 'bayesian') -> dict[str: list[DyadGames]]:
    """
    Extracts all the dyads involving a player. 
    
    Note: A dyad is a series of games between the same two players.

    Arguments:
        • player_uuid: str | int; Unique player identifier
            - or integer index of alphabatized uuids ∈ [0, 72].
        • experiment_num: int; Experiment number 1, 2, or 3.
        • file_paths: FilePaths; Dictionary of file paths.
        • dyad_already_analyzed: bool; If True, extracts dyads already run through the analysis pipeline.
        • analysis_mode: str ('mle' | 'bayesian'); Indicates if the analysis pipeline uses the Bayesian model of MLE.

    Returns:
        • dict[str: list[DyadGames]]; Dictionary of dyads indexed by dyad keys
    """
    if not isinstance(player_uuid, (str, int)):
        raise TypeError(f"player_uuid({player_uuid}) must be a string or integeter, not {type(player_uuid)}.")

    if not experiment_num in (0, 1, 2, 3):
        raise ValueError(f"experiment_num({experiment_num}) must be 1, 2, or 3.")

    if experiment_num == 0:
        demo_mode = True  #HACK REMOVE AFTER DEMO
        if demo_mode:
            histories_file_path = os.path.join(ROOT, 'demo_files', 'processed', 
                                           file_paths['file_names'][f'player_pairs_exper0'])

        else:
            histories_file_path = os.path.join(ROOT, 'processed', 
                                           file_paths['file_names'][f'player_pairs_exper0'])

        with open(histories_file_path, "r", encoding='utf-8') as file:
            player_histories: dict = json.load(file)  

            return {
                dyad_key: dyad_games for dyad_key, dyad_games in player_histories['histories'].items() if player_uuid in dyad_key
            }

    plrs_to_dyads = players_to_dyads(experiment_num=experiment_num, file_paths=file_paths, create_new_file=False)
    if isinstance(player_uuid, int):
        players_lst = list(plrs_to_dyads.keys())
        player_uuid = players_lst[player_uuid % len(players_lst)]

    dyad_keys_for_player = plrs_to_dyads.get(player_uuid, [])
    if not dyad_keys_for_player:
        raise Exception(f"Failed to extract dyad keys for player {player_uuid}.")

    dyads_for_a_plr = {}
    for dyad_key in dyad_keys_for_player:
        dyads_for_a_plr[dyad_key] = get_dyad_data(dyad_key=dyad_key, file_paths=file_paths, 
                                                  experiment_num=experiment_num, analysis_mode=analysis_mode, 
                                                  dyad_already_analyzed=dyad_already_analyzed)

    return dyads_for_a_plr


def fitted_dyads_for_a_player(player_uuid: str | int, experiment_num: int, file_paths: FilePaths) -> dict[str, dict[str, Any]] | None:
    """
    Extracts all the dyads involving a player that have already been assigned loss-minimizing parameters. 
    
    Note: A dyad is a series of games between the same two players.

    Arguments:
        • player_uuid: str | int; Unique player identifier
            - or integer index of alphabatized uuids ∈ [0, 72].
        • experiment_num: int; Experiment number 1, 2, or 3.
        • file_paths: FilePaths; Dictionary of file paths.

    Returns:
        • dict[str: list[DyadGames]] | None; Dictionary of dyads indexed by dyad keys
            - Returns None if the files cannot be accessed.
    """
    plr_file_path = ensure_directory_and_join(base_dir=os.path.join(file_paths["player_fits"], f"experiment_{experiment_num}"), 
                                              file_name=f'{file_paths["file_name_suffix"]}_' + player_uuid + ".json")
    if os.path.exists(plr_file_path):
        with open(plr_file_path, "r", encoding='utf-8') as file:
            fitted_player_dyads = json.load(file)
        return fitted_player_dyads

    return None


def players_to_dyads(experiment_num: int, file_paths: FilePaths, create_new_file: bool = False) -> dict[str: list[str]]:
    """
    Creates or extracts a mapping from player uuids to the dyads they were involved in.

    Arguments:
        • experiment_num: int; Experiment number 1, 2, or 3.
        • file_paths: FilePaths; Dictionary of file paths.
        • create_new_file: bool;
            - If True, overwrites existing files.
            - Otherwise, extracts existing files, if any.

    Returns:
        • plrs_to_dyads: dict[str: list[str]]; Example: {
            player_uuid_01: ['(player_uuid_01, player_uuid_15)', '(player_uuid_01, player_uuid_66)',...],
            player_uuid_01: ['(player_uuid_02, player_uuid_34)', '(player_uuid_01, player_uuid_57)',...],
            ...
        }
    """
    plrs_to_dyads_fname = f'players_to_dyads_exper{experiment_num}'
    full_path = os.path.join(file_paths["processed"], file_paths["file_names"][plrs_to_dyads_fname])
    if not create_new_file and os.path.exists(full_path):
        with open(full_path, "r") as file:
            result = json.load(file)
        return result

    with open(os.path.join(ROOT, 'processed', file_paths['file_names'][
        f'player_pairs_exper{experiment_num}']), "r", encoding='utf-8') as file:
        raw_data = json.load(file)   

    histories: dict = raw_data['histories']
    dyad_keys = [_dyad_key(dyad_key=key) for key in histories.keys()]

    def bot_plr(player_uuid: str) -> bool:
        return any(bot_name in player_uuid for bot_name in ('utilitarian', 'selfish', 'competitive', 'masochistic'))

    plrs_to_dyads = {}
    for dkey in dyad_keys:
        player_uuid_1, player_uuid_2 = _dyad_key(dyad_key=dkey, return_tuple=True) 
        if not bot_plr(player_uuid=player_uuid_1):
            if player_uuid_1 not in plrs_to_dyads:
                plrs_to_dyads[player_uuid_1] = []
            plrs_to_dyads[player_uuid_1].append(dkey)
        if not bot_plr(player_uuid=player_uuid_2):
            if player_uuid_2 not in plrs_to_dyads:
                plrs_to_dyads[player_uuid_2] = []
            plrs_to_dyads[player_uuid_2].append(dkey)

    with open(full_path, "w") as file: 
        json.dump(plrs_to_dyads, file, indent=4)  

    return plrs_to_dyads


def serialize_param_vectors(dyad_games: DyadGames, general_settings: dict[str, Any]) -> DyadGames:
    """
    Convert parameter grid data inside a dyad game list into a JSON-serializable form.

    When `update_method` is `'grid'`, parameter vectors are stored as tuple keys and
    numpy arrays, neither of which is directly JSON-serializable.  This function
    converts tuple keys to their string representations and converts numpy arrays to
    plain Python lists, rounding values to 9 decimal places.

    Arguments:
        • dyad_games: DyadGames
            List of per-round game dicts for a single dyad; modified in place.
        • general_settings: dict[str, Any]
            Must contain `'update_method'`; if it is not `'grid'`, the function returns
            immediately without modification.

    Returns:
        • DyadGames — the same list with grid data converted to serializable types.
    """
    if general_settings.get('update_method') != 'grid':
        return dyad_games
    
    first_choo = dyad_games[0]['chooser']
    first_pred = dyad_games[0]['predictor']

    for idx in range(len(dyad_games)):
        dyad_game = dyad_games[idx]
        for player_uuid in [first_choo, first_pred]:
            for player_role in ['chooser', 'predictor']:
                grid_data: dict = dyad_game.get('parameter_estimates', {}).get(
                    'grid', {}).get(player_uuid, {}).get(player_role, {})
                meta_data = grid_data.get('meta_data', None)
                if meta_data is not None:
                    meta_data['tickvals'] = {key: [round(val, 9) for val in ticks_array.tolist()] if isinstance(ticks_array, np.ndarray) 
                                                else [round(val, 9) for val in list(ticks_array)] for key, ticks_array in meta_data['tickvals'].items()}
                param_vectors = grid_data.get('param_vectors', None)   
                if param_vectors is not None:
                    grid_data['param_vectors'] = {str(tuple(x.item() for x in vect_key)): value for vect_key, value in param_vectors.items()}   

    return dyad_games


def create_unified_dataframe(general_settings: GeneralSettings, file_paths: FilePaths, param_info: ParamInfo, experiment_nums: list[int], print_: bool = True) -> pd.DataFrame | None:
    """
    Create a pandas DataFrame from the given 'histories' JSON structure.
    Each row = one meeting. Stores columns for both the chooser's and predictor's
    parameter estimates, with suffixes like 'c_' for chooser, 'p_' for predictor.

    Arguments:
        • all_histories: list[tuple[str, dict]]; Contains the following:
            - histories_data: dict with 'histories' key => the structure from JSON.
            - experiment_num: str, e.g. 'γ1','γ2','γ3'.

    Returns:
        pd.DataFrame with columns:
          [
            'experiment_num', 'dyad_key', 'chooser', 
            'predictor', 'matching_probability', 'round',
            'abdicated_chooser','abdicated_predictor',
            'payoff_A_chooser','payoff_A_predictor',
            'payoff_B_chooser','payoff_B_predictor',
            'choice','prediction','timestamp',
            'c_Vᵢᵢ','c_Vᵢᵢ_se','c_Vᵢⱼ','c_Vᵢⱼ_se',
            'c_Ƹᵢⱼ','c_Ƹᵢⱼ_se','c_loss', ...
            'p_Vᵢᵢ','p_Vᵢᵢ_se', ... 'p_loss'
          ]
    """
    create_new_file = general_settings.get('create_new_file')
    update_method =   general_settings.get('update_method')
    analysis_mode =   general_settings.get('analysis_mode')
    fname_key =       "all_fitted_data_" + analysis_mode

    if not create_new_file:
        df_new = dataframe(file_paths["processed"], file_paths["file_names"][fname_key])
        if df_new is not None:
            if print_:
                print(df_new)
            return df_new

    if experiment_nums is None:
        experiment_nums = [general_settings.get('experiment_num')]

    elif isinstance(experiment_nums, int):
        experiment_nums = [experiment_nums]

    if isinstance(experiment_nums, list) and all(num in (1, 2, 3) for num in experiment_nums):
        exper_num_and_player_uuids = []
        for num in experiment_nums:
            player_uuids = all_player_uuids(
                file_paths=file_paths, experiment_num=num, only_humans=True)
            nums = [num for player_uuid in player_uuids]
            "List of all player uuids in by experiment number."
            exper_num_and_player_uuids += list(zip(nums, player_uuids))
    else:
        raise ValueError(f"Invalid experiment_nums: {experiment_nums}. Use list of integers.")

    n_players = len(exper_num_and_player_uuids)
    
    histories_by_exper = {num: {} for num in experiment_nums}
    player_info_by_exper  = {num: {} for num in experiment_nums}

    "Iterate through all players appending their parameters to rows."
    for player_idx, (exper_num, player_uuid) in enumerate(exper_num_and_player_uuids):
        print(f'Adding data for player {player_idx} / {n_players}: {player_uuid}')
        dyads_for_this_player = fitted_dyads_for_a_player(
            player_uuid=player_uuid, experiment_num=exper_num, file_paths=file_paths)
        if dyads_for_this_player is None:
            raise Exception(f"Failed to extract data for player {player_uuid}")

        with open(os.path.join(ROOT, file_paths['file_names'][
            f'player_pairs_exper{exper_num}']), "r", encoding='utf-8') as file:
            raw_data: dict[str, dict[str, str]] = json.load(file)

        if 'player_info' not in raw_data:
            raise Exception(f"'player_info' not found!")
        player_info_by_exper[exper_num] = raw_data['player_info']

        for dyad_key, meeting_list in dyads_for_this_player.items():
            dyad_key = _dyad_key(dyad_key=dyad_key, return_tuple=False)
            if dyad_key in histories_by_exper[exper_num]:
                continue

            player_uuid_1, player_uuid_2 = _dyad_key(dyad_key=dyad_key, return_tuple=True)
            player_info_1: dict = player_info_by_exper[exper_num].get(player_uuid_1, {})
            player_info_2: dict = player_info_by_exper[exper_num].get(player_uuid_2, {})
            if player_info_1.get('player_type') == "robot":
                continue
            if player_info_2.get('player_type') == "robot":
                continue

            dyads_for_player_1 = fitted_dyads_for_a_player(
                player_uuid=player_uuid_1, experiment_num=exper_num, file_paths=file_paths)
            if dyads_for_player_1 is None:
                raise Exception(f"Failed to extract data for player 1 {player_uuid_1}")
            dyads_for_player_2 = fitted_dyads_for_a_player(
                player_uuid=player_uuid_2, experiment_num=exper_num, file_paths=file_paths)
            if dyads_for_player_2 is None:
                raise Exception(f"Failed to extract data for player 2 {player_uuid_2}")

            for dyad_key_1 in dyads_for_player_1.keys():
                sorted_dyad_key_1 = _dyad_key(dyad_key=dyad_key_1, return_tuple=False)
                if sorted_dyad_key_1 != dyad_key_1:
                    dyads_for_player_1[sorted_dyad_key_1] = dyads_for_player_1.pop(dyad_key_1)
            for dyad_key_2 in dyads_for_player_2.keys():
                sorted_dyad_key_2 = _dyad_key(dyad_key=dyad_key_2, return_tuple=False)
                if sorted_dyad_key_2 != dyad_key_2:
                    dyads_for_player_2[sorted_dyad_key_2] = dyads_for_player_2.pop(dyad_key_2)

            dyad_games_1 = dyads_for_player_1.get(dyad_key, None)
            dyad_games_2 = dyads_for_player_2.get(dyad_key, None)
            if dyad_games_1 is None or dyad_games_2 is None:
                raise Exception(f"Mismatch between stored data for players {player_uuid_1} and {player_uuid_2}")

            for dyad_game, dyad_game_1, dyad_game_2 in zip(meeting_list, dyad_games_1, dyad_games_2):
                param_est_1 = dyad_game_1.get('parameter_estimates', {}).get(update_method, {}).get(player_uuid_1, {})
                param_est_2 = dyad_game_2.get('parameter_estimates', {}).get(update_method, {}).get(player_uuid_2, {})
                if not param_est_1 or not param_est_2:
                    raise Exception(f"Mismatch between stored data for players {player_uuid_1} and {player_uuid_2}")
                dyad_game['parameter_estimates'][update_method] = {
                    player_uuid_1: param_est_1,
                    player_uuid_2: param_est_2
                }
            histories_by_exper[exper_num][dyad_key] = meeting_list

    all_dfs = []
    for exper_num in experiment_nums:
        rows = []
        histories = histories_by_exper[exper_num]
        player_info = player_info_by_exper[exper_num]
        if not histories or not player_info:
            continue

        dyad_losses, dyad_errors = {}, {}    
        for dyad_key, meeting_list in histories.items():
            dyad_losses[dyad_key] = {}
            dyad_errors[dyad_key] = {'cc': {}, 'cp': {}, 'pc': {}, 'pp': {}}
            for meeting_idx, mtg in enumerate(meeting_list):
                mtg: dict
                rowdict: dict = {}
                chooser_uuid, predictor_uuid = mtg.get('chooser'), mtg.get('predictor')          
                rowdict['experiment_num']       = experiment_num
                rowdict['dyad_key']             = dyad_key
                rowdict['chooser']              = chooser_uuid
                rowdict['predictor']            = predictor_uuid
                player_info_c                   = player_info.get(chooser_uuid, None)
                player_info_p                   = player_info.get(predictor_uuid, None)
                rowdict['player_type_c']        = player_info_c.get('player_type', None)
                rowdict['player_type_p']        = player_info_p.get('player_type', None)       
                rowdict['avatar_shape_c']       = player_info_c.get('avatar_shape', 'unknown')
                rowdict['avatar_shape_p']       = player_info_p.get('avatar_shape', 'generic')      
                rowdict['avatar_color_c']       = player_info_c.get('avatar_color', 'unknown')
                rowdict['avatar_color_p']       = player_info_p.get('avatar_color', 'generic')          

                rowdict['meeting_idx']          = meeting_idx
                rowdict['round']                = mtg.get('round', None)
                rowdict['timestamp']            = mtg.get('timestamp', None)
                rowdict['matching_probability'] = mtg.get('matching_probability', None)
                rowdict['abdicated_chooser']    = mtg.get('abdicated_chooser', False)
                rowdict['abdicated_predictor']  = mtg.get('abdicated_predictor', False)
                rowdict['payoff_A_chooser']     = mtg.get('payoff_A_chooser', None)
                rowdict['payoff_A_predictor']   = mtg.get('payoff_A_predictor', None)
                rowdict['payoff_B_chooser']     = mtg.get('payoff_B_chooser', None)
                rowdict['payoff_B_predictor']   = mtg.get('payoff_B_predictor', None)
                rowdict['choice']               = mtg.get('choice', None)
                rowdict['prediction']           = mtg.get('prediction', None)
                
                "Parse the parameter_estimates dict for both chooser & predictor"
                "Prefixes: c = chooser and p = predictor."
                cc_params, cp_params = {}, {}
                pc_params, pp_params = {}, {} 
                cc_errors, cp_errors = {}, {}                  
                pc_errors, pp_errors = {}, {}
                cc_loss, cp_loss     = None, None
                pc_loss, pp_loss     = None, None

                param_estimates = mtg.get('parameter_estimates', {}).get(update_method, {})
                param_est_choo = param_estimates.get(chooser_uuid, None)
                param_est_pred = param_estimates.get(predictor_uuid, None)
                if param_est_choo:
                    cc_data = param_est_choo.get('chooser', None)
                    if cc_data:
                        for param_key, val in cc_data['params'].items():
                            cc_params[param_key] = val                        
                        cc_loss = cc_data['loss'] if 'loss_final' in cc_data else cc_data.get('output', {}).get('loss_final', None) 
                        if cc_loss:
                            dyad_losses[dyad_key]['cc_loss'] = cc_loss
                        else:
                            cc_loss = dyad_losses[dyad_key].get('cc_loss', None)
                        cc_std_errors = cc_data.get('std_errors', {})
                        if cc_std_errors:
                            for param_key, val in cc_std_errors.items():
                                cc_errors[param_key] = dyad_errors[dyad_key]['cc'][param_key] = val
                        else:
                            for param_key in cc_data['params'].keys():
                                cc_errors[param_key] = dyad_errors[dyad_key]['cc'].get(param_key, None)
                    cp_data = param_est_choo.get('predictor', None)
                    if cp_data:
                        for param_key, val in cp_data['params'].items():
                            cp_params[param_key] = val                        
                        cp_loss = cp_data['loss'] if 'loss_final' in cp_data else cp_data.get('output', {}).get('loss_final', None) 
                        if cp_loss:
                            dyad_losses[dyad_key]['cp_loss'] = cp_loss
                        else:
                            cp_loss = dyad_losses[dyad_key].get('cp_loss', None)
                        cp_std_errors = cp_data.get('std_errors', {})
                        if cp_std_errors:
                            for param_key, val in cp_std_errors.items():
                                cp_errors[param_key] = dyad_errors[dyad_key]['cp'][param_key] = val
                        else:
                            for param_key in cp_data['params'].keys():
                                cp_errors[param_key] = dyad_errors[dyad_key]['cp'].get(param_key, None)
                if param_est_pred:
                    pc_data = param_est_pred.get('chooser', None)
                    if pc_data:
                        for param_key, val in pc_data['params'].items():
                            pc_params[param_key] = val                        
                        pc_loss = pc_data['loss'] if 'loss_final' in pc_data else pc_data.get('output', {}).get('loss_final', None) 
                        if pc_loss:
                            dyad_losses[dyad_key]['pc_loss'] = pc_loss
                        else:
                            pc_loss = dyad_losses[dyad_key].get('pc_loss', None)
                        pc_std_errors = pc_data.get('std_errors', {})
                        if pc_std_errors:
                            for param_key, val in pc_std_errors.items():
                                pc_errors[param_key] = dyad_errors[dyad_key]['pc'][param_key] = val
                        else:
                            for param_key in pc_data['params'].keys():
                                pc_errors[param_key] = dyad_errors[dyad_key]['pc'].get(param_key, None)
                    pp_data = param_est_pred.get('predictor', None)
                    if pp_data:
                        for param_key, val in pp_data['params'].items():
                            pp_params[param_key] = val     
                        pp_loss = pp_data['loss'] if 'loss_final' in pp_data else pp_data.get('output', {}).get('loss_final', None)                
                        if pp_loss:
                            dyad_losses[dyad_key]['pp_loss'] = pp_loss
                        else:
                            pp_loss = dyad_losses[dyad_key].get('pp_loss', None)                        
                        pp_std_errors = pp_data.get('std_errors', {})
                        if pp_std_errors:
                            for param_key, val in pp_std_errors.items():
                                pp_errors[param_key] = dyad_errors[dyad_key]['pp'][param_key] = val
                        else:
                            for param_key in pp_data['params'].keys():
                                pp_errors[param_key] = dyad_errors[dyad_key]['pp'].get(param_key, None)

                for param_key in param_info["keys"]:
                    rowdict[f'cc_{param_key}'] = cc_params.get(param_key, None)
                    rowdict[f'cp_{param_key}'] = cp_params.get(param_key, None)
                    rowdict[f'pc_{param_key}'] = pc_params.get(param_key, None)
                    rowdict[f'pp_{param_key}'] = pp_params.get(param_key, None)

                rowdict['cc_loss'] = cc_loss
                rowdict['cp_loss'] = cp_loss
                rowdict['pc_loss'] = pc_loss
                rowdict['pp_loss'] = pp_loss

                for param_key in param_info["keys"]:    
                    rowdict[f'cc_{param_key}_se'] = cc_errors.get(param_key, None)
                    rowdict[f'cp_{param_key}_se'] = cp_errors.get(param_key, None)
                    rowdict[f'pc_{param_key}_se'] = pc_errors.get(param_key, None)
                    rowdict[f'pp_{param_key}_se'] = pp_errors.get(param_key, None)

                rows.append(rowdict)

        all_dfs.append(pd.DataFrame(rows))

    df = pd.concat(all_dfs, ignore_index=True)

    df.to_csv(os.path.join(file_paths["processed"], file_paths["file_names"][fname_key]), 
              index=False, encoding='utf-8-sig')

    if print_:
        print(df)
    return df


"=========================================================================================="
"=================================== Preprocess Raw Data =================================="
"=========================================================================================="


def preprocessing1(df: pd.DataFrame, column_names: ColumnNames, file_paths: FilePaths, create_new_file: bool = False) -> pd.DataFrame:
    """
    Preprocesses raw data from experiment 1.

    Arguments:
        • df: pandas.DataFrame; The raw data.
        • create_new_file: bool; If true, creates a new dataframe even if one already exists.

    Returns:
        • df: pandas.DataFrame; The preprocessed data.         
    """
    if not create_new_file:
        df_new = dataframe(file_paths["processed"], file_paths["file_names"]['processed_data_exper1'])
        if df_new is not None:
            return df_new

    "Select relevant columns"
    df = df[list(column_names["exper_1A"].keys())]
    df.rename(columns=column_names["exper_1A"], inplace=True)
    df['round'] = df['round'].fillna(-1).astype(int)
    df['player_uuid_p0'] = df['player_uuid_p0'].fillna(-1).astype(int)
    df['player_uuid_p0'] = df['player_uuid_p0'].apply(lambda x: 'robot-' + str(x))

    df.insert(loc=11, column='choice__p0', value=df.apply(lambda row: 'A', axis=1))

    df['prediction__p1'] = df['prediction__p1'].apply(lambda x: 'A' if x == 1 else 'B')   

    df.to_csv(os.path.join(file_paths["processed"], file_paths["file_names"]['processed_data_exper1']), 
              index=False, encoding='utf-8-sig')

    return df


def preprocessing2(df: pd.DataFrame, column_names: ColumnNames, file_paths: FilePaths, create_new_file: bool = False) -> pd.DataFrame:
    """
    Preprocesses raw data from experiment 2.

    Arguments:
        • df: pandas.DataFrame; The raw data.
        • create_new_file: bool; If true, creates a new dataframe even if one already exists.

    Returns:
        • df: pandas.DataFrame; The preprocessed data.         
    """
    if not create_new_file:
        print('cat'), exit()
        df_new = dataframe(file_paths["processed"], file_paths["file_names"]['processed_data_exper2'])
        if df_new is not None:
            return df_new

    "Select relevant columns"
    df = df[list(column_names["exper_2A"].keys())]
    df.rename(columns=column_names["exper_2A"], inplace=True)
    df['round'] = df['round'].fillna(-1).astype(int)

    "Create 'choice' column"
    def choice_col(As: int, Ao: int, Bs: int, Bo: int,
                   Cs: int, Co: int, Rs: int, Ro: int) -> str:
        """
        Determine which option (A or B) the chooser selected by comparing offered payoffs to chosen and rejected payoffs.

        Arguments:
            • As, Ao: int — self and other payoffs for option A.
            • Bs, Bo: int — self and other payoffs for option B.
            • Cs, Co: int — self and other payoffs of the chosen option.
            • Rs, Ro: int — self and other payoffs of the rejected option.

        Returns:
            • str — `'A'` if option A was chosen, `'B'` if option B was chosen, `'N'` if indeterminate.
        """
        if As == Cs and Ao == Co and Bs == Rs and Bo == Ro:
            return 'A'
        elif As == Rs and Ao == Ro and Bs == Cs and Bo == Co:
            return 'B'
        else:
            return 'N'
        
    df['choice__p0'] = df.apply(lambda x: choice_col(
        x.payoffs_A_p0_op, x.payoffs_A_p1_op, x.payoffs_B_p0_op, x.payoffs_B_p1_op, 
        x.payoffs_selected_p0, x.payoffs_selected_p1, x.payoffs_rejected_p0, x.payoffs_rejected_p1), axis=1)    
    
    df['prediction__p1'] = df['prediction__p1'].apply(lambda x: 'A' if x == 1 else 'B')   
    
    "The player uuid for the avatar is a composite of their number, type, and shape."
    df['player_uuid_p0'] = df.apply(lambda x: str(x.player_uuid_p0) + '-' + x.avatar_type.replace(
        ' Avatar', '').lower() + '-' + x.avatar_shape.replace(' ', '-').lower(), axis=1)      

    df = df[column_names["exper_2B"]]

    df.to_csv(os.path.join(file_paths["processed"], file_paths["file_names"]['processed_data_exper2']), 
              index=False, encoding='utf-8-sig')
    
    return df


def preprocessing3(df: pd.DataFrame, column_names: ColumnNames, file_paths: FilePaths, create_new_file: bool = False) -> pd.DataFrame:
    """
    Preprocesses raw data from experiment 3.

    Arguments:
        • df: pandas.DataFrame; The raw data.
        • create_new_file: bool; If true, creates a new dataframe even if one already exists.

    Returns:
        • df: pandas.DataFrame; The preprocessed data.         
    """
    def find_batch_groups(batch_tuples: list[tuple[int]]) -> list[tuple[int]]:
        """
        A union-find / disjoint set data structure can help group connected components.

        Note: In experiment 3, participants played in batches and this function 
        finds groups of subjects that participanted at the same time (same batch).

        1) Initialize a union-find structure for all unique individuals found.
        2) For each tuple, union all the elements.
        3) After processing, extract the connected components.    
        """
        "Collect all unique individuals"
        individuals = set()
        for group in batch_tuples:
            for person in group:
                individuals.add(person)

        "Create a union-find structure"
        parent = {x: x for x in individuals}
        rank = {x: 0 for x in individuals}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rootX = find(x)
            rootY = find(y)
            if rootX != rootY:
                "Union by rank"
                if rank[rootX] < rank[rootY]:
                    parent[rootX] = rootY
                elif rank[rootX] > rank[rootY]:
                    parent[rootY] = rootX
                else:
                    parent[rootY] = rootX
                    rank[rootX] += 1

        "Union all sets within each tuple"
        for group in batch_tuples:
            "If group has more than one person, union them pairwise"
            if len(group) > 1:
                group = list(group)
                base = group[0]
                for other in group[1:]:
                    union(base, other)

        "Once all unions are done, find connected components"
        components = {}
        for x in individuals:
            root = find(x)
            if root not in components:
                components[root] = []
            components[root].append(x)

        "Sort each component and then convert to tuple"
        batch_groups = []
        for comp in components.values():
            batch_groups.append(tuple(sorted(comp)))

        "Sort the list of tuples by their first element (just for a consistent ordering)"
        batch_groups.sort(key=lambda t: t[0])

        return batch_groups

    if not create_new_file:
        print('dog')
        df_new = dataframe(file_paths["processed"], file_paths["file_names"]['processed_data_exper3'])
        if df_new is not None:
            return df_new

    "Select relevant columns"
    df = df[column_names["exper_3A"]]
    df['batch'] = df['batch'].fillna(-1).astype(int)

    "Extract relevant rows for round 1"
    df_rnd_1 = df[df['round'] == 1]
    
    "Create separate columns for player 1 and player 2 UUIDs and types"
    df_rnd_1['player1_uuid'] = df_rnd_1['player_uuids'].apply(lambda x: ast.literal_eval(x)[0])
    df_rnd_1['player2_uuid'] = df_rnd_1['player_uuids'].apply(lambda x: ast.literal_eval(x)[1])
    df_rnd_1['player1_type'] = df_rnd_1['player_types'].apply(lambda x: ast.literal_eval(x)[0])
    df_rnd_1['player2_type'] = df_rnd_1['player_types'].apply(lambda x: ast.literal_eval(x)[1])

    "Create a mapping of batches to participant UUIDs"
    batch_to_uuids = (
        df_rnd_1[df_rnd_1['player1_type'] == 'participant']
        .groupby('batch')['player1_uuid']
        .apply(set)
        .to_dict()
    )

    uuids_to_batches = {}
    for batch, uuids in batch_to_uuids.items():
        for uuid in list(uuids):
            if uuid not in uuids_to_batches:
                uuids_to_batches[uuid] = []
            uuids_to_batches[uuid].append(batch)

    batch_groups = set()
    for batches in uuids_to_batches.values():
        batch_groups.add(tuple(sorted(batches)))

    batch_groups = find_batch_groups(batch_tuples=batch_groups)

    batch_to_timeslot = {}
    for idx, group in enumerate(batch_groups):
        for batch in group:
            batch_to_timeslot[batch] = idx

    df.insert(loc=0, column='timeslot', value=df.apply(
        lambda row: batch_to_timeslot[row.batch], axis=1))

    df.rename(columns={'players_abdicated': 'abdicated_'}, inplace=True)

    for col_name in ['player_uuids', 'player_types', 
                     'avatar_shapes', 'avatar_colors', 'abdicated_']:
        for idx in [0, 1]:
            new_col_name = col_name[:-1] + f'_p{idx}'
            df[new_col_name] = df[col_name].apply(
                lambda x: ast.literal_eval(x)[idx])

    df['matching_probability'] = df['adjacency_matrix'].apply(
        lambda x: ast.literal_eval(x)[0][1]).fillna(-1).astype(float)
    
    "Fixing abdication data"
    df['choice_data__p0'] = df['choice_data__p0'].apply(lambda x: x.replace("'rtype': , ", "'rtype': None, "))
    df['prediction_data__p1'] = df['prediction_data__p1'].apply(lambda x: x.replace("'rtype': , ", "'rtype': None, "))
    df['abdicated_p0'] = df['choice_data__p0'].apply(lambda x: ast.literal_eval(x)['rtype'] is None)
    df['abdicated_p1'] = df['prediction_data__p1'].apply(lambda x: ast.literal_eval(x)['rtype'] is None)

    df = df[column_names["exper_3B"]]

    df.to_csv(os.path.join(file_paths["processed"], file_paths["file_names"]['processed_data_exper3']), 
              index=False, encoding='utf-8-sig')

    return df


def player_histories(df: pd.DataFrame, experiment_num: int, file_paths: FilePaths, create_new_file: bool = False, omit_bot_pairs: bool = True) -> Histories:
    """
    Reformats the experiment data into a dictionary of player pair histories.

    Arguments:
        • df: pd.DataFrame; Preprocessed dataframe.
        • experiment_num: int (literal); Experiment 1, 2, or 3.
        • create_new_file: bool; If true, creates a new JSON even if one already exists.
        • omit_bot_pairs: bool; If true, removes bot-to-bot pairings from 'histories.

    Returns:
        • result: dict; A dictionary containing:
            {
                "histories": {
                    (uuid1, uuid2): [
                       {
                         'chooser': str,
                         'predictor': str,
                         'payoff_A_chooser': int,
                         'payoff_A_predictor': int,
                         'payoff_B_chooser': int,
                         'payoff_B_predictor': int,
                         'choice': 'A' or 'B',
                         'prediction': 'A' or 'B',
                         'abdicated_chooser': bool,
                         'abdicated_predictor': bool,
                         'matching_probability': float,
                         'timestamp': int,
                         'round': int,
                       },
                       ...
                    ],
                    ...
                },
                "player_info": {
                    uuid: {
                        'player_type': 'participant' or 'robot',
                        'avatar_shape': str,
                        'avatar_color': str
                    },
                    ...
                }
            }
    """
    if experiment_num not in [1, 2, 3]:
        raise ValueError(f"experiment_num must be 1, 2, or 3, not {experiment_num}!")

    if not create_new_file:
        full_path = os.path.join(file_paths["processed"], file_paths["file_names"][f'player_pairs_exper{experiment_num}'])
        if os.path.exists(full_path):
            with open(full_path, "r") as file:
                result = json.load(file)
            return result

    if experiment_num in [1, 2]:
        required_columns = [
            'player_uuid_p0', 'player_uuid_p1', 
            'payoffs_A_p0_op', 'payoffs_A_p1_op', 
            'payoffs_B_p0_op', 'payoffs_B_p1_op', 
            'payoffs_A_p0_rp', 'payoffs_A_p1_rp', 
            'payoffs_B_p0_rp', 'payoffs_B_p1_rp', 
            'choice__p0', 'prediction__p1', 
            'round', 
        ]

    else:
        required_columns = [
            'player_uuid_p0', 'player_uuid_p1', 
            'player_type_p0', 'player_type_p1', 
            'avatar_shape_p0', 'avatar_shape_p1', 
            'avatar_color_p0', 'avatar_color_p1', 
            'payoffs_A_p0', 'payoffs_A_p1', 
            'payoffs_B_p0', 'payoffs_B_p1', 
            'choice__p0', 'prediction__p1', 
            'abdicated_p0', 'abdicated_p1', 
            'matching_probability', 
            'timestamp', 'round', 
        ]

    "Validating inputs."
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col} in the dataframe.")

    player_info = {}
    histories = {}
    
    "Iterate over rows."
    for _, row in df.iterrows():
        uuid_p0 = row['player_uuid_p0']
        uuid_p1 = row['player_uuid_p1']

        "Update player_info if not present"
        if uuid_p0 not in player_info:
            player_info[uuid_p0] = {
                'player_type': 'robot',
            }
            if experiment_num == 3:
                player_info[uuid_p0]['avatar_shape'] = row['avatar_shape_p0']
                player_info[uuid_p0]['avatar_color'] = row['avatar_color_p0']

        if uuid_p1 not in player_info:
            player_info[uuid_p1] = {
                'player_type': 'participant',
            }
            if experiment_num == 3:
                player_info[uuid_p1]['avatar_shape'] = row['avatar_shape_p1']
                player_info[uuid_p1]['avatar_color'] = row['avatar_color_p1']

        if omit_bot_pairs and experiment_num == 3:
            "Excluding bot-bot pairs from histories."
            p0_is_bot = player_info[uuid_p0]['player_type'] == 'robot'
            p1_is_bot = player_info[uuid_p1]['player_type'] == 'robot'  
            if p0_is_bot and p1_is_bot:
                continue      

        "Sort the tuple of uuids so that each pair is stored consistently."
        pair_key = tuple(sorted([uuid_p0, uuid_p1]) if experiment_num == 3 else [uuid_p0, uuid_p1])
        pair_history = histories.get(pair_key, [])
        if not pair_history:
            histories[pair_key] = []

        if experiment_num in [1, 2]:
            round_data_op = {
                'chooser': uuid_p0,
                'predictor': uuid_p1,
                'matching_probability': 1.0,
                'payoff_A_chooser': int(row['payoffs_A_p0_op']),
                'payoff_A_predictor': int(row['payoffs_A_p1_op']),
                'payoff_B_chooser': int(row['payoffs_B_p0_op']),
                'payoff_B_predictor': int(row['payoffs_B_p1_op']),
                'choice': row['choice__p0'], 'prediction': None, 
                'round': int(row['round']) * 2,
                'phase': 'op',
            }

            round_data_rp = {
                'chooser': uuid_p0,
                'predictor': uuid_p1,
                'matching_probability': 0.0,
                'payoff_A_chooser': int(row['payoffs_A_p0_rp']),
                'payoff_A_predictor': int(row['payoffs_A_p1_rp']),
                'payoff_B_chooser': int(row['payoffs_B_p0_rp']),
                'payoff_B_predictor': int(row['payoffs_B_p1_rp']),
                'choice': None, 'prediction': row['prediction__p1'], 
                'round': int(row['round']) * 2 + 1,
                'phase': 'rp',
            }

            histories[pair_key].append(round_data_op)
            histories[pair_key].append(round_data_rp)

        else:    
            round_data = {
                'chooser': uuid_p0,
                'predictor': uuid_p1,
                'matching_probability': float(row['matching_probability']),
                'payoff_A_chooser': int(row['payoffs_A_p0']),
                'payoff_A_predictor': int(row['payoffs_A_p1']),
                'payoff_B_chooser': int(row['payoffs_B_p0']),
                'payoff_B_predictor': int(row['payoffs_B_p1']),
                'choice': row['choice__p0'],  # 'A' or 'B'
                'prediction': row['prediction__p1'], # 'A', 'B',
                'abdicated_chooser': bool(row['abdicated_p0']),
                'abdicated_predictor': bool(row['abdicated_p1']),
                'timestamp': int(row['timestamp']),
                'round': int(row['round']),
                'batch': int(row['batch']),
            }

            histories[pair_key].append(round_data)

    "Once done, this has histories for each pair and player_info."

    "Sort each pair's history by round_number just in case"
    for pair in histories:
        histories[pair].sort(key=lambda x: x['timestamp' if experiment_num == 3 else 'round'])
        if experiment_num == 3:
            min_batch = min([meeting['batch'] for meeting in histories[pair]])
            for meeting in histories[pair]:
                batch = meeting['batch'] - min_batch
                meeting['round'] = meeting['round'] + batch * 40
                del meeting['batch']
   
    result = {
        "histories": {f"({pair[0]}, {pair[1]})": val for pair, val in histories.items()},
        "player_info": player_info
    }

    with open(os.path.join(file_paths["processed"], file_paths[
        "file_names"][f'player_pairs_exper{experiment_num}']), "w") as file: 
        json.dump(result, file, indent=4)  

    return result


def all_histories_raw(column_names: ColumnNames, file_paths: FilePaths) -> List[pd.DataFrame]:
    """
    Load all data from all experiments. DEPRICATED Used to load raw data, which is no longer on this repo.
    """
    experiments = [1, 2, 3]
    "Try processed/ first, fall back to raw_data/ (some CSVs live in processed/ rather than raw_data/)."
    pre_dfs = []
    for exper in experiments:
        fname = file_paths["file_names"][f"raw_data_exper{exper}"]
        df = dataframe(file_paths["processed"], fname) or dataframe(file_paths["raw_data"], fname)
        pre_dfs.append(df)
    df1 = preprocessing1(df = pre_dfs[0], column_names = column_names, file_paths = file_paths, create_new_file = False)
    df2 = preprocessing2(df = pre_dfs[1], column_names = column_names, file_paths = file_paths, create_new_file = False)
    df3 = preprocessing3(df = pre_dfs[2], column_names = column_names, file_paths = file_paths, create_new_file = False)
    return [player_histories(df=[df1, df2, df3][exper - 1], experiment_num=exper, file_paths=file_paths,
                                  create_new_file = False) for exper in experiments]  


def all_histories(file_paths: FilePaths, experiment_numbers=[1, 2, 3]) -> List[pd.DataFrame]:
    """
    Load all data from all experiments.
    """
    return [dataframe(file_paths["processed"], file_paths[
        "file_names"][f"processed_data_exper{exper}"]) for exper in experiment_numbers]