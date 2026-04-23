from simulation import *
import hashlib

"=========================================================================================="
"============================== Illustrating Belief Updates ==============================="
"=========================================================================================="

def visualize_bayesian_updates_2d(player_uuid: str | int, counterpart_uuid: str | int, player_role: PlayerRole, general_settings: GeneralSettings, 
                                  utility_settings: UtilitySettings, file_paths: FilePaths, fig_lay: FigLay, n_rounds: int = 5, dark_zero_lines: bool = True) -> None:
    """
    Generate a 3-row × N-column Plotly figure visualizing Bayesian belief updating in 2D.

    For each of the N rounds (columns), three panels are shown:
        Row 1: The observed binary choice plotted in (Vᵢᵢ, Vᵢⱼ) payoff-difference space,
               with the indifference boundary drawn as a line separating A and B regions.
        Row 2: The likelihood heatmap over (Vᵢᵢ, Vᵢⱼ) for that round's observation.
        Row 3: The prior (before the round) and posterior (after the round) probability
               mass distributions over the parameter grid.

    Arguments:
        • player_uuid: str | int
            UUID string or integer index into the sorted player list for the dyad to visualize.
        • counterpart_uuid: str | int
            UUID string or integer index for the chooser whose behavior is being predicted.
        • player_role: PlayerRole
            Role to visualize: 'predictor' shows belief updating; 'chooser' shows choice probabilities.
        • general_settings: GeneralSettings
            Must include 'experiment_num', 'update_method', 'n_bins_per_dimension', and related keys.
        • utility_settings: UtilitySettings
            Boolean toggles selecting the active utility functional form.
        • file_paths: FilePaths
            Directory paths used to locate the player's saved parameter grid files.
        • fig_lay: FigLay
            Figure layout template controlling fonts, colors, colorscales, and sizing.
        • n_rounds: int
            Number of rounds (columns) to display. Rounds are selected from the filtered game list.
        • dark_zero_lines: bool
            If True, the zero lines in the top-row payoff-difference panels are drawn in a
            darker color to make the indifference boundary more visually prominent.

    Returns:
        • None — the figure is exported as an HTML file to file_paths['visuals'].
    """
    "1) Filter the dyad games for relevant rounds."
    experiment_num = general_settings.get('experiment_num', 3)
    player_uuids = prep.all_player_uuids(file_paths=file_paths, experiment_num=experiment_num, only_humans=False)
    if experiment_num == 0:
        player_uuids = [uuid for uuid in player_uuids if 'predictor' in uuid]

    "Convert player_uuid index => actual string"
    if isinstance(player_uuid, int):
        plrs_to_dyads = prep.players_to_dyads(experiment_num=experiment_num,
                                                    file_paths=file_paths, create_new_file=False)
        plr_keys = sorted(list(plrs_to_dyads.keys()))
        if experiment_num == 0:
            plr_keys = [key for key in plr_keys if 'chooser' not in key]
        player_uuid = plr_keys[player_uuid % len(plr_keys)]

    elif isinstance(player_uuid, str):
        if player_uuid not in player_uuids:
            raise IndexError(f"player_uuid '{player_uuid}' not found among {player_uuids}.")

    else:
        raise ValueError(f"player_uuid type {type(player_uuid)} not supported.")

    "Load the player's data file."
    plr_file_path = os.path.join(
        file_paths["player_fits"], f"experiment_{experiment_num}",
        player_uuid + ".json"
    )

    if not os.path.exists(plr_file_path):
        raise FileNotFoundError(f"File not found for player: {player_uuid} => {plr_file_path}")

    with open(plr_file_path, "r", encoding='utf-8') as file:
        player_dyads = json.load(file)
    if player_dyads is None:
        raise Exception(f"Failed to extract dyads for player {player_uuid}")

    dyad_keys = sorted(list(player_dyads.keys()))

    dyad_key = None
    if isinstance(counterpart_uuid, int):
        dyad_key = dyad_keys[counterpart_uuid % len(dyad_keys)]
        plr_uuid_1, plr_uuid_2 = prep._dyad_key(dyad_key=dyad_key, return_tuple=True)
        counterpart_uuid = plr_uuid_2 if plr_uuid_1 == player_uuid else plr_uuid_1
        
    found_counterpart = False
    for dkey in dyad_keys:
        if counterpart_uuid in dkey:
            found_counterpart = True
            break

    if not found_counterpart:
        "Raise a clear error when the counterpart is unavailable."
        raise Exception(f"Counterpart {counterpart_uuid} not in {dyad_keys}")

    if dyad_key is None:
        dyad_key = prep._dyad_key(dyad_key=f"({player_uuid}, {counterpart_uuid})")

    dyad_games = player_dyads[dyad_key]
    "Filter games by the desired role"
    filtered_games = [game for game in dyad_games if game.get(player_role, None) == player_uuid]
    if not filtered_games:
        raise Exception("Failed to extract filtered dyad games for role='{}'.".format(player_role))

    n_filtered_games = len(filtered_games)
    if n_rounds > n_filtered_games:
        n_rounds = n_filtered_games

    "2) Prepare subplots: 3 rows, n_rounds columns"
    fig = make_subplots(
        rows=3, cols=n_rounds,
        horizontal_spacing=0.05, vertical_spacing=0.07,
        subplot_titles=tuple([f"Round {rval}" for rval in range(
            1, n_rounds + 1)] + ["" for rval in range(1, (n_rounds + 1) * 2)])
    )

    max_prior_prob = 0.0
    for game in filtered_games:
        grid_predictor = game.get("parameter_estimates", {}).get(
            "grid", {}).get(player_uuid, {}).get("predictor", {}).get('parameter_vectors', {})        
        for probability in grid_predictor.values():
            if probability > max_prior_prob:
                max_prior_prob = probability

    "3) For each round => create 3 subplots"
    for col_idx, game in enumerate(filtered_games, start=1):
        if col_idx > n_rounds:
            break
        "(A) Top row => observed choice in payoff-difference space."
        payoff_As = game.get("payoff_A_chooser", 0)
        payoff_Bs = game.get("payoff_B_chooser", 0)
        payoff_Ao = game.get("payoff_A_predictor", 0)
        payoff_Bo = game.get("payoff_B_predictor", 0)

        if game.get("choice") == "A":
            payoff_diff_self = payoff_As - payoff_Bs
            payoff_diff_other = payoff_Ao - payoff_Bo
        else:
            payoff_diff_self = payoff_Bs - payoff_As
            payoff_diff_other = payoff_Bo - payoff_Ao

        dot_trace = go.Scatter(
            x=(payoff_diff_self,),
            y=(payoff_diff_other,),
            mode="markers",
            marker=dict(size=16, color="green"),
            showlegend=False,
            hovertemplate=(
                f"<b>Observed Choice:</b> {game.get('choice')}<br>"
                f"ΔSelf: {payoff_diff_self}<br>"
                f"ΔOther: {payoff_diff_other}<extra></extra>"
            )
        )
        fig.add_trace(dot_trace, row=1, col=col_idx)

        """
        (B) Middle row => likelihood heatmap.
        Reconstructs the Nx x Ny array:
            1) Build the param grid from meta_data/tickvals
            2) For each point, compute p(choose A) or 1 - p(choose A)
        """

        "Replicates logic from \"3D\" code but produces a 2D array."
        grid_predictor = game.get("parameter_estimates", {}).get(
            "grid", {}).get(player_uuid, {}).get("predictor", None)

        if not grid_predictor:
            "Skips rounds without grid data."
            continue

        meta_data = grid_predictor.get("meta_data", {})
        tickvals = meta_data.get("tickvals", {})

        if "Vᵢᵢ" not in tickvals or "Vᵢⱼ" not in tickvals:
            continue  # Skips incomplete grid metadata.
 
        "Extract param axes"
        Vii_vals = np.array(tickvals["Vᵢᵢ"], dtype=float)
        Vij_vals = np.array(tickvals["Vᵢⱼ"], dtype=float)
        Nx, Ny = len(Vii_vals), len(Vij_vals)

        "Build a 2D array for likelihood"
        likelihood_2d = np.zeros((Nx, Ny), dtype=float)
        for ix, v_ii in enumerate(Vii_vals):
            for jx, v_ij in enumerate(Vij_vals):
                agent_params = {
                    "Vᵢᵢ": v_ii,
                    "Vᵢⱼ": v_ij
                }
                "Compute p(choose A)."
                _gp = grid_predictor.get('params', {})
                _tau = _gp.get('τ', _gp.get('temp', 1.5))
                "TODO Decide whether visualization should dampen temperature or use the fitted value directly."
                p_choose_A = choice(
                    current_game=game,
                    agent_params=agent_params,
                    utility_settings=utility_settings,
                    softmax_temperature=_tau * 0.75,
                    select=False
                )["model_choose_A"]

                if game.get("choice") == "A":
                    likelihood_2d[ix, jx] = p_choose_A
                else:
                    likelihood_2d[ix, jx] = 1.0 - p_choose_A
        
        "Create the likelihood heatmap."
        likelihood_hm = go.Heatmap(
            x=Vii_vals,
            y=Vij_vals,
            z=likelihood_2d.T,
            colorscale=fig_lay.get("colorscales", ["Viridis"])[0],
            hovertemplate=("Vᵢᵢ: %{x:.3f}, Vᵢⱼ: %{y:.3f}<br>Lik: %{z:.3f}<extra></extra>"),
            showscale=False, zmin=0, zmax=1,
        )
        fig.add_trace(likelihood_hm, row=2, col=col_idx)

        "(C) Bottom row => prior/posterior"
        "Parse param_vectors => build a 2D pmf => fill holes => sum => normalize"
        param_vectors = grid_predictor.get("param_vectors", {})
        prior_2d = np.full((Nx, Ny), np.nan)
        for idx_tuple, prob in param_vectors.items():
            idx_tuple = ast.literal_eval(idx_tuple)
            ix, jx = idx_tuple[0], idx_tuple[1]
            if 0 <= ix < Nx and 0 <= jx < Ny:
                prior_2d[ix, jx] = prob
        "Fill holes"
        if general_settings.get('sample_ratio') < 1:
            prior_2d = gnrl.fill_holes_nd(prior_2d, (Nx, Ny), method="cubic")
        sprob = np.nansum(prior_2d)
        if sprob > 0:
            prior_2d /= sprob

        prior_hm = go.Heatmap(
            x=Vii_vals,
            y=Vij_vals,
            z=prior_2d.T,
            colorscale=fig_lay.get("colorscales", ["Viridis"])[1] if len(fig_lay.get("colorscales", []))>1 else "Plasma",
            zmin=0, zmax=max_prior_prob,
            hovertemplate=("Vᵢᵢ: %{x:.3f}, Vᵢⱼ: %{y:.3f}<br>Prob: %{z:.5f}<extra></extra>"),
            showscale=False
        )
        fig.add_trace(prior_hm, row=3, col=col_idx)

        if col_idx == 1:
            prior_preview = list(copy.deepcopy(prior_2d.T))
            for prior_row in prior_preview:
                print([round(prob, 9) for prob in list(prior_row)])

    "4) Stylistic adjustments"
    fig.update_layout(
        template=fig_lay.get("template", "plotly_dark"),
        title="Bayesian Updates for {} vs. {}".format(player_uuid, counterpart_uuid),
        title_x=fig_lay['title_x'], title_y=fig_lay['title_y'], 
        titlefont_size=fig_lay['titlefont_size'] * 0.5, 
        margin=dict(l=120, r=120, t=120, b=60 + 20 * n_rounds),
    ) 

    range_pay = [-4.1, 4.1]
    tickvals_pay = [-4, -2, 0, 2, 4]
    ticktext_pay_y = [str(val) for val in tickvals_pay]
    ticktext_pay_x = [''] + ticktext_pay_y[1:]

    n_ticks = len(Vii_vals)
    bin_size = 2/n_ticks
    half_bin = bin_size/2

    range_par = [-1 - half_bin, 1 + half_bin]
    tickvals_par = [-1.0, -0.5, 0.0, 0.5, 1.0]
    ticktext_par_y = [str(val) for val in tickvals_par]
    ticktext_par_x = [''] + ticktext_par_y[1:]

    scale_count = 0
    scaleanchors = [
        (f'x{fig_idx}', f'y{fig_idx}') for fig_idx in range(1, (n_rounds * 3) + 1)
    ]
    scaleanchors = []
    for cdx in range(1, n_rounds+1):
        for row_idx in (0, 1, 2):
            scaleanchors.append((f"x{cdx + n_rounds * row_idx}", f"x{cdx + n_rounds * row_idx}"))

    "For each column, style axes"
    for cdx in range(1, n_rounds+1):
        "Top row => payoff diff space"
        fig.update_xaxes(
            title_text="ΔSelf payoff", 
            range=range_pay, tickvals=tickvals_pay, ticktext=ticktext_pay_x,
            zeroline=dark_zero_lines, zerolinewidth=5 if dark_zero_lines else 1,
            scaleanchor=scaleanchors[scale_count][1], scaleratio=1, 
            row=1, col=cdx
        )
        fig.update_yaxes(
            title_text="ΔOther payoff" if cdx == 1 else "", 
            range=range_pay, tickvals=tickvals_pay, ticktext=ticktext_pay_y,
            zeroline=dark_zero_lines, zerolinewidth=5 if dark_zero_lines else 1,
            scaleanchor=scaleanchors[scale_count][0], scaleratio=1, 
            row=1, col=cdx
        )
        scale_count += 1
        "Middle row => likelihood parameter space."
        fig.update_xaxes(
            title_text="",
            range=range_par, tickvals=tickvals_par, ticktext=ticktext_par_x,
            scaleanchor=scaleanchors[scale_count][1], scaleratio=1, 
            row=2, col=cdx
        )
        fig.update_yaxes(
            title_text="Vᵢⱼ (altruism)" if cdx == 1 else "",
            range=range_par, tickvals=tickvals_par, ticktext=ticktext_par_y,
            scaleanchor=scaleanchors[scale_count][0], scaleratio=1, 
            row=2, col=cdx
        )
        scale_count += 1
        "Bottom row => prior/posterior parameter space."
        fig.update_xaxes(
            title_text="Vᵢᵢ (self-interest)",
            range=range_par, tickvals=tickvals_par, ticktext=ticktext_par_x,
            scaleanchor=scaleanchors[scale_count][1], scaleratio=1, 
            row=3, col=cdx
        )
        fig.update_yaxes(
            title_text="Vᵢⱼ (altruism)" if cdx == 1 else "",
            range=range_par, tickvals=tickvals_par, ticktext=ticktext_par_y,
            scaleanchor=scaleanchors[scale_count][0], scaleratio=1, 
            row=3, col=cdx
        )
        scale_count += 1

    "Save or show the figure."
    outdir = os.path.join(file_paths.get("visuals","."), "bayesian_updates_2d")
    os.makedirs(outdir, exist_ok=True)
    file_name = f"bayes2d_{player_uuid}_{counterpart_uuid}.html"
    if experiment_num == 0:
        for replacement in ('chooser', 'predictor', '_', '='):
            file_name = file_name.replace(replacement, '')
    out_path = os.path.join(outdir, file_name)

    if general_settings.get('export_fig'):
        fig.write_html(out_path)
        print(f"Saved 2D Bayesian updates figure to {out_path}")
    else:
        fig.show()

    return fig


def visualize_bayesian_updates_3d(dyad_games_or_key: int | DyadGames, player_uuid: PlayerUUID, fig_lay: Dict[str, Any], file_paths: FilePaths, 
                                  general_settings: GeneralSettings, fix_z_axis: bool = True):
    """
    Generate an interactive 3D two-panel figure with a game-by-game slider.

    For each game where the specified player is the predictor (and did not abdicate), two
    3D surfaces are rendered side by side:
        Left panel: The interpolated prior probability mass grid over (Vᵢᵢ, Vᵢⱼ), with
                    sparse probability sample points overlaid as a scatter.
        Right panel: The likelihood surface over (Vᵢᵢ, Vᵢⱼ) computed from the current
                     game’s payoff structure, with the observed choice marked in red.

    A Plotly slider lets the viewer step through games without re-rendering. The prior shown
    for game k is the posterior from game k-1, so the progression illustrates sequential
    belief updating across the interaction.

    Arguments:
        • dyad_games_or_key: int | DyadGames
            Either a list of game dicts (already loaded) or a dyad key string/int; if a key is
            provided, the dyad data is loaded from file.
        • player_uuid: PlayerUUID
            UUID of the predictor player whose beliefs to visualize.
        • fig_lay: Dict[str, Any]
            Figure layout template controlling colorscales, fonts, and sizing.
        • file_paths: FilePaths
            Directory paths used for loading dyad data and (optionally) exporting the figure.
        • general_settings: GeneralSettings
            Must include ‘update_method’, ‘experiment_num’, and ‘export_fig’.
        • fix_z_axis: bool
            If True, sets the z-axis and color range of the left (prior) panel to [0, global_max],
            where global_max is the maximum probability across all games. If False, the z-axis
            scales independently per game, which can make low-mass distributions more visible.

    Returns:
        • None — the figure is displayed (fig.show()) or exported to HTML per general_settings.
    """
    update_method = general_settings.get('update_method', 'grid')

    if isinstance(dyad_games_or_key, list):
        dyad_games = dyad_games_or_key
    elif general_settings.get('experiment_num') == 0:
        _raw = get_simulated_dyad(file_paths=file_paths, dyad_idx=dyad_games_or_key, n_games=0)
        dyad_games = next(iter(_raw.values()))
    else:
        dyad_games = prep.get_dyad_data(dyad_key=dyad_games_or_key, file_paths=file_paths,
                                              experiment_num=general_settings.get('experiment_num', 3), analysis_mode='bayesian', dyad_already_analyzed=False)

    first_game = dyad_games[0]

    if isinstance(player_uuid, int):
        if player_uuid == 1:
            player_uuid = first_game['chooser']
        elif player_uuid == 2:
            player_uuid = first_game['predictor']
            
    first_choo = first_game['chooser']
    first_pred = first_game['predictor']    
    player_number = 1 if player_uuid == first_choo else 2    
    _uuid_hash = hashlib.md5(f"{first_choo}|{first_pred}".encode()).hexdigest()[:12]
    dyad_name = f"{player_number}_{_uuid_hash}"

    "Filter dyad games: only include games where the player is the predictor and did not abdicate."
    if general_settings.get('experiment_num') == 3:
        filtered_games = [game for game in dyad_games 
                        if game.get('predictor', None) == player_uuid and not game.get('abdicated_predictor', False)]
        if not filtered_games:
            raise ValueError("No games found where the given player is the predictor and did not abdicate.")
        
    else:
        # TODO Extract the choice marker from observation-phase games for experiments 1 and 2.
        filtered_games = [game for game in dyad_games if game.get('predictor', None) == player_uuid]

    "Find the first game with grid data for predictor."
    first_grid_game = None
    for game in filtered_games:
        grid_data = game.get("parameter_estimates", {}).get(update_method, {}).get(player_uuid, {}).get("predictor", {})
        if grid_data and grid_data.get("meta_data", None) is not None:
            first_grid_game = game
            break         
    if first_grid_game is None:
        raise ValueError("No grid data found for predictor in any game for player_uuid: " + player_uuid)
    
    "Extract meta_data and tickvals from the first grid game."
    grid_data = first_grid_game["parameter_estimates"][update_method][player_uuid]["predictor"]
    meta_data = grid_data["meta_data"]
    tickvals = meta_data["tickvals"]

    if "Vᵢᵢ" not in tickvals or "Vᵢⱼ" not in tickvals:
        raise ValueError("Tickvals must contain 'Vᵢᵢ' and 'Vᵢⱼ' keys.")
    
    "Build a common parameter grid."
    Vᵢᵢ_vals = np.array([round(val, 9) for val in tickvals["Vᵢᵢ"]])
    Vᵢⱼ_vals = np.array([round(val, 9) for val in tickvals["Vᵢⱼ"]])
    Vᵢᵢ_mesh, Vᵢⱼ_mesh = np.meshgrid(Vᵢᵢ_vals, Vᵢⱼ_vals, indexing='ij')
    
    """
    If requested, compute the global maximum normalized probability across filtered
    games. This value fixes the z-axis and color axis for the prior surface.
    """
    global_max_prob = None
    if fix_z_axis:
        global_max_prob = 0
        for game in filtered_games:
            grid_predictor = game.get("parameter_estimates", {}).get(
                update_method, {}).get(player_uuid, {}).get("predictor", None)
            if grid_predictor is None:
                continue
            prior_dict: dict = grid_predictor.get("param_vectors", {})
            temp_grid = np.full((len(Vᵢᵢ_vals), len(Vᵢⱼ_vals)), np.nan)
            for idx_tuple, prob in prior_dict.items():
                idx_tuple = ast.literal_eval(idx_tuple)
                idx, jdx = idx_tuple[0], idx_tuple[1]
                if 0 <= idx < len(Vᵢᵢ_vals) and 0 <= jdx < len(Vᵢⱼ_vals):
                    temp_grid[idx, jdx] = prob
            temp_grid = gnrl.fill_holes_nd(input_array=temp_grid, output_shape=(len(Vᵢᵢ_vals), len(Vᵢⱼ_vals)), method="cubic")
            sum_temp = np.nansum(temp_grid)
            temp_grid = temp_grid / sum_temp
            candidate = np.nanmax(temp_grid)
            if candidate > global_max_prob:
                global_max_prob = candidate
        "Fallback when no valid grid contributes to the global maximum."
        if global_max_prob is None or global_max_prob == 0:
            global_max_prob = 1

    "Create the figure with two 3D subplots."
    fig = make_subplots(rows=1, cols=2,
                        horizontal_spacing=0.1,
                        subplot_titles=["Prior Probability Distribution", "Likelihood Probability Distribution"],
                        specs=[[{'type': 'surface'}, {'type': 'surface'}]])
    
    "Apply the shared figure layout."
    fig.update_layout(
        template=fig_lay["template"] if fig_lay else None,
        hoverlabel=dict(font_size=14),
        scene=dict(
            xaxis=dict(title="Vᵢᵢ", zeroline=True, zerolinewidth=4, nticks=9),
            yaxis=dict(title="Vᵢⱼ", zeroline=True, zerolinewidth=4, nticks=9),
            zaxis=dict(title="Probability", zeroline=True, nticks=9,
                       range=[0, global_max_prob] if fix_z_axis else None),
            camera=dict(eye=dict(x=-0.0, y=-1e-5, z=2)),
            aspectmode="cube",
        ),
        scene2=dict(
            xaxis=dict(title="Vᵢᵢ", zeroline=True, zerolinewidth=4, nticks=9),
            yaxis=dict(title="Vᵢⱼ", zeroline=True, zerolinewidth=4, nticks=9),
            zaxis=dict(title="Probability", zeroline=True, nticks=9, range=[0, 1]),
            camera=dict(eye=dict(x=-0.0, y=-1e-5, z=2)),
            aspectmode="cube",
        ),
    )
    "Fix the color axis for scene 1 if requested."
    fig.update_layout(
        coloraxis=dict(
            colorscale=fig_lay["colorscales"][0] if "colorscales" in fig_lay else "Viridis",
            cmin=0, cmax=global_max_prob if fix_z_axis else None,
            showscale=False
        )
    )
    
    "Accumulate traces for each game (for the slider)."
    all_traces = []
    
    "Loop over the filtered games."
    for game_idx, game in enumerate(filtered_games):
        grid_predictor = game.get("parameter_estimates", {}).get(update_method, {}).get(player_uuid, {}).get("predictor", None)
        if grid_predictor is None:
            continue

        "Build the full prior PMF array from the sparse dictionary."
        prior_dict = grid_predictor.get("param_vectors", {})

        "Initialize probability sums and counts for averaging."
        aggregated_probs = {}

        "Aggregate probabilities by Vᵢᵢ and Vᵢⱼ grid indices."
        for idx_tuple, prob in prior_dict.items():
            idx_tuple = ast.literal_eval(idx_tuple)
            idx, jdx = idx_tuple[0], idx_tuple[1]  # Vᵢᵢ and Vᵢⱼ indices.
            if 0 <= idx < len(Vᵢᵢ_vals) and 0 <= jdx < len(Vᵢⱼ_vals):
                if (idx, jdx) not in aggregated_probs:
                    aggregated_probs[(idx, jdx)] = {"prob_sum": 0, "count": 0}
                aggregated_probs[(idx, jdx)]["prob_sum"] += prob
                aggregated_probs[(idx, jdx)]["count"] += 1

        "Create the full_grid with averaged probabilities."
        full_grid = np.full((len(Vᵢᵢ_vals), len(Vᵢⱼ_vals)), np.nan)
        for (idx, jdx), data in aggregated_probs.items():
            full_grid[idx, jdx] = data["prob_sum"] / data["count"]

        "Handle missing values in the grid and normalize."
        full_grid = gnrl.fill_holes_nd(input_array=full_grid, output_shape=(len(Vᵢᵢ_vals), len(Vᵢⱼ_vals)), method="cubic")
        sum_full_grid = np.nansum(full_grid)
        full_grid = full_grid / sum_full_grid
    
        "Prepare sparse prior sample points."
        filtered_data = {}
        Vᵢᵢ_idx = list(tickvals.keys()).index("Vᵢᵢ")
        Vᵢⱼ_idx = list(tickvals.keys()).index("Vᵢⱼ")

        for param_vector, prob in prior_dict.items():
            param_vector = ast.literal_eval(param_vector)
            val_Vᵢᵢ = tickvals["Vᵢᵢ"][param_vector[Vᵢᵢ_idx]]
            val_Vᵢⱼ = tickvals["Vᵢⱼ"][param_vector[Vᵢⱼ_idx]]
            key = (val_Vᵢᵢ, val_Vᵢⱼ)
            if key not in filtered_data:
                filtered_data[key] = []
            filtered_data[key].append(prob)
        unique_points = np.array(list(filtered_data.keys()))
        probabilities = np.array([np.mean(filtered_data[key]) for key in filtered_data])
        scatter_z = [prob / sum_full_grid for prob in probabilities]  # Normalized sparse probabilities.
        
        scatter_x, scatter_y = unique_points[:, 0], unique_points[:, 1]
        
        "Compute likelihood surface over the same grid."
        current_params = grid_predictor.get("params", {})
        Vᵢᵢ_mean = round(current_params.get("Vᵢᵢ", 1), 2)
        Vᵢⱼ_mean = round(current_params.get("Vᵢⱼ", 1), 2)
        Vᵢᵢ_std = current_params.get("Vᵢᵢ_std", 1)
        Vᵢⱼ_std = current_params.get("Vᵢⱼ_std", 1)
        likelihood_surface = np.zeros_like(Vᵢᵢ_mesh, dtype=float)
        for idx in range(len(Vᵢᵢ_vals)):
            for jdx in range(len(Vᵢⱼ_vals)):
                agent_params = {
                    "Vᵢᵢ": Vᵢᵢ_mesh[idx, jdx],
                    "Vᵢⱼ": Vᵢⱼ_mesh[idx, jdx],
                    "Vᵢᵢ_std": Vᵢᵢ_std,
                    "Vᵢⱼ_std": Vᵢⱼ_std,
                }
                p_choose_A = choice(current_game=game, agent_params=agent_params, 
                                    softmax_temperature=general_settings.get('softmax_temperature', 1.5),
                                    utility_settings=utility_settings, select=False)["model_choose_A"]
                if game.get("choice", "A") == "A":
                    likelihood_surface[idx, jdx] = p_choose_A
                else:
                    likelihood_surface[idx, jdx] = 1 - p_choose_A
        
        "Determine the observed choice marker."
        hover_text = f"Observed Choice: <b>"
        payoff_As, payoff_Ao = game.get("payoff_A_chooser", 0), game.get("payoff_A_predictor", 0)
        payoff_Bs, payoff_Bo = game.get("payoff_B_chooser", 0), game.get("payoff_B_predictor", 0)        
        if game.get("choice", "A") == "A":
            payoff_diff_self = (payoff_As - payoff_Bs) / 4.0
            payoff_diff_other = (payoff_Ao - payoff_Bo) / 4.0
            hover_text += f"A:({payoff_As}, {payoff_Ao})</b> over B:({payoff_Bs}, {payoff_Bo})<br>"
        else:
            payoff_diff_self = (payoff_Bs - payoff_As) / 4.0
            payoff_diff_other = (payoff_Bo - payoff_Ao) / 4.0   
            hover_text += f"B:({payoff_As}, {payoff_Ao})</b> over A:({payoff_Bs}, {payoff_Bo})<br>"                 
        observed_choice_point = (payoff_diff_self, payoff_diff_other)
        hover_text += f"Payoff Difference Self (chooser) =     {int(payoff_diff_self*4)}<br>"
        hover_text += f"Payoff Difference Other (predictor) = {int(payoff_diff_self*4)}<extra></extra>"
        observed_z = 0.05  # Slight elevation.
        
        "Build traces for this game."
        game_traces = []
        "Trace 1: Prior surface (left panel)"
        trace_prior = go.Surface(
            z=full_grid.T,
            x=Vᵢᵢ_vals,
            y=Vᵢⱼ_vals,
            colorscale=fig_lay["colorscales"][0],
            opacity=0.8,
            showscale=False,
            name=f"Prior (Game {game_idx})",
            hovertemplate=(
                "Vᵢᵢ: %{x:.3f}; " + f"(μ, σ) = ({Vᵢᵢ_mean}, {round(Vᵢᵢ_std, 2)})<br>" +
                "Vᵢⱼ: %{y:.3f}; " + f"(μ, σ) = ({Vᵢⱼ_mean}, {round(Vᵢⱼ_std, 2)})<br>" +
                "Probability: %{z:.3g}<extra></extra>"
            )                
        )
        game_traces.append(trace_prior)
        
        "Trace 2: Sparse prior points."
        trace_sparse = go.Scatter3d(
            x=scatter_x,
            y=scatter_y,
            z=scatter_z,
            mode="markers",
            marker=dict(
                size=6,
                color=scatter_z,
                colorscale=fig_lay["colorscales"][0],
                opacity=0.9
            ),
            name=f"Prior Probabilities",
            hovertemplate=(
                "Vᵢᵢ: %{x:.3f}; " + f"(μ, σ) = ({Vᵢᵢ_mean}, {round(Vᵢᵢ_std, 2)})<br>" +
                "Vᵢⱼ: %{y:.3f}; " + f"(μ, σ) = ({Vᵢⱼ_mean}, {round(Vᵢⱼ_std, 2)})<br>" +
                "Probability: %{z:.3g}<extra></extra>"
            )              
        )
        game_traces.append(trace_sparse)
        
        "Trace 3: Likelihood surface (right panel)"
        trace_likelihood = go.Surface(
            z=likelihood_surface.T,
            x=Vᵢᵢ_vals,
            y=Vᵢⱼ_vals,
            colorscale=fig_lay["colorscales"][1],
            opacity=0.8,
            showscale=False,
            name=f"Likelihood (Game {game_idx})",
            hovertemplate=(
                "Vᵢᵢ: %{x:.3f}<br>" +
                "Vᵢⱼ: %{y:.3f}<br>" +
                "Probability: %{z:.3g}<extra></extra>"
            )
        )
        game_traces.append(trace_likelihood)
        
        "Trace 4: Observed choice marker."
        trace_choice = go.Scatter3d(
            x=[observed_choice_point[0]],
            y=[observed_choice_point[1]],
            z=[observed_z],
            mode="markers",
            marker=dict(size=8, color="red", symbol="circle"),
            name=f"Counterpart's Choice",
            hovertemplate=hover_text
        )
        game_traces.append(trace_choice)
        
        "Save traces for this game."
        all_traces.append({"traces": game_traces, "grid_predictor": grid_predictor})
    
    if not all_traces:
        raise ValueError("No valid predictor grid data was found in the filtered games.")
    
    "Add all traces to the figure."
    total_traces = 0
    for game_block in all_traces:
        for trace in game_block["traces"]:
            fig.add_trace(trace)
            total_traces += 1
    traces_per_game = len(all_traces[0]["traces"])
    
    "Build slider steps: each step makes only the traces for that game visible."
    slider_steps = []
    for game_idx in range(len(all_traces)):
        visible = [False] * total_traces
        for idx in range(traces_per_game):
            visible[game_idx * traces_per_game + idx] = True
        step = dict(
            label=f"Game {game_idx}",
            method="update",
            args=[{"visible": visible},
                  {"title": f"Bayesian Update Visualization (Game {game_idx})"}]
        )
        slider_steps.append(step)
    
    "Set initial visibility: only traces for the first game."
    init_visible = [False] * total_traces
    for idx in range(traces_per_game):
        init_visible[idx] = True
    for idx, trace in enumerate(fig.data):
        trace.visible = init_visible[idx]
    
    fig.update_layout(
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Game: "},
            pad={"t": 50},
            steps=slider_steps
        )],
        margin=dict(l=100, r=100, t=100, b=100)
    )
    
    "Assign traces to the proper subplot (scene):"
    "Traces 0 and 1 go to scene (left panel), traces 2 and 3 to scene2 (right panel)."
    trace_idx = 0
    for game_block in all_traces:
        for idx, tr in enumerate(game_block["traces"]):
            if idx < 2:
                fig.data[trace_idx].update(scene="scene")
            else:
                fig.data[trace_idx].update(scene="scene2")
            trace_idx += 1
    
    "Use fig_lay's width and height if provided."
    if fig_lay.get("width") and fig_lay.get("height"):
        fig.update_layout(width=fig_lay["width"], height=fig_lay["height"])

    file_name = f"bayesian_update_visualization_{dyad_name}"
    if isinstance(dyad_games_or_key, int):
        file_name += f"_{dyad_games_or_key}"
    file_name += f"{file_paths.get('file_name_suffix', '')}.html"
    visuals_dir = os.path.join(file_paths["visuals"], "bayesian_updates_3d")
    os.makedirs(visuals_dir, exist_ok=True)
    out_path = os.path.join(visuals_dir, file_name)
    if general_settings.get('export_fig'):
        fig.write_html(out_path)
        print(f"Saved {file_name} at {out_path}")
    else:
        fig.show()
    
    return fig


def belief_accuracy_analysis(file_paths: FilePaths, general_settings: GeneralSettings, fig_lay: FigLay, participant_num: int, 
                             fitted_by_player: bool = True, compute_optimum_updates: bool = False, animate_figure: bool = False) -> None:
    """
    Visualize how accurately a predictor's posterior beliefs track their counterpart's true preferences.

    Loads a participant's sequential parameter estimates and compares the UBM posterior mean
    trajectory (how the predictor's beliefs about the chooser evolve game by game) against
    the true fitted parameters of the chooser. Produces a figure showing belief trajectories
    and convergence behavior across the interaction, optionally animated across rounds.

    Arguments:
        • file_paths: FilePaths
            Directory paths used to locate saved parameter grid files and figure output directories.
        • general_settings: GeneralSettings
            Must include 'experiment_num', 'update_method', and related settings.
        • fig_lay: FigLay
            Figure layout template controlling fonts, colors, and sizing.
        • participant_num: int
            Integer index into the sorted player list; used as a convenience alternative to
            specifying a full UUID string.
        • fitted_by_player: bool
            If True, uses parameters fitted per-player (pooled across counterparts).
            If False, uses parameters fitted per dyad (specific to each chooser–predictor pair).
        • compute_optimum_updates: bool
            If True, recomputes the optimal Bayesian posterior updates from scratch rather than
            loading cached grid data. Slower but guarantees fresh computation.
        • animate_figure: bool
            If True, generates an animated figure that steps through rounds.
            If False, generates a static figure showing all rounds at once.

    Returns:
        • None — the figure is displayed or exported as HTML per general_settings['export_fig'].
    """
    avatar_type_to_key = {
        'utilitarian': None,
        'selfish':     None,
        'competitive': None,
        'masochistic': None,
    }

    converge_n_round = {
        'utilitarian': None,
        'selfish':     None,
        'competitive': None,
        'masochistic': None,        
    }

    if fitted_by_player:
        "Parameters are fitted to each participant across multiple dyads."
        plrs_to_dyads = prep.players_to_dyads(experiment_num=2, 
                        file_paths=file_paths, create_new_file=False)

        players_lst = sorted(list(plrs_to_dyads.keys()))
        player_uuid = players_lst[participant_num % len(players_lst)]

        histories = None
        file_name_suffix = file_paths["file_name_suffix"]
        plr_file_path = os.path.join(file_paths["player_fits"], f"experiment_2", 
                                f'{file_name_suffix}_' + player_uuid + ".json")

        if os.path.exists(plr_file_path):
            with open(plr_file_path, "r", encoding='utf-8') as file:
                histories: dict = json.load(file)

        if histories is None:
            raise Exception(f"Failed to extract dyads for player {player_uuid}.")

    else:
        "Parameters are fitted to each dyad and so can differ across dyads for the same participant."
        with open(os.path.join("./", file_paths['file_names'][
            'player_pairs_exper2']), "r", encoding='utf-8') as file:
            raw_data = json.load(file)
        histories: dict = raw_data['histories']
        player_info: dict = raw_data['player_info']
        participants = [player_uuid for player_uuid, info in player_info.items() 
                        if info.get('player_type') == 'participant']
        
        player_uuid = participants[participant_num % len(participants)]

    player_dyad_keys = [dyad_key for dyad_key in histories.keys() if player_uuid in dyad_key]

    for avatar_type in avatar_type_to_key.keys():
        for player_dyad_key in player_dyad_keys:
            if avatar_type in player_dyad_key:
                avatar_type_to_key[avatar_type] = player_dyad_key

    if any(dyad_key is None for dyad_key in avatar_type_to_key.values()):
        raise Exception(f"Failed to extract all four dyads for participant number {participant_num}.")

    "Organize all dyads into a dictionary indexed by counterpart avatar type."
    if fitted_by_player:
        player_dyads = {
            avatar_type: histories[dyad_key] 
            for avatar_type, dyad_key in avatar_type_to_key.items()
        }
    else:
        player_dyads = {
            avatar_type: prep.get_dyad_data(dyad_key=dyad_key, 
            file_paths=file_paths, dyad_already_analyzed=True, experiment_num=2) 
            for avatar_type, dyad_key in avatar_type_to_key.items()
        }

    if compute_optimum_updates:
        "Run discrete model"
        for avatar_type, dyad_games in player_dyads.items():
            temperature = None
            if general_settings.get('temperature_is_param', False):
                first_game = dyad_games[0]
                param_est = first_game.get('parameter_estimates', {}).get(
                    'grid', {}).get(player_uuid, {}).get('predictor', {}).get('params', {})
                temperature = param_est.get('τ', param_est.get('temp', None))
            dyad_games = typo.avatar_posteriors(dyad_games=dyad_games, temperature=temperature, 
                                                 loss_funct_type=general_settings.get('loss_funct_type', 'log'),
                                                 update_method=general_settings.get('update_method', 'grid'))

    player_dyads_data = {}
    for avatar_type, dyad_games in player_dyads.items():
        "Generate data for figure and save in dicts of lists."
        dyad_data = {
            'round': [],
            'phase': [],
            'payoff_A_chooser': [],
            'payoff_A_predictor': [],
            'payoff_B_chooser': [],
            'payoff_B_predictor': [],
            'prediction': [],
            'choice': [],
        }
        param_data_human = {
            'Vᵢᵢ': [],
            'Vᵢⱼ': [],
            'Vᵢᵢ_std': [],
            'Vᵢⱼ_std': [],
        }
        param_data_optimum = {
            'utilitarian': [],
            'selfish':     [],
            'competitive': [],
            'masochistic': [],
        }
        for idx, dyad_game in enumerate(dyad_games):
            param_est = dyad_game.get('parameter_estimates', {}).get(
                'grid', {}).get(player_uuid, {}).get('predictor', {}).get('params', {})
            if dyad_game['phase'] == 'op':
                optimum_update: dict = dyad_game.get('parameter_estimates', {}).get(
                    'grid', {}).get('optimum_update', {})
            else:
                optimum_update = dyad_games[idx-1].get('parameter_estimates', {}).get(
                    'grid', {}).get('optimum_update', {})                 

            avatar_posteriors = optimum_update.get('avatar_posteriors', {})
            if converge_n_round[avatar_type] is None:
                converged = optimum_update.get('converged', False)
                if converged:
                    converge_n_round[avatar_type] = idx

            for data_key in dyad_data.keys():
                dyad_data[data_key].append(dyad_game.get(data_key, None))
            for hum_param_key in param_data_human.keys():
                param_data_human[hum_param_key].append(param_est.get(hum_param_key, None))
            for opt_param_key in param_data_optimum.keys():
                param_data_optimum[opt_param_key].append(avatar_posteriors.get(opt_param_key, None))

        "Merge all data into a single dictionary indexed by counterpart avatar type."
        player_dyads_data[avatar_type] = {**dyad_data, **param_data_human, **param_data_optimum}        

    "Repeat masochistic avatar data so its length matches the other avatar series."
    for key, data_list in player_dyads_data['masochistic'].items():
        player_dyads_data['masochistic'][key] = [val for val in data_list for _ in range(3)]
 
    n_games = len(player_dyads_data['masochistic']['Vᵢᵢ'])

    "Create subplot layout"
    fig = make_subplots(
        rows=2, cols=4,
        specs=[
            [{"rowspan": 2, "colspan": 2}, None, {"type": "bar"}, {"type": "bar"}],
            [None,                         None, {"type": "bar"}, {"type": "bar"}]
        ],
        subplot_titles=(f"Social Preference Belief Updates Over Time For Participant {player_uuid} With Four Avatars", 
                        "Avatar is Utilitarian", "Avatar is Selfish", "Avatar is Competitive", "Avatar is Masochistic")
    )
    avatar_colors = {
        'utilitarian': 'hsla(180, 50%, 50%, 0.9)',
        'selfish':     'hsla(240, 50%, 50%, 0.9)',
        'competitive': 'hsla(300, 50%, 50%, 0.9)',
        'masochistic': 'hsla(360, 50%, 50%, 0.9)', 
    }
    avatar_region_colors = {
        'utilitarian': 'hsla(180, 50%, 50%, 0.4)',
        'selfish':     'hsla(240, 50%, 50%, 0.4)',
        'competitive': 'hsla(300, 50%, 50%, 0.4)',
        'masochistic': 'hsla(360, 50%, 50%, 0.4)', 
    }    
    bar_subplot_row_col = {
        'utilitarian': (1, 3),
        'selfish':     (1, 4),
        'competitive': (2, 3),
        'masochistic': (2, 4),
    }

    slider_steps = []
    type_labels = ['utilitarian', 'selfish', 'competitive', 'masochistic']
    type_labels = [label.capitalize() for label in type_labels]
    for idx in range(n_games):
        n_traces = 0
        "Line plots representing mean parameter belief updates."
        for avatar_type in ['utilitarian','selfish','competitive','masochistic']:

            "Marks regions where parameters are compatible with avatar choices."
            for line_idx in (0, 1):
                fig.add_trace(
                    go.Scatter(
                        x=(0, typo.regions_of_compatible_utility[avatar_type][line_idx][0]),
                        y=(0, typo.regions_of_compatible_utility[avatar_type][line_idx][1]),
                        name=f"{avatar_type} Region", fill=None if line_idx == 0 else 'tonexty',
                        line=dict(color=avatar_region_colors[avatar_type], width=0),
                        marker=dict(size=12), visible=True if idx == 0 else False, 
                        showlegend=False, mode='lines', 
                    ),
                    row=1, 
                    col=1
                )
                n_traces += 1

            Vᵢᵢ_array = player_dyads_data[avatar_type]['Vᵢᵢ']
            Vᵢⱼ_array = player_dyads_data[avatar_type]['Vᵢⱼ']
            Vᵢᵢ_array_std = player_dyads_data[avatar_type]['Vᵢᵢ_std']
            Vᵢⱼ_array_std = player_dyads_data[avatar_type]['Vᵢⱼ_std']  
            Vᵢᵢ_std_prior = round(Vᵢᵢ_array_std[0], 4)
            Vᵢⱼ_std_prior = round(Vᵢⱼ_array_std[0], 4)
            prior_std_str = f"<br>σ(𝑉𝑖𝑖) = {Vᵢᵢ_std_prior}<br>σ(𝑉𝑖𝑗) = {Vᵢⱼ_std_prior}"
            custom_hover_data = [
                typo.distance_to_perfection(Vᵢᵢ=Vᵢᵢ_array[jdx], Vᵢⱼ=Vᵢⱼ_array[jdx], 
                                             avatar_type=avatar_type, return_percent=True)
                for jdx in range(len(Vᵢᵢ_array))
            ]
            custom_hover_subset = custom_hover_data[:idx+1]
            x_line = Vᵢᵢ_array[:idx+1]
            y_line = Vᵢⱼ_array[:idx+1]
            hover_str = f"Avatar is {avatar_type.capitalize()}:<br><br>"
            hover_str += "Social preference<br>belief accuracy is<br>"

            "Lines show parameter beliefs as they update."
            fig.add_trace(
                go.Scatter(
                    x=x_line, y=y_line,
                    mode='lines+markers',
                    marker=dict(size=12), name="",
                    customdata=custom_hover_subset,
                    line=dict(color=avatar_colors[avatar_type], width=8),
                    visible=(True if idx == 0 else False), showlegend=(True if idx==0 else False),
                    hovertemplate=f"{hover_str}%{{customdata:.1f}}%.<br><br>Vᵢᵢ = %{{x:.2f}}<br>Vᵢⱼ = %{{y:.2f}}"
                ),
                row=1, 
                col=1
            )
            n_traces += 1

            "Highlights the current posterior."
            first_game = player_dyads[avatar_type][0]
            optimum_update_data: dict = first_game.get('parameter_estimates', {}).get(
                general_settings.get('update_method', 'grid'), {}).get('optimum_update', None)
            if optimum_update_data is not None:
                model_winner: str = optimum_update_data.get('model_winner')
                continuous_model_loss = optimum_update_data.get('total_continious_model_loss')
                discrete_model_loss = optimum_update_data.get('total_discrete_model_loss')
                if all(val is not None for val in (model_winner, continuous_model_loss, discrete_model_loss)):
                    continuous_model_loss = round(continuous_model_loss, 3)    
                    discrete_model_loss = round(discrete_model_loss, 3) 
                    hover_text = f"Posterior:<br>Vᵢᵢ = {round(x_line[-1], 2)}<br>"
                    hover_text += f"Vᵢⱼ = {round(y_line[-1], 3)}<br><br>Model Loss:<br>"
                    hover_text += f"Continuous: {continuous_model_loss}<br>"
                    hover_text += f"Discrete:     {discrete_model_loss}<br>"
                    hover_text += f"<br>Winning Model:<br>{model_winner.capitalize()}!"
                    "Mark the final posteriors."
                    fig.add_trace(
                        go.Scatter(
                            x=[x_line[-1]], y=[y_line[-1]],
                            customdata=custom_hover_subset, name='',
                            marker=dict(size=22, symbol='circle'), mode='markers',
                            line=dict(color=avatar_colors[avatar_type], width=6),
                            visible=False, showlegend=False,
                            hovertemplate=hover_text
                        ),
                        row=1, 
                        col=1
                    )
                    n_traces += 1 

            "Mark the initial prior."
            fig.add_trace(
                go.Scatter(
                    x=[x_line[0]], y=[y_line[0]],
                    customdata=custom_hover_subset, name='',
                    marker=dict(size=28, symbol='cross'), mode='markers',
                    line=dict(color='hsla(120, 50%, 50%, 0.9)', width=6),
                    visible=(True if idx == 0 else False), showlegend=False,
                    hovertemplate=f"Starting Prior:<br>μ(𝑉𝑖𝑖) = %{{x:.4f}}<br>μ(𝑉𝑖𝑗) = %{{y:.4f}}{prior_std_str}"
                ),
                row=1, 
                col=1
            )
            n_traces += 1   

        "Four bar charts indicate p(avatar type) under the optimum discrete Bayesian model."
        for avatar_type in ['utilitarian','selfish','competitive','masochistic']:
            opt_util = player_dyads_data[avatar_type]['utilitarian'][idx] 
            opt_self = player_dyads_data[avatar_type]['selfish'][idx]     
            opt_comp = player_dyads_data[avatar_type]['competitive'][idx] 
            opt_maso = player_dyads_data[avatar_type]['masochistic'][idx] 
            y_vals = [opt_util, opt_self, opt_comp, opt_maso]
            converge_text = "Not yet converged"
            n_rounds_converged = converge_n_round.get(avatar_type)
            if n_rounds_converged is not None:
                if idx < n_rounds_converged:
                    converge_text = f"<br><br>Will achieve<br>certainty on<br>game {n_rounds_converged}."
                elif idx > n_rounds_converged:
                    converge_text = f"<br><br>Achieved<br>certainty in<br>{n_rounds_converged} games."
                else:
                    converge_text = f"<br><br>Achieved<br>certainty on<br>game {n_rounds_converged}."

            fig.add_trace(
                go.Bar(
                    x=type_labels,
                    y=y_vals,
                    marker=dict(color=[
                        avatar_colors['utilitarian'],
                        avatar_colors['selfish'],
                        avatar_colors['competitive'],
                        avatar_colors['masochistic'],
                    ]),
                    name=f"Posterior:",
                    visible=True if idx == 0 else False,
                    hovertemplate="𝑝(%{x}) = %{y:.2f}" + converge_text,
                    showlegend=False,
                ),
                row=bar_subplot_row_col[avatar_type][0], 
                col=bar_subplot_row_col[avatar_type][1]
            )
            n_traces += 1

        "Controlling which traces are visible per slider step."
        visible_traces = [False] * n_traces * n_games
        for trace_idx in range(n_traces):
            visible_traces[idx * n_traces + trace_idx] = True  

        if animate_figure:
            step = {
                "label": f"Game: {idx}",
                "method": "animate",
                "args": [
                    [str(idx)],  
                    {
                        "frame": {"duration": 10, "redraw": True},
                        "transition": {"duration": 2, 'easing': 'quadratic-in-out'},
                        "mode": "immediate"
                    }
                ]
            }
        else:
            step = dict(
                label=f"Game: {idx}",
                method="update",
                args=[{"visible": visible_traces}]
            )            
        slider_steps.append(step)

    fig.update_layout(
        sliders=[dict(active=0,
            currentvalue={"prefix": ""},
            steps=slider_steps
        )],
        showlegend=False, hoverlabel=dict(font_size=20), 
        margin=dict(l=80, r=80, t=120, b=100), 
        template=fig_lay['template'], 
    )

    if animate_figure:
        total_traces = len(fig.data)  
        frames = []

        for frame_idx in range(n_games):
            "Builds a list of booleans for each trace's visibility."
            visible_flags = [False] * total_traces
            start_index = frame_idx * n_traces
            "Marks traces for the current game as visible."
            for kdx in range(start_index, start_index + n_traces):
                visible_flags[kdx] = True

            "Builds a minimal update for each trace that sets 'visible' and preserves 'type'."
            frame_update = []
            for ldx in range(total_traces):
                trace_type = fig.data[ldx].type
                frame_update.append({"visible": visible_flags[ldx], "type": trace_type})
            
            "Creates a frame with name equal to the frame index."
            frames.append(go.Frame(data=frame_update, name=str(frame_idx)))

        "Attach frames to the figure."
        fig.frames = frames

        fig.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 10, "redraw": True},
                                    "transition": {"duration": 2, 'easing': 'quadratic-in-out'},
                                }
                            ]
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "transition": {"duration": 0},
                                    "mode": "immediate",
                                }
                            ]
                        }
                    ],
                    "pad": {"r": 10, "t": 87},
                    "x": -0.02, "y": 1.04,
                    "xanchor": "right",
                    "yanchor": "top"
                }
            ],
            sliders=[{
                "active": 0,
                "currentvalue": {"prefix": "Frame: "},
                "pad": {"t": 50},
                "steps": slider_steps
            }]
        )

    tickvals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    ticktext_y = [str(val) for val in tickvals]
    ticktext_x = [''] + ticktext_y[1:]

    "Line Plot"
    fig.update_xaxes(range=[-1.002, 1.002], row=1, col=1, tickvals=tickvals, ticktext=ticktext_x, scaleanchor='y1', 
                     scaleratio=1, zerolinewidth=4, title=dict(text="Self-Interest (Vᵢᵢ)", font=dict(size=15)))
    fig.update_yaxes(range=[-1.002, 1.002], row=1, col=1, tickvals=tickvals, ticktext=ticktext_y, scaleanchor='x1', 
                     scaleratio=1, zerolinewidth=4, title=dict(text="Altruism (Vᵢⱼ)", font=dict(size=15)))

    "Utilitarian"
    fig.update_xaxes(title="Possible Avatar Types", row=1, col=3)
    fig.update_yaxes(title="𝑝(avatar type)", range=[0,1], row=1, col=3)
    "Selfish"
    fig.update_xaxes(title="Possible Avatar Types", row=1, col=4)
    fig.update_yaxes(title="𝑝(avatar type)", range=[0,1], row=1, col=4)
    "Competitive"
    fig.update_xaxes(title="Possible Avatar Types", row=2, col=3)
    fig.update_yaxes(title="𝑝(avatar type)", range=[0,1], row=2, col=3)
    "Masochistic"
    fig.update_xaxes(title="Possible Avatar Types", row=2, col=4)
    fig.update_yaxes(title="𝑝(avatar type)", range=[0,1], row=2, col=4)

    "Show or export"
    title_str = "Animated" if animate_figure else "Static"
    title = f"Belief Accuracy Analysis {title_str} {player_uuid}"
    file_name = title.replace(' ', '_') + f"{file_paths['file_name_suffix']}.html"
    if general_settings.get('export_fig'):
        fig.write_html(os.path.join(file_paths["visuals"], "belief_accuracy_analyses", file_name))
        print(f"Saved {file_name}")
    else:
        fig.show()

    return fig


