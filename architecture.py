import hashlib
import time
from visualization import *
from visualization import _hsla
from utilities import compute_conditional_hamming_distance_matrix
from behavioral_distances import *
from behavioral_distances import _fmt_duration, _ampd_distance_name, _classical_mds
from analysis import *

_UNSET = object()   # sentinel: "caller did not provide - read from general_settings"


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
            f"  Current path  : {pretty_path(ic_json_path)}\n"
            f"  Falling back  : {pretty_path(_OLD_REPO_IC_JSON_PATH)}\n"
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
    print(f"Loading IC JSON: {pretty_path(ic_json_path)}")
    with open(ic_json_path, "r", encoding="utf-8-sig") as ic_file_handle:
        ic_data = json.load(ic_file_handle)
    ic_results = ic_data.get("ic_results", {})
    n_models_in_json = len(ic_results)
    print(
        f"  {n_models_in_json} models found in IC JSON "
        f"(utility universe has 505 — {505 - n_models_in_json} not present in this file)."
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
    print(f"Saved: {pretty_path(output_csv_path)}  ({len(combined_fits_df)} rows)")

    return combined_fits_df


"=========================================================================================="
"==================== Participant Cloud Distances and Architecture Embedding =============="
"=========================================================================================="

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
        print(f"Participant model-space centroids loaded from cache: {pretty_path(output_csv_path)}")
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
            f"Model-space embedding not found: {pretty_path(embedding_csv_path)}\n"
            "Run compute_model_space_embedding first."
        )
    model_embedding_df = pd.read_csv(embedding_csv_path, dtype={"utility_bitstring": str})
    print(f"Loaded model-space embedding: {pretty_path(embedding_csv_path)}  ({len(model_embedding_df)} models)")

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
    print(f"Saved: {pretty_path(output_csv_path)}  ({len(centroids_df)} participants)")
    return centroids_df


def compute_participant_cloud_distances(
    general_settings: dict,
    file_paths: dict,
    create_new_file: bool = False,
) -> dict:
    """
    Computes pairwise participant cloud distances using the energy distance formulation.
    Each participant is treated as a BIC-weight distribution over utility models (from
    extract_participant_model_combined_fits); the AMPD matrix (from compute_ampd_matrix)
    provides the distance geometry over model space.

    Two matrices are computed and saved:
      cross_distance_ij   = bic_weights_i @ ampd_matrix @ bic_weights_j
      energy_distance_ij  = sqrt(max(0, 2*cross_ij - within_i - within_j))
    where within_i = bic_weights_i @ ampd_matrix @ bic_weights_i.

    energy_distance(participant, participant) = 0 by construction. The cross matrix is
    also saved as a descriptive similarity measure for downstream stages.

    Arguments:
        • general_settings: dict — used to load the AMPD matrix via
            compute_ampd_matrix.
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
    ampd_matrix_df = compute_ampd_matrix(
        general_settings=general_settings, file_paths=file_paths, create_new_file=False,
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
    print(f"Saved cross distance matrix:  {pretty_path(cross_csv_path)}")
    print(f"Saved energy distance matrix: {pretty_path(energy_csv_path)}")

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
            print(f"Participant architecture embedding loaded from cache: {pretty_path(output_csv_path)}")
            return cached_embedding_df

    energy_distance_csv_path = os.path.join(
        file_paths["processed"],
        "participant_cloud_distance_matrix__metric=ampd_energy.csv",
    )
    if not os.path.exists(energy_distance_csv_path):
        raise FileNotFoundError(
            f"Energy distance matrix not found: {pretty_path(energy_distance_csv_path)}\n"
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
    print(f"Saved: {pretty_path(output_csv_path)}  ({len(embedding_df)} participants, {n_dimensions}D)")
    return embedding_df


def compute_participant_feature_support(
    general_settings: dict,
    file_paths: dict,
    utility_settings: dict,
    create_new_file: bool = False,
) -> pd.DataFrame:
    """
    Computes each participant's BIC-weighted probability that each Boolean utility setting
    is active, using the model cloud BIC weights from extract_participant_model_combined_fits.
    For each participant and each utility setting, P_i(setting=True) = sum_m w_i,m * setting_m_value, 
    which gives the expected presence of that architectural feature weighted by model plausibility.

    Arguments:
        • general_settings: dict — unused directly; kept for consistency with repo pattern.
        • file_paths: dict — must contain 'processed'.
        • utility_settings: dict — the canonical UtilitySettings dict; its keys determine
            which Boolean columns are read from the combined fits CSV.
        • create_new_file: bool (default False) — if False and the output CSV exists, load
            and return it without recomputing.

    Returns:
        • pd.DataFrame — one row per participant, columns: player_uuid, P_<setting> per key
            in utility_settings, effective_number_of_models, model_weight_entropy.
    """
    output_csv_path = os.path.join(
        file_paths["processed"], "participant_feature_support.csv",
    )
    if not create_new_file and os.path.exists(output_csv_path):
        print(f"Participant feature support loaded from cache: {pretty_path(output_csv_path)}")
        return pd.read_csv(output_csv_path)

    combined_fits_df = pd.read_csv(
        os.path.join(file_paths["processed"], "participant_model_combined_fits.csv"),
    )

    settings_cols_present = [col for col in utility_settings.keys() if col in combined_fits_df.columns]
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
    print(f"Saved: {pretty_path(output_csv_path)}  ({len(feature_support_df)} participants, {len(settings_cols_present)} settings)")
    return feature_support_df


"=========================================================================================="
"================= Population Architecture Compression Curve ============================="
"=========================================================================================="

def _exhaustive_search_worker(args: tuple) -> tuple:
    """
    Module-level parallel worker for exhaustive M-architecture set search.

    Evaluates a batch of candidate architecture sets against a participant × model
    score matrix and returns the best-scoring set in the batch.

    Arguments:
        • args: (combo_batch, L)
            combo_batch: list of M-tuples (column indices into L)
            L: np.ndarray, shape (N_participants, N_candidates)

    Returns:
        • (best_score: float, best_set: tuple of ints)
    """
    combo_batch, L = args
    if not combo_batch:
        return np.inf, None
    arr    = np.array(combo_batch, dtype=np.int32)
    "L[:, arr] is (N_participants, B, M); min(axis=2) -> (N_participants, B); sum(axis=0) -> (B,)."
    scores = L[:, arr].min(axis=2).sum(axis=0)
    best_i = int(scores.argmin())
    return float(scores[best_i]), tuple(combo_batch[best_i])


def compute_architecture_compression_curve(
    general_settings: dict,
    file_paths: dict,
    population_top_n_models=_UNSET,
    participant_top_r_models=_UNSET,
    M_max=_UNSET,
    exhaustive_M_max=_UNSET,
    score_basis=_UNSET,
    stopping_criteria=_UNSET,
    marginal_gain_threshold=_UNSET,
    n_consecutive_low_marginal_gains_required=_UNSET,
    cumulative_gain_threshold=_UNSET,
    diagnose_selected_library_redundancy=_UNSET,
    n_workers=_UNSET,
    create_new_file: bool = False,
    param_bds=None,
    utility_settings=None,
) -> pd.DataFrame:
    """
    Population architecture compression curve.

    For M = 1, 2, 3, … finds the library of M utility function architectures minimising
    total population BIC under hard assignment (each participant uses their best-fitting
    architecture in the library).  The compression curve A(M) tracks what fraction of the
    fully-individualised BIC advantage is recovered as M grows.

    Arguments:
        • general_settings: dict; Project-wide settings (must include 'temperature_is_param').
        • file_paths: dict; Must include 'processed', 'visuals', and 'file_names'.
        • population_top_n_models: int | None; Top-N models by aggregate population BIC to
            include as candidates.  None includes all models.
        • participant_top_r_models: int | None; Top-R models per participant by individual
            BIC included as candidates.  None includes all.
        • M_max: int | None; Hard ceiling on M.  None runs until stopping criterion fires
            or M_useful_max is reached.
        • exhaustive_M_max: int; Maximum M for exhaustive combination search (default 4).
            Larger M uses greedy extension + local-swap refinement.
        • score_basis: Literal; Scoring basis for the participant × model matrix.  Primary is
            'ic_equivalent_participant_score', ensuring M=1 matches the IC champion.
        • stopping_criteria: Literal; Criterion highlighted in terminal output.  All five are
            always computed and saved.
        • marginal_gain_threshold: float; ΔA threshold for marginal-gain criterion.
        • n_consecutive_low_marginal_gains_required: int; Consecutive low-gain M values
            required before marginal-gain criterion fires.
        • cumulative_gain_threshold: float; A(M) threshold for cumulative-gain criterion.
        • diagnose_selected_library_redundancy: bool; Compute per-architecture pruning cost,
            assignment counts, and AMPD similarity flags.
        • n_workers: int | None; Worker count for parallel exhaustive search.
            None = cpu_count − 1.
        • create_new_file: bool; If False and the final CSV exists, load and return it without
            recomputation.  Partial CSV enables resume after interruption.
        • param_bds: ParamBounds | None; Required to auto-generate the AMPD matrix when not
            found on disk. If None and the matrix is missing, raises FileNotFoundError.
        • utility_settings: UtilitySettings | None; Same requirement as param_bds.

    Returns:
        • pd.DataFrame: one row per M.
    """

    "Resolve settings: explicit kwargs take priority; fall back to general_settings nested dict."
    ia = general_settings.get('individual_architecture_settings', {})
    if population_top_n_models               is _UNSET: population_top_n_models               = ia.get('population_top_n_models', 120)
    if participant_top_r_models              is _UNSET: participant_top_r_models              = ia.get('participant_top_r_models', 10)
    if M_max                                 is _UNSET: M_max                                 = ia.get('M_max', None)
    if exhaustive_M_max                      is _UNSET: exhaustive_M_max                      = ia.get('exhaustive_M_max', 4)
    if score_basis                           is _UNSET: score_basis                           = ia.get('score_basis', 'ic_equivalent_participant_score')
    if stopping_criteria                     is _UNSET: stopping_criteria                     = ia.get('stopping_criteria', 'kneedle_elbow')
    if marginal_gain_threshold               is _UNSET: marginal_gain_threshold               = ia.get('marginal_gain_threshold', 0.01)
    if n_consecutive_low_marginal_gains_required is _UNSET: n_consecutive_low_marginal_gains_required = ia.get('n_consecutive_low_marginal_gains_required', 1)
    if cumulative_gain_threshold             is _UNSET: cumulative_gain_threshold             = ia.get('cumulative_gain_threshold', 0.80)
    if diagnose_selected_library_redundancy  is _UNSET: diagnose_selected_library_redundancy  = ia.get('diagnose_selected_library_redundancy', True)
    if n_workers                             is _UNSET: n_workers                             = ia.get('n_workers', None)

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
    score_M1       = float(pop_col_scores[k1_col_idx])

    if k1_model_idx == ic_winner_idx:
        print(f"M=1 self-check PASSED: best single architecture = utility index {k1_model_idx},  score = {score_M1:.2f}")
    else:
        ic_col      = cand_col_map.get(ic_winner_idx)
        ic_score_k1 = float(L[:, ic_col].sum()) if ic_col is not None else float('nan')
        print(f"M=1 WARNING: selected utility index {k1_model_idx} (score={score_M1:.2f}) "
              f"differs from IC winner (index {ic_winner_idx}, score={ic_score_k1:.2f}) — may reflect candidate filtering.")

    score_fully_indiv     = float(L.min(axis=1).sum())
    unique_personal_bests = set(int(candidate_model_idxs[jdx]) for jdx in L.argmin(axis=1))
    M_useful_max          = len(unique_personal_bests)
    delta_A_range         = score_M1 - score_fully_indiv
    print(f"Anchor scores: M=1 score = {score_M1:.2f},  fully-individualised score = {score_fully_indiv:.2f},  useful maximum M = {M_useful_max}")

    "=== Resume from partial CSV if present ==="
    completed_M_rows: list = []
    max_completed_M        = 0
    if not create_new_file and os.path.exists(partial_csv):
        try:
            partial_df       = pd.read_csv(partial_csv)
            completed_M_rows = partial_df.to_dict('records')
            completed_Ms     = {int(row['M']) for row in completed_M_rows}
            max_completed_M  = max(completed_Ms) if completed_Ms else 0
            print(f"Resuming from partial run: {len(completed_Ms)} M values already computed (up to M={max_completed_M}).")
        except Exception as _resume_exc:
            print(f"Could not load partial results ({_resume_exc}); starting from scratch.")
            completed_M_rows = []
            max_completed_M  = 0

    if completed_M_rows:
        last_set_idxs     = json.loads(completed_M_rows[-1]['architecture_set_idxs'])
        prev_set_col_idxs = [cand_col_map[model_idx] for model_idx in last_set_idxs if model_idx in cand_col_map]
        current_min_pp    = (L[:, prev_set_col_idxs].min(axis=1)
                             if prev_set_col_idxs else np.full(N_participants, np.inf))
        prev_A_M          = float(completed_M_rows[-1].get('A_M', 0.0))
        low_gain_streak   = 0
        for _cr in reversed(completed_M_rows):
            _da = _cr.get('delta_A_M')
            if _da is None or (isinstance(_da, float) and np.isnan(float(_da))):
                break
            low_gain_streak = (low_gain_streak + 1 if float(_da) < marginal_gain_threshold else 0)
            if float(_da) >= marginal_gain_threshold:
                break
    else:
        prev_set_col_idxs = []
        current_min_pp    = np.full(N_participants, np.inf)
        prev_A_M          = 0.0
        low_gain_streak   = 0

    "=== Search settings ==="
    n_workers_actual = (max(mp.cpu_count() - 1, 1) if n_workers is None
                        else max(1, min(int(n_workers), mp.cpu_count() - 1)))
    BATCH_SIZE = 100_000
    M_run_max  = M_max if M_max is not None else M_useful_max
    start_M    = max_completed_M + 1

    def _gen_batches(combo_iter):
        batch = []
        for item in combo_iter:
            batch.append(item)
            if len(batch) == BATCH_SIZE:
                yield batch
                batch = []
        if batch:
            yield batch

    "=== Main M loop ==="
    curve_start_time = time.time()

    for M in range(start_M, M_run_max + 1):
        M_t0 = time.time()

        if M <= exhaustive_M_max:
            "Exhaustive: evaluate all C(N_candidates, M) sets in batches of BATCH_SIZE."
            search_method  = 'exhaustive'
            best_score_M   = np.inf
            best_set_M_tup = None
            combo_iter     = it.combinations(range(N_candidates), M)
            if n_workers_actual > 1:
                tasks_gen = ((batch, L) for batch in _gen_batches(combo_iter))
                with mp.Pool(processes=n_workers_actual) as pool:
                    for b_score, b_set in pool.imap_unordered(
                            _exhaustive_search_worker, tasks_gen, chunksize=1):
                        if b_score < best_score_M:
                            best_score_M = b_score;  best_set_M_tup = b_set
            else:
                for batch in _gen_batches(combo_iter):
                    b_score, b_set = _exhaustive_search_worker((batch, L))
                    if b_score < best_score_M:
                        best_score_M = b_score;  best_set_M_tup = b_set
            if best_set_M_tup is None:
                best_set_M_tup = tuple(range(min(M, N_candidates)))
            best_set_col_idxs = list(best_set_M_tup)

        else:
            "Greedy extension from M-1 set, followed by local-swap refinement."
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
                print(f"Greedy extension found no improvement at M={M}; stopping search.")
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
            best_score_M      = float(L[:, candidate_set].min(axis=1).sum())

        current_min_pp      = L[:, best_set_col_idxs].min(axis=1)
        prev_set_col_idxs   = list(best_set_col_idxs)
        best_set_model_idxs = [int(candidate_model_idxs[jdx]) for jdx in best_set_col_idxs]

        A_M       = (score_M1 - best_score_M) / delta_A_range if delta_A_range > 0 else 1.0
        delta_A_M = float(A_M - prev_A_M) if M > 1 else float('nan')
        prev_A_M  = A_M

        raw_nll_score_M  = float(L_nll[:, best_set_col_idxs].min(axis=1).sum())
        sum_ibic_score_M = float(L_ibic[:, best_set_col_idxs].min(axis=1).sum())
        ic_equiv_score_M = float(L_ic[:, best_set_col_idxs].min(axis=1).sum())

        k_elapsed     = time.time() - M_t0
        curve_elapsed = time.time() - curve_start_time
        n_done        = M - start_M + 1
        n_todo        = M_run_max - start_M + 1
        eta_str       = (_fmt_duration((n_todo - n_done) / (n_done / curve_elapsed))
                         if n_done > 0 and curve_elapsed > 0 else "unknown")
        delta_str     = f"{delta_A_M:.4f}" if not np.isnan(delta_A_M) else "n/a"
        print(f"M={M}  A(M)={A_M:.4f}  ΔA={delta_str}  "
              f"set={best_set_model_idxs}  ({search_method}, {_fmt_duration(k_elapsed)})  ETA={eta_str}")

        completed_M_rows.append({
            'M':                              M,
            'architecture_set_idxs':          json.dumps(best_set_model_idxs),
            'search_method':                  search_method,
            'score_basis':                    score_basis,
            'raw_nll_score_M':                raw_nll_score_M,
            'sum_individual_BIC_score_M':     sum_ibic_score_M,
            'ic_equivalent_score_M':          ic_equiv_score_M,
            'score_M1':                       score_M1,
            'score_fully_individualized':     score_fully_indiv,
            'A_M':                            A_M,
            'delta_A_M':                      delta_A_M if not np.isnan(delta_A_M) else None,
            'M_useful_max':                   M_useful_max,
            'n_unique_individual_best_models': M_useful_max,
        })
        pd.DataFrame(completed_M_rows).to_csv(partial_csv, index=False, encoding='utf-8-sig')

        if M_max is None:
            if stopping_criteria == 'marginal_gain' and not np.isnan(delta_A_M):
                if delta_A_M < marginal_gain_threshold:
                    low_gain_streak += 1
                    if low_gain_streak >= n_consecutive_low_marginal_gains_required:
                        print(f"Marginal gain stopping criterion reached at M={M}; halting search.")
                        break
                else:
                    low_gain_streak = 0
            elif stopping_criteria == 'cumulative_gain' and A_M >= cumulative_gain_threshold:
                print(f"Cumulative gain stopping criterion reached: A(M={M}) = {A_M:.4f}; halting search.")
                break

    "=== Post-hoc: all five stopping criteria ==="
    curve_df = pd.DataFrame(completed_M_rows)
    M_vals   = curve_df['M'].tolist()
    A_vals   = curve_df['A_M'].tolist()
    n_rows   = len(curve_df)

    da_list  = [row.get('delta_A_M') for row in completed_M_rows]
    d2a_list = [None] * n_rows
    for row_idx in range(2, n_rows):
        da_k, da_km1 = da_list[row_idx], da_list[row_idx - 1]
        if da_k is not None and da_km1 is not None:
            d2a_list[row_idx] = float(da_k - da_km1)
    curve_df['delta2_A_M'] = d2a_list

    M_span  = max(M_vals) - min(M_vals) or 1
    A_max   = max(A_vals) or 1.0
    kneedle = [(a / A_max) - ((k_val - min(M_vals)) / M_span) for k_val, a in zip(M_vals, A_vals)]
    curve_df['kneedle_distance']          = kneedle
    kneedle_best_M                        = M_vals[int(np.argmax(kneedle))]
    curve_df['selected_by_kneedle_elbow'] = [k_val == kneedle_best_M for k_val in M_vals]

    first_low_mg_M = None;  selected_mg_M = None;  mg_streak = 0
    for _ri, _row in curve_df.iterrows():
        _da = _row['delta_A_M']
        if _da is None or (isinstance(_da, float) and np.isnan(float(_da))):
            continue
        if float(_da) < marginal_gain_threshold:
            mg_streak += 1
            if mg_streak >= n_consecutive_low_marginal_gains_required and first_low_mg_M is None:
                first_low_mg_M = int(_row['M']);  selected_mg_M = first_low_mg_M - 1
        else:
            mg_streak = 0
    curve_df['first_M_with_low_marginal_gain'] = first_low_mg_M
    curve_df['selected_M_by_marginal_gain']    = selected_mg_M
    curve_df['selected_by_marginal_gain']      = [k_val == selected_mg_M for k_val in M_vals]

    selected_cg_M = None
    for _ri, _row in curve_df.iterrows():
        if _row['A_M'] >= cumulative_gain_threshold:
            selected_cg_M = int(_row['M']);  break
    curve_df['selected_by_cumulative_gain'] = [k_val == selected_cg_M for k_val in M_vals]

    d2_abs = [abs(val) if val is not None else -np.inf for val in d2a_list]
    mc_idx = int(np.argmax(d2_abs))
    selected_mc_M = M_vals[mc_idx] if d2_abs[mc_idx] > 0 else None
    curve_df['selected_by_max_curvature'] = [k_val == selected_mc_M for k_val in M_vals]

    model_keff_lookup = fits_df.drop_duplicates('utility_idx').set_index('utility_idx')['k_effective']
    meta_bic_vals     = []
    ic_scores_list    = curve_df['ic_equivalent_score_M'].tolist()
    for row_idx, row in curve_df.iterrows():
        set_idxs  = json.loads(row['architecture_set_idxs'])
        mean_keff = float(model_keff_lookup.reindex(set_idxs).mean())
        meta_bic_vals.append(float(
            2 * ic_scores_list[row_idx] + row['M'] * mean_keff * np.log(max(N_participants, 1))
        ))
    curve_df['exploratory_meta_bic'] = meta_bic_vals
    selected_metabic_M               = M_vals[int(np.argmin(meta_bic_vals))]
    curve_df['selected_by_meta_bic'] = [k_val == selected_metabic_M for k_val in M_vals]

    "=== H_M: fraction of explainable improvement from chance to fully-individualised ceiling ==="
    # H(M) = (BIC_chance − score_M) / (BIC_chance − score_fully_individualized)
    # Fixed denominator: BIC_chance and score_fully_individualized do not change with M.
    # BIC_chance = 2 × n_total_games × log(2) — the BIC of a 50/50 random predictor (k=0).
    # This differs from H_form_K in architecture_codebook_summary.csv, which uses a moving
    # denominator (NLL_chance − NLL(K)) and is therefore not bounded [0,1] in the same way.
    n_total_games   = int(fits_df.groupby('player_uuid')['n_combined'].first().sum())
    bic_chance      = 2 * n_total_games * np.log(2)
    bic_chance_denom = bic_chance - score_fully_indiv
    curve_df['H_M'] = (bic_chance - curve_df['ic_equivalent_score_M']) / bic_chance_denom
    curve_df['absolute_gap_closed_versus_chance'] = bic_chance - curve_df['ic_equivalent_score_M']

    "=== AMPD matrix ==="
    ampd_idx_set = set();  ampd_col_set = set();  all_ampd_pos = np.array([])
    ampd_df = compute_ampd_matrix(
        general_settings=general_settings, file_paths=file_paths,
        param_bds=param_bds, utility_settings=utility_settings,
        create_new_file=False,
    )
    ampd_df.index   = ampd_df.index.astype(int)
    ampd_df.columns = ampd_df.columns.astype(int)
    ampd_idx_set    = set(ampd_df.index.tolist())
    ampd_col_set    = set(ampd_df.columns.tolist())
    flat            = ampd_df.values.flatten()
    all_ampd_pos    = flat[~np.isnan(flat) & (flat > 0)]
    print(f"AMPD behavioral-distance matrix loaded: {ampd_df.shape[0]}×{ampd_df.shape[1]}")

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
        M_val        = int(row['M'])
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
                'M': M_val, 'player_uuid': player_uuid,
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
            M_val        = int(row['M'])
            set_idxs     = json.loads(row['architecture_set_idxs'])
            set_col_idxs = [cand_col_map[model_idx] for model_idx in set_idxs if model_idx in cand_col_map]
            if not set_col_idxs:
                continue
            score_M_val = float(L[:, set_col_idxs].min(axis=1).sum())
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
                pruning_cost = float(score_without - score_M_val)
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
                    'M': M_val, 'utility_idx': model_idx, 'equation': equation,
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

        "=== Human-readable summary table: one row per (M, model), all info a researcher needs ==="
        k_level_cols = [
            'M', 'A_M', 'delta_A_M', 'delta2_A_M', 'kneedle_distance',
            'selected_by_kneedle_elbow', 'selected_by_marginal_gain',
            'selected_by_cumulative_gain', 'selected_by_max_curvature', 'selected_by_meta_bic',
            'H_M', 'absolute_gap_closed_versus_chance',
        ]
        ic_bic_lookup        = ic_df.set_index('idx')['BIC'].to_dict()
        summary_df           = diag_df.rename(columns={
            'assigned_n':      'n_players_assigned',
            'assigned_percent': 'pct_players_assigned',
        }).copy()
        summary_df['population_IC_BIC'] = summary_df['utility_idx'].map(ic_bic_lookup)
        curve_k_indexed = curve_df.set_index('M')
        for col in [col for col in k_level_cols if col != 'M' and col in curve_k_indexed.columns]:
            summary_df[col] = summary_df['M'].map(curve_k_indexed[col].to_dict())
        front_cols  = [
            'M_models', 'A_M', 'delta_A_M', 'delta2_A_M', 'kneedle_distance', 'selected_by_kneedle_elbow', 
            'selected_by_marginal_gain', 'selected_by_cumulative_gain', 'selected_by_max_curvature', 
            'selected_by_meta_bic', 'H_M', 'absolute_gap_closed_versus_chance', 'utility_idx', 'k_params', 
            'population_IC_BIC', 'n_players_assigned', 'pct_players_assigned', 'mean_individual_BIC', 
            'pruning_cost', 'pruning_cost_normalized', 'nearest_selected_model_idx', 
            'nearest_selected_model_ampd', 'nearest_selected_model_ampd_percentile',
            'redundancy_warning_level', 'redundancy_score_optional',
        ]
        "Sort within each M by mean_individual_BIC ascending (best-fitting models first)."
        summary_df = summary_df.sort_values(['M', 'mean_individual_BIC'], ascending=[True, True])
        "Utility settings columns come after diagnostics; equation is always last so it spills into empty Excel cells."
        util_cols    = [col for col in summary_df.columns if col not in front_cols and col != 'equation']
        ordered_cols = [col for col in front_cols if col in summary_df.columns] + util_cols + ['equation']
        summary_csv  = os.path.join(proc_dir, 'population_architecture_summary_table.csv')
        summary_df[ordered_cols].to_csv(summary_csv, index=False, encoding='utf-8-sig')
        print(f"Architecture summary table saved: {os.path.basename(summary_csv)}")

    if os.path.exists(partial_csv):
        os.remove(partial_csv)

    return curve_df


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
    generating_model=_UNSET,
    n_agents_grid=_UNSET,
    n_games_grid=_UNSET,
    softmax_temperature=_UNSET,
    candidate_model_selection_mode=_UNSET,
    n_candidate_models=_UNSET,
    ampd_matrix_name_or_path=_UNSET,
    random_seed=_UNSET,
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

    "Resolve settings: explicit kwargs take priority; fall back to general_settings nested dict."
    mr = general_settings.get('model_recovery_settings', {})
    if generating_model               is _UNSET: generating_model               = mr.get('generating_model', 443)
    if n_agents_grid                  is _UNSET: n_agents_grid                  = mr.get('n_agents_grid', None)
    if n_games_grid                   is _UNSET: n_games_grid                   = mr.get('n_games_grid', None)
    if softmax_temperature            is _UNSET: softmax_temperature            = mr.get('softmax_temperature', 0.5)
    if candidate_model_selection_mode is _UNSET: candidate_model_selection_mode = mr.get('candidate_model_selection_mode', 'hamming')
    if n_candidate_models             is _UNSET: n_candidate_models             = mr.get('n_candidate_models', 100)
    if ampd_matrix_name_or_path       is _UNSET: ampd_matrix_name_or_path       = mr.get('ampd_matrix_name_or_path', None)
    if random_seed                    is _UNSET: random_seed                    = mr.get('random_seed', 42)

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
    _sim_results_dir      = str(file_paths['simulations'])
    os.makedirs(_sim_results_dir, exist_ok=True)
    _gitignore_path = os.path.join(_sim_results_dir, '.gitignore')
    if not os.path.exists(_gitignore_path):
        with open(_gitignore_path, 'w', encoding='utf-8') as _gig:
            _gig.write('*\n')
    registry_df   = all_utility_functions_dataframe(file_paths=file_paths, general_settings=general_settings)
    _non_flag_columns = {
        'utility_idx', 'utility_bitstring', 'k_params', 'redundant_with', 'differing_settings',
        'n_data', 'pvar', 'param_norm_sd', 'loss_nll', 'AIC', 'BIC', 'ΔAIC', 'ΔBIC',
        'AIC_rank', 'BIC_rank', 'parents', 'siblings', 'children',
        'ampd_to_best', 'policy_regret_norm_to_best', 'equation',
        'canonical_model',
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
    _stem = _recovery_simulation_stem(
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
            print(f"Model recovery simulation loaded from cache: {pretty_path(output_csv_path)}"
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
            distance_matrix_df = compute_ampd_matrix(
                general_settings=general_settings, file_paths=file_paths,
                param_bds=param_bds, utility_settings=utility_settings, create_new_file=False,
            )
        distance_matrix_df.index   = distance_matrix_df.index.astype(int)
        distance_matrix_df.columns = distance_matrix_df.columns.astype(int)
    else:
        n_registry_models   = len(registry_df)
        hamming_matrix_path = os.path.join(
            processed_dir, f'model_distance_conditional_hamming__n_models={n_registry_models}.csv'
        )
        if not os.path.exists(hamming_matrix_path):
            print("Conditional Hamming distance matrix not found; computing now...")
            compute_conditional_hamming_distance_matrix(
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

    "Load AMPD and conditional Hamming matrices for continuous recovery distance metrics."
    if candidate_model_selection_mode == 'ampd':
        ampd_metrics_df = distance_matrix_df
    else:
        if ampd_matrix_name_or_path is not None:
            _ampd_metr_path = (
                ampd_matrix_name_or_path if os.path.isabs(ampd_matrix_name_or_path)
                else os.path.join(processed_dir, ampd_matrix_name_or_path)
            )
            ampd_metrics_df = pd.read_csv(_ampd_metr_path, index_col=0)
        else:
            ampd_metrics_df = compute_ampd_matrix(
                general_settings=general_settings, file_paths=file_paths,
                param_bds=param_bds, utility_settings=utility_settings, create_new_file=False,
            )
        ampd_metrics_df.index   = ampd_metrics_df.index.astype(int)
        ampd_metrics_df.columns = ampd_metrics_df.columns.astype(int)
        print(f"AMPD matrix loaded for recovery metrics: "
              f"{ampd_metrics_df.shape[0]}×{ampd_metrics_df.shape[1]}")

    try:
        cond_hamming_metrics_df = compute_conditional_hamming_distance_matrix(
            file_paths=file_paths, utility_settings=utility_settings,
        )
        cond_hamming_metrics_df.index   = cond_hamming_metrics_df.index.astype(int)
        cond_hamming_metrics_df.columns = cond_hamming_metrics_df.columns.astype(int)
    except Exception as _ch_err:
        print(f"  Warning: could not load conditional Hamming matrix ({_ch_err}). "
              f"Conditional Hamming metrics will be NaN.")
        cond_hamming_metrics_df = None

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

    "Fallback: if the IC JSON is absent from the repo's bic_aic/, try the original analysis"
    "directory as a convenience for the primary author's machine.  For all other users, the"
    "JSON must exist in bic_aic/ (generated by information_criterion_analysis() or provided"
    "by the authors).  A missing file raises a clear error rather than an opaque crash."
    _fallback_ic_json_path = (
        r"C:\Users\Gregory Stanley\Desktop\U of M\Research Archive\Multiplayer"
        r"\ABM_Simulation\Judgment_Game\Inputs\Iter_Binary_Dictator"
        rf"\bic_aic\All_Utility_Forms_IC_Analysis_Experiment{experiment_num}.json"
    )
    if not os.path.exists(ic_json_path) or os.path.getsize(ic_json_path) < 50_000_000:
        if os.path.exists(_fallback_ic_json_path):
            ic_json_path = _fallback_ic_json_path
        elif not os.path.exists(ic_json_path):
            raise FileNotFoundError(
                f"IC JSON not found at the expected location:\n  {pretty_path(ic_json_path)}\n"
                "To resolve: place All_Utility_Forms_IC_Analysis_Experiment3.json in the\n"
                "bic_aic/ directory of this repo. The file is generated by\n"
                "information_criterion_analysis() or can be obtained from the authors."
            )

    print(f"Loading IC JSON: {os.path.basename(ic_json_path)}")
    with open(ic_json_path, 'r', encoding='utf-8-sig') as ic_file_handle:
        ic_data = json.load(ic_file_handle)
    ic_results = ic_data.get('ic_results', {})

    """
    Find generating model entry by settings tuple — not by utility_idx.
    The IC JSON was built from a different model registry (different total model count),
    so integer indices may differ. The settings tuple is the stable cross-version identity.
    """
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
            f"  IC JSON path:   {pretty_path(ic_json_path)}\n"
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
            player_1_uuid=f"synthetic_agent_{agent_idx}_chooser",
            player_2_uuid=f"synthetic_agent_{agent_idx}_predictor",
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
    print(f"Max-scale synthetic data saved: {pretty_path(max_scale_synthetic_path)}")

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
                str(file_paths['simulations']), 'model_recovery_simulation',
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
            ic_df['cond_hamming_to_truth'] = ic_df['utility_idx'].apply(
                lambda uid: _dist_lookup(cond_hamming_metrics_df, uid, generating_utility_idx)
            )

            _ampd_winner         = _dist_lookup(ampd_metrics_df,         _winner_uid, generating_utility_idx)
            _cond_hamming_winner = _dist_lookup(cond_hamming_metrics_df, _winner_uid, generating_utility_idx)

            from scipy.stats import spearmanr as _spearmanr
            _valid_mask_a = ic_df['ampd_to_truth'].notna()
            if _valid_mask_a.sum() >= 3:
                _rho_a, _pval_a = _spearmanr(
                    ic_df.loc[_valid_mask_a, 'bic_rank_overall'],
                    ic_df.loc[_valid_mask_a, 'ampd_to_truth'],
                )
                _spear_r, _spear_p = float(_rho_a), float(_pval_a)
            else:
                _spear_r, _spear_p = float('nan'), float('nan')

            _valid_mask_h = ic_df['cond_hamming_to_truth'].notna()
            if _valid_mask_h.sum() >= 3:
                _rho_h, _pval_h = _spearmanr(
                    ic_df.loc[_valid_mask_h, 'bic_rank_overall'],
                    ic_df.loc[_valid_mask_h, 'cond_hamming_to_truth'],
                )
                _ch_spear_r, _ch_spear_p = float(_rho_h), float(_pval_h)
            else:
                _ch_spear_r, _ch_spear_p = float('nan'), float('nan')

            ic_df['bic_rank_true_model']           = _bic_rank_true
            ic_df['rank_percentile_true_model']     = _rank_pct_true
            ic_df['ampd_winner_to_truth']           = _ampd_winner
            ic_df['cond_hamming_winner_to_truth']   = _cond_hamming_winner
            ic_df['ampd_rank_spearman_r']           = _spear_r
            ic_df['ampd_rank_spearman_p']           = _spear_p
            ic_df['cond_hamming_rank_spearman_r']   = _ch_spear_r
            ic_df['cond_hamming_rank_spearman_p']   = _ch_spear_p

            "Select and rename columns for the output CSV."
            _output_col_map = {
                'loss': 'nll_population', 'AIC': 'aic_population',
                'BIC': 'bic_population',  'ΔBIC': 'delta_bic', 'n_data': 'n_data_population',
            }
            _keep_cols = [
                'n_agents_fitted', 'n_games_fitted', 'utility_idx', 'true_utility_idx',
                'bic_rank_overall', 'is_generating_model', 'recovered',
                'loss', 'k_params', 'AIC', 'BIC', 'ΔBIC', 'n_data',
                'ampd_to_truth', 'cond_hamming_to_truth',
                'bic_rank_true_model', 'rank_percentile_true_model',
                'ampd_winner_to_truth', 'cond_hamming_winner_to_truth',
                'ampd_rank_spearman_r', 'ampd_rank_spearman_p',
                'cond_hamming_rank_spearman_r', 'cond_hamming_rank_spearman_p',
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
                  f"cond_hamming_winner={_cond_hamming_winner:.0f}  "
                  f"ampd_r={_spear_r:.2f}  "
                  f"ch_r={_ch_spear_r:.2f}  "
                  f"time={_fmt_duration(condition_elapsed)}")

    "Combine all conditions, write final CSV, delete partial."
    all_results_df = (
        pd.concat(accumulated_dataframes, ignore_index=True)
        if accumulated_dataframes else pd.DataFrame()
    )
    all_results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\nModel recovery simulation saved: {pretty_path(output_csv_path)}  ({len(all_results_df)} rows)")
    print(f"Total time: {_fmt_duration(time.time() - total_start_time)}")

    if os.path.exists(partial_csv_path):
        os.remove(partial_csv_path)

    return all_results_df


def plot_model_recovery_simulation(
    general_settings: dict,
    file_paths: dict,
    figure_layout: dict,
    generating_model=_UNSET,
    n_candidate_models=_UNSET,
    candidate_model_selection_mode=_UNSET,
    softmax_temperature=_UNSET,
    n_agents_grid=_UNSET,
    n_games_grid=_UNSET,
    random_seed=_UNSET,
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
      ampd_winner_to_truth          → 1 - ampd                    [invert: 0=identical=best]
      cond_hamming_winner_to_truth  → 1 - cond_hamming / 14        [invert, 14 = max live flags]
      ampd_rank_spearman_r          → (r + 1) / 2                  [map [-1,1] → [0,1]]
      cond_hamming_rank_spearman_r  → (r + 1) / 2                  [map [-1,1] → [0,1]]

    Arguments:
        • general_settings: dict; accepted for API consistency (not currently used).
        • file_paths: dict; must contain 'processed' and 'visuals'.
        • figure_layout: dict; layout settings (template, font, title_size, base_hue).
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
    "Resolve settings: explicit kwargs take priority; fall back to general_settings nested dict."
    mr = general_settings.get('model_recovery_settings', {})
    if generating_model               is _UNSET: generating_model               = mr.get('generating_model', 443)
    if n_candidate_models             is _UNSET: n_candidate_models             = mr.get('n_candidate_models', 100)
    if candidate_model_selection_mode is _UNSET: candidate_model_selection_mode = mr.get('candidate_model_selection_mode', 'hamming')
    if softmax_temperature            is _UNSET: softmax_temperature            = mr.get('softmax_temperature', 0.5)
    if n_agents_grid                  is _UNSET: n_agents_grid                  = mr.get('n_agents_grid', None)
    if n_games_grid                   is _UNSET: n_games_grid                   = mr.get('n_games_grid', None)
    if random_seed                    is _UNSET: random_seed                    = mr.get('random_seed', 42)

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

    base_hue            = figure_layout.get('base_hue', 220)
    base_font_size      = max(8, figure_layout.get('font', {}).get('size', 28) // 2)
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
            'col':         'cond_hamming_winner_to_truth',
            'short_label': 'Cond. Hamming (inv.)',
            'label':       'Conditional Hamming: winner → truth',
            'norm_desc':   '1 − cond_hamming / 14',
            'y_title':     'Conditional Hamming distance  (winner → truth;  live flags only)',
            'y_range':     None,
            'hover_fmt':   '.0f',
        },
        {
            'col':         'ampd_rank_spearman_r',
            'short_label': 'AMPD Spearman r',
            'label':       'AMPD rank-distance Spearman r',
            'norm_desc':   '(r + 1) / 2',
            'y_title':     'Spearman r: BIC rank ↔ AMPD-to-truth  (1 = perfectly ordered)',
            'y_range':     [-1.05, 1.05],
            'hover_fmt':   '.3f',
        },
        {
            'col':         'cond_hamming_rank_spearman_r',
            'short_label': 'Cond. Hamming Spearman r',
            'label':       'Conditional Hamming rank-distance Spearman r',
            'norm_desc':   '(r + 1) / 2',
            'y_title':     'Spearman r: BIC rank ↔ cond. Hamming-to-truth  (1 = perfectly ordered)',
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
        if col == 'cond_hamming_winner_to_truth':
            return 1.0 - min(max(raw, 0.0), 14.0) / 14.0
        if col in ('ampd_rank_spearman_r', 'cond_hamming_rank_spearman_r'):
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

    "Buttons 1..8: individual metrics — always show normalized Block 1 traces, fixed [0,1] axis."
    for m_idx, m_cfg in enumerate(_metric_configs):
        block1_visible = [
            (t_idx // n_agents_count) == m_idx
            for t_idx in range(n_block)
        ]
        block2_visible = [False] * n_block
        yaxis_args = {
            'yaxis.title.text':      'Normalized recovery score  (0 = worst,  1 = perfect recovery)',
            'yaxis.title.font.size': axis_font_size,
            'yaxis.tickfont.size':   axis_font_size,
            'yaxis.range':           [-0.05, 1.05],
            'yaxis.autorange':       False,
            'yaxis.tickmode':        'auto',
        }

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
            font=dict(size=figure_layout.get('title_size', 22) * 2),
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
            font=dict(size=base_font_size + 8),
            bgcolor='white',
            bordercolor=_hsla(hue=0, saturation_percent=0, lightness_percent=60, alpha=0.8),
        )],
        hoverlabel=dict(font=dict(size=base_font_size + 2)),
        template=figure_layout.get('template', 'plotly_white'),
        font=dict(
            family=figure_layout.get('font', {}).get('family', 'Calibri'),
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
        print(f"Model recovery simulation plot saved: {pretty_path(out_path)}")

    return fig
