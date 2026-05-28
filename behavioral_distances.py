import hashlib
import time
from visualization import *
from visualization import _hsla
from utilities import compute_hamming_distance_matrix, compute_conditional_hamming_distance_matrix, all_utility_functions_dataframe


"=========================================================================================="
"============================== Average Model Policy Distance =============================="
"=========================================================================================="


def _build_ampd_cache_path(
    file_paths: FilePaths,
    metric: str,
    parameter_sampling_mode: str,
    parameter_pairing_mode: str,
    player_roles: Optional[List[str]],
    softmax_temperature: float,
    n_games: int,
    n_iters: int,
    random_seed: Optional[int],
) -> str:
    """
    Constructs the canonical cache filename for an AMPD master matrix. The master matrix
    always covers all 505 utility forms; subsets fill in only the relevant cells. This design
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
        • softmax_temperature: float — softmax temperature τ.
        • n_games: int — number of payoff structures evaluated.
        • n_iters: int — number of Monte Carlo parameter draws.
        • random_seed: int | None — seed used; None becomes 'unseeded'.

    Returns:
        • str — full path to the master cache CSV file.
    """
    tau_str = f"{softmax_temperature:.4g}".replace(".", "p")
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
    softmax_temperature: Optional[float] = None,
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
            softmax_temperature when softmax_temperature is None.
        • file_paths: FilePaths
            Must contain 'processed'; used to load the registry when utility_idx inputs
            are given and to load empirical data for realistic sampling.
        • param_bds: dict[str, tuple[float, float]]
            Bounds for all parameters; used to define the canonical reference vector space
            and as fallback for realistic sampling.
        • metric: str (default 'normalized_jsd')
            Distance metric. Currently supported: 'normalized_jsd'.
        • softmax_temperature: float | None
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
            compute_ampd_matrix; maps each parameter name to the full pool of
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
                registry_df = all_utility_functions_dataframe(file_paths=file_paths, general_settings=general_settings)
            row = registry_df[registry_df["utility_idx"] == model_ref]
            if len(row) == 0:
                raise ValueError(f"utility_idx {model_ref} not found in registry.")
            non_flag_cols = {
                "utility_idx", "utility_bitstring", "k_params", "redundant_with",
                "differing_settings", "n_data", "pvar", "param_norm_sd", "loss_nll",
                "AIC", "BIC", "ΔAIC", "ΔBIC", "AIC_rank", "BIC_rank", "parents", 
                "siblings", "children", "ampd_to_best", "policy_regret_norm_to_best", 
                "canonical_model", "equation",
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

    tau = softmax_temperature if softmax_temperature is not None else float(
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


def compute_policy_regret_to_best(
    winner_idx: int,
    model_idxs: List[int],
    general_settings: GeneralSettings,
    file_paths: FilePaths,
    param_bds: Dict[str, Tuple[float, float]],
    softmax_temperature: Optional[float] = None,
    n_games: int = 625,
    n_iters: int = 30,
    random_seed: Optional[int] = None,
    registry_df: Optional[pd.DataFrame] = None,
) -> Dict[int, float]:
    """
    Computes policy_regret_norm_to_best for each model relative to the BIC winner.

    For each Monte Carlo draw (shared parameters + payoff structure):
        regret_m = KL(p_winner ‖ p_m) / KL(p_winner ‖ 0.5)
    where KL(p ‖ 0.5) = log(2) − H(p). Draws where p_winner ≈ 0.5 (denominator < ε)
    are skipped. Values are averaged over all valid draws.

    Arguments:
        • winner_idx: int — utility_idx of the BIC winner.
        • model_idxs: list[int] — utility indices of all models to evaluate.
        • general_settings: GeneralSettings — for param counting and softmax τ.
        • file_paths: FilePaths — must contain 'processed'.
        • param_bds: dict — parameter bounds for uniform sampling.
        • softmax_temperature: float | None — τ; defaults to general_settings value.
        • n_games: int — payoff structures per draw (625 = exhaustive 5^4 grid).
        • n_iters: int — number of parameter draws (default 30).
        • random_seed: int | None — RNG seed.
        • registry_df: pd.DataFrame | None — pass the already-loaded registry to avoid
            re-entering all_utility_functions_dataframe (prevents recursive completeness checks).
            When None, the registry is loaded fresh without triggering the completeness check.

    Returns:
        • dict[int, float] — maps each model_idx to its mean policy regret in [0, ∞).
    """
    import math as _math

    tau = softmax_temperature if softmax_temperature is not None else float(
        general_settings.get("softmax_temperature", 1.5)
    )
    rng = random.Random(random_seed)

    "Load registry and derive flag columns — skip completeness check to prevent recursive calls."
    if registry_df is None:
        registry_df = all_utility_functions_dataframe(file_paths=file_paths)
    _non_flag: set = {
        "utility_idx", "utility_bitstring", "k_params", "redundant_with", 
        "differing_settings", "n_data", "pvar", "param_norm_sd", "loss_nll",
        "AIC", "BIC", "ΔAIC", "ΔBIC", "AIC_rank", "BIC_rank", "parents", 
        "siblings", "children", "ampd_to_best", "policy_regret_norm_to_best", 
        "canonical_model", "equation",
    }
    flag_cols = [col for col in registry_df.columns if col not in _non_flag]

    def _parse_csv_bool(val: Any) -> bool:
        if isinstance(val, str):
            return val.strip().lower() not in ("false", "0", "")
        return bool(val)

    registry_indexed = registry_df.set_index("utility_idx")
    all_needed_idxs = list({winner_idx} | set(model_idxs))
    settings_map: Dict[int, UtilitySettings] = {
        idx: {col: _parse_csv_bool(registry_indexed.loc[idx, col]) for col in flag_cols}
        for idx in all_needed_idxs
        if idx in registry_indexed.index
    }

    "Precompute per-model mean-parameter key lists (cached outside iteration loop)."
    all_mean_keys = [k for k in param_bds if not k.endswith("_std") and "_cov" not in k]
    midpoints = {k: 0.5 * (float(param_bds[k][0]) + float(param_bds[k][1])) for k in all_mean_keys}
    model_mean_keys: Dict[int, List[str]] = {}
    for idx in all_needed_idxs:
        if idx not in settings_map:
            continue
        pi = make_param_info(
            param_bds=param_bds, utility_settings=settings_map[idx],
            general_settings=general_settings, guess_seed=None,
        )
        model_mean_keys[idx] = [k for k in pi["keys"] if not k.endswith("_std") and "_cov" not in k]

    "Build the exhaustive 5^4 payoff grid (or a random sample when n_games < 625)."
    if n_games >= 625:
        payoff_tuples: List[Dict[str, int]] = [
            {"As": a, "Ao": b, "Bs": c, "Bo": d}
            for a, b, c, d in it.product(range(1, 6), repeat=4)
        ]
    else:
        payoff_tuples = [
            {"As": rng.randint(1, 5), "Ao": rng.randint(1, 5),
             "Bs": rng.randint(1, 5), "Bo": rng.randint(1, 5)}
            for _ in range(n_games)
        ]
    payoff_tuples_B: List[Dict[str, int]] = [
        {"As": p["Bs"], "Ao": p["Bo"], "Bs": p["As"], "Bo": p["Ao"]}
        for p in payoff_tuples
    ]

    regret_sum:   Dict[int, float] = {idx: 0.0 for idx in model_idxs}
    regret_count: Dict[int, int]   = {idx: 0   for idx in model_idxs}
    eps = 1e-12

    for _ in range(n_iters):
        full_ref = {k: rng.uniform(float(param_bds[k][0]), float(param_bds[k][1])) for k in all_mean_keys}
        params_map = {
            idx: {k: full_ref.get(k, midpoints[k]) for k in model_mean_keys.get(idx, [])}
            for idx in all_needed_idxs
        }
        winner_settings = settings_map.get(winner_idx)
        if winner_settings is None:
            continue
        params_winner = params_map[winner_idx]

        for payoffs_A, payoffs_B in zip(payoff_tuples, payoff_tuples_B):
            try:
                u_w_A = utility(payoffs=payoffs_A, params=params_winner, utility_settings=winner_settings)
                u_w_B = utility(payoffs=payoffs_B, params=params_winner, utility_settings=winner_settings)
                p_w_raw = softmax_(uA=u_w_A, uB=u_w_B, temperature=tau)
            except Exception:
                continue

            p_w = max(eps, min(1.0 - eps, p_w_raw))
            "Denominator = KL(p_winner ‖ 0.5) = log(2) − H(p_winner)"
            denom = p_w * _math.log(2.0 * p_w) + (1.0 - p_w) * _math.log(2.0 * (1.0 - p_w))
            if denom < eps:
                continue

            for idx in model_idxs:
                if idx not in settings_map or idx not in params_map:
                    continue
                try:
                    u_m_A = utility(payoffs=payoffs_A, params=params_map[idx], utility_settings=settings_map[idx])
                    u_m_B = utility(payoffs=payoffs_B, params=params_map[idx], utility_settings=settings_map[idx])
                    p_m = max(eps, min(1.0 - eps, softmax_(uA=u_m_A, uB=u_m_B, temperature=tau)))
                    kl = p_w * _math.log(p_w / p_m) + (1.0 - p_w) * _math.log((1.0 - p_w) / (1.0 - p_m))
                    regret_sum[idx]   += kl / denom
                    regret_count[idx] += 1
                except Exception:
                    pass

    return {
        idx: (regret_sum[idx] / regret_count[idx] if regret_count[idx] > 0 else float("nan"))
        for idx in model_idxs
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


def _ampd_pair_worker(args: tuple) -> list:
    """
    Module-level parallel worker for AMPD distance matrix computation.

    Receives a batch of model-index pairs plus all parameters needed by
    average_model_policy_distance, computes the AMPD for each pair, and returns
    the results so the master process can merge them into the matrix and save.

    Arguments:
        • args: tuple of (pair_batch, settings_cache, general_settings, file_paths,
              param_bds, metric, softmax_temperature, n_games, n_iters,
              parameter_sampling_mode, parameter_pairing_mode, player_roles,
              random_seed, participant_parameter_pools)

    Returns:
        • list of (utility_idx_i, utility_idx_j, distance) tuples — one per pair.
    """
    (pair_batch, settings_cache, general_settings, file_paths, param_bds,
     metric, softmax_temperature, n_games, n_iters, parameter_sampling_mode,
     parameter_pairing_mode, player_roles, random_seed, participant_parameter_pools) = args

    batch_results = []
    for utility_idx_i, utility_idx_j in pair_batch:
        dist = average_model_policy_distance(
            utility_settings_a=settings_cache[utility_idx_i],
            utility_settings_b=settings_cache[utility_idx_j],
            general_settings=general_settings,
            file_paths=file_paths,
            param_bds=param_bds,
            metric=metric,
            softmax_temperature=softmax_temperature,
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


def compute_ampd_matrix(
    general_settings: GeneralSettings,
    file_paths: FilePaths,
    param_bds: Optional[Dict[str, Tuple[float, float]]] = None,
    utility_settings: Optional[UtilitySettings] = None,
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
    Uses a single master 505×505 file per settings combination. NaN marks uncomputed cells;
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
            both models are in this list. The master matrix still covers all 505 models.
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
        • pd.DataFrame — full 505×505 master matrix indexed and columned by utility_idx.
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

    tau = float(_ampd_cfg.get("softmax_temperature", general_settings.get("softmax_temperature", 1.5)))

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

    "Load full registry (all models) to build the master index — no general_settings to prevent recursive completeness checks."
    registry_df = all_utility_functions_dataframe(file_paths=file_paths)
    all_utility_idxs: List[int] = sorted(registry_df["utility_idx"].astype(int).tolist())
    n_all = len(all_utility_idxs)

    master_path = _build_ampd_cache_path(
        file_paths=file_paths, metric=metric,
        parameter_sampling_mode=parameter_sampling_mode,
        parameter_pairing_mode=parameter_pairing_mode,
        player_roles=player_roles, softmax_temperature=tau, 
        n_games=n_games, n_iters=n_iters,
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
        "utility_idx", "utility_bitstring", "k_params", "redundant_with", 
        "differing_settings", "n_data", "pvar", "param_norm_sd", "loss_nll",
        "AIC", "BIC", "ΔAIC", "ΔBIC", "AIC_rank", "BIC_rank", "parents", 
        "siblings", "children", "ampd_to_best", "policy_regret_norm_to_best", 
        "canonical_model", "equation",
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

    if param_bds is None or utility_settings is None:
        raise ValueError(
            f"compute_ampd_matrix: {n_remaining} pairs still need to be computed but "
            "param_bds and utility_settings were not provided. Pass both to compute the matrix, "
            "or run compute_ampd_matrix() interactively first."
        )

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
        distance_matrix_df = compute_ampd_matrix(
            general_settings=general_settings, file_paths=file_paths, create_new_file=False,
        )

    out_path = os.path.join(
        file_paths["processed"],
        f"model_space_embedding__{distance_name}__dims={n_dimensions}.csv",
    )
    if not create_new_file and os.path.exists(out_path):
        print(f"Model-space embedding loaded from cache: {pretty_path(out_path)}")
        return pd.read_csv(out_path, dtype={"utility_bitstring": str})

    registry_df = all_utility_functions_dataframe(file_paths=file_paths, general_settings=general_settings)
    if require_ic_data:
        registry_df = registry_df[registry_df["BIC"].notna()].copy()
    if "BIC_rank" in registry_df.columns:
        registry_df = registry_df.sort_values("BIC_rank").reset_index(drop=True)

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
    Because compute_ampd_matrix fills both triangles simultaneously, any model with
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
                "need at least 3 for MDS. Run more of compute_ampd_matrix first."
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
    figure_layout: FigLay,
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
        • figure_layout: FigLay — layout constants from config.py.
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
            "siblings", "children", "ampd_to_best", "policy_regret_norm_to_best",
            "mds_x", "mds_y", "mds_z", "mds_w", "equation", "canonical_model",
        }
    ]

    marker_size = int(figure_layout.get("markersize", 16) * 2)
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
                    family=figure_layout.get("font", {}).get("family", "Calibri"),
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
        template=figure_layout.get("template", "plotly_white"),
        title=f"Model-Space MDS — {distance_label} Distances ({n_models} models)",
        titlefont_size=figure_layout["titlefont_size"],
        title_x=0.5,
        font=figure_layout.get("font", {}),
        hoverlabel=figure_layout.get("hoverlabel", {}),
        margin=dict(l=120, r=180, t=140, b=100),
        xaxis=dict(title="MDS Dimension 1", range=mds_axis_range,
                   scaleanchor="y", scaleratio=1, **figure_layout.get("xaxis", {})),
        yaxis=dict(title="MDS Dimension 2", range=mds_axis_range,
                   **figure_layout.get("yaxis", {})),
        annotations=annotations,
        legend=dict(orientation="h", x=0.0, y=-0.15,
                    font=dict(size=figure_layout.get("font", {}).get("size", 20))),
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
                     if "dark" in figure_layout.get("template", "") 
                     else _hsla(hue=0, saturation_percent=0, lightness_percent=94, alpha=0.92)),
            font=dict(size=20, family=figure_layout.get("font", {}).get("family", "Calibri")),
        )])

    out_path = os.path.join(file_paths["visuals"], f"mds_{distance_name}.html")
    fig.write_html(out_path)
    print(f"Model-space MDS saved: {out_path}")
    return fig


def plot_distance_to_winner_vs_delta_bic(
    general_settings: GeneralSettings,
    file_paths: FilePaths,
    figure_layout: FigLay,
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
        • figure_layout: FigLay — layout constants from config.py.
        • distance_matrix_df: pd.DataFrame | None (default None) — square AMPD matrix
            indexed by utility_idx. If None, loaded from general_settings['ampd_settings'].
        • require_ic_data: bool (default True) — if True, exclude models without BIC.

    Returns:
        • go.Figure — also written to visuals/dist_to_winner_vs_dbic.html.
    """
    if distance_matrix_df is None:
        distance_matrix_df = compute_ampd_matrix(
            general_settings=general_settings, file_paths=file_paths, create_new_file=False,
        )

    registry_df = all_utility_functions_dataframe(file_paths=file_paths, general_settings=general_settings)
    if require_ic_data:
        registry_df = registry_df[registry_df["BIC"].notna()].copy()
    if "BIC_rank" in registry_df.columns:
        registry_df = registry_df.sort_values("BIC_rank").reset_index(drop=True)
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

    marker_size = int(figure_layout.get("markersize", 16) * 2)
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
        font=dict(size=20, family=figure_layout.get("font", {}).get("family", "Calibri")),
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
        template=figure_layout.get("template", "plotly_white"),
        title=f"Distance to BIC Winner vs ΔBIC — {corr_label} ({n_valid_pairs} of {len(sub_df)} models computed)",
        titlefont_size=figure_layout["titlefont_size"],
        title_x=0.5,
        font=figure_layout.get("font", {}),
        hoverlabel=figure_layout.get("hoverlabel", {}),
        margin=dict(l=120, r=180, t=140, b=100),
        xaxis=dict(title="AMPD Distance to BIC-Winning Model", **figure_layout.get("xaxis", {})),
        yaxis=dict(title="ΔBIC (vs Best Model)", **figure_layout.get("yaxis", {})),
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
        distance_matrix_df = compute_ampd_matrix(
            general_settings=general_settings, file_paths=file_paths, create_new_file=False,
        )
    if top_ns is None:
        top_ns = [5, 10, 25, 50]

    registry_df = all_utility_functions_dataframe(file_paths=file_paths, general_settings=general_settings)
    if require_ic_data:
        registry_df = registry_df[registry_df["BIC"].notna()].copy()
    if "BIC_rank" in registry_df.columns:
        registry_df = registry_df.sort_values("BIC_rank").reset_index(drop=True)
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
    figure_layout: FigLay,
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
        • figure_layout: FigLay — layout constants from config.py.
        • distance_matrix_df: pd.DataFrame | None (default None) — pairwise distance matrix
            indexed by utility_idx. If None, loaded from general_settings['ampd_settings'].
        • top_n: int (default 50) — number of top-BIC models to include.
        • require_ic_data: bool (default True) — restrict to models with BIC data.

    Returns:
        • go.Figure — also written to visuals/top_model_heatmap_{top_n}.html.
    """
    if distance_matrix_df is None:
        distance_matrix_df = compute_ampd_matrix(
            general_settings=general_settings, file_paths=file_paths, create_new_file=False,
        )

    registry_df = all_utility_functions_dataframe(file_paths=file_paths, general_settings=general_settings)
    if require_ic_data:
        registry_df = registry_df[registry_df["BIC"].notna()].copy()
    if "BIC_rank" in registry_df.columns:
        registry_df = registry_df.sort_values("BIC_rank").reset_index(drop=True)
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
            "need at least 2 for a heatmap. Run more of compute_ampd_matrix first."
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
        template=figure_layout.get("template", "plotly_white"),
        title=f"Pairwise AMPD — Top {actual_n} BIC-Ranked Models",
        titlefont_size=figure_layout["titlefont_size"],
        title_x=0.5,
        font=figure_layout.get("font", {}),
        hoverlabel=figure_layout.get("hoverlabel", {}),
        margin=dict(l=140, r=160, t=140, b=140),
        xaxis=dict(title="Model (BIC Rank)", tickangle=-45, **figure_layout.get("xaxis", {})),
        yaxis=dict(title="Model (BIC Rank)", autorange="reversed",
                   scaleanchor="x", scaleratio=1, **figure_layout.get("yaxis", {})),
    )

    out_path = os.path.join(file_paths["visuals"], f"top_model_heatmap_{actual_n}.html")
    fig.write_html(out_path)
    print(f"Top-model AMPD heatmap saved: {out_path}")
    return fig

