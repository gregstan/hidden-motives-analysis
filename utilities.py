from config import *


def correlation_xy(df: pd.DataFrame, col_name_x: str, col_name_y: str) -> tuple[float, float]:
    """
    Computes the Pearson correlation coefficient and p-value between
    two columns in the DataFrame.

    • df: pd.DataFrame
        - The DataFrame containing the columns to be correlated.
    • col_name_x: str
        - Name of the column to use as the x-values.
    • col_name_y: str
        - Name of the column to use as the y-values.

    Returns:
        • (r_value, p_value) in (float, float).

    Example:
        r_corr, p_val = correlation_xy(df, 'c_Vᵢᵢ', 'p_Vᵢᵢ')
    """
    import scipy as sp # type: ignore
    x_vals: NDArray[np.float64] = np.asarray(df[col_name_x].values, dtype=np.float64)
    y_vals: NDArray[np.float64] = np.asarray(df[col_name_y].values, dtype=np.float64)

    mask: NDArray[np.bool_] = ~np.isnan(x_vals) & ~np.isnan(y_vals)
    if int(mask.sum()) < 2: # type: ignore
        return float('nan'), float('nan')

    r_value, p_value = sp.stats.pearsonr(x_vals[mask], y_vals[mask])
    return r_value, p_value


def fill_holes_nd(input_array: NDArray[np.float64], method: str = "cubic", output_shape: tuple[int] | Any | None = None, fill_value: float = np.nan) -> NDArray[np.float64]:
    """
    Fill missing values (None/NaN) in an nD array via scattered-data interpolation.
    
    Arguments:
        • input_array: NDArray[np.float64]
            An n-dimensional array of shape (d1, d2, ..., dn), containing 
            numeric values and possibly missing values (None or NaN).

        • method: str ("linear", "nearest", "cubic")
            Interpolation method for `scipy.interpolate.griddata`. 

        • output_shape: tuple, optional
            The dimensions of the output array.

        • fill_value: float
            Value passed to `griddata` for extrapolation (points 
            outside the convex hull of known data). Default is np.nan.

    Returns:
        • filled_array: NDArray[np.float64]
            The array of shape=out_shape (or input_array.shape if out_shape=None),
            with missing values replaced by interpolated values.

    Notes:
        • Large nD arrays or large fractions of missing data can make this approach slow, 
          since scattered-data interpolation can span up to (d1*d2*...*dn) points. 
    """
    from scipy.interpolate import griddata, RBFInterpolator # type: ignore
    "1) Convert all None -> np.nan, ensure float"
    arr = np.array(input_array, dtype=float)
    shape_in = arr.shape
    ndim = arr.ndim

    "2) Identify coordinates of valid data (i.e. not NaN)"
    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        "If everything is missing, return zeros or NaNs or whatever"
        print("Warning: Input array contains no valid values. Returning zeros.")
        if output_shape is None:
            return np.zeros_like(arr)
        else:
            return np.zeros(output_shape, dtype=float)

    "Gather 'sample_points' in R^n plus 'sample_values'"
    valid_indices = np.argwhere(valid_mask)  # shape=(N, ndim), N=number of valid points
    sample_values = arr[valid_mask]          # shape=(N,)

    "3) Builds the output grid. If out_shape=None, reuses shape_in. Otherwise uses out_shape."
    if output_shape is None:
        output_shape = shape_in
    
    "Create a regular grid of integer coords in [0..(output_shape[k]-1)] for each dimension k"
    scale_factors: list[Any] = []
    for dim_i in range(ndim):
        if shape_in[dim_i] == 1:
            "Edge case: dimension of size 1 => no scaling"
            scale_factors.append(0.0)  # effectively means everything is index=0
        else:
            scale_factors.append((shape_in[dim_i] - 1.0) / max(1, (output_shape[dim_i] - 1)))

    "Now Create an nD meshgrid of shape=out_shape, with coords in the 'input space.'"
    coords_list: list[Any] = []
    for dim_i in range(ndim):
        if output_shape[dim_i] > 1:
            "e.g. np.linspace(0, shape_in[dim_i]-1, out_shape[dim_i])"
            coords_list.append(
                np.linspace(0, shape_in[dim_i]-1, output_shape[dim_i])
            )
        else:
            "Dimension of size 1 => just [0]"
            coords_list.append(np.array([0.0]))

    "meshgrid => list of length=ndim, each array shape=out_shape"
    mg = np.meshgrid(*coords_list, indexing='ij')  

    "Flattens coordinates for griddata. Shape=(out_shape product, ndim)."
    target_coords = np.stack([m.flatten() for m in mg], axis=-1)

    "4) Interpolate using the chosen method."
    filled_vals: NDArray[np.float64]
    if ndim > 2 and method == 'cubic':
        "For nD cubic-like interpolation, use RBFInterpolator."
        rbf = RBFInterpolator(valid_indices, sample_values, kernel='cubic')
        filled_vals: NDArray[np.float64] = rbf(target_coords)
    else:
        filled_vals = np.asarray(griddata(
            points=valid_indices,
            values=sample_values,
            xi=target_coords,
            method=method,
            fill_value=fill_value
        ), dtype=np.float64)  # <== Explicitly convert to array

    "5) Check for any NaNs left over from interpolation."
    if np.any(np.isnan(filled_vals)):
        "Use nearest neighbor interpolation for points still missing."
        nearest_vals: NDArray[np.float64] = np.asarray(griddata(
            points=valid_indices,
            values=sample_values,
            xi=target_coords,
            method='nearest'
        ), dtype=np.float64)
        "Replace NaNs from the primary interpolation with the nearest neighbor values."
        filled_vals = np.where(np.isnan(filled_vals), nearest_vals, filled_vals)

    "6) Reshape to the output grid shape."
    filled_array = filled_vals.reshape(output_shape)

    return filled_array


def convert_to_nd_arrays(prior_grid: Any, players_and_roles: Dict[str, List[str]] | None = None) -> dict[str, Any]:
    """
    Converts sparse parameter vectors from prior_grid into full n-dimensional arrays (PMFs).

    Arguments:
        • prior_grid: dict; The prior grid containing sparse parameter vectors.
        • players_and_roles: Dict[str, List[str]]; Limits which players and roles to iterate over.
            - Example (only one player in one role): {player_uuid: ['chooser']}

    Returns:
        • prior_grid: dict; Updated prior grid with n-dimensional arrays (PMFs) added.
    """
    meta_data = prior_grid["meta_data"]
    n_bins_per_dimension = meta_data["n_bins_per_dimension"]
    tickvals = meta_data["tickvals"]

    for player_uuid, player_roles in prior_grid["players"].items():
        if players_and_roles is not None and player_uuid not in players_and_roles:
            continue
        for player_role, role_data in player_roles.items():
            if players_and_roles is not None and player_role not \
                in players_and_roles.get(player_uuid, {}):
                continue
            param_vectors = role_data["param_vectors"]

            "Determine the dimensions and create an empty grid"
            grid_shape = tuple(n_bins_per_dimension for _ in tickvals.keys())
            full_grid = np.full(fill_value=np.nan, shape=grid_shape)

            "Fill the grid with probabilities from param_vectors"
            for param_vector, probability in param_vectors.items():
                indices = tuple(param_vector)
                full_grid[indices] = probability

            "Skip if grid has no valid data"
            if np.isnan(full_grid).all():
                print(f"Warning: All values are NaN for {player_uuid}, {player_role}.")
                continue

            "Fill in missing values and normalize"
            full_grid = fill_holes_nd(input_array=full_grid, output_shape=None)
            full_grid /= full_grid.sum() # type: ignore

            "Store the full grid (PMF) back in prior_grid"
            prior_grid["players"][player_uuid][player_role]["PMF"] = full_grid

    return prior_grid


def transform_to_simplex(alpha: np.ndarray) -> np.ndarray:
    """
    Converts an unconstrained real vector alpha into a probability simplex.
    Uses softmax-like exponentiation so all returned values are in [0,1] and sum to 1.

    Arguments:
        • alpha: np.ndarray
            Vector of unconstrained real values.

    Returns:
        • np.ndarray
            The normalized probabilities that sum to 1.
    """
    exp_alpha = np.exp(alpha)
    return exp_alpha / np.sum(exp_alpha)


def numerical_hessian(funct: Callable[[NDArray[np.float64], Any], NDArray[np.float64]], x0: NDArray[np.float64], epsilon: float = 1e-5, *args: Any) -> NDArray[np.float64]:
    """
    Compute approximate Hessian matrix of function funct at x0 using finite differences.
    
    Arguments:
        • funct: object; function
        • x0: NDArray[np.float64]; point at which to compute Hessian
        • epsilon: float; step size for finite difference
        • args: additional arguments passed to funct

    Returns:
        • Hessian: NDArray[np.float64] of shape (len(x0), len(x0))
    """
    num = len(x0)
    hessian = np.zeros((num, num))

    "f_x is the gradient approximations"
    for idx in range(num):
        for jdx in range(num):
            x_ij1 = x0.copy()
            x_ij2 = x0.copy()
            x_ij3 = x0.copy()
            x_ij4 = x0.copy()
            x_ij1[idx] += epsilon
            x_ij1[jdx] += epsilon
            x_ij2[idx] += epsilon
            x_ij2[jdx] -= epsilon
            x_ij3[idx] -= epsilon
            x_ij3[jdx] += epsilon
            x_ij4[idx] -= epsilon
            x_ij4[jdx] -= epsilon

            funct1 = funct(x_ij1, *args)
            funct2 = funct(x_ij2, *args)
            funct3 = funct(x_ij3, *args)
            funct4 = funct(x_ij4, *args)
            hessian[idx, jdx] = (
                funct1 - funct2 - funct3 + funct4
                ) / (4 * epsilon * epsilon)

    return hessian


def should_penalize_parameter_key(parameter_key: str) -> bool:
    """
    Return True iff this parameter should be included in the parameter penalty.
    Excludes temperature and covariance entries; keeps means and stds.
    """
    if parameter_key in ("τ", "temp"):
        return False
    if parameter_key.endswith("_cov"):
        return False
    "Means and stds are penalized (stds with lighter rule); exponents and weights included."
    return True


def parameter_penalty(
    params: dict[str, int | float],
    penalty_weight: float, *,
    penalize_exponents: bool = True,
    penalize_std: bool = False,
    penalize_cov: bool = False,
    penalize_temp: bool = False
) -> float:
    """
    Scale-invariance guard and *parent-fair* regularizer.

    Why this exists
    ----------------
    Many utility variants are (nearly) scale-invariant in the weight means:
    multiplying all active weights by c often leaves the induced choice probabilities
    essentially unchanged. Without a regularizer, optimizers can "wander" to large
    magnitudes and stall. This function applies a small, smooth penalty to keep the
    search in a well-behaved region.

    Why it's fair to parents
    ------------------------
    The penalty is constructed so that **a parent that collapses to its child** by
    setting the extra weights to their child-equivalent values incurs the *same*
    penalty as the child. Concretely:

        • For positivity/negativity pairs on the *same* term (self, altruism, social):
            penalizes the **mean absolute magnitude** of the pair, not each coordinate
            separately. That way, a parent with (V, Λ) = (v, v) or (E, G) = (e, e) pays
            exactly what the child pays with the single weight = v or e.

        • If a negativity mate is absent (only Vᵢᵢ, Vᵢⱼ, or Ƹᵢⱼ exists), falls back to
            the usual L2: penalty += (weight)^2.

        • Exponents (if penalized) are biased toward 1.0 via (γ - 1)^2. To avoid bias
            when moving from one gamma to many, penalizes the **mean** gamma deviation:
                mean_gamma = avg({all gammas present})
                penalty += (#gammas) * (mean_gamma - 1)^2
            This preserves fairness between a child with one γ and a parent with several.

    What this DOES and DOESN’T penalize by default:
        • Mean weights: yes (always; this is the primary stabilizer).
        • Exponents: on by default (set `penalize_exponents=False` to disable).
        • *_std, *_cov, temp: off by default for IC fairness; you can enable later
            when you fit a single winning model.

    Notes:
        • This function intentionally does **not** normalize by the number of parameters.
            Normalizing can subtly reintroduce anti-parent bias by shrinking children more.
        • No special `*args` trickery here — the `*` in the signature only forces keyword
            usage for the boolean switches (helps avoid accidental positional bugs).

    Returns:
        • A scalar penalty = penalty_weight × (sum of term penalties).
    """
    if penalty_weight <= 0:
        return 0.0

    "Helper: safely extract a numeric parameter value from the params dict."
    def parameter_value(param_dict: dict, param_key: str) -> float:
        param_val = param_dict.get(param_key, 0.0)
        return float(param_val) if isinstance(param_val, (int, float)) else 0.0

    def _has(param_dict: dict, *param_keys: str) -> bool:
        return any(param_key in param_dict for param_key in param_keys)

    penalty = 0.0

    "1) Means (self, altruism, social): pairwise mean-abs magnitude with appropriate anchors."

    "Self-interest (anchor = 1.0)."
    Vii = parameter_value(params, "Vᵢᵢ") or parameter_value(params, "Vii")
    Lii = parameter_value(params, "Ʌᵢᵢ") or parameter_value(params, "Λii")
    has_Vii = _has(params, "Vᵢᵢ", "Vii")
    has_Lii = _has(params, "Ʌᵢᵢ", "Λii")

    if has_Vii or has_Lii:
        anchor = 1.0
        if has_Vii and has_Lii:
            mag = (abs(Vii - anchor) + abs(Lii - anchor)) / 2.0
            penalty += mag * mag
        elif has_Vii:
            penalty += (Vii - anchor) ** 2
        else:  # has_Lii only
            penalty += (Lii - anchor) ** 2

    "Altruism (anchor = 0.0)."
    Vij = parameter_value(params, "Vᵢⱼ") or parameter_value(params, "Vij")
    Lij = parameter_value(params, "Ʌᵢⱼ") or parameter_value(params, "Λij")
    has_Vij = _has(params, "Vᵢⱼ", "Vij")
    has_Lij = _has(params, "Ʌᵢⱼ", "Λij")

    if has_Vij and has_Lij:
        mag = (abs(Vij) + abs(Lij)) / 2.0
        penalty += mag * mag
    else:
        if has_Vij:
            penalty += Vij * Vij
        if has_Lij:
            penalty += Lij * Lij

    "Social comparison (anchor = 0.0)."
    Eij = parameter_value(params, "Ƹᵢⱼ") or parameter_value(params, "Eij")
    Gij = parameter_value(params, "Ʒᵢⱼ") or parameter_value(params, "Gij")
    has_E = _has(params, "Ƹᵢⱼ", "Eij")
    has_G = _has(params, "Ʒᵢⱼ", "Gij")

    if has_E and has_G:
        mag = (abs(Eij) + abs(Gij)) / 2.0
        penalty += mag * mag
    else:
        if has_E:
            penalty += Eij * Eij
        if has_G:
            penalty += Gij * Gij

    "2) Exponents (optional): penalize deviation of the mean gamma from 1."
    if penalize_exponents:
        gammas = []
        for param_key in params:
            if param_key.startswith("γ") or param_key.lower().startswith("gamma"):
                val = parameter_value(params, param_key)
                "Keep zeros too — avoids problems with non-numeric values."
                gammas.append(val)
        if gammas:
            mean_gamma = sum(gammas) / len(gammas)
            penalty += len(gammas) * (mean_gamma - 1.0) ** 2  # parent-fair

    "3) Standard deviations (optional)."
    if penalize_std:
        for param_key, param_val in params.items():
            if isinstance(param_val, (int, float)) and isinstance(param_key, str) and param_key.endswith("_std"):
                param_val = float(param_val)
                floor = 1e-3
                if param_val < floor:
                    penalty += 1e6 * (floor - param_val)  # hard push away from zero
                else:
                    penalty += (param_val * param_val) / 10.0

    "4) Covariances (optional)."
    if penalize_cov:
        for param_key, param_val in params.items():
            if isinstance(param_val, (int, float)) and isinstance(param_key, str) and param_key.endswith("_cov"):
                penalty += float(param_val) ** 2

    "5) Temperature (optional)."
    if penalize_temp:
        temperature = parameter_value(params, "τ") or parameter_value(params, "temp")
        if temperature != 0.0:
            penalty += (temperature - 1.5) ** 4

    return penalty_weight * penalty


def compute_statistics(joint_pmf: NDArray[np.float64], 
                       grids: List[NDArray[np.float64]], 
                       param_info: Dict[str, Any],
                       print_warnings: bool = False) -> Dict[str, float]:
    """
    Compute the marginal means, standard deviations, and pairwise covariances for each variable in a joint PMF.

    Parameters:
        • joint_pmf: NDArray[np.float64]
            The n-dimensional joint probability mass function (PMF) defined over the grid.
        • grids: list of NDArray[np.float64]
            A list of 1D arrays containing the tick values for each dimension.
            Each array corresponds to one parameter in the same order as specified in param_info.
        • param_info: Dict[str, Any]
            Dictionary containing parameter information. It should include:
                - "keys": list of parameter names (e.g., ['Vᵢᵢ', 'Vᵢⱼ', 'Ƹᵢⱼ', 'Vᵢᵢ_std', 'Vᵢⱼ_std', 'Ƹᵢⱼ_std']).
                  Only keys not ending with '_std' will be used to label the dimensions for computing means and covariances.
        • print_warnings: bool
            If True, warnings will be printed when encountering invalid PMF or variance computations.

    Returns:
        • stats: dict
            A dictionary mapping parameter names to their computed statistics. For example:
                {
                    'Vᵢᵢ': <mean of Vᵢᵢ>,
                    'Vᵢⱼ': <mean of Vᵢⱼ>,
                    'Ƹᵢⱼ': <mean of Ƹᵢⱼ>,
                    'Vᵢᵢ_std': <standard deviation of Vᵢᵢ>,
                    'Vᵢⱼ_std': <standard deviation of Vᵢⱼ>,
                    'Ƹᵢⱼ_std': <standard deviation of Ƹᵢⱼ>,
                    'Vᵢᵢ_Vᵢⱼ_cov': <covariance between Vᵢᵢ and Vᵢⱼ>,
                    'Vᵢᵢ_Ƹᵢⱼ_cov': <covariance between Vᵢᵢ and Ƹᵢⱼ>,
                    'Vᵢⱼ_Ƹᵢⱼ_cov': <covariance between Vᵢⱼ and Ƹᵢⱼ>
                }
    """
    "Extract the names for the parameters (exclude any keys ending with '_std')"
    parameter_mean_names = [param for param in param_info["keys"] if not param.endswith('_std')]
    num_parameters: int = len(parameter_mean_names)

    "Ensure that the joint PMF is normalized; otherwise, normalize it."
    total_probability_mass: float = float(np.sum(joint_pmf)) #type: ignore   Type of "sum" is partially unknown
    if total_probability_mass <= 0 or np.isnan(total_probability_mass):
        if print_warnings:
            print(f"Warning: Invalid sum {total_probability_mass}. Normalizing.")
        total_probability_mass = 1.0

    normalized_joint_pmf = joint_pmf / total_probability_mass
    "Each coordinate array has the same shape as the joint PMF."
    meshgrid_coordinate_arrays: List[NDArray[np.float64]] = np.meshgrid(*grids, indexing='ij')

    "Initialize lists to store computed means and standard deviations for each parameter."
    computed_means: List[float] = [0.0] * num_parameters
    computed_standard_deviations: List[float] = [0.0] * num_parameters

    "Compute marginal means and variances for each parameter dimension."
    for parameter_index in range(num_parameters):
        coordinate_values: NDArray[np.float64] = meshgrid_coordinate_arrays[parameter_index]
        
        parameter_mean: float = float(np.sum(coordinate_values * normalized_joint_pmf)) #type: ignore
        computed_means[parameter_index] = parameter_mean  

        parameter_variance: float = float(np.sum(((coordinate_values - parameter_mean) ** 2) * normalized_joint_pmf)) #type: ignore
        
        if parameter_variance < 0 or np.isnan(parameter_variance):
            if print_warnings:
                print(f"Warning: Invalid variance for index {parameter_index}.")
            grid_values = grids[parameter_index]
            parameter_variance = ((float(np.max(grid_values)) - float(np.min(grid_values))) ** 2) / 12.0 #type: ignore

        computed_standard_deviations[parameter_index] = np.sqrt(parameter_variance)

    "Compute pairwise covariances between parameters."
    computed_covariances: Dict[str, float] = {}
    for idx in range(num_parameters):
        for jdx in range(idx + 1, num_parameters):
            deviation_product = (meshgrid_coordinate_arrays[idx] - computed_means[idx]) * (meshgrid_coordinate_arrays[jdx] - computed_means[jdx])
            covariance_value: float = float(np.sum(deviation_product * normalized_joint_pmf)) #type: ignore
            covariance_key = f"{parameter_mean_names[idx]}_{parameter_mean_names[jdx]}_cov"
            computed_covariances[covariance_key] = covariance_value

    "Build the final statistics dictionary."
    computed_statistics: Dict[str, float] = {}
    "Add the means."
    computed_statistics.update({param_name: computed_means[idx] for idx, param_name in enumerate(parameter_mean_names)})
    "Add the standard deviations with key format '<parameter>_std'"
    computed_statistics.update({f"{param_name}_std": computed_standard_deviations[idx] for idx, param_name in enumerate(parameter_mean_names)})
    "Add the pairwise covariances."
    computed_statistics.update(computed_covariances)

    return computed_statistics


def _statistics_from_sparse_param_vectors(
    param_vectors: dict[tuple[int, ...], float],
    meta_data: dict[str, Any],
    param_info: dict[str, Any]
) -> dict[str, float]:
    """
    Compute per-parameter means and standard deviations 
    directly from a sparse mass map (no densification).

    Arguments:
        • param_vectors: dict[(i1,...,id)->mass]; sparse posterior/prior on bins
        • meta_data: dict with 'tickvals' and 'n_bins_per_dimension'
        • param_info: dict with 'keys' (means-only keys are used here)

    Returns:
        • dict[str, float]; e.g., {'Vᵢᵢ': μ, 'Vᵢᵢ_std': σ, 'Vᵢⱼ': μ, 'Vᵢⱼ_std': σ, ...}
    """
    tickvals: dict[str, list[float]] = meta_data['tickvals']
    mean_param_keys: list[str] = [
        param_key for param_key in param_info['keys']
        if (not param_key.endswith('_std')) and (not param_key.endswith('_cov'))
    ]

    total_mass: float          = 0.0
    weighted_value_sum:   dict[str, float] = {param_key: 0.0 for param_key in mean_param_keys}
    weighted_squared_sum: dict[str, float] = {param_key: 0.0 for param_key in mean_param_keys}

    for index_tuple, mass in param_vectors.items():
        if mass is None or mass <= 0.0:
            continue
        total_mass += float(mass)
        for dimension_index, param_key in enumerate(mean_param_keys):
            tick_value = float(tickvals[param_key][index_tuple[dimension_index]])
            weighted_value_sum[param_key]   += mass * tick_value
            weighted_squared_sum[param_key] += mass * (tick_value * tick_value)

    if total_mass <= 0.0:
        "Degenerate; return zeros to be safe (should not happen if normalized)"
        zero_result = {param_key: 0.0 for param_key in mean_param_keys}
        zero_result.update({f"{param_key}_std": 0.0 for param_key in mean_param_keys})
        return zero_result

    result: dict[str, float] = {}
    for param_key in mean_param_keys:
        param_mean             = weighted_value_sum[param_key] / total_mass
        mean_of_squared_values = weighted_squared_sum[param_key] / total_mass
        variance               = max(0.0, mean_of_squared_values - param_mean * param_mean)
        result[param_key]            = float(param_mean)
        result[f"{param_key}_std"]   = float(variance ** 0.5)

    return result


def is_positive_semidefinite(matrix: NDArray[np.float64], tol: float = 1e-12) -> bool:
    """
    Returns True if matrix is positive semidefinite (all eigenvalues ≥ -tol).

    Tries a Cholesky decomposition first (fast path for strictly PD matrices); falls back
    to a full eigendecomposition if that fails, which handles the semidefinite boundary.

    Arguments:
        • matrix: NDArray[np.float64] — square symmetric matrix to test.
        • tol: float — tolerance for treating near-zero negative eigenvalues as zero
            (default 1e-12). Accounts for floating-point rounding near the PSD boundary.

    Returns:
        • bool — True if all eigenvalues of matrix are ≥ -tol, False otherwise.
    """
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        eigenvalues: NDArray[np.float64] = np.linalg.eigvalsh(matrix)
        return bool(np.all(eigenvalues >= -tol))


def nearest_psd_matrix(matrix: NDArray[np.float64], min_eigval: float = 0.0) -> NDArray[np.float64]:
    """
    Converts M into a PSD matrix by:
        1) Symmetrizing
        2) Eigen-decomposing
        3) Clamping negative eigenvalues to 'min_eigval' (usually 0 or small)
        4) Reconstructing a PSD matrix

    Returns the PSD-fixed matrix. This map is generally *not* invertible one-to-one.
    """
    "1) Symmetrize"
    M_sym = 0.5 * (matrix + matrix.T)

    "2) Eigen-decompose"
    eigvals, eigvecs = np.linalg.eigh(M_sym)

    "3) Clamp eigenvalues"
    "If min_eigval=0, clamps negatives to 0 => PSD."
    "If min_eigval>0, forces *strict* positive definiteness."
    eigvals_clamped = np.maximum(eigvals, min_eigval)

    "4) Reconstruct"
    M_psd = (eigvecs * eigvals_clamped) @ eigvecs.T
    return M_psd


def validate_covariance_matrix(
    cov_matrix: NDArray[np.float64],
    name: str = "Name of Covariance Matrix",
    fix_invalid_matrices: bool = False,
    min_eigval: float = 1e-10
) -> NDArray[np.float64] | None:
    """
    Checks (and optionally fixes) a covariance matrix to ensure symmetry and
    positive semidefiniteness (PSD).

    Arguments:
        • cov_matrix (NDArray[np.float64]): The covariance matrix to validate.
        • name (str): A name for debugging logs.
        • fix_invalid_matrices (bool): If True, tries to fix negative eigenvalues.
        • min_eigval (float): The minimum eigenvalue cutoff when fixing.

    Returns:
        • NDArray[np.float64] | None:
            - If fix_invalid_matrices=True, always returns a PSD matrix
              (possibly corrected), or None if it's beyond salvage.
            - If fix_invalid_matrices=False, returns the matrix itself if valid,
              else None.
    """
    "1) Check symmetry"
    if not np.allclose(cov_matrix, cov_matrix.T, atol=1e-8):
        "Optionally fix symmetry"
        print(f"[validate_cov] '{name}' not symmetric. Making symmetric via (M + M^T)/2.")
        cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)

    "2) Check eigenvalues"
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  # For symmetric mat
    if np.any(eigenvalues < 0):
        "If there's a negative eigenvalue"
        if not fix_invalid_matrices:
            "Return None => \"invalid\" if not allowed to fix"
            return None
        else:
            print(f"[validate_cov] '{name}' not PSD. Fixing negative eigenvalues.")
            print("  Original eigenvalues:", eigenvalues)
            "Clip them to min_eigval"
            eigenvalues = np.maximum(eigenvalues, min_eigval)
            cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            "If you want to re-check, you can do so here:"
            "Re-checks PSD after repair."
            re_eigs, _ = np.linalg.eigh(cov_matrix)
            if np.any(re_eigs < 0):
                print("  Could not fix it entirely. Returning None.")
                return None

    "3) Return final PSD matrix"
    return cov_matrix


def build_covariation_matrix(param_info: Dict[str, Any], params: Dict[str, float] = {}, 
                             default_to_guesses: bool = False, raise_on_invalid: bool = True) -> NDArray[np.float64]:
    """
    Generates a covariation matrix from param_info.

    Arguments:
        • param_info: Dict[str, Any];
            - Example: { 
                'keys': ['Vᵢᵢ', 'Vᵢⱼ', 'Ƹᵢⱼ', 'Vᵢᵢ_std', 'Vᵢⱼ_std', 'Ƹᵢⱼ_std']
                'bounds': [(-1, 1), (-1, 1), (-1, 1), (1e-12, 4), (1e-12, 4), (1e-12, 4)],
                'guesses': [0.5, 0.6, 0.7, 0.01, 0.02, 0.03], 
                'covar': { 
                    'keys': ['Vᵢᵢ_Vᵢⱼ_cov', 'Vᵢᵢ_Ƹᵢⱼ_cov', 'Vᵢⱼ_Ƹᵢⱼ_cov'],
                    'bounds': [(-16, 16), (-16, 16), (-16, 16)],
                    'guesses': [0, 0, 0],
                }
            }
        • params: Dict[str, float];
            - Example: {
                'Vᵢᵢ': 0.45, 'Vᵢⱼ': -0.02, 'Ƹᵢⱼ': -0.11, 
                'Vᵢᵢ_std': 1.34, 'Vᵢⱼ_std': 0.65, 'Ƹᵢⱼ_std': 0.42
            } 

    Returns:
        • covariance_matrix: 2D Numpy array of floats. 

    Notes:
        • If params is {}, this will construct the matrix using param_info['guesses'] 
            assuming that this function is being called on the first game.    
    """
    if not default_to_guesses:
        for param_key in param_info['keys']:
            if param_key not in params:
                raise ValueError(f"Key: '{param_key}' missing from params: {params}")

        if param_info.get('covar', None) is None:
            raise ValueError(f"param_info missing 'covar' key: {param_info}")

        for cov_key in param_info['covar']['keys']:
            if cov_key not in params:
                raise ValueError(f"Key: '{cov_key}' missing from params: {params}")

    "Extract the relevant keys for the means (do not include '_std')"
    active_keys = [key for key in param_info["keys"] if not key.endswith('_std')]

    "Initialize covariance matrix"
    n_params = len(active_keys)  # Number of social preferences
    covariance_matrix = np.zeros((n_params, n_params))

    "Fill diagonal with variances: use each corresponding std (squared)"
    guesses: Union[List[float], Callable[[], List[float]]] = param_info['guesses']
    for idx, key in enumerate(active_keys):
        std_key = key + '_std'
        if default_to_guesses:
            guess_index: int = param_info['keys'].index(std_key)
            default_guess: float = guesses()[guess_index] if callable(guesses) else guesses[guess_index]
            std_val: float = params.get(std_key, default_guess)
        else:
            std_val = params[std_key]

        covariance_matrix[idx, idx] = std_val ** 2

    "Fill off-diagonals using the free covariance parameters"
    for idx, key1 in enumerate(active_keys):
        for jdx, key2 in enumerate(active_keys[idx + 1:], start=idx + 1):
            cov_key = f"{key1}_{key2}_cov"
            if default_to_guesses:
                cov_val = params.get(
                    cov_key,
                    param_info['covar']['guesses'][param_info['covar']['keys'].index(cov_key)]
                )
            else:
                cov_val = params[cov_key]

            covariance_matrix[idx, jdx] = cov_val
            covariance_matrix[jdx, idx] = cov_val

    "Regularize by adding jitter if necessary"
    jitter = 1e-8
    max_iter = 10
    iter_count = 0
    if raise_on_invalid:
        while not is_positive_semidefinite(covariance_matrix) and iter_count < max_iter:
            covariance_matrix += np.eye(n_params) * jitter
            jitter *= 10
            iter_count += 1
            print(f" ------------------------------------------Jitter--------------------------Jitter----------------------------------")

    if raise_on_invalid and not is_positive_semidefinite(covariance_matrix): 
        raise ValueError(f"Generated covariance matrix is not positive semidefinite: {covariance_matrix}")

    return covariance_matrix


def serialize_opt_result(opt_result: OptimizeResult,
                         duration: float | None = None,
                         loss: float | None = None) -> Dict[str, Any]:
    """
    Convert a scipy.optimize result object into a JSON-serializable dictionary.
    
    Returns:
        • report_dict: dict; A dictionary containing key information from the optimizer report.
        
    Notes:
        • Arrays are converted to lists and any non-serializable objects (like hess_inv) are converted to strings.
    """
    def _json_sanitize(obj):
        "Fast path for primitives"
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        "Numpy scalar types"
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)

        "Numpy arrays"
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        "Mappings"
        if isinstance(obj, dict):
            return {str(key): _json_sanitize(val) for key, val in obj.items()}

        "Sequences"
        if isinstance(obj, (list, tuple)):
            return [_json_sanitize(val) for val in obj]

        "Special SciPy objects (e.g., L-BFGS-B hess_inv OperatorInverseHessian)"
        "or anything else non-serializable: fall back to string"
        try:
            "Some objects (rare) implement array-like protocol"
            return np.asarray(obj).tolist()
        except Exception:
            return str(obj)

    report_dict: Dict[str, Any] = {}

    "OptimizeResult behaves like a dict; use its items to avoid fabricating fields"
    try:
        for key, val in opt_result.items():
            "Be gentle with 'hess_inv' (stringify to avoid massive or opaque structures)"
            if key == "hess_inv":
                report_dict[key] = str(val)
            else:
                report_dict[key] = _json_sanitize(val)
    except Exception:
        "Fallback: probe common attributes defensively (still keep \"only-if-present\")"
        for key in ("fun", "x", "jac", "hess_inv", "grad", "status", "success",
                  "message", "nfev", "nit", "njev", "maxcv",        # SLSQP / COBYLA-ish fields
                  "constr_violation", "barrier_parameter", "cg_niter"):  # trust-constr
            if hasattr(opt_result, key):
                val = getattr(opt_result, key)
                report_dict[key] = str(val) if key == "hess_inv" else _json_sanitize(val)

    "Optional extras"
    if duration is not None:
        report_dict["duration"] = float(duration)
    if loss is not None:
        report_dict["loss"] = float(loss)

    return report_dict


def transform_cov_params(params: dict[str, float], param_info: dict[str, list[str | float | tuple[float]]],
                         huge_loss: float = 1e6, is_psd_tol: float = 1e-12, asymmetry_tol: float = 1e-12, minimum_eigval: float = 1e-6,
                         player_role: str = 'predictor', include_covariance: bool = True, print_: bool = False) -> dict[str, dict[str, float] | str | float | None]:
    """
    Validate and repair a predictor's covariance parameter block so that it is positive semidefinite
    and consistent with the declared parameter bounds.

    If the covariance matrix implied by `params` is not PSD, the nearest PSD matrix (in Frobenius
    norm) is computed and the repaired diagonal standard deviations are checked against the bounds
    stored in `param_info`.  Any standard deviation that falls outside its declared bounds is
    clamped to the nearest boundary.  The repaired parameter values are written back into a copy
    of `params` and returned alongside a diagnostic report string.

    If `player_role != 'predictor'` or `include_covariance` is False, a penalty loss is returned
    immediately without attempting any repair, because covariance parameters only apply to the
    predictor role.

    Arguments:
        • params: dict[str, float]
            Current parameter dict, expected to include `_std` and `_cov` entries that together
            define a covariance matrix via `build_covariation_matrix`.
        • param_info: dict[str, list[str | float | tuple[float]]]
            Full parameter specification including `'keys'` and `'bounds'`; used to look up
            `_std` bounds for boundary enforcement after PSD repair.
        • huge_loss: float
            Penalty value returned when the role or settings make covariance repair inapplicable.
        • is_psd_tol: float
            Tolerance passed to `is_positive_semidefinite`; eigenvalues below this are treated
            as non-positive.
        • asymmetry_tol: float
            Tolerance for asymmetry detection (currently passed to the PSD check infrastructure).
        • minimum_eigval: float
            Minimum eigenvalue enforced when constructing the nearest PSD matrix.
        • player_role: str
            Must be `'predictor'`; any other value triggers an immediate penalty return.
        • include_covariance: bool
            If False, no covariance matrix exists and the function returns a penalty immediately.
        • print_: bool
            If True, prints diagnostic messages when the matrix is altered or bounds are violated.

    Returns:
        • dict with keys:
            - `'params'`: dict[str, float] — repaired parameter dict (or the original if no
              repair was needed).
            - `'report'`: str — human-readable description of any changes made.
            - `'loss'`: float | None — penalty loss if repair was inapplicable or a bound
              violation was found; None if the parameters are valid.
    """
    if player_role != 'predictor':
        report_str = "Warning: Only use transform_cov_params for 'predictors'."
        if print_:
            print(report_str)
        return {
            'params': params,
            'report': report_str,
            'loss': huge_loss
        }

    if not include_covariance:
        report_str = "Warning: Only use transform_cov_params when there are covariance parameters."
        if print_:
            print(report_str)
        return {
            'params': params,
            'report': report_str,
            'loss': huge_loss
        }

    "Ensure covariance matrix is symmetric and PSD."
    cov_matrix = build_covariation_matrix(
        param_info=param_info,
        params=params,
        raise_on_invalid=False  
    )
    altered_cov_matrix = False 
    if not is_positive_semidefinite(matrix=cov_matrix, tol=is_psd_tol):    
        cov_matrix = nearest_psd_matrix(matrix=cov_matrix, min_eigval=minimum_eigval)
        altered_cov_matrix = True

        "Ensure that the altered matrix respects parameter boundaries."
        param_mean_keys = [param_key for param_key in param_info['keys'] if '_std' not in param_key]
        for idx, param_key in enumerate(param_mean_keys):
            std_key = param_key + "_std"
            if std_key not in param_info["keys"]:
                continue  # no std?
            idx_std = param_info["keys"].index(std_key)
            (lower_bound, upper_bound) = param_info["bounds"][idx_std]

            "'Repaired' std is square root of diagonal matrix value at (idx, idx)."
            stdev = np.sqrt(cov_matrix[idx, idx]) 
            if not (lower_bound <= stdev <= upper_bound):
                if lower_bound >= stdev:
                    cov_matrix[idx, idx] = round(lower_bound ** 2, 9)
                else:
                    cov_matrix[idx, idx] = int(upper_bound ** 2)
                    report_str = f"After PSD fix, stdev for param '{param_key}' = {stdev} is "
                    report_str += f"out of user-specified bounds [{lower_bound}, {upper_bound}]."                            
                    if print_:
                        print(report_str)
                    return {
                        'params': params,
                        'report': report_str,
                        'loss': huge_loss
                    }

        "Ensure that the altered matrix is symmetric."
        for idx, key1 in enumerate(param_mean_keys):
            for jdx, key2 in enumerate(param_mean_keys[idx + 1:], start=idx + 1):
                if abs(cov_matrix[idx][jdx] - cov_matrix[jdx][idx]) > asymmetry_tol:
                    cov_matrix[idx][jdx] = cov_matrix[jdx][idx]
                    report_str = "Asymmetry detected in covariance matrix:"
                    if print_:
                        print(report_str)
                        print(cov_matrix)
                    return {
                        'params': params,
                        'report': report_str,
                        'loss': huge_loss
                    }

    if not is_positive_semidefinite(matrix=cov_matrix, tol=is_psd_tol):
        report_str = "[objective] Covariance not PSD => penalty."
        if print_:
            print(report_str)
        
        return {
            'params': params,
            'report': report_str,
            'loss': huge_loss
        }

    try:
        from scipy.stats import multivariate_normal # type: ignore
        param_means = [param_val for param_key, param_val in params.items()
                        if not any(key in param_key for key in ('_std', '_cov', 'τ', 'temp'))]
        multivariate_normal(mean=param_means, cov=cov_matrix, allow_singular=False)
    except np.linalg.LinAlgError:
        report_str = "Failed Multivariate Normal:"
        if print_:
            print(report_str)
            print(cov_matrix)
        return {
            'params': params,
            'report': report_str,
            'loss': huge_loss
        }
    
    if altered_cov_matrix:
        "Updating the param dictionary with altered parameters."
        param_mean_keys = [param_key for param_key in param_info['keys'] if '_std' not in param_key]
        for idx, key1 in enumerate(param_mean_keys):
            params[key1 + '_std'] = math.sqrt(cov_matrix[idx][idx])
            for jdx, key2 in enumerate(param_mean_keys[idx + 1:], start=idx + 1):      
                cov_key = f"{key1}_{key2}_cov"
                params[cov_key] = cov_matrix[idx][jdx]

    return {
        'params': params,
        'report': 'success',
        'loss': None
    }


def equation_to_settings(equation_function: Callable, utility_settings: UtilitySettings, file_paths: FilePaths, create_new_file: bool = False) -> dict:
    """
    Maps string equations to utility settings.
    
    Arguments:
        • equation_function: Callable; Converts utility settings into pretty equations in string form.
        • utility_settings: dict[str, bool]; Boolean flags that define functional forms of utility functions.
        • file_paths: FilePaths; Dictionary of file paths for files in this analysis.
        • create_new_file: bool; 
            - If True, creates a new file even if one already exists. 
            - Otherwise, extracts preexisting file.
    
    Returns:
        • equ_to_setting: Maps equations to settings
    """
    file_name = "equation_to_settings.json"
    file_path = os.path.join(file_paths["processed"], file_name)

    "Return existing model nesting data if possible and desired."
    if not create_new_file and os.path.exists(path=file_path):
        
        with open(file_path, "r", encoding="utf-8") as file:
            equ_to_settings = json.load(file) 

        if isinstance(equation_to_settings, dict) and all(
            isinstance(key, str) for key in equ_to_settings.keys()):
            return equ_to_settings

    utility_setting_varieties = generate_utility_settings(utility_settings=utility_settings, sort_by_k=True)

    equ_to_settings = {}
    for utility_settings_variety in utility_setting_varieties:
        equation = equation_function(utility_settings_variety)
        if isinstance(equation, str) and equation:
            equ_to_settings[equation] = utility_settings_variety

    if not os.path.exists(path=file_paths["processed"]):
        os.makedirs(name=file_paths["processed"])

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(equ_to_settings, file, ensure_ascii=False, indent=4)    

    return equ_to_settings


def convert_utility_settings(utility_settings: Union[Dict[str, bool], Tuple[bool, ...]], into: type = tuple) -> Union[Dict[str, bool], Tuple[bool, ...]]:
    """
    Converts between a dict[str, bool] of utility options and a tuple[bool].
    By design, tuple ordering follows the *insertion order* of keys in the reference.

    Arguments:
        • utility_settings: dict[str, bool] | tuple[bool]; The structure to convert.
        • into: type; Either 'tuple' or 'dict' to indicate the target type. Default is tuple.
        • keys: list[str] | None; Required when converting from tuple->dict to recover names.
            - If converting dict->tuple and 'keys' is None, uses the dict's insertion order.
            - If converting tuple->dict, you *must* provide 'keys' in the canonical order
              used across this codebase (e.g., list(utility_settings_template.keys())).

    Returns:
        • dict[str, bool] or tuple[bool]

    Raises:
        • ValueError if tuple->dict conversion is requested without 'keys'.
    """
    "Standardized utility setting keys in order."
    ordered_keys = (
        'conditional_welfare_mode',
        'reference_dependent_altruism',
        'min_max_rawlsian_leontief',
        'use_exponential_parameters',
        'apply_exponents_to_payoffs',
        'single_exponential_parameter',
        'single_payoffs_not_differences',
        'payoff_ratios_not_differences',
        'reference_dependent_utility',
        'use_negativity_parameters',
        'negativity_social_comparison',
        'fix_self_interest_parameter',
        'include_social_comparison',
        'include_altruism_term',
    )

    if into not in (tuple, dict, int, str):
        raise ValueError("`into` must be either `tuple`, `dict`, `str`, or `int`.")

    if not isinstance(utility_settings, (dict, tuple)):
        raise TypeError(f"utility_settings must be a tuple or dict, not {type(utility_settings)}.")

    if into is int or into is str:
        if isinstance(utility_settings, dict):
            try:
                if into is str:
                    return "".join("1" if bool(utility_settings[key]) else "0" for key in ordered_keys)
                return tuple(int(utility_settings[k]) for k in ordered_keys)
            except KeyError as err:
                for key in ordered_keys:
                    setting = utility_settings.get(key, None)
                    if setting is None:
                        print(f"Missing key from utility_settings: {key}")
                raise KeyError(err)
        else:
            n_ordered_keys, n_settings = len(ordered_keys), len(utility_settings)
            if n_ordered_keys != n_settings:
                raise ValueError(f"N keys ({n_ordered_keys}) ≠ N settings ({n_settings})!") 
            if into is str:
                return "".join("1" if bool(value) else "0" for value in utility_settings.values())         
            return tuple(int(flag) for flag in utility_settings)

    if isinstance(utility_settings, dict) and into is tuple:
        try:
            return tuple(bool(utility_settings[k]) for k in ordered_keys)
        except KeyError as err:
            for key in ordered_keys:
                setting = utility_settings.get(key, None)
                if setting is None:
                    print(f"Missing key from utility_settings: {key}")
            raise KeyError(err)

    if isinstance(utility_settings, tuple) and into is dict:
        n_ordered_keys, n_settings = len(ordered_keys), len(utility_settings)
        if n_ordered_keys != n_settings:
            raise ValueError(f"N keys ({n_ordered_keys}) ≠ N settings ({n_settings})!")
        return {key: bool(val) for key, val in zip(ordered_keys, utility_settings)}

    "Already the requested type"
    return utility_settings  # type: ignore[return-value]


def is_valid_utility_settings(candidate: UtilitySettings, provide_explanation: bool = False) -> bool:
    """
    Boolean-only validator that mirrors the filtering logic in `generate_utility_settings`.
    Returns True iff the candidate passes exactly the same constraints.

    Notes
        • This is a direct transcription of the rules, but expressed as returns rather than 'continue'.
        • Keep this in one place so generation & validation remain consistent.
    """
    "Determines how many social preference parameters are in the equation."
    n_social_preference_params = sum([
        int(not candidate['fix_self_interest_parameter']), 
        int(candidate['include_altruism_term']), 
        int(candidate['include_social_comparison'])
    ])
    if n_social_preference_params == 0:
        explanation = "If there are zero social preference parameters (only payoff(s) for self), then "
        if candidate['use_negativity_parameters'] or candidate['negativity_social_comparison']:
            explanation += "parameters are irrelevant."
            return explanation if provide_explanation else False
        if not candidate['single_exponential_parameter']:
            explanation += "there can only be at most one exponent."
            return explanation if provide_explanation else False
        
    elif n_social_preference_params == 1:
        explanation = "If only one term exists"
        if not candidate['single_exponential_parameter']:
            explanation += ", then there must be only one exponent."
            return explanation if provide_explanation else False
        if candidate['fix_self_interest_parameter']:
            if candidate['use_negativity_parameters'] and not candidate['conditional_welfare_mode']:
                explanation = "Redundant setting: negativity adds no effective parameter when Vᵢᵢ is fixed and no other negative terms are present."
                return explanation if provide_explanation else False

    if candidate['use_negativity_parameters']:
        if not candidate['negativity_social_comparison']:
            if candidate['include_social_comparison']:
                explanation = "If using negativity parameters, then the social comparison term must have a negativity parameter."
                return explanation if provide_explanation else False
        if candidate['single_payoffs_not_differences']:
            explanation = "With single payoffs that are always positive, it does not make sense to use negativity parameters."
            return explanation if provide_explanation else False

    if not candidate['include_social_comparison']:
        if candidate['negativity_social_comparison']:
            explanation = "If no social comparison term, then a nonexistent term cannot have positive and negative sides."
            return explanation if provide_explanation else False
        
    if not candidate['use_exponential_parameters']:
        explanation = "If no exponents, then "
        if not candidate['single_exponential_parameter']:
            explanation += "nonexistent exponents cannot be different for each term."
            return explanation if provide_explanation else False
        if candidate['apply_exponents_to_payoffs']:
            explanation += "nonexistent exponents cannot be applied to payoffs."
            return explanation if provide_explanation else False

    if candidate['single_payoffs_not_differences']:
        explanation = "If using single payoffs, then "
        if candidate['payoff_ratios_not_differences']:
            explanation += "payoffs cannot be expressed as ratios."
            return explanation if provide_explanation else False
        if candidate['reference_dependent_utility']:
            explanation += "payoffs cannot be reference dependent."
            return explanation if provide_explanation else False
        if candidate['apply_exponents_to_payoffs']:
            explanation += "exponents cannot be applied before a transformation that never occurs."
            return explanation if provide_explanation else False

    if candidate['conditional_welfare_mode'] and candidate['min_max_rawlsian_leontief']:
        explanation = "The utility function cannot be both in conditional welfare mode and in min-max form."
        return explanation if provide_explanation else False

    if candidate['reference_dependent_altruism']:
        if not candidate['conditional_welfare_mode']:
            explanation = "Reference-dependent altruism only applies within conditional welfare mode."
            return explanation if provide_explanation else False
        
    if candidate['min_max_rawlsian_leontief']:
        explanation = "If using min-max Rawlsian or Leontief forms, then "
        if candidate['use_negativity_parameters'] or candidate['negativity_social_comparison']:
            explanation += "negativity parameters of any kind cannot be used."
            return explanation if provide_explanation else False
        if candidate['fix_self_interest_parameter']:
            explanation += "a self-interest parameter must be included."
            return explanation if provide_explanation else False

    if candidate['conditional_welfare_mode']:
        explanation = "If using conditional welfare mode, then "
        if candidate['use_negativity_parameters'] or candidate['negativity_social_comparison']:
            explanation += "negativity parameters of any kind cannot be used."
            return explanation if provide_explanation else False
        if candidate['fix_self_interest_parameter']:
            explanation += "a self-interest parameter must be included."
            return explanation if provide_explanation else False
        if candidate['include_social_comparison']:
            explanation += "a social comparison term must not be included."
            return explanation if provide_explanation else False

    return "Success!" if provide_explanation else True


def generate_utility_settings(
    utility_settings: UtilitySettings,
    general_settings: Optional[GeneralSettings] = None,
    file_paths: Optional[FilePaths] = None,
    sort_by_k: bool = False,
    create_new_file: bool = False,
    return_df: bool = False,
    build_equation_function: Optional[Callable] = None,
    **kwargs: Any,
) -> Union[List[UtilitySettings], pd.DataFrame]:
    """
    Returns all valid combinations of utility function settings, subject to the structural
    constraints encoded in `is_valid_utility_settings`. Serves as both the core generator
    and the retrieval interface for the central utility-function registry
    (processed/all_utility_functions.csv).

    When called with only `utility_settings` (and optionally `sort_by_k`), the function
    behaves exactly as before: it generates and returns a list of valid settings dicts.
    The additional parameters unlock registry-based workflows.

    Arguments:
        • utility_settings: UtilitySettings
            A canonical utility settings dict used to seed the Boolean flag universe.
            All 14 Boolean keys must be present; only their key names matter for generation.
        • general_settings: GeneralSettings | None
            Global analysis settings. Passed through to parameter-counting helpers when
            `sort_by_k=True` or `create_new_file=True`.
        • file_paths: FilePaths | None
            Must contain a 'processed' key. When provided and `create_new_file=False`, the
            function attempts to load the registry CSV before falling back to live generation.
            When `create_new_file=True`, `file_paths` is required.
        • sort_by_k: bool (default False)
            If True and returning a list, sort the list by ascending k_params. This default
            preserves backward compatibility with existing callers.
        • create_new_file: bool (default False)
            If True, rebuild processed/all_utility_functions.csv from scratch and return
            the resulting DataFrame or list. Requires `file_paths`.
        • return_df: bool (default False)
            If True, return a pd.DataFrame. If False, return list[UtilitySettings].
            Only meaningful when `file_paths` is provided or `create_new_file=True`.
        • build_equation_function: Callable | None
            If provided, called with a UtilitySettings dict to produce each model's equation
            string when building the registry. Ignored when `create_new_file=False`.

    Returns:
        • list[UtilitySettings] or pd.DataFrame — valid utility settings in requested form.

    Meanings of Utility Options:
        • 'conditional_welfare_mode':
            - If True, weights self and other payoffs differently when ahead versus behind.
            - If False, weights these payoffs the same regardless if ahead or behind.
        • 'reference_dependent_altruism':
            - If True, other's payoffs are weighted differently when one's payoffs are less than a reference point.
            - If False, other's payoffs are weighted the same regardless if one's payoffs are less than a reference point.
        • 'min_max_rawlsian_leontief':
            - If True, uses a Rawlsian of Leontief min-max type functional form.
            - If False, does not use a Rawlsian of Leontief min-max type functional form.
        • 'use_exponential_parameters':
            - If True, all terms have an exponent parameter.
            - If False, all terms lack an exponent parameter.
        • 'apply_exponents_to_payoffs':
            - If True, apply exponents to payoffs before transforming them with differences or ratios.
            - If False, exponents are applied to the bases after payoff differences or ratios have been computed.
        • 'single_exponential_parameter':
            - If True, all terms have the same exponent parameter if they have exponent parameters at all.
            - If False, all terms have unique exponent parmeters if they have exponent parameters at all.
        • 'single_payoffs_not_differences':
            - If True, the base of each term is a single payoff, not a difference between two payoffs.
            - If False, the base of each term is a difference between two payoffs, not a single payoff.
        • 'payoff_ratios_not_differences':
            - If True, the base of each term is a ratio between payoffs, not one payoff or a payoff difference.
            - If False, the base of each term is a single payoff or a difference between payoffs.
        • 'reference_dependent_utility':
            - If True, the utility function is reference dependent, meaning single payoffs are compared to a reference point of 3.
            - If False, the utility function is not reference dependent, meaning that the base can be one payoff or a payoff difference.
        • 'use_negativity_parameters':
            - If True, each parameter has a mirror opposite that only influence utility if the base is negative.
            - If False, each term functions the same way regardless of whether the base is positive or negative.
        • 'negativity_social_comparison':
            - If True, the social comparison term has a negativity side even if other terms do not.
            - If False, the social comparison term lacks a negativity side unless all terms have negativity parameters.
        • 'fix_self_interest_parameter':
            - If True, the self-interest parameter is fixed at 1.
            - If False, the self-interest parameter is free to vary.
        • 'include_social_comparison':
            - If True, the social comparison term is included in the equation.
            - If False, the social comparison is excluded from the equation.
        • 'include_altruism_term':
            - If True, the altruism term is included in the equation.
            - If False, the altruism term is excluded from the equation.

    Rules of Utility Options:
        • If the utility function is reference dependent, then the base is always one payoff minus
            the reference point of 3, regardless of the 'single_payoffs_not_differences' option.
        • The social comparison term is always a difference between two payoffs, regardless of whether
            or not the utility function is reference dependent or uses single payoffs or payoff differences.
        • The social comparison term can have a negativity side even if the other terms lack negativity sides, but
            if the other terms have negativity sides, then the social comparison term must also have a negativity side.
        • The terms have unique exponent parameters if and only if they have exponent parameters at all and they do not
            have a single exponent parameter.
    """
    "If requested, rebuild the registry CSV from scratch and return."
    if create_new_file:
        if file_paths is None:
            raise ValueError(
                "file_paths must be provided when create_new_file=True so the registry "
                "can be written to processed/all_utility_functions.csv."
            )
        registry_df = build_utility_function_registry(
            utility_settings=utility_settings,
            file_paths=file_paths,
            general_settings=general_settings,
            build_equation_function=build_equation_function,
        )
        if return_df:
            return registry_df
        canonical_flag_order = list(convert_utility_settings(utility_settings=utility_settings, into=dict).keys())
        return [
            {col: bool(row[col]) for col in canonical_flag_order}
            for _, row in registry_df.iterrows()
        ]

    "If file_paths is provided, try to load the registry CSV before generating from scratch."
    if file_paths is not None:
        registry_csv_path = os.path.join(file_paths["processed"], "all_utility_functions.csv")
        if os.path.exists(registry_csv_path):
            "Force utility_bitstring to str so leading zeros are preserved after CSV round-trip."
            registry_df = pd.read_csv(registry_csv_path, dtype={"utility_bitstring": str})
            canonical_flag_order = list(convert_utility_settings(utility_settings=utility_settings, into=dict).keys())
            missing_setting_columns = [col for col in canonical_flag_order if col not in registry_df.columns]
            if not missing_setting_columns:
                if return_df:
                    return registry_df
                return [
                    {col: bool(row[col]) for col in canonical_flag_order}
                    for _, row in registry_df.iterrows()
                ]
            print(
                f"WARNING: Registry CSV is missing setting columns: {missing_setting_columns}. "
                f"Falling through to live generation."
            )
        else:
            "Intentional fallback — keep permanently. Regenerates the registry from scratch and warns"
            "the user when all_utility_functions.csv is absent, ensuring the code works on any machine."
            print(
                "WARNING: processed/all_utility_functions.csv not found. "
                "Falling back to live generation. Run generate_utility_settings(..., create_new_file=True) "
                "to build the registry."
            )

    "Core generation path: exhaustively enumerate all valid flag combinations."
    bool_flags = {key: [False, True] for key in utility_settings.keys()}
    all_keys = sorted(bool_flags.keys())
    all_value_combos = it.product(*(bool_flags[k] for k in all_keys))

    valid_combos = []
    for combo in all_value_combos:
        candidate = dict(zip(all_keys, combo))
        if is_valid_utility_settings(candidate=candidate):
            valid_combos.append(candidate)

    if sort_by_k:
        settings_to_k = []
        for combo in valid_combos:
            param_keys = parameter_keys_for_utility_settings(
                utility_settings=combo, general_settings=None
            )
            settings_to_k.append((combo, len(param_keys)))
        sorted_settings_to_k = sorted(settings_to_k, key=lambda x: x[-1])
        valid_combos = [setting_to_k[0] for setting_to_k in sorted_settings_to_k]

    return valid_combos


def build_utility_function_registry(
    utility_settings: UtilitySettings,
    file_paths: FilePaths,
    general_settings: Optional[GeneralSettings] = None,
    build_equation_function: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    Builds the central utility-function registry and writes it to
    processed/all_utility_functions.csv. Every valid utility form receives one canonical
    row with a stable utility_idx, bitstring, parameter count, equation (if available),
    redundancy information, parent/sibling/child relations, and IC columns (if available).

    Sorting order is k_params ascending, then utility_bitstring ascending. This places
    simpler models (children) before more complex ones (parents), which supports the
    warm-starting logic in the IC robustness analysis.

    Arguments:
        • utility_settings: UtilitySettings
            A canonical utility settings dict used to seed generation. All 14 Boolean keys
            must be present; their insertion order defines the canonical bitstring order.
        • file_paths: FilePaths
            Must contain 'processed' (output directory). If 'bic_aic' is also present,
            IC results will be merged from the IC analysis CSV.
        • general_settings: GeneralSettings | None
            Passed to parameter-counting helpers. May be None.
        • build_equation_function: Callable | None
            If provided, called as build_equation_function(settings_dict) to produce the
            human-readable equation string for each utility form. When None, the equation
            column is left blank unless equations can be recovered from an existing IC CSV.

    Returns:
        • pd.DataFrame: the registry, one row per valid utility form, sorted and indexed.
    """
    canonical_flag_order: List[str] = list(convert_utility_settings(
        utility_settings=utility_settings, into=dict
    ).keys())

    "Enumerate all valid settings using the same core loop as the existing generator."
    bool_flags = {key: [False, True] for key in utility_settings.keys()}
    all_keys = sorted(bool_flags.keys())
    all_value_combos = it.product(*(bool_flags[k] for k in all_keys))

    valid_settings_list: List[UtilitySettings] = []
    for combo in all_value_combos:
        candidate = dict(zip(all_keys, combo))
        if is_valid_utility_settings(candidate=candidate):
            valid_settings_list.append(candidate)

    print(f"Registry builder: {len(valid_settings_list)} valid utility forms found.")

    "Build one row per valid setting with k_params and utility_bitstring."
    rows: List[Dict[str, Any]] = []
    for settings_dict in valid_settings_list:
        k_params_value = count_free_parameters(
            utility_settings=settings_dict, general_settings=general_settings
        )
        "Bitstring uses the canonical insertion order from convert_utility_settings, then formatted with dashes."
        raw_bitstring = convert_utility_settings(utility_settings=settings_dict, into=str)
        utility_bitstring = _format_utility_bitstring(raw_bitstring=raw_bitstring)

        row: Dict[str, Any] = {"k_params": k_params_value, "utility_bitstring": utility_bitstring}
        for flag_name in canonical_flag_order:
            row[flag_name] = bool(settings_dict[flag_name])
        rows.append(row)

    registry_df = pd.DataFrame(rows)

    "Sort by k_params ascending, then utility_bitstring ascending for deterministic indexing."
    registry_df = registry_df.sort_values(
        by=["k_params", "utility_bitstring"], ascending=[True, True]
    ).reset_index(drop=True)

    "Assign stable utility_idx as the row position in the sorted order."
    registry_df.insert(0, "utility_idx", registry_df.index)

    "Build a bitstring-to-utility_idx lookup for resolving family relations."
    bitstring_to_utility_idx: Dict[str, int] = dict(
        zip(registry_df["utility_bitstring"], registry_df["utility_idx"])
    )

    "Compute equation strings when a builder is provided."
    if build_equation_function is not None:
        equation_strings: List[str] = []
        for _, registry_row in registry_df.iterrows():
            settings_dict = {col: bool(registry_row[col]) for col in canonical_flag_order}
            try:
                equation_string = build_equation_function(settings_dict)
            except Exception:
                equation_string = ""
            equation_strings.append(equation_string)
        registry_df["equation"] = equation_strings
    else:
        registry_df["equation"] = ""

    "Redundancy columns are populated after the IC merge below, once equations are available."
    registry_df["redundant_with"] = ""
    registry_df["differing_settings"] = ""

    """
    Compute parent, sibling, and child relations via the same O(n²) pairwise approach used in
    model_nesting_adjacency_matrices. Single-pivot enumeration with _apply_minimal_dependent_fixes
    is unreliable here because trailing implications in that function can cascade additional flag
    changes beyond the intended pivot, causing the direct single-flag neighbor to be missed. The
    pairwise approach delegates all relation logic to classify_pair_relation, which is the canonical
    implementation and the source of truth for what constitutes a valid parent/child/sibling pair.
    """
    settings_list: List[UtilitySettings] = [
        {col: bool(registry_row[col]) for col in canonical_flag_order}
        for _, registry_row in registry_df.iterrows()
    ]
    n_models = len(settings_list)

    parents_by_idx: Dict[int, List[int]] = {i: [] for i in range(n_models)}
    siblings_by_idx: Dict[int, List[int]] = {i: [] for i in range(n_models)}
    children_by_idx: Dict[int, List[int]] = {i: [] for i in range(n_models)}

    for row_i in range(n_models):
        for col_j in range(row_i + 1, n_models):
            relation_i_to_j, relation_j_to_i, setting_flipped = classify_pair_relation(
                model_1=settings_list[row_i],
                model_2=settings_list[col_j],
                utility_settings=utility_settings,
                general_settings=general_settings,
            )

            "Flipping these mode-level flags produces neither relationship, consistent with model_nesting_adjacency_matrices."
            if setting_flipped in ("min_max_rawlsian_leontief", "conditional_welfare_mode"):
                continue

            idx_i = int(registry_df.iloc[row_i]["utility_idx"])
            idx_j = int(registry_df.iloc[col_j]["utility_idx"])

            if relation_i_to_j == "parent":
                children_by_idx[row_i].append(idx_j)
                parents_by_idx[col_j].append(idx_i)
            elif relation_i_to_j == "child":
                parents_by_idx[row_i].append(idx_j)
                children_by_idx[col_j].append(idx_i)
            elif relation_i_to_j == "sibling":
                siblings_by_idx[row_i].append(idx_j)
                siblings_by_idx[col_j].append(idx_i)

    registry_df["parents"] = [str(sorted(parents_by_idx[i])) for i in range(n_models)]
    registry_df["siblings"] = [str(sorted(siblings_by_idx[i])) for i in range(n_models)]
    registry_df["children"] = [str(sorted(children_by_idx[i])) for i in range(n_models)]

    "Initialize IC columns as NaN; populated later by information_criterion_analysis or by IC merge below."
    for ic_column_name in ("n_data", "pvar", "param_norm_sd", "loss_nll",
                            "AIC", "BIC", "ΔAIC", "ΔBIC", "AIC_rank", "BIC_rank"):
        registry_df[ic_column_name] = float("nan")

    "Initialize AMPD distance columns as NaN; populated in Stage 3–4."
    for distance_column_name in ("ampd_to_best_rand", "ampd_to_best_real", "policy_regret_norm"):
        registry_df[distance_column_name] = float("nan")

    "Attempt to merge existing IC results from bic_aic/ if the directory and CSV exist."
    if "bic_aic" in file_paths:
        ic_csv_path = os.path.join(
            file_paths["bic_aic"], "All_Utility_Forms_IC_Analysis_Experiment3.csv"
        )
        if os.path.exists(ic_csv_path):
            registry_df = _merge_ic_results_into_registry(
                registry_df=registry_df,
                ic_csv_path=ic_csv_path,
                canonical_flag_order=canonical_flag_order,
                populate_equation_from_ic=(build_equation_function is None),
            )
            print("Registry builder: IC results merged successfully.")
        else:
            print(
                f"Registry builder: IC CSV not found at {ic_csv_path}. "
                "IC columns left blank; run information_criterion_analysis to populate them."
            )

    """
    Compute redundancy columns now that equations are fully populated (either from
    build_equation_function or from the IC merge above). Any rows with a blank equation
    are treated as uniquely identifying — they will not be reported as redundant with anything.
    """
    non_blank_equation_mask = registry_df["equation"].str.len() > 0
    if non_blank_equation_mask.any():
        equation_to_idx_list: Dict[str, List[int]] = {}
        for _, registry_row in registry_df[non_blank_equation_mask].iterrows():
            eq = registry_row["equation"]
            if eq not in equation_to_idx_list:
                equation_to_idx_list[eq] = []
            equation_to_idx_list[eq].append(int(registry_row["utility_idx"]))

        redundant_with_entries: List[str] = []
        differing_settings_entries: List[str] = []
        for _, registry_row in registry_df.iterrows():
            eq = registry_row["equation"]
            if eq == "" or eq not in equation_to_idx_list:
                redundant_with_entries.append("")
                differing_settings_entries.append("")
                continue
            sharing_same_equation = tuple(sorted(equation_to_idx_list[eq]))
            redundant_with_entries.append(str(sharing_same_equation))

            "Find which Boolean settings differ among all models that share this equation."
            group_rows = registry_df[registry_df["utility_idx"].isin(sharing_same_equation)]
            differing_flags: List[str] = []
            for flag_name in canonical_flag_order:
                if group_rows[flag_name].nunique() > 1:
                    differing_flags.append(flag_name)
            differing_settings_entries.append(str(tuple(differing_flags)))

        registry_df["redundant_with"] = redundant_with_entries
        registry_df["differing_settings"] = differing_settings_entries

    "Arrange columns in the canonical order specified in model_recovery_simulation.md."
    ic_columns = [
        "n_data", "pvar", "param_norm_sd", "loss_nll",
        "AIC", "BIC", "ΔAIC", "ΔBIC", "AIC_rank", "BIC_rank",
    ]
    family_columns = ["parents", "siblings", "children"]
    distance_columns = ["ampd_to_best_rand", "ampd_to_best_real", "policy_regret_norm"]
    all_ordered_columns = (
        ["utility_idx", "utility_bitstring", "k_params"]
        + canonical_flag_order
        + ["redundant_with", "differing_settings"]
        + ic_columns
        + family_columns
        + distance_columns
        + ["equation"]
    )
    present_ordered_columns = [col for col in all_ordered_columns if col in registry_df.columns]
    registry_df = registry_df[present_ordered_columns]

    "Write the registry CSV."
    registry_csv_path = os.path.join(file_paths["processed"], "all_utility_functions.csv")
    try:
        registry_df.to_csv(registry_csv_path, index=False, encoding="utf-8-sig")
        print(f"Registry written: {registry_csv_path}  ({len(registry_df)} rows, "
              f"{registry_df['k_params'].max():.0f} max k_params)")
    except (PermissionError, OSError) as write_error:
        print(f"WARNING: Could not write registry to {registry_csv_path}: {write_error}")

    return registry_df


def _merge_ic_results_into_registry(
    registry_df: pd.DataFrame,
    ic_csv_path: str,
    canonical_flag_order: List[str],
    populate_equation_from_ic: bool = False,
) -> pd.DataFrame:
    """
    Merges IC analysis results from an existing bic_aic CSV into the central registry.
    Matches rows by boolean utility settings expressed as a bitstring (canonical flag order),
    not by the legacy idx column. This is robust to ordering differences between the IC CSV
    and the new registry.

    Arguments:
        • registry_df: pd.DataFrame
            The registry DataFrame to update. Modified on a copy; original is not mutated.
        • ic_csv_path: str
            Path to the IC analysis CSV (e.g., All_Utility_Forms_IC_Analysis_Experiment3.csv).
        • canonical_flag_order: list[str]
            Ordered list of Boolean setting column names, defining the bitstring bit order.
        • populate_equation_from_ic: bool (default False)
            If True and the registry's 'equation' column is blank, copy equation strings
            from the IC CSV into the registry.

    Returns:
        • pd.DataFrame: a copy of registry_df with IC columns populated where matches exist.
    """
    try:
        ic_df = pd.read_csv(ic_csv_path)
    except Exception as read_error:
        print(f"WARNING: Could not read IC CSV at {ic_csv_path}: {read_error}")
        return registry_df

    """
    Build a formatted bitstring for each IC row using column names (order-independent).
    The dashed format (XXXX-XXXX-XXXX-XX) must match the registry's utility_bitstring column
    so the merge lookup finds the correct rows.
    """
    def _ic_row_to_bitstring(ic_row: pd.Series) -> str:
        raw = "".join("1" if bool(ic_row[flag]) else "0" for flag in canonical_flag_order)
        return _format_utility_bitstring(raw_bitstring=raw)

    ic_df["utility_bitstring"] = ic_df.apply(_ic_row_to_bitstring, axis=1)

    "The IC CSV uses 'loss' as the NLL column; the registry uses 'loss_nll'."
    if "loss" in ic_df.columns and "loss_nll" not in ic_df.columns:
        ic_df = ic_df.rename(columns={"loss": "loss_nll"})

    ic_merge_column_names = [
        col for col in (
            "n_data", "pvar", "param_norm_sd", "loss_nll",
            "AIC", "BIC", "ΔAIC", "ΔBIC", "AIC_rank", "BIC_rank",
        )
        if col in ic_df.columns
    ]
    if populate_equation_from_ic and "equation" in ic_df.columns:
        ic_merge_column_names.append("equation")

    "Build a lookup from bitstring to IC row values."
    ic_lookup: Dict[str, Dict[str, Any]] = (
        ic_df.set_index("utility_bitstring")[ic_merge_column_names].to_dict(orient="index")
    )

    registry_df = registry_df.copy()
    matched_count = 0
    for registry_row_idx in registry_df.index:
        bitstring = registry_df.at[registry_row_idx, "utility_bitstring"]
        if bitstring in ic_lookup:
            for col in ic_merge_column_names:
                registry_df.at[registry_row_idx, col] = ic_lookup[bitstring][col]
            matched_count += 1

    unmatched_registry_count = len(registry_df) - matched_count
    unmatched_ic_count = len(ic_df) - matched_count
    print(
        f"IC merge: {matched_count} matched, "
        f"{unmatched_registry_count} registry rows unmatched, "
        f"{unmatched_ic_count} IC rows unmatched."
    )

    return registry_df


def select_utility_settings_subset(
    n_models: Optional[int] = None,
    hand_picked_subset: Optional[List[Union[int, UtilitySettings]]] = None,
    selection_mode: str = "random",
    file_paths: Optional[FilePaths] = None,
    general_settings: Optional[GeneralSettings] = None,
    utility_settings: Optional[UtilitySettings] = None,
    include_model_idxs: Optional[List[Union[int, UtilitySettings]]] = None,
    exclude_model_idxs: Optional[List[Union[int, UtilitySettings]]] = None,
    required_settings: Optional[UtilitySettings] = None,
    required_k_params: Optional[List[int]] = None,
    parents_of: Optional[List[Union[int, UtilitySettings]]] = None,
    siblings_of: Optional[List[Union[int, UtilitySettings]]] = None,
    children_of: Optional[List[Union[int, UtilitySettings]]] = None,
    distance_matrix_path: Optional[str] = None,
    random_seed: Optional[int] = None,
) -> List[UtilitySettings]:
    """
    Returns a filtered and/or diversity-selected subset of the 480 valid utility forms,
    drawing from the central registry (processed/all_utility_functions.csv).

    Selection modes:
        'random'            — uniform random sample.
        'random_by_k'       — equal allocation across k_params levels, then random within each.
        'random_by_setting' — coverage sample: ensures both True and False values appear for
                              each Boolean flag before filling remaining slots randomly.
        'hamming'           — max-min greedy: repeatedly adds the model whose minimum Hamming
                              distance to the already-selected set is greatest.
        'ampd'              — max-min greedy using a precomputed AMPD distance matrix
                              (requires distance_matrix_path; computed in Stage 3–4).

    For 'hamming' and 'ampd', if no forced seed is provided, the empirical IC winner
    (lowest AIC_rank) is used as the initial seed; ties broken by utility_idx.

    Arguments:
        • n_models: int | None
            How many models to return. If None or greater than the filtered candidate
            count, all surviving candidates are returned without downsampling.
        • hand_picked_subset: list[int | UtilitySettings] | None
            If provided, this pool is used instead of the full registry. Each element
            may be a utility_idx (int) or a UtilitySettings dict. Filters
            (required_settings, required_k_params, exclude_model_idxs) still apply.
            If n_models is None or >= len(hand_picked_subset), all are returned.
        • selection_mode: str
            One of 'random', 'random_by_k', 'random_by_setting', 'hamming', 'ampd'.
        • file_paths: FilePaths | None
            Must contain key 'processed' pointing to the directory holding
            all_utility_functions.csv.
        • general_settings: GeneralSettings | None
            Passed through; not used internally but accepted for forward compatibility.
        • utility_settings: UtilitySettings | None
            Used to derive canonical flag order if provided; otherwise inferred from
            the registry columns.
        • include_model_idxs: list[int | UtilitySettings] | None
            Models that must appear in the output regardless of the selection mode.
            Excluded models take priority over forced inclusions.
        • exclude_model_idxs: list[int | UtilitySettings] | None
            Models that must never appear in the output.
        • required_settings: UtilitySettings | None
            A partial UtilitySettings dict. Only models matching every specified
            Boolean flag value pass the filter. Omitted flags are unconstrained.
        • required_k_params: list[int] | None
            Only models whose k_params is in this list pass the filter. If None,
            all k values are eligible.
        • parents_of: list[int | UtilitySettings] | None
            Restricts candidates to models listed as parents of the specified models
            in the registry's 'parents' column.
        • siblings_of: list[int | UtilitySettings] | None
            Restricts candidates to models listed as siblings of the specified models.
        • children_of: list[int | UtilitySettings] | None
            Restricts candidates to models listed as children of the specified models.
        • distance_matrix_path: str | None
            Path to a precomputed distance matrix CSV (row/column headers = utility_idx).
            Required when selection_mode='ampd'.
        • random_seed: int | None
            Seed for reproducibility of random and stratified modes.

    Returns:
        • list[UtilitySettings] — one dict per selected model, in selection order.
    """
    import random as _random

    if file_paths is None or "processed" not in file_paths:
        raise ValueError(
            "select_utility_settings_subset requires file_paths['processed'] "
            "pointing to the directory containing all_utility_functions.csv."
        )

    registry_df = pd.read_csv(
        os.path.join(file_paths["processed"], "all_utility_functions.csv"),
        dtype={"utility_bitstring": str},
    )

    "Derive canonical flag order from utility_settings if provided; else infer from registry columns."
    non_flag_columns: set = {
        "utility_idx", "utility_bitstring", "k_params",
        "redundant_with", "differing_settings",
        "n_data", "pvar", "param_norm_sd", "loss_nll",
        "AIC", "BIC", "ΔAIC", "ΔBIC", "AIC_rank", "BIC_rank",
        "parents", "siblings", "children",
        "ampd_to_best_rand", "ampd_to_best_real", "policy_regret_norm",
        "equation",
    }
    if utility_settings is not None:
        canonical_flag_order = list(
            convert_utility_settings(utility_settings=utility_settings, into=dict).keys()
        )
    else:
        canonical_flag_order = [col for col in registry_df.columns if col not in non_flag_columns]

    def _parse_idx_list(cell_value: Any) -> List[int]:
        """Parse a stringified Python list like '[3, 17, 42]' into a list of ints."""
        s = str(cell_value).strip("[]")
        return [int(x.strip()) for x in s.split(",") if x.strip()] if s else []

    def _resolve_to_idx(model_ref: Union[int, UtilitySettings]) -> int:
        """Convert a utility_idx int or a UtilitySettings dict to a utility_idx int."""
        if isinstance(model_ref, int):
            return model_ref
        settings_dict = convert_utility_settings(utility_settings=model_ref, into=dict)
        raw_bit = convert_utility_settings(utility_settings=settings_dict, into=str)
        fmt_bit = _format_utility_bitstring(raw_bitstring=raw_bit)
        match = registry_df[registry_df["utility_bitstring"] == fmt_bit]
        if len(match) == 0:
            raise ValueError(f"UtilitySettings not found in registry: {model_ref}")
        return int(match.iloc[0]["utility_idx"])

    "Build the initial candidate pool by applying required_settings and required_k_params filters."
    candidate_df = registry_df.copy()
    if required_settings is not None:
        required_dict = convert_utility_settings(utility_settings=required_settings, into=dict)
        for flag_name, flag_value in required_dict.items():
            if flag_name in candidate_df.columns:
                candidate_df = candidate_df[candidate_df[flag_name] == flag_value]

    if required_k_params is not None:
        candidate_df = candidate_df[candidate_df["k_params"].isin(required_k_params)]

    "Restrict to family members of specified focal models when any family filter is given."
    family_filter_idxs: Optional[Set[int]] = None

    for family_column, focal_model_list in (
        ("parents", parents_of),
        ("siblings", siblings_of),
        ("children", children_of),
    ):
        if focal_model_list is None:
            continue
        relation_idxs: Set[int] = set()
        for focal_model_ref in focal_model_list:
            focal_idx = _resolve_to_idx(model_ref=focal_model_ref)
            focal_row = registry_df[registry_df["utility_idx"] == focal_idx]
            if len(focal_row):
                relation_idxs.update(
                    _parse_idx_list(cell_value=focal_row.iloc[0][family_column])
                )
        family_filter_idxs = (
            (family_filter_idxs & relation_idxs) if family_filter_idxs is not None else relation_idxs
        )

    if family_filter_idxs is not None:
        candidate_df = candidate_df[candidate_df["utility_idx"].isin(family_filter_idxs)]

    "Apply exclusions; excluded models are removed from the candidate pool."
    exclude_idxs: Set[int] = set()
    if exclude_model_idxs is not None:
        exclude_idxs = {_resolve_to_idx(model_ref=m) for m in exclude_model_idxs}
    candidate_df = candidate_df[~candidate_df["utility_idx"].isin(exclude_idxs)]

    "Forced inclusions are added to the output regardless of selection mode (exclusions take priority)."
    forced_include_idxs: List[int] = []
    if include_model_idxs is not None:
        for model_ref in include_model_idxs:
            forced_idx = _resolve_to_idx(model_ref=model_ref)
            if forced_idx not in exclude_idxs:
                forced_include_idxs.append(forced_idx)

    def _idx_list_to_settings(utility_idx_list: List[int]) -> List[UtilitySettings]:
        """Convert a list of utility_idx values to a list of UtilitySettings dicts."""
        rows = registry_df[registry_df["utility_idx"].isin(utility_idx_list)]
        rows = rows.set_index("utility_idx").loc[
            [i for i in utility_idx_list if i in rows["utility_idx"].values]
        ]
        return [{col: bool(row[col]) for col in canonical_flag_order} for _, row in rows.iterrows()]

    "Handle hand_picked_subset: it replaces the full candidate pool."
    if hand_picked_subset is not None:
        hand_picked_idxs = [_resolve_to_idx(model_ref=m) for m in hand_picked_subset]
        valid_candidate_idxs = set(candidate_df["utility_idx"])
        hand_picked_idxs = [i for i in hand_picked_idxs if i in valid_candidate_idxs]
        if n_models is None or n_models >= len(hand_picked_idxs):
            selected_idxs = hand_picked_idxs
        else:
            selected_idxs = hand_picked_idxs[:n_models]
        for forced_idx in forced_include_idxs:
            if forced_idx not in selected_idxs:
                selected_idxs.append(forced_idx)
        return _idx_list_to_settings(utility_idx_list=selected_idxs)

    "If n_models is None or >= candidate count, return all candidates plus forced inclusions."
    all_candidate_idxs: List[int] = list(candidate_df["utility_idx"])
    if n_models is None or n_models >= len(all_candidate_idxs):
        selected_idxs = list(set(all_candidate_idxs) | set(forced_include_idxs))
        return _idx_list_to_settings(utility_idx_list=selected_idxs)

    if random_seed is not None:
        _random.seed(random_seed)

    "Initialize the selected set with forced inclusions and compute how many more are needed."
    selected_set: List[int] = list(dict.fromkeys(forced_include_idxs))
    remaining_candidates: List[int] = [
        i for i in all_candidate_idxs if i not in set(selected_set)
    ]
    n_to_select = n_models - len(selected_set)

    if n_to_select <= 0:
        return _idx_list_to_settings(utility_idx_list=selected_set[:n_models])

    if selection_mode == "random":
        additional = _random.sample(
            population=remaining_candidates, k=min(n_to_select, len(remaining_candidates))
        )
        selected_set.extend(additional)

    elif selection_mode == "random_by_k":
        """Sample with equal allocation across k_params levels, then fill remaining slots randomly."""
        remaining_df = candidate_df[candidate_df["utility_idx"].isin(remaining_candidates)]
        k_groups: Dict[int, List[int]] = {
            k_val: list(group["utility_idx"])
            for k_val, group in remaining_df.groupby("k_params")
        }
        n_k_levels = len(k_groups)
        per_k = max(1, n_to_select // n_k_levels)
        sampled_from_k: List[int] = []
        for k_val, group_idxs in k_groups.items():
            sampled_from_k.extend(
                _random.sample(population=group_idxs, k=min(per_k, len(group_idxs)))
            )
        _random.shuffle(sampled_from_k)
        selected_set.extend(sampled_from_k[:n_to_select])
        if len(selected_set) < n_models:
            still_needed = n_models - len(selected_set)
            leftover = [i for i in remaining_candidates if i not in set(selected_set)]
            selected_set.extend(
                _random.sample(population=leftover, k=min(still_needed, len(leftover)))
            )

    elif selection_mode == "random_by_setting":
        """Coverage sample: for each Boolean flag, pick at least one True model and one False model
        before filling remaining slots randomly. This maximizes flag-level diversity."""
        covered: Set[int] = set(selected_set)
        coverage_picks: List[int] = []
        remaining_df = candidate_df[candidate_df["utility_idx"].isin(remaining_candidates)]
        for flag_name in canonical_flag_order:
            for flag_value in (True, False):
                if len(covered) + len(coverage_picks) - len(selected_set) >= n_to_select:
                    break
                eligible = remaining_df[
                    (remaining_df[flag_name] == flag_value) &
                    (~remaining_df["utility_idx"].isin(covered))
                ]["utility_idx"].tolist()
                if eligible:
                    pick = _random.choice(eligible)
                    coverage_picks.append(pick)
                    covered.add(pick)
        selected_set.extend(coverage_picks[:n_to_select])
        if len(selected_set) < n_models:
            still_needed = n_models - len(selected_set)
            leftover = [i for i in remaining_candidates if i not in set(selected_set)]
            selected_set.extend(
                _random.sample(population=leftover, k=min(still_needed, len(leftover)))
            )

    elif selection_mode in ("hamming", "ampd"):
        if selection_mode == "ampd":
            if distance_matrix_path is None:
                raise ValueError(
                    "selection_mode='ampd' requires a precomputed AMPD distance matrix CSV. "
                    "Provide distance_matrix_path, or use selection_mode='hamming' as a "
                    "structural-distance proxy until the AMPD matrix is computed (Stage 3–4)."
                )
            dist_matrix = pd.read_csv(distance_matrix_path, index_col=0)
            dist_matrix.index = dist_matrix.index.astype(int)
            dist_matrix.columns = dist_matrix.columns.astype(int)

            def _get_distance(idx_a: int, idx_b: int) -> float:
                return float(dist_matrix.loc[idx_a, idx_b])

        else:
            bits_by_idx: Dict[int, str] = dict(
                zip(registry_df["utility_idx"], registry_df["utility_bitstring"].str.replace("-", ""))
            )

            def _get_distance(idx_a: int, idx_b: int) -> float:
                ba, bb = bits_by_idx[idx_a], bits_by_idx[idx_b]
                return float(sum(a != b for a, b in zip(ba, bb)))

        "Seed the max-min greedy search: use forced includes if available, else the IC winner."
        if not selected_set:
            ic_available = candidate_df[candidate_df["AIC_rank"].notna()]
            if len(ic_available):
                seed_idx = int(ic_available.loc[ic_available["AIC_rank"].idxmin(), "utility_idx"])
            else:
                seed_idx = _random.choice(remaining_candidates)
            selected_set.append(seed_idx)
            remaining_candidates = [i for i in remaining_candidates if i != seed_idx]

        n_to_select = n_models - len(selected_set)
        for _ in range(n_to_select):
            if not remaining_candidates:
                break
            "Add the candidate whose minimum distance to the current selected set is largest."
            best_candidate: Optional[int] = None
            best_min_dist: float = -1.0
            for cand_idx in remaining_candidates:
                min_dist_to_selected = min(
                    _get_distance(cand_idx, sel_idx) for sel_idx in selected_set
                )
                if min_dist_to_selected > best_min_dist:
                    best_min_dist = min_dist_to_selected
                    best_candidate = cand_idx
            if best_candidate is not None:
                selected_set.append(best_candidate)
                remaining_candidates.remove(best_candidate)

    else:
        raise ValueError(
            f"Unknown selection_mode: {selection_mode!r}. "
            f"Valid modes: 'random', 'random_by_k', 'random_by_setting', 'hamming', 'ampd'."
        )

    return _idx_list_to_settings(utility_idx_list=selected_set)


def compute_hamming_distance_matrix(
    file_paths: FilePaths,
    utility_settings: Optional[UtilitySettings] = None,
    create_new_file: bool = False,
) -> pd.DataFrame:
    """
    Computes the pairwise Hamming distance matrix over all valid utility forms in the
    central registry and caches it to processed/. The matrix is symmetric, has a zero
    diagonal, and contains integer values in [0, 14] (one per Boolean flag position).

    The cache filename encodes the number of models so that a matrix computed on a
    different registry version does not silently overwrite or shadow the current one:
        processed/model_distance_hamming__n_models={M}.csv

    Row and column labels are utility_idx values (integers). The matrix can be reloaded
    by select_utility_settings_subset (hamming mode) and all downstream geometry analyses.

    Arguments:
        • file_paths: FilePaths
            Must contain key 'processed' pointing to the directory that holds
            all_utility_functions.csv and where the distance matrix is cached.
        • utility_settings: UtilitySettings | None
            Used to derive canonical flag order when provided. If None, the flag order
            is inferred from the registry columns (non-metadata columns).
        • create_new_file: bool (default False)
            If False and a cached matrix exists for the current registry size, that
            file is loaded and returned. If True, the matrix is recomputed and the
            cache is overwritten.

    Returns:
        • pd.DataFrame — square symmetric matrix indexed and columned by utility_idx.
            All values are non-negative integers; diagonal is zero.
    """
    registry_df = pd.read_csv(
        os.path.join(file_paths["processed"], "all_utility_functions.csv"),
        dtype={"utility_bitstring": str},
    )
    n_models = len(registry_df)
    cache_path = os.path.join(
        file_paths["processed"], f"model_distance_hamming__n_models={n_models}.csv"
    )

    "Return cached matrix when available and not overridden."
    if not create_new_file and os.path.exists(cache_path):
        hamming_df = pd.read_csv(cache_path, index_col=0)
        hamming_df.index = hamming_df.index.astype(int)
        hamming_df.columns = hamming_df.columns.astype(int)
        print(f"Hamming matrix loaded from cache: {cache_path}  ({n_models}×{n_models})")
        return hamming_df

    "Extract raw 14-bit strings (dashes removed) indexed by utility_idx for fast comparison."
    utility_idx_values: List[int] = list(registry_df["utility_idx"].astype(int))
    raw_bits_by_position: List[str] = [
        bits.replace("-", "") for bits in registry_df["utility_bitstring"]
    ]

    "Compute the full upper triangle of pairwise Hamming distances."
    distance_matrix: List[List[int]] = [
        [0] * n_models for _ in range(n_models)
    ]
    for row_index in range(n_models):
        row_bits = raw_bits_by_position[row_index]
        for col_index in range(row_index + 1, n_models):
            col_bits = raw_bits_by_position[col_index]
            hamming_distance = sum(a != b for a, b in zip(row_bits, col_bits))
            distance_matrix[row_index][col_index] = hamming_distance
            distance_matrix[col_index][row_index] = hamming_distance

    hamming_df = pd.DataFrame(
        data=distance_matrix,
        index=utility_idx_values,
        columns=utility_idx_values,
    )

    "Sanity checks: symmetry, zero diagonal, integer values, range [0, 14]."
    assert (hamming_df == hamming_df.T).all().all(), "Hamming matrix is not symmetric."
    assert (hamming_df.values.diagonal() == 0).all(), "Hamming matrix diagonal is not zero."
    assert hamming_df.max().max() <= 14, "Hamming distance exceeds 14 (number of flags)."

    hamming_df.to_csv(cache_path)
    print(
        f"Hamming matrix computed and cached: {cache_path}  "
        f"({n_models}×{n_models}, max distance={int(hamming_df.max().max())})"
    )
    return hamming_df


def compute_conditional_hamming_distance_matrix(
    file_paths: FilePaths,
    utility_settings: Optional[UtilitySettings] = None,
    create_new_file: bool = False,
) -> pd.DataFrame:
    """
    Pairwise conditional Hamming distance matrix over all valid utility forms in the registry.

    Like the raw Hamming matrix, counts Boolean flag mismatches between pairs of models —
    but only at positions where the flag is semantically live in BOTH models. A flag is live
    in a model if flipping that flag (all other flags held fixed) still yields a valid utility
    settings combination per is_valid_utility_settings.

    This corrects raw Hamming's inflation when a parent feature is absent: e.g., two models
    with no exponents and one with per-term exponents should have equal conditional Hamming
    distance to the no-exponent model on the 'single_exponential_parameter' axis, because
    that flag is forced (not live) in the no-exponent model.

    is_valid_utility_settings is the sole source of truth for liveness — if new settings or
    dependency rules are added there, this function adapts automatically.

    Arguments:
        • file_paths: FilePaths — must contain 'processed' pointing to the registry directory.
        • utility_settings: UtilitySettings | None — unused; present for API parity with
            compute_hamming_distance_matrix.
        • create_new_file: bool (default False) — if False and a cached matrix exists, load it.

    Returns:
        • pd.DataFrame — square symmetric matrix indexed and columned by utility_idx.
            Values are non-negative integers; diagonal is zero; max ≤ 14.
    """
    registry_df = pd.read_csv(
        os.path.join(file_paths["processed"], "all_utility_functions.csv"),
        dtype={"utility_bitstring": str},
    )
    n_models  = len(registry_df)
    cache_path = os.path.join(
        file_paths["processed"],
        f"model_distance_conditional_hamming__n_models={n_models}.csv",
    )

    "Return cached matrix when available and not overridden."
    if not create_new_file and os.path.exists(cache_path):
        cond_hamming_df = pd.read_csv(cache_path, index_col=0)
        cond_hamming_df.index   = cond_hamming_df.index.astype(int)
        cond_hamming_df.columns = cond_hamming_df.columns.astype(int)
        print(f"Conditional Hamming matrix loaded from cache: {cache_path}  ({n_models}×{n_models})")
        return cond_hamming_df

    "Build settings dicts from bitstrings via convert_utility_settings (canonical key order)."
    utility_idx_values: List[int] = list(registry_df["utility_idx"].astype(int))
    settings_list: List[dict] = [
        convert_utility_settings(
            tuple(bool(int(c)) for c in row["utility_bitstring"].replace("-", "")),
            into=dict,
        )
        for _, row in registry_df.iterrows()
    ]

    "Precompute live flags per model: a flag is live if flipping it still yields a valid model."
    flag_keys: List[str] = list(settings_list[0].keys())
    live_flags_list: List[set] = []
    for settings in settings_list:
        live = set()
        for key in flag_keys:
            flipped = dict(settings)
            flipped[key] = not flipped[key]
            if is_valid_utility_settings(flipped):
                live.add(key)
        live_flags_list.append(live)

    "Count mismatches only where the flag is live in BOTH models."
    distance_matrix: List[List[int]] = [[0] * n_models for _ in range(n_models)]
    for i in range(n_models):
        settings_i = settings_list[i]
        live_i     = live_flags_list[i]
        for j in range(i + 1, n_models):
            both_live = live_i & live_flags_list[j]
            dist = sum(settings_i[k] != settings_list[j][k] for k in both_live)
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist

    cond_hamming_df = pd.DataFrame(
        data=distance_matrix,
        index=utility_idx_values,
        columns=utility_idx_values,
    )

    assert (cond_hamming_df == cond_hamming_df.T).all().all(), "Matrix is not symmetric."
    assert (cond_hamming_df.values.diagonal() == 0).all(), "Diagonal is not zero."

    cond_hamming_df.to_csv(cache_path)
    print(
        f"Conditional Hamming matrix computed and cached: {cache_path}  "
        f"({n_models}×{n_models}, max distance={int(cond_hamming_df.max().max())})"
    )
    return cond_hamming_df


def identify_redundant_utility_functions(
    utility_settings: UtilitySettings,
    build_equation_function: callable,
    file_paths: dict[str, str],
    compute_ampd_fn: Optional[callable] = None,
    general_settings: Optional[Dict[str, Any]] = None,
    param_bds: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Finds redundant utility functions and reports which settings cause the redundancy.
    Runs two independent checks:

    CHECK 1 — Equation-string redundancy:
        Detects models whose utility equations are algebraically identical (same string).
        These are true duplicates: two different flag combinations that collapse to the
        same functional form.

        Procedure:
            1) Generate all valid utility_settings via generate_utility_settings(…).
            2) Render each as a pretty equation via build_equation_function(…).
            3) For each equation, compute:
                • equation_count: number of occurrences
                • redundant_with: tuple of utility_idx sharing this equation (sorted)
                • differing_settings: tuple of setting names that differ within the group
            4) Write a CSV with one row per utility function, flags in canonical order,
               'equation' rightmost, redundant equations grouped adjacent.

    CHECK 2 — AMPD behavioral redundancy (optional; skipped when compute_ampd_fn is None):
        Detects model pairs whose equations differ algebraically but whose choice
        distributions are empirically indistinguishable across Monte Carlo parameter draws
        and payoff structures. An off-diagonal AMPD value ≤ epsilon is a red flag that
        two nominally distinct models produce identical behavior.

        Adds two columns to the DataFrame (None when compute_ampd_fn is not provided):
            • ampd_zero_with: tuple of utility_idx with AMPD ≤ epsilon; None if none.
            • ampd_zero_differing_settings: flags that differ in each such pair; None if none.

    Arguments:
        • utility_settings: dict[str, bool] | tuple[bool]
            Canonical utility_settings structure; seeds generate_utility_settings and
            determines canonical flag order.
        • build_equation_function: callable
            Takes a dict[str, bool] and returns the pretty equation string.
        • file_paths: dict[str, str]
            Must contain 'processed' for the output CSV location.
        • compute_ampd_fn: callable | None (default None)
            When provided, called as compute_ampd_fn(general_settings=…, file_paths=…,
            param_bds=…, utility_settings=…, create_new_file=False) to obtain or
            compute the 480×480 AMPD distance matrix. When None, Check 2 is skipped
            and no AMPD columns are added — callers that should not trigger a potentially
            slow AMPD computation (e.g. quick_demo.py) should leave this as None.
        • general_settings: dict | None
            Passed through to compute_ampd_fn. Required when compute_ampd_fn is provided.
        • param_bds: dict | None
            Passed through to compute_ampd_fn. Required when compute_ampd_fn is provided.

    Returns:
        • pd.DataFrame; columns:
            ['utility_idx', <flags… in canonical order>, 'equation_count',
             'redundant_with', 'differing_settings', 'equation',
             'ampd_zero_with', 'ampd_zero_differing_settings']
            The last two columns are present only when compute_ampd_fn is provided.
    """
    "Canonical flag order derived from the provided utility_settings"
    canonical_flags: list[str] = list(convert_utility_settings(utility_settings, into=dict).keys())

    "Enumerate all valid settings in canonical order"
    all_settings: list[dict[str, bool]] = generate_utility_settings(utility_settings=utility_settings)

    rows: list[dict[str, Any]] = []
    for utility_index, settings_item in enumerate(all_settings):
        settings_dict = convert_utility_settings(settings_item, into=dict)
        equation_string = build_equation_function(settings_dict)

        row: dict[str, Any] = {"utility_idx": utility_index, "equation": equation_string}
        "Add flags in canonical order"
        for flag_name in canonical_flags:
            row[flag_name] = bool(settings_dict[flag_name])
        rows.append(row)

    df = pd.DataFrame(rows)

    "Count identical equations"
    df["equation_count"] = df.groupby("equation")["utility_idx"].transform("count")

    "Build 'redundant_with': sorted tuple of utility_idx sharing the equation"
    equation_to_indices: dict[str, tuple[int, ...]] = (
        df.groupby("equation")["utility_idx"]
          .apply(lambda s: tuple(sorted(map(int, s.tolist()))))
          .to_dict()
    )
    df["redundant_with"] = df["equation"].map(equation_to_indices)

    "Build 'differing_settings': which flags differ within each equation group"
    def _differing_flags_for_group(group_df: pd.DataFrame) -> tuple[str, ...]:
        differing: list[str] = []
        for flag in canonical_flags:
            "unique values in this group for the flag"
            uniques = group_df[flag].unique()
            if len(uniques) > 1:
                differing.append(flag)
        return tuple(differing)

    differing_by_equation: dict[str, tuple[str, ...]] = (
        df.groupby("equation")
          .apply(_differing_flags_for_group)
          .to_dict()
    )
    df["differing_settings"] = df["equation"].map(differing_by_equation)

    "Sorting so redundant rows are adjacent:"
    "1) equation_count (desc) — put redundancies on top"
    "2) redundant_with (ascending by tuple) — groups adjacent"
    "3) utility_idx (asc) — stable within group"
    df = df.sort_values(
        by=["equation_count", "redundant_with", "utility_idx"],
        ascending=[False, True, True]
    ).reset_index(drop=True)

    "Reorder columns: utility_idx, flags (canonical order), equation_count,"
    "redundant_with, differing_settings, equation (rightmost)"
    ordered_cols = (
        ["utility_idx"] +
        canonical_flags +
        ["equation_count", "redundant_with", "differing_settings", "equation"]
    )
    df = df[ordered_cols]

    "Write CSV"
    out_path = os.path.join(file_paths["processed"], "redundant_utility_functions.csv")
    try: df.to_csv(out_path, index=False, encoding="utf-8-sig")
    except (PermissionError, OSError): pass

    "Console feedback — Check 1"
    n_total = len(df)
    n_unique = df["equation"].nunique()
    if n_unique == n_total:
        print(f"Check 1 (equation strings): All {n_total} utility functions are unique. No redundancies found.")
    else:
        n_redundant = n_total - n_unique
        n_groups_gt1 = int((df["equation_count"] > 1).sum())
        print(f"Check 1 (equation strings): Found {n_redundant} redundant utility functions "
              f"across {n_groups_gt1} duplicated groups.")

    "Check 2 — AMPD behavioral redundancy (skipped when compute_ampd_fn is None)"
    if compute_ampd_fn is not None and general_settings is not None and param_bds is not None:
        _ampd_epsilon = 1e-6
        ampd_matrix: pd.DataFrame = compute_ampd_fn(
            general_settings=general_settings,
            file_paths=file_paths,
            param_bds=param_bds,
            utility_settings=utility_settings,
            create_new_file=False,
        )

        "Map utility_idx → row in df so we can look up flags for any model index"
        idx_to_flags: dict[int, dict[str, bool]] = {
            int(row_["utility_idx"]): {f: bool(row_[f]) for f in canonical_flags}
            for _, row_ in df.iterrows()
        }

        "Collect off-diagonal pairs with AMPD ≤ epsilon"
        ampd_zero_with: dict[int, list[int]] = {}
        n_models = len(ampd_matrix)
        for i_idx, row_label in enumerate(ampd_matrix.index):
            i = int(row_label)
            for j_idx, col_label in enumerate(ampd_matrix.columns):
                j = int(col_label)
                if i >= j:
                    continue   # upper-triangle only; symmetric
                val = ampd_matrix.iloc[i_idx, j_idx]
                if pd.notna(val) and float(val) <= _ampd_epsilon:
                    ampd_zero_with.setdefault(i, []).append(j)
                    ampd_zero_with.setdefault(j, []).append(i)

        "Compute differing flags for each flagged model's zero-AMPD partners"
        def _ampd_differing(model_i: int, partners: list[int]) -> tuple[str, ...]:
            flags_i = idx_to_flags.get(model_i, {})
            differing: set[str] = set()
            for partner in partners:
                flags_j = idx_to_flags.get(partner, {})
                for flag in canonical_flags:
                    if flags_i.get(flag) != flags_j.get(flag):
                        differing.add(flag)
            return tuple(sorted(differing))

        "Attach columns to df"
        df["ampd_zero_with"] = df["utility_idx"].apply(
            lambda idx: tuple(sorted(ampd_zero_with[int(idx)])) if int(idx) in ampd_zero_with else None
        )
        df["ampd_zero_differing_settings"] = df.apply(
            lambda r: _ampd_differing(int(r["utility_idx"]), list(ampd_zero_with[int(r["utility_idx"])]))
                      if int(r["utility_idx"]) in ampd_zero_with else None,
            axis=1,
        )

        "Update CSV with the new columns"
        try: df.to_csv(out_path, index=False, encoding="utf-8-sig")
        except (PermissionError, OSError): pass

        "Console feedback — Check 2"
        n_flagged_pairs = sum(len(v) for v in ampd_zero_with.values()) // 2
        if n_flagged_pairs == 0:
            print(f"Check 2 (AMPD behavioral, ε={_ampd_epsilon:.0e}): "
                  f"All off-diagonal AMPD values > ε. No behavioral redundancies found.")
        else:
            print(f"Check 2 (AMPD behavioral, ε={_ampd_epsilon:.0e}): "
                  f"WARNING — {n_flagged_pairs} off-diagonal pair(s) have AMPD ≤ ε. "
                  f"These models produce statistically indistinguishable behavior:")
            printed: set[tuple[int, int]] = set()
            for model_i, partners in sorted(ampd_zero_with.items()):
                for model_j in sorted(partners):
                    if (min(model_i, model_j), max(model_i, model_j)) in printed:
                        continue
                    printed.add((min(model_i, model_j), max(model_i, model_j)))
                    diff_flags = _ampd_differing(model_i=model_i, partners=[model_j])
                    print(f"  Models ({model_i}, {model_j}): differing flags: {', '.join(diff_flags) or '(none)'}")

    return df


def count_free_parameters(utility_settings: UtilitySettings, general_settings: Optional[Dict[str, Any]] = None) -> int:
    """
    Convenience wrapper that returns the number of *free mean* parameters by default.
    Covariance params are intentionally excluded here (IC counting often focuses on means).
    """
    keys = parameter_keys_for_utility_settings(
        utility_settings=utility_settings,
        general_settings=general_settings,
    )
    "Count only free mean parameters; strip _std and _cov keys that grid/MCMC modes append."
    return len([key for key in keys if not key.endswith('_std') and not key.endswith('_cov')])


def _apply_minimal_dependent_fixes(utility_settings: UtilitySettings, pivot: str) -> UtilitySettings:
    """
    Apply the *minimal* additional flips implied by the rules 
    so a single pivot yields a valid candidate wherever possible.

    Examples of couplings handled:
        • use_exponential_parameters=False    ⇒ single_exponential_parameter=False
        • include_social_comparison=False     ⇒ negativity_social_comparison=False
        • use_negativity_parameters=True      ⇒ negativity_social_comparison=True
        • payoff_ratios_not_differences=True  ⇒ single_payoffs_not_differences=False
        • reference_dependent_utility=True    ⇒ single_payoffs_not_differences=False
        • single_payoffs_not_differences=True ⇒ reference_dependent_utility=False
        • min_max_rawlsian_leontief=True      ⇒ force its specific bundle
        • conditional_welfare_mode=True       ⇒ enforce its bundle
    """
    utility_settings = copy.deepcopy(utility_settings)

    if pivot == 'use_exponential_parameters':
        if not utility_settings['use_exponential_parameters']:
            utility_settings['single_exponential_parameter'] = False
        else:
            utility_settings['single_exponential_parameter'] = True

    if pivot == 'include_social_comparison':
        if not utility_settings['include_social_comparison']:
            utility_settings['negativity_social_comparison'] = False

    if pivot == 'use_negativity_parameters':
        if utility_settings['use_negativity_parameters']:
            utility_settings['negativity_social_comparison'] = True
        else:
            "If no social comparison, must be False"
            if not utility_settings['include_social_comparison']:
                utility_settings['negativity_social_comparison'] = False

    if pivot == 'payoff_ratios_not_differences' and utility_settings['payoff_ratios_not_differences']:
        utility_settings['single_payoffs_not_differences'] = False

    if pivot == 'reference_dependent_utility' and utility_settings['reference_dependent_utility']:
        utility_settings['single_payoffs_not_differences'] = False

    if pivot == 'single_payoffs_not_differences' and utility_settings['single_payoffs_not_differences']:
        utility_settings['reference_dependent_utility'] = False

    if pivot == 'min_max_rawlsian_leontief' and utility_settings['min_max_rawlsian_leontief']:
        "Enforce the 'min-max' bundle from the generator"
        utility_settings['conditional_welfare_mode'] = False
        utility_settings['reference_dependent_altruism'] = False
        utility_settings['use_negativity_parameters'] = False
        utility_settings['negativity_social_comparison'] = False
        utility_settings['fix_self_interest_parameter'] = False
        utility_settings['single_payoffs_not_differences'] = True
        utility_settings['single_exponential_parameter'] = True

    if pivot == 'conditional_welfare_mode' and utility_settings['conditional_welfare_mode']:
        utility_settings['use_negativity_parameters'] = True
        utility_settings['include_social_comparison'] = True
        utility_settings['negativity_social_comparison'] = True
        if not utility_settings['include_altruism_term']:
            "This is required to remain valid in the generator under conditional mode"
            utility_settings['fix_self_interest_parameter'] = False
            utility_settings['single_exponential_parameter'] = True

    "Enforce trailing implications that might be triggered indirectly:"
    "(No extra pivots beyond the minimal implications above.)"
    if not utility_settings['include_social_comparison']:
        utility_settings['negativity_social_comparison'] = False
    if utility_settings['payoff_ratios_not_differences']:
        utility_settings['single_payoffs_not_differences'] = False
    if utility_settings['single_payoffs_not_differences']:
        utility_settings['reference_dependent_utility'] = False
    if not utility_settings['use_exponential_parameters']:
        utility_settings['single_exponential_parameter'] = False
    "Only force negativity_social_comparison=True when social comparison is actually present."
    "The original rule was unconditional, which created invalid candidates (include_social_comparison=False"
    "with negativity_social_comparison=True) and silently broke sibling detection."
    if utility_settings['use_negativity_parameters'] and utility_settings['include_social_comparison']:
        utility_settings['negativity_social_comparison'] = True

    return utility_settings


def _format_utility_bitstring(raw_bitstring: str) -> str:
    """
    Formats a 14-character raw bitstring into XXXX-XXXX-XXXX-XX for human readability
    and Excel safety (the dashes prevent Excel from interpreting the value as an integer
    and silently stripping leading zeros).

    Arguments:
        • raw_bitstring: str
            A 14-character string of '0' and '1' characters in canonical flag order.

    Returns:
        • str — formatted as 'XXXX-XXXX-XXXX-XX' (groups of 4-4-4-2, separated by dashes).
    """
    return f"{raw_bitstring[0:4]}-{raw_bitstring[4:8]}-{raw_bitstring[8:12]}-{raw_bitstring[12:14]}"


def parents_children_of(utility_settings: Union[UtilitySettings, BoolTuple], return_children: bool = True, 
                        return_parents: bool = True, general_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[List[BoolTuple]]]:
    """
    Returns immediate neighbors (by one *pivot* change, allowing dependent fixes).
    Child/Parent is defined by Δk = k(neighbor) - k(focal), which may be > 1 (e.g., exponent toggles).

    Arguments:
        • utility_settings: dict[str, bool] | tuple[bool]; Focal model.
        • return_children: bool; If True, compute simpler neighbors (Δk < 0).
        • return_parents:  bool; If True, compute more complex neighbors (Δk > 0).
        • general_settings: dict[str, Any] | None; Used only for parameter counting, not required.
        • keys: list[str] | None; Required if `utility_settings` is a tuple (to recover names).

    Returns:
        • dict with:
            {'children': [tuple[bool], ...] or [],
             'parents':  [tuple[bool], ...] or None }  # None when not requested
    """
    "Normalize to dict with preserved key order"
    if isinstance(utility_settings, tuple):
        base = convert_utility_settings(utility_settings, into=dict)  # type: ignore
    else:
        base = copy.deepcopy(utility_settings)
    ordered_keys = list(base.keys())

    if not is_valid_utility_settings(base):
        explanation = is_valid_utility_settings(candidate=base, provide_explanation=True)
        raise ValueError(explanation)

    k0 = count_free_parameters(base, general_settings=general_settings)

    candidates: Dict[str, List[BoolTuple]] = {'children': [], 'parents': []}
    seen: set = set()

    "Consider each boolean flag as the single *pivot* to toggle; add dependent fixes minimally"
    for pivot in ordered_keys:
        cand = copy.deepcopy(base)
        cand[pivot] = not cand[pivot]
        cand = _apply_minimal_dependent_fixes(cand, pivot)

        if cand == base:
            continue
        if not is_valid_utility_settings(cand):
            continue

        k1 = count_free_parameters(cand, general_settings=general_settings)
        tup = convert_utility_settings(cand, into=tuple)
        if tup in seen:
            continue
        seen.add(tup)

        if return_children and (k1 < k0):
            candidates['children'].append(tup)
        if return_parents and (k1 > k0):
            candidates['parents'].append(tup)

    if not return_children:
        candidates['children'] = []
    if not return_parents:
        candidates['parents'] = None

    return candidates


def classify_pair_relation(model_1: Union[UtilitySettings, BoolTuple], model_2: Union[UtilitySettings, BoolTuple], 
                           utility_settings: UtilitySettings, general_settings: Optional[Dict[str, Any]] = None, print_: bool = False) -> bool:
    """
    Classify the relation between two utility specifications that differ in exactly one flag.

    Returns (rel_1_to_2, rel_2_to_1, changed_setting) where each relation is one of:
        'parent', 'child', 'sibling', or 'neither'.

    Rules implemented:
        • Only models differing in exactly one boolean setting can be relatives.
        • Flipping 'min_max_rawlsian_leontief' or 'conditional_welfare_mode' alone ⇒ neither.
        • Inside the min–max family, flipping 'include_social_comparison' (Rawlsian ↔ Leontief) ⇒ neither.
        • Siblings: flags that change functional form without changing number of free parameters
            (e.g., apply_exponents_to_payoffs, single_payoffs_not_differences, payoff_ratios_not_differences,
            reference_dependent_*).
        • Parent/child: flags that change the number of free parameters
            (e.g., use_exponential_parameters, single_exponential_parameter, use_negativity_parameters,
            negativity_social_comparison, include_social_comparison (outside min–max), include_altruism_term,
            fix_self_interest_parameter).
        • For conditional-welfare models, flipping include_altruism_term **is** parent/child; the parent has explicit
            altruism weights (Vᵢⱼ, Ʌᵢⱼ). The child ties them to self-interest (1-Vᵢᵢ, 1-Ʌᵢᵢ).
    """
    model_1 = convert_utility_settings(utility_settings=model_1, into=dict)
    model_2 = convert_utility_settings(utility_settings=model_2, into=dict)

    if not is_valid_utility_settings(model_1) or not is_valid_utility_settings(model_2):
        if print_:
            if not is_valid_utility_settings(model_1):
                print(f"Invalid utility settings detected in model 1!")
                print(is_valid_utility_settings(model_2, provide_explanation=True))
                for setting_key, setting_val in model_1.items():
                    setting_key += " " * (30 - len(setting_key))
                    print(f"{setting_key}: {setting_val},")
            if not is_valid_utility_settings(model_2):
                print(f"Invalid utility settings detected in model 2!")
                print(is_valid_utility_settings(model_2, provide_explanation=True))
                for setting_key, setting_val in model_2.items():
                    setting_key += " " * (30 - len(setting_key))
                    print(f"{setting_key}: {setting_val},")
        "Relatives must have valid utility settings"
        return ('neither', 'neither', None) 

    settings_when_flipped_dont_make_relatives = (
        'conditional_welfare_mode', 
        'min_max_rawlsian_leontief'
    )    
    settings_when_flipped_make_siblings = (
        'apply_exponents_to_payoffs',
        'single_payoffs_not_differences',
        'payoff_ratios_not_differences',
        'reference_dependent_altruism',
        'reference_dependent_utility'
    )
    settings_when_flipped_make_children_parents = (
        setting for setting in utility_settings 
        if setting not in settings_when_flipped_make_siblings 
        and setting not in settings_when_flipped_dont_make_relatives
    )
    settings_when_flipped_make_children_parents = (
        'use_exponential_parameters',
        'single_exponential_parameter',
        'use_negativity_parameters',
        'negativity_social_comparison',
        'fix_self_interest_parameter',
        'include_social_comparison',
        'include_altruism_term',
    )

    different_settings = 0
    different_setting = None
    for utility_setting in utility_settings:
        if model_1[utility_setting] != model_2[utility_setting]:
            different_setting = utility_setting
            different_settings += 1    

    if different_settings != 1:
        "Relatives must differ by only one setting."
        return ('neither', 'neither', different_setting)    

    if different_setting in settings_when_flipped_dont_make_relatives:
        "Relatives cannot be created by flipping these settings."
        return ('neither', 'neither', different_setting) 

    k_params_m1 = count_free_parameters(utility_settings=model_1, general_settings=general_settings)
    k_params_m2 = count_free_parameters(utility_settings=model_2, general_settings=general_settings)

    if k_params_m1 == k_params_m2:
        "Determine if models are siblings."

        for utility_setting in utility_settings:
            if model_1[utility_setting] != model_2[utility_setting]:
                if utility_setting not in settings_when_flipped_make_siblings:
                    if model_1['conditional_welfare_mode'] and model_2['conditional_welfare_mode']:
                        if utility_setting == 'include_altruism_term':
                            continue
                    elif model_1['min_max_rawlsian_leontief'] and model_2['min_max_rawlsian_leontief']:
                        if utility_setting in ('include_altruism_term', 'include_social_comparison'):
                            continue
                    "Siblings can only be created by flipping specific utility settings."
                    return ('neither', 'neither', different_setting)

        return ('sibling', 'sibling', different_setting)

    else:
        for utility_setting in utility_settings:
            if model_1[utility_setting] != model_2[utility_setting]:
                if utility_setting not in settings_when_flipped_make_children_parents:        
                    "Parents and children can only be created by flipping specific utility settings."
                    return ('neither', 'neither', different_setting)
                if model_1['min_max_rawlsian_leontief'] and model_2['min_max_rawlsian_leontief']:
                    if utility_setting == 'include_social_comparison':
                        "Rawlsian and Leontief forms are not relatives of any kind."
                        return ('neither', 'neither', different_setting)   

        if k_params_m1 > k_params_m2:
            return ('parent', 'child', different_setting)
        else:
            return ('child', 'parent', different_setting)


def load_fitted_parameters(
    player_uuid: str,
    player_role: str,
    general_settings: Dict[str, Any],
    utility_settings: Union[UtilitySettings, BoolTuple],
    param_info: Dict[str, Any],
    file_paths: Dict[str, Any],
    create_file_name_suffix: object,
    experiment_num: Optional[int] = None
) -> Dict[str, float]:
    """
    Loads the fitted parameters for a given player/role under the specified settings.

    This uses the file-naming convention via `create_file_name_suffix(general_settings, utility_settings)`
    and searches the player-fits directory:
        Iter_Binary_Dictator/player_fits/experiment_{experiment_num}/
    for a file named:
        "<file_name_suffix>_<player_uuid>.json"

    Arguments:
        • player_uuid: str
        • role: str; One of {'chooser','predictor'}
        • general_settings: dict[str, Any]
        • utility_settings: dict[str, bool] | tuple[bool]
        • param_info: dict with at least 'keys' field (used to filter/ordered return)
        • file_paths: the `file_paths` mapping (uses 'outputs' first, then 'inputs' as fallback)
        • experiment_num: Optional[int]; If None, attempts general_settings['experiment_num'].
        • keys: Required only if `utility_settings` is passed as a tuple (for flag-name recovery)

    Returns:
        • dict[str, float]; Ordered to match param_info['keys'] where possible.

    Notes:
        • JSON structure is expected to contain:
              parameter_estimates[update_method][player_uuid][role]['params']
        • Picks the first matching block found during a recursive scan of the JSON payload.
    """
    "Normalize options -> dict to build file name suffix deterministically"
    if isinstance(utility_settings, tuple):
        uopts = convert_utility_settings(utility_settings, into=dict)  # type: ignore
    else:
        uopts = utility_settings

    "Import locally to avoid circulars if these live in different modules"
    try:
        file_name_suffix = create_file_name_suffix(general_settings, uopts)  # type: ignore[name-defined]
    except NameError as err:
        raise NameError("`create_file_name_suffix` must be imported into the current namespace.") from err

    if experiment_num is None:
        experiment_num = int(general_settings.get('experiment_num', 3))

    fname = f"{file_name_suffix}_{player_uuid}.json"

    "Candidate roots to search (prefer outputs, then inputs)"
    roots: List[str] = []
    if 'outputs' in file_paths:
        roots.append(os.path.join(file_paths['outputs'], 'Iter_Binary_Dictator', 
                                  'player_fits', f"experiment_{experiment_num}"))
    if 'inputs' in file_paths:
        roots.append(os.path.join(file_paths['inputs'], 'Iter_Binary_Dictator', 
                                  'player_fits', f"experiment_{experiment_num}"))

    found_path: Optional[str] = None
    for root in roots:
        candidate = os.path.join(root, fname)
        if os.path.exists(candidate):
            found_path = candidate
            break
        "Fallback: look for the exact suffix + uuid among files if the name had changed slightly upstream"
        if os.path.isdir(root):
            for file in os.listdir(root):
                if file.endswith(f"_{player_uuid}.json") and file.startswith(file_name_suffix):
                    found_path = os.path.join(root, file)
                    break
        if found_path is not None:
            break

    if found_path is None:
        raise FileNotFoundError(f"Could not locate fitted-params file '{fname}' in: {roots}")

    with open(found_path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    update_method = general_settings.get('update_method', 'naive')

    def _extract_params(obj: Any) -> Optional[Dict[str, float]]:
        """Recursive finder for parameter_estimates[update_method][player_uuid][role]['params']"""
        if isinstance(obj, dict):
            if 'parameter_estimates' in obj:
                pe = obj['parameter_estimates']
                if isinstance(pe, dict) and update_method in pe:
                    block = pe[update_method]
                    "The file format uses UUID at the next level"
                    if isinstance(block, dict) and player_uuid in block:
                        by_role = block[player_uuid]
                        if isinstance(by_role, dict) and player_role in by_role:
                            rblock = by_role[player_role]
                            if isinstance(rblock, dict) and 'params' in rblock:
                                params = rblock['params']
                                if isinstance(params, dict):
                                    "Coerce to floats"
                                    return {key: float(val) for key, val in params.items()}
            "Recurse dict values"
            for val in obj.values():
                out = _extract_params(val)
                if out is not None:
                    return out
        elif isinstance(obj, list):
            for val in obj:
                out = _extract_params(val)
                if out is not None:
                    return out
        return None

    params_found = _extract_params(payload)
    if params_found is None:
        raise ValueError(
            f"Found '{found_path}' but could not find parameter_estimates"
            f"['{update_method}']['{player_uuid}']['{player_role}']['params']."
        )

    "Order according to param_info['keys'] if provided"
    if isinstance(param_info, dict) and 'keys' in param_info:
        ordered = {}
        for key in param_info['keys']:
            if key in params_found:
                ordered[key] = params_found[key]
        "Also include any extra params from the file (if any)"
        for key, val in params_found.items():
            if key not in ordered:
                ordered[key] = val
        return ordered

    return params_found


def map_child_to_parent_special_param_info(
    child_utility_settings: Union[UtilitySettings, BoolTuple],
    parent_utility_settings: Union[UtilitySettings, BoolTuple],
    child_fitted_parameters: Dict[str, float],
    general_settings: Dict[str, Any],
    param_bds: ParameterBounds,
    build_utility_equation: Callable | None = None
) -> Dict[str, Any]:
    """
    Creates a parent `param_info` that *embeds* the child model as a special case using the
    child's fitted parameters whenever possible, and deterministic embedding values otherwise.

    This is the starting point for "neighborhood" refits: initialize the parent exactly at
    the child's optimum (within the parent space), then allow only the newly introduced
    parent dimensions to relax first, followed by the rest (outside this function).

    Arguments:
        • child_utility_settings: dict[str,bool] | tuple[bool]
        • parent_utility_settings: dict[str,bool] | tuple[bool]
        • child_fitted_parameters: dict[str,float]; keys are parameter names (e.g., 'Vᵢᵢ','γ1',...).
        • param_bds: dict[str, (low, high)]; parameter bounds used by `make_param_info`.
        • general_settings: dict[str,Any]; drives std/covariance conventions.

    Returns:
        • parent_param_info: dict[str, Any] with fields:
            - 'keys'   : ordered parameter names for the parent
            - 'bounds' : bounds aligned with 'keys'
            - 'guesses': a list of floats aligned with 'keys' that embed the child
            - optional 'covar' sub-dict if `include_covariance` is True

    Raises:
        • ValueError if the pair is not a valid child→parent relation under the rules.
        • NotImplementedError for transitions involving `min_max_rawlsian_leontief=True`.
    """
    "Normalize typed inputs to dicts (preserving insertion order of keys)."
    if isinstance(child_utility_settings, tuple):
        child_utility_settings = convert_utility_settings(child_utility_settings, into=dict)  # type: ignore
    if isinstance(parent_utility_settings, tuple):
        parent_utility_settings = convert_utility_settings(parent_utility_settings, into=dict)  # type: ignore

    if not is_valid_utility_settings(child_utility_settings):
        raise ValueError(f"Invalid Child Settings: {is_valid_utility_settings(child_utility_settings, provide_explanation=True)}")

    if not is_valid_utility_settings(parent_utility_settings):
        raise ValueError(f"Invalid Parent Settings: {is_valid_utility_settings(parent_utility_settings, provide_explanation=True)}")

    relation_1_to_2, relation_2_to_1, changed_utility_setting = classify_pair_relation(
        model_1=child_utility_settings,
        model_2=parent_utility_settings,
        general_settings=general_settings,
        utility_settings=child_utility_settings
    )
    if relation_1_to_2 != 'child' or relation_2_to_1 != 'parent':
        if build_utility_equation is not None:
            print(f"Model 1: {build_utility_equation(child_utility_settings)}")
            print(f"Model 2: {build_utility_equation(parent_utility_settings)}")
        raise ValueError(f"Requested mapping is not child→parent. Got: {relation_1_to_2} / {relation_2_to_1} - Flipped Setting: {changed_utility_setting}.")

    "Build a parent param_info scaffold (keys, bounds, covar). Override the guesses deterministically."
    parent_param_info = make_param_info(
        param_bds=param_bds,
        utility_settings=parent_utility_settings,
        general_settings=general_settings,
        guess_seed=None,
    )
    parent_keys: List[str] = list(parent_param_info["keys"])

    "Helper: robust lookup for a child's parameter with sensible embedding defaults."
    def _value_or_default(param_name: str) -> float:
        if param_name in child_fitted_parameters:
            return float(child_fitted_parameters[param_name])
        "Structured defaults for embedding when child lacked this dimension."
        if param_name == 'Vᵢᵢ':
            "If child had fixed self-interest, emulate the same behavior."
            return 1.0
        if param_name in ('Vᵢⱼ', 'Ʌᵢⱼ', 'Ƹᵢⱼ', 'Ʒᵢⱼ'):
            "If the term did not exist before, set to zero."
            return 0.0
        if param_name.startswith('γ'):
            "If the child had no exponents, embedding requires γ ≡ 1."
            return 1.0
        if param_name == 'Ʌᵢᵢ':
            "Without an explicit Vᵢᵢ in the child, the symmetric embedding is 1."
            return 1.0
        return 0.0  # Safe fallback for unexpected keys

    "Collect child's gamma structure for smarter tying."
    child_gamma_keys: List[str] = [key for key in child_fitted_parameters.keys() if key.startswith('γ')]
    child_has_no_gammas: bool = (len(child_gamma_keys) == 0)
    child_has_single_gamma: bool = (len(child_gamma_keys) == 1)
    child_common_gamma_value: Optional[float] = None
    if child_has_single_gamma:
        child_common_gamma_value = float(child_fitted_parameters[child_gamma_keys[0]])

    "Precompute base weights from the child when present (to tie negative/conditioned sides)."
    child_Vii: Optional[float] = child_fitted_parameters.get('Vᵢᵢ', None)
    child_Vij: Optional[float] = child_fitted_parameters.get('Vᵢⱼ', None)
    child_Envy: Optional[float] = child_fitted_parameters.get('Ƹᵢⱼ', None)  # envy
    child_Guilt: Optional[float] = child_fitted_parameters.get('Ʒᵢⱼ', None)  # guilt

    "Pass 1: create a dict of embedded values for the parent means."
    embedded_parent_values: Dict[str, float] = {}
    for param_name in parent_keys:
        if param_name.endswith('_std'):
            "Fill stds later after means are set."
            continue

        if param_name in child_fitted_parameters:
            embedded_parent_values[param_name] = float(child_fitted_parameters[param_name])
            continue

        "New parameter in the parent—choose an embedding value based on structure."
        if param_name == 'Ʌᵢᵢ':
            "Tie to self-interest mean if available; else fall back to fixed value 1.0"
            embedded_parent_values['Ʌᵢᵢ'] = float(child_Vii) if child_Vii is not None else 1.0
            continue

        if param_name == 'Ʌᵢⱼ':
            "Tie to altruism mean if available; else fall back to 0.0"
            embedded_parent_values['Ʌᵢⱼ'] = float(child_Vij) if child_Vij is not None else 0.0
            continue

        if param_name == 'Ʒᵢⱼ':
            "Split envy/guilt: set guilt equal to envy to reproduce the child."
            if child_Guilt is not None:
                embedded_parent_values['Ʒᵢⱼ'] = float(child_Guilt)
            elif child_Envy is not None:
                embedded_parent_values['Ʒᵢⱼ'] = float(child_Envy)
            else:
                embedded_parent_values['Ʒᵢⱼ'] = 0.0
            continue

        if param_name == 'Ƹᵢⱼ':
            "If social comparison is newly introduced, set envy to zero."
            embedded_parent_values['Ƹᵢⱼ'] = 0.0
            continue

        if param_name == 'Vᵢⱼ':
            "If altruism is newly introduced, set to zero."
            embedded_parent_values['Vᵢⱼ'] = 0.0
            continue

        if param_name == 'Vᵢᵢ':
            "If self-interest becomes unfixed in the parent, set to 1.0 to embed the child's fixed case."
            embedded_parent_values['Vᵢᵢ'] = 1.0
            continue

        if param_name.startswith('γ'):
            "Exponent handling:"
            if child_has_no_gammas:
                embedded_parent_values[param_name] = 1.0
            elif child_has_single_gamma and child_common_gamma_value is not None:
                "Tie all parent gammas to child's single gamma"
                embedded_parent_values[param_name] = child_common_gamma_value
            else:
                "Child had multiple gammas; align by name when possible, otherwise default to 1.0"
                embedded_parent_values[param_name] = float(child_fitted_parameters.get(param_name, 1.0))
            continue


        "Generic fallback (should be rare): use structured defaults."
        embedded_parent_values[param_name] = _value_or_default(param_name)

    "Pass 2: fill std keys if required by update method."
    if general_settings.get('update_method') in ('MCMC', 'grid'):
        min_std_guess = 0.5
        for param_name in parent_keys:
            if not param_name.endswith('_std'):
                continue
            base_name = param_name[:-4]
            if (base_name + '_std') in child_fitted_parameters:
                embedded_parent_values[param_name] = float(child_fitted_parameters[base_name + '_std'])
            else:
                embedded_parent_values[param_name] = min_std_guess

    "--- SPECIAL: conditional-welfare child (no explicit altruism) → parent (explicit altruism)"
    if (parent_utility_settings.get('conditional_welfare_mode', False)
        and child_utility_settings.get('conditional_welfare_mode', False)
        and (parent_utility_settings.get('include_altruism_term', False)
        != child_utility_settings.get('include_altruism_term', False))):

        "1) tie altruism means to self-interest means (to reproduce child)"
        Vii = float(child_fitted_parameters.get('Vᵢᵢ', 1.0))
        Lai = float(child_fitted_parameters.get('Ʌᵢᵢ', 0.0))
        if 'Vᵢⱼ' in parent_keys:
            embedded_parent_values['Vᵢⱼ'] = 1.0 - Vii
        if 'Ʌᵢⱼ' in parent_keys:
            embedded_parent_values['Ʌᵢⱼ'] = 1.0 - Lai

        "2) tie altruism curvature to self-interest curvature (if parent exposes γ2)"
        if 'γ2' in parent_keys:
            gamma1 = float(child_fitted_parameters.get('γ1', 1.0))
            embedded_parent_values['γ2'] = gamma1

    "Finally, override guesses with the deterministic embedded vector, ordered by parent_keys."
    parent_param_info["guesses"] = [float(embedded_parent_values[k]) for k in parent_keys]

    "Optional: annotate with a small, human-readable note to facilitate debugging/printing."
    parent_param_info["init_from_child"] = {
        "changed_utility_setting": changed_utility_setting,
        "child_has_exponents": not child_has_no_gammas,
        "child_gamma_keys": child_gamma_keys,
    }

    return parent_param_info


def summarize_nesting_relationship_counts(
    general_settings: Dict[str, Any],
    utility_settings: Dict[str, bool],
    file_paths: Dict[str, str],
    model_nesting_adjacency_matrices: Callable, *,
    create_new_file: bool = False,
    print_: bool = True
) -> Dict[str, int]:
    """
    Compute counts of unique parent–child and sibling–sibling relationships.

    Uses the adjacency lists produced by `model_nesting_adjacency_matrices`.
    If a direct sibling adjacency is not provided by that function, infers
    siblings as (i) same k (free parameter count), (ii) same model family
    (base vs conditional_welfare vs min_max), and (iii) Hamming distance = 1
    over the boolean settings vector.

    Returns:
        • dict with keys:
            - 'n_models'
            - 'n_parent_child_pairs'
            - 'n_sibling_pairs'
    """
    nesting = model_nesting_adjacency_matrices(
        general_settings=general_settings,
        utility_settings=utility_settings,
        file_paths=file_paths,
        create_new_file=create_new_file,
        print_=False
    )

    settings_list: List[Dict[str, bool]] = nesting['settings']
    adj_lists: Dict[str, List[List[int]]] = nesting['adjacency_lists']
    n_models = len(settings_list)

    "---- Parent–child: take from 'parent_of' only (already directed, no double-count) ----"
    parent_child_pairs: set[tuple[int, int]] = set()
    for parent_idx, child_idxs in enumerate(adj_lists.get('parent_of', [[] for _ in range(n_models)])):
        for child_idx in child_idxs:
            if 0 <= child_idx < n_models:
                parent_child_pairs.add((parent_idx, child_idx))

    "---- Siblings: prefer provided adjacency if present; else infer robustly ----"
    "1) If provided by adjacency builder"
    sib_key_candidates = ('siblings_of', 'sibling_of', 'siblings', 'sibling')
    provided_sib_key = next((k for k in sib_key_candidates if k in adj_lists), None)

    sibling_pairs: set[tuple[int, int]] = set()
    if provided_sib_key is not None:
        for model_index, neighbor_indices in enumerate(adj_lists[provided_sib_key]):
            for neighbor_index in neighbor_indices:
                if 0 <= neighbor_index < n_models and model_index != neighbor_index:
                    "store undirected as (min,max)"
                    sibling_pairs.add((min(model_index, neighbor_index), max(model_index, neighbor_index)))
    else:
        "2) Infer siblings:"
        "Convert settings to tuples of 0/1 flags once, and compute k & family tags."
        settings_tuples: List[tuple[int, ...]] = [
            convert_utility_settings(utility_settings=settings, into=tuple) for settings in settings_list
        ]

        def count_k_params(settings: Dict[str, bool]) -> int:
            return count_free_parameters(utility_settings=settings)

        def family_tag(settings: Dict[str, bool]) -> str:
            if settings.get('conditional_welfare_mode', False):
                return "cw"
            if settings.get('min_max_rawlsian_leontief', False):
                return "mm"
            return "base"

        model_k_counts   = [count_k_params(settings) for settings in settings_list]
        model_family_tags = [family_tag(settings) for settings in settings_list]

        def _hamming(tuple_a: tuple[int, ...], tuple_b: tuple[int, ...]) -> int:
            return sum(1 for a, b in zip(tuple_a, tuple_b) if a != b)

        for model_index in range(n_models):
            for other_model_index in range(model_index + 1, n_models):
                "siblings: same k, same family, and differ by exactly one boolean flag"
                if (model_k_counts[model_index] == model_k_counts[other_model_index]) \
                   and (model_family_tags[model_index] == model_family_tags[other_model_index]) \
                   and (_hamming(settings_tuples[model_index], settings_tuples[other_model_index]) == 1):
                    sibling_pairs.add((model_index, other_model_index))

    out = {
        'n_models': n_models,
        'n_parent_child_pairs': len(parent_child_pairs),
        'n_sibling_pairs': len(sibling_pairs),
    }

    if print_:
        print(f"Models: {out['n_models']}")
        print(f"Unique parent–child pairs: {out['n_parent_child_pairs']}")
        print(f"Unique sibling pairs: {out['n_sibling_pairs']}")

    return out


def test_utility_functions(build_utility_equation: Callable, general_settings: GeneralSettings, utility_settings: UtilitySettings, setting_to_flip: str, print_: bool = True) -> None:
    """
    Used to test if the rules for generating utility functions make sense and if build_utility_equation follows those rules. 
    """
    if setting_to_flip not in utility_settings:
        raise ValueError(f"{setting_to_flip} not in {list(utility_settings.keys())}.")
    
    errored = False
    valid = False
    while not valid:
        utility_settings = copy.deepcopy(utility_settings)
        setting_keys = list(utility_settings.keys())
        for idx in range(len(setting_keys)):
            utility_settings[setting_keys[idx]] = random.choice([True, False])
        valid = is_valid_utility_settings(candidate=utility_settings)

    utility_settings_m1 = copy.deepcopy(utility_settings)
    utility_settings_m2 = copy.deepcopy(utility_settings_m1)
    utility_settings_m2[setting_to_flip] = not utility_settings_m1[setting_to_flip]

    if print_: 
        print("")
        for setting_key, setting_val in utility_settings.items():
            setting_key += " " * (30 - len(setting_key))
            print(f"{setting_key}: {setting_val},")    

    equation_m1 = build_utility_equation(utility_settings=utility_settings_m1)
    if print_: print(f"M1: {equation_m1}")

    if not is_valid_utility_settings(candidate=utility_settings_m1):
        if print_: 
            print(f"Invalid utility settings detected M1!")
            print(is_valid_utility_settings(candidate=utility_settings_m1, provide_explanation=True))
        errored = True

    else:
        equation_m2 = build_utility_equation(utility_settings=utility_settings_m2)
        if print_: print(f"M2: {equation_m2}")

        if not is_valid_utility_settings(candidate=utility_settings_m2):
            if print_: 
                print(f"Invalid utility settings detected M2!")
                print(is_valid_utility_settings(candidate=utility_settings_m2, provide_explanation=True))
            errored = True

    relations = classify_pair_relation(model_1=utility_settings_m1, model_2=utility_settings_m2, 
                                            utility_settings=utility_settings, general_settings=general_settings, print_=False)
    if print_: print(relations)

    return (errored, relations)


"=========================================================================================="
"=============================== Pretty Equation Evaluators ==============================="
"=========================================================================================="

import re as _re_eval
import math as _math_eval

"Map subscript characters to plain gamma parameter names for downstream substitution."
_SUB_TO_GAMMA = {"₁": "γ1", "₂": "γ2", "₃": "γ3"}

"Regex fragment matching an exponent suffix: ^γ₁, ^γ₂, ^γ₃, or a numeric literal."
_SC_EXPONENT_TAG = r"\^[^\s)]+?"

"Matches a social comparison term written in grouped form: -Ƹᵢⱼ × (max(envy,0)^p - max(guilt,0)^q)."
_SC_GROUPED = _re_eval.compile(
    rf"""
    -\s*Ƹᵢⱼ\s*[×*]\s*\(\s*
    max\((?P<envy>[^)]*?)\s*,\s*0\)\s*(?P<envy_exponent>{_SC_EXPONENT_TAG})?
    \s*-\s*
    max\((?P<guilt>[^)]*?)\s*,\s*0\)\s*(?P<guilt_exponent>{_SC_EXPONENT_TAG})?
    \s*\)
    """, _re_eval.VERBOSE
)

"Matches a social comparison term written in two-term form: -Ƹᵢⱼ × max(envy,0)^p + Ƹᵢⱼ × max(guilt,0)^q."
_SC_TWOTERM = _re_eval.compile(
    rf"""
    -\s*Ƹᵢⱼ\s*[×*]\s*max\((?P<envy>[^)]*?)\s*,\s*0\)\s*(?P<envy_exponent>{_SC_EXPONENT_TAG})?
    \s*\+\s*Ƹᵢⱼ\s*[×*]\s*max\((?P<guilt>[^)]*?)\s*,\s*0\)\s*(?P<guilt_exponent>{_SC_EXPONENT_TAG})?
    """, _re_eval.VERBOSE
)

def signed_pow(value: float, exponent: float) -> float:
    """
    Sign-preserving power function: returns sign(value) × |value|^exponent.

    Arguments:
        • value: float — the base; its sign is preserved through the exponentiation.
        • exponent: float — the power to raise |value| to.

    Returns:
        • float — sign(value) × |value|^exponent, or 0.0 if value is zero.
    """
    value    = float(value)
    exponent = float(exponent)
    if value == 0.0:
        return 0.0
    return _math_eval.copysign(abs(value) ** exponent, value)

def _canon_sc_grouped_to_twoterm(equation_rhs: str) -> str:
    """
    Rewrites any grouped social comparison term in equation_rhs into two-term form.

    Grouped form:   - Ƹᵢⱼ × (max(envy, 0)^p - max(guilt, 0)^q)
    Two-term form:  - Ƹᵢⱼ × max(envy, 0)^p + Ƹᵢⱼ × max(guilt, 0)^q

    Arguments:
        • equation_rhs: str — right-hand side of a pretty-printed utility equation.

    Returns:
        • str — equation_rhs with grouped SC terms replaced by two-term equivalents.
    """
    def _repl(match: _re_eval.Match) -> str:
        envy_expression  = match.group("envy").strip()
        guilt_expression = match.group("guilt").strip()
        envy_exponent    = match.group("envy_exponent") or ""
        guilt_exponent   = match.group("guilt_exponent") or envy_exponent
        return (
            f"- Ƹᵢⱼ × max({envy_expression}, 0){envy_exponent}"
            f" + Ƹᵢⱼ × max({guilt_expression}, 0){guilt_exponent}"
        )
    return _SC_GROUPED.sub(_repl, equation_rhs)


def _canon_sc_twoterm_to_grouped(equation_rhs: str) -> str:
    """
    Rewrites any two-term social comparison in equation_rhs into grouped form.

    Two-term form:  - Ƹᵢⱼ × max(envy, 0)^p + Ƹᵢⱼ × max(guilt, 0)^q
    Grouped form:   - Ƹᵢⱼ × (max(envy, 0)^p - max(guilt, 0)^q)

    Arguments:
        • equation_rhs: str — right-hand side of a pretty-printed utility equation.

    Returns:
        • str — equation_rhs with two-term SC patterns replaced by grouped equivalents.
    """
    def _repl(match: _re_eval.Match) -> str:
        envy_expression  = match.group("envy").strip()
        guilt_expression = match.group("guilt").strip()
        envy_exponent    = match.group("envy_exponent") or ""
        guilt_exponent   = match.group("guilt_exponent") or envy_exponent
        return (
            f"- Ƹᵢⱼ × (max({envy_expression}, 0){envy_exponent}"
            f" - max({guilt_expression}, 0){guilt_exponent})"
        )
    return _SC_TWOTERM.sub(_repl, equation_rhs)


def canon_sc_both_ways(equation_rhs: str, mode: str = "twoterm") -> str:
    """
    Normalizes symmetric social comparison terms to a canonical target style.

    Applies both conversions in sequence (grouped → two-term → grouped) to reach a
    stable form regardless of which style the input was written in, then enforces
    the requested target style.

    Arguments:
        • equation_rhs: str — right-hand side of a pretty-printed utility equation.
        • mode: str — target style; either "twoterm" (default) or "grouped".

    Returns:
        • str — equation_rhs with all SC terms written in the requested style.
    """
    twoterm_form = _canon_sc_grouped_to_twoterm(equation_rhs)
    grouped_form = _canon_sc_twoterm_to_grouped(twoterm_form)
    if mode == "twoterm":
        return _canon_sc_grouped_to_twoterm(grouped_form)
    return grouped_form

def _is_token_char(character: str) -> bool:
    """
    Returns True if character can appear inside an operator token (variable name, number, etc.).

    Arguments:
        • character: str — a single character from an expression string.

    Returns:
        • bool — True if the character is not a delimiter (space, comma, operator, parenthesis).
    """
    return character not in " \t\r\n,^*/+-()"

def _find_left_operand(expression: str, caret_index: int) -> tuple[int, int]:
    """
    Locates the left operand of the ^ operator at caret_index in expression.

    Handles three cases: a bare token (number or variable name), a parenthesised
    sub-expression, and a named function call such as max(...) or pow_signed(...).

    Arguments:
        • expression: str — the full expression string being parsed.
        • caret_index: int — index of the '^' character in expression.

    Returns:
        • tuple[int, int] — (start, end) character indices of the left operand,
            such that expression[start:end] is the operand text.
    """
    position = caret_index - 1
    while position >= 0 and expression[position].isspace():
        position -= 1

    if position >= 0 and expression[position] == ")":
        paren_depth = 1
        position -= 1
        while position >= 0 and paren_depth > 0:
            if expression[position] == ")":
                paren_depth += 1
            elif expression[position] == "(":
                paren_depth -= 1
            position -= 1
        operand_start = position + 1
        operand_end   = caret_index
        function_name_end   = operand_start
        function_name_start = function_name_end - 1
        while function_name_start >= 0 and expression[function_name_start].isalpha():
            function_name_start -= 1
        function_name_start += 1
        if function_name_start < function_name_end and expression[function_name_end] == "(":
            operand_start = function_name_start
        return operand_start, operand_end

    operand_end   = position + 1
    operand_start = position
    while operand_start >= 0 and _is_token_char(expression[operand_start]):
        operand_start -= 1
    operand_start += 1
    return operand_start, operand_end

def _find_right_operand(expression: str, caret_index: int) -> tuple[int, int]:
    """
    Locates the right operand of the ^ operator at caret_index in expression.

    Handles parenthesised sub-expressions and bare tokens (numbers, variable names).

    Arguments:
        • expression: str — the full expression string being parsed.
        • caret_index: int — index of the '^' character in expression.

    Returns:
        • tuple[int, int] — (start, end) character indices of the right operand,
            such that expression[start:end] is the operand text.
    """
    position    = caret_index + 1
    expr_length = len(expression)
    while position < expr_length and expression[position].isspace():
        position += 1

    if position < expr_length and expression[position] == "(":
        paren_depth = 1
        position += 1
        while position < expr_length and paren_depth > 0:
            if expression[position] == "(":
                paren_depth += 1
            elif expression[position] == ")":
                paren_depth -= 1
            position += 1
        return caret_index + 1, position

    operand_start = position
    while position < expr_length and _is_token_char(expression[position]):
        position += 1
    return operand_start, position

def _replace_powers(expression: str) -> str:
    """
    Replaces all ^ exponentiation operators in expression with pow_signed(...) calls.

    Uses parenthesis-aware operand detection so that compound bases like max(a, b)^γ
    and multi-character exponents are captured correctly.

    Arguments:
        • expression: str — a normalised equation string that may contain ^ operators.

    Returns:
        • str — expression with every a^b replaced by pow_signed(a, b).
    """
    result = expression
    while "^" in result:
        caret_position             = result.find("^")
        left_start,  left_end      = _find_left_operand(result, caret_position)
        right_start, right_end     = _find_right_operand(result, caret_position)
        base_expression            = result[left_start:left_end].strip()
        exponent_expression        = result[right_start:right_end].strip()
        result = (
            result[:left_start]
            + f"pow_signed({base_expression}, {exponent_expression})"
            + result[right_end:]
        )
    return result

def normalize_pretty_rhs_for_eval(rhs_text: str, sc_mode: str = "twoterm") -> str:
    """
    Converts a pretty-printed equation right-hand side into a Python-evaluable string.

    Applies Unicode normalisation, implicit-multiplication insertion, social comparison
    canonicalisation, and ^ → pow_signed(...) substitution in sequence.

    Arguments:
        • rhs_text: str — right-hand side of a pretty-printed utility equation, possibly
            containing Unicode characters, implicit multiplication, and ^ exponents.
        • sc_mode: str — target style for social comparison terms; "twoterm" (default)
            or "grouped". Passed through to canon_sc_both_ways.

    Returns:
        • str — a Python-evaluable expression string using only ASCII operators and
            the pow_signed, max, min, and abs functions.
    """
    "Normalise Unicode dashes and multiplication symbols to ASCII equivalents."
    norm = (rhs_text.replace("\u00A0", " ")
                    .replace("−", "-").replace("–", "-").replace("—", "-")
                    .replace("≥", ">=").replace("≤", "<=").replace("≠", "!=")
                    .replace("×", "*").replace("·", "*").replace("⋅", "*"))
    "Replace square brackets with parentheses."
    norm = norm.replace("[", "(").replace("]", ")")
    "Insert explicit multiplication where a numeric literal immediately precedes an open paren."
    norm = _re_eval.sub(r"(?<![A-Za-z0-9_])(\-?\d+(?:\.\d+)?)\s*\(", r"\1*(", norm)
    norm = norm.replace(")(", ")*(")
    norm = _re_eval.sub(r"\)\s*(\-?\d+(?:\.\d+)?)", r")*\1", norm)
    "Insert explicit multiplication where a function name is stuck to a number or closing paren."
    norm = _re_eval.sub(r"(\d)\s*(max|min)\s*\(", r"\1*\2(", norm)
    norm = _re_eval.sub(r"(\d)(?=(max|min)\()", r"\1*", norm)
    norm = _re_eval.sub(r"\)(?=(max|min)\()", r")*", norm)

    "Canonicalise social comparison terms to a stable target style."
    norm = canon_sc_both_ways(norm, mode=sc_mode)

    "Replace ^ operators with pow_signed(...) calls, respecting parenthesised bases."
    out = _replace_powers(norm)

    "Insert explicit multiplication around any pow_signed calls stuck to numbers or parens."
    out = _re_eval.sub(r"(\d)\s*(pow_signed)\s*\(", r"\1*\2(", out)
    out = _re_eval.sub(r"\)\s*(pow_signed)\s*\(", r")*\1(", out)
    out = _re_eval.sub(r"(\d)(?=(pow_signed)\()", r"\1*", out)
    out = _re_eval.sub(r"\)(?=(pow_signed)\()", r")*", out)
    return out

def eval_pretty_equation_rhs(
    rhs_filled: str,
    decimals: int = 6,
    sc_mode: str = "twoterm",
) -> tuple[float | None, str]:
    """
    Evaluates a pretty-printed, parameter-filled equation right-hand side numerically.

    Normalises the string via normalize_pretty_rhs_for_eval, then evaluates it in a
    restricted namespace containing only max, min, abs, and signed_pow. No builtins
    are exposed to prevent arbitrary code execution.

    Arguments:
        • rhs_filled: str — right-hand side of a utility equation with all parameter
            symbols replaced by their numeric values.
        • decimals: int — number of decimal places to round the result to (default 6).
        • sc_mode: str — social comparison canonicalisation style; passed through to
            normalize_pretty_rhs_for_eval (default "twoterm").

    Returns:
        • tuple[float | None, str] — (value, status) where value is the rounded result
            or None on failure, and status is an empty string on success or an error
            message beginning with "EVAL ERROR:" on failure.
    """
    python_expression = normalize_pretty_rhs_for_eval(rhs_filled, sc_mode=sc_mode)
    safe_namespace = {
        "__builtins__": {},
        "max": max, "min": min, "abs": abs,
        "pow_signed": signed_pow, "signed_pow": signed_pow,
    }
    try:
        result_value = float(eval(python_expression, safe_namespace, {}))
        return round(result_value, decimals), ""
    except Exception as error:
        return None, f"EVAL ERROR: {type(error).__name__}: {error}"


