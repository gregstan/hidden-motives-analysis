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
    if parameter_key == "temp":
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

    # ---- Helpers --------------------------------------------------------------
    def parameter_value(param_dict: dict, param_key: str) -> float:
        param_val = param_dict.get(param_key, 0.0)
        return float(param_val) if isinstance(param_val, (int, float)) else 0.0

    def _has(param_dict: dict, *param_keys: str) -> bool:
        return any(param_key in param_dict for param_key in param_keys)

    penalty = 0.0

    # 1) Means (self, altruism, social): pairwise mean-abs with appropriate anchors

    # --- Self-interest (anchor = 1.0) -----------------------------------------
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

    # --- Altruism (anchor = 0.0) ----------------------------------------------
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

    # --- Social comparison (anchor = 0.0) -------------------------------------
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

    # 2) Exponents (optional): penalize deviation of the MEAN gamma from 1
    if penalize_exponents:
        gammas = []
        for param_key in params:
            if param_key.startswith("γ") or param_key.lower().startswith("gamma"):
                val = parameter_value(params, param_key)
                # keep zeros too, but this avoids weird non-numeric
                gammas.append(val)
        if gammas:
            mean_gamma = sum(gammas) / len(gammas)
            penalty += len(gammas) * (mean_gamma - 1.0) ** 2  # parent-fair

    # 3) Standard deviations (optional)
    if penalize_std:
        for param_key, param_val in params.items():
            if isinstance(param_val, (int, float)) and isinstance(param_key, str) and param_key.endswith("_std"):
                param_val = float(param_val)
                floor = 1e-3
                if param_val < floor:
                    penalty += 1e6 * (floor - param_val)  # hard push away from zero
                else:
                    penalty += (param_val * param_val) / 10.0

    # 4) Covariances (optional)
    if penalize_cov:
        for param_key, param_val in params.items():
            if isinstance(param_val, (int, float)) and isinstance(param_key, str) and param_key.endswith("_cov"):
                penalty += float(param_val) ** 2

    # 5) Temperature (optional)
    if penalize_temp:
        temperature = parameter_value(params, "temp")
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
    # Extract the names for the parameters (exclude any keys ending with '_std')
    parameter_mean_names = [param for param in param_info["keys"] if not param.endswith('_std')]
    num_parameters: int = len(parameter_mean_names)

    # Ensure that the joint PMF is normalized; otherwise, normalize it.
    total_probability_mass: float = float(np.sum(joint_pmf)) #type: ignore   Type of "sum" is partially unknown
    if total_probability_mass <= 0 or np.isnan(total_probability_mass):
        if print_warnings:
            print(f"Warning: Invalid sum {total_probability_mass}. Normalizing.")
        total_probability_mass = 1.0

    normalized_joint_pmf = joint_pmf / total_probability_mass
    # Each coordinate array has the same shape as the joint PMF.
    meshgrid_coordinate_arrays: List[NDArray[np.float64]] = np.meshgrid(*grids, indexing='ij')

    # Initialize lists to store computed means and standard deviations for each parameter.
    computed_means: List[float] = [0.0] * num_parameters
    computed_standard_deviations: List[float] = [0.0] * num_parameters

    # Compute marginal means and variances for each parameter dimension.
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

    # Compute pairwise covariances between parameters.
    computed_covariances: Dict[str, float] = {}
    for idx in range(num_parameters):
        for jdx in range(idx + 1, num_parameters):
            deviation_product = (meshgrid_coordinate_arrays[idx] - computed_means[idx]) * (meshgrid_coordinate_arrays[jdx] - computed_means[jdx])
            covariance_value: float = float(np.sum(deviation_product * normalized_joint_pmf)) #type: ignore
            covariance_key = f"{parameter_mean_names[idx]}_{parameter_mean_names[jdx]}_cov"
            computed_covariances[covariance_key] = covariance_value

    # Build the final statistics dictionary.
    computed_statistics: Dict[str, float] = {}
    # Add the means.
    computed_statistics.update({param_name: computed_means[idx] for idx, param_name in enumerate(parameter_mean_names)})
    # Add the standard deviations with key format '<parameter>_std'
    computed_statistics.update({f"{param_name}_std": computed_standard_deviations[idx] for idx, param_name in enumerate(parameter_mean_names)})
    # Add the pairwise covariances.
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
        k for k in param_info['keys']
        if (not k.endswith('_std')) and (not k.endswith('_cov'))
    ]

    total_mass: float = 0.0
    sum_x:  dict[str, float] = {k: 0.0 for k in mean_param_keys}
    sum_x2: dict[str, float] = {k: 0.0 for k in mean_param_keys}

    for index_tuple, mass in param_vectors.items():
        if mass is None or mass <= 0.0:
            continue
        total_mass += float(mass)
        for dim, param_key in enumerate(mean_param_keys):
            x_val = float(tickvals[param_key][index_tuple[dim]])
            sum_x[param_key]  += mass * x_val
            sum_x2[param_key] += mass * (x_val * x_val)

    if total_mass <= 0.0:
        # Degenerate; return zeros to be safe (should not happen if normalized)
        out_zero = {k: 0.0 for k in mean_param_keys}
        out_zero.update({f"{k}_std": 0.0 for k in mean_param_keys})
        return out_zero

    out: dict[str, float] = {}
    for param_key in mean_param_keys:
        mu  = sum_x[param_key] / total_mass
        ex2 = sum_x2[param_key] / total_mass
        var = max(0.0, ex2 - mu * mu)
        out[param_key] = float(mu)
        out[f"{param_key}_std"] = float(var ** 0.5)

    return out


def is_positive_semidefinite(matrix: NDArray[np.float64], tol: float = 1e-12) -> bool:
    """Check if a matrix is positive semidefinite."""
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        eigenvalues: NDArray[np.float64] = np.linalg.eigvalsh(matrix)
        return bool(np.all(eigenvalues >= -tol))  # Ensure boolean return


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
                        if not any(key in param_key for key in ('_std', '_cov', 'temp'))]
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
        # if candidate['include_social_comparison']:
        #     if candidate['single_payoffs_not_differences']:
        #         explanation += " and that is a social comparison term, then it must apply to payoff differences or ratios."
        #         return explanation if provide_explanation else False
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


def generate_utility_settings(utility_settings: dict[str, bool], sort_by_k: bool = False) -> List[UtilitySettings]:
    """
    Generates all valid combinations of utility function settings, subject to the following constraints:
        • If single_payoffs_not_differences is False, then reference_dependent_utility must be False (irrelevant).
        • If use_exponential_parameters is False, then single_exponential_parameter must be False (irrelevant).
        • If include_social_comparison is False, then negativity_social_comparison must be False (irrelevant).
        • If use_negativity parameters is True, then negativity_social_comparison must be True (irrelevant).
        • All other boolean flags vary independently.

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

    Returns:
        • A list of dict[ str, bool ], where each dict is a valid set of utility options.
    """
    "All booleans that can vary in principle"
    bool_flags = {
        key: [False, True] for key in utility_settings.keys()
    }

    valid_combos = []

    "Exhaustively generate all combinations"
    all_keys = sorted(bool_flags.keys())
    all_value_combos = it.product(*(bool_flags[k] for k in all_keys))

    for combo in all_value_combos:
        candidate = dict(zip(all_keys, combo))

        if is_valid_utility_settings(candidate=candidate):
            "If it passes all constraints, store it"
            valid_combos.append(candidate)

    if sort_by_k:
        settings_to_k = []
        for combo in valid_combos:
            param_keys = parameter_keys_for_utility_settings(utility_settings=combo, general_settings=None)
            settings_to_k.append((combo, len(param_keys)))

        sorted_settings_to_k = sorted(settings_to_k, key=lambda x: x[-1])
        valid_combos = [setting_to_k[0] for setting_to_k in sorted_settings_to_k]

    return valid_combos


def identify_redundant_utility_functions(
        utility_settings: UtilitySettings,
        build_equation_function: callable,
        file_paths: dict[str, str]
    ) -> pd.DataFrame:
    """
    Finds redundant utility functions (identical equations) and reports which settings
    cause the redundancy.

    Procedure:
        1) Generate all valid utility_settings via generate_utility_settings(…).
        2) Render each as a pretty equation via build_equation_function(…).
        3) For each equation, compute:
            • equation_count: number of occurrences
            • redundant_with: tuple of utility_idx sharing this equation (sorted)
            • differing_settings: tuple of setting names that differ within this group (sorted)
        4) Write a CSV with one row per utility function, with the utility flag columns
           in canonical order, 'equation' as the rightmost column, and rows grouped so
           redundant equations are adjacent.

    Arguments:
        • utility_settings: dict[str, bool] | tuple[bool]
            A canonical utility_settings structure used for:
            - seeding generate_utility_settings to obtain the full set
            - deriving canonical flag order (list(utility_settings.keys()))
        • build_equation_function: callable
            Function that takes a dict[str,bool] and returns the pretty equation string.
        • file_paths: dict[str, str]
            Must contain key "bic_aic" for the output CSV location.

    Returns:
        • pd.DataFrame; columns:
            ['utility_idx', <flags… in canonical order>, 'equation_count',
             'redundant_with', 'differing_settings', 'equation']
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

    "Console feedback"
    n_total = len(df)
    n_unique = df["equation"].nunique()
    if n_unique == n_total:
        print(f"All {n_total} utility functions are unique. No redundancies found.")
    else:
        n_redundant = n_total - n_unique
        n_groups_gt1 = int((df["equation_count"] > 1).sum())
        print(f"Found {n_redundant} redundant utility functions across {n_groups_gt1} duplicated groups.")

    return df


def count_free_parameters(
    utility_settings: UtilitySettings,
    general_settings: Optional[Dict[str, Any]] = None
) -> int:
    """
    Convenience wrapper that returns the number of *free mean* parameters by default.
    Covariance params are intentionally excluded here (IC counting often focuses on means).
    """
    keys = parameter_keys_for_utility_settings(
        utility_settings=utility_settings,
        general_settings=general_settings,
    )
    "Strips covariance keys if present; this helper does not add them."
    return len([key for key in keys if not key.endswith('_cov')])


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
    if utility_settings['use_negativity_parameters']:
        utility_settings['negativity_social_comparison'] = True

    return utility_settings


def parents_children_of(
    utility_settings: Union[UtilitySettings, BoolTuple],
    return_children: bool = True,
    return_parents: bool = True,
    general_settings: Optional[Dict[str, Any]] = None
) -> Dict[str, Optional[List[BoolTuple]]]:
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
    # pp.pprint(base)
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
        # random_guesses_are_unique=not general_settings.get('run_in_parallel', True),  # not used for final guesses TODO delete???
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
        for i, nbrs in enumerate(adj_lists[provided_sib_key]):
            for j in nbrs:
                if 0 <= j < n_models and i != j:
                    "store undirected as (min,max)"
                    sibling_pairs.add((min(i, j), max(i, j)))
    else:
        "2) Infer siblings:"
        "Convert settings to tuples of 0/1 flags once, and compute k & family tags."
        settings_tuples: List[tuple[int, ...]] = [
            convert_utility_settings(utility_settings=s, into=tuple) for s in settings_list
        ]

        def k_params(s: Dict[str, bool]) -> int:
            return count_free_parameters(utility_settings=s)

        def family_tag(s: Dict[str, bool]) -> str:
            if s.get('conditional_welfare_mode', False):
                return "cw"
            if s.get('min_max_rawlsian_leontief', False):
                return "mm"
            return "base"

        k_list = [k_params(s) for s in settings_list]
        fam_list = [family_tag(s) for s in settings_list]

        def _hamming(a: tuple[int, ...], b: tuple[int, ...]) -> int:
            return sum(1 for x, y in zip(a, b) if x != y)

        for i in range(n_models):
            for j in range(i + 1, n_models):
                "siblings: same k, same family, and differ by exactly one boolean flag"
                if (k_list[i] == k_list[j]) and (fam_list[i] == fam_list[j]) \
                   and (_hamming(settings_tuples[i], settings_tuples[j]) == 1):
                    sibling_pairs.add((i, j))

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

"Map subscript to plain gamma names if needed downstream"
_SUB_TO_GAMMA = {"₁": "γ1", "₂": "γ2", "₃": "γ3"}
_SC_EXP_TAG   = r"\^[^\s)]+?"  # matches ^γ₁/^γ₂/^γ₃ or numeric after substitution
_SC_GROUPED = _re_eval.compile(
    rf"""
    -\s*Ƹᵢⱼ\s*[×*]\s*\(\s*
    max\((?P<envy>[^)]*?)\s*,\s*0\)\s*(?P<p>{_SC_EXP_TAG})?
    \s*-\s*
    max\((?P<guilt>[^)]*?)\s*,\s*0\)\s*(?P<q>{_SC_EXP_TAG})?
    \s*\)
    """, _re_eval.VERBOSE
)
_SC_TWOTERM = _re_eval.compile(
    rf"""
    -\s*Ƹᵢⱼ\s*[×*]\s*max\((?P<envy>[^)]*?)\s*,\s*0\)\s*(?P<p>{_SC_EXP_TAG})?
    \s*\+\s*Ƹᵢⱼ\s*[×*]\s*max\((?P<guilt>[^)]*?)\s*,\s*0\)\s*(?P<q>{_SC_EXP_TAG})?
    """, _re_eval.VERBOSE
)

def signed_pow(_x: float, _gamma: float) -> float:
    """
    Sign-preserving power: returns sign(x) * |x|^gamma.
    Used to align string evaluation with utility() semantics for non-integer exponents.
    """
    _x = float(_x); _gamma = float(_gamma)
    if _x == 0.0:
        return 0.0
    return _math_eval.copysign(abs(_x) ** _gamma, _x)

def _canon_sc_grouped_to_twoterm(rhs: str) -> str:
    def _repl(m: _re_eval.Match) -> str:
        envy  = m.group("envy").strip()
        guilt = m.group("guilt").strip()
        p     = m.group("p") or ""
        q     = m.group("q") or p
        return f"- Ƹᵢⱼ × max({envy}, 0){p} + Ƹᵢⱼ × max({guilt}, 0){q}"
    return _SC_GROUPED.sub(_repl, rhs)

def _canon_sc_twoterm_to_grouped(rhs: str) -> str:
    def _repl(m: _re_eval.Match) -> str:
        envy  = m.group("envy").strip()
        guilt = m.group("guilt").strip()
        p     = m.group("p") or ""
        q     = m.group("q") or p
        return f"- Ƹᵢⱼ × (max({envy}, 0){p} - max({guilt}, 0){q})"
    return _SC_TWOTERM.sub(_repl, rhs)

def canon_sc_both_ways(rhs: str, mode: str = "twoterm") -> str:
    """
    Canonicalize symmetric SC either way, then enforce a target style.
    mode="twoterm":  - Ƹ × max(envy)^p + Ƹ × max(guilt)^q  (default)
    mode="grouped":  - Ƹ × (max(envy)^p - max(guilt)^q)
    """
    rhs1 = _canon_sc_grouped_to_twoterm(rhs)
    rhs2 = _canon_sc_twoterm_to_grouped(rhs1)
    if mode == "twoterm":
        return _canon_sc_grouped_to_twoterm(rhs2)
    else:
        return rhs2

def _is_token_char(ch: str) -> bool:
    return ch not in " \t\r\n,^*/+-()"

def _find_left_operand(expr: str, caret_index: int) -> tuple[int, int]:
    i = caret_index - 1
    while i >= 0 and expr[i].isspace():
        i -= 1
    if i >= 0 and expr[i] == ")":
        depth = 1; i -= 1
        while i >= 0 and depth > 0:
            if expr[i] == ")": depth += 1
            elif expr[i] == "(": depth -= 1
            i -= 1
        start = i + 1; end = caret_index
        name_end = start; name_start = name_end - 1
        while name_start >= 0 and expr[name_start].isalpha():
            name_start -= 1
        name_start += 1
        if name_start < name_end and expr[name_end] == "(":
            start = name_start
        return start, end
    end = i + 1; start = i
    while start >= 0 and _is_token_char(expr[start]):
        start -= 1
    start += 1
    return start, end

def _find_right_operand(expr: str, caret_index: int) -> tuple[int, int]:
    i = caret_index + 1; n = len(expr)
    while i < n and expr[i].isspace():
        i += 1
    if i < n and expr[i] == "(":
        depth = 1; i += 1
        while i < n and depth > 0:
            if expr[i] == "(": depth += 1
            elif expr[i] == ")": depth -= 1
            i += 1
        return caret_index + 1, i
    start = i
    while i < n and _is_token_char(expr[i]):
        i += 1
    return start, i

def _replace_powers(expr: str) -> str:
    out = expr
    while "^" in out:
        caret = out.find("^")
        ls, le = _find_left_operand(out, caret)
        rs, re = _find_right_operand(out, caret)
        base = out[ls:le].strip()
        exp  = out[rs:re].strip()
        out  = out[:ls] + f"pow_signed({base}, {exp})" + out[re:]
    return out

def normalize_pretty_rhs_for_eval(rhs_text: str, sc_mode: str = "twoterm") -> str:
    """
    Unicode/operator cleanup, canonicalize symmetric SC (both ways), insert implicit '*',
    and convert '^' with parenthesis-aware capture to pow_signed(...).
    """
    "unicode & ops"
    norm = (rhs_text.replace("\u00A0", " ")
                    .replace("−", "-").replace("–", "-").replace("—", "-")
                    .replace("≥", ">=").replace("≤", "<=").replace("≠", "!=")
                    .replace("×", "*").replace("·", "*").replace("⋅", "*"))
    "brackets→parens"
    norm = norm.replace("[", "(").replace("]", ")")
    "implicit multiplication (numbers against '(')"
    norm = _re_eval.sub(r"(?<![A-Za-z0-9_])(\-?\d+(?:\.\d+)?)\s*\(", r"\1*(", norm)
    norm = norm.replace(")(", ")*(")
    norm = _re_eval.sub(r"\)\s*(\-?\d+(?:\.\d+)?)", r")*\1", norm)
    "function calls stuck to numbers or ')'"
    norm = _re_eval.sub(r"(\d)\s*(max|min)\s*\(", r"\1*\2(", norm)
    norm = _re_eval.sub(r"(\d)(?=(max|min)\()", r"\1*", norm)
    norm = _re_eval.sub(r"\)(?=(max|min)\()", r")*", norm)

    "symmetric SC canonicalization both ways; choose stable target"
    norm = canon_sc_both_ways(norm, mode=sc_mode)

    "'^' replacement (captures bases incl. function calls)"
    out = _replace_powers(norm)

    "ensure pow_signed is multiplied when stuck to numbers or ')'"
    out = _re_eval.sub(r"(\d)\s*(pow_signed)\s*\(", r"\1*\2(", out)
    out = _re_eval.sub(r"\)\s*(pow_signed)\s*\(", r")*\1(", out)
    out = _re_eval.sub(r"(\d)(?=(pow_signed)\()", r"\1*", out)
    out = _re_eval.sub(r"\)(?=(pow_signed)\()", r")*", out)
    return out

def eval_pretty_equation_rhs(rhs_filled: str, decimals: int = 6, sc_mode: str = "twoterm") -> tuple[float | None, str]:
    """
    Converts a filled RHS to Python and evaluates it. Returns (value, status).
    """
    python_rhs = normalize_pretty_rhs_for_eval(rhs_filled, sc_mode=sc_mode)
    safe_env = {"__builtins__": {}, "max": max, "min": min, "abs": abs,
                "pow_signed": signed_pow, "signed_pow": signed_pow}
    try:
        value = float(eval(python_rhs, safe_env, {}))
        return round(value, decimals), ""
    except Exception as err:
        return None, f"EVAL ERROR: {type(err).__name__}: {err}"


