from optimization import *

"=========================================================================================="
"===================================== Bayesian Code ======================================"
"=========================================================================================="

def bayesian_update_parametric(old_means: Dict[str, float], old_stds: Dict[str, float], observed_choice: str, game_dict: DyadGame, 
                               choice_func: callable, utility_settings: UtilitySettings, learning_rate: float = 0.4, shrink_std: bool = True, 
                               shrink_factor: float = 0.02, epsilon: float = 1e-4) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Perform a Bayesian update on social preference parameters using numeric gradients of log-likelihood.

    This function updates the means and standard deviations for each parameter in a lightweight 
    parametric Bayesian framework. Gradients of the log-likelihood are calculated for the observed 
    choice ("A" or "B"), and parameters are adjusted based on these gradients. Optionally shrinks 
    standard deviations to reflect reduced uncertainty as new data is incorporated.

    Arguments:
        • old_means: dict
            Dictionary of current parameter means, where keys represent parameters 
            (e.g., 'Vᵢᵢ', 'Vᵢⱼ', 'αᵢⱼ', 'γ1').
        • old_stds: dict
            Dictionary of current parameter standard deviations, where keys are suffixed with '_std' 
            (e.g., 'Vᵢᵢ_std', 'Vᵢⱼ_std').
        • observed_choice: str
            The observed choice made by the player ('A' or 'B').
        • game_dict: dict
            Contains game details, including payoffs, player roles, and other context for the current round.
        • choice_func: callable
            Function that computes the probability of choosing "A" given a parameter dictionary. It 
            should return a dictionary with a key "model_choose_A" containing the probability.
        • utility_settings: UtilitySettings
            Configuration options for utility modeling, such as whether to include reference dependence, 
            negativity, or exponential parameters.
        • learning_rate: float, optional (default: 0.4)
            Step size for updating parameter means based on the gradient.
        • shrink_std: bool, optional (default: True)
            Whether to reduce parameter standard deviations as gradients become smaller, reflecting 
            reduced uncertainty over time.
        • shrink_factor: float, optional (default: 0.02)
            Scaling factor for reducing standard deviations.
        • epsilon: float, optional (default: 1e-4)
            Small value used for numeric gradient approximation.

    Returns:
        • Tuple[Dict[str, float], Dict[str, float]]
            - Updated means: A dictionary with the same structure as `old_means`.
            - Updated standard deviations: A dictionary with the same structure as `old_stds`.

    Notes:
        • This function calculates the log-likelihood of the observed choice based on the provided 
          utility model and parameter values.
        • Gradients are approximated numerically by perturbing each parameter in turn and computing the 
          difference in log-likelihood.
        • Negative parameter values (e.g., exponents) are avoided by enforcing lower bounds during updates.
        • This method is lightweight but assumes independent Gaussian distributions for each parameter.
    """
    "Create 'param_keys' for convenience"
    param_keys_means = [key for key in old_means if not key.endswith('_std')]

    def log_likelihood(param_dict_means: Dict[str,float]) -> float:
        "Merge param_dict_means + old_stds so choice_func ignores std keys"
        merged_params = {}
        for mean_param_key in param_keys_means:
            merged_params[mean_param_key] = param_dict_means[mean_param_key]
        "Keep the old std dev placeholders around:"
        for std_param_key in old_stds:
            merged_params[std_param_key] = old_stds[std_param_key]

        "Get p(A) from choice_func"
        choice_result = choice_func(
            current_game=game_dict,
            agent_params=merged_params,
            utility_settings=utility_settings,
            select_responses=False
        )
        probability_choose_A = choice_result['model_choose_A']
        if observed_choice == 'A':
            return math.log(max(probability_choose_A, 1e-12))
        else:
            return math.log(max(1 - probability_choose_A, 1e-12))

    new_means = dict(old_means)
    new_stds  = dict(old_stds)

    "For each dimension, numeric gradient:"
    for mean_param_key in param_keys_means:
        "Plus."
        plus_means  = copy.deepcopy(old_means)
        plus_means[mean_param_key] += epsilon
        ll_plus  = log_likelihood(plus_means)

        "Minus."
        minus_means = copy.deepcopy(old_means)
        minus_means[mean_param_key] -= epsilon
        ll_minus = log_likelihood(minus_means)

        grad_k = (ll_plus - ll_minus) / (2 * epsilon)

        "Scale by old sigma => bigger uncertainty => bigger move."
        old_sigma = old_stds.get(mean_param_key + '_std', 0.5)
        step_size = learning_rate * old_sigma

        new_means[mean_param_key] = old_means[mean_param_key] + step_size * grad_k

        if 'γ' in mean_param_key:
            "Prevent exponents from going negative"
            if new_means[mean_param_key] < 0.01:
                new_means[mean_param_key] = 0.01

        if shrink_std:
            "Naive shrink."
            shrink_amount = shrink_factor * abs(grad_k)
            new_sigma = old_sigma * max(0.0, 1.0 - shrink_amount)
            new_stds[mean_param_key + '_std'] = new_sigma
        else:
            new_stds[mean_param_key + '_std'] = old_sigma

    return new_means, new_stds


def bayesian_update_mcmc(old_means: Dict[str, float], old_stds: Dict[str, float], observed_choice: str, game_dict: DyadGame, 
                         choice_func: callable, utility_settings: UtilitySettings, param_info: ParamInfo, random_seed: int | None = None, 
                         chain_length: int = 300, burn_in: int = 50, thin: int = 1, proposal_sd: float = 0.35) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Approximate a posterior over the player's parameters after observing a single new choice,
    using a small Metropolis–Hastings MCMC chain. Returns (new_means, new_stds).
    
    NOTE: Depricated. Does not work. 

    Arguments:
    -----------
    old_means, old_stds: Dict[str, float]
        • Prior "mean" and "std" for each parameter (point estimate + uncertainty).
          Example keys:
              'Vᵢᵢ', 'Vᵢⱼ', 'αᵢⱼ', 'βᵢⱼ', 'γ1'  AND
              'Vᵢᵢ_std', 'Vᵢⱼ_std', 'αᵢⱼ_std', 'βᵢⱼ_std', 'γ1_std'
        • Interpreted here as defining a Normal prior for each parameter.

    observed_choice: str
        • The newly observed choice: 'A' or 'B'.

    game_dict: Dict
        • Contains payoffs, roles, etc., for the current round.

    choice_func: callable
        • A function that returns a dict with key 'model_choose_A' or 'model_predict_A' for probability of 'A'.

    utility_settings: UtilitySettings
        • Configuration for utility function (negativity, exponent, etc.).
    
    learning_rate, shrink_std, shrink_factor, epsilon:
        • Present purely for signature compatibility with bayesian_update_parametric().
        • Not used in typical MCMC usage. A fixed-proposal Metropolis-Hastings update is used below.

    Returns:
    -----------
    Tuple[Dict[str,float], Dict[str,float]]:
        (new_means, new_stds),
        where new_means has the same keys as old_means (minus any '_std' suffix),
        and new_stds has the same keys as old_stds.
    
    Notes:
    -----------
    1. In real usage, MCMC would typically run over all trials so far (batch) or re-run
       an entire chain each time with data[0..t], not just a single new observation.
       The single-trial approach is used here for consistency with the existing agent() pipeline.

    2. Interprets old_means and old_stds as a Normal prior:
       p(theta_i) ~ Normal(old_means[i], old_stds[i]^2), then multiplies by the
       likelihood from the single observed_choice.

    3. The parameter domains in existing code suggest that some parameters have bounds,
       e.g. [-1,1] or [0.01,2]. This example enforces those bounds by rejecting proposals
       that lie outside. Real code might do fancier transformations (like log-space for exponents).
    """
    from scipy.stats import norm
    "A) Setup and random seed"
    if not isinstance(random_seed, int):
        random.seed(random_seed)
        np.random.seed(random_seed)

    required_keys = ['keys', 'guesses', 'bounds']
    if not isinstance(param_info, dict) or not all(key in param_info for key in required_keys):
        raise ValueError(f"param_info must be a dictionary with keys {required_keys}.")

    param_keys = param_info.get("keys", list(old_means.keys()))
    param_bounds = param_info.get("bounds", None)

    "Map param -> (lower_bound, upper_bound)"
    param_bounds_map = {param_key: param_bounds[idx] for idx, param_key in enumerate(param_keys)}

    def log_prior(parameter_values_dict: Dict[str, float]) -> float:
        """
        B) Prior function: log of Normal priors from old_means/old_stds
        """
        total_lp = 0.0
        for param_key in param_keys:
            if '_std' not in param_key:
                prior_mean = old_means[param_key]
                sigma_key = param_key + '_std'
                sigma = max(old_stds.get(sigma_key, 0.5), 1e-9)  # Avoid zero or negative.
                x_val = parameter_values_dict[param_key]
                lprior = norm.logpdf(x_val, loc=prior_mean, scale=sigma)
                total_lp += lprior
        return total_lp

    def log_likelihood(parameter_values_dict: Dict[str, float]) -> float:
        """
        C) Likelihood function for single observed choice
        Return log( p( observed_choice | parameter_values ) ).
        """
        "Probability that the model chooses 'A' under these parameters"
        choice_output: dict = choice_func(
            current_game=game_dict,
            agent_params=parameter_values_dict,
            utility_settings=utility_settings,
            select_responses=False
        )

        probability_predict_A = choice_output.get('model_choose_A', None)
        if probability_predict_A is None:
            raise Exception("choice function failed to produce a response.")

        if observed_choice == 'A':
            return math.log(max(probability_predict_A, 1e-12))
        else:
            return math.log(max(1.0 - probability_predict_A, 1e-12))

    def log_posterior(parameter_values_dict: Dict[str, float]) -> float:
        """
        Posterior ~ Prior * Likelihood, in log space => log_prior + log_likelihood.
        Return -inf if out of bounds.
        """
        "Respect param-specific bounds"
        for param_key in param_keys:
            val = parameter_values_dict[param_key]
            lower_bound, upper_bound = param_bounds_map[param_key]
            if val < lower_bound or val > upper_bound:
                return np.clip(a=val, a_min=lower_bound, a_max=upper_bound)
                return -float('inf')

        return log_prior(parameter_values_dict) + log_likelihood(parameter_values_dict)

    "D) Prepare MCMC chain. Start from old_means, clamped to param_info bounds"
    current_params = {**copy.deepcopy(old_means), **copy.deepcopy(old_stds)}
    for param_key in param_keys:
        lower_bound, upper_bound = param_bounds_map[param_key]
        current_params[param_key] = max(lower_bound, min(upper_bound, current_params[param_key]))

    current_log_post = log_posterior(current_params)

    if np.isnan(current_log_post):
        print("NaN detected in log posterior!")
        return old_means, old_stds  

    samples_chain = []
    accepted_count = 0

    "E) Metropolis–Hastings random-walk"
    for idx in range(chain_length):
        proposal_dict = copy.deepcopy(current_params)
        "Random-walk step in each parameter."
        for param_key in param_keys:
            step = random.gauss(0.0, proposal_sd)
            proposal_dict[param_key] += step

        proposal_log_post = log_posterior(proposal_dict)
        acceptance_log_ratio = proposal_log_post - current_log_post

        "Accept or reject"
        if math.log(random.random()) < acceptance_log_ratio:
            current_params = proposal_dict
            current_log_post = proposal_log_post
            accepted_count += 1

        samples_chain.append(copy.deepcopy(current_params))

    acceptance_rate = accepted_count / float(chain_length)
    if general_settings.get('verbose', False):
        print(f"  MCMC acceptance rate: {acceptance_rate:.3f}")

    "F) Convert chain to arrays, discard burn-in, compute mean & std"
    valid_chain = samples_chain[burn_in::thin]
    param_vectors = []
    for sample_dict in valid_chain:
        param_vectors.append([sample_dict[param_key] for param_key in param_keys])
    chain_matrix = np.array(param_vectors)  # Shape: [#samples, #params].

    posterior_means_array = chain_matrix.mean(axis=0)
    posterior_stds_array = chain_matrix.std(axis=0)

    "Convert back to dict"
    new_means, new_stds = {}, {}
    for param_index, param_key in enumerate(param_keys):
        if '_std' in param_key:
            "Let std dev float freely, just ensure it's >= 0"
            new_stds[param_key] = float(max(posterior_stds_array[param_index], 0.0))
        else:
            new_means[param_key] = float(posterior_means_array[param_index])

    "G) Return final posterior dicts"
    return new_means, new_stds


def param_vector_to_pmf_array(param_vectors: Dict[Tuple[int], float],
                              meta_data: Dict[str, Dict[str, Any]],
                              general_settings: GeneralSettings,
                              use_fallback: bool = False) -> NDArray[np.float64]:
    """
    Converts a sparse parameter-coordinate → probability mapping into a dense normalized PMF array.

    Particle-filter mode (use_particle_filter=True): scatter particle probabilities directly into
    a full grid and normalize — no interpolation.
    Full-grid mode: fill a NaN-initialized grid, interpolate missing cells when sample_ratio < 1,
    then normalize.

    Arguments:
        • param_vectors   : dict mapping index-tuples to probability mass (sparse; zero entries may be absent).
        • meta_data       : grid metadata containing 'n_bins_per_dimension', 'tickvals',
                            'sample_ratio', and 'representation'.
        • general_settings: used to check use_particle_filter.
        • use_fallback    : if True, returns a zero array instead of raising when all inputs are NaN.

    Returns:
        • NDArray[np.float64]; normalized PMF array shaped by the meta_data grid dimensions.
    """
    grid_shape = tuple(meta_data["n_bins_per_dimension"] for _ in meta_data["tickvals"].keys())

    if general_settings.get('use_particle_filter', False):
        full_grid = np.zeros(grid_shape, dtype=float)
        for idx_tuple, prob in param_vectors.items():
            if prob > 0:
                full_grid[idx_tuple] += prob
        grid_sum = full_grid.sum()
        if grid_sum > 0:
            full_grid /= grid_sum
        return full_grid

    "Determine the dimensions and create an empty grid"
    full_grid = np.full(fill_value=np.nan, shape=grid_shape)

    "Fill the grid with probabilities from param_vectors"
    for param_vector, probability in param_vectors.items():
        indices = tuple(param_vector)
        if probability < 0:
            probability = 0
        full_grid[indices] = probability

    "Check if grid has any valid data"
    if np.isnan(full_grid).all():
        warning_str = "Warning: All values in param_vectors are NaN."
        print(param_vectors)
        if use_fallback:
            print(warning_str)
            return np.zeros_like(full_grid)
        else:
            raise Exception(warning_str)

    "Only interpolate for the interpolation strategy, not PF"
    if meta_data.get("sample_ratio", 1.0) < 1.0 and meta_data.get("representation", "grid") == "grid":
        n_dimensions = len(grid_shape)
        interp_method = "cubic" if n_dimensions <= 2 else "linear"
        full_grid = gnrl.fill_holes_nd(input_array=full_grid, output_shape=None, method=interp_method)

    full_grid /= full_grid.sum()
    return full_grid


def prior_grid_from_params(param_vals: Dict[str, Dict[str, Dict[str, float]]], param_info: ParamInfo, n_bins_per_dimension: int, 
                           sample_ratio: float = 0.5, covariation_matrix: CovMatDict = None, trust_inputs: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Generates a discrete joint-Gaussian pmf to act as an agent's prior, for each player and 
    role. Randomly samples from a grid of size (n_bins_per_dimension ^ n non-std parameters).

    Then this computes the multivariate normal pdf at each sampled point, possibly 
    incorporate a small 'volume element', and normalize so that it sums to 1.

    Arguments:
        • param_vals: Dict[str, Dict[str, Dict[str, float]]]; Mean and standard deviation parameters for each player in each role.
        • param_info: ParamInfo; Information about parameters used throughout this analysis. Keys: 'keys', 'bounds', and 'guesses'.
        • n_bins_per_dimension: int; The length of each array of tick values for each dimension (all the same length).
        • sample_ratio: float; Determines the number of samples for each dimension. Falls within [0, 1].
        • covariation_matrix: Dict[str, Dict[str, NDArray[np.float64]]]; Covariance matrices for each player and role. 
            - Example {
                "player_1": {
                    "chooser":   np.array([[0.0100, 0.0020, 0.0010], [0.0020, 0.0400, 0.0030], [0.0010, 0.0030, 0.0220]]),
                    "predictor": np.array([[0.0625, 0.0100, 0.0050], [0.0100, 0.0100, 0.0020], [0.0050, 0.0020, 0.0400]])
                },
                "player_2": {
                    "chooser":   np.array([[0.0400, 0.0030, 0.0020], [0.0030, 0.0225, 0.0010], [0.0020, 0.0010, 0.0100]]),
                    "predictor": np.array([[0.0900, 0.0200, 0.0100], [0.0200, 0.0400, 0.0050], [0.0100, 0.0050, 0.0600]])
                }
            }

    Returns:
        • grid_prior: Dict[str, Dict[str, Any]]; Maps randomly sampled parameter vectors to prior probabilities. 
            Example: {
                'meta_data': {
                    'sample_ratio': 0.25,
                    'n_bins_per_dimension': 41,
                    'tickvals': {
                        'Vᵢⱼ': [-1.0, -0.8, -0.6, -0.4, -0.2,  0.0,  0.2,  0.4,  0.6,  0.8,  1.0],
                        'Vᵢᵢ': [-1.0, -0.8, -0.6, -0.4, -0.2,  0.0,  0.2,  0.4,  0.6,  0.8,  1.0],
                        'αᵢⱼ': [-1.0, -0.8, -0.6, -0.4, -0.2,  0.0,  0.2,  0.4,  0.6,  0.8,  1.0]
                    }
                },
                player_uuid: {
                    player_role: {
                        'param_vectors': {
                            (0, 9, 2): 0.87,
                            (7, 3, 4): 0.05,
                            (6, 0, 1): 0.73,
                            (3, 5, 8): 0.11,...
                        }                             
                    },...
                },...
            }    
    """
    if not trust_inputs:
        if not (0 <= sample_ratio <= 1):
            raise ValueError(f"sample_ratio({sample_ratio}) must be between 0 and 1.")

        if not (3 <= n_bins_per_dimension <= 201):
            raise ValueError(f"n_bins_per_dimension({n_bins_per_dimension}) must be between 3 and 201.")

        if covariation_matrix is not None and not isinstance(covariation_matrix, dict):
            raise Exception(f"covariation_matrix should be a dictionary not {type(covariation_matrix)}: {covariation_matrix}.")

        param_val_err = "param_vals must be a dictionary containing parameter values for each player in each role."

        "Ensure param_vals is a dictionary"
        if not isinstance(param_vals, dict):
            raise ValueError(f"{param_val_err} Expected a dictionary, but got: {type(param_vals)}")

        "Validate each player UUID and their roles"
        for player_uuid, roles in param_vals.items():
            if not isinstance(roles, dict):
                raise ValueError(f"{param_val_err} Each player must have a dictionary of roles. Invalid entry for player {player_uuid}: {roles}")

            "Validate each role"
            for role, params in roles.items():
                if params is None:
                    continue

                "Check parameter keys"
                param_keys_in_params = [param_key for param_key in params.keys() if '_cov' not in param_key and param_key not in ('τ', 'temp')]
                if sorted(param_keys_in_params) != sorted(param_info["keys"]):
                    raise ValueError(
                        f"Parameter keys mismatch in player {player_uuid}, role {role}. "
                        f"Expected: {param_info['keys']}, but got: {param_keys_in_params}"
                    )

                "Validate parameter values and bounds"
                for idx, key in enumerate(param_info["keys"]):
                    value = params[key]
                    lower_bound, upper_bound = param_info["bounds"][idx]

                    "Ensure value is numeric"
                    if not isinstance(value, (int, float)):
                        raise ValueError(
                            f"Invalid type for parameter '{key}' in player {player_uuid}, role {role}. "
                            f"Expected numeric type, but got: {type(value)}"
                        )

                    "Ensure value is within bounds"
                    if not (lower_bound <= value <= upper_bound):
                        raise ValueError(
                            f"Parameter '{key}' in player {player_uuid}, role {role} is out of bounds. "
                            f"Value: {value}, Expected bounds: [{lower_bound}, {upper_bound}]"
                        )

    "Generate tick values for each parameter"
    tickvals = {
        param_key: np.linspace(
            start=param_info['bounds'][idx][0], 
            stop=param_info['bounds'][idx][1], 
            num=n_bins_per_dimension
        ) 
        for idx, param_key in enumerate(param_info['keys']) if '_std' not in param_key
    }

    "Store everything in a big dictionary"
    grid_prior = {
        'meta_data': {
            'n_bins_per_dimension': n_bins_per_dimension,
            'sample_ratio': sample_ratio,
            'tickvals': tickvals
        }
    }

    "Precompute the 'volume element' for each dimension (the spacing along that dimension)."
    spacing = {}
    for idx, param_key in enumerate(param_info["keys"]):
        if param_key.endswith('_std'):
            continue
        lower_bound, upper_bound = param_info["bounds"][idx]
        if n_bins_per_dimension > 1:
            spacing[param_key] = (upper_bound - lower_bound) / (n_bins_per_dimension - 1)
        else:
            spacing[param_key] = (upper_bound - lower_bound)  # Fallback if n_bins_per_dimension=1.

    "For each player and role, build a pmf"
    for player_uuid, roles_dict in param_vals.items():
        grid_prior[player_uuid] = {}
        for role_name, params_dict in roles_dict.items():
            if params_dict is None:
                continue
            "1) Extract means & standard deviations for each relevant param"
            means, stds = [], []
            these_param_keys = []
            for idx, param_key in enumerate(param_info["keys"]):
                if param_key.endswith('_std'):
                    continue
                param_mean = params_dict[param_key]
                sigma_key = param_key + '_std'
                sigma_val = params_dict[sigma_key]
                these_param_keys.append(param_key)
                stds.append(sigma_val)
                means.append(param_mean)

            "Degenerate k=0 model: no free parameters, so the prior is a point mass at the single"
            "0-dimensional grid point. scipy multivariate_normal cannot accept empty inputs."
            if not means:
                grid_prior[player_uuid][role_name] = {'param_vectors': {(): 1.0}}
                continue

            "2) Validate and correct the covariance matrix"
            if covariation_matrix is not None and covariation_matrix.get(player_uuid, {}).get(role_name) is not None:  
                cov_matrix = covariation_matrix[player_uuid][role_name]
                if cov_matrix.shape != (len(means), len(means)):
                    err_str = f"Cov matrix for {player_uuid} role={role_name} has shape {cov_matrix.shape}"
                    raise ValueError(f"{err_str}, expected {(len(means), len(means))}.")

                if not gnrl.is_positive_semidefinite(matrix=cov_matrix, tol=1e-12):
                    cov_matrix = gnrl.nearest_psd_matrix(matrix=cov_matrix, min_eigval=0.0)

            else:
                "Default to diagonal matrix."
                cov_matrix = np.diag(np.square(stds)) 

            "3) Create scipy's multivariate normal"
            multivariate_normal_distribution = multivariate_normal(mean=means, cov=cov_matrix, allow_singular=False)

            "4) Sample from the full grid: n_bins_per_dimension^d total points."
            full_axes = [tickvals[param_key] for param_key in these_param_keys]  # Each is length n_bins_per_dimension.
            mesh = np.meshgrid(*full_axes, indexing='ij')  

            "Flattens mesh axes into all_points: shape=(n_bins_per_dimension^d, d)."
            all_points = np.stack([mesh_axis.flatten() for mesh_axis in mesh], axis=-1)

            total_grid_size = all_points.shape[0]  # n_bins_per_dimension^d.
            "Pick a random subset of size (samples_per_dimension^d)."
            desired_sample_size = min(int(sample_ratio * total_grid_size), total_grid_size)

            volume_elem = 1.0
            for param_key in these_param_keys:
                volume_elem *= spacing[param_key]

            "6) Convert each point in 'sampled_points' back to a discrete 'index'."
            if sample_ratio == 1.0: 

                "Use the full grid"
                sampled_points = all_points  # Shape: (n_bins_per_dimension^d, d).
                
                "Precompute PDF values for all grid points"
                pmf_values = multivariate_normal_distribution.pdf(sampled_points)  # Compute the PDF for all grid points at once.
                
                "Precompute indices for all points in the full grid"
                indices = np.stack([
                    np.searchsorted(tickvals[param_key], sampled_points[:, dim_i])
                    for dim_i, param_key in enumerate(these_param_keys)
                ], axis=-1)  # Shape: (n_bins_per_dimension^d, d)
                
                "Construct param_vectors dictionary efficiently"
                param_vectors = {}
                for idx_tuple, pmf_val in zip(map(tuple, indices), pmf_values):
                    "Accumulate probabilities for each index tuple"
                    param_vectors[idx_tuple] = param_vectors.get(idx_tuple, 0.0) + pmf_val * volume_elem
            else:
                sample_indices = random.sample(range(total_grid_size), desired_sample_size)
                sampled_points = all_points[sample_indices, :]  # Shape=(desired_sample_size, d).

                "5) Evaluate the pdf at each sampled point"
                pdf_values = np.atleast_1d(multivariate_normal_distribution.pdf(sampled_points))

                pmf_values = pdf_values * volume_elem

                indices = np.stack([
                    np.searchsorted(tickvals[param_key], sampled_points[:, dim_i])
                    for dim_i, param_key in enumerate(these_param_keys)
                ], axis=-1)   

                param_vectors = {}
                for row_idx, idx_tuple in enumerate(map(tuple, indices)):
                    pmf_val = pmf_values[row_idx]
                    current_val = param_vectors.get(idx_tuple, 0.0)
                    param_vectors[idx_tuple] = current_val + pmf_val

            "7) Normalize so the total sum is 1"
            total_mass = sum(param_vectors.values())
            if total_mass > 0:
                for idx_tuple in param_vectors:
                    param_vectors[idx_tuple] /= total_mass
            else:
                n_bins = len(param_vectors)
                for idx_tuple in param_vectors:
                    param_vectors[idx_tuple] = 1.0 / n_bins if n_bins > 0 else 0.0

            grid_prior[player_uuid][role_name] = {
                'param_vectors': param_vectors,
            }

    return grid_prior


def bayesian_update_grid(prior_array: NDArray[np.float64] | dict[tuple[int, ...], float], meta_data: dict[str, Any], game_dict: dict[str, Any], choice_func: callable, 
                         utility_settings: dict[str, bool], general_settings: GeneralSettings, param_info: dict[str, Any], softmax_temperature: float, no_memory_mode: bool = False) -> dict[str, Any]:
    """
    Perform a single Bayesian posterior update over a discretized parameter grid.

    Implements three sampling regimes controlled by meta_data['sample_ratio'] and
    meta_data['use_particle_filter']:

    Full-grid update (sample_ratio == 1.0):
        Evaluates the likelihood on every grid bin and multiplies by the prior.
        Input prior may be a dense ndarray; if a sparse dict is passed it is densified once.

    Uniform subsample (sample_ratio < 1.0, use_particle_filter == False):
        Draws a uniform subset of grid bins and multiplies by the prior on those bins.
        Input prior is densified once before sampling.

    Particle filter (sample_ratio < 1.0, use_particle_filter == True):
        Maintains a persistent set of particles in bin-index space; state lives in
        meta_data['pf_state']. Each update computes likelihood once per unique bin
        (major speedup), updates particle weights, resamples when ESS drops below the
        threshold, and optionally jitters particles in bin space. Returns a sparse
        posterior mass map and sets meta_data['representation'] = 'particles' so
        downstream code knows not to densify.

    Arguments:
        • prior_array: NDArray[np.float64] | dict[tuple[int, ...], float]
            Either a dense PMF array over the active parameter grid, or a sparse dict
            mapping multi-index tuples to probability masses. The particle filter branch
            keeps the sparse dict sparse (no densification).
        • meta_data: dict
            Required keys:
                'n_bins_per_dimension': int
                'tickvals': dict[str, list[float]] — bin locations per active mean parameter
                'sample_ratio': float in (0, 1] — controls subsampling / particle-count scaling
            Optional particle-filter knobs:
                'use_particle_filter': bool (default True when sample_ratio < 1)
                'pf_max_particles': int (cap on particle count; default 5000)
                'pf_min_particles': int (floor on particle count; default 200)
                'pf_resample_fraction': float in (0, 1] (ESS/N threshold; default 0.5)
                'pf_jitter_sd': float (std-dev of Gaussian jitter in bin units; default 0.0)
                'pf_random_seed': int | None
                'pf_state': persistent particle state dict from the previous update
                'representation': 'grid' | 'particles'
        • game_dict: dict
            A single game payload containing payoffs and the observed choice ('A' or 'B').
        • choice_func: callable
            A function with the signature:
                choice_func(current_game, agent_params, utility_settings, softmax_temperature, select_responses=False)
            Must return a dict with key 'model_choose_A' in [0, 1]. Typically this is response().
        • utility_settings: dict
            Boolean flags defining the active utility functional form (same toggles as utility()).
        • general_settings: GeneralSettings
            Top-level settings dict; used here for 'n_bins_per_dimension' and related knobs.
        • param_info: ParamInfo
            Holds parameter keys, bounds, and guesses. Only mean-parameter keys (those not ending
            in '_std') are active on the grid; standard-deviation keys are excluded from the grid.
        • softmax_temperature: float
            SoftMax temperature τ used during likelihood evaluation.
        • no_memory_mode: bool
            If True, the posterior ignores the prior and is computed from the likelihood alone.
            Used when user wants a single-game belief snapshot with no accumulated history.

    Returns:
        • dict with keys:
            'param_vectors': dict[(i₁,…,iₐ) → float] — sparse posterior mass map over grid bins.
            'meta_data': dict — updated meta_data (includes pf_state, 'representation', etc.).
    """
    "----- Toggle particle filter path when subsampling -----"
    use_particle_filter: bool = bool(meta_data.get("use_particle_filter", True))

    "Active mean-parameter keys; grid size info"
    param_mean_keys = [param_key for param_key in param_info["keys"] if not param_key.endswith("_std")]
    n_bins_per_dimension: int = int(meta_data["n_bins_per_dimension"])
    tickvals: dict[str, list[float]] = meta_data["tickvals"]
    sample_ratio: float = float(meta_data["sample_ratio"])

    n_dims = len(param_mean_keys)
    total_grid_size = n_bins_per_dimension ** n_dims

    "---------------------------- FULL GRID (unchanged) ----------------------------"
    if sample_ratio == 1.0:
        if isinstance(prior_array, dict):
            "Densify once (only for full-grid case)"
            prior_array = param_vector_to_pmf_array(param_vectors=prior_array, meta_data=meta_data, general_settings=general_settings)

        full_axes = [tickvals[param_key] for param_key in param_mean_keys]
        mesh = np.meshgrid(*full_axes, indexing='ij')
        all_points = np.stack([mesh_axis.flatten() for mesh_axis in mesh], axis=-1)
        prior_flat = prior_array.flatten().astype(float)

        "Bin indices for each point"
        indices = np.stack([
            np.searchsorted(tickvals[param_key], all_points[:, dim_idx])
            for dim_idx, param_key in enumerate(param_mean_keys)
        ], axis=-1)

        "Evaluate likelihood once per point"
        likelihoods = np.empty(all_points.shape[0], dtype=float)
        obs_is_A = (game_dict['choice'] == 'A')
        for row_idx in range(all_points.shape[0]):
            agent_params = {param_mean_keys[dim_idx]: float(all_points[row_idx, dim_idx]) for dim_idx in range(n_dims)}
            probability_choose_A = choice_func(current_game=game_dict,
                             agent_params=agent_params,
                             utility_settings=utility_settings,
                             softmax_temperature=softmax_temperature,
                             select_responses=False)['model_choose_A']
            likelihoods[row_idx] = probability_choose_A if obs_is_A else (1.0 - probability_choose_A)

        "Posterior (full-grid)"
        posterior_probs = likelihoods if no_memory_mode else (prior_flat * likelihoods)

        "Accumulate into sparse map keyed by multi-index"
        posterior_param_vectors: dict[tuple[int, ...], float] = {}
        for idx_tuple, mass in zip(map(tuple, indices), posterior_probs):
            posterior_param_vectors[idx_tuple] = posterior_param_vectors.get(idx_tuple, 0.0) + float(mass)

        "Normalize"
        posterior_normalizer = sum(posterior_param_vectors.values())
        if posterior_normalizer > 0:
            for idx_tuple in posterior_param_vectors:
                posterior_param_vectors[idx_tuple] /= posterior_normalizer

        return {'param_vectors': posterior_param_vectors, 'meta_data': meta_data}

    "----------------------- UNIFORM SUBSAMPLE (unchanged) ------------------------"
    if not use_particle_filter:
        if isinstance(prior_array, dict):
            prior_array = param_vector_to_pmf_array(param_vectors=prior_array, meta_data=meta_data, general_settings=general_settings)

        desired_sample_size = max(1, min(int(sample_ratio * total_grid_size), total_grid_size))
        full_axes = [tickvals[param_key] for param_key in param_mean_keys]
        mesh = np.meshgrid(*full_axes, indexing='ij')
        all_points = np.stack([mesh_axis.flatten() for mesh_axis in mesh], axis=-1)

        sample_indices = random.sample(range(total_grid_size), desired_sample_size)
        sampled_points = all_points[sample_indices, :]
        sampled_prior_probs = prior_array.flatten()[sample_indices]

        indices = np.stack([
            np.searchsorted(tickvals[param_key], sampled_points[:, dim_idx])
            for dim_idx, param_key in enumerate(param_mean_keys)
        ], axis=-1)

        obs_is_A = (game_dict['choice'] == 'A')
        likelihoods = np.empty(sampled_points.shape[0], dtype=float)
        for row_idx in range(sampled_points.shape[0]):
            agent_params = {param_mean_keys[dim_idx]: float(sampled_points[row_idx, dim_idx]) for dim_idx in range(n_dims)}
            probability_choose_A = choice_func(current_game=game_dict,
                             agent_params=agent_params,
                             utility_settings=utility_settings,
                             softmax_temperature=softmax_temperature,
                             select_responses=False)['model_choose_A']
            likelihoods[row_idx] = probability_choose_A if obs_is_A else (1.0 - probability_choose_A)

        posterior_probs = likelihoods if no_memory_mode else (sampled_prior_probs * likelihoods)

        posterior_param_vectors: dict[tuple[int, ...], float] = {}
        for idx_tuple, mass in zip(map(tuple, indices), posterior_probs):
            posterior_param_vectors[idx_tuple] = posterior_param_vectors.get(idx_tuple, 0.0) + float(mass)

        posterior_normalizer = sum(posterior_param_vectors.values())
        if posterior_normalizer > 0:
            for idx_tuple in posterior_param_vectors:
                posterior_param_vectors[idx_tuple] /= posterior_normalizer

        return {'param_vectors': posterior_param_vectors, 'meta_data': meta_data}

    "--------------------------- PARTICLE FILTER (FAST) ---------------------------"
    "Particle budget (cap/floor), ESS threshold, and jitter"
    pf_max_particles = int(meta_data.get("pf_max_particles", 5000))
    pf_min_particles = int(meta_data.get("pf_min_particles", 200))
    pf_resample_frac = float(meta_data.get("pf_resample_fraction", 0.5))
    pf_jitter_sd     = float(meta_data.get("pf_jitter_sd", 0.0))
    random_seed         = meta_data.get("pf_random_seed", None)
    rng = np.random.default_rng(random_seed)

    "Interpret sample_ratio as an upper bound on the particle budget, capped by pf_max_particles"
    target_particles = min(int(round(sample_ratio * total_grid_size)), pf_max_particles)
    n_particles = max(pf_min_particles, target_particles, 1)

    "Build per-dimension tick arrays once"
    ticks_list = [np.asarray(tickvals[param_key], dtype=float) for param_key in param_mean_keys]

    def params_for_indices(idx_mat: np.ndarray) -> list[dict[str, float]]:
        "idx_mat shape: (K, n_dims); return K param dicts"
        out: list[dict[str, float]] = []
        "Grab values per dim, vectorized"
        vals_per_dim = [ticks_list[dim_idx][idx_mat[:, dim_idx]] for dim_idx in range(n_dims)]
        for row_idx in range(idx_mat.shape[0]):
            out.append({param_mean_keys[dim_idx]: float(vals_per_dim[dim_idx][row_idx]) for dim_idx in range(n_dims)})
        return out

    "Fetch persistent PF state or initialize"
    pf_state = meta_data.get("pf_state", None)

    if pf_state is None:
        "Initialize from prior WITHOUT densifying:"
        "- if sparse dict: sample keys by their mass"
        "- else dense array: sample flat indices by pmf"
        if isinstance(prior_array, dict):
            prior_keys = np.array(list(prior_array.keys()), dtype=int)          # (M, d)
            prior_weights  = np.array([max(0.0, float(prior_mass)) for prior_mass in prior_array.values()], dtype=float)
            prior_weight_sum = float(prior_weights.sum())
            if prior_weight_sum <= 0:
                prior_weights = np.full(prior_keys.shape[0], 1.0 / max(1, prior_keys.shape[0]))
            else:
                prior_weights /= prior_weight_sum
            ancestor_rows = rng.choice(prior_keys.shape[0], size=n_particles, replace=True, p=prior_weights)
            indices = prior_keys[ancestor_rows]
            weights = np.full(n_particles, 1.0 / n_particles, dtype=float)
        else:
            flat = np.asarray(prior_array, dtype=float).ravel()
            flat_prior_sum = float(flat.sum())
            if flat_prior_sum <= 0:
                flat = np.full_like(flat, 1.0 / max(1, flat.size), dtype=float)
            else:
                flat /= flat_prior_sum
            flat_idx = rng.choice(flat.size, size=n_particles, replace=True, p=flat)
            indices = np.column_stack(np.unravel_index(flat_idx, (n_bins_per_dimension,) * n_dims)).astype(int)
            weights = np.full(n_particles, 1.0 / n_particles, dtype=float)
    else:
        indices = np.asarray(pf_state["indices"], dtype=int)
        weights = np.asarray(pf_state["weights"], dtype=float)
        if indices.shape[0] != weights.shape[0]:
            raise ValueError("pf_state malformed: indices and weights have different lengths.")
        "If the requested particle count changes, resample to match it."
        if indices.shape[0] != n_particles:
            cumulative_weights = np.cumsum(weights)
            positions = (rng.random() + np.arange(n_particles)) / n_particles
            selected_particle_indices = np.searchsorted(cumulative_weights, positions, side="left")
            indices = indices[selected_particle_indices]
            weights = np.full(n_particles, 1.0 / n_particles, dtype=float)

    "-- Compute per-particle likelihood for this game (unique-eval to avoid rework)."
    obs_A = (game_dict['choice'] == 'A')

    "Unique rows of indices; inverse maps back to all particles."
    unique_indices, inverse_map = np.unique(indices, axis=0, return_inverse=True)
    n_unique = unique_indices.shape[0]

    like_unique = np.empty(n_unique, dtype=float)
    "Build agent params only for unique particles"
    unique_params_list = params_for_indices(unique_indices)

    for unique_particle_idx in range(n_unique):
        choice_result = choice_func(
            current_game=game_dict,
            agent_params=unique_params_list[unique_particle_idx],
            utility_settings=utility_settings,
            softmax_temperature=softmax_temperature,
            select_responses=False
        )['model_choose_A']
        like_unique[unique_particle_idx] = choice_result if obs_A else (1.0 - choice_result)

    "Broadcast back to all particles."
    like = like_unique[inverse_map]

    "Weight update (log-stable); if no_memory_mode → likelihood-only"
    if no_memory_mode:
        weights = like
    else:
        log_weights = np.log(weights + 1e-300) + np.log(like + 1e-300)
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)

    "Normalize / fallback uniform"
    weights_normalizer = float(weights.sum())
    if not np.isfinite(weights_normalizer) or weights_normalizer <= 0.0:
        weights = np.full(n_particles, 1.0 / n_particles, dtype=float)
    else:
        weights /= weights_normalizer

    "ESS-based resampling"
    ess = 1.0 / np.sum(weights ** 2)
    if ess < pf_resample_frac * n_particles:
        cumulative_weights = np.cumsum(weights)
        positions = (rng.random() + np.arange(n_particles)) / n_particles
        selected_particle_indices = np.searchsorted(cumulative_weights, positions, side="left")
        indices = indices[selected_particle_indices]
        weights.fill(1.0 / n_particles)

        "Optional jitter in *bin space* (default 0.0 → fast; set >0 to explore)"
        if pf_jitter_sd > 0.0:
            noise = rng.normal(0.0, pf_jitter_sd, size=indices.shape)
            jittered = np.rint(indices.astype(float) + noise).astype(int)
            max_idx = n_bins_per_dimension - 1
            "Reflect at boundaries."
            jittered = np.where(jittered < 0, -jittered, jittered)
            over_boundary = jittered > max_idx
            jittered[over_boundary] = 2 * max_idx - jittered[over_boundary]
            indices = np.clip(jittered, 0, max_idx)

    "Build sparse posterior map by summing weights for identical bins"
    unique_bins, inverse_bin_map = np.unique(indices, axis=0, return_inverse=True)
    mass_per_bin = np.zeros(unique_bins.shape[0], dtype=float)
    np.add.at(mass_per_bin, inverse_bin_map, weights)

    posterior_param_vectors: dict[tuple[int, ...], float] = {
        tuple(row): float(mass) for row, mass in zip(map(tuple, unique_bins), mass_per_bin)
    }
    "Normalize defensively"
    posterior_normalizer = sum(posterior_param_vectors.values())
    if posterior_normalizer > 0:
        for idx_tuple in posterior_param_vectors:
            posterior_param_vectors[idx_tuple] /= posterior_normalizer

    "Persists PF state and marks representation as 'particles' to prevent downstream densification."
    new_meta = dict(meta_data)
    new_meta["pf_state"] = {"indices": indices.tolist(), "weights": weights.tolist()}
    new_meta["representation"] = "particles"

    return {'param_vectors': posterior_param_vectors, 'meta_data': new_meta}


def agent(dyad_games: DyadGames, game_idx_start: int, game_idx_stop: int, general_settings: GeneralSettings,
          initial_params: Dict[str, Dict[str, float]], param_info: ParamInfo, utility_settings: UtilitySettings,
          player_uuid: str | None = None, player_role: str | None = None, select_responses: bool = False, softmax_temperature: float | None = None) -> List[dict]:
    """
    Run the UBM for a single player over a slice of a dyad's games, updating beliefs game by game.

    This is the core sequential inference loop of the Utility Bayesian Model. For each game in
    [game_idx_start, game_idx_stop], agent() retrieves the prior from the previous game, calls
    bayesian_update_grid() (or the parametric / MCMC variant depending on update_method) to
    incorporate the observed choice, stores the resulting posterior, and records the model's
    predicted probability alongside the posterior summary statistics. Results are written in-
    place into the 'parameter_estimates' sub-dict of each game dictionary.

    Arguments:
        • dyad_games: List[dict]
            Ordered list of game dictionaries for one dyad, each containing payoffs, roles,
            and observed choices. The 'parameter_estimates' key is written into each game.
        • game_idx_start: int
            Index of the first game to process (0-based). Clamped to [0, len(dyad_games)-1].
        • game_idx_stop: int
            Index of the last game to process (inclusive). Clamped to len(dyad_games)-1.
        • general_settings: GeneralSettings
            Settings controlling update_method ('grid', 'parametric', 'naive'), sample_ratio,
            n_bins_per_dimension, softmax_temperature, learning_rate, and other UBM parameters.
        • initial_params: Dict[str, Dict[str, float]]
            Starting parameter values keyed by role ('chooser', 'predictor'). These seed the
            prior for game_idx_start before any observations have been seen.
        • param_info: ParamInfo
            Parameter configuration dict with 'keys', 'bounds', and 'guesses'.
        • utility_settings: UtilitySettings
            Boolean toggles selecting the active utility functional form.
        • player_uuid: str | None
            UUID of the player to run. If None, the function may operate on an implicit player.
        • player_role: str | None
            If 'chooser' or 'predictor', only processes games where this player holds that role.
            If None, processes games where the player holds either role.
        • select_responses: bool
            If True, the response function generates a stochastic binary response (0/1) rather than
            a float probability. Used during simulation to generate artificial response data.
            Set False (default) when processing real participant data to avoid overwriting responses.
        • softmax_temperature: float | None
            Overrides general_settings['softmax_temperature'] for the choice probability step.
            If None or out of range, falls back to the temperature in general_settings.

    Returns:
        • List[dict] — the same dyad_games list, with 'parameter_estimates' populated or
          updated in-place for each game in [game_idx_start, game_idx_stop].
    """
    "=== 1) Basic Setup ==="
    num_meetings = len(dyad_games)
    if not isinstance(game_idx_start, int) or game_idx_start < 0:
        game_idx_start = 0
    if not isinstance(game_idx_stop, int) or game_idx_stop >= num_meetings:
        game_idx_stop = num_meetings - 1
    if game_idx_start > game_idx_stop:
        game_idx_start = game_idx_stop
    
    "Extract settings"
    sample_ratio = general_settings.get('sample_ratio', True)
    learning_rate = general_settings.get('learning_rate', True)
    update_method = general_settings.get('update_method', True)
    include_covariance = general_settings.get('include_covariance', True)
    n_bins_per_dimension = general_settings.get('n_bins_per_dimension', True)
    default_softmax_temperature = general_settings.get('softmax_temperature', True)

    "Possibly override if there's a param-based temperature"
    if (not general_settings.get('temperature_is_param', False)
        or not (isinstance(softmax_temperature, (int, float)) and 0 < softmax_temperature <= 3)):
        softmax_temperature = default_softmax_temperature

    "=== 2) Main Loop Over Games ==="
    idx = game_idx_start
    while idx <= game_idx_stop and idx < num_meetings:
        game_dict = dyad_games[idx]
        
        "2a) Figure out which role this player actually occupies in *this* game"
        actual_game_role = None
        if game_dict.get('chooser') == player_uuid:
            actual_game_role = 'chooser'
        elif game_dict.get('predictor') == player_uuid:
            actual_game_role = 'predictor'
        else:
            print(f"Game {idx} player_uuid {player_uuid}")
            game_copy = copy.deepcopy(game_dict)
            del game_copy['parameter_estimates']
            pp.pprint(game_copy)
            raise Exception(f"No role found in game {idx} for player {player_uuid}")

        "assigned_role is what the caller wants us to run"
        assigned_role = player_role

        "2b) Create or get the sub-dicts where parameter estimates will be stored."
        param_estimates = game_dict.setdefault('parameter_estimates', {})
        method_dict = param_estimates.setdefault(update_method, {})
        player_est_dict = method_dict.setdefault(player_uuid, {})

        "2c) Copy forward both roles' parameters from the previous game"
        if idx > 0:
            prev_game = dyad_games[idx - 1]
            prev_method_dict = prev_game.get('parameter_estimates', {}).get(update_method, {})
            prev_player_est = prev_method_dict.get(player_uuid, {})
            "If prev_game had 'chooser' data"
            if 'chooser' in prev_player_est:
                old_chooser_params = prev_player_est['chooser'].get('params', {})
                player_est_dict.setdefault('chooser', {})['params'] = copy.deepcopy(old_chooser_params)

                if update_method == 'grid':
                    if 'param_vectors' in prev_player_est['chooser']:
                        player_est_dict['chooser']['param_vectors'] = prev_player_est['chooser']['param_vectors']
                    if 'meta_data' in prev_player_est['chooser']:
                        player_est_dict['chooser']['meta_data'] = prev_player_est['chooser']['meta_data']

            "If prev_game had 'predictor' data"
            if 'predictor' in prev_player_est:
                old_pred_params = prev_player_est['predictor'].get('params', {})
                player_est_dict.setdefault('predictor', {})['params'] = copy.deepcopy(old_pred_params)

                if update_method == 'grid':
                    if 'param_vectors' in prev_player_est['predictor']:
                        player_est_dict['predictor']['param_vectors'] = prev_player_est['predictor']['param_vectors']
                    if 'meta_data' in prev_player_est['predictor']:
                        player_est_dict['predictor']['meta_data'] = prev_player_est['predictor']['meta_data']

        else:
            "idx == 0 => store initial_params if not already done"
            for plr_role in ('chooser', 'predictor'):
                if plr_role in initial_params:  # E.g. initial_params['chooser'] or .predictor.
                    player_est_dict.setdefault(plr_role, {})['params'] = copy.deepcopy(initial_params[plr_role])

            "If role='predictor' and using grid, build initial prior param_vectors for the predictor."
            "Skip for k=0 models: no free parameters means no belief distribution to maintain."
            _non_std_keys = [_k for _k in param_info.get("keys", []) if not _k.endswith('_std')]
            if update_method == 'grid' and assigned_role != 'chooser' and _non_std_keys:
                pred_sub = player_est_dict.setdefault('predictor', {})
                if 'param_vectors' not in pred_sub or 'meta_data' not in pred_sub:
                    this_pred_params = initial_params.get('predictor', {})
                    param_vals = {player_uuid: {'predictor': this_pred_params}}
                    covar = None
                    if include_covariance:
                        covar = {
                            player_uuid: {
                                'predictor': gnrl.build_covariation_matrix(param_info=param_info,
                                                                           params=this_pred_params, raise_on_invalid=True)
                            }
                        }
                    prior_data = prior_grid_from_params(param_vals=param_vals,
                                                        param_info=param_info,
                                                        n_bins_per_dimension=n_bins_per_dimension,
                                                        sample_ratio=sample_ratio,
                                                        covariation_matrix=covar,
                                                        trust_inputs=False)

                    pred_sub['meta_data'] = prior_data['meta_data']
                    pred_sub['param_vectors'] = prior_data[player_uuid]['predictor']['param_vectors']

        "2d) Decides whether to skip or play."
        if assigned_role is None:
            "If player_role is unspecified, uses whichever role the player has."
            if actual_game_role is None:
                "The player is not in this game => skip"
                idx += 1
                continue
            role_to_play = actual_game_role
        else:
            "Uses assigned_role only."
            if actual_game_role != assigned_role:
                "Skip if the actual game role doesn't match the assigned role"
                idx += 1
                continue
            role_to_play = assigned_role

        "2e) Executes the active role logic."
        role_params_for_this_game = player_est_dict[role_to_play].get('params', {})
        if not role_params_for_this_game:
            "fallback to initial if missing"
            role_params_for_this_game = copy.deepcopy(initial_params.get(role_to_play, {}))
            player_est_dict[role_to_play]['params'] = role_params_for_this_game

        "Make the choice or prediction."
        model_sel_key = "model_choose_A" if role_to_play == 'chooser' else "model_predict_A"
        "Predictor and chooser share the same temperature — asymmetric temperatures would add a parameter without clear theoretical motivation."
        current_temp = softmax_temperature

        choice_output = response(
            current_game=game_dict,
            agent_params=role_params_for_this_game,
            utility_settings=utility_settings,
            softmax_temperature=current_temp,
            select_responses=select_responses
        )

        "Store the model's output for this game."
        player_est_dict[role_to_play]['output'] = {
            model_sel_key: choice_output["model_choose_A"],
            'confidence': choice_output["confidence"]
        }

        if select_responses:
            "Store the sampled response in the game dict — 'choice' for chooser role, 'prediction' for predictor role."
            if role_to_play == 'chooser':
                choice_bit = choice_output["model_choose_A"]
                game_dict["choice"] = "A" if choice_bit == 1 else "B"
            elif role_to_play == 'predictor':
                pred_bit = choice_output["model_choose_A"]
                game_dict["prediction"] = "A" if pred_bit == 1 else "B"

        "2f) Skips update on game 0; otherwise, updates predictor beliefs."
        if idx == 0:
            "No update in first game. Do not overwrite priors. Cannot learn until the first choice is observed."  
            pass  
        else:
            if role_to_play == 'predictor':
                observed_choice = game_dict.get('choice', None)
                predictor_abdicated = game_dict.get('abdicated_predictor', False)
                predictor_learned_something = observed_choice and not predictor_abdicated

                "Do the final update on the last game or if predictor observed a choice"
                if predictor_learned_something or idx == game_idx_stop:
                    old_means = {
                        param_key: param_value
                        for param_key, param_value in role_params_for_this_game.items()
                        if not param_key.endswith('_std')
                    }
                    old_stds = {
                        param_key: param_value
                        for param_key, param_value in role_params_for_this_game.items()
                        if param_key.endswith('_std')
                    }

                    if update_method == 'naive':
                        "This 'naive' model predicts from fixed parameters--no learning."
                        pass

                    elif update_method == 'parametric':
                        new_means, new_stds = bayesian_update_parametric(
                            old_means=old_means, old_stds=old_stds,
                            observed_choice=observed_choice,
                            game_dict=game_dict, choice_func=response,
                            utility_settings=utility_settings,
                            learning_rate=learning_rate
                        )
                        "Store the updated results."
                        updated_params = {}
                        for param_key in param_info["keys"]:
                            if '_std' in param_key:
                                updated_params[param_key] = new_stds.get(param_key, 0.0)
                            else:
                                updated_params[param_key] = new_means.get(param_key, 0.0)
                        player_est_dict['predictor']['params'] = copy.deepcopy(updated_params)

                    elif update_method == 'MCMC':
                        new_means, new_stds = bayesian_update_mcmc(
                            old_means=old_means, old_stds=old_stds,
                            observed_choice=observed_choice,
                            game_dict=game_dict, choice_func=response,
                            utility_settings=utility_settings,
                            param_info=param_info
                        )
                        updated_params = {}
                        for param_key in param_info["keys"]:
                            if '_std' in param_key:
                                updated_params[param_key] = new_stds.get(param_key, 0.0)
                            else:
                                updated_params[param_key] = new_means.get(param_key, 0.0)
                        player_est_dict['predictor']['params'] = copy.deepcopy(updated_params)

                    elif update_method == 'grid':
                        "------------- GRID-BASED UPDATE -------------"
                        "(i) Build or retrieve the “prior” representation"
                        prior_grid_data = None
                        pred_sub = player_est_dict['predictor']
                        prev_vectors = pred_sub.get('param_vectors', None)
                        prev_meta    = pred_sub.get('meta_data', None)

                        if (prev_vectors is not None) and (prev_meta is not None):
                            "Fast path: if previous posterior was produced by the particle filter,"
                            "Keeps it sparse and DOES NOT densify (no interpolation, no qhull)."
                            if isinstance(prev_vectors, dict) and prev_meta.get('representation') == 'particles':
                                prior_grid_data = {
                                    'prior_array': prev_vectors,     # Sparse map: {(i1,...,id): mass}.
                                    'meta_data':   prev_meta
                                }
                            else:
                                "Grid representation: convert to dense PMF array once"
                                prior_array = param_vector_to_pmf_array(
                                    param_vectors=prev_vectors,
                                    meta_data=prev_meta,
                                    general_settings=general_settings
                                )
                                prior_grid_data = {
                                    'prior_array': prior_array,      # Dense ndarray.
                                    'meta_data':   prev_meta
                                }

                        "Fallback (very first game or if previous state missing)"
                        if not prior_grid_data or prior_grid_data.get('prior_array', None) is None:
                            fallback_pred_params = initial_params.get('predictor', {})
                            param_vals = {player_uuid: {'predictor': fallback_pred_params}}
                            covar = None
                            if include_covariance:
                                covar = {
                                    player_uuid: {
                                        'predictor': gnrl.build_covariation_matrix(
                                            param_info=param_info,
                                            params=fallback_pred_params,
                                            raise_on_invalid=True
                                        )
                                    }
                                }
                            fallback_prior_data = prior_grid_from_params(
                                param_vals=param_vals,
                                param_info=param_info,
                                n_bins_per_dimension=n_bins_per_dimension,
                                sample_ratio=sample_ratio,
                                covariation_matrix=covar,
                                trust_inputs=False
                            )
                            "Keep the prior sparse (dict) to avoid densification/interpolation."
                            "Mark representation so downstream knows how to treat it."
                            sparse_prior_vectors = fallback_prior_data[player_uuid][role_to_play]['param_vectors']
                            prior_grid_data = {
                                'prior_array': sparse_prior_vectors,  # Dict: {(i1,...,id): mass}.
                                'meta_data': {**fallback_prior_data['meta_data'], 'representation': 'grid_sparse'}
                            }

                        "(ii) Inject PF knobs into meta_data (so bayesian_update_grid sees them)"
                        meta_for_update = copy.deepcopy(prior_grid_data['meta_data'])
                        meta_for_update['use_particle_filter']  = bool(general_settings.get('use_particle_filter', True))
                        meta_for_update['pf_max_particles']     = int(general_settings.get('pf_max_particles', 5000))
                        meta_for_update['pf_min_particles']     = int(general_settings.get('pf_min_particles', 200))
                        meta_for_update['pf_resample_fraction'] = float(general_settings.get('pf_resample_fraction', 0.5))
                        meta_for_update['pf_jitter_sd']         = float(general_settings.get('pf_jitter_sd', 0.0))  # Default 0.0 for speed.

                        "Used for a trivial non-Bayesian model that forgets all priors."
                        no_memory_mode = general_settings.get('no_memory_mode', False)

                        "(iii) Now do the update"
                        role_params = initial_params.get(player_role, {})
                        likelihood_temp = role_params.get('τ', role_params.get('temp', softmax_temperature))
                        posterior_data = bayesian_update_grid(
                            prior_array=prior_grid_data['prior_array'],   # Dict or ndarray.
                            meta_data=meta_for_update,
                            softmax_temperature=likelihood_temp,
                            utility_settings=utility_settings,
                            general_settings=general_settings,
                            no_memory_mode=no_memory_mode,
                            param_info=param_info,
                            game_dict=game_dict,
                            choice_func=response,
                        )

                        "Store prior parameter stats."
                        if isinstance(prior_grid_data['prior_array'], dict) and prior_grid_data['meta_data'].get('representation') == 'particles':
                            pred_sub['params'] = gnrl._statistics_from_sparse_param_vectors(
                                param_vectors=prior_grid_data['prior_array'],
                                meta_data=prior_grid_data['meta_data'],
                                param_info=param_info
                            )
                        else:
                            tickvals_array = [
                                prior_grid_data['meta_data']["tickvals"][key]
                                for key in prior_grid_data['meta_data']["tickvals"].keys()
                            ]
                            pred_sub['params'] = gnrl.compute_statistics(
                                joint_pmf=prior_grid_data['prior_array'],
                                grids=tickvals_array,
                                param_info=param_info
                            )

                        if 'τ' not in pred_sub['params']:
                            "Storing choice temperature (which is static across rounds)."
                            role_init = initial_params.get(player_role, {})
                            if 'τ' in role_init:
                                pred_sub['params']['τ'] = role_init['τ']
                            elif 'temp' in role_init:
                                pred_sub['params']['τ'] = role_init['temp']
                            else:
                                pred_sub['params']['τ'] = softmax_temperature

                        "(iii) Store the new posterior param_vectors"
                        pred_sub['meta_data'] = posterior_data['meta_data']
                        pred_sub['param_vectors'] = posterior_data['param_vectors']

                        "(iv) On last game, compute final means, std, etc. from the posterior"
                        if idx == game_idx_stop:
                            posterior_meta    = posterior_data['meta_data']
                            posterior_vectors = posterior_data['param_vectors']

                            if isinstance(posterior_vectors, dict) and posterior_meta.get('representation') == 'particles':
                                final_stats = gnrl._statistics_from_sparse_param_vectors(
                                    param_vectors=posterior_vectors,
                                    meta_data=posterior_meta,
                                    param_info=param_info
                                )
                            else:
                                posterior_array = param_vector_to_pmf_array(
                                    param_vectors=posterior_vectors,
                                    meta_data=posterior_meta,
                                    general_settings=general_settings
                                )
                                tickvals_array = [
                                    posterior_meta["tickvals"][key]
                                    for key in posterior_meta["tickvals"].keys()
                                ]
                                final_stats = gnrl.compute_statistics(
                                    joint_pmf=posterior_array,
                                    grids=tickvals_array,
                                    param_info=param_info
                                )

                            pred_sub['posteriors'] = final_stats

                    else:
                        raise ValueError(f"Only supports update_methods 'parametric','MCMC','grid', not {update_method}.")

                else:
                    "Skip update. Previous param data already copied into current game."
                    pass

        "Always stores 'params' for the role in this game."
        if 'params' not in player_est_dict[role_to_play]:
            player_est_dict[role_to_play]['params'] = copy.deepcopy(role_params_for_this_game)

        idx += 1

    return dyad_games


def simulate_dyad(dyad_games: DyadGames, initial_params_p1_p2: List[Dict[str, Dict[str, float]]], param_info: ParamInfo,
                  utility_settings: UtilitySettings, general_settings: GeneralSettings,
                  select_responses: bool = False) -> List[Dict]:
    """
    Simulates a series of binary dictator games between a pair of
    participants, who alternate between roles as chooser and predictor.

    Arguments:
        • dyad_games: List[Dict]; list of games between a participant pair.
            - Each dict is a "meeting" or "game" in the dyad, containing:
                'chooser', 'predictor', 'payoff_A_chooser', 'payoff_B_chooser',
                'payoff_A_predictor', 'payoff_B_predictor', etc.
        • initial_params_p1_p2: List[Dict[str, Dict[str, float]]]; Parameters for both players.
            - The order of the parameter dicts corresponds to the alphabetical order
                of the player uuids, which are extracted from dyad_games.
        • select_responses: bool;
            Passed through to agent(). If True, agent() samples binary responses and writes
            them into game_dict['choice'] / game_dict['prediction']. If False (default),
            only model probabilities are stored — real participant responses are left untouched.

    Returns:
        • dyad_games: list[dict]; The series of games between a pair of players
            with estimated parameters stored within those games for both players.
    """
    "Validating inputs."
    if not isinstance(dyad_games, (list, tuple)):
        raise ValueError(f"dyad_games must be a list, not {type(dyad_games)}!")

    if not all(isinstance(game, dict) for game in dyad_games):
        raise ValueError(f"dyad_games must be a list of dictionaries, not {type(dyad_games[0])}!")

    first_game = dyad_games[0]
    first_chooser  = first_game.get('chooser', None)
    first_predictor = first_game.get('predictor', None)
    if first_chooser is None or first_predictor is None:
        raise ValueError(f"Failed to extract player uuids from the first game: {first_game}.")

    "Checking if initial_params_p1_p2 contains all required keys."
    for param_dict in initial_params_p1_p2:
        for player_role_key in ['chooser', 'predictor']:
            param_dict_role = param_dict.get(player_role_key, None)
            if param_dict_role is not None:
                "param_keys is a 'global variable' at the top of the file."
                for param_key in param_info["keys"]:
                    if param_key not in param_dict_role:
                        raise ValueError(f"{param_key} missing from initial_params_p1_p2.")

    "Sorting player uuids in initial_params_p1_p2 alphabetically."
    player_uuid_1, player_uuid_2 = sorted([first_chooser, first_predictor])

    "Creating a dictionary of player parameters."
    player_params = {
        player_uuid_1: initial_params_p1_p2[0],
        player_uuid_2: initial_params_p1_p2[1]
    }

    "Iterate agent() for both players over dyad_games one game at a time."
    for meeting_idx in range(len(dyad_games)):
        for player_uuid in [player_uuid_1, player_uuid_2]:
            dyad_games = agent(dyad_games=dyad_games, game_idx_start=meeting_idx, game_idx_stop=meeting_idx,
                               initial_params=player_params[player_uuid], param_info=param_info, utility_settings=utility_settings,
                               player_uuid=player_uuid, general_settings=general_settings,
                               select_responses=select_responses)

    return dyad_games


def loss_function_bayes(dyad_games: list[dict[str, Any]], general_settings: Dict[str, Any]) -> list[dict[str, Any]]:
    """
    Computes per-game loss for each (player, role) and stores intermediate values.
    This does NOT accumulate totals across games here—see create_loss_report for that.
    Stores all data in param_estimates[update_method][player_uuid][player_role]['output'].

    Arguments:
        • dyad_games: list[dict[str, Any]]; List of binary dictator games.
        • general_settings: dict[str, Any]; Various settings used throughout this analysis

    For each game and for each (player, role):
        • raw_ssr: (model prediction - actual response)^2
        • raw_neglogprob: -log(predicted_probability_of_observed_action)
        • confidence: from response(); Confidence is the inverse variance of parameters.
        • param_penalty: from parameter_penalty(); Penalizes large parameter absolute values.
        • loss_final: depends on general_settings['confidence_weighted'] (True/False)
            - If True : final = raw_metric * confidence + param_penalty
            - If False: final = raw_metric + param_penalty

    Returns:
        • dyad_games, with each game’s param_estimates updated.
    """
    confidence_weighted = general_settings.get('confidence_weighted', True)
    loss_funct_type = general_settings.get('loss_funct_type', 'log')
    update_method = general_settings.get('update_method', 'naive')
    penalty_weight = general_settings.get('penalty_weight', 0.01)
    
    for game_dict in dyad_games:
        param_estimates: dict = game_dict.setdefault('parameter_estimates', {}).setdefault(update_method, {})

        "For each player/role in this game, compute raw_srr & raw_neglogprob & final loss"
        for player_uuid, role_dict in param_estimates.items():
            for player_role, stats_dict in role_dict.items():
                model_output: dict = stats_dict.setdefault('output', {})
                if not model_output:
                    continue  # No predictions here.

                "Checks whether a predicted probability is available."
                if player_role == 'chooser':
                    selection = game_dict.get('choice', None)      # Actual choice
                    model_select_A = model_output.get('model_choose_A', None)
                    abdicated = game_dict.get('abdicated_chooser', False)
                else:  # 'predictor'
                    selection = game_dict.get('prediction', None) # Actual prediction
                    model_select_A = model_output.get('model_predict_A', None)
                    abdicated = game_dict.get('abdicated_chooser', False) or game_dict.get('abdicated_predictor', False)

                if selection is None or model_select_A is None or abdicated:
                    "Skip if player abdicated response, no response is found, or no model prediction is found."
                    continue

                selection_val = 1 if selection == 'A' else 0

                "Compute raw residuals"
                raw_ssr = (model_select_A - selection_val)**2
                prob_of_observed = model_select_A if selection_val == 1 else (1 - model_select_A)
                if prob_of_observed <= 0:
                    prob_of_observed = 1e-6
                raw_neglogprob = -math.log(prob_of_observed)

                "The chosen 'residual' depends on whether 'ssr' or 'log'"
                if loss_funct_type == 'ssr':
                    raw_residual = raw_ssr
                else:
                    raw_residual = raw_neglogprob

                "Confidence is inverse parameter variance."
                confidence = model_output.get('confidence', 1.0)

                "Penalize large parameter absolute values."
                param_penalty_val = 0.0
                if 'params' in stats_dict:
                    param_penalty_val = gnrl.parameter_penalty(
                        params=stats_dict['params'],
                        penalty_weight=penalty_weight
                    )
                else:
                    raise Exception(f"'params' not found in stats dict even though model produced a prediction.")

                "final loss depending on confidence_weighted"
                if confidence_weighted:
                    loss_final = raw_residual * confidence + param_penalty_val
                else:
                    loss_final = raw_residual + param_penalty_val

                "Store data"
                model_output['raw_ssr'] = raw_ssr
                model_output['raw_neglogprob'] = raw_neglogprob
                model_output['confidence'] = confidence
                model_output['param_penalty'] = param_penalty_val
                model_output['loss_final'] = loss_final

    return dyad_games


def create_loss_report(dyad_games: list[dict[str, Any]], general_settings: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Aggregates per-game data from loss_function_bayes. Sums them up across 
    all games in the dyad for each (player, role), then stores the results 
    in dyad_games[0]['loss_report'][player_uuid][player_role] = { ... }.

    Notes:
        • If there's no per-game data, sums are zero or none. 
        • This function returns the loss report but also stores it within the first game.

    Arguments:
        • dyad_games: list[dict[str, Any]]; List of binary dictator games.
        • general_settings: dict[str, Any]; Various settings used throughout this analysis

    Returns:
        • loss_report: dict[str, dict[str, dict[str, Any]]] = {
            'player_uuid': {
                'player_role': {
                    'n_data': int,
                    'raw_ssr_sum': float,
                    'raw_neglogprob_sum': float,
                    'param_penalty_sum': float,
                    'loss_final_sum': float,
                    'confidence_weighted': bool,
                    'loss_funct_type': str,
                    'penalty_weight': float,
                    'update_method': str,                    
                },...
            },...
        }
    """
    confidence_weighted = general_settings.get('confidence_weighted', True)
    loss_funct_type = general_settings.get('loss_funct_type', 'log')
    update_method = general_settings.get('update_method', 'grid')
    penalty_weight = general_settings.get('penalty_weight', 0.1)

    "Prepare a top-level dictionary in game[0]"
    first_game = dyad_games[0]
    lr_container: dict = first_game.setdefault('loss_report', {})

    "Accumulate sums in a local structure, then store them"
    sums_dict = {}  # { player_uuid : { role: { n_data, raw_ssr, raw_neglogprob, param_penalty, ... } } }
    for game_dict in dyad_games:
        param_estimates: dict = game_dict.get('parameter_estimates', {}).get(update_method, {})
        for player_uuid, role_dict in param_estimates.items():
            if player_uuid not in sums_dict:
                sums_dict[player_uuid] = {}
            for player_role, stats_dict in role_dict.items():
                if not isinstance(stats_dict, dict):
                    continue
                model_out: dict = stats_dict.get('output', {})
                if not model_out:
                    "Skips abdicated responses."
                    continue
                "Store the values from the final call to loss_function_bayes"
                raw_ssr = model_out.get('raw_ssr', 0.0)
                raw_ll = model_out.get('raw_neglogprob', 0.0)
                param_pen = model_out.get('param_penalty', 0.0)
                final_l = model_out.get('loss_final', 0.0)

                "Init if needed"
                if player_role not in sums_dict[player_uuid]:
                    sums_dict[player_uuid][player_role] = {
                        'n_data': 0,
                        'raw_ssr_sum': 0.0,
                        'raw_neglogprob_sum': 0.0,
                        'param_penalty_sum': 0.0,
                        'loss_final_sum': 0.0
                    }
                sums_dict[player_uuid][player_role]['n_data'] += 1
                sums_dict[player_uuid][player_role]['raw_ssr_sum'] += raw_ssr
                sums_dict[player_uuid][player_role]['raw_neglogprob_sum'] += raw_ll
                sums_dict[player_uuid][player_role]['param_penalty_sum'] += param_pen
                sums_dict[player_uuid][player_role]['loss_final_sum'] += final_l

    "Store them in dyad_games[0]['loss_report']"
    for player_uuid, role_data in sums_dict.items():
        lr_container.setdefault(player_uuid, {})
        for player_role, sums in role_data.items():
            "Store sums and relevant settings."
            lr_container[player_uuid][player_role] = {
                "n_data": sums['n_data'],
                "raw_ssr_sum": sums['raw_ssr_sum'],
                "raw_neglogprob_sum": sums['raw_neglogprob_sum'],
                "param_penalty_sum": sums['param_penalty_sum'],
                "loss_final_sum": sums['loss_final_sum'],
                "confidence_weighted": confidence_weighted,
                "loss_funct_type": loss_funct_type,
                "penalty_weight": penalty_weight,
                "update_method": update_method,
            }

    return lr_container


def fit_params_by_player(player_uuid: PlayerUUID, param_info: ParamInfo, utility_settings: UtilitySettings, 
                              file_paths: FilePaths, general_settings: GeneralSettings) -> None:
    """
    Runs an optimization function to find the social preference parameter values that best fit participants' 
    patterns of choices and predictions. This fits parameters by player across all dyads they participated in.
    
    Arguments:
        • player_uuid: str; Identifies the player for whom parameters are being fit.
        • param_info: dict[str, list[Any]]; Stores parameter keys, boundaries, and initial guesses.
        • utility_settings: dict[str, bool]; Defines the functional form of the utility function.
        • file_paths: dict[str, str | dict[str, str]]; Stores all file paths for storing data.
        • general_settings: dict[str, Any]; Various settings used throughout this analysis.

    Returns:
        • None; Instead saves the results within files.
    """
    time_start_fit_plr = time.time()

    "Extract General Settings"
    experiment_num = general_settings.get('experiment_num', 3)
    update_method = general_settings.get('update_method', True)
    include_covariance = general_settings.get('include_covariance', True)
    default_softmax_temperature = general_settings.get('softmax_temperature', True)
    temperature_is_param = general_settings.get('temperature_is_param', True)
    optimization_method = general_settings.get('optimization_method', 'local')

    if update_method in ('naive', 'parametric'):
        for param_key in param_info['keys']:
            if '_std' in param_key:
                raise ValueError(f"Standard deviation parameter {param_key} discovered during {update_method} update method.")

    if isinstance(player_uuid, int):
        plrs_to_dyads = prep.players_to_dyads(experiment_num=experiment_num, 
                                                    file_paths=file_paths, create_new_file=False)
        plr_keys = sorted(list(plrs_to_dyads.keys()))
        player_uuid = plr_keys[player_uuid % len(plr_keys)]

    player_fit_dir = os.path.join(file_paths["player_fits"], f"experiment_{experiment_num}")
    if experiment_num == 0:
        player_fit_name = f"{player_uuid}.json"
    else:
        player_fit_name = f"{file_paths.get('file_name_suffix','')}_{player_uuid}.json"

    plr_file_path = prep.ensure_directory_and_join(base_dir=player_fit_dir, file_name=player_fit_name)

    "Return cached results if they exist; fall through and re-fit if the JSON is corrupt."
    try:
        if not general_settings.get('create_new_file', False) and os.path.exists(plr_file_path):
            with open(plr_file_path, "r", encoding='utf-8') as file:
                fitted_player_dyads = json.load(file)
            return fitted_player_dyads
    except json.decoder.JSONDecodeError as error:
        print(error)

    player_dyads = prep.dyads_for_a_player(player_uuid=player_uuid, experiment_num=experiment_num, file_paths=file_paths, 
                                                 analysis_mode=general_settings.get('analysis_mode', 'bayesian'), dyad_already_analyzed=False)

    "Build role-specific parameter specs. Both roles start from the same param_info base."
    "Predictor keeps _std (prior-width) params; chooser's _std params are stripped below."
    initial_params = {'chooser': {}, 'predictor': {}}
    for player_role in ('chooser', 'predictor'):
        for param_cat in ('keys', 'bounds', 'guesses'):
            if param_cat == 'guesses':
                initial_params[player_role][param_cat] = param_info['guesses']() \
                    if callable(param_info['guesses']) else copy.deepcopy(param_info['guesses'])
            else:
                initial_params[player_role][param_cat] = copy.deepcopy(param_info[param_cat])
            if include_covariance:
                initial_params[player_role][param_cat] += copy.deepcopy(param_info['covar'][param_cat])
        if temperature_is_param and update_method in ('MCMC', 'grid'):
            initial_params[player_role]['keys'] += ['τ']
            initial_params[player_role]['bounds'] += [(0.5, 3.0)]
            initial_params[player_role]['guesses'] += [default_softmax_temperature]

    "Remove standard deviation parameters from chooser's params."
    # Chooser parameters do not include prior standard deviations. In the UBM, standard deviations
    # describe the width of the predictor's belief distribution about the chooser's parameters;
    # the chooser has no analogous internal uncertainty representation.
    for param_key in list(initial_params['chooser'].keys()):
        if '_std' in param_key:
            del initial_params['chooser'][param_key]

    loss_report = {
        'chooser': [],
        'predictor': []
    }

    def optimize_roles(initial_params_for_role: dict[str, list], role_to_fit: str):
        """
        Run the full Bayesian fitting pipeline for a single role across all of a player's dyads.

        Iterates over every dyad in `player_dyads`, calls `agent()` to propagate the given
        parameter vector through the game sequence, then calls `loss_function_bayes()` to
        score the resulting beliefs.  The accumulated total loss is returned and also stored
        in `loss_report[role_to_fit]` for downstream inspection.

        Arguments:
            • initial_params_for_role: dict[str, list]
                Must contain at least a `'keys'` entry listing the parameter names in the
                same order as the values that the optimizer will pass through
                `objective_function`.
            • role_to_fit: str
                Either `'chooser'` or `'predictor'`; determines which role's parameters
                are being optimized and which entries in `loss_report` are updated.

        Returns:
            • float — total loss across all dyads for the given role and parameter vector.
        """

        def objective_function(param_array: NDArray[np.float64]) -> float:
            """
            Convert a flat parameter array into a total loss score across all dyads.

            Called by the optimizer at each evaluation step.  Unpacks `param_array` into
            a named parameter dict, optionally repairs covariance structure via
            `transform_cov_params`, then runs `agent()` and `loss_function_bayes()` over
            every dyad in `player_dyads`, accumulating the per-dyad `loss_final` values
            into a single scalar.

            Arguments:
                • param_array: NDArray[np.float64]
                    Flat array of parameter values in the order defined by
                    `initial_params_for_role['keys']`.  If `temperature_is_param` is True,
                    the final element is treated as `softmax_temperature`.

            Returns:
                • float — total negative log-likelihood loss across all dyads.
            """
            param_array = copy.deepcopy(param_array)
            if isinstance(param_array, np.ndarray):
                param_array = param_array.tolist()

            param_array: list
            if temperature_is_param:
                softmax_temperature = param_array[-1]
            else:
                softmax_temperature = default_softmax_temperature

            role_params = {param_key: param_val for param_key, param_val in zip(initial_params_for_role['keys'], param_array)}

            if include_covariance and role_to_fit == 'predictor':
                safe_pred_params = gnrl.transform_cov_params(params=role_params, param_info=param_info)
                if safe_pred_params['loss'] is not None:
                    print(safe_pred_params['report'])
                    return safe_pred_params['loss']
                role_params = safe_pred_params['params']

            total_plr_loss = 0.0
            aggregated_sums = {
                'n_data': 0,
                'raw_ssr_sum': 0.0,
                'raw_neglogprob_sum': 0.0,
                'param_penalty_sum': 0.0,
                'loss_final_sum': 0.0
            }

            for dyad_key, dyad_games in player_dyads.items():
                "Run agent function with parameters in param_array"
                dyad_games_copy = copy.deepcopy(dyad_games) 
                updated_games = agent(dyad_games=dyad_games_copy,
                                        game_idx_start=0,
                                        game_idx_stop=len(dyad_games)-1,
                                        initial_params={role_to_fit: role_params},
                                        param_info=param_info,
                                        utility_settings=utility_settings,
                                        player_uuid=player_uuid,
                                        player_role=role_to_fit,
                                        general_settings=general_settings,
                                        softmax_temperature=softmax_temperature)
                
                if general_settings.get('include_covariance') and role_to_fit == 'predictor':
                    prior_param_vector = updated_games[0].get('parameter_estimates', {}).get('grid', {}).get(
                        player_uuid, {}).get('predictor', {}).get('param_vectors', None)
                    "Preventing error with degenerate priors that have all zeros and a few huge probabilities."
                    if isinstance(prior_param_vector, dict):
                        for probability in prior_param_vector.values():
                            if probability > 0.2:
                                print(f"Detected massive probability {probability}")
                                return 1e6
                                
                "Compute loss (using loss_function_bayes)."
                updated_games = loss_function_bayes(dyad_games=updated_games, general_settings=general_settings)
                loss_sums: dict = create_loss_report(dyad_games=updated_games, general_settings=general_settings).get(player_uuid, {}).get(role_to_fit, {})
                total_plr_loss += loss_sums.get('loss_final_sum', 0.0)

                "accumulate into aggregated_sums"
                aggregated_sums['n_data'] += loss_sums.get('n_data', 0)
                aggregated_sums['raw_ssr_sum'] += loss_sums.get('raw_ssr_sum', 0.0)
                aggregated_sums['raw_neglogprob_sum'] += loss_sums.get('raw_neglogprob_sum', 0.0)
                aggregated_sums['param_penalty_sum'] += loss_sums.get('param_penalty_sum', 0.0)
                aggregated_sums['loss_final_sum'] += loss_sums.get('loss_final_sum', 0.0)

            "Build the row => param_key => param_val plus the aggregated sums"
            row = {}
            "Add param_key: param_val"
            for pkey, pval in zip(initial_params_for_role['keys'], param_array):
                row[pkey] = pval
            "Add aggregated sums"
            row.update(aggregated_sums)
            loss_report[role_to_fit].append(row)

            return total_plr_loss

        def objective_function_raw_nll(param_array: NDArray[np.float64]) -> float:
            """
            Raw NLL objective for the same parameterization, *without* any parameter penalty.
            This reads the per-dyad raw_neglogprob_sum and aggregates across all dyads.
            """
            param_array = np.asarray(param_array, dtype=float)
            if temperature_is_param:
                softmax_temperature_local = float(param_array[-1])
            else:
                softmax_temperature_local = float(default_softmax_temperature)

            role_params_local = {param_key: float(param_val) for param_key, param_val in zip(initial_params_for_role['keys'], param_array)}
            if include_covariance and role_to_fit == 'predictor':
                safe_pred = gnrl.transform_cov_params(params=role_params_local, param_info=param_info)
                if safe_pred['loss'] is not None:
                    "invalid covariance parametrization — return a large penalty-like number"
                    return float(safe_pred['loss'])
                role_params_local = safe_pred['params']

            total_raw_nll = 0.0
            for dyad_key, dyad_games_local in player_dyads.items():
                games_copy = copy.deepcopy(dyad_games_local)
                "run the agent with these parameters"
                updated_games_local = agent(
                    dyad_games=games_copy,
                    game_idx_start=0,
                    game_idx_stop=len(dyad_games_local)-1,
                    initial_params={role_to_fit: role_params_local},
                    param_info=param_info,
                    utility_settings=utility_settings,
                    player_uuid=player_uuid,
                    player_role=role_to_fit,
                    general_settings=general_settings,
                    softmax_temperature=softmax_temperature_local
                )
                "compute losses"
                updated_games_local = loss_function_bayes(dyad_games=updated_games_local, general_settings=general_settings)
                loss_sums_local: dict = create_loss_report(
                    dyad_games=updated_games_local,
                    general_settings=general_settings
                ).get(player_uuid, {}).get(role_to_fit, {})
                total_raw_nll += float(loss_sums_local.get('raw_neglogprob_sum', 0.0))

            return float(total_raw_nll)

        best_fitting_params = {}
        optimization_results = {}

        bounds = initial_params[role_to_fit]['bounds']
        guesses = initial_params[role_to_fit]['guesses']

        if update_method == 'grid' and role_to_fit == 'predictor' and not general_settings.get('no_memory_mode', False):
            "Using parameters fitted on the 'naive' update method as the initial parameter guesses."
            general_settings_ = copy.deepcopy(general_settings)
            general_settings_['update_method'] = 'naive'
            general_settings_['run_in_parallel'] = True
            file_name_suffix_naive = prep.create_file_name_suffix(general_settings=general_settings_, utility_settings=utility_settings)
            directory_path_naive = ensure_directory_and_join(file_paths['player_fits'], f'experiment_{experiment_num}')
            file_name_naive = file_name_suffix_naive + f"_{player_uuid}.json"
            file_path_naive = prep.ensure_directory_and_join(base_dir=directory_path_naive, file_name=file_name_naive)
            if os.path.exists(file_path_naive):
                player_histories_naive = None
                with open(file_path_naive, "r", encoding="utf-8") as file:
                    player_histories_naive = json.load(file)
                if isinstance(player_histories_naive, dict):
                    dyad_keys = list(player_histories_naive.keys())
                    if len(dyad_keys) > 0:
                        first_dyad = player_histories_naive[dyad_keys[0]]
                        if len(first_dyad) > 0:
                            first_game: dict = first_dyad[0]
                            if first_game.get('predictor') == player_uuid and 'naive' in first_game.get('parameter_estimates', {}):
                                params: dict = first_game['parameter_estimates']['naive'].get(player_uuid, {}).get('predictor', {}).get('params', {})
                                educated_guesses = copy.deepcopy(guesses)
                                for idx, param_key in enumerate(param_info['keys']):
                                    fitted_param = params.get(param_key, None)
                                    if isinstance(fitted_param, (float, int)) and (
                                        param_info['bounds'][idx][0] <= fitted_param <= param_info['bounds'][idx][1]):
                                        educated_guesses[idx] = fitted_param
                                guesses = educated_guesses

        "--- Child→Parent warm-start (consume the function's prepared guesses) -----"
        opt_method_local = str(general_settings.get('optimization_method', 'globloc')).lower()
        warm_pol = general_settings.get("warmstart_policy", {}) or {}
        guesses_before = list(map(float, np.array(guesses, dtype=float)))

        "Record what happened for the JSON report."
        warm_meta = {
            "enabled": bool(warm_pol.get("enabled", True)),
            "phase":   warm_pol.get("phase", "cold"),
            "temperature": float(warm_pol.get("temperature", 0.0)),
            "x_initial_guess_before": guesses_before,
            "x_initial_guess_after":  None,
            "optimization_method_effective": None,
            "selected_child_key": None,
            "exception": None, "used": False,
            "model_bit_string": gnrl.convert_utility_settings(utility_settings=utility_settings, into=str),
            "model_equation": build_utility_equation(utility_settings=utility_settings),
            "model_utility_settings": gnrl.convert_utility_settings(utility_settings=utility_settings, into=dict),
        }

        "--- Degenerate model guard: no free parameters for this role -----------------"
        if len(bounds) == 0:
            "Evaluate once at the empty vector so loss_report is still populated."
            baseline_x_vector = np.array([], dtype=float)
            baseline_loss_value = float(objective_function(baseline_x_vector))

            "Build a minimal \"optimization\" report so downstream code doesn't break."
            gltc_result = {
                "final": {"x": [], "fun": baseline_loss_value},
                "random_search": {"x_initial_guess": []},
                "local": {"success": True, "message": "No free parameters for this role."},
                "warmstart_meta": {
                    **warm_meta,
                    "optimization_method_effective": "none",
                    "x_initial_guess_before": [],
                    "x_initial_guess_after":  []
                }
            }
            optimization_results[role_to_fit] = gltc_result
            best_fitting_params[role_to_fit] = {}  # Nothing to fit for this role.
            return best_fitting_params, optimization_results

        elif len(bounds) == 1:
            "Base case: one free parameter — skip warm-start, still optimize."
            warm_meta["enabled"] = False
            warm_meta["phase"]   = "cold"
            warm_meta["note"] = "One-parameter model: keeping dual_annealing enabled."
            
            "Single-parameter models should always explore globally."
            opt_method_local = "globloc"       

        if warm_meta["enabled"] and warm_meta["phase"] == "warm":
            from analysis import best_fitting_child_parameters_for_parent
            warm_start = best_fitting_child_parameters_for_parent(
                player_uuid=player_uuid,
                player_role=role_to_fit,
                utility_settings_parent=utility_settings,   # Parent is the current model.
                utility_settings=utility_settings,          # OK (universe of flags).
                general_settings=general_settings,
                file_paths=file_paths,
                param_bds=param_bds,                        # Module/global.
                within_ic_analysis=True,
                temperature=warm_meta["temperature"]
            )

            "IMPORTANT: use the parent-space warmstart prepared inside warm_start"
            parent_ws = (warm_start or {}).get("parent_warmstart", {})
            ws_params_for_role = parent_ws.get(player_uuid, {}).get(role_to_fit)

            "Optionally record which child was selected (bitstring)"
            warm_meta["selected_child_key"] = (
                ((warm_start or {}).get("selected_child", {}) or {})
                .get("metadata", {}).get("model_bit_str", None)
            )

            warm_meta["child_equation"] = warm_start.get("metadata", {}).get("U_funct")
            warm_meta["model_equation"] = warm_start.get("metadata", {}).get("parent_equation")

            if isinstance(ws_params_for_role, dict) and ws_params_for_role:
                warm_guess = []
                if "guesses" in ws_params_for_role:
                    warm_guess = ws_params_for_role["guesses"]

                else:
                    for idx, param_key in enumerate(initial_params_for_role["keys"]):
                        param_val = ws_params_for_role.get(param_key, None)
                        if isinstance(param_val, (int, float)):
                            warm_guess.append(float(param_val))
                        else:
                            warm_guess = []
                            break

                if warm_guess:
                    guesses = np.array(warm_guess, dtype=float)
                    warm_meta["used"] = True

                    "If requested, force local-only when using warm-starts "
                    if warm_pol.get("disable_dual_annealing_when_warm", True):
                        opt_method_local = "local"

        warm_meta["x_initial_guess_after"] = list(map(float, np.array(guesses, dtype=float)))
        warm_meta["optimization_method_effective"] = opt_method_local

        if len(bounds) != len(guesses):
            print("Param Keys:", initial_params[role_to_fit]['keys'])
            raise Exception(f"len(bounds) = {len(bounds)} != len(guesses) = {len(guesses)}")

        default_optimization_policy = {
            'n_random_starts'    : 1,
            'maxiter_global'     : 24,
            'maxiter_local'      : 24,
            'maxfun_global'      : 24,
            'maxfun_local'       : 24,
            'run_trust_constr'   : False,
            'dual_annealing_seed': None,
            'trust_maxiter'      : 600,
            'trust_gtol'         : 1e-6,
            'trust_xtol'         : 1e-8,
            'trust_verbose'      : False        
        }

        optimization_policy = general_settings.get('optimization_policy', default_optimization_policy)

        "If exactly one dimension, make global sampling a bit deeper and ensure multiple random starts."
        if len(bounds) == 1:
            optimization_policy['n_random_starts'] = max(optimization_policy.get('n_random_starts', 1), 5)
            maxiter_global = optimization_policy.get('maxiter_global', 24)
            maxfun_global = optimization_policy.get('maxfun_global', 24)
            if isinstance(maxfun_global, int):
                maxiter_global = max(maxiter_global, 64)
            if isinstance(maxfun_global, int):
                maxfun_global = max(maxfun_global, 200)
            optimization_policy['maxiter_global']  = maxiter_global
            optimization_policy['maxfun_global']   = maxfun_global

        "--- Two-step optimizer: robust penalized search, then optional constrained raw-NLL refine ---"
        gltc_result = global_local_then_trust_constr(
            objective_with_penalty = objective_function,           # Existing penalized objective.
            objective_raw_nll      = objective_function_raw_nll,   # New raw NLL objective.
            x_bounds               = bounds,
            x_initial_guess        = guesses,
            optimization_method    = opt_method_local,
            parameter_keys         = initial_params_for_role['keys'],
            local_methods          = optimization_policy.get('local_methods', None),
            n_random_starts        = optimization_policy.get('n_random_starts', 1),
            maxiter_global         = optimization_policy.get('maxiter_global', 24),
            maxiter_local          = optimization_policy.get('maxiter_local', 24),
            maxfun_global          = optimization_policy.get('maxfun_global', 24),
            maxfun_local           = optimization_policy.get('maxfun_local', 24),
            run_trust_constr       = optimization_policy.get('run_trust_constr', False),
            dual_annealing_seed    = optimization_policy.get('dual_annealing_seed', None),
            trust_maxiter          = int(optimization_policy.get('trust_maxiter', 600)),
            trust_gtol             = float(optimization_policy.get('trust_gtol', 1e-6)),
            trust_xtol             = float(optimization_policy.get('trust_xtol', 1e-8)),
            trust_verbose          = bool(optimization_policy.get('trust_verbose', False))
        )

        "later, after gltc_result is created:"
        gltc_result["warmstart_meta"] = {
            **warm_meta,
            "x_initial_guess_before": guesses_before,
            "x_initial_guess_after":  np.array(guesses, float).tolist(),
            "optimization_method_effective": opt_method_local
        }
        "and add the actual initial guess to the random_search section:"
        if "random_search" not in gltc_result:
            gltc_result["random_search"] = {}
        gltc_result["random_search"]["x_initial_guess"] = np.array(guesses, float).tolist()

        "Record the pipeline reports for this role"
        optimization_results[role_to_fit] = gltc_result

        "--- Chooses final point by *raw NLL* among all evaluated points. ----"
        if loss_report[role_to_fit]:
            "(a) best raw NLL seen during *any* penalized objective call"
            min_row = min(loss_report[role_to_fit], key=lambda r: r.get('raw_neglogprob_sum', float('inf')))
            raw_min_seen = float(min_row.get('raw_neglogprob_sum', float('inf')))
            x_minraw = np.array(
                [min_row[param_key] for param_key in initial_params_for_role['keys'] if param_key in min_row],
                dtype=float
            )

            "(b) raw NLL at the optimizer's final x"
            x_final = np.asarray(gltc_result['final']['x'], float)
            raw_at_final = float(objective_function_raw_nll(x_final))

            "(c) if the observed raw-min beats the optimizer's final, override final"
            if raw_min_seen + 1e-8 < raw_at_final:
                gltc_result['final'].update({
                    "x": x_minraw.tolist(),
                    "loss": raw_min_seen,
                    "chosen_optimizer": gltc_result['final'].get("chosen_optimizer", "globloc") + "+raw-min-override"
                })

        best_vector = np.asarray(gltc_result['final']['x'], dtype=float)
        best_fitting_params[role_to_fit] = {
            param_key: float(param_val)
            for param_key, param_val in zip(initial_params_for_role['keys'], best_vector)  
        }

        return best_fitting_params, optimization_results

    "1) Fit all roles: run optimize_roles for predictor (and chooser if experiment 3)."
    best_fitting_params = {}
    optimization_results = {}
    for player_role in ('predictor', 'chooser', ):
        if player_role == 'chooser' and experiment_num != 3:
            continue
        "fit_predictor_role=False skips the expensive UBM pass — used by run_param_recovery_by_k when only chooser recovery is needed."
        if player_role == 'predictor' and not general_settings.get('fit_predictor_role', True):
            continue

        best_fit_params_role, opt_results_role = optimize_roles(
            initial_params_for_role=initial_params[player_role], role_to_fit=player_role)
        best_fitting_params[player_role] = best_fit_params_role[player_role]
        optimization_results[player_role] = opt_results_role[player_role]

    "3) Build & save the CSV. Done for each role, or can be unified if preferred."
    for role_to_fit in ('chooser','predictor'):
        if not loss_report[role_to_fit]:
            continue  # Maybe it is empty for roles not optimized.

        df_loss = pd.DataFrame(loss_report[role_to_fit])
        if "raw_neglogprob_sum" in df_loss.columns:
            df_loss.sort_values("raw_neglogprob_sum", inplace=True)

        base_loss_dir = os.path.join(file_paths["player_fits"], "loss_reports", f"experiment_{experiment_num}")
        if experiment_num == 0:
            file_name_loss = f"Loss-{role_to_fit[0]}-{player_uuid}.csv"
        else:
            file_name_loss = f"Loss{file_paths.get('file_name_suffix','')}-{role_to_fit[0]}-{player_uuid}.csv"

        csv_path = prep.ensure_directory_and_join(base_dir=base_loss_dir, file_name=file_name_loss)
        df_loss.to_csv(csv_path, index=False, encoding='utf-8-sig')

    "2) Final agent pass with best-fit params — re-runs agent over every dyad to capture"
    "   the complete belief trajectories and per-round losses needed for the JSON output."
    fitted_plr_dyads = {}
    for dyad_key, dyad_games in player_dyads.items():

        if temperature_is_param:
            softmax_temperature = default_softmax_temperature
            chooser_params = best_fitting_params.get('chooser', {})
            choice_temp = chooser_params.get('τ', chooser_params.get('temp'))
            if choice_temp is not None:
                softmax_temperature = choice_temp
            else:
                predictor_params = best_fitting_params.get('predictor', {})
                choice_temp = predictor_params.get('τ', predictor_params.get('temp'))
                if choice_temp is not None:
                    softmax_temperature = choice_temp
        else:
            softmax_temperature = default_softmax_temperature

        "Run agent function with parameters in param_array"
        "When predictor fitting was skipped, restrict the replay to chooser rounds only."
        final_pass_role = None if general_settings.get('fit_predictor_role', True) else 'chooser'
        fitted_dyad_games = agent(dyad_games=dyad_games,
                                game_idx_start=0,
                                game_idx_stop=len(dyad_games)-1,
                                initial_params=best_fitting_params,
                                param_info=param_info,
                                utility_settings=utility_settings,
                                player_uuid=player_uuid,
                                player_role=final_pass_role,
                                general_settings=general_settings,
                                softmax_temperature=softmax_temperature)
        
        "Compute loss (using loss_function_bayes)."
        fitted_dyad_games = loss_function_bayes(
            dyad_games=fitted_dyad_games, general_settings=general_settings)

        loss_sum = 0.0
        fitted_dyad_games[0]['loss_report'] = create_loss_report(dyad_games=fitted_dyad_games, general_settings=general_settings)
        loss_sum += fitted_dyad_games[0].get('loss_report', {}).get(player_uuid, {}).get('chooser', {}).get('loss_final_sum', 0.0)
        loss_sum += fitted_dyad_games[0].get('loss_report', {}).get(player_uuid, {}).get('predictor', {}).get('loss_final_sum', 0.0)

        "Making Numpy arrays JSON serializable."
        if update_method == 'grid':
            fitted_dyad_games = prep.serialize_or_drop_param_vectors(
                dyad_games=fitted_dyad_games, general_settings=general_settings)

        fitted_dyad_games[0]['reports'] = {
            'chooser': optimization_results.get('chooser', None),
            'predictor': optimization_results.get('predictor', None)
        }

        "Extract the parameters that minimize raw_neglog_sum from the loss report and put them within fitted_dyad_games[0]['reports']"
        raw_loss_minimizing_params = {}
        for player_role in ('predictor', 'chooser', ):
            raw_loss_minimizing_params[player_role] = {}
            param_loss_list: list[dict[str: int | float]] = copy.deepcopy(loss_report[player_role])
            if len(param_loss_list) > 0:
                if "raw_neglogprob_sum" in param_loss_list[0]:
                    param_loss_list = sorted(param_loss_list, key=lambda loss_row: loss_row.get('raw_neglogprob_sum', 0.0))
                    dict_with_raw_loss_minimizing_params = param_loss_list[0]
                    raw_neglogprob_sum = dict_with_raw_loss_minimizing_params["raw_neglogprob_sum"]
                    for param_key in param_info['keys']:
                        param_val = dict_with_raw_loss_minimizing_params.get(param_key, None)
                        if isinstance(param_val, (int, float)):
                            raw_loss_minimizing_params[player_role][param_key] = param_val
                if len(fitted_dyad_games) > 0:
                    if 'reports' in fitted_dyad_games[0]:
                        if player_role in fitted_dyad_games[0]['reports']:
                            if "final" in fitted_dyad_games[0]['reports'][player_role]:
                                min_raw_neglog_sum = {
                                    "params": raw_loss_minimizing_params[player_role],
                                    "loss": raw_neglogprob_sum
                                }
                                fitted_dyad_games[0]['reports'][player_role]["final"]["min_raw_neglog_sum"] = min_raw_neglog_sum

        "Experiment 2: human vs. avatar dyads — compute avatar-type posteriors for the visualization pipeline."
        if general_settings.get('experiment_num') == 2:
            fitted_dyad_games = typo.avatar_posteriors(
                dyad_games=fitted_dyad_games, update_method=update_method,
                temperature=default_softmax_temperature)

        fitted_plr_dyads[dyad_key] = fitted_dyad_games

    def _serialize_particle_filter_state(dyad_games_by_key: dict[str, DyadGames]) -> None:
        for dyad_games in dyad_games_by_key.values():
            for dyad_game in dyad_games:
                grid_estimates = dyad_game.get('parameter_estimates', {}).get('grid', {})
                for player_estimates in grid_estimates.values():
                    for role_estimates in player_estimates.values():
                        meta_data = role_estimates.get('meta_data')
                        if not meta_data or meta_data.get('representation') != 'particles':
                            continue
                        pf_state = meta_data.get('pf_state')
                        if not pf_state:
                            continue
                        indices = pf_state.get('indices')
                        weights = pf_state.get('weights')
                        if isinstance(indices, np.ndarray):
                            pf_state['indices'] = indices.tolist()
                        if isinstance(weights, np.ndarray):
                            pf_state['weights'] = weights.tolist()
    "Convert any numpy arrays inside PF state to plain lists so json.dump can serialize them."
    _serialize_particle_filter_state(fitted_plr_dyads)

    "Save the fitted results."
    with open(plr_file_path, 'w', encoding='utf-8') as file:
        json.dump(fitted_plr_dyads, file, ensure_ascii=False, indent=4)


def fit_dyad_parameters_bayes(dyad_games: DyadGames, param_info: ParamInfo, utility_settings: UtilitySettings, 
                              file_paths: FilePaths, general_settings: GeneralSettings, fit_choosers_exper_1and2: bool = False) -> list[dict[str, Any]]:
    """
    Fit parameters for each player in a dyad by optimizing the loss function over the series of games.
    
    This function can either fit the chooser and predictor parameters simultaneously or, if
    fit_roles_together is False, iteratively fit one role at a time (i.e., thawed parameters are
    optimized while frozen parameters remain fixed). This is done separately for each player.
    
    Arguments:
        • dyad_games: list[dict]; Sequence of games (meetings) between two players.
        • param_info: ParamInfo; Contains parameter keys, bounds, and initial guesses.
            Example:
                {
                    "keys": ["Vᵢᵢ", "Vᵢⱼ", "αᵢⱼ", "βᵢⱼ", "exp1"]  # Plus '_std' and '_cov' keys if used.
                    "bounds": [(lower, upper), ...],
                    "guesses": callable or list of floats
                }
        • utility_settings: UtilitySettings; Defines utility model configuration.
        • file_paths: dict; File paths for processed data and individual dyad data.
        • general_settings: GeneralSettings;
            Various settings bundled into one dictionary.
            - experiment_num: int; Experiment identifier.
            - update_method: str; e.g., 'grid', 'parametric', etc.
            - create_new_file: bool; If False, load existing fitted data.
            - learning_rate: float; Learning rate for parameter optimization.
            - n_bins_per_dimension: int; Passed to agent() for grid dimensions.
            - sample_ratio: float; Passed to agent(); if 1, use L-BFGS-B; otherwise, use simulated annealing.
            - fit_roles_together: bool; If False, fit each player's roles separately (thaw one role at a time).
            - include_covariance: bool; If True, include covariance parameters (keys with '_cov') in the fitting.
        • fit_choosers_exper_1and2: Only if True, will fit parameters to robotic players in experiments 1 and 2.
    
    Returns:
        • dict; Updated dyad_games with fitted parameters and associated loss, as well as the best parameter estimates.
    """
    time_start_fit_dyad = time.time()

    "Determine dyad file path and attempt to load if create_new_file is False."
    first_game = dyad_games[0]
    first_choo = first_game.get('chooser')
    first_pred = first_game.get('predictor')
    if first_choo is None or first_pred is None:
        raise ValueError(f"Failed to extract player uuids from the first game: {first_game}")
    player_uuids = sorted([first_choo, first_pred])
    
    "Extract General Settings"
    sample_ratio = general_settings.get('sample_ratio', True)
    experiment_num = general_settings.get('experiment_num', 3)
    learning_rate = general_settings.get('learning_rate', True)
    update_method = general_settings.get('update_method', True)
    fit_roles_together = general_settings.get('fit_roles_together', True)
    include_covariance = general_settings.get('include_covariance', True)
    n_bins_per_dimension = general_settings.get('n_bins_per_dimension', True)
    default_softmax_temperature = general_settings.get('softmax_temperature', True)
    temperature_is_param = general_settings.get('temperature_is_param', True)
    
    dyad_file_path = prep._dyad_file_path(dyad_key=tuple(player_uuids), file_paths=file_paths, 
                                                 experiment_num=experiment_num, analysis_mode='bayesian')
    try:
        if not general_settings.get('create_new_file', False) and os.path.exists(dyad_file_path):
            with open(dyad_file_path, "r", encoding='utf-8') as file:
                dyad_history = json.load(file)
            if dyad_history:
                return dyad_history
    except json.decoder.JSONDecodeError as error:
        print(error)

    "--- Get Initial Parameter Guesses ---"
    initial_params = best_initial_guesses(dyad_key=tuple(player_uuids), #NOTE Check if this still works. 
                                          file_paths=file_paths,
                                          param_info=param_info, 
                                          general_settings=general_settings)

    best_params_overall = copy.deepcopy(initial_params)

    def optimize_role(player_uuid: str, role_to_fit: str, fixed_params: dict) -> dict:
        """
        Optimize the parameters for player_uuid for role role_to_fit (thawed parameters),
        while keeping the other role frozen (from fixed_params).
        
        fixed_params: dict with keys 'chooser' and 'predictor' for that player.
        Returns the updated parameter dict for that player.
        """
        def objective_function(free_param_array: NDArray[np.float64]) -> float:
            """Objective function that only modifies the thawed parameters."""
            "Build updated parameter set for this player."
            updated = copy.deepcopy(fixed_params)
            for idx, key in enumerate(free_keys):
                updated[role_to_fit][key] = free_param_array[idx]

            if include_covariance and role_to_fit == 'predictor':
                "Ensure covariance matrix is symmetric and PSD."
                huge_loss      = 1e6
                is_psd_tol     = 1e-12
                asymmetry_tol  = 1e-12
                minimum_eigval = 1e-6
                cov_matrix = gnrl.build_covariation_matrix(
                    param_info=param_info,
                    params=updated[role_to_fit],
                    raise_on_invalid=False  
                )
                altered_cov_matrix = False 
                if not gnrl.is_positive_semidefinite(matrix=cov_matrix, tol=is_psd_tol):    
                    cov_matrix = gnrl.nearest_psd_matrix(matrix=cov_matrix, min_eigval=minimum_eigval)
                    altered_cov_matrix = True

                    "Ensure that the altered matrix respects parameter boundaries."
                    param_mean_keys = [param_key for param_key in param_info['keys'] if '_std' not in param_key]
                    for idx, param_key in enumerate(param_mean_keys):
                        std_key = param_key + "_std"
                        if std_key not in param_info["keys"]:
                            continue  # No std?
                        idx_std = param_info["keys"].index(std_key)
                        (lower_bound, upper_bound) = param_info["bounds"][idx_std]

                        "'Repaired' std is square root of diagonal matrix value at (idx, idx)."
                        stdev = np.sqrt(cov_matrix[idx, idx]) 
                        if not (lower_bound <= stdev <= upper_bound):
                            if lower_bound >= stdev:
                                cov_matrix[idx, idx] = round(lower_bound ** 2, 9)
                            else:
                                cov_matrix[idx, idx] = int(upper_bound ** 2)

                    "Ensure that the altered matrix is symmetric."
                    for idx, key1 in enumerate(param_mean_keys):
                        for jdx, key2 in enumerate(param_mean_keys[idx + 1:], start=idx + 1):
                            if abs(cov_matrix[idx][jdx] - cov_matrix[jdx][idx]) > asymmetry_tol:
                                cov_matrix[idx][jdx] = cov_matrix[jdx][idx]

                if not gnrl.is_positive_semidefinite(matrix=cov_matrix, tol=is_psd_tol):
                    print("[objective] Covariance not PSD => penalty.")
                    return huge_loss

                try:
                    param_means = [param_val for param_key, param_val in updated[role_to_fit].items() 
                                   if not any(key in param_key for key in ('_std', '_cov', 'τ', 'temp'))]
                    multivariate_normal(mean=param_means, cov=cov_matrix, allow_singular=False)
                except np.linalg.LinAlgError:
                    print(f"Failed Multivariate Normal:")
                    print(cov_matrix)
                    return huge_loss
                
                if altered_cov_matrix:
                    "Updating the param dictionary with altered parameters."
                    param_mean_keys = [param_key for param_key in param_info['keys'] if '_std' not in param_key]
                    for idx, key1 in enumerate(param_mean_keys):
                        updated[role_to_fit][key1 + '_std'] = math.sqrt(cov_matrix[idx][idx])
                        for jdx, key2 in enumerate(param_mean_keys[idx + 1:], start=idx + 1):      
                            cov_key = f"{key1}_{key2}_cov"
                            updated[role_to_fit][cov_key] = cov_matrix[idx][jdx]

            "Construct a full parameter dictionary for this player."
            "Uses initial parameters for the other player."
            full_param_dict = {player_uuid: updated}

            "For the other player, use the original guess."
            partner_uuid = [uuid for uuid in player_uuids if uuid != player_uuid][0]
            full_param_dict[partner_uuid] = initial_params[partner_uuid]
            "Run the agent simulation for the entire dyad."
            dyad_copy = copy.deepcopy(dyad_games)

            if temperature_is_param and role_to_fit == 'predictor' and update_method == 'grid':
                pred_pd = full_param_dict[player_uuid]['predictor']
                softmax_temperature = pred_pd.get('τ', pred_pd.get('temp'))
            else:
                softmax_temperature = None

            "agent() is assumed to use the provided initial_params for the given player."
            updated_games = agent(dyad_games=dyad_copy,
                                  game_idx_start=0,
                                  game_idx_stop=len(dyad_copy)-1,
                                  initial_params=full_param_dict[player_uuid],
                                  param_info=param_info,
                                  utility_settings=utility_settings,
                                  player_uuid=player_uuid,
                                  player_role=role_to_fit,
                                  general_settings=general_settings,
                                  softmax_temperature=softmax_temperature)
            "Compute loss (using loss_function_bayes)."
            updated_games = loss_function_bayes(dyad_games=updated_games, general_settings=general_settings)
            loss_val = sum_of_all_loss(updated_games, update_method=update_method,
                                       target_player=player_uuid, target_role=role_to_fit)
            return loss_val
        
        time_start_obj_func = time.time()
        free_keys = [key for key in param_info["keys"]]
        param_bounds = [key for key in param_info["bounds"]]
        if include_covariance and role_to_fit == 'predictor':
            free_keys += [key for key in param_info["covar"]["keys"]]
            param_bounds += [key for key in param_info["covar"]["bounds"]]
        if temperature_is_param:
            free_keys += ['τ']
            param_bounds += [(0.5, 3.0)]

        "Extract initial free parameters for the active (thawed) role."
        free_initial_vector = np.array([fixed_params[role_to_fit][key] for key in free_keys])

        "Choose optimizer based on sample_ratio."
        if sample_ratio == 1:
            optimizer_method = "L-BFGS-B"
            opt_result = minimize(fun=objective_function,
                                  x0=free_initial_vector,
                                  bounds=param_bounds,
                                  method=optimizer_method,
                                  options={'maxiter': 300, 'ftol': 1e-4})
        else:
            "Otherwise, use simulated annealing."
            opt_result = dual_annealing(func=objective_function, bounds=param_bounds, maxiter=50)
        print(f"Role: {role_to_fit}")
        print(opt_result)
        best_free = opt_result.x
        "Update fixed_params with optimized values for role_to_fit."
        for idx, key in enumerate(free_keys):
            fixed_params[role_to_fit][key] = best_free[idx]

        time_stop_obj_func = time.time()
        duration = time_stop_obj_func - time_start_obj_func
        serialized_report = gnrl.serialize_opt_result(opt_result, duration=duration)
        "Store the report in the first game of the dyad under game_dict['reports'][player_uuid][role_to_fit]"
        dyad_games[0].setdefault("reports", {}).setdefault(player_uuid, {})[role_to_fit] = serialized_report

        return fixed_params

    def sum_of_all_loss(dyad_games: list, update_method: str, target_player: str, target_role: str) -> float:
        """Helper to sum losses for a given player/role."""
        total_loss = 0.0
        for game in dyad_games:
            param_data = game.get('parameter_estimates', {}).get(update_method, {})
            if target_player in param_data:
                total_loss += param_data[target_player].get(target_role, {}).get('output', {}).get('loss', 0)
        return total_loss

    "Iterate over players and roles."
    for player in player_uuids:
        if not fit_roles_together:
            if general_settings.get('experiment_num') == 3 or fit_choosers_exper_1and2:
                "First, optimize the chooser parameters for this player."
                best_params_overall[player] = optimize_role(player_uuid=player, role_to_fit="chooser", fixed_params=best_params_overall[player])  
                pp.pprint(best_params_overall[player])
            if general_settings.get('experiment_num') == 3 or first_pred == player:
                "Then, optimize the predictor parameters for this player."
                best_params_overall[player] = optimize_role(player_uuid=player, role_to_fit="predictor", fixed_params=best_params_overall[player])
                pp.pprint(best_params_overall[player])
        else:
            def objective_function_joint(free_param_array: NDArray[np.float64]) -> float:
                """Optimize both roles simultaneously."""
                updated = copy.deepcopy(best_params_overall[player])
                n_params = len(free_keys)
                for idx, key in enumerate(free_keys):
                    updated["chooser"][key] = free_param_array[idx]
                    updated["predictor"][key] = free_param_array[n_params+idx]
                full_param_dict = {player: updated}
                partner = [uuid for uuid in player_uuids if uuid != player][0]
                full_param_dict[partner] = initial_params[partner]
                dyad_copy = copy.deepcopy(dyad_games)
                updated_games = agent(dyad_games=dyad_copy,
                                      game_idx_start=0,
                                      game_idx_stop=len(dyad_copy)-1,
                                      initial_params=full_param_dict[player],
                                      param_info=param_info,
                                      utility_settings=utility_settings,
                                      player_uuid=player,
                                      general_settings=general_settings)
                updated_games = loss_function_bayes(dyad_games=updated_games, general_settings=general_settings)
                loss_val = sum_of_all_loss(updated_games, general_settings=general_settings, target_player=player, target_role="chooser")['loss_final_sum'] + \
                           sum_of_all_loss(updated_games, general_settings=general_settings, target_player=player, target_role="predictor")['loss_final_sum']
                return loss_val
            
            "Create a free vector that concatenates chooser and predictor parameters."
            free_keys = [key for key in param_info["keys"]]
            param_bounds = [key for key in param_info["bounds"]]
            free_initial_vector = np.array([best_params_overall[player]["chooser"][key] for key in free_keys] +
                                            [best_params_overall[player]["predictor"][key] for key in free_keys])
            
            if include_covariance:
                free_keys += [key for key in param_info["covar"]["keys"]]
                param_bounds += [key for key in param_info["covar"]["bounds"]]
                free_initial_vector = np.array([best_params_overall[player]["chooser"][key] 
                                                for key in free_keys] + param_info["covar"]["guesses"] +
                                                [best_params_overall[player]["predictor"][key] 
                                                 for key in free_keys] + param_info["covar"]["guesses"])

            param_bounds_joint = param_bounds * 2

            if sample_ratio == 1:
                opt_result = minimize(fun=objective_function_joint,
                                      x0=free_initial_vector,
                                      bounds=param_bounds_joint,
                                      method="L-BFGS-B",
                                      options={'maxiter': 300, 'ftol': 1e-4})
            else:
                opt_result = dual_annealing(func=objective_function_joint, 
                                            bounds=param_bounds_joint, maxiter=50)                
            best_free = opt_result.x
            print(opt_result)
            n_params = len(free_keys)
            for idx, key in enumerate(free_keys):
                best_params_overall[player]["chooser"][key] = best_free[idx]
                best_params_overall[player]["predictor"][key] = best_free[n_params+idx]
    
            dyad_games[0].setdefault("reports", {}).setdefault(
                player, {})["joint"] = gnrl.serialize_opt_result(opt_result)

    if include_covariance:
        "Ensuring covariance parameters satisfy PSD."
        for player_uuid in player_uuids:
            params = best_params_overall[player_uuid]['predictor']
            cov_matrix = gnrl.build_covariation_matrix(
                param_info=param_info,
                params=params,
                raise_on_invalid=False  
            )
            if not gnrl.is_positive_semidefinite(matrix=cov_matrix, tol=1e-12):             
                cov_matrix = gnrl.nearest_psd_matrix(matrix=cov_matrix, min_eigval=0.0)
                param_mean_keys = [param_key for param_key in param_info['keys'] if '_std' not in param_key]
                for idx, key1 in enumerate(param_mean_keys):
                    params[key1 + '_std'] = math.sqrt(cov_matrix[idx][idx])
                    for jdx, key2 in enumerate(param_mean_keys[idx + 1:], start=idx + 1):      
                        cov_key = f"{key1}_{key2}_cov"
                        params[cov_key] = cov_matrix[idx][jdx]

    "Simulate the dyad using the updated parameters."
    updated_dyad = simulate_dyad(dyad_games=dyad_games,
                                  initial_params_p1_p2=[best_params_overall[player_uuids[0]], 
                                                        best_params_overall[player_uuids[1]]],
                                  param_info=param_info,
                                  utility_settings=utility_settings,
                                  general_settings=general_settings)
    updated_dyad = loss_function_bayes(dyad_games=updated_dyad, general_settings=general_settings)
    
    "Making Numpy arrays JSON serializable."
    if update_method == 'grid':
        updated_dyad = prep.serialize_or_drop_param_vectors(dyad_games=updated_dyad, general_settings=general_settings)

    "Recording the total duration for fitting the dyad."
    time_stop_fit_dyad = time.time()
    duration_fit_dyad = time_stop_fit_dyad - time_start_fit_dyad
    serialized_report = updated_dyad[0].get('reports', None)
    if serialized_report is not None:
        serialized_report['total_duration'] = duration_fit_dyad

    final_loss = 0.0
    dyad_key = prep._dyad_key(dyad_key=(first_choo, first_pred), return_tuple=True)
    for player_uuid in dyad_key:
        for player_role in ('chooser', 'predictor'):
            final_loss += sum_of_all_loss(dyad_games=updated_dyad, general_settings=general_settings, 
                                          target_player=player_uuid, target_role=player_role)['loss_final_sum']

    "Recording the final loss for fitting the dyad."
    if serialized_report is not None:
        serialized_report['final_loss'] = final_loss

    if general_settings.get('experiment_num') == 2:
        converged_on_avatar = False
        avatar_frequencies = copy.deepcopy(typo.avatar_frequencies)
        for dyad_game in updated_dyad:
            payoff_keys = [
                'payoff_A_chooser', 'payoff_A_predictor', 
                'payoff_B_chooser', 'payoff_B_predictor'
            ]
            avatar_choice = dyad_game.get('choice', None)
            if avatar_choice in ("A", "B"):
                payoffs = {payoff_key: dyad_game.get(payoff_key) for payoff_key in payoff_keys}
                avatar_frequencies = typo.bayesian_update_discrete(payoffs=payoffs, choice=avatar_choice, 
                                                            choice_frequencies_by_type=typo.choice_frequencies_by_type, 
                                                            priors=avatar_frequencies, print_=False)        
                if any(posterior == 1 for posterior in avatar_frequencies.values()):
                    converged_on_avatar = True
                param_data: dict = dyad_game.get('parameter_estimates', {}).get(update_method, {})
                param_data['optimum_update'] = {
                    'converged': converged_on_avatar,
                    'avatar_posteriors': avatar_frequencies
                }

    "Save the fitted results."
    with open(dyad_file_path, 'w', encoding='utf-8') as file:
        try:
            json.dump(updated_dyad, file, ensure_ascii=False, indent=4)
        except TypeError as error:
            print(f"Dyad not JSON serializable {error}")
            try:
                reports = updated_dyad[0].get('reports', None)
                if reports is not None:
                    del updated_dyad[0]['reports']
                json.dump(updated_dyad, file, ensure_ascii=False, indent=4)
            except TypeError as error:
                print(f"Dyad still not JSON serializable {error}")                    
    
    print(f"Processed {dyad_key} Loss: {round(final_loss, 6)} Duration: {round(duration_fit_dyad, 6)}")
    return updated_dyad


def _worker_fit_one(args: Any):
    """
    Dispatch function for multiprocessing: fit parameters for a single player or dyad.

    Arguments:
        • args: tuple
            A tuple containing:
            - key: str — player UUID or dyad key identifying what to fit.
            - meeting_list: list[dict] — game records for the dyad.
            - file_paths: FilePaths — paths for saving fit outputs.
            - param_info: ParamInfo — parameter configuration.
            - utility_settings: UtilitySettings — utility model configuration.
            - general_settings: GeneralSettings — analysis mode and runtime settings.

    Returns:
        • str — the key passed in, to confirm which unit was processed.

    Saves:
        • JSON file containing the fitted results, written to file_paths["dyad_data"].
    """
    try:
        key, meeting_list, file_paths, param_info, utility_settings, general_settings = args
        analysis_mode = general_settings.get('analysis_mode', 'bayesian')

        if general_settings.get('analysis_unit') == "player" or general_settings.get('update_method') == 'naive':
            "Fit by player."
            fit_params_by_player(player_uuid=key, param_info=param_info, utility_settings=utility_settings,
                                 file_paths=file_paths, general_settings=general_settings)
        elif analysis_mode == 'bayesian':
            "Fit the dyad."
            fit_dyad_parameters_bayes(dyad_games=meeting_list, param_info=param_info,
                utility_settings=utility_settings, file_paths=file_paths, general_settings=general_settings)
        elif analysis_mode == 'mle':
            "Fit the dyad."
            fit_dyad_parameters_mle(dyad_games=meeting_list, param_info=param_info,
                utility_settings=utility_settings, file_paths=file_paths, general_settings=general_settings)
        else:
            raise ValueError(f"analysis_mode must be 'bayesian' or 'mle', not {analysis_mode}!")

        "Return the key to signal completion."
        return key

    except Exception:
        import traceback
        traceback.print_exc()


def run_analysis_bayes(histories_data: Histories, file_paths: FilePaths, param_info: ParamInfo, utility_settings: UtilitySettings, 
                       general_settings: GeneralSettings, dyads_subset: List[int] | None = None, player_uuids: List[str] | None = None, print_: bool = True) -> Histories:
    """
    Process dyads in parallel or serially, saving results as each dyad completes.
    
    Arguments:
        • histories_data: Dict;
            Contains "histories" (dyad meeting data) and "player_info" at the top level.
        • file_paths: Dict;
            File paths for parameter data and individual dyad data.
        • param_info: ParamInfo;
            Parameter information required for fitting dyads.
        • utility_settings: Dict;
            Utility model configuration options.
        • general_settings: GeneralSettings;
            Various settings bundled into one dictionary.
            - experiment_num: int;
                Experiment identifier for naming output files.
            - create_new_file: bool (default: False);
                If False, reuses existing output files to avoid redundant computation.
            - run_in_parallel: bool;
                If True (default), runs in parallel. If False, runs serially.
        • dyads_subset: List[int];
            Start and stop indices of list of dyads to iterate
            over. Useful, for running analyses in batches.

    Returns:
        • dict; The updated `histories_data` dictionary with all fitted parameters.
    """
    "Extract General Settings"
    experiment_num =  general_settings.get('experiment_num', 3)
    run_in_parallel = general_settings.get('run_in_parallel', True)
    create_new_file = general_settings.get('create_new_file', True)
    analysis_unit =   general_settings.get('analysis_unit', 'player')

    "Prepare output file paths"
    output_file = file_paths["file_names"][f"params_data_exper{experiment_num}_bayes"]
    aggregate_path = prep.ensure_directory_and_join(file_paths["param_data"], output_file)
    if analysis_unit == 'player':
        output_dir = file_paths['player_fits']
    elif analysis_unit == 'dyad':
        output_dir = file_paths["dyad_data"]
    else:
        raise ValueError(f"analysis_unit {analysis_unit} not supported.")

    os.makedirs(output_dir, exist_ok=True)

    "Check if the aggregate file already exists"
    if not create_new_file and os.path.exists(aggregate_path):
        with open(aggregate_path, "r", encoding='utf-8') as file:
            histories_data_fitted = json.load(file)
        if histories_data_fitted:
            if print_:
                print(f"Aggregate data loaded from {pretty_path(aggregate_path)}.")
            return histories_data_fitted

    if analysis_unit == 'player':
        if not isinstance(histories_data, dict):
            raise TypeError(
                f"run_analysis_bayes expected histories_data to be a Histories dict "
                f"(from Social_Preference_Prediction_Pairs_Exper{experiment_num}.json) "
                f"but received {type(histories_data).__name__}. "
                f"Check that main.py is loading the JSON pairs file, not the processed CSV."
            )
        player_info: PlayerInfo = histories_data.get('player_info', None)
        if not player_info:
            raise KeyError(
                f"No 'player_info' key found in histories_data. "
                f"Expected the Histories dict from "
                f"Social_Preference_Prediction_Pairs_Exper{experiment_num}.json "
                f"(found at {file_paths.get('processed', '?')}). "
                f"Top-level keys present: {list(histories_data.keys())[:10]}"
            )

        if not (isinstance(player_uuids, list) and all(isinstance(player_uuid, str) for player_uuid in player_uuids)):
            player_uuids = sorted([player_uuid for player_uuid, info in player_info.items()
                            if info.get('player_type') in ('participant', 'synthetic') or (experiment_num == 0 and 'predictor' in player_uuid)])
            # Experiment 0 is the simulation study; predictor-bot UUIDs are included so the optimizer can attempt to recover their known ground-truth parameters.

        n_items = len(player_uuids)
        args_list = [
            (player_uuid, [], file_paths, param_info, utility_settings, general_settings) 
            for player_uuid in player_uuids
        ]

    elif analysis_unit == 'dyad':
        "Extract dyads from histories_data"
        dyads_dict = histories_data.get('histories', None)
        if not dyads_dict:
            raise Exception("No 'histories' found in histories_data.")

        "Prepare for processing"
        dyad_items = list(dyads_dict.items())

        if isinstance(dyads_subset, list) and len(dyads_subset) == 2:
            if isinstance(dyads_subset[0], int) and isinstance(dyads_subset[1], int):
                "Optinally, selecting a subset of all dyads."
                dyads_subset = [index % len(dyads_subset) for index in dyads_subset]
                dyad_items = dyad_items[dyads_subset[0]: dyads_subset[1]]

        n_items = len(dyad_items)
        args_list = [
            (dkey, meeting_list, file_paths, param_info, utility_settings, general_settings)
            for (dkey, meeting_list) in dyad_items
        ]

    else:
        raise ValueError(f"analysis_unit {analysis_unit} not supported.")

    if run_in_parallel:
        "Process players/dyads in parallel"
    
        "Decide worker count conservatively (leave one core for you)"
        max_procs = max(1, mp.cpu_count() - 1)
        n_items   = len(args_list)
        n_workers = min(max_procs, n_items)
        "Can override via general_settings if preferred:"
        n_workers = general_settings.get('n_workers', n_workers)

        "Choose a chunksize: enough work per task to amortize overhead,"
        "but small enough to keep all workers busy"
        default_chunksize = max(1, math.ceil(n_items / (n_workers * 20))) if n_workers != 0 else 1
        chunksize = int(general_settings.get('mp_chunksize', default_chunksize))

        "Recycle workers periodically to curb leaks / fragmentation"
        maxtasks = int(general_settings.get('maxtasksperchild', 50))

        # Spawn context and worker initializer left as future improvement.
        with mp.Pool(processes=n_workers, maxtasksperchild=maxtasks) as pool:
            for idx, key_returned in enumerate(pool.imap_unordered(_worker_fit_one, args_list, chunksize=chunksize), 1):
                if print_:
                    print(f"Processed {idx} / {n_items} {analysis_unit}s - {key_returned}.")

    else:
        "Process players/dyads serially"
        for idx, args in enumerate(args_list, 1):
            key_returned = _worker_fit_one(args)
            if print_:
                print(f"Processed {idx} / {n_items} {analysis_unit}s - {key_returned}.")

    "Reload all individual player/dyad files and combine into histories_data"
    count = 1
    keys_lst = list(dyads_dict.keys()) if analysis_unit == 'dyad' else player_uuids
    for key in keys_lst:
        if analysis_unit == 'dyad':
            file_path = prep._dyad_file_path(dyad_key=key, file_paths=file_paths, 
                                            experiment_num=experiment_num, analysis_mode='bayesian')
        else:
            file_path = os.path.join(file_paths["player_fits"], f"experiment_{experiment_num}", 
                                 f'{file_paths["file_name_suffix"]}_' + key + ".json")

        try:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding='utf-8') as file:
                    fitted_meeting = json.load(file)
                "Update the 'histories' key in the original data"
                if analysis_unit == 'dyad':
                    histories_data['histories'][key] = fitted_meeting
                else:
                    for uuid, fitted_dyad in fitted_meeting.items():
                        histories_data['histories'][uuid] = fitted_dyad

                if count % 10 == 0 and print_: 
                    print(f"Retrieved {count} / {n_items} {analysis_unit}s - {key}")
                count += 1

        except json.decoder.JSONDecodeError as error:
            print(error)

    for top_key in ('histories', 'player_info'):
        for key in histories_data.get(top_key, {}).keys():
            if isinstance(key, (tuple, list)):
                histories_data.get(top_key, {})[str(key)] = histories_data.get(top_key, {}).pop(key)
                print(f"Tuple key detected in histories_data['{top_key}']: {key}")    

    try:
        "Save the final combined aggregate JSON"
        with open(aggregate_path, "w", encoding='utf-8') as file:
            json.dump(histories_data, file, ensure_ascii=False, indent=4)
        if print_:
            print(f"All {analysis_unit}s processed. Final aggregate data saved to {pretty_path(aggregate_path)}.")
    except TypeError:
        print(f"TypeError detected!")

    return histories_data


