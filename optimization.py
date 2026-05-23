from model import *

"=========================================================================================="
"==================================== Shared Functions ===================================="
"=========================================================================================="

def compute_ic(k_params: int, n_data: int, neg_log_likelihood: float) -> Dict[str, float]:
    """
    Computes AIC and BIC using standard formulas when neg_log_likelihood 
    is already the sum of -log(predicted probability of observed choice).

    Arguments:
        • k_params: int; Number of free parameters in the model.
        • n_data: int; Number of data points.
        • neg_log_likelihood: float; sum of negative log-likelihood
            (i.e., sum of -log(predicted probability of observed)).

    Returns:
        • Dictionary with keys 'AIC' and 'BIC'.
    """
    "AIC = 2*k + 2*NLL"
    AIC = 2 * k_params + 2 * neg_log_likelihood 
    
    "BIC = k ln(n) + 2*NLL"
    if n_data > 1:
        BIC = k_params * math.log(n_data) + 2 * neg_log_likelihood
    else:
        "If n_data <= 1, BIC can be undefined or set large"
        BIC = float('inf')
    
    return {"AIC": AIC, "BIC": BIC}


def global_local_optimization(objective_fn: Callable[[Union[np.ndarray, Sequence[float]]], float], x_bounds: Sequence[tuple[float, float]], x_guesses: Optional[Sequence[float]] = None, 
                              optimization_method: str = 'globloc', maxfun_global: int = None, maxfun_local: int = None, maxiter_global: int = None, maxiter_local: int = None, 
                              n_random_starts: int = 1, da_seed: Optional[int] = None, random_seed: Optional[int] = None, local_methods: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """
    Performs a flexible optimization procedure with optional:
      1) Multiple random starts (plus the user-supplied x_guesses) to find the best initial guess.
      2) Global search using Dual Annealing (if optimization_method != 'local').
      3) Local refine using L-BFGS-B (if optimization_method != 'global').

    The final result merges the random-search outcome, the global stage outcome (if used), 
    the local stage outcome (if used), and picks whichever is better as "final" result.

    Arguments:
        • objective_fn: Callable[[np.ndarray], float];
            The function to minimize; returns scalar loss.
        • x_bounds: Optional[List[Tuple[float, float]]];
            Bounds for each alpha dimension (used by local and global optimizers).
            If None, defaults to [-3, 3] in each dimension (arbitrary).
        • maxiter_global: int;
            Maximum iterations for the global search stage (dual_annealing).
        • maxiter_local: int;
            Maximum iterations for the local search stage (L-BFGS-B).    
        • optimization_method: str;
            One of {'global','local','globloc'}.
        • n_random_starts: int; 
            How many random initial guesses to evaluate (including x_guesses as the 
            first). If n_random_starts == 1, then skip random search beyond x_guesses.

    Returns:
        • A JSON-serializable dictionary:
            - Example: {
                "random_search": { 
                    "n_random_starts": int,
                    "best_guess": list,
                    "best_loss": float,
                    "x_guesses_loss": float,
                    "x_guesses_better_than_random": bool,
                    "duration": float
                },
                "global": {...} or None,
                "local":  {...} or None,
                "final":  {...} # final chosen result with total duration
            }
    """
    def safe_serialize(opt_res: Optional[OptimizeResult], dur: float) -> Optional[Dict[str, Any]]:
        if opt_res is None:
            return None
        report = gnrl.serialize_opt_result(opt_res, duration=dur, loss=float(opt_res.fun))
        "method attribute is present on OptimizeResult for SciPy's minimize"
        if hasattr(opt_res, "method"):
            report["method"] = str(getattr(opt_res, "method"))        
        return report
        
    optimization_method = optimization_method.lower().strip()
    if optimization_method not in ('global', 'local', 'globloc'):
        raise ValueError(
            # optimization_method must be one of {'global', 'local', 'globloc'}, 
            f"but got {optimization_method!r}."
        )

    "Handles the case where local optimization needs x_guesses but x_guesses is None."
    if (optimization_method in ('local', 'globloc')) and (x_guesses is None):
        "Fallback guess: random midpoint of each bound."
        x_guesses = [random.uniform(lo, hi) for (lo, hi) in x_bounds]

    x_guesses = np.array(x_guesses) if x_guesses is not None else None
    x_bounds = np.array(x_bounds, dtype=float)

    if isinstance(random_seed, int) and random_seed > 0:
        random.seed(int(random_seed))
        np.random.seed(int(random_seed))

    "0) Keep track of durations to sum in the end"
    random_dur = 0.0
    global_dur = 0.0
    local_dur  = 0.0

    "1) Random Search"
    time_start_random = time.time()

    guess_candidates = []
    "Evaluate user-provided x_guesses"
    if x_guesses is not None:
        xg_loss = objective_fn(x_guesses)
        guess_candidates.append((x_guesses, xg_loss))
    else:
        xg_loss = float('inf')

    n_to_gen = max(0, n_random_starts - 1)
    for _ in range(n_to_gen):
        this_rand = [random.uniform(lo, hi) for (lo, hi) in x_bounds]
        this_rand = np.array(this_rand, dtype=float)
        cur_loss = objective_fn(this_rand)
        guess_candidates.append((this_rand, cur_loss))

    best_guess_rand, best_loss_rand = min(guess_candidates, key=lambda x: x[1])
    x_guesses_better = bool(xg_loss <= best_loss_rand + 1e-14)

    time_stop_random = time.time()
    random_dur = time_stop_random - time_start_random

    random_search_report = {
        "n_random_starts": n_random_starts,
        "best_guess": best_guess_rand.tolist(),
        "best_loss": float(best_loss_rand),
        "x_guesses_loss": float(xg_loss),
        "x_guesses_better_than_random": x_guesses_better,
        "duration": random_dur
    }

    "2) Global (Simulated Annealing)"
    global_opt_result: Optional[OptimizeResult] = None
    time_global_start = None
    time_global_stop  = None
    if optimization_method != 'local':

        time_global_start = time.time()
        da_kwargs = {
            'func': objective_fn,
            'bounds': x_bounds
        }
        if isinstance(maxiter_global, int):
            da_kwargs['maxiter'] = maxiter_global
        if isinstance(maxfun_global, int):
            da_kwargs['maxfun'] = maxfun_global
        if isinstance(da_seed, int):
            da_kwargs['seed'] = int(da_seed)

        global_opt_result = dual_annealing(**da_kwargs)
        time_global_stop = time.time()
        global_dur = time_global_stop - time_global_start

    "3) Decide best from random vs global"
    "That best becomes the x0 for local (if local is used)"
    best_for_local = best_guess_rand
    best_for_local_loss = best_loss_rand
    if global_opt_result is not None:
        if global_opt_result.fun < best_for_local_loss:
            best_for_local = global_opt_result.x
            best_for_local_loss = global_opt_result.fun

    def _method_options(method: str) -> dict:
        opts: dict = {}
        "common"
        if isinstance(maxiter_local, int):
            opts['maxiter'] = int(maxiter_local)
        "per-method add-ons"
        if method == 'L-BFGS-B':
            opts.setdefault('ftol', 1e-6)
            "no maxfun here"
        elif method == 'TNC':
            opts.setdefault('ftol', 1e-6)
            if isinstance(maxfun_local, int):
                opts['maxfun'] = int(maxfun_local)
        elif method == 'SLSQP':
            opts.setdefault('ftol', 1e-6)
            "SLSQP has 'eps' if you want a step size; no maxfun"
        elif method in ('Powell','Nelder-Mead'):
            if isinstance(maxiter_local, int):
                opts['maxiter'] = int(maxiter_local)
            "scipy exposes 'maxfev' for these"
            if isinstance(maxfun_local, int):
                opts['maxfev'] = int(maxfun_local)
        return opts

    "--- Helper: method capabilities ---"
    _methods_supporting_jac = {"L-BFGS-B", "TNC", "SLSQP"}
    _default_local_methods = tuple(local_methods) if local_methods else ("L-BFGS-B",)

    def _minimize_once(method_name: str, x0: np.ndarray) -> Optional[OptimizeResult]:
        "Per-method options, respecting existing ftol/maxiter/maxfun knobs."

        kwargs = {
            "x0": x0,
            "fun": objective_fn,
            "method": method_name,
            "bounds": x_bounds,
            "options": _method_options(method_name),
        }
        if method_name in _methods_supporting_jac:
            kwargs["jac"] = "2-point"

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"delta_grad == 0\.0\. Check if the approximated function is linear.*",
                module=r"scipy\.optimize\._hessian_update_strategy"
            )

            "just before running local(s)"
            assert best_for_local.shape == np.array(x_bounds, float)[:,0].shape, \
                "Local x0 shape mismatch vs bounds."

            return minimize(**kwargs)

    "4) Possibly run local refine"
    local_opt_result: Optional[OptimizeResult] = None
    time_local_start = None
    time_local_stop  = None

    if optimization_method != 'global':
        time_local_start = time.time()

        best_local_res = None
        best_local_method = None
        for meth in _default_local_methods:
            try:
                res = _minimize_once(meth, np.array(best_for_local, dtype=float))
                if res is None:
                    continue
                if (best_local_res is None) or (res.fun < best_local_res.fun):
                    best_local_res = res
                    best_local_method = meth
            except Exception as err:
                print(f"Optimization Failure {meth}: {err}.")
                pass

        local_opt_result = best_local_res
        time_local_stop = time.time()
        local_dur = time_local_stop - time_local_start

        "# --- debug: print which local method won (if any) ---"
        # if local_opt_result is not None:
        #     print(f"[local winner] {best_local_method}")

    if local_opt_result is not None:
        "store winning method for later inspection"
        local_dict = safe_serialize(local_opt_result, local_dur)
        if local_dict is not None:
            # local_dict["chosen_local_method"] = str(getattr(local_opt_result, "method", "unknown"))
            local_dict["chosen_local_method"] = best_local_method

    "5) Decide final"
    best_loss_final = float('inf')
    best_result: Optional[OptimizeResult] = None
    best_label = None

    "Considers the global result if available."
    if global_opt_result is not None:
        if global_opt_result.fun < best_loss_final:
            best_loss_final = global_opt_result.fun
            best_result = global_opt_result
            best_label = "global"

    "If local is used"
    if local_opt_result is not None:
        if local_opt_result.fun < best_loss_final:
            best_loss_final = local_opt_result.fun
            best_result = local_opt_result
            best_label = "local"

    if best_result is None:
        raise RuntimeError("No optimization step was actually performed. Check method logic.")

    "6) Build final dictionary"
    global_dict = safe_serialize(global_opt_result, global_dur) if (optimization_method != 'local') else None
    local_dict  = safe_serialize(local_opt_result,  local_dur)  if (optimization_method != 'global') else None

    # final => merges total duration
    total_duration = random_dur
    if global_opt_result is not None:
        total_duration += global_dur
    if local_opt_result is not None:
        total_duration += local_dur

    final_dict = safe_serialize(best_result, 0.0)
    if final_dict is not None:
        final_dict["chosen_optimizer"] = best_label
        final_dict["duration"] = total_duration  # now final has the sum of all durations

    output = {
        "random_search": random_search_report,
        "global": global_dict,
        "local": local_dict,
        "final": final_dict
    }
    return output


def global_local_then_trust_constr(objective_with_penalty: Callable[[np.ndarray], float], objective_raw_nll: Callable[[np.ndarray], float], x_bounds: Sequence[tuple[float, float]], 
                                   parameter_keys: Sequence[str], x_initial_guess: Optional[Sequence[float]] = None, n_random_starts: int = 1, maxiter_global: Optional[int] = None, 
                                   maxiter_local: Optional[int] = None, maxfun_global: Optional[int] = None, maxfun_local: Optional[int] = None, dual_annealing_seed: Optional[int] = None, 
                                   run_trust_constr: bool = True, optimization_method='globloc', local_methods: Optional[Sequence[str]] = None, trust_maxiter: int = 600, 
                                   trust_gtol: float = 1e-6, trust_xtol: float = 1e-8, trust_verbose: bool = False) -> Dict[str, Any]:
    """
    Two-stage optimizer that combines broad exploration with constrained local refinement.

    Stage 1 runs global_local_optimization with a penalized objective (objective_with_penalty),
    which includes a soft regularization term that discourages extreme parameter values and keeps
    the search numerically stable across the broad parameter space.

    Stage 2 (optional) takes the Stage-1 solution and refines it using SciPy's trust-constr method
    on the *raw* NLL (no penalty) subject to an L1 unit-norm equality constraint over the social-
    preference weight parameters. This constraint prevents the optimizer from growing weights
    arbitrarily large after the penalty is removed, and is only applied to the six weight parameters
    (Vᵢᵢ, Ʌᵢᵢ, Vᵢⱼ, Ʌᵢⱼ, Ƹᵢⱼ, Ʒᵢⱼ). The final result is whichever stage achieved the lower raw NLL.

    Arguments:
        • objective_with_penalty: Callable[[np.ndarray], float]
            Penalized loss function used during Stage 1 (dual annealing + L-BFGS-B).
        • objective_raw_nll: Callable[[np.ndarray], float]
            Raw negative log-likelihood (no penalty) used for Stage 2 and for comparing stages.
        • x_bounds: Sequence[tuple[float, float]]
            Per-parameter (lower, upper) bounds.
        • parameter_keys: Sequence[str]
            Ordered parameter names corresponding to the elements of the parameter vector.
            Used by build_unitnorm_mask to identify which coordinates to constrain.
        • x_initial_guess: Sequence[float] | None
            Starting parameter vector for Stage 1. If None, a random feasible point is used.
        • n_random_starts: int
            Number of random candidate starting points evaluated before global search.
        • maxiter_global: int | None
            Max iterations for the dual-annealing global search (Stage 1).
        • maxiter_local: int | None
            Max iterations for the L-BFGS-B local refinement (Stage 1).
        • maxfun_global: int | None
            Max function evaluations for dual annealing (Stage 1).
        • maxfun_local: int | None
            Max function evaluations for the local optimizer (Stage 1).
        • dual_annealing_seed: int | None
            Random seed for dual annealing reproducibility.
        • run_trust_constr: bool
            If True, run Stage 2 (trust-constr refinement). If False, return Stage 1 result only.
        • optimization_method: str
            Passed to global_local_optimization: 'global', 'local', or 'globloc'.
        • local_methods: Sequence[str] | None
            List of local optimizer names to try in Stage 1 (e.g., ['L-BFGS-B', 'TNC']).
        • trust_maxiter, trust_gtol, trust_xtol: int, float, float
            Convergence controls for the Stage 2 trust-constr solver.
        • trust_verbose: bool
            If True, trust-constr prints iteration-level diagnostics.

    Returns:
        • dict with keys 'stage1', 'stage2' (None if not run), and 'final', where 'final'
            holds {'chosen_optimizer', 'x', 'loss', 'duration'} for the winning solution.
    """
    "Stage-1: global+local"
    """
    NOTE: Use existing global_local_optimization, but allow a seed for DA.
    To avoid changing its signature, this closes over the seed inside the objective
    by monkey-patching dual_annealing at call-site (shown below).
    """

    "Re-uses function directly; to pass a seed into DA, set 'random_state' inside the internal call."
    "Easiest safe tweak: temporarily wrap dual_annealing with seed if provided."
    from scipy.optimize import dual_annealing as _da_original

    def trust_constr_unitnorm_refine(objective_raw_nll: Callable[[np.ndarray], float], x0: np.ndarray, x_bounds: Sequence[tuple[float, float]], unitnorm_mask: np.ndarray, 
                                    maxiter: int = 600, gtol: float = 1e-6, xtol: float = 1e-8, verbose: bool = False) -> Dict[str, Any]:
        """
        Refines a candidate parameter vector with SciPy's 'trust-constr' under an equality L1
        unit-norm constraint applied ONLY to the coordinates flagged by `unitnorm_mask`.

        Design goals:
            • Optimize the *raw* NLL (no penalty) with robust finite-difference gradients.
            • Enforce parameter bounds strictly (clamp x0 just inside bounds to avoid FD issues).
            • Enforce an L1 equality constraint on the selected coordinates:
                sum(|x[mask]|) == 1.0
            • Suppress benign solver warnings (piecewise objectives can trigger SVD/linear hints).
            • Never crash the caller: return a soft failure dict on exceptions; the pipeline can
            keep the Stage-1 solution when Stage-2 does not improve (or fails).

        Arguments:
            • objective_raw_nll: Callable[[np.ndarray], float]; returns the raw negative log-likelihood.
            • x0: np.ndarray; starting point for trust-constr (will be clamped inside bounds).
            • x_bounds: Sequence[Tuple[float,float]]; lower/upper bounds per coordinate.
            • unitnorm_mask: np.ndarray[bool]; True for coordinates included in the L1 constraint.
            • maxiter, gtol, xtol: trust-constr stopping criteria.
            • verbose: bool; if True, trust-constr runs in verbose mode (3).

        Returns:
            • dict; a SciPy-like result summary:
                {
                    "method": "trust-constr",
                    "x": [...],
                    "fun": float,
                    "nit": int,
                    "success": bool,
                    "message": str,
                    "duration": float
                }
        """
        "--- 1) Bounds & safe start ---------------------------------------------------"
        lower_bounds = np.array([lo for lo, _ in x_bounds], dtype=float)
        upper_bounds = np.array([hi for _, hi in x_bounds], dtype=float)

        "Stay strictly inside the box so finite-difference probing never steps outside."
        tiny = 1e-12
        x0 = np.asarray(x0, dtype=float).copy()
        x0 = np.minimum(np.maximum(x0, lower_bounds + tiny), upper_bounds - tiny)

        "--- 2) L1 equality constraint over the masked coordinates --------------------"
        def _l1_equation(x: np.ndarray) -> float:
            return float(np.sum(np.abs(np.asarray(x, dtype=float)[unitnorm_mask])) - 1.0)

        "No analytic Jacobian/Hessian: trust-constr will use finite differences."
        l1_equality_constraint = NonlinearConstraint(
            fun=_l1_equation,
            lb=0.0,
            ub=0.0,
        )

        bounds_object = Bounds(lower_bounds, upper_bounds)  # bound feasibility is enforced by the solver

        "--- 3) Solver call with robust FD and warning suppression --------------------"
        start_time = time.time()
        try:
            with warnings.catch_warnings():
                "3a) benign messages on non-smooth objectives / interior-point linear solves"
                warnings.filterwarnings(
                    "ignore",
                    message=r"Singular Jacobian matrix\. Using SVD decomposition.*",
                    module=r"scipy\.optimize\._trustregion_constr"
                )
                warnings.filterwarnings(
                    "ignore",
                    message=r"delta_grad == 0\.0\. Check if the approximated function is linear.*",
                    module=r"scipy\.optimize\._hessian_update_strategy"
                )

                "3b) Finite-difference settings that play nicely with box constraints"
                result = minimize(
                    fun=objective_raw_nll,
                    x0=x0,
                    method="trust-constr",
                    jac="2-point",                  # robust FD gradient (no user Jacobian)
                    bounds=bounds_object,
                    constraints=[l1_equality_constraint],
                    options={
                        "maxiter": int(maxiter),
                        "gtol": float(gtol),
                        "xtol": float(xtol),
                        "finite_diff_rel_step": 1e-6,
                        "verbose": 3 if verbose else 0
                    }
                )

            duration = time.time() - start_time
            return {
                "method": "trust-constr",
                "x": np.asarray(result.x, dtype=float).tolist(),
                "fun": float(result.fun),
                "nit": int(result.nit),
                "success": bool(result.success),
                "message": str(result.message),
                "duration": float(duration)
            }

        except Exception as err:
            "Soft failure: caller can keep Stage-1 solution. Do not crash the run."
            duration = time.time() - start_time
            return {
                "method": "trust-constr",
                "x": x0.tolist(),
                "fun": float("inf"),
                "nit": 0,
                "success": False,
                "message": f"trust-constr exception: {err}",
                "duration": float(duration)
            }

    def build_unitnorm_mask(parameter_keys: Sequence[str]) -> np.ndarray:
        """
        Select ONLY the social-preference *weight* coordinates for the L1 unit-norm constraint.

        Included:
            • The six weight parameters (pretty or ASCII aliases):
            {Vᵢᵢ, Ʌᵢᵢ, Vᵢⱼ, Ʌᵢⱼ, Ƹᵢⱼ, Ʒᵢⱼ}  or  {Vii, Λii, Vij, Λij, Eij, Gij}

        Excluded:
            • Temperature ('τ')
            • Any standard deviation / covariance keys ('*_std', '*_cov')
            • All exponent parameters (γ₁/γ₂/γ₃, or ASCII 'gamma*')
            • Any other keys not explicitly recognized as weights

        Returns:
            • np.ndarray[bool]; mask aligned with `parameter_keys`, True for coords
            in the L1 equality constraint, False otherwise.
        """
        def _is_exponent_key(key: str) -> bool:
            k0 = key.strip().lower()
            "both pretty (γ1) and ASCII ('gamma1', 'gamma_1') forms"
            return (k0 in {"γ1", "γ2", "γ3"}) or k0.startswith("gamma")

        def _is_std_or_cov(key: str) -> bool:
            return key.endswith("_std") or key.endswith("_cov")

        def _normalize_ascii_aliases(key: str) -> str:
            "make a lenient ASCII fallback for pretty subscripts/special chars"
            return (key.replace("ᵢ", "i")
                    .replace("ⱼ", "j")
                    .replace("Ʌ", "Λ")
                    .replace("Ƹ", "E")
                    .replace("Ʒ", "G"))

        WEIGHT_SET_PRETTY = {"Vᵢᵢ", "Ʌᵢᵢ", "Vᵢⱼ", "Ʌᵢⱼ", "Ƹᵢⱼ", "Ʒᵢⱼ"}
        WEIGHT_SET_ASCII  = {"Vii", "Λii", "Vij", "Λij", "Eij", "Gij"}

        mask_values: list[bool] = []
        for key in parameter_keys:
            if key in ("τ", "temp"):
                mask_values.append(False); continue
            if _is_std_or_cov(key):
                mask_values.append(False); continue
            if _is_exponent_key(key):
                mask_values.append(False); continue

            "Recognize the six weights in either pretty or ASCII form"
            ascii_key = _normalize_ascii_aliases(key)
            is_weight = (key in WEIGHT_SET_PRETTY) or (ascii_key in WEIGHT_SET_ASCII)
            mask_values.append(bool(is_weight))

        return np.array(mask_values, dtype=bool)

    def normalize_vector_L1(parameter_vector: np.ndarray, unitnorm_mask: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """
        Project 'parameter_vector' onto the L1 unit-simplex over the coordinates selected by unitnorm_mask.
        Only the masked coordinates are normalized; others are left unchanged.
        """
        parameter_vector = np.asarray(parameter_vector, dtype=float).copy()
        masked_values = parameter_vector[unitnorm_mask]
        l1_norm = np.sum(np.abs(masked_values))
        if l1_norm < eps:
            "If everything is ~0, leave as-is; trust-constr will move it."
            return parameter_vector
        parameter_vector[unitnorm_mask] = masked_values / l1_norm
        return parameter_vector

    def _dual_annealing_seeded(**kwargs):
        if dual_annealing_seed is not None:
            kwargs["seed"] = int(dual_annealing_seed)
        return _da_original(**kwargs)

    raw_loss_xinit = None
    if (objective_raw_nll is not None) and (x_initial_guess is not None):
        raw_loss_xinit = float(objective_raw_nll(np.array(x_initial_guess, float)))

    "Patch in a localized way (context-like)"
    import scipy.optimize as _sp_opt
    _saved_da = _sp_opt.dual_annealing
    _sp_opt.dual_annealing = _dual_annealing_seeded
    try:
        stage1_report = global_local_optimization(
            objective_fn=objective_with_penalty,
            x_bounds=x_bounds,
            x_guesses=x_initial_guess,
            optimization_method=optimization_method,
            local_methods=local_methods,
            maxfun_global=maxfun_global,
            maxfun_local=maxfun_local,
            maxiter_global=maxiter_global,
            maxiter_local=maxiter_local,
            n_random_starts=n_random_starts,
            random_seed=dual_annealing_seed,
            da_seed=dual_annealing_seed,
        )
    finally:
        _sp_opt.dual_annealing = _saved_da  # restore

    if "random_search" not in stage1_report:
        stage1_report["random_search"] = {}
    stage1_report["random_search"]["x_initial_guess_raw_loss"] = raw_loss_xinit

    final_stage1 = stage1_report["final"]
    x_best_stage1 = np.array(final_stage1["x"], dtype=float)
    loss_best_stage1_penalized = float(final_stage1["loss"])

    "Stage-2: trust-constr (raw NLL under L1 unit-norm)"
    stage2_report = None
    if run_trust_constr:
        unitnorm_mask = build_unitnorm_mask(parameter_keys=parameter_keys)
        if unitnorm_mask.any():
            x0_for_trust = normalize_vector_L1(x_best_stage1, unitnorm_mask)
            stage2_report = trust_constr_unitnorm_refine(
                objective_raw_nll=objective_raw_nll,
                x0=x0_for_trust,
                x_bounds=x_bounds,
                unitnorm_mask=unitnorm_mask,
                maxiter=trust_maxiter,
                gtol=trust_gtol,
                xtol=trust_xtol,
                verbose=trust_verbose
            )

    "Decide final (by raw NLL)"
    "Compute raw NLL for Stage-1 best (penalty-free) to compare apples-to-apples."
    raw_loss_stage1_best = float(objective_raw_nll(x_best_stage1))

    if stage2_report is not None and (stage2_report["success"] or (stage2_report["fun"] <= raw_loss_stage1_best)):
        final_report = {
            "chosen_optimizer": "trust-constr",
            "x": stage2_report["x"],
            "loss": float(stage2_report["fun"]),
            "duration": float(final_stage1["duration"] + stage2_report["duration"])
        }
    else:
        final_report = {
            "chosen_optimizer": "globloc",
            "x": final_stage1["x"],
            "loss": raw_loss_stage1_best,
            "duration": float(final_stage1["duration"])
        }

    return {
        "stage1": stage1_report,
        "stage2": stage2_report,
        "final": final_report
    }


def best_initial_guesses(dyad_key: str | int, file_paths: FilePaths, param_info: ParamInfo, general_settings: GeneralSettings) -> ParamVals:
    """
    Extract the parameter values for each player in each role from the 
    dataframe for a specific dyad in an iterated binary dictator game.

    Arguments:
        • dyad_key: str | int; The dyad key as a string or integer index.
        • file_paths: Dict[str, str]; File paths for loading the dataframe.
        • param_info: ParamInfo; Information about parameters, including keys and bounds.
        • experiment_num: int; Experiment identifier to filter the dataframe.
        • analysis_mode: str; The type of model being run: 'mle' or 'bayesian'.
        • use_initial_params: bool; If True, extract the first parameters; otherwise, extract the last.

    Returns:
        • Dict[str, Dict[str, Any]]; Parameter values for each player in each role.
    """
    use_only_guesses = False
    "Load and filter the dataframe"
    df = prep.create_unified_dataframe(
        all_histories=None, file_paths=file_paths, param_info=param_info, print_=False,
        create_new_file=False, analysis_mode=general_settings.get('analysis_mode', 'bayesian')
    )
    if df is not None:
        experiment_num = general_settings.get('experiment_num', 3)
        if experiment_num in [1, 2, 3]:
            df = df[df['experiment'] == general_settings.get('experiment_num', 3)]

        "Get the dyad key"
        dyad_keys = list(df['dyad_key'])

    else:
        dyad_keys = []

    if isinstance(dyad_key, int):
        dyad_key = dyad_keys[dyad_key % len(dyad_keys)]
    else:
        dyad_key = prep._dyad_key(dyad_key=dyad_key, return_tuple=False)
    
    player_uuid_1, player_uuid_2 = prep._dyad_key(dyad_key=dyad_key, return_tuple=True)
    if dyad_key not in dyad_keys:
        dyad_key = prep._dyad_key(dyad_key=dyad_key, return_tuple=False, reverse=True)
        if dyad_key not in dyad_keys:
            use_only_guesses = True

    if use_only_guesses:    
        row = {}
    else:
        "Filter the dataframe for the selected dyad"
        df = df[df['dyad_key'] == dyad_key]

        "Sort the dataframe by 'meeting_idx' to ensure correct order"
        df = df.sort_values(by='meeting_idx')
        
        "Determine the row to extract parameters from"
        if general_settings.get('use_initial_params', True):
            row = df.iloc[0]  # First row
        else:
            row = df.iloc[-1]  # Last row

    def extract_params(row, prefix: str, keys: list, guesses: list, use_only_guesses: bool = False) -> dict:
        """
        Helper function to extract parameters with fallback to default guesses.
        """
        if use_only_guesses:
            return {
                key: guess for key, guess in zip(keys, guesses)
            }
        else:
            return {
                key: row.get(f'{prefix}_{key}', guess) if isinstance(row.get(f'{prefix}_{key}'
                        ), (int, float)) and not np.isnan(row.get(f'{prefix}_{key}')) else guess
                for key, guess in zip(keys, guesses)
            }

    "Extract parameters for player_uuid_1"
    params_player_1 = {
        'chooser': extract_params(row, 'cc', param_info['keys'], param_info['guesses'](), use_only_guesses=use_only_guesses),
        'predictor': extract_params(row, 'cp', param_info['keys'], param_info['guesses'](), use_only_guesses=use_only_guesses)
    }

    "Extract parameters for player_uuid_2"
    params_player_2 = {
        'chooser': extract_params(row, 'pc', param_info['keys'], param_info['guesses'](), use_only_guesses=use_only_guesses),
        'predictor': extract_params(row, 'pp', param_info['keys'], param_info['guesses'](), use_only_guesses=use_only_guesses)
    }

    if general_settings.get('include_covariance', False):
        "Adding covariance values to each player's dictionary of initial param guesses."
        for pred_param_dict in [params_player_1['predictor'], params_player_2['predictor']]:
            for cov_key, cov_guess in zip(param_info['covar']['keys'], param_info['covar']['guesses']):
                pred_param_dict[cov_key] = cov_guess

    if general_settings.get('temperature_is_param', True):
        "Adding softmax temperature to each player's dictionary of initial param guesses."
        softmax_temperature = general_settings.get('softmax_temperature', None)
        if softmax_temperature is None:
            raise ValueError("Softmax temperature could not be accessed.")
        for player_params in [params_player_1, params_player_2]:
            for param_dict in player_params.values():
                param_dict['τ'] = softmax_temperature

    if use_only_guesses:
        chooser_uuid = player_uuid_1
        predictor_uuid = player_uuid_2
    else:    
        "Determine the roles of player_uuid_1 and player_uuid_2"
        chooser_uuid = row['chooser']
        predictor_uuid = row['predictor']

    "Assign the extracted parameters to the correct players"
    player_params = {}
    player_params[player_uuid_1] = params_player_1 if player_uuid_1 == chooser_uuid else params_player_2
    player_params[player_uuid_2] = params_player_2 if player_uuid_2 == predictor_uuid else params_player_1

    return player_params


"=========================================================================================="
"======== MLE Code — DEPRECATED; see mle.py. Not called by the active pipeline. =========="
"=========================================================================================="

def compute_std_errors_mle(best_x: NDArray[np.float64], data_rows: List[Dict[str, Any]], param_info: ParamInfo,
                            utility_settings: UtilitySettings, penalty_weight: float) -> Dict[str,float]:
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
    hess = gnrl.numerical_hessian(func_wrapper, best_x)  # Finite-difference approach
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
    "parse param_array => param_dict"
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

        selection = row["selection"]  # 0.0 or 1.0
        if loss_funct_type == "ssr":
            residual = (pA - selection)**2
        elif loss_funct_type == "log":
            residual = -math.log(pA if selection == 'A' else 1 - pA)
            
        total_loss += residual

        total_loss += gnrl.parameter_penalty(params=param_dict, penalty_weight=penalty_weight)

    mean_loss = total_loss / len(data_rows)
    return mean_loss


def extract_one_role_data_mle(dyad_games: DyadGames, player_uuid: PlayerUUID, player_role: PlayerRole) -> List[Dict[str, Any]]:
    """
    Collect per-round data needed for MLE from the dyad's meeting list
    for a single player+role.

    Each returned item is a dict:
      {
        "As": float, "Ao": float,
        "Bs": float, "Bo": float,
        "selection": float in {0.0, 1.0},
        "meeting_idx": int,
        "round": int
      }

    Arguments:
        • dyad_games: list of meeting dictionaries for one dyad.
        • player_uuid: str; the player's UUID
        • player_role: 'chooser' or 'predictor'.

    Returns:
        • List[dict]; each element is one row of data for that role.
    """
    extracted_rows = []
    for meet_idx, meeting in enumerate(dyad_games):
        if meeting.get(player_role) != player_uuid:
            continue

        "Skip abdications"
        if player_role == 'chooser' and meeting.get('abdicated_chooser', False):
            continue
        if player_role == 'predictor' and meeting.get('abdicated_predictor', False):
            continue

        label_str = 'choice' if player_role == 'chooser' else 'prediction'
        label_val = meeting.get(label_str)
        if label_val is None:
            continue

        "Convert 'A'=>1.0, 'B'=>0.0"
        selection = 1.0 if label_val == 'A' else 0.0

        "Payoffs"
        As = meeting.get('payoff_A_chooser', 0.0)
        Ao = meeting.get('payoff_A_predictor', 0.0)
        Bs = meeting.get('payoff_B_chooser', 0.0)
        Bo = meeting.get('payoff_B_predictor', 0.0)

        round_num = meeting.get('round', meet_idx)  # fallback to meet_idx if no 'round'

        extracted_rows.append({
            "As": As, "Ao": Ao,
            "Bs": Bs, "Bo": Bo,
            "selection": selection,
            "meeting_idx": meet_idx,
            "round": round_num
        })

    "Sort by round"
    extracted_rows.sort(key=lambda x: x["round"])
    return extracted_rows


def fit_one_player_one_role_mle(role_data: List[Dict[str, Any]], param_info: ParamInfo, 
                                 utility_settings: UtilitySettings, track_evolution: bool) -> List[Dict[str, Any]]:
    """
    Fit parameters for a single player's single role (e.g., 'chooser')
    across the entire role_data. If track_evolution=True, do iterative
    fits (1..n), storing partial results. Otherwise, do one final fit.

    Returns a list of dicts, each containing:
        {
            'meeting_idx': int,
            'round': int,
            'params': { param_name: float, ... },
            'std_errors': { param_name: float, ... },
            'loss': float
        }
    If track_evolution=False, there's only one item for the full data.
    """
    if not role_data:
        return []

    results_list = []
    n = len(role_data)
    "stage counts: either 1..n for iterative or just [n] for a single final fit"
    stage_indices = range(1, n+1) if track_evolution else [n]
    
    for stage_count in stage_indices:
        subset = role_data[:stage_count]
        best_params, std_errs, final_loss = fit_subset_params_mle(subset, param_info, utility_settings)
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
                           penalty_weight: float = 0.1) -> Tuple[Dict[str,float], Dict[str,float], float]:
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
            - final_loss: float — loss value at the optimum.
    """
    if not subset_data:
        "Trivial fallback"
        best_dict = {k: 0.0 for k in param_info["keys"]}
        err_dict = {k: float('inf') for k in param_info["keys"]}
        return best_dict, err_dict, float('inf')

    "Initial parameter guesses (x0) and parameter bounds"
    if callable(param_info["guesses"]):
        x0 = param_info["guesses"]()
    else:
        x0 = param_info["guesses"]
    bnds = param_info["bounds"]

    def objective_func(x: NDArray[np.float64]) -> float:
        return loss_function_mle(x, subset_data, param_info, utility_settings, penalty_weight)

    "Optimize: Find best fitting parameters."
    result = minimize(objective_func, x0, bounds=bnds, method='L-BFGS-B')
    best_x = result.x
    final_loss = result.fun

    "Compute std errors"
    std_errs = compute_std_errors_mle(best_x, subset_data, param_info, utility_settings, penalty_weight)

    best_params_dict = {key: val for (key, val) in zip(param_info["keys"], best_x)}
    return best_params_dict, std_errs, final_loss


def store_params_in_dyad_mle(dyad_games: DyadGames, player_uuid: PlayerUUID, player_role: PlayerRole, 
                              fit_results: List[Dict[str, Any]], utility_settings: UtilitySettings, general_settings: GeneralSettings) -> List[Dict[str, Any]]:
    """
    Store the MLE fit results in the dyad meeting dictionaries.

    fit_results is a list of items:
        {
            "meeting_idx": int,
            "round": int,
            "params": {...},
            "std_errors": {...},
            "loss": float
        }
    For track_evolution=True, multiple items. For single-shot, just 1.

    Store them in:
        meeting["parameter_estimates"]["mle"][player_uuid][player_role]
        at the relevant "meeting_idx".
    """
    for item in fit_results:
        midx = item["meeting_idx"]
        if midx < 0 or midx >= len(dyad_games):
            continue

        meeting = dyad_games[midx]
        param_est = meeting.setdefault("parameter_estimates", {})
        mle_dict = param_est.setdefault("mle", {})
        plyr_dict = mle_dict.setdefault(player_uuid, {})
        role_dict = plyr_dict.setdefault(player_role, {})

        """
        Overwrite or store. If iterative, it might be better to have a
        list of partial fits but just store the final stage each time.
        """
        role_dict["params"] = item["params"]
        role_dict["std_errors"] = item["std_errors"]

        model_select_A = "model_choose_A" if player_role == "chooser" else "model_predict_A"
        role_dict["output"] = {
            "loss": item["loss"],
            model_select_A: choice(current_game=meeting, agent_params=role_dict["params"], 
                                   softmax_temperature=general_settings.get('softmax_temperature', 1.5), 
                                   utility_settings=utility_settings, select=False)["model_choose_A"]
        }
    
    return dyad_games


def fit_dyad_parameters_mle(dyad_games: List[Dict[str, Any]], param_info: ParamInfo, utility_settings: UtilitySettings,  
                            file_paths: FilePaths, general_settings: GeneralSettings) -> Dict[str, Any]:
    """
    Fit MLE-based social preference parameters for both players in a single dyad.

    This function processes all games (meetings) in a single dyad and fits 
    parameters for each player in each role (chooser and predictor). It stores 
    the fitted parameters and optional iterative/evolutionary fits in the 
    'parameter_estimates' => 'mle' sub-dictionaries within each meeting dict.

    Arguments:
        • dyad_games: List[dict]
            A list of meeting/game dictionaries for this dyad. Each dict includes:
              - 'chooser': str (player UUID)
              - 'predictor': str (player UUID)
              - 'choice': str in {'A','B'} or None
              - 'prediction': str in {'A','B'} or None
              - payoff fields: 'payoff_A_chooser','payoff_A_predictor', ...
              - 'round': int, etc.
        • param_info: ParamInfo
            Configuration for parameter fitting, e.g.:
              {
                "keys": ["Vᵢᵢ","Vᵢⱼ","ε_s"],
                "bounds": [(-1,1),(-1,1),(-1,1)],
                "guesses": lambda: [0.0,0.0,0.0]  # or a list
              }
        • utility_settings: UtilitySettings
            Options for the utility function and modeling approach, e.g.:
              {
                "use_negativity_parameters": True,
                "use_exponential_parameters": False,
                ...
              }
        • track_evolution: bool (default=False)
            If True, do iterative fits, storing partial results in each relevant meeting.
            If False, just do one fit over all data.

    Returns:
        • List[dict]
            The same dyad_games structure, but updated with MLE results in:
              meeting["parameter_estimates"]["mle"][player_uuid][player_role] = {
                "params": {...}, "std_errors": {...}, "loss": ...
              }

    Notes:
        • Each player-role pair is fit *independently.* E.g., 
          (playerA, chooser), (playerA, predictor), (playerB, chooser), (playerB, predictor).
        • If a player never acts in a particular role, skips that pair.
        • If track_evolution=True, performs partial fits (first 1 game, first 2 games, etc.) 
          and store each step’s result at the final meeting used.
    """
    if not dyad_games:
        return dyad_games  # no data => no fitting

    "Extract player uuids and sort alphabetically."
    first_game = dyad_games[0]
    first_choo = first_game.get('chooser')
    first_pred = first_game.get('predictor')
    if not isinstance(first_choo, str) or not isinstance(first_pred, str):
        raise ValueError(f"Failed to extract player uuids from games.")
    player_uuids = sorted([first_choo, first_pred])

    dyad_file_path = prep._dyad_file_path(dyad_key=tuple(player_uuids), file_paths=file_paths, 
                                     experiment_num=general_settings.get('experiment_num', 3), analysis_mode='mle')
    try:
        if not general_settings.get('create_new_file', False):
            if os.path.exists(path=dyad_file_path):
                with open(dyad_file_path, "r", encoding='utf-8') as file:
                    dyad_history = json.load(file)
                if dyad_history:
                    return dyad_history            
    except json.decoder.JSONDecodeError as error:
        print(error)
        pass

    "For [playerA, playerB] × ['chooser','predictor'] gather data => fit => store results."
    for player_uuid in player_uuids:
        for role in ['chooser', 'predictor']:
            "1) Extract role data"
            role_data = extract_one_role_data_mle(dyad_games, player_uuid, role)
            if not role_data:
                continue  # skip if no data for that role

            "2) Fit the data (iterative or single‐shot)"
            fit_results = fit_one_player_one_role_mle(role_data, param_info, utility_settings, general_settings.get('track_evolution', True))

            "3) Store the results in the dyad_games"
            store_params_in_dyad_mle(dyad_games, player_uuid, role, fit_results, utility_settings)
        
    "Save the fitted results."
    with open(dyad_file_path, 'w', encoding='utf-8') as file:
        json.dump(dyad_games, file, ensure_ascii=False, indent=4)

    return dyad_games


def run_analysis_mle(histories_data: Histories, file_paths: FilePaths, param_info: ParamInfo, 
                     utility_settings: UtilitySettings, general_settings: GeneralSettings) -> Dict[str, Any]:
    """
    Run the non-cognitive MLE analysis over all dyads in histories_data["histories"].

    For each dyad, calls fit_dyad_parameters_mle(...) to store MLE parameters 
    in each meeting. Optionally track parameter evolution.

    Arguments:
        • histories_data: dict 
            Must have 'histories': { "(A,B)": [ {meeting}, ... ], ... }
        • experiment_num: int
            E.g. 3, used if you want to name files, etc.
        • param_info: ParamInfo
            Parameter fitting config (keys, bounds, guesses).
        • utility_settings: UtilitySettings
            Utility model config.
        • track_evolution: bool
            If True, iterative fits.
        • run_in_parallel: bool
            If True (default), runs in parallel. If False, runs serially.

    Returns:
        The updated histories_data with MLE results in each dyad's meetings:
          meeting["parameter_estimates"]["mle"][player_uuid][player_role]
    """
    "Extract General Settings"
    experiment_num =  general_settings.get('experiment_num',  3)
    track_evolution = general_settings.get('track_evolution', True)
    run_in_parallel = general_settings.get('run_in_parallel', True)
    create_new_file = general_settings.get('create_new_file', True)

    "Prepare output file paths" 
    output_file = file_paths["file_names"][f"params_data_exper{experiment_num}_{'iter' if track_evolution else 'fit1'}"]
    aggregate_path = os.path.join(file_paths["param_data"], output_file)
    dyad_output_dir = file_paths["dyad_data"]

    "Check if the aggregate file already exists"
    if not create_new_file and os.path.exists(aggregate_path):
        with open(aggregate_path, "r", encoding='utf-8') as file:
            histories_data_fitted = json.load(file)
        if histories_data_fitted:
            print(f"Aggregate data loaded from {aggregate_path}.")
            return histories_data_fitted

    "Extract dyads from histories_data"
    dyads_dict = histories_data.get('histories', None)
    if not dyads_dict:
        raise Exception("No 'histories' found in histories_data.")

    "Prepare for processing"
    dyad_items = list(dyads_dict.items())
    count, n_dyads = 1, len(dyad_items)
    os.makedirs(dyad_output_dir, exist_ok=True)

    args_list = [
        (dkey, meeting_list, file_paths, param_info, utility_settings, general_settings)
        for (dkey, meeting_list) in dyad_items
    ]

    from bayesian import _worker_fit_one
    if run_in_parallel:
        "Process dyads in parallel"
        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            for idx, dkey_returned in enumerate(pool.imap_unordered(_worker_fit_one, args_list), 1):
                print(f"Processed {idx} / {n_dyads} dyads - {dkey_returned}.")
    else:
        "Process dyads serially"
        for idx, args in enumerate(args_list, 1):
            dkey_returned = _worker_fit_one(args)
            print(f"Processed {idx} / {n_dyads} dyads - {dkey_returned}.")

    "Reload all individual dyad files and combine into histories_data"
    for dkey in dyads_dict.keys():
        dyad_file_path = prep._dyad_file_path(dyad_key=dkey, file_paths=file_paths, 
                                         experiment_num=experiment_num, analysis_mode='mle')
        try:
            if os.path.exists(dyad_file_path):
                with open(dyad_file_path, "r", encoding='utf-8') as file:
                    fitted_meeting = json.load(file)
                histories_data['histories'][dkey] = fitted_meeting
                print(f"Retrieved {count} / {n_dyads} dyads - {dkey}")
                count += 1
        except json.decoder.JSONDecodeError as error:
            print(error)

    "Save the final combined aggregate JSON"
    with open(aggregate_path, "w", encoding='utf-8') as file:
        json.dump(histories_data, file, ensure_ascii=False, indent=4)
    print(f"All dyads processed. Final aggregate data saved to {aggregate_path}.")

    return histories_data


