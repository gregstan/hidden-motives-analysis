"=========================================================================================="
"=================================== Importing Libraries =================================="
"=========================================================================================="

"BLAS / OpenMP thread control"
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np, pandas as pd, itertools as it, plotly.graph_objects as go, multiprocessing as mp, \
    datetime as dt, hashlib, warnings, inspect, random, pprint, json, math, copy, time, glob, ast, re
from typing import Callable, Sequence, Optional, TypedDict, \
    Literal, Iterable, Dict, List, Set, Tuple, Union, Any
from scipy.optimize import SR1, minimize, dual_annealing, differential_evolution, \
    OptimizeResult, Bounds, NonlinearConstraint # type: ignore
from scipy.stats import multivariate_normal
from plotly.subplots import make_subplots
from scipy.special import softmax # type: ignore
from numpy.typing import NDArray
from dataclasses import dataclass
from pathlib import Path
dual_annealing: Any
minimize: Any  

"""Suppress Pandas warnings from printing to consol."""
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_colwidth', 22)
pp = pprint.PrettyPrinter(indent=2)

warnings.filterwarnings("ignore", message=r"delta_grad == 0\.0.*", module=r"scipy\.optimize")
warnings.filterwarnings("ignore", message=r"Singular Jacobian matrix.*", module=r"scipy\.optimize")
warnings.filterwarnings(
    "ignore",
    message=r"Values in x were outside bounds during a minimize step",
    category=RuntimeWarning,
    module=r"scipy\.optimize\._optimize"
)

"=========================================================================================="
"====================================== Type Aliases ======================================"
"=========================================================================================="

PlayerUUID = str
PlayerRole = Literal['chooser', 'predictor']
DyadKey = str | tuple[PlayerUUID, PlayerUUID] | list[PlayerUUID]
BoolTuple = Tuple[bool, ...]

ParameterBounds = dict[str, tuple[int | float, int | float]]
FigLay = dict[str, Any]
ColumnNames = dict[str, list[str] | dict[str, str]]

ParamKeys = list[str]
ParamBounds = list[tuple[int | float]]
ParamGuesses = Callable[[], Dict[str, float]]
ParamCovar = list[str]

CovMat = NDArray[np.float64] | None
CovMatDict = dict[str, dict[str, dict[str, CovMat]]] | None

Output = dict[str, float]
Params = dict[str, float]
ParamVectors = dict[tuple[int] | str, float]
MetaData = dict[str, int | float | dict[str, list[float]]]
ParamEst = dict[str, dict[str, dict[str, dict[str, Output | Params | MetaData | ParamVectors]]]]

DyadGame = dict[str, str | int | bool | ParamEst]
DyadGames = list[DyadGame]

PlayerInfo = dict[str, dict[str, str]]
Histories = dict[str, DyadGames | PlayerInfo]
ParamVals = dict[PlayerUUID, dict[PlayerRole, Params]]

PlayerDyads = dict[DyadKey, DyadGames]

WriteMode = Literal["readonly", "resume", "overwrite"]

class FileNames(TypedDict):
    player_pairs_exper1: str
    player_pairs_exper2: str
    player_pairs_exper3: str
    processed_data_exper1: str
    processed_data_exper2: str
    processed_data_exper3: str
    params_data_exper1_iter: str
    params_data_exper2_iter: str
    params_data_exper3_iter: str
    params_data_exper1_fit1: str
    params_data_exper2_fit1: str
    params_data_exper3_fit1: str
    params_hist_exper1_iter: str
    params_hist_exper2_iter: str
    params_hist_exper3_iter: str
    params_hist_exper1_fit1: str
    params_hist_exper2_fit1: str
    params_hist_exper3_fit1: str
    params_data_exper1_bayes: str
    params_data_exper2_bayes: str
    params_data_exper3_bayes: str
    params_hist_exper1_bayes: str
    params_hist_exper2_bayes: str
    params_hist_exper3_bayes: str
    all_fitted_data_bayesian: str
    all_fitted_data_mle: str
    raw_data_exper1: str
    raw_data_exper2: str
    raw_data_exper3: str

class FilePaths(TypedDict):
    raw_data: str
    processed: str
    param_data: str
    dyad_data: str
    visuals: str
    file_names: FileNames  # Nested dictionary

@dataclass
class UniformGuesser:
    bounds: list[tuple[float,float]]
    seed: int | None = None
    def __call__(self) -> list[float]:
        rng = random.Random(self.seed) if self.seed is not None else random
        return [rng.uniform(lo, hi) for (lo,hi) in self.bounds]


class UtilitySettings(TypedDict):
    conditional_welfare_mode: bool
    reference_dependent_altruism: bool
    min_max_rawlsian_leontief: bool
    include_welfare_efficiency_term: bool
    include_relative_income_penalty: bool
    use_exponential_parameters: bool
    apply_exponents_to_payoffs: bool
    uniform_exponential_parameter: bool
    single_payoffs_not_differences: bool
    payoff_ratios_not_differences: bool
    reference_dependent_utility: bool
    use_negativity_parameters: bool
    negativity_social_comparison: bool
    tie_self_interest_and_altruism: bool
    fix_self_interest_parameter: bool
    include_social_comparison: bool
    include_altruism_term: bool


class WarmstartPolicy(TypedDict):
    enabled: bool
    schedule: str
    cold_iters: int
    explore_iters: int
    temperature_low: float
    temperature_high: float
    disable_dual_annealing_when_warm: bool


class OptimizationPolicy(TypedDict, total=False):
    n_random_starts: int
    maxiter_global: int
    maxiter_local: int
    maxfun_global: int
    maxfun_local: int
    run_trust_constr: bool
    dual_annealing_seed: int | None
    trust_maxiter: int
    trust_gtol: float
    trust_xtol: float
    trust_verbose: bool
    local_methods: list[str]


class AmpdSettings(TypedDict, total=False):
    metric: str
    n_games: int
    n_iters: int
    parameter_sampling_mode: str
    parameter_pairing_mode: str
    player_roles: list[str] | None
    random_seed: int | None


class IndividualArchitectureSettings(TypedDict, total=False):
    population_top_n_models: Optional[int]
    participant_top_r_models: Optional[int]
    M_max: Optional[int]
    exhaustive_M_max: int
    score_basis: str
    stopping_criteria: str
    marginal_gain_threshold: float
    n_consecutive_low_marginal_gains_required: int
    cumulative_gain_threshold: float
    diagnose_selected_library_redundancy: bool
    n_workers: Optional[int]


class ModelRecoverySettings(TypedDict, total=False):
    generating_model: int
    n_agents_grid: List[int]
    n_games_grid: List[int]
    softmax_temperature: float
    candidate_model_selection_mode: str
    n_candidate_models: Optional[int]
    ampd_matrix_name_or_path: Optional[str]
    random_seed: int


class RandomSeeds(TypedDict, total=False):
    use_seeds: bool
    seed: int | None


class ParameterRecoverySettings(TypedDict, total=False):
    n_players: int | None        # None → auto-count from experiment 3 human participants
    fit_predictor_role: bool     # True → also run Bayesian predictor fitting (slow); default False
    n_games: int | None       # None → 60 games per synthetic dyad
    k_params_range: tuple     # (k_min, k_max) inclusive range of dimensionalities to test
    n_altruism_steps: int     # Grid points across the altruism (Vᵢⱼ) range
    evenly_space_altruism: bool   # True → grid; False → uniform random sampling
    correlate_all_params: bool    # True → all free mean params; False → altruism only
    random_seed: int | None       # RNG seed; None → unseeded


class GeneralSettings(TypedDict, total=False):
    update_method: str
    analysis_mode: str
    analysis_unit: str
    experiment_num: int
    loss_funct_type: str
    track_evolution: bool
    create_new_file: bool
    run_in_parallel: bool
    include_covariance: bool
    softmax_temperature: float
    optimization_method: str
    confidence_weighted: bool
    use_particle_filter: bool
    guess_params_randomly: bool
    temperature_is_param: bool
    n_bins_per_dimension: int
    optimization_policy: OptimizationPolicy
    fit_roles_together: bool
    use_initial_params: bool
    warmstart_policy: WarmstartPolicy
    penalty_weight: float
    learning_rate: float
    sample_ratio: float
    export_fig: bool
    write_mode: str
    dark_mode: bool
    ampd_settings: AmpdSettings
    individual_architecture_settings: IndividualArchitectureSettings
    model_recovery_settings: ModelRecoverySettings
    parameter_recovery_settings: ParameterRecoverySettings
    random_seeds: RandomSeeds


class RunCodeSettings(TypedDict):
    run_belief_update_simulation: bool
    run_illustrate_belief_updates: bool
    run_alternative_model_contest: bool
    run_typological_bayesian_models: bool
    run_information_criterion_analysis: bool
    run_model_nesting_violation_analysis: bool
    run_parameter_distribution_results: bool
    run_inequality_aversion_analysis: bool
    run_individual_architecture_analysis: bool
    run_model_recovery_simulation: bool
    run_param_recovery_analysis: bool


class ParamCovarInfo(TypedDict):
    keys: list[str]
    bounds: list[tuple[float, float]]
    guesses: list[float]


class ParamInfo(TypedDict, total=False):
    keys: list[str]
    bounds: list[tuple[int | float, int | float]]
    guesses: Any  # UniformGuesser (callable) or list[float]
    covar: ParamCovarInfo


def parameter_keys_for_utility_settings(utility_settings: UtilitySettings, general_settings: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Returns the ordered list of parameter keys for a given set of utility options.

    Arguments:
        • utility_settings: dict[str, bool]
        • general_settings: dict[str, Any] | None; Makes std keys follow MCMC/grid conventions. 

    Returns:
        • list[str]; Ordered parameter names, e.g. ['Vᵢᵢ', 'Vᵢⱼ', 'αᵢⱼ', 'γ1', 'γ2', ...]
    """
    negativity_params = {'Vᵢᵢ': 'λᵢᵢ', 'Vᵢⱼ': 'λᵢⱼ', 'αᵢⱼ': 'βᵢⱼ'}
    param_keys: List[str] = []

    if utility_settings['min_max_rawlsian_leontief']:
        if utility_settings['include_social_comparison']:
            param_keys.append('Vᵢⱼ')
        else:
            param_keys.append('Vᵢᵢ')
            param_keys.append('Vᵢⱼ')

        if utility_settings['use_exponential_parameters']:
            param_keys.append('γ1')
            if not utility_settings['uniform_exponential_parameter']:
                param_keys.append('γ2')

    else:
        if not utility_settings['fix_self_interest_parameter']:
            param_keys.append('Vᵢᵢ')

        if utility_settings['conditional_welfare_mode']:
            param_keys.append('λᵢᵢ')
            if utility_settings['include_altruism_term']:
                param_keys.append('Vᵢⱼ')
                param_keys.append('λᵢⱼ')
            if utility_settings['use_exponential_parameters']:
                param_keys.append('γ1')
                if not utility_settings['uniform_exponential_parameter'] and utility_settings['include_altruism_term']:
                    param_keys.append('γ2')

        elif utility_settings.get('include_welfare_efficiency_term'):
            "Vᵢᵢ already added above; social comparison activates maximin via Vᵢⱼ."
            if utility_settings['include_social_comparison']:
                param_keys.append('Vᵢⱼ')
            if utility_settings['use_exponential_parameters']:
                param_keys.append('γ1')
                if not utility_settings['uniform_exponential_parameter'] and utility_settings['include_social_comparison']:
                    param_keys.append('γ2')

        else:
            # if utility_settings['use_negativity_parameters'] and not utility_settings['fix_self_interest_parameter']:
            if utility_settings['use_negativity_parameters']:
                param_keys.append('λᵢᵢ')

            if utility_settings['include_altruism_term']:
                if not utility_settings.get('tie_self_interest_and_altruism', False):
                    param_keys.append('Vᵢⱼ')
                    if utility_settings['use_negativity_parameters']:
                        param_keys.append('λᵢⱼ')

            if utility_settings['include_social_comparison']:
                param_keys.append('αᵢⱼ')
                if utility_settings['use_negativity_parameters'] or utility_settings['negativity_social_comparison']:
                    param_keys.append('βᵢⱼ')

            if utility_settings.get('include_relative_income_penalty'):
                if 'αᵢⱼ' not in param_keys:
                    param_keys.append('αᵢⱼ')

            if utility_settings['use_exponential_parameters']:
                if utility_settings['uniform_exponential_parameter']:
                    param_keys.append('γ1')
                else:
                    "Match enumeration rule for γ's across present terms"
                    param_keys_ = copy.deepcopy(param_keys)
                    if utility_settings['fix_self_interest_parameter']:
                        param_keys_ = ['Vᵢᵢ'] + param_keys_
                    param_keys += [
                        f'γ{idx + 1}'
                        for idx, key in enumerate(negativity_params.keys())
                        if key in param_keys_
                    ]

    "Append std keys if needed (same convention as make_param_info)"
    if general_settings is not None:
        if general_settings.get('update_method') in ('MCMC', 'grid'):
            param_keys += [f"{key}_std" for key in param_keys]

    return param_keys


def make_param_info(param_bds: dict[str, tuple[int | float, int | float]], utility_settings: dict[str, bool], 
                    general_settings: dict[str, Any], guess_seed: int | None = None, random_guesses_are_unique: bool = True) -> ParamInfo:
    """
    Builds the parameter-specification dictionary (`param_info`) for the current utility function,
    including ordered parameter names, bounds, and initialization guesses, with optional covariance
    parameters. This function is the authoritative source of parameter order used throughout the
    codebase (for fitting, saving, and loading).

    Arguments:
        • param_bds: dict[str, (low, high)];
            Numeric bounds for each parameter name (means and, if present, *_std and covariances).
        • utility_settings: dict[str, bool];
            Boolean switches defining the active functional form (e.g., exponents on/off, negativity, etc.).
        • general_settings: dict[str, Any];
            Global analysis/settings flags. The following keys are respected here:
              – 'update_method' in {'naive','MCMC','grid'}:
                    If 'MCMC' or 'grid', standard deviation parameters ('*_std') are appended
                    for all mean parameters to support hierarchical/sampling updates.
              – 'include_covariance' (bool):
                    If True, pairwise covariance parameters are added for all pairs of mean parameters
                    that also have corresponding *_std bounds in `param_bds`. Bounds are computed from
                    the *_std upper bounds (or a conservative alternative if `wide_cov_bounds=False`).
        • random_guesses_are_unique: bool (default True);
            If True, `param_info["guesses"]` is a callable that draws a fresh random vector of initial
            guesses on each call (within `param_bds`, with a minimum standard deviation floor for *_std).
            If False, `param_info["guesses"]` is a single fixed list of floats (one-time draws).

    Returns:
        • param_info: dict with at least the following keys:
            – 'keys'   : list[str] – ordered parameter names (means first; *_std appended when applicable).
            – 'bounds' : list[(low, high)] – bounds aligned with 'keys'.
            – 'guesses': list[float] or Callable[[], list[float]] – initial guesses aligned with 'keys'.
            – 'covar'  : optional dict with 'keys' (names), 'bounds', and 'guesses' (if include_covariance=True).

    Conventions and invariants:
        • Parameter names use Unicode symbols consistently with the rest of the codebase:
            Vᵢᵢ, Vᵢⱼ, αᵢⱼ, λᵢᵢ, λᵢⱼ, βᵢⱼ, and γ1, γ2, γ3, …
        • The order in 'keys' is *the* canonical order used to interpret vectors passed to optimizers.
          Always derive counts (k) and indexing from this list to avoid drift across components.
        • Standard-deviation keys ('*_std') are appended in the same order as their mean counterparts
          when update_method ∈ {'MCMC','grid'}; covariance keys are added only when requested.
        • Bounds are taken directly from `param_bds` in the same order as 'keys'. It is the caller’s
          responsibility to ensure `param_bds` contains entries for any parameter that can appear under
          the given `utility_settings` and `general_settings`.

    Notes for maintainers:
        • The logic for selecting which mean parameters appear is factored through
          `parameter_keys_for_utility_settings(...)`. If user extends the functional form
          (e.g., additional terms), update that function first and this function will pick up
          the changes automatically.
        • If user changes how standard deviations or covariance parameters are handled, keep the
          ordering stable—many other modules assume that 'keys' order is the single source of truth.
    """
    param_keys = parameter_keys_for_utility_settings(
        utility_settings=utility_settings, general_settings=general_settings
    )
    param_bounds = [param_bds[param_key] for param_key in param_keys]

    if random_guesses_are_unique:
        guesses = UniformGuesser(bounds=param_bounds, seed=guess_seed)  # pickleable
    else:
        rng = random.Random(guess_seed)
        guesses = [rng.uniform(lo, hi) for (lo, hi) in param_bounds]

    param_info = {
        "keys": param_keys,
        "bounds": param_bounds,
        "guesses": guesses,
    }

    if general_settings.get('include_covariance'):
        "Generate 'keys', 'bounds', and 'guesses' for covariance parameters."
        param_info["covar"] = {"keys": [], "guesses": [], "bounds": []}

        "Iterate over all parameter keys to generate covariance bounds"
        for key1, key2 in it.combinations([
            param_key for param_key in param_info["keys"] if "_std" not in param_key], 2):

            "Retrieve corresponding standard deviation keys"
            std_key1, std_key2 = f"{key1}_std", f"{key2}_std"

            "Ensure both standard deviation keys exist in pbds"
            if std_key1 in param_bds and std_key2 in param_bds:
                "Extract upper bounds for standard deviations"
                std1_upper = param_bds[std_key1][1]
                std2_upper = param_bds[std_key2][1]

                "Compute covariance bounds"
                wide_cov_bounds = False
                if wide_cov_bounds:
                    upper_bound = std1_upper * std2_upper
                else:
                    upper_bound = int((std1_upper + std2_upper) / 2) 
                lower_bound = -upper_bound

                "Save bounds for the covariance"
                covariance_key = f"{key1}_{key2}_cov"
                param_info["covar"]["keys"].append(covariance_key)
                param_info["covar"]["bounds"].append((lower_bound, upper_bound))
                param_info["covar"]["guesses"].append(0)

    return param_info


def create_file_name_suffix(general_settings: dict[str, Any], utility_settings: dict[str, bool]) -> str:
    """
    Creates a suffix for file names based on the general settings and utility
    options so that new files only overwrite files with the same settings.

    Only scalar settings (bool, int, float, str) contribute to the suffix.
    Nested dicts (warmstart_policy, optimization_policy, ampd_settings,
    individual_architecture_settings, model_recovery_settings, etc.) are
    automatically skipped — they encode run-specific tuning that should not
    determine which saved files get overwritten.

    Arguments:
        • general_settings: dict[str, Any]; Various settings bundled together.
        • utility_settings: dict[str, bool]; Configures the utility function.

    Returns:
        • str; Added to the end of each file name.
    """
    settings_to_ignore = (
        'run_in_parallel', 'create_new_file', 'write_mode',
        'dark_mode', 'export_fig', 'track_evolution',
        'fit_roles_together', 'use_initial_params', 'learning_rate',
    )

    file_name_suffix = "~"
    for key, val in sorted(general_settings.items()):
        if key in settings_to_ignore:
            continue
        elif key == "penalty_weight":
            abreviated_val = f"{val}"
        elif isinstance(val, bool):
            abreviated_val = f"{int(val)}"
        elif isinstance(val, float):
            abreviated_val = f"{round(val,1)}"
        elif isinstance(val, int):
            abreviated_val = f"{val}"
        elif isinstance(val, str):
            abreviated_val = val[0]
        else:
            continue

        file_name_suffix += abreviated_val

    raw_utility_bits = "".join(str(int(val)) for _, val in sorted(utility_settings.items()))
    fmt_utility_bits = f"{raw_utility_bits[0:4]}-{raw_utility_bits[4:8]}-{raw_utility_bits[8:12]}-{raw_utility_bits[12:17]}"
    file_name_suffix += "--" + fmt_utility_bits

    return file_name_suffix


def add_remove_file_name_suffix(file_paths: FileNames, file_name_suffix: str | None,
                                add_suffix: bool = True) -> dict[str, str | dict[str, str]]:
    """
    Adds or removes a file_name_suffix from the canonical file names in file_paths["file_names"].

    When adding (add_suffix=True):
        • Stores the suffix under file_paths["file_name_suffix"].
        • Appends the suffix before the extension of every file_names entry whose key
          contains "_iter", "_fit1", "_bayes", or "all" — the run-specific output files.
        • File names without a "." are left unchanged.

    When removing (add_suffix=False):
        • Strips everything from the first "~" in the stem onward, restoring the base name.
        • Safe to call on an unsuffixed dict (no-op when "~" is absent).

    Arguments:
        • file_paths      : FileNames; the config file-paths dict (modified in place and returned).
        • file_name_suffix : str | None; suffix from create_file_name_suffix; None = no-op when adding.
        • add_suffix      : bool; True to apply the suffix, False to strip it.

    Returns:
        • dict[str, str | dict[str, str]]; the modified file_paths dict.
    """
    if add_suffix and file_name_suffix is not None:
        file_paths["file_name_suffix"] = file_name_suffix
        for key in file_paths["file_names"].keys():
            if any(substr in key for substr in ["_iter", "_fit1", "_bayes", "all"]):
                file_name: str = file_paths["file_names"][key]
                if "." in file_name:
                    fname, extension = file_name.split(".")
                    file_paths["file_names"][key] = fname + file_name_suffix + "." + extension

    else:
        for key in file_paths["file_names"].keys():
            if any(substr in key for substr in ["_iter", "_fit1", "_bayes", "all"]):
                file_name: str = file_paths["file_names"][key]
                if "." in file_name:
                    fname, extension = file_name.rsplit(".", 1)
                else:
                    fname, extension = file_name, ""
                "Remove everything from the first \"~\" onward"
                if "~" in fname:
                    fname = fname.split("~", 1)[0]
                file_paths["file_names"][key] = fname + "." + extension  

    return file_paths


def ensure_directory_and_join(base_dir: str, file_name: str, max_total_path_len: int = 245, max_file_name_len: int = 79) -> str:
    """
    Create (if needed) and return a safe file path under 'base_dir' for 'file_name'.

    Behavior:
        • Sanitizes the base file name (letters, digits, '.', '_', '-'; others → '~').
        • Preserves the final extension, if present ('.json', '.csv', '.html', etc.).
        • Shortens the file name and/or the full path deterministically using a hash
          when exceeding Windows' legacy path limits (≈260 characters).
        • Creates 'base_dir' recursively (safe for concurrent workers).

    Arguments:
        • base_dir: str; directory to contain the file.
        • file_name: str; proposed file name, possibly long/unwieldy.
        • max_total_path_len: int; safeguard upper bound for full path length.
        • max_file_name_len: int; safeguard upper bound for the file name length.

    Returns:
        • str; a safe absolute or relative path guaranteed to exist up to the parent directory.
    """
    "Make sure base_dir exists (idempotent for parallel workers)."
    os.makedirs(base_dir, exist_ok=True)

    "Split extension once; keep it and sanitize only the stem."
    stem, ext = os.path.splitext(file_name)
    if not ext:
        ext = ""  # keep consistent behavior

    "Sanitize stem (filesystem-friendly)."
    stem_sanitized = re.sub(r"[^A-Za-z0-9._-]+", "~", str(stem))

    "After sanitization and initial shortening"
    if len(stem_sanitized) > max_file_name_len - len(ext):
        "Clamp the stem to max length minus extension length"
        stem_sanitized = stem_sanitized[:max(0, max_file_name_len - len(ext) - 3)] + "~"

    "First-pass shortener for the file name itself."
    file_name_candidate = f"{stem_sanitized}{ext}"

    if len(file_name_candidate) > max_file_name_len:
        digest = hashlib.sha1(file_name_candidate.encode("utf-8")).hexdigest()[:10]
        "Keep both ends to remain informative."
        keep = max_file_name_len - len(ext) - len(digest) - 2  # hyphens around hash
        keep = max(12, keep)  # keep something human-readable
        head = stem_sanitized[: keep // 2]
        tail = stem_sanitized[-(keep - len(head)):]
        file_name_candidate = f"{head}-{digest}-{tail}{ext}"

    "Now check full path length and, if needed, fall back to a compact hashed name."
    full_candidate = os.path.join(base_dir, file_name_candidate)
    if len(full_candidate) > max_total_path_len:
        digest = hashlib.sha1(full_candidate.encode("utf-8")).hexdigest()[:16]
        file_name_candidate = f"{digest}{ext}"
        full_candidate = os.path.join(base_dir, file_name_candidate)

    return full_candidate


"=========================================================================================="
"======================================= Variables ========================================"
"=========================================================================================="

utility_settings: UtilitySettings = {
    'conditional_welfare_mode':        False,
    'reference_dependent_altruism':    False,
    'min_max_rawlsian_leontief':       False,
    'include_welfare_efficiency_term': False,
    'include_relative_income_penalty': False,
    'use_exponential_parameters':      True,
    'apply_exponents_to_payoffs':      False,
    'uniform_exponential_parameter':   False,
    'single_payoffs_not_differences':  False,
    'payoff_ratios_not_differences':   False,
    'reference_dependent_utility':     False,
    'use_negativity_parameters':       False,
    'negativity_social_comparison':    True,
    'tie_self_interest_and_altruism':  False,
    'fix_self_interest_parameter':     False,
    'include_social_comparison':       True,
    'include_altruism_term':           True,
}

experiment_num = 3
run_in_parallel = True
track_evolution = False
create_new_file = False
update_method = 'grid'
analysis_mode = 'bayesian'
analysis_unit = 'player'
n_bins_per_dimension = 5
include_covariance = False
softmax_temperature = 1.0
temperature_is_param = True
guess_params_randomly = False
optimization_method = 'globloc'
confidence_weighted = False
use_particle_filter = True
fit_roles_together = False
use_initial_params = True
loss_funct_type = 'log'
penalty_weight = 0.005
write_mode = 'resume'
learning_rate = 0.8
sample_ratio = 0.05
export_fig = True
dark_mode = False


general_settings: GeneralSettings = {
    'update_method': update_method,
    'analysis_mode': analysis_mode,
    'analysis_unit': analysis_unit,
    'experiment_num': experiment_num,
    'loss_funct_type': loss_funct_type,
    'track_evolution': track_evolution,
    'create_new_file': create_new_file,
    'run_in_parallel': run_in_parallel,
    'include_covariance': include_covariance,
    'softmax_temperature': softmax_temperature,
    'optimization_method': optimization_method,
    'confidence_weighted': confidence_weighted,
    'use_particle_filter': use_particle_filter,
    'guess_params_randomly': guess_params_randomly,
    'temperature_is_param': temperature_is_param,
    'n_bins_per_dimension': n_bins_per_dimension,
    'fit_roles_together': fit_roles_together,
    'use_initial_params': use_initial_params,
    'penalty_weight': penalty_weight,
    'learning_rate': learning_rate,
    'sample_ratio': sample_ratio,
    'export_fig': export_fig,
    'write_mode': write_mode,
    'dark_mode': dark_mode,
    'optimization_policy': {
        'n_random_starts'    : 10,
        'maxiter_global'     : 1000,
        'maxiter_local'      : 1000,
        'maxfun_global'      : 1000,
        'maxfun_local'       : 1000,
        'run_trust_constr'   : False,
        'dual_annealing_seed': None,
        'trust_maxiter'      : 600,
        'trust_gtol'         : 1e-6,
        'trust_xtol'         : 1e-8,
        'trust_verbose'      : False,
        'local_methods'      : ['L-BFGS-B'],
    },
    'warmstart_policy': {
        "enabled"                         : True,
        "schedule"                        : "binary",
        "cold_iters"                      : 4,
        "explore_iters"                   : 4,
        "temperature_low"                 : 0.01,
        "temperature_high"                : 1000.0,
        "disable_dual_annealing_when_warm": True,
    },
    'ampd_settings': {
        'metric':                   'normalized_jsd',
        'n_games':                  625,
        'n_iters':                  30,      # set to 250 for full-precision runs
        'parameter_sampling_mode':  'uniform',
        'parameter_pairing_mode':   'shared',
        'player_roles':             None,
        'random_seed':              None,
        'softmax_temperature':      1.5,     # AMPD-specific τ; independent of global softmax_temperature
    },
    'individual_architecture_settings': {
        'population_top_n_models':                   120,
        'participant_top_r_models':                  10,
        'M_max':                                     None,
        'exhaustive_M_max':                          4,
        'score_basis':                               'ic_equivalent_participant_score',
        'stopping_criteria':                         'kneedle_elbow',
        'marginal_gain_threshold':                   0.01,
        'n_consecutive_low_marginal_gains_required': 1,
        'cumulative_gain_threshold':                 0.80,
        'diagnose_selected_library_redundancy':      True,
        'n_workers':                                 None,
    },
    'model_recovery_settings': {
        'generating_model':               utility_settings,
        'n_agents_grid':                  [73],
        'n_games_grid':                   [30, 60, 90, 120, 150],
        'softmax_temperature':            1.0,
        'candidate_model_selection_mode': 'hamming',
        'n_candidate_models':             603,
        'ampd_matrix_name_or_path':       None,
        'random_seed':                    42,
    },
    'parameter_recovery_settings': {
        'n_players':              None,   # → 73
        'fit_predictor_role':     False,
        'n_games':                None,   # → 60
        'k_params_range':         (1, 9),
        'n_altruism_steps':       7,
        'evenly_space_altruism':  True,
        'correlate_all_params':   False,
        'random_seed':            None,
    },
    'random_seeds': {
        'use_seeds': False,
        'seed':      42,
    },
}

"If use_seeds=False, nullify the seed so all downstream code treats it as unseeded."
if not general_settings['random_seeds']['use_seeds']:
    general_settings['random_seeds']['seed'] = None

"Propagate master seed to the three subsystem seed locations so they all use the same value."
general_settings['ampd_settings']['random_seed']               = general_settings['random_seeds']['seed']
general_settings['model_recovery_settings']['random_seed']     = general_settings['random_seeds']['seed']
general_settings['optimization_policy']['dual_annealing_seed'] = general_settings['random_seeds']['seed']

param_bds: ParameterBounds = {
    'Vᵢᵢ': (-1, 1), 'λᵢᵢ': (-1, 1), 'Vᵢⱼ': (-1, 1), 'λᵢⱼ': (-1, 1), 'αᵢⱼ': (-1, 1), 'βᵢⱼ': (-1, 1), 
    'γ1': (1e-4, 2), 'γ2': (1e-4, 2), 'γ3': (1e-4, 2), 'Vᵢᵢ_std': (1e-2, 4), 'λᵢᵢ_std': (1e-2, 4), 
    'Vᵢⱼ_std': (1e-2, 4), 'λᵢⱼ_std': (1e-2, 4), 'αᵢⱼ_std': (1e-2, 4), 'βᵢⱼ_std': (1e-2, 4), 
    'γ1_std': (1e-2, 1), 'γ2_std': (1e-2, 1), 'γ3_std': (1e-2, 1),
}

param_info = make_param_info(param_bds=param_bds, utility_settings=utility_settings, general_settings=general_settings)

txt_color = "white" if dark_mode else "black"
txtfam = "Calibri"

figure_layout: FigLay = {
    "template": "plotly_dark" if dark_mode else "plotly_white",
    "font": dict(family=txtfam, color=txt_color, size=24),
    "tickfont": dict(family=txtfam, color=txt_color, size=30),
    "titlefont_size": 48, "title_x": 0.5, "title_y": 0.96, "scale": ("x", 1),
    "colorscales": ['Viridis', 'Plasma', 'Inferno', 'matter', 'haline', 'thermal', 'dense', 'Magma'],
    "annotations": {"font":  dict(family=txtfam, color=txt_color, size=34), "showarrow": False}, 
    "xaxis" : {"title_font": dict(family=txtfam, color=txt_color, size=34), 
        "tickfont": dict(size=30, family=txtfam, color=txt_color)},
    "yaxis" : {"title_font": dict(family=txtfam, color=txt_color, size=34), 
        "tickfont": dict(size=30, family=txtfam, color=txt_color)},
    "hoverlabel": dict(font_size=30, font_family=txtfam),
    "markersize": 16,
    "base_hue": 200,
}

ROOT = Path(__file__).resolve().parent


def pretty_path(path) -> str:
    """
    Returns a shortened display path relative to ROOT's parent, prefixed with '...'.
    Falls back to the full string if relpath fails (e.g. different drive on Windows).
    """
    try:
        return '...' + os.path.relpath(str(path), str(ROOT.parent))
    except ValueError:
        return str(path)

file_paths: FilePaths = {
    "raw_data":    ROOT / "raw_data",
    "processed":   ROOT / "processed",
    "param_data":  ROOT / "param_data",
    "player_fits": ROOT / "player_fits",
    "simulations": ROOT / "simulations",
    "dyad_data":   ROOT / "dyad_data",
    "discrete":    ROOT / "discrete",
    "visuals":     ROOT / "visuals",
    "bic_aic":     ROOT / "bic_aic",
    "file_names": {
        "player_pairs_exper0": "Social_Preference_Prediction_Pairs_Exper0.json",
        "player_pairs_exper1": "Social_Preference_Prediction_Pairs_Exper1.json",
        "player_pairs_exper2": "Social_Preference_Prediction_Pairs_Exper2.json",
        "player_pairs_exper3": "Social_Preference_Prediction_Pairs_Exper3.json",
        "processed_data_exper0": "Social_Preference_Prediction_Processed_Results_Exper0.csv",
        "processed_data_exper1": "Social_Preference_Prediction_Processed_Results_Exper1.csv",
        "processed_data_exper2": "Social_Preference_Prediction_Processed_Results_Exper2.csv",
        "processed_data_exper3": "Social_Preference_Prediction_Processed_Results_Exper3.csv",
        "params_data_exper1_iter": "Social_Preference_Prediction_Parameters_Exper1_Iter.json",
        "params_data_exper2_iter": "Social_Preference_Prediction_Parameters_Exper2_Iter.json",
        "params_data_exper3_iter": "Social_Preference_Prediction_Parameters_Exper3_Iter.json",
        "params_data_exper1_fit1": "Social_Preference_Prediction_Parameters_Exper1_Fit1.json",
        "params_data_exper2_fit1": "Social_Preference_Prediction_Parameters_Exper2_Fit1.json",
        "params_data_exper3_fit1": "Social_Preference_Prediction_Parameters_Exper3_Fit1.json",
        "params_hist_exper1_iter": "Social_Preference_Prediction_Parameters_Exper1_Iter.html",
        "params_hist_exper2_iter": "Social_Preference_Prediction_Parameters_Exper2_Iter.html",
        "params_hist_exper3_iter": "Social_Preference_Prediction_Parameters_Exper3_Iter.html",
        "params_hist_exper1_fit1": "Social_Preference_Prediction_Parameters_Exper1_Fit1.html",
        "params_hist_exper2_fit1": "Social_Preference_Prediction_Parameters_Exper2_Fit1.html",
        "params_hist_exper3_fit1": "Social_Preference_Prediction_Parameters_Exper3_Fit1.html",
        "params_data_exper0_bayes": "Social_Preference_Prediction_Parameters_Exper0_Bayes.json",
        "params_data_exper1_bayes": "Social_Preference_Prediction_Parameters_Exper1_Bayes.json",
        "params_data_exper2_bayes": "Social_Preference_Prediction_Parameters_Exper2_Bayes.json",
        "params_data_exper3_bayes": "Social_Preference_Prediction_Parameters_Exper3_Bayes.json",
        "params_hist_exper1_bayes": "Social_Preference_Prediction_Parameters_Exper1_Bayes.html",
        "params_hist_exper2_bayes": "Social_Preference_Prediction_Parameters_Exper2_Bayes.html",
        "params_hist_exper3_bayes": "Social_Preference_Prediction_Parameters_Exper3_Bayes.html",    
        "all_fitted_data_bayesian": "Social_Preference_Prediction_Parameters_Bayes.csv",
        "all_fitted_data_naive": "Social_Preference_Prediction_Parameters_Naive.csv",
        "all_fitted_data_mle": "Social_Preference_Prediction_Parameters_MLE.csv",
        "raw_data_exper1": "Judgment_Game_Data_Experiments_1abcd_Post-exclusions.csv",
        "raw_data_exper2": "Judgment_Game_Data_Experiment_2_Pre-exclusions.csv",
        "raw_data_exper3": "Morality_Game_Iter_bDG_Results_Combined.csv",
        "players_to_dyads_exper0": "SPPP_Exper0_Players_to_Dyad_Keys.json",
        "players_to_dyads_exper1": "SPPP_Exper1_Players_to_Dyad_Keys.json",
        "players_to_dyads_exper2": "SPPP_Exper2_Players_to_Dyad_Keys.json",
        "players_to_dyads_exper3": "SPPP_Exper3_Players_to_Dyad_Keys.json",
        "information_criterion": "All_Utility_Forms_IC_Analysis_Experiment3.csv",
        "problematic_pairs": "parent_worse_than_child_pairs.csv",
        "embedding_equality": "embedding_equality_results.csv",
        "all_minimal_pairs": "minimal_pairs_summary.csv",
    }
}

file_name_suffix = create_file_name_suffix(
    general_settings=general_settings, utility_settings=utility_settings)

file_paths = add_remove_file_name_suffix(
    file_paths=file_paths, file_name_suffix=file_name_suffix, add_suffix=True)

column_names: ColumnNames = {
    "exper_1A": {
        'Trial_Order': 'round', 'Avatar_ID': 'player_uuid_p0', 'Participant_ID': 'player_uuid_p1', 
        'As_OP': 'payoffs_A_p0_op', 'Ao_OP': 'payoffs_A_p1_op', 'Bs_OP': 'payoffs_B_p0_op', 'Bo_OP': 'payoffs_B_p1_op', 
        'As_RP': 'payoffs_A_p0_rp', 'Ao_RP': 'payoffs_A_p1_rp', 'Bs_RP': 'payoffs_B_p0_rp', 'Bo_RP': 'payoffs_B_p1_rp', 
        'Prediction': 'prediction__p1', 'Subjective_Probability': 'subjective_probability', 	
    },
    "exper_2A": {
        'Trial_order': 'round', 'Avatar': 'player_uuid_p0', 'Participant_ID': \
            'player_uuid_p1', 'Avatar_Type': 'avatar_type', 'Avatar_Pic': 'avatar_shape', 
        'As_OP': 'payoffs_A_p0_op', 'Ao_OP': 'payoffs_A_p1_op', 'Bs_OP': 'payoffs_B_p0_op', 'Bo_OP': 'payoffs_B_p1_op', 
        'As_RP': 'payoffs_A_p0_rp', 'Ao_RP': 'payoffs_A_p1_rp', 'Bs_RP': 'payoffs_B_p0_rp', 'Bo_RP': 'payoffs_B_p1_rp', 
        'Cs': 'payoffs_selected_p0', 'Co': 'payoffs_selected_p1', 'Rs': 'payoffs_rejected_p0', 'Ro': 'payoffs_rejected_p1',
        'Prediction': 'prediction__p1', 'Subjective_Probability': 'subjective_probability', 	
    },
    "exper_2B": [
        'round', 'player_uuid_p0', 'player_uuid_p1', 
        'payoffs_A_p0_op', 'payoffs_A_p1_op', 'payoffs_B_p0_op', 'payoffs_B_p1_op', 
        'payoffs_A_p0_rp', 'payoffs_A_p1_rp', 'payoffs_B_p0_rp', 'payoffs_B_p1_rp', 
        'choice__p0', 'prediction__p1', 'subjective_probability', 
    ],
    "exper_3A": [
        'round', 'room', 'batch', 'timestamp', 'title', 'player_uuids', 'player_types', 'avatar_shapes', 
        'avatar_colors', 'players_abdicated', 'adjacency_matrix', 'final_node', 'payoffs_A_p0', 'payoffs_A_p1', 
        'payoffs_B_p0', 'payoffs_B_p1', 'choice__p0', 'prediction__p1', 'choice_data__p0', 'prediction_data__p1'
    ],
        "exper_3B": [
        'timeslot', 'batch', 'round', 'room', 'timestamp', 'title', 'player_uuid_p0', 'player_uuid_p1', 'player_type_p0', 
        'player_type_p1', 'avatar_shape_p0', 'avatar_shape_p1', 'avatar_color_p0', 'avatar_color_p1', 'abdicated_p0', 'abdicated_p1', 
        'matching_probability', 'final_node', 'payoffs_A_p0', 'payoffs_A_p1', 'payoffs_B_p0', 'payoffs_B_p1', 'choice__p0', 'prediction__p1'
    ],
}

valid_analysis_types = [
    "within_player",
    "within_dyad_symmetry",
    "within_dyad_accuracy",
    "across_dyad_one_chooser_many",
    "across_dyad_many_predictors_one_chooser",
    "across_analysis_modes",
]

def _load_canonical_utility_specs() -> dict[str, dict]:
    _new_flags = ('include_welfare_efficiency_term', 'include_relative_income_penalty')
    with open(ROOT / "canonical_utility_settings.json", "r", encoding="utf-8") as _f:
        raw = json.load(_f)
    return {
        name: {**spec, **{flag: False for flag in _new_flags if flag not in spec}}
        for name, spec in raw.items()
    }

CANONICAL_UTILITY_SPECS: dict[str, dict] = _load_canonical_utility_specs()