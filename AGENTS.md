# AGENTS.md — Hidden Motives Analysis

This file orients an AI collaborator to the project, codebase, and collaboration style.
Read this before writing or suggesting any code changes.

---

## 1. What this project is

This repository implements the **Utility Bayesian Model (UBM)** from *Inferring Hidden Motives:
Bayesian Models of Preference Learning in Repeated Dictator Games* (Stanley, Zhang, & Lewis, 2025;
arXiv). The UBM formalizes how an observer (the *predictor*) updates beliefs about another
person's social preferences (altruism, envy, guilt) from their sequential payoff-allocation choices
in iterated binary dictator games.

The paper has been submitted and is currently being revised. The code is close to finished but
still needs cleanup and stylistic consistency work. A follow-up study is planned that integrates
a higher-order Theory of Mind model with social-preference belief updating, working on arbitrary
game trees — the intention is to make the code more like a general-purpose toolkit for that work.

For an accessible orientation, read the [`README.md`](README.md) and the files in [`docs/`](docs/).

---

## 2. Domain vocabulary

Understanding these terms is necessary for reading the code and contributing correctly.

| Term | Meaning |
|------|---------|
| **UBM** | Utility Bayesian Model — the core model; continuously updates beliefs about social preferences from sequential choices |
| **chooser** | The player who makes binary payoff-allocation decisions in each game |
| **predictor** | The observer who watches the chooser and updates beliefs about their preferences; the UBM models the predictor |
| **dyad** | A chooser–predictor pair who play a series of games together |
| **game** | A single round within a dyad: one binary allocation choice by the chooser |
| **dyad_key** | String identifier for a dyad, e.g., `"player_1__player_2"` |
| **player_uuid** | Unique identifier for a player (typically an integer or UUID string) |
| **utility function** | The functional form that determines how much the chooser values different allocations |
| **utility_settings** | Dict of 13+ boolean toggles that define which terms are active in the utility function |
| **param_bds** | Dict of (lower, upper) bounds for each parameter (Vᵢᵢ, Vᵢⱼ, γ1, etc.) |
| **param_info** | Full parameter specification dict: bounds, initial guesses, and optionally covariance |
| **general_settings** | Master config dict controlling experiment_num, analysis_mode, optimization, parallelism, etc. |
| **fig_lay** | Plotly figure layout template; controls fonts, colors, sizing |
| **file_paths** | Dict of directory paths for raw data, processed outputs, figures, etc. |
| **column_names** | Experiment-specific column name mappings for raw data CSV files |
| **Vᵢᵢ** | Self-interest weight in the utility function |
| **Vᵢⱼ** | Social-comparison (altruism) weight |
| **Ʌᵢᵢ / Ʌᵢⱼ** | Negativity counterparts (used when `use_negativity_parameters=True`) |
| **γ1, γ2, γ3** | Curvature/exponent parameters (used when `use_exponential_parameters=True`) |
| **grid update** | Full discretized posterior update over the parameter grid |
| **particle filter** | Approximation to the grid update using a random sample of particles |
| **prior / posterior** | Bayesian terms; prior = beliefs before seeing a game; posterior = beliefs after |
| **agent()** | The core UBM function; runs sequentially over all games in a dyad, updating beliefs each round |
| **bayesian_update_grid()** | Single-round posterior update via grid/particle filter |
| **analysis_mode** | `'bayesian'` or `'mle'`; controls whether the UBM or maximum likelihood is used for fitting |
| **experiment_num** | 1, 2, or 3 — the three empirical experiments in the paper |
| **IC analysis** | Information criterion analysis comparing 480 candidate utility functional forms by AIC/BIC |
| **model nesting** | The hierarchical relationship where simpler utility forms are special cases of richer ones |
| **parent / child model** | In nesting: a parent model reduces to its child when extra parameters take special values |
| **warm-starting** | Initializing optimization from the best previously found parameter values |
| **softmax_temperature** | Controls how deterministic choices are (lower = more deterministic) |
| **NLL** | Negative log-likelihood — the loss function for both MLE and Bayesian fitting |

---

## 3. Key files and folders

```
main.py            ← orchestrator only; imports from analysis and invokes run_code_settings blocks
config.py          ← all settings, paths, type aliases, param specs; single source of truth
model.py           ← core utility/choice functions: utility_term, utility, softmax_, choice,
                       build_utility_equation, make_param_info, parameter_keys_for_utility_settings
optimization.py    ← shared optimization infrastructure: compute_ic, global_local_optimization,
                       global_local_then_trust_constr, best_initial_guesses, warm-starting helpers
bayesian.py        ← Bayesian fitting pipeline: bayesian_update_grid, agent, loss_function_bayes,
                       fit_params_by_player, run_analysis_bayes, _worker_fit_one
simulation.py      ← simulation utilities: create_simulated_dyad, create_simulated_data,
                       run_simulation_recovery_analysis, parameter recovery, update speed analyses
visualization.py   ← all plotting functions and _hsla color helper
analysis.py        ← Stage 5–12 analyses: AMPD, model-space geometry, IC extraction, MDS,
                       compression curve, model recovery simulation
mle.py             ← legacy — MLE fitting pipeline; still callable via analysis_mode='mle' but
                       not used in the paper analyses
preprocessing.py   ← data loading, dyad construction, experiment-specific cleaning
utilities.py       ← general helpers: utility enumeration, model nesting, IC utilities
typological.py     ← discrete/typological Bayesian variants (parallel to the continuous UBM)
quick_demo.py      ← entry point for the reproducible demo; runs a working subset of the pipeline

docs/
  architecture.md  ← high-level architecture notes
  core_function_map.md ← guided map of core functions with explanations of why they exist
  data_dictionary.md   ← data structures and file outputs

raw_data/          ← raw experiment CSVs
processed/         ← cleaned intermediate data
param_data/        ← per-player parameter grids saved between runs
player_fits/       ← JSON files with fitted parameter estimates
dyad_data/         ← per-dyad data files
bic_aic/           ← IC analysis output CSVs
visuals/           ← exported Plotly HTML figures
demo_files/        ← all demo outputs (safe to delete)
plans/             ← saved implementation plans (markdown); see Section 9
```

**Import flow (no circular dependencies):**
```
config.py       →  (nothing from this project)
preprocessing   →  config
utilities       →  config, preprocessing
typological     →  config, preprocessing
model.py        →  config (via *)
optimization.py →  model (via *)
bayesian.py     →  optimization (via *)
simulation.py   →  bayesian (via *)
visualization.py→  simulation (via *)
analysis.py     →  visualization (via *)
mle.py          →  optimization (via *)   [legacy; not imported by any active module]
main.py         →  analysis (via *), mle.run_analysis_mle (explicit; for the one legacy call-site)
```

---

## 4. How to run

For the quick demo:
```bash
python quick_demo.py
```

For the full pipeline, set the `run_code_settings` flags at the bottom of `main.py` and run:
```bash
python main.py
```

All settings live in `config.py`. The `general_settings`, `utility_settings`, and `param_bds`
dicts control all behavior. The defaults in the repo are the settings used in the paper.

---

## 5. Architecture

### The active fitting approach

All active fitting uses the **Bayesian UBM** pipeline in `bayesian.py`:

```
agent() → bayesian_update_grid() → loss_function_bayes() → fit_params_by_player() → run_analysis_bayes()
```

The `agent()` function runs the full sequential belief-updating UBM over all games in a dyad.
`loss_function_bayes()` is the NLL-style loss used in the IC analysis.

**The MLE pipeline in `mle.py` is legacy** — still callable via `analysis_mode='mle'` in
`general_settings`, but not used in the paper analyses. Do not extend it or call it from new
code. The call-site in `main.py` is retained only so the option remains accessible.

### `general_settings` and `utility_settings`

`general_settings` controls: which experiment to run, analysis mode (bayesian vs. mle),
whether to parallelize, how many bins per dimension, particle filter settings, optimization
policy, warm-starting strategy, figure export settings, and file write mode.

`utility_settings` is a dict of 13 boolean toggles that selects among 480 possible utility
functional forms. Every function that computes utilities receives `utility_settings` as a
parameter and reads these toggles to decide which terms to activate. This is what enables the
large-scale IC model comparison without code duplication.

### Optimization

The optimizer chain is: simulated annealing (`dual_annealing`) → L-BFGS-B local refinement,
with an optional trust-region constrained step. Warm-starting from previous fits is controlled
by `warmstart_policy`. All optimization entrypoints share common infrastructure in
`global_local_optimization()` and `global_local_then_trust_constr()`.

### IC analysis

`information_criterion_analysis()` iterates over all 480 combinations of `utility_settings`,
fits each to data, and computes AIC/BIC. Model-nesting-aware warm-starting (child→parent
parameter mappings) prevents nesting violations where a richer model appears to fit worse than
its simpler nested version.

### Pipeline robustness: call generating functions, not raw `pd.read_csv`

**This is a fundamental repo convention — preserve it in all new functions.**

Any function that depends on a CSV produced by another function in this codebase must call
that generating function (with `create_new_file=False`) rather than calling `pd.read_csv`
on the output path directly.  The generating function knows whether its output is complete
and will finish computation or load from cache as appropriate.  A bare `pd.read_csv` call
cannot detect or recover from an incomplete, stale, or missing file — it will either crash
with a parse error or silently return wrong data.

```python
"Good — call the generating function; it handles caching and completeness checks"
combined_fits_df = extract_participant_model_combined_fits(
    general_settings=general_settings,
    file_paths=file_paths,
    create_new_file=False,
)

"Bad — bypasses the generating function's completeness checks and caching logic"
combined_fits_df = pd.read_csv(
    os.path.join(file_paths['processed'], 'participant_model_combined_fits.csv'),
    encoding='utf-8-sig',
)
```

When the generating function has already been called earlier in the same pipeline run,
passing `create_new_file=False` means the second call is essentially free — it reads from
the in-memory cache path without recomputing.

**Corollary — never write a degenerate output file**: if a computation produces an empty
DataFrame (zero rows), do not write it to disk.  An empty CSV causes `pd.read_csv` to
throw `EmptyDataError` in every downstream function.  Check `df.empty` before calling
`to_csv` and print a clear diagnostic if the check fails.

```python
"Good — guard before writing"
if result_df.empty:
    print("Warning: computation produced 0 rows — output CSV not written.")
    return result_df
result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
```

---

### CSV-from-settings pattern

**This is a fundamental repo convention — preserve it in all new functions.**

Functions that produce or consume named CSV files resolve the filename from the globally
accessible settings dicts (`general_settings`, `utility_settings`, `param_bds`) rather than
requiring the caller to construct paths or load DataFrames manually. Callers pass
`general_settings` and `file_paths`; the function derives the exact filename from the settings
that were used to produce the file, opens it, and returns the result.

This pattern is used throughout the IC analysis (`bic_aic/` CSVs), model nesting
(`processed/model_nesting_adjacency_*.csv`), and all Stage 5+ analyses. The AMPD matrix is
the clearest example:

```python
"Caller (main.py) — no path construction, no DataFrame loading"
fig = plot_model_space_mds(
    general_settings=general_settings, file_paths=file_paths, fig_lay=fig_lay,
)

"Inside plot_model_space_mds — auto-resolves the matrix from settings"
if distance_matrix_df is None:
    distance_matrix_df = _load_ampd_matrix_from_settings(general_settings, file_paths)
```

**AMPD-specific helpers** (in `analysis.py`, defined just before the Stage 5 section):

- `_load_ampd_matrix_from_settings(general_settings, file_paths)` — builds the canonical
  AMPD master matrix path from `general_settings['ampd_settings']`, checks it exists, loads
  it, casts index/columns to `int`, and returns the DataFrame.
- `_ampd_distance_name(general_settings)` — returns the `distance_name` label string (e.g.
  `"ampd_uniform_shared"`) derived from the same settings.
- `_build_ampd_cache_path(file_paths, metric, ...)` — builds the actual file path; the two
  helpers above call this internally.

**CSV encoding — non-negotiable rule**: Every `to_csv()` call in this codebase must include
`encoding='utf-8-sig'`, without exception. This BOM variant ensures that Unicode characters
in column names and cell values — Greek letters, mathematical Unicode symbols such as
Vᵢᵢ, Vᵢⱼ, Ʌᵢᵢ, τ, 𝑘, Δ, etc. — survive a round-trip through Excel, Windows file
dialogs, and `pd.read_csv()` without corruption or replacement characters. **A `to_csv()` call
without `encoding='utf-8-sig'` is always a bug in this codebase.** When adding any new CSV
write, treat the encoding argument as mandatory, not optional.

**`general_settings['ampd_settings']`** is a nested dict (added in `config.py`) with keys:
`metric`, `n_games`, `n_iters`, `parameter_sampling_mode`, `parameter_pairing_mode`,
`player_roles`, `random_seed`. All AMPD-specific parameters in `compute_ampd_distance_matrix`
default to `None`; when `None`, the function reads the value from this dict. The settings
dict is the single source of truth so that `compute_ampd_distance_matrix` and every
downstream Stage 5+ function always agree on which file to read and write.

---

## 6. Coding style — read this carefully

This is the most important section for any AI collaborator. Greg's style should be preserved
and extended consistently across the codebase. Do not "clean up" code to match generic Python
conventions.

### Long, descriptive variable names

Variable and argument names should be long and self-documenting. Avoid abbreviations beyond
established domain shorthand (e.g., `dv`, `nll`, `uuid`). **Single-letter variable names are
never acceptable — no exceptions.** This applies to loop indices (`i`, `j`, `k`, `n`),
matrix shorthands (`W`, `D`, `H`, `B`), and any other context. Short names are hard to
search and rename reliably. Use a descriptive suffix if you need to follow a mathematical
convention (e.g., `gram_matrix_B`, `centering_matrix_J`, `n_entities`).

```python
"Good"
chooser_posterior_mean_estimates = ...
fitted_parameter_values_for_all_dyads = ...
loss_for_current_parameter_guess = ...
for participant_idx, player_uuid in enumerate(all_player_uuids):  ...
gram_matrix_B = -0.5 * (centering_matrix_J @ squared_distances @ centering_matrix_J)

"Bad — do not write these"
est = ...
fitted_params = ...
loss = ...
for i, uuid in enumerate(uuids):  ...
W = ...   # weight matrix
B = ...   # gram matrix
```

### Docstrings with bullet-point argument lists

Docstrings use a specific format: a plain-English description, then an `Arguments:` block
where each argument is introduced with `•`, and a `Returns:` block.

```python
def example_function(first_argument: pd.DataFrame, second_argument: bool) -> pd.DataFrame:
    """
    One- to three-sentence description of what the function does.

    Arguments:
        • first_argument: pd.DataFrame
            What this argument is and how it is used.
        • second_argument: bool
            What this argument controls.

    Returns:
        • pd.DataFrame — description of the return value.
    """
```

### Comments as plain strings, not `#` lines

Block comments and section labels are written as standalone string literals on their own line,
not with `#`. This is intentional and should be maintained throughout the codebase. Inline
`#` comments within a line of code are acceptable for very short annotations, but standalone
comment blocks should use the string form.

```python
"Good — this is how block comments are written"
posterior_weights = posterior_weights / posterior_weights.sum()

# Bad — do not introduce standalone hash-comment blocks
```

### Section headers

Major sections use the 90-character separator style already present in the file:
```python
"=========================================================================================="
"===================================== Section Title ======================================"
"=========================================================================================="
```

The middle line must have the title centered within the `=` characters (one space padding on each
side). Use this formula, where **N** = total `=` count in the top row and **T** = character
length of the title text:

```
n_left  = (N - T - 2) // 2      # integer division; the -2 accounts for the two space pads
n_right = N - T - 2 - n_left    # equals n_left + 1 when (N - T - 2) is odd
```

When `N - T - 2` is odd the right side gets the extra `=`. Example: a 90-`=` top row with
"My Title" (8 chars) → `(90 - 8 - 2) // 2 = 40` left and 40 right — even, perfectly centered.
With "My Titles" (9 chars) → `(90 - 9 - 2) // 2 = 39` left and 40 right.

Section headers must be preceded by **two blank lines** and followed by **one blank line**:
```python
<two blank lines>
"=========================================================================================="
"===================================== Section Title ======================================"
"=========================================================================================="
<one blank line>
```

### Always use keyword arguments in function calls

**Every** call to a project-defined function must use keyword arguments — no exceptions.
Positional-only calls are acceptable for standard Python builtins (`len`, `range`, `print`,
`sorted`, `isinstance`, etc.) but never for anything defined in this codebase.

This rule applies equally to:
- calls inside the same module as the definition,
- single-argument calls (e.g., `f(x=value)` not `f(value)`),
- calls where the argument order feels "obvious".

```python
"Good — every argument is named, even single-argument calls"
stable_bot_id(params=params_predictor, player_role='predictor', n_games=n_games)

gnrl.classify_pair_relation(
    model_1=child_utility_settings, model_2=parent_utility_settings,
    utility_settings=utility_settings, general_settings=general_settings,
)

"Bad — positional arguments to project functions are never acceptable"
stable_bot_id(params_predictor, 'predictor', n_games)
gnrl.classify_pair_relation(child_utility_settings, parent_utility_settings)
```

### Comment capitalization and consolidation

- Comments (string literals used as block comments) must start with an upper-case letter, unless
  the very first word is a variable or function name that is lower-case by convention.
- A run of two or more consecutive single-line comment strings must be consolidated into one
  triple-quoted string rather than written as separate `"..."` lines.
- Comments describe what the code *does* and are addressed to a third-party researcher,
  developer, collaborator, or AI assistant who has no prior context for the project — not to
  Greg personally. Write as if explaining to a capable stranger, not as a personal note.

```python
"Good — single comment, upper-case, third-person framing"
"Normalize the posterior so it sums to one before the next update."
posterior_weights = posterior_weights / posterior_weights.sum()

"""
Good — multiple related lines consolidated into one triple-quoted block.
The first step converts each parameter vector to an index tuple so the grid
can be addressed by integer coordinates rather than raw float values.
"""
index_tuples = [param_vector_to_index(v) for v in parameter_vectors]

"Bad — lower-case start (unless first word is a variable name)"
"normalize the posterior..."

"Bad — two separate single-line strings that should be one block"
"Step 1: convert vectors."
"Step 2: build grid."
```

### HSLA color scheme for multi-series figures

All multi-series figures in this codebase use a consistent HSLA-based color scheme built around
`_hsla()` in `visualization.py`. The system mirrors the pattern established in the companion
responsibility-shielding project.

**The rule:** when a figure has N series (bars, violin plots, scatter traces, vertical markers,
etc.), each series gets a hue that is `base_hue + 20 × series_index` degrees. Saturation,
lightness, and alpha stay fixed across series unless deliberately varied for visual emphasis
(e.g., a darker border vs. a lighter fill).

```python
"Standard pattern — iterate series and offset hue by 20° each time"
base_hue = fig_lay.get('base_hue', 200)
for series_index, series_label in enumerate(series_labels):
    fill_color = _hsla(hue=base_hue + 20 * series_index, alpha=0.55)
    line_color = _hsla(hue=base_hue + 20 * series_index, alpha=1.00)
```

**`_hsla(hue, saturation_percent=100, lightness_percent=50, alpha=0.9)`** — defined in
`visualization.py`. Returns a Plotly/CSS-compatible `'hsla(H, S%, L%, a)'` string. Hue is
automatically modulo-360'd, so `_hsla(hue=380)` is the same as `_hsla(hue=20)`.

**`base_hue`** — stored in `fig_lay` (default 200, a medium blue). Every plot function reads
it from `fig_lay.get('base_hue', 200)` so the whole figure set shifts together when the
researcher changes the base color.

**Varying lightness and alpha for fill vs. outline** is the standard way to give a series depth
without breaking the hue ladder:
```python
fill  = _hsla(hue=base_hue + 20 * idx, lightness_percent=55, alpha=0.45)
line  = _hsla(hue=base_hue + 20 * idx, lightness_percent=35, alpha=1.00)
point = _hsla(hue=base_hue + 20 * idx, alpha=0.65)
```

Do not use named CSS colors, raw hex strings, or `rgba()` strings for any color in a figure —
always go through `_hsla()` so that the color scheme stays cohesive and is trivially adjustable
via `base_hue`. The `rgba()` format is permanently banned from plot code; grey neutrals are
expressed as `_hsla(hue=0, saturation_percent=0, lightness_percent=<L>, alpha=<a>)` where
L≈63 for medium grey and L≈78 for light grid lines.

### Terminal output is for third-party readers

All text printed to the terminal must be interpretable by someone with no prior knowledge of
this project's internal structure — a collaborator, a reviewer, or a future maintainer reading
the output cold. This means:

- **No internal shorthand** like `[Stage 9]`, `[S5]`, or `[run_ampd]`. These labels are
  meaningless to anyone outside the immediate development context. Use a short noun phrase
  that describes what is being computed, e.g., `"Architecture compression curve:"` or
  `"AMPD behavioral-distance matrix:"`.
- **No undocumented abbreviations** in message text. Write `utility index` not `utility_idx`,
  `population IC winner` not `IC winner`, `candidate filtering` not just `filtering`.
- **Warnings must be self-explanatory.** A message like `"Warning: could not load AMPD matrix
  ({exc}); AMPD columns will be NaN."` tells the reader what was attempted, what failed, and
  what the consequence is — all without needing to know what AMPD stands for internally.

The guiding question is: *if a collaborating researcher saw this line in a terminal log a year
from now, would they understand what was happening without opening the source file?*

### Progress printing in long-running functions

Any function that is expected to run for more than a few seconds should print
periodic progress feedback so the researcher can monitor status without having to
instrument the code themselves. Good feedback includes: what is currently being
computed, how far along the loop is, and — when estimable — the remaining wall time.

The AMPD matrix computation (`compute_ampd_distance_matrix`) and the IC analysis
(`information_criterion_analysis`) are the canonical examples of this pattern. Study
them before adding progress printing to a new function. Key conventions:

- Print a startup banner that shows the total work to be done (number of pairs,
  players, models, etc.) and the output file path.
- Inside the main loop, print after every *N*th iteration (controlled by a
  `print_every_x_pairs` or similar argument), not on every iteration.
- Each progress line should include: iteration count, completion percentage, elapsed
  time, and estimated time remaining (ETA). Compute ETA as
  `elapsed * (total - done) / done`.
- Print a completion banner at the end (total time, output file, row count, etc.).

```python
"Good — AMPD-style progress line"
elapsed = time.time() - start_time
eta_seconds = elapsed * (n_total_pairs - n_pairs_done) / n_pairs_done
print(
    f"  Pair {n_pairs_done}/{n_total_pairs}  "
    f"({100 * n_pairs_done / n_total_pairs:.1f}%)  "
    f"elapsed {elapsed:.0f}s  ETA {eta_seconds:.0f}s"
)
```

### Inner helper functions

Helper functions that are **only called by one parent function** must be defined as
inner functions (closures) within that parent, not at module level. This keeps the
module namespace clean and makes clear that the helper is not part of any public API.

```python
"Good — _compute_weights is only used inside fit_player, so it lives inside"
def fit_player(data: pd.DataFrame) -> Dict[str, float]:
    def _compute_weights(likelihoods: np.ndarray) -> np.ndarray:
        return likelihoods / likelihoods.sum()
    weights = _compute_weights(likelihoods=raw_likelihoods)
    ...

"Bad — _compute_weights is promoted to module level unnecessarily"
def _compute_weights(likelihoods: np.ndarray) -> np.ndarray:
    return likelihoods / likelihoods.sum()

def fit_player(data: pd.DataFrame) -> Dict[str, float]:
    weights = _compute_weights(likelihoods=raw_likelihoods)
    ...
```

The exception: helpers shared by two or more functions stay at module level (or at
the nearest common enclosing scope). Before moving any `_`-prefixed function inside
a parent, grep to confirm it has exactly one call site.

### Design philosophy

The overriding goal is **maximum readability with minimum working memory overhead**. A reader
should understand what any line does without holding context from twenty lines earlier.

- Prefer clarity over brevity
- Name things for what they *are*, not what they *do*
- Let the variable name carry the meaning so comments are rarely needed
- Do not introduce abstractions or helpers beyond what the task requires

---

## 7. Future work context

The planned follow-up study integrates **higher-order Theory of Mind** (recursive belief
reasoning) with **social preference belief updating** (the current UBM), running on
**arbitrary game trees** rather than iterated binary dictator games. The intent is to
generalize this codebase into a more modular toolkit — the current experiments and
experimental paradigm are one instance of a more general class of problems.

When working on extensions or new analyses, keep this in mind:

- Favor parameterization over hardcoding
- Keep experimental-paradigm-specific logic in `preprocessing.py` and clearly separate from model logic
- Prefer architecture that generalizes to arbitrary game structures rather than assuming dictator games

---

## 8. Module contents

The codebase has been split from the original monolithic `main.py` into focused modules.
`main.py` is now only an orchestrator that sets `run_code_settings` flags and calls `main()`.

| Module | Key sections / functions |
|--------|--------------------------|
| `model.py` | `utility_term`, `utility`, `softmax_`, `choice`, `build_utility_equation`, `make_param_info`, `parameter_keys_for_utility_settings` |
| `optimization.py` | `compute_ic`, `global_local_optimization`, `global_local_then_trust_constr`, `best_initial_guesses`, warm-starting helpers |
| `bayesian.py` | `bayesian_update_grid`, `agent`, `loss_function_bayes`, `fit_params_by_player`, `run_analysis_bayes`, `_worker_fit_one` |
| `simulation.py` | `create_simulated_dyad`, `create_simulated_data`, `stable_bot_id`, `run_simulation_recovery_analysis`, `run_param_recovery_by_k`, `verify_particle_filter_fidelity` |
| `visualization.py` | `_hsla`, all `plot_*` functions for belief updates, parameter distributions, accuracy |
| `analysis.py` | Stage 5: `average_model_policy_distance`, `compute_ampd_distance_matrix`, `compute_model_space_embedding` · Stage 6: `extract_participant_model_combined_fits` · Stage 7: `compute_participant_architecture_embedding`, `compute_participant_feature_support` · Stage 9: `compute_architecture_compression_curve`, `plot_architecture_compression_curve` · Stage 12: `_recovery_fit_worker`, `compute_model_recovery_simulation`, `plot_model_recovery_simulation` |
| `mle.py` | **legacy** — `loss_function_mle`, `fit_one_player_one_role_mle`, `fit_dyad_parameters_mle`, `run_analysis_mle` and helpers; callable via `analysis_mode='mle'` but not used in the paper analyses |

---

## 9. Plans folder

Detailed implementation plans are stored in [`plans/`](plans/) at the project root.

**When to save a plan**: Any plan with more than ~3 stages or with stage-level
implementation specs (function signatures, column names, mathematical formulas,
multi-step algorithms) should be saved here so it can be recovered after context
compression.

**How to save a plan**: Copy the plan document wholesale — do not rewrite or
summarize it. The Claude internal plans directory (`~/.claude/plans/`) is the
authoritative source during a session; when that plan has been approved and is about
to be executed, copy the file byte-for-byte to `plans/<descriptive_name>.md`. The
project copy serves as the permanent record; the Claude copy is ephemeral.

**Naming convention**: Use `snake_case` with a short descriptive label, e.g.,
`participant_fit_extraction_and_realistic_ampd.md`. Match what the plan actually
describes, not the internal Claude task slug.

Current plans:

| File | Contents |
|------|----------|
| [`participant_fit_extraction_and_realistic_ampd.md`](plans/participant_fit_extraction_and_realistic_ampd.md) | Stage 6 extraction of per-participant × per-model fits from IC JSON; `participant_sampled` AMPD mode |
| [`population_architecture_compression_curve.md`](plans/population_architecture_compression_curve.md) | Stage 9 compression curve: scoring basis, candidate filtering, exhaustive + greedy-swap search, stopping criteria, AMPD diagnostics, output files |
