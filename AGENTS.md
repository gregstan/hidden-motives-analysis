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
main.py            ← core model, optimization, simulation, visualization, analyses; ~16,000 lines
config.py          ← all settings, paths, type aliases, param specs; single source of truth
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
```

**Import flow (no circular dependencies):**
```
config.py     →  (nothing from this project)
preprocessing →  config
utilities     →  config, preprocessing
typological   →  config, preprocessing
main.py       →  config (via *), preprocessing (as prep), utilities (as gnrl), typological (as typo)
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

### The two-layer structure

The codebase has two parallel fitting approaches:

- **Bayesian (UBM)**: `agent()` → `bayesian_update_grid()` → `loss_function_bayes()` → `fit_params_by_player()` → `run_analysis_bayes()`
- **MLE**: `loss_function_mle()` → `fit_one_player_one_role_mle()` → `fit_dyad_parameters_mle()` → `run_analysis_mle()`

Both share the same utility and optimization infrastructure.

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

---

## 6. Coding style — read this carefully

This is the most important section for any AI collaborator. Greg's style should be preserved
and extended consistently across the codebase. Do not "clean up" code to match generic Python
conventions.

### Long, descriptive variable names

Variable and argument names should be long and self-documenting. Avoid abbreviations beyond
established domain shorthand (e.g., `dv`, `nll`, `uuid`). Mathematical primitives in
tight computation loops are acceptable exceptions (e.g., `x`, `y` in a lambda).

```python
"Good"
chooser_posterior_mean_estimates = ...
fitted_parameter_values_for_all_dyads = ...
loss_for_current_parameter_guess = ...

"Bad — do not write these"
est = ...
fitted_params = ...
loss = ...
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

The middle line must have an equal number of `=` characters on both sides of the text (with one
space padding on each side). If the available `=` count is odd, the extra `=` goes **after** the
text (right side). Example: a 90-character line with "My Title" (8 chars) has 80 `=` characters
available (after 2 spaces), split 40 left and 40 right — even, so equal. A title with 9 chars
leaves 79, split 39 left and 40 right (extra `=` on right).

### Always use keyword arguments in function calls

All non-trivial function calls must use keyword arguments so a reader can tell at a glance what
each value is. Positional-only calls are acceptable for standard builtins (`len`, `range`, `print`)
but not for project functions.

```python
"Good"
gnrl.classify_pair_relation(
    model_1=child_utility_settings, model_2=parent_utility_settings,
    utility_settings=utility_settings, general_settings=general_settings,
)

"Bad — reader cannot tell which argument is which"
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

## 8. Main sections of `main.py`

The file is organized into logical sections in this order:

| Section | Line range | Content |
|---------|-----------|---------|
| Utility and Choice Functions | 6–749 | `utility_term`, `utility`, `softmax_`, `choice`, `build_utility_equation` |
| Shared Functions | 750–1543 | Optimization helpers: `global_local_optimization`, warm-starting, IC |
| MLE Code | 1544–2003 | MLE loss, fitting, `run_analysis_mle` |
| Bayesian Code | 2004–4789 | `bayesian_update_grid`, `agent`, `loss_function_bayes`, `run_analysis_bayes` |
| Simulation 1–3 | 4790–8218 | Parameter recovery, convergence, update speed |
| Illustrating Belief Updates | 8219–9419 | 2D/3D update visualizations, accuracy analysis |
| Model Validation | 9420–10536 | Typological comparison, model comparison |
| IC Analysis | 10537–12711 | 480-model utility comparison; nesting-aware fitting |
| Nesting Network & Verification | 12712–15000 | Sanity checks, nesting tests, embedding verification |
| Parameter Distribution Results | 15001–15861 | Population-level distributions, correlations |
| Inequality Aversion Analysis | 15862–16224 | Bot competitions, aversion heatmaps |
| Run Code | 16225–end | `run_code_settings` flags + `main()` |
