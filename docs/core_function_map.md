# Core function map — *Inferring Hidden Motives* (Utility‑Bayesian Model)

This document is an optional “code map” that explains where the core modeling ideas live and how the main functions fit together.
The **README** is the canonical quickstart. If you only have 2–3 minutes, start there.

If you’re new to the data structures, see **data_dictionary.md** for a concrete example of the dyad-history format and the key settings dictionaries.



## Quick navigation

| Area | Key functions | Module |
|---|---|---|
| Utility + likelihood | `utility_term`, `utility`, `softmax_`, `choice` | `model.py` |
| Readable equations | `build_utility_equation` | `model.py` |
| Bayesian updating | `prior_grid_from_params`, `bayesian_update_grid` | `bayesian.py` |
| UBM engine | `agent` | `bayesian.py` |
| Objective | `loss_function_bayes`, `create_loss_report` | `bayesian.py` |
| Optimizer | `global_local_optimization` | `optimization.py` |
| Fit loop | `fit_params_by_player`, `run_analysis_bayes` | `bayesian.py` |
| Model comparison | `information_criterion_analysis` | `analysis.py` |
| Architecture curve | `compute_architecture_compression_curve` | `analysis.py` |
| Model recovery | `compute_model_recovery_simulation` | `analysis.py` |
| Nesting graph | `model_nesting_adjacency_matrices` | `analysis.py` |
| Nesting checks | `run_child_parent_probability_equivalence_smoketest` | `analysis.py` |
| String-vs-numeric check | `verify_utility_vs_string_equation` | `utilities.py` |
| Settings / enumeration | `generate_utility_settings`, `make_param_info` | `utilities.py`, `config.py` |

---

## 1) Mental model of the pipeline

At a high level, the project does three things:

1. **Defines a family of utility functions** (480 valid forms) for social preferences.
2. **Implements a Utility‑Bayesian Model (UBM)** where a *predictor* updates beliefs about a *chooser’s* latent social‑preference traits over repeated games.
3. **Fits and compares models** by optimizing parameters to minimize negative log likelihood (plus a nesting‑fair penalty), and then performing a large‑scale information‑criterion comparison.

### 1.1 Core call graph (Bayesian fitting)

```
run_analysis_bayes(...)
  -> fit_params_by_player(...)
     -> global_local_optimization(objective = player_loss)
        -> loss_function_bayes(...)          # evaluates one parameter proposal
           -> agent(...)                     # the UBM “engine” over a dyad’s games
              -> bayesian_update_grid(...)   # grid update or particle filter approx
                 -> prior_grid_from_params(...)
              -> choice(...) / softmax_(...) # likelihood (choice rule)
           -> create_loss_report(...)        # aggregates loss for a dyad
        -> (returns best parameters, saves JSON/CSV reports)
```

### 1.2 Core call graph (model comparison + nesting robustness)

```
information_criterion_analysis(...)
  -> generate_utility_settings(...)        # enumerates all valid utility forms
  -> model_nesting_adjacency_matrices(...) # builds child/parent/sibling graph
  -> for each utility form:
       run_analysis_bayes(...)             # fit the model
  -> iterates multiple “robustness” passes:
       - exploration early (random restarts)
       - exploitation late (warm-start via child->parent parameter transfer)
       - stops when improvements (Δ min loss) taper below a threshold
```

---

## 2) Configuration & model enumeration (config.py)

These functions make the model family explicit and ensure the rest of the code can treat “utility form” as a switchable configuration.

### `parameter_keys_for_utility_settings(utility_settings) -> list[str]`
**Purpose:** Given a particular boolean configuration of `utility_settings` (14 flags), compute the **exact set of free parameters** required by that utility form (e.g., which social‑comparison parameters exist, which exponent parameters exist, whether temperature is a parameter, etc.).

**Why it matters:** This is the bridge between *conceptual model form* (“include altruism term?” “include social comparison?” “single exponent or multiple?”) and the concrete parameter vector passed to optimization and Bayesian updating.

---

### `make_param_info(utility_settings, param_bds, ...) -> dict`
**Purpose:** Build the `param_info` bundle used throughout the project:
- `param_keys` (ordered list of parameter names)
- `param_guesses` (initial values / random-start logic)
- `param_bds` (bounds per parameter)

**Why it matters:** This makes the rest of the pipeline generic: *the same optimizer / likelihood / UBM code can run for any utility form* because `param_info` tells it what parameters exist.

---

## 3) Enumerating the 480 utility functions (utilities.py)

### `generate_utility_settings(utility_settings) -> list[dict]`
**Purpose:** Enumerate the space of candidate utility functions by iterating over all `2^14` boolean combinations of the utility flags and keeping only configurations that are **logically valid** and **non‑redundant**.

**Key idea:** 480 of the `2^14 = 16384` combinations are meaningful models; the rest are logically impossible or collapse to an equivalent model.

**Used by:** `information_criterion_analysis(...)` to define the universe of models.

---

### `is_valid_utility_settings(utility_settings) -> bool`
**Purpose:** Apply the “rules of the model family”:
- rejects conceptual impossibilities (incompatible flag combos)
- rejects redundancies (two different flag settings that yield the same effective model)

**Used by:** `generate_utility_settings(...)`.

---

## 4) Utility function + choice rule (model.py)

These functions implement the likelihood term: given payoffs and parameters, what is the probability of each choice?

### `utility_term(...)`
**Purpose:** Compute one component of utility (e.g., self-payoff term, other-regarding term, social-comparison term) depending on `utility_settings`.

---

### `utility(payoffs, params, utility_settings, ...) -> float`
**Purpose:** The “Swiss‑army‑knife” utility function.
It assembles the full utility value for an option by combining the relevant terms specified by `utility_settings`.

**Used by:** `choice(...)` → `softmax_(...)` → likelihood (NLL).

---

### `softmax_(u_A, u_B, temperature) -> tuple[float, float]`
**Purpose:** Convert utility differences into choice probabilities.
Temperature controls choice noise / stochasticity.

---

### `choice(payoffs_A, payoffs_B, params, utility_settings, ...) -> dict`
**Purpose:** Convenience wrapper that computes utilities for both options and returns choice probabilities (and related metadata used by the predictor/likelihood).

---

### `build_utility_equation(utility_settings) -> str`
**Purpose:** Create a human‑readable **string equation** for the current utility form (for debugging, reporting, and especially the large 476‑model comparison).
This is critical for making the model family interpretable and auditable.

---

## 5) Bayesian updating engine (the UBM) (bayesian.py)

### `prior_grid_from_params(params, param_info, general_settings, ...)`
**Purpose:** Convert param “mean + std” representations into a discrete probability mass function (PMF) over an n‑dimensional grid.

**Used by:** `bayesian_update_grid(...)`.

---

### `bayesian_update_grid(prior_grid, likelihood_grid, general_settings, ...)`
**Purpose:** Perform the Bayes update on a discrete hypothesis space:
- full grid update (exact but slow), **or**
- particle‑filter approximation (fast), depending on `general_settings['use_particle_filter']`.

**Core operation:** posterior ∝ prior × likelihood (with normalization), repeated across rounds.

---

### `agent(dyad_games, chooser_params, predictor_params, general_settings, utility_settings, ...)`
**Purpose:** The **Utility‑Bayesian Model (UBM)**.
Given a sequence of games between two players, it:
1. uses current beliefs (priors) to compute predictions / choice probabilities,
2. observes what happened,
3. updates beliefs (posteriors),
4. feeds posteriors at time *t* into priors at time *t+1* (“self‑perpetuating” across rounds).

**Why it matters:** This is the core cognitive model of social‑preference learning: *how you infer “how nice or nasty someone is” from what they do.*

---

## 6) Loss, penalties, and optimization (utilities.py + bayesian.py + optimization.py)

### `parameter_penalty(params, utility_settings, ...) -> float`  *(utilities.py)*
**Purpose:** Add a regularization term to discourage extreme / degenerate parameterizations **without biasing model comparison**.

**Design constraint (important):** The penalty is constructed to be *nesting‑fair*.
For a child/parent pair, if the parent is set to its child‑equivalent parameterization (child parameters + the parent’s “special” inert values for extra parameters), the **penalty (and thus loss) matches** the child’s penalty.  
This prevents an “anti‑parent” artifact where more expressive models get penalized just for having extra parameters.

---

### `create_loss_report(dyad_games, general_settings) -> dict`  *(main.py)*
**Purpose:** Aggregate per‑round quantities (NLL, penalties, etc.) into a compact loss report **for a single dyad**.
The report is stored on the first game in the dyad so downstream code can access summary metrics without recomputing them.

**Used by:** `loss_function_bayes(...)`.

---

### `loss_function_bayes(dyad_games, chooser_params, predictor_params, general_settings, utility_settings, ...) -> float`
**Purpose:** Compute the objective optimized during fitting:  
**negative log likelihood (from the UBM) + parameter_penalty(...)**.

**Used by:** `global_local_optimization(...)` during parameter fitting.

---

### `global_local_optimization(objective_fn, bounds, general_settings, ...) -> dict`
**Purpose:** Main optimizer used throughout the project:
- **global search** (simulated annealing / dual annealing) to escape local minima
- **local refinement** (L‑BFGS‑B) to converge to a good solution

**Why it matters:** Many subproblems are non‑convex; this “globloc” pattern is robust.

---

### `global_local_then_trust_constr(...)`
**Purpose:** A wrapper around `global_local_optimization` that can optionally enforce constraints (e.g., a norm constraint on parameters) via `trust-constr`.  
Not central to the paper, but kept as an experimentation hook.

---

## 7) Fitting pipeline (player-level) (bayesian.py)

### `fit_params_by_player(player_uuid, ..., general_settings, utility_settings, param_info, file_paths)`
**Purpose:** Fit parameters for one participant **across all dyads they appeared in** (the unit used in the paper).
It:
- loads/collects that player’s dyad histories,
- defines a summed loss across dyads,
- optimizes parameters,
- writes outputs:
  - JSON with best‑fit parameters + belief trajectories (`player_fits/`)
  - CSV loss reports mapping parameter proposals → losses (`loss_reports/`)

**Note:** There is a dyad‑level sibling (fit per dyad) but the paper focuses on player‑level fits.

---

### `run_analysis_bayes(histories_data, general_settings, utility_settings, param_info, file_paths)`
**Purpose:** Orchestrate Bayesian fitting over the dataset:
- iterate over players (or dyads, depending on settings),
- call `fit_params_by_player(...)`,
- manage multiprocessing (when enabled),
- produce the fitted outputs used by downstream analyses and figures.

---

## 8) Model comparison & nesting-violation prevention (utilities.py + analysis.py)

### `classify_pair_relation(child_settings, parent_settings) -> tuple[str|None, str|None]` *(utilities.py)*
**Purpose:** Given two utility-setting configurations, classify their relationship:
- (`'child'`, `'parent'`) if they differ by one toggle and the parent has strictly more parameters,
- (`'sibling'`, `'sibling'`) if they differ by one toggle but have the same parameter count,
- (`None`, None) otherwise.

**Used by:** `model_nesting_adjacency_matrices(...)` and the nesting‑robustness machinery.

---

### `parents_children_of(model_settings, utility_settings_universe) -> dict` *(utilities.py)*
**Purpose:** Enumerate the immediate parents and children (and sometimes siblings) of a model within the model family.

---

### `map_child_to_parent_special_param_info(child_settings, parent_settings, child_params) -> dict` *(utilities.py)*
**Purpose:** Convert a child model’s fitted parameters into a **parent-compatible** parameter dict by adding the parent’s “special” parameters (inert values that make the parent behave exactly like the child).

**Why it matters:** This enables:
- warm-starting parents from children during IC robustness iterations, and
- preventing **nesting violations** (a parent must be able to fit at least as well as its child).

---

### `model_nesting_adjacency_matrices(general_settings, utility_settings, file_paths, ...)` *(main.py)*
**Purpose:** Build and save the graph structure of the model family:
- parent/child edges
- sibling edges  
This is the backbone used to detect and prevent nesting violations.

---

### `best_fitting_model_parameters(...)` *(utilities.py)*
**Purpose:** Helper for retrieving the best-known parameters for a given model (from previous iterations / saved loss reports).

---

### `best_fitting_child_parameters_for_parent(...)` *(utilities.py)*
**Purpose:** Given a parent model, locate its best-performing child(ren) and return child parameters suitable for warm-starting the parent (via `map_child_to_parent_special_param_info`).

---

### `select_child_params_for_parent(...)` *(main.py)*
**Purpose:** Implements the exploration/exploitation policy for child→parent warm-starting during robustness iterations:
- early iterations: exploratory (diverse child seeds)
- later iterations: exploit best child solutions

---

### `information_criterion_analysis(general_settings, utility_settings, file_paths, param_bds, ...)`
**Purpose:** The “colossus” model-comparison routine.
It:
- iterates over all valid utility models,
- fits each model multiple times with random starts,
- tracks each model’s best NLL and information criteria,
- runs a robustness loop until improvements saturate,
- uses the nesting graph + child→parent parameter propagation to eliminate nesting violations.

**Why it matters:** This is what makes the 480‑model comparison feasible and trustworthy.

---

## 9) Validation & sanity checks (analysis.py + utilities.py)

These are the “trust but verify” tools used to ensure the model family behaves correctly.

### `verify_same_inputs_same_outputs_for_children_and_parents(...)`
**Purpose:** Confirm that every child model is a true special case of its parent by checking that the parent model — with its extra parameters set to neutral values — produces the same NLL as the child on real participant data. `equal_loss=False` in the output CSV identifies broken embedding rules; `changed_utility_setting` names the flag responsible.

---

### `run_child_parent_probability_equivalence_smoketest(...)`
**Purpose:** Fast check that for randomly generated payoff pairs and a random child model:
- child choice probabilities match parent choice probabilities
- when the parent is set to the child-equivalent “special parameter” configuration.

---

### `verify_utility_vs_string_equation(utility_function, utility_function_str, ...)`
**Purpose:** Ensure that `build_utility_equation(...)` is not merely cosmetic: its string form matches the numeric utility implementation (within tolerance) over large sets of random games.

---

## 10) “Where should I look first?”

If you only inspect ~200 lines, I’d prioritize:

1. **UBM core:** `agent(...)`, `bayesian_update_grid(...)`, `prior_grid_from_params(...)`
2. **Likelihood:** `utility(...)`, `choice(...)`, `softmax_(...)`
3. **Optimization:** `global_local_optimization(...)`
4. **Nesting robustness:** `map_child_to_parent_special_param_info(...)` + `run_child_parent_probability_equivalence_smoketest(...)`

That combination captures the main modeling assumptions and the engineering that makes the large-scale comparison reliable.
