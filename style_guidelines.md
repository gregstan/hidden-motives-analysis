# Coding Style Guidelines

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
