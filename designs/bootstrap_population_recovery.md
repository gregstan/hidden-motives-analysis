# Plan: Population-Level Parameter Recovery Bootstrap

**Audience:** A Claude Code instance working in the `hidden-motives-analysis` repo with full read/write access.

**Prerequisite read:** `AGENTS.md` (project conventions) and `style_guidelines.md` (coding style). All code written for this analysis must conform to both. The variable-naming rule, the keyword-arguments rule, the string-comments rule, the section-header style, and the HSLA color rule are non-negotiable.

---

## 1. Why this analysis exists

The IC analysis on Experiment 3 selects a seven-parameter utility function as the BIC winner. The existing parameter recovery analysis (`simulation.run_param_recovery_by_k`, output in `simulations/param_recovery_by_k/`) shows that within this winning model the social preference weights recover well but the curvature exponents γ1, γ2, γ3 recover only weakly (correlations 0.39, 0.67, 0.37).

One of the paper's headline findings is that γ2 and γ3 are super-linear at the population level (means ≈ 1.34 and 1.38), contradicting the canonical assumption of diminishing marginal utility. A reviewer at Cognition seeing recovery correlations of 0.39 and 0.37 on the parameters carrying that claim will ask whether the population-level means can be trusted.

Low individual-level correlation is consistent with two underlying realities:

1. **Noise around the right mean.** Individual γ_hat is noisy, but the population mean of γ_hat tracks the population mean of γ_true. Population-level claims survive. Individual interpretation should be cautious.

2. **Systematic bias.** The recovery procedure pulls γ_hat toward some attractor (the L2 penalty, which scores `(mean(γ) - 1)^2`, is the most likely culprit; trade-offs between exponents and weights are another). Population mean of γ_hat is biased relative to population mean of γ_true. The super-linear claim is suspect.

Correlation alone cannot distinguish these. This analysis distinguishes them by computing bias, variance ratio, and regression slope alongside correlation.

The analysis is also useful beyond the immediate γ question. Any utility function, any parameter, any player role can be subjected to the same test. Build it general from the start so it can be reused by future analyses.

## 2. Method overview

Parametric bootstrap from the empirical fitted distribution:

1. For the target utility function and player role, pull the fitted parameter vectors from the human-data IC JSON. Treat each as a "true" parameter vector for one synthetic agent.
2. Generate synthetic experimental data in which each synthetic agent makes choices governed by those parameters, using the same payoff sampling regime as the real experiment.
3. Refit the same utility function to the synthetic data using the standard production fitter.
4. Compare the recovered population to the empirical population, parameter by parameter.

This differs from a classical bootstrap (resampling existing observations with replacement). The randomness comes from synthetic choice noise and payoff sampling, not from resampling participants. Use the term *parametric bootstrap from the empirical distribution* in the writeup to avoid confusion.

## 3. Function signature and design requirements

Build one public function. Place it in `simulation.py` next to `run_param_recovery_by_k` so callers find it where they expect. Suggested name: `run_population_recovery_bootstrap`. The signature must match the style and parameter conventions of the surrounding functions, and every argument must be documented in the bulleted `Arguments:` block per the style guide.

Required arguments and defaults:

- `general_settings: GeneralSettings`, `file_paths: FilePaths`, `param_bds: ParamBounds`, `figure_layout: FigLay`. Standard project arguments.
- `utility_settings: UtilitySettings | None = None`. The utility function to test. When `None`, resolve to the BIC-winning utility settings for `experiment_num = general_settings['experiment_num']` from the IC analysis CSV. **Do not hard-code the k=7 winner as a literal dict.** Resolve it dynamically. See Section 6 below.
- `player_role: Literal['chooser', 'predictor'] = 'chooser'`. Which role's parameters are bootstrapped from. Default chooser.
- `dynamic_predictor: bool = False`. Forwarded to the simulation and fitting pipeline. False by default to match the static fits that produced the population statistics in Section 8 of the paper.
- `parameters_of_interest: list[str] | None = None`. Subset of parameter keys to report metrics for. When `None`, report metrics for every parameter in the utility function (this is the default and the right one for the γ defense; the argument exists so future callers can focus on a subset).
- `n_bootstrap_iterations: int = 1`. Set higher when you want standard errors on the bias. Default 1 because each iteration is expensive.
- `random_seed: int | None = None`. Threaded through to the simulation generator. 
- `enforce_memory_limit: bool = False`. Forwarded to `create_simulated_experiment`.
- `create_new_file: bool | None = None`. Standard cache pattern. Resolve from `general_settings` when `None`.
- `base_hue: int | None = None`. For figure colors per the HSLA rule.

Returns: a tuple `(metrics_dataframe, detailed_results_dict)` matching the return shape of `run_param_recovery_by_k`. The DataFrame has one row per (iteration, parameter) when `n_bootstrap_iterations > 1`, otherwise one row per parameter. The dict holds the per-iteration synthetic-vs-recovered parameter tables for downstream inspection.

## 4. File hygiene (critical)

The largest practical risk is that bootstrap synthetic data files collide with the existing synthetic data files produced by `create_simulated_experiment` for `run_param_recovery_by_k`. Both functions use the same caching pattern, both write to `file_paths['processed']`, and both default their output filename to `file_paths['file_names']['player_pairs_exper3']`. Sharing that filename would silently overwrite the existing recovery-by-k data and contaminate it with bootstrap-specific runs.

The convention to follow is the same one already used elsewhere in the repo: build an edited copy of `file_paths` with a bootstrap-specific filename, and pass that edited copy to the existing pipeline. The existing pipeline does not need changes; only the input file_paths does.

Example pattern:

```python
"""
Build a bootstrap-scoped copy of file_paths that redirects the synthetic-experiment
output (and its derived players_to_dyads cache) to a uniquely named subdirectory and
filename so the bootstrap data never collides with any other synthetic dataset.
"""
file_paths_for_bootstrap = copy.deepcopy(file_paths)
bootstrap_subdir = os.path.join(
    file_paths['processed'], 'population_param_bootstrap',
)
os.makedirs(bootstrap_subdir, exist_ok=True)
file_paths_for_bootstrap['processed'] = bootstrap_subdir

bootstrap_run_label = _bootstrap_run_label(
    utility_settings=resolved_utility_settings,
    player_role=player_role,
    iteration_index=current_iteration_index,
)
file_paths_for_bootstrap['file_names'] = dict(file_paths['file_names'])
file_paths_for_bootstrap['file_names']['player_pairs_exper3'] = (
    f"synthetic_histories_{bootstrap_run_label}.json"
)
file_paths_for_bootstrap['file_names']['players_to_dyads_exper3'] = (
    f"players_to_dyads_{bootstrap_run_label}.json"
)
```

The bootstrap label should encode the utility function (via `convert_utility_settings(..., into=str)`), the player role, the iteration index, and the random seed if one is provided. That way every bootstrap run produces a uniquely named cache file and reruns can resume from cache without ambiguity.

Add a `.gitignore` rule under the bootstrap subdirectory so none of the synthetic-experiment JSON files, the per-player fit JSONs, or the temporary players_to_dyads files end up tracked. The metrics CSV and the figure HTML are the only files worth committing.

```
# .gitignore additions
processed/population_param_bootstrap/synthetic_histories_*.json
processed/population_param_bootstrap/players_to_dyads_*.json
processed/population_param_bootstrap/per_player_fits/
```

Keep the metrics CSV and the figure HTML in `processed/population_param_bootstrap/` and let them be tracked. They are small and worth versioning.

## 5. Reuse what already exists

This is the most important section. Do not reimplement anything below.

- **`create_simulated_experiment(...)`** in `simulation.py` (around line 1739). Produces synthetic experimental data in the same JSON format as the raw human data. The bootstrap uses this as its data generator. Pass in the edited `file_paths_for_bootstrap` from Section 4 and the empirical chooser parameter vectors (see Section 7) for each synthetic player. The existing function samples parameters uniformly from bounds and then overrides the altruism dimension to span a target range. The bootstrap needs a different sampling regime: use the empirical fitted vectors directly, not a uniform sample. The cleanest implementation path is to modify `create_simulated_experiment` to accept an optional `empirical_chooser_parameters: dict[str, dict[str, float]] | None = None` argument that, when provided, replaces the per-player parameter sampling block. When `None`, the existing uniform-sampling-plus-altruism-override behavior is preserved. Document the new argument in the docstring. Do not introduce a parallel data-generation function. See AGENTS.md Section 2 ("Domain vocabulary") and Section 8 ("Module contents") for the existing function map.
- **`run_analysis_bayes(...)`** in `bayesian.py`. The standard fitter. The bootstrap calls this against the synthetic JSON to recover per-player parameters. Same pipeline used by `run_param_recovery_by_k`. Do not bypass it.
- **`convert_utility_settings(...)`** in `utilities.py` (around line 1041). Converts between dict, tuple, string, and int representations of utility settings. Use this for everything settings-related: matching, hashing into filenames, comparing models. Never write a tuple-of-bools or bitstring by hand.
- **`make_param_info(...)`** in `model.py`. Builds the parameter spec for a given utility_settings. Reused by `create_simulated_experiment` and the fitter; the bootstrap should use the same path.
- **`is_valid_utility_settings(...)`** in `utilities.py`. Use this to sanity-check the resolved utility_settings before the simulation starts. Catch malformed inputs early.
- **`extract_participant_model_combined_fits(...)`** in `analysis.py` (Stage 6). The reference example for how to read the IC JSON, how to iterate over models, and how to handle per-player fit data. Read it before writing the empirical-parameter loader in Section 7.

If you find yourself writing a function with more than five lines of logic that has the same shape as something in `simulation.py` or `analysis.py`, stop and grep for it. Reuse beats new code.

## 6. Indexing utility functions correctly

The utility universe was expanded from 476 to 505 models when two settings were added: `include_welfare_efficiency_term` and `include_relative_income_penalty`. The canonical settings dict in `config.utility_settings` is now 16 keys long. Older fits in the IC JSON were produced before the expansion and their tuple-keys are 14-element. The settings dict embedded inside each model entry (`model_entry["utility_settings"]`) may also be 14-key in legacy entries; both new settings are always `False` in those entries.

Rules the bootstrap must follow:

1. **Index utility functions by their settings dict, not by integer or by tuple-string.** Integer indices have shifted because the model count changed; do not trust them across the 476→505 expansion.
2. **When loading the IC JSON, prefer `model_entry["utility_settings"]` over the tuple key.** The dict is self-describing and survives the expansion.
3. **Backfill missing keys to the full 16-key canonical form.** Before any comparison or filename hashing, normalize every settings dict to include all 16 keys, defaulting `include_welfare_efficiency_term=False` and `include_relative_income_penalty=False` for legacy entries.
4. **Use `convert_utility_settings(into=tuple, template=config.utility_settings)`** for any conversion that needs the canonical key order. Use `convert_utility_settings(into=str)` to produce filename-safe bitstrings.
5. **Resolve the BIC winner dynamically** when `utility_settings=None`. Read `bic_aic/All_Utility_Forms_IC_Analysis_Experiment{experiment_num}.csv`, find the row with minimum BIC, read its settings columns, normalize to 16-key form. Print the resolved settings dict and the resolved equation string so the run log shows which model was actually used.

This is the highest-risk section for silent bugs. A run that quietly tested the wrong utility function would invalidate the whole analysis. Print the resolved settings, the resolved equation, and the parameter list at the top of every run.

## 7. Sampling the empirical chooser parameters

The "true" parameter vectors for the bootstrap come from the IC JSON at `bic_aic/All_Utility_Forms_IC_Analysis_Experiment{experiment_num}.json`.

The loader needs to:

1. Open the IC JSON (be aware of the temporary fallback path in `architecture.py` lines ~61-77 in case the local file is incomplete; do not duplicate that block, but consider extracting a small helper if the same fallback is needed here, and check whether one already exists).
2. Iterate over `ic_data["ic_results"]` looking for the model whose `utility_settings` dict, once backfilled to 16 keys, equals the resolved target.
3. From `model_entry["minvec"]`, extract per-player fitted parameters for the requested role. Inspect the JSON structure in advance: the per-player entry contains a `loss` field and likely a `params` or equivalent field with the actual fitted parameter values. The reference reader in `extract_participant_model_combined_fits` only consumes `loss`, so the bootstrap loader is doing something new and the exact field name needs to be confirmed from the JSON itself.
4. Drop any participant with missing or NaN values for the target role.
5. Return a dict mapping `player_uuid → {parameter_key: value}` of length N (typically 73 for Experiment 3 chooser fits).

These empirical vectors become the input to the modified `create_simulated_experiment` per Section 5.

## 8. What to compute and report

For each parameter in `parameters_of_interest` (or for all utility parameters when `None`), compute these metrics by comparing the empirical vectors (treated as "true") to the recovered vectors:

- `correlation_pearson` between empirical and recovered. Sanity-check against the existing `param_recovery_by_k.csv` for the target k.
- `bias = mean(recovered) - mean(empirical)`.
- `bias_normalized = bias / std(empirical)`.
- `variance_ratio = std(recovered) / std(empirical)`.
- `regression_slope`, `regression_intercept`, `regression_r2` from OLS of `recovered ~ a + b * empirical`.

Outputs go to `file_paths['processed']/population_param_bootstrap/`:

- `population_recovery_metrics_{bootstrap_label}.csv`. One row per (iteration, parameter) when `n_bootstrap_iterations > 1`, else one row per parameter. Columns: `iteration_index, parameter_key, n_synthetic_agents, mean_empirical, std_empirical, mean_recovered, std_recovered, correlation_pearson, bias, bias_normalized, variance_ratio, regression_slope, regression_intercept, regression_r2`. Save with `encoding='utf-8-sig'` as required by the style guide.
- `population_recovery_figure_{bootstrap_label}.html`. Two panels rendered via the existing `visualization.py` infrastructure and `_hsla` color helper. Left panel: scatter of empirical vs recovered for each parameter (one color per parameter via `base_hue + 20 * series_index`), identity line overlaid. Right panel: bar chart of `bias_normalized` for each parameter with the zero line bolded; if `n_bootstrap_iterations > 1`, add SE error bars computed across iterations.
- `population_recovery_summary_{bootstrap_label}.md`. Short text summary: which utility function was tested, which role, which N, the resolved BIC, and one sentence per parameter on whether the population mean recovers within the thresholds in Section 9.

The `_label` suffix on filenames encodes the same components used in the cache path so that multiple bootstrap configurations coexist without collision.

## 9. Decision logic for the paper

Use these thresholds when writing the bootstrap result into Section 7.4 of the manuscript. They are starting points; calibrate against what feels defensible for the specific reviewer audience.

**Population means recover.** For each γ parameter, all of:

- `abs(bias_normalized) < 0.2`
- `regression_slope > 0.7`
- When `n_bootstrap_iterations > 1`, the 0 line is within `bias ± 1·SE`.

Conclusion: report the bootstrap as supporting the population-level super-linear claim. One sentence in 7.4: "A parametric bootstrap from the empirical fitted distribution confirms that population-level means of γ1, γ2, and γ3 recover without substantial bias under the same fitting pipeline applied to real human data, even though individual-level recovery correlations are modest. The directional finding that the social exponents γ2 and γ3 exceed 1 on average is therefore supported despite the individual-level noise."

**Population means are biased.** Either `abs(bias_normalized) > 0.3` or `regression_slope < 0.5` for at least one γ parameter.

Conclusion: name the bias, identify the likely source (L2 penalty toward 1, trade-off with other parameters), and report bias-corrected population means alongside the raw estimates. Soften the directional claim: γ2 and γ3 exceed 1 on average, but the point estimates should be read as estimates whose magnitude is partly suppressed by identifiability constraints in the fitting procedure.

**Severe bias.** Either `abs(bias_normalized) > 0.5` or `regression_slope < 0.3` for any γ parameter.

Conclusion: the directional claim itself is in trouble. Discussion needs to engage with the possibility that apparent super-linearity is partly an artifact of the optimization landscape. Do not delete the finding; characterize it honestly and adjust the strength of the language.

## 10. Implementation notes and gotchas

- **Memory.** `create_simulated_experiment` already implements `enforce_memory_limit`. The bootstrap should pass this through. Seven parameters with the default grid bins is well within memory; ten iterations are not memory-limited.
- **Time.** A single fit of one utility function on 73 synthetic agents is roughly equivalent in cost to one row of the existing IC analysis. The full IC analysis fits 505 models; this is 1/505th of that, give or take. Without `dynamic_updating=True`, expect single-digit hours on Greg's machine for one iteration. On Rick's 28-core, under an hour. Multiply by `n_bootstrap_iterations`.
- **The L2 penalty.** The exponent penalty `(mean(γ) - 1)^2` is the most likely source of any bias the bootstrap finds. Worth logging the penalty's contribution to total loss at the end of each run so the diagnostic is in the run log. If the result is biased and you want a follow-up, the natural next experiment is to rerun with the exponent penalty disabled (`λ=0` on just the exponents) and see whether the bias goes away. Do not do this in the first pass. Flag as a follow-up in the summary markdown.
- **`dynamic_predictor: bool = False`.** Default matches the static fit used for the paper's Section 8 statistics. When True, the simulation and fitting both engage the full UBM belief-updating loop for the predictor side. This is the more expensive setting and is not needed for the immediate γ defense. Plumb it through so the function works in both modes.
- **`player_role: Literal['chooser', 'predictor']`.** Default chooser. Predictor-side bootstrap is a separate question (does the predictor's prior get recovered?) and is supported but not the primary use case.
- **Caching.** The bootstrap respects the standard caching pattern. A run with `create_new_file=False` and a populated cache subdirectory loads the cached metrics CSV and returns immediately. A run with `create_new_file=True` regenerates. Encode the iteration index in the cache filename so partial multi-iteration runs can resume.
- **Settings stamp.** Print the resolved utility_settings (full 16-key form) and the resolved equation string at the start of every run. This is non-negotiable. The biggest silent-bug risk in this analysis is testing the wrong utility function.
- **Progress printing.** Follow the AMPD-style progress pattern documented in the style guide: startup banner with total work, periodic per-iteration progress with elapsed and ETA, completion banner with output paths.

## 11. Deliverables checklist

- [ ] New public function `run_population_recovery_bootstrap` in `simulation.py`, conforming to style guidelines and AGENTS.md conventions, taking the arguments listed in Section 3.
- [ ] Optional `empirical_chooser_parameters` argument added to `create_simulated_experiment` so the bootstrap can inject the empirical vectors without duplicating data-generation code.
- [ ] Helper for loading per-player fitted parameters from the IC JSON, with backfill to 16-key canonical form. Lives either as an inner function or as a small module-level helper in `simulation.py` or `utilities.py`, whichever is closer to existing related code.
- [ ] Helper for resolving the BIC-winning utility_settings from the IC analysis CSV when the caller passes `utility_settings=None`. Print the resolved settings.
- [ ] `processed/population_param_bootstrap/` subdirectory, with `.gitignore` rules for the large synthetic JSON files but not for the metrics CSV or figure HTML.
- [ ] Metrics CSV per Section 8, with `encoding='utf-8-sig'`.
- [ ] Figure HTML per Section 8, using `_hsla` colors and the existing visualization conventions.
- [ ] Summary markdown per Section 8.
- [ ] Run log capturing the resolved settings, the resolved equation, N synthetic agents, total time, and the final population-level metrics. Print to stdout in the AMPD-style progress format.

## 12. Final reminder

Read `AGENTS.md` first. Read `style_guidelines.md` second. Then grep before writing. The repo is structured so almost every primitive this analysis needs already exists; the new code should mostly be glue plus the population-level metrics computation. If the resulting file is more than 250 lines long, something has been reimplemented that should have been reused.
