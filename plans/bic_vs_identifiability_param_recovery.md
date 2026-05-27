# BIC vs. Practical Parameter Identifiability: Findings, Implications, and Proposed Remedies

**Written:** 2026-05-26  
**Context:** Hidden-motives-analysis codebase (`run_param_recovery_by_k` pipeline)  
**Status:** Findings in hand; relaxed-optimizer replication pending; paper implications TBD

---

## 1. What We Just Did

We ran `run_param_recovery_by_k` — a three-phase simulation pipeline that:

1. **Generates synthetic data** using the full `k`-parameter utility model for a population of players
2. **Fits the model** to each synthetic player's data using the standard Bayesian UBM optimizer
3. **Computes Pearson 𝑟** between true parameter values (known because we generated the data) and fitted estimates — per-parameter and as an aggregate across all parameters

We ran this for **k = 2 through k = 7**, holding constant:
- n = 73–76 synthetic players per k (padded to a multiple of 4)
- n = 120 games per player
- Chooser role only (predictor role fitting is harder and was deferred)

The results are saved in `simulations/param_recovery_by_k/param_recovery_by_k.csv` and the interactive figure is `param_recovery_by_k.html`.

---

## 2. What We Found

### 2a. Aggregate correlation drops with k

| k | 𝑟 aggregate (chooser) |
|---|----------------------|
| 2 | 0.916 |
| 3 | 0.872 |
| 4 | 0.728 |
| 5 | 0.718 |
| 6 | 0.594 |
| 7 | 0.580 |

The drop is substantial. At k=7, the average correlation across all parameters is only ~0.58, which means the optimizer is recovering less than half the explained variance in parameter space. Whether this reflects genuine non-identifiability or insufficient optimizer budget is the open question (see §4).

### 2b. Within-k, correlations drop by parameter position

At k=7, the per-parameter correlations were approximately:

| Parameter | 𝑟 |
|-----------|-----|
| Vᵢᵢ (self-interest) | 0.809 |
| Vᵢⱼ (altruism) | 0.799 |
| α (inequality aversion, self) | 0.685 |
| β (inequality aversion, other) | 0.515 |
| γ1 | 0.449 |
| γ2 | 0.524 |
| γ3 | 0.275 |

Key observations:
- Vᵢᵢ and Vᵢⱼ recover well across all k values.
- The higher-order parameters (γ1, γ2, γ3) recover poorly.
- **The pattern is not perfectly monotone** — γ2 > γ1, which is diagnostically important (see §3).

### 2c. Altruism (Vᵢⱼ) recovery is consistently high

Vᵢⱼ recovers at 𝑟 ≈ 0.917 at k=2 and remains the best-recovered parameter at each k. This is reassuring for the central claim of the paper (that players infer altruism/social preference). Even at k=7, Vᵢⱼ recovers at 𝑟 = 0.800.

---

## 3. Why Does This Pattern Occur?

### The primary explanation: differential parameter identifiability

Each parameter in the utility function has a different *curvature* in the likelihood surface — how sharply the log-likelihood degrades as you move away from the true value. High curvature → the optimizer reliably finds the true value. Flat curvature → many parameter values explain the data nearly equally well, and the optimizer's solution is noisy.

The structural reason curvature varies across parameters:

- **Vᵢᵢ and Vᵢⱼ** appear in every game, in the main utility term, and drive the primary variance in choice behavior. They carry the most signal per game and dominate the gradient.
- **α and β** (inequality aversion) modulate behavior primarily when payoff *differences* are large. That's a subset of games.
- **γ1, γ2, γ3** are higher-order social comparison terms that only substantially influence choice probabilities in specific game configurations. Their contribution to the gradient is small and context-dependent.

This is a well-studied phenomenon in cognitive and psychophysical modeling. See:
- Wilson & Collins (2019, *eLife*): parameter recovery in reinforcement learning models; they recommend flagging any parameter with 𝑟 < 0.6 as practically non-identifiable at typical sample sizes.
- Gutenkunst et al. (2007, *PLoS Computational Biology*): "sloppy models" — systems where a few parameter combinations (stiff directions) determine most of the behavior, and many orthogonal directions (sloppy directions) have negligible effect. This is almost certainly what we're seeing.
- Scheibehenne et al.: parameter recovery concerns for social preference models specifically.

### Why the non-monotonicity (γ2 > γ1) matters

If the drop were purely an artifact of parameter *order* in the optimization vector, we'd expect a perfectly monotone decline. The fact that γ2 recovers better than γ1 means the pattern reflects genuine differences in how these parameters influence behavior — γ2 must produce more discriminable choice patterns in this payoff structure than γ1 does. This is evidence for the identifiability hypothesis over a pure optimizer-order artifact.

**Caveat:** The non-monotonicity is also consistent with the optimizer searching γ2's region more efficiently (convergence hypothesis). The two hypotheses are not yet distinguished — see §4.

---

## 4. The Open Question: Identifiability vs. Convergence

There are two distinct explanations for the low γ1/γ3 correlations that have very different implications:

### Hypothesis A: Identifiability (the likelihood surface is genuinely flat)

γ1 is a sloppy parameter. The data simply do not contain enough information to pin it down at n=120 games. Running the optimizer for 10× longer would not substantially improve 𝑟 because the likelihood function itself is nearly flat along the γ1 axis. More iterations converge to nearly the same poorly-determined estimate.

**Implication:** γ1 is noise-fitting even within the k=7 model. Individual-level γ1 estimates are unreliable. The model is technically 7-dimensional but effectively lower-dimensional.

### Hypothesis B: Convergence (the optimizer terminates too early)

γ1 has real curvature in the likelihood, but it lies in a direction the optimizer reaches late or explores inefficiently. The standard optimizer settings (maxfun, maxtol) terminate before finding the true minimum. With relaxed tolerances, recovery would improve substantially.

**Implication:** The model is correctly specified and γ1 is recoverable, but the computational budget needs to increase. This is a methodological note rather than a model critique.

### How to distinguish them

**Step 1 (in progress):** Rerun `run_param_recovery_by_k` with relaxed `maxfun`/`maxtol`. If γ1 𝑟 improves substantially (e.g., from 0.449 to 0.7+), hypothesis B is supported.

**Step 2:** After relaxed rerun, examine whether fitted γ1 distributions show *systematic bias* (values cluster at boundaries → convergence issue) or *high variance around truth* (𝑟 is low but unbiased → identifiability issue).

**Step 3:** Profile likelihood analysis — for a sample of synthetic players, compute the log-likelihood as a function of γ1 alone (holding all other parameters at their fitted values). A flat profile → identifiability; a sharp trough that the optimizer missed → convergence.

---

## 5. The BIC Problem

BIC (Bayesian Information Criterion) is the primary model selection tool in this analysis. It penalizes complexity by adding **k · ln(n)** to the negative log-likelihood, where n = number of observations. The intent is to reward parsimony and prevent overfitting.

**Why BIC doesn't fully solve the overfitting problem here:**

### 5a. BIC penalizes by parameter count, not by effective identifiability

BIC treats all k parameters as equally costly and equally informative. It adds ln(120) ≈ 4.79 for each parameter, regardless of whether that parameter is Vᵢᵢ (high leverage, well-identified) or γ1 (low leverage, sloppy). If γ1 improves the training log-likelihood by more than 4.79 units purely by fitting noise, BIC will still prefer k=7 over k=6 — even when γ1 is doing nothing useful for generalization.

### 5b. BIC is asymptotically consistent, not finite-sample optimal

BIC is guaranteed to select the true model as n → ∞. At n = 120, this guarantee doesn't apply. The log-likelihood improvement from adding a sloppy parameter can exceed the BIC penalty by chance, especially across many datasets.

### 5c. Independent-observation assumption may be violated

BIC's n is the number of independent observations. If a player's choices are autocorrelated within a session (carry-over effects, learning, stable strategy), the effective n is less than 120. BIC with inflated n underpenalizes.

### 5d. What BIC does catch

BIC correctly identifies which model *architecture* is best supported across the population — it will reliably prefer k=4 over k=7 if the former generalizes better across players. It's a good tool for comparing aggregate model support. It just can't identify which specific parameters within the winning model are individually reliable.

---

## 6. LOO-CV: A More Honest Alternative

**Leave-one-out cross-validation (LOO-CV)** is the most principled alternative to BIC for this problem. Instead of penalizing by a formula, it directly measures out-of-sample predictive accuracy.

### How LOO-CV works for this model

For each player and each held-out game:
1. Fit the model on all other games for that player (n−1 games)
2. Compute the log probability of the held-out choice given the fitted parameters
3. Sum log probabilities across all held-out games → LOO log-likelihood for that player

Sum across all players → population LOO score. Lower (less negative) is better.

### Why LOO-CV is more honest about overfitting

If γ1 is noise-fitting, the model trained on n−1 games will have a slightly different (and equally noisy) γ1 estimate. The held-out game's choice probability will be no better than the k=6 model's prediction, and might be worse due to worse estimates of the other parameters caused by γ1 competing for variance. LOO-CV will penalize k=7 for this directly, without relying on asymptotic theory.

### Could LOO-CV become a column in the IC results DataFrame?

Yes. The IC analysis already produces a DataFrame indexed by model (utility function specification) with columns including BIC, AIC, log-likelihood, k, n. Adding a LOO column is conceptually straightforward.

**Implementation approach:**
```
For each utility model specification:
    For each player in the dataset:
        For each game g (or a stratified subset):
            Fit model to all games except g
            Record log p(choice_g | fitted params)
    LOO score = sum of all held-out log probabilities
```

### Computational cost

This is the main obstacle. For the current analysis:
- ~84 players per experiment
- 120 games per player
- Each fit takes ~seconds at standard settings, possibly minutes with relaxed optimizer

Naively, LOO-CV requires **84 × 120 = 10,080 fits** per model, versus 84 fits for the full-data analysis. With ~100+ models in the IC comparison, this is computationally prohibitive.

**Practical approximations:**

1. **LOO-CV on a representative subset of models** — only compute LOO for the top-ranked BIC models (e.g., top 10–20). This tests whether BIC and LOO agree on the winner, without rerunning everything.

2. **k-fold CV instead of LOO** — use 10-fold CV (12 held-out games per fold, 10 fits per player instead of 120). Reduces cost by 12×. Less precise but still honest.

3. **Pareto-smoothed LOO (PSIS-LOO)** from the Stan/ArviZ literature — approximates LOO from the full-data posterior without refitting, using importance sampling. Only applicable to Bayesian posteriors (which this model uses), not MLE. This could be integrated into the Bayesian fitting framework with relatively modest effort. Computationally similar cost to a single full fit, not 120 fits.

4. **LOO on a held-out player subset** — fit on 70 players, evaluate on 14. Gives a between-player generalization estimate rather than within-player, but much cheaper.

**Recommended approach for paper:** Use PSIS-LOO if the Bayesian posteriors are available (they are — the grid stores the posterior distribution). Compute it for the top BIC models only. Report as a robustness check rather than the primary criterion. This is computationally feasible and methodologically defensible.

---

## 7. Implications for the Paper

### If Hypothesis B is confirmed (convergence was the problem)

- The parameter recovery results with relaxed optimizer settings become the primary ones to report
- The current results (tight tolerances) become evidence that standard optimizer settings are insufficient for k ≥ 6 — a methodological caveat, not a model critique
- BIC-based model selection stands as-is
- No major rewriting required

### If Hypothesis A is confirmed (identifiability is the problem)

This is the more consequential scenario. Options, roughly in order of increasing disruption:

1. **Report the winning k model but flag low-identifiability parameters** — use BIC to select the model but explicitly state that individual-level estimates of γ1, γ3 are unreliable and should not be interpreted. Restrict individual-level analysis (histograms, correlations, clinical interpretation) to the identifiable parameters only.

2. **Redefine the model comparison to exclude k > threshold** — if γ1 is not identifiable at n=120, then k=7 is not a legitimate competitor at this sample size. Justify excluding it from the BIC comparison on identifiability grounds, not model selection grounds.

3. **Propose a lower effective-k model as the main result** — if k=4 or k=5 gives the best combination of BIC *and* identifiability (all parameters 𝑟 > 0.7), argue this is the right model for this data regime.

4. **Advocate for larger n** — if the study were run with 240+ games per player, γ1 might become identifiable. Frame the current n=120 as a limitation and the identifiability analysis as a diagnostic for future studies.

The third option requires the most rewriting but is the most scientifically defensible position if the data genuinely cannot support k=7. Given the current BIC curve (BIC still decreasing through k=7), the most honest framing may be: "The utility model favored by BIC is k=7, but parameter recovery simulations reveal that individual-level estimates are reliable only for k ≤ [threshold]. We therefore report individual-level findings using the k=[threshold] model and treat the BIC result as evidence about aggregate structural preferences."

---

## 8. Immediate Next Steps

1. **[In progress]** Run `run_param_recovery_by_k` with relaxed `maxfun`/`maxtol` to test Hypothesis B. Compare per-parameter 𝑟 values against the current results.

2. If relaxed results show improvement: update the default optimizer settings for the production run and rerun the full IC analysis.

3. If relaxed results show minimal improvement: accept the identifiability interpretation, decide on paper framing (option 1–3 in §7), and investigate PSIS-LOO as a supplementary check.

4. Consider adding LOO-CV (at minimum, k-fold) to the IC analysis pipeline as a column in the IC DataFrame. Even a coarse 5-fold CV would provide out-of-sample validation that reviewers will find compelling.

5. Consult Wilson & Collins (2019) recovery threshold recommendations for a defensible cutoff (they suggest 𝑟 > 0.6 as a minimum; 𝑟 > 0.7 as comfortable).

---

## 9. Summary

We found that parameter recovery 𝑟 declines with model complexity (k), with the sharpest drops in higher-order social comparison parameters (γ1, γ2, γ3). BIC selects for model fit vs. complexity but cannot detect within-model parameter sloppiness. Two explanations remain open: genuine non-identifiability (the data don't contain enough information) or optimizer under-convergence (more iterations would fix it). The relaxed-tolerance replication currently in progress will distinguish between them. If identifiability is the root cause, paper-level implications include restricting individual-level analysis to identifiable parameters and potentially arguing for a lower effective-k model. LOO-CV is a more honest alternative to BIC that directly tests out-of-sample generalization and could be added as a column in the IC DataFrame, with PSIS-LOO being the most computationally feasible implementation for this Bayesian model.
