# **Inferring Hidden Motives** — `ihm_starter_pack.md`

> **A compact briefing for coding agents and humans**
>
> **Paper:** *Inferring Hidden Motives: Bayesian Models of Preference Learning in Repeated Dictator Games*  
> **Authors:** Gregory Stanley, Jun Zhang, Rick Lewis

---

## **1) What this paper is trying to do**

This paper asks:

> **How do people infer other people’s hidden social motives from behavior over time?**

In repeated binary dictator games, people watch others choose between payoff allocations and then predict what they will do next. The paper models that process as **Bayesian belief updating over continuous social-preference parameters**.

The central thesis is:

- people do **not** seem to think of others as a few fixed moral “types”
- they seem to track **graded uncertainty** over latent motives such as self-interest, altruism, envy, and guilt
- doing that well requires a richer utility representation than classic low-dimensional social-preference models

The result is a **Utility-Bayesian Model (UBM)** plus a large empirical search over **480 utility functions** to determine which utility representation best supports belief updating.

---

## **2) Why this research is needed**

The paper argues that prior work is limited in three big ways:

1. **Narrow payoff spaces**  
   Many studies use only a few fixed allocations, so they do not reveal the full landscape of social preferences.

2. **Tiny model spaces**  
   Most studies compare only a handful of canonical utility functions.

3. **Belief–preference confounding**  
   If you infer beliefs only from participants’ own choices, you mix together:
   - what they themselves value,
   - what strategy they are using,
   - and what they believe their counterpart values.

### **The four mutually reinforcing methodological elements**
The paper’s real methodological claim is that you need these **together**:

1. **Randomized payoff structures** across a broad multidimensional space  
2. **Near-comprehensive comparison of 480 utility forms**  
3. **Explicit elicitation of predictions** to measure beliefs more cleanly  
4. **Bayesian cognitive modeling** of belief updating across repeated interactions  

These reinforce one another:
- rich payoff variation is needed to discriminate utility forms,
- the utility form is needed for the Bayesian likelihood,
- and explicit predictions are needed to fit a model of **beliefs about others** rather than only personal preferences.

---

## **3) Model first: the Utility-Bayesian Model (UBM)**

### **3.1 The inferential problem**
A chooser’s single action is usually ambiguous. A self-favoring choice could reflect:
- high self-interest,
- weak altruism,
- strong envy,
- weak guilt,
- noisy choice,
- or some combination.

So the observer faces an **inverse problem**:

> infer the hidden preference parameters that most likely generated the observed choices.

### **3.2 Latent parameters**
The UBM represents a chooser using continuous parameters such as:

- `V_ii` — self-interest  
- `V_ij` — altruism / direct valuation of the other’s payoff  
- `Ɛ_ij` — envy / aversion to disadvantageous inequality  
- `Ʒ_ij` — guilt / aversion to advantageous inequality  
- `γ` (or `γ1, γ2, γ3`) — nonlinear payoff sensitivity  
- `τ` — SoftMax temperature / perceived choice stochasticity  

A generic utility skeleton is:

$$
U_i(A \mid \theta)
=
V_{ii} f_{\text{self}}(A)
+
V_{ij} f_{\text{other}}(A)
-
Ɛ_{ij} f_{\text{envy}}(A)
-
Ʒ_{ij} f_{\text{guilt}}(A)
$$

where the $f$-terms change across the **480 candidate functional forms**.

### **3.3 Choice rule**
The chooser’s probability of selecting option $A$ is modeled with a SoftMax:

$$
P(A \mid \theta)
=
\frac{\exp(U(A,\theta)/\tau)}
{\exp(U(A,\theta)/\tau)+\exp(U(B,\theta)/\tau)}
$$

- low $\tau$ -> deterministic-looking choices  
- high $\tau$ -> noisy / random-looking choices

### **3.4 Bayesian update**
Beliefs over the chooser’s latent parameters are updated by Bayes’ rule:

$$
p_t(\theta \mid c_{1:t})
\propto
p(c_t \mid \theta)\,p_{t-1}(\theta)
$$

The model’s psychological sequence is:

```text
predict -> observe -> update
```

This is crucial:

- **chooser actions** are the evidence
- **predictor responses** are what the model is fit to
- belief updating is driven by **observed chooser behavior**, not by the predictor’s own prediction errors

### **3.5 Posterior representation**
The posterior is represented either as:

- a **grid-based** probability mass function over discretized parameter space, or
- a **particle filter** approximation for high-dimensional settings

Because grid size scales as $n^k$, full grids become infeasible as parameter count rises.

The paper validates the particle filter directly: across $k=1$ to $9$ and many sampling ratios, PF posteriors correlated **above 0.993** with full-grid posteriors.

### **3.6 Fitting objective**
The model is fit with **negative log likelihood** against participant predictions:

$$
\text{NLL}(\theta)
=
-\sum_{t=1}^{T}\log P(\hat c_t \mid \theta)
$$

Optimization uses:
1. **simulated annealing** for global search
2. **L-BFGS-B** for local refinement

---

## **4) What the belief-update visualizations mean**

Two figures are especially important for understanding the model.

### **Figure 4: heatmap updating**
Each column shows one observed game:

- **top:** likelihood over parameter space  
- **middle:** the observed payoff structure / choice  
- **bottom:** posterior after the update  

Key idea:
- informative choices produce sharp likelihoods
- posteriors become more concentrated as evidence accumulates
- uninformative cases (e.g., identical options) produce flat likelihoods and no update

### **Figure 7: trajectory view**
This figure shows trajectories of mean beliefs in the $(V_{ii}, V_{ij})$ plane for the four bot avatars.

Key idea:
- high prior variance -> faster, larger updates
- low prior variance -> conservative updating
- zig-zag paths arise because payoff structures vary randomly across trials

For a coding agent, these figures explain what the UBM is doing more clearly than the equations alone.

---

## **5) Methods, after the model**

## **5.1 Experiment 1: human-bot validation**
**Goal:** validate the UBM in a controlled setting

- $n = 83$
- participants predict choices of identifiable bot avatars
- four avatar types: **Utilitarian, Selfish, Competitive, Masochistic**
- 8 controlled payoff-difference scenarios
- repeated interactions with the same avatars
- abstract dot displays, not numeric payoffs
- predictions only, no time pressure

Why this exists:
- bot preferences are fixed and known
- this makes it the cleanest test of whether the UBM captures human belief updating

## **5.2 Experiment 2: human-human estimation**
**Goal:** estimate real human parameter distributions

- $n = 73$
- humans alternate between chooser and predictor roles
- groups of 4–8 players
- payoffs sampled from $\{1,2,3,4,5\}$
- this yields **625** possible payoff configurations
- probabilistic rematching with identifiable partners
- private predictions
- 8-second response limit

Why this exists:
- broader payoff coverage
- more natural repeated interaction
- allows estimation of the population distribution of chooser parameters and predictor priors

---

## **6) Simulation validation: the three big checks**

Before comparing the UBM to rival cognitive models, the paper runs a simulation with **945 dyads** over **25 rounds**.

### **What the simulation tests**
1. **Optimizer recovery**  
   Can the fitting pipeline recover the predictor’s true parameter?

2. **Convergence**  
   Do inferred beliefs move toward the chooser’s true latent parameter over time?

3. **Bayesian dynamics**  
   Do prior variance and temperature affect updating in the right direction?

### **Main results**
- **Recovery:** true vs fitted predictor altruism correlated at **$r = 0.755$**
- **Convergence:** belief accuracy increased almost linearly by round, slope $\approx 0.024$, $R^2 \approx 0.958$
- **Bayesian logic:** higher prior variance sped updating; higher temperature tended to slow it

Interpretation:
the UBM is not just mathematically Bayesian on paper — it behaves like a real learner under noisy evidence.

---

## **7) What the UBM beats**

This is one of the strongest parts of the paper.

## **7.1 Non-Bayesian alternatives**
Compared in the human-bot experiment:

- **Stochastic** — random choice
- **No Learning** — priors effectively frozen
- **No Memory** — uses only the immediately preceding choice

Total NLLs:
- **Utility Bayesian:** ~**5811**
- **No Learning:** ~**10684**
- **No Memory:** ~**12058**
- **Stochastic:** ~**13272**

### **Conclusion**
People are not behaving like:
- random guessers,
- fixed-type perceivers,
- or one-step heuristics.

They integrate **prior beliefs + cumulative evidence**.

## **7.2 Discrete Bayesian / typological alternatives**
Also tested:

- **Good vs Evil**
- **Canonical SVO**
- **Perfect Oracle**

These represent others as a few discrete types rather than continuous parameters.

Total NLLs:
- Good vs Evil: ~**14640**
- Canonical SVO: ~**16905**
- Perfect Oracle: ~**17381**

The “Perfect Oracle” result is especially revealing: even a model that starts with the true population frequencies performs badly.

### **Large typological search**
The paper then searches **15,000+** discrete models:
- all 3-type subsets of a 5x5 grid
- all 4-type subsets
- sampled 5- and 6-type subsets

Best discrete result:
- still around **8500+** NLL
- still clearly worse than the UBM

### **Conclusion**
The paper’s central representational claim is:

> human social-preference inference is better described by **graded, high-resolution uncertainty** than by a few rigid moral types.

---

## **8) Why 480 utility functions were compared**

Because the Bayesian likelihood depends on the utility function, the paper treats utility-form choice as an empirical problem.

### **Examples of dimensions varied**
The 14 Boolean utility settings include:

- conditional welfare
- reference-dependent altruism
- Rawlsian / Leontief min–max forms
- exponents on utility terms
- single exponent vs separate exponents
- exponents on payoffs vs on transformed terms
- single payoffs vs payoff differences
- payoff ratios
- reference-dependent utility with $R=3$
- general negativity / loss-aversion parameters
- envy–guilt asymmetry
- fixed vs variable self-interest
- inclusion of altruism
- inclusion of social comparison

This makes the paper unusual: it does **not** just compare a few famous models. It searches a large, structured space.

---

## **9) Pairwise comparison analysis: which ingredients help?**

The paper also asks which design choices tend to help fit **on average**.

### **Model relation types**
This matters for both interpretation and coding.

#### **Parent-child (nested)**
The parent contains the child plus extra flexibility and can collapse to the child at anchor values.

Examples:
- adding altruism
- adding social comparison
- splitting envy and guilt
- freeing self-interest
- going from one exponent to several
- adding negativity parameters

#### **Sibling-sibling**
Same parameter count, one Boolean flip, but neither is a strict special case.

Examples:
- single payoffs vs differences
- payoff ratios vs differences
- exponents on payoffs vs elsewhere
- reference-dependent utility on vs off

#### **None**
Some flips are neither under the paper’s criterion:
- conditional welfare form
- Rawlsian/Leontief min–max

### **Main average lessons**
Strong improvements:
- **use exponents**
- **include altruism**
- **include social comparison**
- **split envy and guilt**
- **let self-interest vary**

Strong harms:
- **payoff ratios**
- **apply exponents directly to payoffs**
- **Rawlsian/Leontief min–max**
- **midpoint reference dependence**

Mostly neutral:
- **single payoffs vs differences**

Important caveat:
these are average effects across the whole 480-model space; some settings can help in restricted subspaces even if they hurt overall.

---

## **10) The winning utility family**

The paper’s substantive conclusion is that classic low-dimensional models are too coarse.

### **What the winner contains**
The best-fitting family has **7 parameters**:

- self-interest
- altruism
- envy
- guilt
- exponent for self term
- exponent for altruism term
- exponent for social-comparison term

This is the important conceptual point:

> **absolute concern for the other** and **relative-position concern** are distinct.

That separation is one of the paper’s main contributions.

### **Costs of simplifying**
Approximate BIC penalties for reducing the winner:

- remove altruism -> **+442**
- symmetrize social comparison -> **+233**
- force a single exponent -> **+193**
- fix self-interest to 1 -> **+135**

So simpler models are not just less flexible; they can become **conceptually misleading** by collapsing distinct motives together.

---

## **11) How many utility architectures does the population need?**

The IC comparison identifies the single best utility architecture for the whole population. But participants may differ not only in preference *magnitudes* — how strongly they weight altruism or self-interest — but in the *structure* of their utility function. A person who is entirely indifferent to inequality requires a categorically different functional form than one who is strongly aversion-motivated.

The **architecture compression curve** addresses this question. For K = 1, 2, 3, …, it finds the set of K utility architectures that minimizes total population BIC under hard assignment: each participant uses whichever of the K architectures fits them best. A(K) measures the fraction of the *fully individualized* BIC advantage — giving every participant their own unique best architecture — that is captured by a K-architecture library.

```
A(1) = 0     (baseline: no individualization with one shared architecture)
A(K) → 1    (ceiling: every participant has their own architecture)
```

The curve's knee — identified by the Kneedle elbow criterion — is the answer to: *how many structurally distinct utility types does the population actually need?*

After the best K is found, **AMPD** (Average Model Policy Distance) is computed for the selected library. This checks behavioral distinctiveness: do the selected architectures make genuinely different predictions across game scenarios, or are they structurally different but functionally near-identical?

The compression curve and the IC analysis answer complementary questions:

- **IC:** what is the best single architectural description of the whole population?
- **Compression curve:** how many distinct architectural types are needed to describe the population?

A library that recovers most of the individualized advantage at K = 2 or K = 3 is more interesting than one requiring K = 10, because it suggests a small number of genuinely distinct social-preference strategies rather than continuous individual variation.

---

## **12) Data adequacy: the model recovery simulation**

A natural question about the IC comparison is whether the experimental design provides enough data to reliably identify the correct utility model. The **model recovery simulation** tests this by treating the IC pipeline itself as the object of study.

**Procedure:**
1. Choose a generating model (the IC winner by default).
2. Draw realistic parameter vectors from the IC-fitted distributions for that model.
3. Simulate synthetic chooser data for a grid of (n_agents × n_games) conditions.
4. Run the full IC candidate comparison on each condition.
5. Measure recovery: did the generating model win population BIC? what was its rank? how large was the BIC gap to the runner-up?

**Key recovery metrics:**
- **Recovery rate** — fraction of synthetic agents for whom the generating model achieves BIC rank 1
- **Mean BIC rank** — mean rank of the generating model among all candidates; 1 = perfect
- **Δ-BIC** — mean BIC gap between the generating model and the runner-up; larger = cleaner separation
- **AMPD to truth** — behavioral distance from the winning model to the generating model; small AMPD means the pipeline selected a model that behaves like the truth even if it is not the exact model
- **Conditional Hamming distance** — structural distance between the winner and truth in terms of active utility flags

The output is a grid of curves across the n_games axis, showing how each metric improves as data increases. These directly answer: *with our experimental design (N games per participant), how reliable is the IC model comparison?*

The simulation also tests a subtler point. If the pipeline does not perfectly recover the generating model, does it at least select a behaviorally similar one? A mismatch in model identity but near-zero AMPD to truth is a mild failure; a structurally and behaviorally distant winner would be a serious problem.

---

## **13) Robustness, nesting, and parent-fair regularization**

This section matters a lot for anyone touching the fitting code.

## **13.1 Robustness analysis**
Because optimization is stochastic, the paper repeats fitting across iterations and tracks:

1. **sum of Δ minimum loss**
2. **sum of rank changes**

Stopping occurs when incremental improvement becomes negligible; stability is reached by about **iteration 11**.

## **13.2 Nesting logic**
A properly optimized parent should never fit worse than its child. If it does, that signals an optimization failure, not a theoretical result.

The nesting graph is built explicitly:
- parent if it contains the child plus extra parameters,
- exactly one Boolean setting differs,
- and the parent reproduces the child at anchor values across all payoff structures.

## **13.3 Parent-fair L2 regularization**
Many utility families are quasi-scale-invariant, so local optimization can wander or stop early.

The paper adds a small penalty:

$$
\mathcal{L}_{\lambda}
=
\text{NLL} + \lambda \cdot \text{penalty}
$$

The key design constraint is **parent fairness**:

> when a parent is set to the child anchor, parent and child should incur the same penalty.

That stabilizes fitting **without** artificially favoring children.

## **13.4 Optimization schedule**
The schedule deliberately moves from exploration to exploitation:

1. early passes: SA + L-BFGS-B, no warm starts  
2. middle passes: child-to-parent warm starts, but random child chosen  
3. late passes: parent starts from its best-fitting child  

This is there to reduce local-minimum problems while preserving the nested model family.

---

## **14) Main empirical results**

## **14.1 Big picture**
The fitted distributions imply:

- strong self-interest
- real but smaller altruism
- asymmetric inequality aversion
- meaningful nonlinear payoff sensitivity
- substantial heterogeneity, including antisocial and self-sacrificial tails

## **14.2 The headline findings**

### **A) Self-interest > altruism**
Chooser means in the main results section:
- $\mu(V_{ii}) \approx 0.679$
- $\mu(V_{ij}) \approx 0.047$

The latest abstract summarizes the imbalance as **roughly seven-to-one**.  
The later results section also reports stronger filtered ratios. The stable qualitative claim is simple:

> humans value their own payoffs much more than others’ payoffs, but altruism is still meaningfully present.

### **B) Guilt > envy**
Chooser means:
- $\mu(Ɛ_{ij}) \approx 0.015$
- $\mu(Ʒ_{ij}) \approx 0.256$

This reverses the canonical Fehr–Schmidt ordering.

The paper’s interpretation is that the “guilt” parameter may partly act like **conditional altruism** or compassionate inequality aversion: people may be especially motivated to raise others **up to their own level**, not necessarily beyond it.

### **C) Term-specific nonlinear sensitivity**
Chooser exponents:
- self: $\gamma_1 \approx 0.83$
- altruism: $\gamma_2 \approx 1.34$
- social comparison: $\gamma_3 \approx 1.38$

Interpretation:
- own-payoff sensitivity is near-linear / mildly concave
- altruism and social comparison are **super-linear**

That is a striking result because it cuts against the usual diminishing-marginal-utility default.

### **D) Minority negative types**
The paper finds more antisocial or self-sacrificial mass than classic low-dimensional models would suggest, including:
- negative altruism / spite
- competitive inequality weights
- negative self-interest / masochistic self-sacrifice

### **E) Weak projection**
Chooser–predictor cross-role correlations are weak and mostly non-significant.

Interpretation:
participants are not simply projecting their own morality onto others.

---

## **15) Why the paper matters**

### **Theoretical contribution**
The paper argues that outcome-based social preferences need more than the classic 2D self-vs-other picture.

Two-dimensional models cannot cleanly separate:
- altruism
- self-interest
- disadvantageous inequality aversion
- advantageous inequality aversion
- nonlinear payoff sensitivity

### **Cumulative-science contribution**
The seven-parameter space functions like a **common psychological coordinate system**:
- every participant can be located in the same space
- future priors can be justified from empirical distributions
- cross-study comparisons become more meaningful

### **Broader relevance**
The paper frames this as relevant to:
- moral psychology
- behavioral game theory
- social cognition
- cooperation research
- and AI systems that need to infer, simulate, or align with human motives

---

## **16) Limits**

This is a model of **outcome-based social preferences**, not all of morality.

It does **not** directly model:
- reciprocity within a round
- intentions
- self-image / social-image
- norms
- multi-step strategic reasoning
- multi-player justice tradeoffs
- real losses around zero

Other important constraints:
- the IC analysis uses a **static approximation** for tractability
- the sample is WEIRD undergraduates
- payoffs are positive only, so “loss aversion” is only approximated around $R=3$

---

## **17) Coding-agent takeaways**

If you are editing code in this repo, preserve these invariants:

1. **Prediction data fit beliefs; chooser actions are the evidence.**
2. **Chooser parameters and predictor parameters are different objects.**
3. **The utility function and Bayesian update are coupled.**
4. **Parent models should not underperform children at anchor values.**
5. **Grid vs PF is a computational approximation choice, not a theory change.**
6. **Do not collapse altruism into social comparison.**
7. **The UBM is an inverse model of hidden motives, not just a generic classifier.**

A good pipeline sketch is:

```text
utility family
    -> likelihood over chooser actions
        -> Bayesian posterior over latent preferences
            -> prediction of future choice
                -> NLL against human predictions
                    -> optimization / model comparison
```

---

## **18) Practical version note**

The manuscript is evolving, so some early summary sentences lag behind later tables/equations.

For exact implementation details, trust in this order:

1. **latest abstract for the headline message**
2. **later results tables / explicit equations**
3. **IC top-model row + nesting logic**
4. **code-level canonical behavior if prose conflicts**

That matters because a few draft summaries appear older than the later technical sections.

---

## **19) Final takeaway**

This paper is trying to make hidden social motives measurable.

Its deepest claim is:

> people do not merely predict others’ actions — they infer a latent moral geometry behind those actions.

The UBM formalizes that idea.  
The 480-model IC analysis makes utility choice empirical.  
The final 7-parameter space turns vague talk about “human nature” into something much closer to a measurable coordinate system.

That is why this project matters.
