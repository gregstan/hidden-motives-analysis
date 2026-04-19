import preprocessing as prep
import typological as typo
import utilities as gnrl
from config import *

"=========================================================================================="
"============================== Utility and Choice Functions =============================="
"=========================================================================================="

def utility_term(payoff_1: float, payoff_2: float, weight_1: float, weight_2: float, exponent: float, 
                 use_negativity_parameters: bool, use_exponential_parameters: bool, 
                 single_payoffs_not_differences: bool, payoff_ratios_not_differences: bool) -> float:
    """
    Compute a single additive term of the utility function using lazy dict dispatch.

    The function dispatches to one of twelve formula variants depending on the three
    boolean flags (negativity, exponential, and payoff-representation). All variants
    are stored as lambdas in a dict and only the selected one is evaluated, avoiding
    unnecessary computation and keeping the branching logic declarative.

    Arguments:
        • payoff_1: float
            The payoff for the focal option (e.g., πᵢᴬ for self-interest, πⱼᴬ for altruism).
        • payoff_2: float
            The payoff for the counterfactual option (e.g., πᵢᴮ).
        • weight_1: float
            Positive-domain weight, applied when payoff_1 > payoff_2 (e.g., Vᵢᵢ, Vᵢⱼ).
        • weight_2: float
            Negative-domain weight, applied when payoff_1 < payoff_2 (e.g., Ʌᵢᵢ, Ʌᵢⱼ).
            Only active when use_negativity_parameters=True.
        • exponent: float
            Curvature exponent γ applied to the payoff base. Exponent of 1 gives a linear term.
        • use_negativity_parameters: bool
            If True, use separate weights for advantage (weight_1) and disadvantage (weight_2).
            If False, the same weight_1 applies to both sides of the payoff comparison.
        • use_exponential_parameters: bool
            If True, raise the payoff base to the power of exponent before multiplying by weight.
            If False, the term is linear in payoff differences/ratios.
        • single_payoffs_not_differences: bool
            If True, the base is the raw payoff_1 (not a difference or ratio).
            If False, the base is payoff_1 − payoff_2 (or the payoff ratio form if that flag is set).
        • payoff_ratios_not_differences: bool
            If True, compute the base as a centered payoff ratio:
                payoff_1 / (payoff_1 + payoff_2) − 0.5
            Takes priority over single_payoffs_not_differences when both are True.

    Returns:
        • float — the scalar value of this utility term.
    """
    def _pow_nonneg(base_value: float, exponent_value: float) -> float:
        """
        Safe power: for nonnegative bases only.
        """
        base_value = float(base_value)
        exponent_value = float(exponent_value)
        if base_value <= 0.0:
            return 0.0
        return base_value ** exponent_value

    term_dict = {
        ("negative", "expone", "payrat"): lambda: weight_1 * _pow_nonneg(max(payoff_1 / (payoff_1 + payoff_2) - 1/2, 0), exponent) - weight_2 * _pow_nonneg(max(payoff_2 / (payoff_1 + payoff_2) - 1/2, 0), exponent),
        ("positive", "expone", "payrat"): lambda: weight_1 * _pow_nonneg(max(payoff_1 / (payoff_1 + payoff_2) - 1/2, 0), exponent) - weight_1 * _pow_nonneg(max(payoff_2 / (payoff_1 + payoff_2) - 1/2, 0), exponent),
        ("negative", "linear", "payrat"): lambda: weight_1 * max(payoff_1 / (payoff_1 + payoff_2) - 1/2, 0) - weight_2 * max(payoff_2 / (payoff_1 + payoff_2) - 1/2, 0),
        ("positive", "linear", "payrat"): lambda: weight_1 * (payoff_1 / (payoff_1 + payoff_2) - 1/2),

        ("negative", "expone", "paydif"): lambda: weight_1 * _pow_nonneg(max(payoff_1 - payoff_2, 0), exponent) - weight_2 * _pow_nonneg(max(payoff_2 - payoff_1, 0), exponent),
        ("positive", "expone", "paydif"): lambda: weight_1 * _pow_nonneg(max(payoff_1 - payoff_2, 0), exponent) - weight_1 * _pow_nonneg(max(payoff_2 - payoff_1, 0), exponent),
        ("negative", "linear", "paydif"): lambda: weight_1 * max(payoff_1 - payoff_2, 0) - weight_2 * max(payoff_2 - payoff_1, 0),
        ("positive", "linear", "paydif"): lambda: weight_1 * (payoff_1 - payoff_2),

        ("negative", "expone", "onepay"): lambda: weight_1 * payoff_1**exponent if payoff_1 >= 0 else -weight_2 * abs(payoff_1)**exponent,
        ("positive", "expone", "onepay"): lambda: weight_1 * payoff_1**exponent if payoff_1 >= 0 else -weight_1 * abs(payoff_1)**exponent,
        ("negative", "linear", "onepay"): lambda: weight_1 * payoff_1 if payoff_1 >= 0 else -weight_2 * abs(payoff_1),
        ("positive", "linear", "onepay"): lambda: weight_1 * payoff_1,
    }
    try:
        use_neg = "negative" if use_negativity_parameters else      "positive"
        use_exp = "expone"   if use_exponential_parameters else     "linear"
        pay_typ = "onepay"   if single_payoffs_not_differences else "paydif"
        if payoff_ratios_not_differences:
            pay_typ = "payrat" 
            
        return term_dict[(use_neg, use_exp, pay_typ)]()

    except ZeroDivisionError as err:
        "Providing debugging information for zero division errors."
        print(f"Key: ({use_negativity_parameters}, {use_exponential_parameters}, {single_payoffs_not_differences}, {payoff_ratios_not_differences})")
        print(f"payoff_1: {payoff_1}, payoff_2: {payoff_2}, weight_1: {weight_1}, weight_2: {weight_2}, exponent: {exponent}")
        raise ZeroDivisionError(err)


def utility(payoffs: dict[str, int], params: dict[str, float], utility_settings: UtilitySettings, separate_terms: bool = False, normalize_conditional_welfare_params: bool = True) -> float:
    """
    Compute the total utility for choosing option A over B given a payoff bundle and parameter set.

    This is the central evaluation function for the UBM. It routes to one of three
    structural families — standard additive, conditional-welfare, or min-max/Rawlsian/Leontief —
    based on the active flags in utility_settings, then sums self-interest, altruism, and social-
    comparison terms by calling utility_term for each.

    Arguments:
        • payoffs: dict[str, int]
            Must contain keys {'As', 'Ao', 'Bs', 'Bo'} mapping option–role combinations to payoffs:
                'As' = payoff to self (chooser) under option A
                'Ao' = payoff to other (predictor) under option A
                'Bs' = payoff to self under option B
                'Bo' = payoff to other under option B
        • params: dict[str, float]
            Social-preference parameter values. Recognized keys (all optional; defaults shown):
                'Vᵢᵢ' (1), 'Ʌᵢᵢ' (0), 'Vᵢⱼ' (0), 'Ʌᵢⱼ' (0),
                'Ƹᵢⱼ' (0), 'Ʒᵢⱼ' (0), 'γ1' (1), 'γ2' (γ1), 'γ3' (γ1).
        • utility_settings: UtilitySettings
            Dict of boolean toggles that select the active functional form. The key flags are:
                'include_altruism_term', 'include_social_comparison', 'use_negativity_parameters',
                'use_exponential_parameters', 'fix_self_interest_parameter',
                'single_payoffs_not_differences', 'payoff_ratios_not_differences',
                'reference_dependent_utility', 'conditional_welfare_mode',
                'min_max_rawlsian_leontief', 'single_exponential_parameter'.
        • separate_terms: bool
            If True, returns a dict {'self_interest': float, 'altruism': float, 'social_comp': float}
            instead of the scalar sum. Useful for inspecting the contribution of each term.
        • normalize_conditional_welfare_params: bool
            Only active when conditional_welfare_mode=True. If True, re-scales weight parameters
            from [−1, 1] to [0, 1] before computing the conditional-welfare utility. This makes
            the parameter space symmetric around 0.5 and avoids negative weights in that mode.

    Returns:
        • float — net utility for choosing A over B (positive → prefer A, negative → prefer B).
          If separate_terms=True, returns dict[str, float] instead.
    """
    Vᵢᵢ = params.get('Vᵢᵢ', 1)
    Ʌᵢᵢ = params.get('Ʌᵢᵢ', 0)
    Vᵢⱼ = params.get('Vᵢⱼ', 0)
    Ʌᵢⱼ = params.get('Ʌᵢⱼ', 0)
    Ƹᵢⱼ = params.get('Ƹᵢⱼ', 0)
    Ʒᵢⱼ = params.get('Ʒᵢⱼ', 0)
    exp1 = params.get('γ1', 1)
    exp2 = params.get('γ2', exp1)
    exp3 = params.get('γ3', exp1)

    "Enforce a single exponent when requested (even if γ2/γ3 are present in params)"
    if utility_settings.get('single_exponential_parameter', False):
        exp2 = exp1
        exp3 = exp1

    payAi, payAj = payoffs['As'], payoffs['Ao']
    payBi, payBj = payoffs['Bs'], payoffs['Bo']
    if utility_settings['reference_dependent_utility']:
        payBi, payBj = 3, 3
    pay1si, pay2si = payAi, payBi
    pay1al, pay2al = payAj, payBj
    pay1sc, pay2sc = payAj, payAi

    if utility_settings.get('apply_exponents_to_payoffs') and utility_settings['use_exponential_parameters']:
        pay1si, pay2si = pay1si ** exp1, pay2si ** exp1
        pay1al, pay2al = pay1al ** exp2, pay2al ** exp2
        pay1sc, pay2sc = pay1sc ** exp3, pay2sc ** exp3
        exp1, exp2, exp3 = 1, 1, 1

    if utility_settings.get('conditional_welfare_mode'):
        "Applies different weights self and other's payoffs when ahead versus behind."

        ref_point = 3 if utility_settings.get('reference_dependent_altruism') else payAj

        if normalize_conditional_welfare_params:
            "Normalize weight parameters from [-1, 1] to [0, 1]"
            Vᵢᵢ, Ʌᵢᵢ = (Vᵢᵢ + 1) / 2, (Ʌᵢᵢ + 1) / 2
            Vᵢⱼ, Ʌᵢⱼ = (Vᵢⱼ + 1) / 2, (Ʌᵢⱼ + 1) / 2

        if payAi >= ref_point:
            "If oneself is ahead, use positivity parameters."
            weight_1_si = 1 if utility_settings['fix_self_interest_parameter'] else Vᵢᵢ
            weight_1_al = Vᵢⱼ if utility_settings['include_altruism_term'] else 1 - Vᵢᵢ
        else:
            "If oneself is behind, use negativity parameters."
            weight_1_si = Ʌᵢᵢ
            weight_1_al = Ʌᵢⱼ if utility_settings['include_altruism_term'] else 1 - Ʌᵢᵢ

        "Self-interest term"
        self_interest = utility_term(payoff_1=pay1si, payoff_2=pay2si, 
                                     weight_1=weight_1_si, weight_2=0, exponent=exp1,  
                                     use_exponential_parameters=utility_settings['use_exponential_parameters'], 
                                     single_payoffs_not_differences=utility_settings['single_payoffs_not_differences'], 
                                     payoff_ratios_not_differences=utility_settings['payoff_ratios_not_differences'],
                                     use_negativity_parameters=False)
        "Altruism term"
        altruism = utility_term(payoff_1=pay1al, payoff_2=pay2al, 
                                weight_1=weight_1_al, weight_2=0, 
                                # exponent=exp2 if utility_settings['include_altruism_term'] else exp1, 
                                exponent=exp2 if utility_settings['single_exponential_parameter'] else exp1, 
                                use_exponential_parameters=utility_settings['use_exponential_parameters'], 
                                single_payoffs_not_differences=utility_settings['single_payoffs_not_differences'], 
                                payoff_ratios_not_differences=utility_settings['payoff_ratios_not_differences'],
                                use_negativity_parameters=False)
        
        if separate_terms:
            return {'self_interest': float(self_interest), 'altruism': float(altruism), 'social_comp': 0.0}
        
        return self_interest + altruism

    elif utility_settings.get('min_max_rawlsian_leontief'):
        "Decide bases and the remaining exponents to apply to the bases."
        if utility_settings['single_payoffs_not_differences']:
            "Single-payoff flavor ignores reference-dependence"
            basei = payAi
            basej = payAj
            exp_i = exp1
            exp_j = (exp1 if utility_settings['single_exponential_parameter'] else exp2)

        elif utility_settings['payoff_ratios_not_differences']:
            "Ratios, centered at 1/2. If we apply γ into payoffs, do it here."
            if utility_settings.get('apply_exponents_to_payoffs') and utility_settings['use_exponential_parameters']:
                "Apply γ into the two payoff arguments that form each ratio"
                ai = (payAi ** exp1)
                bi = (payBi ** exp1)
                aj = (payAj ** (exp1 if utility_settings['single_exponential_parameter'] else exp2))
                bj = (payBj ** (exp1 if utility_settings['single_exponential_parameter'] else exp2))
                basei = ai / (ai + bi) - 0.5
                basej = aj / (aj + bj) - 0.5
                exp_i = exp_j = 1.0  
            else:
                basei = payAi / (payAi + payBi) - 0.5
                basej = payAj / (payAj + payBj) - 0.5
                exp_i = exp1
                exp_j = (exp1 if utility_settings['single_exponential_parameter'] else exp2)

        else:
            "Payoff differences"
            if utility_settings.get('apply_exponents_to_payoffs') and utility_settings['use_exponential_parameters']:
                "Apply γ into the two payoff arguments before subtracting"
                ai = (payAi ** exp1)
                bi = (payBi ** exp1)
                aj = (payAj ** (exp1 if utility_settings['single_exponential_parameter'] else exp2))
                bj = (payBj ** (exp1 if utility_settings['single_exponential_parameter'] else exp2))
                basei = ai - bi
                basej = aj - bj
                exp_i = exp_j = 1.0  
            else:
                basei = payAi - payBi
                basej = payAj - payBj
                exp_i = exp1
                exp_j = (exp1 if utility_settings['single_exponential_parameter'] else exp2)

        "Apply exponents to the bases only if they haven't already been applied"
        if utility_settings['use_exponential_parameters']:
            if not utility_settings.get('apply_exponents_to_payoffs', False):
                "Exponent is on the base: match the string builder’s ReLU-signed rewrite"
                if exp_i != 1:
                    basei = (
                        max(basei, 0.0) ** exp_i
                        - max(-basei, 0.0) ** exp_i
                    )
                if exp_j != 1:
                    basej = (
                        max(basej, 0.0) ** exp_j
                        - max(-basej, 0.0) ** exp_j
                    )

        "Pick min vs max based on social_comparison & altruism"
        if utility_settings['include_social_comparison']:
            "Rawlsian functional forms"
            if utility_settings['include_altruism_term']:
                if separate_terms:
                    return {'self_interest': 0.0, 'altruism': float(Vᵢⱼ * min(basei, basej)), 'social_comp': 0.0}
                return Vᵢⱼ * min(basei, basej)
            else:
                if separate_terms:
                    return {'self_interest': 0.0, 'altruism': float(Vᵢⱼ * max(basei, basej)), 'social_comp': 0.0}
                return Vᵢⱼ * max(basei, basej)
        else:
            "Leontief functional forms"
            if utility_settings['include_altruism_term']:
                if separate_terms:
                    return {'self_interest': 0.0, 'altruism': float(min(Vᵢᵢ * basei, Vᵢⱼ * basej)), 'social_comp': 0.0}
                return min(Vᵢᵢ * basei, Vᵢⱼ * basej)
            else:
                if separate_terms:
                    return {'self_interest': 0.0, 'altruism': float(max(Vᵢᵢ * basei, Vᵢⱼ * basej)), 'social_comp': 0.0}
                return max(Vᵢᵢ * basei, Vᵢⱼ * basej)            

    "Self-interest term"
    self_interest = utility_term(payoff_1=pay1si, payoff_2=pay2si, 
                                 weight_1=1 if utility_settings['fix_self_interest_parameter'] else Vᵢᵢ, 
                                 weight_2=Ʌᵢᵢ if utility_settings['use_negativity_parameters'] else 0.0, 
                                 exponent=exp1, use_negativity_parameters=utility_settings['use_negativity_parameters'], 
                                 use_exponential_parameters=utility_settings['use_exponential_parameters'], 
                                 single_payoffs_not_differences=utility_settings['single_payoffs_not_differences'], 
                                 payoff_ratios_not_differences=utility_settings['payoff_ratios_not_differences'])  

    "Altruism term"
    if utility_settings['include_altruism_term']:
        altruism = utility_term(payoff_1=pay1al, payoff_2=pay2al, weight_1=Vᵢⱼ, exponent=exp2, 
                                weight_2=Ʌᵢⱼ if utility_settings['use_negativity_parameters'] else 0.0, 
                                use_negativity_parameters=utility_settings['use_negativity_parameters'], 
                                use_exponential_parameters=utility_settings['use_exponential_parameters'], 
                                single_payoffs_not_differences=utility_settings['single_payoffs_not_differences'], 
                                payoff_ratios_not_differences=utility_settings['payoff_ratios_not_differences'])  
    else:
        altruism = 0.0

    "Social comparison term"
    if utility_settings['include_social_comparison']:
        social_comp = utility_term(
            payoff_1=pay1sc, payoff_2=pay2sc,
            weight_1=-Ƹᵢⱼ, weight_2=Ʒᵢⱼ if utility_settings['negativity_social_comparison'] else Ƹᵢⱼ,
            exponent=exp3, use_exponential_parameters=utility_settings['use_exponential_parameters'],
            payoff_ratios_not_differences=utility_settings['payoff_ratios_not_differences'],
            use_negativity_parameters=True, single_payoffs_not_differences=False
        )
    else:
        social_comp = 0.0


    if separate_terms:
        return {'self_interest': float(self_interest), 'altruism': float(altruism), 'social_comp': float(social_comp)}

    return self_interest + altruism + social_comp


def softmax_(uA: float, uB: float, temperature: float = 1.5, use_fallback: bool = True) -> float:
    """
    Convert utilities for options A and B into a probability of choosing A via the softmax rule.

    Handles non-finite utility values gracefully by falling back to deterministic boundary
    probabilities rather than crashing — important for robustness during optimization when
    extreme parameter values can produce ±inf utilities.

    Arguments:
        • uA: float
            Utility for option A (output of utility() evaluated on the A-payoff bundle).
        • uB: float
            Utility for option B (output of utility() evaluated on the B-payoff bundle).
        • temperature: float
            SoftMax temperature τ. Higher values flatten the choice distribution toward 0.5;
            lower values make it more deterministic. Must be strictly positive.
        • use_fallback: bool
            If True (default), non-finite utility values are handled with printed warnings
            and sensible boundary probabilities (0.0 or 1.0). If False, raises ValueError
            instead, which is useful for debugging during optimization.

    Returns:
        • float — probability of choosing A, in [0, 1].
    """
    "Ensure temperature is valid"
    if temperature <= 0:
        raise ValueError(f"Temperature must be greater than 0. Received: {temperature}")
    
    "Check if utilities are finite"
    uA_finite = np.isfinite(uA)
    uB_finite = np.isfinite(uB)

    if not uA_finite and not uB_finite:
        "Both utilities are non-finite → No meaningful comparison, return 0.5."
        if use_fallback:
            print(f"Warning: Both uA and uB are non-finite (uA={uA}, uB={uB}). Returning 0.5.")
            return 0.5
        else:
            raise ValueError(f"Both utilities are non-finite: uA={uA}, uB={uB}")

    elif uA_finite and not uB_finite:
        "Only uA is finite → Compare to zero."
        if use_fallback:
            print(f"Warning: uB is non-finite (uB={uB}). Using uA={uA} for decision.")
            return 1.0 if uA > 0 else 0.0
        else:
            raise ValueError(f"Non-finite uB: {uB}")

    elif not uA_finite and uB_finite:
        "Only uB is finite → Compare to zero."
        if use_fallback:
            print(f"Warning: uA is non-finite (uA={uA}). Using uB={uB} for decision.")
            return 0.0 if uB > 0 else 1.0
        else:
            raise ValueError(f"Non-finite uA: {uA}")

    "Create a NumPy array of the utilities, scaled by temperature"
    utilities = np.array([uA, uB]) / temperature

    "Use scipy.special.softmax to calculate the probabilities"
    probabilities = softmax(utilities)

    "Return the probability of choosing A (the first element)"
    return probabilities[0]


def choice(current_game: dict[str, Any], agent_params: Dict[str, float], utility_settings: UtilitySettings, 
           softmax_temperature: float = 1.5, select: bool = False, normalize_conditional_welfare_params: bool = True) -> dict[str, float | Any]:
    """
    Compute the model-predicted probability (or binary decision) for option A in a single game.

    Extracts payoffs from current_game, evaluates utility() for both options under agent_params,
    and passes the resulting utilities through softmax_() to get a choice probability. Optionally
    converts that probability to a binary draw (select=True), which is used during simulation.
    Also computes a confidence score from parameter standard deviations, included in the output
    so downstream code can weight predictions by their certainty.

    Arguments:
        • current_game: dict
            A game dictionary containing at minimum:
                'payoff_A_chooser', 'payoff_A_predictor', 'payoff_B_chooser', 'payoff_B_predictor'.
        • agent_params: dict[str, float]
            Social-preference parameters plus optional standard deviations and temperature.
            Example: {
                'Vᵢᵢ': 0.958, 'Vᵢⱼ': 0.333, 'Ƹᵢⱼ': 0.274, 'γ1': 0.800,
                'Vᵢᵢ_std': 0.5,  'Vᵢⱼ_std': 0.3,  'Ƹᵢⱼ_std': 0.1,  'γ1_std': 0.9,
            }
            Keys ending in '_std' are used to compute the confidence score but not passed to utility().
        • utility_settings: UtilitySettings
            Boolean toggles selecting the active utility functional form (see utility() docs).
        • softmax_temperature: float
            SoftMax temperature τ controlling stochasticity of choices. Higher → flatter distribution.
        • select: bool
            If True, stochastically converts the probability to a binary 1/0 via a random draw.
            If False, returns the raw probability in [0, 1].
        • normalize_conditional_welfare_params: bool
            Passed through to utility(). Only relevant when conditional_welfare_mode=True.

    Returns:
        • dict with keys:
            - 'model_choose_A': float in [0, 1] (probability) or int in {0, 1} (if select=True).
            - 'confidence': float in (0, 1]; inverse-mean-std-based measure of parameter certainty.
    """
    "Identify payoffs from the game."
    payoff_A_chooser =   current_game.get('payoff_A_chooser', None)
    payoff_A_predictor = current_game.get('payoff_A_predictor', None)
    payoff_B_chooser =   current_game.get('payoff_B_chooser', None)
    payoff_B_predictor = current_game.get('payoff_B_predictor', None)
    
    if any(payoff is None for payoff in (payoff_A_chooser, payoff_A_predictor, payoff_B_chooser, payoff_B_predictor)):
        raise Exception(f"Failed to extract payoff in game {current_game}.")

    payoffsA = {'As': payoff_A_chooser, 'Ao': payoff_A_predictor, 'Bs': payoff_B_chooser, 'Bo': payoff_B_predictor}
    payoffsB = {'As': payoff_B_chooser, 'Ao': payoff_B_predictor, 'Bs': payoff_A_chooser, 'Bo': payoff_A_predictor}

    "Compute utilities based on payoffs and parameters."
    utilityA = utility(payoffs=payoffsA, params=agent_params, utility_settings=utility_settings, normalize_conditional_welfare_params=normalize_conditional_welfare_params)
    utilityB = utility(payoffs=payoffsB, params=agent_params, utility_settings=utility_settings, normalize_conditional_welfare_params=normalize_conditional_welfare_params)

    "Compute choice probability via SoftMax."
    p_choose_A = softmax_(uA=utilityA, uB=utilityB, temperature=softmax_temperature)

    "Compute 'confidence' as inverse mean variance"
    std_params = [agent_params[pkey] for pkey in agent_params if pkey.endswith('_std')]
    if not std_params:
        confidence = 1.0  # Default confidence if no valid parameters
    else:
        # confidence = np.exp(-np.mean(std_params))
        confidence = math.exp(-np.mean(std_params))

    if select:
        "Convert to binary 1/0"
        random_draw = random.random()
        p_choose_A = 1 if random_draw < p_choose_A else 0        

    return {"model_choose_A": p_choose_A, "confidence": confidence}


def build_utility_equation(utility_settings: Dict[str, bool], option: str = "A") -> str:
    """
    Build a human-readable symbolic string representation of the active utility function.

    Used for figure labels, printed output, and the IC analysis table. The string mirrors the
    mathematical expression that utility() would evaluate numerically, including Greek parameter
    symbols (Vᵢᵢ, Ʌᵢᵢ, γ₁, etc.) and payoff notation (πᵢᴬ, πⱼᴮ, etc.). It respects all
    active flags in utility_settings: altruism terms, social comparison, negativity parameters,
    exponent parameters, payoff-difference vs. ratio form, and the conditional-welfare and
    min-max/Rawlsian/Leontief structural families.

    Arguments:
        • utility_settings: Dict[str, bool]
            The same toggle dict passed to utility(). May be supplied as either a dict or a
            tuple (the tuple form used internally during the IC analysis sweep); tuples are
            automatically converted before processing.
        • option: str
            Which option's utility to express: 'A' produces 'Uᵢ(A) = …', 'B' produces 'Uᵢ(B) = …'.
            Controls which payoff symbols are assigned to self vs. other.

    Returns:
        • str — the formatted utility equation, e.g. 'Uᵢ(A) = Vᵢᵢ(πᵢᴬ − πᵢᴮ) + Vᵢⱼ(πⱼᴬ − πⱼᴮ)'.
    """
    if isinstance(utility_settings, tuple):
        utility_settings = gnrl.convert_utility_settings(utility_settings=utility_settings, into=dict)

    "Extract booleans"
    con_welf = utility_settings.get('conditional_welfare_mode', False)
    min_max  = utility_settings.get('min_max_rawlsian_leontief', False)
    loss_av  = utility_settings.get('use_negativity_parameters', False)
    use_exp  = utility_settings.get('use_exponential_parameters', False)
    pay_expo = utility_settings.get('apply_exponents_to_payoffs', False)
    ref_dep  = utility_settings.get('reference_dependent_utility', False)
    ref_alt  = utility_settings.get('reference_dependent_altruism', False)
    one_pay  = utility_settings.get('single_payoffs_not_differences', False)
    pay_rats = utility_settings.get('payoff_ratios_not_differences', False)
    one_exp  = utility_settings.get('single_exponential_parameter', False)
    la_socc  = utility_settings.get('negativity_social_comparison', False)
    fix_self = utility_settings.get('fix_self_interest_parameter', False)
    soc_comp = utility_settings.get('include_social_comparison', False)
    alt_term = utility_settings.get('include_altruism_term', False)
    
    def _simplify_nonnegatives(pretty: str) -> str:
        """
        Simplify ReLU expressions when the argument is a single payoff or '3':
        • max(π·, 0)^γₖ → π·^γₖ
        • max(π·, 0)    → π·
        • max(-π·,0)^γₖ → 0
        • max(-π·,0)    → 0
        """
        toks = r"(πᵢᴬ|πⱼᴬ|πᵢᴮ|πⱼᴮ|3)"
        "With exponent"
        pretty = re.sub(rf"max\(\s*{toks}\s*,\s*0\s*\)\s*\^γ([₁₂₃])", r"\1^γ\2", pretty)
        pretty = re.sub(rf"max\(\s*-\s*{toks}\s*,\s*0\s*\)\s*\^γ([₁₂₃])", r"0", pretty)
        "Without exponent"
        pretty = re.sub(rf"max\(\s*{toks}\s*,\s*0\s*\)", r"\1", pretty)
        pretty = re.sub(rf"max\(\s*-\s*{toks}\s*,\s*0\s*\)", r"0", pretty)
        "Clean up trivial '× 0' terms that may remain"
        pretty = re.sub(r"\s*[+−-]\s*[^+−]*×\s*0(\^γ[₁₂₃])?", "", pretty)
        return pretty

    def _apply_gamma_to_payoffs(pretty_equation: str, utility_settings: UtilitySettings) -> str:
        """
        Post-render rewrite used only when:
            use_exponential_parameters=True and apply_exponents_to_payoffs=True.
        It rewrites outer ^γt into per-payoff exponents inside the bracket/parenthesis/max.

        Handles:
            • (πX - πY)^γt                → (πX^γt - πY^γt)
            • max(πX - πY, 0)^γt          → max(πX^γt - πY^γt, 0)
            • (πX - 3)^γt, (3 - πX)^γt    → (πX^γt - 3^γt), (3^γt - πX^γt)
            • max(πX - 3, 0)^γt           → max(πX^γt - 3^γt, 0)
            • max(3 - πX, 0)^γt           → max(3^γt - πX^γt, 0)
            • [inner]^γt                  → [inner with π·^γt and 3^γt]
            • max([inner], 0)^γt          → max([inner with π·^γt and 3^γt], 0)
            • max(-[inner], 0)^γt         → max(-[inner with π·^γt and 3^γt], 0)
        """
        if not utility_settings.get('apply_exponents_to_payoffs', False):
            return pretty_equation

        out = pretty_equation

        subs = "₁₂₃"
        PHI  = r"π[ᵢⱼ][ᴬᴮ]"  # πᵢᴬ, πⱼᴬ, πᵢᴮ, πⱼᴮ

        def raise_inside(inner: str, t: str) -> str:
            "Raise all π tokens"
            inner = re.sub(rf"{PHI}", lambda m: f"{m.group(0)}^γ{t}", inner)

            "Raise bare '3' tokens (avoid touching '1/3' etc., and don't re-raise 3^γ)"
            "1) Start-of-string '3'"
            inner = re.sub(rf"^3(?!\^γ)", f"3^γ{t}", inner)
            "2) '3' between non-word chars"
            inner = re.sub(rf"(?<=\W)3(?!\^γ)(?=\W)", f"3^γ{t}", inner)
            "3) End-of-string '3'"
            inner = re.sub(rf"(?<=\W)3(?!\^γ)$", f"3^γ{t}", inner)

            return inner

        "(π - π)^γt"
        out = re.sub(rf"\(\s*({PHI})\s*-\s*({PHI})\s*\)\s*\^γ([{subs}])",
                    lambda m: f"({m.group(1)}^γ{m.group(3)} - {m.group(2)}^γ{m.group(3)})", out)

        "(π - 3)^γt and (3 - π)^γt"
        out = re.sub(rf"\(\s*({PHI})\s*-\s*3\s*\)\s*\^γ([{subs}])",
                    lambda m: f"({m.group(1)}^γ{m.group(2)} - 3^γ{m.group(2)})", out)
        out = re.sub(rf"\(\s*3\s*-\s*({PHI})\s*\)\s*\^γ([{subs}])",
                    lambda m: f"(3^γ{m.group(2)} - {m.group(1)}^γ{m.group(2)})", out)

        "max(π - π, 0)^γt"
        out = re.sub(rf"max\(\s*({PHI})\s*-\s*({PHI})\s*,\s*0\s*\)\s*\^γ([{subs}])",
                    lambda m: f"max({m.group(1)}^γ{m.group(3)} - {m.group(2)}^γ{m.group(3)}, 0)", out)

        "max(π - 3, 0)^γt and max(3 - π, 0)^γt"
        out = re.sub(rf"max\(\s*({PHI})\s*-\s*3\s*,\s*0\s*\)\s*\^γ([{subs}])",
                    lambda m: f"max({m.group(1)}^γ{m.group(2)} - 3^γ{m.group(2)}, 0)", out)
        out = re.sub(rf"max\(\s*3\s*-\s*({PHI})\s*,\s*0\s*\)\s*\^γ([{subs}])",
                    lambda m: f"max(3^γ{m.group(2)} - {m.group(1)}^γ{m.group(2)}, 0)", out)

        "[ ... ]^γt"
        out = re.sub(rf"\[\s*([^\]]+?)\s*\]\s*\^γ([{subs}])",
                    lambda m: "[" + raise_inside(m.group(1), m.group(2)) + "]", out)

        "max([ ... ], 0)^γt"
        out = re.sub(rf"max\(\s*\[\s*([^\]]+?)\s*\]\s*,\s*0\s*\)\s*\^γ([{subs}])",
                    lambda m: "max([" + raise_inside(m.group(1), m.group(2)) + "], 0)", out)

        "max(-[ ... ], 0)^γt"
        out = re.sub(rf"max\(\s*-\s*\[\s*([^\]]+?)\s*\]\s*,\s*0\s*\)\s*\^γ([{subs}])",
                    lambda m: "max(-[" + raise_inside(m.group(1), m.group(2)) + "], 0)", out)

        "Also: ( - [ … ] )^γt  (if present anywhere)"
        out = re.sub(rf"\(\s*-\s*\[\s*([^\]]+?)\s*\]\s*\)\s*\^γ([{subs}])",
                    lambda m: "(-[" + raise_inside(m.group(1), m.group(2)) + "])", out)

        out = _simplify_nonnegatives(out)

        return out
    
    def term(term_type: str) -> str:
        """Creates all three terms for the equation."""

        "generate weights, bases, and operators."
        if term_type == "self-interest":
            weight1, weight2 = f"{Vᵢᵢ}", "Ʌᵢᵢ"
            operator1, operator2 = "", " - "
            base1 = f"{payAi} - {payBi}" if not one_pay or ref_dep else payAi
            base2 = f"{payBi} - {payAi}" if not one_pay or ref_dep else payAi

            "Setting to use payoff ratios, not differences or single payoffs."
            if pay_rats:
                base1 = f"[{payAi} / ({payAi} + {payBi}) - 1/2]" 
                base2 = f"[{payBi} / ({payBi} + {payAi}) - 1/2]" 

        elif term_type == "altruism":
            if not alt_term:
                return ""
            weight1, weight2 = "Vᵢⱼ", "Ʌᵢⱼ"
            operator1, operator2 = " + ", " - "
            base1 = f"{payAj} - {payBj}" if not one_pay or ref_dep else payAj
            base2 = f"{payBj} - {payAj}" if not one_pay or ref_dep else payAj  
        
            "Setting to use payoff ratios, not differences or single payoffs."
            if pay_rats:
                base1 = f"[{payAj} / ({payAj} + {payBj}) - 1/2]" 
                base2 = f"[{payBj} / ({payBj} + {payAj}) - 1/2]" 

        elif term_type == "social_comparison":
            if not soc_comp:
                return ""

            "Bases for envy (other - self) and guilt (self - other)"
            if pay_rats:
                envy_base  = f"[{payAj} / ({payAi} + {payAj}) - 1/2]"
                guilt_base = f"[{payAi} / ({payAi} + {payAj}) - 1/2]"
            else:
                envy_base  = f"{payAj} - {payAi}"
                guilt_base = f"{payAi} - {payAj}"

            if use_exp:
                exp_tag = "^γ₁" if one_exp else "^γ₃"
            else:
                exp_tag = ""

            if la_socc:
                "Two-sided: (-Ƹ)·max(envy,0)^γ - (−Ƹ)·max(guilt,0)^γ == -Ƹ*max(envy)^γ + Ƹ*max(guilt)^γ"
                left  = f"- Ƹᵢⱼ × max({envy_base}, 0){exp_tag}"
                right = f"- Ʒᵢⱼ × max({guilt_base}, 0){exp_tag}"
                return f" {left} {right}"
            else:
                "Symmetric: single weight on (envy - guilt)"
                left  = f"max({envy_base}, 0){exp_tag}"
                right = f"max({guilt_base}, 0){exp_tag}"
                return f" - Ƹᵢⱼ × ({left} + {right})"

        else:
            error_str_end = "Use 'self-interest', 'altruism', or 'social_comparison'."
            raise ValueError(f"term_type {term_type} not supported. {error_str_end}")

        "Place bases within parentheses or max operators."
        if (term_type == "social_comparison" and la_socc) or loss_av:
            "comparison = '0.5' if pay_rats and term_type != 'social_comparison' else '0'"
            comparison = "0"
            if fix_self and term_type == "self-interest":
                base1 = f"max({base1}, {comparison})" 
            else:
                base1 = f" × max({base1}, {comparison})"
            base2 = f" × max({base2}, {comparison})"
        elif term_type == "social_comparison":
            base1 = f"({base1})"
            base2 = f"({base2})"
        elif pay_rats or (fix_self and term_type == "self-interest" and not use_exp):
            base1 = f"{base1}"
            base2 = f"{base2}"
        else:
            base1 = f"({base1})"
            base2 = f"({base2})"

        "Create exponent parameters, if any."
        if use_exp:
            exponent = "^γ₁"
            if not one_exp:
                if term_type == "altruism":
                    exponent = "^γ₂"
                elif term_type == "social_comparison":
                    exponent = "^γ₃"
        else:
            exponent = ""

        "Join operators, weights, bases, and exponents."
        group1 = f"{operator1}{weight1}{base1}{exponent}"
        group2 = f"{operator2}{weight2}{base2}{exponent}"

        "Use both groups if negativity parameters are included."
        if (loss_av and not (term_type == "self-interest" and fix_self)) or (la_socc and term_type == "social_comparison"):
            "if loss_av or (la_socc and term_type == 'social_comparison'):"
            return group1 + group2
        else:
            return group1

    "Make self-interest param a variable or a constant."
    Vᵢᵢ = "" if fix_self else "Vᵢᵢ"

    "Create payoffs and utility term."
    if option == "A":
        utility_ = "Uᵢ(A) = "
        payAi, payAj = "πᵢᴬ", "πⱼᴬ"
        payBi, payBj = "πᵢᴮ", "πⱼᴮ"
    elif option == "B":
        utility_ = "Uᵢ(B) = "
        payAi, payAj = "πᵢᴮ", "πⱼᴮ"
        payBi, payBj = "πᵢᴬ", "πⱼᴬ"
    else:
        raise ValueError(f"option {option} not supported. Use 'A' or 'B'.")
    
    if ref_dep:
        "Use 3 as reference point for reference dependent utility functions."
        payBi, payBj = "3", "3"

    if con_welf:

        if pay_expo:
            payAi, payBi = f"{payAi}^γ₁", f"{payBi}^γ₁"
            if one_exp:
                payAj, payBj = f"{payAj}^γ₁", f"{payBj}^γ₁"
            else:
                payAj, payBj = f"{payAj}^γ₂", f"{payBj}^γ₂"

        base_self = f"({payAi} - {payBi})" if not one_pay or ref_dep else f"({payAi})"
        base_altr = f"({payAj} - {payBj})" if not one_pay or ref_dep else f"({payAj})"
        if pay_rats:
            base_self = f"[{payAi} / ({payAi} + {payBi}) - 1/2]"
            base_altr = f"[{payAj} / ({payAj} + {payBj}) - 1/2]"

        "Apply exponents to bases."
        if use_exp and not pay_expo:
            base_self = f"{base_self}^γ₁" 
            if one_exp:
                base_altr = f"{base_altr}^γ₁" 
            else:
                base_altr = f"{base_altr}^γ₂" 

        if not one_pay or ref_dep:
            base_self = f" × {base_self}"
            base_altr = f" × {base_altr}"

        ahead_self = f"Vᵢᵢ" + base_self
        behind_self = f"Ʌᵢᵢ" + base_self
        if alt_term:
            ahead_altr = f"Vᵢⱼ" + base_altr
            behind_altr = f"Ʌᵢⱼ" + base_altr
        else:
            ahead_altr = f"(1 - Vᵢᵢ)" + base_altr
            behind_altr = f"(1 - Ʌᵢᵢ)" + base_altr

        if ref_alt:
            payAj = "3" 

        return f"{utility_}{ahead_self} + {ahead_altr} if {payAi} ≥ {payAj} else {behind_self} + {behind_altr}"
       
    elif min_max:

        if pay_expo:
            payAi, payBi = f"{payAi}^γ₁", f"{payBi}^γ₁"
            if one_exp:
                payAj, payBj = f"{payAj}^γ₁", f"{payBj}^γ₁"
            else:
                payAj, payBj = f"{payAj}^γ₂", f"{payBj}^γ₂"

        if one_pay:
            basei = f"{payAi}"
            basej = f"{payAj}"
        elif pay_rats:
            basei = f"{payAi} / ({payAi} + {payBi}) - 1/2"
            basej = f"{payAj} / ({payAj} + {payBj}) - 1/2"
        else:
            basei = f"{payAi} - {payBi}"
            basej = f"{payAj} - {payBj}"

        "Apply exponents to bases."
        if use_exp and not pay_expo:
            basei = f"({basei})^γ₁"
            if one_exp:
                basej = f"({basej})^γ₁"
            else:
                basej = f"({basej})^γ₂"


        "--- Avoid complex numbers: split parenthetic powers into ReLU-signed powers ---"
        "Only for the min-max family when the exponent is applied to the BASE (not payoffs)."
        def _split_parenthetic_power_to_relu(expr_with_pow: str) -> str:
            """
            Turn '(inner)^γₖ' or '[inner]^γₖ' into:
                (max(inner, 0)^γₖ - max(-(inner), 0)^γₖ)
            This preserves sign for any real γₖ>0 and never yields complex numbers.
            """
            "Parenthesis form"
            expr_with_pow = re.sub(
                r"\(\s*([^\)]+?)\s*\)\s*\^γ([₁₂₃])",
                lambda m: f"({m.group(1)}^γ{m.group(2)}" if one_pay else f"(max({m.group(1)}, 0)^γ{m.group(2)} - max(-({m.group(1)}), 0)^γ{m.group(2)})",
                expr_with_pow
            )
            "Bracket form"
            expr_with_pow = re.sub(
                r"\[\s*([^\]]+?)\s*\]\s*\^γ([₁₂₃])",
                lambda m: f"({m.group(1)}^γ{m.group(2)}" if one_pay else f"(max({m.group(1)}, 0)^γ{m.group(2)} - max(-({m.group(1)}), 0)^γ{m.group(2)})",
                expr_with_pow
            )
            return expr_with_pow

        if use_exp and not pay_expo:
            basei = _split_parenthetic_power_to_relu(basei)
            basej = _split_parenthetic_power_to_relu(basej)

        if soc_comp:
            "Rawlsian family"
            if alt_term:
                utility_ += f"Vᵢⱼ × min({basei}, {basej})"
            else:
                utility_ += f"Vᵢⱼ × max({basei}, {basej})"

        else:
            "Leontief family (min/max of WEIGHTED bases)"
            if alt_term:
                utility_ += f"min(Vᵢᵢ × ({basei}), Vᵢⱼ × ({basej}))"
            else:
                utility_ += f"max(Vᵢᵢ × ({basei}), Vᵢⱼ × ({basej}))"

        return utility_

    return _apply_gamma_to_payoffs(
        utility_ + term("self-interest") + term("altruism") + term("social_comparison"), 
        utility_settings)

