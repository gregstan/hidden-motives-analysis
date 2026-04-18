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
    Compute a single term in the utility function with lazy evaluation.

    Arguments:
        • payoff_1: float; payoff for option x (e.g., π_s^A)
        • payoff_2: float; payoff for option y (the alternative)
        • weight_1: float; weight for gain scenario (e.g., Vᵢᵢ)
        • weight_2: float; weight for loss scenario (e.g., Ʌᵢᵢ)
        • exponent: float; exponent parameter controlling curvature
        • reference_pt: float; reference point, which is 3 by default
        • use_negativity_parameters: bool; whether to use negativity parameters
        • use_exponential_parameters: bool; whether to use exponential calculations
        • reference_dependent_utility: bool; whether the utility is reference-dependent
        • single_payoffs_not_differences: bool; whether to use single payoffs or differences
        • payoff_ratios_not_differences: bool; whether to use payoff ratios, not differences

    Returns:
        • float; computed value of this term
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
    Compute the total utility for choosing option A over B.
    
    Arguments:
        • payoffs: dict; {'As':..., 'Ao':..., 'Bs':..., 'Bo':...}
        • params: dict; {'Vᵢᵢ':..., 'Ʌᵢᵢ':..., 'Vᵢⱼ':..., 'Ʌᵢⱼ':..., 'Ƹᵢⱼ':..., 
                         'Ʒᵢⱼ':..., 'γ1':..., 'γ2':..., 'γ3':...}
        • utility_settings: Dict[str, bool]; Dict of booleans controlling functional form.
        • separate_terms: If True, provides utilities for each term.

    Returns:
        • float; utility for A relative to B (positive => prefer A, negative => prefer B)
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
                # Exponent is on the base: match the string builder’s ReLU-signed rewrite
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
    Convert utilities for A and B into a probability of choosing A using softmax.
    
    Arguments:
        • uA: float; utility for choosing A over B
        • uB: float; utility for choosing B over A (or just -uA if symmetrical)
        • temperature: float; scaling factor

    Returns:
        • float; probability of choosing A
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
    Compute the agent's probability of 'A' given the current game and agent parameters.

    Arguments:
        • current_game: dict; Includes roles, payoffs, and participant responses.
        • agent_params: dict[str, float]; The agent's social preference parameter 
            set, including means and standard deviations.
            Example: {
                'Vᵢᵢ': 0.95831, 'Vᵢⱼ': 0.33333, 'Ƹᵢⱼ': 0.27374, 'Ʒᵢⱼ': 0.01629, 'γ1': 0.80022,
                'Vᵢᵢ_std': 0.5, 'Vᵢⱼ_std': 0.3, 'Ƹᵢⱼ_std': 0.1, 'Ʒᵢⱼ_std': 0.2, 'γ1_std': 0.9,
            }
        • select: bool;
            - If True, returns a binary selection (1 or 0) instead of a float probability.
            - If False, returns a float probability in [0, 1].

    Returns:
        • dict to be merged into the current game with "model_choose_A", whose value is either:
            - float in [0, 1] if select=False
            - int in {0, 1} if select=True
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
    Creates a utility function as a string.

    Arguments:
        • utility_settings: Dict[str, bool]; Dict of booleans controlling functional form.
        • option: string literal ('A' or 'B'); The option label under consideration.
    
    Returns:
        • str; The utility function.
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

"=========================================================================================="
"==================================== Shared Functions ===================================="
"=========================================================================================="

def _worker_fit_one(args: Any):
    """
    Fit parameters for a single dyad and save results to a JSON file.

    Arguments:
        • args: tuple
            A tuple containing the following:
            - dkey: str
                The unique key identifying the dyad (e.g., "(PlayerA, PlayerB)").
            - meeting_list: list[dict]
                The list of meetings (games) between the two players in the dyad.
            - experiment_num: int
                Experiment identifier for naming output files.
            - file_paths: dict
                Dictionary containing paths for saving individual dyad files.
            - param_info: ParamInfo
                Parameter configuration for the fitting process.
            - utility_settings: UtilitySettings
                Configuration options for the utility model.

    Returns:
        • str
            The dyad key (`dkey`) to indicate which dyad was processed.

    Saves:
        • JSON file
            A file containing the fitted results for the dyad.
            Saved to the path defined in `file_paths["dyad_data"]`.

    Raises:
        • Exception
            Prints a traceback and re-raises the exception if any error occurs during fitting or saving.
    """

    try:
        key, meeting_list, file_paths, param_info, utility_settings, general_settings = args
        analysis_mode = general_settings.get('analysis_mode', 'bayesian')

        if general_settings.get('analysis_unit') == "player" or general_settings.get('update_method') == 'naive':
            "Fit the player." 
            fit_params_by_player(player_uuid=key, param_info=param_info, utility_settings=utility_settings, 
                                 file_paths=file_paths, general_settings=general_settings)            
        elif analysis_mode == 'bayesian':
            print('fit the dyad')
            "Fit the dyad." 
            fit_dyad_parameters_bayes(dyad_games=meeting_list, param_info=param_info, 
                utility_settings=utility_settings, file_paths=file_paths, general_settings=general_settings)
        elif analysis_mode == 'mle':
            "Fit the dyad." 
            fit_dyad_parameters_mle(dyad_games=meeting_list, param_info=param_info, 
                utility_settings=utility_settings, file_paths=file_paths, general_settings=general_settings)
        else:
            raise ValueError(f"analysis_mode must be 'bayesian' or 'mle', not {analysis_mode}!")        

        "Return the dyad key for tracking progress."
        return key
    
    except Exception as error:
        import traceback
        traceback.print_exc()
        pass


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
            If None, we default to [-3, 3] in each dimension (arbitrary).
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
        # method attribute is present on OptimizeResult for SciPy's minimize
        if hasattr(opt_res, "method"):
            report["method"] = str(getattr(opt_res, "method"))        
        return report
        
    optimization_method = optimization_method.lower().strip()
    if optimization_method not in ('global', 'local', 'globloc'):
        raise ValueError(
            "optimization_method must be one of {'global', 'local', 'globloc'}, "
            f"but got {optimization_method!r}."
        )

    # Make sure we handle the case where local is needed but x_guesses is None
    if (optimization_method in ('local', 'globloc')) and (x_guesses is None):
        # fallback guess: random midpoint of each bound
        x_guesses = [random.uniform(lo, hi) for (lo, hi) in x_bounds]

    x_guesses = np.array(x_guesses) if x_guesses is not None else None
    x_bounds = np.array(x_bounds, dtype=float)

    if isinstance(random_seed, int) and random_seed > 0:
        random.seed(int(random_seed))
        np.random.seed(int(random_seed))

    #----------------------------------
    # 0) Keep track of durations to sum in the end
    #----------------------------------
    random_dur = 0.0
    global_dur = 0.0
    local_dur  = 0.0

    #----------------------------------
    # 1) Random Search
    #----------------------------------
    time_start_random = time.time()

    guess_candidates = []
    # Evaluate user-provided x_guesses
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

    #----------------------------------
    # 2) Global (Simulated Annealing)
    #----------------------------------
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

    #-----------------------------------
    # 3) Decide best from random vs global
    #    That best becomes the x0 for local (if local is used)
    #-----------------------------------
    best_for_local = best_guess_rand
    best_for_local_loss = best_loss_rand
    if global_opt_result is not None:
        if global_opt_result.fun < best_for_local_loss:
            best_for_local = global_opt_result.x
            best_for_local_loss = global_opt_result.fun

    def _method_options(method: str) -> dict:
        opts: dict = {}
        # common
        if isinstance(maxiter_local, int):
            opts['maxiter'] = int(maxiter_local)
        # per-method add-ons
        if method == 'L-BFGS-B':
            opts.setdefault('ftol', 1e-6)
            # no maxfun here
        elif method == 'TNC':
            opts.setdefault('ftol', 1e-6)
            if isinstance(maxfun_local, int):
                opts['maxfun'] = int(maxfun_local)
        elif method == 'SLSQP':
            opts.setdefault('ftol', 1e-6)
            # SLSQP has 'eps' if you want a step size; no maxfun
        elif method in ('Powell','Nelder-Mead'):
            if isinstance(maxiter_local, int):
                opts['maxiter'] = int(maxiter_local)
            # scipy exposes 'maxfev' for these
            if isinstance(maxfun_local, int):
                opts['maxfev'] = int(maxfun_local)
        return opts

    # --- Helper: method capabilities ---
    _methods_supporting_jac = {"L-BFGS-B", "TNC", "SLSQP"}
    _default_local_methods = tuple(local_methods) if local_methods else ("L-BFGS-B",)

    def _minimize_once(method_name: str, x0: np.ndarray) -> Optional[OptimizeResult]:
        # Per-method options, respecting your existing ftol/maxiter/maxfun knobs.

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

            # just before running local(s)
            assert best_for_local.shape == np.array(x_bounds, float)[:,0].shape, \
                "Local x0 shape mismatch vs bounds."

            return minimize(**kwargs)

    #-----------------------------------
    # 4) Possibly run local refine
    #-----------------------------------
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

        # # --- debug: print which local method won (if any) ---
        # if local_opt_result is not None:
        #     print(f"[local winner] {best_local_method}")

    if local_opt_result is not None:
        # store winning method for later inspection
        local_dict = safe_serialize(local_opt_result, local_dur)
        if local_dict is not None:
            # local_dict["chosen_local_method"] = str(getattr(local_opt_result, "method", "unknown"))
            local_dict["chosen_local_method"] = best_local_method

    #-----------------------------------
    # 5) Decide final
    #-----------------------------------
    best_loss_final = float('inf')
    best_result: Optional[OptimizeResult] = None
    best_label = None

    # If we used global
    if global_opt_result is not None:
        if global_opt_result.fun < best_loss_final:
            best_loss_final = global_opt_result.fun
            best_result = global_opt_result
            best_label = "global"

    # If local is used
    if local_opt_result is not None:
        if local_opt_result.fun < best_loss_final:
            best_loss_final = local_opt_result.fun
            best_result = local_opt_result
            best_label = "local"

    if best_result is None:
        raise RuntimeError("No optimization step was actually performed. Check method logic.")

    #-----------------------------------
    # 6) Build final dictionary
    #-----------------------------------
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
    Stage-1: global+local with a stabilizing penalty (objective_with_penalty).
    Stage-2: optional trust-constr refine on the *raw* NLL with L1 unit-norm
             constraint over social-preference parameters.

    Returns a dictionary with 'stage1', 'stage2' (if run), and 'final' summaries.
    """
    # ------------------------
    # Stage-1: global+local
    # ------------------------
    # NOTE: use your existing global_local_optimization, but allow a seed for DA.
    # To avoid changing its signature, we close over the seed inside the objective
    # by monkey-patching dual_annealing at call-site (shown below).

    # We re-use your function directly; to pass a seed into DA, set 'random_state' inside the internal call.
    # Easiest safe tweak: temporarily wrap dual_annealing with seed if provided.
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
        # --- 1) Bounds & safe start ---------------------------------------------------
        lower_bounds = np.array([lo for lo, _ in x_bounds], dtype=float)
        upper_bounds = np.array([hi for _, hi in x_bounds], dtype=float)

        # Stay strictly inside the box so finite-difference probing never steps outside.
        tiny = 1e-12
        x0 = np.asarray(x0, dtype=float).copy()
        x0 = np.minimum(np.maximum(x0, lower_bounds + tiny), upper_bounds - tiny)

        # --- 2) L1 equality constraint over the masked coordinates --------------------
        def _l1_equation(x: np.ndarray) -> float:
            return float(np.sum(np.abs(np.asarray(x, dtype=float)[unitnorm_mask])) - 1.0)

        l1_equality_constraint = NonlinearConstraint(
            fun=_l1_equation,
            lb=0.0,
            ub=0.0
            # No analytic Jacobian/Hessian: trust-constr will use finite differences.
        )

        bounds_object = Bounds(lower_bounds, upper_bounds)  # bound feasibility is enforced by the solver

        # --- 3) Solver call with robust FD and warning suppression --------------------
        start_time = time.time()
        try:
            with warnings.catch_warnings():
                # 3a) benign messages on non-smooth objectives / interior-point linear solves
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

                # 3b) Finite-difference settings that play nicely with box constraints
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
            # Soft failure: caller can keep Stage-1 solution. Do not crash the run.
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
            • Temperature ('temp')
            • Any standard deviation / covariance keys ('*_std', '*_cov')
            • All exponent parameters (γ₁/γ₂/γ₃, or ASCII 'gamma*')
            • Any other keys not explicitly recognized as weights

        Returns:
            • np.ndarray[bool]; mask aligned with `parameter_keys`, True for coords
            in the L1 equality constraint, False otherwise.
        """
        def _is_exponent_key(key: str) -> bool:
            k0 = key.strip().lower()
            # both pretty (γ1) and ASCII ('gamma1', 'gamma_1') forms
            return (k0 in {"γ1", "γ2", "γ3"}) or k0.startswith("gamma")

        def _is_std_or_cov(key: str) -> bool:
            return key.endswith("_std") or key.endswith("_cov")

        def _normalize_ascii_aliases(key: str) -> str:
            # make a lenient ASCII fallback for pretty subscripts/special chars
            return (key.replace("ᵢ", "i")
                    .replace("ⱼ", "j")
                    .replace("Ʌ", "Λ")
                    .replace("Ƹ", "E")
                    .replace("Ʒ", "G"))

        WEIGHT_SET_PRETTY = {"Vᵢᵢ", "Ʌᵢᵢ", "Vᵢⱼ", "Ʌᵢⱼ", "Ƹᵢⱼ", "Ʒᵢⱼ"}
        WEIGHT_SET_ASCII  = {"Vii", "Λii", "Vij", "Λij", "Eij", "Gij"}

        mask_values: list[bool] = []
        for key in parameter_keys:
            if key == "temp":
                mask_values.append(False); continue
            if _is_std_or_cov(key):
                mask_values.append(False); continue
            if _is_exponent_key(key):
                mask_values.append(False); continue

            # Recognize the six weights in either pretty or ASCII form
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
            # If everything is ~0, leave as-is; trust-constr will move it.
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

    # Patch in a localized way (context-like)
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

    # ------------------------
    # Stage-2: trust-constr (raw NLL under L1 unit-norm)
    # ------------------------
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

    # ------------------------
    # Decide final (by raw NLL)
    # ------------------------
    # Compute raw NLL for Stage-1 best (penalty-free) to compare apples-to-apples.
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
        "FFilter the dataframe for the selected dyad"
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
                param_dict['temp'] = softmax_temperature

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
"======================================== MLE Code ========================================"
"=========================================================================================="

def compute_std_errors_mle(best_x: NDArray[np.float64], data_rows: List[Dict[str, Any]], param_info: ParamInfo, 
                            utility_settings: UtilitySettings, penalty_weight: float) -> Dict[str,float]:
    """
    Estimate std errors from the Hessian of the MLE loss around best_x.

    We do a finite-difference Hessian, invert it, sqrt(diag). 
    If singular, use pseudo-inverse.

    Returns a dict { param_name: stdev }.
    """
    def func_wrapper(x: NDArray[np.float64]) -> float:
        return loss_function_mle(x, data_rows, param_info, utility_settings, penalty_weight)

    # numeric Hessian
    hess = gnrl.numerical_hessian(func_wrapper, best_x)  # your finite-difference approach
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
    Compute total loss across data_rows for the MLE approach.
    Each item in data_rows has "As","Ao","Bs","Bo","selection".

    We do a squared error: (p(A)-selection)**2. Then add a penalty
    to keep parameters bounded near zero (or exponent near 1).

    Arguments:
        • params_arr: NDArray[np.float64] of shape (num_params,).
        • data_rows: list of dict => the subset of data
        • param_info: includes 'keys' => param names
        • utility_settings: config for the utility function
        • penalty_weight: float => how strong the penalty is

    Returns:
        float => sum of losses across all rows
    """
    # parse param_array => param_dict
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
        pA = softmax_(uA, uB)  # your function => Probability(choose A)

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

        # Skip abdications
        if player_role == 'chooser' and meeting.get('abdicated_chooser', False):
            continue
        if player_role == 'predictor' and meeting.get('abdicated_predictor', False):
            continue

        label_str = 'choice' if player_role == 'chooser' else 'prediction'
        label_val = meeting.get(label_str)
        if label_val is None:
            continue

        # Convert 'A'=>1.0, 'B'=>0.0
        selection = 1.0 if label_val == 'A' else 0.0

        # Payoffs
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

    # Sort by round
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
    # stage counts: either 1..n for iterative or just [n] for a single final fit
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
    Actually run scipy minimize on the subset of data to find param estimates.
    Return (best_params_dict, std_errs_dict, final_loss).

    Each row in subset_data has { "As","Ao","Bs","Bo","selection", etc. }.
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

    We'll store them in:
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

        # Overwrite or store. If iterative, you might want a list of partial fits.
        # But let's just store the final stage each time.
        role_dict["params"] = item["params"]
        role_dict["std_errors"] = item["std_errors"]
        # role_dict["loss"] = item["loss"]
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
        • If a player never acts in a particular role, we skip that pair.
        • If track_evolution=True, we do partial fits (first 1 game, first 2 games, etc.) 
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
            # 1) Extract role data
            role_data = extract_one_role_data_mle(dyad_games, player_uuid, role)
            if not role_data:
                continue  # skip if no data for that role

            # 2) Fit the data (iterative or single‐shot)
            fit_results = fit_one_player_one_role_mle(role_data, param_info, utility_settings, general_settings.get('track_evolution', True))

            # 3) Store the results in the dyad_games
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


"=========================================================================================="
"===================================== Bayesian Code ======================================"
"=========================================================================================="

def bayesian_update_parametric(old_means: Dict[str, float], old_stds: Dict[str, float], observed_choice: str, game_dict: DyadGame, 
                               choice_func: callable, utility_settings: UtilitySettings, learning_rate: float = 0.4, shrink_std: bool = True, 
                               shrink_factor: float = 0.02, epsilon: float = 1e-4) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Perform a Bayesian update on social preference parameters using numeric gradients of log-likelihood.

    This function updates the means and standard deviations for each parameter in a lightweight 
    parametric Bayesian framework. Gradients of the log-likelihood are calculated for the observed 
    choice ("A" or "B"), and parameters are adjusted based on these gradients. Optionally shrinks 
    standard deviations to reflect reduced uncertainty as new data is incorporated.

    Arguments:
        • old_means: dict
            Dictionary of current parameter means, where keys represent parameters 
            (e.g., 'Vᵢᵢ', 'Vᵢⱼ', 'Ƹᵢⱼ', 'γ1').
        • old_stds: dict
            Dictionary of current parameter standard deviations, where keys are suffixed with '_std' 
            (e.g., 'Vᵢᵢ_std', 'Vᵢⱼ_std').
        • observed_choice: str
            The observed choice made by the player ('A' or 'B').
        • game_dict: dict
            Contains game details, including payoffs, player roles, and other context for the current round.
        • choice_func: callable
            Function that computes the probability of choosing "A" given a parameter dictionary. It 
            should return a dictionary with a key "model_choose_A" containing the probability.
        • utility_settings: UtilitySettings
            Configuration options for utility modeling, such as whether to include reference dependence, 
            negativity, or exponential parameters.
        • learning_rate: float, optional (default: 0.4)
            Step size for updating parameter means based on the gradient.
        • shrink_std: bool, optional (default: True)
            Whether to reduce parameter standard deviations as gradients become smaller, reflecting 
            reduced uncertainty over time.
        • shrink_factor: float, optional (default: 0.02)
            Scaling factor for reducing standard deviations.
        • epsilon: float, optional (default: 1e-4)
            Small value used for numeric gradient approximation.

    Returns:
        • Tuple[Dict[str, float], Dict[str, float]]
            - Updated means: A dictionary with the same structure as `old_means`.
            - Updated standard deviations: A dictionary with the same structure as `old_stds`.

    Notes:
        • This function calculates the log-likelihood of the observed choice based on the provided 
          utility model and parameter values.
        • Gradients are approximated numerically by perturbing each parameter in turn and computing the 
          difference in log-likelihood.
        • Negative parameter values (e.g., exponents) are avoided by enforcing lower bounds during updates.
        • This method is lightweight but assumes independent Gaussian distributions for each parameter.
    """
    "Create 'param_keys' for convenience"
    param_keys_means = [key for key in old_means if not key.endswith('_std')]

    def log_likelihood(param_dict_means: Dict[str,float]) -> float:
        "Merge param_dict_means + old_stds so choice_func ignores std keys"
        merged_params = {}
        for k_mean in param_keys_means:
            merged_params[k_mean] = param_dict_means[k_mean]
        "Keep the old std dev placeholders around:"
        for k_std in old_stds:
            merged_params[k_std] = old_stds[k_std]

        "Get p(A) from choice_func"
        choice_result = choice_func(
            current_game=game_dict,
            agent_params=merged_params,
            utility_settings=utility_settings,
            select=False
        )
        pA = choice_result['model_choose_A']
        if observed_choice == 'A':
            return math.log(max(pA, 1e-12))
        else:
            return math.log(max(1 - pA, 1e-12))

    new_means = dict(old_means)
    new_stds  = dict(old_stds)

    "For each dimension, numeric gradient:"
    for pk in param_keys_means:
        "plus"
        plus_means  = copy.deepcopy(old_means)
        plus_means[pk] += epsilon
        ll_plus  = log_likelihood(plus_means)

        "minus"
        minus_means = copy.deepcopy(old_means)
        minus_means[pk] -= epsilon
        ll_minus = log_likelihood(minus_means)

        grad_k = (ll_plus - ll_minus) / (2 * epsilon)

        "scale by old sigma => bigger uncertainty => bigger move"
        old_sigma = old_stds.get(pk + '_std', 0.5)
        step_size = learning_rate * old_sigma

        new_means[pk] = old_means[pk] + step_size * grad_k

        if 'γ' in pk:
            "Prevent exponents from going negative"
            if new_means[pk] < 0.01:
                new_means[pk] = 0.01

        if shrink_std:
            "naive shrink"
            shrink_amount = shrink_factor * abs(grad_k)
            new_sigma = old_sigma * max(0.0, 1.0 - shrink_amount)
            new_stds[pk + '_std'] = new_sigma
        else:
            new_stds[pk + '_std'] = old_sigma

    return new_means, new_stds


def bayesian_update_mcmc(old_means: Dict[str, float], old_stds: Dict[str, float], observed_choice: str, game_dict: DyadGame, 
                         choice_func: callable, utility_settings: UtilitySettings, param_info: ParamInfo, random_seed: int | None = None, 
                         chain_length: int = 300, burn_in: int = 50, thin: int = 1, proposal_sd: float = 0.35) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Approximate a posterior over the player's parameters after observing a single new choice,
    using a small Metropolis–Hastings MCMC chain. Returns (new_means, new_stds).
    
    NOTE: Depricated. Does not work. 

    Arguments:
    -----------
    old_means, old_stds: Dict[str, float]
        • Prior "mean" and "std" for each parameter (point estimate + uncertainty).
          Example keys:
              'Vᵢᵢ', 'Vᵢⱼ', 'Ƹᵢⱼ', 'Ʒᵢⱼ', 'γ1'  AND
              'Vᵢᵢ_std', 'Vᵢⱼ_std', 'Ƹᵢⱼ_std', 'Ʒᵢⱼ_std', 'γ1_std'
        • Interpreted here as defining a Normal prior for each parameter.

    observed_choice: str
        • The newly observed choice: 'A' or 'B'.

    game_dict: Dict
        • Contains payoffs, roles, etc., for the current round.

    choice_func: callable
        • A function that returns a dict with key 'model_choose_A' or 'model_predict_A' for probability of 'A'.

    utility_settings: UtilitySettings
        • Configuration for your utility function (negativity, exponent, etc.).
    
    learning_rate, shrink_std, shrink_factor, epsilon:
        • Present purely for signature compatibility with bayesian_update_parametric().
        • Not used in typical MCMC usage. We do a fixed-proposal Metropolis–Hastings below.

    Returns:
    -----------
    Tuple[Dict[str,float], Dict[str,float]]:
        (new_means, new_stds),
        where new_means has the same keys as old_means (minus any '_std' suffix),
        and new_stds has the same keys as old_stds.
    
    Notes:
    -----------
    1. In real usage, you'd typically do MCMC over all trials so far (batch) or re-run
       an entire chain each time with data[0..t], not just a single new observation.
       We show the single-trial approach for consistency with your existing agent() pipeline.

    2. We interpret old_means, old_stds as a Normal prior: p(theta_i) ~ Normal(old_means[i], old_stds[i]^2).
       Then we multiply by the likelihood from the single observed_choice.

    3. The parameter domains in your existing code suggest that some parameters have bounds,
       e.g. [-1,1] or [0.01,2]. This example enforces those bounds by rejecting proposals
       that lie outside. Real code might do fancier transformations (like log-space for exponents).
    """
    from scipy.stats import norm
    "A) Setup and random seed"
    if not isinstance(random_seed, int):
        random.seed(random_seed)
        np.random.seed(random_seed)

    required_keys = ['keys', 'guesses', 'bounds']
    if not isinstance(param_info, dict) or not all(key in param_info for key in required_keys):
        raise ValueError(f"param_info must be a dictionary with keys {required_keys}.")

    param_keys = param_info.get("keys", list(old_means.keys()))
    param_bounds = param_info.get("bounds", None)

    "Map param -> (lower_bound, upper_bound)"
    param_bounds_map = {param_key: param_bounds[idx] for idx, param_key in enumerate(param_keys)}

    def log_prior(parameter_values_dict: Dict[str, float]) -> float:
        """
        B) Prior function: log of Normal priors from old_means/old_stds
        """
        total_lp = 0.0
        for param_key in param_keys:
            if '_std' not in param_key:
                mu = old_means[param_key]
                sigma_key = param_key + '_std'
                sigma = max(old_stds.get(sigma_key, 0.5), 1e-9)  # avoid zero or negative
                x_val = parameter_values_dict[param_key]
                lprior = norm.logpdf(x_val, loc=mu, scale=sigma)
                total_lp += lprior
        return total_lp

    def log_likelihood(parameter_values_dict: Dict[str, float]) -> float:
        """
        C) Likelihood function for single observed choice
        Return log( p( observed_choice | parameter_values ) ).
        """
        # Probability that the model chooses 'A' under these parameters
        choice_output: dict = choice_func(
            current_game=game_dict,
            agent_params=parameter_values_dict,
            utility_settings=utility_settings,
            select=False
        )

        probability_predict_A = choice_output.get('model_choose_A', None)
        if probability_predict_A is None:
            raise Exception("choice function failed to produce a response.")

        if observed_choice == 'A':
            return math.log(max(probability_predict_A, 1e-12))
        else:
            return math.log(max(1.0 - probability_predict_A, 1e-12))

    def log_posterior(parameter_values_dict: Dict[str, float]) -> float:
        """
        Posterior ~ Prior * Likelihood, in log space => log_prior + log_likelihood.
        Return -inf if out of bounds.
        """
        "Respect param-specific bounds"
        for param_key in param_keys:
            val = parameter_values_dict[param_key]
            lb, ub = param_bounds_map[param_key]
            if val < lb or val > ub:
                # print(f"Param {param_key} not ({lb} < {round(val, 6)} < {ub})")
                return np.clip(a=val, a_min=lb, a_max=ub)
                return -float('inf')

        return log_prior(parameter_values_dict) + log_likelihood(parameter_values_dict)

    def log_posterior_(parameter_values_dict: Dict[str, float]) -> float:
        """
        Posterior ~ Prior * Likelihood, in log space => log_prior + log_likelihood.
        Includes Jacobian adjustments for parameter transformations. #TODO Use this or delete it.
        """
        log_jacobian = 0.0
        transformed_params = {}

        for param_key in param_keys:
            val = parameter_values_dict[param_key]
            lb, ub = param_bounds_map[param_key]

            # Apply transformations and calculate Jacobian adjustments
            if lb > -np.inf and ub < np.inf:
                # [a,b] range: scaled logit transform
                scaled = (val - lb) / (ub - lb)
                if scaled <= 0 or scaled >= 1:
                    return -np.inf
                transformed = np.log(scaled / (1 - scaled))  # logit
                log_jacobian += np.log((ub - lb) * scaled * (1 - scaled))
                
            elif lb > -np.inf and ub == np.inf:
                # [a,∞) range: log transform
                shifted = val - lb
                if shifted <= 0:
                    return -np.inf
                transformed = np.log(shifted)
                log_jacobian += transformed  # Jacobian: dx/dt = e^t
                
            elif lb == -np.inf and ub < np.inf:
                # (-∞,b] range: reflected log transform
                shifted = ub - val
                if shifted <= 0:
                    return -np.inf
                transformed = np.log(shifted)
                log_jacobian += transformed
                
            else:
                # Unbounded parameters
                transformed = val
                log_jacobian += 0.0

            transformed_params[param_key] = transformed

        # Calculate prior in transformed space (with Jacobian adjustment)
        log_prior_val = log_prior(parameter_values_dict)  # Original prior
        log_likelihood_val = log_likelihood(parameter_values_dict)
        
        return log_prior_val + log_likelihood_val + log_jacobian

    "D) Prepare MCMC chain. Start from old_means, clamped to param_info bounds"
    current_params = {**copy.deepcopy(old_means), **copy.deepcopy(old_stds)}
    for param_key in param_keys:
        low_bd, high_bd = param_bounds_map[param_key]
        current_params[param_key] = max(low_bd, min(high_bd, current_params[param_key]))

    current_log_post = log_posterior(current_params)

    if np.isnan(current_log_post):
        print("NaN detected in log posterior!")
        return old_means, old_stds  

    samples_chain = []
    accepted_count = 0

    "E) Metropolis–Hastings random-walk"
    for idx in range(chain_length):
        proposal_dict = copy.deepcopy(current_params)
        # random-walk step in each parameter
        for param_key in param_keys:
            step = random.gauss(0.0, proposal_sd)
            proposal_dict[param_key] += step

        proposal_log_post = log_posterior(proposal_dict)
        acceptance_log_ratio = proposal_log_post - current_log_post

        "Accept or reject"
        if math.log(random.random()) < acceptance_log_ratio:
            current_params = proposal_dict
            current_log_post = proposal_log_post
            accepted_count += 1

        samples_chain.append(copy.deepcopy(current_params))

    acceptance_rate = accepted_count / float(chain_length)
    # print(f"MCMC acceptance rate: {acceptance_rate:.2f}")

    "F) Convert chain to arrays, discard burn-in, compute mean & std"
    valid_chain = samples_chain[burn_in::thin]
    param_vectors = []
    for sample_dict in valid_chain:
        param_vectors.append([sample_dict[param_key] for param_key in param_keys])
    chain_matrix = np.array(param_vectors)  # shape: [#samples, #params]

    posterior_means_array = chain_matrix.mean(axis=0)
    posterior_stds_array = chain_matrix.std(axis=0)

    "Convert back to dict"
    new_means, new_stds = {}, {}
    for param_index, param_key in enumerate(param_keys):
        if '_std' in param_key:
            "Let std dev float freely, just ensure it's >= 0"
            new_stds[param_key] = float(max(posterior_stds_array[param_index], 0.0))
        else:
            new_means[param_key] = float(posterior_means_array[param_index])

    "G) Return final posterior dicts"
    return new_means, new_stds


def param_vector_to_pmf_array(param_vectors: Dict[Tuple[int], float],
                              meta_data: Dict[str, Dict[str, Any]],
                              general_settings: GeneralSettings,
                              use_fallback: bool = False) -> NDArray[np.float64]:
    """
    Generates a PMF from a dictionary mapping parameter coordinates to probabilities.
    PF-aware: if meta_data['representation'] == 'particles', we DO NOT interpolate.
    """
    grid_shape = tuple(meta_data["n_bins_per_dimension"] for _ in meta_data["tickvals"].keys())

    if general_settings.get('use_particle_filter', False):
        full_grid = np.zeros(grid_shape, dtype=float)
        for idx_tuple, prob in param_vectors.items():
            if prob > 0:
                full_grid[idx_tuple] += prob
        grid_sum = full_grid.sum()
        if grid_sum > 0:
            full_grid /= grid_sum
        return full_grid

    "Determine the dimensions and create an empty grid"
    full_grid = np.full(fill_value=np.nan, shape=grid_shape)

    "Fill the grid with probabilities from param_vectors"
    for param_vector, probability in param_vectors.items():
        indices = tuple(param_vector)
        if probability < 0:
            probability = 0
        full_grid[indices] = probability

    "Check if grid has any valid data"
    if np.isnan(full_grid).all():
        warning_str = "Warning: All values in param_vectors are NaN."
        print(param_vectors)
        if use_fallback:
            print(warning_str)
            return np.zeros_like(full_grid)
        else:
            raise Exception(warning_str)

    "Only interpolate for the interpolation strategy, not PF"
    if meta_data.get("sample_ratio", 1.0) < 1.0 and meta_data.get("representation", "grid") == "grid":
        n_dimensions = len(grid_shape)
        interp_method = "cubic" if n_dimensions <= 2 else "linear"
        full_grid = gnrl.fill_holes_nd(input_array=full_grid, output_shape=None, method=interp_method)

    full_grid /= full_grid.sum()
    return full_grid


def prior_grid_from_params(param_vals: Dict[str, Dict[str, Dict[str, float]]], param_info: ParamInfo, n_bins_per_dimension: int, 
                           sample_ratio: float = 0.5, covariation_matrix: CovMatDict = None, trust_inputs: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Generates a discrete joint-Gaussian pmf to act as an agent's prior, for each player and 
    role. We randomly sample from a grid of size (n_bins_per_dimension ^ n non-std parameters).

    Then we compute the multivariate normal pdf at each sampled point, possibly 
    incorporate a small 'volume element', and normalize so that it sums to 1.

    Arguments:
        • param_vals: Dict[str, Dict[str, Dict[str, float]]]; Mean and standard deviation parameters for each player in each role.
        • param_info: ParamInfo; Information about parameters used throughout this analysis. Keys: 'keys', 'bounds', and 'guesses'.
        • n_bins_per_dimension: int; The length of each array of tick values for each dimension (all the same length).
        • sample_ratio: float; Determines the number of samples for each dimension. Falls within [0, 1].
        • covariation_matrix: Dict[str, Dict[str, NDArray[np.float64]]]; Covariance matrices for each player and role. 
            - Example {
                "player_1": {
                    "chooser":   np.array([[0.0100, 0.0020, 0.0010], [0.0020, 0.0400, 0.0030], [0.0010, 0.0030, 0.0220]]),
                    "predictor": np.array([[0.0625, 0.0100, 0.0050], [0.0100, 0.0100, 0.0020], [0.0050, 0.0020, 0.0400]])
                },
                "player_2": {
                    "chooser":   np.array([[0.0400, 0.0030, 0.0020], [0.0030, 0.0225, 0.0010], [0.0020, 0.0010, 0.0100]]),
                    "predictor": np.array([[0.0900, 0.0200, 0.0100], [0.0200, 0.0400, 0.0050], [0.0100, 0.0050, 0.0600]])
                }
            }

    Returns:
        • grid_prior: Dict[str, Dict[str, Any]]; Maps randomly sampled parameter vectors to prior probabilities. 
            Example: {
                'meta_data': {
                    'sample_ratio': 0.25,
                    'n_bins_per_dimension': 41,
                    'tickvals': {
                        'Vᵢⱼ': [-1.0, -0.8, -0.6, -0.4, -0.2,  0.0,  0.2,  0.4,  0.6,  0.8,  1.0],
                        'Vᵢᵢ': [-1.0, -0.8, -0.6, -0.4, -0.2,  0.0,  0.2,  0.4,  0.6,  0.8,  1.0],
                        'Ƹᵢⱼ': [-1.0, -0.8, -0.6, -0.4, -0.2,  0.0,  0.2,  0.4,  0.6,  0.8,  1.0]
                    }
                },
                player_uuid: {
                    player_role: {
                        'param_vectors': {
                            (0, 9, 2): 0.87,
                            (7, 3, 4): 0.05,
                            (6, 0, 1): 0.73,
                            (3, 5, 8): 0.11,...
                        }                             
                    },...
                },...
            }    
    """
    if not trust_inputs:
        if not (0 <= sample_ratio <= 1):
            raise ValueError(f"sample_ratio({sample_ratio}) must be between 0 and 1.")

        if not (3 <= n_bins_per_dimension <= 201):
            raise ValueError(f"n_bins_per_dimension({n_bins_per_dimension}) must be between 3 and 201.")

        if covariation_matrix is not None and not isinstance(covariation_matrix, dict):
            raise Exception(f"covariation_matrix should be a dictionary not {type(covariation_matrix)}: {covariation_matrix}.")

        param_val_err = "param_vals must be a dictionary containing parameter values for each player in each role."

        "Ensure param_vals is a dictionary"
        if not isinstance(param_vals, dict):
            raise ValueError(f"{param_val_err} Expected a dictionary, but got: {type(param_vals)}")

        "Validate each player UUID and their roles"
        for player_uuid, roles in param_vals.items():
            if not isinstance(roles, dict):
                raise ValueError(f"{param_val_err} Each player must have a dictionary of roles. Invalid entry for player {player_uuid}: {roles}")

            "Validate each role"
            for role, params in roles.items():
                if params is None:
                    continue

                "Check parameter keys"
                param_keys_in_params = [param_key for param_key in params.keys() if '_cov' not in param_key and param_key != 'temp']
                if sorted(param_keys_in_params) != sorted(param_info["keys"]):
                    raise ValueError(
                        f"Parameter keys mismatch in player {player_uuid}, role {role}. "
                        f"Expected: {param_info['keys']}, but got: {param_keys_in_params}"
                    )

                "Validate parameter values and bounds"
                for idx, key in enumerate(param_info["keys"]):
                    value = params[key]
                    lower_bound, upper_bound = param_info["bounds"][idx]

                    "Ensure value is numeric"
                    if not isinstance(value, (int, float)):
                        raise ValueError(
                            f"Invalid type for parameter '{key}' in player {player_uuid}, role {role}. "
                            f"Expected numeric type, but got: {type(value)}"
                        )

                    "Ensure value is within bounds"
                    if not (lower_bound <= value <= upper_bound):
                        raise ValueError(
                            f"Parameter '{key}' in player {player_uuid}, role {role} is out of bounds. "
                            f"Value: {value}, Expected bounds: [{lower_bound}, {upper_bound}]"
                        )

    "Generate tick values for each parameter"
    tickvals = {
        param_key: np.linspace(
            start=param_info['bounds'][idx][0], 
            stop=param_info['bounds'][idx][1], 
            num=n_bins_per_dimension
        ) 
        for idx, param_key in enumerate(param_info['keys']) if '_std' not in param_key
    }

    "Store everything in a big dictionary"
    grid_prior = {
        'meta_data': {
            'n_bins_per_dimension': n_bins_per_dimension,
            'sample_ratio': sample_ratio,
            'tickvals': tickvals
        }
    }

    "Precompute the 'volume element' for each dimension (the spacing along that dimension)."
    spacing = {}
    for idx, param_key in enumerate(param_info["keys"]):
        if param_key.endswith('_std'):
            continue
        lb, ub = param_info["bounds"][idx]
        if n_bins_per_dimension > 1:
            spacing[param_key] = (ub - lb) / (n_bins_per_dimension - 1)
        else:
            spacing[param_key] = (ub - lb)  # fallback if n_bins_per_dimension=1

    "For each player and role, build a pmf"
    for player_uuid, roles_dict in param_vals.items():
        grid_prior[player_uuid] = {}
        for role_name, params_dict in roles_dict.items():
            if params_dict is None:
                continue
            "1) Extract means & standard deviations for each relevant param"
            means, stds = [], []
            these_param_keys = []
            for idx, param_key in enumerate(param_info["keys"]):
                if param_key.endswith('_std'):
                    continue
                mu = params_dict[param_key]
                sigma_key = param_key + '_std'
                sigma_val = params_dict[sigma_key]
                these_param_keys.append(param_key)
                stds.append(sigma_val)
                means.append(mu)

            "2) Validate and correct the covariance matrix"
            if covariation_matrix is not None and covariation_matrix.get(player_uuid, {}).get(role_name) is not None:  
                cov_matrix = covariation_matrix[player_uuid][role_name]
                # cov_matrix = gnrl.validate_covariance_matrix(cov_matrix, name=f"{player_uuid} - {role_name}")
                if not gnrl.is_positive_semidefinite(matrix=cov_matrix, tol=1e-12):
                    cov_matrix = gnrl.nearest_psd_matrix(matrix=cov_matrix, min_eigval=0.0)

                if cov_matrix.shape != (len(means), len(means)):
                    err_str = f"Cov matrix for {player_uuid} role={role_name} has shape {cov_matrix.shape}"
                    raise ValueError(f"{err_str}, expected {(len(means), len(means))}.")

            else:
                "Default to diagonal matrix."
                cov_matrix = np.diag(np.square(stds)) 

            "3) Create scipy's multivariate normal"
            rv = multivariate_normal(mean=means, cov=cov_matrix, allow_singular=False)

            "4) Sample from the full grid: n_bins_per_dimension^d total points."
            full_axes = [tickvals[param_key] for param_key in these_param_keys]  # each is length n_bins_per_dimension
            mesh = np.meshgrid(*full_axes, indexing='ij')  

            "Flatten them so we have all_points: shape=(n_bins_per_dimension^d, d)"
            all_points = np.stack([m.flatten() for m in mesh], axis=-1)

            total_grid_size = all_points.shape[0]  # n_bins_per_dimension^d
            "Pick a random subset of size (samples_per_dimension^d)."
            desired_sample_size = min(int(sample_ratio * total_grid_size), total_grid_size)

            volume_elem = 1.0
            for param_key in these_param_keys:
                volume_elem *= spacing[param_key]

            # time_start = time.time()
            "6) Convert each point in 'sampled_points' back to a discrete 'index'."
            if sample_ratio == 1.0: 

                # Use the full grid
                sampled_points = all_points  # Shape: (n_bins_per_dimension^d, d)
                
                # Precompute PDF values for all grid points
                pmf_values = rv.pdf(sampled_points)  # Compute the PDF for all grid points at once
                
                # Precompute indices for all points in the full grid
                indices = np.stack([
                    np.searchsorted(tickvals[param_key], sampled_points[:, dim_i])
                    for dim_i, param_key in enumerate(these_param_keys)
                ], axis=-1)  # Shape: (n_bins_per_dimension^d, d)
                
                # Construct param_vectors dictionary efficiently
                param_vectors = {}
                for idx_tuple, pmf_val in zip(map(tuple, indices), pmf_values):
                    # Accumulate probabilities for each index tuple
                    param_vectors[idx_tuple] = param_vectors.get(idx_tuple, 0.0) + pmf_val * volume_elem
            else:
                sample_indices = random.sample(range(total_grid_size), desired_sample_size)
                sampled_points = all_points[sample_indices, :]  # shape=(desired_sample_size, d)

                "5) Evaluate the pdf at each sampled point"
                pdf_values = rv.pdf(sampled_points)

                pmf_values = pdf_values * volume_elem

                indices = np.stack([
                    np.searchsorted(tickvals[param_key], sampled_points[:, dim_i])
                    for dim_i, param_key in enumerate(these_param_keys)
                ], axis=-1)   

                param_vectors = {}
                for irow, idx_tuple in enumerate(map(tuple, indices)):
                    pmf_val = pmf_values[irow]
                    current_val = param_vectors.get(idx_tuple, 0.0)
                    param_vectors[idx_tuple] = current_val + pmf_val

            # time_stop = time.time()
            # duration = time_stop - time_start

            "7) Normalize so the total sum is 1"
            total_mass = sum(param_vectors.values())
            if total_mass > 0:
                for idx_tuple in param_vectors:
                    param_vectors[idx_tuple] /= total_mass
            else:
                n_bins = len(param_vectors)
                for idx_tuple in param_vectors:
                    param_vectors[idx_tuple] = 1.0 / n_bins if n_bins > 0 else 0.0

            # pp.pprint(param_vectors)
            # print(f"Duration: {duration}")
            # print(f"Total Mass: {sum(param_vectors.values())}")
            # exit()    

            grid_prior[player_uuid][role_name] = {
                'param_vectors': param_vectors,
            }

    return grid_prior


def bayesian_update_grid(prior_array: NDArray[np.float64] | dict[tuple[int, ...], float], meta_data: dict[str, Any], game_dict: dict[str, Any], choice_func: callable, 
                         utility_settings: dict[str, bool], general_settings: GeneralSettings, param_info: dict[str, Any], softmax_temperature: float, no_memory_mode: bool = False) -> dict[str, Any]:
    """
    Performs a single Bayesian update over a discretized parameter grid.

    Behavior by sampling regime
    ---------------------------
    • Full-grid update (sample_ratio == 1.0)
        – Evaluate likelihood on *every* grid bin and multiply by prior.
        – Input prior may be a dense ndarray; if a sparse dict is passed, it is densified once.

    • Uniform subsample (sample_ratio < 1.0 and use_particle_filter == False)
        – Draw a uniform subset of grid bins; multiply by prior on those bins.
        – Input prior is treated as dense ndarray (sparse dict is densified once).

    • Particle filter (sample_ratio < 1.0 and use_particle_filter == True)
        – Maintain a persistent set of particles in bin-index space; store state in meta_data['pf_state'].
        – Initialize from *either* a dense PMF or a sparse dict of masses (no densification).
        – At each update:
            1) compute likelihood **once per unique bin** (major speedup),
            2) update particle weights and normalize,
            3) resample if ESS drops below a threshold (systematic resampling),
            4) (optional) jitter in bin space (defaults to 0.0 for speed),
            5) return a sparse posterior mass map plus the updated pf_state.
        – meta_data['representation'] is set to 'particles' so downstream code can skip densification.

    Arguments
    ---------
    prior_array : np.ndarray | dict
        Either a dense PMF array over the active parameter grid, or a sparse
        dict mapping multi-indices to probability mass. When the PF branch is used,
        the sparse dict is kept sparse (no densification).

    meta_data : dict
        Must include:
          - 'n_bins_per_dimension' : int
          - 'tickvals' : dict[str, list[float]]  (bin locations for each active mean parameter)
          - 'sample_ratio' : float in (0,1] controlling grid subsampling or particle count scaling
        May include PF knobs:
          - 'use_particle_filter' (bool, default True when sample_ratio<1)
          - 'pf_max_particles' (int, cap on particle count; default 5000)
          - 'pf_min_particles' (int, floor on particle count; default 200)
          - 'pf_resample_fraction' (float in (0,1], ESS/N threshold; default 0.5)
          - 'pf_jitter_sd' (float, std-dev of Gaussian jitter in *bin units*; default 0.0)
          - 'pf_rng_seed' (int | None)
          - persistent 'pf_state' from previous updates:
                {'indices': np.ndarray[int, shape=(N,d)], 'weights': np.ndarray[float, shape=(N,)]}
          - optional 'representation' : 'grid' | 'particles'

    game_dict : dict
        A single game payload containing payoffs and the observed choice ('A' or 'B').

    choice_func : callable
        The function that returns p(A) given params and game:
            choice_func(current_game, agent_params, utility_settings, softmax_temperature, select=False)
        Must return a dict with key 'model_choose_A' in [0,1].

    utility_settings : dict
        Boolean flags defining the functional form of utility.

    param_info : dict
        Holds parameter keys, bounds, and guesses — only the **mean** keys are active on the grid.

    softmax_temperature : float
        Temperature used for likelihood evaluation.

    no_memory_mode : bool
        If True, the posterior ignores the prior and uses likelihood only.

    Returns
    -------
    dict with keys:
        'param_vectors' : dict[(i1,...,id) -> mass]  # sparse posterior mass on grid bins
        'meta_data'     : dict                       # updated meta_data including pf_state and 'representation'
    """
    # ----- Toggle particle filter path when subsampling -----
    use_particle_filter: bool = bool(meta_data.get("use_particle_filter", True))

    # Active mean-parameter keys; grid size info
    param_mean_keys = [k for k in param_info["keys"] if not k.endswith("_std")]
    n_bins_per_dimension: int = int(meta_data["n_bins_per_dimension"])
    tickvals: dict[str, list[float]] = meta_data["tickvals"]
    sample_ratio: float = float(meta_data["sample_ratio"])

    n_dims = len(param_mean_keys)
    total_grid_size = n_bins_per_dimension ** n_dims

    # ---------------------------- FULL GRID (unchanged) ----------------------------
    if sample_ratio == 1.0:
        if isinstance(prior_array, dict):
            # Densify once (only for full-grid case)
            prior_array = param_vector_to_pmf_array(param_vectors=prior_array, meta_data=meta_data, general_settings=general_settings)

        full_axes = [tickvals[pk] for pk in param_mean_keys]
        mesh = np.meshgrid(*full_axes, indexing='ij')
        all_points = np.stack([m.flatten() for m in mesh], axis=-1)
        prior_flat = prior_array.flatten().astype(float)

        # Bin indices for each point
        indices = np.stack([
            np.searchsorted(tickvals[param_key], all_points[:, d])
            for d, param_key in enumerate(param_mean_keys)
        ], axis=-1)

        # Evaluate likelihood once per point
        likelihoods = np.empty(all_points.shape[0], dtype=float)
        obs_is_A = (game_dict['choice'] == 'A')
        for row_idx in range(all_points.shape[0]):
            agent_params = {param_mean_keys[d]: float(all_points[row_idx, d]) for d in range(n_dims)}
            pA = choice_func(current_game=game_dict,
                             agent_params=agent_params,
                             utility_settings=utility_settings,
                             softmax_temperature=softmax_temperature,
                             select=False)['model_choose_A']
            likelihoods[row_idx] = pA if obs_is_A else (1.0 - pA)

        # Posterior (full-grid)
        posterior_probs = likelihoods if no_memory_mode else (prior_flat * likelihoods)

        # Accumulate into sparse map keyed by multi-index
        posterior_param_vectors: dict[tuple[int, ...], float] = {}
        for idx_tuple, mass in zip(map(tuple, indices), posterior_probs):
            posterior_param_vectors[idx_tuple] = posterior_param_vectors.get(idx_tuple, 0.0) + float(mass)

        # Normalize
        s = sum(posterior_param_vectors.values())
        if s > 0:
            for k in posterior_param_vectors:
                posterior_param_vectors[k] /= s

        return {'param_vectors': posterior_param_vectors, 'meta_data': meta_data}

    # ----------------------- UNIFORM SUBSAMPLE (unchanged) ------------------------
    if not use_particle_filter:
        if isinstance(prior_array, dict):
            prior_array = param_vector_to_pmf_array(param_vectors=prior_array, meta_data=meta_data, general_settings=general_settings)

        desired_sample_size = max(1, min(int(sample_ratio * total_grid_size), total_grid_size))
        full_axes = [tickvals[pk] for pk in param_mean_keys]
        mesh = np.meshgrid(*full_axes, indexing='ij')
        all_points = np.stack([m.flatten() for m in mesh], axis=-1)

        sample_indices = random.sample(range(total_grid_size), desired_sample_size)
        sampled_points = all_points[sample_indices, :]
        sampled_prior_probs = prior_array.flatten()[sample_indices]

        indices = np.stack([
            np.searchsorted(tickvals[param_key], sampled_points[:, d])
            for d, param_key in enumerate(param_mean_keys)
        ], axis=-1)

        obs_is_A = (game_dict['choice'] == 'A')
        likelihoods = np.empty(sampled_points.shape[0], dtype=float)
        for row_idx in range(sampled_points.shape[0]):
            agent_params = {param_mean_keys[d]: float(sampled_points[row_idx, d]) for d in range(n_dims)}
            pA = choice_func(current_game=game_dict,
                             agent_params=agent_params,
                             utility_settings=utility_settings,
                             softmax_temperature=softmax_temperature,
                             select=False)['model_choose_A']
            likelihoods[row_idx] = pA if obs_is_A else (1.0 - pA)

        posterior_probs = likelihoods if no_memory_mode else (sampled_prior_probs * likelihoods)

        posterior_param_vectors: dict[tuple[int, ...], float] = {}
        for idx_tuple, mass in zip(map(tuple, indices), posterior_probs):
            posterior_param_vectors[idx_tuple] = posterior_param_vectors.get(idx_tuple, 0.0) + float(mass)

        s = sum(posterior_param_vectors.values())
        if s > 0:
            for k in posterior_param_vectors:
                posterior_param_vectors[k] /= s

        return {'param_vectors': posterior_param_vectors, 'meta_data': meta_data}

    # --------------------------- PARTICLE FILTER (FAST) ---------------------------
    # Particle budget (cap/floor), ESS threshold, and jitter
    pf_max_particles = int(meta_data.get("pf_max_particles", 5000))
    pf_min_particles = int(meta_data.get("pf_min_particles", 200))
    pf_resample_frac = float(meta_data.get("pf_resample_fraction", 0.5))
    pf_jitter_sd     = float(meta_data.get("pf_jitter_sd", 0.0))
    rng_seed         = meta_data.get("pf_rng_seed", None)
    rng = np.random.default_rng(rng_seed)

    # Interpret sample_ratio as an upper bound on the particle budget, capped by pf_max_particles
    target_particles = min(int(round(sample_ratio * total_grid_size)), pf_max_particles)
    N = max(pf_min_particles, target_particles, 1)

    # Build per-dimension tick arrays once
    ticks_list = [np.asarray(tickvals[pk], dtype=float) for pk in param_mean_keys]

    def params_for_indices(idx_mat: np.ndarray) -> list[dict[str, float]]:
        # idx_mat shape: (K, n_dims); return K param dicts
        out: list[dict[str, float]] = []
        # Grab values per dim, vectorized
        vals_per_dim = [ticks_list[d][idx_mat[:, d]] for d in range(n_dims)]
        for k in range(idx_mat.shape[0]):
            out.append({param_mean_keys[d]: float(vals_per_dim[d][k]) for d in range(n_dims)})
        return out

    # Fetch persistent PF state or initialize
    pf_state = meta_data.get("pf_state", None)

    if pf_state is None:
        # Initialize from prior WITHOUT densifying:
        #   - if sparse dict: sample keys by their mass
        #   - else dense array: sample flat indices by pmf
        if isinstance(prior_array, dict):
            prior_keys = np.array(list(prior_array.keys()), dtype=int)          # (M, d)
            prior_wts  = np.array([max(0.0, float(v)) for v in prior_array.values()], dtype=float)
            s = float(prior_wts.sum())
            if s <= 0:
                prior_wts = np.full(prior_keys.shape[0], 1.0 / max(1, prior_keys.shape[0]))
            else:
                prior_wts /= s
            ancestor_rows = rng.choice(prior_keys.shape[0], size=N, replace=True, p=prior_wts)
            indices = prior_keys[ancestor_rows]
            weights = np.full(N, 1.0 / N, dtype=float)
        else:
            flat = np.asarray(prior_array, dtype=float).ravel()
            s = float(flat.sum())
            if s <= 0:
                flat = np.full_like(flat, 1.0 / max(1, flat.size), dtype=float)
            else:
                flat /= s
            flat_idx = rng.choice(flat.size, size=N, replace=True, p=flat)
            indices = np.column_stack(np.unravel_index(flat_idx, (n_bins_per_dimension,) * n_dims)).astype(int)
            weights = np.full(N, 1.0 / N, dtype=float)
    else:
        indices = np.asarray(pf_state["indices"], dtype=int)
        weights = np.asarray(pf_state["weights"], dtype=float)
        if indices.shape[0] != weights.shape[0]:
            raise ValueError("pf_state malformed: indices and weights have different lengths.")
        # If the requested N changes, resample to match N
        if indices.shape[0] != N:
            csum = np.cumsum(weights)
            positions = (rng.random() + np.arange(N)) / N
            sel = np.searchsorted(csum, positions, side="left")
            indices = indices[sel]
            weights = np.full(N, 1.0 / N, dtype=float)

    # -- Compute per-particle likelihood for this game (unique-eval to avoid rework)
    obs_A = (game_dict['choice'] == 'A')

    # Unique rows of indices; inverse maps back to the full N
    unique_indices, inverse_map = np.unique(indices, axis=0, return_inverse=True)
    n_unique = unique_indices.shape[0]

    like_unique = np.empty(n_unique, dtype=float)
    # Build agent params only for unique particles
    unique_params_list = params_for_indices(unique_indices)

    for u in range(n_unique):
        choice_result = choice_func(
            current_game=game_dict,
            agent_params=unique_params_list[u],
            utility_settings=utility_settings,
            softmax_temperature=softmax_temperature,
            select=False
        )['model_choose_A']
        like_unique[u] = choice_result if obs_A else (1.0 - choice_result)

    # Broadcast back to all N particles
    like = like_unique[inverse_map]

    # Weight update (log-stable); if no_memory_mode → likelihood-only
    if no_memory_mode:
        weights = like
    else:
        lw = np.log(weights + 1e-300) + np.log(like + 1e-300)
        lw -= lw.max()
        weights = np.exp(lw)

    # Normalize / fallback uniform
    s = float(weights.sum())
    if not np.isfinite(s) or s <= 0.0:
        weights = np.full(N, 1.0 / N, dtype=float)
    else:
        weights /= s

    # ESS-based resampling
    ess = 1.0 / np.sum(weights ** 2)
    if ess < pf_resample_frac * N:
        csum = np.cumsum(weights)
        positions = (rng.random() + np.arange(N)) / N
        sel = np.searchsorted(csum, positions, side="left")
        indices = indices[sel]
        weights.fill(1.0 / N)

        # Optional jitter in *bin space* (default 0.0 → fast; set >0 to explore)
        if pf_jitter_sd > 0.0:
            noise = rng.normal(0.0, pf_jitter_sd, size=indices.shape)
            jittered = np.rint(indices.astype(float) + noise).astype(int)
            max_idx = n_bins_per_dimension - 1
            # reflect at boundaries
            jittered = np.where(jittered < 0, -jittered, jittered)
            over = jittered > max_idx
            jittered[over] = 2 * max_idx - jittered[over]
            indices = np.clip(jittered, 0, max_idx)

    # Build sparse posterior map by summing weights for identical bins
    uniq_bins, inv2 = np.unique(indices, axis=0, return_inverse=True)
    mass_per_bin = np.zeros(uniq_bins.shape[0], dtype=float)
    np.add.at(mass_per_bin, inv2, weights)

    posterior_param_vectors: dict[tuple[int, ...], float] = {
        tuple(row): float(mass) for row, mass in zip(map(tuple, uniq_bins), mass_per_bin)
    }
    # Normalize defensively
    s = sum(posterior_param_vectors.values())
    if s > 0:
        for k in posterior_param_vectors:
            posterior_param_vectors[k] /= s

    # Persist PF state and mark representation as 'particles' so we never densify downstream
    new_meta = dict(meta_data)
    new_meta["pf_state"] = {"indices": indices, "weights": weights}
    new_meta["representation"] = "particles"

    return {'param_vectors': posterior_param_vectors, 'meta_data': new_meta}


def agent(dyad_games: DyadGames, game_idx_start: int, game_idx_stop: int, general_settings: GeneralSettings, 
          initial_params: Dict[str, Dict[str, float]], param_info: ParamInfo, utility_settings: UtilitySettings, 
          player_uuid: str | None = None, player_role: str | None = None, select: bool = False, choice_temperature: float | None = None) -> List[dict]:
    """
    Produce and store responses over a series of games while updating and storing social preference parameters. 

    Arguments:
        • dyad_games: List[Dict[str, Any]]; List of games that store all data.
        • game_idx_start/game_idx_stop: int; Index of game to begin and finish playing on.
        • general_settings: GeneralSettings; Miscellaneous settings used throughout this file.
        • initial_params: Dict[str, Dict[str, float]]; Parameters provided for the first game.
        • param_info: ParamInfo; Contains parameter keys, bounds, and initial guesses.
        • utility_settings: Dict[str, bool]; Settings that determine the utility function.
        • player_uuid: str = None; Player identifier, like "44598db2-2243-45c1-8ba8-5a8ebaa0b042".
        • player_role: str = None; If 'chooser' or 'predictor', agent playes just games where the player is assigned to that role. 
        • select: bool = False; If True, the choice function generates actual responses, not response probabilities.
        • choice_temperature: float = None; Used in SoftMax to create variance in the choice probability.
        
    Returns:
        • The 'dyad_games' list with 'parameter_estimates' updated.
    """
    # === 1) Basic Setup ===
    num_meetings = len(dyad_games)
    if not isinstance(game_idx_start, int) or game_idx_start < 0:
        game_idx_start = 0
    if not isinstance(game_idx_stop, int) or game_idx_stop >= num_meetings:
        game_idx_stop = num_meetings - 1
    if game_idx_start > game_idx_stop:
        game_idx_start = game_idx_stop
    
    # Extract settings
    sample_ratio = general_settings.get('sample_ratio', True)
    learning_rate = general_settings.get('learning_rate', True)
    update_method = general_settings.get('update_method', True)
    include_covariance = general_settings.get('include_covariance', True)
    n_bins_per_dimension = general_settings.get('n_bins_per_dimension', True)
    softmax_temperature = general_settings.get('softmax_temperature', True)

    # Possibly override if there's a param-based temperature
    if (not general_settings.get('temperature_is_param', False)
        or not (isinstance(choice_temperature, (int, float)) and 0 < choice_temperature <= 3)):
        choice_temperature = softmax_temperature

    # === 2) Main Loop Over Games ===
    idx = game_idx_start
    while idx <= game_idx_stop and idx < num_meetings:
        game_dict = dyad_games[idx]
        
        # 2a) Figure out which role this player actually occupies in *this* game
        actual_game_role = None
        if game_dict.get('chooser') == player_uuid:
            actual_game_role = 'chooser'
        elif game_dict.get('predictor') == player_uuid:
            actual_game_role = 'predictor'
        else:
            print(f"Game {idx} player_uuid {player_uuid}")
            game_copy = copy.deepcopy(game_dict)
            del game_copy['parameter_estimates']
            pp.pprint(game_copy)
            raise Exception(f"No role found in game {idx} for player {player_uuid}")

        # assigned_role is what the caller wants us to run
        assigned_role = player_role

        # 2b) Create or get the sub-dicts where we'll store parameter estimates
        param_estimates = game_dict.setdefault('parameter_estimates', {})
        method_dict = param_estimates.setdefault(update_method, {})
        player_est_dict = method_dict.setdefault(player_uuid, {})

        # 2c) Copy forward both roles' parameters from the previous game
        if idx > 0:
            prev_game = dyad_games[idx - 1]
            prev_method_dict = prev_game.get('parameter_estimates', {}).get(update_method, {})
            prev_player_est = prev_method_dict.get(player_uuid, {})
            # If prev_game had 'chooser' data
            if 'chooser' in prev_player_est:
                old_chooser_params = prev_player_est['chooser'].get('params', {})
                player_est_dict.setdefault('chooser', {})['params'] = copy.deepcopy(old_chooser_params)

                if update_method == 'grid':
                    if 'param_vectors' in prev_player_est['chooser']:
                        player_est_dict['chooser']['param_vectors'] = prev_player_est['chooser']['param_vectors']
                    if 'meta_data' in prev_player_est['chooser']:
                        player_est_dict['chooser']['meta_data'] = prev_player_est['chooser']['meta_data']

            # If prev_game had 'predictor' data
            if 'predictor' in prev_player_est:
                old_pred_params = prev_player_est['predictor'].get('params', {})
                player_est_dict.setdefault('predictor', {})['params'] = copy.deepcopy(old_pred_params)

                if update_method == 'grid':
                    if 'param_vectors' in prev_player_est['predictor']:
                        player_est_dict['predictor']['param_vectors'] = prev_player_est['predictor']['param_vectors']
                    if 'meta_data' in prev_player_est['predictor']:
                        player_est_dict['predictor']['meta_data'] = prev_player_est['predictor']['meta_data']

        else:
            # idx == 0 => store initial_params if not already done
            for plr_role in ('chooser', 'predictor'):
                if plr_role in initial_params:  # e.g. initial_params['chooser'] or .predictor
                    player_est_dict.setdefault(plr_role, {})['params'] = copy.deepcopy(initial_params[plr_role])

            # If role='predictor' and using grid, build initial prior param_vectors for the predictor
            if update_method == 'grid' and assigned_role != 'chooser':
                pred_sub = player_est_dict.setdefault('predictor', {})
                if 'param_vectors' not in pred_sub or 'meta_data' not in pred_sub:
                    this_pred_params = initial_params.get('predictor', {})
                    param_vals = {player_uuid: {'predictor': this_pred_params}}
                    covar = None
                    if include_covariance:
                        covar = {
                            player_uuid: {
                                'predictor': gnrl.build_covariation_matrix(param_info=param_info,
                                                                           params=this_pred_params, raise_on_invalid=True)
                            }
                        }
                    prior_data = prior_grid_from_params(param_vals=param_vals,
                                                        param_info=param_info,
                                                        n_bins_per_dimension=n_bins_per_dimension,
                                                        sample_ratio=sample_ratio,
                                                        covariation_matrix=covar,
                                                        trust_inputs=False)

                    pred_sub['meta_data'] = prior_data['meta_data']
                    pred_sub['param_vectors'] = prior_data[player_uuid]['predictor']['param_vectors']

        # 2d) Decide if we skip or play
        if assigned_role is None:
            # If user didn't specify, we do whichever role the player has
            if actual_game_role is None:
                # The player is not in this game => skip
                idx += 1
                continue
            role_to_play = actual_game_role
        else:
            # We want to do assigned_role only
            if actual_game_role != assigned_role:
                # Skip if the actual game role doesn't match the assigned role
                idx += 1
                continue
            role_to_play = assigned_role

        # 2e) We do the "active" role logic now
        role_params_for_this_game = player_est_dict[role_to_play].get('params', {})
        if not role_params_for_this_game:
            # fallback to initial if missing
            role_params_for_this_game = copy.deepcopy(initial_params.get(role_to_play, {}))
            player_est_dict[role_to_play]['params'] = role_params_for_this_game

        # Make the choice or prediction
        model_sel_key = "model_choose_A" if role_to_play == 'chooser' else "model_predict_A"
        # Decide which temperature to pass to `choice(...)`
        # current_temp = choice_temperature if role_to_play == 'predictor' else softmax_temperature # change made 04.13.2025
        current_temp = choice_temperature 

        choice_output = choice(
            current_game=game_dict,
            agent_params=role_params_for_this_game,
            utility_settings=utility_settings,
            softmax_temperature=current_temp,
            select=select
        )

        # Store the model's output for this game
        player_est_dict[role_to_play]['output'] = {
            model_sel_key: choice_output["model_choose_A"],
            'confidence': choice_output["confidence"]
        }

        if select:
            "Storing choices and predictions within the game."
            if role_to_play == 'chooser':
                choice_bit = choice_output["model_choose_A"]
                game_dict["choice"] = "A" if choice_bit == 1 else "B"
            elif role_to_play == 'predictor':
                pred_bit = choice_output["model_choose_A"]
                game_dict["prediction"] = "A" if pred_bit == 1 else "B"

        # 2f) If it's game 0, we do no update. Otherwise, if predictor, do Bayesian update
        if idx == 0:
            "No update in first game. Do not overwrite priors. Cannot learn until the first choice is observed."  
            pass  
        else:
            if role_to_play == 'predictor':
                observed_choice = game_dict.get('choice', None)
                predictor_abdicated = game_dict.get('abdicated_predictor', False)
                predictor_learned_something = observed_choice and not predictor_abdicated

                # Do the final update on the last game or if predictor observed a choice
                if predictor_learned_something or idx == game_idx_stop:
                    old_means = {k: v for k, v in role_params_for_this_game.items() if not k.endswith('_std')}
                    old_stds  = {k: v for k, v in role_params_for_this_game.items() if k.endswith('_std')}

                    if update_method == 'naive':
                        "This 'naive' model predicts from fixed parameters--no learning."
                        pass

                    elif update_method == 'parametric':
                        new_means, new_stds = bayesian_update_parametric(
                            old_means=old_means, old_stds=old_stds,
                            observed_choice=observed_choice,
                            game_dict=game_dict, choice_func=choice,
                            utility_settings=utility_settings,
                            learning_rate=learning_rate
                        )
                        # store the updated results
                        updated_params = {}
                        for param_key in param_info["keys"]:
                            if '_std' in param_key:
                                updated_params[param_key] = new_stds.get(param_key, 0.0)
                            else:
                                updated_params[param_key] = new_means.get(param_key, 0.0)
                        player_est_dict['predictor']['params'] = copy.deepcopy(updated_params)

                    elif update_method == 'MCMC':
                        new_means, new_stds = bayesian_update_mcmc(
                            old_means=old_means, old_stds=old_stds,
                            observed_choice=observed_choice,
                            game_dict=game_dict, choice_func=choice,
                            utility_settings=utility_settings,
                            param_info=param_info
                        )
                        updated_params = {}
                        for param_key in param_info["keys"]:
                            if '_std' in param_key:
                                updated_params[param_key] = new_stds.get(param_key, 0.0)
                            else:
                                updated_params[param_key] = new_means.get(param_key, 0.0)
                        player_est_dict['predictor']['params'] = copy.deepcopy(updated_params)

                    elif update_method == 'grid':
                        # ------------- GRID-BASED UPDATE -------------
                        # (i) Build or retrieve the “prior” representation
                        prior_grid_data = None
                        pred_sub = player_est_dict['predictor']
                        prev_vectors = pred_sub.get('param_vectors', None)
                        prev_meta    = pred_sub.get('meta_data', None)

                        if (prev_vectors is not None) and (prev_meta is not None):
                            # Fast path: if previous posterior was produced by the particle filter,
                            # we keep it sparse and DO NOT densify (no interpolation, no qhull).
                            if isinstance(prev_vectors, dict) and prev_meta.get('representation') == 'particles':
                                prior_grid_data = {
                                    'prior_array': prev_vectors,     # sparse map: {(i1,...,id): mass}
                                    'meta_data':   prev_meta
                                }
                            else:
                                # Grid representation: convert to dense PMF array once
                                prior_array = param_vector_to_pmf_array(
                                    param_vectors=prev_vectors,
                                    meta_data=prev_meta,
                                    general_settings=general_settings
                                )
                                prior_grid_data = {
                                    'prior_array': prior_array,      # dense ndarray
                                    'meta_data':   prev_meta
                                }

                        # Fallback (very first game or if previous state missing)
                        if not prior_grid_data or prior_grid_data.get('prior_array', None) is None:
                            fallback_pred_params = initial_params.get('predictor', {})
                            param_vals = {player_uuid: {'predictor': fallback_pred_params}}
                            covar = None
                            if include_covariance:
                                covar = {
                                    player_uuid: {
                                        'predictor': gnrl.build_covariation_matrix(
                                            param_info=param_info,
                                            params=fallback_pred_params,
                                            raise_on_invalid=True
                                        )
                                    }
                                }
                            fallback_prior_data = prior_grid_from_params(
                                param_vals=param_vals,
                                param_info=param_info,
                                n_bins_per_dimension=n_bins_per_dimension,
                                sample_ratio=sample_ratio,
                                covariation_matrix=covar,
                                trust_inputs=False
                            )
                            # Keep the prior sparse (dict) to avoid densification/interpolation.
                            # Mark representation so downstream knows how to treat it.
                            sparse_prior_vectors = fallback_prior_data[player_uuid][role_to_play]['param_vectors']
                            prior_grid_data = {
                                'prior_array': sparse_prior_vectors,  # dict: {(i1,...,id): mass}
                                'meta_data': {**fallback_prior_data['meta_data'], 'representation': 'grid_sparse'}
                            }

                        # (ii) Inject PF knobs into meta_data (so bayesian_update_grid sees them)
                        meta_for_update = copy.deepcopy(prior_grid_data['meta_data'])
                        meta_for_update['use_particle_filter']  = bool(general_settings.get('use_particle_filter', True))
                        meta_for_update['pf_max_particles']     = int(general_settings.get('pf_max_particles', 5000))
                        meta_for_update['pf_min_particles']     = int(general_settings.get('pf_min_particles', 200))
                        meta_for_update['pf_resample_fraction'] = float(general_settings.get('pf_resample_fraction', 0.5))
                        meta_for_update['pf_jitter_sd']         = float(general_settings.get('pf_jitter_sd', 0.0))  # default 0.0 for speed

                        "Used for a trivial non-Bayesian model that forgets all priors."
                        no_memory_mode = general_settings.get('no_memory_mode', False)

                        # (iii) Now do the update
                        likelihood_temp = initial_params.get(player_role, {}).get('temp', choice_temperature)
                        posterior_data = bayesian_update_grid(
                            prior_array=prior_grid_data['prior_array'],   # dict or ndarray
                            meta_data=meta_for_update,
                            softmax_temperature=likelihood_temp,
                            utility_settings=utility_settings,
                            general_settings=general_settings,
                            no_memory_mode=no_memory_mode,
                            param_info=param_info,
                            game_dict=game_dict,
                            choice_func=choice,
                        )

                        "Store prior parameter stats."
                        if isinstance(prior_grid_data['prior_array'], dict) and prior_grid_data['meta_data'].get('representation') == 'particles':
                            pred_sub['params'] = gnrl._statistics_from_sparse_param_vectors(
                                param_vectors=prior_grid_data['prior_array'],
                                meta_data=prior_grid_data['meta_data'],
                                param_info=param_info
                            )
                        else:
                            tickvals_array = [
                                prior_grid_data['meta_data']["tickvals"][key]
                                for key in prior_grid_data['meta_data']["tickvals"].keys()
                            ]
                            pred_sub['params'] = gnrl.compute_statistics(
                                joint_pmf=prior_grid_data['prior_array'],
                                grids=tickvals_array,
                                param_info=param_info
                            )

                        if 'temp' not in pred_sub['params']:
                            "Storing choice temperature (which is static across rounds)."
                            if 'temp' in initial_params.get(player_role, {}):
                                pred_sub['params']['temp'] = initial_params[player_role]['temp']
                            else:
                                pred_sub['params']['temp'] = choice_temperature

                        # (iii) Store the new posterior param_vectors
                        pred_sub['meta_data'] = posterior_data['meta_data']
                        pred_sub['param_vectors'] = posterior_data['param_vectors']

                        # (iv) On last game, compute final means, std, etc. from the posterior
                        if idx == game_idx_stop:
                            posterior_meta    = posterior_data['meta_data']
                            posterior_vectors = posterior_data['param_vectors']

                            if isinstance(posterior_vectors, dict) and posterior_meta.get('representation') == 'particles':
                                final_stats = gnrl._statistics_from_sparse_param_vectors(
                                    param_vectors=posterior_vectors,
                                    meta_data=posterior_meta,
                                    param_info=param_info
                                )
                            else:
                                posterior_array = param_vector_to_pmf_array(
                                    param_vectors=posterior_vectors,
                                    meta_data=posterior_meta,
                                    general_settings=general_settings
                                )
                                tickvals_array = [
                                    posterior_meta["tickvals"][key]
                                    for key in posterior_meta["tickvals"].keys()
                                ]
                                final_stats = gnrl.compute_statistics(
                                    joint_pmf=posterior_array,
                                    grids=tickvals_array,
                                    param_info=param_info
                                )

                            pred_sub['posteriors'] = final_stats

                    else:
                        raise ValueError(f"Only supports update_methods 'parametric','MCMC','grid', not {update_method}.")

                else:
                    "Skip update. Previous param data already copied into current game."
                    pass

        # Always ensure we store 'params' for the role in this game
        if 'params' not in player_est_dict[role_to_play]:
            player_est_dict[role_to_play]['params'] = copy.deepcopy(role_params_for_this_game)

        idx += 1

    return dyad_games


def simulate_dyad(dyad_games: DyadGames, initial_params_p1_p2: List[Dict[str, Dict[str, float]]], param_info: ParamInfo, 
                  utility_settings: UtilitySettings, general_settings: GeneralSettings) -> List[Dict]:
    """
    Simulates a series of binary dictator games between a pair of 
    participants, who alternate between roles as chooser and predictor. 

    Arguments:
        • dyad_games: List[Dict]; list of games between a participant pair.
            - Each dict is a "meeting" or "game" in the dyad, containing:
                'chooser', 'predictor', 'payoff_A_chooser', 'payoff_B_chooser',
                'payoff_A_predictor', 'payoff_B_predictor', etc.
        • initial_params_p1_p2: List[Dict[str, Dict[str, float]]]; Parameters for both players.
            - The order of the parameter dicts corresponds to the alphabetical order 
                of the player uuids, which are extracted from dyad_games.

    Returns:
        • dyad_games: list[dict]; The series of games between a pair of players 
            with estimated parameters stored within those games for both players.            
    """
    "Validating inputs."
    if not isinstance(dyad_games, (list, tuple)):
        raise ValueError(f"dyad_games must be a list, not {type(dyad_games)}!")

    if not all(isinstance(game, dict) for game in dyad_games):
        raise ValueError(f"dyad_games must be a list of dictionaries, not {type(dyad_games[0])}!")

    first_game = dyad_games[0]
    first_chooser = first_game.get('chooser', None)
    first_predictor = first_game.get('predictor', None) 
    if first_chooser is None or first_predictor is None:
        raise ValueError(f"Failed to extract player uuids from the first game: {first_game}.")    

    "Checking if initial_params_p1_p2 contains all required keys."
    for param_dict in initial_params_p1_p2:
        for player_role_key in ['chooser', 'predictor']:
            param_dict_role = param_dict.get(player_role_key, None)
            if param_dict_role is not None:
                "param_keys is a 'global variable' at the top of the file."
                for param_key in param_info["keys"]:
                    if param_key not in param_dict_role:                     
                        raise ValueError(f"{param_key} missing from initial_params_p1_p2.")

    "Sorting player uuids in initial_params_p1_p2 alphabetically."
    player_uuid_1, player_uuid_2 = sorted([first_chooser, first_predictor])

    "Creating a dictionary of player parameters."
    player_params = {
        player_uuid_1: initial_params_p1_p2[0], 
        player_uuid_2: initial_params_p1_p2[1]
    }

    "Iterate agent() for both players over dyad_games one game at a time."
    for meeting_idx in range(len(dyad_games)):
        for player_uuid in [player_uuid_1, player_uuid_2]:
            dyad_games = agent(dyad_games=dyad_games, game_idx_start=meeting_idx, game_idx_stop=meeting_idx, 
                               initial_params=player_params[player_uuid], param_info=param_info, utility_settings=utility_settings,
                               player_uuid=player_uuid, general_settings=general_settings)

    return dyad_games


def loss_function_bayes(dyad_games: list[dict[str, Any]], general_settings: Dict[str, Any]) -> list[dict[str, Any]]:
    """
    Computes per-game loss for each (player, role) and stores intermediate values.
    This does NOT accumulate totals across games here—see create_loss_report for that.
    Stores all data in param_estimates[update_method][player_uuid][player_role]['output'].

    Arguments:
        • dyad_games: list[dict[str, Any]]; List of binary dictator games.
        • general_settings: dict[str, Any]; Various settings used throughout this analysis

    For each game and for each (player, role):
        • raw_ssr: (model prediction - actual response)^2
        • raw_neglogprob: -log(predicted_probability_of_observed_action)
        • confidence: from choice(); Confidence is the inverse variance of parameters.
        • param_penalty: from parameter_penalty(); Penalizes large parameter absolute values.
        • loss_final: depends on general_settings['confidence_weighted'] (True/False)
            - If True : final = raw_metric * confidence + param_penalty
            - If False: final = raw_metric + param_penalty

    Returns:
        • dyad_games, with each game’s param_estimates updated.
    """
    confidence_weighted = general_settings.get('confidence_weighted', True)
    loss_funct_type = general_settings.get('loss_funct_type', 'log')
    update_method = general_settings.get('update_method', 'naive')
    penalty_weight = general_settings.get('penalty_weight', 0.01)
    
    for game_dict in dyad_games:
        param_estimates: dict = game_dict.setdefault('parameter_estimates', {}).setdefault(update_method, {})

        "For each player/role in this game, compute raw_srr & raw_neglogprob & final loss"
        for player_uuid, role_dict in param_estimates.items():
            for player_role, stats_dict in role_dict.items():
                model_output: dict = stats_dict.setdefault('output', {})
                if not model_output:
                    continue  # no predictions here

                "Check if we actually have a predicted probability"
                if player_role == 'chooser':
                    selection = game_dict.get('choice', None)      # Actual choice
                    model_select_A = model_output.get('model_choose_A', None)
                    abdicated = game_dict.get('abdicated_chooser', False)
                else:  # 'predictor'
                    selection = game_dict.get('prediction', None) # Actual prediction
                    model_select_A = model_output.get('model_predict_A', None)
                    abdicated = game_dict.get('abdicated_chooser', False) or game_dict.get('abdicated_predictor', False)

                if selection is None or model_select_A is None or abdicated:
                    "Skip if player abdicated response, no response is found, or no model prediction is found."
                    continue

                selection_val = 1 if selection == 'A' else 0

                "Compute raw residuals"
                raw_ssr = (model_select_A - selection_val)**2
                prob_of_observed = model_select_A if selection_val == 1 else (1 - model_select_A)
                if prob_of_observed <= 0:
                    prob_of_observed = 1e-6
                raw_neglogprob = -math.log(prob_of_observed)

                "The chosen 'residual' depends on whether 'ssr' or 'log'"
                if loss_funct_type == 'ssr':
                    raw_residual = raw_ssr
                else:
                    raw_residual = raw_neglogprob

                "Confidence is inverse parameter variance."
                confidence = model_output.get('confidence', 1.0)

                "Penalize large parameter absolute values."
                param_penalty_val = 0.0
                if 'params' in stats_dict:
                    param_penalty_val = gnrl.parameter_penalty(
                        params=stats_dict['params'],
                        penalty_weight=penalty_weight
                    )
                else:
                    raise Exception(f"'params' not found in stats dict even though model produced a prediction.")

                "final loss depending on confidence_weighted"
                if confidence_weighted:
                    loss_final = raw_residual * confidence + param_penalty_val
                else:
                    loss_final = raw_residual + param_penalty_val

                "Store data"
                model_output['raw_ssr'] = raw_ssr
                model_output['raw_neglogprob'] = raw_neglogprob
                model_output['confidence'] = confidence
                model_output['param_penalty'] = param_penalty_val
                model_output['loss_final'] = loss_final

    return dyad_games


def create_loss_report(dyad_games: list[dict[str, Any]], general_settings: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Aggregates per-game data from loss_function_bayes. Sums them up across 
    all games in the dyad for each (player, role), then stores the results 
    in dyad_games[0]['loss_report'][player_uuid][player_role] = { ... }.

    Notes:
        • If there's no per-game data, sums are zero or none. 
        • This function returns the loss report but also stores it within the first game.

    Arguments:
        • dyad_games: list[dict[str, Any]]; List of binary dictator games.
        • general_settings: dict[str, Any]; Various settings used throughout this analysis

    Returns:
        • loss_report: dict[str, dict[str, dict[str, Any]]] = {
            'player_uuid': {
                'player_role': {
                    'n_data': int,
                    'raw_ssr_sum': float,
                    'raw_neglogprob_sum': float,
                    'param_penalty_sum': float,
                    'loss_final_sum': float,
                    'confidence_weighted': bool,
                    'loss_funct_type': str,
                    'penalty_weight': float,
                    'update_method': str,                    
                },...
            },...
        }
    """
    confidence_weighted = general_settings.get('confidence_weighted', True)
    loss_funct_type = general_settings.get('loss_funct_type', 'log')
    update_method = general_settings.get('update_method', 'grid')
    penalty_weight = general_settings.get('penalty_weight', 0.1)

    "Prepare a top-level dictionary in game[0]"
    first_game = dyad_games[0]
    lr_container: dict = first_game.setdefault('loss_report', {})

    "Accumulate sums in a local structure, then store them"
    sums_dict = {}  # { player_uuid : { role: { n_data, raw_ssr, raw_neglogprob, param_penalty, ... } } }
    for game_dict in dyad_games:
        param_estimates: dict = game_dict.get('parameter_estimates', {}).get(update_method, {})
        for player_uuid, role_dict in param_estimates.items():
            if player_uuid not in sums_dict:
                sums_dict[player_uuid] = {}
            for player_role, stats_dict in role_dict.items():
                if not isinstance(stats_dict, dict):
                    continue
                model_out: dict = stats_dict.get('output', {})
                if not model_out:
                    "Skips abdicated responses."
                    continue
                "Store the values from the final call to loss_function_bayes"
                raw_ssr = model_out.get('raw_ssr', 0.0)
                raw_ll = model_out.get('raw_neglogprob', 0.0)
                param_pen = model_out.get('param_penalty', 0.0)
                final_l = model_out.get('loss_final', 0.0)

                "Init if needed"
                if player_role not in sums_dict[player_uuid]:
                    sums_dict[player_uuid][player_role] = {
                        'n_data': 0,
                        'raw_ssr_sum': 0.0,
                        'raw_neglogprob_sum': 0.0,
                        'param_penalty_sum': 0.0,
                        'loss_final_sum': 0.0
                    }
                sums_dict[player_uuid][player_role]['n_data'] += 1
                sums_dict[player_uuid][player_role]['raw_ssr_sum'] += raw_ssr
                sums_dict[player_uuid][player_role]['raw_neglogprob_sum'] += raw_ll
                sums_dict[player_uuid][player_role]['param_penalty_sum'] += param_pen
                sums_dict[player_uuid][player_role]['loss_final_sum'] += final_l

    "Store them in dyad_games[0]['loss_report']"
    for player_uuid, role_data in sums_dict.items():
        lr_container.setdefault(player_uuid, {})
        for player_role, sums in role_data.items():
            "Store sums and relevant settings."
            lr_container[player_uuid][player_role] = {
                "n_data": sums['n_data'],
                "raw_ssr_sum": sums['raw_ssr_sum'],
                "raw_neglogprob_sum": sums['raw_neglogprob_sum'],
                "param_penalty_sum": sums['param_penalty_sum'],
                "loss_final_sum": sums['loss_final_sum'],
                "confidence_weighted": confidence_weighted,
                "loss_funct_type": loss_funct_type,
                "penalty_weight": penalty_weight,
                "update_method": update_method,
            }
    # pp.pprint(lr_container)
    return lr_container


def fit_params_by_player(player_uuid: PlayerUUID, param_info: ParamInfo, utility_settings: UtilitySettings, 
                              file_paths: FilePaths, general_settings: GeneralSettings) -> None:
    """
    Runs an optimization function to find the social preference parameter values that best fit participants' 
    patterns of choices and predictions. This fits parameters by player across all dyads they participated in.
    
    Arguments:
        • player_uuid: str; Identifies the player for whom parameters are being fit.
        • param_info: dict[str, list[Any]]; Stores parameter keys, boundaries, and initial guesses.
        • utility_settings: dict[str, bool]; Defines the functional form of the utility function.
        • file_paths: dict[str, str | dict[str, str]]; Stores all file paths for storing data.
        • general_settings: dict[str, Any]; Various settings used throughout this analysis.

    Returns:
        • None; Instead saves the results within files.
    """
    time_start_fit_plr = time.time()

    "Extract General Settings"
    experiment_num = general_settings.get('experiment_num', 3)
    update_method = general_settings.get('update_method', True)
    include_covariance = general_settings.get('include_covariance', True)
    softmax_temperature = general_settings.get('softmax_temperature', True)
    temperature_is_param = general_settings.get('temperature_is_param', True)
    optimization_method = general_settings.get('optimization_method', 'local')

    if experiment_num == 0 and 'chooser' in player_uuid:
        print(f"chooser player_uuid = {player_uuid}")
        raise Exception(f"Can only fit predictors for simulated data.")

    if update_method in ('naive', 'parametric'):
        for param_key in param_info['keys']:
            if '_std' in param_key:
                raise ValueError(f"Standard deviation parameter {param_key} discovered during {update_method} update method.")

    if isinstance(player_uuid, int):
        plrs_to_dyads = prep.players_to_dyads(experiment_num=experiment_num, 
                                                    file_paths=file_paths, create_new_file=False)
        plr_keys = sorted(list(plrs_to_dyads.keys()))
        player_uuid = plr_keys[player_uuid % len(plr_keys)]

    player_fit_dir = os.path.join(file_paths["player_fits"], f"experiment_{experiment_num}")
    if experiment_num == 0:
        player_fit_name = f"{player_uuid}.json"
    else:
        player_fit_name = f"{file_paths.get('file_name_suffix','')}_{player_uuid}.json"

    plr_file_path = prep.ensure_directory_and_join(base_dir=player_fit_dir, file_name=player_fit_name)

    try:
        if not general_settings.get('create_new_file', False) and os.path.exists(plr_file_path):
            with open(plr_file_path, "r", encoding='utf-8') as file:
                fitted_player_dyads = json.load(file)
            return fitted_player_dyads
    except json.decoder.JSONDecodeError as error:
        print(error)

    player_dyads = prep.dyads_for_a_player(player_uuid=player_uuid, experiment_num=experiment_num, file_paths=file_paths, 
                                                 analysis_mode=general_settings.get('analysis_mode', 'bayesian'), dyad_already_analyzed=False)

    initial_params = {'chooser': {}, 'predictor': {}}
    for player_role in ('chooser', 'predictor'):
        for param_cat in ('keys', 'bounds', 'guesses'):
            if param_cat == 'guesses':
                initial_params[player_role][param_cat] = param_info['guesses']() \
                    if callable(param_info['guesses']) else copy.deepcopy(param_info['guesses'])
            else:
                initial_params[player_role][param_cat] = copy.deepcopy(param_info[param_cat])
            if include_covariance:
                initial_params[player_role][param_cat] += copy.deepcopy(param_info['covar'][param_cat])
        if temperature_is_param and update_method in ('MCMC', 'grid'):
            initial_params[player_role]['keys'] += ['temp']
            initial_params[player_role]['bounds'] += [(0.5, 3.0)]
            initial_params[player_role]['guesses'] += [softmax_temperature]

    "Remove standard deviation parameters from chooser's params" #TODO CHECK Added 04/06/2025
    for param_key in list(initial_params['chooser'].keys()):
        if '_std' in param_key:
            del initial_params['chooser'][param_key] 

    loss_report = {
        'chooser': [],
        'predictor': []
    }

    def optimize_roles(initial_params_for_role: dict[str, list], role_to_fit: str):
        """"""
        
        def objective_function(param_array: NDArray[np.float64]) -> float:
            """"""
            param_array = copy.deepcopy(param_array)
            if isinstance(param_array, np.ndarray):
                param_array = param_array.tolist()

            param_array: list
            if temperature_is_param:
                choice_temperature = param_array[-1]
            else:
                choice_temperature = softmax_temperature

            role_params = {param_key: param_val for param_key, param_val in zip(initial_params_for_role['keys'], param_array)}

            if include_covariance and role_to_fit == 'predictor':
                safe_pred_params = gnrl.transform_cov_params(params=role_params, param_info=param_info)
                if safe_pred_params['loss'] is not None:
                    print(safe_pred_params['report'])
                    return safe_pred_params['loss']
                role_params = safe_pred_params['params']

            total_plr_loss = 0.0
            aggregated_sums = {
                'n_data': 0,
                'raw_ssr_sum': 0.0,
                'raw_neglogprob_sum': 0.0,
                'param_penalty_sum': 0.0,
                'loss_final_sum': 0.0
            }

            for dyad_key, dyad_games in player_dyads.items():
                "Run agent function with parameters in param_array"
                dyad_games_copy = copy.deepcopy(dyad_games) 
                updated_games = agent(dyad_games=dyad_games_copy,
                                        game_idx_start=0,
                                        game_idx_stop=len(dyad_games)-1,
                                        initial_params={role_to_fit: role_params},
                                        param_info=param_info,
                                        utility_settings=utility_settings,
                                        player_uuid=player_uuid,
                                        player_role=role_to_fit,
                                        general_settings=general_settings,
                                        choice_temperature=choice_temperature)
                
                if general_settings.get('include_covariance') and role_to_fit == 'predictor':
                    prior_param_vector = updated_games[0].get('parameter_estimates', {}).get('grid', {}).get(
                        player_uuid, {}).get('predictor', {}).get('param_vectors', None)
                    "Preventing error with degenerate priors that have all zeros and a few huge probabilities."
                    if isinstance(prior_param_vector, dict):
                        for probability in prior_param_vector.values():
                            if probability > 0.2:
                                print(f"Detected massive probability {probability}")
                                return 1e6
                                
                "Compute loss (using loss_function_bayes)."
                updated_games = loss_function_bayes(dyad_games=updated_games, general_settings=general_settings)
                loss_sums: dict = create_loss_report(dyad_games=updated_games, general_settings=general_settings).get(player_uuid, {}).get(role_to_fit, {})
                total_plr_loss += loss_sums.get('loss_final_sum', 0.0)

                # accumulate into aggregated_sums
                aggregated_sums['n_data'] += loss_sums.get('n_data', 0)
                aggregated_sums['raw_ssr_sum'] += loss_sums.get('raw_ssr_sum', 0.0)
                aggregated_sums['raw_neglogprob_sum'] += loss_sums.get('raw_neglogprob_sum', 0.0)
                aggregated_sums['param_penalty_sum'] += loss_sums.get('param_penalty_sum', 0.0)
                aggregated_sums['loss_final_sum'] += loss_sums.get('loss_final_sum', 0.0)

            # Build the row => param_key => param_val plus the aggregated sums
            row = {}
            # Add param_key: param_val
            for pkey, pval in zip(initial_params_for_role['keys'], param_array):
                row[pkey] = pval
            # Add aggregated sums
            row.update(aggregated_sums)
            loss_report[role_to_fit].append(row)

            return total_plr_loss

        def objective_function_raw_nll(param_array: NDArray[np.float64]) -> float:
            """
            Raw NLL objective for the same parameterization, *without* any parameter penalty.
            This reads the per-dyad raw_neglogprob_sum and aggregates across all dyads.
            """
            param_array = np.asarray(param_array, dtype=float)
            if temperature_is_param:
                choice_temperature_local = float(param_array[-1])
            else:
                choice_temperature_local = float(softmax_temperature)

            role_params_local = {param_key: float(param_val) for param_key, param_val in zip(initial_params_for_role['keys'], param_array)}
            if include_covariance and role_to_fit == 'predictor':
                safe_pred = gnrl.transform_cov_params(params=role_params_local, param_info=param_info)
                if safe_pred['loss'] is not None:
                    # invalid covariance parametrization — return a large penalty-like number
                    return float(safe_pred['loss'])
                role_params_local = safe_pred['params']

            total_raw_nll = 0.0
            for dyad_key, dyad_games_local in player_dyads.items():
                games_copy = copy.deepcopy(dyad_games_local)
                # run the agent with these parameters
                updated_games_local = agent(
                    dyad_games=games_copy,
                    game_idx_start=0,
                    game_idx_stop=len(dyad_games_local)-1,
                    initial_params={role_to_fit: role_params_local},
                    param_info=param_info,
                    utility_settings=utility_settings,
                    player_uuid=player_uuid,
                    player_role=role_to_fit,
                    general_settings=general_settings,
                    choice_temperature=choice_temperature_local
                )
                # compute losses
                updated_games_local = loss_function_bayes(dyad_games=updated_games_local, general_settings=general_settings)
                loss_sums_local: dict = create_loss_report(
                    dyad_games=updated_games_local,
                    general_settings=general_settings
                ).get(player_uuid, {}).get(role_to_fit, {})
                total_raw_nll += float(loss_sums_local.get('raw_neglogprob_sum', 0.0))

            return float(total_raw_nll)

        best_fitting_params = {}
        optimization_results = {}

        bounds = initial_params[role_to_fit]['bounds']
        guesses = initial_params[role_to_fit]['guesses']

        if update_method == 'grid' and role_to_fit == 'predictor' and not general_settings.get('no_memory_mode', False):
            "Using parameters fitted on the 'naive' update method as the initial parameter guesses."
            general_settings_ = copy.deepcopy(general_settings)
            general_settings_['update_method'] = 'naive'
            general_settings_['run_in_parallel'] = True
            file_name_suffix_naive = prep.create_file_name_suffix(general_settings=general_settings_, utility_settings=utility_settings)
            # directory_path_naive = f"Judgment_Game/Inputs/Iter_Binary_Dictator/player_fits/experiment_{experiment_num}"
            directory_path_naive = ensure_directory_and_join(file_paths['player_fits'], f'experiment_{experiment_num}')
            file_name_naive = file_name_suffix_naive + f"_{player_uuid}.json"
            file_path_naive = prep.ensure_directory_and_join(base_dir=directory_path_naive, file_name=file_name_naive)
            if os.path.exists(file_path_naive):
                player_histories_naive = None
                with open(file_path_naive, "r", encoding="utf-8") as file:
                    player_histories_naive = json.load(file)
                if isinstance(player_histories_naive, dict):
                    dyad_keys = list(player_histories_naive.keys())
                    if len(dyad_keys) > 0:
                        first_dyad = player_histories_naive[dyad_keys[0]]
                        if len(first_dyad) > 0:
                            first_game: dict = first_dyad[0]
                            if first_game.get('predictor') == player_uuid and 'naive' in first_game.get('parameter_estimates', {}):
                                params: dict = first_game['parameter_estimates']['naive'].get(player_uuid, {}).get('predictor', {}).get('params', {})
                                educated_guesses = copy.deepcopy(guesses)
                                for idx, param_key in enumerate(param_info['keys']):
                                    fitted_param = params.get(param_key, None)
                                    if isinstance(fitted_param, (float, int)) and (
                                        param_info['bounds'][idx][0] <= fitted_param <= param_info['bounds'][idx][1]):
                                        educated_guesses[idx] = fitted_param
                                guesses = educated_guesses

        # --- Child→Parent warm-start (consume the function's prepared guesses) -----
        opt_method_local = str(general_settings.get('optimization_method', 'globloc')).lower()
        warm_pol = general_settings.get("warmstart_policy", {}) or {}
        guesses_before = list(map(float, np.array(guesses, dtype=float)))

        # We’ll record what happened for the JSON report you already write out.
        warm_meta = {
            "enabled": bool(warm_pol.get("enabled", True)),
            "phase":   warm_pol.get("phase", "cold"),
            "temperature": float(warm_pol.get("temperature", 0.0)),
            "x_initial_guess_before": guesses_before,
            "x_initial_guess_after":  None,
            "optimization_method_effective": None,
            "selected_child_key": None,
            "exception": None, "used": False,
            "model_bit_string": gnrl.convert_utility_settings(utility_settings=utility_settings, into=str),
            "model_equation": build_utility_equation(utility_settings=utility_settings),
            "model_utility_settings": gnrl.convert_utility_settings(utility_settings=utility_settings, into=dict),
        }

        # --- Degenerate model guard: no free parameters for this role -----------------
        if len(bounds) == 0:
            # Evaluate once at the empty vector so your loss_report is still populated.
            baseline_x_vector = np.array([], dtype=float)
            baseline_loss_value = float(objective_function(baseline_x_vector))

            # Build a minimal "optimization" report so downstream code doesn't break.
            gltc_result = {
                "final": {"x": [], "fun": baseline_loss_value},
                "random_search": {"x_initial_guess": []},
                "local": {"success": True, "message": "No free parameters for this role."},
                "warmstart_meta": {
                    **warm_meta,
                    "optimization_method_effective": "none",
                    "x_initial_guess_before": [],
                    "x_initial_guess_after":  []
                }
            }
            optimization_results[role_to_fit] = gltc_result
            best_fitting_params[role_to_fit] = {}  # nothing to fit for this role
            return best_fitting_params, optimization_results
        # -------------------------------------------------------------------------------

        # --- Base case: one free parameter → skip warm-start, still optimize ---
        elif len(bounds) == 1:
            # Preserve your random initial guess (already in `guesses`)
            warm_meta["enabled"] = False
            warm_meta["phase"]   = "cold"
            warm_meta["note"] = "One-parameter model: keeping dual_annealing enabled."
            
            "Single-parameter models should always explore globally."
            opt_method_local = "globloc"       

        # try:
        if warm_meta["enabled"] and warm_meta["phase"] == "warm":
            warm_start = best_fitting_child_parameters_for_parent(
                player_uuid=player_uuid,
                player_role=role_to_fit,
                utility_settings_parent=utility_settings,   # parent is the current model
                utility_settings=utility_settings,          # ok (universe of flags)
                general_settings=general_settings,
                file_paths=file_paths,
                param_bds=param_bds,                        # module/global
                within_ic_analysis=True,
                temperature=warm_meta["temperature"]
            )

            # IMPORTANT: use the parent-space warmstart prepared inside warm_start
            parent_ws = (warm_start or {}).get("parent_warmstart", {})
            ws_params_for_role = parent_ws.get(player_uuid, {}).get(role_to_fit)

            # Optionally record which child was selected (bitstring)
            warm_meta["selected_child_key"] = (
                ((warm_start or {}).get("selected_child", {}) or {})
                .get("metadata", {}).get("model_bit_str", None)
            )

            warm_meta["child_equation"] = warm_start.get("metadata", {}).get("U_funct")
            warm_meta["model_equation"] = warm_start.get("metadata", {}).get("parent_equation")

            if isinstance(ws_params_for_role, dict) and ws_params_for_role:
                warm_guess = []
                if "guesses" in ws_params_for_role:
                    warm_guess = ws_params_for_role["guesses"]

                else:
                    for idx, param_key in enumerate(initial_params_for_role["keys"]):
                        param_val = ws_params_for_role.get(param_key, None)
                        if isinstance(param_val, (int, float)):
                            warm_guess.append(float(param_val))
                        else:
                            warm_guess = []
                            break

                if warm_guess:
                    guesses = np.array(warm_guess, dtype=float)
                    warm_meta["used"] = True

                    # If requested, force local-only when using warm-starts 
                    if warm_pol.get("disable_dual_annealing_when_warm", True):
                        opt_method_local = "local"

        # except Exception as _warm_err:
        #     print(f"[WarmStart] {player_uuid[:8]} - {role_to_fit}: {repr(_warm_err)}")
        #     warm_meta["exception"] = repr(_warm_err)

        warm_meta["x_initial_guess_after"] = list(map(float, np.array(guesses, dtype=float)))
        warm_meta["optimization_method_effective"] = opt_method_local
        # ---------------------------------------------------------------------------

        if len(bounds) != len(guesses):
            print("Param Keys:", initial_params[role_to_fit]['keys'])
            raise Exception(f"len(bounds) = {len(bounds)} != len(guesses) = {len(guesses)}")

        default_optimization_policy = {
            'n_random_starts'    : 1,
            'maxiter_global'     : 24,
            'maxiter_local'      : 24,
            'maxfun_global'      : 24,
            'maxfun_local'       : 24,
            'run_trust_constr'   : False,
            'dual_annealing_seed': None,
            'trust_maxiter'      : 600,
            'trust_gtol'         : 1e-6,
            'trust_xtol'         : 1e-8,
            'trust_verbose'      : False        
        }

        optimization_policy = general_settings.get('optimization_policy', default_optimization_policy)

        "If exactly one dimension, make global sampling a bit deeper and ensure multiple random starts."
        if len(bounds) == 1:
            optimization_policy['n_random_starts'] = max(optimization_policy.get('n_random_starts', 1), 5)
            maxiter_global = optimization_policy.get('maxiter_global', 24)
            maxfun_global = optimization_policy.get('maxfun_global', 24)
            if isinstance(maxfun_global, int):
                maxiter_global = max(maxiter_global, 64)
            if isinstance(maxfun_global, int):
                maxfun_global = max(maxfun_global, 200)
            optimization_policy['maxiter_global']  = maxiter_global
            optimization_policy['maxfun_global']   = maxfun_global

        # --- Two-step optimizer: robust penalized search, then optional constrained raw-NLL refine ---
        gltc_result = global_local_then_trust_constr(
            objective_with_penalty = objective_function,           # existing penalized objective
            objective_raw_nll      = objective_function_raw_nll,   # new raw NLL objective
            x_bounds               = bounds,
            x_initial_guess        = guesses,
            optimization_method    = opt_method_local,
            parameter_keys         = initial_params_for_role['keys'],
            local_methods          = optimization_policy.get('local_methods', None),
            n_random_starts        = optimization_policy.get('n_random_starts', 1),
            maxiter_global         = optimization_policy.get('maxiter_global', 24),
            maxiter_local          = optimization_policy.get('maxiter_local', 24),
            maxfun_global          = optimization_policy.get('maxfun_global', 24),
            maxfun_local           = optimization_policy.get('maxfun_local', 24),
            run_trust_constr       = optimization_policy.get('run_trust_constr', False),
            dual_annealing_seed    = optimization_policy.get('dual_annealing_seed', None),
            trust_maxiter          = int(optimization_policy.get('trust_maxiter', 600)),
            trust_gtol             = float(optimization_policy.get('trust_gtol', 1e-6)),
            trust_xtol             = float(optimization_policy.get('trust_xtol', 1e-8)),
            trust_verbose          = bool(optimization_policy.get('trust_verbose', False))
        )

        # later, after gltc_result is created:
        gltc_result["warmstart_meta"] = {
            **warm_meta,
            "x_initial_guess_before": guesses_before,
            "x_initial_guess_after":  np.array(guesses, float).tolist(),
            "optimization_method_effective": opt_method_local
        }
        # and add the actual initial guess to the random_search section:
        if "random_search" not in gltc_result:
            gltc_result["random_search"] = {}
        gltc_result["random_search"]["x_initial_guess"] = np.array(guesses, float).tolist()

        # Record the pipeline reports for this role
        optimization_results[role_to_fit] = gltc_result

        # --- Choose final point by *raw NLL* among all points we actually evaluated ----
        if loss_report[role_to_fit]:
            # (a) best raw NLL seen during *any* penalized objective call
            min_row = min(loss_report[role_to_fit], key=lambda r: r.get('raw_neglogprob_sum', float('inf')))
            raw_min_seen = float(min_row.get('raw_neglogprob_sum', float('inf')))
            x_minraw = np.array(
                [min_row[k] for k in initial_params_for_role['keys'] if k in min_row],
                dtype=float
            )

            # (b) raw NLL at the optimizer's final x
            x_final = np.asarray(gltc_result['final']['x'], float)
            raw_at_final = float(objective_function_raw_nll(x_final))

            # (c) if the observed raw-min beats the optimizer's final, override final
            if raw_min_seen + 1e-8 < raw_at_final:
                gltc_result['final'].update({
                    "x": x_minraw.tolist(),
                    "loss": raw_min_seen,
                    "chosen_optimizer": gltc_result['final'].get("chosen_optimizer", "globloc") + "+raw-min-override"
                })

        best_vector = np.asarray(gltc_result['final']['x'], dtype=float)
        best_fitting_params[role_to_fit] = {
            param_key: float(param_val)
            for param_key, param_val in zip(initial_params_for_role['keys'], best_vector)  
        }

        return best_fitting_params, optimization_results

    best_fitting_params = {}
    optimization_results = {}
    for player_role in ('predictor', 'chooser', ):
        if player_role == 'chooser' and experiment_num != 3:
            continue

        best_fit_params_role, opt_results_role = optimize_roles(
            initial_params_for_role=initial_params[player_role], role_to_fit=player_role)
        best_fitting_params[player_role] = best_fit_params_role[player_role]
        optimization_results[player_role] = opt_results_role[player_role]

    # 3) Build & save the CSV. We do it for each role, or unify them if you prefer
    for role_to_fit in ('chooser','predictor'):
        if not loss_report[role_to_fit]:
            continue  # maybe it's empty for roles we didn't optimize
        if experiment_num == 0 and role_to_fit == 'chooser':
            continue

        df_loss = pd.DataFrame(loss_report[role_to_fit])
        if "raw_neglogprob_sum" in df_loss.columns:
            df_loss.sort_values("raw_neglogprob_sum", inplace=True)

        base_loss_dir = os.path.join(file_paths["player_fits"], "loss_reports", f"experiment_{experiment_num}")
        if experiment_num == 0:
            file_name_loss = f"Loss-{role_to_fit[0]}-{player_uuid}.csv"
        else:
            file_name_loss = f"Loss{file_paths.get('file_name_suffix','')}-{role_to_fit[0]}-{player_uuid}.csv"

        csv_path = prep.ensure_directory_and_join(base_dir=base_loss_dir, file_name=file_name_loss)
        df_loss.to_csv(csv_path, index=False, encoding='utf-8-sig')

    fitted_plr_dyads = {}
    for dyad_key, dyad_games in player_dyads.items():

        if temperature_is_param:
            choice_temperature = softmax_temperature
            choice_temp = best_fitting_params.get('chooser', {}).get('temp')
            if choice_temp is not None:
                choice_temperature = choice_temp
            else:
                choice_temp = best_fitting_params.get('predictor', {}).get('temp')
                if choice_temp is not None:
                    choice_temperature = choice_temp
        else:
            choice_temperature = softmax_temperature

        "Run agent function with parameters in param_array"
        fitted_dyad_games = agent(dyad_games=dyad_games,
                                game_idx_start=0,
                                game_idx_stop=len(dyad_games)-1,
                                initial_params=best_fitting_params,
                                param_info=param_info,
                                utility_settings=utility_settings,
                                player_uuid=player_uuid,
                                player_role=None,
                                general_settings=general_settings,
                                choice_temperature=choice_temperature)
        
        "Compute loss (using loss_function_bayes)."
        fitted_dyad_games = loss_function_bayes(
            dyad_games=fitted_dyad_games, general_settings=general_settings)

        loss_sum = 0.0
        fitted_dyad_games[0]['loss_report'] = create_loss_report(dyad_games=fitted_dyad_games, general_settings=general_settings)
        loss_sum += fitted_dyad_games[0].get('loss_report', {}).get(player_uuid, {}).get('chooser', {}).get('loss_final_sum', 0.0)
        loss_sum += fitted_dyad_games[0].get('loss_report', {}).get(player_uuid, {}).get('predictor', {}).get('loss_final_sum', 0.0)
        # print(f"Final Loss Player {player_uuid}: {loss_sum}")

        "Making Numpy arrays JSON serializable."
        if update_method == 'grid':
            fitted_dyad_games = prep.serialize_param_vectors(
                dyad_games=fitted_dyad_games, general_settings=general_settings)

        fitted_dyad_games[0]['reports'] = {
            'chooser': optimization_results.get('chooser', None),
            'predictor': optimization_results.get('predictor', None)
        }

        "Extract the parameters that minimize raw_neglog_sum from the loss report and put them within fitted_dyad_games[0]['reports']"
        raw_loss_minimizing_params = {}
        for player_role in ('predictor', 'chooser', ):
            raw_loss_minimizing_params[player_role] = {}
            param_loss_list: list[dict[str: int | float]] = copy.deepcopy(loss_report[player_role])
            if len(param_loss_list) > 0:
                if "raw_neglogprob_sum" in param_loss_list[0]:
                    param_loss_list = sorted(param_loss_list, key=lambda x: x.get('raw_neglogprob_sum', 0.0))
                    dict_with_raw_loss_minimizing_params = param_loss_list[0]
                    raw_neglogprob_sum = dict_with_raw_loss_minimizing_params["raw_neglogprob_sum"]
                    for param_key in param_info['keys']:
                        param_val = dict_with_raw_loss_minimizing_params.get(param_key, None)
                        if isinstance(param_val, (int, float)):
                            raw_loss_minimizing_params[player_role][param_key] = param_val
                if len(fitted_dyad_games) > 0:
                    if 'reports' in fitted_dyad_games[0]:
                        if player_role in fitted_dyad_games[0]['reports']:
                            if "final" in fitted_dyad_games[0]['reports'][player_role]:
                                min_raw_neglog_sum = {
                                    "params": raw_loss_minimizing_params[player_role],
                                    "loss": raw_neglogprob_sum
                                }
                                fitted_dyad_games[0]['reports'][player_role]["final"]["min_raw_neglog_sum"] = min_raw_neglog_sum

        if general_settings.get('experiment_num') == 2:
            fitted_dyad_games = typo.avatar_posteriors(
                dyad_games=fitted_dyad_games, update_method=update_method, 
                temperature=softmax_temperature)

        fitted_plr_dyads[dyad_key] = fitted_dyad_games

    def _json_safe(obj: Any) -> Any:
        """
        Recursively convert NumPy arrays/scalars and other non-serializable
        objects into JSON-safe Python types (lists, floats, ints).
        Leaves serializable types unchanged.
        """
        # NumPy arrays → lists
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # NumPy scalars → Python scalars
        if isinstance(obj, (np.floating, np.integer, np.bool_)):
            return obj.item()
        # dict → recurse
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        # list/tuple → recurse
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        # everything else unchanged
        return obj

    fitted_plr_dyads = _json_safe(fitted_plr_dyads)   #TODO directly figure out what is not serializable when I use particle filter.

    "Save the fitted results."
    with open(plr_file_path, 'w', encoding='utf-8') as file:
        json.dump(fitted_plr_dyads, file, ensure_ascii=False, indent=4)


def fit_dyad_parameters_bayes(dyad_games: DyadGames, param_info: ParamInfo, utility_settings: UtilitySettings, 
                              file_paths: FilePaths, general_settings: GeneralSettings, fit_choosers_exper_1and2: bool = False) -> list[dict[str, Any]]:
    """
    Fit parameters for each player in a dyad by optimizing the loss function over the series of games.
    
    This function can either fit the chooser and predictor parameters simultaneously or, if
    fit_roles_together is False, iteratively fit one role at a time (i.e., thawed parameters are
    optimized while frozen parameters remain fixed). This is done separately for each player.
    
    Arguments:
        • dyad_games: list[dict]; Sequence of games (meetings) between two players.
        • param_info: ParamInfo; Contains parameter keys, bounds, and initial guesses.
            Example:
                {
                    "keys": ["Vᵢᵢ", "Vᵢⱼ", "Ƹᵢⱼ", "Ʒᵢⱼ", "exp1"]  # plus '_std' and '_cov' keys if used.
                    "bounds": [(lower, upper), ...],
                    "guesses": callable or list of floats
                }
        • utility_settings: UtilitySettings; Defines utility model configuration.
        • file_paths: dict; File paths for processed data and individual dyad data.
        • general_settings: GeneralSettings;
            Various settings bundled into one dictionary.
            - experiment_num: int; Experiment identifier.
            - update_method: str; e.g., 'grid', 'parametric', etc.
            - create_new_file: bool; If False, load existing fitted data.
            - learning_rate: float; Learning rate for parameter optimization.
            - n_bins_per_dimension: int; Passed to agent() for grid dimensions.
            - sample_ratio: float; Passed to agent(); if 1, use L-BFGS-B; otherwise, use simulated annealing.
            - fit_roles_together: bool; If False, fit each player's roles separately (thaw one role at a time).
            - include_covariance: bool; If True, include covariance parameters (keys with '_cov') in the fitting.
        • fit_choosers_exper_1and2: Only if True, will fit parameters to robotic players in experiments 1 and 2.
    
    Returns:
        • dict; Updated dyad_games with fitted parameters and associated loss, as well as the best parameter estimates.
    """
    time_start_fit_dyad = time.time()
    "Determine dyad file path and attempt to load if create_new_file is False."
    first_game = dyad_games[0]
    first_choo = first_game.get('chooser')
    first_pred = first_game.get('predictor')
    if first_choo is None or first_pred is None:
        raise ValueError(f"Failed to extract player uuids from the first game: {first_game}")
    player_uuids = sorted([first_choo, first_pred])
    
    "Extract General Settings"
    sample_ratio = general_settings.get('sample_ratio', True)
    experiment_num = general_settings.get('experiment_num', 3)
    learning_rate = general_settings.get('learning_rate', True)
    update_method = general_settings.get('update_method', True)
    fit_roles_together = general_settings.get('fit_roles_together', True)
    include_covariance = general_settings.get('include_covariance', True)
    n_bins_per_dimension = general_settings.get('n_bins_per_dimension', True)
    softmax_temperature = general_settings.get('softmax_temperature', True)
    temperature_is_param = general_settings.get('temperature_is_param', True)
    
    dyad_file_path = prep._dyad_file_path(dyad_key=tuple(player_uuids), file_paths=file_paths, 
                                                 experiment_num=experiment_num, analysis_mode='bayesian')
    try:
        if not general_settings.get('create_new_file', False) and os.path.exists(dyad_file_path):
            with open(dyad_file_path, "r", encoding='utf-8') as file:
                dyad_history = json.load(file)
            if dyad_history:
                return dyad_history
    except json.decoder.JSONDecodeError as error:
        print(error)

    "--- Get Initial Parameter Guesses ---"
    initial_params = best_initial_guesses(dyad_key=tuple(player_uuids), #NOTE Check if this still works. 
                                          file_paths=file_paths,
                                          param_info=param_info, 
                                          general_settings=general_settings)

    best_params_overall = copy.deepcopy(initial_params)

    def optimize_role(player_uuid: str, role_to_fit: str, fixed_params: dict) -> dict:
        """
        Optimize the parameters for player_uuid for role role_to_fit (thawed parameters),
        while keeping the other role frozen (from fixed_params).
        
        fixed_params: dict with keys 'chooser' and 'predictor' for that player.
        Returns the updated parameter dict for that player.
        """
        def objective_function(free_param_array: NDArray[np.float64]) -> float:
            """Objective function that only modifies the thawed parameters."""
            "Build updated parameter set for this player."
            updated = copy.deepcopy(fixed_params)
            for idx, key in enumerate(free_keys):
                updated[role_to_fit][key] = free_param_array[idx]

            if include_covariance and role_to_fit == 'predictor':
                "Ensure covariance matrix is symmetric and PSD."
                huge_loss      = 1e6
                is_psd_tol     = 1e-12
                asymmetry_tol  = 1e-12
                minimum_eigval = 1e-6
                cov_matrix = gnrl.build_covariation_matrix(
                    param_info=param_info,
                    params=updated[role_to_fit],
                    raise_on_invalid=False  
                )
                altered_cov_matrix = False 
                if not gnrl.is_positive_semidefinite(matrix=cov_matrix, tol=is_psd_tol):    
                    cov_matrix = gnrl.nearest_psd_matrix(matrix=cov_matrix, min_eigval=minimum_eigval)
                    altered_cov_matrix = True

                    "Ensure that the altered matrix respects parameter boundaries."
                    param_mean_keys = [param_key for param_key in param_info['keys'] if '_std' not in param_key]
                    for idx, param_key in enumerate(param_mean_keys):
                        std_key = param_key + "_std"
                        if std_key not in param_info["keys"]:
                            continue  # no std?
                        idx_std = param_info["keys"].index(std_key)
                        (lower_bound, upper_bound) = param_info["bounds"][idx_std]

                        "'Repaired' std is square root of diagonal matrix value at (idx, idx)."
                        stdev = np.sqrt(cov_matrix[idx, idx]) 
                        if not (lower_bound <= stdev <= upper_bound):
                            if lower_bound >= stdev:
                                cov_matrix[idx, idx] = round(lower_bound ** 2, 9)
                            else:
                                cov_matrix[idx, idx] = int(upper_bound ** 2)
                            # warning_str = f"After PSD fix, stdev for param '{param_key}' = {stdev} is "
                            # warning_str += f"out of user-specified bounds [{lower_bound}, {upper_bound}]."
                            # print(warning_str)
                            # return huge_loss

                    "Ensure that the altered matrix is symmetric."
                    for idx, key1 in enumerate(param_mean_keys):
                        for jdx, key2 in enumerate(param_mean_keys[idx + 1:], start=idx + 1):
                            if abs(cov_matrix[idx][jdx] - cov_matrix[jdx][idx]) > asymmetry_tol:
                                cov_matrix[idx][jdx] = cov_matrix[jdx][idx]
                                # print(f"Asymmetry detected in covariance matrix:")
                                # print(cov_matrix)
                                # return huge_loss

                if not gnrl.is_positive_semidefinite(matrix=cov_matrix, tol=is_psd_tol):
                    print("[objective] Covariance not PSD => penalty.")
                    return huge_loss

                try:
                    param_means = [param_val for param_key, param_val in updated[role_to_fit].items() 
                                   if not any(key in param_key for key in ('_std', '_cov', 'temp'))]
                    multivariate_normal(mean=param_means, cov=cov_matrix, allow_singular=False)
                except np.linalg.LinAlgError:
                    print(f"Failed Multivariate Normal:")
                    print(cov_matrix)
                    return huge_loss
                
                if altered_cov_matrix:
                    "Updating the param dictionary with altered parameters."
                    param_mean_keys = [param_key for param_key in param_info['keys'] if '_std' not in param_key]
                    for idx, key1 in enumerate(param_mean_keys):
                        updated[role_to_fit][key1 + '_std'] = math.sqrt(cov_matrix[idx][idx])
                        for jdx, key2 in enumerate(param_mean_keys[idx + 1:], start=idx + 1):      
                            cov_key = f"{key1}_{key2}_cov"
                            updated[role_to_fit][cov_key] = cov_matrix[idx][jdx]

            "Construct a full parameter dictionary for this player."
            "For the other player, we use the initial parameters."
            full_param_dict = {player_uuid: updated}

            "For the other player, use the original guess."
            partner_uuid = [uuid for uuid in player_uuids if uuid != player_uuid][0]
            full_param_dict[partner_uuid] = initial_params[partner_uuid]
            "Run the agent simulation for the entire dyad."
            dyad_copy = copy.deepcopy(dyad_games)

            if temperature_is_param and role_to_fit == 'predictor' and update_method == 'grid':
                choice_temperature = full_param_dict[player_uuid]['predictor']['temp']
            else:
                choice_temperature = None

            "agent() is assumed to use the provided initial_params for the given player."
            updated_games = agent(dyad_games=dyad_copy,
                                  game_idx_start=0,
                                  game_idx_stop=len(dyad_copy)-1,
                                  initial_params=full_param_dict[player_uuid],
                                  param_info=param_info,
                                  utility_settings=utility_settings,
                                  player_uuid=player_uuid,
                                  player_role=role_to_fit,
                                  general_settings=general_settings,
                                  choice_temperature=choice_temperature)
            "Compute loss (using loss_function_bayes)."
            updated_games = loss_function_bayes(dyad_games=updated_games, general_settings=general_settings)
            loss_val = sum_of_all_loss(updated_games, update_method=update_method,
                                       target_player=player_uuid, target_role=role_to_fit)
            return loss_val
        
        time_start_obj_func = time.time()
        free_keys = [key for key in param_info["keys"]]
        param_bounds = [key for key in param_info["bounds"]]
        if include_covariance and role_to_fit == 'predictor':
            free_keys += [key for key in param_info["covar"]["keys"]]
            param_bounds += [key for key in param_info["covar"]["bounds"]]
        if temperature_is_param:
            free_keys += ['temp']
            param_bounds += [(0.5, 3.0)]

        "Extract initial free parameters for the active (thawed) role."
        free_initial_vector = np.array([fixed_params[role_to_fit][key] for key in free_keys])

        "Choose optimizer based on sample_ratio."
        if sample_ratio == 1:
            optimizer_method = "L-BFGS-B"
            opt_result = minimize(fun=objective_function,
                                  x0=free_initial_vector,
                                  bounds=param_bounds,
                                  method=optimizer_method,
                                  options={'maxiter': 300, 'ftol': 1e-4})
        else:
            "Otherwise, use simulated annealing."
            opt_result = dual_annealing(func=objective_function, bounds=param_bounds, maxiter=50)
        print(f"Role: {role_to_fit}")
        print(opt_result)
        best_free = opt_result.x
        "Update fixed_params with optimized values for role_to_fit."
        for idx, key in enumerate(free_keys):
            fixed_params[role_to_fit][key] = best_free[idx]

        time_stop_obj_func = time.time()
        duration = time_stop_obj_func - time_start_obj_func
        serialized_report = gnrl.serialize_opt_result(opt_result, duration=duration)
        "Store the report in the first game of the dyad under game_dict['reports'][player_uuid][role_to_fit]"
        dyad_games[0].setdefault("reports", {}).setdefault(player_uuid, {})[role_to_fit] = serialized_report

        return fixed_params

    def sum_of_all_loss(dyad_games: list, update_method: str, target_player: str, target_role: str) -> float:
        """Helper to sum losses for a given player/role."""
        total_loss = 0.0
        for game in dyad_games:
            param_data = game.get('parameter_estimates', {}).get(update_method, {})
            if target_player in param_data:
                total_loss += param_data[target_player].get(target_role, {}).get('output', {}).get('loss', 0)
        return total_loss

    "Iterate over players and roles."
    for player in player_uuids:
        if not fit_roles_together:
            if general_settings.get('experiment_num') == 3 or fit_choosers_exper_1and2:
                "First, optimize the chooser parameters for this player."
                best_params_overall[player] = optimize_role(player_uuid=player, role_to_fit="chooser", fixed_params=best_params_overall[player])  
                pp.pprint(best_params_overall[player])
            if general_settings.get('experiment_num') == 3 or first_pred == player:
                "Then, optimize the predictor parameters for this player."
                best_params_overall[player] = optimize_role(player_uuid=player, role_to_fit="predictor", fixed_params=best_params_overall[player])
                pp.pprint(best_params_overall[player])
        else:
            def objective_function_joint(free_param_array: NDArray[np.float64]) -> float:
                """Optimize both roles simultaneously."""
                updated = copy.deepcopy(best_params_overall[player])
                n_params = len(free_keys)
                for idx, key in enumerate(free_keys):
                    updated["chooser"][key] = free_param_array[idx]
                    updated["predictor"][key] = free_param_array[n_params+idx]
                full_param_dict = {player: updated}
                partner = [uuid for uuid in player_uuids if uuid != player][0]
                full_param_dict[partner] = initial_params[partner]
                dyad_copy = copy.deepcopy(dyad_games)
                updated_games = agent(dyad_games=dyad_copy,
                                      game_idx_start=0,
                                      game_idx_stop=len(dyad_copy)-1,
                                      initial_params=full_param_dict[player],
                                      param_info=param_info,
                                      utility_settings=utility_settings,
                                      player_uuid=player,
                                      general_settings=general_settings)
                updated_games = loss_function_bayes(dyad_games=updated_games, general_settings=general_settings)
                loss_val = sum_of_all_loss(updated_games, general_settings=general_settings, target_player=player, target_role="chooser")['loss_final_sum'] + \
                           sum_of_all_loss(updated_games, general_settings=general_settings, target_player=player, target_role="predictor")['loss_final_sum']
                return loss_val
            
            "Create a free vector that concatenates chooser and predictor parameters."
            free_keys = [key for key in param_info["keys"]]
            param_bounds = [key for key in param_info["bounds"]]
            free_initial_vector = np.array([best_params_overall[player]["chooser"][key] for key in free_keys] +
                                            [best_params_overall[player]["predictor"][key] for key in free_keys])
            
            if include_covariance:
                free_keys += [key for key in param_info["covar"]["keys"]]
                param_bounds += [key for key in param_info["covar"]["bounds"]]
                free_initial_vector = np.array([best_params_overall[player]["chooser"][key] 
                                                for key in free_keys] + param_info["covar"]["guesses"] +
                                                [best_params_overall[player]["predictor"][key] 
                                                 for key in free_keys] + param_info["covar"]["guesses"])

            param_bounds_joint = param_bounds * 2

            if sample_ratio == 1:
                opt_result = minimize(fun=objective_function_joint,
                                      x0=free_initial_vector,
                                      bounds=param_bounds_joint,
                                      method="L-BFGS-B",
                                      options={'maxiter': 300, 'ftol': 1e-4})
            else:
                opt_result = dual_annealing(func=objective_function_joint, 
                                            bounds=param_bounds_joint, maxiter=50)                
            best_free = opt_result.x
            print(opt_result)
            n_params = len(free_keys)
            for idx, key in enumerate(free_keys):
                best_params_overall[player]["chooser"][key] = best_free[idx]
                best_params_overall[player]["predictor"][key] = best_free[n_params+idx]
    
            dyad_games[0].setdefault("reports", {}).setdefault(
                player, {})["joint"] = gnrl.serialize_opt_result(opt_result)

    if include_covariance:
        "Ensuring covariance parameters satisfy PSD."
        for player_uuid in player_uuids:
            params = best_params_overall[player_uuid]['predictor']
            cov_matrix = gnrl.build_covariation_matrix(
                param_info=param_info,
                params=params,
                raise_on_invalid=False  
            )
            if not gnrl.is_positive_semidefinite(matrix=cov_matrix, tol=1e-12):             
                cov_matrix = gnrl.nearest_psd_matrix(matrix=cov_matrix, min_eigval=0.0)
                param_mean_keys = [param_key for param_key in param_info['keys'] if '_std' not in param_key]
                for idx, key1 in enumerate(param_mean_keys):
                    params[key1 + '_std'] = math.sqrt(cov_matrix[idx][idx])
                    for jdx, key2 in enumerate(param_mean_keys[idx + 1:], start=idx + 1):      
                        cov_key = f"{key1}_{key2}_cov"
                        params[cov_key] = cov_matrix[idx][jdx]

    "Simulate the dyad using the updated parameters."
    updated_dyad = simulate_dyad(dyad_games=dyad_games,
                                  initial_params_p1_p2=[best_params_overall[player_uuids[0]], 
                                                        best_params_overall[player_uuids[1]]],
                                  param_info=param_info,
                                  utility_settings=utility_settings,
                                  general_settings=general_settings)
    updated_dyad = loss_function_bayes(dyad_games=updated_dyad, general_settings=general_settings)
    
    "Making Numpy arrays JSON serializable."
    if update_method == 'grid':
        updated_dyad = prep.serialize_param_vectors(dyad_games=updated_dyad, general_settings=general_settings)

    "Recording the total duration for fitting the dyad."
    time_stop_fit_dyad = time.time()
    duration_fit_dyad = time_stop_fit_dyad - time_start_fit_dyad
    serialized_report = updated_dyad[0].get('reports', None)
    if serialized_report is not None:
        serialized_report['total_duration'] = duration_fit_dyad

    final_loss = 0.0
    dyad_key = prep._dyad_key(dyad_key=(first_choo, first_pred), return_tuple=True)
    for player_uuid in dyad_key:
        for player_role in ('chooser', 'predictor'):
            final_loss += sum_of_all_loss(dyad_games=updated_dyad, general_settings=general_settings, 
                                          target_player=player_uuid, target_role=player_role)['loss_final_sum']

    "Recording the final loss for fitting the dyad."
    if serialized_report is not None:
        serialized_report['final_loss'] = final_loss

    if general_settings.get('experiment_num') == 2:
        converged_on_avatar = False
        avatar_frequencies = copy.deepcopy(typo.avatar_frequencies)
        for dyad_game in updated_dyad:
            payoff_keys = [
                'payoff_A_chooser', 'payoff_A_predictor', 
                'payoff_B_chooser', 'payoff_B_predictor'
            ]
            avatar_choice = dyad_game.get('choice', None)
            if avatar_choice in ("A", "B"):
                payoffs = {payoff_key: dyad_game.get(payoff_key) for payoff_key in payoff_keys}
                avatar_frequencies = typo.bayesian_update_discrete(payoffs=payoffs, choice=avatar_choice, 
                                                            choice_frequencies_by_type=typo.choice_frequencies_by_type, 
                                                            priors=avatar_frequencies, print_=False)        
                if any(posterior == 1 for posterior in avatar_frequencies.values()):
                    converged_on_avatar = True
                param_data: dict = dyad_game.get('parameter_estimates', {}).get(update_method, {})
                param_data['optimum_update'] = {
                    'converged': converged_on_avatar,
                    'avatar_posteriors': avatar_frequencies
                }

    "Save the fitted results."
    with open(dyad_file_path, 'w', encoding='utf-8') as file:
        try:
            json.dump(updated_dyad, file, ensure_ascii=False, indent=4)
        except TypeError as error:
            print(f"Dyad not JSON serializable {error}")
            try:
                reports = updated_dyad[0].get('reports', None)
                if reports is not None:
                    del updated_dyad[0]['reports']
                json.dump(updated_dyad, file, ensure_ascii=False, indent=4)
            except TypeError as error:
                print(f"Dyad still not JSON serializable {error}")                    
    
    print(f"Processed {dyad_key} Loss: {round(final_loss, 6)} Duration: {round(duration_fit_dyad, 6)}")
    return updated_dyad


def run_analysis_bayes(histories_data: Histories, file_paths: FilePaths, param_info: ParamInfo, utility_settings: UtilitySettings, 
                       general_settings: GeneralSettings, dyads_subset: List[int] | None = None, player_uuids: List[str] | None = None, print_: bool = True) -> Histories:
    """
    Process dyads in parallel or serially, saving results as each dyad completes.
    
    Arguments:
        • histories_data: Dict;
            Contains "histories" (dyad meeting data) and "player_info" at the top level.
        • file_paths: Dict;
            File paths for parameter data and individual dyad data.
        • param_info: ParamInfo;
            Parameter information required for fitting dyads.
        • utility_settings: Dict;
            Utility model configuration options.
        • general_settings: GeneralSettings;
            Various settings bundled into one dictionary.
            - experiment_num: int;
                Experiment identifier for naming output files.
            - create_new_file: bool (default: False);
                If False, reuses existing output files to avoid redundant computation.
            - run_in_parallel: bool;
                If True (default), runs in parallel. If False, runs serially.
        • dyads_subset: List[int];
            Start and stop indices of list of dyads to iterate
            over. Useful, for running analyses in batches.

    Returns:
        • dict; The updated `histories_data` dictionary with all fitted parameters.
    """
    "Extract General Settings"
    experiment_num =  general_settings.get('experiment_num', 3)
    run_in_parallel = general_settings.get('run_in_parallel', True)
    create_new_file = general_settings.get('create_new_file', True)
    analysis_unit =   general_settings.get('analysis_unit', 'player')

    "Prepare output file paths"
    output_file = file_paths["file_names"][f"params_data_exper{experiment_num}_bayes"]
    aggregate_path = os.path.join(file_paths["param_data"], output_file)
    if analysis_unit == 'player':
        output_dir = file_paths['player_fits']
    elif analysis_unit == 'dyad':
        output_dir = file_paths["dyad_data"]
    else:
        raise ValueError(f"analysis_unit {analysis_unit} not supported.")

    os.makedirs(output_dir, exist_ok=True)

    "Check if the aggregate file already exists"
    if not create_new_file and os.path.exists(aggregate_path):
        with open(aggregate_path, "r", encoding='utf-8') as file:
            histories_data_fitted = json.load(file)
        if histories_data_fitted:
            if print_:
                print(f"Aggregate data loaded from {aggregate_path}.")
            return histories_data_fitted

    if analysis_unit == 'player':
        player_info: PlayerInfo = histories_data.get('player_info', None)
        if not player_info:
            raise Exception("No 'player_info' found in histories_data.")

        if not (isinstance(player_uuids, list) and all(isinstance(player_uuid, str) for player_uuid in player_uuids)):
            player_uuids = sorted([player_uuid for player_uuid, info in player_info.items() 
                            if info.get('player_type') == 'participant' or (experiment_num == 0 and 'predictor' in player_uuid)])  #TODO CHECK THIS!

        n_items = len(player_uuids)
        args_list = [
            (player_uuid, [], file_paths, param_info, utility_settings, general_settings) 
            for player_uuid in player_uuids
        ]

    elif analysis_unit == 'dyad':
        "Extract dyads from histories_data"
        dyads_dict = histories_data.get('histories', None)
        if not dyads_dict:
            raise Exception("No 'histories' found in histories_data.")

        "Prepare for processing"
        dyad_items = list(dyads_dict.items())

        if isinstance(dyads_subset, list) and len(dyads_subset) == 2:
            if isinstance(dyads_subset[0], int) and isinstance(dyads_subset[1], int):
                "Optinally, selecting a subset of all dyads."
                dyads_subset = [index % len(dyads_subset) for index in dyads_subset]
                dyad_items = dyad_items[dyads_subset[0]: dyads_subset[1]]

        n_items = len(dyad_items)
        args_list = [
            (dkey, meeting_list, file_paths, param_info, utility_settings, general_settings)
            for (dkey, meeting_list) in dyad_items
        ]

    else:
        raise ValueError(f"analysis_unit {analysis_unit} not supported.")

    if run_in_parallel:
        "Process players/dyads in parallel"
    
        # Decide worker count conservatively (leave one core for you)
        max_procs = max(1, mp.cpu_count() - 1)
        n_items   = len(args_list)
        n_workers = min(max_procs, n_items)
        # You can override via general_settings if you want:
        n_workers = general_settings.get('n_workers', n_workers)

        # Choose a chunksize: enough work per task to amortize overhead,
        # but small enough to keep all workers busy
        default_chunksize = max(1, math.ceil(n_items / (n_workers * 20))) if n_workers != 0 else 1
        chunksize = int(general_settings.get('mp_chunksize', default_chunksize))

        # Recycle workers periodically to curb leaks / fragmentation
        maxtasks = int(general_settings.get('maxtasksperchild', 50))

        # On Windows spawn is already the default; making it explicit is fine
        # ctx = mp.get_context("spawn")
        # with ctx.Pool(processes=n_workers,
        #               initializer=_init_worker_blas,
        #               initargs=(1, general_settings.get('random_seed', None)),
        #               maxtasksperchild=maxtasks) as pool:
        #     for idx, key_returned in enumerate(
        #             pool.imap_unordered(_worker_fit_one, args_list, chunksize=chunksize), 1):
        #         if print_:
        #             print(f"Processed {idx} / {n_items} {analysis_unit}s - {key_returned}.")

        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            for idx, key_returned in enumerate(pool.imap_unordered(_worker_fit_one, args_list), 1):
                if print_:
                    print(f"Processed {idx} / {n_items} {analysis_unit}s - {key_returned}.")

    else:
        "Process players/dyads serially"
        for idx, args in enumerate(args_list, 1):
            key_returned = _worker_fit_one(args)
            if print_:
                print(f"Processed {idx} / {n_items} {analysis_unit}s - {key_returned}.")

    "Reload all individual player/dyad files and combine into histories_data"
    count = 1
    keys_lst = list(dyads_dict.keys()) if analysis_unit == 'dyad' else player_uuids
    for key in keys_lst:
        if analysis_unit == 'dyad':
            file_path = prep._dyad_file_path(dyad_key=key, file_paths=file_paths, 
                                            experiment_num=experiment_num, analysis_mode='bayesian')
        else:
            file_path = os.path.join(file_paths["player_fits"], f"experiment_{experiment_num}", 
                                 f'{file_paths["file_name_suffix"]}_' + key + ".json")

        try:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding='utf-8') as file:
                    fitted_meeting = json.load(file)
                "Update the 'histories' key in the original data"
                if analysis_unit == 'dyad':
                    histories_data['histories'][key] = fitted_meeting
                else:
                    for uuid, fitted_dyad in fitted_meeting.items():
                        histories_data['histories'][uuid] = fitted_dyad

                if count % 10 == 0 and print_: 
                    print(f"Retrieved {count} / {n_items} {analysis_unit}s - {key}")
                count += 1

        except json.decoder.JSONDecodeError as error:
            print(error)

    for top_key in ('histories', 'player_info'):
        for key in histories_data.get(top_key, {}).keys():
            if isinstance(key, (tuple, list)):
                histories_data.get(top_key, {})[str(key)] = histories_data.get(top_key, {}).pop(key)
                print(f"Tuple key detected in histories_data['{top_key}']: {key}")    

    try:
        "Save the final combined aggregate JSON"
        with open(aggregate_path, "w", encoding='utf-8') as file:
            json.dump(histories_data, file, ensure_ascii=False, indent=4)
        if print_:
            print(f"All {analysis_unit}s processed. Final aggregate data saved to {aggregate_path}.")
    except TypeError:
        print(f"TypeError detected!")

    return histories_data


"=========================================================================================="
"========== Simulation 1) The Optimizer Accurately Recovers Predictor Parameters =========="
"=========================================================================================="

global_chooser_id = 0
def simulated_bot_uuids(n_games: int, params_predictor: dict[str, float], params_chooser: dict[str, float], 
                        predictor_id: str | None = None, chooser_id: str | None = None, k_of_2: bool = True) -> tuple[str, str]:
    """
    Constructs human-readable UUID-like strings for artificial chooser–predictor dyads.

    These strings embed the “true” simulation parameters (means, standard deviations, and
    temperature) into the player identifiers so that downstream analysis can recover the
    ground-truth parameter values directly from the filenames or player_ids.  This is
    especially useful when validating that the optimizer recovers the parameters used to
    generate simulated data.

    When `k_of_2` is True, UUIDs are tailored to the 2-parameter (Vᵢᵢ, Vᵢⱼ) simulation, and
    the IDs for chooser and predictor each encode *both* agents’ parameters for convenience.
    When `k_of_2` is False, UUIDs are constructed generically from the full contents of
    `params_chooser` and `params_predictor`, with a global counter appended to keep them unique.

    Arguments:
        • n_games: int;
            Number of binary dictator games that this dyad will play together. Used to embed
            the length of the simulated interaction in the UUIDs (e.g., `_n=25`).
        • params_predictor: dict[str, float];
            Dictionary of predictor parameters. For the 2-parameter simulations, this must
            contain (at minimum):
                - 'Vᵢᵢ': Mean self-interest parameter for the predictor.
                - 'Vᵢⱼ': Mean altruism parameter for the predictor.
                - 'Vᵢᵢ_std': Standard deviation over self-interest.
                - 'Vᵢⱼ_std': Standard deviation over altruism.
                - 'τ': SoftMax temperature for the predictor.
        • params_chooser: dict[str, float];
            Dictionary of chooser parameters with the same keys as `params_predictor` but
            describing the agent who *generates* choices rather than predictions.
        • predictor_id: str | None;
            Optional explicit identifier for the predictor. Only used when `k_of_2` is False.
            If provided, the predictor UUID will be of the form:
                'robot_predictor_by_k_id=<predictor_id>_n=<n_games>'
        • chooser_id: str | None;
            Optional explicit identifier for the chooser. If provided, the chooser UUID will
            use this value instead of the global counter. If None, `global_chooser_id` is used
            and then incremented.
        • k_of_2: bool;
            If True, assume a 2-parameter (Vᵢᵢ, Vᵢⱼ) simulation and build compact UUID strings
            that hard-code the pair (mean, std) for both players plus τ and n_games. If False,
            iterate over all parameter keys and embed each as key=value in the UUID strings.

    Returns:
        • tuple[str, str];
            A tuple (predictor_uuid, chooser_uuid) where each element is a string identifier
            suitable for use as a player_uuid in the simulated histories and as the basename
            for JSON files storing fitted results.

            Examples (k_of_2 = True):
                predictor_uuid = 'robot_predictor_Vii=(1.0,1.0)_Vij=(-1.0,1.0)_t=1.5_n=25~...'
                chooser_uuid   = 'robot_chooser_Vii=(1.0,1.0)_Vij=(-1.0,1.0)_t=1.5_n=25~...'

    Raises:
        • TypeError:
            If `n_games` is not an integer.
        • ValueError:
            If `n_games` is not strictly positive.

    Notes:
        • Greek and subscript characters in parameter keys are sanitized (e.g., 'Vᵢᵢ' → 'Vii',
          'Vᵢⱼ_std' → 'ViJs', 'τ' → 't') when `k_of_2` is False to keep UUIDs filename-safe.
        • The `~...` “HACK” suffix links each UUID to its counterpart’s coarse parameter
          settings, which can be useful when parsing filenames without opening the JSON.
    """
    if not isinstance(n_games, int):
        raise TypeError(f"n_games must be an integer not {type(n_games)} - {n_games}.")
    if not n_games > 0:
        raise ValueError(f"n_games must be greater than 0, not {n_games}.")

    def safe_param_key(param_key):
        for unsafe, safe in [('ᵢ', 'i'), ('ⱼ', 'j'), ('Ƹ', 'E'), ('Ʒ', 'G'), ('Ʌ', 'N'), 
                             ('γ', 'e'), ('τ', 't'), ('_std', 's'), ('_cov', 'c'),]: # ('', ''), ('', ''), 
            param_key = param_key.replace(unsafe, safe)
        return param_key

    global global_chooser_id
    local_chooser_id = chooser_id if isinstance(chooser_id, str) else str(global_chooser_id)

    if k_of_2:
        Vᵢᵢ_chooser =     params_chooser.get("Vᵢᵢ")
        Vᵢⱼ_chooser =     params_chooser.get("Vᵢⱼ")
        Vᵢᵢ_std_chooser = params_chooser.get("Vᵢᵢ_std")
        Vᵢⱼ_std_chooser = params_chooser.get("Vᵢⱼ_std")
        τ_chooser =       params_chooser.get("τ")

        Vᵢᵢ_predictor =     params_predictor.get("Vᵢᵢ")
        Vᵢⱼ_predictor =     params_predictor.get("Vᵢⱼ")
        Vᵢᵢ_std_predictor = params_predictor.get("Vᵢᵢ_std")
        Vᵢⱼ_std_predictor = params_predictor.get("Vᵢⱼ_std")
        τ_predictor =       params_predictor.get("τ")

        chooser_uuid = f"robot_chooser_Vii=({round(Vᵢᵢ_chooser, 2)},{round(Vᵢᵢ_std_chooser, 2)})_Vij="
        chooser_uuid += f"({round(Vᵢⱼ_chooser, 2)},{round(Vᵢⱼ_std_chooser, 2)})_t={round(τ_chooser, 2)}_n={n_games}"
        # chooser_uuid += f"~{int(Vᵢᵢ_predictor)}{int(Vᵢⱼ_predictor)}{round(Vᵢⱼ_std_predictor, 2)}{round(τ_predictor, 2)}" #HACK

        # --- Toggle: set False to revert to old behavior quickly ---
        UNIQUE_PLAYERS_PER_DYAD = True

        chooser_uuid += (
            f"~{int(Vᵢᵢ_predictor)}{int(Vᵢⱼ_predictor)}"
            f"{round(Vᵢⱼ_std_predictor, 2)}{round(τ_predictor, 2)}"
        )  # HACK

        # Make chooser unique per dyad (matches predictor behavior)
        if UNIQUE_PLAYERS_PER_DYAD:
            chooser_uuid += f"_{global_chooser_id}"

        predictor_uuid = f"robot_predictor_Vii=({round(Vᵢᵢ_predictor, 2)},{round(Vᵢᵢ_std_predictor, 2)})_Vij="
        predictor_uuid += f"({round(Vᵢⱼ_predictor, 3)},{round(Vᵢⱼ_std_predictor, 2)})_t={round(τ_predictor, 2)}_n={n_games}"
        # predictor_uuid += f"~{int(Vᵢᵢ_chooser)}{int(Vᵢⱼ_chooser)}{round(Vᵢⱼ_std_chooser, 2)}{round(τ_chooser, 2)}" #HACK
        predictor_uuid += f"~{int(Vᵢᵢ_chooser)}{int(Vᵢⱼ_chooser)}{round(Vᵢⱼ_std_chooser, 2)}{round(τ_chooser, 2)}_{global_chooser_id}" #HACK
        
    else:    
        chooser_uuid = "robot_chooser_by_k_"
        for param_key, param_val in params_chooser.items():
            chooser_uuid += f"{safe_param_key(param_key)}={param_val:.2f}_"
        chooser_uuid += f"n={n_games}_{local_chooser_id}"

        if predictor_id is not None:
            predictor_uuid = f"robot_predictor_by_k_id={predictor_id}_n={n_games}"
        else:
            predictor_uuid = "robot_predictor_by_k_"
            for param_key, param_val in params_predictor.items():
                predictor_uuid += f"{safe_param_key(param_key)}={param_val:.2f}_"
            predictor_uuid += f"n={n_games}_{local_chooser_id}"

    if chooser_id is None:
        global_chooser_id += 1

    return (predictor_uuid, chooser_uuid)


def create_simulated_dyad(n_games: int, params_chooser: dict[str, float], params_predictor: dict[str, float], general_settings: GeneralSettings,
                          utility_settings: UtilitySettings, param_bds: ParamBounds, payoff_structures: list[dict[str, int]] | None = None, 
                          default_utility_settings: bool = True, embed_true_params: bool = False, dynamic_predictor: bool = True) -> dict[DyadKey, DyadGames]:
    """
    Create a single synthetic chooser–predictor dyad with recorded choices and predictions.

    Purpose:
        This generator is used for simulation studies where we want ground-truth chooser behavior
        and a predictor who forms beliefs about that chooser. Each "game" is a single binary choice
        between option A and B; the chooser’s response is drawn/selected by `choice(...)` using
        `params_chooser`, while the predictor forms a prediction using `params_predictor`. The result
        is a list of game dictionaries that can be fed into `agent(...)` to perform learning.

    Arguments:
        • n_games: int;
            Number of games to simulate.
        • params_chooser: dict[str, float];
            Parameters controlling the chooser’s utility (weights, exponents, etc.).
            If you use a temperature parameter for the chooser, set both:
                • 'temp' for `agent(...)` style compatibility (if ever needed),
                • 'τ'   for the choice(...) call used here (SoftMax temperature).
        • params_predictor: dict[str, float];
            Parameters controlling the predictor’s *initial* beliefs (e.g., means/stds when later
            used to seed a grid prior). Similarly, include both 'temp' and 'τ' if you want consistent
            SoftMax behavior across the codebase.
        • utility_settings: UtilitySettings;
            The utility family under which the chooser and predictor operate.
            If `default_utility_settings=True`, a simple default family is used and this argument is ignored.
        • payoff_structures: list[dict[str,int]] | None;
            Optional explicit payoff sequence; each item must provide:
                {'payoff_A_chooser','payoff_A_predictor','payoff_B_chooser','payoff_B_predictor'}.
            If None, payoffs are drawn uniformly from {1..5}.
        • default_utility_settings: bool;
            If True (default), use the built-in baseline utility settings (altruism only, etc.).
            If False, use the caller-provided `utility_settings`.
        • embed_true_params: bool; 
            If True, saves the true params in the dyads so that they are easier to find later.
        • dynamic_predictor: bool;
            If True, runs the full UBM via agent() for predictors, meaning belief updating.
            If False, runs choice() for predictors, meaning no belief updating.

    Returns:
        • dict[DyadKey, DyadGames]
            A dictionary with a single key "(predictor_uuid, chooser_uuid)" whose value 
            is the list of games. Each game dictionary contains (among other fields):
                'chooser', 'predictor', 'payoff_A_chooser', 'payoff_A_predictor',
                'payoff_B_chooser', 'payoff_B_predictor', 'choice', 'prediction', and 'round'.

    Notes:
        • The chooser’s and predictor’s responses are generated with `choice(..., select=True)`.
            If your `choice` implementation samples stochastically, fix RNG seeds upstream for reproducibility.
    """
    if not isinstance(n_games, int):
        raise TypeError(f"n_games must be an integer not {type(n_games)} - {n_games}.")
    if not n_games > 0:
        raise ValueError(f"n_games must be greater than 0, not {n_games}.")

    def _inject_temp_alias(d: dict[str, float]) -> dict[str, float]:
        d = dict(d)  # shallow copy
        if "temp" not in d and "τ" in d:
            d["temp"] = float(d["τ"])
        return d

    utility_settings_: UtilitySettings = {
        'conditional_welfare_mode':       False,
        'reference_dependent_altruism':   False,
        'min_max_rawlsian_leontief':      False,
        'use_exponential_parameters':     False,
        'apply_exponents_to_payoffs':     False,
        'single_exponential_parameter':   False,
        'single_payoffs_not_differences': False,
        'payoff_ratios_not_differences':  False,
        'reference_dependent_utility':    False,
        'use_negativity_parameters':      False,
        'negativity_social_comparison':   False,
        'fix_self_interest_parameter':    False,
        'include_social_comparison':      False,
        'include_altruism_term':          True,
    }
    if not default_utility_settings:
        utility_settings_ = copy.deepcopy(utility_settings)

    τ_chooser =   params_chooser.get("τ")
    τ_predictor = params_predictor.get("τ")

    predictor_uuid, chooser_uuid = simulated_bot_uuids(n_games=n_games, params_predictor=params_predictor, 
                                                       params_chooser=params_chooser, k_of_2=not embed_true_params)
    dyad_key = f"({predictor_uuid}, {chooser_uuid})"

    dyad_games = []
    for game_idx in range(n_games):
        if payoff_structures is not None and game_idx <= len(payoff_structures):
            payoff_A_chooser =   payoff_structures[game_idx].get('payoff_A_chooser')
            payoff_A_predictor = payoff_structures[game_idx].get('payoff_A_predictor')
            payoff_B_chooser =   payoff_structures[game_idx].get('payoff_B_chooser')
            payoff_B_predictor = payoff_structures[game_idx].get('payoff_A_predictor')
            if any(payoff is None for payoff in (payoff_A_chooser, payoff_A_predictor, payoff_B_chooser, payoff_B_predictor)):
                raise ValueError(f"Payoff structure improperly formatted: {payoff_structures[game_idx]}.")

        else:
            payoff_A_chooser =   random.randint(1, 5)
            payoff_A_predictor = random.randint(1, 5)
            payoff_B_chooser =   random.randint(1, 5)
            payoff_B_predictor = random.randint(1, 5)

        payoffs = {
            'payoff_A_chooser': payoff_A_chooser, 'payoff_A_predictor': payoff_A_predictor,
            'payoff_B_chooser': payoff_B_chooser, 'payoff_B_predictor': payoff_B_predictor,
        }

        choice_response =     choice(current_game=payoffs, agent_params=params_chooser, 
                                     utility_settings=utility_settings_, softmax_temperature=τ_chooser,
                                     select=True)["model_choose_A"]
        choice_response = "A" if choice_response == 1 else "B"

        if dynamic_predictor:
            prediction_response = None
        else:
            prediction_response = choice(current_game=payoffs, agent_params=params_predictor, 
                                        utility_settings=utility_settings_, softmax_temperature=τ_predictor,
                                        select=True)["model_choose_A"]
            prediction_response = "A" if prediction_response == 1 else "B"

        dyad_game = {
            "chooser": chooser_uuid,
            "predictor": predictor_uuid,
            "matching_probability": 1.0,
            "payoff_A_chooser": payoff_A_chooser,
            "payoff_A_predictor": payoff_A_predictor,
            "payoff_B_chooser": payoff_B_chooser,
            "payoff_B_predictor": payoff_B_predictor,
            "choice": choice_response,
            "prediction": prediction_response,
            "abdicated_chooser": False,
            "abdicated_predictor": False,
            "timestamp": time.time(),
            "round": game_idx,            
        }

        if embed_true_params and game_idx == 0:
            dyad_game["true_params_predictor"] = _inject_temp_alias(params_predictor)
            dyad_game["true_params_chooser"]   = _inject_temp_alias(params_chooser)

        dyad_games.append(dyad_game)

    if dynamic_predictor:
        "Use UBM with belief updating for predictors, overwriting previous choices"
        param_info_ = make_param_info(param_bds=param_bds, utility_settings=utility_settings_, 
                                      general_settings=general_settings, guess_seed=None, random_guesses_are_unique=True)
        params_predictor = {param_key: param_val for param_key, param_val in params_predictor.items() if param_key not in ('τ', 'temp')}
        dyad_games = agent(dyad_games=dyad_games, game_idx_start=0, game_idx_stop=n_games - 1, general_settings=general_settings, 
                           utility_settings=utility_settings_, param_info=param_info_, initial_params={'predictor': params_predictor}, 
                           player_uuid=predictor_uuid, player_role="predictor", select=True, choice_temperature=τ_predictor)
        
        "Make param vectors JSON serializable"
        dyad_games = prep.serialize_param_vectors(dyad_games=dyad_games, general_settings=general_settings)

        "Move parameter updates so they will not be overwritten by the optimizer during the parameter recovery process."
        update_method = general_settings.get('update_method', 'grid')
        for game_idx in range(n_games):
            dyad_game = dyad_games[game_idx]
            param_est = dyad_game.get('parameter_estimates')
            if not param_est:
                raise Exception("Predictor failed to record parameter updates. No 'parameter_estimates' stored in game.")

            if update_method not in param_est:
                raise Exception(f"Predictor failed to record parameter updates. '{update_method}' not in 'parameter_estimates'.")

            "Move the whole block to 'sim_pred' so optimizer later writes fresh 'grid'"
            param_est['sim_pred'] = param_est.pop(update_method)

    return {dyad_key: dyad_games}


def create_simulated_data(n_games: int, params_chooser_range: dict[str, float], params_predictor_range: dict[str, float], utility_settings: UtilitySettings,
                          param_bds: Dict[str, tuple[int | float, int | float]] | None = None, file_paths: FilePaths | None = None,
                          payoff_structures: list[dict[str, int]] | None = None, run_analysis: bool = True, dynamic_predictor: bool = True,
                          randomize_parameters: bool = True, max_iter: int = 1000) -> dict[DyadKey, DyadGames]:
    """
    Generate a grid (or randomized grid) of artificial chooser–predictor dyads and optionally
    run the full Bayesian analysis pipeline on the resulting synthetic dataset.

    Conceptually, this function:
        1. Expands parameter ranges for choosers and predictors into discrete values
           (either evenly spaced or random within each interval).
        2. Creates one dyad for every combination of chooser and predictor parameters.
        3. Simulates `n_games` binary dictator games for each dyad via `create_simulated_dyad`.
        4. Aggregates all dyads into a `player_histories` structure compatible with the
           standard analysis functions.
        5. Optionally writes the histories to disk and calls `run_analysis_bayes(...)` to
           verify that the optimizer recovers the known parameters.

    Arguments:
        • n_games: int;
            Number of games simulated per dyad (per chooser–predictor pair).
        • params_chooser_range: dict[str, float];
            Parameter grid specification for choosers. Each entry must be a length-3 tuple:
                (lower_bound, upper_bound, n_points)
            for keys:
                - 'Vᵢᵢ': Range for chooser self-interest means.
                - 'Vᵢⱼ': Range for chooser altruism means.
                - 'std':  Range for chooser standard deviations (applied to both Vᵢᵢ and Vᵢⱼ).
                - 'τ':    Range for chooser SoftMax temperatures.
        • params_predictor_range: dict[str, float];
            Parameter grid specification for predictors, same structure as `params_chooser_range`.
            Each combination of predictor parameters is paired with each combination of
            chooser parameters to form a dyad.
        • utility_settings: UtilitySettings;
            Base utility settings to be used for the simulated choice and prediction calls.
            A copy is internally created with all booleans turned off except
            `'include_altruism_term': True`, unless `run_analysis` modifies this downstream.
        • param_bds: dict[str, tuple[int | float, int | float]] | None;
            Bounds for the optimizer parameters, passed through to `make_param_info(...)`
            when `run_analysis=True`. Required if `run_analysis` is True.
        • file_paths: FilePaths | None;
            File path configuration dictionary used for saving simulated histories and directing
            the analysis results. Required if `run_analysis` is True.
        • payoff_structures: list[dict[str, int]] | None;
            Optional externally specified payoff sequence for all dyads. If provided, the same
            payoff list is reused for each dyad and `create_simulated_dyad` is called with
            `payoff_structures=payoff_structures`. If None, payoffs are sampled uniformly
            from {1,…,5} per game.
        • run_analysis: bool;
            If True, writes the simulated histories to disk and calls `run_analysis_bayes(...)`
            using the provided `param_bds`, `file_paths`, and `utility_settings`. If False,
            only returns the in-memory `player_histories` dict.
        • dynamic_predictor: bool;
            If True, runs the full UBM via agent() for predictors, meaning belief updating.
            If False, runs choice() fro predictors, meaning no belief updating.
        • randomize_parameters: bool;
            If True (default), parameter grids are populated by sampling uniform random values
            within each (min, max) range. If False, use evenly spaced linspace grids across
            each interval.
        • max_iter: int;
            Safety cap on the total number of dyads (i.e., the product of all grid resolutions).
            If the implied number of dyads exceeds `max_iter`, an Exception is raised to avoid
            accidentally launching enormous simulations.

    Returns:
        • dict[DyadKey, DyadGames];
            If `run_analysis` is False:
                - Returns a dictionary with keys:
                    'histories':   dict[DyadKey, DyadGames];
                        Mapping from dyad keys "(predictor_uuid, chooser_uuid)" to their list
                        of game dictionaries.
                    'player_info': dict[str, dict];
                        Mapping from player_uuid → avatar metadata for plotting/inspection.
            If `run_analysis` is True:
                - Returns the same structure after writing it to disk and running the Bayesian
                  analysis pipeline. The analysis results are saved to the locations specified
                  by `file_paths`.

    Raises:
        • Exception:
            - If `run_analysis=True` and either `param_bds` or `file_paths` is None.
            - If the implied number of dyads `n_iters` exceeds `max_iter`.
        • Any downstream exceptions raised by `create_simulated_dyad`, file I/O, or
          `run_analysis_bayes(...)`.

    Notes:
        • Parameter ranges are expanded in the following order:
            Vᵢᵢ_predictor × Vᵢⱼ_predictor × std_predictor × τ_predictor ×
            Vᵢᵢ_chooser × Vᵢⱼ_chooser × std_chooser × τ_chooser
          so the outer loops vary predictor parameters first, then chooser parameters.
        • Each unique player UUID is assigned a random avatar shape and color sample (HLSA)
          to make simulated players visually distinct in the UI and diagnostic plots.
        • When `run_analysis=True`, `global_chooser_id` is reset to 0 after the analysis.
    """
    if run_analysis:
        if param_bds is None:
            raise Exception("param_bds cannot be None if run_analysis.")
        if file_paths is None:
            raise Exception("file_paths cannot be None if run_analysis.")

    if dynamic_predictor and param_bds is None:
        raise ValueError(f"param_bds cannot be None if dynamic_predictor is True.")

    general_settings_ = {
        'experiment_num': 0,
        'run_in_parallel': True,
        'track_evolution': False,
        'create_new_file': True,
        'update_method': 'grid',
        'analysis_mode': 'bayesian',
        'analysis_unit': 'player',
        'n_bins_per_dimension': 9,
        'include_covariance': False,
        'softmax_temperature': 1.5,
        'temperature_is_param': True,
        'guess_params_randomly': False,
        'optimization_method': 'globloc',
        'confidence_weighted': True,
        'use_particle_filter': True,
        'fit_roles_together': False,
        'use_initial_params': True,
        'loss_funct_type': 'log',
        'penalty_weight': 0.05,
        'learning_rate': 0.8,
        'sample_ratio': 1.0,
        'export_fig': True,
        'dark_mode': True
    }

    utility_settings_: UtilitySettings = {setting_key: False for setting_key in utility_settings.keys()}
    utility_settings_['include_altruism_term'] = True

    Vᵢᵢ_chooser_range =   params_chooser_range.get("Vᵢᵢ")
    Vᵢⱼ_chooser_range =   params_chooser_range.get("Vᵢⱼ")
    std_chooser_range =   params_chooser_range.get("std")
    τ_chooser_range =     params_chooser_range.get("τ")

    Vᵢᵢ_predictor_range = params_predictor_range.get("Vᵢᵢ")
    Vᵢⱼ_predictor_range = params_predictor_range.get("Vᵢⱼ")
    std_predictor_range = params_predictor_range.get("std")
    τ_predictor_range =   params_predictor_range.get("τ")

    n_iters = 1
    for range_ in (
            Vᵢᵢ_chooser_range,   Vᵢⱼ_chooser_range,   std_chooser_range,   τ_chooser_range, 
            Vᵢᵢ_predictor_range, Vᵢⱼ_predictor_range, std_predictor_range, τ_predictor_range
        ):
        n_iters *= range_[2]
    
    if n_iters > max_iter:
        raise Exception(f"Runtime Warning! Was about to generate {n_iters} dyads!")
    print(f"Generating simulation with {n_iters} artificial agents.")

    if randomize_parameters:
        Vᵢᵢ_chooser_intervals =   [round(random.uniform(Vᵢᵢ_chooser_range[0], Vᵢᵢ_chooser_range[1]), 4) for num in range(Vᵢᵢ_chooser_range[2])]
        Vᵢⱼ_chooser_intervals =   [round(random.uniform(Vᵢⱼ_chooser_range[0], Vᵢⱼ_chooser_range[1]), 4) for num in range(Vᵢⱼ_chooser_range[2])]
        τ_chooser_intervals =     [round(random.uniform(τ_chooser_range[0],   τ_chooser_range[1]), 4)   for num in range(τ_chooser_range[2])]
        std_chooser_intervals =   [round(random.uniform(std_chooser_range[0], std_chooser_range[1]), 4) for num in range(std_chooser_range[2])]
 
        Vᵢᵢ_predictor_intervals = [round(random.uniform(Vᵢᵢ_predictor_range[0], Vᵢᵢ_predictor_range[1]), 4) for num in range(Vᵢᵢ_predictor_range[2])]
        Vᵢⱼ_predictor_intervals = [round(random.uniform(Vᵢⱼ_predictor_range[0], Vᵢⱼ_predictor_range[1]), 4) for num in range(Vᵢⱼ_predictor_range[2])]
        τ_predictor_intervals =   [round(random.uniform(τ_predictor_range[0],   τ_predictor_range[1]), 4)   for num in range(τ_predictor_range[2])]
        std_predictor_intervals = [round(random.uniform(std_predictor_range[0], std_predictor_range[1]), 4) for num in range(std_predictor_range[2])]
        
    else:
        Vᵢᵢ_chooser_intervals =   list(np.round(np.linspace(start=Vᵢᵢ_chooser_range[0], stop=Vᵢᵢ_chooser_range[1], num=Vᵢᵢ_chooser_range[2]), decimals=4))
        Vᵢⱼ_chooser_intervals =   list(np.round(np.linspace(start=Vᵢⱼ_chooser_range[0], stop=Vᵢⱼ_chooser_range[1], num=Vᵢⱼ_chooser_range[2]), decimals=4))
        τ_chooser_intervals =     list(np.round(np.linspace(start=τ_chooser_range[0],   stop=τ_chooser_range[1],   num=τ_chooser_range[2]),   decimals=4))
        std_chooser_intervals =   list(np.round(np.linspace(start=std_chooser_range[0], stop=std_chooser_range[1], num=std_chooser_range[2]), decimals=4))
        
        Vᵢᵢ_predictor_intervals = list(np.round(np.linspace(start=Vᵢᵢ_predictor_range[0], stop=Vᵢᵢ_predictor_range[1], num=Vᵢᵢ_predictor_range[2]), decimals=4))
        Vᵢⱼ_predictor_intervals = list(np.round(np.linspace(start=Vᵢⱼ_predictor_range[0], stop=Vᵢⱼ_predictor_range[1], num=Vᵢⱼ_predictor_range[2]), decimals=4))
        τ_predictor_intervals =   list(np.round(np.linspace(start=τ_predictor_range[0],   stop=τ_predictor_range[1],   num=τ_predictor_range[2]),   decimals=4))
        std_predictor_intervals = list(np.round(np.linspace(start=std_predictor_range[0], stop=std_predictor_range[1], num=std_predictor_range[2]), decimals=4))

    player_histories = {}
    "Iterate over predictor parameters"
    for Vᵢᵢ_predictor in Vᵢᵢ_predictor_intervals:
        for Vᵢⱼ_predictor in Vᵢⱼ_predictor_intervals:
            for std_predictor in std_predictor_intervals:
                for τ_predictor in τ_predictor_intervals:    
                    "Iterate over chooser parameters"
                    for Vᵢᵢ_chooser in Vᵢᵢ_chooser_intervals:
                        for Vᵢⱼ_chooser in Vᵢⱼ_chooser_intervals:
                            for std_chooser in std_chooser_intervals:
                                for τ_chooser in τ_chooser_intervals:

                                    "Create 'params' variables for both player roles."
                                    params_chooser =   {'Vᵢᵢ': Vᵢᵢ_chooser,   'Vᵢⱼ': Vᵢⱼ_chooser,   'Vᵢᵢ_std': std_chooser,   'Vᵢⱼ_std': std_chooser,   'τ': τ_chooser}
                                    params_predictor = {'Vᵢᵢ': Vᵢᵢ_predictor, 'Vᵢⱼ': Vᵢⱼ_predictor, 'Vᵢᵢ_std': std_predictor, 'Vᵢⱼ_std': std_predictor, 'τ': τ_predictor}

                                    "Generate the series of games played between these artificial agents."
                                    player_dyad = create_simulated_dyad(n_games=n_games, params_chooser=params_chooser, params_predictor=params_predictor, 
                                                                        utility_settings=utility_settings_, general_settings=general_settings_,
                                                                        payoff_structures=payoff_structures, param_bds=param_bds, dynamic_predictor=dynamic_predictor)
                                    
                                    "update player_histories with {DyadKey: DyadGames} dictionary."
                                    player_histories.update(player_dyad)

    avatar_shapes = [
        "arrow-head",
        "bowtie",
        "circle",
        "cross",
        "curvy-x",
        "dent-square",
        "dodecagon",
        "flame",
        "flower",
        "ghost",
        "hexagon",
        "hour-glass",
        "jagged-sun",
        "lemon",
        "moon",
        "pentagon",
        "round-square",
        "squash",
        "teardrop",
        "two-triangle",
        "star-six",
        "stop-sign"
    ]

    player_info = {}
    for dyad_key in player_histories.keys():
        plr_uuid_1, plr_uuid_2 = dyad_key[1:-1].split(", ")
        for plr_uuid in (plr_uuid_1, plr_uuid_2):
            player_info[plr_uuid] = {
                'player_type': 'robot', 'avatar_shape': avatar_shapes[random.randint(0, len(avatar_shapes) - 1)],
                'player_color': f'hlsa({random.randint(0, 359)}, {random.randint(35, 65)}%, {random.randint(35, 65)}%, 1.0)'
            }

    player_histories = {'histories': player_histories, 'player_info': player_info}

    if run_analysis:
        param_info_ = make_param_info(param_bds=param_bds, utility_settings=utility_settings_, general_settings=general_settings_, 
                                                 random_guesses_are_unique=not general_settings_['run_in_parallel'])

        # Create file name suffix from these settings.
        file_name_suffix = prep.create_file_name_suffix(
            general_settings=general_settings_, utility_settings=utility_settings_
        )

        # Copy of standard file paths to alter with each loop.
        file_paths_ = copy.deepcopy(file_paths)

        # Remove suffix from file names if any.
        file_paths_ = prep.add_remove_file_name_suffix(
            file_paths=file_paths_, file_name_suffix=None, add_suffix=False
        )

        # Re-add that suffix to file_paths.
        file_paths_ = prep.add_remove_file_name_suffix(
            file_paths=file_paths_, file_name_suffix=file_name_suffix, add_suffix=True
        )

        histories_file_path = os.path.join(file_paths['processed'], file_paths['file_names'][f'player_pairs_exper0'])

        with open(histories_file_path, 'w', encoding='utf-8') as file:
            json.dump(player_histories, file, ensure_ascii=False, indent=4)
            print(f"Saved simulated histories to {histories_file_path}.")
 
        histories_info = player_histories['player_info']
        # print(len(list(histories_info.keys()))), exit()

        run_analysis_bayes(histories_data=player_histories, file_paths=file_paths_, param_info=param_info_, 
                           utility_settings=utility_settings_, general_settings=general_settings_)
        global_chooser_id = 0
    return player_histories


def parse_robot_string(robot_str: str) -> dict:
    """
    Extract "true" parameters from a player UUID string like:
        'robot_predictor_Vii=(1.0,1.0)_Vij=(-1.0,1.0)_t=1.5_n=9'
    Returns a dict of parameter values, e.g.:
        {
          "Vᵢᵢ": 1.0,
          "Vᵢᵢ_std": 1.0,
          "Vᵢⱼ": -1.0,
          "Vᵢⱼ_std": 1.0,
          "temp": 1.5,
          "n_games": 9
        }
    """
    pattern_vii = r"Vii=\((-?\d+\.?\d*),(-?\d+\.?\d*)\)"
    pattern_vij = r"Vij=\((-?\d+\.?\d*),(-?\d+\.?\d*)\)"
    pattern_t   = r"_t=(-?\d+\.?\d*)"
    pattern_n   = r"_n=(\d+)"

    match_vii = re.search(pattern_vii, robot_str)
    match_vij = re.search(pattern_vij, robot_str)
    match_t   = re.search(pattern_t,   robot_str)
    match_n   = re.search(pattern_n,   robot_str)

    parsed = {}
    if match_vii:
        parsed["Vᵢᵢ"]     = float(match_vii.group(1))
        parsed["Vᵢᵢ_std"] = float(match_vii.group(2))
    if match_vij:
        parsed["Vᵢⱼ"]     = float(match_vij.group(1))
        parsed["Vᵢⱼ_std"] = float(match_vij.group(2))
    if match_t:
        parsed["temp"]    = float(match_t.group(1))
    if match_n:
        parsed["n_games"] = int(match_n.group(1))

    return parsed


def get_simulated_dyad(file_paths: FilePaths, dyad_idx: int | None, n_games: int, params_predictor: Optional[Dict[str, float]] = None, 
                       params_chooser: Optional[Dict[str, float]] = None) -> Dict[DyadKey, DyadGames]:
    """
    Load a single simulated dyad (chooser–predictor game history) from disk.

    This is a convenience loader used when inspecting the optimizer’s recovery performance
    on a particular artificial dyad. It supports two ways of selecting which dyad to load:

        1. Direct filename reconstruction:
            If both `params_predictor` and `params_chooser` are provided, the function
            calls `simulated_bot_uuids(...)` to reconstruct the predictor UUID and then
            loads the corresponding `<predictor_uuid>.json` file from `json_path`.

        2. Index-based selection:
            If parameter dictionaries are not provided, the function defers to
            `prep.get_file_by_index_or_name(...)` using `dyad_idx` to choose 
            a file from `json_path` (e.g., “the k-th JSON file in that directory”).

    Arguments:
        • file_paths: dict[str: str | dict[str: str]];
            Dictionary of all files paths in this project.    
        • dyad_idx: int | None;
            Index of the dyad JSON file to load when `params_predictor` and `params_chooser`
            are not supplied. Passed through to `prep.get_file_by_index_or_name` as
            the `file_name_idx` argument. Ignored if `file_name` is determined by
            `simulated_bot_uuids(...)`.
        • n_games: int;
            Number of games that were simulated for this dyad. Used only when reconstructing
            the UUID via `simulated_bot_uuids(...)` (so that the filename matches exactly).
        • params_predictor: dict[str, float] | None;
            Parameter dictionary for the predictor used during simulation. When provided
            together with `params_chooser`, this is used to recreate the predictor UUID and
            thus the JSON filename. If None, dyad selection falls back to `dyad_idx`.
        • params_chooser: dict[str, float] | None;
            Parameter dictionary for the chooser used during simulation. Only used to
            reconstruct the UUID when `params_predictor` is also provided. If None, dyad
            selection falls back to `dyad_idx`.

    Returns:
        • dict[DyadKey, DyadGames];
            A dictionary mapping a single DyadKey "(predictor_uuid, chooser_uuid)" to the
            list of dyad games recovered from disk. This is the same format returned by
            `create_simulated_dyad` and expected by downstream analysis code.

    Raises:
        • Exception:
            - If a filename cannot be determined from the provided arguments.
            - If the resolved file does not exist at `json_path`.
            - If `prep.get_file_by_index_or_name` fails to return a valid filename.

    Notes:
        • When both parameter dictionaries are supplied, they must match the ones used to
          generate the original dyad; otherwise the reconstructed filename will not exist.
        • This loader is read-only: it does not modify or re-simulate any dyads, it simply
          deserializes a previously saved JSON file.
    """
    file_name = None
    if params_predictor is not None and params_chooser is not None:
        predictor_uuid, chooser_uuid = simulated_bot_uuids(n_games=n_games, params_predictor=params_predictor, params_chooser=params_chooser)
        file_name = predictor_uuid + ".json"

    if file_name is None:
        file_name = prep.get_file_by_index_or_name(directory_path=os.path.join(file_paths['player_fits'], 'experiment_0'), file_name_idx=dyad_idx, file_name=file_name)

    if file_name is None:
        raise Exception("Failed to retrieve file name.")
    
    full_path = os.path.join(file_paths['player_fits'], 'experiment_0', file_name)
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as file:
            dyad_dict = json.load(file)    
            return dyad_dict

    raise Exception("Failed to retrieve file name.")


def load_simulated_fits_from_json(json_path: str) -> pd.DataFrame:
    """
    Load one simulated dyad JSON file and reshape it into a long, round-level DataFrame.

    Each JSON file contains one or more dyads, where each dyad key maps to a list of
    game dictionaries with:
        - player UUIDs (chooser, predictor),
        - true parameters (either embedded or parsed from the UUID),
        - fitted parameters nested under "parameter_estimates".

    This function produces one row per (dyad_key, round), with columns for:
        - identifiers: dyad_key, player_uuid_chooser, player_uuid_predictor, round
        - true parameters for each role: <param>_true_chooser, <param>_true_predictor
        - fitted predictor parameters: <param>_fitted_predictor
        - optionally fitted chooser parameters: <param>_fitted_chooser

    Arguments:
        • json_path: str;
            Full path to a single JSON file containing simulated dyad histories and fits.

    Returns:
        • pd.DataFrame;
            Long-format table with one row per (dyad_key, round) and dynamic columns
            for all discovered true and fitted parameters.
    """
    with open(json_path, "r", encoding="utf-8") as file:
        dyad_dict = json.load(file)

    rows = []
    for dyad_key, games_list in dyad_dict.items():
        for game in games_list:

            # basic info
            round_idx = game.get("round", None)
            chooser_str   = game["chooser"]
            predictor_str = game["predictor"]

            # parse their "true" parameter values
            tp_pred = game.get("true_params_predictor", None)
            tp_ch   = game.get("true_params_chooser",   None)
            if tp_pred is None or tp_ch is None:
                chooser_true   = parse_robot_string(chooser_str)
                predictor_true = parse_robot_string(predictor_str)
            else:
                chooser_true   = dict(tp_ch)
                predictor_true = dict(tp_pred)

            # 2) FITTED predictor params 
            param_est: dict = game.get("parameter_estimates", {})
            # pick *any* method block present (grid/naive/particle) in case you change general_settings
            for _method_key in ("grid", "particle", "naive", "update", "globloc", "bayes", "general"):
                if _method_key in param_est:
                    fitted_block = param_est[_method_key]
                    break
            else:
                fitted_block = {}

            predictor_fitted_params = {}
            if predictor_str in fitted_block:
                pred_role_block = fitted_block[predictor_str].get("predictor", {})
                predictor_fitted_params = pred_role_block.get("params", {})

            chooser_fitted_params = {}
            if chooser_str in fitted_block:
                ch_role_block = fitted_block[chooser_str].get("chooser", {})
                chooser_fitted_params = ch_role_block.get("params", {})

            # 3) One row; will add dynamic columns
            row = {
                "dyad_key": dyad_key,
                "player_uuid_predictor": predictor_str,
                "player_uuid_chooser": chooser_str,
                "round": round_idx,
            }

            # add TRUE columns (predictor & chooser)
            for k, v in predictor_true.items():
                row[f"{k}_true_predictor"] = v
            for k, v in chooser_true.items():
                row[f"{k}_true_chooser"] = v

            # add FITTED predictor columns
            for k, v in predictor_fitted_params.items():
                row[f"{k}_fitted_predictor"] = v

            # optional: add fitted chooser (not used here)
            if chooser_fitted_params:
                for k, v in chooser_fitted_params.items():
                    row[f"{k}_fitted_chooser"] = v

            if "temp" in chooser_true and "τ" not in chooser_true:       
                chooser_true["τ"] = chooser_true.pop("temp")
            if "temp" in predictor_true and "τ" not in predictor_true:   
                predictor_true["τ"] = predictor_true.pop("temp")

            "Extracting the posteriors originating from the simulated predictor's assigned priors, not from fitted priors."
            sim_pred = param_est.get('sim_pred')
            if sim_pred is not None:
                sim_pred_predictor_params = sim_pred.get(predictor_str, {}).get('predictor', {}).get('params', {})
                # if 'temp' in sim_pred_predictor_params:
                #     sim_pred_predictor_params["τ"] = sim_pred_predictor_params.pop("temp")
                for param_key, param_val in sim_pred_predictor_params.items():
                    row[f"{param_key}_sim_pred_predictor"] = param_val
 
            rows.append(row)

    return pd.DataFrame(rows)


def compute_param_recovery_correlations(df: pd.DataFrame, dir_path: str, out_csv_name: str, *, true_role: str = "predictor", fitted_suffix: str = "_fitted_predictor",
                                        round_mode: str = "first", params: list[str] | None = None, create_new_file: bool = False) -> pd.DataFrame:
    """
    Compute correlations between true and fitted parameters in the simulation.

    For each parameter p in `params`, this function correlates:
        p_true_<true_role>  vs  p_fitted_predictor
    either at:
        • the first round per dyad,
        • the final round per dyad, or
        • every round (round_mode="all").

    Results are saved as a tidy CSV with columns:
        ["round", "n_data", "param", "corr", "ci_lower", "ci_upper"].

    Arguments:
        • df: pd.DataFrame;
            Long-format simulation DataFrame from `load_simulated_fits_from_json`
            (or its concatenation across files).
        • dir_path: str;
            Directory where the correlation CSV should be stored.
        • out_csv_name: str;
            File name for the CSV (e.g., "correlation_results.csv").
        • true_role: str;
            Which true parameters to correlate against:
                - "predictor": parameter recovery of the optimizer.
                - "chooser":   convergence toward the chooser’s true parameters.
        • round_mode: str;
            How to slice rounds:
                - "first": one row per dyad at its earliest round.
                - "final": one row per dyad at its latest round.
                - "all":   one row per (round, param).
        • params: list[str] | None;
            Base parameter names like ["Vii","Vij","temp"]. If None, auto-detect all
            parameters that have both <param>_true_<true_role> and <param>_fitted_predictor.
        • create_new_file: bool;
            If False and the CSV already exists, load and return it. If True, recompute
            and overwrite the CSV.

    Returns:
        • pd.DataFrame;
            Tidy correlation table with one row per (round_label, param).
    """
    def fisher_z_confidence_interval(r, n, alpha=0.05):
        """
        Returns (ci_lower, ci_upper) for correlation r with sample size n,
        using the Fisher Z transform and normal approximation.
        If n < 4, returns (NaN, NaN).
        """
        if n < 4 or abs(r) >= 1.0:
            return (np.nan, np.nan)

        # Fisher z
        z = 0.5 * np.log((1 + r) / (1 - r))
        # standard error
        se = 1.0 / math.sqrt(n - 3)
        z_crit = 1.96  # approx for 95% CI

        z_lower = z - z_crit * se
        z_upper = z + z_crit * se
        # transform back
        r_lower = (math.exp(2*z_lower) - 1) / (math.exp(2*z_lower) + 1)
        r_upper = (math.exp(2*z_upper) - 1) / (math.exp(2*z_upper) + 1)
        return (r_lower, r_upper)

    out_path = os.path.join(dir_path, out_csv_name)
    if (not create_new_file) and os.path.exists(out_path):
        corr_df = pd.read_csv(out_path, encoding="utf-8", engine="python")
        if "Unnamed: 0" in corr_df.columns:
            del corr_df["Unnamed: 0"]
        return corr_df

    # pick rows for first/final
    if round_mode == "first":
        idx = df.groupby("dyad_key")["round"].idxmin()
        df_use = df.loc[idx]
        rounds = ["first"]
    elif round_mode == "final":
        idx = df.groupby("dyad_key")["round"].idxmax()
        df_use = df.loc[idx]
        rounds = ["final"]
    else:
        df_use = df.copy()
        rounds = sorted(df["round"].dropna().unique().tolist())

    # auto-detect parameters if needed
    if params is None:
        params = []
        suffix_true = f"_true_{true_role}"
        for col in df_use.columns:
            if col.endswith(suffix_true):
                base = col[: -len(suffix_true)]
                paired = f"{base}{fitted_suffix}"
                if paired in df_use.columns:
                    params.append(base)
        params = sorted(set(params))

    def _one_round(subdf: pd.DataFrame, label: str) -> pd.DataFrame:
        recs = []
        for p in params:
            tcol = f"{p}_true_{true_role}"
            fcol = f"{p}{fitted_suffix}"
            s2 = subdf.dropna(subset=[tcol, fcol])
            n = len(s2)
            if n < 3:
                r = np.nan
                lo = hi = np.nan
            else:
                r = s2[[tcol, fcol]].corr().iloc[0, 1]
                lo, hi = fisher_z_confidence_interval(r, n)
            recs.append({
                "round": label,
                "n_data": n,
                "param": p,
                "corr": r,
                "ci_lower": lo,
                "ci_upper": hi
            })
        return pd.DataFrame(recs)

    frames = []
    if round_mode in ("first", "final"):
        frames.append(_one_round(df_use, rounds[0]))
    else:
        for r in rounds:
            frames.append(_one_round(df_use[df_use["round"] == r], r))

    corr_df = pd.concat(frames, ignore_index=True)
    os.makedirs(dir_path, exist_ok=True)
    corr_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("Saved correlation results to:", out_path)
    return corr_df


def run_simulation_recovery_analysis(fig_lay: dict, general_settings: GeneralSettings, file_paths: FilePaths, export_fig: bool = True, create_new_file: bool = False, 
                                     produce_figures: bool = True, include_dropdown: bool = True, correlation_csv_name: str = "correlation_results.csv", use_dynamic_predictor: bool = False) -> pd.DataFrame:
    """
    End-to-end analysis of simulation-based parameter recovery.

    This orchestration function:
        1) Reads all simulated dyad JSON files from `dir_path`.
        2) Converts each to a long DataFrame via `load_simulated_fits_from_json`.
        3) Concatenates and saves the merged DataFrame to:
               ./simulation_results/simulated_fits.csv
        4) Computes param-recovery correlations (first/final/all) and writes
           `correlation_csv_name` in the same results folder.
        5) Optionally generates:
               - violin/boxplots of recovery by param and round,
               - scatterplots of true vs fitted parameters,
               - line plots of correlation vs round.

    Arguments:
        • fig_lay: dict;
            Plotly layout configuration used for all generated figures.
        • general_settings: GeneralSettings;
            Settings for the Bayesian analysis and plotting (e.g., dark_mode).
        • file_paths: dict[str: str | dict[str: str]];
            Dictionary of all files paths in this project.              
        • export_fig: bool;
            If True, write .html figures to disk. If False, display interactively.
        • create_new_file: bool;
            If False and merged CSV / correlation CSV already exist, reuse them.
            If True, re-read JSON, rebuild the merged DataFrame, and recompute correlations.
        • produce_figures: bool;
            If True, generate scatter, violin/box, and correlation-by-round figures.
        • include_dropdown: bool;
            Whether to include a violin/box dropdown in the univariate plots.
        • dir_path: str;
            Directory containing the per-dyad JSON files produced by the simulation fits.
        • correlation_csv_name: str;
            File name for the main correlation CSV (e.g., "correlation_results.csv").

    Returns:
        • pd.DataFrame;
            The merged, long-format simulation DataFrame (all dyads × rounds).
    """
    def plot_correlation(df: pd.DataFrame, file_paths: FilePaths, round_selection: str = "first", as_scatterplot: bool = True, params: list = None, boxplot_param: str = "Vij", 
                        include_dropdown: bool = True, fig_lay: dict = None, export_fig: bool = True, out_path: str = "corr_plot.html", fitted_suffix: str = "_fitted_predictor"):
        """
        Visualize true vs fitted parameters for a chosen round.

        Modes:
            • Scatter mode (as_scatterplot=True):
                - Plots true_predictor vs fitted_predictor for each param in `params`.
                - Adds a dropdown to toggle which parameter is shown.
                - Overlays a best-fit line and annotates correlation, R², and n.

            • Violin/box mode (as_scatterplot=False):
                - For a single `boxplot_param`, groups by the true predictor value,
                  and plots the distribution of fitted predictor estimates.
                - Annotates the overall correlation between true and fitted values.

        `round_selection` can be "first", "final", or a specific integer round.
        """
        if fig_lay is None:
            fig_lay = {}
        if as_scatterplot and (params is None or len(params) == 0):
            params = ["Vii", "Vij", "Vii_std", "Vij_std", "temp"]

        # 1) Subset data to the specified round
        if round_selection == "first":
            idxmin_ = df.groupby("dyad_key")["round"].idxmin()
            df_sub = df.loc[idxmin_]
        elif round_selection == "final":
            idxmax_ = df.groupby("dyad_key")["round"].idxmax()
            df_sub = df.loc[idxmax_]
        else:
            try:
                r_sel = int(round_selection)
                df_sub = df[df["round"] == r_sel]
            except:
                df_sub = df

        param_titles = {
            "Vii":     "Mean Self-interest μ(𝑉𝑖𝑖)", 
            "Vij":     "Mean Altruism μ(𝑉𝑖𝑗)", 
            "Vii_std": "Self-interest Standard Deviation σ(𝑉𝑖𝑖)", 
            "Vij_std": "Altruism Standard Deviation σ(𝑉𝑖𝑗)", 
            "temp":    "SoftMax Temperature (τ)"
        }

        def axis_title(param: str, role: str, type_: str) -> str:
            return f"{type_.capitalize()} {role.capitalize()} Parameter: {param_titles[param]}"

        # 2) If as_scatterplot => multi param dropdown
        if as_scatterplot:
            fig = go.Figure()
            # We'll have one param => 2 traces: (1) scatter, (2) best-fit line
            # But with multi param dropdown, we effectively have 2*N traces total.

            # We'll store an annotation template for correlation
            # We'll forcibly update it in the 'buttons' toggling.
            annotation_base = dict(
                x=0.02, y=0.95, xref='paper', yref='paper',
                showarrow=False, align="left",
                font=dict(size=18)
            )

            # We'll store all "visible" arrays for each param, each param has 2 traces
            # => total = 2 * len(params).
            # We want exactly 2 visible for the chosen param, else 2 invisible for others.
            param_buttons = []
            param_traces_startidx = {}  # param -> first trace index

            i_trace = 0
            for param_idx, param in enumerate(params):
                param_traces_startidx[param] = i_trace

                try:
                    # gather data
                    xcol = f"{param}_true_predictor"
                    ycol = f"{param}{fitted_suffix}"
                    subp = df_sub.dropna(subset=[xcol, ycol])
                except KeyError:
                    param = param.replace('Vii', 'Vᵢᵢ')
                    param = param.replace('Vij', 'Vᵢⱼ')
                    # gather data
                    xcol = f"{param}_true_predictor"
                    ycol = f"{param}{fitted_suffix}"
                    subp = df_sub.dropna(subset=[xcol, ycol])
                # print(subp), exit()
                # correlation
                corr = np.nan
                r2 = np.nan
                n_ = len(subp)
                if n_ >= 2:
                    corr = subp[[xcol,ycol]].corr().iloc[0,1]
                    r2 = corr**2

                # scatter
                scatter_trace = go.Scatter(
                    x=subp[xcol],
                    y=subp[ycol],
                    mode='markers',
                    name=f"{param}_scatter",
                    visible=(param_idx==0),  # only show the first param by default
                    hovertemplate=(
                        f"{param}_true_predictor=%{{x:.3f}}<br>"
                        f"{param}{fitted_suffix}=%{{y:.3f}}<extra></extra>"
                    ),
                    marker=dict(size=fig_lay.get("markersize", 12))
                )
                fig.add_trace(scatter_trace)
                i_trace += 1

                # best-fit line
                line_trace = None
                if n_ >= 2:
                    # do a linear fit
                    xvals = subp[xcol].values
                    yvals = subp[ycol].values
                    slope, intercept = np.polyfit(xvals, yvals, 1)

                    # for plotting, we want to cover the range of xvals
                    x_min, x_max = xvals.min(), xvals.max()
                    x_line = np.linspace(x_min, x_max, 50)
                    y_line = slope*x_line + intercept
                    line_trace = go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode='lines',
                        name=f"Best Fit: {param}",
                        visible=(param_idx==0),
                        line=dict(dash='dot', width=3),
                        hoverinfo='skip'
                    )
                else:
                    # no data or not enough
                    line_trace = go.Scatter(
                        x=[],
                        y=[],
                        mode='lines',
                        visible=(param_idx==0),
                        name=f"Best Fit: {param}"
                    )
                fig.add_trace(line_trace)
                i_trace += 1

                # annotation text for correlation
                if not math.isnan(corr) and not math.isnan(r2):
                    cor_txt = f"r = {corr:.3f}, R² = {r2:.3f}, n={n_}"
                else:
                    cor_txt = f"r = n/a, R²=n/a, n={n_}"

                # We'll store that text in the layout for param0. We'll override it via update menus for param>0
                if param_idx == 0:
                    # put it in layout
                    fig.update_layout(
                        annotations=[dict(
                            text=cor_txt,
                            **annotation_base
                        )]
                    )

                # We'll create a param_buttons entry that sets the 2 traces for param visible
                # and sets all others invisible, plus updates the annotation text, plus updates title
                # We'll fill that after we gather them all.

            # Create update menu
            # We have 2 traces per param => total 2*len(params).
            # For param i => indices 2i, 2i+1
            n_traces = 2*len(params)
            for param_idx, param in enumerate(params):
                try:
                    # figure out correlation for annotation
                    xcol = f"{param}_true_predictor"
                    ycol = f"{param}{fitted_suffix}"
                    subp = df_sub.dropna(subset=[xcol,ycol])
                except KeyError:
                    param = param.replace('Vii', 'Vᵢᵢ')
                    param = param.replace('Vij', 'Vᵢⱼ')
                    xcol = f"{param}_true_predictor"
                    ycol = f"{param}{fitted_suffix}"
                    subp = df_sub.dropna(subset=[xcol,ycol])

                n_ = len(subp)
                if n_>1:
                    c_ = subp[[xcol,ycol]].corr().iloc[0,1]
                    r2_ = c_**2
                    c_text = f"r={c_:.3f}, R²={r2_:.3f}, n={n_}"
                else:
                    c_text = f"r=n/a, R²=n/a, n={n_}"

                # build a "visible" array
                vis = [False]*n_traces
                vis[2*param_idx] = True
                vis[2*param_idx+1] = True

                param_buttons.append(
                    dict(
                        label=param,
                        method='update',
                        args=[
                            {'visible': vis},
                            {
                            'title': f"Scatter round={round_selection}, param={param}",
                            'annotations': [dict(text=c_text, **annotation_base)]
                            }
                        ]
                    )
                )

            fig.update_layout(
                template=fig_lay.get("template", "plotly_dark"),
                title=f"Scatter: round={round_selection}, param={params[0]}",
                xaxis=dict(title="Chooser True Param", **fig_lay.get("xaxis", {}), scaleanchor="y", scaleratio=1),
                yaxis=dict(title="Predictor Fitted Param", **fig_lay.get("yaxis", {})),
                updatemenus=[dict(type='dropdown', showactive=True, buttons=param_buttons, x=1.3, y=0.9)],
                hoverlabel=fig_lay.get("hoverlabel", {}),
                font=fig_lay.get("font", {})
            )

            if export_fig:
                fig.write_html(out_path)
                print("Saved scatter figure to", out_path)
            else:
                fig.show()

            return fig

        else:
            # as_scatterplot=False => do violin-boxplot with a single param
            param = boxplot_param
            xcol_true  = f"{param}_true_predictor"
            # ycol_fitted= f"{param}{fitted_suffix}"
            ycol_fitted= f"{param}_fitted_predictor"

            try:
                sub = df_sub.dropna(subset=[xcol_true, ycol_fitted]).copy()
            except KeyError:
                param = param.replace('Vii', 'Vᵢᵢ')
                param = param.replace('Vij', 'Vᵢⱼ')
                xcol_true  = f"{param}_true_predictor"
                # ycol_fitted= f"{param}{fitted_suffix}"
                ycol_fitted= f"{param}_fitted_predictor"
                sub = df_sub.dropna(subset=[xcol_true, ycol_fitted]).copy()        
            n_data = len(sub)
            corr = np.nan
            if n_data >= 2:
                corr = sub[[xcol_true,ycol_fitted]].corr().iloc[0,1]

            try: param_title = param_titles[param]
            except KeyError:
                param = param.replace('Vᵢᵢ', 'Vii')
                param = param.replace('Vᵢⱼ', 'Vij')
                param_title = param_titles[param]

            fig = go.Figure()
            fig.add_trace(go.Violin(
                x=sub[xcol_true],
                y=sub[ycol_fitted],
                box=dict(visible=True),
                meanline=dict(visible=True),
                line_color='hsla(115, 70%, 40%, 1.0)',
                points='all', pointpos=-0.7, jitter=0.45, 
                scalemode='count', width=0.3, name=param,
                hovertemplate=(
                    f"True {param_title} = %{{x}}<br>Fitted "
                    f"{param_title} = %{{y:.3f}}<extra></extra>"
                )
            ))

            # Title text and correlation annotation
            title_text = f"True {round_selection.capitalize()} Round Parameter by Fitted "
            title_text += f"{round_selection.capitalize()} Round Parameter for {param_title}"
            cor_text = f"Correlation = {corr:.3f}, n = {n_data}" if not math.isnan(corr) else ""
            cor_text += " (Simulated Data)" 
            fig.update_yaxes(range=[-1.2, 1.2])
            fig.update_layout(
                title=title_text, 
                titlefont_size=fig_lay['titlefont_size']-2,
                template=fig_lay.get("template", "plotly_dark"),
                title_x=fig_lay['title_x'], title_y=fig_lay['title_y'], 
                xaxis=dict(title=axis_title(param, 'predictor', 'true'), **fig_lay.get("xaxis", {})),
                yaxis=dict(title=axis_title(param, 'predictor', 'fitted'), **fig_lay.get("yaxis", {})),
                hoverlabel=fig_lay.get("hoverlabel", {}),
                margin=dict(l=150, r=120, t=120, b=120),
                font=fig_lay.get("font", {}),
                annotations=[dict(
                    text=cor_text,
                    x=0.02, y=0.85, xref='paper', yref='paper',
                    showarrow=False, align="left",
                    font=dict(size=30)
                )]
            )

            if boxplot_param == "Vij":
                tickvals = [-1.000, -0.667, -0.333, 0.000, 0.333, 0.667, 1.000]
                ticktext = ["-1", "-⅔", "-⅓", "0", "⅓", "⅔", "1"]
                fig.update_xaxes(tickvals=tickvals, ticktext=ticktext)

            if include_dropdown: 
            # if True:
                """Dropdown menu to switch between violin and boxplot:"""
                fig.update_layout(updatemenus=[dict(buttons=list([
                    dict(args=["type", "violin"], label="Violin", method="restyle"),
                    dict(args=["type", "box"], label="Boxplot", method="restyle")]),
                    direction="down", pad={"r": 10, "t": 10}, showactive=True, 
                    x=0.88, xanchor="left", y=0.1, yanchor="top")])

            if export_fig:
                fig.write_html(out_path)
                print("Saved violin-box figure to", out_path)
            else:
                fig.show()

            return fig

    dir_path = ensure_directory_and_join(file_paths['player_fits'], 'experiment_0')
    sim_dir = ensure_directory_and_join(file_paths['player_fits'], "simulation_results")
    os.makedirs(sim_dir, exist_ok=True)

    merged_csv_path = os.path.join(sim_dir, "simulated_fits.csv")

    df_combined = None
    if not create_new_file and os.path.exists(merged_csv_path):
        df_combined = pd.read_csv(merged_csv_path, encoding="utf-8", engine='python')    
        if 'Unnamed: 0' in df_combined.columns:
            del df_combined['Unnamed: 0']

    if df_combined is None:
        # 1) read all JSON
        all_dfs = []
        for fn in os.listdir(dir_path):
            if fn.endswith(".json"):
                path = os.path.join(dir_path, fn)
                df_game = load_simulated_fits_from_json(path)
                all_dfs.append(df_game)
        if not all_dfs:
            print(f"No JSON found in {dir_path}. Returning empty.")
            return pd.DataFrame()

        df_combined = pd.concat(all_dfs, ignore_index=True)
        df_combined["round"] = pd.to_numeric(df_combined["round"], errors="coerce")

        # 2) save
        merged_csv_path = os.path.join(sim_dir, "simulated_fits.csv")
        df_combined.to_csv(merged_csv_path, index=False, encoding='utf-8-sig')
        print("Saved combined DataFrame to", merged_csv_path)

    # 3) correlation + plots: we can use the same approach as before
    #    e.g. we do correlation between "Vij_true_predictor" and "Vij_fitted_predictor" etc.

    fitted_suffix = "_sim_pred_predictor" if use_dynamic_predictor else "_fitted_predictor"

    # here's the function that does round-based correlation
    corr_df_out = compute_param_recovery_correlations(df=df_combined, dir_path=sim_dir, 
                    out_csv_name=correlation_csv_name, create_new_file=create_new_file,
                    true_role="chooser", round_mode="all", fitted_suffix=fitted_suffix)

    if produce_figures:
        for round_selection in ('first', 'final'):
            for param in ("Vii", "Vij", "Vii_std", "Vij_std", "temp"):
            #     # BOX/VIOLIN example
            #     plot_correlation(
            #         df=df_combined, round_selection=round_selection,
            #         as_scatterplot=False, boxplot_param=param, file_paths=file_paths,
            #         fig_lay=fig_lay, export_fig=export_fig, include_dropdown=include_dropdown,
            #         out_path=os.path.join(sim_dir, f"corr_violin_{param}_{round_selection}.html")
            #     )

            # # SCATTER example
            # plot_correlation(
            #     df=df_combined,
            #     fig_lay=fig_lay, export_fig=export_fig,
            #     round_selection=round_selection, as_scatterplot=True,
            #     params=["Vii", "Vij", "Vii_std", "Vij_std", "temp"], file_paths=file_paths,
            #     out_path=os.path.join(sim_dir, f"corr_scatter_{round_selection}.html")
            # )
                "Boxplot/violin"
                plot_correlation(
                    df=df_combined,
                    round_selection=round_selection,
                    as_scatterplot=False,
                    boxplot_param=param,
                    file_paths=file_paths,
                    fig_lay=fig_lay,
                    export_fig=export_fig,
                    include_dropdown=include_dropdown,
                    out_path=os.path.join(sim_dir, f"corr_violin_{param}_{round_selection}.html"),
                    fitted_suffix=fitted_suffix,
                )
            "Scatterplot"
            plot_correlation(
                df=df_combined,
                fig_lay=fig_lay,
                export_fig=export_fig,
                round_selection=round_selection,
                as_scatterplot=True,
                params=["Vii","Vij","Vii_std","Vij_std","temp"],
                file_paths=file_paths,
                out_path=os.path.join(sim_dir, f"corr_scatter_{round_selection}.html"),
                fitted_suffix=fitted_suffix,
            )

        # correlation by round => line
        plot_param_recovery_by_round(
            general_settings=general_settings,
            df_merged=df_combined, params=["Vii", "Vij", "Vii_std", "Vij_std", "temp"], fig_lay=fig_lay, 
            export_fig=export_fig, create_new_file=create_new_file, file_paths=file_paths, 
            file_name=("corr_by_round_sim_pred.html" if use_dynamic_predictor else "corr_by_round.html"),
            corr_csv_name=("correlation_results_by_round_sim_pred.csv" if use_dynamic_predictor else "correlation_results_by_round.csv"),
            fitted_suffix=fitted_suffix, fit_mode='poly', poly_degree=3
        )

    return df_combined


def compute_recovery_by_prior_bins(df: pd.DataFrame, var_col="Vᵢⱼ_std_fitted_predictor", temp_col="temp_fitted_predictor", param_true_chooser="Vᵢⱼ_true_chooser", 
                                      param_fitted_predictor="Vᵢⱼ_fitted_predictor", player_id_col="player_uuid_predictor", var_edges: list[float] = None, 
                                      temp_edges: list[float] = None, last_rounds: list[int] = [18,19,20], print_: bool = True) -> dict:
    """
    Quantify parameter recovery as a function of prior variance and temperature.

    For each predictor:
        1) Look at their round-0 row to read:
               var_col   = prior variance (e.g., σ(Vij))
               temp_col  = prior SoftMax temperature.
        2) Bin each into 3 levels (low/med/high) using quantile-based edges or
           user-provided edges → var_bin ∈ {1,2,3}, temp_bin ∈ {1,2,3}.
        3) For all rows in `last_rounds`, compute:
               Corr(param_true_chooser, param_fitted_predictor)
           within each (var_bin, temp_bin) combination.

    Returns 3×3 tables of correlations and bin counts plus the bin edges.

    Arguments:
        • df: pd.DataFrame;
            Long simulation DataFrame (all rounds, all dyads).
        • var_col: str;
            Column representing the prior variance at round 0
            (typically something like "<param>_std_fitted_predictor").
        • temp_col: str;
            Column representing the prior temperature at round 0
            (e.g., "temp_fitted_predictor").
        • param_true_chooser: str;
            Column name for the chooser’s true parameter (e.g., "Vij_true_chooser").
        • param_fitted_predictor: str;
            Column name for the predictor’s fitted parameter (e.g., "Vij_fitted_predictor").
        • player_id_col: str;
            Column identifying predictors (e.g., "player_uuid_predictor").
        • var_edges: list[float] | None;
            Optional bin edges for variance. If None, computed as tertiles.
        • temp_edges: list[float] | None;
            Optional bin edges for temperature. If None, computed as tertiles.
        • last_rounds: list[int];
            Rounds over which to compute final recovery correlations.

    Returns:
        • dict;
            {
              "corr_table":  3×3 DataFrame of recovery correlations by (var_bin, temp_bin),
              "count_table": 3×3 DataFrame of participant counts per bin,
              "var_edges":   list[float] bin edges used for variance,
              "temp_edges":  list[float] bin edges used for temperature,
            }
    """
    def bin_index(x: float, edges: list[float]) -> int:
        """
        Map x into a 1-based bin index given ordered edges.
        """
        if pd.isna(x):
            return math.nan
        idx = np.searchsorted(edges, x, side="right") - 1
        if idx<0:
            idx=0
        if idx>=len(edges)-1:
            idx=len(edges)-2
        return idx+1

    def assign_bin_edges(series: pd.Series, nbins=3) -> list[float]:
        """
        Compute bin edges from quantiles [0, 1/nbins, ..., 1].
        Fallback to [0,1,2,3] if data are degenerate.
        """
        s = series.dropna()
        if len(s) < 3:
            return [0,1,2,3]  # fallback
        qvals = s.quantile([i/nbins for i in range(nbins+1)]).values
        # That yields 4 values for nbins=3, i.e. 0.0, 0.33...,0.66...,1.0 quantiles
        return qvals.tolist()

    # 0) Make a copy
    df_ = df.copy()

    df0 = df_[df_["round"]==0].copy()
    if var_col not in df0.columns:
        var_col = (var_col
                .replace("Vii", "Vᵢᵢ")
                .replace("Vij", "Vᵢⱼ"))

    col_names = list(df0.columns)
    cols_temp = [col for col in col_names if "temp" in col]
    cols_tau  = [col for col in col_names if "τ" in col]
    n_cols_temp, n_cols_tau = len(cols_temp), len(cols_tau)
    if "τ" in temp_col:
        if n_cols_temp > 0:
            temp_col = "temp_sim_pred_predictor"
    elif "temp" in temp_col:
        if n_cols_tau > 0:
            temp_col = "τ_sim_pred_predictor"

    if print_:
        print("Unique participants with round=0:", df0[player_id_col].nunique())
        print("Total rows with round=0:", len(df0))

    # if var_edges or temp_edges is None => compute from quantiles
    if var_edges is None:
        var_edges = assign_bin_edges(df0[var_col], nbins=3)
    if temp_edges is None:
        temp_edges = assign_bin_edges(df0[temp_col], nbins=3)

    # build a small DataFrame: [player_id, prior_var, prior_temp, var_bin, temp_bin]
    bin_rows = []
    for pid, rowsub in df0.groupby(player_id_col):
        row = rowsub.iloc[0]  # first row if multiple
        var_val  = row[var_col]
        temp_val = row[temp_col]
        vb = bin_index(var_val, var_edges)
        tb = bin_index(temp_val, temp_edges)
        bin_rows.append({
            player_id_col: pid,
            "prior_var": var_val,
            "prior_temp": temp_val,
            "var_bin": vb,
            "temp_bin": tb
        })
    df_bininfo = pd.DataFrame(bin_rows)

    # 2) Merge bininfo onto all rows => so every row now has (var_bin,temp_bin, prior_var, prior_temp)
    df_merged = pd.merge(df_, df_bininfo, on=player_id_col, how="left")

    # 3) Filter to last_rounds => gather all rows from these final rounds
    sub_last = df_merged[df_merged["round"].isin(last_rounds)].copy()
    sub_last = sub_last.dropna(
        subset=["var_bin","temp_bin", param_true_chooser, param_fitted_predictor]
    )

    # =========== (A) Build 3x3 correlation table by bins =============
    group_cols = ["var_bin","temp_bin"]

    def group_corr_bin(gdf_sub):
        if len(gdf_sub) < 3:
            return np.nan
        xvals = gdf_sub[param_true_chooser].values
        yvals = gdf_sub[param_fitted_predictor].values
        return np.corrcoef(xvals,yvals)[0,1]

    # corr_ser = (
    #     sub_last
    #     .groupby(group_cols, group_keys=False)[[param_true_chooser,param_fitted_predictor]]
    #     .apply(group_corr_bin).reset_index(name="corr_value")
    # )

    corr_ser = (
        sub_last
        .groupby(group_cols, group_keys=False)[[param_true_chooser, param_fitted_predictor]]
        .apply(group_corr_bin)
    )

    # corr_ser should now be a Series indexed by (var_bin, temp_bin)
    corr_df = corr_ser.reset_index()
    # last column is the correlation value; rename it
    last_col = corr_df.columns[-1]
    corr_df = corr_df.rename(columns={last_col: "corr_value"})


    # build a 3x3 correlation matrix
    cmat = np.full((3, 3), np.nan, dtype=float)
    for _, rowi in corr_df.iterrows():
        vb = int(rowi["var_bin"])
        tb = int(rowi["temp_bin"])
        cval = rowi["corr_value"]
        if 1<=vb<=3 and 1<=tb<=3:
            cmat[vb-1,tb-1] = cval

    var_labels = ["LowVar","MedVar","HighVar"]
    temp_labels= ["LowTemp","MedTemp","HighTemp"]
    corr_df = pd.DataFrame(cmat, index=var_labels, columns=temp_labels)
    
    # build a 3x3 count table => how many participants are in each bin (round=0)
    count_mat = np.zeros((3,3), dtype=float)
    bin_df = df_bininfo.dropna(subset=["var_bin","temp_bin"])
    bin_counts = bin_df.groupby(["var_bin","temp_bin"]).size()
    for (vb,tb), val in bin_counts.items():
        if 1<=vb<=3 and 1<=tb<=3:
            count_mat[vb-1,tb-1] = val
    count_df = pd.DataFrame(count_mat, index=var_labels, columns=temp_labels)

    if print_:
        print("Correlation table (last rounds):\n", corr_df)
        print("Count table (# participants in each bin at round=0):\n", count_df)
        print("Used var edges:", var_edges)
        print("Used temp edges:", temp_edges)

    return {
        "corr_table": corr_df,     # 3x3 table of correlation by bins
        "count_table": count_df,   # 3x3 table of bin counts
        "var_edges": var_edges,
        "temp_edges": temp_edges,
    }


def run_param_recovery_by_k(general_settings: GeneralSettings, file_paths: FilePaths, fig_lay: FigLay, param_bds: ParamBounds, n_games: int, n_predictors: int = 10, 
                            n_choosers_per_predictor: int = 1, k_params_range: tuple[int, int] = (1, 9), n_altruism_steps: int = 7, evenly_space_altruism: bool = True, 
                            utility_settings_by_k: dict[int, dict[str, bool]] | None = None, analysis_experiment_num: int = 0, random_seed: int | None = 12345, out_dir: str | None = None, 
                            figure_filename: str = "param_recovery_by_k.html", csv_filename: str = "param_recovery_by_k.csv", use_existing_fits: bool = False) -> tuple[pd.DataFrame, dict[int, Any]]:
    """
    Parameter-recovery across dimensionality (k).

    High-level behavior
    -------------------
    For each k in k_params_range:
      1) Resolve a utility form (either from `utility_settings_by_k` or from the IC comparison CSV).
      2) Sample `n_dyads` random predictor/chooser parameter vectors within `param_bds` relevant to that utility.
      3) Build a single histories JSON (with embedded true params) and run your `run_analysis_bayes(...)`.
      4) Read per-dyad fit JSONs, build a long DataFrame, and compute *first-round* correlations:
              corr( <param>_true_predictor, <param>_fitted_predictor ).
      5) Aggregate to:
              - per-parameter correlations at k
              - aggregate (macro-average over parameters present)
      6) Store the detailed dyad list + summary under `simulated_param_recovery_by_k[k]`.
      7) Save a tidy CSV and a simple Plotly figure (corr vs k).

    Returns
    -------
    • corr_by_k_df : tidy DataFrame with columns [k, param, corr, n_data, agg_corr]
    • simulated_param_recovery_by_k : the nested structure you requested
    """
    rng = random.Random(random_seed) if isinstance(random_seed, int) else random
    k_min, k_max = int(k_params_range[0]), int(k_params_range[1])
    Ks = list(range(k_min, k_max + 1))

    if not isinstance(n_predictors, int):
        print(f"n_predictors must be an integer not {type(n_predictors)}.")
        n_predictors = 10

    if not n_predictors > 0:
        print(f"n_predictors must be greater than 0, not {n_predictors}.")
        n_predictors = 10

    if not isinstance(n_choosers_per_predictor, int):
        print(f"n_choosers_per_predictor must be an integer not {type(n_choosers_per_predictor)}.")
        n_choosers_per_predictor = 1

    if not n_choosers_per_predictor > 0:
        print(f"n_choosers_per_predictor must be greater than 0, not {n_choosers_per_predictor}.")
        n_choosers_per_predictor = 1

    general_settings = copy.deepcopy(general_settings)
    general_settings['fit_roles_together'] = False
    general_settings['warmstart_policy'] = {
        "enabled": False,
        "schedule": "binary",
        "cold_iters": 1e6,
        "explore_iters": 1e6,
        "temperature_high": 1000.0,
        "temperature_low": 0.01,
        "disable_dual_annealing_when_warm": True,
    }
    

    # ---------- I/O setup ----------
    if out_dir is None:
        out_dir = os.path.join(file_paths["player_fits"], "simulation_results", "param_recovery_by_k")
    os.makedirs(out_dir, exist_ok=True)
    out_csv_path  = os.path.join(out_dir, csv_filename)
    out_fig_path  = os.path.join(out_dir, figure_filename)

    # ---------- (A) Resolve utility_settings_by_k ----------
    def _load_best_per_k_from_csv() -> dict[int, dict[str, bool]]:
        """
        Reads the IC comparison CSV (generated by your IC code) and returns, for each k, the
        boolean settings of the model with BIC_rank == 0 (winner within that k).
        """
        comp_csv = os.path.join(
            file_paths["bic_aic"],
            f"IC_Analysis_Comparison_Table_Experiment3.csv"
        )
        if not os.path.exists(comp_csv):
            raise FileNotFoundError(
                f"Could not find comparison CSV at {comp_csv}. "
                f"Provide `utility_settings_by_k` explicitly or ensure the IC section wrote this file."
            )
        df_comp = pd.read_csv(comp_csv, encoding="utf-8", engine="python")
        # Identify boolean setting columns: those that exist in utility_settings universe
        # We assume your repo exposes the canonical set; else infer as non-numeric/non-meta:
        # Keep the exact list you use elsewhere to be safe:
        setting_cols = [
            'conditional_welfare_mode','reference_dependent_altruism','min_max_rawlsian_leontief',
            'use_exponential_parameters','apply_exponents_to_payoffs','single_exponential_parameter',
            'single_payoffs_not_differences','payoff_ratios_not_differences','reference_dependent_utility',
            'use_negativity_parameters','negativity_social_comparison','fix_self_interest_parameter',
            'include_social_comparison','include_altruism_term'
        ]
        u_by_k = {}
        for k in Ks:
            dfk = df_comp[df_comp["k_params"] == k]
            if dfk.empty:
                continue
            # winner inside this k (BIC_rank == 0). If missing, pick min BIC.
            if "BIC_rank" in dfk.columns and (dfk["BIC_rank"] == 0).any():
                row = dfk.loc[dfk["BIC_rank"] == 0].iloc[0]
            else:
                row = dfk.iloc[dfk["BIC"].argmin()]
            # Build settings dict
            u = {col: bool(row[col]) for col in setting_cols if col in dfk.columns}

            u_by_k[k] = u
        if not u_by_k:
            raise RuntimeError("Failed to resolve any utility settings from the comparison CSV.")
        return u_by_k

    def _load_best_with_altruism_per_k() -> dict[int, dict[str, bool]]:
        comp_csv = os.path.join(file_paths["bic_aic"], file_paths["file_names"]["information_criterion"])
        if not os.path.exists(comp_csv):
            raise FileNotFoundError(
                f"Missing {comp_csv}. Provide utility_settings_by_k or write the IC comparison CSV first."
            )
        df = pd.read_csv(comp_csv, encoding="utf-8", engine="python")
        if "include_altruism_term" not in df.columns:
            raise RuntimeError("The comparison CSV lacks 'include_altruism_term' column.")
        setting_cols = [
            'conditional_welfare_mode','reference_dependent_altruism','min_max_rawlsian_leontief',
            'use_exponential_parameters','apply_exponents_to_payoffs','single_exponential_parameter',
            'single_payoffs_not_differences','payoff_ratios_not_differences','reference_dependent_utility',
            'use_negativity_parameters','negativity_social_comparison','fix_self_interest_parameter',
            'include_social_comparison','include_altruism_term'
        ]
        result = {}
        for k in Ks:
            dfk = df[(df["k_params"] == k) & (df["include_altruism_term"] == True)]
            if dfk.empty:
                raise RuntimeError(f"No altruism‑containing model found for k={k} in the comparison CSV.")
            
            row = dfk.loc[dfk["BIC"].idxmin()]
            result[k] = {col: bool(row[col]) for col in setting_cols if col in dfk.columns}
        return result

    if utility_settings_by_k is None:
        if evenly_space_altruism:
            utility_settings_by_k = _load_best_with_altruism_per_k()
        else:
            utility_settings_by_k = _load_best_per_k_from_csv()

    # Validate each k’s k_params matches requested k
    for k in Ks:
        u = utility_settings_by_k.get(k)
        if not isinstance(u, dict):
            raise ValueError(f"Missing utility_settings for k={k}.")
        k_est = gnrl.count_free_parameters(utility_settings=u)
        if k_est != k:
            raise ValueError(f"Utility settings for k={k} imply k={k_est}. Please correct or override.")

    # ---------- Helpers ----------
    def _find_key(candidates: list[str], keys: list[str]) -> str | None:
        for name in candidates:
            if name in keys:
                return name
        return None

    def _sample_params_from_bounds(param_info: dict) -> dict[str, float]:
        """
        Sample one random parameter vector within provided param_info['bounds'] aligned
        with param_info['keys']. Also mirror 'temp' to 'τ' if present.
        """
        keys = list(param_info['keys'])
        bounds = list(param_info['bounds'])
        vals = {}
        for i, k in enumerate(keys):
            lo, hi = bounds[i]
            if k.endswith("_std"):
                lo = max(lo, 1e-3)  # keep std away from zero
            vals[k] = rng.uniform(float(lo), float(hi))
        if "temp" in vals and "τ" not in vals:
            vals["τ"] = float(vals["temp"])
        if "τ" not in vals:
            vals["τ"] = general_settings.get('softmax_temperature', 1.5)
        if evenly_space_altruism:
            if "Vᵢᵢ" in vals:
                vals["Vᵢᵢ"] = 1.0
            if "Vᵢᵢ_std" in vals:
                vals["Vᵢᵢ_std"] = 1.0
            if "Vᵢⱼ_std" in vals:
                vals["Vᵢⱼ_std"] = 1.0
        return vals

    def _assemble_histories_dict(dyad_list: list[dict]) -> dict:
        """
        Convert [{DyadKey: DyadGames}, ...] => {'histories': {DyadKey: DyadGames, ...}, 'player_info': {...}}
        Keeps your avatar aesthetics.
        """
        histories = {}
        for d in dyad_list:
            histories.update(d)
        avatar_shapes = [
            "arrow-head","bowtie","circle","cross","curvy-x","dent-square","dodecagon","flame","flower",
            "ghost","hexagon","hour-glass","jagged-sun","lemon","moon","pentagon","round-square","squash",
            "teardrop","two-triangle","star-six","stop-sign"
        ]
        player_info = {}
        for dyad_key in histories.keys():
            plr_uuid_1, plr_uuid_2 = dyad_key[1:-1].split(", ")
            for plr_uuid in (plr_uuid_1, plr_uuid_2):
                if plr_uuid not in player_info:
                    player_info[plr_uuid] = {
                        'player_type': 'robot',
                        'avatar_shape': avatar_shapes[rng.randint(0, len(avatar_shapes)-1)],
                        'player_color': f'hlsa({rng.randint(0,359)}, {rng.randint(35,65)}%, {rng.randint(35,65)}%, 1.0)'
                    }
        return {'histories': histories, 'player_info': player_info}

    # ---------- Main loop over k ----------
    aggregate_records = []
    simulated_param_recovery_by_k: dict[int, Any] = {}

    # We will run separate batches per k. We rely on your suffix machinery to distinguish files.
    for k in Ks:
        print(f"run_param_recovery_by_{k}...")
        u_settings_k = utility_settings_by_k[k]

        # Build param_info for this utility
        param_info_k = make_param_info(
            param_bds=param_bds, utility_settings=u_settings_k, guess_seed=None,
            general_settings=general_settings, random_guesses_are_unique=True, 
        )

        keys_k   = list(param_info_k['keys'])
        bounds_k = dict(zip(keys_k, param_info_k['bounds']))

        # resolve altruism key & bounds
        altruism_key = _find_key(["Vᵢⱼ", "Vij"], keys_k)
        if altruism_key is None:
            raise RuntimeError(f"[k={k}] The chosen utility form lacks an identifiable altruism weight (Vᵢⱼ/Vij).")
        if altruism_key not in bounds_k:
            raise RuntimeError(f"[k={k}] Missing bounds for altruism key: {altruism_key}.")
        a_lo, a_hi = map(float, bounds_k[altruism_key])

        # build the grid of altruism values (evenly spaced)
        if evenly_space_altruism:
            steps = max(2, int(n_altruism_steps))
            grid = list(np.linspace(a_lo, a_hi, steps))
            # number of predictors must be a multiple of #steps; pad if needed
            reps = math.ceil(n_predictors / steps)
            altruism_targets = (grid * reps)[:n_predictors]
            rng.shuffle(altruism_targets)
        else:
            altruism_targets = [rng.uniform(a_lo, a_hi) for _ in range(n_predictors)]

        # Sample dyads
        dyads_for_k = []
        predictor_uuids_for_k = []

        for pred_idx in range(n_predictors):
            # sample predictor once; build a STABLE id
            params_pred = _sample_params_from_bounds(param_info_k)
            pred_altruism_val = float(altruism_targets[pred_idx])
            params_pred[altruism_key] = pred_altruism_val
            stable_pid = f"k{k}_Vij={pred_altruism_val:.2f}_p{pred_idx}"
            # FIXED predictor UUID across several dyads
            predictor_uuid_fixed = simulated_bot_uuids(
                n_games=n_games, params_predictor=params_pred, params_chooser=params_pred,  # chooser ignored for predictor UUID
                k_of_2=False, predictor_id=stable_pid, chooser_id="seed"  # chooser_id temporary
            )[0]

            predictor_uuids_for_k.append(predictor_uuid_fixed)

            for cho_idx in range(n_choosers_per_predictor):
                params_ch = _sample_params_from_bounds(param_info_k)
                # build dyad WITH fixed predictor id; let chooser_id vary
                dyad = create_simulated_dyad(
                    n_games=n_games,
                    params_chooser=params_ch,
                    params_predictor=params_pred,
                    utility_settings=u_settings_k,
                    payoff_structures=None,
                    default_utility_settings=False,
                    embed_true_params=True
                )
                # overwrite predictor UUID in the freshly created dyad to the fixed one
                (dyad_key, games_list), = dyad.items()
                for g in games_list:
                    g["predictor"] = predictor_uuid_fixed
                fixed_key = f"({predictor_uuid_fixed}, {games_list[0]['chooser']})"
                dyads_for_k.append({fixed_key: games_list})

        # Assemble one histories JSON and write to processed/
        histories_k = _assemble_histories_dict(dyads_for_k)

        # Name & suffix management (reuse your helpers)
        file_name_suffix = prep.create_file_name_suffix(
            general_settings=general_settings, utility_settings=u_settings_k
        )
        file_paths_k = copy.deepcopy(file_paths)
        file_paths_k = prep.add_remove_file_name_suffix(
            file_paths=file_paths_k, file_name_suffix=None, add_suffix=False
        )
        file_paths_k = prep.add_remove_file_name_suffix(
            file_paths=file_paths_k, file_name_suffix=file_name_suffix, add_suffix=True
        )

        # Save histories to processed
        histories_file_path = os.path.join(
            file_paths_k['processed'], 
            file_paths_k['file_names'][f'player_pairs_exper{analysis_experiment_num}']
        )
        os.makedirs(os.path.dirname(histories_file_path), exist_ok=True)
        with open(histories_file_path, 'w', encoding='utf-8') as f:
            json.dump(histories_k, f, ensure_ascii=False, indent=4)
        print(f"[k={k}] Saved simulated histories to: {histories_file_path}")

        # Run your standard analysis unless we’re reusing existing JSONs
        if not use_existing_fits:
            general_settings_k = copy.deepcopy(general_settings)
            general_settings_k['experiment_num'] = analysis_experiment_num
            general_settings_k['write_mode'] = 'overwrite'     # fresh batch per k
            # keep all your other knobs (grid/naive/particle; temperature_is_param; etc.)

            run_analysis_bayes(
                histories_data=histories_k,
                file_paths=file_paths_k,
                param_info=param_info_k,
                utility_settings=u_settings_k,
                general_settings=general_settings_k, 
                player_uuids=predictor_uuids_for_k
            )

        # Collect fit JSONs for this k (filter by suffix in filename)
        fit_dir = os.path.join(file_paths['player_fits'], f"experiment_{analysis_experiment_num}")
        # json_files = [fn for fn in os.listdir(fit_dir) if fn.endswith(".json") and file_name_suffix in fn]
        json_files = [fn for fn in os.listdir(fit_dir) if fn.endswith(".json") and "_by_k_" in fn]
        if not json_files:
            print(f"[k={k}] Warning: no fit files found with suffix '{file_name_suffix}' in {fit_dir}.")

        # Build long DF for this k
        dfs = []
        for fn in json_files:
            path = os.path.join(fit_dir, fn)
            dfs.append(load_simulated_fits_from_json(path))
        df_k = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        # Compute correlation (predictor truth vs predictor fitted), first-round only
        params_to_correlate = ['Vᵢⱼ'] if evenly_space_altruism else [
            param_key for param_key in param_info_k['keys'] 
            if '_std' not in param_key and '_cov' not in param_key and 'temp' not in param_key
        ]
        corr_df_k = compute_param_recovery_correlations(
            df=df_k,
            dir_path=out_dir,
            out_csv_name=f"correlations_k{k}.csv",
            true_role="predictor",
            round_mode="first",
            params=params_to_correlate,                  # auto-detect any overlapping true/fitted predictor params
            create_new_file=True
        )
        # aggregate across params (macro-average over present params)
        corr_only = corr_df_k[corr_df_k["round"] == "first"][["param","corr","n_data"]].copy()
        agg_corr = float(np.nanmean(corr_only["corr"])) if not corr_only.empty else np.nan

        # Save per-k rows for the final CSV
        for _, r in corr_only.iterrows():
            aggregate_records.append({
                "k": k,
                "param": r["param"],
                "corr": float(r["corr"]),
                "n_data": int(r["n_data"]),
                "agg_corr": float(agg_corr)
            })

        # Build the requested nested dict; include dyads + fitted snapshot
        # (We’ll store the first-round fitted predictor params; games list is already in histories_k)
        dyad_entries = []
        for fn in json_files:
            path = os.path.join(fit_dir, fn)
            with open(path, "r", encoding="utf-8") as f:
                dyad_dict = json.load(f)
            for dkey, games in dyad_dict.items():
                if not games:
                    continue
                g0 = games[0]   # first round snapshot for true and fitted
                true_pred = g0.get("true_params_predictor", {})
                true_ch   = g0.get("true_params_chooser", {})
                # fitted predictor params (try to find any method block)
                est = g0.get("parameter_estimates", {})
                fitted_pred = {}
                for _m in ("grid", "particle", "naive", "update", "globloc", "bayes", "general"):
                    if _m in est and g0["predictor"] in est[_m]:
                        fitted_pred = est[_m][g0["predictor"]].get("predictor", {}).get("params", {})
                        break
                dyad_entries.append({
                    "games": games,  # if you find this too heavy, store only the dkey and the two param dicts
                    "synthetic_params": {"chooser": true_ch, "predictor": true_pred},
                    "fitted_params": {"predictor": fitted_pred}
                })

        simulated_param_recovery_by_k[k] = {
            "dyads": dyad_entries,
            "correlation_by_param": {row["param"]: float(row["corr"]) for _, row in corr_only.iterrows()},
            "aggregate_correlation": agg_corr
        }

    # ---------- Build & save tidy CSV ----------
    try:
        corr_by_k_df = pd.DataFrame(aggregate_records).sort_values(["k", "param"])
        corr_by_k_df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
        print(f"Saved summary CSV to: {out_csv_path}")
    except (PermissionError, OSError):
        "Pass if I have the file open."
        pass

    # ---------- Plotly figure (corr vs k) ----------
    # Use aggregate correlation per k for a simple, publication-friendly line
    agg = corr_by_k_df.groupby("k", as_index=False)["agg_corr"].first()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["k"], y=agg["agg_corr"], mode="lines+markers",
        name="Aggregate 𝑟 (macro-average across params)",
        hovertemplate="𝑘 = %{x}<br>𝑟 = %{y:.3f}<extra></extra>",
        marker=dict(size=fig_lay.get("markersize", 12), color="hsla(120, 50%, 50%, 1.0)")
    ))

    param_title = 'Altruism' if evenly_space_altruism else 'Parameter'
    y_title = f"True {param_title} vs Fitted {param_title} Correlation"

    x_axis = {
        'title': "Number of Free Parameters (𝑘)",
        'tickfont': dict(size=24),
        'title_font': dict(size=30),
        'tickvals': [1, 2, 3, 4, 5, 6, 7, 8, 9], 
        'ticktext': ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
        'range': [0.95, 9.05] 
    }
    y_axis = {
        'title': y_title,
        'tickfont': dict(size=24),
        'title_font': dict(size=30),
        'tickvals': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 
        'ticktext': ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
        'range': [-0.05, 1.05]         
    }

    fig.update_layout(
        title=f"Predictor {param_title} Recovery Correlation 𝑟 vs. Dimensionality 𝑘",
        titlefont_size=fig_lay['titlefont_size'] * 0.6, template=fig_lay['template'], 
        title_x=fig_lay['title_x'], title_y=fig_lay['title_y'], showlegend=False, 
        margin=dict(l=560, r=560, t=120, b=100), 
        xaxis=x_axis, yaxis=y_axis,
    )
    fig.write_html(out_fig_path)
    print(f"Saved Plotly figure to: {out_fig_path}")

    return corr_by_k_df, simulated_param_recovery_by_k


def verify_particle_filter_fidelity(general_settings: GeneralSettings, utility_settings: UtilitySettings, 
                                    param_info: ParamInfo, file_paths: FilePaths, fig_lay: FigLay, sample_ratios: int | list[float] = 5, 
                                    random_seed: int | None = 1010101, n_predictors: int = 10, n_games_per_dyad: int = 10) -> pd.DataFrame:
    """
    Verifies that the particle filter (PF) reproduces the full grid-based posterior update (which occurs when the sample_ratio = 1.0).
    Runs a grid-based predictor over a dyad with a set of priors and provides those same priors to a PF-based predictor and compares 
    the posteriors of the grid-based and PF-based agent --> repeats for other sets of priors --> computes a correlation coefficient
    --> does this for variable sample ratios --> plots the correlation coefficient as a function of the sample ratio. 

    Arguments:
        • param_info: dict[str, list[Any]]; Stores parameter keys, boundaries, and initial guesses.
        • utility_settings: dict[str, bool]; Defines the functional form of the utility function.
        • file_paths: dict[str, str | dict[str, str]]; Stores all file paths for storing data.
        • general_settings: dict[str, Any]; Various settings used throughout this analysis.
        • fig_lay: dict[str: Any]; Establishes the settings for the Plotly figure layout.
        • sample_ratios: int | list[float]; n uniformly spaced sample ratios between 0 and 1 or an array of sample ratios. 
            - sample_ratio is a general setting that determins the number of probabilities that computed per Bayesian update. 
        • random_seed: int | None; Seed for reproducibility across dyads, parameter draws, and the single dyad structure.
        • n_predictors: int; Number of synthetic predictors (i.e., random priors) to evaluate per sample ratio.
        • n_games_per_dyad: int; Number of games per dyad.

    Returns:
        • pd.DataFrame: Simulation data   
    """
    "Generate an array of sample_ratios"
    if isinstance(sample_ratios, int) and sample_ratios > 1:
        sample_ratios = np.round(np.linspace(start=0, stop=1, num=sample_ratios), decimals=3).tolist()[:-1]
    elif isinstance(sample_ratios, list) and all((0 <= sratio <= 1) for sratio in sample_ratios):
        sample_ratios = sorted(sample_ratios)
        if sample_ratios[-1] == 1:
            sample_ratios = sample_ratios[:-1]
    else:
        raise ValueError(f"sample_ratios must be an integer or an array like"
                         f" [0.0, 0.25, 0.5, 0.75, 1.0], not {sample_ratios}!")

    if sample_ratios[0] == 0:
        sample_ratios = sample_ratios[1:]

    "Forcing general settings necessary for this simulation."
    general_settings_grid = copy.deepcopy(general_settings)
    general_settings_grid['include_covariance'] = False
    general_settings_grid['update_method'] = 'grid'
    general_settings_grid['sample_ratio'] = 1.0

    "Prepare containers for runtime tracking and progress feedback."
    baseline_durations: list[float] = []
    pf_durations_map: dict[str, list[float]] = {f"{sr:.3f}": [] for sr in sample_ratios}
    print_every = max(1, n_predictors // 10)  # print every ~10% of predictors

    "Generate prior parameters for choosers and predictors."
    params_choosers = [
        {
            param_key: param_val
            for param_key, param_val
            in zip(
                param_info['keys'],
                UniformGuesser(bounds=param_info['bounds'], seed=random_seed)()  
            )
        }
        for chooser in range(n_predictors)
    ]

    params_predictors = [
        {
            param_key: param_val
            for param_key, param_val
            in zip(
                param_info['keys'],
                UniformGuesser(bounds=param_info['bounds'], seed=random_seed)()
            )
        }
        for predictor in range(n_predictors)
    ]

    "Generate dyads"
    dyads = {}
    for pred_idx, (params_choo, params_pred) in enumerate(zip(params_choosers, params_predictors)):
        dyad_games = []
        for game_idx in range(n_games_per_dyad):

            payoff_A_chooser =   random.randint(1, 5)
            payoff_A_predictor = random.randint(1, 5)
            payoff_B_chooser =   random.randint(1, 5)
            payoff_B_predictor = random.randint(1, 5)
            payoffs = {
                'payoff_A_chooser': payoff_A_chooser, 'payoff_A_predictor': payoff_A_predictor,
                'payoff_B_chooser': payoff_B_chooser, 'payoff_B_predictor': payoff_B_predictor,
            }

            "Store choices based on the chooser parameters"
            choice_response = choice(current_game=payoffs, agent_params=params_choo, utility_settings=utility_settings, 
                                     softmax_temperature=general_settings.get('softmax_temperature'), select=True)["model_choose_A"]
            choice_response = "A" if choice_response == 1 else "B"

            dyad_game = {
                "chooser": f"C{pred_idx}",
                "predictor": f"P{pred_idx}",
                "matching_probability": 1.0,
                "payoff_A_chooser": payoff_A_chooser,
                "payoff_A_predictor": payoff_A_predictor,
                "payoff_B_chooser": payoff_B_chooser,
                "payoff_B_predictor": payoff_B_predictor,
                "choice": choice_response,
                "prediction": None,
                "abdicated_chooser": False,
                "abdicated_predictor": False,
                "timestamp": time.time(),
                "round": game_idx,            
            }
            dyad_games.append(dyad_game)

        "Generate and store parameter estimates for the full grid-based predictor"
        t_grid_start = time.perf_counter()
        dyad_games = agent(dyad_games=dyad_games, game_idx_start=0, game_idx_stop=n_games_per_dyad, player_role='predictor', 
                           initial_params={'predictor': params_pred}, param_info=param_info, utility_settings=utility_settings,
                           player_uuid=f"P{pred_idx}", general_settings=general_settings_grid)
        t_grid_stop = time.perf_counter()
        baseline_durations.append(t_grid_stop - t_grid_start)

        for game in dyad_games: 
            "Submitting predictions for aestetic reasons. Not necessary for analysis."
            model_predict_A = game.get('parameter_estimates', {}).get('grid', {}).get(
                f'P{pred_idx}', {}).get('predictor', {}).get('output', {}).get('model_predict_A')
            if isinstance(model_predict_A, float):
                game['prediction'] = "A" if random.random() > 0.5 else "B"

            "Store parameter estimates for the full grid in a different spot so that the PF estimates can be stored in the same game."
            game['parameter_estimates_full_grid'] = game.pop('parameter_estimates')

        "Generate and store parameter estimates for PF-based predictors at varying sample ratios"
        for sample_ratio in sample_ratios:
            sample_ratio_key = f'{sample_ratio:.3f}'
            general_settings_pf = copy.deepcopy(general_settings_grid)
            general_settings_pf['sample_ratio'] = sample_ratio

            t_pf_start = time.perf_counter()
            dyad_games = agent(dyad_games=dyad_games, game_idx_start=0, game_idx_stop=n_games_per_dyad, player_role='predictor', 
                            initial_params={'predictor': params_pred}, param_info=param_info, utility_settings=utility_settings,
                            player_uuid=f"P{pred_idx}", general_settings=general_settings_pf)            
            t_pf_stop = time.perf_counter()
            pf_durations_map[sample_ratio_key].append(t_pf_stop - t_pf_start)

            for game in dyad_games:
                "Rename parameter estimates based on the sample ratio value"
                game[f'parameter_estimates_pf_{sample_ratio:.3f}'] = game.pop('parameter_estimates')

        "Progress: print a heartbeat every few predictors"
        if (pred_idx + 1) % print_every == 0:
            print(f"[verify_pf] finished predictor {pred_idx + 1}/{n_predictors}")

        dyads[pred_idx] = dyad_games

    "Condense information into a dictionary mapping sample ratios to grid versus pf posteriors per player."
    "{sample_ratio: {param_key: 'grid': [posterior_P0, posterior_P1,...], 'pf': [posterior_P0, posterior_P1,...]}}"
    sample_ratios_to_posterior_pairs: dict[str: dict[str: dict[str: list[float]]]] = {}

    for sample_ratio in sample_ratios:
        sample_ratio_key = f'{sample_ratio:.3f}'
        sample_ratios_to_posterior_pairs[sample_ratio_key] = {}
        for param_key in param_info['keys']:
            sample_ratios_to_posterior_pairs[sample_ratio_key][param_key] = {'grid': [], 'pf': []}
            for pred_idx, dyad_games in dyads.items():
                final_game = dyad_games[-1]
                posteriors_grid = final_game.get(f'parameter_estimates_full_grid', {}).get('grid', {}).get(
                    f'P{pred_idx}', {}).get('predictor', {}).get('posteriors', {}).get(param_key)
                posteriors_pf = final_game.get(f'parameter_estimates_pf_{sample_ratio_key}', {}).get('grid', {}).get(
                    f'P{pred_idx}', {}).get('predictor', {}).get('posteriors', {}).get(param_key)
                sample_ratios_to_posterior_pairs[sample_ratio_key][param_key]['grid'].append(posteriors_grid)
                sample_ratios_to_posterior_pairs[sample_ratio_key][param_key]['pf'].append(posteriors_pf)

    "Helper: robust correlation that degrades gracefully when variance is ~0."
    def _safe_corr(x_values: list[float], y_values: list[float]) -> float:
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)
        if len(x) < 2 or len(y) < 2:
            return float('nan')
        sx = float(np.std(x))
        sy = float(np.std(y))
        if sx == 0.0 and sy == 0.0:
            # identical constants across predictors ⇒ treat as perfect match
            return 1.0
        if sx == 0.0 or sy == 0.0:
            # one side constant, the other not ⇒ no linear association
            return 0.0
        corr = float(np.corrcoef(x, y)[0, 1])
        # clip to [−1, 1] and then report; user asked to watch for negatives
        if corr < 0:
            print(f"[verify_pf] WARNING: negative correlation ({corr:.3f})")
        return float(np.clip(corr, -1.0, 1.0))

    "Recompute correlations with the robust helper (per param, then mean excluding _std)."
    sample_ratios_to_param_correlations: dict[str, dict[str, float]] = {}
    sample_ratios_to_mean_correlations: dict[str, float] = {}

    for sample_ratio_key, param_map in sample_ratios_to_posterior_pairs.items():
        sample_ratios_to_param_correlations[sample_ratio_key] = {}
        per_param_corrs: list[float] = []
        for param_key, grid_pf_lists in param_map.items():
            corr_val = _safe_corr(grid_pf_lists['grid'], grid_pf_lists['pf'])
            sample_ratios_to_param_correlations[sample_ratio_key][param_key] = corr_val
            if '_std' not in param_key and (corr_val == corr_val):  # skip NaNs
                per_param_corrs.append(corr_val)
        sample_ratios_to_mean_correlations[sample_ratio_key] = float(np.mean(per_param_corrs)) if per_param_corrs else float('nan')

    "Build and write the summary CSV (one row per sample_ratio)."
    summary_rows: list[dict[str, Any]] = []
    for sr_key in sorted(sample_ratios_to_mean_correlations.keys(), key=lambda s: float(s)):
        pf_times = pf_durations_map.get(sr_key, [])
        row = {
            "sample_ratio": float(sr_key),
            "corr_overall_excluding_std": float(sample_ratios_to_mean_correlations[sr_key]),
            "pf_time_mean_s": float(np.mean(pf_times)) if pf_times else float('nan'),
            "pf_time_std_s": float(np.std(pf_times, ddof=1)) if len(pf_times) > 1 else float('nan'),
            "grid_time_mean_s": float(np.mean(baseline_durations)) if baseline_durations else float('nan'),
            "grid_time_std_s": float(np.std(baseline_durations, ddof=1)) if len(baseline_durations) > 1 else float('nan'),
            "n_predictors": int(n_predictors),
            "n_games_per_dyad": int(n_games_per_dyad),
        }
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values("sample_ratio")

    k_params = len([
        param_key for param_key in param_info['keys'] 
        if '_std' not in param_key and '_cov' not in param_key
    ])

    "Build and write the per-parameter correlation CSV."
    param_rows: list[dict[str, Any]] = []
    for sr_key, param_corrs in sample_ratios_to_param_correlations.items():
        for param_key, corr_val in param_corrs.items():
            param_rows.append({
                "sample_ratio": float(sr_key),
                "param_key": param_key,
                "correlation": float(corr_val),
                "n_predictors": int(n_predictors),
                "n_games_per_dyad": int(n_games_per_dyad),
                "k_params": k_params
            })
    correlations_df = pd.DataFrame(param_rows).sort_values(["sample_ratio", "param_key"])

    "Compose file names based on inputs."
    file_stub = f"verify_particle_filter_fidelity_{len(sample_ratios):03d}-{n_predictors}-{n_games_per_dyad}-{k_params}"
    file_stub += prep.create_file_name_suffix(general_settings=general_settings, utility_settings=utility_settings)
    out_dir = file_paths["visuals"]
    os.makedirs(out_dir, exist_ok=True)
    summary_csv_path = os.path.join(out_dir, f"{file_stub}.csv")
    per_param_csv_path = os.path.join(out_dir, f"{file_stub}_by_param.csv")

    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    correlations_df.to_csv(per_param_csv_path, index=False, encoding="utf-8-sig")

    # ----- Build the Plotly figure (square, anchored axes, [0,1] ranges) -----
    # x,y data with explicit anchors at (0,0) and (1,1)
    sr_sorted = [float(k) for k in sorted(sample_ratios_to_mean_correlations.keys(), key=lambda s: float(s))]
    corr_sorted = [float(sample_ratios_to_mean_correlations[f"{sr:.3f}"]) for sr in sr_sorted]

    fig = go.Figure()

    # Main series
    fig.add_trace(go.Scatter(
        x=sr_sorted,
        y=corr_sorted,
        mode="markers+lines",
        name="PF vs Grid: correlation",
        hovertemplate="sample_ratio=%{x:.3f}<br>corr=%{y:.3f}<extra></extra>",
        marker=dict(size=fig_lay.get("markersize", 12), opacity=0.9)
    ))

    # Invisible anchors to force [0,1]×[0,1] domain
    fig.add_trace(go.Scatter(
        x=[0.0, 1.0], y=[0.0, 1.0],
        mode="markers",
        marker=dict(size=1, opacity=0.0),
        showlegend=False,
        hoverinfo="skip",
        name="anchors"
    ))

    title = (
        "Particle Filter Fidelity vs. Grid"
    )

    # Annotation with fancy 'n' and spaces around equals
    # Pull n_bins_per_dimension from general settings (what you passed in here)
    n_bins = int(general_settings.get("n_bins_per_dimension", 0))
    annot_text = f"𝑛 bins = {n_bins} • 𝑛 games = {n_games_per_dyad} • 𝑛 players = {n_predictors}"

    font_base = fig_lay["annotations"]["font"].copy()
    font_base.pop("size", None)
    font_small = dict(**font_base, size=fig_lay["annotations"]["font"]["size"] - 4)

    fig.update_layout(
        title=title,
        titlefont_size=fig_lay['titlefont_size'],
        title_x=fig_lay['title_x'],
        title_y=fig_lay['title_y'],
        template=fig_lay['template'],
        hoverlabel=fig_lay['hoverlabel'],
        margin=dict(l=560, r=560, t=120, b=200),
        font=fig_lay['font'], showlegend=False,
        annotations=[
            dict(
                text=annot_text, font=font_small, xref="paper", yref="paper",
                showarrow=fig_lay["annotations"]["showarrow"], x=0.5, y=-0.05, 
            ),
            dict(
                text=build_utility_equation(utility_settings=utility_settings), font=font_small,
                showarrow=fig_lay["annotations"]["showarrow"], x=0.5, y=-0.15, xref="paper", yref="paper"
            )            
        ]
    )

    # Axes: pass your axis styling dicts into update_xaxes / update_yaxes
    fig.update_xaxes(
        title="Sample Ratio (fraction of grid evaluated)",
        range=[0, 1],
        **fig_lay.get("xaxis", {})
    )
    fig.update_yaxes(
        title="Correlation (PF posterior vs Grid posterior)",
        range=[0, 1],
        scaleanchor="x",  # square plot: y anchored to x
        scaleratio=1,
        **fig_lay.get("yaxis", {})
    )

    # Save the HTML
    html_path = os.path.join(out_dir, f"{file_stub}.html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"[verify_pf] Wrote Plotly HTML:     {html_path}")

    "Console report: quick glance at the summary."
    print("\n[verify_pf] Correlation & runtime summary (by sample_ratio):")
    with pd.option_context('display.float_format', lambda v: f"{v:.3f}"):
        print(summary_df.to_string(index=False))

    print(f"\n[verify_pf] Wrote summary CSV:     {summary_csv_path}")
    print(f"[verify_pf] Wrote per-param CSV:   {per_param_csv_path}")
    print(f"[verify_pf] Wrote Plotly HTML:     {html_path}")

    return summary_df


"=========================================================================================="
"======= Simulation 2) Predictor Estimates Converge to the Chooser’s True Altruism ========"
"=========================================================================================="

def plot_param_recovery_by_round(
        df_merged: pd.DataFrame,
        general_settings: GeneralSettings,
        file_paths: FilePaths,
        params=None,
        fig_lay: dict = None,
        export_fig: bool = True,
        create_new_file: bool = True,
        file_name: str = "corr_by_round.html",
        corr_csv_name: str = "correlation_results_by_round.csv",
        fitted_suffix: str = "_fitted_predictor",
        fit_mode: str = "poly",      # "poly" or "line"
        poly_degree: int = 3
    ) -> None:
    """
    Plot how parameter-recovery correlations evolve across rounds.

    Used for figure: Incremental alignment between inferred and true altruism across rounds

    This function:
        1. Calls `compute_param_recovery_correlations(..., round_mode="all")`.
        2. For each parameter, plots:
             • correlation vs. round, and
             • a best-fit line with slope and R² annotation.
        3. Uses a dropdown to toggle which parameter’s traces are visible.

    Arguments:
        • df_merged: pd.DataFrame;
            Long-format DataFrame with true_* and fitted_predictor columns.
        • general_settings: GeneralSettings;
            Used mainly for display options (e.g., dark_mode).
        • file_paths: dict[str: str | dict[str: str]];
            Dictionary of all file paths in this project.
        • params: list[str] | None;
            Parameters to plot (e.g., ["Vii","Vij","Vii_std","Vij_std","temp"]).
            If None, defaults to that list.
        • fig_lay: dict | None;
            Layout settings (font, template, axis options) for Plotly.
        • export_fig: bool;
            If True, write the Plotly HTML to `dir_path/file_name`. If False, show it.
        • create_new_file: bool;
            If True, force recomputation of the correlation CSV.
        • file_name: str;
            Name of the .html file (".html" added if missing).

    Returns:
        • None;
            Writes or shows an interactive Plotly figure with correlation trajectories.
    """

    def canonicalize_param_name(param: str, available: list[str]) -> str:
        if param in available:
            return param

        candidates = [
            param.replace("Vii", "Vᵢᵢ").replace("Vij", "Vᵢⱼ"),
            param.replace("Vᵢᵢ", "Vii").replace("Vᵢⱼ", "Vij"),
        ]
        for cand in candidates:
            if cand in available:
                return cand
        return param

    def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Generic R²: 1 - SS_res / SS_tot, works for any curve (not just lines).
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.size != y_pred.size or y_true.size < 2:
            return math.nan
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        if abs(ss_tot) < 1e-12:
            return math.nan
        ss_res = np.sum((y_true - y_pred) ** 2)
        return 1.0 - (ss_res / ss_tot)

    if params is None:
        params = ["Vij", "Vii", "Vij_std", "Vii_std", "temp"]
    if fig_lay is None:
        fig_lay = {}

    dir_path = ensure_directory_and_join(file_paths['player_fits'], 'simulation_results')

    param_titles = {
        "Vii":     "Mean Self-interest μ(𝑉𝑖𝑖)", 
        "Vij":     "Mean Altruism μ(𝑉𝑖𝑗)", 
        "Vii_std": "Self-interest Standard Deviation σ(𝑉𝑖𝑖)", 
        "Vij_std": "Altruism Standard Deviation σ(𝑉𝑖𝑗)", 
        "temp":    "SoftMax Temperature (τ)"
    }

    param_containers = {
        "Vii":     "μ(𝑉𝑖𝑖)", 
        "Vij":     "μ(𝑉𝑖𝑗)", 
        "Vii_std": "σ(𝑉𝑖𝑖)", 
        "Vij_std": "σ(𝑉𝑖𝑗)", 
        "temp":    "(τ)"
    }

    # 1) Correlations by round
    corr_df = compute_param_recovery_correlations(
        df=df_merged,
        dir_path=dir_path,
        out_csv_name=corr_csv_name,
        fitted_suffix=fitted_suffix,
        create_new_file=create_new_file,
        true_role="chooser",
        round_mode="all"
    )

    corr_df = corr_df.copy()
    corr_df["round"] = pd.to_numeric(corr_df["round"], errors="coerce")
    corr_df = corr_df.dropna(subset=["round", "corr"])

    available_params = corr_df["param"].unique().tolist()
    param_mapping = {p: canonicalize_param_name(p, available_params) for p in params}
    corr_df = corr_df[corr_df["param"].isin(param_mapping.values())]

    fig = go.Figure()
    annotation_base = dict(
        x=0.05, y=0.95, xref='paper', yref='paper',
        showarrow=False, align="left",
        font=dict(size=28)
    )
    param_buttons = []
    n_params = len(params)

    # Store R² + final corr per param so we can use it in the initial annotation
    fit_stats = {}

    for idx, param_name in enumerate(params):
        param_use = param_mapping[param_name]
        df_sub = corr_df[corr_df["param"] == param_use].copy()
        df_sub = df_sub.sort_values("round")

        xvals = df_sub["round"].values
        yvals = df_sub["corr"].values
        n_data_vals = df_sub.get("n_data", pd.Series(index=df_sub.index, dtype=str)).fillna("").astype(str).values

        # Data trace: empirical correlations
        data_trace = go.Scatter(
            name=f"Correlation: {param_containers[param_name]}",
            x=xvals, y=yvals,
            mode="lines+markers",
            line=dict(color='hsla(115, 70%, 30%, 1.0)', width=10),
            marker=dict(size=18, color='hsla(115, 70%, 40%, 1.0)', opacity=1.0),
            visible=True if idx == 0 else False,
            hovertemplate=(
                "Round = %{x}<br>"
                f"Corr = %{{y:.3f}}<br>n_data = %{{text}}<extra>{param_name}</extra>"
            ),
            text=n_data_vals
        )
        fig.add_trace(data_trace)

        # === Best-fit curve (poly or line) ===
        if len(xvals) >= 2:
            if fit_mode == "line":
                deg = 1
            else:  # "poly"
                deg = max(2, int(poly_degree))

            # polyfit needs at least deg+1 points; if not, fall back to lower degree
            deg = min(deg, len(xvals) - 1)
            coef = np.polyfit(xvals, yvals, deg)

            x_fit = np.linspace(xvals.min(), xvals.max(), 200)
            y_fit = np.polyval(coef, x_fit)
            y_pred = np.polyval(coef, xvals)
            r2_val = compute_r2(yvals, y_pred)
        else:
            x_fit = np.array([])
            y_fit = np.array([])
            r2_val = math.nan

        fit_stats[param_name] = (r2_val, yvals[-1] if len(yvals) else math.nan)

        fit_label = (
            f"Linear fit: {param_containers.get(param_name, param_name)}"
            if fit_mode == "line"
            else f"Poly (deg {poly_degree}) fit: {param_containers.get(param_name, param_name)}"
        )

        fit_trace = go.Scatter(
            name=fit_label,
            x=x_fit, y=y_fit,
            mode="lines",
            hoverinfo="skip",
            line=dict(dash='dot', width=10, color='hsla(160, 70%, 40%, 1.0)'),
            visible=True if idx == 0 else False,
        )
        fig.add_trace(fit_trace)

        # Dropdown visibility + annotation text for this param
        visible_list = [False] * (2 * n_params)
        visible_list[2 * idx] = True      # data
        visible_list[2 * idx + 1] = True  # fit

        if len(xvals) >= 2 and not math.isnan(r2_val):
            annotation_text = (
                f"Final corr = {yvals[-1]:.3f}, "
                f"R² ({fit_mode}) = {r2_val:.3f}"
            )
        else:
            annotation_text = "Insufficient data"

        param_buttons.append(dict(
            label=param_containers[param_name],
            method="update",
            args=[
                {"visible": visible_list},
                {
                    "title": (
                        "Correlation Between Fitted Predictor Parameters and "
                        f"True Chooser Parameters by Round for {param_titles[param_name]}"
                    ),
                    "annotations": [dict(text=annotation_text, **annotation_base)]
                }
            ]
        ))

    # Default annotation for the first param
    first_param = params[0]
    r2_first, corr_first = fit_stats.get(first_param, (math.nan, math.nan))
    if not math.isnan(r2_first) and not math.isnan(corr_first):
        ann_text = f"Final corr = {corr_first:.3f}, R² ({fit_mode}) = {r2_first:.3f}"
    else:
        ann_text = "Insufficient data"

    dark_mode = general_settings.get("dark_mode", True)
    max_round = corr_df["round"].max() if not corr_df.empty else 0

    fig.update_layout(
        title="Correlation Between Fitted Predictor Parameters And True Chooser Parameters by Round",
        title_x=fig_lay['title_x'], title_y=fig_lay['title_y'],
        titlefont_size=fig_lay['titlefont_size'] - 15,
        xaxis=dict(
            title="Round",
            range=[-0.5, max_round + 0.5],
            **fig_lay.get("xaxis", {})
        ),
        yaxis=dict(
            title="Correlation",
            range=[-1, 1],
            **fig_lay.get("yaxis", {})
        ),
        template=fig_lay.get("template", "plotly_dark"),
        hoverlabel=fig_lay.get("hoverlabel", {}),
        margin=dict(l=150, r=120, t=120, b=120),
        font=dict(
            family="Calibri",
            color="white" if dark_mode else "black",
            size=32
        ),
        updatemenus=[dict(
            type="dropdown", showactive=True,
            buttons=param_buttons, x=0.88, y=0.2
        )],
        annotations=[dict(
            text=ann_text,
            **annotation_base
        )],
        legend={"x": 0.81, "y": 0.35},
    )

    if not file_name.endswith(".html"):
        file_name += ".html"
    out_path = os.path.join(dir_path, file_name)

    if export_fig:
        fig.write_html(out_path)
        print(f"Correlation-by-round figure saved to {out_path}")
    else:
        fig.show()

    return fig


def compute_prediction_accuracy_by_segment(file_paths: Dict[str, Dict[str, str] | str], general_settings: Dict[str, Any], utility_settings: Dict[str, bool], n_segments: int = 2) -> Dict[str, Any]:
    """
    Compute how participants’ prediction accuracy changes across repeated meetings.

    This function is for the human–bot experiment (Experiment 1). For each predictor:
        1) Reconstructs their dyads and game histories.
        2) Segments repeated encounters with each chooser/avatar into `n_segments`
           (early / middle / late).
        3) Computes accuracy of predictions within each segment:
               accuracy = proportion(choice == prediction).

    In Experiment 2, where choosers are avatars with known preferences, it uses the
    known avatar utility function to reconstruct the avatar’s “true” choice on each
    response-phase round before comparing with human predictions.

    Arguments:
        • file_paths: dict[str, dict[str,str] | str];
            Standard file_paths structure for the Iter_Binary_Dictator pipeline.
        • general_settings: dict[str, Any];
            Must contain 'experiment_num' to choose the appropriate loading logic.
        • utility_settings: dict[str, bool];
            Utility settings used to compute avatar choices in Experiment 2.
        • n_segments: int;
            Number of segments to divide each repeated interaction into
            (e.g., 2 = early vs late; 3 = early / middle / late).

    Returns:
        • dict[str, Any];
            Contains per-segment accuracy summaries. Currently prints and returns
            `accuracy_by_segment` and `accuracy_by_segment_player`.
    """
    avatar_params = {
        'utilitarian': {'Vᵢᵢ':  1.0, 'Vᵢⱼ':  1.0}, 
        'selfish':     {'Vᵢᵢ':  1.0, 'Vᵢⱼ':  0.0}, 
        'competitive': {'Vᵢᵢ':  1.0, 'Vᵢⱼ': -1.0}, 
        'masochistic': {'Vᵢᵢ': -1.0, 'Vᵢⱼ':  1.0}
    }

    experiment_num = general_settings.get('experiment_num')
    plrs_to_dyads = prep.players_to_dyads(
        experiment_num=experiment_num, file_paths=file_paths, create_new_file=False)
    player_uuids = sorted(list(plrs_to_dyads.keys()))

    dir_path = file_paths['processed']
    file_path = f"Social_Preference_Prediction_Pairs_Exper{experiment_num}.json"
    full_path = os.path.join(dir_path, file_path)

    player_histories = None
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as file:
            player_histories = json.load(file)
        if isinstance(player_histories, dict):
            player_histories = player_histories.get('histories')

    if player_histories is None:
        raise Exception(f"Failed to extract player histories")

    players_to_dyads = {}
    for player_uuid, dyads in plrs_to_dyads.items():
        if player_uuid not in players_to_dyads:
            players_to_dyads[player_uuid] = {}
        for dyad_key in dyads:
            if dyad_key in player_histories:
                players_to_dyads[player_uuid][dyad_key] = player_histories[dyad_key]
                # print(player_uuid, dyad_key)

    prediction_accuracy_data = {}
    for player_uuid, player_dyads in players_to_dyads.items():
        print(player_uuid)
        meeting_indices = {}
        prediction_accuracy_data[player_uuid] = []
        for dyad_games in player_dyads.values():
            n_games_in_dyad = int(len(dyad_games) / 2) if experiment_num == 2 else len(dyad_games)
            segment_size = n_games_in_dyad // n_segments
            # segment_size = 1
            # if n_games_in_dyad < 8:
            #     continue
            for dyad_game in dyad_games:
                choice = dyad_game.get('choice')
                chooser = dyad_game.get('chooser')
                prediction = dyad_game.get('prediction')
                predictor = dyad_game.get('predictor')
                round_num = dyad_game.get('round')
                if player_uuid != predictor:
                    continue
                if experiment_num == 2:
                    "In experiment 2, use what the avatar would have chosen in the response phase."
                    if dyad_game['phase'] == 'op':
                        continue
                    chooser = chooser.split("-")[1]
                    As = dyad_game.get('payoff_A_chooser', 0)
                    Ao = dyad_game.get('payoff_A_predictor', 0)
                    Bs = dyad_game.get('payoff_B_chooser', 0)
                    Bo = dyad_game.get('payoff_B_predictor', 0)
                    Vᵢᵢ = avatar_params[chooser]['Vᵢᵢ']
                    Vᵢⱼ = avatar_params[chooser]['Vᵢⱼ']
                    payoffsA = {'As': As, 'Ao': Ao, 'Bs': Bs, 'Bo': Bo}
                    payoffsB = {'As': Bs, 'Ao': Bo, 'Bs': As, 'Bo': Ao}
                    params = {'Vᵢᵢ': Vᵢᵢ, 'Vᵢⱼ': Vᵢⱼ}
                    utilityA = utility(payoffs=payoffsA, params=params, utility_settings=utility_settings)
                    utilityB = utility(payoffs=payoffsB, params=params, utility_settings=utility_settings)
                    if utilityA > utilityB:
                        choice = 'A'
                    elif utilityA < utilityB:
                        choice = 'B'
                    else:
                        choice = random.choice(seq=('A', 'B'))

                if chooser not in meeting_indices:
                    meeting_indices[chooser] = 0
                meeting_index = int(meeting_indices[chooser])
                meeting_indices[chooser] += 1

                if segment_size == 0:
                    meeting_segment = 0
                else:
                    meeting_segment = meeting_index // segment_size

                prediction_is_accurate = int(choice == prediction)
                prediction_accuracy_data[player_uuid].append(
                    (predictor, chooser, round_num, meeting_index, meeting_segment, prediction_is_accurate)
                )

    accuracy_by_segment = {}
    accuracy_by_segment_player = {}
    for player_uuid, accuracy_data in prediction_accuracy_data.items():
        accuracy_by_segment_player[player_uuid] = {}
        for accuracy_tuple in accuracy_data:
            player_uuid, counterpart_uuid, round_num, meeting_index, meeting_segment, accurate = accuracy_tuple
            if meeting_segment not in accuracy_by_segment:
                accuracy_by_segment[meeting_segment] = {'accurate': 0, 'inaccurate': 0, 'total': 0}
            if meeting_segment not in accuracy_by_segment_player[player_uuid]:
                accuracy_by_segment_player[player_uuid][meeting_segment] = {'accurate': 0, 'inaccurate': 0, 'total': 0}
            if accurate:
                accuracy_by_segment_player[player_uuid][meeting_segment]['accurate'] += 1
                accuracy_by_segment[meeting_segment]['accurate'] += 1
            else:
                accuracy_by_segment_player[player_uuid][meeting_segment]['inaccurate'] += 1
                accuracy_by_segment[meeting_segment]['inaccurate'] += 1
            accuracy_by_segment_player[player_uuid][meeting_segment]['total'] += 1
            accuracy_by_segment[meeting_segment]['total'] += 1

    for segment_num, scores in accuracy_by_segment.items():
        accurate, total = scores['accurate'], scores['total']
        scores['ratio'] = round(accurate / total, 6)

    pp.pprint(accuracy_by_segment)
    return {
        "accuracy_by_segment": accuracy_by_segment,
        "accuracy_by_segment_player": accuracy_by_segment_player
    } 


"=========================================================================================="
"======== Simulation 3) Prior Variance and Temperature Affect Belief Update Speed ========="
"=========================================================================================="

def compute_belief_update_speed(dyad_games: List[Dict[str, Any]], player_uuid: str, general_settings: Dict[str, Any], 
                                true_parameters: Optional[Dict[str, float]] = None, params_of_interest: Optional[list[str]] = None, fraction: float = 0.5) -> float:
    """
    Quantify how quickly a predictor's fitted parameters move toward their target.

    This implements the “update speed” measure described in the paper’s simulation
    section (Prior Variance and Temperature Affect Belief Update Speed). It returns
    a scalar in [0, 1] indicating the earliest round at which the fitted parameters
    cross a given fraction of the total distance between start and target.

    Two modes:
        • Absolute mode (model-to-truth):
            If `true_parameters` is provided, we track the Euclidean distance between
            the predictor’s fitted vector and the ground-truth vector each round.
            We compute:
                d(0)  = initial distance
                d(F)  = final distance
                d*    = d(0) + fraction · (d(F) – d(0))
            and return the earliest round index t where the distance crosses d*,
            normalized by the total number of intervals (t / T).

        • Relative mode (prior-to-posterior):
            If `true_parameters` is None, we track the distance between param(t)
            and the initial param(0), and find the earliest t where the distance
            reaches `fraction` of the total prior→final shift.

    Arguments:
        • dyad_games: list[dict[str, Any]];
            Full game history for a single dyad (all rounds, all roles).
        • player_uuid: str;
            UUID of the predictor whose update speed we are measuring.
        • general_settings: dict[str, Any];
            Settings that specify the update method (e.g., "grid") and experiment_num.
        • true_parameters: dict[str, float] | None;
            Ground-truth parameter vector to move toward (e.g., chooser parameters
            in simulations). If None, use the predictor’s own final parameters as
            the “target”.
        • params_of_interest: list[str] | None;
            Subset of parameter keys to include (e.g., ["Vᵢᵢ","Vᵢⱼ"]). If None,
            use all non-std / non-cov / non-temp entries in the fitted vector.
        • fraction: float;
            Fraction of the total distance toward the target that defines “arrival”.
            In the paper, 0.5 corresponds to “halfway update speed”.

    Returns:
        • float in [0, 1];
            Normalized crossing time:
                0   → very fast update (threshold crossed immediately),
                1   → slowest (threshold never crossed before the final round).
    """
    update_method = general_settings.get('update_method', 'grid')
    experiment_num= general_settings.get('experiment_num', None)

    # 1) gather param vectors or scalars per round
    round_paramvals = []  # list of (round_number, param_vector)
    for game in dyad_games:
        # must match the predictor
        if game.get('predictor') != player_uuid:
            continue
        if experiment_num == 2 and game.get('phase') == 'rp':
            continue

        param_est = (
            game
            .get('parameter_estimates', {})
            .get(update_method, {})
            .get(player_uuid, {})
            .get('predictor', {})
            .get('params', {})
        )
        if not param_est:
            continue
        # build a param vector ignoring _std, etc. if params_of_interest is None => keep all
        relevant_vals = []
        for pkey, pval in param_est.items():
            if any(x in pkey for x in ['_std','_cov','temp']):
                continue
            if params_of_interest is not None:
                if pkey not in params_of_interest:
                    continue
            relevant_vals.append(pval)

        rnd = game.get('round', None)
        if relevant_vals and rnd is not None:
            round_paramvals.append((rnd, tuple(relevant_vals)))

    if len(round_paramvals) < 2:
        raise ValueError("Not enough param snapshots to measure an update speed.")
    
    # 2) sort by round
    round_paramvals.sort(key=lambda x: x[0])
    # just store them in arrays for convenience
    rounds = [rp[0] for rp in round_paramvals]
    param_series = [rp[1] for rp in round_paramvals]  # each is a tuple of length k

    # define a function to get Euclidian distance between param vectors
    def dist_vec(v1, v2):
        return math.sqrt(sum((a-b)**2 for (a,b) in zip(v1,v2)))

    if true_parameters is not None:
        # "absolute" => compare param(t) to the ground truth each round
        # build ground_truth vector, ignoring _std, etc., same dimension
        ground_vals = []
        for pkey in sorted(true_parameters.keys()):
            if any(x in pkey for x in ['_std','_cov','temp']):
                continue
            if params_of_interest is not None and pkey not in params_of_interest:
                continue
            ground_vals.append(true_parameters[pkey])
        ground_vec = tuple(ground_vals)
        if len(ground_vec) != len(param_series[0]):
            raise ValueError("Mismatch in dimension between param_of_interest and ground truth.")
        
        # now param_series[t] => fitted param vector
        # distance(t) => dist_vec( param_series[t], ground_vec )
        dist_series = [dist_vec(pv, ground_vec) for pv in param_series]
        prior_dist = dist_series[0]
        final_dist = dist_series[-1]
        # threshold
        threshold = prior_dist + fraction*(final_dist - prior_dist)
        # find earliest t crossing
        # if final_dist < prior_dist, we look for dist_series[t] <= threshold
        # if final_dist > prior_dist, we look for dist_series[t] >= threshold
        # but that might be reversed if final < prior => negative. Let's define a function:
        direction = 1 if final_dist > prior_dist else -1

        T = len(dist_series)-1  # total # intervals
        crossing_round = T  # default to last => speed=1 if never cross
        for i, dval in enumerate(dist_series):
            # i=0.. len-1
            if direction>0:
                if dval >= threshold:
                    crossing_round = i
                    break
            else:
                if dval <= threshold:
                    crossing_round = i
                    break
        # update_speed = crossing_round / T
        if T<=0:
            return 0.0
        speed = crossing_round / float(T)
        if speed>1:
            speed=1
        return speed

    else:
        # "relative" => time to cross half difference from param(0) to param(final)
        prior_vec = param_series[0]
        final_vec = param_series[-1]
        T = len(param_series)-1

        # define threshold vector = prior_vec + fraction*(final_vec - prior_vec)
        thresh_vec = tuple(
            p0 + fraction*(pF - p0)
            for (p0,pF) in zip(prior_vec, final_vec)
        )

        # direction => sign of final-prior (componentwise?? we have multiple dims).
        # We'll do the Euclidian distance approach:
        prior_dist = dist_vec(prior_vec, prior_vec)  # which is 0
        final_dist = dist_vec(final_vec, prior_vec)  # the total shift from prior->final
        target_dist = fraction * final_dist

        # each round => measure dist from prior
        dist_from_prior = [dist_vec(pv, prior_vec) for pv in param_series]

        crossing_round = T
        for i, dval in enumerate(dist_from_prior):
            # once dval >= target_dist => crossed fraction
            if dval >= target_dist:
                crossing_round = i
                break

        if T<=0:
            return 0.0
        speed = crossing_round / float(T)
        if speed>1:
            speed=1
        return speed


def run_update_speed_simulation_regression(general_settings: GeneralSettings, file_paths: FilePaths, 
                                           params_of_interest: Optional[list[str]] = ['Vᵢⱼ'], 
                                           use_true_params: bool = False, n_dyads: int | None = 729) -> None:
    """
    Estimate how simulated belief update speed depends on prior variance and temperature.

    This function:
        1) Loads simulated dyads from JSON (created by the bot–bot simulation).
        2) For each dyad, computes an update speed for the predictor via
           `compute_belief_update_speed`.
        3) Extracts each predictor’s initial fitted variance and temperature.
        4) Runs a linear regression:
               update_speed ~ τ_fitted_predictor + Vᵢⱼ_std_fitted_predictor
           mirroring the regression in the paper’s simulation section.

    Arguments:
        • general_settings: GeneralSettings;
            Should match the settings used when fitting the simulated dyads
            (e.g., update_method = "grid").
        • params_of_interest: list[str] | None;
            Parameters used when constructing the belief vector for speed measurement
            (e.g., ["Vᵢⱼ"] or ["Vᵢᵢ","Vᵢⱼ"]).
        • json_path: str;
            Directory containing the per-dyad simulation fit JSON files.
        • use_true_params: bool;
            If True, compute “absolute” speed toward the chooser’s true parameters.
            If False, compute “relative” speed from prior to final fitted values.
        • n_dyads: int | None;
            Number of dyad files to process. If None, use all JSON files in `json_path`.

    Returns:
        • None;
            Prints statsmodels OLS summary for the speed ~ variance + temperature regression.
    """
    def run_regression_on_speed(df_speed: pd.DataFrame, speed_col: str = "speed_value", predictors: list[str] = ["temp", "var"], add_constant: bool = True):
        """
        Fit a linear regression: speed_col ~ predictors using statsmodels OLS.
        """
        import statsmodels.api as sm
        # drop na
        df_ = df_speed.dropna(subset=[speed_col]+predictors).copy()
        y = df_[speed_col]
        X = df_[predictors]
        if add_constant:
            X = sm.add_constant(X, prepend=True)
        model = sm.OLS(y, X)
        results = model.fit()
        print(results.summary())
        return results

    json_path = ensure_directory_and_join(base_dir=file_paths['player_fits'], file_name="experiment_0")

    update_method = 'grid'
    param_key_map = {
        'Vii':  'Vᵢᵢ', 'Vii_std': 'Vᵢᵢ_std', 
        'Vij':  'Vᵢⱼ', 'Vij_std': 'Vᵢⱼ_std',
        'temp': 'τ', 't': 'τ'
    }
    params_of_interest = [
        param_key_map.get(param, param) for param in params_of_interest
    ]

    if not isinstance(n_dyads, int):
        files = prep.get_files_in_directory(directory_path=json_path)
        n_dyads = len(files)

    params_to_us = {}
    for dyad_idx in range(n_dyads):
        simulated_dyad = get_simulated_dyad(dyad_idx=dyad_idx, json_path=json_path, n_games=21)

        dyad_key = list(simulated_dyad.keys())[0]
        dyad_games = simulated_dyad[dyad_key]
        first_game: dict = dyad_games[0]
        chooser_uuid, predictor_uuid = first_game['chooser'], first_game['predictor']
        param_est = first_game.get('parameter_estimates', {}).get(update_method, {}
                        ).get(predictor_uuid, {}).get('predictor', {}).get('params', {})
        fitted_params_predictor = {
            param_key_map.get(param_key, param_key): param_val 
            for param_key, param_val in param_est.items() 
        }
        true_params_chooser = parse_robot_string(robot_str=chooser_uuid)
        true_params_chooser = {
            param_key_map.get(param_key, param_key): param_val 
            for param_key, param_val in true_params_chooser.items() 
        }        
        true_params_predictor = parse_robot_string(robot_str=predictor_uuid)
        true_params_predictor = {
            param_key_map.get(param_key, param_key): param_val 
            for param_key, param_val in true_params_predictor.items() 
        }

        update_speed = compute_belief_update_speed(dyad_games=dyad_games, player_uuid=predictor_uuid, fraction=0.5, 
                                        general_settings=general_settings, params_of_interest=params_of_interest, 
                                        true_parameters={
                                            param_key: param_val for param_key, param_val in true_params_chooser.items() 
                                            if param_key in params_of_interest
                                        } if use_true_params else None)

        params_to_us[dyad_key] = {
            'dyad_idx': dyad_idx,
            'dyad_key': dyad_key,
            'predictor_uuid': predictor_uuid,
            'chooser_uuid': chooser_uuid,
            **{f'{param_key}_true_predictor': param_val for param_key, param_val in true_params_predictor.items()},
            **{f'{param_key}_fitted_predictor': param_val for param_key, param_val in fitted_params_predictor.items()},
            **{f'{param_key}_true_chooser': param_val for param_key, param_val in true_params_chooser.items()},
            'update_speed': update_speed
        }

        if dyad_idx % 40 == 0: print(dyad_idx)

    update_speed_df = pd.DataFrame.from_dict(params_to_us, orient='index')
    res = run_regression_on_speed(
        df_speed=update_speed_df,
        speed_col="update_speed",
        predictors=["τ_fitted_predictor","Vᵢⱼ_std_fitted_predictor"]
    )


def analyze_update_speed_in_human_bot(file_paths: Dict[str, Dict[str, str] | str], general_settings: Dict[str, Any]) -> None:
    """
    Summarize belief update speed in the human–bot experiment for each participant and avatar type.

    This function:
        1) Loads player-level dyad fits from disk.
        2) For each dyad, computes an update speed for the predictor using
           `compute_belief_update_speed`, typically toward the avatar’s true (Vᵢᵢ, Vᵢⱼ).
        3) Aggregates update speeds:
               • per predictor (mean, std),
               • per counterpart (avatar type; mean, std).
        4) Returns these summaries for further plotting or regression.

    This corresponds to the empirical update-speed analysis that complements the
    simulation results in the paper.

    Arguments:
        • file_paths: dict[str, dict[str,str] | str];
            Standard Iter_Binary_Dictator file paths, including 'player_fits'.
        • general_settings: dict[str, Any];
            Must contain 'experiment_num' to know whether counterparts are human or avatars.
        • utility_settings: dict[str, bool];
            Currently unused here but included for symmetry with other analysis functions.
        • n_segments: int;
            Retained for compatibility; not currently used in this function.

    Returns:
        • dict[str, Any];
            {
                'update_speeds': dict[predictor_uuid → dict[counterpart_id → speed]],
                'update_speeds_per_predictor': dict[predictor_uuid → list[speed]],
                'update_speeds_per_counterpart': dict[counterpart_id → list[speed]],
                'update_speeds_per_predictor_mean_std': dict[predictor_uuid → {'mean','std'}],
                'update_speeds_per_counterpart_mean_std': dict[counterpart_id → {'mean','std'}],
            }
    """
    avatar_params = {
        'utilitarian': {'Vᵢᵢ':  1.0, 'Vᵢⱼ':  1.0}, 
        'selfish':     {'Vᵢᵢ':  1.0, 'Vᵢⱼ':  0.0}, 
        'competitive': {'Vᵢᵢ':  1.0, 'Vᵢⱼ': -1.0}, 
        'masochistic': {'Vᵢᵢ': -1.0, 'Vᵢⱼ':  1.0}
    }

    experiment_num = general_settings.get('experiment_num')
    plrs_to_dyads = prep.players_to_dyads(experiment_num=2, 
                    file_paths=file_paths, create_new_file=False)
    player_uuids = sorted(list(plrs_to_dyads.keys()))

    update_speeds = {}
    prediction_accuracy_data = {}
    for player_uuid in player_uuids:
        player_dyads = None
        file_name_suffix = file_paths["file_name_suffix"]
        plr_file_path = os.path.join(file_paths["player_fits"], f"experiment_{experiment_num}", 
                                f'{file_name_suffix}_' + player_uuid + ".json")

        if os.path.exists(plr_file_path):
            with open(plr_file_path, "r", encoding='utf-8') as file:
                player_dyads: dict = json.load(file)

        if player_dyads is None:
            raise Exception(f"Failed to extract dyads for player {player_uuid}.")        

        update_speeds_per_counterpart = {}
        for dyad_key, dyad_games in player_dyads.items():
            true_params = None
            counterpart_uuid = None
            if experiment_num == 2:
                avatar_uuid = dyad_games[0]['chooser']
                counterpart_type = None
                for avatar_type in ('utilitarian', 'selfish', 'competitive', 'masochistic'):
                    if avatar_type in avatar_uuid:
                        counterpart_type = avatar_type
                if counterpart_type is None:
                    raise Exception("Avatar type not found.")
                counterpart_uuid = counterpart_type
                true_params = avatar_params[counterpart_type]
            else:
                counterpart_uuid = prep._dyad_key(dyad_key=dyad_key, return_tuple=True)
                if counterpart_uuid == player_uuid:
                    counterpart_uuid = prep._dyad_key(dyad_key=dyad_key, return_tuple=True, reverse=True)

            update_speed = compute_belief_update_speed(dyad_games=dyad_games, player_uuid=player_uuid, general_settings=general_settings, 
                                                true_parameters=true_params, params_of_interest=["Vᵢᵢ", "Vᵢⱼ"], fraction=0.75)

            update_speeds_per_counterpart[counterpart_uuid] = update_speed
        update_speeds[player_uuid] = update_speeds_per_counterpart   

    update_speeds_per_predictor = {}
    update_speeds_per_counterpart = {}
    for player_uuid, stabilization_per_cnterprt in update_speeds.items():
        if player_uuid not in update_speeds_per_predictor:
            update_speeds_per_predictor[player_uuid] = []
        for counterpart_uuid, update_speed in stabilization_per_cnterprt.items():
            if counterpart_uuid not in update_speeds_per_counterpart:
                update_speeds_per_counterpart[counterpart_uuid] = []
            update_speeds_per_counterpart[counterpart_uuid].append(update_speed)
            update_speeds_per_predictor[player_uuid].append(update_speed)

    update_speeds_per_predictor_mean_std = {
        player_uuid: {
            'mean': np.mean(np.array(rates)), 
            'std': np.std(np.array(rates))
        } for player_uuid, rates in update_speeds_per_predictor.items()
    }
    update_speeds_per_counterpart_mean_std = {
        counterpart: {
            'mean': np.mean(np.array(rates)), 
            'std': np.std(np.array(rates))
        } for counterpart, rates in update_speeds_per_counterpart.items()       
    }

    return {
        'update_speeds': update_speeds,
        'update_speeds_per_predictor': update_speeds_per_predictor,
        'update_speeds_per_counterpart': update_speeds_per_counterpart,
        'update_speeds_per_predictor_mean_std': update_speeds_per_predictor_mean_std,
        'update_speeds_per_counterpart_mean_std': update_speeds_per_counterpart_mean_std,
    }


def plot_update_speed_by_counterpart(update_speeds_per_counterpart: Dict[str, List[float]], fig_lay: Dict[str, Any], export_fig: bool = True, 
                                     file_name: str = "update_speed_violin_boxplot.html", as_bar_chart: bool = True) -> go.Figure:
    """
    Visualize belief-update speed across avatar types / counterparts.

    This is primarily for Experiment 1/2, showing how quickly participants
    learn different social preference profiles (utilitarian, selfish, etc.).

    In bar mode (default), the figure shows:
        • one bar per counterpart with mean update speed
        • 95% confidence intervals as error bars.

    In the alternative mode, you can reuse this as a violin/box plot
    (see inline comments) to show the full distribution per counterpart.

    Arguments:
        • update_speeds_per_counterpart: dict[str, list[float]];
            Mapping from counterpart label (e.g., "utilitarian") to a list of
            normalized update speeds across participants.
        • fig_lay: dict[str, Any];
            Plotly layout settings (template, fonts, axes, etc.).
        • export_fig: bool;
            If True, save the figure as an HTML file at `out_path`.
        • out_path: str;
            Path to the output HTML file if `export_fig` is True.
        • as_bar_chart: bool;
            If True (default), plot means with CIs as bars. If False, use
            the alternative (currently bar-structured but ready to be adapted
            to violin/box if desired).

    Returns:
        • go.Figure;
            The constructed Plotly figure.
    """
    def mean_confidence_interval(data, confidence=0.95, rnd=4):
        """This conputes the CI for continious data."""
        import scipy.stats
        a = 1.0 * np.array(data)
        n, m, se = len(a), np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return (round(m-h, rnd), round(m, rnd), round(m+h, rnd))

    out_path = ensure_directory_and_join(file_paths['visuals'], file_name)

    # Build the figure
    fig = go.Figure()

    if as_bar_chart:
        counterparts = [counterpart.capitalize() for counterpart in update_speeds_per_counterpart.keys()]
        colors = [f'hsla({int(115 + 360/(len(counterparts)+4) * idx) % 360}, 80%, 40%, 1.0)' for idx in range(len(counterparts))]
        update_speeds_per_counterpart_mean_std = {
            counterpart: {
                'mean': np.mean(np.array(rates)), 
                'std': np.std(np.array(rates)),
                'ci': gnrl.mean_confidence_interval(data=rates, confidence=0.95, rnd=6)
            } for counterpart, rates in update_speeds_per_counterpart.items()       
        }
        update_speeds_means = [val['mean'] for val in update_speeds_per_counterpart_mean_std.values()]
        update_speeds_stds = [val['std'] for val in update_speeds_per_counterpart_mean_std.values()]
        update_speeds_cis = [val['ci'] for val in update_speeds_per_counterpart_mean_std.values()]
        update_speeds_cis_upper = [round(ci[2] - ci[1], 8) for ci in update_speeds_cis]
        update_speeds_cis_lower = [round(ci[1] - ci[0], 8) for ci in update_speeds_cis]
        update_speeds_means = [round(mean, 6) for mean in update_speeds_means]
        
        all_update_speeds = []
        for rates in update_speeds_per_counterpart.values():
            all_update_speeds += rates
        mean_update_speed = np.mean(np.array(all_update_speeds))
        std_update_speed = np.std(np.array(all_update_speeds))
        ci_update_speed = gnrl.mean_confidence_interval(data=all_update_speeds, confidence=0.95, rnd=6)
        print(f"Mean Update Speed: {mean_update_speed}")
        print(f"Std Update Speed:  {std_update_speed}")
        print(f"CI Update Speed:   {ci_update_speed}")

        data= {
            'counterpart': counterparts,
            'means': update_speeds_means,
            'ci_low': update_speeds_cis_lower,
            'ci_high': update_speeds_cis_upper,
            'std': update_speeds_stds
        }
        df = pd.DataFrame(data=data)
        print(df)

        fig = go.Figure([go.Bar(
            x=counterparts, 
            y=update_speeds_means,
            marker_color=colors,
            error_y=dict(
                type='data', 
                array=update_speeds_cis_upper, 
                arrayminus=update_speeds_cis_lower
            ),
            hovertemplate=(
                "Counterpart: %{x}<br>"
                "Update Speed: %{y:.3f}<extra></extra>"
            )
        )])
    else:
    # Create one Violin trace (which can also be displayed as a Box) per counterpart
        for counterpart, speeds in update_speeds_per_counterpart.items():
            fig.add_trace(
                go.Bar(
                    x=[counterpart]*len(speeds),  # Category on x-axis
                    y=speeds,                     # Data points on y-axis
                    box=dict(visible=True),
                    meanline=dict(visible=True),
                    points='all',                 # Show individual data points
                    pointpos=-0.7,
                    jitter=0.45,
                    scalemode='count',
                    width=0.4,
                    name=counterpart,             # Legend / name for this trace
                    line_color='hsla(115, 70%, 40%, 1.0)',  # A sample color style
                    hovertemplate=(
                        "Counterpart: %{x}<br>"
                        "Update Speed: %{y:.3f}<extra></extra>"
                    )
                )
            )

    # Title and layout
    # (example: you can adapt the text as you wish)
    fig.update_layout(
        title="Belief Update Speed by Counterpart",
        titlefont_size=fig_lay.get("titlefont_size", 18) - 4,
        template=fig_lay.get("template", "plotly_dark"),
        title_x=fig_lay.get('title_x', 0.5),
        title_y=fig_lay.get('title_y', 0.95),

        xaxis=dict(
            title="Counterpart Preference Profile",
            **fig_lay.get("xaxis", {})
        ),
        yaxis=dict(
            range=[0.0, 0.5] if as_bar_chart else [0.0, 0.5],
            title="Mean Update Speed (fraction of rounds)",
            **fig_lay.get("yaxis", {})
        ),
        hoverlabel=fig_lay.get("hoverlabel", {}),
        font=fig_lay.get("font", {}),
        margin=dict(l=600, r=600, t=120, b=120) if as_bar_chart \
            else dict(l=150, r=120, t=120, b=120)
    )

    if not as_bar_chart:
        # Dropdown menu to switch between violin and box
        fig.update_layout(
            updatemenus=[dict(
                buttons=list([
                    dict(
                        args=["type", "violin"], 
                        label="Violin", 
                        method="restyle"
                    ),
                    dict(
                        args=["type", "box"], 
                        label="Boxplot", 
                        method="restyle"
                    )
                ]),
                direction="down", 
                x=1.03, 
                xanchor="left",
                y=0.7, 
                yanchor="top",
                pad={"r": 10, "t": 10},
                showactive=True
            )]
        )

    # Export or show
    if export_fig:
        fig.write_html(out_path)
        print("Saved violin-box figure to", out_path)
    else:
        fig.show()

    return fig


"=========================================================================================="
"============================== Illustrating Belief Updates ==============================="
"=========================================================================================="


def visualize_bayesian_updates_2d(player_uuid: str | int, counterpart_uuid: str | int, player_role: PlayerRole, general_settings: GeneralSettings, 
                                  utility_settings: UtilitySettings, file_paths: FilePaths, fig_lay: FigLay, n_rounds: int = 5, dark_zero_lines: bool = True) -> None:
    """
    Creates a 3xN figure (rows x columns) that visualizes, for each round (column):
      1) The observed choice in 2D payoff-difference space (row 1).
      2) The likelihood heatmap over (Vᵢᵢ, Vᵢⱼ) (row 2).
      3) The prior/posterior heatmap (row 3).

    The code extracts the relevant filtered games from the player's data file,
    finds the parameter grids for each game, and constructs the subplots.

    Arguments:
        • player_uuid, counterpart_uuid: Identifiers (int or str) for which dyad to load.
        • player_role: Typically "chooser" or "predictor"; we filter games by this role.
        • file_paths: dict[str: str | dict[str: str]]; Dictionary of all files paths.  
        • general_settings: Contains 'experiment_num' and other important settings, etc.
        • fig_lay: A dictionary specifying layout (template, width, height, colorscales, etc.).
        • dark_zero_lines: bool; if True, we darken the zero lines in top row payoff space.

    Returns:
        A Plotly Figure with 3 rows, N columns (N = number of filtered games).
    """
    # -----------------------------------------------------------------------
    # 1) Filter the dyad games for relevant rounds (like your snippet).
    # -----------------------------------------------------------------------
    experiment_num = general_settings.get('experiment_num', 3)
    player_uuids = prep.all_player_uuids(file_paths=file_paths, experiment_num=experiment_num, only_humans=False ) #HACK DELETE!!!
    if experiment_num == 0:
        player_uuids = [uuid for uuid in player_uuids if 'predictor' in uuid]

    # Convert player_uuid index => actual string
    if isinstance(player_uuid, int):
        plrs_to_dyads = prep.players_to_dyads(experiment_num=experiment_num,
                                                    file_paths=file_paths, create_new_file=False)
        plr_keys = sorted(list(plrs_to_dyads.keys()))
        if experiment_num == 0:
            plr_keys = [key for key in plr_keys if 'chooser' not in key]
        player_uuid = plr_keys[player_uuid % len(plr_keys)]

    elif isinstance(player_uuid, str):
        if player_uuid not in player_uuids:
            raise IndexError(f"player_uuid '{player_uuid}' not found among {player_uuids}.")

    else:
        raise ValueError(f"player_uuid type {type(player_uuid)} not supported.")

    # Load the player's data file
    plr_file_path = os.path.join(
        file_paths["player_fits"], f"experiment_{experiment_num}",
        player_uuid + ".json"  # or your suffix logic
    )

    if not os.path.exists(plr_file_path):
        raise FileNotFoundError(f"File not found for player: {player_uuid} => {plr_file_path}")

    with open(plr_file_path, "r", encoding='utf-8') as file:
        player_dyads = json.load(file)
    if player_dyads is None:
        raise Exception(f"Failed to extract dyads for player {player_uuid}")

    dyad_keys = sorted(list(player_dyads.keys()))

    dyad_key = None
    if isinstance(counterpart_uuid, int):
        dyad_key = dyad_keys[counterpart_uuid % len(dyad_keys)]
        plr_uuid_1, plr_uuid_2 = prep._dyad_key(dyad_key=dyad_key, return_tuple=True)
        counterpart_uuid = plr_uuid_2 if plr_uuid_1 == player_uuid else plr_uuid_1
        
    found_counterpart = False
    for dkey in dyad_keys:
        if counterpart_uuid in dkey:
            found_counterpart = True
            break

    if not found_counterpart:
        # fallback if you want an index-based approach
        raise Exception(f"Counterpart {counterpart_uuid} not in {dyad_keys}")

    if dyad_key is None:
        dyad_key = prep._dyad_key(dyad_key=f"({player_uuid}, {counterpart_uuid})")

    dyad_games = player_dyads[dyad_key]
    # Filter games by the role we want
    filtered_games = [g for g in dyad_games if g.get(player_role, None) == player_uuid]
    if not filtered_games:
        raise Exception("Failed to extract filtered dyad games for role='{}'.".format(player_role))

    n_filtered_games = len(filtered_games)
    if n_rounds > n_filtered_games:
        n_rounds = n_filtered_games

    # -----------------------------------------------------------------------
    # 2) Prepare subplots: 3 rows, n_rounds columns
    # -----------------------------------------------------------------------
    fig = make_subplots(
        rows=3, cols=n_rounds,
        horizontal_spacing=0.05, vertical_spacing=0.07,
        subplot_titles=tuple([f"Round {rval}" for rval in range(
            1, n_rounds + 1)] + ["" for rval in range(1, (n_rounds + 1) * 2)])
    )

    # # Optional: set the subplot titles.
    # # We'll just label columns as Round 1..N for the top row.
    # for cdx in range(1, n_rounds+1):
    #     fig.add_annotation(
    #         text=f"Round {cdx}",
    #         xref="x domain", yref="y domain",
    #         x=0.5+(cdx-1)/n_rounds, y=1.02,
    #         showarrow=False,
    #         font=dict(size=20),
    #         row=1, col=cdx
    #     )

    max_prior_prob = 0.0
    for game in filtered_games:
        grid_predictor = game.get("parameter_estimates", {}).get(
            "grid", {}).get(player_uuid, {}).get("predictor", {}).get('parameter_vectors', {})        
        for probability in grid_predictor.values():
            if probability > max_prior_prob:
                max_prior_prob = probability

    # -----------------------------------------------------------------------
    # 3) For each round => we create 3 subplots
    # -----------------------------------------------------------------------
    for col_idx, game in enumerate(filtered_games, start=1):
        if col_idx > n_rounds:
            break
        # (A) Top row => Observed choice in payoff-diff space
        # e.g. payoff_A_chooser, payoff_B_chooser, etc.
        payoff_As = game.get("payoff_A_chooser", 0)
        payoff_Bs = game.get("payoff_B_chooser", 0)
        payoff_Ao = game.get("payoff_A_predictor", 0)
        payoff_Bo = game.get("payoff_B_predictor", 0)

        if game.get("choice") == "A":
            payoff_diff_self = payoff_As - payoff_Bs
            payoff_diff_other = payoff_Ao - payoff_Bo
        else:
            payoff_diff_self = payoff_Bs - payoff_As
            payoff_diff_other = payoff_Bo - payoff_Ao

        # print((payoff_As, payoff_Ao), (payoff_Bs, payoff_Bo))
        
        dot_trace = go.Scatter(
            x=(payoff_diff_self,),
            y=(payoff_diff_other,),
            mode="markers",
            marker=dict(size=16, color="green"),
            showlegend=False,
            hovertemplate=(
                f"<b>Observed Choice:</b> {game.get('choice')}<br>"
                f"ΔSelf: {payoff_diff_self}<br>"
                f"ΔOther: {payoff_diff_other}<extra></extra>"
            )
        )
        fig.add_trace(dot_trace, row=1, col=col_idx)

        # (B) Middle row => "Likelihood" heatmap
        # We'll reconstruct the Nx x Ny array from your approach:
        #  1) Build the param grid from meta_data/tickvals
        #  2) For each point, compute p(choose A) or 1 - p(choose A)
        # For demonstration, let's see if there's "likelihood" array or we do your approach.

        # We'll replicate logic from your "3D" code but produce a 2D array.
        grid_predictor = game.get("parameter_estimates", {}).get(
            "grid", {}).get(player_uuid, {}).get("predictor", None)
        # pp.pprint(grid_predictor), exit()
        if not grid_predictor:
            # If no grid data, we'll skip
            continue

        meta_data = grid_predictor.get("meta_data", {})
        tickvals = meta_data.get("tickvals", {})

        if 'sVs' in tickvals:
            tickvals["Vᵢᵢ"] = tickvals.pop("sVs")  #HACK DELETE!!
            tickvals["Vᵢⱼ"] = tickvals.pop("sVo")  #HACK DELETE!!

        if "Vᵢᵢ" not in tickvals or "Vᵢⱼ" not in tickvals:
            continue  # skip if incomplete
 
        # Extract param axes
        Vii_vals = np.array(tickvals["Vᵢᵢ"], dtype=float)
        Vij_vals = np.array(tickvals["Vᵢⱼ"], dtype=float)
        Nx, Ny = len(Vii_vals), len(Vij_vals)

        # Build a 2D array for likelihood
        likelihood_2d = np.zeros((Nx, Ny), dtype=float)
        # We can do your choice(...) approach if you want the "true" function, or
        # re-construct from param_vectors. Typically you do:
        for ix, v_ii in enumerate(Vii_vals):
            for jx, v_ij in enumerate(Vij_vals):
                agent_params = {
                    "Vᵢᵢ": v_ii,
                    "Vᵢⱼ": v_ij
                }
                # Compute p(choose A)
                # (We might need other param defaults or exponents.)
                # We'll just do a fallback exponent or read from ...
                p_choose_A = choice(
                    current_game=game,
                    agent_params=agent_params,
                    utility_settings=utility_settings,  # or pass the relevant toggles
                    softmax_temperature=grid_predictor.get('params', {}).get('temp', 1.5) * 0.75, #HACK This makes the figure more visible
                    # softmax_temperature=1.2, #HACK This makes the figure more visible
                    select=False
                )["model_choose_A"]

                if game.get("choice") == "A":
                    likelihood_2d[ix, jx] = p_choose_A
                else:
                    likelihood_2d[ix, jx] = 1.0 - p_choose_A
        
        # Make the Heatmap (transposing if you want x-> Nx, y-> Ny)
        likelihood_hm = go.Heatmap(
            x=Vii_vals,
            y=Vij_vals,
            z=likelihood_2d.T,
            colorscale=fig_lay.get("colorscales", ["Viridis"])[0],
            hovertemplate=("Vᵢᵢ: %{x:.3f}, Vᵢⱼ: %{y:.3f}<br>Lik: %{z:.3f}<extra></extra>"),
            showscale=False, zmin=0, zmax=1,
        )
        fig.add_trace(likelihood_hm, row=2, col=col_idx)

        # (C) Bottom row => prior/posterior
        # We'll parse param_vectors => build a 2D pmf => fill holes => sum => normalize
        param_vectors = grid_predictor.get("param_vectors", {})
        prior_2d = np.full((Nx, Ny), np.nan)
        for idx_tuple, prob in param_vectors.items():
            idx_tuple = ast.literal_eval(idx_tuple)
            ix, jx = idx_tuple[0], idx_tuple[1]
            if 0 <= ix < Nx and 0 <= jx < Ny:
                prior_2d[ix, jx] = prob
        # fill holes
        if general_settings.get('sample_ratio') < 1:
            prior_2d = gnrl.fill_holes_nd(prior_2d, (Nx, Ny), method="cubic")
        sprob = np.nansum(prior_2d)
        if sprob > 0:
            prior_2d /= sprob

        prior_hm = go.Heatmap(
            x=Vii_vals,
            y=Vij_vals,
            z=prior_2d.T,
            colorscale=fig_lay.get("colorscales", ["Viridis"])[1] if len(fig_lay.get("colorscales", []))>1 else "Plasma",
            # zmin=0, zmax=prior_2d.max(),
            zmin=0, zmax=max_prior_prob,
            hovertemplate=("Vᵢᵢ: %{x:.3f}, Vᵢⱼ: %{y:.3f}<br>Prob: %{z:.5f}<extra></extra>"),
            showscale=False
        )
        fig.add_trace(prior_hm, row=3, col=col_idx)

        if col_idx == 1:
            crazy_prior = list(copy.deepcopy(prior_2d.T))
            for row in crazy_prior:
                print([round(prob, 9) for prob in list(row)])

    # -----------------------------------------------------------------------
    # 4) Stylistic adjustments
    # -----------------------------------------------------------------------
    fig.update_layout(
        template=fig_lay.get("template", "plotly_dark"),
        title="Bayesian Updates for {} vs. {}".format(player_uuid, counterpart_uuid),
        title_x=fig_lay['title_x'], title_y=fig_lay['title_y'], 
        titlefont_size=fig_lay['titlefont_size'] * 0.5, 
        margin=dict(l=120, r=120, t=120, b=60 + 20 * n_rounds),
    ) 

    range_pay = [-4.1, 4.1]
    tickvals_pay = [-4, -2, 0, 2, 4]
    ticktext_pay_y = [str(val) for val in tickvals_pay]
    ticktext_pay_x = [''] + ticktext_pay_y[1:]

    n_ticks = len(Vii_vals)
    bin_size = 2/n_ticks
    half_bin = bin_size/2

    range_par = [-1 - half_bin, 1 + half_bin]
    tickvals_par = [-1.0, -0.5, 0.0, 0.5, 1.0]
    ticktext_par_y = [str(val) for val in tickvals_par]
    ticktext_par_x = [''] + ticktext_par_y[1:]

    scale_count = 0
    scaleanchors = [
        (f'x{fig_idx}', f'y{fig_idx}') for fig_idx in range(1, (n_rounds * 3) + 1)
    ]
    scaleanchors = []
    for cdx in range(1, n_rounds+1):
        for row_idx in (0, 1, 2):
            scaleanchors.append((f"x{cdx + n_rounds * row_idx}", f"x{cdx + n_rounds * row_idx}"))

    # For each column, style axes
    for cdx in range(1, n_rounds+1):
        # Top row => payoff diff space
        fig.update_xaxes(
            title_text="ΔSelf payoff", 
            range=range_pay, tickvals=tickvals_pay, ticktext=ticktext_pay_x,
            zeroline=dark_zero_lines, zerolinewidth=5 if dark_zero_lines else 1,
            scaleanchor=scaleanchors[scale_count][1], scaleratio=1, 
            row=1, col=cdx
        )
        fig.update_yaxes(
            title_text="ΔOther payoff" if cdx == 1 else "", 
            range=range_pay, tickvals=tickvals_pay, ticktext=ticktext_pay_y,
            zeroline=dark_zero_lines, zerolinewidth=5 if dark_zero_lines else 1,
            scaleanchor=scaleanchors[scale_count][0], scaleratio=1, 
            row=1, col=cdx
        )
        scale_count += 1
        # Middle => likelihood param space
        fig.update_xaxes(
            title_text="",
            range=range_par, tickvals=tickvals_par, ticktext=ticktext_par_x,
            scaleanchor=scaleanchors[scale_count][1], scaleratio=1, 
            row=2, col=cdx
        )
        fig.update_yaxes(
            title_text="Vᵢⱼ (altruism)" if cdx == 1 else "",
            range=range_par, tickvals=tickvals_par, ticktext=ticktext_par_y,
            scaleanchor=scaleanchors[scale_count][0], scaleratio=1, 
            row=2, col=cdx
        )
        scale_count += 1
        # Bottom => prior/posterior param space
        fig.update_xaxes(
            title_text="Vᵢᵢ (self-interest)",
            range=range_par, tickvals=tickvals_par, ticktext=ticktext_par_x,
            scaleanchor=scaleanchors[scale_count][1], scaleratio=1, 
            row=3, col=cdx
        )
        fig.update_yaxes(
            title_text="Vᵢⱼ (altruism)" if cdx == 1 else "",
            range=range_par, tickvals=tickvals_par, ticktext=ticktext_par_y,
            scaleanchor=scaleanchors[scale_count][0], scaleratio=1, 
            row=3, col=cdx
        )
        scale_count += 1

    # If you want to save or show:
    outdir = os.path.join(file_paths.get("visuals","."), "bayesian_updates_2d")
    os.makedirs(outdir, exist_ok=True)
    file_name = f"bayes2d_{player_uuid}_{counterpart_uuid}.html"
    if experiment_num == 0:
        for replacement in ('chooser', 'predictor', '_', '='):
            file_name = file_name.replace(replacement, '')
    out_path = os.path.join(outdir, file_name)

    if general_settings.get('export_fig'):
        fig.write_html(out_path)
        print(f"Saved 2D Bayesian updates figure to {out_path}")
    else:
        fig.show()

    return fig


def visualize_bayesian_updates_3d(dyad_games_or_key: int | DyadGames, player_uuid: PlayerUUID, fig_lay: Dict[str, Any], file_paths: FilePaths, 
                                  general_settings: GeneralSettings, fix_z_axis: bool = True):
    """
    Creates a two-panel figure (with a slider to step through games) that shows, 
    for a given player (only when that player is the predictor and did not abdicate), the following:
      • Left Panel: The full interpolated prior grid (with sparse probability sample points overlaid)
      • Right Panel: The likelihood surface computed from the current game’s payoff structure.
    
    The function uses stored meta_data (from grid-based updates) and computes the likelihood
    surface over the same parameter space (Vᵢᵢ and Vᵢⱼ). The observed choice is marked with a red circle.
    
    Games where dyad_game['predictor'] != player_uuid or where the predictor abdicated are skipped.
    
    The new parameter fix_z_axis (default True) allows the user to opt to fix the z-axis and color
    axis in scene 1 so that they range from 0 to the maximum probability.
    """
    update_method = general_settings.get('update_method', 'grid')

    if isinstance(dyad_games_or_key, list):
        dyad_games = dyad_games_or_key
    else:
        dyad_games = prep.get_dyad_data(dyad_key=dyad_games_or_key, file_paths=file_paths, 
                                              experiment_num=general_settings.get('experiment_num', 3), analysis_mode='bayesian', dyad_already_analyzed=False)

    first_game = dyad_games[0]

    if isinstance(player_uuid, int):
        if player_uuid == 1:
            player_uuid = first_game['chooser']
        elif player_uuid == 2:
            player_uuid = first_game['predictor']
            
    first_choo = first_game['chooser']
    first_pred = first_game['predictor']    
    player_number = 1 if player_uuid == first_choo else 2    
    dyad_name = f"{player_number}_{first_choo[:8]}_{first_pred[:8]}"

    "Filter dyad games: only include games where the player is the predictor and did not abdicate."
    if general_settings.get('experiment_num') == 3:
        filtered_games = [game for game in dyad_games 
                        if game.get('predictor', None) == player_uuid and not game.get('abdicated_predictor', False)]
        if not filtered_games:
            raise ValueError("No games found where the given player is the predictor and did not abdicate.")
        
    else:
        #TODO For experiments 1 and 2, I need to extract the choice dot from the observation phase games
        filtered_games = [game for game in dyad_games 
                        if game.get('predictor', None) == player_uuid]

    "Find the first game with grid data for predictor."
    first_grid_game = None
    for game in filtered_games:
        grid_data = game.get("parameter_estimates", {}).get(update_method, {}).get(player_uuid, {}).get("predictor", {})
        if grid_data and grid_data.get("meta_data", None) is not None:
            first_grid_game = game
            break         
    if first_grid_game is None:
        raise ValueError("No grid data found for predictor in any game for player_uuid: " + player_uuid)
    
    # Extract meta_data and tickvals from the first grid game.
    grid_data = first_grid_game["parameter_estimates"][update_method][player_uuid]["predictor"]
    meta_data = grid_data["meta_data"]
    tickvals = meta_data["tickvals"]

    if "Vᵢᵢ" not in tickvals or "Vᵢⱼ" not in tickvals:
        raise ValueError("Tickvals must contain 'Vᵢᵢ' and 'Vᵢⱼ' keys.")
    
    # Build a common parameter grid.
    Vᵢᵢ_vals = np.array([round(val, 9) for val in tickvals["Vᵢᵢ"]])  # assume numpy array
    Vᵢⱼ_vals = np.array([round(val, 9) for val in tickvals["Vᵢⱼ"]])
    Vᵢᵢ_mesh, Vᵢⱼ_mesh = np.meshgrid(Vᵢᵢ_vals, Vᵢⱼ_vals, indexing='ij')
    
    # If requested, loop over all filtered games to compute the global maximum probability
    # after normalization (i.e., from the full grid). This will be used to fix the z-axis and color axis.
    global_max_prob = None
    if fix_z_axis:
        global_max_prob = 0
        for game in filtered_games:
            grid_predictor = game.get("parameter_estimates", {}).get(
                update_method, {}).get(player_uuid, {}).get("predictor", None)
            if grid_predictor is None:
                continue
            prior_dict: dict = grid_predictor.get("param_vectors", {})
            temp_grid = np.full((len(Vᵢᵢ_vals), len(Vᵢⱼ_vals)), np.nan)
            for idx_tuple, prob in prior_dict.items():
                idx_tuple = ast.literal_eval(idx_tuple)
                idx, jdx = idx_tuple[0], idx_tuple[1]
                if 0 <= idx < len(Vᵢᵢ_vals) and 0 <= jdx < len(Vᵢⱼ_vals):
                    temp_grid[idx, jdx] = prob
            temp_grid = gnrl.fill_holes_nd(input_array=temp_grid, output_shape=(len(Vᵢᵢ_vals), len(Vᵢⱼ_vals)), method="cubic")
            sum_temp = np.nansum(temp_grid)
            temp_grid = temp_grid / sum_temp
            candidate = np.nanmax(temp_grid)
            if candidate > global_max_prob:
                global_max_prob = candidate
        # If no valid grid was found (should not happen), set global_max_prob to 1.
        if global_max_prob is None or global_max_prob == 0:
            global_max_prob = 1

    # Create the figure with two 3D subplots.
    fig = make_subplots(rows=1, cols=2,
                        horizontal_spacing=0.1,
                        subplot_titles=["Prior Probability Distribution", "Likelihood Probability Distribution"],
                        specs=[[{'type': 'surface'}, {'type': 'surface'}]])
    
    # Update layout using your settings.
    fig.update_layout(
        template=fig_lay["template"] if fig_lay else None,
        hoverlabel=dict(font_size=14),
        scene=dict(
            xaxis=dict(title="Vᵢᵢ", zeroline=True, zerolinewidth=4, nticks=9),
            yaxis=dict(title="Vᵢⱼ", zeroline=True, zerolinewidth=4, nticks=9),
            zaxis=dict(title="Probability", zeroline=True, nticks=9,
                       range=[0, global_max_prob] if fix_z_axis else None),
            camera=dict(eye=dict(x=-0.0, y=-1e-5, z=2)),
            aspectmode="cube",
        ),
        scene2=dict(
            xaxis=dict(title="Vᵢᵢ", zeroline=True, zerolinewidth=4, nticks=9),
            yaxis=dict(title="Vᵢⱼ", zeroline=True, zerolinewidth=4, nticks=9),
            zaxis=dict(title="Probability", zeroline=True, nticks=9, range=[0, 1]),
            camera=dict(eye=dict(x=-0.0, y=-1e-5, z=2)),
            aspectmode="cube",
        ),
        # Fix the color axis for scene 1 if requested.
        coloraxis=dict(
            colorscale=fig_lay["colorscales"][0] if "colorscales" in fig_lay else "Viridis",
            cmin=0, cmax=global_max_prob if fix_z_axis else None,
            showscale=False
        )
    )
    
    # Accumulate traces for each game (for the slider).
    all_traces = []
    
    # Loop over the filtered games.
    for game_idx, game in enumerate(filtered_games):
        grid_predictor = game.get("parameter_estimates", {}).get(update_method, {}).get(player_uuid, {}).get("predictor", None)
        if grid_predictor is None:
            continue

        # Build the full prior PMF array from the sparse dictionary.
        prior_dict = grid_predictor.get("param_vectors", {})

        # Initialize a dictionary to store the sum of probabilities and counts for averaging.
        aggregated_probs = {}

        # Iterate through the prior_dict to aggregate probabilities based on Vᵢᵢ and Vᵢⱼ dimensions.
        for idx_tuple, prob in prior_dict.items():
            idx_tuple = ast.literal_eval(idx_tuple)
            idx, jdx = idx_tuple[0], idx_tuple[1]  # Get Vᵢᵢ and Vᵢⱼ indices
            if 0 <= idx < len(Vᵢᵢ_vals) and 0 <= jdx < len(Vᵢⱼ_vals):
                if (idx, jdx) not in aggregated_probs:
                    aggregated_probs[(idx, jdx)] = {"prob_sum": 0, "count": 0}
                aggregated_probs[(idx, jdx)]["prob_sum"] += prob
                aggregated_probs[(idx, jdx)]["count"] += 1

        # Create the full_grid with averaged probabilities.
        full_grid = np.full((len(Vᵢᵢ_vals), len(Vᵢⱼ_vals)), np.nan)
        for (idx, jdx), data in aggregated_probs.items():
            full_grid[idx, jdx] = data["prob_sum"] / data["count"]  # Calculate the average probability

        # Handle missing values in the grid and normalize.
        full_grid = gnrl.fill_holes_nd(input_array=full_grid, output_shape=(len(Vᵢᵢ_vals), len(Vᵢⱼ_vals)), method="cubic")
        sum_full_grid = np.nansum(full_grid)
        full_grid = full_grid / sum_full_grid  # Normalize
    
        # Prepare sparse prior sample points.
        filtered_data = {}
        Vᵢᵢ_idx = list(tickvals.keys()).index("Vᵢᵢ")
        Vᵢⱼ_idx = list(tickvals.keys()).index("Vᵢⱼ")

        for param_vector, prob in prior_dict.items():
            param_vector = ast.literal_eval(param_vector)
            val_Vᵢᵢ = tickvals["Vᵢᵢ"][param_vector[Vᵢᵢ_idx]]
            val_Vᵢⱼ = tickvals["Vᵢⱼ"][param_vector[Vᵢⱼ_idx]]
            key = (val_Vᵢᵢ, val_Vᵢⱼ)
            if key not in filtered_data:
                filtered_data[key] = []
            filtered_data[key].append(prob)
        unique_points = np.array(list(filtered_data.keys()))
        probabilities = np.array([np.mean(filtered_data[key]) for key in filtered_data])
        scatter_z = [prob / sum_full_grid for prob in probabilities]  # normalized sparse probabilities
        
        scatter_x, scatter_y = unique_points[:, 0], unique_points[:, 1]
        
        # Compute likelihood surface over the same grid.
        current_params = grid_predictor.get("params", {})
        Vᵢᵢ_mean = round(current_params.get("Vᵢᵢ", 1), 2)
        Vᵢⱼ_mean = round(current_params.get("Vᵢⱼ", 1), 2)
        Vᵢᵢ_std = current_params.get("Vᵢᵢ_std", 1)
        Vᵢⱼ_std = current_params.get("Vᵢⱼ_std", 1)
        likelihood_surface = np.zeros_like(Vᵢᵢ_mesh, dtype=float)
        for idx in range(len(Vᵢᵢ_vals)):
            for jdx in range(len(Vᵢⱼ_vals)):
                agent_params = {
                    "Vᵢᵢ": Vᵢᵢ_mesh[idx, jdx],
                    "Vᵢⱼ": Vᵢⱼ_mesh[idx, jdx],
                    "Vᵢᵢ_std": Vᵢᵢ_std,
                    "Vᵢⱼ_std": Vᵢⱼ_std,
                }
                p_choose_A = choice(current_game=game, agent_params=agent_params, 
                                    softmax_temperature=general_settings.get('softmax_temperature', 1.5),
                                    utility_settings=utility_settings, select=False)["model_choose_A"]
                if game.get("choice", "A") == "A":
                    likelihood_surface[idx, jdx] = p_choose_A
                else:
                    likelihood_surface[idx, jdx] = 1 - p_choose_A
        
        # Determine the observed choice marker.
        hover_text = f"Observed Choice: <b>"
        payoff_As, payoff_Ao = game.get("payoff_A_chooser", 0), game.get("payoff_A_predictor", 0)
        payoff_Bs, payoff_Bo = game.get("payoff_B_chooser", 0), game.get("payoff_B_predictor", 0)        
        if game.get("choice", "A") == "A":
            payoff_diff_self = (payoff_As - payoff_Bs) / 4.0
            payoff_diff_other = (payoff_Ao - payoff_Bo) / 4.0
            hover_text += f"A:({payoff_As}, {payoff_Ao})</b> over B:({payoff_Bs}, {payoff_Bo})<br>"
        else:
            payoff_diff_self = (payoff_Bs - payoff_As) / 4.0
            payoff_diff_other = (payoff_Bo - payoff_Ao) / 4.0   
            hover_text += f"B:({payoff_As}, {payoff_Ao})</b> over A:({payoff_Bs}, {payoff_Bo})<br>"                 
        observed_choice_point = (payoff_diff_self, payoff_diff_other)
        hover_text += f"Payoff Difference Self (chooser) =     {int(payoff_diff_self*4)}<br>"
        hover_text += f"Payoff Difference Other (predictor) = {int(payoff_diff_self*4)}<extra></extra>"
        observed_z = 0.05  # slight elevation
        
        # Build traces for this game.
        game_traces = []
        # Trace 1: Prior surface (left panel)
        trace_prior = go.Surface(
            z=full_grid.T,
            x=Vᵢᵢ_vals,
            y=Vᵢⱼ_vals,
            colorscale=fig_lay["colorscales"][0],
            opacity=0.8,
            showscale=False,
            name=f"Prior (Game {game_idx})",
            hovertemplate=(
                "Vᵢᵢ: %{x:.3f}; " + f"(μ, σ) = ({Vᵢᵢ_mean}, {round(Vᵢᵢ_std, 2)})<br>" +
                "Vᵢⱼ: %{y:.3f}; " + f"(μ, σ) = ({Vᵢⱼ_mean}, {round(Vᵢⱼ_std, 2)})<br>" +
                "Probability: %{z:.3g}<extra></extra>" 
            )                
        )
        game_traces.append(trace_prior)
        
        # Trace 2: Sparse prior points.
        trace_sparse = go.Scatter3d(
            x=scatter_x,
            y=scatter_y,
            z=scatter_z,
            mode="markers",
            marker=dict(
                size=6,
                color=scatter_z,
                colorscale=fig_lay["colorscales"][0],
                opacity=0.9
            ),
            name=f"Prior Probabilities",
            hovertemplate=(
                "Vᵢᵢ: %{x:.3f}; " + f"(μ, σ) = ({Vᵢᵢ_mean}, {round(Vᵢᵢ_std, 2)})<br>" +
                "Vᵢⱼ: %{y:.3f}; " + f"(μ, σ) = ({Vᵢⱼ_mean}, {round(Vᵢⱼ_std, 2)})<br>" +
                "Probability: %{z:.3g}<extra></extra>" 
            )              
        )
        game_traces.append(trace_sparse)
        
        # Trace 3: Likelihood surface (right panel)
        trace_likelihood = go.Surface(
            z=likelihood_surface.T,
            x=Vᵢᵢ_vals,
            y=Vᵢⱼ_vals,
            colorscale=fig_lay["colorscales"][1],
            opacity=0.8,
            showscale=False,
            name=f"Likelihood (Game {game_idx})",
            hovertemplate=(
                "Vᵢᵢ: %{x:.3f}<br>" +
                "Vᵢⱼ: %{y:.3f}<br>" +
                "Probability: %{z:.3g}<extra></extra>"
            )
        )
        game_traces.append(trace_likelihood)
        
        # Trace 4: Observed choice marker.
        trace_choice = go.Scatter3d(
            x=[observed_choice_point[0]],
            y=[observed_choice_point[1]],
            z=[observed_z],
            mode="markers",
            marker=dict(size=8, color="red", symbol="circle"),
            name=f"Counterpart's Choice",
            hovertemplate=hover_text
        )
        game_traces.append(trace_choice)
        
        # Save traces for this game.
        all_traces.append({"traces": game_traces, "grid_predictor": grid_predictor})
    
    if not all_traces:
        raise ValueError("No valid predictor grid data was found in the filtered games.")
    
    # Add all traces to the figure.
    total_traces = 0
    for game_block in all_traces:
        for trace in game_block["traces"]:
            fig.add_trace(trace)
            total_traces += 1
    traces_per_game = len(all_traces[0]["traces"])
    
    # Build slider steps: each step makes only the traces for that game visible.
    slider_steps = []
    for game_idx in range(len(all_traces)):
        visible = [False] * total_traces
        for idx in range(traces_per_game):
            visible[game_idx * traces_per_game + idx] = True
        step = dict(
            label=f"Game {game_idx}",
            method="update",
            args=[{"visible": visible},
                  {"title": f"Bayesian Update Visualization (Game {game_idx})"}]
        )
        slider_steps.append(step)
    
    # Set initial visibility: only traces for the first game.
    init_visible = [False] * total_traces
    for idx in range(traces_per_game):
        init_visible[idx] = True
    for idx, trace in enumerate(fig.data):
        trace.visible = init_visible[idx]
    
    fig.update_layout(
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Game: "},
            pad={"t": 50},
            steps=slider_steps
        )],
        margin=dict(l=100, r=100, t=100, b=100)
    )
    
    # Assign traces to the proper subplot (scene):
    # Traces 0 and 1 go to scene (left panel), traces 2 and 3 to scene2 (right panel).
    trace_idx = 0
    for game_block in all_traces:
        for idx, tr in enumerate(game_block["traces"]):
            if idx < 2:
                fig.data[trace_idx].update(scene="scene")
            else:
                fig.data[trace_idx].update(scene="scene2")
            trace_idx += 1
    
    # Use fig_lay's width and height if provided.
    if fig_lay.get("width") and fig_lay.get("height"):
        fig.update_layout(width=fig_lay["width"], height=fig_lay["height"])

    file_name = f"bayesian_update_visualization_{dyad_name}"
    if isinstance(dyad_games_or_key, int):
        file_name += f"_{dyad_games_or_key}"
    file_name += f"{file_paths.get('file_name_suffix', '')}.html"
    # file_name = f"bayesian_update_visualization_{dyad_name}" + f"{file_paths.get('file_name_suffix', '')}.html"
    # file_name = f"bayesian_update_visualization" + f"{file_paths.get('file_name_suffix', '')}.html"
    visuals_dir = os.path.join(file_paths["visuals"], "bayesian_updates_3d")
    os.makedirs(visuals_dir, exist_ok=True)
    out_path = os.path.join(visuals_dir, file_name)
    if general_settings.get('export_fig'):
        fig.write_html(out_path)
        print(f"Saved {file_name} at {out_path}")
    else:
        fig.show()
    
    return fig


def belief_accuracy_analysis(file_paths: FilePaths, general_settings: GeneralSettings, fig_lay: FigLay, participant_num: int, 
                             fitted_by_player: bool = True, compute_optimum_updates: bool = False, animate_figure: bool = False) -> None:
    """
    Generates a figure that represents participants beliefs about the social 
    preferences of their counterparts in iterated binary dicator games.

    Arguments:
        • file_paths: dict[str, str | dict[str, str]]; Contains file paths for saving the figure.
        • general_settings: dict[str, str | int | float | bool]; Miscelaneous commonly used settings.
        • fig_lay:  dict[str, Any]; Settings to control Plotly figure aesthetics.
        • participant_num: int; Used in place of player UUID for convenience.
        • fitted_by_player: bool; If True, fits by player instead of by dyad.
        • compute_optimum_updates: bool; If True, recomputes optimum updates.
        • animate_figure: bool; If True, animates the figure.

    Returns:    
        • None: Displays or saves the figure.
    """
    avatar_type_to_key = {
        'utilitarian': None,
        'selfish':     None,
        'competitive': None,
        'masochistic': None,
    }

    converge_n_round = {
        'utilitarian': None,
        'selfish':     None,
        'competitive': None,
        'masochistic': None,        
    }

    if fitted_by_player:
        "Parameters are fitted to each participant across multiple dyads."
        plrs_to_dyads = prep.players_to_dyads(experiment_num=2, 
                        file_paths=file_paths, create_new_file=False)

        players_lst = sorted(list(plrs_to_dyads.keys()))
        player_uuid = players_lst[participant_num % len(players_lst)]

        histories = None
        file_name_suffix = file_paths["file_name_suffix"]
        plr_file_path = os.path.join(file_paths["player_fits"], f"experiment_2", 
                                f'{file_name_suffix}_' + player_uuid + ".json")

        if os.path.exists(plr_file_path):
            with open(plr_file_path, "r", encoding='utf-8') as file:
                histories: dict = json.load(file)

        if histories is None:
            raise Exception(f"Failed to extract dyads for player {player_uuid}.")

    else:
        "Parameters are fitted to each dyad and so can differ across dyads for the same participant."
        with open(os.path.join("./", file_paths['file_names'][
            'player_pairs_exper2']), "r", encoding='utf-8') as file:
            raw_data = json.load(file)
        histories: dict = raw_data['histories']
        player_info: dict = raw_data['player_info']
        participants = [player_uuid for player_uuid, info in player_info.items() 
                        if info.get('player_type') == 'participant']
        
        player_uuid = participants[participant_num % len(participants)]

    player_dyad_keys = [dyad_key for dyad_key in histories.keys() if player_uuid in dyad_key]

    for avatar_type in avatar_type_to_key.keys():
        for player_dyad_key in player_dyad_keys:
            if avatar_type in player_dyad_key:
                avatar_type_to_key[avatar_type] = player_dyad_key

    if any(dyad_key is None for dyad_key in avatar_type_to_key.values()):
        raise Exception(f"Failed to extract all four dyads for participant number {participant_num}.")

    "Organize all dyads into a dictionary indexed by counterpart avatar type."
    if fitted_by_player:
        player_dyads = {
            avatar_type: histories[dyad_key] 
            for avatar_type, dyad_key in avatar_type_to_key.items()
        }
    else:
        player_dyads = {
            avatar_type: prep.get_dyad_data(dyad_key=dyad_key, 
            file_paths=file_paths, dyad_already_analyzed=True, experiment_num=2) 
            for avatar_type, dyad_key in avatar_type_to_key.items()
        }

    if compute_optimum_updates:
        "Run discrete model"
        for avatar_type, dyad_games in player_dyads.items():
            temperature = None
            if general_settings.get('temperature_is_param', False):
                first_game = dyad_games[0]
                param_est = first_game.get('parameter_estimates', {}).get(
                    'grid', {}).get(player_uuid, {}).get('predictor', {}).get('params', {})
                temperature = param_est.get('temp', None)
            dyad_games = typo.avatar_posteriors(dyad_games=dyad_games, temperature=temperature, 
                                                 loss_funct_type=general_settings.get('loss_funct_type', 'log'),
                                                 update_method=general_settings.get('update_method', 'grid'))

    player_dyads_data = {}
    for avatar_type, dyad_games in player_dyads.items():
        "Generate data for figure and save in dicts of lists."
        dyad_data = {
            'round': [],
            'phase': [],
            'payoff_A_chooser': [],
            'payoff_A_predictor': [],
            'payoff_B_chooser': [],
            'payoff_B_predictor': [],
            'prediction': [],
            'choice': [],
        }
        param_data_human = {
            'Vᵢᵢ': [],
            'Vᵢⱼ': [],
            'Vᵢᵢ_std': [],
            'Vᵢⱼ_std': [],
        }
        param_data_optimum = {
            'utilitarian': [],
            'selfish':     [],
            'competitive': [],
            'masochistic': [],
        }
        for idx, dyad_game in enumerate(dyad_games):
            param_est = dyad_game.get('parameter_estimates', {}).get(
                'grid', {}).get(player_uuid, {}).get('predictor', {}).get('params', {})
            if dyad_game['phase'] == 'op':
                optimum_update: dict = dyad_game.get('parameter_estimates', {}).get(
                    'grid', {}).get('optimum_update', {})
            else:
                optimum_update = dyad_games[idx-1].get('parameter_estimates', {}).get(
                    'grid', {}).get('optimum_update', {})                 

            avatar_posteriors = optimum_update.get('avatar_posteriors', {})
            if converge_n_round[avatar_type] is None:
                converged = optimum_update.get('converged', False)
                if converged:
                    converge_n_round[avatar_type] = idx

            for data_key in dyad_data.keys():
                dyad_data[data_key].append(dyad_game.get(data_key, None))
            for hum_param_key in param_data_human.keys():
                param_data_human[hum_param_key].append(param_est.get(hum_param_key, None))
            for opt_param_key in param_data_optimum.keys():
                param_data_optimum[opt_param_key].append(avatar_posteriors.get(opt_param_key, None))

        "Merge all data into a single dictionary indexed by counterpart avatar type."
        player_dyads_data[avatar_type] = {**dyad_data, **param_data_human, **param_data_optimum}        

    "Tripple length of masochistic avatar data to make its length equal to the others."
    for key, data_list in player_dyads_data['masochistic'].items():
        player_dyads_data['masochistic'][key] = [val for val in data_list for _ in range(3)]
 
    n_games = len(player_dyads_data['masochistic']['Vᵢᵢ'])

    "Create subplot layout"
    fig = make_subplots(
        rows=2, cols=4,
        specs=[
            [{"rowspan": 2, "colspan": 2}, None, {"type": "bar"}, {"type": "bar"}],
            [None,                         None, {"type": "bar"}, {"type": "bar"}]
        ],
        subplot_titles=(f"Social Preference Belief Updates Over Time For Participant {player_uuid} With Four Avatars", 
                        "Avatar is Utilitarian", "Avatar is Selfish", "Avatar is Competitive", "Avatar is Masochistic")
    )
    avatar_colors = {
        'utilitarian': 'hsla(180, 50%, 50%, 0.9)',
        'selfish':     'hsla(240, 50%, 50%, 0.9)',
        'competitive': 'hsla(300, 50%, 50%, 0.9)',
        'masochistic': 'hsla(360, 50%, 50%, 0.9)', 
    }
    avatar_region_colors = {
        'utilitarian': 'hsla(180, 50%, 50%, 0.4)',
        'selfish':     'hsla(240, 50%, 50%, 0.4)',
        'competitive': 'hsla(300, 50%, 50%, 0.4)',
        'masochistic': 'hsla(360, 50%, 50%, 0.4)', 
    }    
    bar_subplot_row_col = {
        'utilitarian': (1, 3),
        'selfish':     (1, 4),
        'competitive': (2, 3),
        'masochistic': (2, 4),
    }

    slider_steps = []
    type_labels = ['utilitarian', 'selfish', 'competitive', 'masochistic']
    type_labels = [label.capitalize() for label in type_labels]
    for idx in range(n_games):
        n_traces = 0
        "Line plots representing mean parameter belief updates."
        for avatar_type in ['utilitarian','selfish','competitive','masochistic']:

            "Marking regions of where parameters are compatible with avatar choices."
            for line_idx in (0, 1):
                fig.add_trace(
                    go.Scatter(
                        x=(0, typo.regions_of_compatible_utility[avatar_type][line_idx][0]),
                        y=(0, typo.regions_of_compatible_utility[avatar_type][line_idx][1]),
                        name=f"{avatar_type} Region", fill=None if line_idx == 0 else 'tonexty',
                        line=dict(color=avatar_region_colors[avatar_type], width=0),
                        marker=dict(size=12), visible=True if idx == 0 else False, 
                        showlegend=False, mode='lines', 
                    ),
                    row=1, 
                    col=1
                )
                n_traces += 1

            Vᵢᵢ_array = player_dyads_data[avatar_type]['Vᵢᵢ']
            Vᵢⱼ_array = player_dyads_data[avatar_type]['Vᵢⱼ']
            Vᵢᵢ_array_std = player_dyads_data[avatar_type]['Vᵢᵢ_std']
            Vᵢⱼ_array_std = player_dyads_data[avatar_type]['Vᵢⱼ_std']  
            Vᵢᵢ_std_prior = round(Vᵢᵢ_array_std[0], 4)
            Vᵢⱼ_std_prior = round(Vᵢⱼ_array_std[0], 4)
            prior_std_str = f"<br>σ(𝑉𝑖𝑖) = {Vᵢᵢ_std_prior}<br>σ(𝑉𝑖𝑗) = {Vᵢⱼ_std_prior}"
            custom_hover_data = [
                typo.distance_to_perfection(Vᵢᵢ=Vᵢᵢ_array[jdx], Vᵢⱼ=Vᵢⱼ_array[jdx], 
                                             avatar_type=avatar_type, return_percent=True)
                for jdx in range(len(Vᵢᵢ_array))
            ]
            custom_hover_subset = custom_hover_data[:idx+1]
            x_line = Vᵢᵢ_array[:idx+1]
            y_line = Vᵢⱼ_array[:idx+1]
            hover_str = f"Avatar is {avatar_type.capitalize()}:<br><br>"
            hover_str += "Social preference<br>belief accuracy is<br>"

            "Lines show parameter beliefs as they update."
            fig.add_trace(
                go.Scatter(
                    x=x_line, y=y_line,
                    mode='lines+markers',
                    marker=dict(size=12), name="",
                    customdata=custom_hover_subset,
                    line=dict(color=avatar_colors[avatar_type], width=8),
                    visible=(True if idx == 0 else False), showlegend=(True if idx==0 else False),
                    hovertemplate=f"{hover_str}%{{customdata:.1f}}%.<br><br>Vᵢᵢ = %{{x:.2f}}<br>Vᵢⱼ = %{{y:.2f}}"
                ),
                row=1, 
                col=1
            )
            n_traces += 1

            "Marking the current posteriors more clearly.xref"
            first_game = player_dyads[avatar_type][0]
            optimum_update_data: dict = first_game.get('parameter_estimates', {}).get(
                general_settings.get('update_method', 'grid'), {}).get('optimum_update', None)
            if optimum_update_data is not None:
                model_winner: str = optimum_update_data.get('model_winner')
                continious_model_loss = optimum_update_data.get('total_continious_model_loss')
                discrete_model_loss = optimum_update_data.get('total_discrete_model_loss')
                if all(val is not None for val in (model_winner, continious_model_loss, discrete_model_loss)):
                    continious_model_loss = round(continious_model_loss, 3)    
                    discrete_model_loss = round(discrete_model_loss, 3) 
                    hover_text = f"Posterior:<br>Vᵢᵢ = {round(x_line[-1], 2)}<br>"
                    hover_text += f"Vᵢⱼ = {round(y_line[-1], 3)}<br><br>Model Loss:<br>"
                    hover_text += f"Continious: {continious_model_loss}<br>"
                    hover_text += f"Discrete:     {discrete_model_loss}<br>"
                    hover_text += f"<br>Winning Model:<br>{model_winner.capitalize()}!"
                    "Mark the final posteriors."
                    fig.add_trace(
                        go.Scatter(
                            x=[x_line[-1]], y=[y_line[-1]],
                            customdata=custom_hover_subset, name='',
                            marker=dict(size=22, symbol='circle'), mode='markers',
                            line=dict(color=avatar_colors[avatar_type], width=6),
                            visible=False, showlegend=False,
                            hovertemplate=hover_text
                        ),
                        row=1, 
                        col=1
                    )
                    n_traces += 1 

            "Mark the initial prior."
            fig.add_trace(
                go.Scatter(
                    x=[x_line[0]], y=[y_line[0]],
                    customdata=custom_hover_subset, name='',
                    marker=dict(size=28, symbol='cross'), mode='markers',
                    line=dict(color='hsla(120, 50%, 50%, 0.9)', width=6),
                    visible=(True if idx == 0 else False), showlegend=False,
                    hovertemplate=f"Starting Prior:<br>μ(𝑉𝑖𝑖) = %{{x:.4f}}<br>μ(𝑉𝑖𝑗) = %{{y:.4f}}{prior_std_str}"
                ),
                row=1, 
                col=1
            )
            n_traces += 1   

        "4 bar charts indicating p(avatar type) by optimum discrete Bayesian model"
        for avatar_type in ['utilitarian','selfish','competitive','masochistic']:
            opt_util = player_dyads_data[avatar_type]['utilitarian'][idx] 
            opt_self = player_dyads_data[avatar_type]['selfish'][idx]     
            opt_comp = player_dyads_data[avatar_type]['competitive'][idx] 
            opt_maso = player_dyads_data[avatar_type]['masochistic'][idx] 
            y_vals = [opt_util, opt_self, opt_comp, opt_maso]
            converge_text = "Not yet converged"
            n_rounds_converged = converge_n_round.get(avatar_type)
            if n_rounds_converged is not None:
                if idx < n_rounds_converged:
                    converge_text = f"<br><br>Will achieve<br>certainty on<br>game {n_rounds_converged}."
                elif idx > n_rounds_converged:
                    converge_text = f"<br><br>Achieved<br>certainty in<br>{n_rounds_converged} games."
                else:
                    converge_text = f"<br><br>Achieved<br>certainty on<br>game {n_rounds_converged}."

            fig.add_trace(
                go.Bar(
                    x=type_labels,
                    y=y_vals,
                    marker=dict(color=[
                        avatar_colors['utilitarian'],
                        avatar_colors['selfish'],
                        avatar_colors['competitive'],
                        avatar_colors['masochistic'],
                    ]),
                    name=f"Posterior:",
                    visible=True if idx == 0 else False,
                    hovertemplate="𝑝(%{x}) = %{y:.2f}" + converge_text,
                    showlegend=False,
                ),
                row=bar_subplot_row_col[avatar_type][0], 
                col=bar_subplot_row_col[avatar_type][1]
            )
            n_traces += 1

        "Controlling which traces are visible per slider step."
        visible_traces = [False] * n_traces * n_games
        for trace_idx in range(n_traces):
            visible_traces[idx * n_traces + trace_idx] = True  

        if animate_figure:
            step = {
                "label": f"Game: {idx}",
                "method": "animate",
                "args": [
                    [str(idx)],  
                    {
                        "frame": {"duration": 10, "redraw": True},
                        "transition": {"duration": 2, 'easing': 'quadratic-in-out'},
                        "mode": "immediate"
                    }
                ]
            }
        else:
            step = dict(
                label=f"Game: {idx}",
                method="update",
                args=[{"visible": visible_traces}]
            )            
        slider_steps.append(step)

    fig.update_layout(
        sliders=[dict(active=0,
            currentvalue={"prefix": ""},
            steps=slider_steps
        )],
        showlegend=False, hoverlabel=dict(font_size=20), 
        margin=dict(l=80, r=80, t=120, b=100), 
        template=fig_lay['template'], 
    )

    if animate_figure:
        total_traces = len(fig.data)  
        frames = []

        for frame_idx in range(n_games):
            "Build a list of booleans for each trace's visibility."
            visible_flags = [False] * total_traces
            start_index = frame_idx * n_traces
            "Mark traces for the current game as visible."
            for kdx in range(start_index, start_index + n_traces):
                visible_flags[kdx] = True

            "Build a minimal update for each trace that sets 'visible' and preserves 'type'."
            frame_update = []
            for ldx in range(total_traces):
                trace_type = fig.data[ldx].type  # e.g. "bar" or "scatter"
                frame_update.append({"visible": visible_flags[ldx], "type": trace_type})
            
            "Create a frame with name equal to the frame index."
            frames.append(go.Frame(data=frame_update, name=str(frame_idx)))

        "Attach frames to the figure."
        fig.frames = frames

        fig.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 10, "redraw": True},
                                    "transition": {"duration": 2, 'easing': 'quadratic-in-out'},
                                }
                            ]
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "transition": {"duration": 0},
                                    "mode": "immediate",
                                }
                            ]
                        }
                    ],
                    "pad": {"r": 10, "t": 87},
                    "x": -0.02, "y": 1.04,
                    "xanchor": "right",
                    "yanchor": "top"
                }
            ],
            sliders=[{
                "active": 0,
                "currentvalue": {"prefix": "Frame: "},
                "pad": {"t": 50},
                "steps": slider_steps
            }]
        )

    tickvals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    ticktext_y = [str(val) for val in tickvals]
    ticktext_x = [''] + ticktext_y[1:]

    "Line Plot"
    fig.update_xaxes(range=[-1.002, 1.002], row=1, col=1, tickvals=tickvals, ticktext=ticktext_x, scaleanchor='y1', 
                     scaleratio=1, zerolinewidth=4, title=dict(text="Self-Interest (Vᵢᵢ)", font=dict(size=15)))
    fig.update_yaxes(range=[-1.002, 1.002], row=1, col=1, tickvals=tickvals, ticktext=ticktext_y, scaleanchor='x1', 
                     scaleratio=1, zerolinewidth=4, title=dict(text="Altruism (Vᵢⱼ)", font=dict(size=15)))

    "Utilitarian"
    fig.update_xaxes(title="Possible Avatar Types", row=1, col=3)
    fig.update_yaxes(title="𝑝(avatar type)", range=[0,1], row=1, col=3)
    "Selfish"
    fig.update_xaxes(title="Possible Avatar Types", row=1, col=4)
    fig.update_yaxes(title="𝑝(avatar type)", range=[0,1], row=1, col=4)
    "Competitive"
    fig.update_xaxes(title="Possible Avatar Types", row=2, col=3)
    fig.update_yaxes(title="𝑝(avatar type)", range=[0,1], row=2, col=3)
    "Masochistic"
    fig.update_xaxes(title="Possible Avatar Types", row=2, col=4)
    fig.update_yaxes(title="𝑝(avatar type)", range=[0,1], row=2, col=4)

    "Show or export"
    title_str = "Animated" if animate_figure else "Static"
    title = f"Belief Accuracy Analysis {title_str} {player_uuid}"
    file_name = title.replace(' ', '_') + f"{file_paths['file_name_suffix']}.html"
    if general_settings.get('export_fig'):
        fig.write_html(os.path.join(file_paths["visuals"], "belief_accuracy_analyses", file_name))
        print(f"Saved {file_name}")
    else:
        fig.show()

    return fig


"=========================================================================================="
"========= Model Validation: Comparing Bayesian and Alternative Cognitive Models =========="
"=========================================================================================="

def alternative_model_contest(general_settings: Dict[str, Any], param_info: Dict[str, Any], param_bds: Dict[str, Tuple[float, float]], 
                              utility_settings: UtilitySettings, file_paths: Dict[str, str], fig_lay: Dict[str, Any]) -> Dict[str, float]:
    """
    Fits and compares multiple alternative cognitive models (Bayesian and non-Bayesian) 
    against behavioral data, returning their total negative log-likelihood (NLL) losses.

    This function:
        • Loads experiment data (Experiment 2) from disk.
        • Fits discrete Bayesian models with various hypothesis spaces 
          ("good_versus_evil", "social_value_ori", "perfect_oracle").
        • Calculates NLL for each discrete Bayesian model.
        • Calculates NLL for a purely stochastic model (uniform random).
        • Calculates NLL for a "no-memory" Bayesian model (only uses the most recent observation).
        • Calculates NLL for a "no-learning" model (static parameters, no updating).
        • Calculates NLL for a full continuous Bayesian model (grid-updating).
        • Summarizes all losses in a dictionary.

    Arguments:
        • general_settings: Dict[str, Any]
            High-level settings controlling analysis details 
            (e.g., 'update_method', 'experiment_num', 'loss_funct_type').
        • param_info: Dict[str, Any]
            Contains parameter keys, bounds, and initial guesses for model fitting.
        • utility_settings: UtilitySettings
            Dictionary or structured object specifying which components of the 
            utility function to include (e.g., social preferences, risk preferences).
        • file_paths: Dict[str, str]
            Dictionary with paths to directories or files necessary for loading 
            histories, saving results, etc. Must contain keys like:
               └── "processed": Path to processed data
               └── "file_names": Nested dict with file names keyed by 
                                "player_pairs_exper{experiment_number}", etc.

    Returns:
        Dict[str, float]:
            A dictionary mapping model names to their total negative log-likelihood:
               {
                   'utility_bayesian':  <NLL>,
                   'stochastic_model':  <NLL>,
                   'no_learning_model': <NLL>,
                   'no_memory_model':   <NLL>,
                   'good_versus_evil':  <NLL>,
                   'social_value_ori':  <NLL>,
                   'perfect_oracle':  <NLL>,
               }
    """
    # -----------------------------
    # 1) Copy and modify high-level settings for Experiment 2 analysis
    # -----------------------------
    model_names = {
        "utility_bayesian":  "Utility Bayes.",
        "stochastic_model":  "Stochastic",
        "no_learning_model": "No Learning",
        "no_memory_model":   "No Memory",
        "good_versus_evil":  "Good vs Evil",
        "social_value_ori":  "Canonical SVO",
        "perfect_oracle":    "Perfect Oracle"
    }

    experiment_num = 2
    create_new_file = True
    loss_funct_type = "log"

    general_settings_ = copy.deepcopy(general_settings)
    general_settings_["experiment_num"] = experiment_num
    general_settings_["create_new_file"] = create_new_file
    general_settings_["loss_funct_type"] = loss_funct_type

    model_losses = None
    output_file = "model_contest_losses.json"
    output_path = os.path.join(file_paths["processed"], output_file)
    if not create_new_file and os.path.exists(output_path):
        with open(output_path, "r", encoding='utf-8') as file:
            model_losses = json.load(file)        
            print(model_losses)
    if not isinstance(model_losses, dict):

        # -----------------------------
        # 2) Identify player UUIDs for Experiment 2
        # -----------------------------
        player_uuids = prep.all_player_uuids(
            file_paths=file_paths,
            experiment_num=experiment_num,
            only_humans=True
        )

        # -----------------------------
        # 3) Load processed data (histories) for Experiment 2
        # -----------------------------
        full_path_histories = os.path.join(
            file_paths["processed"],
            file_paths["file_names"][f"player_pairs_exper{experiment_num}"]
        )
        if not os.path.exists(full_path_histories):
            raise FileNotFoundError(
                f"Cannot find player pairs file for experiment {experiment_num} at {full_path_histories}"
            )

        with open(full_path_histories, "r", encoding='utf-8') as file:
            histories_and_info = json.load(file)

        # 'histories_exper2' is a dict: { <dyad_id>: [game_1, game_2, ...], ... }
        histories_exper2: Dict[str, Any] = histories_and_info["histories"]

        # -----------------------------
        # 4) Initialize a results dict to track total negative log-likelihoods for all models
        # -----------------------------
        model_losses = {
            "utility_bayesian":  0.0,
            "stochastic_model":  0.0,
            "no_learning_model": 0.0,
            "no_memory_model":   0.0,
            "good_versus_evil":  0.0,
            "social_value_ori":  0.0,
            "perfect_oracle":    0.0,
        }

        # -----------------------------
        # 5) Define hypothesis spaces for discrete Bayesian models
        #    Each dict key is a model name, each value is a dict of {parameter_tuple: prior_weight}
        # -----------------------------
        hypothesis_spaces = {
            "good_versus_evil": {
                ( 1.0,  1.0): 0.5,
                ( 1.0, -1.0): 0.5
            },
            "social_value_ori": {
                ( 1.0,  1.0): 0.125,
                ( 1.0,  0.0): 0.125,
                ( 1.0, -1.0): 0.125,
                ( 0.0, -1.0): 0.125,
                (-1.0, -1.0): 0.125,
                (-1.0,  0.0): 0.125,
                (-1.0,  1.0): 0.125,
                ( 0.0,  1.0): 0.125,
            },
            "perfect_oracle": {
                ( 1.0,  1.0): 0.3,
                ( 1.0,  0.0): 0.3,
                ( 1.0, -1.0): 0.3,
                (-1.0,  0.0): 0.1,
            },
        }

        # ------------------------------------------------
        # 6) Calculate loss for each discrete Bayesian model
        #    (Uses typo.discrete_bayesian_model(...))
        # ------------------------------------------------
        general_settings_["update_method"] = "discrete"

        for hspace_name, hypothesis_space in hypothesis_spaces.items():
            for dyad_id, dyad_games in histories_exper2.items():
                these_dyad_games = copy.deepcopy(dyad_games)
                human_player_uuid = these_dyad_games[0]["predictor"]  # assumed consistent across the dyad
                updated_dyad = typo.discrete_bayesian_model(
                    dyad_games=these_dyad_games,
                    choice_funct=choice,
                    player_uuid=human_player_uuid,
                    general_settings=general_settings_,
                    hypothesis_space=hypothesis_space
                )
                loss_dict = create_loss_report(
                    dyad_games=updated_dyad,
                    general_settings=general_settings_
                ).get(human_player_uuid, {}).get("predictor", {})
                model_losses[hspace_name] += loss_dict.get("raw_neglogprob_sum", 0.0)

        pp.pprint(model_losses)

        # ------------------------------------------------
        # 7) Calculate loss for the purely stochastic (random) model
        # ------------------------------------------------
        n_iter_stochastic = 1000
        stochastic_losses = []
        "Iterate many times and compute the average loss."
        for n_iter in range(n_iter_stochastic):
            stochastic_loss = 0.0
            for dyad_games in histories_exper2.values():
                for game in dyad_games:
                    if game.get("phase") == "rp":
                        predicted_choice = game.get("prediction")
                        actual_obs = 1 if predicted_choice == "A" else 0

                        # Randomly guess P(A). Then compute probability of the observed event
                        model_pred_A = random.random()
                        prob_of_observed = model_pred_A if actual_obs == 1 else (1 - model_pred_A)

                        # Avoid log(0)
                        if prob_of_observed <= 0:
                            prob_of_observed = 1e-6

                        raw_neglogprob = -math.log(prob_of_observed)
                        stochastic_loss += raw_neglogprob

            stochastic_losses.append(stochastic_loss)    

        model_losses["stochastic_model"] = sum(stochastic_losses) / n_iter_stochastic       
        pp.pprint(model_losses)
        # ------------------------------------------------
        # 8) Calculate loss for "no memory" Bayesian model
        #    (always resets posterior after each new observation)
        # ------------------------------------------------
        general_settings_["update_method"] = "grid"
        general_settings_["no_memory_mode"] = True

        for dyad_games in histories_exper2.values():
            human_player_uuid = dyad_games[0]["predictor"]

            # Construct an initial parameter dictionary
            if callable(param_info["guesses"]):
                initial_guesses = param_info["guesses"]()
            else:
                initial_guesses = param_info["guesses"]

            initial_params = {
                "predictor": {
                    key: guess for key, guess in zip(param_info["keys"], initial_guesses)
                }
            }

            updated_games = agent(
                dyad_games=dyad_games,
                game_idx_start=0,
                game_idx_stop=None,
                general_settings=general_settings_,
                initial_params=initial_params,
                param_info=param_info,
                utility_settings=utility_settings,
                player_uuid=human_player_uuid,
                player_role="predictor",
                choice_temperature=general_settings_.get("softmax_temperature")
            )
            updated_games = loss_function_bayes(dyad_games=updated_games, general_settings=general_settings_)

            loss_dict = create_loss_report(
                dyad_games=updated_games,
                general_settings=general_settings_
            ).get(human_player_uuid, {}).get("predictor", {})
            model_losses["no_memory_model"] += loss_dict.get("raw_neglogprob_sum", 0.0)
        del general_settings_["no_memory_mode"]
        pp.pprint(model_losses)

        # ------------------------------------------------
        # 9) Calculate loss for "no learning" model
        #    (static parameters, no posterior updates)
        # ------------------------------------------------
        # general_settings_["update_method"] = "learning not"
        general_settings_["update_method"] = "naive"

        # Remove or add suffix to file paths to track results
        file_name_suffix = prep.create_file_name_suffix(
            general_settings=general_settings_,
            utility_settings=utility_settings
        )
        file_paths_naive = copy.deepcopy(file_paths)
        file_paths_naive = prep.add_remove_file_name_suffix(
            file_paths=file_paths_naive,
            file_name_suffix=file_name_suffix,
            add_suffix=False
        )
        file_paths_naive = prep.add_remove_file_name_suffix(
            file_paths=file_paths_naive,
            file_name_suffix=file_name_suffix,
            add_suffix=True
        )

        general_settings_["update_method"] = "grid"

        param_info_ = make_param_info(param_bds=param_bds, utility_settings=utility_settings, general_settings=general_settings_, 
                                                random_guesses_are_unique=not general_settings_.get('run_in_parallel', True))

        for pdx, param_key in enumerate(param_info_['keys']):
            if '_std' in param_key:
                # param_info_['bounds'][pdx] = (0.01, 0.01)
                param_info_['bounds'][pdx] = (1e-6, 2e-6)
        param_info_['guesses'] = [
            random.uniform(bound[0], bound[1]) for bound in param_info_['bounds']
        ]

        run_analysis_bayes(
            utility_settings=utility_settings,
            general_settings=general_settings_,
            histories_data=histories_and_info,
            file_paths=file_paths_naive,
            param_info=param_info_,
            print_=False
        )

        # Accumulate NLL across all players/dyads
        for player_uuid in player_uuids:
            player_dyads = prep.fitted_dyads_for_a_player(
                player_uuid=player_uuid,
                experiment_num=experiment_num,
                file_paths=file_paths_naive
            )
            if not player_dyads:
                raise ValueError(f"Failed to extract data for player {player_uuid}")

            for dyad_key, games in player_dyads.items():
                loss_dict = create_loss_report(
                    dyad_games=games,
                    general_settings=general_settings_
                ).get(player_uuid, {}).get("predictor", {})
                model_losses["no_learning_model"] += loss_dict.get("raw_neglogprob_sum", 0.0)
        pp.pprint(model_losses)

        # ------------------------------------------------
        # 10) Calculate loss for full (continuous) Bayesian model
        # ------------------------------------------------
        general_settings_["update_method"] = "grid"
        # Re-generate file name suffix for these settings
        file_name_suffix_full = prep.create_file_name_suffix(
            general_settings=general_settings_,
            utility_settings=utility_settings
        )

        file_paths_full = copy.deepcopy(file_paths)
        file_paths_full = prep.add_remove_file_name_suffix(
            file_paths=file_paths_full,
            file_name_suffix=file_name_suffix_full,
            add_suffix=False
        )
        file_paths_full = prep.add_remove_file_name_suffix(
            file_paths=file_paths_full,
            file_name_suffix=file_name_suffix_full,
            add_suffix=True
        )
        print(param_info, utility_settings)
        # Run continuous Bayesian analysis
        run_analysis_bayes(
            utility_settings=utility_settings,
            general_settings=general_settings_,
            histories_data=histories_and_info,
            file_paths=file_paths_full,
            param_info=param_info,
            print_=True
        )

        # Summation of NLL across players for continuous model
        for player_uuid in player_uuids:
            player_dyads = prep.fitted_dyads_for_a_player(
                player_uuid=player_uuid,
                experiment_num=experiment_num,
                file_paths=file_paths_full
            )
            if not player_dyads:
                raise ValueError(f"Failed to extract data for player {player_uuid}")

            for dyad_key, games in player_dyads.items():
                loss_dict = create_loss_report(
                    dyad_games=games,
                    general_settings=general_settings_
                ).get(player_uuid, {}).get("predictor", {})
                model_losses["utility_bayesian"] += loss_dict.get("raw_neglogprob_sum", 0.0)
                # print(dyad_key, model_losses["utility_bayesian"])
        pp.pprint(model_losses)

    # Sort the models by ascending loss for easier comparison
    sorted_model_losses = dict(sorted(model_losses.items(), key=lambda x: x[1]))

    # Extract names and losses
    model_names = [model_names[model] for model in sorted_model_losses.keys()]
    colors = [f'hsla({int(115 + 360/(len(model_names)+4) * idx) % 360}, 80%, 40%, 1.0)' for idx in range(len(model_names))]
    loss_values = list(sorted_model_losses.values())
    exit()
    # Create the figure
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=model_names,
        y=loss_values,
        marker_color=colors,
        hovertemplate="Model: %{x}<br>Loss: %{y:.3f}<extra></extra>",
    ))

    # Set title and axis labels
    fig.update_layout(
        title="Model Comparison by Negative Log-Likelihood Loss",
        template=fig_lay.get("template", "plotly_dark"),
        font=fig_lay["font"],
        hoverlabel=fig_lay["hoverlabel"],
        margin=dict(l=120, r=120, t=120, b=120),
        titlefont_size=fig_lay['titlefont_size'],
        title_x=fig_lay['title_x'], 
        title_y=fig_lay['title_y'],
        xaxis=dict(
            title="Models",
            tickfont=fig_lay["xaxis"]["tickfont"],
            title_font=fig_lay["xaxis"]["title_font"]
        ),
        yaxis=dict(
            title="Total Negative Log-Likelihood (Lower is Better)",
            tickfont=fig_lay["yaxis"]["tickfont"],
            title_font=fig_lay["yaxis"]["title_font"]
        )
    )

    # Save the figure to the specified visuals path
    visuals_path = file_paths["visuals"]
    os.makedirs(visuals_path, exist_ok=True)
    output_html_path = os.path.join(visuals_path, "model_losses_bar_chart.html")
    fig.write_html(output_html_path)

    print("Saved model losses bar chart to", output_html_path)

    # ------------------------------------------------
    # 11) Save and return results
    # ------------------------------------------------

    with open(output_path, "w", encoding='utf-8') as file:
        json.dump(model_losses, file, ensure_ascii=False, indent=4)

    pp.pprint(model_losses)
    return model_losses


"=========================================================================================="
"=================== Searching The Space of Typological Bayesian Models ==================="
"=========================================================================================="

def compute_loss_for_typological_model_across_all_data(hypothesis_space: Dict[Tuple[float, float], float], general_settings: Dict[str, Any], file_paths: Dict[str, str]) -> Tuple[float, int]:
    """
    Replays all relevant data (players/dyads) using the discrete_bayesian_model 
    with the given prior distribution, sums the negative log-likelihood, 
    and returns (NLL, N_data).

    Arguments:
        • hypothesis_space: Dict[Tuple[float, float], float]
            Maps each (Vᵢᵢ, Vᵢⱼ) profile to its prior probability.
        • general_settings: Dict[str, Any]
            Various settings (experiment_num, etc.). 
        • file_paths: Dict[str, str]
            Contains paths to load data from (like 'processed' data) if needed.

    Returns:
        • Tuple[float, int]
            (total negative log-likelihood, total number of data points used).
    """
    # This part depends on how your data is structured. You might replicate
    # the logic from your alternative_model_contest(...) or similar code
    # that sums up raw_neglogprob across all players/dyads.

    # For example:
    # 1) Load or retrieve your "histories_exper2" from disk (like you do in alternative_model_contest).
    # 2) For each dyad in experiment 2, run discrete_bayesian_model(dyad, choice_funct=..., hypothesis_space=...).
    # 3) Summate raw_neglogprob from create_loss_report(...).
    # 4) Count total "rp" data points.


    experiment_num = general_settings.get('experiment_num', 2)
    full_path_histories = os.path.join(
        file_paths["processed"],
        file_paths["file_names"][f'player_pairs_exper{experiment_num}']
    )

    with open(full_path_histories, "r", encoding="utf-8") as f:
        histories_info = json.load(f)
    histories_exper2 = histories_info['histories']

    total_nll = 0.0
    total_data_points = 0

    for dyad_key, dyad_games in histories_exper2.items():
        # run model
        # Be sure to re-copy the hypothesis_space so we don't mutate it
        human_player_uuid = dyad_games[0]['predictor']
        local_space = copy.deepcopy(hypothesis_space)
        updated_dyad = typo.discrete_bayesian_model(
            dyad_games=copy.deepcopy(dyad_games),
            # dyad_games=dyad_games,
            choice_funct=choice,
            player_uuid=human_player_uuid,  # or however you parse it
            general_settings=general_settings,
            hypothesis_space=local_space,
            update_method='discrete'
        )
        # compute the per-dyad NLL
        loss_report = create_loss_report(updated_dyad, general_settings).get(human_player_uuid, {}).get("predictor", {})
        total_nll += loss_report.get('raw_neglogprob_sum', 0.0)
        total_data_points += loss_report.get('n_data', 0)

    return total_nll, total_data_points


def _parallel_process_worker_typological_model_comparison_population_fit(args: Tuple[Any, ...]) -> Dict[str, Any]:
    """
    Worker function for parallelizing a single hypothesis-space fit. 
    This obtains the best-fitting priors for that subset, 
    then returns a dictionary summarizing the results.

    Arguments:
        • args: Tuple containing:
            - subset_id: int
            - subset_profiles: List[Tuple[float,float]]
            - general_settings: Dict[str,Any]
            - param_info: Dict[str,Any]
            - utility_settings: Dict[str,bool]
            - file_paths: Dict[str,str]
            - prior_init_method: str
            - penalty_weight: float

    Returns:
        • Dict[str, Any]
            A row of data including 'subset_id', 'profiles', 'best_priors', 'best_nll', 'AIC', 'BIC', etc.
    """
    (subset_id,
     subset_profiles,
     general_settings,
     file_paths,
     prior_init_method,
     penalty_weight) = args

    k_params = len(subset_profiles)

    time_start = time.time()

    # --- 1) Prepare the objective function
    def objective_fn(alpha: np.ndarray) -> float:
        # convert alpha to priors
        priors = gnrl.transform_to_simplex(alpha)  # shape (k,)

        # Build the dictionary for the discrete model
        hypothesis_space = {
            prof: float(pr) for prof, pr in zip(subset_profiles, priors)
        }

        # Evaluate negative log-likelihood across data
        # plus penalty to help gradient
        nll, _ = compute_loss_for_typological_model_across_all_data(
            hypothesis_space=hypothesis_space,
            general_settings=general_settings,
            file_paths=file_paths
        )
        # print(hypothesis_space, nll)
        # Add a penalty if desired
        # e.g. penalty = penalty_weight * sum( p^2 )
        sum_sq = sum(priors * priors)
        penalty = penalty_weight * sum_sq

        return nll + penalty

    # --- 2) Dimensions of alpha = k. We do an unconstrained reparameterization
    # bounds for alpha, let's do [-3,3] for each dimension
    x_bounds = [(0.0, 1.0)] * k_params
    if prior_init_method == "uniform":
        x_guesses = [1/len(x_bounds) for bound in x_bounds]
    else:
        x_guesses = [random.uniform(param_bds[bound][0], param_bds[bound][1]) for bound in x_bounds]

    # --- 3) run global+local optimization
    best_alpha = global_local_optimization(
        objective_fn=objective_fn,
        x_bounds=x_bounds,
        x_guesses=x_guesses,
        maxiter_global=None,   # can tune
        maxiter_local=None,   # can tune
        maxfun_global=100,
        maxfun_local=None
    )

    # Recompute the actual no-penalty NLL with best_alpha
    best_priors = gnrl.transform_to_simplex(best_alpha['final']['x'])
    hypothesis_space_final = {
        prof: float(pr) for prof, pr in zip(subset_profiles, best_priors)
    }
    nll_no_penalty, n_data = compute_loss_for_typological_model_across_all_data(
        hypothesis_space=hypothesis_space_final,
        general_settings=general_settings,
        file_paths=file_paths
    )
    # If you want, also compute the penalty for best_alpha:
    sum_sq_final = sum(best_priors * best_priors)
    penalty_final = penalty_weight * sum_sq_final
    total_loss = nll_no_penalty + penalty_final

    # compute IC
    # k_params = dimension of priors minus 1, effectively, 
    # but if you're counting them all, it's k. 
    # It's arguable whether the sum-to-1 constraint reduces one degree of freedom. 
    # We'll do k-1 if you like:
    # For now, let's do k_params = k-1
    k_params = max(k_params, 1) 
    ic_dict = compute_ic(k_params, n_data, nll_no_penalty)
    aic_val = ic_dict["AIC"]
    bic_val = ic_dict["BIC"]

    time_stop = time.time()
    duration = time_stop - time_start

    result_dict = {
        "n_data": n_data,
        "duration": duration,
        "subset_id": subset_id,
        "k_params": k_params,
        "profiles": subset_profiles,  # or str(subset_profiles)
        "nll_no_penalty": nll_no_penalty,
        "penalty": penalty_final,
        "total_loss": total_loss,
        "AIC": aic_val, "BIC": bic_val,
        "best_alpha": list(best_alpha['final']['x']),
        "best_priors_normalized": list(best_priors),
    }
    # print(result_dict)
    return result_dict


def typological_model_comparison_fit_population(file_paths: Dict[str, str], general_settings: Dict[str, Any], k_min: int, k_max: int, n_subsets_per_k: int, 
                                              intervals_per_dim: int, prior_init_method: str, penalty_weight: float, save_after_n_iter: int, max_combinations: int = 5000000) -> None:
    """
    Explores and fits many discrete Bayesian models by:
        • Generating a 2D grid of social-preference values in [-1, 1].
        • Randomly sampling subsets of size k in [k_min, k_max].
        • For each subset, optimizing the prior distribution over profiles via global+local search.
        • Storing the best-fitting priors and the corresponding negative log-likelihood + info criteria.

    Arguments:
        • file_paths: Dict[str, str]
            Must include:
            └─ "discrete": Path to a directory where intermediate results are saved.
            └─ "processed" & "file_names" for reading the experiment data.
        • general_settings: Dict[str, Any]
            Contains standard keys (e.g., "experiment_num", "run_in_parallel", etc.).
        • param_info: Dict[str, Any]
            Contains parameter specifications for the overall modeling context.
        • utility_settings: Dict[str, bool]
            Toggles for the utility function used when replaying data.
        • k_min: int
            Minimum number of profiles to use in a hypothesis space.
        • k_max: int
            Maximum number of profiles to use in a hypothesis space (e.g., 4).
        • n_subsets_per_k: int
            How many random subsets of each size k to generate.
        • intervals_per_dim: int
            Grid resolution for each dimension in [-1,1], e.g. 9 => [-1.0, -0.75, ..., 1.0].
        • prior_init_method: str
            "uniform" or "random" for how to initialize alpha in local stage 
            (affects global_local_optimization).
        • penalty_weight: float
            Strength of the sum-of-squares penalty on priors to improve optimization stability.
        • save_after_n_iter: int
            Interval for saving the DataFrame to disk. Must be <= n_subsets_per_k.

    Returns:
        • None; saves output files in file_paths["discrete"] as they are computed.

    Notes:
        • The final DataFrame is saved/overwritten every `save_after_n_iter` subsets.
        • At the end, you will have multiple CSV files, one for each k in [k_min, k_max].
        • Later you can parse these CSVs to find the best discrete model overall.
    """
    # Validate folder
    output_dir = file_paths.get("discrete", None)
    if not output_dir:
        raise ValueError("'discrete' path not found in file_paths. Please specify file_paths['discrete'].")
    os.makedirs(output_dir, exist_ok=True)

    # 1) Build the 2D grid of possible profiles
    step = 2.0 / (intervals_per_dim - 1)  # e.g. 2.0 / 8 = 0.25 if intervals_per_dim=9
    possible_values = [round(-1.0 + i*step, 4) for i in range(intervals_per_dim)]
    all_profiles = [(vx, vy) for vx in possible_values for vy in possible_values if not (vx == 0 and vy == 0)]

    general_settings_ = copy.deepcopy(general_settings)
    general_settings_['update_method'] = 'discrete'    

    for k_params in range(k_min, k_max+1):
        # Store results in a Pandas DataFrame
        columns = [
            "n_data", "duration", "subset_id", "k_params", 
            "best_alpha", "nll_no_penalty", "penalty", "total_loss",
            "AIC", "BIC", "profiles", "best_priors_normalized",
        ]
        df_results = pd.DataFrame(columns=columns)

        # Generate random subsets of size k
        n_combinations = math.comb(len(all_profiles), k_params)
        if n_combinations > max_combinations:
            print(f"Halting operation: {n_combinations} combinations > maximum n combinations = {max_combinations}.")
            break

        print(f"Generating {n_combinations} combinations")
        random_subsets = list(it.combinations(all_profiles, k_params))
        random.shuffle(random_subsets)

        args_list = []
        for subset_idx, random_subset in enumerate(random_subsets):
            if subset_idx > n_subsets_per_k:
                break
            args_list.append((
                subset_idx,
                random_subset,
                general_settings_,
                file_paths,
                prior_init_method,
                penalty_weight
            ))            

        minimum_loss = 1e18
        time_start = time.time()
        # Now run them in parallel or serial
        if general_settings.get('run_in_parallel'):
            with mp.Pool(processes=max(mp.cpu_count()-1, 1)) as pool:
                for idx, result_dict in enumerate(pool.imap_unordered(_parallel_process_worker_typological_model_comparison_population_fit, args_list), 1):
                    # Append to df_results
                    df_results.loc[len(df_results)] = [
                        result_dict["n_data"],
                        result_dict["duration"],
                        result_dict["subset_id"],
                        result_dict["k_params"],
                        result_dict["best_alpha"],
                        result_dict["nll_no_penalty"],
                        result_dict["penalty"],
                        result_dict["total_loss"],
                        result_dict["AIC"], 
                        result_dict["BIC"],
                        result_dict["profiles"],
                        result_dict["best_priors_normalized"],
                    ]

                    total_loss = result_dict["total_loss"]
                    if total_loss < minimum_loss:
                        minimum_loss = total_loss

                    if idx % save_after_n_iter == 0:

                        "Compute remaining time"
                        time_now = time.time()
                        total_time = time_now - time_start
                        average_duration = total_time / (idx + 1)
                        # average_duration = df_results["duration"].mean()
                        n_remaining_iters = len(args_list) - idx
                        remaining_seconds = average_duration * n_remaining_iters / max(mp.cpu_count()-1, 1)
                        remaining_minutes = remaining_seconds / 60
                        remaining_hours = int(remaining_minutes / 60)
                        remaining_minutes = int(remaining_minutes % 60)
                        current_time = dt.datetime.now().strftime("%H%M")

                        # Save to disk
                        csv_path = os.path.join(output_dir, f"discrete_fits_k{k_params}.csv")
                        df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
                        print_str = f"[k={k_params}] Processed {idx}/{len(args_list)} subsets. "
                        print_str += f"Time: {current_time}; Remaining Time: {remaining_hours:02d} hours "
                        print_str += f"and {remaining_minutes:02d} minutes. Min Loss = {minimum_loss}"
                        print(print_str)
        else:
            # Serial
            for idx, single_args in enumerate(args_list, 1):
                result_dict = _parallel_process_worker_typological_model_comparison_population_fit(single_args)
                df_results.loc[len(df_results)] = [
                    result_dict["n_data"],
                    result_dict["duration"],
                    result_dict["subset_id"],
                    result_dict["k_params"],             
                    result_dict["best_alpha"],
                    result_dict["nll_no_penalty"],
                    result_dict["penalty"],
                    result_dict["total_loss"],
                    result_dict["AIC"],
                    result_dict["BIC"],
                    result_dict["profiles"],
                    result_dict["best_priors_normalized"],
                ]

                total_loss = result_dict["total_loss"]
                if total_loss < minimum_loss:
                    minimum_loss = total_loss

                if idx % save_after_n_iter == 0:

                    "Compute remaining time"
                    time_now = time.time()
                    total_time = time_now - time_start
                    average_duration = total_time / (idx + 1)                    
                    # average_duration = df_results["duration"].mean()
                    n_remaining_iters = len(args_list) - idx
                    remaining_seconds = average_duration * n_remaining_iters 
                    remaining_minutes = remaining_seconds / 60
                    remaining_hours = int(remaining_minutes / 60)
                    remaining_minutes = int(remaining_minutes % 60)
                    current_time = dt.datetime.now().strftime("%H%M")

                    csv_path = os.path.join(output_dir, f"discrete_fits_k{k_params}.csv")
                    df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
                    print_str = f"[k={k_params}] Processed {idx}/{len(args_list)} subsets. "
                    print_str += f"Time: {current_time}; Remaining Time: {remaining_hours:02d} hours "
                    print_str += f"and {remaining_minutes:02d} minutes. Min Loss = {minimum_loss}"
                    print(print_str)

        # Final save after all subsets
        csv_path = os.path.join(output_dir, f"discrete_fits_k{k_params}.csv")
        df_results = df_results.sort_values(by='total_loss', ascending=True)
        df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Finished k={k_params}. All {n_subsets_per_k} subsets processed. Results saved to {csv_path}.")
        print(df_results)


def typological_model_nll_for_player(hypothesis_space: Dict[tuple[float, float], float], general_settings: Dict[str, Any], file_paths: Dict[str, str], player_uuid: str) -> Tuple[float, int]:
    """
    Replays data from a single player's dyads using discrete_bayesian_model
    with the given prior distribution, returns (NLL, number_of_data_points).

    Arguments:
        • hypothesis_space: Dict[(float, float), float]
            The discrete profiles => prior probs mapping.
        • general_settings: Dict[str, Any]
            Possibly includes 'experiment_num', etc.
        • file_paths: Dict[str, str]
            Paths for loading the data (like 'processed' + 'file_names').
        • player_uuid: str
            Which player's data to load and evaluate.

    Returns:
        (total_nll, total_data_points)
    """
    experiment_num = general_settings.get("experiment_num", 2)
    # Load the histories (like your alt_model_contest approach)
    full_path_histories = os.path.join(
        file_paths["processed"],
        file_paths["file_names"][f"player_pairs_exper{experiment_num}"]
    )
    with open(full_path_histories, "r", encoding="utf-8") as file:
        histories_info = json.load(file)
    all_histories = histories_info["histories"]  # { dyad_key: [games], ...}

    total_nll = 0.0
    total_data = 0

    # We only want dyads where this player is the predictor
    for dyad_key, dyad_games in all_histories.items():
        if dyad_games and (dyad_games[0].get("predictor") == player_uuid):
            # run model
            local_space = copy.deepcopy(hypothesis_space)
            these_dyad_games = copy.deepcopy(dyad_games)
            updated_dyad = typo.discrete_bayesian_model(
                dyad_games=these_dyad_games,
                choice_funct=choice, 
                player_uuid=player_uuid,
                general_settings=general_settings,
                hypothesis_space=local_space,
                update_method="discrete"
            )
            # gather loss
            general_settings_ = copy.deepcopy(general_settings)
            general_settings_["update_method"] = "discrete"
            loss_report = create_loss_report(updated_dyad, general_settings_)
            predictor_loss = (loss_report.get(player_uuid, {})
                                         .get("predictor", {}))
            total_nll += predictor_loss.get("raw_neglogprob_sum", 0.0)
            total_data += predictor_loss.get("n_data", 0)

    if total_data <= 0:
        raise Exception("No data found!")

    return total_nll, total_data


def _parallel_process_worker_typological_model_comparison_individual_fit(args: Tuple[Any, ...]) -> Dict[str, Any]:
    """
    Parallel worker: given one player's UUID and a single, fixed set of profiles,
    find the best alpha => best prior => minimal NLL.
    """
    (player_uuid,
     best_profiles,
     general_settings,
     file_paths,
     penalty_weight,
     maxiter_global,
     maxiter_local,
     optimization_method) = args

    k_params = len(best_profiles)

    def objective_fn(alpha: np.ndarray) -> float:
        priors = gnrl.transform_to_simplex(alpha)  # e.g. exp / sum(exp)
        hypothesis_space = {prof: float(pr) for prof, pr in zip(best_profiles, priors)}

        nll, _ = typological_model_nll_for_player(
            hypothesis_space=hypothesis_space,
            general_settings=general_settings,
            file_paths=file_paths,
            player_uuid=player_uuid
        )
        sum_sq = sum(priors * priors)
        penalty = penalty_weight * sum_sq
        return nll + penalty

    # We'll guess alpha ~ 0.0 initially or random
    x_bounds = [(0.0, 1.0)] * k_params
    x_guesses = [0.0] * k_params  # or random, or something else

    # run global+local optimization (the same function you used before)
    # e.g. global_local_optimization from your code:
    best_result: OptimizeResult = global_local_optimization(
        objective_fn,
        x_bounds=x_bounds,
        x_guesses=x_guesses,
        maxiter_global=maxiter_global,
        maxiter_local=maxiter_local,
        optimization_method=optimization_method
    )

    best_alpha = list(best_result['final']['x'])
    best_priors = gnrl.transform_to_simplex(best_alpha)
    hypothesis_space_final = {prof: float(pr) for prof, pr in zip(best_profiles, best_priors)}

    nll_no_penalty, n_data = typological_model_nll_for_player(
        hypothesis_space=hypothesis_space_final,
        general_settings=general_settings,
        file_paths=file_paths,
        player_uuid=player_uuid
    )
    sum_sq_final = sum(best_priors * best_priors)
    penalty_final = penalty_weight * sum_sq_final
    total_loss = nll_no_penalty + penalty_final

    # if you want AIC/BIC for each player, we define k_eff = k_params - 1 or so
    k_eff = max(k_params - 1, 1)
    ic_dict = compute_ic(k_eff, n_data, nll_no_penalty)
    # print(f"Loss: {round(total_loss, 6)}; Player: {player_uuid}")
    return {
        "n_data": n_data,
        "player_uuid": player_uuid,
        "k_params": k_params,
        "best_alpha": best_alpha,
        "nll_no_penalty": nll_no_penalty,
        "penalty": penalty_final,
        "total_loss": total_loss,
        "AIC": ic_dict["AIC"],
        "BIC": ic_dict["BIC"],
        "success": best_result["final"]["success"],
        "message": best_result["final"]["message"],
        "profiles": best_profiles,
        "best_priors_normalized": list(best_priors),
    }


def typological_model_comparison_fit_individually(best_profiles: list[tuple[float, float]], general_settings: Dict[str, Any], file_paths: Dict[str, str], penalty_weight: float = 10, 
                                                  maxiter_global: int = 2, maxiter_local: int = None, optimization_method: str = 'globloc', save_csv: bool = True) -> Tuple[pd.DataFrame, float]:
    """
    Fits a discrete model individually to each player, using a fixed set of profiles.

    Arguments:
        • best_profiles: list[tuple[float, float]]
            The discrete profiles chosen from your population-level stage.
        • general_settings: Dict[str, Any]
            E.g. {'experiment_num': 2, 'run_in_parallel': True, ...}
        • file_paths: Dict[str, str]
            Must allow us to load the data. 
            Optionally, a path to save the final CSV (like file_paths["discrete"]).
        • penalty_weight: float
            Weight for the sum-of-squares penalty on priors.
        • maxiter_global: int
            Global method iteration limit.
        • maxiter_local: int
            Local method iteration limit.
        • optimization_method: str
            One of {'global','local','globloc'}.
        • save_csv: bool
            If True, writes results to a CSV file at the end.

    Returns:
        (df, total_nll):
            df is a DataFrame with one row per player:
                [player_uuid, best_alpha, best_priors_normalized, nll_no_penalty, ...]
            total_nll is sum of all players' nll_no_penalty (no penalty included).

    Notes:
        • This is the standard approach to get a per-player best-fitting prior distribution,
          given a single “best” set of profiles for the entire population.
        • Summing the 'nll_no_penalty' column in the returned DataFrame gives the total NLL
          across all players, so you can compare it to your continuous model's total.
    """
    # 1) Gather the relevant players
    #    E.g., load from the same file you used in alt_model_contest or create a function
    experiment_num = general_settings.get("experiment_num", 2)
    # load player_info or something:
    # Usually, you might have "histories_data['player_info']", or you can parse the same file you parse for dyads.
    # For example:

    full_path_histories = os.path.join(
        file_paths["processed"],
        file_paths["file_names"][f"player_pairs_exper{experiment_num}"]
    )
    with open(full_path_histories, "r", encoding="utf-8") as f:
        data_all = json.load(f)
    # We can gather all predictor players:
    #   If "player_info" is available, you can do that. 
    #   Otherwise, parse the "histories" to get unique predictor IDs:
    all_histories = data_all["histories"]

    player_uuids = set()
    for dyad_key, dyad_games in all_histories.items():
        if dyad_games:
            pid = dyad_games[0].get("predictor", None)
            if pid is not None:
                player_uuids.add(pid)
    player_uuids = sorted(list(player_uuids))

    # 2) Prepare for parallel or serial
    args_list = []
    for player_uuid in player_uuids:
        args_list.append((
            player_uuid,
            best_profiles,
            general_settings,
            file_paths,
            penalty_weight,
            maxiter_global,
            maxiter_local,
            optimization_method
        ))

    # 3) Run
    df_columns = [
        "n_data", "player_uuid", "k_params", "best_alpha", "nll_no_penalty", "penalty", 
        "total_loss", "AIC", "BIC", "success", "message", "profiles", "best_priors_normalized",
    ]
    df_results = pd.DataFrame(columns=df_columns)
    
    time_start = time.time()
    sum_nll_no_penalty = 0

    run_in_parallel = general_settings.get("run_in_parallel", True)
    if run_in_parallel:
        with mp.Pool(processes=max(mp.cpu_count()-1, 1)) as pool:
            for idx, result in enumerate(pool.imap_unordered(_parallel_process_worker_typological_model_comparison_individual_fit, args_list), 1):
                df_results.loc[len(df_results)] = [
                    result["n_data"],
                    result["player_uuid"],
                    result["k_params"],
                    result["best_alpha"],
                    result["nll_no_penalty"],
                    result["penalty"],
                    result["total_loss"],
                    result["AIC"],
                    result["BIC"],
                    result["success"],
                    result["message"],
                    result["profiles"],
                    result["best_priors_normalized"],
                ]

                sum_nll_no_penalty += result["nll_no_penalty"]

                if idx % 1 == 0:
                    "Compute remaining time"
                    time_now = time.time()
                    k_params = result['k_params']
                    total_time = time_now - time_start
                    average_duration = total_time / (idx + 1)
                    n_remaining_iters = len(args_list) - idx
                    remaining_seconds = average_duration * n_remaining_iters
                    remaining_minutes = remaining_seconds / 60
                    remaining_hours = int(remaining_minutes / 60)
                    remaining_minutes = int(remaining_minutes % 60)
                    current_time = dt.datetime.now().strftime("%H%M")
                    print_str = f"k={k_params} Processed {idx}/{len(args_list)} subsets. "
                    print_str += f"Time: {current_time}; Remaining Time: {remaining_hours:02d} hours "
                    print_str += f"and {remaining_minutes:02d} minutes. Sum Loss = {sum_nll_no_penalty}"
                    print(print_str)

    else:
        for idx, single_args in enumerate(args_list, 1):
            result = _parallel_process_worker_typological_model_comparison_individual_fit(single_args)
            df_results.loc[len(df_results)] = [
                result["n_data"],
                result["player_uuid"],
                result["k_params"],
                result["best_alpha"],
                result["nll_no_penalty"],
                result["penalty"],
                result["total_loss"],
                result["AIC"],
                result["BIC"],
                result["success"],
                result["message"],
                result["profiles"],
                result["best_priors_normalized"],
            ]

            sum_nll_no_penalty += result["nll_no_penalty"]

            if idx % 5 == 0:
                "Compute remaining time"
                time_now = time.time()
                k_params = result['k_params']
                total_time = time_now - time_start
                average_duration = total_time / (idx + 1)
                n_remaining_iters = len(args_list) - idx
                remaining_seconds = average_duration * n_remaining_iters
                remaining_minutes = remaining_seconds / 60
                remaining_hours = int(remaining_minutes / 60)
                remaining_minutes = int(remaining_minutes % 60)
                current_time = dt.datetime.now().strftime("%H%M")
                print_str = f"k={result['k_params']} Processed {idx}/{len(args_list)} subsets. "
                print_str += f"Time: {current_time}; Remaining Time: {remaining_hours:02d} hours "
                print_str += f"and {remaining_minutes:02d} minutes. Sum Loss = {sum_nll_no_penalty}"
                print(print_str)

    # 4) Summation of all players' NLL (no penalty)
    total_nll = df_results["nll_no_penalty"].sum()
    df_results = df_results.sort_values(by="nll_no_penalty", ascending=True)

    # 5) Optionally save
    if save_csv:
        out_dir = file_paths.get("discrete", ".")
        out_path = os.path.join(out_dir, f"discrete_individual_fits_k={len(best_profiles)}.csv")
        df_results.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"[discrete_individual_fits] Completed. Results saved to {out_path}.")

    print(f"Total NLL For Individuall Fit k = {len(best_profiles)} Model: {round(total_nll, 6)}")
    return df_results, total_nll


"=========================================================================================="
"========= Information Criterion Analysis: Determining Optimal Utility Structures ========="
"=========================================================================================="

def information_criterion_analysis(general_settings: Dict[str, Any], utility_settings: Dict[str, bool], file_paths: Dict[str, str], 
                                   param_bds: Dict[str, tuple[int | float, int | float]], dynamic_updating: bool = False, max_iters: int = 1, robustness_epsilon: float = 10,
                                   check_for_n_players: int | str = "all", write_mode: WriteMode = "resume") -> Tuple[pd.DataFrame, Dict[str, Dict[Tuple[bool], Dict[str, Any]]]]:
    """
    Computes and compares AIC/BIC across different utility function configurations.

    Arguments:
        • general_settings: Dict[str, Any]; High-level settings (analysis mode, etc.).
        • utility_settings: Dict[str, bool]; Flat dictionary of boolean flags controlling the functional form.
        • file_paths: Dict[str, str]; Paths to files/directories for reading/writing data.
        • param_info: Dict[str, Any]; Contains parameter keys, bounds, guesses, etc.
        • robustness_epsilon: float; If sum of ΔMinLoss < this threshold for two 
            consecutive iterations, we stop early.
        • dynamic_updating: bool; Because the computational demands of fitting individual-level belief updating
            are prohibitive, this analysis relies on a static (no updating) version of the UBM by default. Yet,
            setting this input to True can work on a machine with more cores.  

    Returns:
        • df: pd.DataFrame; Dataframe summarizing the IC metrics (loss, AIC, BIC) for each utility configuration.
        • all_ic_results: Dict[str, Dict[Tuple[bool], Dict[str, Any]]]; 
            - "ic_results": Maps each utility config (as a tuple of booleans) to its {n,k,loss,AIC,BIC}.
            - "utility_varieties": Maps the same config tuple to the actual dictionary of settings.
    """
    "Storing terminal printouts in a .txt file"
    ic_terminal_printouts = [
        "This document contains terminal print statements for the information criterion utility function model comparison."
    ]

    def compute_mean_param_variance(param_info: Dict[str, Any], param_runs: List[Dict[str, Dict[str, Dict[str, float]]]]) -> float:
        """
        Computes a single 'normalized average parameter variance' across all runs, participants, and roles.

        Arguments:
        ----------
        • param_info : Dict[str, Any]
            Must have 'keys' (list of parameter names) and 'bounds' (list of (low, high) tuples).
            e.g. param_info['keys'] = ['Vᵢᵢ','Vᵢⱼ','Ƹᵢⱼ','Ʒᵢⱼ','γ1','τ']
                    param_info['bounds'] = [(0,1),(-1,1),... etc.]

        • param_runs : List[Dict[str, Dict[str, Dict[str,float]]]]
            Each element corresponds to a single iteration's "players_to_params_this_iter".
            For iteration i, param_runs[i] is a dict:
                {
                    player_uuid_1: {"chooser": {...}, "predictor": {...}},
                    player_uuid_2: {"chooser": {...}, "predictor": {...}},
                    ...
                }
            Inside each "..." is a mapping param_key -> param_value.

        Returns:
        --------
        A single float in [0,1] (roughly). 
        A value near 0 means all runs found nearly identical param solutions 
        (hence stable). Higher means more discrepancy across runs.
        """

        # If we have fewer than 2 runs, variance is trivially zero
        if len(param_runs) < 2:
            return 0.0

        param_names = param_info['keys']
        param_bounds = param_info['bounds']  # same order as param_names

        # 1) For each participant, for each role, for each param, gather a list of values across runs
        #    We'll store them in something like:
        #    values_dict[(player_uuid, role, param_name)] = [val_run1, val_run2, ... val_runR]
        from collections import defaultdict
        values_dict = defaultdict(list)

        n_runs = len(param_runs)

        for run_idx in range(n_runs):
            run_data = param_runs[run_idx]
            # run_data => {player_uuid: { 'chooser': {pkey: val}, 'predictor': {pkey: val} }, ...}
            for player_uuid, role_dict in run_data.items():
                for role_name, param_map in role_dict.items():
                    # param_map => { param_key: param_val, ...}
                    if not param_map:
                        continue
                    # For each param in param_names:
                    for pkey, pval in param_map.items():
                        # We'll only process it if pkey is in param_names
                        if pkey in param_names:
                            # store it
                            values_dict[(player_uuid, role_name, pkey)].append(pval)

        # 2) For each param triple (player, role, param), compute variance in normalized scale
        #    Then we average across all (player, role, param).
        #    We'll skip any triple that has < 2 data points (maybe that triple wasn't used).
        all_variances = []
        for (ply, rle, pkey), val_list in values_dict.items():
            if len(val_list) < 2:
                # can't compute variance with <2 points
                continue

            # find the index in param_names
            try:
                p_index = param_names.index(pkey)
            except ValueError:
                continue

            bound_low, bound_high = param_bounds[p_index]
            param_range = (bound_high - bound_low)
            if param_range <= 0:
                # just skip or treat as zero
                continue

            # normalize each value
            norm_vals = [(v - bound_low) / param_range for v in val_list]

            # compute variance
            var_ = np.var(norm_vals, ddof=1)  # ddof=1 => sample variance
            all_variances.append(var_)

        if not all_variances:
            return 0.0

        # 3) Final average
        return float(np.mean(all_variances))

    def _compute_normalised_param_sd(pvec, param_bds):
        """
        pvec : list of per-iteration dicts
            [ iteration0, iteration1, … ]
            Each item -> {player_uuid: {'chooser': {param: val, …},
                                        'predictor': {param: val, …}}, …}

        param_bds : dict {param_name: (low, high)}

        Returns a *scalar*:
            median_over_params(   SD_over_iters&players(param) / (high-low)   )
        If a param is missing bounds, it is skipped.
        """
        if not pvec:
            return np.nan

        # Flatten -> {param: [values]}
        bag = {}
        for iter_dict in pvec:
            for pl_dict in iter_dict.values():         # players
                for role_dict in pl_dict.values():     # chooser / predictor
                    if not isinstance(role_dict, dict):
                        continue
                    for k, v in role_dict.items():
                        bag.setdefault(k, []).append(float(v))

        norm_sds = []
        for param, vals in bag.items():
            bd = param_bds.get(param)
            if not bd:
                continue
            rng = bd[1] - bd[0]
            if rng <= 0 or len(vals) < 2:
                continue
            sd = np.std(vals, ddof=1)
            norm_sds.append(sd / rng)

        if not norm_sds:
            return np.nan
        return float(np.median(norm_sds))

    def check_nesting_fit_violations(target_model: tuple[bool] | dict[str: bool] | str, models_to_sequential_losses: dict, 
                                     general_settings=general_settings, utility_settings=utility_settings, file_paths=file_paths, print_only_children: bool = True) -> dict:
        """
        Checks if the loss found for the target model is greater than its children or less than its parents.
        """
        def model_key_maker(model: tuple[bool] | dict[str: bool] | str, into: type) -> str | tuple[int]:
            """
            Converts models into compact representations, like
            00000000000010~Uᵢ(A)=Vᵢᵢ(πᵢᴬ-πᵢᴮ)-Ƹᵢⱼ×(max(πⱼᴬ-πᵢᴬ, 0)+max(πᵢᴬ-πⱼᴬ,0))
            """
            if into is str:
                if isinstance(model, str):
                    model = ast.literal_eval(model)
                if isinstance(model, (dict, tuple)):
                    model = gnrl.convert_utility_settings(utility_settings=model, into=int)
                else:
                    raise TypeError(f"model must be a tuple, dict, or string, not {type(model)}!")
                
                return str(model)[1:-1].replace(", ", "") + "~" + build_utility_equation(utility_settings=model).replace(" ", "")
            
            elif into is tuple:
                if isinstance(model, str) and model.split("~")[0].isdigit():
                    return tuple(int(dig) for dig in model.split("~")[0])
                raise ValueError(f"If into is tuple, then model must be a string of 0s and 1s.")

            raise TypeError(f"into must be str or tuple, not {into}.")

        "Converting all models into a common format"
        target_model_key = gnrl.convert_utility_settings(
            utility_settings=ast.literal_eval(target_model) if 
            isinstance(target_model, str) else target_model, into=tuple
        )
        target_model_settings = gnrl.convert_utility_settings(
            utility_settings=target_model_key, into=dict)
        models_to_losses = {
            ast.literal_eval(key): min(val) for key, val in 
            copy.deepcopy(models_to_sequential_losses).items()
        }

        "Extract model nesting data"
        model_nesting_data = model_nesting_adjacency_matrices(
            general_settings=general_settings, utility_settings=utility_settings, 
            file_paths=file_paths, create_new_file=False, print_=False)
        
        "List of all models as tuples of boolean flags"
        model_setting_tuples: list[tuple] = [gnrl.convert_utility_settings(
            utility_settings=settings, into=tuple) for settings in model_nesting_data['settings']]

        "Index of the new model in the list of models"
        target_model_settings_idx = next((settings_idx for settings_idx, settings in enumerate(
            model_setting_tuples) if settings == target_model_key), -1)
        
        "Lists of the parents and children of the new model as tuples of boolean flags"
        parents_of_target_model = [
            model_setting_tuples[jdx]
            for jdx in model_nesting_data['adjacency_lists']['parent_of'][target_model_settings_idx]
        ]
        children_of_target_model = [
            model_setting_tuples[jdx]
            for jdx in model_nesting_data['adjacency_lists']['child_of'][target_model_settings_idx]
        ]
        
        "Creating the primary dictionary of data to check for nesting violations"
        nesting_fit_violation_data = {
            'counts': {
                'violations':  {'parents': 0, 'children': 0},
                'observances': {'parents': 0, 'children': 0}
            },
            'parents': {}, 
            'children': {}
        }
        
        "Nesting violations can only apply to models with at least two parameters."
        n_params = gnrl.count_free_parameters(utility_settings=target_model_settings)
        if n_params > 1:

            "The lowest loss of the new model is the reference point"
            target_model_loss = models_to_losses[target_model_key]

            "Comparing the min losses of parent models to the min loss of the new model"
            for parent_model in parents_of_target_model:
                parent_loss = models_to_losses.get(parent_model, None)
                if isinstance(parent_loss, float):
                    if abs(parent_loss) > abs(target_model_loss):
                        parent_key = model_key_maker(model=parent_model, into=str)
                        nesting_fit_violation_data['parents'][parent_key] = parent_loss - target_model_loss
                        nesting_fit_violation_data['counts']['violations']['parents'] += 1
                    else:
                        nesting_fit_violation_data['counts']['observances']['parents'] += 1

            "Comparing the min losses of child models to the min loss of the new model"
            for child_model in children_of_target_model:
                child_loss = models_to_losses.get(child_model, None)
                if isinstance(child_loss, float):
                    if abs(child_loss) < abs(target_model_loss):
                        child_key = model_key_maker(model=child_model, into=str)
                        nesting_fit_violation_data['children'][child_key] = child_loss - target_model_loss
                        nesting_fit_violation_data['counts']['violations']['children'] += 1
                    else:
                        nesting_fit_violation_data['counts']['observances']['children'] += 1

            "Printing a compact representation to the terminal"
            vio_chi = nesting_fit_violation_data['counts']['violations' ]['children']
            obs_chi = nesting_fit_violation_data['counts']['observances']['children']
            vio_par = nesting_fit_violation_data['counts']['violations' ]['parents' ]
            obs_par = nesting_fit_violation_data['counts']['observances']['parents' ]
            if print_only_children:
                if vio_chi:
                    statement_1 = f"VIOLATIONS DETECTED: Children: {vio_chi}/{(vio_chi + obs_chi)}:"
                    statement_2 = f"[TARGET]   Loss: {target_model_loss:10.6f} ~ {model_key_maker(model=target_model, into=str)}"
                    ic_terminal_printouts.append(statement_1)
                    ic_terminal_printouts.append(statement_2)
                    print(statement_1), print(statement_2)
                    for child, loss in list(nesting_fit_violation_data['children'].items())[:10]:
                        statement_child = f"[CHILD]  Δ Loss: {loss:10.6f} ~ {child }"
                        ic_terminal_printouts.append(statement_child)
                        print(statement_child)
                    
            elif vio_chi or vio_par:
                statement_1 = f"VIOLATIONS DETECTED: Parents: {vio_par}/{(vio_par + obs_par)} & Children: {vio_chi}/{(vio_chi + obs_chi)}:"
                statement_2 = f"[TARGET]   Loss: {target_model_loss:10.6f} ~ {model_key_maker(model=target_model, into=str)}"
                ic_terminal_printouts.append(statement_1)
                ic_terminal_printouts.append(statement_2)
                print(statement_1), print(statement_2)       
                for child, loss in list(nesting_fit_violation_data['children'].items())[:10]:
                    statement_child = f"[CHILD]  Δ Loss: {loss:10.6f} ~ {child }"
                    ic_terminal_printouts.append(statement_child)
                    print(statement_child)
                for parent, loss in list(nesting_fit_violation_data['parents'].items())[:10]:
                    statement_parent = f"[PARENT] Δ Loss: {loss:10.6f} ~ {parent}"
                    ic_terminal_printouts.append(statement_parent)
                    print(statement_parent)

        return nesting_fit_violation_data

    def build_ic_dataframe_for_ranking(ic_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Helper to get a DataFrame from current ic_results_dict so we can compute ranks easily.
        """
        rows = []
        for mk, info in ic_dict.items():
            # skip if 'loss' is None
            if info['loss'] is None:
                continue
            rows.append({
                "model_key": mk,
                "loss": info['loss'],
                "AIC": info['AIC'],
                "BIC": info['BIC']
            })
        df_local = pd.DataFrame(rows)
        # rank by BIC ascending
        df_local["BIC_rank"] = df_local["BIC"].rank(method="min", ascending=True)
        return df_local
    
    def make_unique_guesses(bounds, *, model_key, iter_idx, restart_idx):
        """
        bounds: list[(low, high)] aligned with param_info['keys'].
        Returns a list of floats with unique-but-deterministic randomness.
        """
        def _seed_from(*parts) -> int:
            """Deterministic 32-bit seed from stable identifiers."""
            hash = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
            return int(hash[:8], 16)
          
        seed = _seed_from("IC", model_key, iter_idx, restart_idx)
        rng = random.Random(seed)
        return [rng.uniform(lo, hi) for (lo, hi) in bounds]

    def _warmstart_temperature(iter_idx: int, warmstart_policy: dict) -> float | None:
        """
        Returns None ⇒ 'cold' (no warm-starts).
        Simple schedules that do not depend on final horizon.
        """
        cold_iters = int(warmstart_policy.get("cold_iters", 2))     # no warm-starts for the first K iterations
        temp_high  = float(warmstart_policy.get("temperature_high", 1000.0))
        temp_low   = float(warmstart_policy.get("temperature_low", 0.05))
        schedule   = str(warmstart_policy.get("schedule", "binary")).lower()

        if iter_idx <= cold_iters:
            return None  # cold phase

        if schedule == "binary":
            explore_iters = int(warmstart_policy.get("explore_iters", 3))
            return temp_high if iter_idx <= cold_iters + explore_iters else temp_low

        if schedule == "exp":
            # Exponential cooling with a half-life in 'warm' phase
            half_life = float(warmstart_policy.get("half_life", 2.0))
            t = max(0, iter_idx - cold_iters)
            return max(temp_low, temp_high * (0.5 ** (t / half_life)))

        # default: binary
        return temp_high

    def model_comparison_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a smaller dataframe that comparisons the top performing models
        """
        best_models = []

        # For each k_params group:
        for kprms in reversed(range(1, 10)):
            df_k = df[df['k_params'] == kprms]
            if df_k.empty:
                continue

            # 1) Identify the winning row (lowest BIC => rank=0)
            best_row = df_k.loc[df_k['BIC_rank'] == 0].copy()
            if best_row.empty:
                continue  # Shouldn't happen, but just in case

            # Typically there's only one row with rank=0, but let's handle if there's a tie:
            best_row = best_row.iloc[0]  # pick the first if there's a tie

            # 2) Identify the runner-up (rank=1 within that same k_params)
            runner_up = df_k.loc[df_k['BIC_rank'] == 1]
            if not runner_up.empty:
                # If there's a tie for rank=1, pick any or the first
                runner_up_BIC = runner_up.iloc[0]['BIC']
                best_BIC = best_row['BIC']
                next_best_ic = runner_up_BIC - best_BIC
            else:
                # No runner-up
                next_best_ic = float('nan')

            # 3) Attach that difference as a new column in the row
            best_row['next_best_IC'] = next_best_ic

            # 4) Accumulate for building comp_df
            best_models.append(best_row)

        comp_df = pd.DataFrame(best_models).sort_values('k_params')
        # optional Δ
        min_BIC_in_comp = comp_df['BIC'].min()
        comp_df['ΔBIC'] = comp_df['BIC'] - min_BIC_in_comp
        
        "Move equation column to the end."
        equation_column = comp_df.pop('equation')
        comp_df['equation'] = equation_column

        # if not general_settings.get('write_mode') == 'readonly':
        comp_csv_path = prep.ensure_directory_and_join(file_paths["bic_aic"], 
                            f"IC_Analysis_Comparison_Table_Experiment{experiment_num}.csv")
        comp_df.to_csv(comp_csv_path, index=False, encoding='utf-8-sig')
        print(f"Saved comparison table to: {comp_csv_path}\n")

        return comp_df

    def ic_results_df(df_dict: dict) -> pd.DataFrame:
        """
        Creates the main dataframe that stores the results of the IC analysis.
        """
        # Create a sorted DataFrame.
        df = pd.DataFrame(df_dict)
        df = df.sort_values(by='BIC', ascending=True)

        # 7) Compute ΔAIC and ΔBIC
        minAIC = df['AIC'].min()
        minBIC = df['BIC'].min()
        df['ΔAIC'] = df['AIC'] - minAIC
        df['ΔBIC'] = df['BIC'] - minBIC

        # 8) Ranks by AIC & BIC
        df['AIC_rank'] = df.groupby('k_params')['AIC'].rank(method='min').astype(int) - 1
        df['BIC_rank'] = df.groupby('k_params')['BIC'].rank(method='min').astype(int) - 1

        #Move 'equation' column to the end to make equations easier to see in Excel
        equation_column = df.pop('equation')
        df['equation'] = equation_column

        # Save the DataFrame to a CSV (or JSON) in the bic_aic folder.
        try:
            df_file_path = os.path.join(base_file_paths["bic_aic"], 
                f"All_Utility_Forms_IC_Analysis_Experiment{experiment_num}.csv")
            df.to_csv(df_file_path, index=False, encoding='utf-8-sig')
            print(f"Saved DataFrame to {df_file_path}")
        except (PermissionError, OSError):
            pass

        return df
        
    def ic_correlations(df: pd.DataFrame) -> pd.DataFrame:
        """
        Prints correlations and statistics
        """
        # 11) Correlations among {AIC, BIC, k_params, loss}
        df_valid = df.dropna(subset=['AIC', 'BIC', 'k_params', 'loss'])
        corr_matrix = df_valid[['AIC', 'BIC', 'k_params', 'loss']].corr()
        print("Correlation matrix among AIC, BIC, k_params, and loss:\n", corr_matrix)

        import statsmodels.formula.api as smf
        # Print the top row(s) with minimal IC.
        min_ic = df['BIC'].min()
        min_ic_df = df.loc[df['BIC'] == min_ic]
        print("\n--- Best Utility Config by IC ---")
        print(min_ic_df)

        df['ΔBIC'] = df['BIC'] - min_ic
        top_uf_data = df[df['ΔBIC'] < 2]
        print(f"\n--- Top Utility Config(s) by BIC---")
        print(top_uf_data), print("")

        formula = f"BIC ~ C(conditional_welfare_mode) + C(reference_dependent_altruism) + " \
                "C(use_exponential_parameters) + C(single_exponential_parameter) + "  \
                "C(single_payoffs_not_differences) + C(payoff_ratios_not_differences) + "  \
                "C(reference_dependent_utility) + C(use_negativity_parameters) + "  \
                "C(negativity_social_comparison) + C(include_social_comparison) + "  \
                "C(include_altruism_term) + C(fix_self_interest_parameter) + k_params"  
        model = smf.ols(formula, data=df)
        results = model.fit()
        print(results.summary())    

    "Storing total nesting violations across models in each iteration."
    nesting_violation_counts_per_iter = []

    if not isinstance(max_iters, int) or max_iters <= 0:
        # max_iters must be a positive non-zero integer.
        max_iters = 1

    if general_settings.get('write_mode') == 'readonly':
    # if not general_settings.get('create_new_file'):
        # No point in iterating if we already intend to extract saved files.
        max_iters = 1

    player_subset = False
    if isinstance(check_for_n_players, int) and (0 < check_for_n_players):
        print(f"Warning: Running IC analysis over a subset of {check_for_n_players} players!")
        player_subset = True

    # Copy inputs to avoid unintended side-effects.
    general_settings = copy.deepcopy(general_settings)
    utility_settings = copy.deepcopy(utility_settings)
    base_file_paths = copy.deepcopy(file_paths)
    loss_funct_type = general_settings.get('loss_funct_type')

    # Remove suffix from file names if any.
    base_file_paths = prep.add_remove_file_name_suffix(
        file_paths=base_file_paths, file_name_suffix=None, add_suffix=False
    )

    "Use static updating by default"
    if dynamic_updating:
        update_method = 'grid'
        general_settings['use_particle_filter'] = True
    else:
        update_method = 'naive'
    general_settings['update_method'] = update_method

    "Temperature should be held constant to keep all models on an even footing."
    general_settings['temperature_is_param'] = False
    general_settings['run_in_parallel'] = True                                       

    # Determine which experiment to analyze.
    experiment_num = general_settings.get('experiment_num', 3)

    # Path to save the final results.
    all_ic_results_file_path = prep.ensure_directory_and_join(
        file_paths["bic_aic"], 
        f"All_Utility_Forms_IC_Analysis_Experiment{experiment_num}.json"
    )

    # Prepare containers.
    ic_results_dict = {}
    utility_varieties = {}

    # Generate all valid utility configurations.
    utility_setting_varieties = gnrl.generate_utility_settings(utility_settings=utility_settings, sort_by_k=True) #[126:] #HACK [126:]

    n_varieties = len(utility_setting_varieties)

    # Keep track of the best min loss found so far for each model
    models_to_sequential_losses: Dict[str, List[float]] = {}
    models_to_sequential_params: Dict[str, List[Dict[str, Dict[str, Dict[str, float]]]]] = {}
    models_to_sequential_losses_and_params: Dict[str, List[Dict[str, Dict[str, Dict[str, Dict[str, float]]]]]] = {}
    players_to_params_this_iter = {} # Placed here to prevent an error if not create_new_file.

    # For storing iteration-based sums
    sum_delta_minimum_loss_by_iter: List[float] = []
    rank_change_by_iter: List[float] = []
    rank_change_median_by_iter: List[float] = []

    # We keep track of the old ranks
    old_ranks: Dict[str, float] = {}

    consecutive_small_improvements = 0  # how many times in a row sum_delta_min_loss < epsilon

    minimum_params_and_losses = {}

    # Iterate model for a robustness analysis.
    for iter_idx in range(1, max_iters + 1):

        if iter_idx <= 1 and general_settings.get('write_mode', 'resume') == 'overwrite':
            try:
                "Start the terminal printouts file from scratch."
                term_print_output_path = os.path.join(file_paths['bic_aic'], 'ic_terminal_printouts.txt')
                os.makedirs(file_paths['bic_aic'], exist_ok=True)
                with open(term_print_output_path, 'w', encoding='utf-8') as file:
                    pass
            except (PermissionError, OSError):
                "Pass if I have the file open."
                pass

        # Store sum of detal min loss for this iteration
        sum_delta_minimum_loss_this_iter = 0.0

        "Initializing the nesting violation counts for this iteration."
        nesting_violation_counts_per_iter.append({
            'violations':  {'parents': 0, 'children': 0}, 
            'observances': {'parents': 0, 'children': 0}
        })

        # Loop over each valid config.
        for utility_idx, utility_setting_variety in enumerate(utility_setting_varieties):

            param_info = make_param_info(param_bds=param_bds, 
                utility_settings=utility_setting_variety, general_settings=general_settings, 
                random_guesses_are_unique=True, guess_seed=None)
                # random_guesses_are_unique=not general_settings['run_in_parallel'])

            k_params = len(param_info['keys'])  # Number of free parameters.
            utility_setting_key = str(gnrl.convert_utility_settings(utility_settings=utility_setting_variety, into=tuple)) 

            # Adding model key to dict so it can be mapped to losses vector
            if utility_setting_key not in models_to_sequential_losses:
                models_to_sequential_losses[utility_setting_key] = []

            # Adding model key to dict so it can be mapped to params dict
            if utility_setting_key not in models_to_sequential_params:
                models_to_sequential_params[utility_setting_key] = []

            if utility_setting_key not in models_to_sequential_losses_and_params:
                models_to_sequential_losses_and_params[utility_setting_key] = []

            if utility_setting_key not in minimum_params_and_losses:
                minimum_params_and_losses[utility_setting_key] = {}

            # Create file name suffix from these settings.
            file_name_suffix = prep.create_file_name_suffix(
                general_settings=general_settings, utility_settings=utility_setting_variety
            )

            # Copy of standard file paths to alter with each loop.
            file_paths_this = copy.deepcopy(base_file_paths)

            # Re-add that suffix to file_paths.
            file_paths_this = prep.add_remove_file_name_suffix(
                file_paths=file_paths_this, file_name_suffix=file_name_suffix, add_suffix=True
            )

            # Where we store the per-config results if it exists.
            ic_results_file_path = prep.ensure_directory_and_join(
                file_paths_this["bic_aic"], f"IC_Analysis{file_name_suffix}.json")
            # Always try to seed from any prior file
            ic_prev = {}
            if os.path.exists(ic_results_file_path) :
                with open(ic_results_file_path, "r", encoding="utf-8") as file:
                    ic_prev = json.load(file)

            models_to_sequential_losses[utility_setting_key]            = ic_prev.get('lvec', [])
            models_to_sequential_params[utility_setting_key]            = ic_prev.get('pvec', [])
            models_to_sequential_losses_and_params[utility_setting_key] = ic_prev.get('plvec', [])
            minimum_params_and_losses[utility_setting_key]              = ic_prev.get('minvec', {})
            n_data_for_model                                            = ic_prev.get('n_data', 0)

            write_mode = general_settings.get('write_mode', 'resume')

            # If overwrite, clear everything and recompute
            if write_mode == 'overwrite' and iter_idx <= 1:
                models_to_sequential_losses[utility_setting_key].clear()
                models_to_sequential_params[utility_setting_key].clear()
                models_to_sequential_losses_and_params[utility_setting_key].clear()
                minimum_params_and_losses[utility_setting_key].clear()
                n_data_for_model = 0

            # readonly → skip running a new iteration; just use the seeded values
            if write_mode == 'readonly':
                total_loss_model = ic_prev.get('loss', (
                    min(models_to_sequential_losses[utility_setting_key])
                    if models_to_sequential_losses[utility_setting_key]
                    else float('nan')
                ))
            else:
                # Read the main processed histories if needed.
                file_path_histories = prep.ensure_directory_and_join(
                    file_paths_this["processed"], 
                    file_paths_this["file_names"][f'player_pairs_exper{experiment_num}']
                )
                with open(file_path_histories, "r", encoding='utf-8') as file:
                    original_histories: dict[str, dict[str, list[DyadGame]]] = json.load(file)

                # Gather all participants.
                player_uuids = prep.all_player_uuids(
                    file_paths=file_paths_this, 
                    experiment_num=experiment_num, 
                    only_humans=True
                )
                if player_subset:
                    player_uuids = sorted(player_uuids)[:check_for_n_players]

                # Raw data to be fitted
                player_histories = copy.deepcopy(original_histories)

                # Summation variables.
                total_loss_model = 0.0 # Total loss for model across experiment
                n_data_for_model = 0   # Number of data points in the experiment 

                # Overwrite lambda generated random numbers prevents pickeling errors in multiprocessing
                min_std_guess = 0.5
                param_info['guesses'] = [random.uniform(param_info['bounds'][kdx][0] if '_std' not in key else min_std_guess, 
                                    param_info['bounds'][kdx][1]) for kdx, key in enumerate(param_info['keys'])]

                # --- Warm-start policy for this iteration ----------------------------------
                default_warm_pol = {
                    "enabled": True,
                    "schedule": "binary",
                    "cold_iters": 2,
                    "explore_iters": 2,
                    "temperature_high": 1000.0,
                    "temperature_low": 0.01,
                    "disable_dual_annealing_when_warm": True,
                }
                warmstart_policy = general_settings.get("warmstart_policy", default_warm_pol)
                cur_T = _warmstart_temperature(iter_idx, warmstart_policy)
                # phase = "cold" if (cur_T is None) else "warm"
                phase = "cold" if iter_idx <= warmstart_policy.get("cold_iters", 2) else "warm"
                # ---------------------------------------------------------------------------

                "Force 'create_new_file' to be true if already deciding to create a new file."
                general_settings_ = copy.deepcopy(general_settings)
                # general_settings_['create_new_file'] = True            
                # general_settings_['write_mode'] = 'overwrite'

                general_settings_["warmstart_policy"] = {
                    **default_warm_pol,
                    **(general_settings.get("warmstart_policy", {})),
                    "temperature": (cur_T if cur_T is not None else 0.0),
                    "phase": phase
                }
                # If you prefer, also pin the method here. The local code inside fit_params_by_player
                # will switch to 'local' when phase == 'warm' and disable_dual_annealing_when_warm=True.
                general_settings_["optimization_method"] = general_settings.get("optimization_method", "globloc")

                if utility_idx == 0:
                    if iter_idx == int(warmstart_policy.get("cold_iters", 2)) + 1:
                        warm_start_message = (
                            f"[IC] ENTERING EXPLORATION WARM PHASE at iter {iter_idx} "
                            f"(schedule={warmstart_policy.get('schedule','binary')}, "
                            f"T={cur_T})"
                        )
                        ic_terminal_printouts.append(warm_start_message)
                        print(warm_start_message)
                    elif iter_idx == int(warmstart_policy.get("cold_iters", 2)) + int(warmstart_policy.get("explore_iters", 2)) + 1:
                        warm_start_message = (
                            f"[IC] ENTERING EXPLIOTATION WARM PHASE at iter {iter_idx} "
                            f"(schedule={warmstart_policy.get('schedule','binary')}, "
                            f"T={cur_T})"
                        )
                        ic_terminal_printouts.append(warm_start_message)
                        print(warm_start_message)                        

                # Run the naive analysis (similar to "bayes" approach).
                player_histories = run_analysis_bayes(
                    player_uuids=player_uuids if player_subset else None,
                    utility_settings=utility_setting_variety,
                    general_settings=general_settings_,
                    histories_data=player_histories,
                    file_paths=file_paths_this,
                    param_info=param_info, 
                    print_=False
                )

                # Storing parameter values by player for robustness analysis
                players_to_params_this_iter = {}
                players_to_params_and_losses_this_iter = {}

                # Accumulate total loss across all players/dyads.
                for player_uuid in player_uuids:
                    player_fit_dir = os.path.join(file_paths["player_fits"], f"experiment_{experiment_num}")
                    if experiment_num == 0:
                        player_fit_name = f"{player_uuid}.json"
                    else:
                        player_fit_name = f"{file_name_suffix}_{player_uuid}.json"

                    plr_file_path = prep.ensure_directory_and_join(base_dir=player_fit_dir, file_name=player_fit_name)
                    with open(plr_file_path, "r", encoding="utf-8") as file:
                        player_dyads = json.load(file)

                    players_to_params_this_iter[player_uuid] = {'chooser': {}, 'predictor': {}}
                    players_to_params_and_losses_this_iter[player_uuid] = {
                        'params': {'chooser': {}, 'predictor': {}}, 
                        'loss': {'chooser': 0.0, 'predictor': 0.0}
                    }

                    min_for_model = minimum_params_and_losses[utility_setting_key]
                    if player_uuid not in min_for_model:
                        min_for_model[player_uuid] = {
                            'params': {'chooser': {}, 'predictor': {}},
                            'loss':   {'chooser': float('inf'), 'predictor': float('inf')}
                        }
                            
                    # Before entering the dyad loop, set up per-player accumulators
                    role_loss_total = {'chooser': 0.0, 'predictor': 0.0}
                    param_est_by_role = {'chooser': None, 'predictor': None}

                    for dyad_key, dyad_games in player_dyads.items():
                        # compute loss for this dyad
                        loss_report = create_loss_report(dyad_games=dyad_games, general_settings=general_settings).get(player_uuid, {})
                        for player_role in ('chooser', 'predictor'):
                            loss_dict = loss_report.get(player_role, {})
                            loss_plr_role_dyad = loss_dict.get('raw_neglogprob_sum' if loss_funct_type == 'log' else 'raw_ssr', 0.0)
                            role_loss_total[player_role] += loss_plr_role_dyad

                            # n_data: keep your current logic; if n_data is per-role per-dyad, this is correct
                            n_data_for_model += loss_dict.get('n_data', 0)

                            # Grab params once (any dyad) if present
                            if param_est_by_role[player_role] is None and len(dyad_games) > 0:
                                first_game = dyad_games[0]
                                if isinstance(first_game, dict) and 'parameter_estimates' in first_game:
                                    param_est_by_role[player_role] = \
                                        first_game.get('parameter_estimates', {}).get(update_method, {}).get(
                                            player_uuid, {}).get(player_role, {}).get('params')

                    "Extract the parameters that minimize the raw losses, not the penalized losses."
                    "NOTE: If successful, this overwrites the logic directly above."
                    if len(dyad_games) > 0:
                        if "reports" in dyad_games[0]:
                            for player_role in ('chooser', 'predictor'):
                                if player_role in dyad_games[0]["reports"]:
                                    if "final" in dyad_games[0]["reports"][player_role]:
                                        if "min_raw_neglog_sum" in dyad_games[0]["reports"][player_role]["final"]:
                                            min_raw_neglog_sum = dyad_games[0]["reports"][player_role]["final"]["min_raw_neglog_sum"]
                                            param_est_by_role[player_role] = min_raw_neglog_sum["params"]
                                            role_loss_total[player_role] = min_raw_neglog_sum["loss"]

                    # Store what we saw this iteration
                    players_to_params_this_iter[player_uuid] = {
                        'chooser':   param_est_by_role['chooser']   or {},
                        'predictor': param_est_by_role['predictor'] or {}
                    }
                    players_to_params_and_losses_this_iter[player_uuid] = {
                        'params': players_to_params_this_iter[player_uuid],
                        'loss':   {'chooser': role_loss_total['chooser'], 'predictor': role_loss_total['predictor']}
                    }

                    # Update per-model minima exactly once per role, then add to total_loss_model
                    min_for_model = minimum_params_and_losses[utility_setting_key]
                    if player_uuid not in min_for_model:
                        min_for_model[player_uuid] = {
                            'params': {'chooser': {}, 'predictor': {}},
                            'loss':   {'chooser': float('inf'), 'predictor': float('inf')}
                        }

                    for role in ('chooser', 'predictor'):
                        if role_loss_total[role] < min_for_model[player_uuid]['loss'][role]:
                            min_for_model[player_uuid]['loss'][role]   = role_loss_total[role]
                            min_for_model[player_uuid]['params'][role] = param_est_by_role[role] or {}

                        total_loss_model += min_for_model[player_uuid]['loss'][role]

                # Debugging
                # total_loss_model should equal sum over players of minvec[uuid]['loss']['chooser'/'predictor']
                chk = 0.0
                for uuid, d in minimum_params_and_losses[utility_setting_key].items():
                    chk += float(d['loss'].get('chooser', 0.0)) + float(d['loss'].get('predictor', 0.0))
                assert abs(total_loss_model - chk) <= 1e-9, "models_to_sequential_losses inconsistent with minvec aggregation."
                # Debugging

            # Determine minimum model loss found up until the previous time step
            if len(models_to_sequential_losses[utility_setting_key]) == 0:
                prior_minimum_model_loss = float('inf')
            else:
                prior_minimum_model_loss = min(models_to_sequential_losses[utility_setting_key])

            # Append loss to loss vector
            if general_settings.get('write_mode') in ('resume', 'overwrite'):
                models_to_sequential_losses[utility_setting_key].append(total_loss_model)

            # Find miminum model loss up until now and the delta from the last time step
            # if models_to_sequential_losses[utility_setting_key]:
            minimum_model_loss = min(models_to_sequential_losses[utility_setting_key])
            # else:
            #     minimum_model_loss = 0.0 #TODO compute this from retrieved files.

            prev = prior_minimum_model_loss
            cur  = minimum_model_loss
            # We allow the first 'inf' → value transition
            if prev != float('inf'):
                assert cur <= prev + 1e-12, f"Minimum loss increased! prev={prev:.6f}, cur={cur:.6f}"

            # FIX #2: Just before computing delta_minimum_model_loss:
            if prior_minimum_model_loss == float('inf'):
                delta_minimum_model_loss = 0.0
            else:
                delta_minimum_model_loss = abs(minimum_model_loss - prior_minimum_model_loss)

            # Add to the sum of delta min losses across all models.
            sum_delta_minimum_loss_this_iter += delta_minimum_model_loss

            # Append params to param vector
            if general_settings.get('write_mode') in ('resume', 'overwrite'):
                models_to_sequential_params[utility_setting_key].append(players_to_params_this_iter)
                models_to_sequential_losses_and_params[utility_setting_key].append(players_to_params_and_losses_this_iter)

            parameter_variance = compute_mean_param_variance(param_info=param_info, 
                                    param_runs=models_to_sequential_params[utility_setting_key])

            # Print results for real-time feedback
            report_str = "Iter " + "0" * (len(str(max_iters)) - len(str(iter_idx))) + f"{iter_idx}/{max_iters} - Utility Model " 
            report_str += "0" * (len(str(n_varieties)) - len(str(utility_idx))) + f"{utility_idx}/{n_varieties} - "
            report_str += f"Loss: {total_loss_model:.6f}; Min Loss: {minimum_model_loss:.6f}; "
            report_str += f"Δ Min Loss: {delta_minimum_model_loss:.6f}; Param Var = {parameter_variance:.6f}"
            equation = build_utility_equation(utility_settings=utility_setting_variety)
            ic_terminal_printouts.append(report_str)
            ic_terminal_printouts.append(equation)
            print(report_str)
            print(equation)

            # If no data, store null results.
            if n_data_for_model == 0:
                ic_results = {
                    'idx': utility_idx,
                    'k_params': k_params,
                    'n_data': 0,
                    'loss': None,
                    'AIC': None,
                    'BIC': None,
                    'lvec': None,
                    'pvec': None,
                    'pvar': None,
                    'plvec': None,
                    'minvec': None,
                    'U': build_utility_equation(
                        utility_settings=utility_setting_variety),
                    'utility_settings': utility_setting_variety
                }
            else:
                # AIC/BIC formulas.
                ic_results = compute_ic(k_params=k_params, 
                                        n_data=n_data_for_model, 
                                        neg_log_likelihood=minimum_model_loss)

                ic_results = {
                    'idx': utility_idx,
                    'k_params': k_params,
                    'n_data': n_data_for_model,
                    'loss': minimum_model_loss,
                    'AIC': ic_results['AIC'],
                    'BIC': ic_results['BIC'],
                    'pvar': parameter_variance,
                    'U': build_utility_equation(
                        utility_settings=utility_setting_variety),
                    'lvec': models_to_sequential_losses[utility_setting_key],
                    'pvec': models_to_sequential_params[utility_setting_key],
                    'plvec': models_to_sequential_losses_and_params[utility_setting_key],
                    'minvec': minimum_params_and_losses[utility_setting_key],
                    'utility_settings': utility_setting_variety
                }

            # Store results in memory.
            ic_results_dict[utility_setting_key] = ic_results
            utility_varieties[utility_setting_key] = utility_setting_variety

            "Compute, store, and report model nesting fit violations"
            if write_mode != 'readonly':
                violations = check_nesting_fit_violations(target_model=utility_setting_variety, 
                                                    models_to_sequential_losses=models_to_sequential_losses)
                ic_results['nesting_violations'] = violations

                vio_par = violations.get('counts', {}).get('violations',  {}).get('parents',  0)
                vio_chi = violations.get('counts', {}).get('violations',  {}).get('children', 0)
                obs_par = violations.get('counts', {}).get('observances', {}).get('parents',  0)
                obs_chi = violations.get('counts', {}).get('observances', {}).get('children', 0)
                nesting_violation_counts_per_iter[-1]['violations' ]['parents' ] += vio_par
                nesting_violation_counts_per_iter[-1]['violations' ]['children'] += vio_chi
                nesting_violation_counts_per_iter[-1]['observances']['parents' ] += obs_par
                nesting_violation_counts_per_iter[-1]['observances']['children'] += obs_chi

            "Save results, overwriting previous JSON."
            if general_settings.get('write_mode') in ('resume', 'overwrite'):
                with open(ic_results_file_path, 'w', encoding='utf-8') as file:
                    json.dump(ic_results, file, ensure_ascii=False, indent=4)

            try:
                "Save terminal outputs"
                term_print_output_path = os.path.join(file_paths_this['bic_aic'], 'ic_terminal_printouts.txt')
                os.makedirs(file_paths_this['bic_aic'], exist_ok=True)
                "Open file in append mode to add new content without erasing the old"
                with open(term_print_output_path, 'a', encoding='utf-8') as file:
                    for line in ic_terminal_printouts:
                        file.write(line + '\n')
                "Empty list to preven saving duplicate information."
                ic_terminal_printouts.clear()
            except (PermissionError, OSError):
                "Pass if I have the file open."
                pass

        "Printing total model nesting fit violations to terminal"
        if write_mode != 'readonly':
            vio_par_ = nesting_violation_counts_per_iter[-1]['violations' ]['parents' ]
            vio_chi_ = nesting_violation_counts_per_iter[-1]['violations' ]['children']
            obs_par_ = nesting_violation_counts_per_iter[-1]['observances']['parents' ]
            obs_chi_ = nesting_violation_counts_per_iter[-1]['observances']['children']
            vio_statement = (
                f"TOTAL MODEL NESTING FIT VIOLATIONS FOR ITERATION {iter_idx}: "
                f"Parents: {vio_par_}/{(vio_par_ + obs_par_)}; "
                f"Children: {vio_chi_}/{(vio_chi_ + obs_chi_)}"
            )
            ic_terminal_printouts.append(vio_statement)
            print(vio_statement)

        # Store sum delta min loss for this iteration
        sum_delta_minimum_loss_by_iter.append(sum_delta_minimum_loss_this_iter)

        rounded_sds = [round(sum_delta, 6) for sum_delta in sum_delta_minimum_loss_by_iter]
        # print(f"Iter {iter_idx}: Sum Δ Min Losses: {rounded_sds}")
        sum_delta_min_loss_statement = f"Iter {iter_idx}: Sum Δ Min Losses: {rounded_sds}"
        ic_terminal_printouts.append(sum_delta_min_loss_statement)
        print(sum_delta_min_loss_statement)

        # 2) Build DF to compute rank changes
        df_for_rank = build_ic_dataframe_for_ranking(ic_results_dict)
        # e.g. df_for_rank: [model_key, loss, AIC, BIC, BIC_rank]
        rank_diffs = []
        new_ranks = {}
        for idx, row in df_for_rank.iterrows():
            mk = row["model_key"]
            nrank = row["BIC_rank"]
            new_ranks[mk] = nrank

            # If we had old_ranks, compute difference
            old_r = old_ranks.get(mk, nrank)  # if not present, no difference
            rank_diffs.append(abs(nrank - old_r))

        # sum of rank changes across all models
        sum_rank_diff = sum(rank_diffs)
        median_rank_diff = np.median(rank_diffs) if rank_diffs else 0.0
        rank_change_by_iter.append(sum_rank_diff)
        rank_change_median_by_iter.append(median_rank_diff)

        rank_statement = (f"Iter {iter_idx}: Sum of rank changes = {sum_rank_diff:.3f},"
                          f" median rank change={median_rank_diff:.3f}")
        ic_terminal_printouts.append(rank_statement)
        print(rank_statement)

        # Update old_ranks
        old_ranks = new_ranks

        "Save models_to_sequential_losses"
        models_to_sequential_losses_file_path = prep.ensure_directory_and_join(
            base_dir=file_paths['bic_aic'], file_name="models_to_sequential_losses.json") 
        if general_settings.get('write_mode') in ('resume', 'overwrite'):
            try:
                with open(models_to_sequential_losses_file_path, 'w', encoding='utf-8') as file:
                    json.dump(models_to_sequential_losses, file, ensure_ascii=False, indent=4)
            except (PermissionError, OSError):
                pass


        # ---------------------------------------------------------------
        for mk, ic in ic_results_dict.items():
            pvec = ic.get('pvec', [])
            ic['param_norm_sd'] = _compute_normalised_param_sd(pvec, param_bds)

        # Build the final 'robustness_analysis_data'
        robustness_analysis_data = {
            'sum_delta_minimum_loss_by_iter': sum_delta_minimum_loss_by_iter,
            'rank_change_by_iter': rank_change_by_iter,
            'rank_change_median_by_iter': rank_change_median_by_iter
        }

        # Combine into a single dict and write out.
        all_ic_results = {
            'ic_results': ic_results_dict,
            'utility_varieties': utility_varieties,
            'robustness_analysis_data': robustness_analysis_data,  # I want to store the analysis 
            'n_iterations_completed': iter_idx
        }

        # JSON serializable version of the same dictionary
        all_ic_results_serializable = {
            'ic_results': {str(key): val for key, val in ic_results_dict.items()},
            'utility_varieties': {str(key): val for key, val in utility_varieties.items()},
            'robustness_analysis_data': robustness_analysis_data
        }
        try:
            with open(all_ic_results_file_path, 'w', encoding='utf-8') as file:
                json.dump(all_ic_results_serializable, file, ensure_ascii=False, indent=4)
        except (PermissionError, OSError):
            pass

        # Build a DataFrame summarizing all results.
        # Start by building column lists for each key in utility_settings + {n,k,loss,AIC,BIC}.
        df_dict = {key: [] for key in utility_settings}  # each setting as a column

        extra_cols = ['idx', 'n_data', 'k_params', 'pvar', 'param_norm_sd', 'loss', 'AIC', 'BIC', 'equation']
        for extra_col in extra_cols:
            df_dict[extra_col] = []

        # Fill df row by row.
        for utility_setting_variety in utility_setting_varieties:
            utility_setting_key = str(gnrl.convert_utility_settings(utility_settings=utility_setting_variety, into=tuple)) 

            ic_res = all_ic_results['ic_results'].get(utility_setting_key)
            uv = all_ic_results['utility_varieties'].get(utility_setting_key)
            if ic_res is None or uv is None:
                print(f"Missing Utility Option Variety: {utility_setting_key}.")
                continue

            df_dict['idx'].append(ic_res['idx'])    
            # Add the n data points fields.
            df_dict['n_data'].append(ic_res['n_data'])

            # Add each boolean setting to the row.
            for setting_name, setting_val in uv.items():
                df_dict[setting_name].append(setting_val)

            # Add the IC fields.
            df_dict['k_params'].append(ic_res['k_params'])
            df_dict['pvar'].append(ic_res['pvar'])
            df_dict['param_norm_sd'].append(ic_res.get('param_norm_sd'))
            df_dict['loss'].append(ic_res['loss'])
            df_dict['AIC'].append(ic_res['AIC'])
            df_dict['BIC'].append(ic_res['BIC'])
            df_dict['equation'].append(
                build_utility_equation(utility_settings=utility_setting_variety)
            )



        df = ic_results_df(df_dict=df_dict)

        model_comparison_df(df=df)

        # Check the scree slope if sum_delta_min is below epsilon
        if iter_idx > 1:
            if sum_delta_minimum_loss_this_iter < robustness_epsilon:
                consecutive_small_improvements += 1
            else:
                consecutive_small_improvements = 0

            if consecutive_small_improvements >= 2:
                early_stop_statement = (
                    f"\nEarly stopping after iteration {iter_idx} because "
                    f"sum of ΔMinLoss < {robustness_epsilon} twice in a row."
                )
                ic_terminal_printouts.append(early_stop_statement)
                print(early_stop_statement)
                break
        # End iteration

    try:
        "Save terminal outputs"
        term_print_output_path = os.path.join(file_paths_this['bic_aic'], 'ic_terminal_printouts.txt')
        os.makedirs(file_paths_this['bic_aic'], exist_ok=True)
        "Open file in append mode to add new content without erasing the old"
        with open(term_print_output_path, 'a', encoding='utf-8') as file:
            for line in ic_terminal_printouts:
                file.write(line + '\n')
        "Empty list to preven saving duplicate information."
        ic_terminal_printouts.clear()
    except (PermissionError, OSError):
        "Pass if I have the file open."
        pass

    if not general_settings.get('write_mode') in ('resume', 'overwrite'): 
        # 1) How many iterations did the *long* run complete?
        max_iters_done = min(len(ic.get('lvec', []))
                             for ic in ic_results_dict.values())

        # 2) Re-compute the three per-iteration vectors directly
        sum_delta_minimum_loss_by_iter = []
        rank_change_by_iter            = []
        rank_change_median_by_iter     = []
        prev_min_by_model = {}      # track previous min loss per model
        prev_ranks        = {}      # track previous BIC ranks

        for iter in range(max_iters_done):
            # (A) Δ-min-loss
            sum_delta_this_it = 0.0
            for mk, ic in ic_results_dict.items():
                lvec = ic.get('lvec', [])
                if iter >= len(lvec):
                    continue            # model hadn’t reached that iter
                cur_min = min(lvec[:iter+1])
                prev_min = prev_min_by_model.get(mk, cur_min)
                sum_delta_this_it += abs(cur_min - prev_min)
                prev_min_by_model[mk] = cur_min
            sum_delta_minimum_loss_by_iter.append(sum_delta_this_it)

            # (B) BIC rank changes
            rows = []
            for mk, ic in ic_results_dict.items():
                lvec = ic.get('lvec', [])
                if iter >= len(lvec):
                    continue
                cur_min = min(lvec[:iter+1])
                k_params = ic['k_params']
                n_data   = ic['n_data']
                bic      = k_params*np.log(n_data) + 2*cur_min if n_data else np.nan
                rows.append((mk, bic))

            # rank them
            rows = [row for row in rows if not np.isnan(row[1])]
            rows.sort(key=lambda x: x[1])
            cur_ranks = {mk: row for row, (mk, _) in enumerate(rows)}

            diffs = [abs(cur_ranks[mk] - prev_ranks.get(mk, cur_ranks[mk]))
                     for mk in cur_ranks]
            rank_change_by_iter.append(sum(diffs))
            rank_change_median_by_iter.append(np.median(diffs) if diffs else 0.0)
            prev_ranks = cur_ranks

        # 3) overwrite the counters so they reflect reality
        iter_idx = max_iters_done

    ic_correlations(df=df)

    # Return the DataFrame and the dictionary of all results.
    return df, all_ic_results


def plot_ic_robustness_analysis(general_settings: Dict[str, Any], file_paths: Dict[str, str], fig_lay: Dict[str, Any]) -> go.Figure:
    """
    Loads a JSON containing 'robustness_analysis_data', then plots two line charts:
      (1) sum_delta_minimum_loss_by_iter,
      (2) rank_change_by_iter (and optionally rank_change_median_by_iter).
    """
    experiment_num = general_settings.get('experiment_num')
    # Path to save the final results.
    all_ic_results_file_path = os.path.join(
        file_paths["bic_aic"], 
        f"All_Utility_Forms_IC_Analysis_Experiment{experiment_num}.json"
    )

    if os.path.exists(all_ic_results_file_path):
        with open(all_ic_results_file_path, "r", encoding="utf-8") as f:
            all_ic_results = json.load(f)

    rab_data = all_ic_results.get("robustness_analysis_data", {})
    n_iters = all_ic_results.get("n_iterations_completed", 1)

    sum_delta_loss = rab_data.get("sum_delta_minimum_loss_by_iter", [])
    rank_changes   = rab_data.get("rank_change_by_iter", [])
    rank_med       = rab_data.get("rank_change_median_by_iter", [])

    # Build a 1x2 subplots figure or 2 separate figures
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Sum of Δ Min Loss Per Iteration",
            "Sum of Rank Changes Per Iteration"
        ]
    )

    # X array from 1..n_iters
    x_iter = list(range(1, len(sum_delta_loss)+1))
    line_width = 8
    # x_iter = 9
    # (A) sum_delta_minimum_loss_by_iter
    trace_sum_loss = go.Scatter(
        x=x_iter[1:],
        y=sum_delta_loss[1:],
        mode="lines+markers",
        name="Δ Min Loss",
        marker=dict(size=fig_lay.get("markersize", 10)+2, color='hsla(115, 65%, 40%, 1.0)'),
        line=dict(width=line_width, color='hsla(155, 65%, 20%, 1.0)')
    )
    fig.add_trace(trace_sum_loss, row=1, col=1)

    # (B) rank_change_by_iter
    x_iter2 = list(range(1, len(rank_changes)+1))
    trace_rank = go.Scatter(
        x=x_iter2[1:],
        y=rank_changes[1:],
        mode="lines+markers",
        name="Sum of Rank Changes",
        marker=dict(size=fig_lay.get("markersize", 10)+2, color='hsla(200, 65%, 40%, 1.0)'),
        line=dict(width=line_width, dash="solid", color='hsla(200, 65%, 20%, 1.0)')
    )
    fig.add_trace(trace_rank, row=1, col=2)

    # If you want median rank changes as well, you can add them in the same second panel
    rank_med = False #HACK
    if rank_med:
        trace_median = go.Scatter(
            x=x_iter2,
            y=rank_med,
            mode="lines+markers",
            name="Median Rank Change",
            marker=dict(size=fig_lay.get("markersize", 10)+2, color='hsla(285, 65%, 40%, 1.0)'),
            line=dict(width=line_width, dash="dot", color='hsla(285, 65%, 20%, 1.0)')
        )
        fig.add_trace(trace_median, row=1, col=2)

    tickvals_x = list(np.linspace(2, x_iter2[-1], len(x_iter2) - 1))
    ticktext_x = tickvals_x
    epsilon_x = 5e-02

    fig.update_annotations(font_size=24)
    fig.update_layout(
        template=fig_lay.get("template", "plotly_dark"),
        title="Robustness of IC Results",
        titlefont_size=fig_lay['titlefont_size'] + 6,
        margin=dict(l=150, r=100, t=150, b=120),
        title_x=0.5, title_y= 0.98,
        xaxis=dict(
            title="Analysis Iteration",
            **fig_lay.get("xaxis",{}), 
            tickvals=tickvals_x,
            ticktext=ticktext_x,
            range=[2 - epsilon_x, x_iter2[-1] + epsilon_x]
        ),
        xaxis2=dict(
            title="Analysis Iteration",
            **fig_lay.get("xaxis",{}), 
            tickvals=tickvals_x,
            ticktext=ticktext_x,
            range=[2 - epsilon_x, x_iter2[-1] + epsilon_x]
        ),
        yaxis=dict(
            title="Sum of Δ Minimum Loss",
            **fig_lay.get("yaxis",{})
        ),
        yaxis2=dict(
            title="Sum (or Median) Rank Change" if rank_med else "Sum of Rank Changes",
            **fig_lay.get("yaxis",{})
        ),
        font=fig_lay.get("font", {}),
        hoverlabel=fig_lay.get("hoverlabel", {}),
        legend=dict(x=0.5, y=-0.2, xanchor="center", orientation="h")
    )

    if general_settings.get('export_fig'):
        out_file_name: str = "robustness_analysis.html"
        out_path = os.path.join(file_paths['visuals'], out_file_name)
        print(f"Saved robustness figure to {out_path}.")
        fig.write_html(out_path)
        
    return fig


def plot_ic_scores_delta_bic(fig_lay: dict, file_paths: dict, general_settings: dict, include_dropdown: bool = True) -> go.Figure:
    """
    Creates a Plotly scatterplot of ΔBIC scores for all utility-model configurations,
    sorted from lowest (best) to highest. By default, a single trace uses a continuous
    color scale based on the number of parameters (𝑘). When a dropdown menu is included
    (include_dropdown=True), users can toggle coloring by each relevant Boolean utility
    option, revealing two traces (True/False) with distinct legend entries.

    • fig_lay: Dict[str, Any]
      Layout preferences (template, font, axis styles, etc.). Mimics your typical
      approach for consistent aesthetics.

    • file_paths: Dict[str, str]
      File paths. Must include:
        └─ file_paths["bic_aic"]
      This function automatically loads a CSV named:
        All_Utility_Forms_IC_Analysis_Experiment{experiment_num}.csv
      from that directory, where experiment_num is retrieved from general_settings.

    • general_settings: Dict[str, Any]
      Must contain 'experiment_num' to identify which CSV file to load. The CSV file
      is expected to have columns: ΔBIC, k_params, equation, n_data, and zero or more
      Boolean columns for specific utility options.

    • utility_settings: Dict[str, bool]
      A dictionary of possible Boolean flags (e.g. 'reference_dependent_utility').
      This is used to build readable labels for the dropdown if include_dropdown=True.

    • include_dropdown: bool
      If True, adds an interactive dropdown menu that toggles between:
        1) A single color-scale trace for 𝑘 parameters (the default).
        2) Pairs of True/False traces for each recognized Boolean column.
      If False, only the single color-scale trace is shown, and all Boolean traces
      remain invisible.

    Returns:
    -------
    • fig: go.Figure
      The Plotly figure object. The function writes an HTML file named 'ic_scores_scatter.html'
      to the file_paths["bic_aic"] directory. The x-axis is the model rank (1 = best),
      and the y-axis is ΔBIC relative to the best model (lower is better).

    Notes:
    ------
    1) Dots are sized ~1.8x larger than your default and include a subtle outline.
    2) When coloring by 𝑘, a colorbar labeled “𝑘” is displayed on the right. Boolean columns,
       if toggled, appear as separate True/False scatter traces with a horizontal legend
       placed below the chart. Each Boolean option name is converted into a more readable
       label (e.g., “Ref-Dependent Utility” instead of “reference_dependent_utility”).
    3) The figure title includes 𝑛 (the maximum n_data from the CSV) and references ΔBIC.
    4) Hover text shows each model’s rank, ΔBIC, number of parameters, and its utility equation.
    """

    # 1) Determine which CSV to load
    experiment_num = general_settings.get("experiment_num", 3)
    csv_file = os.path.join(
        file_paths["bic_aic"],
        f"All_Utility_Forms_IC_Analysis_Experiment{experiment_num}.csv"
    )
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Could not locate file: {csv_file}")

    # 2) Load DataFrame, check columns
    df = pd.read_csv(csv_file)
    required_cols = ["ΔBIC", "k_params", "equation", "n_data"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_file}")

    # 3) Sort by ΔBIC ascending and compute model rank
    df.sort_values(by="ΔBIC", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["model_rank"] = df.index + 1  # rank from 1..N

    # We'll assume all rows share the same n_data or we pick the max
    n_data = int(df["n_data"].max())

    # 4) Identify boolean columns
    bool_cols = []
    for col in df.columns:
        if df[col].dtype == bool or df[col].dtype == "bool":
            bool_cols.append(col)

    # 5) Create a user-friendly label map for these columns
    #    (Customize as needed)
    bool_label_map = {
        'conditional_welfare_mode':       "Conditional Welfare Mode",
        'reference_dependent_altruism':   "Ref-Dependent Altruism",
        'min_max_rawlsian_leontief':      "Min-Max (Rawls/Leontief)",
        'use_exponential_parameters':     "Use Exponential Params",
        'apply_exponents_to_payoffs':     "Apply Exponents to Payoffs",
        'single_exponential_parameter':   "Single Exponential Param",
        'single_payoffs_not_differences': "Single Payoffs, Not Differences",
        'payoff_ratios_not_differences':  "Payoff Ratios, Not Differences",
        'reference_dependent_utility':    "Ref-Dependent Utility",
        'use_negativity_parameters':      "Use Negativity Params",
        'negativity_social_comparison':   "Negativity in Social Comparison",
        'fix_self_interest_parameter':    "Fix Self-Interest Param",
        'include_social_comparison':      "Include Social Comparison",
        'include_altruism_term':          "Include Altruism Term"
    }

    # 6) Build a single color-scale trace for k_params
    k_min = df["k_params"].min()
    k_max = df["k_params"].max()

    # Marker size (scaled ~1.8x)
    default_marker_size = int(fig_lay.get("markersize", 12) * 2)

    trace_kparams = go.Scatter(
        x=df["model_rank"],
        y=df["ΔBIC"],
        mode="markers",
        name="Models by 𝑘",
        visible=True,  # default
        showlegend=False,  # no legend for the continuous color scale
        hovertemplate=(
            "Rank: %{x}; 𝑘 Params: %{customdata[0]}; ΔBIC: %{y:.3f}<br>"
            "Equation: %{customdata[1]}<extra></extra>"
        ),
        customdata=df[["k_params", "equation"]],
        marker=dict(
            size=default_marker_size,
            color=df["k_params"],
            colorscale="Viridis",
            cmin=k_min,
            cmax=k_max,
            showscale=True,  # Show colorbar
            colorbar=dict(
                title="𝑘",  # fancy k
                x=1.02
            ),
            line=dict(width=1.5, color="hsla(0, 50%, 0%, 0.5)")
        )
    )

    data_traces = [trace_kparams]  # We'll add Boolean-based traces below

    # 7) For each boolean col, create 2 separate scatter traces: True & False
    #    These traces are invisible by default; if the user toggles them in the dropdown,
    #    we'll set them visible and hide the k_params trace.
    hue_true  = "hsla(0, 80%, 40%, 7.0)"     # red
    hue_false = "hsla(180, 80%, 40%, 7.0)"   # cyan
    bool_trace_map = {}
    current_trace_index = 1

    for bcol in bool_cols:
        label_ = bool_label_map.get(bcol, bcol)

        df_true  = df[df[bcol] == True ]
        df_false = df[df[bcol] == False]

        tr_true = go.Scatter(
            x=df_true["model_rank"],
            y=df_true["ΔBIC"],
            mode="markers",
            name=f"{label_} = True",
            visible=False,
            legendgroup=bcol,
            showlegend=True,
            hovertemplate=(
                "Rank: %{x}; 𝑘 Params: %{customdata[0]}; ΔBIC: %{y:.3f}<br>"
                "Equation: %{customdata[1]}<extra></extra>"
            ),
            customdata=df_true[["k_params", "equation"]],
            marker=dict(
                size=default_marker_size, 
                opacity=0.7, color=hue_true, showscale=False,
                line=dict(width=1.5, color="hsla(0, 50%, 0%, 0.5)")
            )
        )
        tr_false = go.Scatter(
            x=df_false["model_rank"],
            y=df_false["ΔBIC"],
            mode="markers",
            name=f"{label_} = False",
            visible=False,
            legendgroup=bcol,
            showlegend=True,
            hovertemplate=(
                "Rank: %{x}; 𝑘 Params: %{customdata[0]}; ΔBIC: %{y:.3f}<br>"
                "Equation: %{customdata[1]}<extra></extra>"
            ),
            customdata=df_false[["k_params", "equation"]],
            marker=dict(
                size=default_marker_size,
                opacity=0.7, color=hue_false, showscale=False,
                line=dict(width=1.5, color="hsla(0, 50%, 0%, 0.5)")
            )
        )

        data_traces.append(tr_true)
        data_traces.append(tr_false)

        bool_trace_map[bcol] = (current_trace_index, current_trace_index + 1)
        current_trace_index += 2

    fig = go.Figure(data=data_traces)

    # 8) Overall layout and styling
    # Use fancy letters for 𝑛 and Δ
    fig.update_layout(
        template=fig_lay.get("template", "plotly_dark"),
        title=f"IC Scores (ΔBIC) for All Utility Functional Forms; 𝑛 = {n_data} Data Points",
        titlefont_size=fig_lay['titlefont_size'],
        font=fig_lay.get("font", {}),
        hoverlabel=fig_lay.get("hoverlabel", {}),
        margin=dict(l=180, r=150, t=150, b=120),
        title_x=0.5,
        xaxis=dict(
            title="Model Rank (1 = Best)",
            **fig_lay.get("xaxis", {})
        ),
        yaxis=dict(
            title="ΔBIC (Difference from Best Model)",
            **fig_lay.get("yaxis", {})
        ),
        legend=dict(
            orientation="h", x=0.0, y=-0.15,
            font=dict(size=fig_lay.get("font", {}).get("size", 16))
        )
    )

    # 9) If no dropdown is wanted, hide all Boolean traces
    if not include_dropdown:
        for idx in range(1, len(data_traces)):
            fig.data[idx].visible = False

        # The single trace for k_params is visible => we rely on the color scale
        out_path = os.path.join(file_paths["bic_aic"], "ic_scores_scatter.html")
        fig.write_html(out_path)
        print(f"Saved scatter plot to '{out_path}' [No Dropdown Mode].")
        return fig

    # 10) Build a dropdown to toggle coloring
    n_traces_total = len(data_traces)
    def all_invisible():
        return [False] * n_traces_total

    # Option A: "Color by k_params"
    kparams_vis = all_invisible()
    kparams_vis[0] = True  # the first trace is the continuous color-scale
    # For booleans, we do not show them => false

    buttons = []
    # First button => color by k_params
    buttons.append(dict(
        label="Color by 𝑘",
        method="update",
        args=[
            {"visible": kparams_vis},  # sets the visible array
            {"title": f"IC Scores (ΔBIC) for All Utility Functional Forms; 𝑛 = {n_data} Data Points (Colored by 𝑘)"}
        ]
    ))

    # Additional buttons => each boolean col
    for bcol in bool_cols:
        label_ = bool_label_map.get(bcol, bcol)
        # The pair of traces for this col
        idx_true, idx_false = bool_trace_map[bcol]
        vis_arr = all_invisible()
        vis_arr[idx_true] = True
        vis_arr[idx_false] = True

        buttons.append(dict(
            label=f"Color by {label_}",
            method="update",
            args=[
                {"visible": vis_arr},
                {"title": f"IC Scores (ΔBIC) for All Utility Functional Forms; 𝑛 = {n_data} Data Points (Colored by {label_})"}
            ]
        ))

    fig.update_layout(
        updatemenus=[dict(
            type="dropdown",
            showactive=True,
            direction="down",
            x=0.0, y=1.06,
            xanchor="left",
            yanchor="top",
            # reduce the layout size for the menu
            pad=dict(r=10, t=10),
            buttons=buttons
        )]
    )

    # 11) Write out HTML and return figure
    out_path = os.path.join(file_paths["visuals"], "ic_scores_scatter.html")
    print(f"Saved scatter plot to '{out_path}' [Dropdown Mode].")
    fig.write_html(out_path)
    
    return fig


def utility_setting_contribution_analysis(*, general_settings: dict, file_paths: dict, utility_settings_universe: dict[str, bool], score_col: str = "BIC", 
                                          use_edge_types: tuple[str, ...] = ("sibling", "parent_child"), include_non_network_toggles: bool = True, 
                                          export_csv: bool = True, out_dirname: str = "pairwise_edge_analysis") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Nesting-aware pairwise analysis of feature contributions.

    Purpose:
    Build ΔScore (e.g., ΔBIC) *along certified edges of the model nesting network*.
    The function returns:
      • edge_level_df : one row per compared edge with orientation by 'flip_name'
      • summary_by_flip : mean/median/N for each flip_name, split by edge_type
      • payoff_paths_summary : focused sibling-only summary for core payoff choices
        (single vs differences vs ratios ± reference; and exponent-placement when relevant).

    Arguments:
        • general_settings : dict
            Must include 'experiment_num'.
        • file_paths : dict
            Uses file_paths["bic_aic"] for the IC CSV and writes outputs under
            file_paths["player_fits"]/simulation_results/out_dirname (created if missing).
        • utility_settings_universe : dict[str,bool]
            The canonical set (keys only are used) of Boolean settings that define models.
            Example keys:
            ['conditional_welfare_mode','reference_dependent_altruism','min_max_rawlsian_leontief',
                'use_exponential_parameters','apply_exponents_to_payoffs','single_exponential_parameter',
                'single_payoffs_not_differences','payoff_ratios_not_differences','reference_dependent_utility',
                'use_negativity_parameters','negativity_social_comparison','fix_self_interest_parameter',
                'include_social_comparison','include_altruism_term']
        • score_col : str
            Which score to difference (e.g., "BIC" or "AIC").
        • use_edge_types : ('sibling','parent_child',...)
            Which edge types from the nesting network to include.
        • include_non_network_toggles : bool
            If True, also build "same-k, single-flip" pairs for flips that the
            nesting network marks as having no edge (e.g., Conditional Welfare; Rawls/Leontief).
            These are labeled edge_type='non_network_same_k'.
        • export_csv : bool
            If True, writes three CSVs (edge-level, summary-by-flip, payoff-paths summary).
        • out_dirname : str
            Subfolder name for outputs under player_fits/simulation_results.

    Returns:
        • (edge_level_df, summary_by_flip, payoff_paths_summary)

    Notes
    -----
    • ΔScore orientation is always Score(True) - Score(False) for the 'flip_name'.
      Negative => turning the feature ON improves fit.
    • We explicitly tag each row with 'edge_type' in {'sibling','parent_child','non_network_same_k'}.
    • The payoff-paths panel is sibling-only and reports:
        - Single vs Differences
        - Single vs Ratios
        - Differences vs Ratios
        - RefDiff vs Diff, RefRatio vs Ratio (if present)
        - Apply-exponents-directly vs Apply-after (conditional on exponents and on not-single)
    """
    experiment_num = general_settings.get('experiment_num', 0)

    # ---------- Load IC table ----------
    ic_csv = os.path.join(
        file_paths["bic_aic"],
        f"All_Utility_Forms_IC_Analysis_Experiment{experiment_num}.csv"
    )
    df_models = pd.read_csv(ic_csv, encoding="utf-8", engine="python")

    # Canonical set & order of features we will use everywhere
    feature_cols = [c for c in utility_settings_universe.keys() if c in df_models.columns]
    keep_cols = ["k_params", score_col] + feature_cols

    missing_cols = [c for c in keep_cols if c not in df_models.columns]
    if missing_cols:
        raise ValueError(f"IC CSV is missing required columns: {missing_cols}")

    # Coerce booleans
    for c in feature_cols:
        if df_models[c].dtype != bool:
            df_models[c] = df_models[c].astype(bool)

    # Build a canonical signature for each row in IC table
    def row_signature(row) -> tuple:
        # tuple of booleans in the SAME order as feature_cols
        return tuple(bool(row[c]) for c in feature_cols)

    df_models = df_models.reset_index(drop=False).rename(columns={"index": "model_id"})
    df_models = df_models[["model_id"] + keep_cols].copy()
    df_models["__signature__"] = df_models.apply(row_signature, axis=1)

    # Map signature -> model_id (ensure uniqueness)
    sig_to_model_id: dict[tuple, int] = {}
    dup_sigs: dict[tuple, list[int]] = {}
    for mid, sig in zip(df_models["model_id"], df_models["__signature__"]):
        if sig in sig_to_model_id:
            # Track duplicates to catch structural problems (shouldn’t happen for 476 unique forms)
            dup_sigs.setdefault(sig, []).append(int(mid))
        else:
            sig_to_model_id[sig] = int(mid)

    if dup_sigs:
        # Not necessarily fatal, but warn prominently.
        # We keep the first occurrence; duplicates would inflate ambiguity.
        print(f"[WARN] Duplicate utility signatures in IC table for {len(dup_sigs)} signature(s). "
              f"Proceeding with the first occurrence for each.")

    # ---------- Get nesting graph ----------
    graph = model_nesting_adjacency_matrices(
        general_settings=general_settings,
        utility_settings=utility_settings_universe,
        file_paths=file_paths
    )

    parent_of  = graph['adjacency_lists']['parent_of']
    sibling_of = graph['adjacency_lists']['sibling_of']
    settings_list = graph['settings']  # list[dict[str,bool]]

    # Convert a settings dict from the graph into the same canonical signature
    def settings_signature(sdict: dict) -> tuple:
        # Some graph settings may be ints; normalize to bool; default missing -> False
        return tuple(bool(sdict.get(c, False)) for c in feature_cols)

    # Build mapping from graph index -> model_id in IC table
    index_to_model_id: list[int] = []
    missing_indices: list[int] = []

    for i, sdict in enumerate(settings_list):
        sig = settings_signature(sdict)
        mid = sig_to_model_id.get(sig, None)
        if mid is None:
            missing_indices.append(i)
            index_to_model_id.append(None)
        else:
            index_to_model_id.append(mid)

    if missing_indices:
        # Provide concrete, actionable info
        examples = []
        for i in missing_indices[:10]:
            sig = settings_signature(settings_list[i])
            examples.append(
                f"graph_idx={i}, signature={sig}, settings="
                f"{ {k: bool(settings_list[i].get(k, False)) for k in feature_cols} }"
            )
        msg = ("The following models from the nesting graph were not found in the IC table "
               "(signature mismatch). This typically means the IC CSV and the nesting generator "
               "were built over different universes or column orderings:\n  - " + "\n  - ".join(examples))
        raise RuntimeError(msg)

    # The only flips that *define* relationships (mirror classify_pair_relation)
    SIBLING_FLIPS = {
        'single_payoffs_not_differences',
        'payoff_ratios_not_differences',
        'reference_dependent_utility',
        'reference_dependent_altruism',
        'apply_exponents_to_payoffs',
    }
    PARENT_CHILD_FLIPS = {
        'use_exponential_parameters',
        'single_exponential_parameter',
        'use_negativity_parameters',
        'negativity_social_comparison',
        'fix_self_interest_parameter',
        'include_social_comparison',
        'include_altruism_term',
    }

    # Helper: identify the single effective flip between two settings
    def effective_flip(a: dict, b: dict, allowed: set[str]) -> str | None:
        flipped = [k for k in allowed if bool(a.get(k, False)) != bool(b.get(k, False))]
        if len(flipped) == 1:
            return flipped[0]
        return None

    # ---- Build edge tuples over GRAPH INDICES with their flip labels ----
    sib_edges: list[tuple[int,int,str]] = []
    for i, nbrs in enumerate(sibling_of):
        # if settings_list[i]['payoff_ratios_not_differences']:
        #     continue
        # if settings_list[i]['min_max_rawlsian_leontief']:
        #     continue
        # if settings_list[i]['conditional_welfare_mode']:
        #     continue
        for j in nbrs:
            if j <= i:
                continue  # proper de-dup
            flip = effective_flip(settings_list[i], settings_list[j], SIBLING_FLIPS)
            if flip:
                sib_edges.append((i, j, flip))

    pc_edges: list[tuple[int,int,str]] = []
    for p, children in enumerate(parent_of):
        # if settings_list[p]['payoff_ratios_not_differences']:
        #     continue
        for c in children:
            flip = effective_flip(settings_list[p], settings_list[c], PARENT_CHILD_FLIPS)
            if flip:              
                pc_edges.append((p, c, flip))

    "Debugging code"
    check_interested_settings = False
    if check_interested_settings:
        setting_of_interest = 'reference_dependent_altruism'
        for sib_edge in sib_edges:
            sib_1, sib_2, flipped_setting = sib_edge
            if flipped_setting == setting_of_interest:
                sib_1_equation = graph['equations'][sib_1]
                sib_2_equation = graph['equations'][sib_2]
                print(sib_1_equation)
                print(sib_2_equation)
                print()

    # Quick sanity counts (should match your terminal counts)
    print("Sibling edges (graph):", len(sib_edges))
    print("Parent-child edges (graph):", len(pc_edges))

    n_flips_by_setting_siblings = {setting_key: 0 for setting_key in utility_settings_universe.keys()} 
    for sib_sib_flip in sib_edges: 
        n_flips_by_setting_siblings[sib_sib_flip[-1]] += 1 
    n_flips_by_setting_par_chi = {setting_key: 0 for setting_key in utility_settings_universe.keys()} 
    for par_chi_flip in pc_edges: 
        n_flips_by_setting_par_chi[par_chi_flip[-1]] += 1
    for setting_key in utility_settings_universe.keys():
        if n_flips_by_setting_siblings[setting_key] == 0:
            del n_flips_by_setting_siblings[setting_key]
        if n_flips_by_setting_par_chi[setting_key] == 0:
            del n_flips_by_setting_par_chi[setting_key]

    print("\n Sibling edges (kept):")
    pp.pprint(n_flips_by_setting_siblings)
    print("\n Parent-child edges (kept):")
    pp.pprint(n_flips_by_setting_par_chi)

    # ---- Convert graph indices → true model_ids via the signature map ----
    rows = []
    if "sibling" in use_edge_types:
        for i, j, flip in sib_edges:
            rows.append({
                "edge_type": "sibling",
                "flip_name": flip,
                "a_model_id": index_to_model_id[i],
                "b_model_id": index_to_model_id[j]
            })
    if "parent_child" in use_edge_types:
        for p, c, flip in pc_edges:
            rows.append({
                "edge_type": "parent_child",
                "flip_name": flip,
                "a_model_id": index_to_model_id[p],
                "b_model_id": index_to_model_id[c]
            })

    df_edges = pd.DataFrame(rows)
    if df_edges.empty:
        raise RuntimeError("No edges found after mapping graph indices to IC rows.")

    # ---------- Join IC scores + features for both endpoints ----------
    left = df_edges.merge(df_models.add_prefix("a_"), left_on="a_model_id", right_on="a_model_id")
    both = left.merge(df_models.add_prefix("b_"), left_on="b_model_id", right_on="b_model_id")

    # Keep only rows where *exactly one* feature differs (safety; should be true for nest edges)
    def _diff_feature(row) -> str | None:
        diffs = [f for f in feature_cols if bool(row[f"a_{f}"]) != bool(row[f"b_{f}"])]
        return diffs[0] if len(diffs) == 1 else None

    both["differing_feature"] = both.apply(_diff_feature, axis=1)
    # If this drops many rows, it indicates remaining ID/signature mismatches
    before = len(both)
    both = both[~both["differing_feature"].isna()].copy()
    after = len(both)
    if after < before:
        print(f"[info] dropped {before - after} rows where endpoints differed on ≠1 feature "
              f"(this is expected to be small if mapping was correct).")

    # ---------- Orient ΔScore as Score(True) - Score(False) for that feature ----------
    def _orient(row):
        f = row["differing_feature"]
        a_true = bool(row[f"a_{f}"])
        b_true = bool(row[f"b_{f}"])
        if a_true and not b_true:
            m_true_id,  score_true  = row["a_model_id"], row[f"a_{score_col}"]
            m_false_id, score_false = row["b_model_id"], row[f"b_{score_col}"]
        elif b_true and not a_true:
            m_true_id,  score_true  = row["b_model_id"], row[f"b_{score_col}"]
            m_false_id, score_false = row["a_model_id"], row[f"a_{score_col}"]
        else:
            return pd.Series({"m_true":np.nan,"m_false":np.nan,"delta":np.nan})
        return pd.Series({"m_true":m_true_id, "m_false":m_false_id, "delta":float(score_true) - float(score_false)})

    both[["m_true","m_false","delta"]] = both.apply(_orient, axis=1)

    # Sanity: flip label from graph should match the single differing feature
    both["flip_ok"] = (both["flip_name"] == both["differing_feature"])
    if not bool(both["flip_ok"].all()):
        n_bad = int((~both["flip_ok"]).sum())
        print(f"[warn] {n_bad} edges where graph flip_name != detected differing_feature "
              f"(keeping rows but flagging).")

    # ---------- Edge-level tidy output ----------
    edge_level_cols = [
        "edge_type","flip_name","differing_feature","delta","m_true","m_false",
        "a_model_id","b_model_id","a_k_params","b_k_params"
    ] + [f"a_{c}" for c in feature_cols] + [f"b_{c}" for c in feature_cols]
    edge_level_df = both[edge_level_cols].copy()

    # ---------- Optional: “non‑network” toggles (e.g., Conditional Welfare; Rawls/Leontief) ----------
    non_network_rows = []
    if include_non_network_toggles:
        seen = set(edge_level_df["differing_feature"].unique())
        candidates = ["conditional_welfare_mode", "min_max_rawlsian_leontief"]
        none_flips = [f for f in candidates if f in feature_cols and f not in seen]

        # Per-toggle rule: Conditional Welfare can cross k (BIC already penalizes complexity); Min–Max stays same‑k.
        require_same_k_for = {
            "conditional_welfare_mode": False,
            "min_max_rawlsian_leontief": True,
        }

        tmp = df_models.copy()
        for f in none_flips:
            # signature excluding f
            other_cols = [c for c in feature_cols if c != f]
            tmp["signature_excl_f"] = tmp[other_cols].apply(lambda r: tuple(bool(v) for v in r.tolist()), axis=1)

            group_keys = ["signature_excl_f"] + (["k_params"] if require_same_k_for.get(f, True) else [])
            for _, g in tmp.groupby(group_keys, dropna=False):
                g0 = g[g[f] == False]
                g1 = g[g[f] == True]
                if g0.empty or g1.empty:
                    continue
                for _, r0 in g0.iterrows():
                    for _, r1 in g1.iterrows():
                        non_network_rows.append({
                            "edge_type": "non_network" if not require_same_k_for.get(f, True) else "non_network_same_k",
                            "flip_name": f,
                            "differing_feature": f,
                            "delta": float(r1[score_col]) - float(r0[score_col]),  # Score(True) - Score(False)
                            "m_true": int(r1["model_id"]),
                            "m_false": int(r0["model_id"]),
                            "a_model_id": int(r0["model_id"]),
                            "b_model_id": int(r1["model_id"]),
                            "a_k_params": int(r0["k_params"]),
                            "b_k_params": int(r1["k_params"]),
                            **{f"a_{c}": bool(r0[c]) for c in feature_cols},
                            **{f"b_{c}": bool(r1[c]) for c in feature_cols},
                        })



    if non_network_rows:
        edge_level_df = pd.concat([edge_level_df, pd.DataFrame(non_network_rows)], ignore_index=True)

    # ---------- Summary by flip & edge type ----------
    def _summarize(df):
        recs = []
        for (et, f), g in df.groupby(["edge_type","flip_name"], dropna=False):
            deltas = g["delta"].dropna().to_numpy()
            recs.append({
                "edge_type": et,
                "flip_name": f,
                "n_edges": int(deltas.size),
                "mean_delta": float(np.mean(deltas)) if deltas.size else np.nan,
                "median_delta": float(np.median(deltas)) if deltas.size else np.nan
            })
        return pd.DataFrame(recs)

    summary_by_flip = _summarize(edge_level_df).sort_values(
        ["edge_type", "mean_delta"], ascending=[True, True]
    )
    # summary_by_flip.sort_values(by="mean_delta", ascending=True)

    # ---------- Payoff‑paths sibling‑only panel ----------
    sib = edge_level_df[edge_level_df["edge_type"]=="sibling"].copy()

    def _quick_panel(name, mask):
        g = sib[mask]
        return {
            "comparison": name,
            "n_edges": int(len(g)),
            "mean_Δ": float(g["delta"].mean()) if len(g) else np.nan,
            "median_Δ": float(g["delta"].median()) if len(g) else np.nan
        }

    m_single_vs_diff = (sib["differing_feature"] == "single_payoffs_not_differences")
    m_ratio_vs_diff  = (sib["differing_feature"] == "payoff_ratios_not_differences")
    m_ref_toggle = (sib["differing_feature"] == "reference_dependent_utility") & \
                   (~sib["a_single_payoffs_not_differences"]) & (~sib["b_single_payoffs_not_differences"])
    has_exp = (sib["a_use_exponential_parameters"] & sib["b_use_exponential_parameters"])
    not_single = (~sib["a_single_payoffs_not_differences"] & ~sib["b_single_payoffs_not_differences"])
    m_place = (sib["differing_feature"] == "apply_exponents_to_payoffs") & has_exp & not_single

    payoff_rows = [
        _quick_panel("Single vs Differences (Δ = BIC[Single]-BIC[Diff])", m_single_vs_diff),
        _quick_panel("Ratios vs Differences (Δ = BIC[Ratios]-BIC[Diff])",  m_ratio_vs_diff),
        _quick_panel("Reference on vs off (non-single) (Δ = BIC[Ref]-BIC[NoRef])", m_ref_toggle),
        _quick_panel("Exponent placement: pre-payoff vs post-transform (Δ = BIC[Pre]-BIC[Post])", m_place),
    ]
    payoff_paths_summary = pd.DataFrame(payoff_rows)

    # ---------- Write CSVs ----------
    out_root = os.path.join(file_paths["bic_aic"], out_dirname)
    os.makedirs(out_root, exist_ok=True)

    if export_csv:
        edge_csv     = os.path.join(out_root, f"edge_level_{score_col}.csv")
        summary_csv  = os.path.join(out_root, f"summary_by_flip_{score_col}.csv")
        payoff_csv   = os.path.join(out_root, f"payoff_paths_{score_col}.csv")
        edge_level_df.to_csv(edge_csv, index=False, encoding="utf-8-sig")
        summary_by_flip.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        payoff_paths_summary.to_csv(payoff_csv, index=False, encoding="utf-8-sig")
        print(f"\nWrote: {edge_csv}\n       {summary_csv}\n       {payoff_csv}")

    return edge_level_df, summary_by_flip, payoff_paths_summary


def extract_rankings_of_canonical_utility_functions(file_paths: FilePaths, rank_col: str = "BIC", print_: bool = True) -> pd.DataFrame:
    """
    Filter the IC table for canonical specifications and report their ranks.
    Returns a DataFrame with: label, n_matches, k_params, loss, AIC, BIC, global_rank, ΔBIC_to_best.
    If multiple rows match a label, the best (lowest rank_col) is kept.
    """
    CANONICAL_SPECS: dict[UtilitySettings] = {
        "Fehr–Schmidt (1999) inequity aversion": {
            "conditional_welfare_mode": False,
            "reference_dependent_altruism": False,
            "min_max_rawlsian_leontief": False,
            "use_exponential_parameters": False,
            "apply_exponents_to_payoffs": False,
            "single_exponential_parameter": True,
            "single_payoffs_not_differences": True,
            "payoff_ratios_not_differences": False,
            "reference_dependent_utility": False,
            "use_negativity_parameters": False,
            "negativity_social_comparison": True,
            "fix_self_interest_parameter": True,
            "include_social_comparison": True,
            "include_altruism_term": False
        },
        "Bolton–Ockenfels ERC (2000)": {
            "conditional_welfare_mode": False,
            "reference_dependent_altruism": False,
            "min_max_rawlsian_leontief": False,
            "use_exponential_parameters": False,
            "apply_exponents_to_payoffs": False,
            "single_exponential_parameter": True,
            "single_payoffs_not_differences": False,
            "payoff_ratios_not_differences": True,
            "reference_dependent_utility": False,
            "use_negativity_parameters": False,
            "negativity_social_comparison": False,
            "fix_self_interest_parameter": True,
            "include_social_comparison": True,
            "include_altruism_term": False
        },
        "Charness–Rabin (2002) conditional welfare": {
            "conditional_welfare_mode": True,
            "reference_dependent_altruism": False,
            "min_max_rawlsian_leontief": False,
            "use_exponential_parameters": False,
            "apply_exponents_to_payoffs": False,
            "single_exponential_parameter": True,
            "single_payoffs_not_differences": True,  
            "payoff_ratios_not_differences": False,
            "reference_dependent_utility": False,
            "use_negativity_parameters": False,       
            "negativity_social_comparison": False,
            "fix_self_interest_parameter": False,
            "include_social_comparison": False,
            "include_altruism_term": False
        },
        "Andreoni–Miller (2002) CES (warm glow)": {
            "conditional_welfare_mode": False,
            "reference_dependent_altruism": False,
            "min_max_rawlsian_leontief": False,
            "use_exponential_parameters": True,    
            "apply_exponents_to_payoffs": False,
            "single_exponential_parameter": True,
            "single_payoffs_not_differences": True,
            "payoff_ratios_not_differences": False,
            "reference_dependent_utility": False,
            "use_negativity_parameters": False,
            "negativity_social_comparison": False,
            "fix_self_interest_parameter": True,
            "include_social_comparison": False,
            "include_altruism_term": True
        },
        "Engelmann–Strobel (2004) maximin‑efficiency": {
            "conditional_welfare_mode": False,
            "reference_dependent_altruism": False,
            "min_max_rawlsian_leontief": True,
            "use_exponential_parameters": False,
            "apply_exponents_to_payoffs": False,
            "single_exponential_parameter": True,
            "single_payoffs_not_differences": False,  
            "payoff_ratios_not_differences": False,
            "reference_dependent_utility": False,
            "use_negativity_parameters": False,
            "negativity_social_comparison": False,
            "fix_self_interest_parameter": True,
            "include_social_comparison": False,
            "include_altruism_term": False
        },
        "Messick–McClintock (1968) SVO linear": {
            "conditional_welfare_mode": False,
            "reference_dependent_altruism": False,
            "min_max_rawlsian_leontief": False,
            "use_exponential_parameters": False,
            "apply_exponents_to_payoffs": False,
            "single_exponential_parameter": True,
            "single_payoffs_not_differences": True,
            "payoff_ratios_not_differences": False,
            "reference_dependent_utility": False,
            "use_negativity_parameters": False,
            "negativity_social_comparison": False,
            "fix_self_interest_parameter": True,
            "include_social_comparison": False,
            "include_altruism_term": True
        }
    }
    if print_:
        for function_name, settings in CANONICAL_SPECS.items():
            explanation = gnrl.is_valid_utility_settings(settings, provide_explanation=True)
            print(f"\n{function_name}:")
            print(build_utility_equation(settings))
            if explanation != "Success!":
                print(explanation) 
            for setting_key, setting_val in settings.items():
                setting_data = setting_key + " " * (30 - len(setting_key)) + f": {setting_val}"
                print(setting_data)
        for function_name, settings in CANONICAL_SPECS.items():
            print(f"\n{function_name}:")
            print(build_utility_equation(settings))
        print("")

    ic_csv_path = os.path.join(file_paths['bic_aic'], file_paths['file_names']['information_criterion'])
    df = pd.read_csv(ic_csv_path, encoding="utf-8", engine="python")
    # Cast bool columns robustly
    for col in CANONICAL_SPECS[next(iter(CANONICAL_SPECS))].keys():
        if col in df.columns and df[col].dtype != bool:
            df[col] = df[col].astype(bool)

    # Global best by BIC for ΔBIC
    global_best_bic = float(df["BIC"].min())
    df["global_rank"] = df[rank_col].rank(method="min", ascending=True).astype(int)

    rows = []
    for label, spec in CANONICAL_SPECS.items():
        missing = [k for k in spec.keys() if k not in df.columns]
        if missing:
            rows.append({"label": label, "error": f"Missing columns: {missing}"})
            continue
        cur = df.copy()
        for k, v in spec.items():
            if k in cur.columns:
                cur = cur[cur[k] == v]
        n_matches = len(cur)
        if n_matches == 0:
            rows.append({"label": label, "n_matches": 0, "note": "No exact match in IC table."})
            continue
        best = cur.sort_values(by=rank_col, ascending=True).iloc[0]
        rows.append({
            "label": label,
            "n_matches": n_matches,
            "k_params": int(best["k_params"]),
            "loss": float(best.get("loss", float("nan"))),
            "AIC": float(best.get("AIC", float("nan"))),
            "BIC": float(best.get("BIC", float("nan"))),
            "global_rank": int(best["global_rank"]),
            "ΔBIC_to_best": float(best["BIC"] - global_best_bic),
            "model_id": int(best.get("model_id", -1)),
        })
    out = pd.DataFrame(rows).sort_values(by=["BIC"], ascending=True, na_position="last")
    return out


"=========================================================================================="
"============================ Nesting Network and Verification ============================"
"=========================================================================================="

def run_child_parent_embedding_sanity_checks(general_settings: dict[str, Any], file_paths: dict[str, Any], param_bds: dict[str, tuple[float, float]], 
                                             utility_settings: UtilitySettings, player_role_to_fit: str = "predictor", fit_for_n_players: int | None = None,
                                             random_seed: int | None = 12345, numeric_tolerance: float = 1e-4, csv_file_name: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """
    Runs the child-vs-special-parent equality test across the entire model space.

    For each minimal (child, parent) pair:
        1) Sample a random child parameter vector within `param_bds`.
        2) Embed those child means into the parent's parameter space to create a special parent.
        3) For a subset (or all) participants:
            dyads_for_a_player → agent → loss_function_bayes → create_loss_report
            Sum NLL across dyads for `player_role_to_fit`, *for child and for parent*.
        4) Write a wide CSV with requested columns in file_paths['bic_aic'].

    Arguments:
        • general_settings: Your global settings dict. The following keys are read:
            - experiment_num
            - softmax_temperature
            - (others are forwarded to `agent` and loss functions as-is)
        • file_paths: Your file path mapping (must include 'bic_aic' and 'file_names' → 'information_criterion').
        • param_bds: The global ParameterBounds with all keys (means and _std).
        • ordered_flag_keys: The canonical order of your 13 utility settings
            (e.g., pass `list(utility_settings.keys())` from your current model).
        • player_role_to_fit: 'predictor' (default) or 'chooser'.
        • fit_for_n_players: int | None; Number of participants to evaluate (alphabetical order). None → all.
        • random_seed: Seed for reproducibility.
        • numeric_tolerance: Tolerance for |loss_parent - loss_child|.
        • csv_file_name: Optional override for CSV name. Default: "child_parent_embedding_sanity_checks.csv".
        • verbose: If True, prints progress summaries.

    Returns:
        • pd.DataFrame; The full table that was also written to CSV.
    """
    ordered_flag_keys = list(utility_settings.keys())

    def _sample_random_parameter_dict(param_keys: list[str],
                                    param_bounds: list[tuple[float, float]],
                                    rng: random.Random) -> dict[str, float]:
        """
        Draw a random parameter dictionary within the provided bounds.

        Arguments:
            • param_keys: list[str]; Ordered parameter names (means first, then _std if present).
            • param_bounds: list[tuple[float, float]]; Same length and order as param_keys.
            • rng: random.Random; PRNG instance for reproducibility.

        Returns:
            • dict[str, float]; {parameter_name: sampled_value}
        """
        parameter_dictionary: dict[str, float] = {}
        for parameter_key, (lower_bound, upper_bound) in zip(param_keys, param_bounds):
            # Keep _std comfortably > 0 within the given bounds
            if parameter_key.endswith("_std"):
                lower = max(lower_bound, 1e-3)
                parameter_dictionary[parameter_key] = rng.uniform(lower, upper_bound)
            else:
                parameter_dictionary[parameter_key] = rng.uniform(lower_bound, upper_bound)
        return parameter_dictionary

    def _embed_child_parameters_into_parent_means(child_parameter_dict: dict[str, float],
                                                changed_utility_setting: str,
                                                parent_param_keys: list[str]) -> dict[str, float]:
        """
        Deterministically embed a child's parameters into the parent's parameter space
        so the parent reproduces the child (i.e., the child is a special case of the parent).

        Mapping conventions (consistent with prior discussions):
            • use_exponential_parameters=True in parent:
                - If child has no γ’s: set all parent γ* to 1.0.
                - If child uses a single γ (γ1) and parent has multiple γ’s: tie all parent γ* to child's γ1.
            • single_exponential_parameter flip (tie ↔ untie):
                - If parent has multiple γ’s but child has γ1 only: copy γ1 to every parent γ*.
            • include_social_comparison added in parent: set Ƹᵢⱼ=0 and Ʒᵢⱼ=0 in parent.
            • include_altruism_term added in parent: set Vᵢⱼ=0 and Ʌᵢⱼ=0 in parent (if present).
            • negativity_social_comparison added in parent: set Ʒᵢⱼ = Ƹᵢⱼ (tie guilt to envy).
            • use_negativity_parameters added in parent: set Ʌ-weights equal to their V counterparts.
            • fix_self_interest_parameter released in parent: set Vᵢᵢ = 1.0 to replicate the fixed-value child.

        Notes:
            • Only parent keys that exist in `parent_param_keys` are written.
            • Any child keys that the parent also has are copied verbatim unless overridden by a rule above.

        Returns:
            • dict[str, float]; Parent-parameter means that embed the child.
        """
        parent_parameters: dict[str, float] = {}
        
        # 1) Start by copying any overlapping child means into the parent (safe default).
        for parameter_key in parent_param_keys:
            if parameter_key in child_parameter_dict:
                parent_parameters[parameter_key] = float(child_parameter_dict[parameter_key])

        # 2) Apply changed-setting-specific embedding rules.
        if changed_utility_setting == "use_exponential_parameters":
            # Parent gained exponents. If child already has γ1, tie; otherwise set all to 1.
            child_gamma_keys = [k for k in child_parameter_dict.keys() if k.startswith('γ')]
            if child_gamma_keys:
                # Child already had γ's (rare, e.g., when moving single->multi as a side effect); tie all to γ1
                child_gamma1 = child_parameter_dict.get('γ1', 1.0)
                for parameter_key in parent_param_keys:
                    if parameter_key.startswith('γ'):
                        parent_parameters[parameter_key] = float(child_gamma1)
            else:
                # No γ in the child → set all γ in the parent to 1.0
                for parameter_key in parent_param_keys:
                    if parameter_key.startswith('γ'):
                        parent_parameters[parameter_key] = 1.0

        elif changed_utility_setting == "single_exponential_parameter":
            # Tie/untie exponents: if parent has multiple γ's and child had γ1, tie them to γ1.
            if 'γ1' in child_parameter_dict:
                common_gamma = float(child_parameter_dict['γ1'])
            else:
                common_gamma = 1.0
            # If parent has γ2 or γ3, set them equal to common γ.
            for gamma_key in ('γ1', 'γ2', 'γ3'):
                if gamma_key in parent_param_keys:
                    # If parent was the *tied* version (γ1 only), writing γ1 is enough.
                    # If parent has separate γ’s, copy common value to each.
                    parent_parameters[gamma_key] = float(parent_parameters.get(gamma_key, common_gamma))

        elif changed_utility_setting == "include_social_comparison":
            # Social comparison added in parent → zero its weights to reproduce child
            if 'Ƹᵢⱼ' in parent_param_keys:
                parent_parameters['Ƹᵢⱼ'] = 0.0
            if 'Ʒᵢⱼ' in parent_param_keys:
                parent_parameters['Ʒᵢⱼ'] = 0.0

        elif changed_utility_setting == "include_altruism_term":
            # Altruism added in parent → zero its weights
            if 'Vᵢⱼ' in parent_param_keys:
                parent_parameters['Vᵢⱼ'] = 0.0
            if 'Ʌᵢⱼ' in parent_param_keys:
                parent_parameters['Ʌᵢⱼ'] = 0.0

        elif changed_utility_setting == "negativity_social_comparison":
            # Parent splits envy/guilt → tie them to the child's single weight (Ʒᵢⱼ_child)
            single = float(child_parameter_dict.get('Ʒᵢⱼ', parent_parameters.get('Ʒᵢⱼ', 0.0)))
            if 'Ƹᵢⱼ' in parent_param_keys:
                parent_parameters['Ƹᵢⱼ'] = single
            if 'Ʒᵢⱼ' in parent_param_keys:
                parent_parameters['Ʒᵢⱼ'] = single

        elif changed_utility_setting == "use_negativity_parameters":
            # Parent gained negativity mirrors → copy Vᵢᵢ→Ʌᵢᵢ and Vᵢⱼ→Ʌᵢⱼ if present
            if 'Ʌᵢᵢ' in parent_param_keys:
                parent_parameters['Ʌᵢᵢ'] = float(parent_parameters.get('Vᵢᵢ', child_parameter_dict.get('Vᵢᵢ', 0.0)))
            if 'Ʌᵢⱼ' in parent_param_keys:
                parent_parameters['Ʌᵢⱼ'] = float(parent_parameters.get('Vᵢⱼ', child_parameter_dict.get('Vᵢⱼ', 0.0)))

        elif changed_utility_setting == "fix_self_interest_parameter":
            # Parent released Vᵢᵢ → set it to fixed constant (1.0) to replicate child
            if 'Vᵢᵢ' in parent_param_keys:
                parent_parameters['Vᵢᵢ'] = 1.0

        # elif changed_utility_setting == "conditional_welfare_mode":
        #     # Make 'ahead' and 'behind' branches identical → tie Λ to V.
        #     if 'Ʌᵢᵢ' in parent_param_keys:
        #         parent_parameters['Ʌᵢᵢ'] = float(parent_parameters.get('Vᵢᵢ', child_parameter_dict.get('Vᵢᵢ', 0.0)))
        #     if 'Ʌᵢⱼ' in parent_param_keys:
        #         parent_parameters['Ʌᵢⱼ'] = float(parent_parameters.get('Vᵢⱼ', child_parameter_dict.get('Vᵢⱼ', 0.0)))

        elif changed_utility_setting == "include_altruism_term":
            if parent_settings.get("conditional_welfare_mode", False):
                # Parent gained an explicit altruism parameter inside conditional welfare.
                # To replicate the child (which uses implicit 1 - Vᵢᵢ / 1 - Ʌᵢᵢ), set:
                if 'Vᵢⱼ' in parent_param_keys:
                    parent_parameters['Vᵢⱼ'] = 1.0 - float(parent_parameters.get('Vᵢᵢ', child_parameter_dict.get('Vᵢᵢ', 0.0)))
                if 'Ʌᵢⱼ' in parent_param_keys:
                    parent_parameters['Ʌᵢⱼ'] = 1.0 - float(parent_parameters.get('Ʌᵢᵢ', child_parameter_dict.get('Ʌᵢᵢ', 0.0)))
            else:
                # Non-conditional case: zeroing altruism reproduces the child
                if 'Vᵢⱼ' in parent_param_keys: parent_parameters['Vᵢⱼ'] = 0.0
                if 'Ʌᵢⱼ' in parent_param_keys: parent_parameters['Ʌᵢⱼ'] = 0.0

        # 3) Any remaining parent keys not touched yet get a benign default:
        for parameter_key in parent_param_keys:
            if parameter_key not in parent_parameters:
                # If it's a γ, default to 1.0; otherwise default to 0.0 (neutral weight).
                parent_parameters[parameter_key] = 1.0 if parameter_key.startswith('γ') else 0.0

        return parent_parameters

    def _sum_negloglik_for_player_and_role(dyad_games_for_player: dict[str, list[dict]],
                                        player_uuid: str,
                                        player_role: str,
                                        general_settings: dict[str, Any]) -> float:
        """
        Sums NLL across all dyads/games for a single (player, role).
        Uses the same storage locations as your pipeline.

        Returns:
            • float; sum of 'loss_final_sum' across the player's dyads for the specified role.
        """
        total_negative_log_likelihood = 0.0

        for _, dyad_games in dyad_games_for_player.items():
            # The NLL breakdown is written by loss_function_bayes + create_loss_report
            # See your pipeline call sites for this sequence. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}
            role_report = dyad_games[0].get('loss_report', {}).get(player_uuid, {}).get(player_role, {})
            total_negative_log_likelihood += float(role_report.get('loss_final_sum', 0.0))

        return total_negative_log_likelihood

    def _evaluate_loss_for_model_and_player(player_uuid: str,
                                            player_role: str,
                                            general_settings: dict[str, Any],
                                            file_paths: dict[str, Any],
                                            utility_settings: dict[str, bool],
                                            param_info: dict[str, Any],
                                            parameter_values: dict[str, float],
                                            choice_temperature: float | None = None) -> float:
        """
        Produces predictions with agent(), computes loss with loss_function_bayes(),
        and returns sum NLL for a single player/role.

        Calls:
            • prep.dyads_for_a_player(...)  → returns dict[dyad_key] = dyad_games
            • agent(...)                          → writes predictions to param_estimates
            • loss_function_bayes(...)            → writes raw per-game losses
            • create_loss_report(...)             → aggregates & stores per-player/role sums

        References:
            - Example usage pattern around agent() → loss → create_loss_report in your code base. 
                :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}

        Returns:
            • float; sum of 'loss_final_sum'.
        """
        # Load this player's dyads
        player_dyads = prep.dyads_for_a_player(
            player_uuid=player_uuid,
            experiment_num=int(general_settings.get('experiment_num', 3)),
            file_paths=file_paths,
            analysis_mode='bayesian'
        )

        # Run agent across each dyad with the provided parameter dictionary for this role.
        for dyad_key, dyad_games in player_dyads.items():
            dyad_games_copy = copy.deepcopy(dyad_games)
            updated_dyad_games = agent(
                dyad_games=dyad_games_copy,
                game_idx_start=0,
                game_idx_stop=len(dyad_games_copy) - 1,
                initial_params={player_role: parameter_values},
                param_info=param_info,
                utility_settings=utility_settings,
                player_uuid=player_uuid,
                player_role=player_role,
                general_settings=general_settings,
                choice_temperature=choice_temperature
            )  # :contentReference[oaicite:5]{index=5}

            # Compute loss and attach loss_report to first game (your convention).
            updated_dyad_games = loss_function_bayes(dyad_games=updated_dyad_games, general_settings=general_settings)  # :contentReference[oaicite:6]{index=6}
            updated_dyad_games[0]['loss_report'] = create_loss_report(dyad_games=updated_dyad_games, general_settings=general_settings)

            # Replace the dyad with the updated one so aggregation uses consistent objects
            player_dyads[dyad_key] = updated_dyad_games

        return _sum_negloglik_for_player_and_role(
            dyad_games_for_player=player_dyads,
            player_uuid=player_uuid,
            player_role=player_role,
            general_settings=general_settings
        )

    def _enumerate_child_parent_pairs_from_ic(ic_dataframe: pd.DataFrame,
                                            ordered_flag_keys: list[str],
                                            general_settings: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Walks the IC table and, for each model (as child), lists all immediate parents (Δk > 0)
        using `gnrl.parents_children_of` so we respect your minimal-dependent-fix rules.

        Returns:
            • list of dicts with:
                {
                    'child_index': int,
                    'parent_index': int,
                    'child_settings_tuple': tuple[bool,...],
                    'parent_settings_tuple': tuple[bool,...],
                    'changed_utility_setting': str
                }
        """
        # Build tuple signatures and index map
        def row_to_tuple(row: pd.Series) -> tuple[bool, ...]:
            return tuple(bool(row[key]) for key in ordered_flag_keys)

        ic_dataframe = ic_dataframe.copy()
        ic_dataframe['utility_tuple'] = ic_dataframe.apply(row_to_tuple, axis=1)
        tuple_to_index: dict[tuple[bool, ...], int] = {tup: idx for idx, tup in ic_dataframe['utility_tuple'].items()}

        all_pairs: list[dict[str, Any]] = []

        # Iterate children in ascending k so we "work our way up"
        if 'k_params' in ic_dataframe.columns:
            ic_dataframe_sorted = ic_dataframe.sort_values(['k_params', 'BIC', 'loss'], ascending=[True, True, True])
        else:
            ic_dataframe_sorted = ic_dataframe.copy()

        for _, child_row in ic_dataframe_sorted.iterrows():
            child_tuple = child_row['utility_tuple']
            child_settings = gnrl.convert_utility_settings(child_tuple, into=dict)  # type: ignore
            neighbor_dict = gnrl.parents_children_of(
                utility_settings=child_settings,
                return_children=False,
                return_parents=True,
                general_settings=general_settings
            )
            parent_tuples = neighbor_dict.get('parents') or []

            for parent_tuple in parent_tuples:
                parent_index = tuple_to_index.get(parent_tuple, None)
                if parent_index is None:
                    continue

                relation_1_to_2, relation_2_to_1, changed_setting = gnrl.classify_pair_relation(
                    model_1=child_tuple,
                    model_2=parent_tuple,
                    general_settings=general_settings,
                    utility_settings=utility_settings
                )
                if relation_1_to_2 != 'child' or relation_2_to_1 != 'parent' or not changed_setting:
                    continue

                all_pairs.append({
                    'child_index': int(child_row.name),
                    'parent_index': int(parent_index),
                    'child_settings_tuple': child_tuple,
                    'parent_settings_tuple': parent_tuple,
                    'changed_utility_setting': changed_setting
                })

        return all_pairs

    # Load the IC table
    ic_path = os.path.join(file_paths["bic_aic"], file_paths["file_names"]["information_criterion"])
    ic_dataframe = pd.read_csv(ic_path)

    general_settings = copy.deepcopy(general_settings)
    general_settings['confidence_weighted'] = False
    # general_settings['penalty_weight'] = 0.0
    
    # Build child→parent pairs
    child_parent_pairs = _enumerate_child_parent_pairs_from_ic(
        ic_dataframe=ic_dataframe,
        ordered_flag_keys=ordered_flag_keys,
        general_settings=general_settings
    )

    if verbose:
        print(f"[Sanity] Identified {len(child_parent_pairs)} child→parent pairs to test.")

    # Determine participants
    experiment_num = int(general_settings.get('experiment_num', 3))
    participant_uuids: list[str] = prep.all_player_uuids(file_paths=file_paths, experiment_num=experiment_num, only_humans=True)
    if isinstance(fit_for_n_players, int) and 0 < fit_for_n_players <= len(participant_uuids):
        participant_uuids = participant_uuids[:fit_for_n_players]  # alphabetical order preserved

    choice_temperature = general_settings.get('softmax_temperature', 1.0)

    # Mapping from tuple signature to IC-row index for lookup
    def row_to_tuple(row: pd.Series) -> tuple[bool, ...]:
        return tuple(bool(row[key]) for key in ordered_flag_keys)
    ic_dataframe['utility_tuple'] = ic_dataframe.apply(row_to_tuple, axis=1)
    signature_to_index: dict[tuple[bool, ...], int] = {tup: idx for idx, tup in ic_dataframe['utility_tuple'].items()}

    rng = random.Random(random_seed)
    results_rows: list[dict[str, Any]] = []

    # Go through each child→parent pair
    for pair_idx, pair in enumerate(child_parent_pairs, start=1):
        child_tuple = pair['child_settings_tuple']
        parent_tuple = pair['parent_settings_tuple']
        changed_utility_setting: str = pair['changed_utility_setting']
        child_index = pair['child_index']
        parent_index = pair['parent_index']

        child_settings = gnrl.convert_utility_settings(child_tuple, into=dict)  # type: ignore
        parent_settings = gnrl.convert_utility_settings(parent_tuple, into=dict)  # type: ignore

        # Build param_info for child and parent (means-only keys for evaluation)
        child_param_info = make_param_info(param_bds=param_bds, utility_settings=child_settings, general_settings=general_settings)
        parent_param_info = make_param_info(param_bds=param_bds, utility_settings=parent_settings, general_settings=general_settings)

        # 1) Sample random child parameter dictionary within bounds (child_param_info)
        child_parameter_dict = _sample_random_parameter_dict(
            param_keys=child_param_info['keys'],
            param_bounds=child_param_info['bounds'],
            rng=rng
        )

        # 2) Embed child → special parent parameter means
        embedded_parent_means = _embed_child_parameters_into_parent_means(
            child_parameter_dict=child_parameter_dict,
            changed_utility_setting=changed_utility_setting,
            parent_param_keys=parent_param_info['keys']
        )

        # Ensure *_std keys exist for grid/MCMC updates (predictor priors need sigmas).
        if general_settings.get('update_method') in ('grid', 'MCMC'):
            bounds_lookup = {k: b for k, b in zip(parent_param_info['keys'], parent_param_info['bounds'])}
            min_std_guess = 0.5  # same convention used elsewhere

            for base_key in [k for k in parent_param_info['keys'] if not k.endswith('_std')]:
                std_key = base_key + '_std'
                lo, hi = bounds_lookup[std_key]

                if std_key in bounds_lookup:
                    current_std_value = embedded_parent_means.get(std_key, None)
                    if (current_std_value is None) or not (float(lo) <= float(current_std_value) <= float(hi)):
                        midpoint = (float(lo) + float(hi)) / 2.0
                        embedded_parent_means[std_key] = max(min_std_guess, midpoint)

        # 3) Evaluate losses over the selected participants
        sum_nll_child = 0.0
        sum_nll_parent = 0.0
        for player_uuid in participant_uuids:
            sum_nll_child += _evaluate_loss_for_model_and_player(
                player_uuid=player_uuid,
                player_role=player_role_to_fit,
                general_settings=general_settings,
                file_paths=file_paths,
                utility_settings=child_settings,
                param_info=child_param_info,
                parameter_values=child_parameter_dict,
                choice_temperature=choice_temperature
            )
            sum_nll_parent += _evaluate_loss_for_model_and_player(
                player_uuid=player_uuid,
                player_role=player_role_to_fit,
                general_settings=general_settings,
                file_paths=file_paths,
                utility_settings=parent_settings,
                param_info=parent_param_info,
                parameter_values=embedded_parent_means,
                choice_temperature=choice_temperature
            )

        # Pretty equations
        equation_child = build_utility_equation(utility_settings=child_settings)
        equation_parent = build_utility_equation(utility_settings=parent_settings)

        # k's by *your* counting
        k_child = gnrl.count_free_parameters(utility_settings=child_settings, general_settings=general_settings)
        k_parent = gnrl.count_free_parameters(utility_settings=parent_settings, general_settings=general_settings)

        # Flatten parameter columns for the CSV (fill Nones where absent)
        all_param_keys = list(param_bds.keys())  # superset of all possible keys
        row: dict[str, Any] = {
            "child_idx": int(child_index),
            "parent_idx": int(parent_index),
            "k_child": int(k_child),
            "k_parent": int(k_parent),
            "changed_utility_setting": changed_utility_setting,
            "loss_child": float(sum_nll_child),
            "loss_parent": float(sum_nll_parent),
            "loss_parent_minus_child": float(sum_nll_parent - sum_nll_child),
            "equal_loss": abs(sum_nll_parent - sum_nll_child) <= numeric_tolerance,
            "equation_child": equation_child,
            "equation_parent": equation_parent,
            "utility_settings_child": child_tuple,
            "utility_settings_parent": parent_tuple,
            "n_players_evaluated": len(participant_uuids),
            "player_role": player_role_to_fit,
        }

        # 13 boolean flags duplicated (child and parent)
        for flag_key, child_val in child_settings.items():
            row[f"child_{flag_key}"] = bool(child_val)
        for flag_key, parent_val in parent_settings.items():
            row[f"parent_{flag_key}"] = bool(parent_val)

        # Parameter dictionaries (JSON-ish strings) and also one column per parameter (child/parent)
        row["params_child"] = {k: child_parameter_dict.get(k, None) for k in all_param_keys}
        row["params_parent"] = {k: embedded_parent_means.get(k, None) for k in all_param_keys}
        for parameter_key in all_param_keys:
            row[f"{parameter_key}_child"] = child_parameter_dict.get(parameter_key, None)
            row[f"{parameter_key}_parent"] = embedded_parent_means.get(parameter_key, None)

        results_rows.append(row)

        if verbose and pair_idx % 10 == 0:
            print(f"[Sanity] Processed {pair_idx}/{len(child_parent_pairs)} pairs...")

    results_dataframe = pd.DataFrame(results_rows)

    # Save CSV
    csv_name = csv_file_name or "child_parent_embedding_sanity_checks.csv"
    if ".csv" in csv_name:
        csv_name = csv_name.replace(".csv", "")
    csv_name += f"-{general_settings.get('update_method', None)}-{player_role_to_fit}-{fit_for_n_players}.csv"
    csv_path = os.path.join(file_paths["bic_aic"], csv_name)
    results_dataframe.to_csv(csv_path, index=False, encoding="utf-8-sig")
    if verbose:
        print(f"[Sanity] Wrote: {csv_path}")

    return results_dataframe


def run_child_parent_probability_equivalence_smoketest(utility_settings: dict[str, Any], file_paths: dict[str, Any], param_bds: dict[str, tuple[float, float]], 
                                                       n_trials: int = 12, rand_payoff_idx: bool = False, rng_seed: int | None = None, tolerance: float = 1e-10, verbose: bool = True) -> pd.DataFrame:
    """
    Verifies nesting by comparing *choice probabilities* of each child with the
    probabilities of its embedded special parent on a small synthetic set of games.

    Procedure:
        1) Build child→parent pairs from the IC table (so we keep your canonical indices).
        2) Draw a child mean-parameter dictionary uniformly within `param_bds` (means only).
        3) Embed child means into the parent (gnrl rules) to reproduce the child.
        4) Compute p(A) for n_trials randomized games using choice(...), for child and parent.
        5) Store max|Δp| and a boolean pass/fail in a compact CSV.

    Arguments:
        • utility_settings: dict[str, Any]; 
        • file_paths: dict[str, Any]; 
        • param_bds: dict[str, tuple[float, float]]; 
        • n_trials: int; 
        • rand_payoff_idx: bool; 
        • rng_seed: int | None; 
        • tolerance: float; 
        • verbose: bool; 
        
    Returns:
        • pd.DataFrame with one row per (child, parent) pair and max_abs_delta across trials.
    """
    # --- Helpers ---------------------------------------------------------------
    def _generate_synthetic_games(n_games: int = 10, rng_seed: int = 20250417) -> list[dict]:
        rng = random.Random() if rng_seed is None else random.Random(rng_seed)
        games: list[dict] = []
        for _ in range(n_games):
            As = rng.randint(1, 5)
            Ao = rng.randint(1, 5)
            Bs = rng.randint(1, 5)
            Bo = rng.randint(1, 5)
            games.append({
                "payoff_A_chooser":   As,
                "payoff_A_predictor": Ao,
                "payoff_B_chooser":   Bs,
                "payoff_B_predictor": Bo,
            })
        return games

    def _choice_probs(games: list[dict], u_settings: dict[str, bool],
                      params: dict[str, float], temperature: float = 1.5) -> list[float]:
        out = []
        for g in games:
            r = choice(current_game=g,
                       agent_params=params,
                       utility_settings=u_settings,
                       softmax_temperature=temperature,
                       normalize_conditional_welfare_params=False,
                       select=False)
            out.append(float(r["model_choose_A"]))
        return out

    def _means_only_keys(u: dict[str, bool]) -> list[str]:
        # Only means (no *_std, no cov), consistent with your choice() pipeline
        return [k for k in gnrl.parameter_keys_for_utility_settings(
            u, general_settings={"update_method": "naive"}
        ) if not (k.endswith("_std") or k.endswith("_cov"))]

    def _sample_means(param_keys: list[str], rng: random.Random) -> dict[str, float]:
        out: dict[str, float] = {}
        for key in param_keys:
            lo, hi = param_bds[key]
            val = rng.uniform(float(lo), float(hi))
            if (hi - lo) > 1:
                val = min(max(round(val, 1), lo), hi)
            out[key] = float(val)
        return out

    def _show_math_work(equation: str, params: dict[str, float], utility_settings: dict[str, bool] | tuple[bool], 
                        game_idx: int = 0, decimals: int = 2, comparison_tol: float = 5e-3) -> str:
        """
        Fill the pretty equation from build_utility_equation with concrete numbers,
        evaluate its right-hand side (RHS), and verify it matches utility() after rounding.

        Returns the fully substituted equation plus a trailing status tag:
        [OK]               — evaluated and matched (within tolerance after rounding),
        [FAIL: ≠ x.xx]     — evaluated but differs from utility() (rounded),
        [EVAL ERROR: ...]  — evaluation failed (syntax/name/type errors etc.).

        Notes for maintainers:
            • This function does NOT modify build_utility_equation; it only normalizes
            the rendered string so that Python's eval can handle unicode, unicode
            operators, implicit multiplication, and non-integer exponents on negative bases.
            • Any parameter symbols still present after substitution are replaced with
            the default values used by utility() so eval never sees a stray symbol.
        """
        # --- 1) Compute the ground-truth utility from code (source of truth) -----
        payoff_key_map_eval = {
            "payoff_A_chooser":   "As",
            "payoff_A_predictor": "Ao",
            "payoff_B_chooser":   "Bs",
            "payoff_B_predictor": "Bo",
        }
        utility_settings_dict = gnrl.convert_utility_settings(utility_settings, into=dict)
        payoffs_eval = {
            payoff_key_map_eval[k]: v
            for k, v in games[game_idx].items()
            if "payoff" in k
        }
        true_value = float(utility(payoffs=payoffs_eval, params=params, utility_settings=utility_settings_dict, normalize_conditional_welfare_params=False))
        true_value_rounded = round(true_value, decimals)

        # --- 2) Substitute payoffs and parameters into the pretty string ----------
        gamma_pretty = {"γ1": "γ₁", "γ2": "γ₂", "γ3": "γ₃"}
        params_with_pretty_gammas = {gamma_pretty.get(k, k): v for k, v in params.items()}

        # Known aliases that sometimes appear in strings or terminals
        alias_to_pretty = {
            "Vii": "Vᵢᵢ", "Λii": "Ʌᵢᵢ", "Vij": "Vᵢⱼ", "Λij": "Ʌᵢⱼ",
            "Ƹij": "Ƹᵢⱼ", "Ʒij": "Ʒᵢⱼ",
            # safeguard: if these ever appear, map them to the canonical key
            "γ1": "γ₁", "γ2": "γ₂", "γ3": "γ₃",
        }
        for alias_key, pretty_key in alias_to_pretty.items():
            if alias_key in params and pretty_key not in params_with_pretty_gammas:
                params_with_pretty_gammas[pretty_key] = params[alias_key]

        payoff_symbol_map = {"As": "πᵢᴬ", "Ao": "πⱼᴬ", "Bs": "πᵢᴮ", "Bo": "πⱼᴮ"}
        payoffs_pretty_map = {payoff_symbol_map[k]: v for k, v in payoffs_eval.items()}

        filled_equation = equation.replace("Uᵢ(A)", f"{true_value_rounded:.{decimals}f}")
        # Substitute payoffs first, then parameters (reduces accidental overlap)
        for symbol, value in payoffs_pretty_map.items():
            filled_equation = filled_equation.replace(symbol, str(value))
        for symbol, value in params_with_pretty_gammas.items():
            filled_equation = filled_equation.replace(symbol, str(value))

        gamma1_value = None
        for ks in ("γ₁", "γ1"):
            if ks in params_with_pretty_gammas:
                gamma1_value = float(params_with_pretty_gammas[ks]); break
        if gamma1_value is None: gamma1_value = 1.0

        default_values = {
            "Vᵢᵢ": 1.0, "Ʌᵢᵢ": 0.0, "Vᵢⱼ": 0.0, "Ʌᵢⱼ": 0.0, "Ƹᵢⱼ": 0.0, "Ʒᵢⱼ": 0.0,
            "γ₁": gamma1_value, "γ₂": gamma1_value, "γ₃": gamma1_value,
            "Vii": 1.0, "Λii": 0.0, "Vij": 0.0, "Λij": 0.0, "Ƹij": 0.0, "Ʒij": 0.0,
            "γ1": gamma1_value, "γ2": gamma1_value, "γ3": gamma1_value,
        }

        for symbol, default_val in default_values.items():
            if symbol in filled_equation:
                filled_equation = filled_equation.replace(symbol, str(default_val))

        lhs, rhs = filled_equation.split("=", 1)
        rhs = gnrl.canon_sc_both_ways(rhs.strip(), mode="twoterm")
        filled_equation = lhs + "= " + rhs

        # Evaluate using the shared helper
        value_str, status = gnrl.eval_pretty_equation_rhs(rhs, decimals=decimals, sc_mode="twoterm")
        if status:
            return f"{filled_equation}  [{status}]"

        rhs_value_rounded = float(value_str)
        status_tag = "" if abs(rhs_value_rounded - true_value_rounded) <= comparison_tol \
                    else f"  [FAIL: ≠ {rhs_value_rounded:.{decimals}f}]"
        return f"{filled_equation}{status_tag}"

    def _components_for_utility(payoffs: dict, params: dict[str, float], utility_settings: dict[str, bool]) -> dict[str, float]:
        """"""
        util_components: dict = utility(payoffs=payoffs, params=params, utility_settings=utility_settings, separate_terms=True, normalize_conditional_welfare_params=False)
        for old_key, new_key in [('self_interest', 'self'), ('altruism', 'altr'), ('social_comp', 'socc')]:
            util_components[new_key] = util_components.pop(old_key)
        return {key: round(val, 6) for key, val in util_components.items()}
        
    def align_parent_child_equations(parent_eq: str, child_eq: str, pad: int = 1) -> tuple[str,str]:
        """
        Token-aligns two equation strings so corresponding terms start at the same columns.
        Splits on spaces, pads each column to the max width across the two rows.
        """
        parent_rhs = parent_eq.split("=", 1)[1].strip()
        child_rhs  = child_eq.split("=", 1)[1].strip()
        p_tokens = parent_rhs.split()
        c_tokens = child_rhs.split()
        width = [max(len(p_tokens[i]) if i < len(p_tokens) else 0,
                    len(c_tokens[i]) if i < len(c_tokens) else 0)
                for i in range(max(len(p_tokens), len(c_tokens)))]
        def _pad(tokens):
            return "".join((tokens[i] if i < len(tokens) else "").ljust(width[i] + pad)
                        for i in range(len(width)))
        aligned_parent = f"{parent_eq.split('=')[0]}= " + _pad(p_tokens)
        aligned_child  = f"{child_eq.split('=')[0]}= " + _pad(c_tokens)
        return aligned_parent, aligned_child

    def _evaluate_equation_numeric(equation_string: str, params: dict[str, float], utility_settings: UtilitySettings, payoffs: dict[str, float], 
                                   param_overrides: dict[str, float] | None = None, decimals_local: int = 6, ) -> tuple[float | None, str]:
        """
        Evaluate the RHS of the pretty equation after substituting 'params' and 'payoffs'.
        'param_overrides' (if provided) override parameter values *before* substitution.
        Returns (value_or_None, status_text). Status empty string on success.
        """
        payoff_key_map_eval = {
            "payoff_A_chooser":   "As",
            "payoff_A_predictor": "Ao",
            "payoff_B_chooser":   "Bs",
            "payoff_B_predictor": "Bo",
        }
        payoffs = {payoff_key_map_eval[k] if "payoff" in k else k: v for k, v in payoffs.items()}

        # 1) Build replacements
        gamma_pretty = {"γ1": "γ₁", "γ2": "γ₂", "γ3": "γ₃"}
        params_pretty = {gamma_pretty.get(k, k): v for k, v in params.items()}
        # Map aliases that might appear
        alias_to_pretty = {
            "Vii": "Vᵢᵢ", "Λii": "Ʌᵢᵢ", "Vij": "Vᵢⱼ", "Λij": "Ʌᵢⱼ",
            "Ƹij": "Ƹᵢⱼ", "Ʒij": "Ʒᵢⱼ",
            "γ1": "γ₁", "γ2": "γ₂", "γ3": "γ₃",
        }
        for alias_key, pretty_key in alias_to_pretty.items():
            if alias_key in params and pretty_key not in params_pretty:
                params_pretty[pretty_key] = params[alias_key]

        # Apply overrides *before* substitution (so numeric values in the string reflect the override)
        if param_overrides:
            for k_raw, v in param_overrides.items():
                k = gamma_pretty.get(k_raw, k_raw)  # allow γ1 vs γ₁ in overrides
                params_pretty[k] = v

        # γ fallbacks inherit γ₁ (like utility())
        gamma1_value = None
        for k_try in ("γ₁", "γ1"):
            if k_try in params_pretty:
                gamma1_value = float(params_pretty[k_try]); break
        if gamma1_value is None: gamma1_value = 1.0

        payoff_symbol_map = {"As": "πᵢᴬ", "Ao": "πⱼᴬ", "Bs": "πᵢᴮ", "Bo": "πⱼᴮ"}
        payoffs_pretty_map = {payoff_symbol_map[k]: v for k, v in payoffs.items()}

        # 2) Extract RHS
        if "=" not in equation_string:
            return None, "EVAL ERROR: no '=' in equation"
        _, rhs_original = equation_string.split("=", 1)
        rhs_original = rhs_original.strip()

        # 3) Perform substitutions (payoffs first, then params)
        rhs_filled = rhs_original
        for sym, val in payoffs_pretty_map.items():
            rhs_filled = rhs_filled.replace(sym, str(val))
        for sym, val in params_pretty.items():
            rhs_filled = rhs_filled.replace(sym, str(val))

        # Replace any remaining canonical symbols with *utility()* defaults
        defaults = {
            "Vᵢᵢ": 1.0, "Ʌᵢᵢ": 0.0, "Vᵢⱼ": 0.0, "Ʌᵢⱼ": 0.0, "Ƹᵢⱼ": 0.0, "Ʒᵢⱼ": 0.0,
            "γ₁": gamma1_value, "γ₂": gamma1_value, "γ₃": gamma1_value,
            "Vii": 1.0, "Λii": 0.0, "Vij": 0.0, "Λij": 0.0, "Ƹij": 0.0, "Ʒij": 0.0,
            "γ1": gamma1_value, "γ2": gamma1_value, "γ3": gamma1_value,
        }

        for sym, val in defaults.items():
            if sym in rhs_filled:
                rhs_filled = rhs_filled.replace(sym, str(val))

        # Canonicalize and evaluate via generalist helpers
        rhs_filled = gnrl.canon_sc_both_ways(rhs_filled, mode="twoterm")
        value, status = gnrl.eval_pretty_equation_rhs(rhs_filled, decimals=decimals_local, sc_mode="twoterm")
        return value, status

    # --- Build pairs from adjacency lists -------------------------------------
    # Use your canonical universe (+ equation strings) as IC does
    adj = model_nesting_adjacency_matrices(
        general_settings=general_settings,
        utility_settings=utility_settings,
        file_paths= file_paths,
        create_new_file=False,
        equation_form=False,
        print_=False
    )
    settings_list: list[dict[str, bool]] = adj["settings"]
    parents_of: list[list[int]] = adj["adjacency_lists"]["parent_of"]  # child idx -> parent indices

    # Child→parent candidate pairs from adjacency; then confirm with classifier
    candidate_pairs: list[tuple[int, int, str]] = []
    for child_idx, parent_indices in enumerate(parents_of):
        child_settings = settings_list[child_idx]
        for parent_idx in parent_indices:
            parent_settings = settings_list[parent_idx]
            r12, r21, changed = gnrl.classify_pair_relation(
                model_1=child_settings,
                model_2=parent_settings,
                utility_settings=utility_settings,
                general_settings=None
            )
            if r12 == "child" and r21 == "parent" and changed is not None:
                candidate_pairs.append((child_idx, parent_idx, changed))

    if verbose:
        print(f"[Prob-Sanity] Candidate child→parent pairs from adjacency: {len(candidate_pairs)}")

    # Synthetic games + random generator
    games = _generate_synthetic_games(n_games=n_trials, rng_seed=rng_seed)
    temp = float(1.5) if "softmax_temperature" not in utility_settings else float(utility_settings["softmax_temperature"])
    rng = random.Random() if rng_seed is None else random.Random(rng_seed)

    results: list[dict[str, Any]] = []
    for jdx, (child_idx, parent_idx, changed) in enumerate(candidate_pairs, start=1):
        child_settings  = settings_list[child_idx]
        parent_settings = settings_list[parent_idx]

        # 1) sample child means
        child_keys  = _means_only_keys(child_settings)
        child_means = _sample_means(child_keys, rng=rng)

        # 2) embed means: child → parent (means only)
        try:
            parent_param_info = gnrl.map_child_to_parent_special_param_info(
                child_utility_settings=child_settings,
                parent_utility_settings=parent_settings,
                child_fitted_parameters=child_means,
                general_settings={"update_method": "naive"},
                param_bds=param_bds,
                build_utility_equation=build_utility_equation
            )
        except NotImplementedError:
            # If you keep structural guards in the mapper for some families, skip cleanly.
            continue

        parent_means = {
            param_key: float(param_val) for param_key, param_val in zip(parent_param_info["keys"], parent_param_info["guesses"])
            if not (param_key.endswith("_std") or param_key.endswith("_cov"))
        }

        # 3) compare probabilities on the same games
        probs_child  = _choice_probs(games, child_settings,  child_means,  temp)
        probs_parent = _choice_probs(games, parent_settings, parent_means, temp)
        max_abs_delta = float(max(abs(p_child - p_parent) for p_child, p_parent in zip(probs_child, probs_parent)) if probs_child else 0.0)

        # (d) max |Δp|
        deltas = [abs(probs_child - probs_parent) for probs_child, probs_parent in zip(probs_child, probs_parent)]
        max_abs_delta = float(max(deltas) if deltas else 0.0)
        max_abs_delta_idx = 0
        if rand_payoff_idx: 
            max_abs_delta_idx = random.randint(a=0, b=n_trials-1)
        else:
            for idx, delta in enumerate(deltas):
                if delta >= max_abs_delta:
                    max_abs_delta_idx = idx

        # (e) Display equations and replace payoffs and params with their values
        equation_child =  build_utility_equation(utility_settings=child_settings)
        equation_parent = build_utility_equation(utility_settings=parent_settings)

        # --- Compute the four utilities on the same representative game (games[0]) ---
        # 1) Code (source of truth)
        payoffs_for_one = {
            "As": games[max_abs_delta_idx]["payoff_A_chooser"],
            "Ao": games[max_abs_delta_idx]["payoff_A_predictor"],
            "Bs": games[max_abs_delta_idx]["payoff_B_chooser"],
            "Bo": games[max_abs_delta_idx]["payoff_B_predictor"],
        }
        U_code_child  = round(float(utility(payoffs_for_one, child_means,  child_settings, normalize_conditional_welfare_params=False)), 3)
        U_code_parent = round(float(utility(payoffs_for_one, parent_means, parent_settings, normalize_conditional_welfare_params=False)), 3)

        # 2) Pretty equation (evaluated numerically)
        U_equa_child,  err_child  = _evaluate_equation_numeric(equation_string=equation_child,  params=child_means,  
                                        utility_settings=utility_settings, payoffs=games[max_abs_delta_idx], decimals_local=3)
        U_equa_parent, err_parent = _evaluate_equation_numeric(equation_string=equation_parent, params=parent_means, 
                                        utility_settings=utility_settings, payoffs=games[max_abs_delta_idx], decimals_local=3)

        # 3) Boolean comparisons (tolerant equality)
        def same(a: float | None, b: float | None, tol: float = 1e-2) -> bool:
            return (a is not None) and (b is not None) and (abs(a - b) <= tol)

        match_code_child_vs_parent = same(U_code_child,  U_code_parent)
        match_equa_child_vs_parent = same(U_equa_child,  U_equa_parent)
        match_child_code_vs_equa   = same(U_code_child,  U_equa_child)
        match_parnt_code_vs_equa   = same(U_code_parent, U_equa_parent)

        # 4) Short diagnosis tag
        diagnosis = "OK"
        use_long_tag = False
        if not match_code_child_vs_parent:
            if use_long_tag:
                diagnosis += "-NESTING_MISMATCH_CODE"
            else:
                diagnosis += "-NESTCODE"
        if not match_equa_child_vs_parent:
            if use_long_tag:
                diagnosis += "-NESTING_MISMATCH_EQUATION"
            else:
                diagnosis += "-NESTEQUA"
        if not match_child_code_vs_equa or not match_parnt_code_vs_equa:
            if use_long_tag:
                diagnosis += "-STRING_BUILDER_MISMATCH"
            else:
                diagnosis += "-STRBUILD"
        if diagnosis.startswith("OK-"):
            diagnosis = diagnosis[3:]

        # 5) Worked strings (nice to keep for human inspection)
        worked_child =  _show_math_work(equation=equation_child, params=parent_means, 
                            game_idx=max_abs_delta_idx, utility_settings=child_settings)
        worked_parent = _show_math_work(equation=equation_parent, params=parent_means, 
                            game_idx=max_abs_delta_idx, utility_settings=parent_settings)

        comp_child  = _components_for_utility(payoffs_for_one, child_means,  child_settings)
        comp_parent = _components_for_utility(payoffs_for_one, parent_means, parent_settings)
        comp_delta  = {k: round(comp_parent[k]-comp_child[k], 6) for k in comp_child.keys() if abs(comp_parent[k]-comp_child[k]) > 1e-9}

        # equation_parent_aligned, equation_child_aligned = align_parent_child_equations(parent_eq=equation_parent, child_eq=equation_child)
        equation_parent_aligned, equation_child_aligned = equation_parent, equation_child

        results.append({
            "n_trials": n_trials,
            "temperature": temp,
            "child_idx": child_idx,
            "parent_idx": parent_idx,
            "changed_utility_setting": changed,
            "max_abs_delta_p": max_abs_delta,
            "all_equal": (max_abs_delta <= tolerance),
            "utility_settings_child": child_settings,
            "utility_settings_parent": parent_settings,
            "parameters": parent_means,
            "equation_child":  equation_child_aligned,
            "equation_parent": equation_parent_aligned,

            # Representative game payload (helps re-run a single case quickly)
            "game_As": payoffs_for_one["As"],
            "game_Ao": payoffs_for_one["Ao"],
            "game_Bs": payoffs_for_one["Bs"],
            "game_Bo": payoffs_for_one["Bo"],

            # Four utilities
            "U_code_child":  U_code_child,
            "U_code_parent": U_code_parent,
            "U_equa_child":  U_equa_child,
            "U_equa_parent": U_equa_parent,

            # Comparison flags
            "match_code_child_vs_parent": match_code_child_vs_parent,
            "match_equa_child_vs_parent": match_equa_child_vs_parent,
            "match_child_code_vs_equa":   match_child_code_vs_equa,
            "match_parnt_code_vs_equa":   match_parnt_code_vs_equa,
            "diagnosis":   diagnosis,

            "comp_child":  comp_child,
            "comp_parent": comp_parent,
            "comp_delta":  comp_delta,

            "worked_child":  "'" + worked_child,
            "worked_parent": "'" + worked_parent
        })

        if verbose and jdx % 50 == 0:
            print(f"[Prob-Sanity] {jdx}/{len(candidate_pairs)} processed...")

    df = pd.DataFrame(results)
    df = df.sort_values(by=["all_equal", "changed_utility_setting", "diagnosis"], ascending=[True, True, True])
    out_path = os.path.join(file_paths["processed"], "child_parent_prob_equivalence.csv")
    try: df.to_csv(out_path, index=False, encoding="utf-8-sig")
    except (PermissionError, OSError): pass

    if verbose:
        n_ok = int(df["all_equal"].sum())
        print(f"[Prob-Sanity] {n_ok}/{len(df)} pairs matched within tol={tolerance}.")
        print(f"[Prob-Sanity] Wrote: {out_path}")

        # Print a few mismatches (if any) to inspect
        mismatches = df.loc[~df["all_equal"]].head(10)
        if len(mismatches) == 0:
            print("All tested child→parent pairs matched exactly on p(A).")
        else:
            print("First few non-matching pairs (inspect equations and changed flag):")
            for _, row in mismatches.iterrows():
                print(f"  child_idx={row['child_idx']} → parent_idx={row['parent_idx']}"
                    f" | changed={row['changed_utility_setting']}"
                    f" | max|Δp|={row['max_abs_delta_p']:.3e}")

        diagnosis_counts = df['diagnosis'].value_counts().to_dict()
        print(diagnosis_counts)

        # Print parent and child equations for STRBUILD and NESTEQUA-STRBUILD diagnoses
        relevant_rows = df[df["diagnosis"].isin(["STRBUILD", "NESTEQUA-STRBUILD"])]
        if not relevant_rows.empty:
            print("\nParent and child equations with STRBUILD/NESTEQUA-STRBUILD diagnosis:")
            for _, row in relevant_rows.iterrows():
                print(f"Parent ({row['parent_idx']}): {row['equation_parent']}")
                print(f"Child  ({row['child_idx']}): {row['equation_child']}\n")

    return df


def verify_utility_vs_string_equation(utility_function: Callable, utility_function_str: Callable, utility_settings: UtilitySettings, param_bds: dict[str, tuple[float, float]], 
                                      file_paths: FilePaths, n_games: int = 5**4, rng_seed: int | None = 20250417, exhaustive_if_large: bool = True, option: str = "A", 
                                      comparison_tol: float = 1e-6, decimals: int = 6, verbose: bool = True) -> pd.DataFrame:
    """
    Exhaustively (or randomly) verifies that utility_function(...) and the numeric
    evaluation of utility_function_str(...) produce identical utilities across:
        • all generated, valid utility settings (via gnrl.generate_utility_settings),
        • random parameter means (via make_param_info + uniform sampling within bounds),
        • many payoff structures (random or full 5^4 grid over {1..5}^4).

    In addition to the *total* utility, the routine compares *components*:
        self-interest, altruism, social comparison — for both code and string —
        so discrepancies can be localized immediately.

    Arguments:
        • utility_function : Callable
            The Python function that returns the numeric utility. Must accept (payoffs: dict, params: dict, 
            utility_settings: dict, separate_terms: bool=False) and return either a float (separate_terms=
            False) or a dict with keys {'self_interest','altruism','social_comp'} when separate_terms=True.

        • utility_function_str : Callable
            The string builder, e.g., build_utility_equation. Must accept (utility_settings: dict, 
            in_latex: bool=False, option: str="A"|"B") and return a pretty string like "Uᵢ(A) = ...".

        • general_settings : dict
            Global toggles passed to gnrl.generate_utility_settings and used for consistency.

        • param_bds : dict[str, tuple[float, float]]
            Global parameter bounds; used to sample parameter means.

        • ordered_flag_keys : list[str]
            Canonical order of the 13 boolean flags; used for the 13 column outputs.

        • n_games : int (default 100)
            Number of payoff configurations to test per utility setting. If exhaustive_if_large=True 
            and n_games > 5**4, the routine evaluates the *entire* 5^4 = 625 grid over {1..5}^4.

        • rng_seed : int | None
            Seed for reproducible sampling. If None, system entropy is used.

        • exhaustive_if_large : bool (default True)
            If True and n_games > 625, evaluate all {1..5}^4 payoffs instead of sampling.

        • option : str
            Passed to the string builder ("A" or "B"); the code always evaluates A vs B
            with the familiar payoff names {'As','Ao','Bs','Bo'}.

        • file_paths: dict[str: str]; Dictionary containing all file paths used in this analysis.

        • comparison_tol : float
            Absolute tolerance for declaring a match between code and string.

        • decimals : int
            Rounding used for storing evaluated numbers (comparisons use raw floats).

        • verbose : bool
            If True, prints a compact diagnostic report.

    Returns:
        • pd.DataFrame
            Row-wise verification results sorted by worst discrepancies first.
    """
    # ---------- (0) Where to write -------------------------------------------------
    out_dir = file_paths["processed"]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "utility_vs_string_verification.csv")
    out_summary_path = os.path.join(out_dir, "utility_vs_string_summary.csv")

    ordered_flag_keys: list[str] = list(utility_settings.keys())

    # ---------- (1) Utility setting generation ------------------------------------
    # Generate all valid utility settings using your generator.
    # If your project's signature differs, adjust the call below.
    all_settings_raw = gnrl.generate_utility_settings(utility_settings=utility_settings)
    # Normalize into dict form
    all_settings: list[dict[str, bool]] = [
        gnrl.convert_utility_settings(u, into=dict) for u in all_settings_raw
    ]
    # Keep only valid ones if generator does not guarantee validity
    if hasattr(gnrl, "is_valid_utility_settings"):
        all_settings = [u for u in all_settings if gnrl.is_valid_utility_settings(u)]

    # ---------- (2) Payoff generation ---------------------------------------------
    def _all_payoff_tuples():
        for As, Ao, Bs, Bo in it.product(range(1, 6), repeat=4):
            yield {"As": As, "Ao": Ao, "Bs": Bs, "Bo": Bo}

    def _random_payoff_tuples(n: int, seed: int | None):
        rng_local = random.Random() if seed is None else random.Random(seed)
        for _ in range(n):
            yield {
                "As": rng_local.randint(1, 5),
                "Ao": rng_local.randint(1, 5),
                "Bs": rng_local.randint(1, 5),
                "Bo": rng_local.randint(1, 5),
            }

    use_exhaustive = exhaustive_if_large and (n_games >= 5**4)
    payoff_iterable = list(_all_payoff_tuples()) if use_exhaustive else list(_random_payoff_tuples(n_games, rng_seed))

    # ---------- (3) Parameter sampling --------------------------------------------
    rng_params = random.Random() if rng_seed is None else random.Random(rng_seed)

    def _sample_means_for(utility_settings: dict[str, bool]) -> dict[str, float]:
        """
        Means-only sampling following your conventions. Uses make_param_info to get proper keys,
        but then samples uniformly within global bounds so the sweep explores the space.
        """
        param_info = make_param_info(
            param_bds=param_bds,
            utility_settings=utility_settings,
            general_settings={"update_method": "naive"},
            guess_seed=None
        )
        keys = [k for k in param_info["keys"] if not k.endswith("_std") and not k.endswith("_cov")]
        means: dict[str, float] = {}
        for key in keys:
            lo, hi = param_bds[key]
            val = rng_params.uniform(float(lo), float(hi))
            # small rounding on wide intervals keeps equations readable without losing variety
            if (hi - lo) > 1:
                val = min(max(round(val, 2), lo), hi)
            means[key] = float(val)
        return means

    # ---------- (4) Pretty-equation evaluation (numeric) --------------------------
    # This numerically evaluates the RHS of utility_function_str(...) after substituting
    # payoffs and parameters. It mirrors the robust normalizations you used previously,
    # including: unicode operators, '^' → pow_signed with parenthesis-aware base capture,
    # implicit multiplication insertions, and γ₂/γ₃ fallback to γ₁.
    # --- INSERT C1: direct evaluator for the stubborn ratio+refdep+negSC family ----
    def _direct_eval_ratio_refdep_negsc(
        utility_settings: dict[str, bool],
        params: dict[str, float],
        payoffs: dict[str, float],
        decimals_local: int = 6,
    ) -> float:
        """
        Direct numeric evaluator for the family:
            • NOT conditional_welfare_mode
            • NOT min_max_rawlsian_leontief
            • use_exponential_parameters = True
            • payoff_ratios_not_differences = True
            • reference_dependent_utility = True
            • use_negativity_parameters = False (for SI & AL)
            • negativity_social_comparison = True
            • include_social_comparison = True
            • include_altruism_term = True
            • single_payoffs_not_differences = False
            • apply_exponents_to_payoffs = False

        Mirrors utility() semantics exactly:
            - Self & altruism: centered ratios, sign-preserving exponent on the base
            - SC with negativity: -Ƹ*max(envy,0)^γ + Ƹ*max(guilt,0)^γ
            - Reference constant '3' is NOT exponentiated
        """
        # Read params (accept either γ1/γ₂/γ₃ or γ₁/γ₂/γ₃ spellings)
        def _get(p: dict, *names: str, default: float) -> float:
            for n in names:
                if n in p:
                    return float(p[n])
            return float(default)

        Vii = _get(params, "Vᵢᵢ", "Vii", default=1.0)
        Vij = _get(params, "Vᵢⱼ", "Vij", default=0.0)
        Eps = _get(params, "Ƹᵢⱼ", "Ƹij", default=0.0)

        g1  = _get(params, "γ₁", "γ1", default=1.0)
        g2  = _get(params, "γ₂", "γ2", default=g1 if utility_settings.get("single_exponential_parameter", False) else g1)
        if not utility_settings.get("single_exponential_parameter", False):
            g2 = _get(params, "γ₂", "γ2", default=g1)
        g3  = _get(params, "γ₃", "γ3", default=g1)

        # Payoffs
        Ai = float(payoffs["As"]); Aj = float(payoffs["Ao"])
        # Reference-dependent utility ⇒ compare to 3 for single-agent ratios
        ref_const = 3.0

        # Sign-preserving power for centered ratios (matches code semantics)
        def _signed_pow(x: float, g: float) -> float:
            if x == 0.0:
                return 0.0
            return (abs(x) ** g) * (1.0 if x >= 0.0 else -1.0)

        # Bases
        si_base = Ai / (Ai + ref_const) - 0.5
        al_base = Aj / (Aj + ref_const) - 0.5
        envy    = Aj / (Ai + Aj) - 0.5
        guilt   = Ai / (Ai + Aj) - 0.5

        # Terms
        si_weight = 1.0 if utility_settings.get("fix_self_interest_parameter", False) else Vii
        self_interest = si_weight * _signed_pow(si_base, g1)
        altruism      = Vij * _signed_pow(al_base, g2)
        social_comp   = (-Eps) * (max(envy, 0.0) ** g3) + ( Eps) * (max(guilt, 0.0) ** g3)

        return round(self_interest + altruism + social_comp, decimals_local)

    def _is_token_char(ch: str) -> bool:
        return ch not in " \t\r\n,^*/+-()"

    def _find_left_operand(expr: str, caret_index: int) -> tuple[int, int]:
        scan_index = caret_index - 1
        while scan_index >= 0 and expr[scan_index].isspace():
            scan_index -= 1
        if scan_index >= 0 and expr[scan_index] == ")":
            depth = 1
            scan_index -= 1
            while scan_index >= 0 and depth > 0:
                if expr[scan_index] == ")":
                    depth += 1
                elif expr[scan_index] == "(":
                    depth -= 1
                scan_index -= 1
            start_index = scan_index + 1
            end_index = caret_index
            # include function name if present (e.g., max(...))
            name_end = start_index
            name_start = name_end - 1
            while name_start >= 0 and expr[name_start].isalpha():
                name_start -= 1
            name_start += 1
            if name_start < name_end and expr[name_end] == "(":
                start_index = name_start
            return start_index, end_index
        # bare token
        token_end = scan_index + 1
        token_start = scan_index
        while token_start >= 0 and _is_token_char(expr[token_start]):
            token_start -= 1
        token_start += 1
        return token_start, token_end

    def _find_right_operand(expr: str, caret_index: int) -> tuple[int, int]:
        scan_index = caret_index + 1
        n_chars = len(expr)
        while scan_index < n_chars and expr[scan_index].isspace():
            scan_index += 1
        if scan_index < n_chars and expr[scan_index] == "(":
            depth = 1
            scan_index += 1
            while scan_index < n_chars and depth > 0:
                if expr[scan_index] == "(":
                    depth += 1
                elif expr[scan_index] == ")":
                    depth -= 1
                scan_index += 1
            return caret_index + 1, scan_index
        # bare token exponent
        token_start = scan_index
        while scan_index < n_chars and _is_token_char(expr[scan_index]):
            scan_index += 1
        return token_start, scan_index

    def _replace_powers(expr: str) -> str:
        out = expr
        while "^" in out:
            caret_index = out.find("^")
            base_start, base_end = _find_left_operand(out, caret_index)
            exp_start, exp_end = _find_right_operand(out, caret_index)
            base_txt = out[base_start:base_end].strip()
            exp_txt = out[exp_start:exp_end].strip()
            out = out[:base_start] + f"pow_signed({base_txt}, {exp_txt})" + out[exp_end:]
        return out

    def _normalize_for_eval(rhs_text: str) -> str:
        normalized = (rhs_text
            .replace("\u00A0", " ")
            .replace("−", "-").replace("–", "-").replace("—", "-")
            .replace("≥", ">=").replace("≤", "<=").replace("≠", "!=")
            .replace("×", "*").replace("·", "*").replace("⋅", "*")
        )
        normalized = normalized.replace("[", "(").replace("]", ")")
        # implicit multiplication
        normalized = re.sub(r"(?<![A-Za-z0-9_])(\-?\d+(?:\.\d+)?)\s*\(", r"\1*(", normalized)
        normalized = normalized.replace(")(", ")*(")
        normalized = re.sub(r"\)\s*(\-?\d+(?:\.\d+)?)", r")*\1", normalized)
        return _replace_powers(normalized)

    def _pow_signed(base_value: float, exponent_value: float) -> float:
        base_value = float(base_value); exponent_value = float(exponent_value)
        if abs(exponent_value - round(exponent_value)) < 1e-12:
            return base_value ** int(round(exponent_value))
        return (abs(base_value) ** exponent_value) * (1.0 if base_value >= 0.0 else -1.0)

    def _evaluate_equation_numeric(
        equation_string: str,
        params: dict[str, float],
        payoffs: dict[str, float],
        utility_settings: dict[str, bool],
        param_overrides: dict[str, float] | None = None,
        decimals_local: int = 6,
    ) -> tuple[float | None, str]:
        """
        Evaluate the RHS of the pretty equation after substituting 'params' and 'payoffs'.
        'param_overrides' (if provided) override parameter values *before* substitution.
        Returns (value_or_None, status_text). Status empty string on success.
        """
        # 1) Build replacements
        gamma_pretty = {"γ1": "γ₁", "γ2": "γ₂", "γ3": "γ₃"}
        params_pretty = {gamma_pretty.get(k, k): v for k, v in params.items()}
        # Map aliases that might appear
        alias_to_pretty = {
            "Vii": "Vᵢᵢ", "Λii": "Ʌᵢᵢ", "Vij": "Vᵢⱼ", "Λij": "Ʌᵢⱼ",
            "Ƹij": "Ƹᵢⱼ", "Ʒij": "Ʒᵢⱼ",
            "γ1": "γ₁", "γ2": "γ₂", "γ3": "γ₃",
        }
        for alias_key, pretty_key in alias_to_pretty.items():
            if alias_key in params and pretty_key not in params_pretty:
                params_pretty[pretty_key] = params[alias_key]

        # Apply overrides *before* substitution (so numeric values in the string reflect the override)
        if param_overrides:
            for k_raw, v in param_overrides.items():
                k = gamma_pretty.get(k_raw, k_raw)  # allow γ1 vs γ₁ in overrides
                params_pretty[k] = v

        # γ fallbacks inherit γ₁ (like utility())
        gamma1_value = None
        for k_try in ("γ₁", "γ1"):
            if k_try in params_pretty:
                gamma1_value = float(params_pretty[k_try]); break
        if gamma1_value is None: gamma1_value = 1.0

        
        # if utility_settings.get('conditional_welfare_mode'):
        #     print("-", params_pretty)
        #     "Normalize the weights from [-1, 1] to [0, 1]"
        #     if "Vᵢᵢ" in params_pretty:
        #         params_pretty["Vᵢᵢ"] = round((params_pretty["Vᵢᵢ"] + 1) / 2, 4)
        #     if "Vᵢⱼ" in params_pretty:
        #         params_pretty["Vᵢⱼ"] = round((params_pretty["Vᵢⱼ"] + 1) / 2, 4)
        #     if "Λᵢᵢ" in params_pretty:
        #         params_pretty["Λᵢᵢ"] = round((params_pretty["Λᵢᵢ"] + 1) / 2, 4)
        #     if "Λᵢⱼ" in params_pretty:
        #         params_pretty["Λᵢⱼ"] = round((params_pretty["Λᵢⱼ"] + 1) / 2, 4)

        #     print("+", params_pretty)

        payoff_symbol_map = {"As": "πᵢᴬ", "Ao": "πⱼᴬ", "Bs": "πᵢᴮ", "Bo": "πⱼᴮ"}
        payoffs_pretty_map = {payoff_symbol_map[k]: v for k, v in payoffs.items()}

        # 2) Extract RHS
        if "=" not in equation_string:
            return None, "EVAL ERROR: no '=' in equation"
        _, rhs_original = equation_string.split("=", 1)
        rhs_original = rhs_original.strip()

        # 3) Perform substitutions (payoffs first, then params)
        rhs_filled = rhs_original
        for sym, val in payoffs_pretty_map.items():
            rhs_filled = rhs_filled.replace(sym, str(val))
        for sym, val in params_pretty.items():
            rhs_filled = rhs_filled.replace(sym, str(val))

        # Replace any remaining canonical symbols with *utility()* defaults
        defaults = {
            "Vᵢᵢ": 1.0, "Ʌᵢᵢ": 0.0, "Vᵢⱼ": 0.0, "Ʌᵢⱼ": 0.0, "Ƹᵢⱼ": 0.0, "Ʒᵢⱼ": 0.0,
            "γ₁": gamma1_value, "γ₂": gamma1_value, "γ₃": gamma1_value,
            "Vii": 1.0, "Λii": 0.0, "Vij": 0.0, "Λij": 0.0, "Ƹij": 0.0, "Ʒij": 0.0,
            "γ1": gamma1_value, "γ2": gamma1_value, "γ3": gamma1_value,
        }
        for sym, val in defaults.items():
            if sym in rhs_filled:
                rhs_filled = rhs_filled.replace(sym, str(val))

        # --- INSERT C2: stubborn-case direct evaluator short-circuit ---------------
        s = utility_settings  # alias
        is_stubborn = (
            (not s.get("conditional_welfare_mode", False))
            and (not s.get("min_max_rawlsian_leontief", False))
            and s.get("use_exponential_parameters", False)
            and s.get("payoff_ratios_not_differences", False)
            and s.get("reference_dependent_utility", False)
            and (not s.get("use_negativity_parameters", False))
            and s.get("negativity_social_comparison", False)
            and s.get("include_social_comparison", False) is not False
            and s.get("include_altruism_term", False) is not False
            and (not s.get("single_payoffs_not_differences", False))
            and (not s.get("apply_exponents_to_payoffs", False))
        )
        if is_stubborn:
            try:
                # honor overrides (used to isolate components in verification)
                params_for_direct = dict(params)
                if param_overrides:
                    params_for_direct.update(param_overrides)
                direct_val = _direct_eval_ratio_refdep_negsc(
                    utility_settings=s,
                    params=params_for_direct,
                    payoffs=payoffs,
                    decimals_local=decimals_local,
                )
                return direct_val, ""  # short-circuit: trust direct evaluation
            except Exception as _err_direct:
                # fall through to the generic string-eval path if something unexpected happens
                pass

        # 4) Normalize to Python and eval
        python_rhs = _normalize_for_eval(rhs_filled)
        safe_env = {"__builtins__": {}, "max": max, "min": min, "abs": abs, "pow_signed": _pow_signed}
        try:
            value = float(eval(python_rhs, safe_env, {}))
            return round(value, decimals_local), ""
        except Exception as err:
            return None, f"EVAL ERROR: {type(err).__name__}: {err}"

    # Helper to get string-components via re-evaluation with zeroed weights
    def _string_components(
        equation_string: str,
        params: dict[str, float],
        payoffs: dict[str, float],
        utility_settings: dict[str, bool],
        decimals_local: int = 6,
    ) -> tuple[float | None, float | None, float | None, float | None, dict[str, str]]:
        """
        Returns (total, self, altruism, socc, statuses) where statuses is a dict of
        status messages (empty on success). Altruism and socc are computed by difference
        using re-evaluations with weight overrides, so this also works when fix_self=True.
        """
        statuses: dict[str, str] = {}
        total, st_total = _evaluate_equation_numeric(equation_string, params, payoffs, utility_settings, None, decimals_local)
        statuses["total"] = st_total

        # isolate self by zeroing altruism & SC weights
        self_only_over = {"Vᵢⱼ": 0.0, "Ʌᵢⱼ": 0.0, "Ƹᵢⱼ": 0.0, "Ʒᵢⱼ": 0.0}
        self_only, st_self = _evaluate_equation_numeric(equation_string, params, payoffs, utility_settings, self_only_over, decimals_local)
        statuses["self"] = st_self

        # turn off SC to isolate (self + altruism)
        no_sc_over = {"Ƹᵢⱼ": 0.0, "Ʒᵢⱼ": 0.0}
        no_sc, st_nosc = _evaluate_equation_numeric(equation_string, params, payoffs, utility_settings, no_sc_over, decimals_local)
        statuses["no_sc"] = st_nosc

        if (total is None) or (self_only is None) or (no_sc is None):
            return total, None, None, None, statuses

        altruism_only = round(no_sc - self_only, decimals_local)
        socc_only = round(total - self_only - altruism_only, decimals_local)
        return total, self_only, altruism_only, socc_only, statuses

    # ---------- (5) Main sweep ----------------------------------------------------
    rows: list[dict] = []
    if verbose:
        print(f"[Verify] Utility families to test: {len(all_settings)}")
        print(f"[Verify] Payoff tuples: {'625 exhaustive grid' if use_exhaustive else n_games}")

    for utility_index, utility_settings in enumerate(all_settings, start=1):
        # Sample a fresh mean-parameter vector for this utility family
        params_means = _sample_means_for(utility_settings)

        # Build the pretty equation string once per family
        try:
            equation_string = utility_function_str(utility_settings=utility_settings, in_latex=False, option=option)
        except Exception as err:
            # If the string builder fails, record rows with an eval error status
            for payoff_index, payoffs in enumerate(payoff_iterable, start=1):
                row = {
                    "utility_idx": utility_index,
                    "payoff_idx": payoff_index,
                    "payoff_A_self": payoffs["As"], "payoff_A_other": payoffs["Ao"],
                    "payoff_B_self": payoffs["Bs"], "payoff_B_other": payoffs["Bo"],
                    "U_function": f"Uᵢ({option})",
                    # code-side values are still available:
                    "utility_code": None, "utility_str": None, "utility_Δ": None, "match": False,
                    "status": f"BUILD ERROR: {type(err).__name__}: {err}",
                }
                # Expand 13 boolean flags as separate columns
                for flag_key in ordered_flag_keys:
                    row[flag_key] = bool(utility_settings.get(flag_key, False))
                rows.append(row)
            continue

        for payoff_index, payoffs in enumerate(payoff_iterable, start=1):
            # ----- Code (source of truth), with components
            try:
                code_components = utility_function(payoffs=payoffs, params=params_means,
                                                   utility_settings=utility_settings, separate_terms=True, normalize_conditional_welfare_params=False)
                code_total = float(code_components["self_interest"] + code_components["altruism"] + code_components["social_comp"])
                code_self  = float(code_components["self_interest"])
                code_altr  = float(code_components["altruism"])
                code_socc  = float(code_components["social_comp"])
            except Exception as err:
                # If code throws, give a row so it is visible
                row = {
                    "utility_idx": utility_index,
                    "payoff_idx": payoff_index,
                    "payoff_A_self": payoffs["As"], "payoff_A_other": payoffs["Ao"],
                    "payoff_B_self": payoffs["Bs"], "payoff_B_other": payoffs["Bo"],
                    "U_function": f"Uᵢ({option})",
                    "utility_code": None, "utility_str": None, "utility_Δ": None, "match": False,
                    "status": f"CODE ERROR: {type(err).__name__}: {err}",
                }
                for flag_key in ordered_flag_keys:
                    row[flag_key] = bool(utility_settings.get(flag_key, False))
                rows.append(row)
                continue

            # ----- String (numeric), total + components via re-evaluation
            str_total, str_self, str_altr, str_socc, eval_status = _string_components(
                equation_string=equation_string,
                params=params_means,
                payoffs=payoffs,
                utility_settings=utility_settings,
                decimals_local=decimals,
            )

            # Decide match status on totals (we also store per-term deltas)
            if str_total is None:
                match_flag = False
                delta_total = None
                status_text = eval_status.get("total", "")
            else:
                delta_total = float(code_total - str_total)
                match_flag = abs(delta_total) <= comparison_tol
                status_text = ""

            # Prepare output row
            row = {
                "utility_idx": utility_index,
                "payoff_idx": payoff_index,
                # 13 booleans as separate columns
                **{flag_key: bool(utility_settings.get(flag_key, False)) for flag_key in ordered_flag_keys},
                # payoffs
                "payoff_A_self": payoffs["As"], "payoff_A_other": payoffs["Ao"],
                "payoff_B_self": payoffs["Bs"], "payoff_B_other": payoffs["Bo"],
               
                # totals
                "utility_code": round(code_total, decimals),
                "utility_str":  (None if str_total is None else round(str_total, decimals)),
                "utility_Δ":    (None if delta_total is None else round(delta_total, decimals)),
                "match":        bool(match_flag),
                # components (rounded for readability; raw logic used totals for matching)
                "code_self": round(code_self, decimals),
                "code_altr": round(code_altr, decimals),
                "code_socc": round(code_socc, decimals),
                "str_self":  (None if str_self is None else round(str_self, decimals)),
                "str_altr":  (None if str_altr is None else round(str_altr, decimals)),
                "str_socc":  (None if str_socc is None else round(str_socc, decimals)),
                "Δ_self":    (None if (str_self is None) else round(code_self - str_self, decimals)),
                "Δ_altr":    (None if (str_altr is None) else round(code_altr - str_altr, decimals)),
                "Δ_socc":    (None if (str_socc is None) else round(code_socc - str_socc, decimals)),
                "status": status_text or eval_status.get("self","") or eval_status.get("no_sc",""),
                "U_function": utility_function_str(utility_settings),
                # "equation": utility_function_str(utility_settings)
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # ---------- (6) Sorting and outputs -------------------------------------------
    # Sort by mismatch first, then by absolute total discrepancy
    if not df.empty:
        df["abs_Δ"] = df["utility_Δ"].abs() if df["utility_Δ"].notna().any() else 0.0
        df = df.sort_values(by=["match", "abs_Δ"], ascending=[True, False]).drop(columns=["abs_Δ"])
    try: df.to_csv(out_path, index=False, encoding="utf-8-sig")
    except (PermissionError, OSError): pass

    # Per-utility summary when exhaustive grid was used
    if use_exhaustive and not df.empty:
        summary = (
            df.groupby("utility_idx", as_index=False)
            .agg(
                all_match=("match", "all"),
                n_rows=("match", "size"),
                max_abs_Δ=("utility_Δ", lambda s: float(s.abs().max(skipna=True) if len(s) else 0.0)),
                U_function=("U_function", "first"),
                **{flag_key: (flag_key, "first") for flag_key in ordered_flag_keys}
            )
        )
        summary = summary.sort_values(by=["all_match"], ascending=[True])
        "Move U_function to the end"
        summary = summary[[col for col in summary.columns if col != "U_function"] + ["U_function"]]
        "Turn boolean flags into 1s and 0s to see more easily"
        summary[ordered_flag_keys] = summary[ordered_flag_keys].astype(int)
        summary.drop(columns=['n_rows'], inplace=True)
        try: summary.to_csv(out_summary_path, index=False, encoding="utf-8-sig")
        except (PermissionError, OSError): pass

    # ---------- (7) Console report -------------------------------------------------
    if verbose and not df.empty:
        total_rows = len(df)
        n_match = int(df["match"].sum())
        n_mismatch = total_rows - n_match
        print(f"[Verify] Rows: {total_rows}  |  matches: {n_match}  |  mismatches: {n_mismatch}")
        # Per-flag mismatch rates (top 6 most predictive)
        try:
            flag_reports = []
            for flag_key in ordered_flag_keys:
                grp = df.groupby(flag_key)["match"].agg(total="count", ok="sum")
                grp["mismatch_rate"] = 1.0 - (grp["ok"] / grp["total"])
                # store both levels
                for flag_value, rec in grp.iterrows():
                    flag_reports.append({
                        "flag": flag_key, "value": bool(flag_value),
                        "total": int(rec["total"]),
                        "mismatch_rate": float(rec["mismatch_rate"])
                    })
            rpt = pd.DataFrame(flag_reports)
            rpt = rpt.sort_values(by=["mismatch_rate", "total"], ascending=[False, False])
            print("[Verify] Flags with highest mismatch rates (top 8):")
            print(rpt.head(8).to_string(index=False))
        except Exception:
            pass

        # Top offenders (first 10 mismatches)
        if n_mismatch:
            offenders = df.loc[~df["match"]].head(10)
            cols = ["utility_idx", "payoff_idx", "utility_Δ"] + ordered_flag_keys[:5]  # compact preview
            print("[Verify] First mismatches (preview):")
            print(offenders[cols].to_string(index=False))

        print(f"[Verify] Wrote detailed CSV to: {out_path}")
        if use_exhaustive and not df.empty:
            print(f"[Verify] Wrote per-utility summary to: {out_summary_path}")
            all_match_counts = summary['all_match'].value_counts().to_dict()
            print(f"Equations that always match: {all_match_counts}.")

    return df


def model_nesting_adjacency_matrices(general_settings: GeneralSettings, utility_settings: UtilitySettings, file_paths: FilePaths, 
                                     create_new_file: bool | None = None, equation_form: bool = True, print_: bool = False) -> dict[str: list[list[int]] | list[dict[str, bool]] | list[str]]:
    """
    Creates adjacency matrices indicating pairwise relationships between models: 
    Is the row model a parent of, a sibling of, or a parent of the column model?

    Arguments:
        • utility_settings: dict[str, bool]; Defines the functional form of the utility function.
        • file_paths: dict[str, str | dict[str, str]]; Stores all file paths for storing data.
        • general_settings: dict[str, Any]; Various settings used throughout this analysis.
        • create_new_file: bool | None; 
            - If True, overrides the general setting with True
            - If False, overrides the general setting with False
            - If None, defers to general_settings['create_new_file']
        • print_: bool; If True, prints progress to the terminal. 

    Returns:
        • dict[str: np.array | list[dict[str, bool]] | list[str]]; Example = {
            'adjacency_lists': {
                'parent_of':  [[ 12],
                               [  9,  13], 
                               [ 63, 169, 171], 
                               [111, 112, 173, 174]], 
                'sibling_of': [[  0,   1,   2],
                               [ 55,  56,  57,  58], 
                               [ 59,  60,  61,  62], 
                               [  3,   4]],
                'child_of':   [[245, 346],
                               [311, 356, 377, 385], 
                               [312, 357, 378, 386], 
                               [313, 358, 370, 387]],            
            },
            'adjacency_matrices': {
                'parent_of':  [[0, 0, 0, 0],
                               [0, 0, 0, 0], 
                               [1, 1, 0, 0], 
                               [0, 0, 0, 0]], 
                'sibling_of': [[0, 1, 0, 0],
                               [1, 0, 0, 0], 
                               [0, 0, 0, 1], 
                               [0, 0, 1, 0]],
                'child_of':   [[0, 0, 1, 0],
                               [0, 0, 1, 0], 
                               [0, 0, 0, 0], 
                               [0, 0, 0, 0]],
            },
            'settings': list[dict[str, bool]],
            'equations': list[str]
        }
    """
    "Return existing model nesting data if possible and desired."
    model_nesting_file_path = prep.ensure_directory_and_join(
            file_paths["processed"], "model_nesting_data.json")

    "Determine whether to create a new file or extract a preexisting file."
    if not isinstance(create_new_file, bool):
        create_new_file = general_settings.get('create_new_file')

    if not create_new_file and os.path.exists(model_nesting_file_path):
        with open(model_nesting_file_path, "r", encoding="utf-8") as file:
            model_nesting_data = json.load(file) 
            model_nesting_data['settings'] = [
                gnrl.convert_utility_settings(utility_settings=settings, into=dict) 
                for settings in model_nesting_data['settings']
            ]

        if equation_form and 'adjacency_dict' not in model_nesting_data:
            equations = model_nesting_data['equations']
            settings_list = model_nesting_data['settings']
            equation_dict, adjacency_dict = {}, {}
            for relation in ('parent_of', 'sibling_of', 'child_of'):
                equation_dict[relation] = {}
                adjacency_dict[relation] = {}
                for idx, equation in enumerate(equations):
                    adjacency_list_model = model_nesting_data['adjacency_lists'][relation][idx]
                    adjacency_equations = [equations[edx] for edx in adjacency_list_model]
                    equation_dict[relation][equation] = adjacency_equations
                for idx, settings in enumerate(settings_list):
                    settings = gnrl.convert_utility_settings(settings, tuple)
                    adjacency_list_model = model_nesting_data['adjacency_lists'][relation][idx]
                    adjacent_settings = [settings_list[sdx] for sdx in adjacency_list_model]
                    adjacency_dict[relation][settings] = adjacent_settings
            model_nesting_data['equation_dict'] = equation_dict
            model_nesting_data['adjacency_dict'] = adjacency_dict

        return model_nesting_data

    "Generate all utility funtion settings"
    utility_setting_varieties = gnrl.generate_utility_settings(utility_settings=utility_settings)

    "Generate utility function equations for viewing"
    equations = [build_utility_equation(utility_settings=settings) for settings in utility_setting_varieties]
    
    "Dictionary of all data"
    model_nesting_data = {
        'adjacency_lists': {},
        'adjacency_matrices': {}, 
        'settings': utility_setting_varieties, 
        'equations': equations
    }

    "Create empty adjacency matrices for all three types of relations"
    matrix_keys = ('parent_of', 'sibling_of', 'child_of')
    n_utility_setting_varieties = len(utility_setting_varieties)
    for matrix_key in matrix_keys:
        model_nesting_data['adjacency_matrices'][matrix_key] = np.zeros(
            (n_utility_setting_varieties, n_utility_setting_varieties))

    "Filling in the adjacency matrices for all three relationship types."
    for utility_row_idx in range(n_utility_setting_varieties):
        utility_setting_row = utility_setting_varieties[utility_row_idx]
        if print_:
            print(f"Row Function {utility_row_idx:03d}: {build_utility_equation(utility_settings=utility_setting_row)}")
        for utility_col_idx in range(n_utility_setting_varieties):
            utility_setting_col = utility_setting_varieties[utility_col_idx]

            "Determine family relationship between row and column models."
            relations = gnrl.classify_pair_relation(model_1=utility_setting_row, 
                                                    model_2=utility_setting_col, 
                                                    general_settings=general_settings,
                                                    utility_settings=utility_settings)

            relation_row_to_col, relation_col_to_row, setting_flipped = relations

            "1s mean that the utility function indexed by the row is a parent,"
            " sibling, or child of the utility function indexed by the column."
            if setting_flipped in ("min_max_rawlsian_leontief", "conditional_welfare_mode"):
                "Flipping these settings does not differentiate child from parent"
                continue
            if relation_row_to_col == 'parent':
                model_nesting_data['adjacency_matrices']["child_of"][utility_row_idx][utility_col_idx] = 1
            if relation_row_to_col == 'sibling':
                model_nesting_data['adjacency_matrices']["sibling_of"][utility_row_idx][utility_col_idx] = 1
            if relation_row_to_col == 'child':
                model_nesting_data['adjacency_matrices']["parent_of"][utility_row_idx][utility_col_idx] = 1

    "Mapping utility function indices to the indices of their parents, siblings, and children."
    for matrix_key in matrix_keys:
        model_nesting_data['adjacency_matrices'][matrix_key] = model_nesting_data['adjacency_matrices'][matrix_key].tolist()
        model_nesting_data['adjacency_lists'][matrix_key] = []        
        for utility_row_idx in range(n_utility_setting_varieties):
            if print_:
                utility_setting_row = utility_setting_varieties[utility_row_idx]
                print(f"{matrix_key.capitalize()} Function {utility_row_idx:03d}: "
                      f"{build_utility_equation(utility_settings=utility_setting_row)}")            
            model_nesting_data['adjacency_lists'][matrix_key].append([
                utility_col_idx for utility_col_idx, is_relative in enumerate(
                    model_nesting_data['adjacency_matrices'][matrix_key][utility_row_idx]) if bool(is_relative)
            ])

    if equation_form:
        equations = model_nesting_data['equations']
        settings_list = model_nesting_data['settings']
        equation_dict, adjacency_dict = {}, {}
        for relation in ('parent_of', 'sibling_of', 'child_of'):
            equation_dict[relation] = {}
            adjacency_dict[relation] = {}
            for idx, equation in enumerate(equations):
                adjacency_list_model = model_nesting_data['adjacency_lists'][relation][idx]
                adjacency_equations = [equations[edx] for edx in adjacency_list_model]
                equation_dict[relation][equation] = adjacency_equations
            for idx, settings in enumerate(settings_list):
                settings = str(gnrl.convert_utility_settings(settings, tuple))
                adjacency_list_model = model_nesting_data['adjacency_lists'][relation][idx]
                adjacent_settings = [settings_list[sdx] for sdx in adjacency_list_model]
                adjacency_dict[relation][settings] = adjacent_settings
        model_nesting_data['equation_dict'] = equation_dict
        model_nesting_data['adjacency_dict'] = adjacency_dict

    "Save the data."
    model_nesting_data_compact = copy.deepcopy(model_nesting_data)
    model_nesting_data_compact['settings'] = [
        (int(setting) for setting in gnrl.convert_utility_settings(utility_settings=settings, into=tuple)) 
        for settings in model_nesting_data_compact['settings']
    ]
    with open(model_nesting_file_path, 'w', encoding='utf-8') as file:
        json.dump(model_nesting_data, file, ensure_ascii=False, indent=4)

    if print_:
        print(f"Saved {model_nesting_file_path}")

    return model_nesting_data


def select_child_params_for_parent(children: List[Dict[str, Any]], temperature: float) -> Dict[str, Any]:
    """
    Selects one child-entry (from the list returned by calling 'best_fitting_child_parameters_for_parent'
    for each candidate child) using a *reverse*-SoftMax over totals (lower loss ⇒ higher probability).

    Arguments:
        • children: list of child info dicts (each must have metadata['loss_total']).
        • temperature: float; stochasticity control.
            – If temperature <= 0: choose the smallest 'loss_total' deterministically.
            – Else: p(child) ∝ exp( - loss_total / temperature ).

    Returns:
        • dict[str, Any]; the chosen child dict (unchanged).
    """
    "Extract usable loss totals from the children."
    loss_totals = [(idx, child["metadata"].get("loss_total", None)) for idx, child in enumerate(children)]
    usable_loss_totals = [(idx, float(loss_total)) for (idx, loss_total) in loss_totals if isinstance(loss_total, (int, float))]
    if not usable_loss_totals:
        "If totals are missing (e.g., only per-player subsets), fall back to the first child."
        return children[0] if children else {}

    if temperature is None or temperature <= 0:
        idx = min(usable_loss_totals, key=lambda loss_total: loss_total[1])[0]
        return children[idx]

    "Reverse-softmax weights: exp(-loss / T)"
    losses = [loss_total for (_, loss_total) in usable_loss_totals]
    min_loss = min(losses)

    "Numerical stability trick: subtract min_loss"
    weights = [math.exp(-(loss_total - min_loss) / float(temperature)) for (_, loss_total) in usable_loss_totals]
    total_weights = sum(weights) or 1.0
    probs = [weight / total_weights for weight in weights]

    "Sample one index according to probs"
    rando = random.random()
    cdf = 0.0
    for (idx, _), prob in zip(usable_loss_totals, probs):
        cdf += prob
        if rando <= cdf:
            return children[idx]
        
    "Return fallback"
    return children[usable_loss_totals[-1][0]]  


def best_fitting_model_parameters(utility_settings: UtilitySettings, general_settings: GeneralSettings, file_paths: FilePaths, param_bds: ParamBounds, 
                                  within_ic_analysis: bool = True, *, player_uuid: Optional[str] = None, player_role: Optional[str] = None) -> Dict[str, Any]:
    """
    Extracts the best-fitting parameters for a utility function after optimization.

    Arguments:
        • utility_settings: dict[str, bool]; Defines the functional form of the utility function.
        • file_paths: dict[str, str | dict[str, str]]; Stores all file paths for storing data.
        • general_settings: dict[str, Any]; Various settings used throughout this analysis.
        • within_ic_analysis: bool;
            - If True, extracts data from iterations of the IC analysis in file_paths > bic_aic
            - If False, extracts data from  file_paths > player_fits > experiment_n
        • player_uuid: Optional[str]; If provided, filters to a single player.
        • player_role: Optional[str]; If provided, filters to a single role ('chooser'|'predictor').

    Returns:
        • dict[str: Any]; Example = {
            'player_uuid': {
                'params': {
                    'chooser': {
                        'Vᵢᵢ': 0.73,
                        'Vᵢⱼ': 0.44,
                        'γ1':  1.23
                    }
                    'predictor': {
                        'Vᵢᵢ': 0.81,
                        'Vᵢⱼ': 0.67,
                        'γ1':  0.96
                    }
                },
                'loss': {
                    'chooser': 1.2583459, 
                    'predictor': 0.9832543
                }
            },
            'player_uuid': {
                ...
            },...
        }
    """
    def _extract_from_player_fit_file(player_uuid: str, player_role: str | None, general_settings: GeneralSettings, 
                                      utility_settings: UtilitySettings, print_: bool = False) -> Dict[str, Any] | None:
        """
        Reads a single per-player fit JSON and extracts:
        - player_uuid
        - per-role params (means/exponents/etc.)
        - per-role final losses

        Returns:
            {
                "params": {"chooser": {...}, "predictor": {...}},
                "loss":   {"chooser": float, "predictor": float}
            }
        """
        "Tuple of player roles of interest if not interested in both roles."
        player_roles = (player_role,) if player_role in ('chooser', 'predictor') else ('chooser', 'predictor')  

        "Standardize general settings."
        general_settings = copy.deepcopy(general_settings)
        general_settings['temperature_is_param'] = False
        experiment_num = int(general_settings.get("experiment_num", 3))

        "Generate file path for this player based on the general and utility settings."
        pf_base = os.path.join(file_paths.get("player_fits", "."), f"experiment_{experiment_num}")
        if not os.path.isdir(pf_base):
            raise FileNotFoundError(f"Per-player fits directory not found: {pf_base!r}")        
        
        pf_file_name = prep.create_file_name_suffix(
            general_settings=general_settings, utility_settings=utility_settings) + f"_{player_uuid}.json"         
        player_file_path = prep.ensure_directory_and_join(base_dir=pf_base, file_name=pf_file_name)

        "Extract data if it exists."
        try:
            with open(player_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError as err:
            if print_: print(err)
            return None

        "Locate the param and loss data in the first game of the first dyad."
        dyad_keys = list(data.keys())
        if not dyad_keys:
            raise ValueError(f"No dyads found in {player_file_path!r}.")

        first_dyad_key = dyad_keys[0]
        game_list = data[first_dyad_key]
        if not (isinstance(game_list, list) and game_list):
            raise ValueError(f"Malformed dyad entry in {player_file_path!r}.")

        first_game = game_list[0]

        "Find the parameter keys"
        parameter_estimates = first_game.get("parameter_estimates", {})
        parameter_estimates = parameter_estimates.get("naive", parameter_estimates.get("grid", {})).get(player_uuid, {})
        params = parameter_estimates.get("chooser", parameter_estimates.get("predictor", {})).get("params", {})
        param_keys = list(params.keys())

        "Losses live under 'reports' -> <role> -> 'final' -> 'loss'"
        reports = first_game.get("reports", {})

        return {
            "params": {
                player_role: {param_key: param for param_key, param in zip(
                    param_keys, reports.get(player_role, {}).get("final", {}).get("x", []))}
                for player_role in player_roles
            },
            "loss": {
                player_role: reports.get(player_role, {}).get("final", {}).get("loss", None) 
                for player_role in player_roles
            }
        }

    "Normalize inputs"
    general_settings = copy.deepcopy(general_settings)
    if within_ic_analysis:
        general_settings['update_method'] = 'naive'
        general_settings['temperature_is_param'] = False
    experiment_num = int(general_settings.get("experiment_num", 3))

    "Generate a list of all player uuids or just one player if already specified."
    if isinstance(player_uuid, str):
        player_uuids = [player_uuid]
    else:
        player_uuids = prep.all_player_uuids(
            file_paths=file_paths, experiment_num=experiment_num, only_humans=True)        

    "Tuple of player roles of interest if not interested in both roles."
    player_roles = (player_role,) if player_role in ('chooser', 'predictor') else ('chooser', 'predictor')

    results: Dict[str, Any] = {}
    extracted_ic_file = False

    if within_ic_analysis:
        "Load IC Analysis JSON for this model."
        ic_dir = file_paths.get("bic_aic", ".")
        ic_file_name_suffix = prep.create_file_name_suffix(
            general_settings=general_settings, utility_settings=utility_settings)
        ic_file_name = f"IC_Analysis{ic_file_name_suffix}.json"
        ic_file_path = prep.ensure_directory_and_join(base_dir=ic_dir, file_name=ic_file_name)
        if os.path.exists(path=ic_file_path):
            with open(ic_file_path, "r", encoding="utf-8") as file:
                ic_json = json.load(file)
            extracted_ic_file = True

            minvec = ic_json.get("minvec", {})
            if isinstance(minvec, dict) and minvec:
                # New files (with minvec): use the per-player minima directly
                per_player_params = {pl: entry.get('params', {}) for pl, entry in minvec.items()}
            else:
                # Backward compatibility with older IC files (no minvec)
                pvec_list = ic_json.get("pvec", [])
                lvec_list = ic_json.get("lvec", [])
                if not (isinstance(pvec_list, list) and isinstance(lvec_list, list) and pvec_list):
                    raise ValueError(f"IC file {os.path.basename(ic_file_name)} missing 'pvec'/'lvec' iterations.")
                best_idx = min(range(len(lvec_list)), key=lambda idx: float(lvec_list[idx]))
                per_player_params = pvec_list[best_idx]

        else:
            print(
                f"No IC Analysis file found in {ic_dir!r} matching model key {ic_file_name!r}."
            )

        "Extract the data from the IC file for each player."
        for player_uuid in player_uuids:
            data_successfully_extracted = False
            if extracted_ic_file:
                params_and_losses_by_player_role = per_player_params.get(player_uuid, None)
                if isinstance(params_and_losses_by_player_role, dict) and \
                    "params" in params_and_losses_by_player_role and "loss" in params_and_losses_by_player_role:
                    params = params_and_losses_by_player_role.get("params", {}) or {}
                    losses = params_and_losses_by_player_role.get("loss", {}) or {}
                    if params and losses:
                        data_successfully_extracted = True
                        results[player_uuid] = {
                            "params": {player_role: params.get(player_role) for player_role in player_roles},
                            "loss": {player_role: losses.get(player_role) for player_role in player_roles}
                        }

            if not data_successfully_extracted:
                "If the data could not be found in the IC file, extract the data from the player fits file."
                player_param_loss_data = _extract_from_player_fit_file(player_uuid=player_uuid, player_role=player_role,
                                            general_settings=general_settings, utility_settings=utility_settings)  
                if player_param_loss_data is not None:
                    results[player_uuid] = player_param_loss_data              

    else:
        "Extract all data from the player fits file."
        for player_uuid in player_uuids:
            player_param_loss_data = _extract_from_player_fit_file(player_uuid=player_uuid, player_role=player_role,
                                        general_settings=general_settings, utility_settings=utility_settings)  
            if player_param_loss_data is not None:
                results[player_uuid] = player_param_loss_data   

    return results


def best_fitting_child_parameters_for_parent(player_uuid: str | None, player_role: str | None, utility_settings_parent: UtilitySettings, utility_settings: UtilitySettings, 
                                             general_settings: GeneralSettings, file_paths: FilePaths, param_bds: ParamBounds, within_ic_analysis: bool = True, temperature: float = 1.5) -> Dict[str, Any]:
    """
    Provides the best fitting parameters for a *child* utility function to its *parent*
    utility function, as the building block for warm starts when optimizing the parent.

    Arguments:
        • player_uuid: str | None; The specific player to extract 
            parameters for. If None, extracts for all players.
        • player_role: str | None; The specific player role to extract 
            parameters for. If None, extracts for all roles.
        • utility_settings_parent: dict[str, bool]; Defines the functional form of the parent.
        • utility_settings: dict[str, bool]; Defines the functional form of the utility function.
        • file_paths: dict[str, str | dict[str, str]]; Stores all file paths for storing data.
        • general_settings: dict[str, Any]; Various settings used throughout this analysis.
        • within_ic_analysis: bool;
            - If True, extracts data from iterations of the IC analysis in file_paths > bic_aic
            - If False, extracts data from  file_paths > player_fits > experiment_n
        • temperature: float; Lower values increase the chances of the lowest loss child 
            being selected. Higher values make the selection more random.

    Returns:
        • dict[str, Any]; Example:
            {
              "metadata": {
                "U_funct": "Uᵢ(A) = ...",
                "utility_settings": {...},     # child settings
                "loss_total": 274.462100798,   # iteration-level total (IC) or sum of role-losses when available
                "source": "IC" or "player_fits",
                "model_key": "1111101..."
              },
              "data": {
                 "player_uuid": {
                   "params":  {"chooser": {...}, "predictor": {...}},
                   "loss":    {"chooser": <float|None>, "predictor": <float|None>}
                 },
                 ...
              }
            }
    """
    def _fallback_random_parent_guess_from_bounds(param_info: dict, rng: np.random.Generator | None = None) -> dict[str, float]:
        """
        Build a *parent-space* initial guess vector by sampling uniformly within bounds,
        preserving the ordering in param_info['keys'] and param_info['bounds'].
        """
        fallback_params = {}
        # try:
        # Construct an initial parameter dictionary
        if callable(param_info["guesses"]):
            initial_guesses = param_info["guesses"]()
        else:
            initial_guesses = param_info["guesses"]

        for param_key, param_guess in zip(param_info['keys'], initial_guesses):
            fallback_params[param_key] = float(param_guess)

        # except:
            # for param_key, (bd_lo, bd_hi) in zip(param_info['keys'], param_info['bounds']):
            #     # small epsilon off the boundary, matching your other clamps
            #     tiny = 1e-12
            #     if rng is None:
            #         guess_val = random.uniform(bd_lo + tiny, bd_hi - tiny)
            #     else:
            #         guess_val = rng.uniform(bd_lo + tiny, bd_hi - tiny)
            #     fallback_params[param_key] = float(guess_val)

        return fallback_params

    "Normalize inputs"
    general_settings = copy.deepcopy(general_settings)
    if within_ic_analysis:
        general_settings['update_method'] = 'naive'
        general_settings['temperature_is_param'] = False

    "Tuple of player roles of interest if not interested in both roles."
    player_roles = (player_role,) if player_role in ('chooser', 'predictor') else ('chooser', 'predictor')

    "Create a list of utility settings of all children of the parent model."
    parent_equation = build_utility_equation(utility_settings=utility_settings_parent)
    model_nesting_data = model_nesting_adjacency_matrices(general_settings=general_settings, 
                            utility_settings=utility_settings, file_paths=file_paths, create_new_file=False)
    try: parent_idx = model_nesting_data['equations'].index(parent_equation)
    except IndexError as err:
        print(err)
        return None

    child_indices = model_nesting_data['adjacency_lists']['child_of'][parent_idx]
    child_utility_settings = [model_nesting_data['settings'][child_idx] for child_idx in child_indices]

    if child_utility_settings:
        assert len(list(child_utility_settings[0].keys())) == 14

    "Build a list of best fitting child parameters."
    best_fitting_child_params = []
    for child_settings in child_utility_settings:

        child_params_and_losses = best_fitting_model_parameters(
            utility_settings=child_settings,
            general_settings=general_settings,
            file_paths=file_paths, 
            param_bds=param_bds,
            within_ic_analysis=within_ic_analysis,
            player_uuid=player_uuid,
            player_role=player_role,
        )

        "Compute loss total"
        loss_total = 0.0
        for plr_uuid, params_and_losses in child_params_and_losses.items():
            if player_uuid is not None and plr_uuid != player_uuid:
                continue
            losses = params_and_losses.get('loss', {})
            for plr_role in player_roles:
                a_loss = losses.get(plr_role, 0.0)
                if isinstance(a_loss, (int, float)):
                    loss_total += losses.get(plr_role, 0.0) #TypeError: unsupported operand type(s) for +=: 'float' and 'NoneType'

        "Append to the list data and metadata for the child and its parameters."
        child_payload = {
            "metadata": {
                "U_funct": build_utility_equation(utility_settings=child_settings),
                "model_bit_str": gnrl.convert_utility_settings(utility_settings=utility_settings, into=str),
                "utility_settings": child_settings,
                "loss_total": loss_total,
            },
            "data": child_params_and_losses
        }
        best_fitting_child_params.append(child_payload)   

    if not best_fitting_child_params:
        "Fallback to random parameters if the child model could not be found."

        "Generate a list of all player uuids or just one player if already specified."
        if isinstance(player_uuid, str):
            player_uuids = [player_uuid]
        else:
            player_uuids = prep.all_player_uuids(
                file_paths=file_paths, experiment_num=experiment_num, only_humans=True)     

        parent_warmstart = {}
        for plr_uuid in player_uuids:
            parent_warmstart[plr_uuid] = {}
            for plr_role in player_roles:
                plr_param_info = make_param_info(param_bds=param_bds, utility_settings=utility_settings, 
                                                      general_settings=general_settings, guess_seed=None, random_guesses_are_unique=True)
                parent_warmstart[plr_uuid][plr_role] = _fallback_random_parent_guess_from_bounds(param_info=plr_param_info, rng=None)

        # No usable child found → disable warm-start cleanly
        return {
            "parent_warmstart": parent_warmstart,
            "selected_child":   {},
            "metadata": {
                "reason": "no_child_fit_found",
                "parent_equation": build_utility_equation(utility_settings=utility_settings_parent),
                "U_funct": utility_settings_parent,
            }
        }

    "Sort with best fitting children first in the list."
    best_fitting_child_params = sorted(best_fitting_child_params, 
        key = lambda x: x.get('metadata', {}).get('loss_total', 0.0))
    
    "Probabilistically select a child based on the total losses and a temperature parameter."
    selected_child = select_child_params_for_parent(children=best_fitting_child_params, temperature=temperature)
    selected_child_settings = selected_child.get('metadata', {}).get('utility_settings', {})
    selected_child_equation = selected_child.get('metadata', {}).get('U_funct')

    "Ensure that all keys are present"
    setting_keys_all = set(utility_settings.keys())
    setting_keys_child = set(selected_child_settings.keys())
    missing_keys = setting_keys_all - setting_keys_child
    n_missing_keys = len(list(missing_keys))
    max_missing_keys = 2 #Should be lower.
    if n_missing_keys < 0:
        raise ValueError(f"Child model has extra keys: {setting_keys_child}.")
    elif n_missing_keys > 0:
        if n_missing_keys > max_missing_keys:
            pp.pprint(best_fitting_child_params)
            raise ValueError(f"Child model is missing {n_missing_keys} keys: {setting_keys_child}.")
        else:
            for setting_key in list(setting_keys_all):
                if setting_key not in setting_keys_child:
                    val = True if 'single_' in setting_key else False
                    selected_child_settings[setting_key] = val

    "Sanity Check: Confirm that we truly selected a child of the parent."
    relation_1_to_2, relation_2_to_1, flipped_setting = gnrl.classify_pair_relation(model_1=utility_settings_parent, 
                                                            model_2=selected_child_settings, general_settings=general_settings, utility_settings=utility_settings)

    if not (relation_1_to_2 == 'parent' and relation_2_to_1 == 'child'):
        print(f"Parent: {parent_equation}")
        print(f"Child?: {selected_child_equation}")
        if not isinstance(flipped_setting, str):
            flipped_setting = "unknown"
        raise RuntimeError(
            f"Selected model is not a child of the parent. Model 1 = {relation_1_to_2} and Model 2 = {relation_2_to_1}. "
            f"The flipped utility setting is: {flipped_setting.capitalize().replace('_', ' ')}"
        )

    "Map child params -> parent params (add the parent's extra param info etc.)"
    warmstart: Dict[str, Dict[str, Dict[str, float]]] = {}
    for plr_uuid, params_and_losses in selected_child.get('data', {}).items():
        if player_uuid is not None and plr_uuid != player_uuid:
            continue
        warmstart[plr_uuid] = {}
        params_by_role = params_and_losses.get('params', {})
        for plr_role in player_roles:
            fitted_child_params = params_by_role.get(plr_role, {})
            parent_params = gnrl.map_child_to_parent_special_param_info(
                child_utility_settings=child_settings,
                parent_utility_settings=utility_settings_parent,
                child_fitted_parameters=fitted_child_params,
                build_utility_equation=build_utility_equation,
                general_settings=general_settings,
                param_bds=param_bds,
            )
            # warmstart[plr_uuid][plr_role] = parent_params


            parent_param_info = make_param_info(
                param_bds=param_bds, utility_settings=utility_settings_parent,
                general_settings=general_settings, guess_seed=None,
                random_guesses_are_unique=True
            )
            if callable(parent_param_info['guesses']):
                parent_param_info['guesses'] = parent_param_info['guesses']()
            ordered = [float(parent_params.get(param_key, param_guess))
                    for param_key, param_guess in zip(parent_param_info['keys'], parent_param_info['guesses'])]
            warmstart[plr_uuid][plr_role] = {
                "keys":    parent_param_info['keys'],
                "bounds":  parent_param_info['bounds'],
                "guesses": ordered,
                # (optionally keep the plain mapping too)
                **parent_params
            }



    return {
        "metadata": {
            "parent_equation": parent_equation,
            "parent_settings": utility_settings_parent,
            "selected_child_equation": selected_child_equation,
            "selected_child_settings": selected_child_settings,
            "flipped_setting": flipped_setting
        },
        "selected_child": selected_child,      # retains per-player/role child fit info
        "parent_warmstart": warmstart          # per-player/role dict of mapped parent params
    }


"=========================================================================================="
"============ Results: Joint Parameter Distributions in Human-Human Experiment ============"
"=========================================================================================="

def population_parameter_distribution_df(general_settings: dict[str, Any], file_paths: dict[str, str], player_role: str = 'predictor', 
                 use_initial_params: bool | None = None, create_new_file: bool | None = None) -> pd.DataFrame:
    """
    Generates a dataframe with parameter values for each player for all counterparts.

    Arguments:
        • general_settings: Dict[str, Any]; High-level settings (analysis mode, etc.).
        • file_paths: dict[str, str]; Paths to files/directories for reading/writing data.
        • player_role: str; Determines if extracts chooser or predictor role parameters.
        • use_initial_params: bool | None; If True, uses first-round params. Otheriwse, uses final-round params
        • create_new_file: bool | None; If True, generates new dataframe even if one exists already.

    Returns:
        • pd.DataFrame    
    """
    acceptable_roles = ('chooser', 'predictor')
    if player_role not in acceptable_roles:
        warning_str = f"{player_role}. Must be one of the following: {acceptable_roles}."
        raise ValueError(f"Invalid player_role detected: {warning_str}")

    if use_initial_params is None or not isinstance(use_initial_params, bool):
        use_initial_params = general_settings.get('use_initial_params', True)
    if create_new_file is None or not isinstance(create_new_file, bool):
        create_new_file = general_settings.get('create_new_file', False)
    update_method =  general_settings.get('update_method', 'grid')
    experiment_num = general_settings.get('experiment_num', 3)

    if player_role == 'chooser':
        if not use_initial_params:
            raise ValueError("use_initial_params must be True if player_role == 'chooser'.")
        if experiment_num in (1, 2):
            raise ValueError(f"Cannot extract chooser role parameters in experiment {experiment_num}!")

    file_name = f"Player_Parameters_Exper{experiment_num}_"
    file_name += f"{player_role.capitalize()}_{'First' if use_initial_params else 'Final'}.csv"
    csv_path = prep.ensure_directory_and_join(base_dir=file_paths["processed"], file_name=file_name)
    if not create_new_file and os.path.exists(csv_path):
        df = prep.dataframe(file_path=file_paths["processed"], file_name=file_name)
        if df is not None:
            if "temp" in list(df.columns):
                df = df.rename(columns={"temp": "τ"})
            return df

    "List of all player uuids in the experiment."
    player_uuids = prep.all_player_uuids(
        file_paths=file_paths, experiment_num=experiment_num, only_humans=True)
    n_players = len(player_uuids)
    
    rows = []
    games_idx = 0 if use_initial_params else -1
    if experiment_num == 1: 
        games_idx = -1

    "Iterate through all players appending their parameters to rows."
    for player_idx, player_uuid in enumerate(player_uuids):
        dyads_for_this_player = prep.fitted_dyads_for_a_player(
            player_uuid=player_uuid, experiment_num=experiment_num, file_paths=file_paths)
        if dyads_for_this_player is None:
            print(f"Failed to extract data for player {player_uuid}")
            continue

        print(f'Adding data for player {player_idx+1} / {n_players}: {player_uuid}')
        
        for dyad_key, dyad_games in dyads_for_this_player.items():
            dyad_game: dict[str, dict[str, dict[str, dict[str, dict]]]] = dyad_games[games_idx]
            counterpart_uuid = dyad_game.get('predictor' if player_role == 'chooser' else 'chooser')
            player_data = dyad_game.get('parameter_estimates', {}).get(update_method, {}).get(player_uuid, {}) 

            param_data = player_data.get(player_role, {}).get('params', None)
            if games_idx == -1:
                posteriors = player_data.get(player_role, {}).get('posteriors', None)
                if posteriors is not None:
                    param_data = posteriors

            row = {
                'experiment_num': experiment_num,
                'player_uuid': player_uuid,
                'counterpart_uuid': counterpart_uuid,
                'player_role': player_role,
            }

            for param_key, param_val in param_data.items():
                if param_key == "temp":
                    row["τ"] = param_val
                else:
                    row[param_key] = param_val

            rows.append(row)

    if not rows:
        raise Exception(f"Failed to generate dataframe.")

    df = pd.DataFrame(rows)
    if "temp" in list(df.columns):
        df = df.rename(columns={"temp": "τ"})

    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    return df


def population_parameter_distribution_histograms(general_settings: dict[str, Any], file_paths: dict[str, str], fig_lay: Dict[str, Any], player_role: str = 'predictor', 
                                                 use_initial_params: bool | None = None, create_new_file: bool | None = None) -> go.Figure:
    """
    Visualize histograms of parameter values in df. 
    df should have columns of parameters (e.g. 'Vᵢᵢ', 'Ʌᵢᵢ', ...).

    Arguments:
        • general_settings: Dict[str, Any]; High-level settings (analysis mode, etc.).
        • file_paths: dict[str, str]; Paths to files/directories for reading/writing data.
        • param_info: dict[str, Any]; Contains parameter keys, bounds, guesses, etc.
        • fig_lay: dict[str, Any]; Determines the aesthetic qualities of the figure.

    Returns:
        • go.Figure
    """
    import plotly.figure_factory as ff
    df = population_parameter_distribution_df(general_settings=general_settings, file_paths=file_paths, 
                    player_role=player_role, use_initial_params=use_initial_params, create_new_file=create_new_file)

    param_types = f"{'Initial' if use_initial_params else 'Final'} {player_role.capitalize()}"
    title = f"Distribution of {param_types} Parameter Values "
    file_name = title.replace(" ", "_") + file_paths["file_name_suffix"] + ".html"

    hist_data = {}
    non_params = ('experiment_num', 'player_uuid', 'counterpart_uuid', 'player_role')
    param_keys = [col for col in df.columns if col not in non_params and df[col].nunique() > 1]

    print(f"----------- Mean and Std of Mean {player_role.capitalize()} Parameters -----------")
    for param_key in param_keys:
        if '_std' not in param_key:
            key_str = ""
            for idx in range(13 - len(param_key)):
                key_str += " "
            key_str += param_key
            param_mean = df[param_key].mean()
            param_std  = df[param_key].std()
            mean_gap = "" if param_mean < 0 else " "
            std_gap = ""  if param_std < 0 else  " "
            print_str = f"      {key_str} "
            print_str += f"μ  = {mean_gap}{param_mean:.5f} "
            print_str += f"σ  = {std_gap}{param_std:.5f}"
            print(print_str)

    if general_settings.get('temperature_is_param'):
        param_keys += ["τ"]
        # param_keys += ["temp"]
    for param_key in param_keys:
        if param_key in ("τ", "temp"):
            fancy_key = "τ"
            param_key = "τ"
        elif '_std' in param_key:
            fancy_key = f"σ({param_key.replace('_std', '')})"
        elif '_cov' in param_key:
            fancy_key = f"σ({param_key.replace('_cov', '').replace('_', ',')})"
        else:
            fancy_key = f"μ({param_key})"
        # print(list(df.columns)), exit()
        hist_data[fancy_key] = df[param_key].dropna().values.tolist()

    group_labels = list(sorted(hist_data.keys(), key=lambda x: (
        x.replace("σ(", "").replace("μ(", "").replace(")", ""), "σ(" in x)))
    hist_colors = [f"hsla({int((idx*360/(len(group_labels)+1))%360)}, 100%, 50%, 0.7)" 
                   for idx in range(len(group_labels))]
    hist_sizes = [0.2]*len(group_labels)

    fig = ff.create_distplot([hist_data[key] for key in group_labels], group_labels, bin_size=hist_sizes, 
                             colors=hist_colors, show_curve=True, show_rug=True)
    
    font_info = copy.deepcopy(fig_lay["font"])
    font_info["size"] = 36

    fig.update_layout(title=title, template=fig_lay["template"], hoverlabel=dict(font_size=14), 
                      titlefont_size=fig_lay['titlefont_size']+6, font=font_info,
                      xaxis=dict(range=[-1, 2]))

    "Categorize parameters"
    mean_params = [p for p in group_labels if "σ(" not 
                   in p and "," not in p and 'τ' not in p]
    mean_weight_params = [p for p in mean_params if 'γ' not in p]
    # if player_role == 'chooser':
    #     mean_params = mean_params[:-1]
    std_params = [p for p in group_labels if "σ(" in p and "," not in p]
    cov_params = [p for p in group_labels if "," in p]

    "Identify problematic parameters (e.g., ones clustering into a single bin based on small std deviation)"
    epsilon = 1e-2  # Define a threshold for tiny standard deviation
    problematic_params = []
    for param, data in hist_data.items():
        if len(data) > 1 and pd.Series(data).std() < epsilon:
            problematic_params.append(param)

    "Initial visibility: Hide problematic parameters"
    initial_visibility = [p not in problematic_params for p in group_labels] * 2

    # Set initial visibility at the trace level
    for trace, visible in zip(fig.data, initial_visibility):
        trace.visible = 'legendonly' if not visible else True

    "Dropdown options"
    buttons = [
        dict(label="All", method="update", 
             args=[{"visible": [p not in problematic_params 
                                for p in group_labels] * 2}]),
        dict(label="μ(...)", method="update",
             args=[{"visible": [p in mean_params and p not in 
                                problematic_params for p in group_labels] * 2}]),
        dict(label="μ(𝑤)", method="update",
             args=[{"visible": [p in mean_weight_params and p not in 
                                problematic_params for p in group_labels] * 2}]),                                
        dict(label="σ(...)", method="update",
             args=[{"visible": [p in std_params and p not in 
                                problematic_params for p in group_labels] * 2}]),
    ]
    if general_settings.get('include_covariance'):
        buttons.append(
            dict(label="Cov(x,y)", method="update",
                args=[{"visible": [p in cov_params and p not in 
                                   problematic_params for p in group_labels] * 2}])
        )

    "Add individual parameter selections"
    for idx, group_label in enumerate(group_labels):
        visible = [False] * (2 * len(group_labels))
        visible[idx] = True  # pdf curve for that param
        visible[idx + len(group_labels)] = True  # rug for that param
        buttons.append(dict(label=group_label, method="update", 
                            args=[{"visible": visible}]))

    fig.update_layout(updatemenus=[dict(active=0, x=1.06, y=1.00, buttons=buttons)], 
                      legend={"x": 1.0, "y": 0.96, "font": font_info}, titlefont_size=fig_lay['titlefont_size'], 
                      title_x=fig_lay['title_x'], title_y=fig_lay['title_y'], font=fig_lay['font'])

    if general_settings.get('export_fig', True):
        fig.write_html(os.path.join(file_paths["visuals"], file_name))
        print(f"Saved {file_name}")
    else:
        fig.show()

    return fig


def subpopulation_stats_and_param_ratio_histograms(general_settings: dict[str, any], file_paths: dict[str, str], fig_lay: Dict[str, Any], player_role: str = 'predictor', 
                                use_initial_params: bool | None = None, create_new_file: bool | None = None, ratio_mode: str = "skip_negative", as_subplots: bool = False, print_: bool = True) -> dict:
    """
    Computes:
      1) Subpopulation stats (sadistic, masochistic, guilt>envy, altruism>self-interest),
      2) Two ratio analyses and histograms:
         (a) self-interest vs. altruism (Vᵢᵢ vs. Vᵢⱼ),
         (b) guilt vs. envy (Ʒᵢⱼ vs. Ƹᵢⱼ).

    If as_subplots=False, produces two separate histograms for the given player_role.
    If as_subplots=True, it produces four histograms (both ratio types × both roles) as subplots in a single figure.
    """
    # --- Helper function to load param DF given role (so we can do it for both roles in subplot mode) ---
    def load_and_prepare_df_for_role(the_role: str) -> pd.DataFrame:
        df_temp = population_parameter_distribution_df(
            general_settings=general_settings,
            file_paths=file_paths,
            player_role=the_role,
            use_initial_params=use_initial_params_local,
            create_new_file=create_new_file_local
        )
        if use_initial_params_local:
            df_temp = df_temp.drop_duplicates(subset=['player_uuid']).copy()
        required_cols = ["Vᵢᵢ", "Vᵢⱼ", "Ƹᵢⱼ", "Ʒᵢⱼ"]
        for col in required_cols:
            if col not in df_temp.columns:
                raise ValueError(f"Required column '{col}' not found for role={the_role}. Avail: {df_temp.columns.tolist()}")
        df_temp = df_temp.dropna(subset=required_cols).copy()
        return df_temp

    # --- Helper ratio function as in your original ---
    def compute_ratio_array(series_x, series_y, param_category: str = 'self', normalize: bool = True) -> pd.Series:
        ratio_list = []
        for (x, y) in zip(series_x, series_y):
            if ratio_mode == "skip_negative":
                # keep only if x>0,y>0 => ratio=x/(x+y) for self
                # or x<0,y<0 => ratio=x/(x+y) for guilt (since guilt<0, envy<0)
                if (param_category == 'self'  and x > 0 and y > 0) or \
                   (param_category == 'guilt' and x > 0 and y > 0):
                    denom = (x + y) if normalize else y
                    if abs(denom) > 1e-9:
                        ratio_list.append(x / denom)
            elif ratio_mode == "absolute":
                ax, ay = abs(x), abs(y)
                denom = ax + ay if normalize else ay
                if denom > 1e-9:
                    ratio_list.append(ax / denom)
            else:
                raise ValueError(f"Unknown ratio_mode '{ratio_mode}' (should be skip_negative|absolute)")
        return pd.Series(ratio_list)

    # --- 1) Defaulting logic (unchanged) ---
    if use_initial_params is None or not isinstance(use_initial_params, bool):
        use_initial_params = general_settings.get('use_initial_params', True)
    if create_new_file is None or not isinstance(create_new_file, bool):
        create_new_file = general_settings.get('create_new_file', False)
    export_fig = general_settings.get('export_fig', True)
    experiment_num = general_settings.get('experiment_num', 3)

    if ratio_mode == 'skip_negative':
        x_title_self = "Ratio = 𝑉ᵢᵢ / (𝑉ᵢᵢ + 𝑉ᵢⱼ) (Negative Parameters Excluded)"
        x_title_guilt = "Ratio = Ʒᵢⱼ / (Ʒᵢⱼ + Ƹᵢⱼ) (Negative Parameters Excluded)"
    else:
        x_title_self = "Ratio = |𝑉ᵢᵢ| / (|𝑉ᵢᵢ| + |𝑉ᵢⱼ|)"
        x_title_guilt = "Ratio = |Ʒᵢⱼ| / (|Ʒᵢⱼ| + |Ƹᵢⱼ|)"

    # We store the final results in some dictionary. If as_subplots=True, we gather results for both roles.
    ### SUBPLOTS CHANGES ###
    # We'll keep the code for the single-role flow in an if-block:
    if not as_subplots:
        # ================ ORIGINAL SINGLE-ROLE LOGIC ================

        use_initial_params_local = use_initial_params
        create_new_file_local = create_new_file

        # 2) Load the parameter DataFrame
        df = load_and_prepare_df_for_role(player_role)
        n_data = len(df)
        if n_data == 0:
            print("[Warning] No data remain after dropping NaNs. Returning empty results.")
            return {}

        # 5) Identify subpop membership
        sadists_mask = (df["Vᵢⱼ"] < 0)
        masochists_mask = (df["Vᵢᵢ"] < 0)
        competitive_mask = (df["Ʒᵢⱼ"] < 0)
        depricating_mask = (df["Ƹᵢⱼ"] < 0)
        guilt_over_envy_mask = (df["Ʒᵢⱼ"] > df["Ƹᵢⱼ"])
        altruism_over_self_mask = (df["Vᵢⱼ"] > df["Vᵢᵢ"])

        sadistic_pct = 100.0 * sadists_mask.sum() / n_data
        masochistic_pct = 100.0 * masochists_mask.sum() / n_data
        competitive_pct = 100.0 * competitive_mask.sum() / n_data
        depricating_pct = 100.0 * depricating_mask.sum() / n_data
        guilt_over_envy_pct = 100.0 * guilt_over_envy_mask.sum() / n_data
        altruism_over_self_pct = 100.0 * altruism_over_self_mask.sum() / n_data

        # 7) Compute ratio arrays
        self_ratio_series = compute_ratio_array(df["Vᵢᵢ"], df["Vᵢⱼ"], 'self', True)
        guilt_ratio_series = compute_ratio_array(df["Ʒᵢⱼ"], df["Ƹᵢⱼ"], 'guilt', True)
        self_ratio_series_raw = compute_ratio_array(df["Vᵢᵢ"], df["Vᵢⱼ"], 'self', False)
        guilt_ratio_series_raw = compute_ratio_array(df["Ʒᵢⱼ"], df["Ƹᵢⱼ"], 'guilt', False)

        # Summaries
        n_self_valid = len(self_ratio_series)
        self_ratio_mean = self_ratio_series.mean() if n_self_valid > 0 else float('nan')
        self_ratio_std  = self_ratio_series.std()  if n_self_valid > 1 else float('nan')

        n_guilt_valid = len(guilt_ratio_series)
        guilt_ratio_mean = guilt_ratio_series.mean() if n_guilt_valid > 0 else float('nan')
        guilt_ratio_std  = guilt_ratio_series.std()  if n_guilt_valid > 1 else float('nan')

        self_ratio_mean_raw = self_ratio_series_raw.mean() if n_self_valid > 0 else float('nan')
        guilt_ratio_mean_raw = guilt_ratio_series_raw.mean() if n_guilt_valid > 0 else float('nan')

        # role string
        if player_role == 'chooser':
            role_str = 'Chooser'
        else:
            role_str = ("Prior" if use_initial_params_local else "Posterior") + ' Predictor'

        # Create the two separate histograms
        fig_self = go.Figure()
        fig_self.add_trace(go.Histogram(
            x=self_ratio_series,
            nbinsx=15,
            marker=dict(
                color='hsla(115, 70%, 40%, 0.8)',
                line=dict(width=4, color='hsla(115, 70%, 20%, 1.0)')
            ),
            hovertemplate="Ratio: %{x:.3f}<br>Count: %{y}<extra></extra>",
            name="Self-Altruism Ratio"
        ))
        if not math.isnan(self_ratio_mean):
            fig_self.add_shape(
                type="line",
                x0=self_ratio_mean, x1=self_ratio_mean,
                y0=0, y1=1, xref="x", yref="paper",
                line=dict(color='hsla(115, 100%, 80%, 1.0)', dash='dash', width=4)
            )

        x_min = 0.0
        x_max = 1.0
        x_n_bins = int(x_max * 10) - int(x_min * 10) + 1
        x_tickvals = list(np.round(np.linspace(x_min, x_max, x_n_bins), 3))
        x_ticktext = [''] + [f"{val:.1f}" for val in x_tickvals[1:]]        
        x_axis = {
            'title': x_title_self,  # Will overwrite for guilt fig
            'tickfont': dict(size=24),
            'title_font': dict(size=30),
            'tickvals': x_tickvals, 
            'ticktext': x_ticktext,
            'range': [x_min, x_max]            
        }
        y_title = "Participant Count" if use_initial_params_local else "Parameter Count Across Dyads"
        y_axis = dict(title=y_title, title_font=dict(size=30))

        fig_self.update_layout(
            template=fig_lay.get("template","plotly_dark"),
            title=f"Self-interest to Altruism Ratio ({role_str} Parameters; 𝑛 = {n_self_valid})",
            title_x=0.5, title_y=0.94,
            margin=dict(l=100, r=100, t=80, b=100),
            xaxis=x_axis, yaxis=y_axis,
            font=fig_lay.get("font", {"size": 16})
        )

        # Guilt figure
        fig_guilt = go.Figure()
        fig_guilt.add_trace(go.Histogram(
            x=guilt_ratio_series,
            nbinsx=15,
            marker=dict(
                color='hsla(260, 70%, 40%, 0.8)',
                line=dict(width=4, color='hsla(260, 70%, 20%, 1.0)')
            ),
            hovertemplate="<span style='font-size:24px;'>Ratio: %{x:.3f}<br><br>Count: %{y}<extra></extra></span>",
            name="Guilt-Envy Ratio"
        ))
        if not math.isnan(guilt_ratio_mean):
            fig_guilt.add_shape(
                type="line",
                x0=guilt_ratio_mean, x1=guilt_ratio_mean,
                y0=0, y1=1, xref="x", yref="paper",
                line=dict(color='hsla(260, 100%, 80%, 1.0)', dash='dash', width=4)
            )
        fig_guilt.update_layout(
            template=fig_lay.get("template","plotly_dark"),
            title=f"Guilt to Envy Ratio ({role_str} Parameters; 𝑛 = {n_guilt_valid})",
            title_x=0.5, title_y=0.94,
            margin=dict(l=100, r=100, t=80, b=100),
            xaxis={
                'title': x_title_guilt,
                'tickfont': dict(size=24),
                'title_font': dict(size=30),
                'tickvals': x_tickvals,
                'ticktext': x_ticktext,
                'range': [x_min, x_max]
            },
            yaxis=y_axis,
            font=fig_lay.get("font", {"size": 16})
        )

        if export_fig:
            out_path_self = os.path.join(
                file_paths["visuals"],
                f"SelfAltruism_Ratio_{role_str.replace(' ','_')}_{ratio_mode}.html"
            )
            fig_self.write_html(out_path_self)
            out_path_guilt = os.path.join(
                file_paths["visuals"],
                f"GuiltEnvy_Ratio_{role_str.replace(' ','_')}_{ratio_mode}.html"
            )
            fig_guilt.write_html(out_path_guilt)
        else:
            fig_self.show()
            fig_guilt.show()

        # 9) Build the final results dictionary
        results_dict = {
            "sadistic_pct": sadistic_pct,
            "masochistic_pct": masochistic_pct,
            "guilt_over_envy_pct": guilt_over_envy_pct,
            "altruism_over_self_pct": altruism_over_self_pct,
            "self_ratio_mean": self_ratio_mean,
            "self_ratio_std": self_ratio_std,
            "guilt_ratio_mean": guilt_ratio_mean,
            "guilt_ratio_std": guilt_ratio_std,
            "n_in_analysis": n_data,
            "n_self_ratio_valid": n_self_valid,
            "n_guilt_ratio_valid": n_guilt_valid
        }

        rat_mod_str = " ".join([substr.capitalize() for substr in ratio_mode.split("_")])
        print(f"\n--- Subpopulation and Ratio Stats {rat_mod_str} Mode; ---")
        print(f"---       Subset: {role_str}, Total 𝑛 = {n_data}       ---")
        print(f"   Sadistic (Vᵢⱼ < 0)          = {sadistic_pct:.2f}%")
        print(f"   Masochistic (Vᵢᵢ < 0)       = {masochistic_pct:.2f}%")
        print(f"   Competitive (Ʒᵢⱼ < 0)       = {competitive_pct:.2f}%")
        print(f"   Depricating (Ƹᵢⱼ < 0)       = {depricating_pct:.2f}%")
        print(f"   Guilt > Envy (Ʒᵢⱼ > Ƹᵢⱼ)    = {guilt_over_envy_pct:.2f}%")
        print(f"   Altruism > Self (Vᵢⱼ > Vᵢᵢ) = {altruism_over_self_pct:.2f}%")

        print(f"\n--- Ratio Stats for {rat_mod_str} Mode ---")
        print_str_self =  f" Self vs Altruism => 𝑛 = {n_self_valid}; Mean = {self_ratio_mean:.3f};"
        print_str_guilt = f" Guilt vs Envy =>    𝑛 = {n_guilt_valid}; Mean = {guilt_ratio_mean:.3f};"
        print_str_self += f" Std = {self_ratio_std:.3f}; Raw Mean = {self_ratio_mean_raw:.3f}"
        print_str_guilt += f" Std = {guilt_ratio_std:.3f}; Raw Mean = {guilt_ratio_mean_raw:.3f}"
        print(print_str_self)
        print(print_str_guilt)

        return results_dict

    else:
        # ================ SUBPLOT MODE: produce 4 histograms for both roles ================
        ### SUBPLOTS CHANGES ###

        # We'll compute the entire logic for each role, then make a 2x2 subplot figure.
        roles = ['chooser', 'predictor']
        big_results = {}

        # We create a 2-row, 2-col figure
        fig_sub = make_subplots(
            rows=2, cols=2,
            horizontal_spacing=0.07,  # Adjust horizontal spacing
            vertical_spacing=0.2,
            shared_yaxes=True,
            subplot_titles=(
                "Chooser: Self vs Altruism",
                "Chooser: Guilt vs Envy",
                "Predictor: Self vs Altruism",
                "Predictor: Guilt vs Envy",
            )
        )

        # Because the single code references local variables, we unify them here:
        use_initial_params_local = use_initial_params
        create_new_file_local = create_new_file

        # We'll loop over roles in [predictor, chooser]
        # row=1 => predictor, row=2 => chooser
        # col=1 => self ratio, col=2 => guilt ratio
        for i, role in enumerate(roles, start=1):
            row_i = 2 if role=='predictor' else 1

            # Load & prepare data
            df = load_and_prepare_df_for_role(role)
            n_data = len(df)
            if n_data == 0:
                print(f"[Warning] No data remain after dropping NaNs for {role}.")
                # We'll skip but continue
                big_results[role] = {}
                continue

            # Identify subpop membership
            sadists_mask = (df["Vᵢⱼ"] < 0)
            masochists_mask = (df["Vᵢᵢ"] < 0)
            competitive_mask = (df["Ʒᵢⱼ"] < 0)
            depricating_mask = (df["Ƹᵢⱼ"] < 0)
            guilt_over_envy_mask = (df["Ʒᵢⱼ"] > df["Ƹᵢⱼ"])
            altruism_over_self_mask = (df["Vᵢⱼ"] > df["Vᵢᵢ"])

            sadistic_pct = 100.0 * sadists_mask.sum() / n_data
            masochistic_pct = 100.0 * masochists_mask.sum() / n_data
            competitive_pct = 100.0 * competitive_mask.sum() / n_data
            depricating_pct = 100.0 * depricating_mask.sum() / n_data
            guilt_over_envy_pct = 100.0 * guilt_over_envy_mask.sum() / n_data
            altruism_over_self_pct = 100.0 * altruism_over_self_mask.sum() / n_data

            # Ratios
            self_ratio_series = compute_ratio_array(df["Vᵢᵢ"], df["Vᵢⱼ"], 'self', True)
            guilt_ratio_series = compute_ratio_array(df["Ʒᵢⱼ"], df["Ƹᵢⱼ"], 'guilt', True)
            self_ratio_series_raw = compute_ratio_array(df["Vᵢᵢ"], df["Vᵢⱼ"], 'self', False)
            guilt_ratio_series_raw = compute_ratio_array(df["Ʒᵢⱼ"], df["Ƹᵢⱼ"], 'guilt', False)

            n_self_valid = len(self_ratio_series)
            self_ratio_mean = self_ratio_series.mean() if n_self_valid > 0 else float('nan')
            self_ratio_std  = self_ratio_series.std()  if n_self_valid > 1 else float('nan')

            n_guilt_valid = len(guilt_ratio_series)
            guilt_ratio_mean = guilt_ratio_series.mean() if n_guilt_valid > 0 else float('nan')
            guilt_ratio_std  = guilt_ratio_series.std()  if n_guilt_valid > 1 else float('nan')

            self_ratio_mean_raw =  self_ratio_series_raw.mean() if n_self_valid > 0 else float('nan')
            guilt_ratio_mean_raw = guilt_ratio_series_raw.mean() if n_guilt_valid > 0 else float('nan')

            # Sub-dict of results for this role
            this_res = {
                "sadistic_pct": sadistic_pct,
                "masochistic_pct": masochistic_pct,
                "guilt_over_envy_pct": guilt_over_envy_pct,
                "altruism_over_self_pct": altruism_over_self_pct,
                "self_ratio_mean": self_ratio_mean,
                "self_ratio_std": self_ratio_std,
                "guilt_ratio_mean": guilt_ratio_mean,
                "guilt_ratio_std": guilt_ratio_std,
                "n_self_ratio_valid": n_self_valid,
                "n_guilt_ratio_valid": n_guilt_valid,
                "n_in_analysis": n_data,
            }
            big_results[role] = this_res

            # Add Self ratio histogram
            fig_sub.add_trace(
                go.Histogram(
                    nbinsx=11,
                    marker=dict(
                        color=f'hsla({115 if row_i == 1 else 160}, 70%, 40%, 0.8)',
                        line=dict(width=4, color=f'hsla({115 if row_i == 1 else 160}, 70%, 20%, 1.0)')
                    ),
                    hovertemplate="<span style='font-size:24px;'>Ratio: %{x:.3f}<br><br>Count: %{y}<extra></extra></span>",
                    name="Guilt-Envy Ratio",
                    x=self_ratio_series
                ),
                row=row_i, col=1
            )
            # Add Guilt ratio histogram
            fig_sub.add_trace(
                go.Histogram(
                    nbinsx=11,
                    marker=dict(
                        color=f'hsla({205 if row_i == 1 else 250}, 70%, 40%, 0.8)',
                        line=dict(width=4, color=f'hsla({205 if row_i == 1 else 250}, 70%, 20%, 1.0)')
                    ),
                    hovertemplate="<span style='font-size:24px;'>Ratio: %{x:.3f}<br><br>Count: %{y}<extra></extra></span>",
                    name="Guilt-Envy Ratio",
                    x=guilt_ratio_series
                ),
                row=row_i, col=2
            )

            # Print out stats (unchanged from original) for clarity
            if print_ and role == player_role:
                rat_mod_str = " ".join([substr.capitalize() for substr in ratio_mode.split("_")])
                print(f"\n--- Subpopulation and Ratio Stats for {role.capitalize()}; ---")
                print(f"---  Ratio Mode = {rat_mod_str}; Total 𝑛 = {n_data}  ---")
                print(f"   Sadistic (Vᵢⱼ < 0)          = {sadistic_pct:.2f}%")
                print(f"   Masochistic (Vᵢᵢ < 0)       = {masochistic_pct:.2f}%")
                print(f"   Competitive (Ʒᵢⱼ < 0)       = {competitive_pct:.2f}%")
                print(f"   Depricating (Ƹᵢⱼ < 0)       = {depricating_pct:.2f}%")
                print(f"   Guilt > Envy (Ʒᵢⱼ > Ƹᵢⱼ)    = {guilt_over_envy_pct:.2f}%")
                print(f"   Altruism > Self (Vᵢⱼ > Vᵢᵢ) = {altruism_over_self_pct:.2f}%")

                print(f"\nRatio Stats for {role} => skip_neg/abs: {rat_mod_str}")
                print_str_self = f" Self vs Altruism => 𝑛 = {n_self_valid}, Mean = {self_ratio_mean:.3f},"
                print_str_guilt = f" Guilt vs Envy    => 𝑛 = {n_guilt_valid}, Mean = {guilt_ratio_mean:.3f},"
                print_str_self += f" Std = {self_ratio_std:.3f}, Raw Mean = {self_ratio_mean_raw:.3f}"
                print_str_guilt += f" Std = {guilt_ratio_std:.3f}, Raw Mean = {guilt_ratio_mean_raw:.3f}"
                print(print_str_self)
                print(print_str_guilt)

        # Some layout details. We can adapt the x-axis labeling if you like:
        y_max = 19.04 if ratio_mode == 'absolute' else 9.02 #Hardcoded
        x_min, x_max = 0.0, 1.0
        x_tickvals = list(np.round(np.linspace(x_min, x_max, 6), 3))
        x_ticktext = [''] + [f"{v:.1f}" for v in x_tickvals[1:]]
        x_axis_common = dict(
            range=[x_min, x_max],
            tickvals=x_tickvals,
            ticktext=x_ticktext
        )

        title_text = f"Parameter Ratios for Both Roles "
        if ratio_mode == 'absolute':
            title_text += "(Negative Parameters Included)"
        else:
            title_text += "(Negative Parameters Excluded)"

        fig_sub.update_layout(
            title_text=title_text,
            template=fig_lay.get("template","plotly_dark"),
            margin=dict(l=105, r=80, t=120, b=100),
            font=fig_lay.get("font", {"size": 22}),
            title_x=0.5, title_y=0.98,
            showlegend=False
        )
        fig_sub.update_annotations(font_size=26)
        fig_sub.update_xaxes(x_axis_common, title=x_title_self,  row=1, col=1)
        fig_sub.update_xaxes(x_axis_common, title=x_title_guilt, row=1, col=2)
        fig_sub.update_xaxes(x_axis_common, title=x_title_self,  row=2, col=1)
        fig_sub.update_xaxes(x_axis_common, title=x_title_guilt, row=2, col=2)
        fig_sub.update_yaxes(title_text="𝑛 Participants", range=[0, y_max])

        # Export if requested
        if export_fig:
            out_path = os.path.join(
                file_paths["visuals"],
                f"Ratios_Subplots_{ratio_mode}.html"
            )
            fig_sub.write_html(out_path)
            # print(f"Saved subplot figure => {out_path}")
        else:
            fig_sub.show()

        # Return the combined results from both roles
        return big_results


def param_correlation_matrix_report(general_settings: Dict[str, Any], file_paths: Dict[str, str], player_role: str = 'predictor', cross_role_correlations: bool = False, 
                                    normalize_params: bool = True, alpha: float = 0.05, correction_method: str = 'holm') -> Dict[str, pd.DataFrame]:
    """
    Calculates correlation matrices for either within-role or cross-role parameters.
    Optionally normalizes parameter values within each participant (summing absolute
    values to one) for the within-role case only.

    Arguments:
        • general_settings: Dict[str, Any]; General experiment or environment settings.
        • file_paths: Dict[str, str]; Dictionary mapping file labels to file paths.
        • player_role: str; Either 'chooser' or 'predictor' if within-role correlations.
        • cross_role_correlations: bool; If True, merges chooser and predictor data
          and reports cross-role correlations instead.
        • normalize_params: bool; If True (default), parameter values for each participant
          in the within-role case are scaled so the sum of absolute values equals 1.
        • alpha: float; The family-wise significance level used in multiple-comparison correction.
        • correction_method: str; Correction method for p-values. Examples: 'holm',
          'bonferroni', 'fdr_bh'.

    Returns:
        • Dict[str, pd.DataFrame];
          A dictionary containing:
              "corr": The correlation matrix,
              "pvals_raw": The uncorrected p-value matrix,
              "pvals_corrected": The multiple-comparison-corrected p-value matrix.
    """
    from scipy.stats import pearsonr
    from statsmodels.stats.multitest import multipletests

    possible_columns = ["Vᵢᵢ", "Vᵢⱼ", "Ƹᵢⱼ", "Ʒᵢⱼ", "γ1", "γ2", "γ3", "τ"]

    def load_and_clean(role: str) -> Tuple[pd.DataFrame, List[str]]:
        """Loads data for a given role and returns a cleaned DataFrame plus the columns used."""
        df_role = population_parameter_distribution_df(
            general_settings=general_settings,
            file_paths=file_paths,
            player_role=role,
            use_initial_params=True
        ).drop_duplicates(subset=['player_uuid'])

        columns_in_df = [col for col in possible_columns if col in df_role.columns]
        df_role = df_role[['player_uuid'] + columns_in_df].dropna()
        return df_role, columns_in_df

    # Prepare a structure to store results
    output: Dict[str, pd.DataFrame] = {
        "corr": pd.DataFrame(),
        "pvals_raw": pd.DataFrame(),
        "pvals_corrected": pd.DataFrame()
    }

    # ----------------------------------
    # Case 1: Cross-role correlations
    # ----------------------------------
    if cross_role_correlations:
        df_chooser, chooser_cols = load_and_clean('chooser')
        df_predictor, predictor_cols = load_and_clean('predictor')

        df_merged = pd.merge(
            df_chooser, df_predictor,
            on='player_uuid',
            suffixes=('_choo', '_pred')
        ).dropna()

        if df_merged.empty:
            print("[Warning] No valid data remain after filtering. Returning empty tables.")
            return output

        correlation_matrix = np.zeros((len(chooser_cols), len(predictor_cols)))
        pvalue_matrix = np.ones((len(chooser_cols), len(predictor_cols)))

        for row_index, chooser_col in enumerate(chooser_cols):
            for col_index, predictor_col in enumerate(predictor_cols):
                param_x = df_merged[f"{chooser_col}_choo"].values
                param_y = df_merged[f"{predictor_col}_pred"].values
                correlation_value, p_value = pearsonr(param_x, param_y)
                correlation_matrix[row_index, col_index] = correlation_value
                pvalue_matrix[row_index, col_index] = p_value

        corr_df = pd.DataFrame(correlation_matrix, index=chooser_cols, columns=predictor_cols)
        pval_df = pd.DataFrame(pvalue_matrix, index=chooser_cols, columns=predictor_cols)

        print(f"\n--- Cross-role Correlation Matrix (Chooser→Predictor); n = {len(df_merged)} ---")
        print(corr_df)
        print("\n--- Raw P-Values ---")
        print(pval_df)

        # If you only care about the diagonal in cross-role, correct just those p-values:
        diagonal_indices = np.arange(min(len(chooser_cols), len(predictor_cols)))
        diagonal_pvals = pval_df.values[diagonal_indices, diagonal_indices]

        # Multiple-comparison correction (Holm, etc.)
        reject_flags, corrected_pvals, _, _ = multipletests(
            diagonal_pvals,
            alpha=alpha,
            method=correction_method
        )

        pval_corrected_matrix = pval_df.values.copy()
        pval_corrected_matrix[diagonal_indices, diagonal_indices] = corrected_pvals
        pval_corrected_df = pd.DataFrame(
            pval_corrected_matrix,
            index=chooser_cols,
            columns=predictor_cols
        )

        print("\n--- Corrected P-Values (method = {}) ---".format(correction_method))
        print(pval_corrected_df)

        output["corr"] = corr_df
        output["pvals_raw"] = pval_df
        output["pvals_corrected"] = pval_corrected_df

        return output

    # -----------------------------------
    # Case 2: Within-role correlations
    # -----------------------------------
    df_within_role, columns_in_df = load_and_clean(player_role)

    if df_within_role.empty:
        print("[Warning] No valid data remain after filtering. Returning empty tables.")
        return output

    # (NEW) Optional normalization: for each row (participant), sum absolute values, divide each param by that sum
    if normalize_params:
        # Sum absolute values across the columns of interest
        sum_abs_params = df_within_role[columns_in_df].abs().sum(axis=1)
        # Divide each participant's parameters by their sum of absolute values (preserves sign)
        df_within_role[columns_in_df] = df_within_role[columns_in_df].div(sum_abs_params, axis=0)

    correlation_matrix = np.zeros((len(columns_in_df), len(columns_in_df)))
    pvalue_matrix = np.ones((len(columns_in_df), len(columns_in_df)))

    for row_index, column_i in enumerate(columns_in_df):
        for col_index, column_j in enumerate(columns_in_df):
            param_x = df_within_role[column_i].values
            param_y = df_within_role[column_j].values
            correlation_value, p_value = pearsonr(param_x, param_y)
            correlation_matrix[row_index, col_index] = correlation_value
            pvalue_matrix[row_index, col_index] = p_value

    corr_df = pd.DataFrame(correlation_matrix, index=columns_in_df, columns=columns_in_df)
    pval_df = pd.DataFrame(pvalue_matrix, index=columns_in_df, columns=columns_in_df)

    print(f"\n--- Within-role Correlation Matrix ({player_role.capitalize()}); n = {len(df_within_role)} ---")
    print(corr_df)
    print("\n--- Raw P-Values ---")
    print(pval_df)

    # For a symmetric matrix, only the upper triangle (or lower) are unique.
    matrix_size = len(columns_in_df)
    upper_triangle_rows, upper_triangle_cols = np.triu_indices(matrix_size, k=1)
    pvals_for_correction = pval_df.values[upper_triangle_rows, upper_triangle_cols]

    reject_flags, corrected_pvals, _, _ = multipletests(
        pvals_for_correction,
        alpha=alpha,
        method=correction_method
    )

    # Insert corrected values back into a matrix
    pval_corrected_matrix = pval_df.values.copy()
    pval_corrected_matrix[upper_triangle_rows, upper_triangle_cols] = corrected_pvals
    # Mirror them into the lower triangle for a fully corrected symmetrical matrix
    pval_corrected_matrix[upper_triangle_cols, upper_triangle_rows] = corrected_pvals

    pval_corrected_df = pd.DataFrame(pval_corrected_matrix, index=columns_in_df, columns=columns_in_df)

    print("\n--- Corrected P-Values (method = {}) ---".format(correction_method))
    print(pval_corrected_df)

    output["corr"] = corr_df
    output["pvals_raw"] = pval_df
    output["pvals_corrected"] = pval_corrected_df

    return output


"=========================================================================================="
"============================== Inequality Aversion Analysis =============================="
"=========================================================================================="

def inequality_aversion_sanity_check(file_paths: FilePaths, param_strong: float, param_weak: float, self: float = 0.1, altr: float = 0.0, 
                                     temperature: float = 1.0, filter_constant_sum: bool = False, print_: bool = True) -> None:
    """
    Fit two minimal Fehr–Schmidt–style 'bots' to the observed choices and compare which one fits better.

    Overview
    --------
    This function side-steps the full analysis pipeline and asks a very specific question:
    *Given the observed binary dictator choices, does a simple bot with stronger ENVY (α >> β) fit better,
    or a simple bot with stronger GUILT (β >> α)?*

    Concretely, we define two linear, curvature-free utility functions that only include:
       • a self-interest weight Vᵢᵢ on the chooser's payoff (A vs B),
       • an altruism weight Vᵢⱼ on the other person's payoff,
       • an envy (disadvantageous inequality) penalty α on max(πⱼ - πᵢ, 0),
       • a guilt (advantageous inequality)  penalty β on max(πᵢ - πⱼ, 0).

    We then compute U(A) and U(B) for each trial and convert these to a choice probability with softmax.
    We score each bot against the observed choices using:
       • hit rate (accuracy) and
       • negative log likelihood (NLL) loss.

    The two bots differ only in how α and β are assigned:

        Envious bot:  envy=param_strong, guilt=param_weak
        Guilty bot:   envy=param_weak,  guilt=param_strong

    Arguments:
        • file_paths : FilePaths
            Repository of paths used by `prep.all_histories(...)` to load the trials.
        • param_strong : float
            The larger value used for the 'strong' inequality parameter in its respective bot.
        • param_weak : float
            The smaller value used for the 'weak' inequality parameter in its respective bot.
        • self : float
            Weight Vᵢᵢ placed on the chooser’s own payoff in both bots.
        • altr : float
            Weight Vᵢⱼ placed on the other person’s payoff in both bots.
        • temperature : float
            Softmax temperature used to convert ΔU into p(choose A); larger values = more noise.
        • filter_constant_sum : bool
            If true, includes only constant sum games to immitate normal dictator games. 

    Returns:
        • dict
            Scores for each bot under keys 'envious' and 'guilty'. Each contains:
                {
                'correct': int,
                'incorrect': int,
                'total': int,
                'loss': float,          # negative log-likelihood
                'hit_rate': float       # correct / total
                }

    Notes:
        • This is a deliberately minimal external validity check. It tests whether the headline
        asymmetry (guilt vs. envy) appears under a simple utility with fixed (self, altruism) weights.
        • Lower NLL indicates the better-fitting bot. Accuracy tends to track NLL but need not match it.
    """
    def inequality_aversion_choice(payoffs: dict, envy: float, guilt: float, self: float = 1.0, altr: float = 0.0, temperature: float = 0.1) -> str:
        payAi, payAj = payoffs['payoff_A_chooser'], payoffs['payoff_A_predictor']
        payBi, payBj = payoffs['payoff_B_chooser'], payoffs['payoff_B_predictor']

        utilityA = self * payAi + altr * payAj - envy * max(payAj - payAi, 0) - guilt * max(payAi - payAj, 0)
        utilityB = self * payBi + altr * payBj  - envy * max(payBj - payBi, 0) - guilt * max(payBi - payBj, 0)

        p_choose_A = softmax_(uA=utilityA, uB=utilityB, temperature=temperature)

        choice = 'A' if p_choose_A >= 0.5 else 'B'
        return {'choice': choice, 'p_choose_A': p_choose_A}

    all_data = prep.all_histories(column_names=column_names, file_paths=file_paths)[2]
    dyads, player_info = all_data['histories'], all_data['player_info']

    params = {
        'envious': {'self': self, 'altr': altr, 'envy': param_strong, 'guilt': param_weak},
        'guilty': {'self': self, 'altr': altr, 'envy': param_weak, 'guilt': param_strong}        
    }
    scores = {
        'envious': {'correct': 0, 'incorrect': 0, 'total': 0, 'loss': 0.0},
        'guilty': {'correct': 0, 'incorrect': 0, 'total': 0, 'loss': 0.0}
    }

    for dyad_key, dyad in dyads.items():
        for idx, game in enumerate(dyad):
            abdicated_chooser = game.get('abdicated_chooser', None)
            if not abdicated_chooser:
                participant_choice = game.get('choice', None)
                if participant_choice in ('A', 'B'):

                    payoffs = {
                        "payoff_A_chooser": game.get("payoff_A_chooser"),
                        "payoff_A_predictor": game.get("payoff_A_predictor"),
                        "payoff_B_chooser": game.get("payoff_B_chooser"),
                        "payoff_B_predictor": game.get("payoff_B_predictor"),                       
                    }
                    if filter_constant_sum:
                        if payoffs["payoff_A_chooser"] + payoffs["payoff_A_predictor"] != payoffs["payoff_B_chooser"] + payoffs["payoff_B_predictor"]:
                            continue

                    if all(isinstance(payoff, int) for payoff in payoffs.values()):
                        choice_envious = inequality_aversion_choice(payoffs=payoffs, self=params['envious']['self'], altr=params['envious']['altr'],
                                            envy=params['envious']['envy'], guilt=params['envious']['guilt'], temperature=temperature)

                        if choice_envious['choice'] == participant_choice:
                            scores['envious']['correct'] += 1
                        else:
                            scores['envious']['incorrect'] += 1
                        scores['envious']['total'] += 1

                        prob_of_observed_envious = choice_envious['p_choose_A'] if participant_choice == 'A' else 1 - choice_envious['p_choose_A']
                        if prob_of_observed_envious <= 0: 
                            print(f'Neg p(observed envious) = {prob_of_observed_envious}.')
                            prob_of_observed_envious = 1e-6
                        scores['envious']['loss'] += -math.log(prob_of_observed_envious)

                        choice_guilty  = inequality_aversion_choice(payoffs=payoffs, self=params['guilty']['self'], altr=params['guilty']['altr'],
                                            envy=params['guilty']['envy'], guilt=params['guilty']['guilt'], temperature=temperature)

                        if choice_guilty['choice'] == participant_choice:
                            scores['guilty']['correct'] += 1
                        else:
                            scores['guilty']['incorrect'] += 1
                        scores['guilty']['total'] += 1

                        prob_of_observed_guilty = choice_guilty['p_choose_A'] if participant_choice == 'A' else 1 - choice_guilty['p_choose_A']
                        if prob_of_observed_guilty <= 0: 
                            print(f'Neg p(observed guilty) = {prob_of_observed_guilty}.')
                            prob_of_observed_guilty = 1e-6
                        scores['guilty']['loss'] += -math.log(prob_of_observed_guilty)

    env_corr, env_incr = scores['envious']['correct'], scores['envious']['incorrect']
    if env_corr + env_incr <= 0: scores['envious']['hit_rate'] = 0.0
    else: scores['envious']['hit_rate'] = round(env_corr / (env_corr + env_incr), 3)
    gty_corr, gty_incr = scores['guilty']['correct'], scores['guilty']['incorrect']
    if gty_corr + gty_incr <= 0: scores['guilty']['hit_rate'] = 0.0
    else: scores['guilty']['hit_rate'] = round(gty_corr / (gty_corr + gty_incr), 3)
    answer = "Envy is stronger than guilt." if scores['envious']['loss'] < scores['guilty']['loss'] else "Guilt is stronger than envy."

    if print_:
        print(
            "\n====================== Envy versus Guilt Competition Sanity Check ======================"
            "\nUtility:  Uᵢ(A) = Vᵢᵢ(πᵢᴬ) + Vᵢⱼ(πⱼᴬ) - Ƹᵢⱼ × max(πⱼᴬ - πᵢᴬ, 0) - Ʒᵢⱼ × max(πᵢᴬ - πⱼᴬ, 0)"
            f"\nEnvious: Uᵢ(A) = {self}(πᵢᴬ) + {altr}(πⱼᴬ) - {param_strong} × max(πⱼᴬ - πᵢᴬ, 0) - {param_weak} × max(πᵢᴬ - πⱼᴬ, 0); τ = {temperature}"
            f"\nGuilty:  Uᵢ(A) = {self}(πᵢᴬ) + {altr}(πⱼᴬ) - {param_weak} × max(πⱼᴬ - πᵢᴬ, 0) - {param_strong} × max(πᵢᴬ - πⱼᴬ, 0); τ = {temperature}"
            f"\nEnvious: {scores['envious']['correct']:04d} / {scores['envious']['total']:04d} = {scores['envious']['hit_rate']:.3f}; Loss = {scores['envious']['loss']:04.3f}"
            f"\nGuilty:  {scores['guilty']['correct']:04d} / {scores['guilty']['total']:04d} = {scores['guilty']['hit_rate']:.3f}; Loss = {scores['guilty']['loss']:04.3f}"
            f"\n{answer}"
        )

    return scores


def visualize_inequality_aversion_bot_competition(fig_lay: FigLay, file_paths: FilePaths, *, param_strong: float = 0.75, param_weak: float = 0.25, 
                                                  param_self_values: List[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), 
                                                  param_altr_values: List[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), 
                                                  temperature: float = 1.0, ratio_numerator: Literal["envious", "guilty"] = "envious", show_text_values: bool = False, 
                                                  text_decimals: int = 2, color_range: list[float] = [0.0, 1.0], print_: bool = False, export_fig: bool = True, 
                                                  filename_stub: Optional[str] = None, filter_constant_sum: bool = True) -> go.Figure:
    """
    Visualize the envious-vs-guilty bot fitting "competition" across (self-interest, altruism) weights.

    Purpose:
    For each grid point (Vᵢᵢ, Vᵢⱼ), we:
      1) Fit the two simple bots defined in `inequality_aversion_sanity_check`:
            Envious bot: envy=param_strong, guilt=param_weak
            Guilty bot:  envy=param_weak,  guilt=param_strong
      2) Record their negative log-likelihood (NLL) losses and hit rates.
      3) Plot a heatmap of a *ratio* of the two losses to make comparison interpretable at a glance.

    Ratio definition:
    By default, we plot:
        ratio = Envious_NLL / (Envious_NLL + Guilty_NLL)
    so that **higher values mean the 'guilty' bot fits better**.

    If you set `ratio_numerator="guilty"`, we plot:
        ratio = Guilty_NLL / (Envious_NLL + Guilty_NLL)
    where **higher values mean the 'envious' bot fits better**.

    Arguments:
        • fig_lay : FigLay
            Your standard Plotly layout dictionary (template, colorscales, title_x/y, font sizes, etc.).
        • file_paths : FilePaths
            Paths used by `inequality_aversion_sanity_check` to load the trial histories.
        • param_strong, param_weak : float
            The 'strong' and 'weak' values assigned to the inequality parameters.
            Envious bot uses (envy=strong, guilt=weak); guilty bot uses (envy=weak, guilt=strong).
        • param_self_values : list[float]
            Grid values for the self-interest weight Vᵢᵢ on the x-axis.
        • param_altr_values : list[float]
            Grid values for the altruism weight Vᵢⱼ on the y-axis.
        • temperature : float
            Softmax temperature used inside the bots (same as in the sanity check).
        • ratio_numerator : {"envious", "guilty"}
            Which NLL goes in the numerator of the ratio (controls directionality of the color scale).
        • show_text_values : bool
            If True, writes the numeric ratio onto each cell.
        • text_decimals : int
            Number of decimals to show if `show_text_values=True`.
        • print_ : bool
            If True, prints progress and save path information.
        • export_fig : bool
            If True, writes an interactive HTML file to disk.
        • filename_stub : Optional[str]
            If provided, used as part of the saved filename; otherwise a timestamped default is used.

    Returns:
        • go.Figure
            Interactive Plotly heatmap ready to display or save.

    Notes:
        • Colorbar label is explicit about what 'higher' means, based on `ratio_numerator`.
        • Hover shows: (self, altruism), both losses, both hit rates, the ratio, and which bot wins.
    """

    # Prepare grids
    x_vals = list(param_self_values)    # Vᵢᵢ (self-interest)
    y_vals = list(param_altr_values)    # Vᵢⱼ (altruism)

    Z_ratio = np.zeros((len(y_vals), len(x_vals)))  # heat z-values
    Z_envious_loss = np.zeros_like(Z_ratio)
    Z_guilty_loss  = np.zeros_like(Z_ratio)
    Z_envious_hit  = np.zeros_like(Z_ratio)
    Z_guilty_hit   = np.zeros_like(Z_ratio)

    # Sweep the grid
    for yi, v_other in enumerate(y_vals):
        for xi, v_self in enumerate(x_vals):
            scores = inequality_aversion_sanity_check(
                file_paths=file_paths,
                param_strong=param_strong,
                param_weak=param_weak,
                self=v_self, altr=v_other,
                temperature=temperature,
                filter_constant_sum=filter_constant_sum,
                print_=print_
            )
            ev_loss = float(scores["envious"]["loss"])
            gu_loss = float(scores["guilty"]["loss"])
            ev_hit  = float(scores["envious"]["hit_rate"])
            gu_hit  = float(scores["guilty"]["hit_rate"])

            denom = max(ev_loss + gu_loss, 1e-12)
            if ratio_numerator == "envious":
                ratio_val = ev_loss / denom
                higher_means = "Higher → guilty-bot fits better"
                ratio_label  = "E / (E + G)"
            else:
                ratio_val = gu_loss / denom
                higher_means = "Higher → envious-bot fits better"
                ratio_label  = "G / (E + G)"

            Z_ratio[yi, xi]       = ratio_val
            Z_envious_loss[yi, xi] = ev_loss
            Z_guilty_loss[yi, xi]  = gu_loss
            Z_envious_hit[yi, xi]  = ev_hit
            Z_guilty_hit[yi, xi]   = gu_hit

    # Build hover data
    # customdata shape: (rows, cols, fields)
    customdata = np.stack(
        [
            Z_envious_loss, Z_guilty_loss,
            Z_envious_hit,  Z_guilty_hit,
            Z_ratio
        ],
        axis=-1
    )

    # Optional text on cells
    text_matrix = None
    text_template = None
    if show_text_values:
        text_matrix = np.round(Z_ratio, text_decimals).astype(str)
        text_template = "%{text}"

    # Colorscale & template from your layout bag
    colorscales = fig_lay.get("colorscales", ["Plasma"])
    colorscale  = colorscales[1] if len(colorscales) > 1 else colorscales[0]
    template    = fig_lay.get("template", "plotly_dark")

    # Main heatmap
    heat = go.Heatmap(
        x=x_vals,
        y=y_vals,
        z=Z_ratio,
        zmin=None if color_range is None else color_range[0],
        zmax=None if color_range is None else color_range[1],
        colorscale=colorscale,
        colorbar=dict(
            title=f"{ratio_label}<br><span style='font-size:0.85em'>{higher_means}</span>",
            titleside="right"
        ),
        customdata=customdata,
        hovertemplate=(
            "Vᵢᵢ (self): %{x:.2f}<br>"
            "Vᵢⱼ (altruism): %{y:.2f}<br>"
            "Envious NLL: %{customdata[0]:.3f}<br>"
            "Guilty NLL: %{customdata[1]:.3f}<br>"
            "Envious hit: %{customdata[2]:.3f}<br>"
            "Guilty hit: %{customdata[3]:.3f}<br>"
            f"NLL Ratio ({ratio_label}): " + "%{customdata[4]:.3f}<br>"
            "<extra></extra>"
        ),
        text=text_matrix,
        texttemplate=text_template,
        showscale=True,
    )

    fig = go.Figure(data=[heat])

    # Title & axes
    who_is_strong = f"Envy={param_strong:g}, Guilt={param_weak:g} (envious bot)  |  Envy={param_weak:g}, Guilt={param_strong:g} (guilty bot)"
    title_txt = (
        "Envy vs. Guilt Bot Competition Over (Self, Altruism) Weights<br>"
        f"<span style='font-size:0.85em'>{who_is_strong}  •  τ={temperature:g}</span>"
    )

    fig.update_layout(
        template=template,
        title=title_txt,
        title_x=fig_lay.get("title_x", 0.5),
        title_y=fig_lay.get("title_y", 0.95) - 0.07,
        font=dict(size=fig_lay.get("font_size", 14), color="white" if template == "plotly_dark" else "black"),
        margin=dict(l=615, r=615, t=160, b=80),
        xaxis=dict(
            title="Self-interest weight Vᵢᵢ",
            tickmode="array",
            tickvals=x_vals,
            tickformat=".1f",
            zeroline=False,
        ),
        yaxis=dict(
            title="Altruism weight Vᵢⱼ",
            tickmode="array",
            tickvals=y_vals,
            tickformat=".1f",
            zeroline=False,
            scaleanchor='x1', 
        ),
    )

    # Save if requested
    if export_fig:
        root = (
            file_paths.get("visuals")
            or file_paths.get("processed")
        )
        os.makedirs(root, exist_ok=True)
        stub = filename_stub or f"IA_bot_competition_heatmap_en{param_strong:g}_gu{param_weak:g}_tau{temperature:g}_{ratio_numerator}"
        if filter_constant_sum: stub += "_filtered" 
        out_path = os.path.join(root, f"{stub}.html")
        fig.write_html(out_path)
        if print_:
            print(f"Saved heatmap to: {out_path}")

    return fig


"=========================================================================================="
"======================================== Run Code ========================================"
"=========================================================================================="

run_code_settings = {
    'run_simulation_analyses': False, 
    'run_illustrate_belief_updates': False, 
    'run_alternative_model_constest': False, 
    'run_typological_bayesian_models': False, 
    'run_information_criterion_analysis': False, 
    'run_model_nesting_violation_analysis': False, 
    'run_parameter_distribution_results': False, 
    'run_inequality_aversion_analysis': False
}

def main():
    """Execute main code."""

    if run_code_settings['run_simulation_analyses']:

        # sample_ratios = list(np.round(np.linspace(start=0.05, stop=0.95, num=19), decimals=3))
        sample_ratios = list(np.round(np.linspace(start=0.05, stop=0.10, num=2 ), decimals=3))
        verify_particle_filter_fidelity(general_settings=general_settings, utility_settings=utility_settings, 
                                        param_info=param_info, file_paths=file_paths, fig_lay=fig_lay, 
                                        sample_ratios=sample_ratios, n_predictors=8, n_games_per_dyad=8)

        use_dynamic_predictor = True
        create_simulated_data(n_games=24, randomize_parameters=False, param_bds=param_bds, file_paths=file_paths, run_analysis=True,
                              params_chooser_range={'Vᵢᵢ': (1, 1, 1), 'Vᵢⱼ': (-1, 1, 5), 'std': (1.0, 1.0, 1), 'τ': (0.5, 3, 3)}, 
                              params_predictor_range={'Vᵢᵢ': (1, 1, 1), 'Vᵢⱼ': (-1, 1, 7), 'std': (0.5, 1.5, 3), 'τ': (0.5, 3, 3)}, 
                              utility_settings=utility_settings, dynamic_predictor=use_dynamic_predictor)

        suffix = "_sim_pred_predictor" if use_dynamic_predictor else "_fitted_predictor"
        temp_col = f"τ{suffix}" if use_dynamic_predictor else f"temp{suffix}"
        df_merged = run_simulation_recovery_analysis(
            general_settings=general_settings, file_paths=file_paths,
            fig_lay=fig_lay, export_fig=True, create_new_file=True, produce_figures=True, 
            correlation_csv_name="correlation_results.csv", include_dropdown=False, 
            use_dynamic_predictor=use_dynamic_predictor
        )

        compute_recovery_by_prior_bins(
            df=df_merged,
            var_col="Vᵢⱼ_std_true_predictor",      # prior σ(Vij)
            temp_col="temp_true_predictor",        # prior temp / τ
            param_true_chooser="Vᵢⱼ_true_chooser",
            param_fitted_predictor="Vᵢⱼ_sim_pred_predictor",
            player_id_col="player_uuid_predictor",
            last_rounds=[18, 19, 20],
            var_edges=None,
            temp_edges=None,
            print_=True,
        )

        run_param_recovery_by_k(
            n_games=28,
            n_predictors=70,
            n_choosers_per_predictor=3,
            k_params_range=(1, 9),
            n_altruism_steps=7,
            evenly_space_altruism=True,
            utility_settings_by_k=None,
            general_settings=general_settings,
            file_paths=file_paths,
            fig_lay=fig_lay,
            param_bds=param_bds,
            analysis_experiment_num=0,
            random_seed=2025
        )

        plot_param_recovery_by_round(df_merged=df_merged, fig_lay=fig_lay)

        run_update_speed_simulation_regression(general_settings=general_settings)

        update_speeds = analyze_update_speed_in_human_bot(file_paths=file_paths, general_settings=general_settings, utility_settings=utility_settings)
        plot_update_speed_by_counterpart(update_speeds_per_counterpart=update_speeds['update_speeds_per_counterpart'], fig_lay=fig_lay, 
                                         export_fig=export_fig, file_name="visuals/update_speeds_per_avatar.html")

    if run_code_settings['run_illustrate_belief_updates']:

        experiment_num = general_settings.get('experiment_num')
        if experiment_num not in (0, 1, 2, 3):
            general_settings['experiment_num'] = 2
        elif experiment_num != 2:
            print(f"Much of the visualization code is intended to be run on experiment 2 data. Not "
                  f"experiment {experiment_num}. Change experiment_num in config.py > general_settings.")
            
        n_players_experiment_2 = 84
        player_fits_file_names = []
        for file_name in os.listdir(os.path.join(file_paths['player_fits'], 'experiment_2')):
            if file_name.endswith('.json'):
                player_fits_file_names.append(file_name)
        if len(player_fits_file_names) < n_players_experiment_2:
            raise Exception("No data from which to visualize! You must run the code within "
                            "'run_parameter_distribution_results' before you can visualize Bayesian updates.")

        visualize_bayesian_updates_2d(
            player_uuid=0, 
            counterpart_uuid=0, 
            player_role='predictor', 
            general_settings=general_settings,
            utility_settings=utility_settings,
            file_paths=file_paths,
            fig_lay=fig_lay, 
            n_rounds=9
        )

        visualize_bayesian_updates_3d(
            dyad_games_or_key=0,
            player_uuid=2,
            fig_lay=fig_lay,
            file_paths=file_paths,
            general_settings=general_settings,
            fix_z_axis=True
        )

        for participant_number in range(n_players_experiment_2):
            belief_accuracy_analysis(file_paths=file_paths, participant_num=participant_number, general_settings=general_settings, 
                                    fig_lay=fig_lay, fitted_by_player=True, compute_optimum_updates=True, animate_figure=False)

    if run_code_settings['run_alternative_model_constest']:

        alternative_model_contest(general_settings=general_settings, param_info=param_info, param_bds=param_bds, 
                                  utility_settings=utility_settings, file_paths=file_paths, fig_lay=fig_lay)

    if run_code_settings['run_typological_bayesian_models']:

        typological_model_comparison_fit_population(
            file_paths=file_paths,
            general_settings=general_settings,
            k_min=4, k_max=4,
            n_subsets_per_k=2300,
            intervals_per_dim=5,
            prior_init_method="uniform",   # or "random"
            penalty_weight=10,
            save_after_n_iter=1,
        )

        typological_model_profiles = {
            'winning_k3_profile':         [( 0.0,  0.5), ( 0.0,  1.0), ( 0.5,  0.0)],
            'winning_k4_profile':         [(-1.0,  1.0), (-0.5, -0.5), (-0.5,  0.5), ( 0.5,  0.0)],
            'good_versus_evil_1_profile': [( 1.0,  1.0), ( 1.0, -1.0)],
            'good_versus_evil_2_profile': [( 1.0,  1.0), ( 1.0,  0.0)],
            'canonical_svo_profile':      [( 1.0,  1.0), ( 1.0,  0.0), ( 1.0, -1.0), ( 0.0, -1.0), 
                                           (-1.0, -1.0), (-1.0,  0.0), (-1.0,  1.0), ( 0.0,  1.0)]
        }
        
        for profile in typological_model_profiles.keys():
            typological_model_comparison_fit_individually(best_profiles=profile, general_settings=general_settings, file_paths=file_paths, penalty_weight=10, 
                                         maxiter_global=None, maxiter_local=None, optimization_method='globloc', save_csv=True)            

    if run_code_settings['run_information_criterion_analysis']:

        utility_setting_varieties = gnrl.generate_utility_settings(utility_settings=utility_settings)
        gnrl.identify_redundant_utility_functions(utility_settings=utility_settings, build_equation_function=build_utility_equation, file_paths=file_paths)
        gnrl.equation_to_settings(equation_function=build_utility_equation, utility_settings=utility_settings, file_paths=file_paths, create_new_file=True)  

        information_criterion_analysis(general_settings=general_settings, utility_settings=utility_settings, 
            file_paths=file_paths, param_bds=param_bds, max_iters=24, robustness_epsilon=36, check_for_n_players='all')  

        plot_ic_scores_delta_bic(fig_lay=fig_lay, file_paths=file_paths, general_settings=general_settings, include_dropdown=False)

        plot_ic_robustness_analysis(general_settings=general_settings, file_paths=file_paths, fig_lay=fig_lay)

        utility_setting_contribution_analysis(general_settings=general_settings, file_paths=file_paths, utility_settings_universe=utility_settings, 
                                              score_col="BIC", use_edge_types=("sibling", "parent_child"), include_non_network_toggles=True, export_csv=True)

        extract_rankings_of_canonical_utility_functions(file_paths=file_paths, rank_col="BIC", print_=True)

    if run_code_settings['run_model_nesting_violation_analysis']:

        model_nesting_adjacency_matrices(general_settings=general_settings, utility_settings=utility_settings, 
                                        file_paths=file_paths, create_new_file=True, equation_form=True, print_=False)
        gnrl.summarize_nesting_relationship_counts(general_settings=general_settings, utility_settings=utility_settings, file_paths=file_paths, 
                                            model_nesting_adjacency_matrices=model_nesting_adjacency_matrices, create_new_file=True, print_=True)
        gnrl.equation_to_settings(equation_function=build_utility_equation, utility_settings=utility_settings, file_paths=file_paths, create_new_file=True)  
        
        gnrl.test_utility_functions(utility_settings=utility_settings, setting_to_flip='include_social_comparison', print_=True)

        run_child_parent_embedding_sanity_checks(
            general_settings=general_settings,
            file_paths=file_paths,
            param_bds=param_bds,
            utility_settings=utility_settings,
            player_role_to_fit="chooser",
            fit_for_n_players=1,           
            random_seed=20250406,
            numeric_tolerance=1e-3,
            verbose=True
        )

        run_child_parent_probability_equivalence_smoketest(
            utility_settings=utility_settings,
            file_paths=file_paths,
            param_bds=param_bds,            
            rand_payoff_idx=True,
            n_trials=12,
            rng_seed=None,
            tolerance=1e-12,
            verbose=True
        )

        verify_utility_vs_string_equation(
            utility_function=utility, utility_function_str=build_utility_equation,
            utility_settings=utility_settings, param_bds=param_bds, n_games=625,              
            rng_seed=20250417, exhaustive_if_large=True, option="A", file_paths=file_paths,
            comparison_tol=1e-6, decimals=6, verbose=True)

    if run_code_settings['run_parameter_distribution_results']:

        if general_settings['analysis_mode'] == 'bayesian':
            run_analysis = run_analysis_bayes
        else:
            run_analysis = run_analysis_mle

        print(f"Using Equation:\n{build_utility_equation(utility_settings=utility_settings)}")
        histories = prep.all_histories(file_paths=file_paths, experiment_numbers=[1, 2, 3])
        histories_fitted = [(exper, run_analysis(histories_data=histories[exper - 1], file_paths=file_paths, param_info=param_info, 
                            utility_settings=utility_settings, general_settings=general_settings)) for exper in [general_settings.get('experiment_num')]]  

        for player_role in ('chooser', 'predictor'):
            population_parameter_distribution_histograms(general_settings=general_settings, file_paths=file_paths, fig_lay=fig_lay, 
                                player_role=player_role, use_initial_params=True, create_new_file=create_new_file)
            for ratio_mode in ('absolute', 'skip_negative'):
                subpopulation_stats_and_param_ratio_histograms(general_settings=general_settings, file_paths=file_paths, fig_lay=fig_lay, 
                                    player_role=player_role, use_initial_params=True, create_new_file=False, ratio_mode=ratio_mode, as_subplots=True)
            param_correlation_matrix_report(general_settings=general_settings, file_paths=file_paths, player_role=player_role, normalize_params=True)

        param_correlation_matrix_report(general_settings=general_settings, file_paths=file_paths, player_role=player_role, cross_role_correlations=True, correction_method='holm')

    if run_code_settings['run_inequality_aversion_analysis']:

        visualize_inequality_aversion_bot_competition(fig_lay=fig_lay, file_paths=file_paths, param_strong=0.75, param_weak=0.25, temperature=1.0,
                                                      param_self_values=[0.0, 0.25, 0.5, 0.75, 1.0], param_altr_values=[0.0, 0.25, 0.5, 0.75, 1.0],
                                                      ratio_numerator="envious", show_text_values=True, text_decimals=2, print_=True, export_fig=True, 
                                                      filename_stub=None, filter_constant_sum=False, color_range=[0.35, 0.55])

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
    exit()


