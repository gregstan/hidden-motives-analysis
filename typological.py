import copy, math
import numpy as np
import pprint as pp
from scipy.special import softmax # type: ignore

avatar_frequencies = {
    'utilitarian': 0.3,
    'selfish':     0.3,
    'competitive': 0.3,
    'masochistic': 0.1
}

avatar_params = {
    'utilitarian': ( 1,  1),
    'selfish':     ( 1,  0),
    'competitive': ( 1, -1),
    'masochistic': (-1,  0)
}

choice_frequencies_by_type = {
    (-2,  4): {
        'utilitarian': 16,
        'selfish':      0,
        'competitive':  0,
        'masochistic':  3
    },
    ( 2,  4): {
        'utilitarian': 16,
        'selfish':     16,
        'competitive':  0,
        'masochistic':  0
    },
    ( 4,  2): {
        'utilitarian': 16,
        'selfish':     16,
        'competitive': 16,
        'masochistic':  0
    },
    ( 4, -2): {
        'utilitarian': 16,
        'selfish':     16,
        'competitive': 16,
        'masochistic':  0
    },
    ( 2, -4): {
        'utilitarian':  0,
        'selfish':     16,
        'competitive': 16,
        'masochistic':  0
    },
    (-2, -4): {
        'utilitarian':  0,
        'selfish':      0,
        'competitive': 16,
        'masochistic':  3
    },
    (-4, -2): {
        'utilitarian':  0,
        'selfish':      0,
        'competitive':  0,
        'masochistic':  3
    },
    (-4,  2): {
        'utilitarian':  0,
        'selfish':      0,
        'competitive':  0,
        'masochistic':  3
    }
}

regions_of_compatible_utility = {
    'utilitarian': [( 2,  4), ( 4,  2)],
    'selfish':     [( 4,  2), ( 4, -2)],
    'competitive': [( 4, -2), ( 2, -4)],
    'masochistic': [(-4, -2), (-4,  2)],
}


def softmax_(uA: float, uB: float, temperature: float) -> float:
    """
    Convert two option utilities into a SoftMax choice probability for option A.

    Conceptually, this is the discrete analogue of the SoftMax likelihood used in
    the continuous UBM, but restricted to a 2-option (A vs B) setting.

    Arguments:
        • uA: float; Subjective utility of option A.
        • uB: float; Subjective utility of option B.
        • temperature: float; SoftMax temperature τ (higher = more random).

    Returns:
        • float; Probability p(choose A | uA, uB, τ).
    """    
    "Create a NumPy array of the utilities, scaled by temperature"
    utilities = np.array([uA, uB]) / temperature

    "Use scipy.special.softmax to calculate the probabilities"
    probabilities = softmax(utilities)

    "Return the probability of choosing A (the first element)"
    return probabilities[0]


def choice_probability_discrete(current_game: dict[str, None], agent_params: dict[str, float], 
                                utility_settings: dict, temperature: float | None = None) -> float:
    """
    Compute p(choose A) for a 2-parameter (Vᵢᵢ, Vᵢⱼ) utility in the typological model.

    This is the “small” utility model used for discrete types in the typological
    Bayesian comparison: each avatar type is represented by (Vᵢᵢ, Vᵢⱼ), possibly
    with exponents, and choice probabilities are derived either deterministically
    (argmax) or via a SoftMax.

    Arguments:
        • current_game: dict;
            Contains payoffs for chooser and predictor:
            {'payoff_A_chooser','payoff_A_predictor','payoff_B_chooser','payoff_B_predictor'}.
        • agent_params: dict[str, float];
            Discrete utility parameters, typically:
                - 'Vᵢᵢ': self-interest weight,
                - 'Vᵢⱼ': altruism weight,
                and optionally 'γ1','γ2' for exponents.
        • utility_settings: dict;
            Simple flags controlling whether utilities use single payoffs vs
            differences and whether to subtract a reference point.
        • temperature: float | None;
            If None, return deterministic choice (0/0.5/1); otherwise use SoftMax.

    Returns:
        • float; Probability p(choose A | current_game, agent_params, temperature).
    """
    "Identify payoffs from the game."
    payoff_A_chooser =   current_game.get('payoff_A_chooser', None)
    payoff_A_predictor = current_game.get('payoff_A_predictor', None)
    payoff_B_chooser =   current_game.get('payoff_B_chooser', None)
    payoff_B_predictor = current_game.get('payoff_B_predictor', None)
    Vᵢᵢ = agent_params.get('Vᵢᵢ', None)
    Vᵢⱼ = agent_params.get('Vᵢⱼ', None)

    if any(val is None for val in (Vᵢᵢ, Vᵢⱼ, payoff_A_chooser, payoff_A_predictor, payoff_B_chooser, payoff_B_predictor)):
        raise Exception(f"Failed to extract value in game {current_game}.")

    if utility_settings.get('single_payoffs_not_differences'):
        base_A_self, base_A_other = payoff_A_chooser, payoff_A_predictor
        base_B_self, base_B_other = payoff_B_chooser, payoff_B_predictor
    elif utility_settings.get('reference_dependent_utility'):
        base_A_self, base_A_other = payoff_A_chooser - 3, payoff_A_predictor - 3
        base_B_self, base_B_other = payoff_B_chooser - 3, payoff_B_predictor - 3
    else:
        base_A_self = payoff_A_chooser - payoff_B_chooser
        base_B_self = payoff_B_chooser - payoff_A_chooser
        base_A_other = payoff_A_predictor - payoff_B_predictor
        base_B_other = payoff_B_predictor - payoff_A_predictor

    exp_self = agent_params.get('γ1', 1)
    exp_other = agent_params.get('γ2', agent_params.get('γ1', 1))

    utilityA = Vᵢᵢ * base_A_self ** exp_self + Vᵢⱼ * base_A_other ** exp_other
    utilityB = Vᵢᵢ * base_B_self ** exp_self + Vᵢⱼ * base_B_other ** exp_other

    if temperature is None:
        if utilityA > utilityB: 
            return 1.0
        elif utilityA < utilityB: 
            return 0.0
        else:
            return 0.5
    
    else:
        return softmax_(uA=utilityA, uB=utilityB, temperature=temperature)


def p_choice_given_avatar_type(pds: int, pdo: int, avatar_type: str, 
                               choice_frequencies_by_type: dict, 
                               game_phase: str | None = 'op',
                               temperature: float | None = 1.5) -> float:
    """
    Likelihood p(choice | avatar_type) for a typological avatar.

    In the human-bot experiment, each avatar_type (utilitarian, selfish, etc.)
    has a known preference profile. This function returns the probability that
    an avatar of a given type chooses the observed option, given payoff
    differences (pds, pdo):

        • In observation phase ("op"): uses empirical choice frequencies from
          `choice_frequencies_by_type`.
        • In response phase ("rp"): uses the 2-parameter utility + SoftMax
          (`choice_probability_discrete`) with the canonical (Vᵢᵢ, Vᵢⱼ) for that avatar.

    Arguments:
        • pds: int; Δ payoff to self for the chosen option (πᵢᴬ−πᵢᴮ or vice versa).
        • pdo: int; Δ payoff to other for the chosen option.
        • avatar_type: str; One of the four avatar labels.
        • choice_frequencies_by_type: dict;
            Mapping from (pds,pdo) → {avatar_type → frequency}.
        • game_phase: str | None;
            'op' (observation) or 'rp' (response) to choose likelihood source.
        • temperature: float | None;
            SoftMax temperature used in "rp" mode.

    Returns:
        • float; p(choice | avatar_type, pds, pdo).
    """    
    if game_phase == 'rp' and not (isinstance(temperature, float) and temperature > 0.0):
        raise Exception(f"Temperature must be provided for computing choice probabilities during response phase games.")

    if game_phase == 'rp':
        current_game = {
            'payoff_A_chooser': pds, 'payoff_A_predictor': pdo, 
            'payoff_B_chooser':   0, 'payoff_B_predictor':   0
        }
        agent_params = {
            'Vᵢᵢ': avatar_params[avatar_type][0], 
            'Vᵢⱼ': avatar_params[avatar_type][1]
        }
        return choice_probability_discrete(current_game=current_game, 
                                           agent_params=agent_params, 
                                           temperature=temperature,
                                           utility_settings={})
    
    else:
        choice_frequency = choice_frequencies_by_type[(pds, pdo)][avatar_type]
        return 1.0 if choice_frequency > 0 else 0.0


def p_avatar_type(priors: dict, avatar_type: str) -> float:
    """
    Prior probability p(avatar_type) in a typological Bayesian model.

    Arguments:
        • priors: dict[str, float]; Current prior over avatar types.
        • avatar_type: str; Avatar label (e.g., 'utilitarian').

    Returns:
        • float; Prior weight for that type.
    """    
    return priors[avatar_type]


def p_make_choice(pds: int, pdo: int, priors: dict, choice_frequencies_by_type: dict, temperature: float | None = None, game_phase: str | None = None) -> float:
    """
    Marginal likelihood p(choice | priors) under a typological model.

    Sums over all avatar types in the hypothesis space:
        p(choice) = Σ_type p(choice | type) · p(type)

    Arguments:
        • pds: int; Δ payoff to self for the chosen option.
        • pdo: int; Δ payoff to other for the chosen option.
        • priors: dict[str, float]; Prior/posterior over avatar types.
        • choice_frequencies_by_type: dict; See `p_choice_given_avatar_type`.
        • temperature: float | None; SoftMax temperature in response-phase mode.
        • game_phase: str | None; 'op' or 'rp'.

    Returns:
        • float; p(choice | priors).
    """    
    return sum(
        [p_choice_given_avatar_type(pds=pds, pdo=pdo, avatar_type=avatar, game_phase=game_phase,
                                    choice_frequencies_by_type=choice_frequencies_by_type, temperature=temperature) * 
                                    p_avatar_type(priors=priors, avatar_type=avatar)
            for avatar in priors.keys()]
    )


def p_avatar_type_given_choice(pds: int, pdo: int, avatar_type: str, priors: dict, 
                               choice_frequencies_by_type: dict, temperature: float | None = None) -> float:
    """
    Posterior p(avatar_type | choice) via Bayes’ rule in the typological model.

    Uses:
        p(type | choice) ∝ p(choice | type) · p(type),
    where p(choice | type) is computed by `p_choice_given_avatar_type` and the
    normalizing constant p(choice) by `p_make_choice`.

    Arguments:
        • pds: int; Δ payoff to self for the chosen option.
        • pdo: int; Δ payoff to other for the chosen option.
        • avatar_type: str; Type being updated (utilitarian, selfish, etc.).
        • priors: dict[str, float]; Current prior over types.
        • choice_frequencies_by_type: dict; Likelihood table.
        • temperature: float | None; SoftMax temperature for response-phase games.

    Returns:
        • float; Posterior probability p(avatar_type | choice).
    """    
    p_choice_given_type = p_choice_given_avatar_type(pds=pds, pdo=pdo, avatar_type=avatar_type, 
                                                     choice_frequencies_by_type=choice_frequencies_by_type, 
                                                     temperature=temperature)
    p_avatar = p_avatar_type(priors=priors, avatar_type=avatar_type)
    p_choice = p_make_choice(pds=pds, pdo=pdo, priors=priors, temperature=temperature, 
                             choice_frequencies_by_type=choice_frequencies_by_type)
    
    return round((p_choice_given_type * p_avatar) / p_choice, 12)


def p_avatar_types_given_choice(pds: int, pdo: int, priors: dict, temperature: float | None = None, choice_frequencies_by_type=choice_frequencies_by_type) -> float:
    """
    Posterior over all avatar types given an observed choice.

    Convenience wrapper that applies `p_avatar_type_given_choice` to every
    type in the current prior dictionary.

    Arguments:
        • pds: int; Δ payoff to self for the chosen option.
        • pdo: int; Δ payoff to other for the chosen option.
        • priors: dict[str, float]; Current prior over types.
        • temperature: float | None; SoftMax temperature for response-phase games.
        • choice_frequencies_by_type: dict; Likelihood table.

    Returns:
        • dict[str, float]; Posterior p(type | choice) for each avatar type.
    """    
    return {
        avatar: p_avatar_type_given_choice(pds=pds, pdo=pdo, avatar_type=avatar, priors=priors, 
                                           temperature=temperature, choice_frequencies_by_type=choice_frequencies_by_type)
        for avatar in priors.keys()
    }


def bayesian_update_discrete(payoffs: dict[str, int], choice: str, 
                             choice_frequencies_by_type: dict, priors: dict, 
                             temperature: float | None = None, print_: bool = False) -> float:
    """
    One-step typological Bayesian update for avatar-type posteriors.

    Given a single observed choice in a binary dictator game, this function:
        1) Converts payoffs into payoff differences (pds, pdo) for the chosen option.
        2) Computes p(type | choice) for each avatar type using the discrete
           likelihood (`p_choice_given_avatar_type`).
        3) Optionally prints the full Bayes’ rule decomposition for inspection.

    This is used to build the “Perfect Oracle”, canonical SVO, and other
    typological Bayesian models compared against the continuous UBM.

    Arguments:
        • payoffs: dict[str, int];
            Contains 'payoff_A_chooser','payoff_A_predictor','payoff_B_chooser','payoff_B_predictor'.
        • choice: str;
            'A' or 'B' – the option actually chosen by the avatar/agent.
        • choice_frequencies_by_type: dict;
            Likelihood table for observation-phase games.
        • priors: dict[str, float];
            Prior/posterior over avatar types before this observation.
        • temperature: float | None;
            SoftMax temperature used in response-phase likelihoods (if any).
        • print_: bool;
            If True, prints the Bayes’ rule breakdown for each type.

    Returns:
        • dict[str, float]; Updated posterior over avatar types.
    """
    "Identify payoffs from the game."
    payoff_A_self =  payoffs.get('payoff_A_chooser', None)
    payoff_A_other = payoffs.get('payoff_A_predictor', None)
    payoff_B_self =  payoffs.get('payoff_B_chooser', None)
    payoff_B_other = payoffs.get('payoff_B_predictor', None)    

    pds = payoff_A_self - payoff_B_self if choice == "A" else payoff_B_self - payoff_A_self
    pdo = payoff_A_other - payoff_B_other if choice == "A" else payoff_B_other - payoff_A_other

    if print_:
        for avatar in priors.keys():
            p_choice_given_type = p_choice_given_avatar_type(pds=pds, pdo=pdo, avatar_type=avatar, 
                                    choice_frequencies_by_type=choice_frequencies_by_type, 
                                    temperature=temperature)
            p_avatar = p_avatar_type(priors=priors, avatar_type=avatar)
            p_choice = p_make_choice(pds=pds, pdo=pdo, priors=priors, 
                        choice_frequencies_by_type=choice_frequencies_by_type, 
                        temperature=temperature)            
            p_type_given_choice = p_avatar_type_given_choice(
                pds=pds, pdo=pdo, avatar_type=avatar, priors=priors, 
                choice_frequencies_by_type=choice_frequencies_by_type, 
                temperature=temperature)
            if avatar == 'selfish': avatar = 'selfish    '
            print_str = "p(avatar|choice) = [p(choice|avatar) × p(avatar)] / p(choice) = "
            print_str += f"p({avatar}|{(pds, pdo)}) = [p({(pds, pdo)}|{avatar}) × p({avatar})] / p({(pds, pdo)}) = "
            print_str += f"[{round(p_choice_given_type, 8)} × {round(p_avatar, 8)}] / {round(p_choice, 8)} = {round(p_type_given_choice, 8)}"
            print(print_str)

    return p_avatar_types_given_choice(pds=pds, pdo=pdo, priors=priors, 
                                       choice_frequencies_by_type=choice_frequencies_by_type)


def sum_of_all_loss(dyad_games: list, update_method: str, target_player: str, target_role: str) -> float:
    """
    Sum the continuous-model loss across all games for a given player and role.

    Used to compare the total negative log-likelihood (or other loss) of the
    continuous UBM-based model against the typological model within a dyad.

    Arguments:
        • dyad_games: list[dict]; Full game list for a dyad.
        • update_method: str; Key under 'parameter_estimates' (e.g., "grid").
        • target_player: str; UUID of the player.
        • target_role: str; 'predictor' or 'chooser' within the estimates.

    Returns:
        • float; Sum of loss_final across all applicable games.
    """
    total_loss = 0.0
    for game in dyad_games:
        param_data = game.get('parameter_estimates', {}).get(update_method, {})
        if target_player in param_data:
            total_loss += param_data[target_player].get(target_role, {}).get('output', {}).get('loss_final', 0)
    return total_loss


def avatar_posteriors(dyad_games: list[dict], update_method: str = 'grid', avatar_frequencies: dict = avatar_frequencies, 
                      choice_frequencies_by_type: dict = choice_frequencies_by_type, temperature: float | None = None, loss_funct_type: str = 'log') -> list[dict]:
    """
    Run a typological avatar-type model alongside the continuous model for one dyad.

    This function:
        • Tracks a simple 4-type avatar prior (utilitarian, selfish, competitive, masochistic).
        • After each avatar choice in the previous game, updates this prior with
          `bayesian_update_discrete`.
        • During prediction rounds, computes a typological prediction and its loss
          vs the participant’s prediction.
        • Accumulates a total discrete-model loss and compares it to the continuous
          model’s total loss (from `sum_of_all_loss`), marking which model “wins”.

    This corresponds to the discrete typological models in the paper’s model
    comparison section (Perfect Oracle, canonical SVO, etc.).

    Arguments:
        • dyad_games: list[dict];
            All games in one dyad (observation and response phases).
        • update_method: str;
            Key for continuous-model estimates in 'parameter_estimates'.
        • avatar_frequencies: dict[str, float];
            Initial prior over avatar types (typically matching the experimental mix).
        • choice_frequencies_by_type: dict;
            Likelihoods p(choice | type) for observation-phase games.
        • temperature: float | None;
            SoftMax temperature used when calling the utility-based avatar model.
        • loss_funct_type: str;
            'log' for negative log-likelihood (default) or 'ssr' for squared error.

    Returns:
        • list[dict]; The dyad_games list with an 'optimum_update' block added
          under 'parameter_estimates'[update_method], including posterior traces,
          discrete-model loss, continuous-model loss, and a model_winner flag.
    """
    participant_uuid = None
    total_discrete_model_loss = 0.0
    payoff_keys = [
        'payoff_A_chooser', 'payoff_A_predictor', 
        'payoff_B_chooser', 'payoff_B_predictor'
    ]        
    converged_on_avatar = False
    avatar_frequencies = copy.deepcopy(avatar_frequencies)
    for idx, dyad_game in enumerate(dyad_games):
        discrete_model_loss = None
        discrete_model_prediction = None
        participant_prediction = dyad_game.get('prediction', None)
        
        if participant_uuid is None:
            predictor = dyad_game.get('predictor', None)
            if predictor is not None:
                participant_uuid = predictor

        if participant_prediction in ("A", "B"):
            "Submit model prediction"
            prediction_val = 1 if participant_prediction == 'A' else 0
            pds_current = dyad_game.get('payoff_A_chooser', 0) - dyad_game.get('payoff_B_chooser', 0)
            pdo_current = dyad_game.get('payoff_A_predictor', 0) - dyad_game.get('payoff_B_predictor', 0)

            discrete_model_prediction = p_make_choice(pds=pds_current, pdo=pdo_current, priors=avatar_frequencies, 
                                    choice_frequencies_by_type=choice_frequencies_by_type, temperature=temperature, game_phase=dyad_game.get('phase'))

            "Compute loss"
            if loss_funct_type == "ssr":
                discrete_model_loss = (discrete_model_prediction - prediction_val)**2
            elif loss_funct_type == "log":
                residual = discrete_model_prediction if participant_prediction == 'A' else 1 - discrete_model_prediction
                if residual <= 0: 
                    "Prevent math domain value error and generate large loss."
                    residual = 1e-6
                discrete_model_loss = -math.log(residual)
            else:
                raise ValueError(f"Unsupported loss_funct_type: {loss_funct_type}. Use 'ssr' or 'log'.") 

            if discrete_model_loss is not None:
                total_discrete_model_loss += discrete_model_loss

        param_data: dict = dyad_game.get('parameter_estimates', {}).get(update_method, {})

        if idx > 0:    
            previous_game = dyad_games[idx-1]        
            avatar_choice = previous_game.get('choice', None)
            if avatar_choice in ("A", "B"):
                payoffs = {payoff_key: previous_game.get(payoff_key) for payoff_key in payoff_keys}
                avatar_frequencies = bayesian_update_discrete(payoffs=payoffs, choice=avatar_choice, 
                                                            choice_frequencies_by_type=choice_frequencies_by_type, 
                                                            priors=avatar_frequencies, temperature=temperature, print_=False)        
                if any(posterior == 1 for posterior in avatar_frequencies.values()):
                    converged_on_avatar = True
                
        param_data['optimum_update'] = {
            'loss': discrete_model_loss,
            'model_prediction': discrete_model_prediction,
            'converged': converged_on_avatar,
            'avatar_posteriors': avatar_frequencies
        }   

    total_continious_model_loss = sum_of_all_loss(dyad_games=dyad_games, update_method=update_method, 
                                                  target_player=participant_uuid, target_role='predictor')

    "Add total_loss to first game"
    first_game = dyad_games[0]
    param_data: dict = first_game.get('parameter_estimates', {}).get(update_method, {})
    if 'optimum_update' not in param_data:
        param_data['optimum_update'] = {}
    param_data['optimum_update']['total_continious_model_loss'] = total_continious_model_loss
    param_data['optimum_update']['total_discrete_model_loss'] = total_discrete_model_loss
    if total_continious_model_loss is not None:
        if total_continious_model_loss > total_discrete_model_loss:
            param_data['optimum_update']['model_winner'] = "discrete"
        else:
            param_data['optimum_update']['model_winner'] = "continious"

    return dyad_games   


def discrete_bayesian_model(dyad_games: list[dict], choice_funct: callable, player_uuid: str, 
                            general_settings: dict, hypothesis_space: dict[tuple[float]: float], 
                            update_method: str = 'discrete') -> list[float]:
    """
    Run a general discrete-parameter Bayesian model over a hypothesis space.

    This is the generic typological Bayesian learner used in the paper’s
    continuous vs discrete comparison: instead of four named avatar types,
    this considers a finite set of (Vᵢᵢ, Vᵢⱼ) profiles with prior probabilities,
    updates them based on observed choices, and evaluates prediction loss.

    Arguments:
        • dyad_games: List[Dict[str, Any]]; List of games that store all data.
        • choice_funct: callable; Function that generates prediction probabilities.
        • general_settings: GeneralSettings; Miscellaneous settings used throughout this file.
        • player_uuid: str = None; Player identifier, like "44598db2-2243-45c1-8ba8-5a8ebaa0b042".
        • hypothesis_space: dict[tuple[float]: float]; Maps parameter typles[Vᵢᵢ, Vᵢⱼ] to probabilities.
            - Example: {
                ( 1.0,  1.0): 0.125,
                ( 1.0,  0.0): 0.125,
                ( 1.0, -1.0): 0.125,
                ( 0.0, -1.0): 0.125,
                (-1.0, -1.0): 0.125,
                (-1.0,  0.0): 0.125,
                (-1.0,  1.0): 0.125,
                ( 0.0,  1.0): 0.125,
            }
        • update_method: str; Key in dyad_game['parameter_estimates'] 
            nested dictionary where hypothesis data is saved.

    Returns:
        • The 'dyad_games' list with 'parameter_estimates' updated.    
    """
    if general_settings.get('experiment_num') != 2:
        raise Exception("This function is intended for experiment number 2.")

    "Ensure that priors sum to 1."
    sum_hypotheses = sum(hypothesis_space.values())
    if sum_hypotheses != 1:
        hypothesis_space = {profile: prob / sum_hypotheses for profile, prob in hypothesis_space.items()}
        
    "Extract softmax temperature from settings."
    softmax_temperature = general_settings.get('softmax_temperature', 1.5)

    "Define hard-coded utility settings."
    utility_settings_ = {
        'conditional_welfare_mode':       False,
        'reference_dependent_altruism':   False,
        'min_max_rawlsian_leontief':      False,
        'use_exponential_parameters':     False,
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

    "Iterate over all games in the dyad"
    for dyad_game in dyad_games:
        if dyad_game.get('predictor') != player_uuid:
            continue

        "Store previous posterior"
        if 'parameter_estimates' not in dyad_game:
            dyad_game['parameter_estimates'] = {}
        if update_method not in dyad_game['parameter_estimates']:
            dyad_game['parameter_estimates'][update_method] = {}
        if player_uuid not in dyad_game['parameter_estimates'][update_method]:
            dyad_game['parameter_estimates'][update_method][player_uuid] = {}
        if 'predictor' not in dyad_game['parameter_estimates'][update_method][player_uuid]:
            dyad_game['parameter_estimates'][update_method][player_uuid]['predictor'] = {}
        param_est = dyad_game['parameter_estimates'][update_method][player_uuid]['predictor']
        param_est['params'] = copy.deepcopy(hypothesis_space)

        "Compute likelihood p(choose A | avatar profile)"
        p_choice_given_profile = [
            choice_funct(current_game=dyad_game, agent_params={'Vᵢᵢ': profile_key[0], 'Vᵢⱼ': profile_key[1]},
                         utility_settings=utility_settings_, softmax_temperature=softmax_temperature, select_responses=False)
                        ['model_choose_A'] for profile_key in hypothesis_space.keys()
        ]
        if dyad_game.get('choice') == "B":
            p_choice_given_profile = [1 - prob for prob in p_choice_given_profile]

        "List of prior probabilities for each avatar profile."
        p_profile = list(hypothesis_space.values())

        "Unnormalized Posterior."
        p_profile_given_choice = [likelihood * prior for likelihood, 
                                    prior in zip(p_choice_given_profile, p_profile)]
        
        "Marginal probability p(choose A)"
        p_choice = sum(p_profile_given_choice)

        game_phase = dyad_game.get('phase')
        if game_phase == "op":
            "Do Bayesian update in observation phases only."

            "Normalize posterior by p(choose A)"
            p_profile_given_choice = [posterior / p_choice for posterior in p_profile_given_choice]

            "Update hypothesis space with posterior"
            hypothesis_space = {
                profile: posterior for profile, posterior in 
                zip(hypothesis_space.keys(), p_profile_given_choice)
            }

        elif game_phase == "rp":
            "Submit prediction in response phases only."
            model_predict_A = p_choice 

            "Extract participant's prediction response data."
            prediction = dyad_game.get('prediction', None) 

            "Skip if player abdicated response, no response is found, or no model prediction is found."
            abdicated = dyad_game.get('abdicated_chooser', False) or dyad_game.get('abdicated_predictor', False)
            if prediction is None or model_predict_A is None or abdicated:
                continue

            "Map prediction 'A' to 1 and 'B' to 0."
            prediction_val = 1 if prediction == 'A' else 0

            "Compute loss as negative log likelihood."
            prob_of_observed = model_predict_A if prediction_val == 1 else (1 - model_predict_A)
            if prob_of_observed <= 0:
                prob_of_observed = 1e-6
            raw_neglogprob = -math.log(prob_of_observed)

            "Store data"
            param_est['output'] = {
                'model_predict_A': model_predict_A,
                'raw_neglogprob': raw_neglogprob,
            }

    return dyad_games


def distance_to_perfection(Vᵢᵢ: float, Vᵢⱼ: float, avatar_type: str, return_percent: bool = False) -> float:
    """
    Computes the angular distance between the given parameters (Vᵢᵢ, Vᵢⱼ) 
    and the optimal parameters for the given avatar type, then normalizes 
    the distance to a scale of 0 (perfect) to 1 (maximally incorrect).

    Arguments:
        • Vᵢᵢ: float - Self-interest parameter.
        • Vᵢⱼ: float - Altruism parameter.
        • avatar_type: str - One of ['utilitarian', 'selfish', 'competitive', 'masochistic'].
        • return_percent: bool - If True, returns percent correct instead of normalized distance.

    Returns:
        • float - Normalized angular distance (0 to 1) OR percent correct (if return_percent=True).
    """
    "Define true parameter vectors (ideal angles)"
    true_parameters = {
        "utilitarian":  45,  # 1.5 o'clock position
        "selfish":       0,  # 3   o'clock position
        "competitive": 315,  # 4.5 o'clock position
        "masochistic": 180   # 9   o'clock position
    }
    
    "Get the target angle in degrees"
    if avatar_type not in true_parameters:
        raise ValueError(f"Invalid avatar type '{avatar_type}'. Must be one of {list(true_parameters.keys())}")

    target_angle = np.radians(true_parameters[avatar_type])  # Convert degrees to radians

    "Compute the angle of the given (Vᵢᵢ, Vᵢⱼ) vector"
    angle = np.arctan2(Vᵢⱼ, Vᵢᵢ)  # Radians in range [-π, π]

    "Ensure positive angle in range [0, 2π] for correct comparison"
    if angle < 0:
        angle += 2 * np.pi

    "Compute absolute angular difference"
    angle_diff = abs(angle - target_angle)

    "Normalize to [0, π] (max difference is 180 degrees = π radians)"
    angle_diff = min(angle_diff, 2 * np.pi - angle_diff)  # Ensures shortest angular distance

    "Normalize to [0,1], where 0 = perfect match, 1 = complete opposite (π radians = 180 degrees)"
    normalized_distance = round(angle_diff / np.pi, 3)

    "Convert to percent correct if requested"
    if return_percent:
        percent_correct = round((1 - normalized_distance) * 100, 1)
        return percent_correct

    return normalized_distance


