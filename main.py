from mle import run_analysis_mle
from behavioral_distances import *
from architecture import *
from analysis import *

"=========================================================================================="
"======================================== Run Code ========================================"
"=========================================================================================="

run_code_settings: RunCodeSettings = {
    'run_simulation_analyses':              True,
    'run_illustrate_belief_updates':        True,
    'run_alternative_model_contest':        True,
    'run_typological_bayesian_models':      True,
    'run_information_criterion_analysis':   True,
    'run_model_nesting_violation_analysis': True,
    'run_individual_architecture_analysis': True,
    'run_model_recovery_simulation':        True,
    'run_parameter_distribution_results':   True,
    'run_inequality_aversion_analysis':     True,
}

def main():
    """Execute main code."""

    run_population_recovery_bootstrap(
        general_settings=general_settings,
        figure_layout=figure_layout,
        file_paths=file_paths,
        param_bds=param_bds,
        n_bootstrap_iterations=10,
        create_new_file=True,
        n_games=60,
    )
    exit()
    "Apply master random seed when reproducibility mode is enabled."
    _master_seed = general_settings.get('random_seeds', {}).get('seed', None)
    if _master_seed is not None:
        random.seed(_master_seed)
        np.random.seed(_master_seed)
        print(f"Reproducibility mode active. Master random seed: {_master_seed}.")

    "Ensure all output directories exist — safe on a clean clone or a new machine."
    for _dir_key in ('processed', 'param_data', 'player_fits', 'dyad_data', 'discrete', 'visuals', 'bic_aic'):
        os.makedirs(str(file_paths[_dir_key]), exist_ok=True)

    if run_code_settings['run_simulation_analyses']:

        sample_ratios = list(np.round(np.linspace(start=0.05, stop=0.95, num=19), decimals=3))
        verify_particle_filter_fidelity(general_settings=general_settings, utility_settings=utility_settings,
                                        param_info=param_info, file_paths=file_paths, figure_layout=figure_layout,
                                        sample_ratios=sample_ratios, n_predictors=8, n_games_per_dyad=8)

        use_dynamic_predictor = True
        create_simulated_data(n_games=24, randomize_parameters=False, param_bds=param_bds, file_paths=file_paths, run_analysis=True,
                              params_chooser_range={'Vᵢᵢ': (1, 1, 1), 'Vᵢⱼ': (-1, 1, 5), 'std': (1.0, 1.0, 1), 'τ': (0.5, 3, 3)},
                              params_predictor_range={'Vᵢᵢ': (1, 1, 1), 'Vᵢⱼ': (-1, 1, 7), 'std': (0.5, 1.5, 3), 'τ': (0.5, 3, 3)},
                              utility_settings=utility_settings, dynamic_predictor=use_dynamic_predictor)

        df_merged = run_parameter_recovery_simulation(
            general_settings=general_settings, file_paths=file_paths,
            figure_layout=figure_layout, export_fig=True, create_new_file=True, 
            produce_figures=True, correlation_csv_name="correlation_results.csv", 
            include_dropdown=False, use_dynamic_predictor=use_dynamic_predictor,
        )

        "Table A: predictor τ bins — kept for reference (was the original call)."
        tabulate_recovery_correlations_by_prior_bins(
            df=df_merged,
            temp_col="τ_true_predictor",
            var_col="Vᵢⱼ_std_true_predictor",
            param_true_chooser="Vᵢⱼ_true_chooser",
            param_fitted_predictor="Vᵢⱼ_belief_predictor",
            player_id_col="player_uuid_predictor",
            last_rounds=[21, 22, 23],
            var_edges=None,
            temp_edges=None,
            file_paths=None,
            print_=True,
        )

        "Table B: chooser τ bins — face-valid version: higher chooser randomness → slower recovery."
        bins_chooser_tau = tabulate_recovery_correlations_by_prior_bins(
            df=df_merged,
            file_paths=file_paths,
            temp_col="τ_true_chooser",
            var_col="Vᵢⱼ_std_true_predictor",
            param_true_chooser="Vᵢⱼ_true_chooser",
            param_fitted_predictor="Vᵢⱼ_belief_predictor",
            player_id_col="player_uuid_predictor",
            last_rounds=[21, 22, 23],
            temp_edges=None,
            var_edges=None,
            print_=True,
        )

        plot_parameter_recovery_correlation(
            df=None, file_paths=file_paths, round_selection='first',
            as_scatterplot=False, boxplot_param="Vij",
            include_dropdown=False, figure_layout=figure_layout, 
            export_fig=True, fitted_suffix="_belief_predictor",
        )

        plot_param_recovery_correlation_by_round(df_merged=df_merged, figure_layout=figure_layout, general_settings=general_settings, file_paths=file_paths)
        run_update_speed_simulation_regression(general_settings=general_settings, file_paths=file_paths)

        update_speeds = analyze_update_speed_in_human_on_bot_study(
            file_paths=file_paths, general_settings=general_settings, utility_settings=utility_settings,
        )
        plot_update_speed_by_counterpart(
            update_speeds_per_counterpart=update_speeds['update_speeds_per_counterpart'],
            figure_layout=figure_layout, export_fig=export_fig, file_name="visuals/update_speeds_per_avatar.html",
        )

        run_param_recovery_by_k(
            general_settings=general_settings,
            file_paths=file_paths,
            figure_layout=figure_layout,
            param_bds=param_bds,
        )

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
            raise Exception("No data from which to visualize! Must run the code within "
                            "'run_parameter_distribution_results' before you can visualize Bayesian updates.")

        visualize_bayesian_updates_2d(
            player_uuid=0,
            counterpart_uuid=0,
            player_role='predictor',
            general_settings=general_settings,
            utility_settings=utility_settings,
            file_paths=file_paths,
            figure_layout=figure_layout,
            n_rounds=9,
        )

        visualize_bayesian_updates_3d(
            dyad_games_or_key=0,
            player_uuid=2,
            figure_layout=figure_layout,
            file_paths=file_paths,
            general_settings=general_settings,
            fix_z_axis=True,
        )

        for participant_number in range(n_players_experiment_2):
            belief_accuracy_analysis(
                file_paths=file_paths, participant_num=participant_number,
                general_settings=general_settings, figure_layout=figure_layout,
                fitted_by_player=True, compute_optimum_updates=True, animate_figure=False,
            )

    if run_code_settings['run_alternative_model_contest']:

        alternative_model_contest(
            general_settings=general_settings, param_info=param_info, param_bds=param_bds,
            utility_settings=utility_settings, file_paths=file_paths, figure_layout=figure_layout,
        )

    if run_code_settings['run_typological_bayesian_models']:

        typological_model_comparison_fit_population(
            file_paths=file_paths,
            general_settings=general_settings,
            k_min=4, k_max=4,
            n_subsets_per_k=2300,
            intervals_per_dim=5,
            prior_init_method="uniform",
            penalty_weight=10,
            save_after_n_iter=1,
        )

        typological_model_profiles = {
            'winning_k3_profile':         [( 0.0,  0.5), ( 0.0,  1.0), ( 0.5,  0.0)],
            'winning_k4_profile':         [(-1.0,  1.0), (-0.5, -0.5), (-0.5,  0.5), ( 0.5,  0.0)],
            'good_versus_evil_1_profile': [( 1.0,  1.0), ( 1.0, -1.0)],
            'good_versus_evil_2_profile': [( 1.0,  1.0), ( 1.0,  0.0)],
            'canonical_svo_profile':      [( 1.0,  1.0), ( 1.0,  0.0), ( 1.0, -1.0), ( 0.0, -1.0),
                                           (-1.0, -1.0), (-1.0,  0.0), (-1.0,  1.0), ( 0.0,  1.0)],
        }

        for profile in typological_model_profiles.keys():
            typological_model_comparison_fit_individually(
                best_profiles=profile, general_settings=general_settings, file_paths=file_paths,
                penalty_weight=10, maxiter_global=None, maxiter_local=None,
                optimization_method='globloc', save_csv=True,
            )

    if run_code_settings['run_information_criterion_analysis']:

        utility_setting_varieties = gnrl.generate_utility_settings(utility_settings=utility_settings)
        gnrl.identify_redundant_utility_functions(
            utility_settings=utility_settings,
            build_equation_function=build_utility_equation,
            file_paths=file_paths,
            compute_ampd_fn=compute_ampd_matrix,
            general_settings=general_settings,
            param_bds=param_bds,
        )
        gnrl.equation_to_settings(
            equation_function=build_utility_equation, utility_settings=utility_settings,
            file_paths=file_paths, create_new_file=True,
        )

        dynamic_updating = mp.cpu_count() >= 10 and general_settings['run_in_parallel']

        information_criterion_analysis(
            general_settings=general_settings, utility_settings=utility_settings,
            file_paths=file_paths, param_bds=param_bds, max_iters=24, robustness_epsilon=36,
            check_for_n_players='all', dynamic_updating=dynamic_updating,
        )

        plot_ic_scores_delta_bic(figure_layout=figure_layout, file_paths=file_paths, general_settings=general_settings, include_dropdown=False)
        plot_ic_robustness_analysis(general_settings=general_settings, file_paths=file_paths, figure_layout=figure_layout)

        utility_setting_contribution_analysis(
            general_settings=general_settings, file_paths=file_paths, utility_settings_universe=utility_settings,
            score_col="BIC", use_edge_types=("sibling", "parent_child"), include_non_network_toggles=True, export_csv=True,
        )

        extract_rankings_of_canonical_utility_functions(file_paths=file_paths, rank_col="BIC", print_=True)

    if run_code_settings['run_model_nesting_violation_analysis']:

        model_nesting_adjacency_matrices(
            general_settings=general_settings, utility_settings=utility_settings,
            file_paths=file_paths, create_new_file=True, equation_form=True, print_=False,
        )
        gnrl.summarize_nesting_relationship_counts(
            general_settings=general_settings, utility_settings=utility_settings, file_paths=file_paths,
            model_nesting_adjacency_matrices=model_nesting_adjacency_matrices, create_new_file=True, print_=True,
        )
        gnrl.equation_to_settings(
            equation_function=build_utility_equation, utility_settings=utility_settings,
            file_paths=file_paths, create_new_file=True,
        )
        gnrl.test_utility_functions(utility_settings=utility_settings, setting_to_flip='include_social_comparison', print_=True)

        verify_same_inputs_same_outputs_for_children_and_parents(
            general_settings=general_settings,
            utility_settings=utility_settings,
            player_role_to_fit="chooser",
            numeric_tolerance=1e-3,
            file_paths=file_paths,
            param_bds=param_bds,
            fit_for_n_players=1,
            random_seed=None,
            verbose=True,
        )

        run_child_parent_probability_equivalence_smoketest(
            utility_settings=utility_settings,
            file_paths=file_paths,
            param_bds=param_bds,
            rand_payoff_idx=True,
            random_seed=None,
            tolerance=1e-12,
            verbose=True,
            n_trials=12,
        )

        verify_utility_vs_string_equation(
            utility_function=utility, utility_function_str=build_utility_equation,
            utility_settings=utility_settings, param_bds=param_bds, n_games=625,
            random_seed=None, exhaustive_if_large=True, file_paths=file_paths,
            comparison_tol=1e-6, decimals=6, verbose=True,
        )

    if run_code_settings['run_individual_architecture_analysis']:

        "Ensure per-participant × per-model combined fits are cached."
        extract_participant_model_combined_fits(
            general_settings=general_settings,
            file_paths=file_paths,
            create_new_file=False,
        )

        "Ensure the AMPD distance matrix exists; generate if not already cached."
        compute_ampd_matrix(
            general_settings=general_settings,
            file_paths=file_paths,
            param_bds=param_bds,
            utility_settings=utility_settings,
            create_new_file=False,
        )

        compute_architecture_compression_curve(
            general_settings=general_settings,
            file_paths=file_paths,
            create_new_file=False,
        )

        plot_architecture_compression_curve(
            general_settings=general_settings,
            file_paths=file_paths,
            figure_layout=figure_layout,
        )

    if run_code_settings['run_model_recovery_simulation']:

        "Generate the AMPD matrix first when AMPD-based candidate selection is requested."
        mr = general_settings.get('model_recovery_settings', {})
        if mr.get('candidate_model_selection_mode', 'hamming') == 'ampd':
            compute_ampd_matrix(
                general_settings=general_settings,
                file_paths=file_paths,
                param_bds=param_bds,
                utility_settings=utility_settings,
                create_new_file=False,
            )

        compute_model_recovery_simulation(
            general_settings=general_settings,
            file_paths=file_paths,
            param_bds=param_bds,
            utility_settings=utility_settings,
            create_new_file=False,
        )

        plot_model_recovery_simulation(
            general_settings=general_settings,
            file_paths=file_paths,
            figure_layout=figure_layout,
        )

    if run_code_settings['run_parameter_distribution_results']:

        if general_settings['analysis_mode'] == 'bayesian':
            run_analysis = run_analysis_bayes
        else:
            run_analysis = run_analysis_mle

        print(f"Using Equation:\n{build_utility_equation(utility_settings=utility_settings)}")
        experiment_num = general_settings.get('experiment_num')
        player_pairs_path = os.path.join(
            file_paths["processed"],
            file_paths["file_names"][f"player_pairs_exper{experiment_num}"]
        )
        with open(player_pairs_path, "r", encoding='utf-8') as histories_file:
            histories_and_info = json.load(histories_file)
        histories_fitted = [(experiment_num, run_analysis(
            histories_data=histories_and_info, file_paths=file_paths,
            param_info=param_info, utility_settings=utility_settings,
            general_settings=general_settings,
        ))]

        for player_role in ('chooser', 'predictor'):
            population_parameter_distribution_histograms(
                general_settings=general_settings, file_paths=file_paths, figure_layout=figure_layout,
                player_role=player_role, use_initial_params=True, create_new_file=create_new_file,
            )
            for ratio_mode in ('absolute', 'skip_negative'):
                subpopulation_stats_and_param_ratio_histograms(
                    general_settings=general_settings, file_paths=file_paths, figure_layout=figure_layout,
                    player_role=player_role, use_initial_params=True, create_new_file=False,
                    ratio_mode=ratio_mode, as_subplots=True,
                )
            param_correlation_matrix_report(
                general_settings=general_settings, file_paths=file_paths,
                player_role=player_role, normalize_params=True,
            )

        param_correlation_matrix_report(
            general_settings=general_settings, file_paths=file_paths,
            player_role=player_role, cross_role_correlations=True, correction_method='holm',
        )

    if run_code_settings['run_inequality_aversion_analysis']:

        visualize_inequality_aversion_bot_competition(
            figure_layout=figure_layout, file_paths=file_paths,
            param_strong=0.75, param_weak=0.25, temperature=1.0,
            param_self_values=[0.0, 0.25, 0.5, 0.75, 1.0],
            param_altr_values=[0.0, 0.25, 0.5, 0.75, 1.0],
            ratio_numerator="envious", show_text_values=True, text_decimals=2,
            print_=True, export_fig=True, filename_stub=None,
            filter_constant_sum=False, color_range=[0.35, 0.55],
        )


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
    exit()
