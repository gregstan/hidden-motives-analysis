from main import *
import utilities as gnrl
import preprocessing as prep
from config import *

"=========================================================================================="
"=================================== Demo Configuration ==================================="
"=========================================================================================="

analysis_options = {
    'light_mode': True,           # True: fast/small versions. False: full scale (some sections take hours or months).

    # ── No external data required ────────────────────────────────────────────────────────
    'run_model_demos':              True,  # All 476 utility equations, Bayesian core checks, make_param_info.
    'run_nesting_tests':            True,  # Model nesting adjacency; equivalence and embedding sanity checks.

    # ── Synthetic simulation data ────────────────────────────────────────────────────────
    'run_simulation':               True,  # Parameter recovery simulation (core paper result, Figures 5–6).
    'run_particle_filter_test':     True,  # Particle filter vs full-grid posterior fidelity check.
    'run_recovery_by_k':            True,  # Recovery accuracy across model complexity levels (k params).
    'run_update_speed_analysis':    True,  # Belief update speed regression over simulated dyads.
    'visualize_belief_updates':     True,  # Interactive 3D Bayesian update plots (uses simulation data).

    # ── Requires raw experiment data ─────────────────────────────────────────────────────
    'run_model_comparison':         False, # Alternative model contest + typological model comparison.
    'run_ic_analysis':              False, # IC utility comparison: 5 forms in light mode, all 476 in full.
    'run_parameter_distribution':   False, # Population parameter distributions and correlations.
    'run_inequality_aversion':      False, # Inequality aversion bot competition heatmaps.
}

"=========================================================================================="
"==================================== Demo Entry Point ===================================="
"=========================================================================================="


def run_quick_demo(analysis_options: dict[str, bool]) -> None:
    """
    Runs a comprehensive demo of the Utility Bayesian Model (UBM) codebase.

    Covers all major analytical components of the paper in manuscript order: the utility
    model and equations, Bayesian inference machinery, model nesting infrastructure,
    parameter recovery simulations, particle filter validation, belief-update visualization,
    alternative model comparison, IC utility function comparison across 476 forms,
    population parameter distributions, and inequality aversion analysis.

    Arguments:
        • analysis_options: dict[str, bool]
            Controls which sections run and at what scale. Toggle 'light_mode' for
            fast versions of all sections. Toggle individual flags to include or exclude
            specific sections. Sections marked 'requires raw experiment data' will print
            a loud warning and skip if the raw CSV files are not found.

    Notes:
        • All demo outputs are written under demo_files/ and can be safely deleted.
        • Sections 'run_simulation' through 'visualize_belief_updates' use synthetic data only.
        • 'visualize_belief_updates' requires 'run_simulation' to have produced data first.
        • 'run_model_comparison', 'run_ic_analysis', 'run_parameter_distribution', and
            'run_inequality_aversion' require raw experiment CSVs in raw_data/.
        • NEVER set light_mode=False for 'run_ic_analysis' without expecting a multi-week run.
    """

    light_mode = analysis_options['light_mode']

    "=========================================================================================="
    "========================================= Setup =========================================="
    "=========================================================================================="

    "Apply light-mode speed settings to general_settings."
    general_settings['experiment_num'] = 0
    if light_mode:
        general_settings['use_particle_filter'] = True
        general_settings['n_bins_per_dimension'] = 5
        general_settings['run_in_parallel']      = True

    "Demo file paths: all outputs land in demo_files/ (safe to delete; never overwrites real results)."
    demo_root = ROOT / "demo_files"
    demo_file_paths = {
        "raw_data":    ROOT / "raw_data",
        "bic_aic":     demo_root / "bic_aic",
        "processed":   demo_root / "processed",
        "player_fits": demo_root / "player_fits",
        "param_data":  demo_root / "param_data",
        "dyad_data":   demo_root / "dyad_data",
        "discrete":    demo_root / "discrete",
        "visuals":     demo_root / "visuals",
        "file_names": {
            "player_pairs_exper0":       "Social_Preference_Prediction_Pairs_Exper0.json",
            "params_data_exper0_bayes":  "Social_Preference_Prediction_Parameters_Exper0_Bayes.json",
            "information_criterion":     "All_Utility_Forms_IC_Analysis_Experiment3.csv",
            **file_paths["file_names"],
        },
    }

    "Demo real-data file paths: reads from the true raw/processed directories but writes all"
    "outputs to demo_files/. This protects precomputed files (e.g., the IC analysis that took"
    "a month to run) from being overwritten by a smaller-scale demo run."
    demo_real_file_paths = {
        **file_paths,
        "bic_aic":     demo_root / "bic_aic",
        "player_fits": demo_root / "player_fits",
        "param_data":  demo_root / "param_data",
        "discrete":    demo_root / "discrete",
        "visuals":     demo_root / "visuals",
    }

    "Create all required output directories."
    experiment_num_for_distributions = general_settings.get('experiment_num', 3)
    required_demo_dirs = [
        demo_root / "bic_aic",
        demo_root / "processed",
        demo_root / "param_data",
        demo_root / "dyad_data",
        demo_root / "discrete",
        demo_root / "visuals" / "bayesian_updates_3d",
        demo_root / "visuals" / "inequality_aversion",
        demo_root / "player_fits" / "experiment_0",
        demo_root / "player_fits" / "loss_reports" / "experiment_0",
        demo_root / "player_fits" / "simulation_results",
        demo_root / "player_fits" / f"experiment_{experiment_num_for_distributions}",
        demo_root / "player_fits" / "loss_reports" / f"experiment_{experiment_num_for_distributions}",
    ]
    for directory in required_demo_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    "=========================================================================================="
    "===== Section 1: Utility Model — Equations, Parameters, and Core Bayesian Components ====="
    "=========================================================================================="
    "Manuscript context: Sections 2 and 3.1–3.2. No data required."
    "Functions exercised: build_utility_equation, utility, make_param_info, is_valid_utility_settings,"
    "generate_utility_settings, classify_pair_relation, verify_utility_vs_string_equation."

    if analysis_options['run_model_demos']:
        print("\n" + "=" * 70)
        print("SECTION 1: Utility model equations and Bayesian core demos")
        print("=" * 70)

        "Validate the active utility settings and show what parameters they imply."
        settings_are_valid = gnrl.is_valid_utility_settings(utility_settings, provide_explanation=True)
        print(f"\nis_valid_utility_settings(utility_settings) → {settings_are_valid}")

        demo_param_info = make_param_info(param_bds=param_bds, utility_settings=utility_settings)
        print(f"\nmake_param_info → parameter keys: {demo_param_info['keys']}")

        "Count and list all valid utility configurations."
        all_utility_setting_varieties = gnrl.generate_utility_settings(utility_settings=utility_settings, sort_by_k=False)
        print(f"\ngenerate_utility_settings → {len(all_utility_setting_varieties)} valid utility configurations")

        "Print the utility equation string for every configuration."
        print("\nAll utility equations via build_utility_equation (all 476 forms):")
        for equation_idx, utility_setting_variety in enumerate(all_utility_setting_varieties):
            equation_str = build_utility_equation(utility_settings=utility_setting_variety)
            print(f"  Equation {equation_idx:03d}: {equation_str}")

        "Demonstrate classify_pair_relation on a parent-child pair and a sibling pair."
        child_utility_settings   = {**utility_settings, 'include_social_comparison': False}
        sibling_utility_settings = {**utility_settings, 'single_payoffs_not_differences': True}
        parent_child_relation = gnrl.classify_pair_relation(child_utility_settings, utility_settings)
        sibling_relation      = gnrl.classify_pair_relation(utility_settings, sibling_utility_settings)
        print(f"\nclassify_pair_relation examples:")
        print(f"  + social_comparison added: '{parent_child_relation}'  (expected: 'parent_child')")
        print(f"  single_payoffs vs diffs:   '{sibling_relation}'       (expected: 'sibling')")

        "Verify that utility() and build_utility_equation() produce identical choice probabilities."
        n_verification_games = 20 if light_mode else 625
        print(f"\nRunning verify_utility_vs_string_equation on {n_verification_games} games across all 476 forms...")
        verify_utility_vs_string_equation(
            utility_function=utility,
            utility_function_str=build_utility_equation,
            utility_settings=utility_settings,
            param_bds=param_bds,
            n_games=n_verification_games,
            rng_seed=20250420,
            exhaustive_if_large=False,
            option="A",
            file_paths=demo_file_paths,
            comparison_tol=1e-6,
            decimals=6,
            verbose=True,
        )

    "=========================================================================================="
    "============== Section 2: Model Nesting Infrastructure and Validity Checks ==============="
    "=========================================================================================="
    "Manuscript context: Section 4.4 (nesting logic and parent-fair regularization)."
    "Functions exercised: model_nesting_adjacency_matrices, summarize_nesting_relationship_counts,"
    "test_utility_functions, run_child_parent_probability_equivalence_smoketest,"
    "run_child_parent_embedding_sanity_checks."

    if analysis_options['run_nesting_tests']:
        print("\n" + "=" * 70)
        print("SECTION 2: Model nesting adjacency matrices and validity checks")
        print("=" * 70)

        model_nesting_adjacency_matrices(
            general_settings=general_settings,
            utility_settings=utility_settings,
            file_paths=demo_file_paths,
            create_new_file=True,
            equation_form=True,
            print_=True,
        )

        gnrl.summarize_nesting_relationship_counts(
            general_settings=general_settings,
            utility_settings=utility_settings,
            file_paths=demo_file_paths,
            model_nesting_adjacency_matrices=model_nesting_adjacency_matrices,
            create_new_file=True,
            print_=True,
        )

        gnrl.test_utility_functions(
            build_utility_equation=build_utility_equation,
            general_settings=general_settings,
            utility_settings=utility_settings,
            setting_to_flip='include_social_comparison',
            print_=True,
        )

        run_child_parent_probability_equivalence_smoketest(
            utility_settings=utility_settings,
            file_paths=demo_file_paths,
            param_bds=param_bds,
            rand_payoff_idx=True,
            n_trials=12 if light_mode else 50,
            rng_seed=20250420,
            tolerance=1e-12,
            verbose=True,
        )

        fit_for_n_players = 1 if light_mode else 3
        run_child_parent_embedding_sanity_checks(
            general_settings=general_settings,
            file_paths=demo_file_paths,
            param_bds=param_bds,
            utility_settings=utility_settings,
            player_role_to_fit="chooser",
            fit_for_n_players=fit_for_n_players,
            random_seed=20250420,
            numeric_tolerance=1e-3,
            verbose=True,
        )

    "=========================================================================================="
    "============== Section 3: Simulation — Parameter Recovery (Figures 5 and 6) =============="
    "=========================================================================================="
    "Manuscript context: Section 3.3.6 (simulation validation, parameter recovery)."
    "Functions exercised: create_simulated_data, run_simulation_recovery_analysis,"
    "compute_recovery_by_prior_bins, plot_param_recovery_by_round."

    if analysis_options['run_simulation']:
        print("\n" + "=" * 70)
        print("SECTION 3: Parameter recovery simulation")
        print("=" * 70)

        if light_mode:
            params_chooser_range   = {'Vᵢᵢ': (1, 1, 1), 'Vᵢⱼ': (-1, 1, 7), 'std': (1.0, 1.0, 1), 'τ': (0.5, 0.5, 1)}
            params_predictor_range = {'Vᵢᵢ': (1, 1, 1), 'Vᵢⱼ': (-1, 1, 7), 'std': (1.0, 1.0, 1), 'τ': (0.5, 0.5, 1)}
        else:
            params_chooser_range   = {'Vᵢᵢ': (1, 1, 1), 'Vᵢⱼ': (-1, 1, 5), 'std': (1.0, 1.0, 1), 'τ': (0.5, 3, 3)}
            params_predictor_range = {'Vᵢᵢ': (1, 1, 1), 'Vᵢⱼ': (-1, 1, 7), 'std': (0.5, 1.5, 3), 'τ': (0.5, 3, 3)}

        create_simulated_data(
            n_games=24,
            randomize_parameters=False,
            param_bds=param_bds,
            file_paths=demo_file_paths,
            run_analysis=True,
            params_chooser_range=params_chooser_range,
            params_predictor_range=params_predictor_range,
            utility_settings=utility_settings,
            dynamic_predictor=True,
        )

        df_merged = run_simulation_recovery_analysis(
            general_settings=general_settings,
            file_paths=demo_file_paths,
            fig_lay=fig_lay,
            export_fig=True,
            create_new_file=True,
            produce_figures=True,
            correlation_csv_name="correlation_results.csv",
            include_dropdown=False,
            use_dynamic_predictor=True,
        )

        if df_merged is not None:
            compute_recovery_by_prior_bins(
                df=df_merged,
                var_col="Vᵢⱼ_std_true_predictor",
                temp_col="temp_true_predictor",
                param_true_chooser="Vᵢⱼ_true_chooser",
                param_fitted_predictor="Vᵢⱼ_sim_pred_predictor",
                player_id_col="player_uuid_predictor",
                last_rounds=[18, 19, 20],
                var_edges=None,
                temp_edges=None,
                print_=True,
            )

            plot_param_recovery_by_round(df_merged=df_merged, fig_lay=fig_lay)

    "=========================================================================================="
    "=============== Section 4: Particle Filter Fidelity vs Full-Grid Posterior ==============="
    "=========================================================================================="
    "Manuscript context: Section 3.3.5 (particle filter validation, correlation > 0.993)."
    "Functions exercised: verify_particle_filter_fidelity."

    if analysis_options['run_particle_filter_test']:
        print("\n" + "=" * 70)
        print("SECTION 4: Particle filter vs full-grid posterior fidelity")
        print("=" * 70)

        if light_mode:
            particle_filter_sample_ratios = [0.1]
            particle_filter_n_predictors  = 4
            particle_filter_n_games       = 6
        else:
            particle_filter_sample_ratios = list(np.round(np.linspace(start=0.05, stop=0.95, num=19), decimals=3))
            particle_filter_n_predictors  = 20
            particle_filter_n_games       = 12

        verify_particle_filter_fidelity(
            general_settings=general_settings,
            utility_settings=utility_settings,
            param_info=param_info,
            file_paths=demo_file_paths,
            fig_lay=fig_lay,
            sample_ratios=particle_filter_sample_ratios,
            n_predictors=particle_filter_n_predictors,
            n_games_per_dyad=particle_filter_n_games,
        )

    "=========================================================================================="
    "============ Section 5: Parameter Recovery Across Model Complexity (k params) ============"
    "=========================================================================================="
    "Manuscript context: Section 3.3.6 (how recovery quality varies with model dimensionality)."
    "Functions exercised: run_param_recovery_by_k."

    if analysis_options['run_recovery_by_k']:
        print("\n" + "=" * 70)
        print("SECTION 5: Parameter recovery across model complexity (k params)")
        print("=" * 70)

        run_param_recovery_by_k(
            n_games                  = 8 if light_mode else 28,
            n_predictors             = 4 if light_mode else 70,
            n_choosers_per_predictor = 2 if light_mode else 3,
            k_params_range           = (1, 3) if light_mode else (1, 9),
            n_altruism_steps         = 3 if light_mode else 7,
            evenly_space_altruism    = True,
            utility_settings_by_k    = None,
            general_settings         = general_settings,
            file_paths               = demo_file_paths,
            fig_lay                  = fig_lay,
            param_bds                = param_bds,
            analysis_experiment_num  = 0,
            random_seed              = 20250420,
        )

    "=========================================================================================="
    "======================== Section 6: Belief Update Speed Analysis ========================="
    "=========================================================================================="
    "Manuscript context: Section 3.3.6 (prior variance and temperature predict update speed)."
    "Functions exercised: run_update_speed_simulation_regression."
    "Requires: simulation data from Section 3 (run_simulation must have run first)."

    if analysis_options['run_update_speed_analysis']:
        print("\n" + "=" * 70)
        print("SECTION 6: Belief update speed regression (synthetic simulation data)")
        print("=" * 70)

        simulation_json_present = (
            demo_file_paths["processed"] / demo_file_paths["file_names"]["player_pairs_exper0"]
        ).exists()
        if not simulation_json_present:
            print(
                "\nWARNING: No simulation data found for update speed analysis.\n"
                f"Expected: {demo_file_paths['processed'] / demo_file_paths['file_names']['player_pairs_exper0']}\n"
                "Set analysis_options['run_simulation'] = True and rerun to generate it first."
            )
        else:
            run_update_speed_simulation_regression(
                general_settings=general_settings,
                file_paths=demo_file_paths,
                params_of_interest=['Vᵢⱼ'],
                use_true_params=False,
                n_dyads=49 if light_mode else None,
            )

    "=========================================================================================="
    "============= Section 7: 3D Bayesian Belief Update Visualization (Figure 4) =============="
    "=========================================================================================="
    "Manuscript context: Section 3.3.4 and Figure 4 (posterior concentration over sequential games)."
    "Functions exercised: visualize_bayesian_updates_3d."
    "Requires: simulation data from Section 3 (run_simulation must have run first)."

    if analysis_options['visualize_belief_updates']:
        print("\n" + "=" * 70)
        print("SECTION 7: 3D Bayesian belief update visualization")
        print("=" * 70)

        simulation_pairs_path = (
            demo_file_paths["processed"] / demo_file_paths["file_names"]["player_pairs_exper0"]
        )
        if not simulation_pairs_path.exists():
            print(
                "\nWARNING: No simulation data found for visualization.\n"
                f"Expected: {simulation_pairs_path}\n"
                "Set analysis_options['run_simulation'] = True and rerun to generate it first."
            )
        else:
            general_settings['update_method'] = 'sim_pred'
            n_total_dyads        = 49 if light_mode else 945
            n_dyads_to_visualize = 9 if light_mode else 45
            dyad_indices = sorted(random.sample(range(n_total_dyads), n_dyads_to_visualize))
            for dyad_idx in dyad_indices:
                visualize_bayesian_updates_3d(
                    dyad_games_or_key=dyad_idx,
                    player_uuid=2,
                    fig_lay=fig_lay,
                    file_paths=demo_file_paths,
                    general_settings=general_settings,
                    fix_z_axis=True,
                )

    "=========================================================================================="
    "=========== Section 8: Alternative Model Competition + Typological Comparison ============"
    "=========================================================================================="
    "Manuscript context: Sections 3.4–3.5 (UBM vs non-Bayesian and discrete-type alternatives)."
    "Functions exercised: alternative_model_contest, typological_model_comparison_fit_individually."
    "Requires: raw experiment data (Experiments 1 and 2) in raw_data/."

    if analysis_options['run_model_comparison']:
        print("\n" + "=" * 70)
        print("SECTION 8: Alternative model contest + typological model comparison")
        print("=" * 70)

        required_raw_data_files = [
            file_paths["file_names"]["raw_data_exper1"],
            file_paths["file_names"]["raw_data_exper2"],
        ]
        missing_raw_data_files = [
            f for f in required_raw_data_files
            if not (ROOT / "raw_data" / f).exists()
        ]
        if missing_raw_data_files:
            print(
                "\n" + "!" * 70 + "\n"
                "CRITICAL WARNING: Raw experiment data files not found:\n"
                + "".join(f"  {ROOT / 'raw_data' / f}\n" for f in missing_raw_data_files)
                + "Section 8 cannot run without these files.\n"
                "Raw participant data should always be present — this is unexpected.\n"
                + "!" * 70 + "\n"
            )
        else:
            model_comparison_settings = {**general_settings, 'experiment_num': 2}

            alternative_model_contest(
                general_settings=model_comparison_settings,
                param_info=param_info,
                param_bds=param_bds,
                utility_settings=utility_settings,
                file_paths=demo_real_file_paths,
                fig_lay=fig_lay,
            )

            winning_k3_profile = [( 0.0,  0.5), ( 0.0,  1.0), ( 0.5,  0.0)]
            typological_model_comparison_fit_individually(
                best_profiles=winning_k3_profile,
                general_settings=model_comparison_settings,
                file_paths=demo_real_file_paths,
                penalty_weight=10,
                maxiter_global=5 if light_mode else None,
                maxiter_local=5 if light_mode else None,
                optimization_method='globloc',
                save_csv=True,
            )

    "=========================================================================================="
    "======= Section 9: Information Criterion Utility Function Comparison (476 Models) ========"
    "=========================================================================================="
    "Manuscript context: Section 4 (near-comprehensive IC comparison across 476 utility forms)."
    "Functions exercised: gnrl.identify_redundant_utility_functions, gnrl.equation_to_settings,"
    "information_criterion_analysis, plot_ic_scores_delta_bic, plot_ic_robustness_analysis,"
    "utility_setting_contribution_analysis, extract_rankings_of_canonical_utility_functions."
    "Requires: raw experiment 3 data. WARNING: full mode (light_mode=False) takes weeks."

    if analysis_options['run_ic_analysis']:
        print("\n" + "=" * 70)
        print("SECTION 9: IC utility function comparison")
        print(f"  Mode: {'5 representative forms (light)' if light_mode else 'all 476 forms (FULL — may take weeks)'}")
        print("=" * 70)

        raw_data_exper3_path = ROOT / "raw_data" / file_paths["file_names"]["raw_data_exper3"]
        if not raw_data_exper3_path.exists():
            print(
                "\n" + "!" * 70 + "\n"
                "CRITICAL WARNING: Raw experiment 3 data not found at:\n"
                f"  {raw_data_exper3_path}\n"
                "Section 9 cannot run without this file.\n"
                "Raw participant data should always be present — this is unexpected.\n"
                + "!" * 70 + "\n"
            )
        else:
            ic_general_settings = {**general_settings, 'experiment_num': 3}

            gnrl.identify_redundant_utility_functions(
                utility_settings=utility_settings,
                build_equation_function=build_utility_equation,
                file_paths=demo_real_file_paths,
            )
            gnrl.equation_to_settings(
                equation_function=build_utility_equation,
                utility_settings=utility_settings,
                file_paths=demo_real_file_paths,
                create_new_file=True,
            )

            "In light mode, select one representative form for each k level from k=1 to k=5."
            "In full mode, pass None so information_criterion_analysis generates all 476."
            if light_mode:
                all_varieties_by_k = gnrl.generate_utility_settings(
                    utility_settings=utility_settings, sort_by_k=True
                )
                ic_demo_varieties = []
                seen_k_values     = set()
                for settings_variety in all_varieties_by_k:
                    k_count = len(parameter_keys_for_utility_settings(utility_settings=settings_variety))
                    if k_count not in seen_k_values and k_count <= 5:
                        ic_demo_varieties.append(settings_variety)
                        seen_k_values.add(k_count)
                    if len(ic_demo_varieties) >= 5:
                        break
                print(f"  Running IC analysis on {len(ic_demo_varieties)} utility forms (one per k, k=1..5)")
            else:
                ic_demo_varieties = None

            information_criterion_analysis(
                general_settings=ic_general_settings,
                utility_settings=utility_settings,
                file_paths=demo_real_file_paths,
                param_bds=param_bds,
                max_iters=1 if light_mode else 24,
                robustness_epsilon=36,
                check_for_n_players=2 if light_mode else 'all',
                utility_setting_varieties=ic_demo_varieties,
            )

            plot_ic_scores_delta_bic(
                fig_lay=fig_lay,
                file_paths=demo_real_file_paths,
                general_settings=ic_general_settings,
                include_dropdown=False,
            )

            plot_ic_robustness_analysis(
                general_settings=ic_general_settings,
                file_paths=demo_real_file_paths,
                fig_lay=fig_lay,
            )

            utility_setting_contribution_analysis(
                general_settings=ic_general_settings,
                file_paths=demo_real_file_paths,
                utility_settings_universe=utility_settings,
                score_col="BIC",
                use_edge_types=("sibling", "parent_child"),
                include_non_network_toggles=True,
                export_csv=True,
            )

            extract_rankings_of_canonical_utility_functions(
                file_paths=demo_real_file_paths,
                rank_col="BIC",
                print_=True,
            )

    "=========================================================================================="
    "============ Section 10: Population Parameter Distributions and Correlations ============="
    "=========================================================================================="
    "Manuscript context: Section 5 (parameter estimates, cross-role correlations, ratios)."
    "Functions exercised: run_analysis_bayes (or mle), population_parameter_distribution_histograms,"
    "subpopulation_stats_and_param_ratio_histograms, param_correlation_matrix_report."
    "Requires: raw experiment data. The fitting step is computationally expensive."

    if analysis_options['run_parameter_distribution']:
        print("\n" + "=" * 70)
        print("SECTION 10: Population parameter distribution results")
        print("=" * 70)

        experiment_num_for_distributions = general_settings.get('experiment_num', 3)
        raw_data_key  = f"raw_data_exper{experiment_num_for_distributions}"
        raw_data_path = ROOT / "raw_data" / file_paths["file_names"][raw_data_key]
        if not raw_data_path.exists():
            print(
                "\n" + "!" * 70 + "\n"
                f"CRITICAL WARNING: Raw experiment {experiment_num_for_distributions} data not found at:\n"
                f"  {raw_data_path}\n"
                "Section 10 cannot run without this file.\n"
                "Raw participant data should always be present — this is unexpected.\n"
                + "!" * 70 + "\n"
            )
        else:
            distribution_settings  = {**general_settings, 'experiment_num': experiment_num_for_distributions}
            run_analysis_function  = run_analysis_bayes if general_settings['analysis_mode'] == 'bayesian' else run_analysis_mle

            print(f"Active utility equation: {build_utility_equation(utility_settings=utility_settings)}\n")
            all_histories_data = prep.all_histories(file_paths=file_paths, experiment_numbers=[1, 2, 3])
            run_analysis_function(
                histories_data=all_histories_data[experiment_num_for_distributions - 1],
                file_paths=demo_real_file_paths,
                param_info=param_info,
                utility_settings=utility_settings,
                general_settings=distribution_settings,
            )

            for player_role in ('chooser', 'predictor'):
                population_parameter_distribution_histograms(
                    general_settings=distribution_settings,
                    file_paths=demo_real_file_paths,
                    fig_lay=fig_lay,
                    player_role=player_role,
                    use_initial_params=True,
                    create_new_file=True,
                )
                for ratio_mode in ('absolute', 'skip_negative'):
                    subpopulation_stats_and_param_ratio_histograms(
                        general_settings=distribution_settings,
                        file_paths=demo_real_file_paths,
                        fig_lay=fig_lay,
                        player_role=player_role,
                        use_initial_params=True,
                        create_new_file=False,
                        ratio_mode=ratio_mode,
                        as_subplots=True,
                    )
                param_correlation_matrix_report(
                    general_settings=distribution_settings,
                    file_paths=demo_real_file_paths,
                    player_role=player_role,
                    normalize_params=True,
                )

            param_correlation_matrix_report(
                general_settings=distribution_settings,
                file_paths=demo_real_file_paths,
                player_role='predictor',
                cross_role_correlations=True,
                correction_method='holm',
            )

    "=========================================================================================="
    "================ Section 11: Inequality Aversion Bot Competition Heatmaps ================"
    "=========================================================================================="
    "Manuscript context: Section 5.4 (envy vs guilt asymmetry competition)."
    "Functions exercised: visualize_inequality_aversion_bot_competition."
    "Requires: raw experiment data (to load trial histories)."

    if analysis_options['run_inequality_aversion']:
        print("\n" + "=" * 70)
        print("SECTION 11: Inequality aversion bot competition heatmaps")
        print("=" * 70)

        raw_data_exper1_path = ROOT / "raw_data" / file_paths["file_names"]["raw_data_exper1"]
        if not raw_data_exper1_path.exists():
            print(
                "\n" + "!" * 70 + "\n"
                "CRITICAL WARNING: Raw experiment 1 data not found at:\n"
                f"  {raw_data_exper1_path}\n"
                "Section 11 cannot run without this file.\n"
                "Raw participant data should always be present — this is unexpected.\n"
                + "!" * 70 + "\n"
            )
        else:
            visualize_inequality_aversion_bot_competition(
                fig_lay=fig_lay,
                file_paths=demo_real_file_paths,
                param_strong=0.75,
                param_weak=0.25,
                temperature=1.0,
                param_self_values=[0.0, 0.25, 0.5, 0.75, 1.0],
                param_altr_values=[0.0, 0.25, 0.5, 0.75, 1.0],
                ratio_numerator="envious",
                show_text_values=True,
                text_decimals=2,
                print_=True,
                export_fig=True,
                filename_stub=None,
                filter_constant_sum=False,
                color_range=[0.35, 0.55],
            )


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    run_quick_demo(analysis_options=analysis_options)
