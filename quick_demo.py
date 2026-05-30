from main import *
import utilities as gnrl
from config import *

"=========================================================================================="
"=================================== Demo Configuration ==================================="
"=========================================================================================="

analysis_options = {
    'light_mode':                   True,   # True: fast/small versions. False: full scale (some sections take hours or months).

    # ── No external data required ────────────────────────────────────────────────────────
    'run_model_demos':              False,  # All 505 utility equations, Bayesian core checks, make_param_info.
    'run_nesting_tests':            False,  # Model nesting adjacency; equivalence and embedding sanity checks.

    # ── Synthetic simulation data ────────────────────────────────────────────────────────
    'run_simulation':               False,  # Parameter recovery simulation (core paper result, Figures 5–6).
    'run_particle_filter_test':     False,  # Particle filter vs full-grid posterior fidelity check.
    'run_recovery_by_k':            False,  # Recovery accuracy across model complexity levels (k params).
    'run_update_speed_analysis':    False,  # Belief update speed regression over simulated dyads.
    'visualize_belief_updates':     False,  # Interactive 3D Bayesian update plots (uses simulation data).

    # ── Requires raw experiment data ─────────────────────────────────────────────────────
    'run_model_comparison':         False,  # Alternative model contest + typological model comparison.
    'run_ic_analysis':              True,  # IC utility comparison: 5 forms in light mode, all 505 in full.

    # ── Requires IC results in bic_aic/ ──────────────────────────────────────────────────
    'run_individual_architecture':  True,  # Architecture compression curve: how many utility types does the population need?
    'run_model_recovery':           False,  # Model recovery simulation: data adequacy check for the IC pipeline.

    # ── Requires raw experiment data ─────────────────────────────────────────────────────
    'run_parameter_distribution':   True,  # Population parameter distributions and correlations.
    'run_inequality_aversion':      True,  # Inequality aversion bot competition heatmaps.
} 

"=========================================================================================="
"==================================== Demo Entry Point ===================================="
"=========================================================================================="

def _section_header(text: str, width: int = 70) -> str:
    padding = width - 2 - len(text)
    if padding < 0:
        return text
    left  = padding // 2
    right = padding - left
    return "=" * left + " " + text + " " + "=" * right


def run_quick_demo(analysis_options: dict[str, bool]) -> None:
    """
    Runs a comprehensive demo of the Utility Bayesian Model (UBM) codebase.

    Covers all major analytical components of the paper in manuscript order: the utility
    model and equations, Bayesian inference machinery, model nesting infrastructure,
    parameter recovery simulations, particle filter validation, belief-update visualization,
    alternative model comparison, IC utility function comparison across 505 forms,
    individual architecture compression curve, model recovery simulation,
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
        • 'run_model_comparison', 'run_ic_analysis', and 'run_parameter_distribution' require
            raw experiment CSVs in raw_data/. 'run_inequality_aversion' is parametric only.
        • NEVER set light_mode=False for 'run_ic_analysis' without expecting a multi-week run.
    """

    light_mode = analysis_options['light_mode']

    "=========================================================================================="
    "========================================= Setup =========================================="
    "=========================================================================================="

    "Apply light-mode speed settings to general_settings."
    general_settings['experiment_num'] = 0
    if light_mode:
        general_settings['use_particle_filter']  = True
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

    """
    Demo real-data file paths: reads from the true raw/processed directories but writes all
    outputs to demo_files/. This protects precomputed files (e.g., the IC analysis that took
    a month to run) from being overwritten by a smaller-scale demo run.
    """
    demo_real_file_paths = {
        **file_paths,
        "processed":   demo_root / "processed",
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
        demo_root / "simulations",
        demo_root / "player_fits" / "experiment_2",
        demo_root / "player_fits" / "loss_reports" / "experiment_2",
        demo_root / "player_fits" / "experiment_3",
        demo_root / "player_fits" / "loss_reports" / "experiment_3",
        demo_root / "player_fits" / f"experiment_{experiment_num_for_distributions}",
        demo_root / "player_fits" / "loss_reports" / f"experiment_{experiment_num_for_distributions}",
    ]
    for directory in required_demo_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    def _seed_demo_processed(demo_processed: Path, real_processed: Path, file_names: dict) -> None:
        """Copy read-only input files from real processed/ into demo_files/processed/ on first run."""
        import shutil
        seed_keys = [k for k in file_names if k.startswith(("player_pairs_exper", "processed_data_exper"))]
        for key in seed_keys:
            fname = file_names[key]
            src   = real_processed / fname
            dest  = demo_processed / fname
            if src.exists() and not dest.exists():
                shutil.copy2(src, dest)
                print(f"  [demo seed] copied {fname} → demo_files/processed/")
        registry_src  = real_processed / "all_utility_functions.csv"
        registry_dest = demo_processed  / "all_utility_functions.csv"
        if registry_src.exists() and not registry_dest.exists():
            shutil.copy2(registry_src, registry_dest)
            print("  [demo seed] copied all_utility_functions.csv → demo_files/processed/")

    _seed_demo_processed(
        demo_processed=demo_root / "processed",
        real_processed=ROOT / "processed",
        file_names=file_paths["file_names"],
    )

    "=========================================================================================="
    "===== Section 1: Utility Model — Equations, Parameters, and Core Bayesian Components ====="
    "=========================================================================================="
    """
    Manuscript context: Sections 2 and 3.1–3.2. No data required.
    Functions exercised: build_utility_equation, utility, make_param_info, is_valid_utility_settings,
    generate_utility_settings, classify_pair_relation, verify_utility_vs_string_equation.
    """

    if analysis_options['run_model_demos']:
        print("\n" + _section_header("SECTION 1: Utility model equations and Bayesian core demos"))

        "Validate the active utility settings and show what parameters they imply."
        settings_are_valid = gnrl.is_valid_utility_settings(utility_settings, provide_explanation=True)
        print(f"\nis_valid_utility_settings(utility_settings) → {settings_are_valid}")

        demo_param_info = make_param_info(param_bds=param_bds, utility_settings=utility_settings, general_settings=general_settings)
        print(f"\nmake_param_info → parameter keys: {demo_param_info['keys']}")

        "Count and list all valid utility configurations."
        all_utility_setting_varieties = gnrl.generate_utility_settings(utility_settings=utility_settings, sort_by_k=False)
        print(f"\ngenerate_utility_settings → {len(all_utility_setting_varieties)} valid utility configurations")

        "Print the utility equation string for every configuration."
        print("\nAll utility equations via build_utility_equation (all 505 forms):")
        for equation_idx, utility_setting_variety in enumerate(all_utility_setting_varieties):
            equation_str = build_utility_equation(utility_settings=utility_setting_variety)
            print(f"  Equation {equation_idx:03d}: {equation_str}")

        "Demonstrate classify_pair_relation on a parent-child pair and a sibling pair."
        child_utility_settings   = {**utility_settings, 'include_social_comparison': False}
        sibling_utility_settings = {**utility_settings, 'single_payoffs_not_differences': True}
        parent_child_relation = gnrl.classify_pair_relation(
            model_1=child_utility_settings, model_2=utility_settings,
            utility_settings=utility_settings, general_settings=general_settings,
        )
        sibling_relation = gnrl.classify_pair_relation(
            model_1=utility_settings, model_2=sibling_utility_settings,
            utility_settings=utility_settings, general_settings=general_settings,
        )
        print(f"\nclassify_pair_relation examples:")
        pc_rel, pc_rev, pc_flag = parent_child_relation
        sib_rel, sib_rev, sib_flag = sibling_relation
        print(f"  + social_comparison added: '{pc_rel}' / '{pc_rev}'  (expected: 'child' / 'parent', flag={pc_flag})")
        print(f"  single_payoffs vs diffs:   '{sib_rel}' / '{sib_rev}'  (expected: 'sibling' / 'sibling', flag={sib_flag})")

        "Verify that utility() and build_utility_equation() produce identical choice probabilities."
        n_verification_games = 20 if light_mode else 625
        print(f"\nRunning verify_utility_vs_string_equation on {n_verification_games} games across all 505 forms...")
        verify_utility_vs_string_equation(
            utility_function=utility,
            utility_function_str=build_utility_equation,
            utility_settings=utility_settings,
            param_bds=param_bds,
            n_games=n_verification_games,
            random_seed=None,
            exhaustive_if_large=False,
            option="A",
            file_paths=demo_file_paths,
            comparison_tol=1e-6,
            decimals=6,
            verbose=True,
        )

        "Demo the two new structural families added in the 505-model expansion."
        print("\n── New family 1: Welfare Efficiency (Engelmann-Strobel) ──")
        "Build minimal welfare efficiency settings: only the flags that matter for this family."
        _welf_base = {**utility_settings,
            'include_welfare_efficiency_term': True,
            'include_altruism_term':           False,   # blocked by validator when combined with soc_comp
            'include_social_comparison':       False,
            'use_exponential_parameters':      False,
            'single_payoffs_not_differences':  True,
        }
        welf_basic = _welf_base
        welf_full  = {**_welf_base, 'include_social_comparison': True}   # adds Vᵢⱼ×min(πᵢ,πⱼ) — full E&S
        welf_exp   = {**welf_full,  'use_exponential_parameters': True}
        print(f"  E&S basic (no maximin)  : {build_utility_equation(welf_basic)}")
        print(f"  E&S full (with maximin) : {build_utility_equation(welf_full)}")
        print(f"  E&S full + exponents    : {build_utility_equation(welf_exp)}")

        print("\n── New family 2: Relative Income Penalty (Bolton-Ockenfels ERC) ──")
        "Build self-contained RIP settings; inherit only structural flags from base."
        _rip_base = {**utility_settings,
            'include_relative_income_penalty': True,
            'include_altruism_term':           False,
            'include_social_comparison':       False,
            'use_exponential_parameters':      False,
            'single_payoffs_not_differences':  True,
        }
        rip_basic     = _rip_base
        rip_canonical = {**_rip_base, 'fix_self_interest_parameter': True,
                         'use_exponential_parameters': True, 'single_exponential_parameter': True}
        rip_two_exp   = {**_rip_base, 'fix_self_interest_parameter': False,
                         'use_exponential_parameters': True, 'single_exponential_parameter': False}
        print(f"  RIP basic                 : {build_utility_equation(rip_basic)}")
        print(f"  Canonical ERC (B-O 2000)  : {build_utility_equation(rip_canonical)}")
        print(f"  RIP two exponents (Vᵢᵢ+pen): {build_utility_equation(rip_two_exp)}")

    "=========================================================================================="
    "============== Section 2: Model Nesting Infrastructure and Validity Checks ==============="
    "=========================================================================================="
    """
    Manuscript context: Section 4.4 (nesting logic and parent-fair regularization).
    Functions exercised: model_nesting_adjacency_matrices, summarize_nesting_relationship_counts,
    test_utility_functions, run_child_parent_probability_equivalence_smoketest,
    verify_same_inputs_same_outputs_for_children_and_parents.
    """

    if analysis_options['run_nesting_tests']:
        print("\n" + _section_header("SECTION 2: Model nesting adjacency matrices and validity checks"))

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
            random_seed=None,
            tolerance=1e-12,
            verbose=True,
        )

        fit_for_n_players = 1 if light_mode else 3
        verify_same_inputs_same_outputs_for_children_and_parents(
            general_settings=general_settings,
            file_paths=demo_file_paths,
            param_bds=param_bds,
            utility_settings=utility_settings,
            player_role_to_fit="chooser",
            fit_for_n_players=fit_for_n_players,
            random_seed=None,
            numeric_tolerance=1e-3,
            verbose=True,
        )

    "=========================================================================================="
    "============== Section 3: Simulation — Parameter Recovery (Figures 5 and 6) =============="
    "=========================================================================================="
    """
    Manuscript context: Section 3.3.6 (simulation validation, parameter recovery).
    Functions exercised: create_simulated_data, run_parameter_recovery_simulation,
    tabulate_recovery_correlations_by_prior_bins, plot_param_recovery_correlation_by_round.
    """

    if analysis_options['run_simulation']:
        print("\n" + _section_header("SECTION 3: Parameter recovery simulation"))

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

        df_merged = run_parameter_recovery_simulation(
            general_settings=general_settings,
            file_paths=demo_file_paths,
            figure_layout=figure_layout,
            export_fig=True,
            create_new_file=True,
            produce_figures=True,
            correlation_csv_name="correlation_results.csv",
            include_dropdown=False,
            use_dynamic_predictor=True,
        )

        if df_merged is not None:
            plot_param_recovery_correlation_by_round(
                df_merged=df_merged,
                general_settings=general_settings,
                file_paths=demo_file_paths,
                figure_layout=figure_layout,
            )

    "=========================================================================================="
    "=============== Section 4: Particle Filter Fidelity vs Full-Grid Posterior ==============="
    "=========================================================================================="
    """
    Manuscript context: Section 3.3.5 (particle filter validation, correlation > 0.993).
    Functions exercised: verify_particle_filter_fidelity.
    """

    if analysis_options['run_particle_filter_test']:
        print("\n" + _section_header("SECTION 4: Particle filter vs full-grid posterior fidelity"))

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
            figure_layout=figure_layout,
            sample_ratios=particle_filter_sample_ratios,
            n_predictors=particle_filter_n_predictors,
            n_games_per_dyad=particle_filter_n_games,
        )

    "=========================================================================================="
    "============ Section 5: Parameter Recovery Across Model Complexity (k params) ============"
    "=========================================================================================="
    """
    Manuscript context: Section 3.3.6 (how recovery quality varies with model dimensionality).
    Functions exercised: run_param_recovery_by_k.
    """

    if analysis_options['run_recovery_by_k']:
        print("\n" + _section_header("SECTION 5: Parameter recovery across model complexity (k params)"))

        run_param_recovery_by_k(
            n_games                  = 8 if light_mode else 28,
            n_predictors             = 4 if light_mode else 70,
            n_choosers_per_predictor = 2 if light_mode else 3,
            k_params_range           = (1, 3) if light_mode else (1, 9),
            n_altruism_steps         = 3 if light_mode else 7,
            evenly_space_altruism    = True,
            utility_settings_by_k    = None,
            general_settings         = general_settings,
            file_paths               = {**demo_file_paths, "bic_aic": file_paths["bic_aic"]},
            figure_layout                  = figure_layout,
            param_bds                = param_bds,
            analysis_experiment_num  = 0,
            random_seed              = None,
        )

    "=========================================================================================="
    "======================== Section 6: Belief Update Speed Analysis ========================="
    "=========================================================================================="
    """
    Manuscript context: Section 3.3.6 (prior variance and temperature predict update speed).
    Functions exercised: run_update_speed_simulation_regression.
    Requires: simulation data from Section 3 (run_simulation must have run first).
    """

    if analysis_options['run_update_speed_analysis']:
        print("\n" + _section_header("SECTION 6: Belief update speed regression (synthetic simulation data)"))

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
                n_dyads=49 if light_mode else None,
            )

    "=========================================================================================="
    "============= Section 7: 3D Bayesian Belief Update Visualization (Figure 4) =============="
    "=========================================================================================="
    """
    Manuscript context: Section 3.3.4 and Figure 4 (posterior concentration over sequential games).
    Functions exercised: visualize_bayesian_updates_3d.
    Requires: simulation data from Section 3 (run_simulation must have run first).
    """

    if analysis_options['visualize_belief_updates']:
        print("\n" + _section_header("SECTION 7: 3D Bayesian belief update visualization"))

        simulation_pairs_path = (
            demo_file_paths["processed"] / demo_file_paths["file_names"]["player_pairs_exper0"]
        )
        fits_dir = os.path.join(str(demo_file_paths['player_fits']), 'experiment_0')
        basic_files = sorted([
            f for f in os.listdir(fits_dir)
            if f.startswith('synthetic_predictor_') and f.endswith('.json')
        ]) if os.path.isdir(fits_dir) else []

        if not simulation_pairs_path.exists() or not basic_files:
            print(
                "\nWARNING: No simulation data found for visualization.\n"
                f"Expected: {pretty_path(simulation_pairs_path)}\n"
                "Set analysis_options['run_simulation'] = True and rerun to generate it first."
            )
        else:
            "Check that at least one file has been analyzed (has parameter_estimates with grid data)."
            _analyzed = False
            for _f in basic_files[:10]:
                _sp = os.path.join(fits_dir, _f)
                if os.path.getsize(_sp) > 20:
                    with open(_sp, encoding='utf-8') as _fh:
                        _sd = json.load(_fh)
                    _sg = next(iter(_sd.values()))
                    if any('parameter_estimates' in g for g in _sg):
                        _analyzed = True
                        break

            if not _analyzed:
                print(
                    "\nWARNING: Simulation files exist but analysis has not been run.\n"
                    "Set analysis_options['run_simulation'] = True and rerun so that\n"
                    "grid posteriors are computed before visualizing belief updates.\n"
                    "Note: if you have already run the full analysis pipeline with a\n"
                    "different file_paths config, those player_fits folders can be used\n"
                    "as a fallback by pointing demo_file_paths['player_fits'] to them."
                )
            else:
                general_settings['update_method'] = 'sim_pred'
                n_dyads_to_visualize = min(9 if light_mode else 45, len(basic_files))
                sample_files = random.sample(basic_files, n_dyads_to_visualize)
                visualized = 0
                for fname in sample_files:
                    fpath = os.path.join(fits_dir, fname)
                    try:
                        with open(fpath, encoding='utf-8') as fh:
                            dyad_dict = json.load(fh)
                        dyad_games = next(iter(dyad_dict.values()))
                        visualize_bayesian_updates_3d(
                            dyad_games_or_key=dyad_games,
                            player_uuid=2,
                            figure_layout=figure_layout,
                            file_paths=demo_file_paths,
                            general_settings=general_settings,
                            fix_z_axis=True,
                        )
                        visualized += 1
                    except (ValueError, KeyError):
                        pass
                print(f"Visualized {visualized} / {len(sample_files)} sampled dyads.")

    "=========================================================================================="
    "=========== Section 8: Alternative Model Competition + Typological Comparison ============"
    "=========================================================================================="
    """
    Manuscript context: Sections 3.4–3.5 (UBM vs non-Bayesian and discrete-type alternatives).
    Functions exercised: alternative_model_contest, typological_model_comparison_fit_individually.
    Requires: raw experiment data (Experiments 1 and 2) in raw_data/.
    """

    if analysis_options['run_model_comparison']:
        print("\n" + _section_header("SECTION 8: Alternative model contest + typological model comparison"))

        "Section 8 uses Experiment 2 data. Look in processed/ first, then raw_data/."
        "Experiment 1 data is not needed by alternative_model_contest or typological fitting."
        exper2_filename = file_paths["file_names"]["raw_data_exper2"]
        exper2_path = next(
            (d / exper2_filename for d in (ROOT / "processed", ROOT / "raw_data")
             if (d / exper2_filename).exists()),
            None
        )
        if exper2_path is None:
            print(
                "\n" + "!" * 70 + "\n"
                f"CRITICAL WARNING: Experiment 2 data file not found:\n"
                f"  {exper2_filename}\n"
                f"Searched in:\n"
                f"  {ROOT / 'processed'}\n"
                f"  {ROOT / 'raw_data'}\n"
                "Section 8 cannot run without this file.\n"
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
                figure_layout=figure_layout,
                check_for_n_players=2 if light_mode else 'all',
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
    "======= Section 9: Information Criterion Utility Function Comparison (505 Models) ========"
    "=========================================================================================="
    """
    Manuscript context: Section 4 (near-comprehensive IC comparison across 505 utility forms).
    Functions exercised: gnrl.identify_redundant_utility_functions, gnrl.equation_to_settings,
    information_criterion_analysis, plot_ic_scores_delta_bic, plot_ic_robustness_analysis,
    utility_setting_contribution_analysis, extract_rankings_of_canonical_utility_functions.
    Requires: raw experiment 3 data. WARNING: full mode (light_mode=False) takes weeks.
    """

    if analysis_options['run_ic_analysis']:
        mode_label = "5 representative forms (light)" if light_mode else "all 505 forms (FULL — may take weeks)"
        print("\n" + _section_header(f"SECTION 9: IC utility function comparison [{mode_label}]"))

        exper3_pairs_path = (
            demo_real_file_paths["processed"] / file_paths["file_names"]["player_pairs_exper3"]
        )
        if not exper3_pairs_path.exists():
            print(
                "\n" + "!" * 70 + "\n"
                "CRITICAL WARNING: Processed experiment 3 histories not found at:\n"
                f"  {pretty_path(exper3_pairs_path)}\n"
                "Section 9 cannot run without this file.\n"
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

            """
            In light mode, select one representative form for each k level from k=1 to k=9.
            In full mode, pass None so information_criterion_analysis generates all 505.
            """
            if light_mode:
                all_varieties_by_k = gnrl.generate_utility_settings(
                    utility_settings=utility_settings, sort_by_k=True
                )
                ic_demo_varieties = []
                k_counts          = {}
                n_per_k           = 3
                for settings_variety in all_varieties_by_k:
                    k = len(parameter_keys_for_utility_settings(utility_settings=settings_variety))
                    if k_counts.get(k, 0) < n_per_k:
                        ic_demo_varieties.append(settings_variety)
                        k_counts[k] = k_counts.get(k, 0) + 1

                "Ensure all canonical forms are always included so extract_rankings_of_ can find them."
                existing_sigs = {tuple(sorted(s.items())) for s in ic_demo_varieties}
                canonical_added = 0
                for spec in CANONICAL_UTILITY_SPECS.values():
                    if tuple(sorted(spec.items())) not in existing_sigs:
                        ic_demo_varieties.append(spec)
                        existing_sigs.add(tuple(sorted(spec.items())))
                        canonical_added += 1

                k_range = f"k={min(k_counts)}..{max(k_counts)}" if k_counts else "none"
                print(f"  Running IC analysis on {len(ic_demo_varieties)} utility forms "
                      f"({n_per_k} per k, {k_range}) + {canonical_added} canonical forms added)")
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
                figure_layout=figure_layout,
                file_paths=demo_real_file_paths,
                general_settings=ic_general_settings,
                include_dropdown=False,
            )

            plot_ic_robustness_analysis(
                general_settings=ic_general_settings,
                file_paths=demo_real_file_paths,
                figure_layout=figure_layout,
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
                canonical_specs=CANONICAL_UTILITY_SPECS,
            )

    "=========================================================================================="
    "====== Section 10: Individual Architecture Compression Curve (M Utility Types) ==========="
    "=========================================================================================="
    """
    Manuscript context: Section 4.5 (how many structurally distinct utility types describe the population?).
    Functions exercised: extract_participant_model_combined_fits, compute_architecture_compression_curve,
    plot_architecture_compression_curve.
    Requires: IC results in bic_aic/ (participant_model_combined_fits.csv is generated on first run).
    NOTE: light_mode uses only 5 candidate models and M_max=2; for real results run with light_mode=False
    and real IC data.
    """

    if analysis_options.get('run_individual_architecture'):
        print("\n" + _section_header("SECTION 10: Individual architecture compression curve"))

        ic_json_path = ROOT / "bic_aic" / "All_Utility_Forms_IC_Analysis_Experiment3.json"
        if not ic_json_path.exists():
            print(
                "\n" + "!" * 70 + "\n"
                "CRITICAL WARNING: IC JSON not found at:\n"
                f"  {pretty_path(ic_json_path)}\n"
                "Section 10 cannot run without this file. Generate it by running\n"
                "information_criterion_analysis() (Section 9), or obtain it from the authors.\n"
                + "!" * 70 + "\n"
            )
        else:
            "File paths for IC-dependent analyses: reads real bic_aic/, writes to demo_files/."
            demo_ic_file_paths = {**demo_real_file_paths, "bic_aic": ROOT / "bic_aic"}

            if light_mode:
                ia_settings = {
                    **general_settings.get('individual_architecture_settings', {}),
                    'population_top_n_models': 5,
                    'participant_top_r_models': 3,
                    'M_max': 2,
                    'exhaustive_M_max': 2,
                    'n_workers': 1,
                }
                arch_general_settings = {**general_settings, 'individual_architecture_settings': ia_settings, 'experiment_num': 3}
            else:
                arch_general_settings = {**general_settings, 'experiment_num': 3}

            extract_participant_model_combined_fits(
                general_settings=arch_general_settings,
                file_paths=demo_ic_file_paths,
                create_new_file=False,
            )
            compute_architecture_compression_curve(
                general_settings=arch_general_settings,
                file_paths=demo_ic_file_paths,
                create_new_file=True,
            )
            plot_architecture_compression_curve(
                general_settings=arch_general_settings,
                file_paths=demo_ic_file_paths,
                figure_layout=figure_layout,
            )

    "=========================================================================================="
    "=========== Section 11: Model Recovery Simulation (IC Pipeline Data Adequacy) ============"
    "=========================================================================================="
    """
    Manuscript context: Section 4.6 (how many games and participants are needed for the IC pipeline
    to reliably recover the generating model?).
    Functions exercised: compute_model_recovery_simulation, plot_model_recovery_simulation.
    Requires: IC JSON in bic_aic/ (supplies generating-model parameter distributions).
    NOTE: light_mode uses 2 agents and 2 game counts; full run uses general_settings defaults.
    """

    if analysis_options.get('run_model_recovery'):
        print("\n" + _section_header("SECTION 11: Model recovery simulation"))

        ic_json_path = ROOT / "bic_aic" / "All_Utility_Forms_IC_Analysis_Experiment3.json"
        if not ic_json_path.exists():
            print(
                "\n" + "!" * 70 + "\n"
                "CRITICAL WARNING: IC JSON not found at:\n"
                f"  {pretty_path(ic_json_path)}\n"
                "Section 11 cannot run without this file. Generate it by running\n"
                "information_criterion_analysis() (Section 9), or obtain it from the authors.\n"
                + "!" * 70 + "\n"
            )
        else:
            demo_ic_file_paths = {**demo_real_file_paths, "bic_aic": ROOT / "bic_aic"}

            if light_mode:
                mr_settings = {
                    **general_settings.get('model_recovery_settings', {}),
                    'n_agents_grid':     [2],
                    'n_games_grid':      [5, 10],
                    'n_candidate_models': 5,
                }
                recovery_general_settings = {**general_settings, 'model_recovery_settings': mr_settings}
            else:
                recovery_general_settings = general_settings

            compute_model_recovery_simulation(
                general_settings=recovery_general_settings,
                file_paths=demo_ic_file_paths,
                param_bds=param_bds,
                utility_settings=utility_settings,
                create_new_file=True,
            )
            plot_model_recovery_simulation(
                general_settings=recovery_general_settings,
                file_paths=demo_ic_file_paths,
                figure_layout=figure_layout,
            )

    "=========================================================================================="
    "============ Section 12: Population Parameter Distributions and Correlations ============="
    "=========================================================================================="
    """
    Manuscript context: Section 5 (parameter estimates, cross-role correlations, ratios).
    Functions exercised: run_analysis_bayes (or mle), population_parameter_distribution_histograms,
    subpopulation_stats_and_param_ratio_histograms, param_correlation_matrix_report.
    Requires: raw experiment data. The fitting step is computationally expensive.
    """

    if analysis_options['run_parameter_distribution'] and not light_mode:
        print("\n" + _section_header("SECTION 12: Population parameter distribution results"))

        experiment_num_for_distributions = general_settings.get('experiment_num', 3)
        distribution_settings = {**general_settings, 'experiment_num': experiment_num_for_distributions}
        run_analysis_function = run_analysis_bayes if general_settings['analysis_mode'] == 'bayesian' else run_analysis_mle

        exper_pairs_key  = f"player_pairs_exper{experiment_num_for_distributions}"
        exper_pairs_path = demo_real_file_paths["processed"] / file_paths["file_names"][exper_pairs_key]
        if not exper_pairs_path.exists():
            print(
                "\n" + "!" * 70 + "\n"
                f"CRITICAL WARNING: Processed experiment {experiment_num_for_distributions} histories not found at:\n"
                f"  {pretty_path(exper_pairs_path)}\n"
                "Section 10 cannot run without this file.\n"
                + "!" * 70 + "\n"
            )
        else:
            print(f"Active utility equation: {build_utility_equation(utility_settings=utility_settings)}\n")
            with open(exper_pairs_path, "r") as _f:
                pairs_data = json.load(_f)
            run_analysis_function(
                histories_data=pairs_data,
                file_paths=demo_real_file_paths,
                param_info=param_info,
                utility_settings=utility_settings,
                general_settings=distribution_settings,
            )

            for player_role in ('chooser', 'predictor'):
                population_parameter_distribution_histograms(
                    general_settings=distribution_settings,
                    file_paths=demo_real_file_paths,
                    figure_layout=figure_layout,
                    player_role=player_role,
                    use_initial_params=True,
                    create_new_file=True,
                )
                for ratio_mode in ('absolute', 'skip_negative'):
                    subpopulation_stats_and_param_ratio_histograms(
                        general_settings=distribution_settings,
                        file_paths=demo_real_file_paths,
                        figure_layout=figure_layout,
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
    "================ Section 13: Inequality Aversion Bot Competition Heatmaps ================"
    "=========================================================================================="
    """
    Manuscript context: Section 5.4 (envy vs guilt asymmetry competition).
    Functions exercised: visualize_inequality_aversion_bot_competition.
    Requires: nothing (purely parametric — no participant data).
    """

    if analysis_options['run_inequality_aversion']:
        print("\n" + _section_header("SECTION 13: Inequality aversion bot competition heatmaps"))

        visualize_inequality_aversion_bot_competition(
            figure_layout=figure_layout,
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
