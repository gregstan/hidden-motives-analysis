from main import create_simulated_data, run_simulation_recovery_analysis, visualize_bayesian_updates_3d, \
    build_utility_equation, model_nesting_adjacency_matrices, run_child_parent_embedding_sanity_checks, \
    run_child_parent_probability_equivalence_smoketest
    
import utilities as gnrl
from config import *

analysis_options = {
    'light_mode': True,
    'run_simulation': True,
    'visualize_belief_updates': True,
    'run_model_nesting_tests': True,
}

def run_quick_demo(analysis_options: dict[str, bool]) -> None:
    """
    Runs a demo of Greg's code for Dr. Falk Lieder.

    Arguments:
        • analysis_options: dict[str: bool]; Controls which analyses are demonstrated.
            - light_mode: 
                - If True, runs a small quick version of the simulation (recommended)
                - Otherwise, runs the full version that takes about two hours running on five cores
            - run_simulation: If True, runs the parameter recovery simulation, where the optimizer 
                attempts to find the parameters assigned to artificial agents.
            - visualize_belief_updates: If True, generates Plotly html files within **INSERT PATH** 
                to visualize Bayesian updates from the simulation in 3D.
            - run_model_nesting_tests: If True, generates all valid utility functions, creates a 
                graph of model family relationships (parent, child, sibling), and verifies that 
                they produce identical choice probabilities and losses in the same games.

    Notes:
        • It is easiest to stick with the default values. 
        • 'visualize_belief_updates' assumes preexisting data from the simulation.  
        • All of these analyses will save data files on your machine, which you can delete afterwards.           
    """
    general_settings['experiment_num'] = 0
    file_paths = {
        "processed":   ROOT / "demo_files" / "processed",
        "player_fits": ROOT / "demo_files" / "player_fits",
        "param_data":  ROOT / "demo_files" / "param_data", 
        "visuals":     ROOT / "demo_files" / "visuals", 
        "file_names": {
            "player_pairs_exper0": "Social_Preference_Prediction_Pairs_Exper0.json",
            "params_data_exper0_bayes": "Social_Preference_Prediction_Parameters_Exper0_Bayes.json",
        }     
    }
    
    "Creating folders"
    demo_root = ROOT / "demo_files"
    required_dirs = [
        demo_root / "processed",
        demo_root / "param_data",
        demo_root / "visuals" / "bayesian_update_3d",
        demo_root / "player_fits" / "experiment_0",
        demo_root / "player_fits" / "loss_reports" / "experiment_0",
        demo_root / "player_fits" / "simulation_results",
    ]
    for dir in required_dirs:
        dir.mkdir(parents=True, exist_ok=True)

    if analysis_options['run_simulation']:
        print("Running 'run_simulation' branch:")
        if analysis_options['light_mode']:
            params_chooser_range =   {'Vᵢᵢ': (1, 1, 1), 'Vᵢⱼ': (-1, 1, 7), 'std': (1.0, 1.0, 1), 'τ': (0.5, 0.5, 1)}
            params_predictor_range = {'Vᵢᵢ': (1, 1, 1), 'Vᵢⱼ': (-1, 1, 7), 'std': (1.0, 1.0, 1), 'τ': (0.5, 0.5, 1)}
        else:
            params_chooser_range =   {'Vᵢᵢ': (1, 1, 1), 'Vᵢⱼ': (-1, 1, 5), 'std': (1.0, 1.0, 1), 'τ': (0.5, 3, 3)}
            params_predictor_range = {'Vᵢᵢ': (1, 1, 1), 'Vᵢⱼ': (-1, 1, 7), 'std': (0.5, 1.5, 3), 'τ': (0.5, 3, 3)}

        create_simulated_data(n_games=24, randomize_parameters=False, param_bds=param_bds, file_paths=file_paths, run_analysis=True,
                                   params_chooser_range=params_chooser_range, params_predictor_range=params_predictor_range, 
                                   utility_settings=utility_settings, dynamic_predictor=True)

        run_simulation_recovery_analysis(
            general_settings=general_settings, file_paths=file_paths,
            fig_lay=fig_lay, export_fig=True, create_new_file=True, produce_figures=True, 
            correlation_csv_name="correlation_results.csv", include_dropdown=False, 
            use_dynamic_predictor=True
        )

    if analysis_options['visualize_belief_updates']:
        print("Running 'visualize_belief_updates' branch:")

        pairs_path = file_paths["processed"] / file_paths["file_names"]["player_pairs_exper0"]
        if not pairs_path.exists():
            print(
                "No simulation data found for visualization.\n"
                f"Expected: {pairs_path}\n"
                "Run with analysis_options['run_simulation']=True first."
            )
            return

        general_settings['update_method'] = 'sim_pred'
        total_dyads = 49 if analysis_options['light_mode'] else 945
        n_samples = 9 if analysis_options['light_mode'] else 45
        dyad_indices = sorted(random.sample(range(total_dyads), n_samples))
        for dyad_idx in dyad_indices:
            visualize_bayesian_updates_3d(
                dyad_games_or_key=dyad_idx,
                player_uuid=2,
                fig_lay=fig_lay,
                file_paths=file_paths,
                general_settings=general_settings,
                fix_z_axis=True, 
            )

    if analysis_options['run_model_nesting_tests']:
        print("Running 'run_model_nesting_tests' branch:")
        
        model_nesting_adjacency_matrices(general_settings=general_settings, utility_settings=utility_settings, 
                                        file_paths=file_paths, create_new_file=True, equation_form=True, print_=True)
        
        gnrl.summarize_nesting_relationship_counts(general_settings=general_settings, utility_settings=utility_settings, file_paths=file_paths, 
                                            model_nesting_adjacency_matrices=model_nesting_adjacency_matrices, create_new_file=True, print_=True)
        
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


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    run_quick_demo(analysis_options=analysis_options)
