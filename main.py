import argparse
import os
from src.run_federated import run_full_experiment
from src.run_centralized import run_centralized
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Federated Waveform Inversion experiments.")
    
    # Add the config_path argument
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the OmegaConf YAML configuration file."
    )
    
    # Add the process_id argument
    parser.add_argument(
        "--process_id",
        type=int,
        required=False,
        choices=[1, 2],
        help="Process ID: 1 for families ['CF', 'CV'], 2 for families ['FF', 'FV']"
    )

    # Add the process_id argument
    parser.add_argument(
        "--family",
        type=str,
        required=False,
        choices=['CF', 'CV', 'FF', 'FV'],
        help="Family: ['CF', 'CV', 'FF', 'FV']"
    )
    
    
    # Add the run_name argument
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        choices=['main', 'centralized'],
        help="Run name: 'main' for main experiments, 'centralized' for centralized experiments"
    )

    # Parse arguments
    args = parser.parse_args()
    
    target_families_env = os.environ.get("RFL_TARGET_FAMILIES")
    target_instances_env = os.environ.get("RFL_TARGET_INSTANCES")

    # Call the main experiment function with all required arguments
    if args.run_name == 'main':
        # Determine targeted families/instances for federated run
        # CLI --family overrides env RFL_TARGET_FAMILIES
        cli_target_families = [args.family] if args.family else None
        resolved_target_families = cli_target_families if cli_target_families else (target_families_env.split(',') if target_families_env else None)

        # If both family and process_id are provided, we let the runner split 12/13 automatically,
        # so do not pass target_instances unless explicitly set via env.
        resolved_target_instances = (
            [int(x) for x in target_instances_env.split(',')]
            if target_instances_env else None
        )

        run_full_experiment(
            config_path=args.config_path,
            process_id=args.process_id,
            run_name=args.run_name,
            target_families=resolved_target_families,
            target_instances=resolved_target_instances,
        )

    elif args.run_name == 'centralized':
        run_centralized(
            config_path=args.config_path,
            process_id=args.process_id,
            run_name=args.run_name,
            family=args.family,
        )