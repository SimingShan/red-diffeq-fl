import argparse
from run import run_full_experiment

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
        required=True,
        choices=[1, 2],
        help="Process ID: 1 for families ['CF', 'CV'], 2 for families ['FF', 'FV']"
    )
    
    # Add the run_name argument
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        choices=['main', 'tuning'],
        help="Run name: 'main' for main experiments, 'tuning' for hyperparameter tuning"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the main experiment function with all required arguments
    run_full_experiment(
        config_path=args.config_path,
        process_id=args.process_id,
        run_name=args.run_name
    )