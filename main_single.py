import argparse
from run_single import run_full_experiment

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
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the main experiment function with the provided path
    run_full_experiment(config_path=args.config_path)