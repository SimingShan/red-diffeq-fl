import argparse
from run_resume import run_resume_experiment

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
    # Optional explicit resume directory; if omitted, it will be inferred from config
    parser.add_argument(
        "--resume_dir",
        type=str,
        required=False,
        default=None,
        help=(
            "Existing run directory to resume (contains intermediate_results/). If omitted, "
            "the script will infer the most recent matching run directory under config.path.output_path."
        ),
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the resume function
    run_resume_experiment(config_path=args.config_path, resume_dir=args.resume_dir)