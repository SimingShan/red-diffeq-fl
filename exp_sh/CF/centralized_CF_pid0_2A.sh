#!/bin/bash
#SBATCH --job-name=centralized_CF_pid0_2A
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --chdir=/home/ss5235/scratch_pi_ll2247/ss5235/red-diffeq-fl/red-diffeq-fl

module reset
module load miniconda
conda activate /home/ss5235/project_pi_ll2247/ss5235/conda_envs/fwi

python main.py --run_name centralized --family CF --process_id 0 --config_path configs/main_exp/diff/config_2A.yml --batch_size max


