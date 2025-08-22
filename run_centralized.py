import os
import sys
from omegaconf import OmegaConf
from configs.config_utils_centralized import AppConfig, load_config
import numpy as np
from scripts.flwr.centralized_baseline import *
from scripts.data_utils import data_trans, pytorch_ssim
from scripts.diffusion_models.diffusion_model import *
from scripts.pde_solvers.pde_solver import *
import pickle
from datetime import datetime
import glob

def run_centralized(config_path: str, process_id: int, run_name = 'Centralized_Baseline'):
    assert process_id is not None, "process_id is required"
    assert process_id in [1,2], "process_id must be 1 (CF, CV) or 2 (FF, FV)"
    assert config_path is not None, "config_path is required"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    config = OmegaConf.load(config_path)
    gt_seismic_data_path = config["path"]["gt_seismic_data_path"]
    velocity_data_path = config["path"]["velocity_data_path"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir_name = f"{run_name}_{config.experiment.regularization}_{config.experiment.reg_lambda}_{config.experiment.scenario_flag}"
    # Check for existing directories
    existing_dirs = glob.glob(os.path.join(config.path.output_path, f"{base_dir_name}_*"))
    if existing_dirs:
        # Use the most recent existing directory
        main_output_dir = max(existing_dirs, key=os.path.getctime)
        print(f"Using existing directory: {main_output_dir}")
    else:
        # Create new directory
        main_output_dir = os.path.join(config.path.output_path, f"{base_dir_name}_{timestamp}")
        os.makedirs(main_output_dir, exist_ok=True)
        print(f"Created new directory: {main_output_dir}")

    diffusion_args = {
                'dim': config.diffusion.dim, 'dim_mults': config.diffusion.dim_mults, 
                'flash_attn': config.diffusion.flash_attn, 'channels': config.diffusion.channels,
                'image_size': config.diffusion.image_size, 'timesteps': config.diffusion.timesteps, 
                'sampling_timesteps': config.diffusion.sampling_timesteps, 
                'objective': config.diffusion.objective
                }

    unet_model = Unet(
        dim=diffusion_args.get('dim'),
        dim_mults=diffusion_args.get('dim_mults'),
        flash_attn=diffusion_args.get('flash_attn'),
        channels=diffusion_args.get('channels')
    )

    diffusion = GaussianDiffusion(
        unet_model,
        image_size=diffusion_args.get('image_size'),
        timesteps=diffusion_args.get('timesteps'),
        sampling_timesteps=diffusion_args.get('sampling_timesteps'),
        objective=diffusion_args.get('objective')
    ).to(device)

    model_path = config.path.model_path
    diffusion.load_state_dict(torch.load(model_path, map_location=device)['model'])
    diffusion.eval()
    ssim_loss = pytorch_ssim.SSIM(window_size=11)  
    red_diffeq = RED_DiffEq(diffusion, data_trans, pytorch_ssim)

    if process_id == 1:
        families = ['CF', 'CV']
    elif process_id == 2:
        families = ['FF', 'FV']

    family_to_vm = {fam: np.load(f"{velocity_data_path}/{fam}.npy", mmap_mode="r") for fam in families}
    family_to_gt = {fam: np.load(f"{gt_seismic_data_path}/{fam}.npy", mmap_mode="r") for fam in families}

    for family in families:
        print(f"\n--- Starting Family: {family} ---")
        
        # Loop over 10 instances in the family
        for i in range(10):
            print(f"-- Running Instance: {family}{i} --")
            test_data = torch.from_numpy(family_to_gt[family][i:i+1,:,:,:]).float().to(device)
            test_vm = torch.from_numpy(family_to_vm[family][i:i+1,:,:,:]).float().to(device)
            test_init_vm = data_trans.prepare_initial_model(test_vm, initial_type="smoothed", sigma=10)
            test_init_vm = torch.nn.functional.pad(test_init_vm, (1, 1, 1, 1), mode="constant", value=0)
            ctx = {'n_grid': config.forward.n_grid, 'nt': config.forward.nt, 'dx': config.forward.dx, 
                    'nbc': config.forward.nbc, 'dt': config.forward.dt, 'f': config.forward.f,
                    'sz': config.forward.sz, 'gz': config.forward.gz, 'ng': config.forward.ng,
                    'ns': config.forward.ns}

            fwi_forward = FWIForward(ctx, device, normalize=True, 
                                    v_denorm_func=data_trans.v_denormalize, 
                                    s_norm_func=data_trans.s_normalize_none)

            result = red_diffeq.sample(mu = test_init_vm,
                                        mu_true=test_vm,
                                        y=test_data,
                                        ts=5000,
                                        lr=0.03,
                                        reg_lambda=0.75,
                                        fwi_forward=fwi_forward,
                                        scenario=config.experiment.scenario_flag
                                        )  
            result_filename = os.path.join(main_output_dir, f"{family}_{i}_result.pkl")
            with open(result_filename, 'wb') as f:
                pickle.dump(result, f)
            print(f"Result saved to: {result_filename}")
        print(f"--- Finished Family: {family} ---")
    print(f"\n--- EXPERIMENT COMPLETE ---")
    print(f"Process {process_id} completed: {len(families)} families, {len(families) * 10} instances")
    print(f"Individual results saved in: {main_output_dir}")


