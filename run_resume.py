# ========================
# Standard library imports
# ========================
from datetime import datetime
import os
import re
import pickle
from typing import Dict, List, Optional, Tuple, Set

# =========================
# Third-party library imports
# =========================
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from omegaconf import OmegaConf

import flwr as fl
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg, FedAvgM, FedProx

# =========================
# Local/project imports
# =========================
from configs.config_utils import load_config

from scripts.data_utils.data_trans import (
    s_denormalize,
    s_normalize,
    s_normalize_none,
    v_denormalize,
    v_normalize,
)
from scripts.data_utils import data_trans, pytorch_ssim

from scripts.diffusion_models.diffusion_model import Unet, GaussianDiffusion
from scripts.flwr.flwr_client import client_fn_factory
from scripts.flwr.flwr_evaluation import get_evaluate_fn
from scripts.flwr.flwr_utils import tensor_to_ndarrays, ndarrays_to_tensor
from scripts.pde_solvers.solver import FWIForward


def average_histories(histories: list):
    if not histories:
        return {}

    # Average losses
    avg_loss = np.mean([loss for _, loss in histories[0].losses_centralized], axis=0).tolist()

    # Average metrics
    avg_metrics = {}
    metric_keys = histories[0].metrics_centralized.keys()
    for key in metric_keys:
        stacked_metrics = np.array([h.metrics_centralized[key] for h in histories])
        avg_metric = np.mean([value for _, value in stacked_metrics[0]], axis=0).tolist()
        avg_metrics[key] = avg_metric

    return {"losses_centralized": avg_loss, "metrics_centralized": avg_metrics}


def _parse_done_pairs(intermediate_path: str) -> Set[Tuple[str, int]]:
    done: Set[Tuple[str, int]] = set()
    if not os.path.isdir(intermediate_path):
        return done
    pattern = re.compile(r"^(?P<family>[A-Za-z]+)_(?P<idx>\d+)_result\.pkl$")
    for fname in os.listdir(intermediate_path):
        if not fname.endswith("_result.pkl"):
            continue
        m = pattern.match(fname)
        if not m:
            continue
        family = m.group("family")
        idx = int(m.group("idx"))
        # Consider a run as done only if the pickle can be loaded successfully
        fpath = os.path.join(intermediate_path, fname)
        try:
            with open(fpath, "rb") as f:
                _ = pickle.load(f)
        except Exception:
            # Skip corrupted or partially written files; they will be recomputed
            continue
        done.add((family, idx))
    return done


def _load_all_pickled_runs(intermediate_path: str):
    runs = []
    for fname in sorted(os.listdir(intermediate_path)):
        if not fname.endswith("_result.pkl"):
            continue
        fpath = os.path.join(intermediate_path, fname)
        try:
            with open(fpath, "rb") as f:
                run = pickle.load(f)
                runs.append(run)
        except Exception:
            # Skip corrupted entries
            continue
    return runs


def _infer_resume_dir_from_config(config) -> str:
    """
    Infer the resume directory from the config by scanning the configured output path
    for a directory matching: "{strategy}_{regularization}_{scenario}_YYYYMMDD_HHMMSS".
    If multiple matches exist, prefer the one with the latest timestamp; if timestamps
    are not parseable, fall back to directory modification time.
    """
    base_dir = os.path.abspath(config.path.output_path)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Output path not found: {base_dir}")

    prefix = f"{config.experiment.strategy}_{config.experiment.regularization}_{config.experiment.scenario_flag}_"

    candidates = []
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if not os.path.isdir(full_path):
            continue
        if not name.startswith(prefix):
            continue
        # Must contain intermediate_results to be a valid run dir
        if not os.path.isdir(os.path.join(full_path, "intermediate_results")):
            continue
        # Try to parse the timestamp suffix
        ts_part = name[len(prefix):]
        dt_obj = None
        try:
            dt_obj = datetime.strptime(ts_part, "%Y%m%d_%H%M%S")
        except Exception:
            dt_obj = None
        candidates.append((dt_obj, os.path.getmtime(full_path), full_path))

    if not candidates:
        raise FileNotFoundError(
            f"No matching run directories under {base_dir} with prefix '{prefix}' and an 'intermediate_results' subfolder."
        )

    # Prefer parsable timestamps, otherwise fall back to mtime
    with_ts = [c for c in candidates if c[0] is not None]
    if with_ts:
        with_ts.sort(key=lambda x: x[0], reverse=True)
        return with_ts[0][2]

    # Fallback: sort by modification time
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][2]


def run_resume_experiment(config_path: str, resume_dir: str):
    """
    Resume an interrupted experiment by skipping already completed (family, instance)
    runs found under resume_dir/intermediate_results and executing only the missing ones.
    After completion, recompute and overwrite the aggregated results file in resume_dir.
    """
    config = OmegaConf.load(config_path)
    # Match performance-related settings from the fast path
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    mp.set_start_method('spawn', force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Allow resume_dir to be inferred from config if not provided
    if resume_dir is None or str(resume_dir).strip() == "":
        main_output_dir = _infer_resume_dir_from_config(config)
    else:
        main_output_dir = os.path.abspath(resume_dir)
        if not os.path.isdir(main_output_dir):
            raise FileNotFoundError(f"Resume directory not found: {main_output_dir}")

    print("--- RESUMING EXPERIMENT RUN ---")
    print(f"Resuming into: {main_output_dir}")

    intermediate_path = os.path.join(main_output_dir, "intermediate_results")
    os.makedirs(intermediate_path, exist_ok=True)

    # Shared components
    ctx = {'n_grid': config.forward.n_grid, 'nt': config.forward.nt, 'dx': config.forward.dx,
           'nbc': config.forward.nbc, 'dt': config.forward.dt, 'f': config.forward.f,
           'sz': config.forward.sz, 'gz': config.forward.gz, 'ng': config.forward.ng}
    fwi_forward = FWIForward(ctx, device, normalize=True, v_denorm_func=v_denormalize, s_norm_func=s_normalize_none)

    server_diffusion_model = None
    diffusion_state_dict = None
    diffusion_args = None
    if config.experiment.regularization == "Diffusion":
        diffusion_args = {
            'dim': config.diffusion.dim,
            'dim_mults': config.diffusion.dim_mults,
            'flash_attn': config.diffusion.flash_attn,
            'channels': config.diffusion.channels,
            'image_size': config.diffusion.image_size,
            'timesteps': config.diffusion.timesteps,
            'sampling_timesteps': config.diffusion.sampling_timesteps,
            'objective': config.diffusion.objective,
        }

        unet_model = Unet(
            dim=diffusion_args.get('dim'),
            dim_mults=diffusion_args.get('dim_mults'),
            flash_attn=diffusion_args.get('flash_attn'),
            channels=diffusion_args.get('channels'),
        )

        diffusion = GaussianDiffusion(
            unet_model,
            image_size=diffusion_args.get('image_size'),
            timesteps=diffusion_args.get('timesteps'),
            sampling_timesteps=diffusion_args.get('sampling_timesteps'),
            objective=diffusion_args.get('objective'),
        ).to(device)

        checkpoint = torch.load(config.path.model_path, map_location=device, weights_only=True)
        state_dict = checkpoint.get('model', checkpoint)
        diffusion.load_state_dict(state_dict)
        diffusion.eval()
        server_diffusion_model = diffusion
        diffusion_state_dict = diffusion.state_dict()

    ssim_loss = pytorch_ssim.SSIM(window_size=11)

    # Factory for fit config
    def fit_config_fn(server_round: int):
        return {
            "server_round": server_round,
            "local_epochs": config.federated.local_epochs,
            "local_lr": config.federated.local_lr,
            "total_rounds": config.federated.num_rounds,
            "regularization": config.experiment.regularization,
        }

    families = ['CF', 'CV', 'FF', 'FV']
    done_pairs = _parse_done_pairs(intermediate_path)
    print(f"Detected completed runs: {len(done_pairs)}")

    # Execute missing runs
    for family in families:
        print(f"\n--- Family: {family} ---")
        for i in range(10):
            if (family, i) in done_pairs:
                print(f"Skipping {family}_{i} (already done)")
                continue

            print(f"-- Running Instance: {family}{i} --")

            # Load data for this run
            vm_data = torch.tensor(np.load(f"{config.path.velocity_data_path}/{family}.npy")[i:i+1, :]).float()
            gt_seismic_data = torch.tensor(np.load(f"{config.path.gt_seismic_data_path}/{family}.npy")[i:i+1, :]).float().to(device)
            client_data_list = []
            for client_idx in range(config.experiment.num_clients):
                client_data = np.load(f"{config.path.client_seismic_data_path}/client{client_idx+1}/{family}.npy")[i:i+1, :]
                client_data_list.append(torch.tensor(client_data).float().to(device))

            # Initial model
            initial_model = data_trans.prepare_initial_model(vm_data, initial_type='smoothed', sigma=config.forward.initial_sigma)
            initial_model = F.pad(initial_model, (1, 1, 1, 1), "constant", 0)

            # Store for final model
            final_parameters_store: Dict = {}

            # Evaluation function
            evaluate_fn = get_evaluate_fn(
                model_shape=initial_model.shape,
                seismic_data=gt_seismic_data,
                mu_true=vm_data,
                fwi_forward=fwi_forward,
                data_trans=data_trans,
                ssim_loss=ssim_loss,
                device=device,
                total_rounds=config.federated.num_rounds,
                final_params_store=final_parameters_store,
                config=config,
                diffusion_model=server_diffusion_model,
            )

            # Strategy
            strategy_class = {"FedAvg": FedAvg, "FedAvgM": FedAvgM, "FedProx": FedProx}[config.experiment.strategy]
            strategy = strategy_class(
                fraction_fit=1.0,
                min_fit_clients=config.experiment.num_clients,
                min_available_clients=config.experiment.num_clients,
                evaluate_fn=evaluate_fn,
                fraction_evaluate=0.0,
                on_fit_config_fn=fit_config_fn,
                initial_parameters=ndarrays_to_parameters(tensor_to_ndarrays(initial_model.to(device))),
                server_momentum=config.experiment.server_momentum,
            )

            # Clients
            client_fn_instance = client_fn_factory(
                partitions=client_data_list,
                config=config,
                device=device,
                fwi_forward=fwi_forward,
                data_trans=data_trans,
                ssim_loss=ssim_loss,
                diffusion_args=diffusion_args,
                diffusion_state_dict=diffusion_state_dict,
            )

            # Use the same temp directory convention as the fast run
            ray_temp_dir = "/tmp/rfl"
            os.makedirs(ray_temp_dir, exist_ok=True)

            # Cap CPU threads to avoid oversubscription (match fast run)
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            torch.set_num_threads(1)

            # Run one simulation
            history = fl.simulation.start_simulation(
                client_fn=client_fn_instance,
                num_clients=config.experiment.num_clients,
                config=fl.server.ServerConfig(num_rounds=config.federated.num_rounds),
                strategy=strategy,
                client_resources={
                    "num_gpus": config.resources.num_gpus_per_client,
                    "num_cpus": config.resources.num_cpus_per_client,
                } if device.type == "cuda" else {},
                ray_init_args={"include_dashboard": False, "_temp_dir": ray_temp_dir},
            )

            # Save individual result
            saved_ndarrays = final_parameters_store["final_model"]
            final_model = ndarrays_to_tensor(saved_ndarrays, device)
            run_result = {
                "family": family,
                "instance": i,
                "final_model": final_model.cpu().detach().numpy(),
                "history": history,
            }
            intermediate_filename = os.path.join(intermediate_path, f"{family}_{i}_result.pkl")
            print(f"Saving intermediate result to: {intermediate_filename}")
            with open(intermediate_filename, 'wb') as f:
                pickle.dump(run_result, f)

    # Recompute aggregated results from all pickles present
    print("\n--- Recomputing aggregated results ---")
    all_runs = _load_all_pickled_runs(intermediate_path)

    # Organize by family
    by_family: Dict[str, List] = {}
    for rr in all_runs:
        fam = rr.get("family")
        by_family.setdefault(fam, []).append(rr["history"])

    all_results = {
        'individual_runs': all_runs,
        'family_histories': {fam: average_histories(hists) for fam, hists in by_family.items()},
        'overall_history': average_histories([rr['history'] for rr in all_runs]) if all_runs else {},
    }

    final_filename = os.path.join(main_output_dir, "aggregated_results.pkl")
    with open(final_filename, 'wb') as f:
        pickle.dump(all_results, f)

    print("--- RESUME COMPLETE ---")
    print(f"Aggregated file written to: {final_filename}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Resume Federated Waveform Inversion experiments.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the OmegaConf YAML configuration file.")
    parser.add_argument(
        "--resume_dir",
        type=str,
        required=False,
        default=None,
        help=(
            "Existing run directory to resume (contains intermediate_results/). "
            "If omitted, the script will infer the most recent matching run directory under config.path.output_path."
        ),
    )
    args = parser.parse_args()
    run_resume_experiment(config_path=args.config_path, resume_dir=args.resume_dir)



