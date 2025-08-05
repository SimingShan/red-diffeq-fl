import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import flwr as fl
from flwr.common import Context, Metrics, NDArrays, Scalar, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from scripts.diffusion_models.diffusion_model import *
from .flwr_utils import *
from .regularizations import *

class FwiClient(fl.client.NumPyClient):
    def __init__(self, cid: str, device: torch.device,
                 fwi_forward, data_trans, ssim_loss,
                 local_data: torch.Tensor,
                 num_total_clients: int,
                 diffusion_state_dict: Optional[dict] = None,
                 diffusion_model_structure_args: Optional[dict] = None,
                 config = None):

        self.cid = int(cid)
        self.device = device
        self.fwi_forward = fwi_forward
        self.data_trans = data_trans
        self.ssim_loss = ssim_loss
        self.local_data = local_data.to(self.device)
        self.diffusion_model = None
        self.num_total_clients = num_total_clients
        self.scenario_flag = config.experiment.scenario_flag
        if diffusion_state_dict is not None and diffusion_model_structure_args is not None:
            unet_model = Unet(
                dim=diffusion_model_structure_args.get('dim', 64),
                dim_mults=diffusion_model_structure_args.get('dim_mults', (1, 2, 4, 8)),
                flash_attn=diffusion_model_structure_args.get('flash_attn', False),
                channels=diffusion_model_structure_args.get('channels', 1)
            )
            self.diffusion_model = GaussianDiffusion(
                unet_model,
                image_size=diffusion_model_structure_args.get('image_size', 72),
                timesteps=diffusion_model_structure_args.get('timesteps', 1000),
                sampling_timesteps=diffusion_model_structure_args.get('sampling_timesteps', 250),
                objective=diffusion_model_structure_args.get('objective', 'pred_noise')
            ).to(self.device)
            self.diffusion_model.load_state_dict(diffusion_state_dict)
            self.diffusion_model.eval()

        if self.diffusion_model:
            print(f"Client {self.cid}: FWI with Diffusion regularization, using local data partition.")
        else:
            print(f"Client {self.cid}: No Diffusion model provided/recreated.")

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return []

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        local_model = ndarrays_to_tensor(parameters, self.device).requires_grad_(True)
        
        local_epochs: int = int(config["local_epochs"])
        local_lr: float = float(config["local_lr"])
        total_rounds: int = int(config["total_rounds"])
        server_round: int = int(config["server_round"])
        regularization: str = str(config["regularization"])

        global_step = (server_round - 1) * local_epochs
        optimizer_local = torch.optim.Adam([local_model], lr=local_lr)
        scheduler_local = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_local,
            T_max=local_epochs * total_rounds,
            eta_min=0.0
        )
    
        for _ in range(global_step):
            scheduler_local.step() # fast forward learning rate
        
        for epoch in range(local_epochs):
            optimizer_local.zero_grad()
            model_input = local_model[:, :, 1:-1, 1:-1]
            
            predicted_seismic = self.fwi_forward(model_input, client_idx=self.cid, 
                                                 scenario=self.scenario_flag,
                                                 num_clients=self.num_total_clients)

            seismic_loss = l1_loss_fn(self.local_data.float(), predicted_seismic.float())
    
            diffusion_loss = torch.tensor(0.0, device=self.device)
            if regularization == "Diffusion":
                batch_size = self.local_data.shape[0]
                time = torch.randint(0, self.diffusion_model.num_timesteps, (1,)).item()
                time_cond = torch.full((batch_size,), time, device=self.device, dtype=torch.long)
                time_tensor = torch.full((batch_size,), time, device=self.device, dtype=torch.long)
                sigma_x0 = 0.0001
                noise_x0 = torch.randn_like(local_model)
                x0_pred = local_model + sigma_x0 * noise_x0
                noise = torch.randn_like(local_model)
                x_t = self.diffusion_model.q_sample(x0_pred, t=time_tensor, noise=noise)
                self_cond = x0_pred if self.diffusion_model.self_condition else None
                model_predictions = self.diffusion_model.model_predictions(
                    x_t, time_cond, self_cond, clip_x_start=True, rederive_pred_noise=True
                )
                pred_noise, x_start = model_predictions.pred_noise, model_predictions.pred_x_start
                et = pred_noise.detach()
                diffusion_loss = torch.mul((et - noise).detach(), local_model).mean()
                total_loss = seismic_loss + 0.75 * diffusion_loss # we fix lambda = 0.75
            elif regularization == "Total_Variation":
                reg_loss = total_variation_loss(model_input)
                total_loss = seismic_loss + 0.1 * reg_loss
            elif regularization == "Tiknov":
                reg_loss = tikhonov_loss(model_input)
                total_loss = seismic_loss + 0.1 * reg_loss
            elif regularization == None:
                total_loss = seismic_loss
                
            total_loss.backward()
            optimizer_local.step()
            scheduler_local.step() 
            local_model.data.clamp_(-1, 1)
    
        updated_ndarrays = tensor_to_ndarrays(local_model)
        num_examples = self.local_data.numel()
        
        return updated_ndarrays, num_examples, {
            "seismic_loss": float(seismic_loss.item()),
            "diffusion_loss": float(diffusion_loss.item()),
            "total_loss": float(total_loss.item())
        }

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        return 0.0, 0, {}
    
def client_fn_factory(partitions: List[torch.Tensor], 
                      config, device, fwi_forward,
                      data_trans, ssim_loss,
                      diffusion_state_dict: Optional[dict] = None,
                      diffusion_args: Optional[dict] = None
                      ):

    def client_fn(context: Context) -> fl.client.Client:

        client_id = int(context.node_config["partition-id"])
        local_data_for_client = partitions[client_id]

        fwi_client_instance = FwiClient(
            cid=str(client_id),
            device=device,
            fwi_forward=fwi_forward,
            data_trans=data_trans,
            ssim_loss=ssim_loss,
            local_data=local_data_for_client,
            num_total_clients=config.experiment.num_clients,
            diffusion_state_dict=diffusion_state_dict,
            diffusion_model_structure_args=diffusion_args,
            config=config
        )
        
        return fwi_client_instance.to_client()

    return client_fn

def fit_config_fn(server_round, config=None):
    return {
        "server_round": server_round, "local_epochs": config.federated.local_epochs,
        "local_lr": config.federated.local_lr, 
        "total_rounds": config.federated.num_rounds, 
        "regularization": config.experiment.regularization
    }