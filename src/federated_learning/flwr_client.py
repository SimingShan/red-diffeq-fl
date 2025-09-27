import torch
from typing import Dict, List, Tuple, Optional
import flwr as fl
from flwr.common import Context, NDArrays, Scalar
from src.diffusion_models.diffusion_model import *
from src.utils import *
from src.full_waveform_inversion import *
from src.federated_learning.flwr_utils import *
import pickle

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
        self.model_history = []
        self.epoch_metrics_history = []
        self.round_data = {} 
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
        # The initial model is passed in as parameters
        return []

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        local_model = ndarrays_to_tensor(parameters, self.device).requires_grad_(True)
        
        local_epochs: int = int(config["local_epochs"])
        local_lr: float = float(config["local_lr"])
        total_rounds: int = int(config["total_rounds"])
        server_round: int = int(config["server_round"])
        # Do not cast to string; preserve None so baseline (no regularization) works
        regularization = config["regularization"]
        reg_lambda = config["reg_lambda"]
        loss_type = 'l1'
        # Build the regularization method for this round based on server-provided config
        regularization_method = Regularization_method(regularization, self.diffusion_model)
        global_step = (server_round - 1) * local_epochs
        optimizer_local = torch.optim.Adam([local_model], lr=local_lr)
        scheduler_local = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_local,
            T_max=local_epochs * total_rounds,
            eta_min=0.0
        )
    
        for _ in range(global_step):
            scheduler_local.step() # fast forward learning rate
        epoch_metrics = []
        initial_model = ndarrays_to_tensor(parameters, self.device)
        self.model_history.append({
            "round": config["server_round"],
            "epoch": 0,
            "model": tensor_to_ndarrays(initial_model)
        })
        results_dict = ResultsDict(self.data_trans, self.ssim_loss, loss_type=loss_type, regularization_method=regularization_method, reg_lambda=reg_lambda)
        for epoch in range(local_epochs):
            optimizer_local.zero_grad()
            model_input = local_model[:, :, 1:-1, 1:-1]
            predicted_seismic = self.fwi_forward(model_input, client_idx=self.cid, 
                                                 scenario=self.scenario_flag,
                                                 num_clients=self.num_total_clients)

            # Calculate loss across the entire batch
            seismic_loss = l1_loss_fn(self.local_data.float(), predicted_seismic.float())
            # Observation and regularization losses
            loss_obs = results_dict.calcualte_seismic_loss(predicted_seismic, self.local_data, loss_type)
            raw_reg_loss = results_dict.calcualte_raw_reg_loss(local_model, reg_lambda)
            total_loss = results_dict.calcualte_total_loss(loss_obs, raw_reg_loss, reg_lambda)

            epoch_metrics.append({
                    "epoch": epoch,
                    "seismic_loss": float(seismic_loss.item()),
                    "total_loss": float(total_loss.item()),
                    "reg_loss": float(raw_reg_loss.item())}
                    )

            total_loss.backward()
            optimizer_local.step()
            scheduler_local.step() 
            local_model.data.clamp_(-1, 1)

        if config["server_round"] % 10 == 0:
            self.model_history.append({
                "round": config["server_round"],
                "epoch": epoch + 1, 
                "model": tensor_to_ndarrays(local_model)
            })

        updated_ndarrays = tensor_to_ndarrays(local_model)
        num_examples = self.local_data.numel()
        
        client_round_data = {
            "epoch_metrics": epoch_metrics,
            "model_history": self.model_history.copy(),
        }
        serialized_data = pickle.dumps(client_round_data)
        self.model_history.clear()
        
        return updated_ndarrays, num_examples, {
            "seismic_loss": float(seismic_loss.item()),
            "total_loss": float(total_loss.item()),
            "reg_loss": float(raw_reg_loss.item()),
            "detailed_data": serialized_data
        }

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        return 0.0, 0, {}
    
    def get_round_data(self, round_number: int) -> Optional[Dict]:
        """Get stored data for a specific round."""
        return self.round_data.get(round_number, None)
    
    def get_all_round_data(self) -> Dict:
        """Get all stored round data."""
        return self.round_data
    

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
