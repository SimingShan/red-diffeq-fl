import torch
from flwr.common import NDArrays
from typing import List, Tuple
import numpy as np
import pickle
from flwr.common import Metrics
from flwr.common import Context
import flwr as fl
from typing import Optional

def ndarrays_to_tensor(ndarrays: NDArrays, device: torch.device) -> torch.Tensor:
    """Convert Flower NDArrays (List[np.ndarray]) to a PyTorch Tensor."""
    if not ndarrays:
         raise ValueError("Received empty NDArrays list")
    tensor = torch.tensor(ndarrays[0], dtype=torch.float32).to(device)
    return tensor

def tensor_to_ndarrays(tensor: torch.Tensor) -> NDArrays:
    """Convert a PyTorch Tensor to Flower NDArrays (List[np.ndarray])."""
    ndarrays = [tensor.cpu().detach().numpy()]
    return ndarrays

def l1_loss_fn(x, y):
    return torch.mean(torch.abs(x - y))

def get_fit_config_fn(config):
    """Factory function to create the on_fit_config_fn."""
    def fit_config_fn(server_round: int):
        return {
            "server_round": server_round,
            "local_epochs": config.federated.local_epochs,
            "local_lr": config.federated.local_lr,
            "total_rounds": config.federated.num_rounds,
            "regularization": config.experiment.regularization,
            "reg_lambda": config.experiment.reg_lambda
        }
    return fit_config_fn


def fit_metrics_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate client metrics."""
    aggregated = {}
    
    # Store deserialized detailed data separately
    detailed_data_store = []

    for client_id, client_metrics in metrics:
        for key, value in client_metrics.items():
            if key == "detailed_data":
                # Deserialize the data and store it
                deserialized = pickle.loads(value)
                detailed_data_store.append({"client_id": client_id, "data": deserialized})
                continue
            
            if key not in aggregated:
                aggregated[key] = []
            aggregated[key].append(value)
    
    # Calculate means and stds for simple numeric metrics
    final_metrics = {}
    for key, values in aggregated.items():
        if isinstance(values[0], (int, float)):
            final_metrics[f"{key}_mean"] = sum(values) / len(values)
            final_metrics[f"{key}_std"] = np.std(values)
        else:
            final_metrics[key] = values
            
    if detailed_data_store:
        final_metrics["detailed_client_data"] = detailed_data_store
    
    return final_metrics
