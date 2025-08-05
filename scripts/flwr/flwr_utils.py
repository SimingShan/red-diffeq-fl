import torch
import numpy
from flwr.common import Metrics, NDArrays, Scalar, Parameters, ndarrays_to_parameters, parameters_to_ndarrays

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

def l2_loss_fn(x, y):
    return torch.mean((x-y)**2)