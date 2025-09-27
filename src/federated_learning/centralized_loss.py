import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from src.regularization_methods.benchmarks import total_variation_loss, tikhonov_loss
from src.diffusion_models.diffusion_model import extract

# Define the loss functions 
l1 = nn.L1Loss()
l2 = nn.MSELoss()

def scenario_aware_seismic_loss(y, predicted_seismic, scenario):
    """
    Compute L1 loss only on source–receiver blocks that exist in the federated setting.
    The mask is built as a UNION of per-client coverage.
    Shapes: (B, num_sources, T, num_receivers)
    """
    if y.shape != predicted_seismic.shape:
        raise ValueError(f"Shape mismatch: y {y.shape} vs pred {predicted_seismic.shape}")

    device = y.device
    mask = torch.zeros_like(y, dtype=y.dtype, device=device)

    # Build union-of-coverage mask per scenario
    if scenario == '2A':
        mask[:, :3,  :, :35] = 1  # client 0
        mask[:, 3:5, :, 35:] = 1  # client 1
    elif scenario == '2B':
        mask[:, :3,  :, :35] = 1  # client 0
        mask[:, 2:5, :, 35:] = 1  # client 1 (overlaps source 2)
    elif scenario == '2C':
        mask[:, :3,  :, :35] = 1  # client 0
        mask[:, 3:6, :, 35:] = 1  # client 1
    elif scenario == '3A':
        mask[:, :4,  :, :24]  = 1  # client 0
        mask[:, 3:7, :, 24:47] = 1  # client 1 (overlaps 3,6)
        mask[:, 6:10, :, 47:] = 1  # client 2
    elif scenario == '3B':
        mask[:, 0:3, :, :24]   = 1  # client 0
        mask[:, 3:7, :, 24:47] = 1  # client 1
        mask[:, 7:10, :, 47:]   = 1  # client 2
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    # Consistent scaling across scenarios
    diff = (y - predicted_seismic).abs() * mask
    denom = mask.sum().clamp_min(1.0)  # number of active elements

    return diff.sum() / denom

