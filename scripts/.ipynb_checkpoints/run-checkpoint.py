import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from functools import partial
import copy
import time
from typing import Dict, List, Tuple, Optional
import flwr as fl
from flwr.common import Metrics, NDArrays, Scalar, Parameters, ndarrays_to_parameters, parameters_to_ndarrays

import os
import tempfile
import pickle
from datetime import datetime
from flwr.server.strategy import FedAvg, FedProx, FedAvgM, FedOpt
from diffusion_models.diffusion_model import *
from pde_solvers.pde_solver_5clients import FWIForward 
from torch.optim import Adam
from scripts.data_trans import v_normalize, v_denormalize, s_normalize_none, s_normalize, s_denormalize
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ctx = {
    'n_grid': 70, 'nt': 1000, 'dx': 10, 'nbc': 120, 'dt': 1e-3, 'f': 15,
    'sz': 10, 'gz': 10, 'ng': 70, 'ns': 5
}

fwi_forward = FWIForward(ctx, device, normalize=True, v_denorm_func=v_denormalize, s_norm_func=s_normalize_none)
diffusion_args = {
    'dim': 64, 'dim_mults': (1, 2, 4, 8), 'flash_attn': False, 'channels': 1,
    'image_size': 72, 'timesteps': 1000, 'sampling_timesteps': 250, 'objective': 'pred_noise'
}

print("Creating diffusion model structure...")
unet_model = Unet(
    dim=diffusion_args.get('dim', 64),
    dim_mults=diffusion_args.get('dim_mults', (1, 2, 4, 8)),
    flash_attn=diffusion_args.get('flash_attn', False),
    channels=diffusion_args.get('channels', 1)
)

diffusion = GaussianDiffusion(
    unet_model,
    image_size=diffusion_args.get('image_size', 72),
    timesteps=diffusion_args.get('timesteps', 1000),
    sampling_timesteps=diffusion_args.get('sampling_timesteps', 250),
    objective=diffusion_args.get('objective', 'pred_noise')
).to(device)

