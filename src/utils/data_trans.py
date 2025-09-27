### This script contains all the necessary code for the data transformation ###
import torch
import numpy as np
from scipy.ndimage import gaussian_filter

def v_normalize(v):
    """Normalize velocity values to [-1, 1]"""
    return (((v - 1500) / 3000) * 2) - 1

def v_denormalize(v_norm):
    """Denormalize velocity values from [-1, 1] to original range"""
    return ((v_norm + 1) / 2) * 3000 + 1500

def s_normalize_none(s):
    """Keep the data in original scale"""
    return s
    
def s_normalize(s):
    """Normalize seismic data to [-1, 1]"""
    return (((s + 20) / 80) * 2) - 1

def s_denormalize(s_norm):
    """Un-normalize the seismic data back to [-20, 60]"""
    return ((s_norm + 1) / 2) * 80 - 20

def add_noise_to_seismic(y, std):
    assert std >= 0, "The standard deviation of the noise must be greater than 0"
    if std == 0:
        return y
    else:
        y = y.detach().cpu().numpy()
        noise = np.random.normal(0, std, y.shape)
        y_noisy = y + noise
        y = torch.tensor(y_noisy).float()
        return y

def prepare_initial_model(v_true, initial_type=None, sigma=None, linear_coeff=1.0):
    assert initial_type in ['smoothed', 'homogeneous', 'linear'], "please choose from 'smoothed' 'homogeneous', and 'linear'"
    v = v_true.clone()
    v_np = v.cpu().numpy()
    v_np = v_normalize(v_np)
    
    if initial_type == 'smoothed':
        v_blurred = gaussian_filter(v_np, sigma=sigma)
    elif initial_type == 'homogeneous':
        min_top_row = np.min(v_np[0, 0, 0, :])
        v_blurred = np.full_like(v_np, min_top_row)
    elif initial_type == 'linear':
        # Get velocity range from true model
        v_min = np.min(v_np)
        v_max = np.max(v_np)
        
        # Create linear gradient with depth
        height = v_np.shape[2]
        depth_gradient = np.linspace(v_min, v_max, height)
        
        # Expand dimensions to match input shape
        depth_gradient = depth_gradient.reshape(-1, 1)  # Make it (height, 1)
        v_blurred = np.tile(depth_gradient, (1, v_np.shape[3]))  # Repeat for width
        v_blurred = v_blurred.reshape(1, 1, height, -1)  # Reshape to (1, 1, height, width)
    
    # Move to the same device as input tensor to avoid device mismatch
    v_blurred = torch.tensor(v_blurred).float().to(v_true.device)
    return v_blurred

def missing_trace(y, num_missing):
    assert num_missing >= 0, "The number of missing traces must be greater than 0"
    if num_missing == 0:
        return y
    else:
        y_np = y.detach().cpu().numpy()
        batch_size, num_sources, time_samples, num_traces = y.shape
        y_missing = y_np.copy()
        for b in range(batch_size):
            for s in range(num_sources):
                missing_indices = np.random.choice(num_traces, num_missing, replace=False)
                y_missing[b, s, :, missing_indices] = 0
        y = torch.tensor(y_missing).float()
    return y