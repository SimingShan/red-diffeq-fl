### This script contains all the necessary code for the data transformation ###
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
### Normalization (Velocity Map)
def v_normalize(v):
    """Normalize velocity values to [-1, 1]"""
    return (((v - 1500) / 3000) * 2) - 1

def v_denormalize(v_norm):
    """Denormalize velocity values from [-1, 1] to original range"""
    return ((v_norm + 1) / 2) * 3000 + 1500


### Normalization (Seismic Data)
def s_normalize_none(s):
    """Keep the data in original scale"""
    return s
    
def s_normalize(s):
    """Normalize seismic data to [-1, 1]"""
    return (((s + 20) / 80) * 2) - 1

def s_denormalize(s_norm):
    """Un-normalize the seismic data back to [-20, 60]"""
    return ((s_norm + 1) / 2) * 80 - 20

### Add Noise To The Seismic Data ###
def add_noise_to_seismic(y, std):
    # Convert PyTorch tensor to NumPy array and remove batch dimension
    y_np = y.detach().cpu().numpy()
    
    # Generate noise
    noise = np.random.normal(0, std, y_np.shape)
    
    # Add noise to normalized data
    y_noisy = y_np + noise
    
    return y_noisy

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
    
    v_blurred = torch.tensor(v_blurred).float().to('cuda')
    return v_blurred

### Prepare the missing trace seismic Data ###
def missing_trace(y, num_missing=10):
    """
    Simulate missing traces in seismic data.
    
    :param y: Seismic data with shape (batch_size, num_sources, time_samples, num_traces)
    :param num_missing: Number of traces to set to zero
    :return: Seismic data with missing traces
    """
    print(f"Input shape: {y.shape}")
    batch_size, num_sources, time_samples, num_traces = y.shape
    
    # Create a copy of the input data to avoid modifying the original
    y_missing = y.copy()
    
    for b in range(batch_size):
        for s in range(num_sources):
            # Randomly select traces to set to zero
            missing_indices = np.random.choice(num_traces, num_missing, replace=False)
            y_missing[b, s, :, missing_indices] = 0
    
    print(f"Number of traces set to zero per batch and source: {num_missing}")
    return y_missing