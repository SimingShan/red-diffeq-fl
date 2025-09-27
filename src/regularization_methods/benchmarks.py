import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from functools import partial

def total_variation_loss(mu):
    """Computes Total Variation Loss for the velocity map (mu)."""
    diff_x = torch.abs(mu[:, :, :, 1:] - mu[:, :, :, :-1])  # Horizontal differences
    diff_y = torch.abs(mu[:, :, 1:, :] - mu[:, :, :-1, :])  # Vertical differences
    tv_loss = torch.mean(diff_x) + torch.mean(diff_y)
    return tv_loss

def tikhonov_loss(mu):
    """Computes L2 regularization (Tikhonov) for the velocity map (mu)."""
    diff_x = mu[:, :, :, 1:] - mu[:, :, :, :-1] # Horizontal differences
    diff_y = mu[:, :, 1:, :] - mu[:, :, :-1, :] # Vertical differences    
    l2_loss = torch.mean(diff_x ** 2) + torch.mean(diff_y ** 2)
    
    return l2_loss