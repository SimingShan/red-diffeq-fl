from pydantic import BaseModel
from typing import Optional

class PathConfig(BaseModel):
    velocity_data_path: str
    seismic_data_path: str
    model_path: str
    output_path: str

class FederatedConfig(BaseModel):
    num_rounds: int
    local_epochs: int
    local_lr: float

class ExperimentConfig(BaseModel):
    strategy: str
    regularization: Optional[str] = None
    server_momentum: float  #for FedAvgM
    num_clients: int
    scenario_flag: Optional[str] = None

class DiffusionConfig(BaseModel):
    dim: int
    dim_mults: tuple
    flash_attn: bool
    channels: int
    image_size: int
    timesteps: int
    sampling_timesteps: int
    objective: str
    
class ForwardConfig(BaseModel):
    initial_sigma: int
    n_grid: int
    nt: int
    dx: int
    nbc: int
    dt: float
    f: int
    sz: int
    gz: int
    ng: int 
    ns: int


class AppConfig(BaseModel):
    path: PathConfig
    federated: FederatedConfig
    experiment: ExperimentConfig
    diffusion: DiffusionConfig
    forward: ForwardConfig
