from pydantic import BaseModel
from typing import Optional
import yaml
class PathConfig(BaseModel):
    velocity_data_path: str
    client_seismic_data_path: str
    gt_seismic_data_path: str
    model_path: str
    output_path: str

class ExperimentConfig(BaseModel):
    regularization: Optional[str] = None
    reg_lambda: float
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
    sx: Optional[list] = None

class AppConfig(BaseModel):
    path: PathConfig
    experiment: ExperimentConfig
    diffusion: DiffusionConfig
    forward: ForwardConfig

def load_config(path):
    with open(path) as f:
        raw_config = yaml.safe_load(f)
    return AppConfig(**raw_config)

