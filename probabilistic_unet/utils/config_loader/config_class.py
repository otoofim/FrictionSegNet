from dataclasses import dataclass
from typing import List

from pathlib import Path


@dataclass
class DatasetConfig:
    input_img_dim: List[int]
    reducedCategories: bool
    cityscapesRootPath: Path
    mapillaryRootPath: Path
    RSCDRootPath: Path
    ACDCRootPath: Path
    volvoRootPath: Path
    RSCD_cat: List[str]
    MapillSubsample: float

    def __post_init__(self):
        # Convert string paths to Path objects
        self.cityscapesRootPath = Path(self.cityscapesRootPath)
        self.mapillaryRootPath = Path(self.mapillaryRootPath)
        self.RSCDRootPath = Path(self.RSCDRootPath)
        self.ACDCRootPath = Path(self.ACDCRootPath)
        self.volvoRootPath = Path(self.volvoRootPath)


@dataclass
class PretrainedConfig:
    enable: bool
    which_model: str
    model_add: Path

    def __post_init__(self):
        self.model_add = Path(self.model_add)


@dataclass
class ContinueTrainingConfig:
    enable: bool
    wandb_id: str
    which_model: str


@dataclass
class CustomizedGECOConfig:
    enable: bool
    goal_fri: float
    goal_seg: float
    alpha: float
    beta: float
    step_size: float
    lambda_s: float
    lambda_f: float
    speedup: int
    beta_min: float
    beta_max: float


@dataclass
class GECOConfig:
    enable: bool
    goal_seg: float
    goal_fri: float
    step_size: float
    alpha: float
    speedup: int
    beta_init: float


@dataclass
class ProjectConfig:
    run_name: str
    project_name: str
    model_add: Path = None
    entity: str
    device: str
    device_name: str
    latent_dim: int
    beta: float
    batch_size: int
    num_samples: int
    momentum: float
    epochs: int
    learning_rate: float
    pos_weight: List[float]
    lossType: str
    datasetConfig: DatasetConfig
    pretrained: PretrainedConfig
    continue_tra: ContinueTrainingConfig
    customized_GECO: CustomizedGECOConfig
    GECO: GECOConfig
