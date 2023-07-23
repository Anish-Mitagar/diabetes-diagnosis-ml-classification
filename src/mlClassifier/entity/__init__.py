from dataclasses import dataclass
from pathlib import Path
from box.config_box import ConfigBox

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    save_path: Path
    preprocessor_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    save_path_best_acc: Path
    save_path_best_f1: Path
    neural_network_params: ConfigBox
    random_forest_params: ConfigBox
    xg_boost_params: ConfigBox
    cat_boost_params: ConfigBox
    ada_boost_params: ConfigBox

@dataclass(frozen=True)
class ModelValidationConfig:
    root_dir: Path
    save_dir: Path
    data_path: Path
    best_acc_model_path_pth: Path
    best_acc_model_path_pkl: Path
    best_f1_model_path_pth: Path
    best_f1_model_path_pkl: Path