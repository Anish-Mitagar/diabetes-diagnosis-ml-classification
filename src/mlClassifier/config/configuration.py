from mlClassifier.constants import *
from mlClassifier.utils.common import read_yaml, create_directories
from mlClassifier.entity import (DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig, ModelValidationConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            save_path=config.save_path,
            preprocessor_path=config.preprocessor_path
        )

        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        

        create_directories([config.root_dir])
        create_directories([config.save_path_best_acc])
        create_directories([config.save_path_best_f1])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            save_path_best_acc = config.save_path_best_acc,
            save_path_best_f1 = config.save_path_best_f1,
            neural_network_params = self.params.neural_network_params,
            random_forest_params = self.params.random_forest_params,
            xg_boost_params = self.params.xg_boost_params,
            cat_boost_params = self.params.cat_boost_params,
            ada_boost_params = self.params.ada_boost_params
        )

        return model_trainer_config
    
    def get_model_validation_config(self) -> ModelValidationConfig:
        config = self.config.model_validation
        

        create_directories([config.root_dir])

        model_validation_config = ModelValidationConfig(
            root_dir=config.root_dir,
            save_dir=config.save_dir,
            data_path=config.data_path,
            best_acc_model_path_pth=config.best_acc_model_path_pth,
            best_acc_model_path_pkl=config.best_acc_model_path_pkl,
            best_f1_model_path_pth=config.best_f1_model_path_pth,
            best_f1_model_path_pkl=config.best_f1_model_path_pkl
        )

        return model_validation_config