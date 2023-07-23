from mlClassifier.config.configuration import ConfigurationManager
from mlClassifier.components.model_validation import ModelValidation

class ModelValidationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_model_validation_config()
        data_transformation = ModelValidation(config=data_validation_config)
        data_transformation.validate_model_performance()