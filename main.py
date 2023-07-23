from mlClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from mlClassifier.pipeline.stage_02_data_transformation import DataTransformationPipeline
from mlClassifier.pipeline.stage_03_model_trainer import ModelTrainerPipeline
from mlClassifier.pipeline.stage_04_model_validation import ModelValidationPipeline
from mlClassifier.utils.exception import CustomException
import sys
from mlClassifier.logging import logger


STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx================================================================================x")
except Exception as e:
    logger.exception(str(CustomException(e, sys)))
    raise CustomException(e, sys)

STAGE_NAME = "Data Transformation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_transformation = DataTransformationPipeline()
   data_transformation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx================================================================================x")
except Exception as e:
    logger.exception(str(CustomException(e, sys)))
    raise CustomException(e, sys)

STAGE_NAME = "Model Trainer stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   model_trainer = ModelTrainerPipeline()
   model_trainer.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx================================================================================x")
except Exception as e:
    logger.exception(str(CustomException(e, sys)))
    raise CustomException(e, sys)

STAGE_NAME = "Model Validation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   model_validation = ModelValidationPipeline()
   model_validation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx================================================================================x")
except Exception as e:
    logger.exception(str(CustomException(e, sys)))
    raise CustomException(e, sys)