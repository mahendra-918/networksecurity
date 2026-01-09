from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig,DataValidationConfig,DataTransformationConfig
import sys
from networksecurity.components.data_transformation import DataTransformation

if __name__ == "__main__":
    try:
        traning_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(traning_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("initiated the data ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
        logging.info("Data initiation completed")
        data_validation_config = DataValidationConfig(traning_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact,data_validation_config)
        logging.info("Initiate the data validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("data validation is completed")
        print(data_validation_artifact)
        data_transformation_config=DataTransformationConfig(traning_pipeline_config)
        logging.info("data Transformation started")
        data_transformation=DataTransformation(data_validation_artifact,data_transformation_config)
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("data Transformation completed")

        
        
    except Exception as e:
        raise NetworkSecurityException(e,sys)