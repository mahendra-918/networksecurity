from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig,DataValidationConfig
import sys

if __name__ == "__main__":
    try:
        traningpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(traningpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("initiated the data ingestion")
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        print(dataingestionartifact)
        logging.info("Data initiation completed")
        datavalidationconfig = DataValidationConfig(traningpipelineconfig)
        data_validation = DataValidation(dataingestionartifact,datavalidationconfig)
        logging.info("Initiate the data validation")
        datavalidationartifact = data_validation.initiate_data_validation()
        logging.info("data validation is completed")
        
        
    except Exception as e:
        raise NetworkSecurityException(e,sys)