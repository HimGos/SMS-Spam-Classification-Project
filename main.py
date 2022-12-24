from classifier.exception import SpamException
from classifier.entity import config_entity
from classifier.entity import artifact_entity
from classifier.components.data_ingestion import DataIngestion
from classifier.components.data_validation import DataValidation
from classifier.components.data_transformation import DataTransformation


if __name__ == "__main__":

    # Training Pipeline Config
    training_pipeline_config = config_entity.TrainingPipelineConfig()

    # Data Ingestion
    data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
    print(data_ingestion_config.to_dict())
    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

    # Data Validation
    data_validation_config = config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
    data_validation = DataValidation(data_validation_config=data_validation_config,
                                     data_ingestion_artifact=data_ingestion_artifact)
    data_validation_artifact = data_validation.initiate_data_validation()

    # Data Transformation
    data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
    data_transformation = DataTransformation(data_transformation_config=data_transformation_config,
                                             data_ingestion_artifact=data_ingestion_artifact)
    data_transformation_artifact = data_transformation.initiate_data_transformation()
