from classifier.exception import SpamException
from classifier.entity import config_entity
from classifier.entity import artifact_entity
from classifier.components.data_ingestion import DataIngestion


if __name__ == "__main__":
    training_pipeline_config = config_entity.TrainingPipelineConfig()
    data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
    print(data_ingestion_config.to_dict())
    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
