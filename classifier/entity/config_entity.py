from classifier.exception import SpamException
from classifier.logger import logging
from datetime import datetime
import os, sys

FILE_NAME = "spam_classifier.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TRANSFORMER_OBJECT_FILE_NAME = "transformer.pkl"
MODEL_FILE_NAME = "model.pkl"


class TrainingPipelineConfig:

    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(), "artifact", f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception as e:
            raise SpamException(e, sys)


class DataIngestionConfig:

    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        try:
            self.database_name = "SpamProject"
            self.collection_name = "classifier"
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir , "data_ingestion")
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir, "feature_store", FILE_NAME)
            self.train_file_path = os.path.join(self.data_ingestion_dir, "dataset", TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir, "dataset", TEST_FILE_NAME)
            self.test_size = 0.2
        except Exception as e:
            raise SpamException(e, sys)

    def to_dict(self)->dict:
        try:
            return self.__dict__
        except Exception as e:
            raise SpamException(e, sys)


class DataValidationConfig: ...


class DataTransformationConfig: ...


class ModelTrainerConfig: ...


class ModelEvaluationConfig: ...


class ModelPusherConfig: ...
