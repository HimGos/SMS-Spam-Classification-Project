from classifier.exception import SpamException
from datetime import datetime
import os, sys

FILE_NAME = "spam_classifier.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TRANSFORMER_OBJECT_FILE_NAME = "transformer.pkl"
MODEL_FILE_NAME = "model.pkl"


class TrainingPipelineConfig:

    def __int__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(), "artifact", f"{datetime.now().strftime('%m%d%Y_%H%M%S')}")
        except Exception as e:
            raise SpamException(e, sys)


class DataIngestionConfig: ...


class DataValidationConfig: ...


class DataTransformationConfig: ...


class ModelTrainerConfig: ...


class ModelEvaluationConfig: ...


class ModelPusherConfig: ...
