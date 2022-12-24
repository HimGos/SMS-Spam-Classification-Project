from classifier.entity import artifact_entity, config_entity
from classifier.exception import SpamException
from classifier.logger import logging
from classifier import utils
from sklearn.preprocessing import LabelEncoder
from classifier.config import TARGET_COLUMN
from typing import Optional
import pandas as pd
import numpy as np
import os, sys
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')


class DataTransformation:

    def __init__(self, data_transformation_config: config_entity.DataTransformationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise SpamException(e, sys)

    @classmethod
    def transform_text(cls):
        pass

    def initiate_data_transformation(self) -> artifact_entity.DataTransformationArtifact:
        try:
            # Reading train and test file
            logging.info("Reading train and test files")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # Selecting input feature for train and test df
            logging.info("Selecting input features in both dfs")
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis=1)

            # Selecting target feature for train and test df
            logging.info("Selecting output feature in both dfs")
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info("Encoding...")
            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)

            # Transformation on target columns
            target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)


        except Exception as e:
            raise SpamException(e, sys)
