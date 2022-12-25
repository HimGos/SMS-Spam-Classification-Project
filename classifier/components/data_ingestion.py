from classifier import utils
from classifier.exception import SpamException
from classifier.logger import logging
from classifier.entity import config_entity
from classifier.entity import artifact_entity
from sklearn.model_selection import train_test_split
import os
import sys
import pandas as pd


class DataIngestion:

    def __init__(self, data_ingestion_config: config_entity.DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20} Data Ingestion {'<<'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise SpamException(e, sys)

    def initiate_data_ingestion(self) -> artifact_entity.DataIngestionArtifact:

        try:
            logging.info("Importing Collection data as pandas dataframe")
            # Importing collection as pandas dataframe
            df: pd.DataFrame = utils.get_collection_as_df(
                database_name=self.data_ingestion_config.database_name,
                collection_name=self.data_ingestion_config.collection_name)

            logging.info("Save data in feature store")

            # Save data in feature store
            logging.info("Create feature store if not available")
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)
            logging.info(f"Save entire DataFrame to feature store folder")
            # Save df to feature store folder
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path, index=False, header=True)

            logging.info(f"Split dataset into train and test set")
            # Splitting data into train and test
            train_df, test_df = train_test_split(df, random_state=42, test_size=self.data_ingestion_config.test_size)

            logging.info("Create dataset directory folder if not available")
            # Creating dataset directory to save train and test files
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir, exist_ok=True)

            logging.info(f"Save train and test dfs to dataset folder")
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path, index=False, header=True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path, index=False, header=True)

            # Prepare artifact

            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path)

            logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise SpamException(e, sys)
