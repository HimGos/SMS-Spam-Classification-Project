from classifier.entity import artifact_entity, config_entity
from classifier.logger import logging
from classifier.exception import SpamException
from classifier import utils
import sys
import pandas as pd


class DataValidation:

    def __init__(self,
                 data_validation_config: config_entity.DataValidationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = dict()
        except Exception as e:
            raise SpamException(e, sys)

    def is_required_columns_exist(self, base_df: pd.DataFrame, current_df: pd.DataFrame, report_key_name: str) -> bool:
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns
            if len(base_columns) != len(current_columns):
                self.validation_error[report_key_name] = len(current_columns)
                return False
            return True
        except Exception as e:
            raise SpamException(e, sys)

    def is_columns_datatypes_same(self, base_df: pd.DataFrame, current_df: pd.DataFrame, report_key_name: str) -> bool:
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns
            for column in range(len(base_columns)):
                if base_df[base_columns[column]].dtype != current_df[current_columns[column]].dtype:
                    self.validation_error[report_key_name] = f"Different datatype in current df: " \
                                                             f"{current_df[current_columns[column]].dtype} "
                    return False
            return True
        except Exception as e:
            raise SpamException(e, sys)

    def initiate_data_validation(self) -> artifact_entity.DataValidationArtifact:
        try:
            logging.info("Reading Base Df")
            base_df = pd.read_csv(self.data_validation_config.base_file_path, sep='\t')

            logging.info("Reading Train df")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info("Reading Test df")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # Checking if required columns exist
            logging.info(f"Does all required columns exist in train df")
            train_df_columns_status = self.is_required_columns_exist(base_df=base_df,
                                                                     current_df=train_df,
                                                                     report_key_name='missing_columns_within_train_df')
            logging.info(f"Does all required columns exist in test df")
            test_df_columns_status = self.is_required_columns_exist(base_df=base_df,
                                                                    current_df=test_df,
                                                                    report_key_name='missing_columns_within_test_df')

            if train_df_columns_status:
                logging.info("All columns are available in train df")
                self.validation_error['Train_df_columns_status'] = 'All Columns Available in Train DataFrame'
            if test_df_columns_status:
                logging.info("All columns are available in test df")
                self.validation_error['Test_df_columns_status'] = 'All Columns Available in Test DataFrame'

            # Checking if all datatypes same
            logging.info("Does all columns in train df are of same type as base df")
            train_df_dtypes = self.is_columns_datatypes_same(base_df=base_df,
                                                             current_df=train_df,
                                                             report_key_name='datatypes_in_train_df')
            logging.info("Does all columns in test df are of same type as base df")
            test_df_dtypes = self.is_columns_datatypes_same(base_df=base_df,
                                                            current_df=test_df,
                                                            report_key_name='datatypes_in_test_df')
            if train_df_dtypes:
                logging.info("Datatypes of train df are ok")
                self.validation_error['Train_df_datatypes'] = 'Same Datatypes of Train df columns'
            if test_df_dtypes:
                logging.info("Datatypes of test df are ok")
                self.validation_error['Test_df_datatypes'] = 'Same Datatypes of Test df columns'

            # Writing report
            logging.info("Writing report in yaml file")
            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path,
                                  data=self.validation_error)
            data_validation_artifact = artifact_entity.DataValidationArtifact(
                report_file_path=self.data_validation_config.report_file_path)
            logging.info(f"Data Validation artifact: {data_validation_artifact}")

            return data_validation_artifact
        except Exception as e:
            raise SpamException(e, sys)
