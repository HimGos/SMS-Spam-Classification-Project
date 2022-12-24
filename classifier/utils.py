import numpy as np
import pandas as pd
from classifier.exception import SpamException
from classifier.logger import logging
from classifier.config import mongo_client
import os
import sys
import yaml
import dill


def get_collection_as_df(database_name:str, collection_name:str) -> pd.DataFrame:
    """
    Description: This function return collection as dataframe
    =========================================================
    Params:
    database_name: database name
    collection_name: collection name
    =========================================================
    return Pandas dataframe of a collection
    """
    try:
        logging.info(f"Reading data from database: {database_name} and collection: {collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"Found Columns: {df.columns}")
        if "_id" in df.columns:
            logging.info("Dropping column: _id")
            df = df.drop("_id", axis=1)
            logging.info(f"Row and column in df: {df.shape}")
        df = df.drop_duplicates()
        return df
    except Exception as e:
        raise SpamException(e, sys)


def write_yaml_file(file_path, data: dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)
        with open(file_path, "w") as file_writer:
            yaml.dump(data, file_writer)
    except Exception as e:
        raise SpamException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of utils")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Exiting the save_object method of utils")
    except Exception as e:
        raise SpamException(e, sys)


def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} doesn't exist")
        with open(file_path, "rb") as file_obj:
            dill.load(file_obj)
    except Exception as e:
        raise SpamException(e, sys)


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    :param file_path: str location of file to save
    :param array: np.array data to save
    :return: None
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise SpamException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    Load Numpy array data from file
    :param file_path: str location of file to load
    :return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise SpamException(e, sys) from e


def preprocessing_input_text():
    pass


