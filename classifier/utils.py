import pandas as pd
from classifier.exception import SpamException
from classifier.logger import logging
from classifier.config import mongo_client
import os
import sys


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
        return df
    except Exception as e:
        raise SpamException(e, sys)


def preprocessing_input_text():
    pass
