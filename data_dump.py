import pymongo
import pandas as pd
import json

from classifier.config import mongo_client

DATA_FILE_PATH = "SMSSpamCollection.csv"
DATABASE_NAME = "SpamProject"
COLLECTION_NAME = "classifier"

if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH, sep='\t', names=['Label', 'Message'])
    print(f"Rows and Columns: {df.shape}")

    # Convert Dataframe to json format to dump these record in mongodb
    df.reset_index(drop=True, inplace=True)
    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    # Insert converted json record to mongodb
    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)


