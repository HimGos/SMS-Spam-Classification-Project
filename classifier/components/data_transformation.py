from classifier.entity import artifact_entity, config_entity
from classifier.exception import SpamException
from classifier.logger import logging
from classifier import utils
from sklearn.preprocessing import LabelEncoder
from classifier.config import TARGET_COLUMN
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional
import pandas as pd
import numpy as np
import os, sys
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')

porter_stemmer = PorterStemmer()

class DataTransformation:

    def __init__(self, data_transformation_config: config_entity.DataTransformationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise SpamException(e, sys)

    def transform_text(self, text: str) -> str:
        try:

            text = text.lower()
            text = nltk.word_tokenize(text)
            corpus = []
            for word in text:
                if word.isalnum():
                    corpus.append(word)
            text = corpus[:]
            corpus.clear()
            for word in text:
                if word not in stopwords.words('english') and word not in string.punctuation:
                    corpus.append(word)
            text = corpus[:]
            corpus.clear()
            for word in text:
                corpus.append(porter_stemmer.stem(word))
            return " ".join(corpus)
        except Exception as e:
            raise SpamException(e, sys)

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

            # Transformation on target column
            target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)

            # Transformation on input column
            input_feature_train_df['Message'] = input_feature_train_df['Message'].apply(self.transform_text)
            input_feature_test_df['Message'] = input_feature_test_df['Message'].apply(self.transform_text)

            # Vectorizing input columns
            logging.info("Vectorizing...")
            tfidf = TfidfVectorizer(max_features=3000)
            input_feature_train_arr = tfidf.fit_transform(input_feature_train_df['Message']).toarray()
            input_feature_test_arr = tfidf.fit_transform(input_feature_test_df['Message']).toarray()

            # target encoder
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            # Save Numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)

            # Save objects
            utils.save_object(file_path=self.data_transformation_config.target_encoder_path,
                              obj=label_encoder)
            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
                              obj=tfidf)

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path=self.data_transformation_config.transformed_train_path,
                transformed_test_path=self.data_transformation_config.transformed_test_path,
                target_encoder_path=self.data_transformation_config.target_encoder_path
            )

            logging.info(f"Data Transformation Object {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise SpamException(e, sys)
