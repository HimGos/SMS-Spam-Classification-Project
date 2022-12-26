from classifier.entity import artifact_entity, config_entity
from classifier.logger import logging
from classifier.exception import SpamException
from classifier.predictor import ModelResolver
from classifier.utils import load_object
from classifier.utils import transform_text
from sklearn.metrics import precision_score
from classifier.config import TARGET_COLUMN
import pandas as pd
import os, sys


class ModelEvaluation:

    def __init__(self,
                 model_eval_config: config_entity.ModelEvaluationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact: artifact_entity.ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*20} Model Evaluation {'<<'*20}")
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise SpamException(e, sys)

    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
        try:
            # If saved model folder has model then we will compare
            # which model folder is best trained
            logging.info("If saved model folder has model then we will compare"
                         " which model folder is best trained")
            latest_dir_path = None  #self.model_resolver.get_latest_dir_path()
            if latest_dir_path is None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(
                    is_model_accepted=True, improved_accuracy=None)
                logging.info(f"Model Evaluation Artifact: {model_eval_artifact}")
                return model_eval_artifact

            # Finding location of transformer model and target encoder
            logging.info("Finding location of transformer, model and target encoder")
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()
            target_encoder_path = self.model_resolver.get_latest_target_encoder_path()

            #Loading previously trained objects
            logging.info("Loading Previously trained transformer, model & target encoder")
            transformer = load_object(file_path=transformer_path)
            model = load_object(file_path=model_path)
            target_encoder = load_object(file_path=target_encoder_path)

            # Currently Trained Model objects
            logging.info("Currently trained model objects")
            current_transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            current_model = load_object(file_path=self.model_trainer_artifact.model_path)
            current_target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            test_df = pd.read_csv(self.data_ingestion_artifact.feature_store_file_path)
            target_df = test_df[TARGET_COLUMN]
            # Accuracy using previous trained model
            y_true = target_encoder.transform(target_df)
            input_df = test_df.drop('Label', axis=1)
            input_df['Message'] = input_df['Message'].apply(transform_text)
            input_arr = transformer.transform(input_df['Message']).toarray()
            y_pred = model.predict(input_arr)
            previous_model_score = precision_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"Precision Score using previous model: {previous_model_score}")

            # Accuracy using Current Trained Model
            y_true = current_target_encoder.transform(target_df)
            input_df = test_df.drop('Label', axis=1)
            input_df['Message'] = input_df['Message'].apply(transform_text)
            input_arr = current_transformer.transform(input_df['Message']).toarray()
            y_pred = current_model.predict(input_arr)
            current_model_score = precision_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"Precision Score using Current model: {current_model_score}")

            # Comparing Models
            if current_model_score <= previous_model_score:
                logging.info(f"Current Trained model is not better than previous model")
                raise Exception(f"Current Trained model is not better than previous model")

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
                                          improved_accuracy=current_model_score-previous_model_score)
            logging.info(f"Model evaluation artifact: {model_eval_artifact}")
            return model_eval_artifact
        except Exception as e:
            raise SpamException(e, sys)
