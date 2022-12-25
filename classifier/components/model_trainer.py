from classifier.entity import artifact_entity, config_entity
from classifier.logger import logging
from classifier.exception import SpamException
from classifier import utils
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score
import sys
from sklearn.model_selection import train_test_split


class ModelTrainer:

    def __init__(self, model_trainer_config: config_entity.ModelTrainerConfig,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise SpamException(e, sys)

    def train_model(self, x, y):
        try:
            multi_nb = MultinomialNB()
            multi_nb.fit(x, y)
            return multi_nb
        except Exception as e:
            raise SpamException(e, sys)

    def initiate_model_trainer(self,) -> artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"Loading train and test array")
            # train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            # test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)
            data_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_file_path)

            logging.info("Splitting input and target feature from both train and test arr")
            # X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            # X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            input_arr, target_arr = data_arr[:, :-1], data_arr[:, -1]

            X_train, X_test, y_train, y_test = train_test_split(input_arr, target_arr,
                                                                random_state=42, test_size=0.2)

            logging.info(f"X_train shape:{X_train.shape}, X_test shape:{X_test.shape}")
            logging.info(f"y_train shape:{y_train.shape}, y_test shape:{y_test.shape}")

            logging.info(f"Train the model")
            model = self.train_model(x=X_train, y=y_train)

            logging.info(f"Calculating Accuracy Train Score")
            yhat_train = model.predict(X_train)
            accuracy_train_score = accuracy_score(y_true=y_train, y_pred=yhat_train)

            logging.info(f"Calculating Accuracy Test Score")
            yhat_test = model.predict(X_test)
            accuracy_test_score = accuracy_score(y_true=y_test, y_pred=yhat_test)
            precision_test_score = precision_score(y_true=y_test, y_pred=yhat_test)
            logging.info(f"Precision Score: {precision_test_score}")

            logging.info(f"Train Score: {accuracy_train_score} and Test Score: {accuracy_test_score}")
            if accuracy_test_score < self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give "
                                f"expected accuracy: {self.model_trainer_config.expected_score}"
                                f"Model Actual score: {accuracy_test_score}")

            logging.info(f"Checking if our model is overfitting or not")
            diff = abs(accuracy_train_score - accuracy_test_score)

            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score diff: {diff} if more than overfitting threshold"
                                f"{self.model_trainer_config.overfitting_threshold}")

            # Save trained Model
            logging.info(f"Saving model object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            # Prepare the artifact
            logging.info(f"Prepare the artifact")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path,
                                                                          accuracy_train_score=accuracy_train_score,
                                                                          accuracy_test_score=accuracy_test_score)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise SpamException(e, sys)
