from classifier.pipeline.training_pipeline import start_training_pipeline
from classifier.utils import transform_text
from classifier.predictor import ModelResolver
from classifier.utils import load_object
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score,confusion_matrix
from classifier.logger import logging
import dill
import os, sys
import pandas as pd


if __name__ == "__main__":

    # start_training_pipeline()

    df = pd.read_csv("SMSSpamCollection.csv", sep='\t', names=['Label', 'Message'])

    model_resolver = ModelResolver(model_registry="saved_models")
    transformer = load_object(file_path=model_resolver.get_latest_transformer_path())
    model = load_object(file_path=model_resolver.get_latest_model_path())
    target_encoder = load_object(file_path=model_resolver.get_latest_target_encoder_path())
    print(model_resolver.get_latest_target_encoder_path())
    print(type(target_encoder))

    # with open(r"artifact/12262022__123250/model_pusher/saved_models/transformer.pkl", "rb") as file:
    #     transformer = dill.load(file)
    #
    # with open(r"artifact/12262022__114130/data_transformation/target_encoder/target_encoder.pkl", "rb") as file:
    #     target_encoder = dill.load(file)
    #
    # with open(r"artifact/12262022__114130/model_trainer/model/model.pkl", "rb") as file:
    #     model = dill.load(file)
    # print(type(model))
    # print(type(transformer))

    input_df = df.drop('Label', axis=1)
    input_df['Message'] = input_df['Message'].apply(transform_text)
    input_df_arr = transformer.transform(input_df['Message']).toarray()

    target_df = df['Label']
    target_df_arr = target_encoder.transform(target_df)

    X_train, X_test, y_train, y_test = train_test_split(input_df_arr, target_df_arr,
                                                        random_state=42, test_size=0.2)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    logging.info(f"Accuracy: {accuracy}, Precision: {precision}")
    logging.info(f"Confusion matrix: {confusion}")
