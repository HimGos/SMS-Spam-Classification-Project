from classifier.pipeline.training_pipeline import start_training_pipeline
from classifier.utils import transform_text
from classifier.predictor import ModelResolver
from classifier.utils import load_object
from PIL import Image
import streamlit as st


if __name__ == "__main__":

    # start_training_pipeline()

    model_resolver = ModelResolver(model_registry="saved_models")
    transformer = load_object(file_path=model_resolver.get_latest_transformer_path())
    model = load_object(file_path=model_resolver.get_latest_model_path())

    st.set_page_config(page_title="Spam Classifier")
    st.title("Spam Classifier")
    st.write("by - [Himanshu Goswami](https://github.com/HimGos)")

    # Taking Input Message
    input_sms = st.text_area("Enter the message...")

    if st.button('Predict'):

        # Transforming Text
        transform_sms = transform_text(input_sms)
        # Vectorizing using TF IDF
        vector_sms = transformer.transform([transform_sms])
        # Predicting using MultinomialNB
        result = model.predict(vector_sms)[0]

        if result == 1:
            image = Image.open('.streamlit/Spam HD - bg removed.jpg')
            st.image(image)
        else:
            image = Image.open('.streamlit/Not Spam HD - bg removed.jpg')
            st.image(image)

    # UI Customization
    st.markdown(""" <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style> """, unsafe_allow_html=True)




