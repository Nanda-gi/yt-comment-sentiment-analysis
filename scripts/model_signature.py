import mlflow
import pytest
import pandas as pd
import pickle
from mlflow.tracking import MlflowClient
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
nltk.download('stopwords')
nltk.download('wordnet')

def processing(text):
  # lower the text
  if pd.isnull(text):
    return ""
  else:
    text=text.lower()
  # remove URL from text
  text=re.sub(r'https?://\S+|www\.\S+','',text)
  # remove newline from text
  text=re.sub(r'\n','',text)
  # remove aplhanumeric from text
  text=re.sub(r'[^a-zA-Z0-9\s!?.,]',"",text)
  # remove stopwords from text
  stop_words=list(set(stopwords.words("english")))
  no_stop_words_sentiment=set(stop_words)-{'not','but','yet','however','no'}
  text=" ".join([word for word in text.split(" ") if word not in no_stop_words_sentiment])
  # do the lemmatization
  lemmatizer=WordNetLemmatizer()
  text=" ".join([lemmatizer.lemmatize(y) for y in text.split(" ")])
  return text


# Set your remote tracking URI
mlflow.set_tracking_uri("http://ec2-13-61-13-214.eu-north-1.compute.amazonaws.com:5000/")

@pytest.mark.parametrize("model_name, stage, vectorizer_path", [
    ("yt_chrome_plugin_model", "staging", "vectorizer.pkl"),  # Replace with your actual model name and vectorizer path
])
def test_model_with_vectorizer(model_name, stage, vectorizer_path):
    client = MlflowClient()

    # Get the latest version in the specified stage
    latest_version_info = client.get_latest_versions(model_name, stages=[stage])
    latest_version = latest_version_info[0].version if latest_version_info else None

    assert latest_version is not None, f"No model found in the '{stage}' stage for '{model_name}'"

    try:
        # Load the latest version of the model
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        model_signature = model.metadata.get_input_schema()
        print("Model expected schema:", model_signature)


        # Load the vectorizer
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        vectorizer_features = vectorizer.get_feature_names_out()
        print("Vectorizer feature names:", vectorizer_features)

        # Create a dummy input for the model
        input_text = "weekend work youtube video"
        input_text = processing(input_text)
        input_data = vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=vectorizer.get_feature_names_out())  # <-- Use correct feature names

        # Predict using the model
        prediction = model.predict(input_df)

        # Verify the input shape matches the vectorizer's feature output
        assert input_df.shape[1] == len(vectorizer.get_feature_names_out()), "Input feature count mismatch"

        # Verify the output shape (assuming binary classification with a single output)
        assert len(prediction) == input_df.shape[0], "Output row count mismatch"

        print(f"Model '{model_name}' version {latest_version} successfully processed the dummy input.")

    except Exception as e:
        pytest.fail(f"Model test failed with error: {e}")

   
