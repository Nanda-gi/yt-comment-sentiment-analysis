import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
import pickle
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

mlflow.set_tracking_uri("http://ec2-13-61-13-214.eu-north-1.compute.amazonaws.com:5000/")



client = MlflowClient()
model_name="yt_chrome_plugin_model"
stage="staging"
vectorizer_path="vectorizer.pkl"
    # Get the latest version in the specified stage
latest_version_info = client.get_latest_versions(model_name, stages=[stage])
latest_version = latest_version_info[0].version if latest_version_info else None

    

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

   
   
