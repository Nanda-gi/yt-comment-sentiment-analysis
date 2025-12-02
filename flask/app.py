import mlflow
from flask import Flask,request,jsonify
from flask_cors import CORS
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import pickle

app=Flask(__name__)
CORS(app)

def load_model_vectorizer(model_name,model_version,vectorizer_path):
    try:
        mlflow.set_tracking_uri("http://ec2-13-61-13-214.eu-north-1.compute.amazonaws.com:5000/")
        model_uri=f'models:/{model_name}/{model_version}'
        model=mlflow.pyfunc.load_model(model_uri)
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        return model,vectorizer
    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")
        raise e

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


model,vectorizer=load_model_vectorizer('mymodel',"1",'vectorizer.pkl')
print("model,vectorizer loaded successfully")

@app.route('/')
def home():
    return "Welcome to our flask api"


@app.route("/predict",methods=['POST'])
def predict():
    data=request.json
    comments=data.get('comments')
    if not comments:
        return jsonify({'error':'no_comments provided'})
    preprocessed_comment=[processing(comment) for comment in comments]
    vectorized_comment=vectorizer.transform(preprocessed_comment)
    vectorized_df = pd.DataFrame(vectorized_comment.toarray(), columns=vectorizer.get_feature_names_out())
    expected_columns = model.metadata.get_input_schema().input_names()
    vectorized_df = vectorized_df.reindex(columns=expected_columns, fill_value=0)
    vectorized_df = vectorized_df.astype(float)
    try:
        predict=model.predict(vectorized_df).tolist()
        response=[{'comment':comment,"sentiment":sentiment} for comment,sentiment in zip(preprocessed_comment,predict)]
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)