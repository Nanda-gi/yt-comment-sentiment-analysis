import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
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

def read_data(url):
    df = pd.read_csv(url)
    return df
def save_data(datapath,train_data,test_data):
    os.makedirs(datapath, exist_ok=True)
    train_data.to_csv(os.path.join(datapath,"reddit_train.csv"),index=False)
    test_data.to_csv(os.path.join(datapath,"reddit_test.csv"),index=False)

def main():
   train_data=read_data('data\\raw\\reddit_train.csv')
   test_data=read_data('data\\raw\\reddit_test.csv')
   train_data['clean_comment']=train_data['clean_comment'].apply(processing)
   test_data['clean_comment']=test_data['clean_comment'].apply(processing)
   save_data('data\\interim',train_data,test_data)

if __name__=='__main__':
   main()