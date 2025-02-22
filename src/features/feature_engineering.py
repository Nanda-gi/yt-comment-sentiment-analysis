import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml
import pickle

n_gram=tuple(yaml.safe_load(open('params.yaml','r'))['feature_engineering']['n_gram'])
max_features=yaml.safe_load(open('params.yaml','r'))['feature_engineering']['max_features']
   

train_data=pd.read_csv('C:\\Users\\HP\\Desktop\\Youtube comment sentiment analysis\\data\\interim\\reddit_train.csv')
test_data=pd.read_csv('C:\\Users\\HP\\Desktop\\Youtube comment sentiment analysis\\data\\interim\\reddit_test.csv')
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)
x_train=train_data['clean_comment'].values
x_test=test_data['clean_comment'].values
y_train=train_data['category'].values
y_test=test_data['category'].values

vectorizer=TfidfVectorizer(ngram_range=n_gram,max_features=max_features)

x_train_vec=vectorizer.fit_transform(x_train)
x_test_vec=vectorizer.transform(x_test)

train_df=pd.DataFrame(x_train_vec.toarray())
train_df['label']=y_train

test_df=pd.DataFrame(x_test_vec.toarray())
test_df['label']=y_test

datapath='C:\\Users\\HP\\Desktop\\Youtube comment sentiment analysis\\data\\processed'
os.makedirs(datapath,exist_ok=True)
train_df.to_csv(os.path.join(datapath,"reddit_train.csv"),index=False)
test_df.to_csv(os.path.join(datapath,"reddit_test.csv"),index=False)

pickle.dump(vectorizer,open('vectorizer.pkl','wb'))
