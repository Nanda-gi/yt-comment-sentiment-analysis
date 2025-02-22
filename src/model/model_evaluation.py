from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score,classification_report,confusion_matrix
import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle
import json
model=pickle.load(open('best_model.pkl','rb'))
test_data=pd.read_csv('C:\\Users\\HP\\Desktop\\Youtube comment sentiment analysis\\data\\processed\\reddit_test.csv')
x_test=test_data.iloc[:,0:-1]
y_test=test_data.iloc[:,-1]

y_pred=model.predict(x_test)
y_pred_proba=model.predict_proba(x_test)
accuracy=accuracy_score(y_test,y_pred)
report=classification_report(y_test,y_pred)
precision = precision_score(y_test, y_pred,average='weighted')
recall = recall_score(y_test, y_pred,average='weighted')


metrices={'accuracy':accuracy,'precision':precision,'recall':recall}

with open ("metrices.json",'w') as f:
    json.dump(metrices,f,indent=4)
