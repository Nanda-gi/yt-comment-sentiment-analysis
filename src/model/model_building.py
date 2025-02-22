from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import pickle
import yaml
n_estimators=yaml.safe_load(open('params.yaml','r'))['model_building']['n_estimators']
learning_rate=yaml.safe_load(open('params.yaml','r'))['model_building']['learning_rate']
max_depth=yaml.safe_load(open('params.yaml','r'))['model_building']['max_depth']
train_data=pd.read_csv('C:\\Users\\HP\\Desktop\\Youtube comment sentiment analysis\\data\\processed\\reddit_train.csv')
x_train=train_data.iloc[:,0:-1]
y_train=train_data.iloc[:,-1]
best_model= LGBMClassifier( objective ='multiclass',
        num_class = 3,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        metric='multi_logloss',
        is_unbalance=True,
        class_weight='balanced'
        )
best_model.fit(x_train,y_train)
pickle.dump(best_model,open('best_model.pkl','wb'))