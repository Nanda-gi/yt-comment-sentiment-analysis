import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml

def load_params(params_path):
    test_size=yaml.safe_load(open(params_path,'r'))['data_ingestion']['test_size']
    return test_size

def read_data(url):
    df = pd.read_csv(url)
    return df
def process_data(df):
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df = df[~(df['clean_comment'].str.strip() == '')]
    df['clean_comment']=df['clean_comment'].str.replace("\n"," ",regex=True)
    df=df[~(df['clean_comment'].str.strip()=="")]
    return df
def save_data(datapath,train_data,test_data):
    
    os.makedirs(datapath)
    train_data.to_csv(os.path.join(datapath,"reddit_train.csv"),index=False)
    test_data.to_csv(os.path.join(datapath,"reddit_test.csv"),index=False)

def main():
    test_size=load_params('params.yaml')
    df=read_data('https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')
    df=process_data(df)
    train_data,test_data=train_test_split(df,test_size=test_size,random_state=42)
    datapath=os.path.join("data","raw")
    save_data(datapath,train_data,test_data)

if __name__=="__main__":
    main()
