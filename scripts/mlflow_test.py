import mlflow
import random
mlflow.set_tracking_uri('http://ec2-51-20-129-94.eu-north-1.compute.amazonaws.com:5000/')
with mlflow.start_run() as run:
    
    mlflow.log_param("param1",random.randint(1,100))
    mlflow.log_param("param2",random.random())

    print("logged parameter")