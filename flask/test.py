import mlflow.pyfunc

mlflow.set_tracking_uri("http://ec2-13-61-13-214.eu-north-1.compute.amazonaws.com:5000/")
model_name='mymodel'
model_version='43'
model_uri=f'models:/{model_name}/{model_version}'
print(mlflow.pyfunc.get_model_dependencies(model_uri=model_uri))
