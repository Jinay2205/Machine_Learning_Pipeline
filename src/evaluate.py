import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from urllib.parse import urlparse

## Setting up the environment variables

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/jinayshah2205/MachineLearningPipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "jinayshah2205"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "a2a803435450f3bd83a1ab45707e961c2c93124b"

params = yaml.safe_load(open("params.yaml"))['train']

def evaluate(data_path,model_path) :
    data = pd.read_csv(data_path)
    X = data.drop('Outcome',axis=1)
    y = data['Outcome']

    mlflow.set_tracking_uri('https://dagshub.com/jinayshah2205/MachineLearningPipeline.mlflow')

    ## Load the pickle file 
    model = pickle.load(open(model_path,mode="rb"))
    predictions = model.predict(X)
    accuracy = accuracy_score(y,predictions)

    ## Log metrics
    mlflow.log_metric("accuracy" , accuracy)
    print(f'Model accuracy : {accuracy}')

if __name__ == "__main__" :
    evaluate(params['data'],params['model'])