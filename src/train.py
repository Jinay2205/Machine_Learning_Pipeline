import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
import pickle
import yaml
import mlflow
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from mlflow.models import infer_signature
import os
from urllib.parse import urlparse

## Setting up the environment variables

os.environ['MLFLOW_TRACKING_URI'] = "MLFLOW TRACKING URI"
os.environ['MLFLOW_TRACKING_USERNAME'] = "MLFLOW USERNAME"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "MLFLOW TRACKING PASSWORD"

def hyperparameter_tuning(X_train,y_train,param_gird) :
    model = RandomForestClassifier() 
    grid_search = GridSearchCV(model,param_grid=param_gird,n_jobs=-1,cv=3,verbose=True)
    grid_search.fit(X_train,y_train)
    return grid_search

## Load the parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))['train']

def train(data_path,model_path,random_state,n_estimators,max_depth) :
    data = pd.read_csv(data_path)
    X = data.drop('Outcome',axis=1)
    y = data['Outcome']

    mlflow.set_tracking_uri("YOUR TRACKING URI")
    with mlflow.start_run() :
        ## Split data
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=random_state)
        signature = infer_signature(X_train,y_train)

        ## Define hyperparameter grid
        param_gird = {
            "n_estimators" : [100,200],
            "max_depth" : [5,10,None],
            "min_samples_split" : [2,5],
            "min_samples_leaf" : [1,2],
        }
        ## Perform hyperparameter tuning
        grid_search = hyperparameter_tuning(X_train,y_train,param_gird)

        ## Get best model
        best_model = grid_search.best_estimator_

        ## Predict and evaluate the model
        y_preds = best_model.predict(X_test)
        accuracy = accuracy_score(y_test,y_preds)
        print(f"Accuracy : {accuracy}")

        ## Log additional metrics
        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_param("best_n_estimators",grid_search.best_params_['n_estimators'])
        mlflow.log_param("best_max_depth",grid_search.best_params_['max_depth'])
        mlflow.log_param("best_min_samples_split",grid_search.best_params_['min_samples_split'])
        mlflow.log_param("best_min_samples_leaf",grid_search.best_params_['min_samples_leaf'])

        ## Log the confusion matrix and classification report
        cm = confusion_matrix(y_test,y_preds)
        cr = classification_report(y_preds,y_preds)

        mlflow.log_text(str(cm),"confusion_matrix.txt")
        mlflow.log_text(cr,"classification_report.txt")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file" : 
            mlflow.sklearn.log_model(best_model,"model",registered_model_name = "best model")

        else : 
            mlflow.sklearn.log_model(best_model,"model",signature=signature)  

        ## Create the directory to save model
        os.makedirs(os.path.dirname(model_path),exist_ok=True)

        filename = model_path
        pickle.dump(best_model,open(filename,'wb'))
        print(f'Model saved to {model_path}')

if __name__ == "__main__" :
    train(params['data'],params['model'],params['random_state'],params['n_estimators'],params['max_depth'])
