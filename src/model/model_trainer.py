import sys, os
from typing import Tuple
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, roc_auc_score)
import pandas as pd
import numpy as np
import yaml
import dill
import json
import mlflow
import dagshub
from dotenv import load_dotenv

from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
)

from src.logger import logging
from src.constants import *

load_dotenv()
# -------------------------------------------------------------------------------------
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{DAGSHUB_URL}/{REPO_OWNER}/{REPO_NAME}.mlflow')
#dagshub.init(repo_owner='pradkris07', repo_name='Diabetes-Health-Indicators', mlflow=True)
# ----------------------------------------------------------------------------------

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise
        
def create_final_model_yaml(mod_yaml: dict, params: dict,  score):

    try:
        logging.info('Checking the model.yaml and updating')
        with open(MODEL_FINAL_CONFIG_PATH, "r") as f:
            data = yaml.safe_load(f)

        # If file doesn't have a list yet, create one
        if not isinstance(data, list):
            data = [data]
            data.append({
                    "estimator": mod_yaml['estimator'],
                    "n_estimators": params['n_estimators'],
                    "max_depth": params['max_depth'],
                    "learning_rate": params['learning_rate'],
                    "accuracy": score
                })
        else:
            print(data)
            existing_estimators = {entry.get("estimator") for entry in data}

            # Append new dictionary
            if existing_estimators and mod_yaml["estimator"] in existing_estimators:
                curr_accuracy = [est.get("accuracy") for est in data if est.get("estimator") == mod_yaml["estimator"]]
                print(curr_accuracy)
                if curr_accuracy[0] >= score:
                    logging.info(f"Estimator '{mod_yaml['estimator']}' already exists. Not adding.")
                    return False  # not appended
                else:
                    data = [existing for existing in data if existing.get("estimator") != mod_yaml["estimator"]]
                    data.append({
                        "estimator": mod_yaml['estimator'],
                        "n_estimators": params['n_estimators'],
                        "max_depth": params['max_depth'],
                        "learning_rate": params['learning_rate'],
                        "accuracy": score
                    })
            else:
                data.append({
                    "estimator": mod_yaml['estimator'],
                    "n_estimators": params['n_estimators'],
                    "max_depth": params['max_depth'],
                    "learning_rate": params['learning_rate'],
                    "accuracy": score
                })

        # Save back
        with open(MODEL_FINAL_CONFIG_PATH, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
    except Exception as e:
        logging.error('Unexpected error while saving yaml file: %s', e)
        raise

def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method")

    try:
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")

    except Exception as e:
        logging.error('Failed to save the model : %s', e)
        print(f"Error: {e}")
        
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        logging.error('Failed to load the data array : %s', e)
        print(f"Error: {e}")

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise

def get_model_and_report_gdb(model_yaml:  dict, train: np.array, test: np.array):
    try:
        logging.info("Training Gradient Boosting Classifier with specified parameters")
        
        # Splitting the train and test data into features and target variables
        x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
        logging.info("train-test split done")
        
        n_estimators = model_yaml['hyperparameter']['n_estimators']
        max_depth = model_yaml['hyperparameter']['max_depth']
        learning_rate = model_yaml['hyperparameter']['learning_rate']
        
        param_distributions = {'n_estimators': n_estimators,
                               'max_depth': max_depth,
                               'learning_rate': learning_rate
                               }
        print( n_estimators, max_depth, learning_rate)
        # Randomized Search instead of Grid Search
        clf = GradientBoostingClassifier(random_state=RANDOM_STATE)
        random_search = RandomizedSearchCV( clf, param_distributions=param_distributions,
                   n_iter=N_ITERATION,               # Number of random parameter combinations to try
                   cv=CV_VALUE,                    # 3-fold CV (faster than 5-fold)
                   scoring='accuracy',
                   random_state=RANDOM_STATE
                   )
        
        # Fit the model
        random_search.fit(x_train, y_train)

        # Get the best cross-validation score
        best_score = random_search.best_score_

        # Get the best combination of hyperparameters
        best_params = random_search.best_params_
        
        print('Best Accuracy:', best_score)
        print('Best Params:', best_params)
        
        # Predict the labels for the test data after fit using x_train
        model = GradientBoostingClassifier(n_estimators= best_params['n_estimators']
                                          ,max_depth= best_params['max_depth']
                                          ,learning_rate= best_params['learning_rate']
                                          ,random_state=RANDOM_STATE)
        model.fit(x_train, y_train)
        
        y_pred = model.predict(x_test)
        accuracy_test = accuracy_score(y_test , y_pred)

        # Compute additional metrics like Precision , recall , F!-score , and ROC AUC
        precision = precision_score(y_test, y_pred , zero_division = 0)
        recall = recall_score(y_test , y_pred , zero_division = 0)
        f1 = f1_score(y_test , y_pred, zero_division = 0)
        roc_auc = roc_auc_score(y_test , model.predict_proba(x_test)[:, 1])

        # Create a metrics DataFrame
        df_metrics = {
            'Test Accuracy': accuracy_test,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC AUC': roc_auc
        }
        
        # Create a estimator Dataframe
        df_estimator = pd.DataFrame({
            'Estimator': [model_yaml['estimator']],
            'n_estimators': [best_params['n_estimators']],
            'max_depth': [best_params['max_depth']],
            'learning_rate': [best_params['learning_rate']]
        })
        
        create_final_model_yaml(model_yaml, best_params, round(accuracy_test,5))
        print(df_estimator)
        return model, df_metrics
    except Exception as e:
        logging.error('Failed to complete getting model info and report process: %s', e)
        print(f"Error: {e}")
        raise

def main():
    logging.info("Entered initiate_model_trainer method")
    """
    Description :   This function initiates the model training steps
    
    Output      :   Returns model trainer artifact
    On Failure  :   Write an exception log and then raise an exception
    """
    mlflow.set_experiment(MLFLOW_EXP_NAME)
    with mlflow.start_run() as run:  # Start an MLflow run
        try:
            print("------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component")
            # Load transformed train and test data
            train_arr = load_numpy_array_data(file_path=os.path.join(DATA_PATH, PROCESSED_OBJECT_DIR,TRAIN_ARRAYNAME))
            test_arr = load_numpy_array_data(file_path=os.path.join(DATA_PATH, PROCESSED_OBJECT_DIR,TEST_ARRAYNAME))
            logging.info("train-test data loaded")

            # getting the models from model.yaml
            model_config = load_params(params_path=MODEL_CONFIG_PATH)
            for each in model_config['models']:
                model_name = each['name']
                if model_name == GDB_ESTIMATOR:
                    print('Values for GDB are ',each)
                    gdb_model,gdb_metrics = get_model_and_report_gdb(model_yaml=each, train=train_arr, test=test_arr)
                    # Check if the model's accuracy meets the expected threshold
                    if accuracy_score(test_arr[:, -1], gdb_model.predict(test_arr[:, :-1])) < EXPECTED_ACCURACY:
                        logging.info("No model found with score above the base score")
                        raise Exception("No model found with score above the base score")

                    # Save the final model object that includes both preprocessing and the trained model
                    logging.info("Saving new model as performace is better than previous one.")
                    #my_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model)
                    save_object(GDB_MODEL_PATH, gdb_model)
                    # save the metrics of the run
                    
                    save_metrics(gdb_metrics, GDB_METRICS_PATH)
                    # Log metrics to MLflow
                    for metric_name, metric_value in gdb_metrics.items():
                        mlflow.log_metric(metric_name, metric_value)
                    
                    # Log model parameters to MLflow
                    if hasattr(gdb_model, 'get_params'):
                        params = gdb_model.get_params()
                        print("Value of params are ",params)
                        print("Value of params items are ",params.items())
                        for param_name, param_value in params.items():
                            mlflow.log_param(param_name, param_value)  

                    # Log model to MLflow
                    mlflow.sklearn.log_model(gdb_model, "model")
            
                    # Save model info
                    save_model_info(run.info.run_id, "model", GDB_INFO_PATH)
            
                    # Log the metrics file to MLflow
                    mlflow.log_artifact(GDB_METRICS_PATH)                            
                   
            # Train model and get metrics
            #get_model_object_and_report(train=train_arr, test=test_arr)
            logging.info("Model object and artifact loaded.")
            
            # Load preprocessing object
            #preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            #logging.info("Preprocessing obj loaded.")

            
            # saving the model to a new path
            #if os.path.exists(self.model_pusher_config.model_final_dir) and os.path.isfile(self.model_pusher_config.model_final_dir):
            #    logging.info("Prod file existing")
            #else:
            #    save_object(self.model_pusher_config.model_final_dir, my_model)
            #    logging.info("Saved final model object that includes both preprocessing and the trained model")

            # Create and return the ModelTrainerArtifact
            #model_trainer_artifact = ModelTrainerArtifact(
            #    trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            #    metric_artifact=metric_artifact,
            #)
            #logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        
        except Exception as e:
            logging.error('Failed to complete the model training process: %s', e)
            print(f"Error: {e}")
            
if __name__ == '__main__':
    main()
