# register model

import json
import mlflow
import yaml
import logging
from src.logger import logging
import os
import dagshub

from dotenv import load_dotenv
from src.constants import * 

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")



# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
load_dotenv()

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{DAGSHUB_URL}/{REPO_OWNER}/{REPO_NAME}.mlflow')
# -------------------------------------------------------------------------------------


# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/vikashdas770/YT-Capstone-Project.mlflow')
# dagshub.init(repo_owner='vikashdas770', repo_name='YT-Capstone-Project', mlflow=True)
# -------------------------------------------------------------------------------------

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


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition the model to "Staging" stage
        #client = mlflow.tracking.MlflowClient()
        #client.transition_model_version_stage(
        #    name=model_name,
        #    version=model_version.version,
        #    stage="Production"
        #)
        
        logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_config = load_params(params_path=MODEL_CONFIG_PATH)
        for each in model_config['models']:
            model_name = each['name']
            if model_name == GDB_ESTIMATOR:
                model_info_path = GDB_INFO_PATH
                model_info = load_model_info(model_info_path)
                print(model_info)
                model_name = GDB_MODEL_PATH.split('\\')[1].split('.')[0]
                register_model(model_name, model_info)
            elif model_name == RFC_ESTIMATOR:
                model_info_path = RFC_INFO_PATH
                model_info = load_model_info(model_info_path)
                model_name = RFC_MODEL_PATH.split('\\')[1].split('.')[0]
                register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()