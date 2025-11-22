# data ingestion
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import os
from sklearn.model_selection import train_test_split
import yaml
import logging
from src.logger import *
from src.constants import *


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

def load_data(data_url: str, data_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        external_data_path = os.path.join(data_path, EXTERNAL_FOLDER_NAME)
        if os.path.isfile(os.path.join(external_data_path, DATA_FILENAME)):
            df = pd.read_csv(os.path.join(external_data_path, DATA_FILENAME))
            logging.info('Data loaded from %s', os.path.join(external_data_path, DATA_FILENAME))
        else:
            df = pd.read_csv(data_url)
            os.makedirs(external_data_path, exist_ok=True)
            df.to_csv(os.path.join(external_data_path, DATA_FILENAME), index=False)
            logging.info('Datafile saved to %s', external_data_path)
        logging.info('Data loaded')
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to retrieve the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, RAW_FOLDER_NAME)
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, TRAIN_FILENAME), index=False)
        test_data.to_csv(os.path.join(raw_data_path, TEST_FILENAME), index=False)
        logging.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        # test_size = 0.2
        
        df = load_data(data_url=DATA_URL, data_path=DATA_PATH)
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=RANDOM_STATE)
        save_data(train_data, test_data, data_path=DATA_PATH)
    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()