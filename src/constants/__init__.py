import os
from datetime import datetime

# Constants for log configuration
LOG_DIR = 'logs'
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3  # Number of backup log files to keep

# Constants for run configuration
RANDOM_STATE = 42

# Constants for data ingestion
DATA_URL = "https://raw.githubusercontent.com/pradkris07/Datasets/refs/heads/main/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
DATA_PATH = './data'
EXTERNAL_FOLDER_NAME = 'external'
DATA_FILENAME = 'diabetic.csv'
RAW_FOLDER_NAME = 'raw'
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"