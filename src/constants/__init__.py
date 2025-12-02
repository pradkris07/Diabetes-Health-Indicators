import os
from datetime import datetime

# Constants for log configuration
LOG_DIR = 'logs'
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3  # Number of backup log files to keep

# Constants for run configuration
RANDOM_STATE = 42
MLFLOW_EXP_NAME = "DVC-Test"
DAGSHUB_URL = "https://dagshub.com"
REPO_OWNER = "pradkris07"
REPO_NAME = "Diabetes-Health-Indicators"

# Constants for data ingestion
TEST_SIZE = 0.25
DATA_URL = "https://raw.githubusercontent.com/pradkris07/Datasets/refs/heads/main/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
DATA_PATH = './data'
EXTERNAL_FOLDER_NAME = 'external'
DATA_FILENAME = 'diabetic.csv'
RAW_FOLDER_NAME = 'raw'
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
SAMPLE_FILENAME= "sample.csv"
SAMPLE_SIZE = 25

# Constants for data validation
SCHEMA_FILE_PATH = os.path.join("src/config", "schema.yaml")
REPORT_FOLDER_NAME = 'report'
REPORT_FILENAME = 'report.yaml'
BIN_COL = 'binary_columns'
NUM_COL = 'numerical_columns'
COLUMNS = 'columns'
VALID_STATUS = 'validation_status'
MESSAGE = "message"

# Constants for data transformation
TARGET_COLUMN = 'Diabetes_binary'
QUANTILE_CONST = 1.5
TRANS_COL = 'outliers'
SAMPLE_STRATEGY = 'minority'
PROCESSED_OBJECT_DIR = 'processed'
PREPROCSSING_OBJECT_FILE_NAME = os.path.join("models","preprocessing.pkl")

# Constants for Model Trainer
TRAIN_ARRAYNAME = "train.npy"
TEST_ARRAYNAME = "test.npy"
MODEL_CONFIG_PATH = os.path.join("src/config", "model.yaml")
N_ITERATION = 5
CV_VALUE = 3
EXPECTED_ACCURACY = 0.9

# Model names
MODEL_FINAL_CONFIG_PATH = os.path.join("models", "final.yaml")
GDB_ESTIMATOR = 'Gradient Boosting'
GDB_MODEL_PATH = os.path.join("models", "GDB_model.pkl")
GDB_METRICS_PATH = os.path.join("reports", "GradientBoostingMetrics.json")
GDB_INFO_PATH = os.path.join("reports", "GradientBoostingExpInfo.json")
#
RFC_ESTIMATOR = 'Random Forest'
RFC_MODEL_PATH = os.path.join("models", "RFC_model.pkl")
RFC_METRICS_PATH = os.path.join("reports", "RandomForestMetrics.json")
RFC_INFO_PATH = os.path.join("reports", "RandomForestExpInfo.json")
#
KNN_ESTIMATOR = 'K Nearest Neighbour'
KNN_MODEL_PATH = os.path.join("models", "KNN_model.pkl")
KNN_METRICS_PATH = os.path.join("reports", "KNNeighboursMetrics.json")
KNN_INFO_PATH = os.path.join("reports", "KNNeighboursExpInfo.json")
#
SVC_ESTIMATOR = 'Support Vectors Classifier'
SVC_MODEL_PATH = os.path.join("models", "SVC_model.pkl")
SVC_METRICS_PATH = os.path.join("reports", "SVCMetrics.json")
SVC_INFO_PATH = os.path.join("reports", "SVCExpInfo.json")
#
GNB_ESTIMATOR = 'Gaussian NB Classifier'
GNB_MODEL_PATH = os.path.join("models", "GNB_model.pkl")
GNB_METRICS_PATH = os.path.join("reports", "GaussianNBMetrics.json")
GNB_INFO_PATH = os.path.join("reports", "GaussianNBExpInfo.json")