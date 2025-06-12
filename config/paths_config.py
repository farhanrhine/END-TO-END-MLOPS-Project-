import os

ARTIFACTS_DIR = "./artifacts" # Directory to store artifacts
RAW_DATA_PATH = os.path.join(ARTIFACTS_DIR,"raw","data.csv")# Path to the raw data file
INGESTED_DATA_DIR = os.path.join(ARTIFACTS_DIR,"ingested_data")
TRAIN_DATA_PATH = os.path.join(INGESTED_DATA_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(INGESTED_DATA_DIR,"test.csv")

PROCESSED_DIR = os.path.join(ARTIFACTS_DIR, "processed_data")# Directory to store processed data
PROCESSED_DATA_PATH = os.path.join(ARTIFACTS_DIR, "processed_data", "processed_train.csv")# Path to the processed train data file

ENGINEERED_DIR = os.path.join(ARTIFACTS_DIR, "engineered_data")# Directory to store feature engineered data
ENGINEERED_DATA_PATH = os.path.join(ARTIFACTS_DIR, "engineered_data", "final_df.csv")# Path to the feature engineered data file

PARAMS_PATH = os.path.join("./config", "params.json")# Path to the hyperparameters file
MODEL_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "models", "trained_model.pkl")# Path to the saved model file