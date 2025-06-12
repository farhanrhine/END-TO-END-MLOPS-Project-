import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.helpers import *
import sys

logger = get_logger(__name__)
# Directory to store feature engineered data
class FeatureEngineer:
    def __init__(self):
        self.data_path = PROCESSED_DATA_PATH
        self.df = None
        self.label_mapping={}
# load data 
    def load_data(self):
        try:
            logger.info("Loading data")
            self.df = pd.read_csv(self.data_path)
            logger.info("Data loaded sucesffuly")
        except Exception as e:
            logger.error(f"Error while loading data {e}")
            raise CustomException("Error while loading data" , sys)
        
    def feature_construction(self):
        try:
            logger.info("Starting Feature Construction")
            self.df['Total Delay'] = self.df['Departure Delay in Minutes'] + self.df['Arrival Delay in Minutes']
            self.df['Delay Ratio'] = self.df['Total Delay'] / (self.df['Flight Distance'] + 1)
            logger.info("Feature Construction done sucesfully")

        except Exception as e:
            logger.error(f"Error while feature construction {e}")
            raise CustomException("Error while faeture construction" , sys)
    
    def bin_age(self):
        try:
            logger.info("Starting Binning of Age Column")
            self.df['Age Group'] = pd.cut(self.df['Age'], bins=[0, 18, 30, 50, 100], labels=['Child', 'Youngster', 'Adult', 'Senior'])
            logger.info("Binning of Age Column Sucesfull")
        except Exception as e:
            logger.error(f"Error while binning {e}")
            raise CustomException("Error while binning" , sys)
        
    def label_encoding(self):
        try:
            columns_to_encode = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction', 'Age Group']
            logger.info(f"Performing label encoding on {columns_to_encode}")

            self.df , self.label_mapping=label_encode(self.df , columns_to_encode)

            for col,mapping in self.label_mapping.items():
                logger.info(f"Mapping for {col} : {mapping}")
            logger.info("Label Encoding successful")

        except Exception as e:
            logger.error(f"Error while label encoding {e}")
            raise CustomException("Error while label encoding" , sys)
    

    def feature_selection(self):
        try:
            logger.info("Starting Feature Selection")
            X = self.df.drop(columns='satisfaction')
            y = self.df['satisfaction']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            mutual_info = mutual_info_classif(X_train, y_train, discrete_features=True)

            mutual_info_df = pd.DataFrame({
                                    'Feature': X.columns,
                                    'Mutual Information': mutual_info
                                    }).sort_values(by='Mutual Information', ascending=False)
            
            logger.info(f"Mutual Information Table is : \n{mutual_info_df}")

            top_features = mutual_info_df.head(12)['Feature'].tolist()

            self.df = self.df[top_features + ['satisfaction']]
            logger.info(f"Top features : {top_features}")
            logger.info("Feature Selection Sucesfull")

        except Exception as e:
            logger.error(f"Error while feature slection {e}")
            raise CustomException("Error while feature selection" , sys)
    
    def save_data(self):
        try:
            logger.info("Saving your data ...")
            os.makedirs(ENGINEERED_DIR,exist_ok=True)
            self.df.to_csv(ENGINEERED_DATA_PATH,index=False)
            logger.info(f"Data Saved succesfully at {ENGINEERED_DATA_PATH}")

        except Exception as e:
            logger.error(f"Error while saving data {e}")
            raise CustomException("Error while saving data" , sys)
    # Run the feature engineering pipeline
    def run(self):
        try:
            logger.info("Starting your feature engineering pipeline")
            self.load_data()
            self.feature_construction()
            self.bin_age()
            self.label_encoding()
            self.feature_selection()
            self.save_data()
            logger.info("Your Feature Engineering pipeline successfully done..")

        except Exception as e:
            logger.error(f"Error in Feature Engineering pipeline {e}")
            raise CustomException("Error in Feature Engineering pipeline" , sys)

        finally:
            logger.info("End of Feature Engineering pipeline")

# This script is the entry point for the feature engineering process.
# It initializes the FeatureEngineer class and runs the feature engineering pipeline.
if __name__ == "__main__":
    feature_engineer = FeatureEngineer()
    feature_engineer.run()
        


        
        

