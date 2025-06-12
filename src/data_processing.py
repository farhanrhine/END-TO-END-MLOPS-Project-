import pandas as pd
from config.paths_config import *
from src.logger import get_logger
from src.custom_exception import CustomException
import sys

logger = get_logger(__name__)

class DataProcessor:

    def __init__(self):
        self.train_path = TRAIN_DATA_PATH
        self.processed_data_path = PROCESSED_DATA_PATH

    def load_data(self):
        try:
            logger.info("Data Processinng Started")
            df = pd.read_csv(self.train_path)
            logger.info(f"Data read succesfull : Data shape : {df.shape}")
            return df
        except Exception as e:
            logger.error("Problem while Loading Data")
            raise CustomException("Error while loading data : ",sys)
        
    def drop_unnecessary_columns(self, df, columns):
        try:
            logger.info(f"Dropping Unnecesary Columns :  {columns}")
            df = df.drop(columns = columns, axis=1)
            logger.info(f"Columns dropped Sucesfully : Shape = {df.shape}")
            return df
        except Exception as e:
            logger.error("Problem while dropping columns")
            raise CustomException("Error while sropping columns : ",sys)
    # Handle outliers using IQR method
    def handle_outliers(self, df , columns):
        try:
            logger.info(f"Handling outliers : Columns = {columns}")
            for column in columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            
            logger.info(f"Outliers handled  Sucesffuully : {df.shape}")
            return df
        
        except Exception as e:
            logger.error("Problem while Outlier handling")
            raise CustomException("Error while outlier handling : ",sys)
    # # Handle null values by filling with median
    def handle_null_values(self,df,columns):
        try:
            logger.info("Handling null values")
            df[columns] = df[columns].fillna(df[columns].median())
            logger.info(f"Missing values handled sucessfully : Shape == {df.shape}")
            return df
        
        except Exception as e:
            logger.error("Problem while null values handling")
            raise CustomException("Error while null values handling : ",sys)
    # Save the processed data to a CSV file
    def save_data(self,df):
        try:
            os.makedirs(PROCESSED_DIR , exist_ok=True)
            df.to_csv(self.processed_data_path , index=False)
            logger.info("Processed data saved sucesfully")
        
        except Exception as e:
            logger.error("Problem while saving data")
            raise CustomException("Error while saving data : ",sys)
        
    # Main function to run the data processing pipeline
    def run(self):
        try:
            logger.info("Starting the pipeline of Data Procesing")

            df = self.load_data()
            df = self.drop_unnecessary_columns(df , ["MyUnknownColumn","id"])
            columns_to_handel= ['Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes', 'Checkin service']
            df = self.handle_outliers(df , columns_to_handel)

            df = self.handle_null_values(df , 'Arrival Delay in Minutes')

            self.save_data(df) # Save the processed data

            logger.info("Data Processing Pipeline CCompleted Successfully")

        except CustomException as ce:
            logger.error(f"Error ocuured in Data Processing Pipleine : {str(ce)}")

# this will run the data processing pipeline when this script is executed
if __name__=="__main__":
    processor = DataProcessor()
    processor.run()




        
        
        
    