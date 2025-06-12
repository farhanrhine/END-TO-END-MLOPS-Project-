import os #for manage folder and files path
import sys
import csv # store extract data in .csv format
import mysql.connector # connect mysql and python
from mysql.connector import Error # # its manage mysql side error
from config.db_config import DB_CONFIG # to take all configuration from database
from src.logger import get_logger # its help to keep logger details
from src.custom_exception import CustomException


logger = get_logger(__name__)

class MySQLDataExtractor:

    def __init__(self,db_config):
        self.host = db_config["host"]
        self.user = db_config["user"]
        self.password = db_config["password"]
        self.database = db_config["database"]
        self.table_name = db_config['table_name']
        self.connection =None

        logger.info("Your Database configuration has been set up")

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host = self.host,
                user = self.user,
                password = self.password,
                database = self.database
            )
            if self.connection.is_connected():
                logger.info("Successfully connected to the Database")
                
        except Error as e:
            raise CustomException(f"Error while while connecting to the Database: {e}", error_detail=sys)
        
    def disconnect(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Disconnected to the Database")
        
    # main fuction
    def extract_to_csv(self, output_folder = "./artifacts/raw"):
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()
            
            # set a cursor for query
            cursor = self.connection.cursor()
            query = f"SELECT * FROM {self.table_name}"
            cursor.execute(query)

            # fetch each row and columns from db
            rows = cursor.fetchall()

            columns = [desc[0] for desc in cursor.description]

            logger.info("Data Fetched Successfully")

            # store data this automatically makedirectory where you said
            os.makedirs(output_folder, exist_ok=True)
            csv_file_path = os.path.join(output_folder,"data.csv") # create a file artifacts/raw/data.csv

            with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(columns)
                writer.writerows(rows)
                logger.info(f"Data Successfully saved to {csv_file_path}")

        except Error as e:
            raise CustomException(f"Error in extracting database due to SQL: {e}", error_detail=sys)
        
        except CustomException as ce: # check python error
            logger.error(str(ce))

        finally:
            if 'cursor' in locals():
                cursor.close()
            self.disconnect()

if __name__  == "__main__":
    try:
        extractor = MySQLDataExtractor(DB_CONFIG)
        extractor.extract_to_csv()
    except CustomException as ce:
        logger.error(str(ce))

