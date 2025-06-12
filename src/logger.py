import logging # py inbuild
import os # its store all logs
from datetime import datetime

#1. create basic configure
Logs_Dir = "logs"
os.makedirs(Logs_Dir, exist_ok=True) # its check, if not then create logs_Dir folder

# create new file inside Logs_Dir folder and keep each date logs details seperate
Log_File = os.path.join(Logs_Dir, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")
# store all above configuration
logging.basicConfig(
    filename=Log_File,
    #format = "%(asctime)s - %(levelname)s -%(message)s",
    format = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    level= logging.INFO
    )
'''acstime = date & time
       levelname = keep {
                            INFO = basic message 
                            warning = give warning 
                            Error = give error 
                            Debug = 
                            Critical
                        }
        level = include only INFO, Warning, Info because only these are => INFO (debug and critical NOT include)
       
'''


def get_logger(name):
    logger = logging.getLogger(name) # its create logger 
    logger.setLevel(logging.INFO)
    return logger


