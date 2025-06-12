from src.logger import get_logger
from src.custom_exception import CustomException
import sys

logger = get_logger(__name__)

def divide_numbers(a,b):
    try:
        result = a/b
        logger.info("Divideding the numbers")
        return result
    except Exception as e:
        logger.error('Error Occured, WHERE? ')
        raise CustomException("Division by zero",sys)
    
    
if __name__ == "__main__": # below all code executed
    try:
        logger.info('Starting main program')
        divide_numbers(10,0)
    except CustomException as ce:
        logger.error(str(ce))
    finally:
        logger.info("end of program")





# logger.info('Logging is being done')
# logger.error("Error occured")
# logger.warning('Warning occured')