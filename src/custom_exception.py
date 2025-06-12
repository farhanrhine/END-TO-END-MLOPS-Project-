import traceback # extract error
import sys # for interacting current python environments

# i want inbuild and custom exception both
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message) # its inherite this from inbuild exception class
        self.error_message = self.get_detailed_error_message(error_message,error_detail)


    @staticmethod # it help us to used any method  without creating objects for CustomException class
    def get_detailed_error_message(error_message, error_detail:sys):
        _,_,exception_traceback = error_detail.exc_info() # i skip type and value only take traceback details
        file_name = exception_traceback.tb_frame.f_code.co_filename
        line_number = exception_traceback.tb_lineno
        return f"Error in {file_name}, line {line_number} : {error_message}"
    
    # magic method  -- > its give detail message when i used str(CustomException)
    def __str__(self):
        return self.error_message
    
str(CustomException)