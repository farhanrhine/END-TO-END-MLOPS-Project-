from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import xgboost as xgb

import pandas as pd
from config.paths_config import *
from src.logger import get_logger
from src.custom_exception import CustomException
import matplotlib.pyplot as plt
import time
import sys

from torch.utils.tensorboard import SummaryWriter # used for logging metrics and visualizations inside tensorboard

logger = get_logger(__name__)


class ModelSelection:

    def __init__(self,data_path):
        self.data_path = data_path # engineered data path
        run_id = time.strftime("%Y%m%d-%H%M%S") # to create a unique run id based on current time for every run
        # Initialize TensorBoard writer with a unique log directory
        self.writer = SummaryWriter(log_dir=f"tensorboard_logs/run_{run_id}")

# Initialize all models
        self.models= {
                    'Logistic Regression': LogisticRegression(),
                    'Random Forest': RandomForestClassifier(n_estimators=50, n_jobs=-1),
                    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50),
                    'AdaBoost': AdaBoostClassifier(n_estimators=50),
                    'Support Vector Classifier': SVC(),
                    'K-Nearest Neighbors': KNeighborsClassifier(),
                    'Naive Bayes': GaussianNB(),
                    'Decision Tree': DecisionTreeClassifier(),
                    'LightGBM': lgb.LGBMClassifier(),
                    'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss')
                }
        
        self.results = {}
    
    def load_data(self):
        try:
            logger.info("Loading CSV file")
            df = pd.read_csv(self.data_path)   #100% data = df
            df_sample = df.sample(frac=0.05, random_state=42)   #5% data 0.05 == df_sample, 10% data = 0.1

            X = df_sample.drop(columns='satisfaction')   # input Features
            y = df_sample['satisfaction']   # Target variable

            logger.info("Data loaded and sampled sucesfully")
            return X,y
        
        except Exception as e:
            raise CustomException("Error while loading data " , sys)
        
    def split_data(self , X , y):
        try:
            logger.info("Splitting data")
            return train_test_split(X, y, test_size=0.2, random_state=42) # 80% data for training and 20% for testing with return 4 things  X_train, X_test, y_train, y_test
        
        except Exception as e:
            raise CustomException("Error while splitting data " , sys)
        
    def log_confusion_matrix(self , y_true , y_pred , step , model_name):
        cm = confusion_matrix(y_true,y_pred)

        fig , ax = plt.subplots(figsize=(5,5))
        ax.matshow(cm , cmap=plt.cm.Blues , alpha=0.7)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j,y=i,s=cm[i,j] , va="center" ,ha="center")
        
        plt.xlabel("Predicted Labels")
        plt.ylabel("True/Actual Labels")
        plt.title(f"Confusion matrix for {model_name}")

        self.writer.add_figure(f"Confusion_Matrix/{model_name}" , fig , global_step=step)
        plt.close(fig)

        
    def train_and_evaluate(self,X_train,X_test,y_train,y_test):
        try:
            logger.info("Training and evalation started")
            for idx , (name,model) in enumerate(self.models.items()):
                model.fit(X_train,y_train)
                y_pred=model.predict(X_test)

                accuracy = accuracy_score(y_test,y_pred)
                precision = precision_score(y_test,y_pred,average="weighted",zero_division=0)
                recall = recall_score(y_test,y_pred,average="weighted",zero_division=0)
                f1 = f1_score(y_test,y_pred,average="weighted",zero_division=0)

                self.results[name] = {
                    'accuracy' : accuracy,
                    'precision' : precision,
                    'recall' : recall,
                    "f1_score" : f1
                }

                logger.info(f"{name} trained sucesfully"
                            f"Metrics : Accuracy : {accuracy} , Precision : {precision} , Recall : {recall} , F1 Score : {f1}")
                # Log metrics to TensorBoard
                self.writer.add_scalar(f'Accuracy/{name}' , accuracy , idx) # idx = index of the model in the models dictionary
                self.writer.add_scalar(f'Precision/{name}' , precision , idx)
                self.writer.add_scalar(f'Recall/{name}' , recall , idx)
                self.writer.add_scalar(f'F1_score/{name}' , f1 , idx)

                self.writer.add_text('Model Details ' , f"Name : {name}"  f"Metrics : Accuracy : {accuracy} , Precision : {precision} , Recall : {recall} , F1 Score : {f1}" )

                self.log_confusion_matrix(y_test , y_pred , idx , name)

            self.writer.close()
        
        except Exception as e:
            raise CustomException("Error while training and evaluation " , sys)

    # run pipeline
    # This function will be called the run method of the entire pipeline
    def run(self):
        try:
            logger.info("Starting ModelSelection pipeline")
            X,y = self.load_data() 

            X_train , X_test , y_train , y_test =self.split_data(X,y)   

            self.train_and_evaluate(X_train , X_test , y_train , y_test)

            logger.info("ModelSelection pipeline completed successfully")

        except Exception as e:
            logger.error("Error in the pipeline")
            raise CustomException("Error in the pipeline " , sys)

#  This is the entry point of the script
#  It will create an instance of the ModelSelection class and call the run method        
if __name__ =="__main__":
    model_selection = ModelSelection(ENGINEERED_DATA_PATH)
    model_selection.run()
    







        

