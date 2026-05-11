import os
import sys

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass()
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'trained_model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("Split Training and Testing Data")
            X_train,y_train,X_test,y_test = (train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1])

            #model dictionary

            models = {
                "RandomForest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Nearest Neighbors": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoost": CatBoostRegressor(),
                "AdaBoost Classifier": AdaBoostRegressor()
            }

            model_report:dict = evaluate_models(X=X_train,y=y_train,models=models,preprocessor_path=preprocessor_path)

        except Exception as e:
            raise CustomException(str(e))





