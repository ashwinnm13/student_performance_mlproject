import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):
        '''

        This function is responsible for Data Transformation

        '''
        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            #Imputer - handle missing values
            #
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            logging.info("Numerical columns Encoding completed.....")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehot",OneHotEncoder()),
                    ("scaler",StandardScaler)
                  ]
            )

            logging.info("Categorical columns Encoding completed.....")

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info(f"read Training data completed.....")
            logging.info(f"Obtaining Preprocessor Object.....")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score","reading_score"]

            input_feature_train_data=train_data.drop(columns=[target_column_name],axis=1)
            target_feature_train_data=train_data[target_column_name]

            input_feature_test_data=test_data.drop(columns=[target_column_name],axis=1)
            target_feature_test_data=test_data[target_column_name]

            logging.info(f"Applying Preprocessor Object on training and testing dataframe.....")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_data)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_data)

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_data)
            ]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_data)]

            logging.info(f"Saving Preprocessing Data.....")

            save_object(

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:






