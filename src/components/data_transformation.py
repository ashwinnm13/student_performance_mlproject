import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''

        try:
            numerical_columns = ["writing_score", "reading_score"]

            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(str(e), sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Clean column names
            train_df.columns = train_df.columns.str.replace(" ", "_")
            train_df.columns = train_df.columns.str.replace("/", "_")

            test_df.columns = test_df.columns.str.replace(" ", "_")
            test_df.columns = test_df.columns.str.replace("/", "_")

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"

            # Split input and target features
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training and testing dataframe"
            )

            # Transform data
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )

            # Combine transformed input and target column
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(str(e), sys)


# STEP 1:
# Read training and testing CSV datasets using pandas

# STEP 2:
# Clean column names by replacing spaces and '/' with '_'
# Example:
# "math score" -> "math_score"
# "race/ethnicity" -> "race_ethnicity"

# STEP 3:
# Define numerical columns
# Numerical columns contain continuous numeric values

# STEP 4:
# Define categorical columns
# Categorical columns contain text/category values

# STEP 5:
# Create Numerical Pipeline
# This pipeline handles preprocessing for numerical features

# STEP 6:
# Numerical Pipeline -> Handle missing values
# Replace missing numerical values using median strategy

# STEP 7:
# Numerical Pipeline -> Apply Standard Scaling
# Scale numerical features using:
# z = (x - mean) / standard deviation

# STEP 8:
# Create Categorical Pipeline
# This pipeline handles preprocessing for categorical features

# STEP 9:
# Categorical Pipeline -> Handle missing values
# Replace missing categorical values using most frequent value

# STEP 10:
# Categorical Pipeline -> Apply One Hot Encoding
# Convert categorical text values into numerical binary vectors

# STEP 11:
# Categorical Pipeline -> Apply Scaling
# Scale encoded categorical features
# with_mean=False is used because OneHotEncoder creates sparse matrices

# STEP 12:
# Create ColumnTransformer
# Apply:
# - numerical pipeline to numerical columns
# - categorical pipeline to categorical columns

# STEP 13:
# Separate input features (X) and target feature (y)
# X -> independent variables
# y -> target/output variable (math_score)

# STEP 14:
# Remove target column from input features using drop()

# STEP 15:
# Apply fit_transform() on training data
# fit() learns:
# - median values
# - scaling parameters
# - category mappings
# transform() applies preprocessing

# STEP 16:
# Apply transform() on testing data
# Uses preprocessing rules learned from training data

# STEP 17:
# Combine transformed features and target column using np.c_

# STEP 18:
# Save preprocessing object as preprocessor.pkl
# Used later during model prediction and deployment

# STEP 19:
# Return:
# - transformed training array
# - transformed testing array
# - preprocessor object path