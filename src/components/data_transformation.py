import sys, os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_transformer_object(self):
        """
        This Function is resposible for data transfrmation
        """
        try:
            # Differentiating on the basis of two type of columns
            numerical_columns = [
                "writing_score",
                "reading_score"
            ]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            
            # Creating transformation pipeline for numerical columns
            numerical_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="median")),
                    ("Scaler", StandardScaler(with_mean=False))
                ]
            )
            
            # Creating transformation pipeline for categorical columns
            categorical_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="most_frequent")),
                    ("One Hot Encoder", OneHotEncoder()),
                    ("Scaler", StandardScaler(with_mean=False))
                ]
            )
            
            # Logging the columns to transform and scale
            logging.info(f"Categorical Columns: {categorical_columns}")
            logging.info(f"Numerical Columns: {numerical_columns}")
            
            # Combining both pipelines
            preprocessor = ColumnTransformer(
                [
                    ("Numerical Pipeline", numerical_pipeline, numerical_columns),
                    ("Categorical Pipeline", categorical_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Logging the start of the reading of the datasets
            logging.info("Reading the train and test data initiated")
            
            # Reading the train and test dataset
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Logging the completion of the reading of the datasets
            logging.info("Reading the train and test data completed")
            
            preprocessing_obj = self.get_transformer_object()
            
            target_column_name = "math_score"
            
            input_feature_train_df = train_df.drop(target_column_name, axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = train_df.drop(target_column_name, axis=1)
            target_feature_test_df = train_df[target_column_name]
            
            # Applying the preprocessing
            logging.info("Applying preprocessing object on training and testing dataframe")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            # Logging the completion and saving of the preprocessing
            logging.info("Saved Prepocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr, test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)