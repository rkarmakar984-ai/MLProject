import os, sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

# Using dataclass decorator
@dataclass
class DataIngestionConfig:
    # Clean default path for storing the training dataset subset used to fit models
    train_data_path: str = os.path.join("artifacts", "train.csv")
    # Clean default path for storing the testing dataset subset reserved for evaluation
    test_data_path: str = os.path.join("artifacts", "test.csv")
    # Clean default path for storing a backup copy of the raw source data
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        
        try:
            # Loading the raw dataset from a local path into a pandas DataFrame
            df = pd.read_csv("notebook/data/stud.csv")
            # Logging the successful read operation
            logging.info("Read the dataset as dataframe")
            
            # Directory Management of the backup dataset
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            # Exporting the backup dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            # Logging the train test split initiation
            logging.info("Initiating the spliting the dataset")
            # Spliting the dataset
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
            
            # Directory Management and export of the training dataset
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            # Directory Management and export of the testing dataset
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            # Logging the split completion
            logging.info("Ingestion of the data is completed")
            
            return(
                self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
            )
            
        # Exception Handeling
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()