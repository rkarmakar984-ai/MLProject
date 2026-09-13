import os, sys
import pandas as pd
import requests
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

# Using dataclass decorator
@dataclass
class DataIngestionConfig:
    data_source: str = "csv"  # "csv" or "api"
    api_url: str | None = None
    source_data_path: str = os.path.join("notebook", "data", "stud.csv")
    # Clean default path for storing the training dataset subset used to fit models
    train_data_path: str = os.path.join("artifacts", "train.csv")
    # Clean default path for storing the testing dataset subset reserved for evaluation
    test_data_path: str = os.path.join("artifacts", "test.csv")
    # Clean default path for storing a backup copy of the raw source data
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self, ingestion_config=None):
        self.ingestion_config = ingestion_config or DataIngestionConfig()
    
    # Finding the actual row data inside an API JSON response.
    def _extract_records(self, payload):
        if isinstance(payload, list):
            return payload

        if not isinstance(payload, dict):
            raise ValueError("API response must be a JSON object or list")

        preferred_keys = ["data", "records", "results", "items"]

        for key in preferred_keys:
            value = payload.get(key)

            if isinstance(value, list):
                return value

            if isinstance(value, dict):
                for nested_key in preferred_keys:
                    nested_value = value.get(nested_key)
                    if isinstance(nested_value, list):
                        return nested_value

        raise ValueError(
            "Could not find a record list. Configure the API response key explicitly."
        )
    
    # Fetching the source dataset and returning it as a pandas DataFrame.
    def _load_data(self):
        # For CSV source file
        if self.ingestion_config.data_source == "csv":
            return pd.read_csv(self.ingestion_config.source_data_path)
        # For api source
        if self.ingestion_config.data_source == "api":
            if not self.ingestion_config.api_url:
                raise ValueError("api_url is required when data_source is 'api'")

            response = requests.get(self.ingestion_config.api_url, timeout=30)
            response.raise_for_status()
            payload = response.json()
            records = self._extract_records(payload)
            return pd.json_normalize(records)
        
        # Error for source not being api or csv file
        raise ValueError("data_source must be either 'csv' or 'api'")
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        
        try:
            # Loading the raw dataset from a local path into a pandas DataFrame
            df = self._load_data()
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