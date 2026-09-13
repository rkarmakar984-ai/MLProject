from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def run_training_pipeline():
    # Data Ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    
    # Data Transformation
    transformation = DataTransformation()
    X_train, Y_train, X_test, Y_test, preprocessor = (
        transformation.initiate_data_transformation(
            train_path,
            test_path
        )
    )
    
    # MOdel Training
    trainer = ModelTrainer()
    final_test_r2 = trainer.initiate_model_training(
        X_train,
        Y_train,
        X_test,
        Y_test,
        preprocessor
    )
    
    return final_test_r2

if __name__ == "__main__":
    score = run_training_pipeline()
    print(f"Final test R2: {score:.4f}")