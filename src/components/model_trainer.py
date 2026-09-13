import os, sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.base import clone
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self, X_train, Y_train, X_test, Y_test, preprocessor):
        try:
            logging.info("Starting regression model selection on training data")
            logging.info("Training rows: %d; test rows reserved: %d", len(X_train), len(X_test))

            # Setting up 5-Fold Cross Validation
            cv = KFold(
                n_splits=5,
                shuffle=True,
                random_state=42,
            )
            logging.info("Configured 5-fold cross-validation with shuffle=True and random_state=42")
            
            # List of Models
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "Support Vector Regressor": SVR(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
                "Random Forest Regressor": RandomForestRegressor(
                    random_state=42,
                    n_jobs=-1,
                ),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
                "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
                "CatBoost Regressor": CatBoostRegressor(
                    verbose=False,
                    random_seed=42,
                ),
                "XGBoost Regressor": XGBRegressor(
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                ),
            }
            
            # Model Hyperparameters list
            tuning_spaces = {
                "Linear Regression": {
                    # Nothing to tune here...
                },

                "Lasso": {
                    "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
                },

                "Ridge": {
                    "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
                },

                "Support Vector Regressor": {
                    "model__C": [0.1, 1.0, 10.0, 100.0],
                    "model__epsilon": [0.01, 0.1, 0.2],
                    "model__kernel": ["linear", "rbf"],
                },

                "K-Neighbors Regressor": {
                    "model__n_neighbors": [3, 5, 7, 9, 11],
                    "model__weights": ["uniform", "distance"],
                    "model__p": [1, 2],
                },

                "Decision Tree Regressor": {
                    "model__max_depth": [None, 5, 10, 15, 20],
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 2, 4],
                },

                "Random Forest Regressor": {
                    "model__n_estimators": [200, 400, 600],
                    "model__max_depth": [None, 10, 20, 30],
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 2, 4],
                    "model__max_features": ["sqrt", "log2"],
                },

                "AdaBoost Regressor": {
                    "model__n_estimators": [50, 100, 200, 300],
                    "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
                },

                "Gradient Boosting Regressor": {
                    "model__n_estimators": [100, 200, 300],
                    "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__max_depth": [2, 3, 4],
                    "model__subsample": [0.8, 1.0],
                },

                "CatBoost Regressor": {
                    "model__iterations": [300, 500, 800],
                    "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__depth": [4, 6, 8, 10],
                    "model__l2_leaf_reg": [1, 3, 5, 7],
                },

                "XGBoost Regressor": {
                    "model__n_estimators": [200, 400, 600],
                    "model__max_depth": [3, 5, 7],
                    "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__subsample": [0.8, 1.0],
                    "model__colsample_bytree": [0.8, 1.0],
                },
            }
            
            default_cv_scores = {}
            for name, model in models.items():
                logging.info("Evaluating default candidate with 5-fold CV: %s", name)
                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", clone(preprocessor)),
                        ("model", clone(model)),
                    ]
                )

                scores = cross_val_score(
                    estimator=pipeline,
                    X=X_train,
                    y=Y_train,
                    scoring="r2",
                    cv=cv,
                )

                default_cv_scores[name] = {
                    "mean_r2": scores.mean(),
                    "std_r2": scores.std(),
                }
                logging.info(
                    "Default CV complete for %s: mean R2=%.4f, std=%.4f",
                    name,
                    scores.mean(),
                    scores.std(),
                )
                
            top_three_names = [name for name, metrics in sorted(
                    default_cv_scores.items(),
                    key=lambda item: item[1]["mean_r2"],
                    reverse=True,
                )[:3]
            ]
            logging.info("Selected top 3 candidates by mean CV R2: %s", top_three_names)
            
            # Hyperparameter tuning top 3 selected models
            tuned_searches = {}
            for name in top_three_names:
                logging.info("Starting hyperparameter tuning for: %s", name)
                # Setting the pipeline up
                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", clone(preprocessor)),
                        ("model", clone(models[name])),
                    ]
                )
                # Operating the random search cross validation
                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=tuning_spaces[name],
                    n_iter=20,
                    scoring="r2",
                    cv=cv,
                    random_state=42,
                    n_jobs=1,
                    refit=True,
                )
                search.fit(X_train, Y_train)
                # Storing the search results
                tuned_searches[name] = search
                logging.info(
                    "Tuning complete for %s: best CV R2=%.4f; best parameters=%s",
                    name,
                    search.best_score_,
                    search.best_params_,
                )
                
            # Finding the best model
            best_tuned_name = max(
                tuned_searches,
                key=lambda name: tuned_searches[name].best_score_,
            )
            best_search = tuned_searches[best_tuned_name]
            best_pipeline = best_search.best_estimator_
            logging.info(
                "Selected final model: %s with tuned CV R2=%.4f",
                best_tuned_name,
                best_search.best_score_,
            )
            
            # Evaluate the best selected model once on the test set
            Y_test_pred = best_pipeline.predict(X_test)
            final_test_r2 = r2_score(Y_test, Y_test_pred)
            logging.info(
                "Final untouched test-set R2 for %s: %.4f",
                best_tuned_name,
                final_test_r2,
            )
            
            if final_test_r2 < 0.60:
                message = (
                    f"Model rejected: final test R2 ({final_test_r2:.4f}) "
                    "is below the required threshold of 0.60."
                )
                logging.warning(message)
                raise ValueError(message)
            
            # Saving the complete fitted pipeline
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_pipeline,
            )
            logging.info(
                "Saved final pipeline to: %s",
                self.model_trainer_config.trained_model_file_path,
            )

            return final_test_r2
            
        except Exception as e:
            logging.exception("Model training failed")
            raise CustomException(e, sys)