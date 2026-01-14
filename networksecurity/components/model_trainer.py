import os
import sys
import mlflow
import dagshub
from urllib.parse import urlparse

from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data, evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

# Initialize DagsHub
dagshub.init(repo_owner='mahendra-918', repo_name='networksecurity', mlflow=True)

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def track_mlflow(self, best_model, train_metric, test_metric):
        """
        Logs both training and testing metrics to a single MLflow run.
        """
        try:
            # DagsHub Tracking URI
            mlflow.set_registry_uri("https://dagshub.com/mahendra-918/networksecurity.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            with mlflow.start_run():
                # Log Training Metrics
                mlflow.log_metric("train_f1_score", train_metric.f1_score)
                mlflow.log_metric("train_precision", train_metric.precision_score)
                mlflow.log_metric("train_recall", train_metric.recall_score)

                # Log Testing Metrics
                mlflow.log_metric("test_f1_score", test_metric.f1_score)
                mlflow.log_metric("test_precision", test_metric.precision_score)
                mlflow.log_metric("test_recall", test_metric.recall_score)

                # Model logging logic
                if tracking_url_type_store != "file":
                    # FIX: Pass a STRING "NetworkSecurityModel", not the model object itself
                    mlflow.sklearn.log_model(
                        sk_model=best_model, 
                        artifact_path="model", 
                        registered_model_name="NetworkSecurityModel"
                    )
                else:
                    mlflow.sklearn.log_model(sk_model=best_model, artifact_path="model")
                    
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def train_model(self, X_train, y_train, x_test, y_test):
        try:
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
            params = {
                "Decision Tree": {'criterion': ['gini', 'entropy', 'log_loss']},
                "Random Forest": {'n_estimators': [8, 16, 32, 128, 256]},
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Logistic Regression": {},
                "AdaBoost": {
                    'learning_rate': [.1, .01, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=x_test, y_test=y_test,
                models=models, param=params
            )
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            # Get Metrics
            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

            y_test_pred = best_model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            # MLflow Tracking (Now combined into one call)
            self.track_mlflow(best_model, classification_train_metric, classification_test_metric)

            # Save the Model & Preprocessor
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            # Create the estimator object
            Network_Model = NetworkModel(preprocessor=preprocessor, model=best_model)

            # FIX: Use the object 'Network_Model', not the class 'NetworkModel'
            save_object(self.model_trainer_config.trained_model_file_path, obj=Network_Model)
            
            # Save for Pusher
            os.makedirs("final_model", exist_ok=True)
            save_object("final_model/model.pkl", best_model)

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            return self.train_model(x_train, y_train, x_test, y_test)
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)