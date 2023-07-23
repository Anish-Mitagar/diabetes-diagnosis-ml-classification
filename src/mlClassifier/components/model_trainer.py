import numpy as np 
import torch
import pandas as pd
from mlClassifier.entity import ModelTrainerConfig
from mlClassifier.utils.common import save_object
import collections
from mlClassifier.utils.ml_functions import (train_Grid_CV_CatBoostClassifier, 
                                             train_Grid_CV_Neural_Network_Classifier, 
                                             train_Grid_CV_AdaBoostClassifier,
                                             train_Grid_CV_RandomForestClassifier,
                                             train_Grid_CV_XGBClassifier)
from mlClassifier.logging import logger

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):

        df = pd.read_csv(self.config.data_path)

        X, y = df.drop(columns=['diabetes'],axis=1), df['diabetes']

        X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
        y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32).reshape(-1, 1)

        best_acc = -np.inf
        best_acc_model = None
        best_f1 = -np.inf
        best_f1_model = None
       
        logger.info(f"Started Grid Search Cross Validation for Neural Network")
        nn_results = train_Grid_CV_Neural_Network_Classifier(self.config.neural_network_params, X_tensor, y_tensor)
        if nn_results["best_acc"][0] > best_acc:
            best_acc, best_acc_model = nn_results["best_acc"][0], nn_results["best_acc"][1]
        if nn_results["best_f1"][0] > best_f1:
            best_f1, best_f1_model = nn_results["best_f1"][0], nn_results["best_f1"][1]  
        logger.info("Finished Grid Search Cross Validation for Neural Network")

        logger.info(f"Started Grid Search Cross Validation for Random Forest")
        random_forest_results = train_Grid_CV_RandomForestClassifier(self.config.random_forest_params, X, y)
        if random_forest_results["best_acc"][0] > best_acc:
            best_acc, best_acc_model = random_forest_results["best_acc"][0], random_forest_results["best_acc"][1] 
        if random_forest_results["best_f1"][0] > best_f1:
            best_f1, best_f1_model = random_forest_results["best_f1"][0], random_forest_results["best_f1"][1]  
        logger.info(f"Finished Grid Search Cross Validation for Random Forest")

        logger.info(f"Started Grid Search Cross Validation for XGB Classifier")
        xgb_results = train_Grid_CV_XGBClassifier(self.config.xg_boost_params, X, y)
        if xgb_results["best_acc"][0] > best_acc:
            best_acc, best_acc_model = xgb_results["best_acc"][0], xgb_results["best_acc"][1]  
        if xgb_results["best_f1"][0] > best_f1: 
            best_f1, best_f1_model = xgb_results["best_f1"][0], xgb_results["best_f1"][1]
        logger.info(f"Finished Grid Search Cross Validation for XGB Classifier")

        logger.info(f"Started Grid Search Cross Validation for CatBoost Classifier")
        cat_boost_results = train_Grid_CV_CatBoostClassifier(self.config.cat_boost_params, X, y)
        if cat_boost_results["best_acc"][0] > best_acc:
            best_acc, best_acc_model = cat_boost_results["best_acc"][0], cat_boost_results["best_acc"][1]  
        if cat_boost_results["best_f1"][0] > best_f1:
            best_f1, best_f1_model = cat_boost_results["best_f1"][0], cat_boost_results["best_f1"][1]  
        logger.info(f"Finished Grid Search Cross Validation for CatBoost Classifier")

        logger.info(f"Started Grid Search Cross Validation for AdaBoost Classifier")
        ada_boost_results = train_Grid_CV_AdaBoostClassifier(self.config.ada_boost_params, X, y)
        if ada_boost_results["best_acc"][0] > best_acc: 
            best_acc, best_acc_model = ada_boost_results["best_acc"][0], ada_boost_results["best_acc"][1]  
        if ada_boost_results["best_f1"][0] > best_f1:
            best_f1, best_f1_model = ada_boost_results["best_f1"][0], ada_boost_results["best_f1"][1]  
        logger.info(f"Finished Grid Search Cross Validation for AdaBoost Classifier")

        if type(best_acc_model) is collections.OrderedDict:
            torch.save(best_acc_model, "artifacts/model_trainer/best_acc_model/model.pth")
        else:
            save_object(file_path="artifacts/model_trainer/best_acc_model/model.pkl", obj = best_acc_model)
        logger.info(f"Successfully saved model for best accuracy")

        if type(best_f1_model) is collections.OrderedDict:
            torch.save(best_f1_model, "artifacts/model_trainer/best_f1_model/model.pth")
        else:
            save_object(file_path="artifacts/model_trainer/best_f1_model/model.pkl", obj = best_acc_model)
        logger.info(f"Successfully saved model for best f1 score")