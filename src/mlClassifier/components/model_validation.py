import torch 
import pandas as pd
from mlClassifier.entity import ModelValidationConfig
from os.path import exists
from mlClassifier.utils.common import load_object, save_json
from sklearn import metrics  
from pathlib import Path
from mlClassifier.utils.ml_functions import Deep

class ModelValidation:
    def __init__(self, config: ModelValidationConfig):
        self.config = config

    def validate_model_performance(self):
        df = pd.read_csv(self.config.data_path)
        X, y = df.drop(columns=['diabetes'],axis=1), df['diabetes']
        X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
        y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32).reshape(-1, 1)

        results = {"Best_Accuracy_Model_Metrics": None, "Best_F1_Model_Metrics": None}

        if exists(self.config.best_acc_model_path_pkl):
            model = load_object(self.config.best_acc_model_path_pkl)
            y_pred = model.predict(X)
            acc, f1, precision, recall = metrics.accuracy_score(y, y_pred), metrics.f1_score(y, y_pred), metrics.precision_score(y, y_pred), metrics.recall_score(y, y_pred)
            results['Best_Accuracy_Model_Metrics'] = {"Accuracy_Score": acc, "F1_Score": f1, "Precision_Score": precision, "Recall_Score": recall}
        elif exists(self.config.best_acc_model_path_pth):
            model = Deep()
            model.load_state_dict(torch.load(self.config.best_acc_model_path_pth))
            model.eval()
            y_pred_tensor = model(X_tensor)
            y_pred_tensor = y_pred_tensor.round().detach().cpu().numpy()
            y_tensor_np = y_tensor.numpy()
            acc, f1, precision, recall = metrics.accuracy_score(y_tensor_np, y_pred_tensor), metrics.f1_score(y_tensor_np, y_pred_tensor), metrics.precision_score(y_tensor_np, y_pred_tensor), metrics.recall_score(y_tensor_np, y_pred_tensor)
            results['Best_Accuracy_Model_Metrics'] = {"Accuracy_Score": acc, "F1_Score": f1, "Precision_Score": precision, "Recall_Score": recall}
        else:
            print("Error")

        if exists(self.config.best_f1_model_path_pkl):
            model = load_object(self.config.best_f1_model_path_pkl)
            y_pred = model.predict(X)
            acc, f1, precision, recall = metrics.accuracy_score(y, y_pred), metrics.f1_score(y, y_pred), metrics.precision_score(y, y_pred), metrics.recall_score(y, y_pred)
            results['Best_F1_Model_Metrics'] = {"Accuracy_Score": acc, "F1_Score": f1, "Precision_Score": precision, "Recall_Score": recall}
        elif exists(self.config.best_f1_model_path_pth):
            model = Deep()
            model.load_state_dict(torch.load(self.config.best_f1_model_path_pth))
            model.eval()
            y_pred_tensor = model(X_tensor)
            y_pred_tensor = y_pred_tensor.round().detach().cpu().numpy()
            y_tensor_np = y_tensor.numpy() 
            acc, f1, precision, recall = metrics.accuracy_score(y_tensor_np, y_pred_tensor), metrics.f1_score(y_tensor_np, y_pred_tensor), metrics.precision_score(y_tensor_np, y_pred_tensor), metrics.recall_score(y_tensor_np, y_pred_tensor)
            results['Best_F1_Model_Metrics'] = {"Accuracy_Score": acc, "F1_Score": f1, "Precision_Score": precision, "Recall_Score": recall}
        else:
            print("Error")

        save_json(path = Path(self.config.save_dir), data = results)