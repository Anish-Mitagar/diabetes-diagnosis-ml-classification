from mlClassifier.constants import *
import sys
import pandas as pd
from mlClassifier.utils.exception import CustomException
from mlClassifier.utils.common import load_object
import os
from os.path import exists
import torch
from mlClassifier.utils.ml_functions import Deep

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            if exists(MODEL_FILE_PATH) and exists(MODEL_FILE_PATH2):
                model = load_object(MODEL_FILE_PATH)
                model2 = load_object(MODEL_FILE_PATH2)
                preprocessor=load_object(PREPROCESSOR_FILE_PATH)
                data_scaled=preprocessor.transform(features)
                preds=model.predict(data_scaled)[0]
                preds2=model2.predict(data_scaled)[0]
                return [preds, preds2]
            elif exists(TORCH_FILE_PATH) and exists(TORCH_FILE_PATH2):
                model = Deep()
                model.load_state_dict(torch.load(TORCH_FILE_PATH))
                model.eval()
                model2 = Deep()
                model2.load_state_dict(torch.load(TORCH_FILE_PATH2))
                model2.eval()
                preprocessor=load_object(PREPROCESSOR_FILE_PATH)
                data_scaled=preprocessor.transform(features)
                X_tensor = torch.tensor(data_scaled, dtype=torch.float32)
                preds = int(model(X_tensor).round().detach().numpy()[0])
                preds2 = int(model2(X_tensor).round().detach().numpy()[0])
                return [preds, preds2]
            else:
                return ["No model is currently loaded", "No model is currently loaded"]
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(  self,
        gender: str,
        age: float,
        hypertension: int,
        heart_disease: int,
        smoking_history: str,
        bmi: float,
        hba1c_level: float,
        blood_glucose_level: int):

        self.gender = gender
        self.age = age
        self.hypertension = hypertension
        self.heart_disease = heart_disease
        self.smoking_history = smoking_history
        self.bmi = bmi
        self.hba1c_level = hba1c_level
        self.blood_glucose_level = blood_glucose_level

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "age": [self.age],
                "hypertension": [self.hypertension],
                "heart_disease": [self.heart_disease],
                "smoking_history": [self.smoking_history],
                "bmi": [self.bmi],
                "HbA1c_level": [self.hba1c_level],
                "blood_glucose_level": [self.blood_glucose_level]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)