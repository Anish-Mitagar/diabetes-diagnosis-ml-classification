from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from mlClassifier.pipeline.predict import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/train", methods=['GET','POST'])
def trainRoute():
    os.system("python3 main.py")
    return "Training done successfully!"

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData (
            gender=request.form.get('gender'),
            age=float(request.form.get('age')),
            hypertension=int(request.form.get('hypertension')),
            heart_disease=int(request.form.get('heart_disease')),
            smoking_history=request.form.get('smoking_history'),
            bmi=float(request.form.get('bmi')),
            hba1c_level=float(request.form.get('hba1c_level')),
            blood_glucose_level=request.form.get('blood_glucose_level')
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results1=results[0], results2=results[1])

if __name__=="__main__":
    app.run(host='0.0.0.0', port=8080)    