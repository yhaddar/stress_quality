from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("./models/stress_quality.pkl")
scaler = joblib.load("./models/stress_scaler.pkl")
le = joblib.load("./models/stress_le.pkl")

app = FastAPI(title="Stress Prediction API")

class StressInput(BaseModel):
    Study_Hours_Per_Day: float
    Extracurricular_Hours_Per_Day: float
    Sleep_Hours_Per_Day: float
    Social_Hours_Per_Day: float
    Physical_Activity_Hours_Per_Day: float
    GPA: float

@app.get("/")
def read_root():
    return {"message": "Stress Prediction API is running"}

@app.post("/predict")
def predict_stress(data: StressInput):
    input_array = np.array([[data.Study_Hours_Per_Day,
                             data.Extracurricular_Hours_Per_Day,
                             data.Sleep_Hours_Per_Day,
                             data.Social_Hours_Per_Day,
                             data.Physical_Activity_Hours_Per_Day,
                             data.GPA]])
    
    input_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_scaled)
    
    prediction_label = le.inverse_transform(prediction.astype(int))[0]
    
    return {"message": prediction_label}
