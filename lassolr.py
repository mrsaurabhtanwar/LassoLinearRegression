from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib


app = FastAPI(title="LassoCV Regression")

@app.get("/")
def home():
    return f"Welcome to the LassoCV prediction model home page."

class UserInput(BaseModel):
    temperature : float
    rh : float
    ws : float
    rain : float
    ffmc : float
    dc : float
    isi : float
    bui : float
    not_fire : bool
try:
    scaler = joblib.load("StandardLassoCV.joblib")
    model = joblib.load("LassoCVlinearRegression.joblib")
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")

@app.post("/prediction")
def prediction(data: UserInput):
    try:
        scaled_input = scaler.transform([[
            data.temperature,
            data.rh,
            data.ws,
            data.rain,
            data.ffmc,
            data.dc,
            data.bui,
            data.isi,
            data.not_fire
        ]])
        Prediction = model.predict(scaled_input)
        predicted_fwi = Prediction[0]
        return f"Predicted FWI is : {predicted_fwi:.2f}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predition failed: {e}")

    
