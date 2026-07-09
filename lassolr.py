from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime
import joblib
import time

from sqlalchemy import create_engine, String, Integer, Float, Column, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session


engine = create_engine("sqlite:///fwi-prediction.db", connect_args={"check_same_thread":False})
sessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
class FWI(Base):
    __tablename__ = "fwi-prediction-table"
    id = Column(Integer, primary_key=True)
    temperature = Column(Float, nullable=False)
    rh = Column(Float, nullable=False)
    ws = Column(Float, nullable=False)
    rain = Column(Float, nullable=False)
    ffmc = Column(Float, nullable=False)
    dc = Column(Float, nullable=False)
    isi = Column(Float, nullable=False)
    bui = Column(Float, nullable=False)
    fire = Column(Boolean, nullable=False)
    pred_fwi = Column(Float, nullable=False)
    datetime = Column(DateTime, nullable=False)
    latency = Column(Float, nullable=False)
    error = Column(String)

Base.metadata.create_all(engine)

def get_db():
    db = sessionLocal()
    try:
        yield db
    finally:
        db.close()

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
def prediction(data: UserInput, db: Session = Depends(get_db)):
    current_dt = datetime.now()
    start_time = time.time()
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
        latency = time.time() - start_time

        db.add(
            FWI(
                temperature = data.temperature,
                rh = data.rh,
                ws = data.ws,
                rain = data.rain,
                ffmc = data.ffmc,
                dc = data.dc,
                isi = data.isi,
                bui = data.bui,
                fire = data.not_fire,
                pred_fwi = predicted_fwi,
                datetime = current_dt,
                latency = latency,
                error = None
            )
        )
        db.commit()
        return f"Predicted FWI is : {predicted_fwi:.2f}"
    except Exception as e:
        db.rollback()
        db.add(
            FWI(
                temperature = data.temperature,
                rh = data.rh,
                ws = data.ws,
                rain = data.rain,
                ffmc = data.ffmc,
                dc = data.dc,
                isi = data.isi,
                bui = data.bui,
                fire = data.not_fire,
                pred_fwi = 0,
                datetime = current_dt,
                latency = 0,
                error = str(e)
            )
        )
        db.commit()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    
