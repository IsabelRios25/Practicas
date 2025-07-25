from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

modelo = joblib.load("modelo.pkl")

class Item(BaseModel):
    sqft_lot: float
    floors: float
    waterfront: float
    view: float
    condition: float
    grade : float
    sqft_living15: float
    sqft_lot15: float
    total_sqft : float
    bathrooms_per_bedroom: float
    renovated: float
    house_age: float 
    luxury : float

@app.post("/predict")
async def predict(item: Item):
    data = np.array([[item.sqft_lot,
        item.floors,
        item.waterfront,
        item.view,
        item.condition,
        item.grade,
        item.sqft_living15,
        item.sqft_lot15,
        item.total_sqft,
        item.bathrooms_per_bedroom,
        item.renovated,
        item.house_age,
        item.luxury]])
    prediction = modelo.predict(data)
    return {"prediction": float(prediction[0])}