from fastapi import FastAPI
from pydantic import  BaseModel, Field,computed_field, field_validator
from typing import  Literal, Annotated
import pickle
import pandas as pd
from fastapi.responses import  JSONResponse
from schema.user_input import UserInput
from schema.prediction_response import PredictionResponse
from model.predict import predict_output, ML_VERSION, model

app = FastAPI()

@app.get('/')
def home():
    return {
        "message":"This is Insurance Premium API",        
        }


@app.get('/health')
def health_check():
    return {
        "status": "OK",
        "Version": ML_VERSION,
        "model_load":model is not None
        }

@app.post('/predict', response_model=PredictionResponse)
def predict_premium(data: UserInput):

    input_user = {
        'bmi':data.bmi,
        'age_group':data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa':data.income_lpa,
        'occupation': data.occupation
    }

    try:
        prediction = predict_output(input_user)

        return JSONResponse(status_code=200, content=prediction)
    except Exception as e:
        return JSONResponse(status_code=500, content=str(e))
