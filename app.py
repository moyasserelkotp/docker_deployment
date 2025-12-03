from fastapi import FastAPI
import pickle 
import numpy as np

app = FastAPI()


# Load the trained model

model = None
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.get("/predict")
async def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}    

