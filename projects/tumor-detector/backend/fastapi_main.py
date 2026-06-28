from fastapi import FastAPI, UploadFile, File
from functions import predict_tumor

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        contents = await file.read()
        prediction = predict_tumor(contents)

        return {"Prediction": prediction}

    except Exception as e:
        return {"error": str(e)}