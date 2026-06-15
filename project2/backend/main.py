from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from typing import Annotated

# Load trained pipeline
model = joblib.load("fake_news_svm_model.pkl")

# Define input schema
class NewsData(BaseModel):
    Article: Annotated[str, Field(... , description="The text of the news article to classify")]

app = FastAPI()

@app.post("/predict")
def predict(data: NewsData):
    try:
        # df = pd.DataFrame([data.model_dump()])

        prediction = model.predict([data.Article])
        label = ' Real' if prediction[0] == 1 else ' Fake'
        return {"Predicted_Class": label}
    except Exception as e:
        return {"error": str(e)}