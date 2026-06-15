from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Annotated
from functions import summarize_document

class summarize_text(BaseModel):
    Article: Annotated[str, Field(... , description="The article you want to summarize")]

app = FastAPI()

@app.post("/predict")
def predict(data: summarize_text):
    try:
        user_file = data.Article
        summary = summarize_document(user_file)
        return {"Summarize_text": summary}
    except Exception as e:
        return {"error": str(e)}