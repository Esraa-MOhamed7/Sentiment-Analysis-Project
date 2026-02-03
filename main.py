from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Sentiment API is running. Use /predict to analyze text."}

model_loaded = joblib.load("sentiment_model.pkl")
vectorizer_loaded = joblib.load("vectorizer.pkl")

# Request schema
class TextInput(BaseModel):
    text: str

def predict_sentiment(text: str):
    text_vec = vectorizer_loaded.transform([text])
    pred = model_loaded.predict(text_vec)[0]
    return "Positive" if pred == 1 else "Negative"

# API endpoint
@app.post("/predict")
def predict(input: TextInput):
    sentiment = predict_sentiment(input.text)
    return {"text": input.text, "sentiment": sentiment}


