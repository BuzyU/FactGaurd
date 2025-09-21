from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from mangum import Mangum  # Serverless adapter

app = FastAPI()
handler = Mangum(app)

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

class AnalyzeRequest(BaseModel):
    text: str

@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    candidate_labels = ["reliable", "unreliable", "mixed"]
    result = classifier(request.text, candidate_labels)
    return {
        "status": "success",
        "result": {
            "text": request.text,
            "predicted_label": result['labels'][0],
            "score": float(result['scores'][0])
        }
    }
