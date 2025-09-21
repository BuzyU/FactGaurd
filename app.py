from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Zero-shot classifier (pretrained, no training needed)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Input schema
class AnalyzeRequest(BaseModel):
    text: str

# API endpoint
@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    try:
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
    except Exception as e:
        return {"status": "failed", "error": str(e)}
