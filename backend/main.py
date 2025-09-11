from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import asyncio
import logging
from datetime import datetime
import json

# Try to import enhanced pipeline, fall back to basic if ML dependencies fail
try:
    from enhanced_ai_pipeline import EnhancedFactCheckPipeline
    AI_PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced AI pipeline not available: {e}")
    try:
        from ai_pipeline import FactCheckPipeline as EnhancedFactCheckPipeline
        AI_PIPELINE_AVAILABLE = True
    except ImportError as e2:
        print(f"Basic AI pipeline also not available: {e2}")
        AI_PIPELINE_AVAILABLE = False
        # Create a mock pipeline for basic functionality
        class MockFactCheckPipeline:
            def __init__(self):
                self.available = False
            
            async def analyze_text(self, text, user_id):
                return {
                    "overall_verdict": "Needs Review",
                    "claims": [{
                        "text": text[:100] + "..." if len(text) > 100 else text,
                        "evidence": [{
                            "title": "AI Pipeline Unavailable",
                            "snippet": "Enhanced AI analysis is temporarily unavailable. Basic heuristic analysis only.",
                            "stance": "neutral",
                            "highlights": []
                        }]
                    }],
                    "educational_tips": [
                        "Verify information from multiple reliable sources",
                        "Check the publication date and author credentials",
                        "Look for supporting evidence and citations"
                    ]
                }
            
            async def analyze_url(self, url, user_id):
                return await self.analyze_text(f"Content from: {url}", user_id)
            
            async def start_training(self, dataset, user_id):
                return "training_unavailable"
            
            async def get_models_status(self):
                return {"status": "unavailable", "reason": "ML dependencies not installed"}
        
        EnhancedFactCheckPipeline = MockFactCheckPipeline

from auth_middleware import verify_firebase_token

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FactGuard API",
    description="AI-powered misinformation detection and education tool",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI pipeline
fact_checker = EnhancedFactCheckPipeline()

class AnalyzeTextRequest(BaseModel):
    text: str

class AnalyzeUrlRequest(BaseModel):
    url: str

class AnalyzeResponse(BaseModel):
    overall_verdict: str  # True, False, Misleading, Needs Review
    claims: List[Dict[str, Any]]  # with evidence, stance, highlights
    educational_tips: List[str]
    processing_time: float

class TrainRequest(BaseModel):
    dataset: List[Dict[str, str]]  # [{"claim": "...", "evidence": "...", "label": "support"}]

@app.get("/")
async def root():
    return {"message": "FactGuard API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/analyze-text", response_model=AnalyzeResponse)
async def analyze_text(
    request: AnalyzeTextRequest,
    user_id: str = Depends(verify_firebase_token)
):
    """
    Analyze text for misinformation using AI pipeline
    Requires Firebase authentication
    """
    try:
        start_time = datetime.now()
        
        # Process the content through AI pipeline
        result = await fact_checker.analyze_text(
            text=request.text,
            user_id=user_id
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AnalyzeResponse(
            overall_verdict=result["overall_verdict"],
            claims=result["claims"],
            educational_tips=result["educational_tips"],
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")

@app.post("/analyze-url", response_model=AnalyzeResponse)
async def analyze_url(
    request: AnalyzeUrlRequest,
    user_id: str = Depends(verify_firebase_token)
):
    """
    Analyze URL content for misinformation using AI pipeline
    Requires Firebase authentication
    """
    try:
        start_time = datetime.now()
        
        # Process the content through AI pipeline
        result = await fact_checker.analyze_url(
            url=request.url,
            user_id=user_id
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AnalyzeResponse(
            overall_verdict=result["overall_verdict"],
            claims=result["claims"],
            educational_tips=result["educational_tips"],
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error analyzing URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"URL analysis failed: {str(e)}")

@app.post("/train")
async def train_model(
    request: TrainRequest,
    user_id: str = Depends(verify_firebase_token)
):
    """
    Fine-tune stance model with uploaded dataset
    Requires Firebase authentication
    """
    try:
        # Validate dataset format
        for item in request.dataset:
            if not all(key in item for key in ["claim", "evidence", "label"]):
                raise HTTPException(status_code=400, detail="Invalid dataset format")
            if item["label"] not in ["support", "contradict", "neutral"]:
                raise HTTPException(status_code=400, detail="Invalid label. Must be: support, contradict, or neutral")
        
        # Start training process
        training_id = await fact_checker.start_training(
            dataset=request.dataset,
            user_id=user_id
        )
        
        return {
            "message": "Training started successfully",
            "training_id": training_id,
            "status": "started",
            "dataset_size": len(request.dataset)
        }
        
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/models/status")
async def get_models_status():
    """Get status of loaded AI models"""
    return await fact_checker.get_models_status()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
