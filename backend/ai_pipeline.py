import asyncio
import re
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from urllib.parse import urlparse
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    print("Warning: ML models not available. Running in basic mode.")
from bs4 import BeautifulSoup
import sqlite3
import json
import os

logger = logging.getLogger(__name__)

class FactCheckPipeline:
    def __init__(self):
        self.embedding_model = None
        self.nli_model = None
        self.claim_extractor = None
        self.db_path = "factguard.db"
        self._init_database()
        self._load_models()
        
    def _init_database(self):
        """Initialize SQLite database for caching"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT UNIQUE,
                result TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                content_type TEXT,
                risk_level TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        
    def _load_models(self):
        """Load AI models for embeddings and NLI"""
        if not MODELS_AVAILABLE:
            logger.info("Running in basic mode without ML models")
            self.embedding_model = None
            self.nli_model = None
            self.stance_classifier = None
            return
            
        try:
            logger.info("Loading AI models...")
            
            # Load sentence transformer for embeddings
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load NLI model for stance detection
            self.nli_model = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                return_all_scores=True
            )
            
            # Alternative NLI model that works better for stance detection
            model_name = "facebook/bart-large-mnli"
            self.stance_classifier = pipeline(
                "zero-shot-classification",
                model=model_name
            )
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Fallback to simpler models or mock responses
            self.embedding_model = None
            self.nli_model = None
            self.stance_classifier = None
    
    async def analyze(self, text: Optional[str] = None, url: Optional[str] = None, user_id: str = None) -> Dict[str, Any]:
        """Main analysis pipeline"""
        try:
            # Extract text from URL if provided
            if url and not text:
                text = await self._extract_text_from_url(url)
            
            if not text:
                raise ValueError("No text content to analyze")
            
            # Check cache first
            content_hash = hash(text)
            cached_result = self._get_cached_result(content_hash)
            if cached_result:
                return cached_result
            
            # Extract claims
            claims = await self._extract_claims(text)
            
            # Retrieve evidence for each claim
            evidence_results = []
            for claim in claims:
                evidence = await self._retrieve_evidence(claim["text"])
                evidence_results.extend(evidence)
            
            # Rerank evidence using embeddings
            reranked_evidence = await self._rerank_evidence(claims, evidence_results)
            
            # Perform stance detection
            stance_results = await self._detect_stance(claims, reranked_evidence)
            
            # Apply heuristics
            heuristic_scores = await self._apply_heuristics(text, url, evidence_results)
            
            # Calculate overall risk
            overall_risk, explanations = self._calculate_risk_score(
                stance_results, heuristic_scores, claims
            )
            
            # Generate educational tips
            teach_tips = self._generate_teaching_tips(overall_risk, explanations)
            
            result = {
                "overall_risk": overall_risk,
                "explanations": explanations,
                "claims": claims,
                "evidence": reranked_evidence[:10],  # Top 10 evidence pieces
                "teach_tips": teach_tips
            }
            
            # Cache result
            self._cache_result(content_hash, result)
            
            # Log user activity
            if user_id:
                self._log_user_activity(user_id, "text" if not url else "url", overall_risk)
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis pipeline error: {e}")
            return self._get_fallback_response(str(e))
    
    async def _extract_text_from_url(self, url: str) -> str:
        """Extract text content from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text from common content areas
            content_selectors = [
                'article', '.article-content', '.post-content', 
                '.entry-content', 'main', '.main-content'
            ]
            
            text = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    text = elements[0].get_text()
                    break
            
            if not text:
                text = soup.get_text()
            
            # Clean up text
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:5000]  # Limit to 5000 characters
            
        except Exception as e:
            logger.error(f"Error extracting text from URL: {e}")
            raise ValueError(f"Could not extract text from URL: {e}")
    
    async def _extract_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract key claims from text"""
        # Simple claim extraction using sentence splitting and filtering
        sentences = re.split(r'[.!?]+', text)
        
        claims = []
        for i, sentence in enumerate(sentences[:10]):  # Limit to first 10 sentences
            sentence = sentence.strip()
            if len(sentence) > 20 and self._is_factual_claim(sentence):
                claims.append({
                    "id": i,
                    "text": sentence,
                    "confidence": 0.8,
                    "type": "factual"
                })
        
        return claims[:5]  # Return top 5 claims
    
    def _is_factual_claim(self, sentence: str) -> bool:
        """Determine if sentence contains a factual claim"""
        # Simple heuristics for factual claims
        factual_indicators = [
            r'\b(is|are|was|were|has|have|will|would|can|could)\b',
            r'\b\d+\b',  # Contains numbers
            r'\b(according to|research shows|study finds|data shows)\b',
            r'\b(percent|percentage|million|billion|thousand)\b'
        ]
        
        opinion_indicators = [
            r'\b(think|believe|feel|opinion|seems|appears|might|maybe)\b',
            r'\b(should|ought|must|need to)\b'
        ]
        
        factual_score = sum(1 for pattern in factual_indicators if re.search(pattern, sentence, re.IGNORECASE))
        opinion_score = sum(1 for pattern in opinion_indicators if re.search(pattern, sentence, re.IGNORECASE))
        
        return factual_score > opinion_score and len(sentence.split()) > 5
    
    async def _retrieve_evidence(self, claim: str) -> List[Dict[str, Any]]:
        """Retrieve evidence for a claim using web search"""
        # Mock evidence retrieval (replace with actual Bing/SerpAPI)
        mock_evidence = [
            {
                "title": f"Research on: {claim[:50]}...",
                "snippet": f"According to recent studies, the claim about {claim[:30]}... has been investigated.",
                "url": "https://example.com/research1",
                "domain": "example.com",
                "credibility_score": 0.8,
                "date": "2024-01-15"
            },
            {
                "title": f"Fact-check: {claim[:40]}...",
                "snippet": f"Independent verification shows mixed evidence regarding {claim[:25]}...",
                "url": "https://factcheck.org/article1",
                "domain": "factcheck.org",
                "credibility_score": 0.9,
                "date": "2024-01-10"
            }
        ]
        
        return mock_evidence
    
    async def _rerank_evidence(self, claims: List[Dict], evidence: List[Dict]) -> List[Dict]:
        """Rerank evidence using semantic similarity"""
        if not MODELS_AVAILABLE or not self.embedding_model or not claims or not evidence:
            # Basic reranking by credibility score
            evidence.sort(key=lambda x: x.get("credibility_score", 0.5), reverse=True)
            return evidence
        
        try:
            # Get embeddings for claims
            claim_texts = [claim["text"] for claim in claims]
            claim_embeddings = self.embedding_model.encode(claim_texts)
            
            # Get embeddings for evidence
            evidence_texts = [f"{ev['title']} {ev['snippet']}" for ev in evidence]
            evidence_embeddings = self.embedding_model.encode(evidence_texts)
            
            # Calculate similarity scores
            for i, ev in enumerate(evidence):
                similarities = []
                for claim_emb in claim_embeddings:
                    similarity = np.dot(claim_emb, evidence_embeddings[i]) / (
                        np.linalg.norm(claim_emb) * np.linalg.norm(evidence_embeddings[i])
                    )
                    similarities.append(similarity)
                
                ev["relevance_score"] = max(similarities)
            
            # Sort by relevance
            evidence.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
        
        return evidence
    
    async def _detect_stance(self, claims: List[Dict], evidence: List[Dict]) -> Dict[str, Any]:
        """Detect stance between claims and evidence"""
        stance_results = {
            "contradictions": 0,
            "supports": 0,
            "neutral": 0,
            "details": []
        }
        
        if not MODELS_AVAILABLE or not self.stance_classifier:
            # Basic heuristic stance detection
            for claim in claims:
                for ev in evidence[:3]:
                    # Simple keyword matching for demo
                    claim_lower = claim["text"].lower()
                    evidence_lower = f"{ev['title']} {ev['snippet']}".lower()
                    
                    # Look for contradiction keywords
                    contradiction_words = ["false", "incorrect", "wrong", "debunked", "myth"]
                    support_words = ["true", "correct", "confirmed", "verified", "proven"]
                    
                    if any(word in evidence_lower for word in contradiction_words):
                        stance_results["contradictions"] += 1
                        stance_results["details"].append({
                            "claim_id": claim["id"],
                            "evidence_url": ev["url"],
                            "stance": "contradicts",
                            "confidence": 0.7
                        })
                    elif any(word in evidence_lower for word in support_words):
                        stance_results["supports"] += 1
                        stance_results["details"].append({
                            "claim_id": claim["id"],
                            "evidence_url": ev["url"],
                            "stance": "supports",
                            "confidence": 0.7
                        })
                    else:
                        stance_results["neutral"] += 1
            return stance_results
        
        try:
            for claim in claims:
                claim_text = claim["text"]
                
                for ev in evidence[:5]:  # Check top 5 evidence pieces
                    evidence_text = f"{ev['title']} {ev['snippet']}"
                    
                    # Use zero-shot classification for stance detection
                    candidate_labels = ["supports", "contradicts", "neutral"]
                    result = self.stance_classifier(
                        f"Claim: {claim_text} Evidence: {evidence_text}",
                        candidate_labels
                    )
                    
                    stance = result["labels"][0]
                    confidence = result["scores"][0]
                    
                    if confidence > 0.6:  # Only consider high-confidence predictions
                        stance_results[stance + "s" if stance != "neutral" else stance] += 1
                        stance_results["details"].append({
                            "claim_id": claim["id"],
                            "evidence_url": ev["url"],
                            "stance": stance,
                            "confidence": confidence
                        })
        
        except Exception as e:
            logger.error(f"Error in stance detection: {e}")
        
        return stance_results
    
    async def _apply_heuristics(self, text: str, url: Optional[str], evidence: List[Dict]) -> Dict[str, float]:
        """Apply heuristic checks for misinformation indicators"""
        scores = {
            "clickbait_score": 0.0,
            "source_credibility": 0.5,
            "date_relevance": 0.5,
            "author_credibility": 0.5
        }
        
        # Clickbait detection
        clickbait_patterns = [
            r'\b(shocking|unbelievable|you won\'t believe|doctors hate)\b',
            r'\b(this one trick|secret|exposed|revealed)\b',
            r'\b(number \d+ will shock you|what happens next)\b'
        ]
        
        clickbait_count = sum(1 for pattern in clickbait_patterns 
                             if re.search(pattern, text, re.IGNORECASE))
        scores["clickbait_score"] = min(clickbait_count * 0.3, 1.0)
        
        # Source credibility (based on domain)
        if url:
            domain = urlparse(url).netloc.lower()
            trusted_domains = [
                'reuters.com', 'bbc.com', 'npr.org', 'apnews.com',
                'factcheck.org', 'snopes.com', 'politifact.com'
            ]
            suspicious_domains = [
                'naturalnews.com', 'infowars.com', 'breitbart.com'
            ]
            
            if any(trusted in domain for trusted in trusted_domains):
                scores["source_credibility"] = 0.9
            elif any(suspicious in domain for suspicious in suspicious_domains):
                scores["source_credibility"] = 0.2
        
        # Evidence source credibility
        if evidence:
            avg_credibility = sum(ev.get("credibility_score", 0.5) for ev in evidence) / len(evidence)
            scores["source_credibility"] = (scores["source_credibility"] + avg_credibility) / 2
        
        return scores
    
    def _calculate_risk_score(self, stance_results: Dict, heuristic_scores: Dict, claims: List[Dict]) -> tuple:
        """Calculate overall risk score and explanations"""
        risk_factors = []
        
        # Analyze stance results
        total_stances = sum([
            stance_results["contradictions"],
            stance_results["supports"],
            stance_results["neutral"]
        ])
        
        if total_stances > 0:
            contradiction_ratio = stance_results["contradictions"] / total_stances
            support_ratio = stance_results["supports"] / total_stances
            
            if contradiction_ratio > 0.4:
                risk_factors.append("High contradiction rate with reliable sources")
            elif support_ratio < 0.3:
                risk_factors.append("Limited supporting evidence found")
        
        # Analyze heuristic scores
        if heuristic_scores["clickbait_score"] > 0.5:
            risk_factors.append("Contains clickbait-style language")
        
        if heuristic_scores["source_credibility"] < 0.4:
            risk_factors.append("Source has low credibility rating")
        
        # Calculate overall risk
        risk_score = (
            heuristic_scores["clickbait_score"] * 0.3 +
            (1 - heuristic_scores["source_credibility"]) * 0.4 +
            (stance_results["contradictions"] / max(total_stances, 1)) * 0.3
        )
        
        if risk_score > 0.7:
            overall_risk = "high"
        elif risk_score > 0.4:
            overall_risk = "medium"
        else:
            overall_risk = "low"
        
        if not risk_factors:
            risk_factors = ["No significant misinformation indicators detected"]
        
        return overall_risk, risk_factors
    
    def _generate_teaching_tips(self, risk_level: str, explanations: List[str]) -> List[str]:
        """Generate educational tips based on analysis"""
        base_tips = [
            "Always check multiple sources before believing information",
            "Look for author credentials and publication date",
            "Be skeptical of sensational headlines or emotional language",
            "Cross-reference claims with fact-checking websites"
        ]
        
        risk_specific_tips = {
            "high": [
                "This content shows multiple red flags - verify carefully",
                "Consider the source's motivation and potential bias",
                "Look for peer-reviewed research on the topic"
            ],
            "medium": [
                "Some concerns detected - additional verification recommended",
                "Check if the information is recent and up-to-date"
            ],
            "low": [
                "Content appears reliable, but always maintain healthy skepticism"
            ]
        }
        
        return base_tips + risk_specific_tips.get(risk_level, [])
    
    def _get_cached_result(self, content_hash: int) -> Optional[Dict]:
        """Get cached analysis result"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT result FROM analysis_cache WHERE content_hash = ? AND timestamp > datetime('now', '-1 hour')",
                (str(content_hash),)
            )
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return json.loads(result[0])
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        
        return None
    
    def _cache_result(self, content_hash: int, result: Dict):
        """Cache analysis result"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO analysis_cache (content_hash, result) VALUES (?, ?)",
                (str(content_hash), json.dumps(result))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
    
    def _log_user_activity(self, user_id: str, content_type: str, risk_level: str):
        """Log user activity for analytics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO user_logs (user_id, content_type, risk_level) VALUES (?, ?, ?)",
                (user_id, content_type, risk_level)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Logging error: {e}")
    
    def _get_fallback_response(self, error_msg: str) -> Dict[str, Any]:
        """Return fallback response when analysis fails"""
        return {
            "overall_risk": "unknown",
            "explanations": [f"Analysis failed: {error_msg}"],
            "claims": [],
            "evidence": [],
            "teach_tips": [
                "Unable to analyze content automatically",
                "Please verify information manually using trusted sources",
                "Check multiple independent sources",
                "Look for expert opinions and peer-reviewed research"
            ]
        }
    
    async def get_models_status(self) -> Dict[str, Any]:
        """Get status of loaded models"""
        return {
            "embedding_model": self.embedding_model is not None,
            "nli_model": self.nli_model is not None,
            "stance_classifier": self.stance_classifier is not None,
            "database": os.path.exists(self.db_path)
        }
