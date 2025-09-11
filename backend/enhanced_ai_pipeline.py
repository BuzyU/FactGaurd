import asyncio
import re
import requests
import wikipedia
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from urllib.parse import urlparse
import os
import json
import sqlite3
import difflib
import uuid
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    print("Warning: ML models not available. Install with: pip install sentence-transformers transformers torch")
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class EnhancedFactCheckPipeline:
    def __init__(self):
        self.embedding_model = None
        self.stance_classifier = None
        self.db_path = "factguard.db"
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")
        self._init_database()
        self._load_models()
        
    def _init_database(self):
        """Initialize SQLite database"""
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
            CREATE TABLE IF NOT EXISTS training_jobs (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                dataset_size INTEGER,
                status TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at DATETIME
            )
        ''')
        conn.commit()
        conn.close()
        
    def _load_models(self):
        """Load AI models"""
        if not MODELS_AVAILABLE:
            logger.info("Running in basic mode without ML models")
            return
            
        try:
            logger.info("Loading enhanced AI models...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.stance_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            logger.info("Enhanced models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.embedding_model = None
            self.stance_classifier = None
    
    async def analyze_text(self, text: str, user_id: str) -> Dict[str, Any]:
        """Analyze text content"""
        try:
            # Extract claims
            claims = await self._extract_claims(text)
            
            # Retrieve evidence for each claim
            all_evidence = []
            for claim in claims:
                evidence = await self._retrieve_evidence_enhanced(claim["text"])
                all_evidence.extend(evidence)
            
            # Rerank evidence
            reranked_evidence = await self._rerank_evidence_enhanced(claims, all_evidence)
            
            # Perform stance detection with highlights
            stance_results = await self._detect_stance_enhanced(claims, reranked_evidence)
            
            # Compute verdict
            verdict = self._compute_verdict(stance_results, claims)
            
            # Add evidence and highlights to claims
            enhanced_claims = self._add_evidence_to_claims(claims, reranked_evidence, stance_results)
            
            # Generate educational tips
            tips = self._generate_educational_tips(verdict, stance_results)
            
            return {
                "overall_verdict": verdict,
                "claims": enhanced_claims,
                "educational_tips": tips
            }
            
        except Exception as e:
            logger.error(f"Text analysis error: {e}")
            return self._get_fallback_response(str(e))
    
    async def analyze_url(self, url: str, user_id: str) -> Dict[str, Any]:
        """Analyze URL content"""
        try:
            # Extract text from URL
            text = await self._extract_text_from_url(url)
            
            # Analyze the extracted text
            result = await self.analyze_text(text, user_id)
            
            return result
            
        except Exception as e:
            logger.error(f"URL analysis error: {e}")
            return self._get_fallback_response(str(e))
    
    async def _extract_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract factual claims from text"""
        sentences = re.split(r'[.!?]+', text)
        claims = []
        
        for i, sentence in enumerate(sentences[:10]):
            sentence = sentence.strip()
            if len(sentence) > 20 and self._is_factual_claim(sentence):
                claims.append({
                    "id": i,
                    "text": sentence,
                    "confidence": 0.8,
                    "type": "factual"
                })
        
        return claims[:5]
    
    def _is_factual_claim(self, sentence: str) -> bool:
        """Determine if sentence contains a factual claim"""
        factual_indicators = [
            r'\b(is|are|was|were|has|have|will|would|can|could)\b',
            r'\b\d+\b',
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
    
    async def _retrieve_evidence_enhanced(self, claim: str) -> List[Dict[str, Any]]:
        """Retrieve evidence using Google CSE and Wikipedia"""
        evidence = []
        
        # Google Custom Search
        if self.google_api_key and self.google_cse_id:
            google_results = await self._search_google_cse(claim)
            evidence.extend(google_results)
        
        # Wikipedia fallback
        wiki_results = await self._search_wikipedia(claim)
        evidence.extend(wiki_results)
        
        return evidence[:10]
    
    async def _search_google_cse(self, query: str) -> List[Dict[str, Any]]:
        """Search using Google Custom Search Engine"""
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_api_key,
                "cx": self.google_cse_id,
                "q": query,
                "num": 5
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get("items", []):
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("link", ""),
                    "domain": urlparse(item.get("link", "")).netloc,
                    "credibility_score": self._assess_domain_credibility(urlparse(item.get("link", "")).netloc),
                    "source": "google_cse"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Google CSE search error: {e}")
            return []
    
    async def _search_wikipedia(self, query: str) -> List[Dict[str, Any]]:
        """Search Wikipedia for evidence"""
        try:
            # Search for relevant pages
            search_results = wikipedia.search(query, results=3)
            results = []
            
            for title in search_results:
                try:
                    page = wikipedia.page(title)
                    summary = wikipedia.summary(title, sentences=2)
                    
                    results.append({
                        "title": page.title,
                        "snippet": summary,
                        "url": page.url,
                        "domain": "wikipedia.org",
                        "credibility_score": 0.9,
                        "source": "wikipedia"
                    })
                except wikipedia.exceptions.DisambiguationError as e:
                    # Take the first option
                    try:
                        page = wikipedia.page(e.options[0])
                        summary = wikipedia.summary(e.options[0], sentences=2)
                        results.append({
                            "title": page.title,
                            "snippet": summary,
                            "url": page.url,
                            "domain": "wikipedia.org",
                            "credibility_score": 0.9,
                            "source": "wikipedia"
                        })
                    except:
                        continue
                except:
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return []
    
    def _assess_domain_credibility(self, domain: str) -> float:
        """Assess domain credibility"""
        trusted_domains = [
            'reuters.com', 'bbc.com', 'npr.org', 'apnews.com',
            'factcheck.org', 'snopes.com', 'politifact.com',
            'nature.com', 'science.org', 'nejm.org'
        ]
        
        suspicious_domains = [
            'naturalnews.com', 'infowars.com', 'breitbart.com'
        ]
        
        domain_lower = domain.lower()
        
        if any(trusted in domain_lower for trusted in trusted_domains):
            return 0.95
        elif any(suspicious in domain_lower for suspicious in suspicious_domains):
            return 0.2
        elif domain_lower.endswith('.edu') or domain_lower.endswith('.gov'):
            return 0.9
        else:
            return 0.6
    
    async def _rerank_evidence_enhanced(self, claims: List[Dict], evidence: List[Dict]) -> List[Dict]:
        """Rerank evidence using semantic similarity"""
        if not MODELS_AVAILABLE or not self.embedding_model:
            evidence.sort(key=lambda x: x.get("credibility_score", 0.5), reverse=True)
            return evidence
        
        try:
            claim_texts = [claim["text"] for claim in claims]
            claim_embeddings = self.embedding_model.encode(claim_texts)
            
            evidence_texts = [f"{ev['title']} {ev['snippet']}" for ev in evidence]
            evidence_embeddings = self.embedding_model.encode(evidence_texts)
            
            for i, ev in enumerate(evidence):
                similarities = []
                for claim_emb in claim_embeddings:
                    similarity = np.dot(claim_emb, evidence_embeddings[i]) / (
                        np.linalg.norm(claim_emb) * np.linalg.norm(evidence_embeddings[i])
                    )
                    similarities.append(similarity)
                
                ev["relevance_score"] = max(similarities)
                ev["combined_score"] = (ev["relevance_score"] * 0.7 + ev["credibility_score"] * 0.3)
            
            evidence.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
            return evidence
            
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            return evidence
    
    async def _detect_stance_enhanced(self, claims: List[Dict], evidence: List[Dict]) -> Dict[str, Any]:
        """Enhanced stance detection with token-level highlighting"""
        stance_results = {
            "stances": [],
            "summary": {"supports": 0, "contradicts": 0, "neutral": 0}
        }
        
        for claim in claims:
            claim_text = claim["text"]
            
            for ev in evidence[:5]:
                evidence_text = f"{ev['title']} {ev['snippet']}"
                
                if MODELS_AVAILABLE and self.stance_classifier:
                    # Use ML model
                    candidate_labels = ["supports", "contradicts", "neutral"]
                    result = self.stance_classifier(
                        f"Claim: {claim_text} Evidence: {evidence_text}",
                        candidate_labels
                    )
                    
                    stance = result["labels"][0]
                    confidence = result["scores"][0]
                else:
                    # Fallback heuristic
                    stance, confidence = self._heuristic_stance_detection(claim_text, evidence_text)
                
                # Generate token-level highlights
                highlights = self._generate_highlights(claim_text, evidence_text, stance)
                
                stance_info = {
                    "claim_id": claim["id"],
                    "evidence_url": ev["url"],
                    "evidence_title": ev["title"],
                    "stance": stance,
                    "confidence": confidence,
                    "highlights": highlights
                }
                
                stance_results["stances"].append(stance_info)
                stance_results["summary"][stance] += 1
        
        return stance_results
    
    def _heuristic_stance_detection(self, claim: str, evidence: str) -> tuple:
        """Fallback heuristic stance detection"""
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()
        
        contradiction_words = ["false", "incorrect", "wrong", "debunked", "myth", "not true"]
        support_words = ["true", "correct", "confirmed", "verified", "proven", "accurate"]
        
        contradiction_score = sum(1 for word in contradiction_words if word in evidence_lower)
        support_score = sum(1 for word in support_words if word in evidence_lower)
        
        if contradiction_score > support_score:
            return "contradicts", 0.7
        elif support_score > contradiction_score:
            return "supports", 0.7
        else:
            return "neutral", 0.6
    
    def _generate_highlights(self, claim: str, evidence: str, stance: str) -> List[Dict[str, Any]]:
        """Generate token-level highlights between claim and evidence"""
        claim_tokens = claim.split()
        evidence_tokens = evidence.split()
        
        # Use difflib to find differences
        matcher = difflib.SequenceMatcher(None, claim_tokens, evidence_tokens)
        
        highlights = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace' or tag == 'delete':
                highlights.append({
                    "type": "claim_unique",
                    "text": " ".join(claim_tokens[i1:i2]),
                    "position": {"start": i1, "end": i2},
                    "context": "claim"
                })
            if tag == 'replace' or tag == 'insert':
                highlights.append({
                    "type": "evidence_unique",
                    "text": " ".join(evidence_tokens[j1:j2]),
                    "position": {"start": j1, "end": j2},
                    "context": "evidence"
                })
        
        return highlights[:10]  # Limit highlights
    
    def _compute_verdict(self, stance_results: Dict, claims: List[Dict]) -> str:
        """Compute overall verdict based on stance analysis"""
        summary = stance_results["summary"]
        total = sum(summary.values())
        
        if total == 0:
            return "Needs Review"
        
        contradict_ratio = summary["contradicts"] / total
        support_ratio = summary["supports"] / total
        
        if contradict_ratio > 0.6:
            return "False"
        elif contradict_ratio > 0.3:
            return "Misleading"
        elif support_ratio > 0.6:
            return "True"
        else:
            return "Needs Review"
    
    def _add_evidence_to_claims(self, claims: List[Dict], evidence: List[Dict], stance_results: Dict) -> List[Dict]:
        """Add evidence and stance information to claims"""
        enhanced_claims = []
        
        for claim in claims:
            claim_stances = [s for s in stance_results["stances"] if s["claim_id"] == claim["id"]]
            
            enhanced_claim = {
                **claim,
                "evidence": [],
                "stance_summary": {"supports": 0, "contradicts": 0, "neutral": 0}
            }
            
            for stance_info in claim_stances[:3]:  # Top 3 evidence pieces per claim
                evidence_item = next((e for e in evidence if e["url"] == stance_info["evidence_url"]), None)
                if evidence_item:
                    enhanced_claim["evidence"].append({
                        **evidence_item,
                        "stance": stance_info["stance"],
                        "confidence": stance_info["confidence"],
                        "highlights": stance_info["highlights"]
                    })
                    enhanced_claim["stance_summary"][stance_info["stance"]] += 1
            
            enhanced_claims.append(enhanced_claim)
        
        return enhanced_claims
    
    def _generate_educational_tips(self, verdict: str, stance_results: Dict) -> List[str]:
        """Generate educational tips based on analysis"""
        base_tips = [
            "Always verify information through multiple independent sources",
            "Check the credibility and expertise of the source",
            "Look for peer-reviewed research and official statements",
            "Be aware of confirmation bias - seek out opposing viewpoints"
        ]
        
        verdict_tips = {
            "False": [
                "This content contains false information - avoid sharing",
                "Look for fact-checking websites that have debunked similar claims"
            ],
            "Misleading": [
                "This content mixes truth with misleading information",
                "Pay attention to context and nuance in complex topics"
            ],
            "True": [
                "While this appears accurate, continue to verify important claims",
                "Even true information can be taken out of context"
            ],
            "Needs Review": [
                "More evidence is needed to verify these claims",
                "Consider waiting for more information before forming conclusions"
            ]
        }
        
        return base_tips + verdict_tips.get(verdict, [])
    
    async def _extract_text_from_url(self, url: str) -> str:
        """Extract text content from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()
            
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
            
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:5000]
            
        except Exception as e:
            logger.error(f"Error extracting text from URL: {e}")
            raise ValueError(f"Could not extract text from URL: {e}")
    
    async def start_training(self, dataset: List[Dict[str, str]], user_id: str) -> str:
        """Start training process"""
        training_id = str(uuid.uuid4())
        
        # Save training job to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO training_jobs (id, user_id, dataset_size, status) VALUES (?, ?, ?, ?)",
            (training_id, user_id, len(dataset), "started")
        )
        conn.commit()
        conn.close()
        
        # In a real implementation, this would start an async training process
        # For now, we'll just return the training ID
        logger.info(f"Training job {training_id} started with {len(dataset)} samples")
        
        return training_id
    
    def _get_fallback_response(self, error_msg: str) -> Dict[str, Any]:
        """Return fallback response when analysis fails"""
        return {
            "overall_verdict": "Needs Review",
            "claims": [],
            "educational_tips": [
                f"Analysis failed: {error_msg}",
                "Please verify information manually using trusted sources",
                "Check multiple independent sources",
                "Look for expert opinions and peer-reviewed research"
            ]
        }
    
    async def get_models_status(self) -> Dict[str, Any]:
        """Get status of loaded models"""
        return {
            "models_available": MODELS_AVAILABLE,
            "embedding_model": self.embedding_model is not None,
            "stance_classifier": self.stance_classifier is not None,
            "google_cse_configured": bool(self.google_api_key and self.google_cse_id),
            "database": os.path.exists(self.db_path)
        }
