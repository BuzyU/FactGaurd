# FactGuard - AI-Powered Misinformation Detection

FactGuard is a comprehensive **AI-powered misinformation detection and education tool** that helps users verify the authenticity of information through advanced machine learning techniques and evidence-based analysis.

---

## 🚀 Features

* **AI-Powered Analysis** – Extract key claims, retrieve evidence, and detect misinformation using state-of-the-art NLP models
* **Multi-Source Verification** – Cross-reference claims with reliable sources and fact-checking databases
* **Risk Assessment** – Get clear risk levels (Low / Medium / High) with detailed explanations
* **Educational Tips** – Learn how to verify information and improve media literacy skills
* **Firebase Authentication** – Secure login with Email/Password and Google Sign-In
* **Real-time Analysis** – Fast processing with caching for improved performance
* **Modern UI** – Beautiful, responsive interface built with React and Tailwind CSS

---

## 🏗️ Project Structure

```
FactGuard/
├── backend/                 # FastAPI backend with AI pipeline
│   ├── main.py              # FastAPI application entry point
│   ├── ai_pipeline.py       # Core AI analysis pipeline
│   ├── auth_middleware.py   # Firebase authentication middleware
│   └── requirements.txt     # Python dependencies
├── frontend/                # React frontend application
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── contexts/        # React contexts (Auth)
│   │   ├── firebase/        # Firebase configuration
│   │   └── main.jsx         # Application entry point
│   ├── package.json         # Node.js dependencies
│   └── index.html           # HTML template
├── scripts/                 # AI training and utilities
│   ├── train.py             # Model training script
│   └── requirements.txt     # Training dependencies
└── README.md                # This file
```

---

## 🛠️ Tech Stack

### Backend

* **FastAPI** – High-performance Python web framework
* **Transformers** – Hugging Face models for NLP
* **Sentence Transformers** – Semantic similarity & embeddings
* **Firebase Admin** – Authentication and user management
* **SQLite** – Local database for caching and logs
* **BeautifulSoup** – Content extraction from URLs

### Frontend

* **React 18** – Modern React with hooks & context
* **Vite** – Next-gen frontend tooling
* **Tailwind CSS** – Utility-first styling
* **Firebase** – Authentication (Email/Password + Google)
* **Axios** – API communication
* **React Router** – Client-side routing
* **Lucide React** – Icons

### AI/ML

* **BART** – Zero-shot stance detection
* **MiniLM** – Lightweight embeddings
* **Custom Models** – Fine-tuned domain-specific models

---

## 📋 Prerequisites

* **Python 3.8+**
* **Node.js 16+**
* **npm or yarn**
* **Git**

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd FactGuard
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
# Activate virtual environment
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Backend will be running at `http://localhost:8000`

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend will be available at `http://localhost:3000`

### 4. Firebase Configuration

1. Create a Firebase project at [Firebase Console](https://console.firebase.google.com/)
2. Enable Email/Password & Google authentication
3. Update `frontend/src/firebase/config.js` with your keys
4. Download the service account key → save as `backend/firebase-service-account.json`

---

## 🔧 Configuration

Create `.env` in **backend/**:

```env
ENVIRONMENT=development
FIREBASE_PROJECT_ID=factguard-5f9f2
BING_API_KEY=your_bing_api_key_here
SERP_API_KEY=your_serp_api_key_here
```

---

## 🧠 AI Pipeline

1. **Claim Extraction** – Identify factual claims using NLP heuristics
2. **Evidence Retrieval** – Retrieve supporting/contradicting sources
3. **Semantic Reranking** – Rank evidence by relevance
4. **Stance Detection** – Classify stance using `facebook/bart-large-mnli`
5. **Heuristic Analysis** – Detect clickbait, check credibility
6. **Risk Assessment** – Aggregate results into Low/Medium/High

---

## 📊 API Endpoints

### `POST /analyze`

Analyze text or URL.

**Request:**

```json
{
  "text": "Text to analyze",
  "url": "https://example.com/article"
}
```

**Response:**

```json
{
  "overall_risk": "medium",
  "claims": [...],
  "evidence": [...],
  "teach_tips": [...],
  "processing_time": 2.34
}
```

### `GET /health`

Health check endpoint.

### `GET /models/status`

Check status of AI models.

---

## 🧪 Testing

### Backend

```bash
cd backend
pytest
```

### Frontend

```bash
cd frontend
npm test
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add feature"`
4. Push: `git push origin feature-name`
5. Open a PR

**Guidelines:**

* Python → PEP 8 + docstrings
* JavaScript → ESLint + Prettier
* Add tests for new features
* Update API docs if changed

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

* Hugging Face 🤗
* Firebase 🔥
* React & Vite ⚛️
* Tailwind CSS 🎨
* All contributors 🙌

---

**FactGuard** – Empowering users with AI-driven fact-checking and media literacy.

---

FactGuard – Empowering users with AI-driven fact-checking and media literacy.
👉 Try it here: https://factgaurd.onrender.com/

---
