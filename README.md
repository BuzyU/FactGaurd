# FactGuard - AI-Powered Misinformation Detection

FactGuard is a comprehensive AI-powered misinformation detection and education tool that helps users verify information authenticity through advanced machine learning techniques and evidence-based analysis.

## 🚀 Features

- **AI-Powered Analysis**: Extract key claims, retrieve evidence, and detect misinformation using state-of-the-art NLP models
- **Multi-Source Verification**: Cross-reference claims with reliable sources and fact-checking databases
- **Risk Assessment**: Get clear risk levels (Low/Medium/High) with detailed explanations
- **Educational Tips**: Learn how to verify information and develop media literacy skills
- **Firebase Authentication**: Secure login with email/password and Google Sign-In
- **Real-time Analysis**: Fast processing with caching for improved performance
- **Modern UI**: Beautiful, responsive interface built with React and Tailwind CSS

## 🏗️ Architecture

```
FactGuard/
├── backend/                 # FastAPI backend with AI pipeline
│   ├── main.py             # FastAPI application entry point
│   ├── ai_pipeline.py      # Core AI analysis pipeline
│   ├── auth_middleware.py  # Firebase authentication middleware
│   └── requirements.txt    # Python dependencies
├── frontend/               # React frontend application
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── contexts/       # React contexts (Auth)
│   │   ├── firebase/       # Firebase configuration
│   │   └── main.jsx        # Application entry point
│   ├── package.json        # Node.js dependencies
│   └── index.html          # HTML template
├── scripts/                # AI training and utilities
│   ├── train.py           # Model training script
│   └── requirements.txt   # Training dependencies
└── README.md              # This file
```

## 🛠️ Tech Stack

### Backend
- **FastAPI**: High-performance Python web framework
- **Transformers**: Hugging Face transformers for NLP models
- **Sentence Transformers**: Semantic similarity and embeddings
- **Firebase Admin**: Authentication and user management
- **SQLite**: Local database for caching and logging
- **BeautifulSoup**: Web scraping for URL content extraction

### Frontend
- **React 18**: Modern React with hooks and context
- **Vite**: Fast build tool and development server
- **Tailwind CSS**: Utility-first CSS framework
- **Firebase**: Authentication (email/password + Google)
- **Axios**: HTTP client for API communication
- **React Router**: Client-side routing
- **Lucide React**: Beautiful icon library

### AI/ML
- **BART**: Zero-shot classification for stance detection
- **MiniLM**: Lightweight embeddings for semantic similarity
- **Custom Models**: Fine-tunable models for domain-specific tasks

## 📋 Prerequisites

- **Python 3.8+**
- **Node.js 16+**
- **npm or yarn**
- **Git**

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd FactGuard
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the backend server
python main.py
```

The backend will be available at `http://localhost:8000`

### 3. Frontend Setup

```bash
# Navigate to frontend directory (in a new terminal)
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

### 4. Firebase Configuration

The Firebase configuration is already included in the project. For production deployment, you may want to:

1. Create your own Firebase project at [Firebase Console](https://console.firebase.google.com/)
2. Enable Authentication with Email/Password and Google providers
3. Update the configuration in `frontend/src/firebase/config.js`
4. Download the service account key and place it as `backend/firebase-service-account.json`

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
ENVIRONMENT=development
FIREBASE_PROJECT_ID=factguard-5f9f2
BING_API_KEY=your_bing_api_key_here  # Optional: for real web search
SERP_API_KEY=your_serp_api_key_here  # Optional: for real web search
```

### Model Configuration

The AI pipeline uses the following models by default:
- **Embeddings**: `all-MiniLM-L6-v2`
- **Stance Detection**: `facebook/bart-large-mnli`
- **Claim Extraction**: Rule-based with NLP heuristics

## 🧠 AI Pipeline Details

### 1. Claim Extraction
- Identifies factual claims from input text
- Filters out opinions and subjective statements
- Uses linguistic patterns and NLP heuristics

### 2. Evidence Retrieval
- Searches for relevant evidence (currently mock data)
- Can be extended with Bing Search API or SerpAPI
- Ranks sources by credibility and relevance

### 3. Semantic Reranking
- Uses sentence transformers for semantic similarity
- Reorders evidence by relevance to claims
- Improves quality of evidence matching

### 4. Stance Detection
- Determines if evidence supports, contradicts, or is neutral to claims
- Uses zero-shot classification with BART model
- Provides confidence scores for predictions

### 5. Heuristic Analysis
- Detects clickbait language patterns
- Evaluates source credibility
- Checks for misinformation indicators

### 6. Risk Assessment
- Combines all analysis results
- Generates overall risk score (Low/Medium/High)
- Provides detailed explanations

## 🎯 Training Custom Models

### Prepare Training Data

Create training data in JSON format:

```json
// stance_data.json
[
  {
    "claim": "Vaccines are safe and effective",
    "evidence": "Clinical trials show 95% efficacy",
    "label": "supports"
  }
]

// claims_data.json
[
  {
    "text": "Scientists have proven that water is wet",
    "label": "claim"
  }
]

// misinfo_data.json
[
  {
    "text": "Peer-reviewed study shows promising results",
    "label": "reliable"
  }
]
```

### Train Models

```bash
# Navigate to scripts directory
cd scripts

# Install training dependencies
pip install -r requirements.txt

# Create sample data (for testing)
python train.py --create_sample_data

# Train stance detection model
python train.py --task stance --data_path ./data/stance_data.json --epochs 5

# Train claim extraction model
python train.py --task claim_extraction --data_path ./data/claims_data.json --epochs 3

# Train misinformation classifier
python train.py --task misinformation --data_path ./data/misinfo_data.json --epochs 4
```

### Training Options

```bash
python train.py \
  --task stance \
  --data_path ./data/stance_data.json \
  --output_dir ./models \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --use_wandb  # Optional: for experiment tracking
```

## 🚀 Deployment

### Local Deployment

1. **Backend**: Run `python main.py` in the backend directory
2. **Frontend**: Run `npm run build && npm run preview` in the frontend directory

### Cloud Deployment Options

#### Option 1: Render
```bash
# Backend deployment
# 1. Connect your GitHub repo to Render
# 2. Set build command: pip install -r requirements.txt
# 3. Set start command: python main.py
# 4. Set environment variables

# Frontend deployment
# 1. Set build command: npm install && npm run build
# 2. Set publish directory: dist
```

#### Option 2: Fly.io
```bash
# Install flyctl
# Create fly.toml configuration
# Deploy with: fly deploy
```

#### Option 3: Google Cloud Run
```bash
# Create Dockerfile
# Build and push to Container Registry
# Deploy to Cloud Run
```

### Environment Setup for Production

```env
ENVIRONMENT=production
FIREBASE_PROJECT_ID=your-project-id
BING_API_KEY=your-bing-api-key
ALLOWED_ORIGINS=https://your-frontend-domain.com
```

## 📊 API Documentation

### Authentication
All API endpoints require Firebase authentication. Include the ID token in the Authorization header:

```
Authorization: Bearer <firebase-id-token>
```

### Endpoints

#### `POST /analyze`
Analyze text or URL for misinformation.

**Request:**
```json
{
  "text": "Text to analyze",  // Optional
  "url": "https://example.com/article"  // Optional
}
```

**Response:**
```json
{
  "overall_risk": "medium",
  "explanations": ["2 sources contradict the main claim"],
  "claims": [
    {
      "id": 0,
      "text": "Extracted claim text",
      "confidence": 0.85,
      "type": "factual"
    }
  ],
  "evidence": [
    {
      "title": "Evidence title",
      "snippet": "Evidence snippet",
      "url": "https://source.com",
      "domain": "source.com",
      "credibility_score": 0.9,
      "relevance_score": 0.8
    }
  ],
  "teach_tips": [
    "Always check multiple sources",
    "Look for author credentials"
  ],
  "processing_time": 2.34
}
```

#### `GET /health`
Health check endpoint.

#### `GET /models/status`
Get status of loaded AI models.

## 🧪 Testing

### Backend Testing
```bash
cd backend
python -m pytest tests/  # If tests are implemented
```

### Frontend Testing
```bash
cd frontend
npm test
```

### Manual Testing
1. Start both backend and frontend
2. Register a new account or login
3. Test text analysis with sample misinformation
4. Test URL analysis with news articles
5. Verify risk assessments and explanations

## 🔍 Troubleshooting

### Common Issues

#### Backend Issues
- **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
- **Firebase errors**: Check Firebase configuration and service account key
- **Model loading errors**: Models will download automatically on first run (may take time)

#### Frontend Issues
- **Build errors**: Clear node_modules and reinstall with `rm -rf node_modules && npm install`
- **Authentication errors**: Verify Firebase configuration
- **API connection errors**: Ensure backend is running on port 8000

#### Performance Issues
- **Slow analysis**: First run downloads models (1-2GB), subsequent runs are faster
- **Memory issues**: Reduce batch size in model configuration
- **Cache issues**: Clear SQLite database file `factguard.db`

### Debug Mode
Set `ENVIRONMENT=development` in backend `.env` file for detailed logging and authentication bypass.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit your changes: `git commit -am 'Add feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

### Development Guidelines
- Follow PEP 8 for Python code
- Use ESLint and Prettier for JavaScript/React code
- Add docstrings to Python functions
- Write unit tests for new features
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for transformer models
- Firebase for authentication services
- React and Vite communities
- Tailwind CSS for styling utilities
- All contributors and testers

## 📞 Support

For support, please:
1. Check the troubleshooting section
2. Search existing GitHub issues
3. Create a new issue with detailed information
4. Join our community discussions

---

**FactGuard** - Empowering users to make informed decisions through AI-powered fact-checking and media literacy education.
"# FactGaurd" 
#   F a c t G a u r d  
 #   F a c t G a u r d  
 