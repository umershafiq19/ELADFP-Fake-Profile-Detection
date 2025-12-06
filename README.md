# ELADFP: Social Media Fake Profile Detection


> **Ensemble Learning Approach for Detecting Fake Profiles on Instagram**

An intelligent system that combines Random Forest, Logistic Regression, and XGBoost to identify fraudulent social media accounts with high accuracy. Built with modern MLOps practices including experiment tracking, model versioning, and production-ready deployment.

---

## 📋 Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Performance](#model-performance)
- [MLflow & DVC](#mlflow--dvc)
- [Project Structure](#project-structure)
- [Team](#team)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

- 🤖 **Ensemble Machine Learning**: Combines RF, LR, and XGBoost for robust predictions
- 🎨 **Interactive Web Interface**: Beautiful, responsive UI built with Tailwind CSS
- 📊 **Real-time Feature Calculation**: Automatic computation of derived features
- 🔍 **AI-Powered Explanations**: Human-readable reasoning using Google Gemini API
- 📈 **Experiment Tracking**: MLflow integration for model versioning and monitoring
- 💾 **Data Version Control**: DVC for dataset and model management
- ⚡ **RESTful API**: Flask backend with modular architecture
- 🎯 **High Accuracy**: Ensemble approach achieves superior performance
- 🔒 **Input Validation**: Comprehensive error handling and data validation

---

## 🎬 Demo

### Web Interface


### Sample Prediction
```json
{
  "prediction": {
    "is_fake": 1
  },
  "confidence": {
    "real_profile_prob": 0.081,
    "fake_profile_prob": 0.919
  },
  "reasoning": "High following/follower ratio with minimal posts suggests bot behavior...",
  "message": "⚠️ FAKE PROFILE DETECTED!"
}
```

---

## 🛠 Technology Stack

### Backend
- **Flask 2.3.0** - Web framework
- **Scikit-learn 1.2.0** - Machine learning
- **XGBoost 1.7.6** - Gradient boosting
- **Pandas 1.5.3** - Data manipulation
- **NumPy 1.24.0** - Numerical computing

### MLOps
- **MLflow 2.0.0** - Experiment tracking
- **DVC 3.0.0** - Data version control
- **Git & GitHub** - Code versioning

### Frontend
- **HTML5 & CSS3** - Structure and styling
- **Tailwind CSS 3.0** - Utility-first CSS
- **Vanilla JavaScript** - Client-side logic

### AI Integration
- **Google Generative AI 0.3.0** - Explanation generation

---

## 🏗 Architecture

```
┌─────────────────┐
│  Web Interface  │
│   (Frontend)    │
└────────┬────────┘
         │
         ↓
┌─────────────────┐      ┌──────────────┐
│   Flask API     │─────→│   MLflow     │
│   (Backend)     │      │  Tracking    │
└────────┬────────┘      └──────────────┘
         │
         ↓
┌─────────────────┐
│  Feature Eng.   │
│  Preprocessing  │
└────────┬────────┘
         │
         ↓
┌─────────────────┐      ┌──────────────┐
│ Ensemble Model  │←─────│     DVC      │
│  RF + LR + XGB  │      │ Model Store  │
└────────┬────────┘      └──────────────┘
         │
         ↓
┌─────────────────┐
│  AI Reasoning   │
│  (Gemini API)   │
└─────────────────┘
```

---

## 📥 Installation

### Prerequisites
- Python 3.11+
- Git
- pip (Python package manager)

### Step 1: Clone Repository
```bash
git clone https://github.com/umershafiq19/ELADFP-Fake-Profile-Detection.git
cd ELADFP-Fake-Profile-Detection
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Pull Model with DVC
```bash
dvc pull
```

### Step 5: Configure Environment Variables
Create `backend/.env` file:
```env
FLASK_ENV=development
FLASK_DEBUG=True
GEMINI_API_KEY=your_gemini_api_key_here
API_HOST=0.0.0.0
API_PORT=5001
```

---

## 🚀 Usage

### Start Flask Backend
```bash
# From project root
python -m backend.app
```

Backend will run on: `http://localhost:5001`

### Open Frontend
Simply open `frontend/index.html` in your web browser, or:
```bash
# Windows
start frontend/index.html

# Linux/Mac
open frontend/index.html
```

### Make Predictions

**Via Web Interface:**
1. Fill in profile information
2. Click "Predict Profile"
3. View results with confidence scores and AI explanation

**Via API:**
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "profile pic": 1,
    "nums/length username": 0.0,
    "fullname words": 2,
    "nums/length fullname": 0.0,
    "name==username": 0,
    "description length": 24,
    "external URL": 0,
    "private": 0,
    "#posts": 3,
    "#followers": 45,
    "#following": 4693
  }'
```

---

## 📚 API Documentation

### Endpoints

#### `POST /predict`
Predict if a profile is fake or real.

**Request Body:**
```json
{
  "profile pic": 1,
  "nums/length username": 0.0,
  "fullname words": 2,
  "nums/length fullname": 0.0,
  "name==username": 0,
  "description length": 24,
  "external URL": 0,
  "private": 0,
  "#posts": 3,
  "#followers": 45,
  "#following": 4693
}
```

**Response:**
```json
{
  "prediction": {"is_fake": 1},
  "confidence": {
    "real_profile_prob": 0.081,
    "fake_profile_prob": 0.919
  },
  "reasoning": "AI-generated explanation...",
  "message": "⚠️ FAKE PROFILE DETECTED!"
}
```

#### `GET /health`
Check API health status.

**Response:**
```json
{
  "status": "ELADFP Flask API Running",
  "model_loaded": true,
  "timestamp": "2025-12-06T20:00:00"
}
```

#### `GET /`
API documentation and metadata.

---

## 📊 Model Performance

### Ensemble Components
- **Random Forest**: 100 estimators, max_depth=10
- **Logistic Regression**: C=1.0, max_iter=1000
- **XGBoost**: 100 estimators, max_depth=5, learning_rate=0.1

### Key Features
1. Profile picture presence
2. Username numeric ratio
3. Fullname word count
4. Bio length
5. Followers-to-following ratio
6. Posts per follower
7. Log-transformed engagement metrics



## 🔬 MLflow & DVC

### View MLflow Experiments
```bash
# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns

# Open browser
http://localhost:5000
```

**Or programmatically:**
```bash
python scripts/view_mlflow_data.py
```

### DVC Operations

**Pull data/models:**
```bash
dvc pull
```

**Add new data:**
```bash
dvc add training/data/new_dataset.csv
git add training/data/new_dataset.csv.dvc
git commit -m "Add new dataset"
git push
```

**Check status:**
```bash
dvc status
```

---

## 📁 Project Structure

```
ELADFP-Fake-Profile-Detection/
├── backend/
│   ├── app.py                          # Main Flask application
│   ├── models/
│   │   └── best_eladfp_model.pkl      # Trained ensemble model
│   ├── routes/
│   │   ├── predict.py                 # Prediction endpoints
│   │   ├── health.py                  # Health checks
│   │   └── mlflow_logging.py          # MLflow integration
│   ├── preprocessing/
│   │   ├── feature_calculator.py      # Feature engineering
│   │   └── validators.py              # Input validation
│   └── ai_reasoning/
│       └── gemini_analyzer.py         # AI explanations
├── frontend/
│   └── index.html                     # Web interface
├── config/
│   └── mlflow_config.yaml            # MLflow configuration
├── scripts/
│   ├── register_existing_model.py    # Model registration
│   └── view_mlflow_data.py           # View MLflow data
├── training/
│   ├── data/                         # Datasets (tracked by DVC)
│   └── mlflow_trainer.py             # Training script
├── mlruns/                           # MLflow artifacts
├── .dvc/                             # DVC configuration
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
```

---

## 👥 Team

**Group Members:**
- **Umer Shafiq** (21L-5401) - [GitHub](https://github.com/umershafiq19)
- **Ibtisam Ali** (21L-5187)

**Instructor:**
- **Usman Anwer**

**Institution:** FAST-NUCES Lahore

**Course:** Machine Learning / Data Science

**Year:** 2024-2025

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset: [Instagram Fake Spammer Genuine Accounts Dataset](https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-accounts)
- Google Gemini API for AI explanations
- MLflow & DVC communities for excellent tools
- Flask and Scikit-learn documentation

---

## 📞 Contact

For questions or feedback, please reach out:

- **Email**: [umershafiq603@gmail.com]
- **GitHub Issues**: [Create an issue](https://github.com/umershafiq19/ELADFP-Fake-Profile-Detection/issues)
- **LinkedIn**: [[Your LinkedIn](https://www.linkedin.com/in/umershafiq9/)]

---

## 🔮 Future Enhancements

- [ ] Deploy to AWS (EC2 + S3)
- [ ] Add authentication system
- [ ] Implement batch prediction UI
- [ ] Real-time profile scraping
- [ ] Network analysis features
- [ ] Multi-platform support (Twitter, Facebook)
- [ ] Mobile app version
- [ ] Docker containerization

---

<div align="center">
  <p>Made with ❤️ by ELADFP Team</p>
  <p>⭐ Star this repo if you find it helpful!</p>
</div>
