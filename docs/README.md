# Learning Style Identification using Semi-Supervised Learning

## 🚀 Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://learning-style-identification-using-semi-supervised-learning-6.streamlit.app/)

## 📊 Features

- **4 State-of-the-Art ML Models**: KAN, TabNet, SS-VAE, and CatBoost
- **18 Behavioral Features**: Comprehensive learning style analysis
- **AI-Powered Study Plans**: Personalized roadmaps using Google Gemini AI
- **Template-Based Fallback**: Works offline without API limits
- **Interactive UI**: Beautiful Streamlit interface with radar charts

## 🎯 Learning Dimensions

- **Visual/Verbal**: How you prefer to receive information
- **Active/Reflective**: Your approach to processing information
- **Sensing/Intuitive**: Your preference for concrete vs abstract
- **Sequential/Global**: Your learning progression style

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## ▶️ Run Locally

```bash
streamlit run app.py
```

## 🔑 API Key Setup (Optional)

For AI-powered study plans, get a free Gemini API key from:
https://aistudio.google.com/app/apikey

Create a `.env` file:
```
GEMINI_API_KEY=your_key_here
```

## 📦 Project Structure

```
├── app.py              # Main Streamlit application
├── sota_models.py      # SOTA model implementations (KAN, TabNet, SS-VAE)
├── train_model.py      # Model training pipeline
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🎓 How It Works

1. **Input**: Answer 18 questions about your learning behaviors
2. **Prediction**: 4 SOTA models analyze your responses
3. **Results**: Get your learning style profile with confidence scores
4. **Study Plan**: Generate personalized learning roadmap (AI-powered)

## 🧠 Models Used

- **KAN (Kolmogorov-Arnold Networks)**: Neural architecture for complex patterns
- **TabNet**: Google's attention-based tabular learning
- **SS-VAE**: Semi-Supervised Variational Autoencoder
- **CatBoost**: Gradient boosting for categorical features

## 📄 License

MIT License

## 👤 Author

Manas Dutt
