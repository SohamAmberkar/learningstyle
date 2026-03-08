# 🚀 Streamlit Cloud Deployment Guide

## Step-by-Step Instructions

### 1. **Go to Streamlit Cloud**
Visit: https://share.streamlit.io/

### 2. **Sign in with GitHub**
- Click "Sign in with GitHub"
- Authorize Streamlit to access your repositories

### 3. **Deploy New App**
- Click "New app" button
- Fill in the form:
  - **Repository**: `manasdutt2003/Learning-Style-Identification-using-semi-supervised-learning`
  - **Branch**: `main`
  - **Main file path**: `app.py`

### 4. **Advanced Settings (Optional)**
Click "Advanced settings" to:
- Add **Secrets** (for Gemini API key):
  ```toml
  GEMINI_API_KEY = "your_api_key_here"
  ```
- Set Python version: `3.10` or `3.11`

### 5. **Deploy!**
- Click "Deploy"
- Wait 2-5 minutes for deployment
- Your app will be live at: `https://learning-style-identification-using-semi-supervised-learning-6.streamlit.app/`

## 🔑 Adding Gemini API Key (Recommended)

In the "Advanced settings" → "Secrets" section, add:

```toml
GEMINI_API_KEY = "AIzaSyDBgISdC9dlz0uWP5trjx13vL7Va53TlOQ"
```

Then update `app.py` to read from Streamlit secrets:

```python
# Replace this line in app.py:
os.environ.get('GEMINI_API_KEY', '')

# With:
st.secrets.get('GEMINI_API_KEY', os.environ.get('GEMINI_API_KEY', ''))
```

## 📝 Notes

- **Free tier**: 1 GB RAM, shared CPU
- **Model files**: Upload `.joblib` files via GitHub LFS if needed (or retrain on cloud)
- **Deployment time**: Usually 2-5 minutes
- **Auto-updates**: Pushes to `main` branch trigger redeployment

## ⚠️ Important

Since we excluded `.joblib` model files from Git, you have two options:

**Option A**: Upload models via GitHub LFS
```bash
git lfs install
git lfs track "*.joblib"
git add .gitattributes
git add *.joblib
git commit -m "Add model files"
git push
```

**Option B**: Retrain models on first run (add training logic in app startup)

## 🎉 You're Done!

Your app will be publicly accessible and automatically update when you push to GitHub!
