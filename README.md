# Learning Style Identification via CPL-LS

This project implements a **Semi-Supervised Curriculum Pseudo-Labeling (CPL-LS)** pipeline to identify Felder-Silverman Learning Styles (FSLSM) from student behavior data (e.g., Moodle VLE interactions). It includes an automated machine learning training pipeline with Optuna hyperparameter tuning, model ensembling (XGBoost, CatBoost, KAN, TabNet), and an interactive **Explainable AI (XAI)** Streamlit dashboard for panel presentations.

## 📂 Project Structure

```text
├── data/
│   └── data_fs1.csv                 # Raw student behavioral dataset with ground truth labels
├── docs/                            
│   └── cpl_process_diagram.png      # Architectural diagram of the pipeline
├── models/
│   └── sota_semi_supervised_models.joblib # Trained models, test splits, and metadata
├── src/
│   ├── app.py                       # The 5-tab Streamlit Dashboard
│   ├── model_definitions.py         # PyTorch KAN implementations & wrappers
│   └── train_semi_supervised.py     # Core training pipeline with Optuna & CPL-LS algorithm
├── curriculum_self_training.py      # The custom CPL-LS module implementation
├── requirements.txt                 # Pinned Python dependencies
└── README.md                        # This file
```

---

## 🚀 Setup Instructions (Windows)

### 1. Create a Virtual Environment
It is highly recommended to run this project in an isolated virtual environment to prevent dependency conflicts.

Open Windows PowerShell or Command Prompt in the project folder and run:
```powershell
python -m venv venv
```

### 2. Activate the Virtual Environment
Activate the environment before installing any packages.
```powershell
# In PowerShell:
.\venv\Scripts\Activate.ps1

# In Command Prompt (cmd.exe):
.\venv\Scripts\activate.bat
```
*(You should see `(venv)` appear at the start of your terminal prompt).*

### 3. Install Dependencies
With the environment active, install all required machine learning and web UI libraries:
```powershell
pip install -r requirements.txt
```

---

## 🧠 1. Running the Training Pipeline

Before launching the dashboard, you must train the models. The pipeline will automatically apply SMOTE for class imbalance, perform Fuzzy C-Means feature augmentation, tune hyperparameters using Optuna (Hungarian algorithm portfolio matching), and train the CPL-LS engines.

Run the training script:
```powershell
python src/train_semi_supervised.py
```
*Note: This process may take several minutes as it trains XGBoost, CatBoost, PyTorch KAN, and TabNet across 10-15 Optuna trials for each of the 4 learning style dimensions.*

You will know it is complete when you see: `✅ CPL-LS pipeline complete. Models saved to ...\models\sota_semi_supervised_models.joblib`

---

## 📊 2. Launching the XAI Dashboard

Once the models are trained and saved, you can launch the interactive Streamlit dashboard. 

Run this command:
```powershell
streamlit run src/app.py
```

This will automatically open a new tab in your default web browser (usually `http://localhost:8501`).

### Dashboard Features for Panel Presentations:
1. **Predict (Student View):** Interactive sliders to simulate a student's VLE behavior and predict their learning style. Supports batch CSV student uploads.
2. **Model Transparency:** View the exact cross-validation accuracy of all participating algorithms and visually track the dynamic Curriculum Pseudo-Labeling threshold evolution.
3. **Explainability (SHAP):** Live generation of Shapley additive explanations (global feature importance) to prove the physiological basis of the predictions.
4. **Dataset & Features:** Defense of the `data_fs1.csv` distribution and the necessity of SMOTE for handling class imbalance.
5. **CPL-LS Architecture:** Mathematical breakdown of the dynamic threshold pacing functions and architectural diagrams.
