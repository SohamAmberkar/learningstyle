import joblib
import pandas as pd
import uvicorn
import shap
import numpy as np
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# Import custom model classes to ensure pickle loading works
import model_definitions 

app = FastAPI(title="FSLSM Learning Style Predictor", version="2.0")

# Add CORS middleware to allow the frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MODEL_FILE = '../models/sota_semi_supervised_models.joblib'

# Global storage
models = {}
explainers = {}

def load_system():
    """Load SOTA models and initialize SHAP explainers"""
    global models, explainers
    
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Error: '{MODEL_FILE}' not found. Please run 'train_semi_supervised.py' first.")
        return

    try:
        # Load the dictionary of best models per dimension
        # Structure: {dim_name: {'model': model_obj, 'accuracy': float, 'algorithm': str}}
        models = joblib.load(MODEL_FILE)
        print(f"✅ Loaded SOTA models for dimensions: {list(models.keys())}")
        
        # Initialize Explainers (Best Effort)
        for name, info in models.items():
            model_wrapper = info['model']
            algo = info['algorithm']
            
            # Extract base estimator from SelfTrainingClassifier if applicable
            if hasattr(model_wrapper, 'base_estimator_'):
                estimator = model_wrapper.base_estimator_
            else:
                estimator = model_wrapper
            
            try:
                # TreeExplainer works well for XGBoost and CatBoost
                if 'XGB' in algo or 'CatBoost' in algo:
                    explainers[name] = shap.TreeExplainer(estimator)
                else:
                    # KAN, TabNet, etc. might require KernelExplainer (too slow) or DeepExplainer
                    # Skipping for performance/stability in this demo
                    explainers[name] = None
            except Exception as e:
                print(f"⚠️ Could not initialize SHAP for {name} ({algo}): {e}")
                explainers[name] = None
                
        print("✅ SHAP explainers initialized where possible.")
        
    except Exception as e:
        print(f"❌ Critical Error loading models: {e}")

# Initialize on startup
load_system()

# Define the data structure for your input
class StudentData(BaseModel):
    T_image: float = 0
    T_video: float = 0
    T_read: float = 0
    T_audio: float = 0
    T_hierarchies: float = 0
    T_powerpoint: float = 0
    T_concrete: float = 0
    T_result: float = 0
    N_standard_questions_correct: float = 0
    N_msgs_posted: float = 0
    T_solve_excercise: float = 0
    N_group_discussions: float = 0
    Skipped_los: float = 0
    N_next_button_used: float = 0
    T_spent_in_session: float = 0
    N_questions_on_details: float = 0
    N_questions_on_outlines: float = 0

# Helper function to get top reasons from SHAP values
def get_top_reasons(shap_values, feature_names, top_n=2):
    try:
        # Handle different SHAP return types
        if isinstance(shap_values, list):
            # For classification, it might return a list of arrays (one per class)
            # We usually care about the positive class (index 1) or the predicted class
            # Here we just take the one with highest magnitude sum or just the second one (Class 1)
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values

        if len(sv.shape) > 1:
            sv = sv[0] # Take first sample
            
        # Create DataFrame for easier sorting
        feature_importance = pd.DataFrame(list(zip(feature_names, sv)), columns=['feature', 'shap_value'])
        feature_importance['abs_value'] = feature_importance['shap_value'].abs()
        top_features = feature_importance.nlargest(top_n, 'abs_value')
        
        reasons = []
        for _, row in top_features.iterrows():
            impact = "positive" if row['shap_value'] > 0 else "negative"
            reasons.append(f"'{row['feature']}' had a strong {impact} impact.")
        return reasons
    except Exception as e:
        print(f"Error generating reasons: {e}")
        return ["Detailed explanation unavailable."]

@app.post("/predict")
def predict_learning_style(data: StudentData):
    if not models:
        return {"error": "Models not loaded. Service unavailable."}
    
    # Prepare input DataFrame (ensure column order matches training)
    # Note: Using dict() to preserve mapping. Training script used full feature set.
    # We must ensure the input_df has standard columns or the model might complain if untrained features are missing.
    # However, XGB/CatB usually handle this if named columns are used.
    # The training script used raw features from ../data/data_fs1.csv.
    # We must match the schema.
    # Let's hope the Pydantic model matches the training data columns! 
    # (Based on app.py logic, it seems these are the features)
    
    input_df = pd.DataFrame([data.dict()])
    
    results = {}
    
    # Dimensions mapping
    # Note: ../data/data_fs1.csv columns were dropped: 'learning_style', 'visual_verbal' etc.
    # The features remaining are the ones trained on.
    
    for name, info in models.items():
        if name not in ['visual_verbal', 'sensing_intuitive', 'active_reflective', 'sequential_global']:
             # Handle short names if used in training script keys (visual, sensing...)
             pass
             
        model = info['model']
        algo = info['algorithm']
        
        try:
            # Predict
            proba = model.predict_proba(input_df)[0]
            pred_class = int(np.argmax(proba))
            confidence = float(proba[pred_class] * 100)
            
            # Define labels (Assuming 1=First Term, 0=Second Term based on typical encoding)
            # visual_verbal: 1=Visual, 0=Verbal (Check training script mapping)
            # Mapping from training script:
            # visual_verbal: 1 if x in [0,1,2] else 0. 0,1,2 implies Visual side usually. 
            # Let's assume standard FSLSM: Visual(1)/Verbal(0)
            
            labels = {
                'visual_verbal': ('Verbal', 'Visual'), # 0, 1
                'sensing_intuitive': ('Intuitive', 'Sensing'),
                'active_reflective': ('Reflective', 'Active'),
                'sequential_global': ('Global', 'Sequential')
            }
            # Fallback for short keys
            if name == 'visual': mapping = ('Verbal', 'Visual')
            elif name == 'sensing': mapping = ('Intuitive', 'Sensing')
            elif name == 'active': mapping = ('Reflective', 'Active')
            elif name == 'sequential': mapping = ('Global', 'Sequential')
            else: mapping = labels.get(name, ('Class 0', 'Class 1'))
            
            predicted_style = mapping[1] if pred_class == 1 else mapping[0]
            
            # SHAP
            reasons = []
            if explainers.get(name):
                shap_vals = explainers[name].shap_values(input_df)
                reasons = get_top_reasons(shap_vals, input_df.columns)
            else:
                reasons = [f"Prediction by {algo} model."]

            results[name] = {
                "style": predicted_style,
                "percentage": round(confidence, 2),
                "reasons": reasons,
                "algorithm": algo
            }
            
        except Exception as e:
            print(f"Prediction error for {name}: {e}")
            results[name] = {"error": str(e)}

    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
