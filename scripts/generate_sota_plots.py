import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import joblib
from pytorch_tabnet.tab_model import TabNetClassifier
from captum.attr import IntegratedGradients
import os

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Load Data
df = pd.read_csv('../data/data_fs1.csv')
# Re-engineer features to match training
df['visual_verbal'] = df['learning_style'].apply(lambda x: 1 if x in [0, 1, 2] else 0)
df['sensing_intuitive'] = df['learning_style'].apply(lambda x: 1 if x in [1, 3, 5] else 0)
df['active_reflective'] = df['learning_style'].apply(lambda x: 1 if x in [2, 3, 4] else 0)
df['sequential_global'] = df['learning_style'].apply(lambda x: 1 if x in [0, 4, 5] else 0)

X_raw = df.drop([
    'learning_style', 'visual_verbal', 'sensing_intuitive', 
    'active_reflective', 'sequential_global'
], axis=1)

# Helper to apply FCM (copy of logic)
import skfuzzy as fuzz
def apply_fuzzy_c_means(X, n_clusters=8, fuzziness=2.0):
    data_for_fcm = X.values.astype(np.float64).T
    try:
        cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(
            data_for_fcm, c=n_clusters, m=fuzziness, error=0.005, maxiter=100
        )
        membership_grades = u.T
        fcm_features = pd.DataFrame(
            membership_grades, 
            columns=[f'FCM_M_{i+1}' for i in range(n_clusters)], 
            index=X.index
        )
        return pd.concat([X, fcm_features], axis=1)
    except:
        return X

X_enhanced = apply_fuzzy_c_means(X_raw)
feature_names = X_enhanced.columns.tolist()

# 2. Load Visual Model (TabNet)
model_path = 'tabnet_model_visual.zip' # Default save format for TabNet
if not os.path.exists(model_path):
    print("❌ Model file not found. Please run train_sota.py first.")
    exit()

clf = TabNetClassifier()
clf.load_model(model_path)

# 3. Interpret with Captum
print("Generating Captum explanations...")

# Wrapper to make TabNet compatible with Captum
# Captum expects a torch.nn.Module and tensors. TabNetClassifier is a wrapper.
# We need to access the underlying network.
torch_model = clf.network
torch_model.eval()

# Select a sample (first 5 for average)
X_sample = torch.tensor(X_enhanced.iloc[:5].values, dtype=torch.float32).to(device)

# Integrated Gradients
ig = IntegratedGradients(torch_model)

# Compute attribution
# TabNet forward returns (logits, M_loss), we need just logits for Captum
def forward_func(inputs):
    return torch_model(inputs)[0] # Extract logits

attributions, delta = ig.attribute(X_sample, target=0, return_convergence_delta=True)
attributions = attributions.cpu().detach().numpy()

# Average attributions across samples
mean_attributions = np.mean(attributions, axis=0)

# 4. Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_names, mean_attributions)
plt.xlabel("Average Integrated Gradients Attribution")
plt.title("Feature Importance (Captum - Integrated Gradients)")
plt.tight_layout()
plt.savefig('sota_captum_plot.png')
print("✅ Captum plot saved to sota_captum_plot.png")
