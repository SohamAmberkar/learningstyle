<<<<<<< HEAD
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os
import sys
import subprocess # This is the missing import statement

# 1. Ensure shap is installed
print("Checking for shap installation...")
try:
    import shap
    print("✅ shap is already installed.")
except ImportError:
    print("shap not found. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    print("✅ shap installed successfully.")
    import shap

# 2. Load the data from your CSV file
df = pd.read_csv('../data/data_fs1.csv')

# 3. Separate features and target
X = df.drop('learning_style', axis=1)
y = df['learning_style']

# 4. Split the data to get X_test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Load the saved FSLSM models
try:
    models = joblib.load('fslsm_models.joblib')
    print("✅ FSLSM models loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'fslsm_models.joblib' not found. Please run 'train_model.py' first.")
    sys.exit(1)

# We will focus on one of the models, for example, the 'visual' model
model = models['visual']
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 6. Generate a SHAP summary plot (bar) for overall feature importance
print("Generating SHAP summary plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('SHAP Summary Plot for Visual/Verbal Model')
plt.savefig('shap_summary_plot.png', bbox_inches='tight')
plt.close()
print("✅ SHAP summary plot generated and saved as 'shap_summary_plot.png'")

# 7. Generate a SHAP force plot for a single prediction (e.g., the first test sample)
print("Generating SHAP force plot...")
sample_index = 0
shap.initjs()
shap_value_single = explainer.shap_values(X_test.iloc[[sample_index]])
force_plot_html = shap.force_plot(
    explainer.expected_value,
    shap_value_single,
    X_test.iloc[[sample_index]],
    show=False
).html()

# Save the force plot to an HTML file
with open('shap_force_plot.html', 'w') as f:
    f.write(force_plot_html)
=======
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os
import sys
import subprocess # This is the missing import statement

# 1. Ensure shap is installed
print("Checking for shap installation...")
try:
    import shap
    print("✅ shap is already installed.")
except ImportError:
    print("shap not found. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    print("✅ shap installed successfully.")
    import shap

# 2. Load the data from your CSV file
df = pd.read_csv('../data/data_fs1.csv')

# 3. Separate features and target
X = df.drop('learning_style', axis=1)
y = df['learning_style']

# 4. Split the data to get X_test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Load the saved FSLSM models
try:
    models = joblib.load('fslsm_models.joblib')
    print("✅ FSLSM models loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'fslsm_models.joblib' not found. Please run 'train_model.py' first.")
    sys.exit(1)

# We will focus on one of the models, for example, the 'visual' model
model = models['visual']
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 6. Generate a SHAP summary plot (bar) for overall feature importance
print("Generating SHAP summary plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('SHAP Summary Plot for Visual/Verbal Model')
plt.savefig('shap_summary_plot.png', bbox_inches='tight')
plt.close()
print("✅ SHAP summary plot generated and saved as 'shap_summary_plot.png'")

# 7. Generate a SHAP force plot for a single prediction (e.g., the first test sample)
print("Generating SHAP force plot...")
sample_index = 0
shap.initjs()
shap_value_single = explainer.shap_values(X_test.iloc[[sample_index]])
force_plot_html = shap.force_plot(
    explainer.expected_value,
    shap_value_single,
    X_test.iloc[[sample_index]],
    show=False
).html()

# Save the force plot to an HTML file
with open('shap_force_plot.html', 'w') as f:
    f.write(force_plot_html)
>>>>>>> a54f0a3a2ff90fb292dab62bad3424f7ccd55b48
print("✅ SHAP force plot generated and saved as 'shap_force_plot.html'")
