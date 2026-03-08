import joblib
import sys
import os

# Add current directory to path just in case
sys.path.append(os.getcwd())

print(f"Python: {sys.version}")
print(f"CWD: {os.getcwd()}")

try:
    print("Attempting import from model_definitions...")
    from model_definitions import KANClassifier, SklearnPyTorchWrapper
    print("Import successful.")
except ImportError as e:
    print(f"Import failed: {e}")

try:
    print("Loading model file...")
    models = joblib.load('sota_semi_supervised_models.joblib')
    print("✅ Model loaded successfully!")
    print("Keys:", models.keys())
    for k, v in models.items():
        print(f" - {k}: {v.get('algorithm', 'Unknown')}")
except Exception as e:
    print("❌ Model loading FAILED.")
    import traceback
    traceback.print_exc()
