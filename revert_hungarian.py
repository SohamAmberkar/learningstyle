import os
import sys
import joblib

# Add src to path so the custom classes (like SklearnPyTorchWrapper) unpickle correctly
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import warnings
warnings.filterwarnings('ignore')

model_path = 'models/sota_semi_supervised_models.joblib'
print(f"Loading {model_path}...")
trained_models = joblib.load(model_path)

print("\nReverting to Greedy Max Accuracy assignment...")
total_acc = 0
for dim, data in trained_models.items():
    # results is a dict: "AlgoName": (accuracy, model)
    all_results = data['all_results']
    
    # Find algorithm with highest accuracy
    best_algo, (best_acc, best_model) = max(all_results.items(), key=lambda x: x[1][0])
    
    # Overwrite the portfolio selection with greedy maximum
    trained_models[dim]['algorithm'] = best_algo
    trained_models[dim]['accuracy'] = best_acc
    trained_models[dim]['model'] = best_model
    total_acc += best_acc
    
    print(f"  {dim:25s} -> {best_algo:25s} (Acc: {best_acc:.4f})")

joblib.dump(trained_models, model_path)
print(f"\nDone. Saved greedy winner payload. Total accuracy: {total_acc:.4f}")
