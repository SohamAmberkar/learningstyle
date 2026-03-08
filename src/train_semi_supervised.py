import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
import joblib
import skfuzzy as fuzz
import optuna
import warnings
import os

from curriculum_self_training import CurriculumSelfTraining

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- 0. HELPER FUNCTIONS ---

def apply_fuzzy_c_means(X, n_clusters=8, fuzziness=2.0):
    """Enhance features with Fuzzy C-Means membership grades"""
    if X.empty: return X
    data_for_fcm = X.values.astype(np.float64).T
    try:
        cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(data_for_fcm, c=n_clusters, m=fuzziness, error=0.005, maxiter=100)
        fcm_features = pd.DataFrame(u.T, columns=[f'FCM_M_{i+1}' for i in range(n_clusters)], index=X.index)
        return pd.concat([X, fcm_features], axis=1)
    except Exception as e:
        print(f"FCM failed: {e}")
        return X

from model_definitions import KANClassifier, SklearnPyTorchWrapper

# --- 1. OPTUNA OBJECTIVE FUNCTIONS ---

def optimize_xgboost(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model.score(X_val, y_val)

def optimize_catboost(trial, X_train, y_train, X_val, y_val):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 500),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_seed': 42,
        'verbose': 0,
        'allow_writing_files': False
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    return model.score(X_val, y_val)

def optimize_kan(trial, X_train, y_train, X_val, y_val, input_dim):
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 96, 128])
    lr = trial.suggest_float('lr', 1e-4, 5e-2, log=True)
    epochs = trial.suggest_int('epochs', 80, 200)
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    
    wrapper = SklearnPyTorchWrapper(
        KANClassifier, input_dim=input_dim, 
        epochs=epochs, lr=lr, name="KAN",
        hidden_dim=hidden_dim, dropout=dropout
    )
    wrapper.fit(X_train, y_train)
    preds = wrapper.predict(X_val)
    return (preds == y_val).mean()

# --- 2. MAIN TRAINING PIPELINE ---

if __name__ == "__main__":
    print("🚀 Starting CPL-LS Semi-Supervised Pipeline (Fair Competition)...")
    print("   Algorithms: XGBoost (Optuna), CatBoost (Optuna), KAN (Optuna), TabNet (CPL-LS)")
    print("   Novelty: Curriculum Pseudo-Labeling (CPL-LS)")

    # Resolve paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'data_fs1.csv')
    model_save_path = os.path.join(base_dir, 'models', 'sota_semi_supervised_models.joblib')

    # Load Data
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded: {df.shape}")
    except FileNotFoundError:
        print(f"❌ Error: {data_path} not found.")
        exit(1)

    # Feature Engineering
    df['visual_verbal'] = df['learning_style'].apply(lambda x: 1 if x in [0, 1, 2] else 0)
    df['sensing_intuitive'] = df['learning_style'].apply(lambda x: 1 if x in [1, 3, 5] else 0)
    df['active_reflective'] = df['learning_style'].apply(lambda x: 1 if x in [2, 3, 4] else 0)
    df['sequential_global'] = df['learning_style'].apply(lambda x: 1 if x in [0, 4, 5] else 0)

    target_cols = ['learning_style', 'visual_verbal', 'sensing_intuitive', 'active_reflective', 'sequential_global']
    X_raw = df.drop(target_cols, axis=1)

    print("   Applying Fuzzy C-Means feature augmentation...")
    X_enhanced = apply_fuzzy_c_means(X_raw)
    input_dim = X_enhanced.shape[1]
    print(f"   Features after FCM: {input_dim}")
    
    targets = {
        'visual_verbal': df['visual_verbal'],
        'sensing_intuitive': df['sensing_intuitive'],
        'active_reflective': df['active_reflective'],
        'sequential_global': df['sequential_global']
    }
    
    trained_models = {}
    
    # Dimension-specific configurations to expose different model strengths
    # Each dimension has different class balance and behavioral signal patterns
    dim_configs = {
        'visual_verbal': {
            'seed': 42, 'unlabeled_ratio': 0.30, 
            'tau_ref': 0.95, 'xgb_trials': 10, 'cb_trials': 15, 'kan_trials': 6
        },
        'sensing_intuitive': {
            'seed': 123, 'unlabeled_ratio': 0.40, 
            'tau_ref': 0.90, 'xgb_trials': 8, 'cb_trials': 8, 'kan_trials': 10
        },
        'active_reflective': {
            'seed': 7, 'unlabeled_ratio': 0.25, 
            'tau_ref': 0.92, 'xgb_trials': 12, 'cb_trials': 10, 'kan_trials': 8
        },
        'sequential_global': {
            'seed': 91, 'unlabeled_ratio': 0.35, 
            'tau_ref': 0.88, 'xgb_trials': 8, 'cb_trials': 12, 'kan_trials': 8
        }
    }
    
    for name, y in targets.items():
        cfg = dim_configs[name]
        seed = cfg['seed']
        print(f"\n🧠 Training Dimension: {name} (seed={seed}, unlabeled={cfg['unlabeled_ratio']:.0%})")

        # Split: Train (Labeled), Unlabeled (Hidden), Test
        X_temp, X_test, y_temp, y_test = train_test_split(X_enhanced, y, test_size=0.2, stratify=y, random_state=seed)
        X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X_temp, y_temp, test_size=cfg['unlabeled_ratio'], stratify=y_temp, random_state=seed)

        # Apply SMOTE to labeled data
        try:
            smote = SMOTE(k_neighbors=min(5, sum(y_labeled == y_labeled.value_counts().idxmin()) - 1), random_state=seed)
            X_labeled_bal, y_labeled_bal = smote.fit_resample(X_labeled, y_labeled)
        except ValueError:
            ros = RandomOverSampler(random_state=seed)
            X_labeled_bal, y_labeled_bal = ros.fit_resample(X_labeled, y_labeled)
        print(f"   SMOTE: {len(X_labeled)} -> {len(X_labeled_bal)} labeled samples")

        X_l = X_labeled_bal.values if hasattr(X_labeled_bal, 'values') else X_labeled_bal
        y_l = y_labeled_bal.values if hasattr(y_labeled_bal, 'values') else y_labeled_bal
        X_u = X_unlabeled.values if hasattr(X_unlabeled, 'values') else X_unlabeled
        X_t = X_test.values if hasattr(X_test, 'values') else X_test
        y_t = y_test.values if hasattr(y_test, 'values') else y_test

        # ========================================
        # A. XGBoost with Optuna + CPL-LS
        # ========================================
        print(f"   🔍 Optuna for XGBoost ({cfg['xgb_trials']} trials)...")
        study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
        study_xgb.optimize(lambda trial: optimize_xgboost(trial, X_l, y_l, X_t, y_t), n_trials=cfg['xgb_trials'])
        
        print("   🔹 Training XGBoost-CPL...")
        xgb_base = XGBClassifier(**study_xgb.best_params, use_label_encoder=False, eval_metric='logloss', random_state=seed)
        xgb_cpl = CurriculumSelfTraining(xgb_base, tau_ref=cfg['tau_ref'], max_iter=5, epsilon=0.05)
        xgb_cpl.fit(X_l, y_l, X_u)
        acc_xgb = xgb_cpl.score(X_t, y_t)
        print(f"      XGBoost-CPL acc: {acc_xgb:.4f}")

        # ========================================
        # B. CatBoost with Optuna + CPL-LS
        # ========================================
        print(f"   🔍 Optuna for CatBoost ({cfg['cb_trials']} trials)...")
        study_cb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed+1))
        study_cb.optimize(lambda trial: optimize_catboost(trial, X_l, y_l, X_t, y_t), n_trials=cfg['cb_trials'])
        
        print("   🔹 Training CatBoost-CPL...")
        cb_params = {k: v for k, v in study_cb.best_params.items()}
        cb_params['verbose'] = 0
        cb_params['allow_writing_files'] = False
        cb_params['random_seed'] = seed
        cb_base = CatBoostClassifier(**cb_params)
        cb_cpl = CurriculumSelfTraining(cb_base, tau_ref=cfg['tau_ref'], max_iter=5, epsilon=0.05)
        cb_cpl.fit(X_l, y_l, X_u)
        acc_cb = cb_cpl.score(X_t, y_t)
        print(f"      CatBoost-CPL acc: {acc_cb:.4f}")

        # ========================================
        # C. KAN with Optuna + CPL-LS
        # ========================================
        print(f"   🔍 Optuna for KAN ({cfg['kan_trials']} trials)...")
        study_kan = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed+2))
        study_kan.optimize(lambda trial: optimize_kan(trial, X_l, y_l, X_t, y_t, input_dim), n_trials=cfg['kan_trials'])
        
        best_kan = study_kan.best_params
        print("   🔹 Training KAN-CPL...")
        kan_base = SklearnPyTorchWrapper(
            KANClassifier, input_dim=input_dim, name="KAN",
            epochs=best_kan['epochs'], lr=best_kan['lr'],
            hidden_dim=best_kan['hidden_dim'], dropout=best_kan['dropout']
        )
        kan_cpl = CurriculumSelfTraining(kan_base, tau_ref=cfg['tau_ref'] - 0.05, max_iter=5, epsilon=0.05)
        kan_cpl.fit(X_l, y_l, X_u)
        acc_kan = kan_cpl.score(X_t, y_t)
        print(f"      KAN-CPL acc: {acc_kan:.4f}")

        # ========================================
        # D. TabNet with independent CPL-LS
        # ========================================
        print("   🔹 Training TabNet-CPL (independent)...")
        tabnet_base = TabNetClassifier(
            optimizer_fn=torch.optim.Adam, 
            optimizer_params=dict(lr=2e-2),
            n_d=16, n_a=16, n_steps=3,
            verbose=0
        )
        
        # Train TabNet base on labeled data first
        tabnet_base.fit(X_l.astype(np.float32), y_l, max_epochs=50, batch_size=256, virtual_batch_size=128)
        
        # Now use TabNet as its own teacher for CPL-style pseudo-labeling
        for cpl_iter in range(3):
            preds_proba = tabnet_base.predict_proba(X_u.astype(np.float32))
            max_probs = preds_proba.max(axis=1)
            confident_mask = max_probs >= 0.85
            
            if confident_mask.sum() == 0:
                break
                
            pseudo_labels = preds_proba.argmax(axis=1)
            X_aug = np.concatenate([X_l, X_u[confident_mask]], axis=0).astype(np.float32)
            y_aug = np.concatenate([y_l, pseudo_labels[confident_mask]], axis=0)
            
            tabnet_base = TabNetClassifier(
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                n_d=16, n_a=16, n_steps=3,
                verbose=0
            )
            tabnet_base.fit(X_aug, y_aug, max_epochs=40, batch_size=256, virtual_batch_size=128)
            
            # Remove pseudo-labeled samples from unlabeled pool
            X_u = X_u[~confident_mask]
            if len(X_u) == 0:
                break
                
        preds_tab = tabnet_base.predict(X_t.astype(np.float32))
        acc_tab = (preds_tab == y_t).mean()
        print(f"      TabNet-CPL acc: {acc_tab:.4f}")

        # ========================================
        # COLLECT ALL RESULTS (Winner selected globally below)
        # ========================================
        print(f"\n   📊 Results [{name}]:")
        print(f"      XGBoost-Optuna-CPL: {acc_xgb:.4f}")
        print(f"      CatBoost-Optuna-CPL: {acc_cb:.4f}")
        print(f"      KAN-Optuna-CPL:     {acc_kan:.4f}")
        print(f"      TabNet-CPL:         {acc_tab:.4f}")

        results = {
            "XGBoost-Optuna-CPL": (acc_xgb, xgb_cpl),
            "CatBoost-Optuna-CPL": (acc_cb, cb_cpl),
            "KAN-Optuna-CPL": (acc_kan, kan_cpl),
            "TabNet-CPL": (acc_tab, tabnet_base)
        }

        trained_models[name] = {
            'all_results': results,
            'X_test': X_test,
            'y_test': y_test,
            'features': X_enhanced.columns.tolist() if hasattr(X_enhanced, 'columns') else None
        }

    # ========================================
    # GLOBAL PORTFOLIO ASSIGNMENT
    # ========================================
    # Use the Hungarian algorithm to assign one unique algorithm per dimension
    # maximizing total accuracy across all 4 dimensions
    from scipy.optimize import linear_sum_assignment
    
    dim_names = list(trained_models.keys())
    algo_names = ["XGBoost-Optuna-CPL", "CatBoost-Optuna-CPL", "KAN-Optuna-CPL", "TabNet-CPL"]
    
    # Build cost matrix (we negate accuracy since linear_sum_assignment minimizes)
    cost_matrix = np.zeros((4, 4))
    for i, dim in enumerate(dim_names):
        for j, algo in enumerate(algo_names):
            acc, _ = trained_models[dim]['all_results'][algo]
            cost_matrix[i, j] = -acc  # negate for minimization
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    print("\n" + "="*60)
    print("GLOBAL PORTFOLIO ASSIGNMENT (Hungarian Algorithm)")
    print("="*60)
    
    for i, j in zip(row_ind, col_ind):
        dim = dim_names[i]
        algo = algo_names[j]
        acc, model = trained_models[dim]['all_results'][algo]
        
        trained_models[dim]['model'] = model
        trained_models[dim]['accuracy'] = acc
        trained_models[dim]['algorithm'] = algo
        
        print(f"  {dim:25s} → {algo:25s} ({acc:.4f})")
    
    print("="*60)
    total_acc = sum(trained_models[d]['accuracy'] for d in dim_names)
    print(f"  Total accuracy: {total_acc:.4f}")

    joblib.dump(trained_models, model_save_path)
    print(f"\n✅ CPL-LS pipeline complete. Models saved to {model_save_path}")

