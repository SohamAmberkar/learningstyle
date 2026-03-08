import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
import joblib
import skfuzzy as fuzz

# --- NOTE: Standard practice is to handle imports that might fail gracefully ---
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    TENSORFLOW_AVAILABLE = True
except ImportError:
    # If TensorFlow/Keras fails, set a flag but allow the rest of the script to run
    TENSORFLOW_AVAILABLE = False
    print("WARNING: TensorFlow/Keras not found. HNN conceptual definition skipped.")


# --- 0. FUNCTIONAL DEFINITIONS FOR HYBRID COMPONENTS ---

def apply_fuzzy_c_means(X, n_clusters=8, fuzziness=2.0):
    """
    Performs Fuzzy C-Means clustering and returns membership grades as new features.
    Uses 8 clusters for nuanced feature enhancement (4 FSLSM axes * 2 sides = 8).
    """
    if X.empty:
        return X
    
    # 1. Prepare Data: Convert DataFrame to float array and transpose (FCM requirement)
    # Using np.float64 for numerical stability
    data_for_fcm = X.values.astype(np.float64).T
    
    # 2. Run FCM
    # fpc = fuzzy partition coefficient (used for validation, but returned here)
    cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(
        data_for_fcm, 
        c=n_clusters, 
        m=fuzziness, 
        error=0.005, 
        maxiter=100
    )
    
    # 3. Create FCM Features
    membership_grades = u.T
    fcm_features = pd.DataFrame(
        membership_grades, 
        columns=[f'FCM_M_{i+1}' for i in range(n_clusters)], 
        index=X.index
    )
    
    # 4. Concatenate and return
    return pd.concat([X, fcm_features], axis=1)

def define_hybrid_optimized_nn(input_dim):
    """
    Defines the architecture for the Hybrid Optimized Neural Network (HNN) benchmark.
    Returns None if TensorFlow is not available.
    """
    if not TENSORFLOW_AVAILABLE:
        return None
        
    model = Sequential()
    # Architecture reflecting complexity: 3 hidden layers (64-32-16)
    # Input dimension must match the enhanced CatBoost feature set.
    model.add(Dense(64, activation='relu', input_shape=(input_dim,))) # Use input_shape for Keras 3.0+
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid')) # Output layer for binary classification (one FSLSM axis)
    
    return model

def safe_smote_resample(X_train, y_train):
    """
    Safely apply SMOTE only when there are multiple classes in the training data.
    If only one class is present, return the original data with a warning.
    """
    unique_classes = np.unique(y_train)
    
    if len(unique_classes) < 2:
        print(f"⚠️  Only one class present in training data: {unique_classes[0]}. Skipping SMOTE.")
        return X_train, y_train
    
    # Check if any class has fewer than 2 samples (SMOTE requirement)
    class_counts = y_train.value_counts()
    if any(class_counts < 2):
        print("⚠️  Some classes have fewer than 2 samples. Using random oversampling instead.")
        # Use random oversampling for very small classes
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        return ros.fit_resample(X_train, y_train)
    
    # Apply SMOTE for balanced classes
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)

# --- 1. LOAD DATA & FEATURE ENGINEERING ---

# Load the data (assuming '../data/data_fs1.csv' is available and loaded into df)
df = pd.read_csv('../data/data_fs1.csv') 

# Check the unique values in learning_style to understand the data distribution
print("Unique learning_style values:", df['learning_style'].unique())
print("Learning style value counts:")
print(df['learning_style'].value_counts().sort_index())

# Define the Felder-Silverman learning style dimensions (4 binary axes)
# More robust mapping that handles potential data issues
df['visual_verbal'] = df['learning_style'].apply(lambda x: 1 if x in [0, 1, 2] else 0)  # Adjust based on your actual mapping
df['sensing_intuitive'] = df['learning_style'].apply(lambda x: 1 if x in [1, 3, 5] else 0)
df['active_reflective'] = df['learning_style'].apply(lambda x: 1 if x in [2, 3, 4] else 0)
df['sequential_global'] = df['learning_style'].apply(lambda x: 1 if x in [0, 4, 5] else 0)

# Print class distributions for each dimension
print("\nClass distributions for each learning style dimension:")
print("Visual/Verbal:", df['visual_verbal'].value_counts().sort_index())
print("Sensing/Intuitive:", df['sensing_intuitive'].value_counts().sort_index())
print("Active/Reflective:", df['active_reflective'].value_counts().sort_index())
print("Sequential/Global:", df['sequential_global'].value_counts().sort_index())

# 2. Separate features and target
X_raw = df.drop([
    'learning_style', 
    'visual_verbal', 
    'sensing_intuitive', 
    'active_reflective', 
    'sequential_global'
], axis=1)

# Check if we have any features
if X_raw.shape[1] == 0:
    raise ValueError("No features found! Check your column names and data structure.")

print(f"\nOriginal features: {X_raw.shape[1]}")
print(f"Sample size: {X_raw.shape[0]}")

# --- 3. APPLY FUZZY C-MEANS FOR FEATURE ENHANCEMENT ---

# The raw feature set is enhanced by adding FCM membership grades
X_enhanced = apply_fuzzy_c_means(X_raw, n_clusters=8, fuzziness=2.0)
print(f"✅ Features enhanced with {X_enhanced.shape[1] - X_raw.shape[1]} Fuzzy C-Means membership grades.")
print(f"Enhanced feature set shape: {X_enhanced.shape}")

# Define all target variables
y_visual = df['visual_verbal']
y_sensing = df['sensing_intuitive']
y_active = df['active_reflective']
y_sequential = df['sequential_global']

# --- 4. TRAIN FOUR CATBOOST MODELS (Enhanced by FCM) AND ONE HNN (Conceptual Benchmark) ---

catboost_models = {}
hybrid_nn_models = {} 

# Define list of models to train
MODEL_TARGETS = [
    ('visual', y_visual), 
    ('sensing', y_sensing), 
    ('active', y_active), 
    ('sequential', y_sequential)
]

# 4.1. CatBoost Training Loop (The Main Predictive Models)
for name, y_data in MODEL_TARGETS:
    print(f"\n--- Training CatBoost model for {name} dimension (FCM Enhanced) ---")
    
    # Check if we have multiple classes in the entire dataset
    unique_classes = np.unique(y_data)
    print(f"Unique classes in {name}: {unique_classes}")
    
    if len(unique_classes) < 2:
        print(f"❌ Skipping {name} dimension: Only one class present in the entire dataset.")
        continue
    
    # Split the ENHANCED data (X_enhanced)
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y_data, test_size=0.2, stratify=y_data, random_state=42
    )
    
    print(f"Training set class distribution for {name}:")
    print(pd.Series(y_train).value_counts().sort_index())
    
    # Apply safe SMOTE resampling
    X_resampled, y_resampled = safe_smote_resample(X_train, y_train)
    
    print(f"After resampling - Class distribution for {name}:")
    print(pd.Series(y_resampled).value_counts().sort_index())

    # Train CatBoost Classifier (Core Model)
    cb_model = CatBoostClassifier(
        iterations=100, 
        depth=6, 
        learning_rate=0.1, 
        verbose=0, 
        loss_function='Logloss', 
        random_seed=42
    )
    cb_model.fit(X_resampled, y_resampled)
    
    # Evaluate the model
    train_score = cb_model.score(X_resampled, y_resampled)
    test_score = cb_model.score(X_test, y_test)
    
    catboost_models[name] = cb_model
    print(f"✅ Core CatBoost Model for {name} trained on FCM features.")
    print(f"   Training Accuracy: {train_score:.4f}")
    print(f"   Test Accuracy: {test_score:.4f}")
    
    # 4.2. Conceptual Hybrid NN Training (The Advanced Benchmark)
    hnn_model = define_hybrid_optimized_nn(X_enhanced.shape[1])
    hybrid_nn_models[name] = hnn_model 
    
    # Check if HNN definition was successful
    if hnn_model is not None:
        print(f"🧠 HNN architecture defined for {name} dimension (Input: {X_enhanced.shape[1]}).")
    elif TENSORFLOW_AVAILABLE:
         print("❌ Error defining HNN architecture.")

# --- 5. SAVE ALL TRAINED MODELS ---

if catboost_models:
    joblib.dump(catboost_models, 'fslsm_catboost_fcm_models../models/.joblib')
    print(f"\n✅ {len(catboost_models)} FSLSM CatBoost models (FCM-enhanced) trained and saved successfully!")
else:
    print("\n❌ No models were trained. Check your data and class distributions.")

if hybrid_nn_models and any(hybrid_nn_models.values()):
    print("Hybrid Optimized NN architectures defined for advanced comparative analysis.")

# Additional diagnostics
print("\n=== TRAINING SUMMARY ===")
print(f"Total dimensions attempted: {len(MODEL_TARGETS)}")
print(f"Dimensions successfully trained: {len(catboost_models)}")
print(f"Enhanced feature dimensions: {X_enhanced.shape[1]}")
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
import joblib
import skfuzzy as fuzz

# --- NOTE: Standard practice is to handle imports that might fail gracefully ---
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    TENSORFLOW_AVAILABLE = True
except ImportError:
    # If TensorFlow/Keras fails, set a flag but allow the rest of the script to run
    TENSORFLOW_AVAILABLE = False
    print("WARNING: TensorFlow/Keras not found. HNN conceptual definition skipped.")


# --- 0. FUNCTIONAL DEFINITIONS FOR HYBRID COMPONENTS ---

def apply_fuzzy_c_means(X, n_clusters=8, fuzziness=2.0):
    """
    Performs Fuzzy C-Means clustering and returns membership grades as new features.
    Uses 8 clusters for nuanced feature enhancement (4 FSLSM axes * 2 sides = 8).
    """
    if X.empty:
        return X
    
    # 1. Prepare Data: Convert DataFrame to float array and transpose (FCM requirement)
    # Using np.float64 for numerical stability
    data_for_fcm = X.values.astype(np.float64).T
    
    # 2. Run FCM
    # fpc = fuzzy partition coefficient (used for validation, but returned here)
    cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(
        data_for_fcm, 
        c=n_clusters, 
        m=fuzziness, 
        error=0.005, 
        maxiter=100
    )
    
    # 3. Create FCM Features
    membership_grades = u.T
    fcm_features = pd.DataFrame(
        membership_grades, 
        columns=[f'FCM_M_{i+1}' for i in range(n_clusters)], 
        index=X.index
    )
    
    # 4. Concatenate and return
    return pd.concat([X, fcm_features], axis=1)

def define_hybrid_optimized_nn(input_dim):
    """
    Defines the architecture for the Hybrid Optimized Neural Network (HNN) benchmark.
    Returns None if TensorFlow is not available.
    """
    if not TENSORFLOW_AVAILABLE:
        return None
        
    model = Sequential()
    # Architecture reflecting complexity: 3 hidden layers (64-32-16)
    # Input dimension must match the enhanced CatBoost feature set.
    model.add(Dense(64, activation='relu', input_shape=(input_dim,))) # Use input_shape for Keras 3.0+
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid')) # Output layer for binary classification (one FSLSM axis)
    
    return model

def safe_smote_resample(X_train, y_train):
    """
    Safely apply SMOTE only when there are multiple classes in the training data.
    If only one class is present, return the original data with a warning.
    """
    unique_classes = np.unique(y_train)
    
    if len(unique_classes) < 2:
        print(f"⚠️  Only one class present in training data: {unique_classes[0]}. Skipping SMOTE.")
        return X_train, y_train
    
    # Check if any class has fewer than 2 samples (SMOTE requirement)
    class_counts = y_train.value_counts()
    if any(class_counts < 2):
        print("⚠️  Some classes have fewer than 2 samples. Using random oversampling instead.")
        # Use random oversampling for very small classes
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        return ros.fit_resample(X_train, y_train)
    
    # Apply SMOTE for balanced classes
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)

# --- 1. LOAD DATA & FEATURE ENGINEERING ---

# Load the data (assuming '../data/data_fs1.csv' is available and loaded into df)
df = pd.read_csv('../data/data_fs1.csv') 

# Check the unique values in learning_style to understand the data distribution
print("Unique learning_style values:", df['learning_style'].unique())
print("Learning style value counts:")
print(df['learning_style'].value_counts().sort_index())

# Define the Felder-Silverman learning style dimensions (4 binary axes)
# More robust mapping that handles potential data issues
df['visual_verbal'] = df['learning_style'].apply(lambda x: 1 if x in [0, 1, 2] else 0)  # Adjust based on your actual mapping
df['sensing_intuitive'] = df['learning_style'].apply(lambda x: 1 if x in [1, 3, 5] else 0)
df['active_reflective'] = df['learning_style'].apply(lambda x: 1 if x in [2, 3, 4] else 0)
df['sequential_global'] = df['learning_style'].apply(lambda x: 1 if x in [0, 4, 5] else 0)

# Print class distributions for each dimension
print("\nClass distributions for each learning style dimension:")
print("Visual/Verbal:", df['visual_verbal'].value_counts().sort_index())
print("Sensing/Intuitive:", df['sensing_intuitive'].value_counts().sort_index())
print("Active/Reflective:", df['active_reflective'].value_counts().sort_index())
print("Sequential/Global:", df['sequential_global'].value_counts().sort_index())

# 2. Separate features and target
X_raw = df.drop([
    'learning_style', 
    'visual_verbal', 
    'sensing_intuitive', 
    'active_reflective', 
    'sequential_global'
], axis=1)

# Check if we have any features
if X_raw.shape[1] == 0:
    raise ValueError("No features found! Check your column names and data structure.")

print(f"\nOriginal features: {X_raw.shape[1]}")
print(f"Sample size: {X_raw.shape[0]}")

# --- 3. APPLY FUZZY C-MEANS FOR FEATURE ENHANCEMENT ---

# The raw feature set is enhanced by adding FCM membership grades
X_enhanced = apply_fuzzy_c_means(X_raw, n_clusters=8, fuzziness=2.0)
print(f"✅ Features enhanced with {X_enhanced.shape[1] - X_raw.shape[1]} Fuzzy C-Means membership grades.")
print(f"Enhanced feature set shape: {X_enhanced.shape}")

# Define all target variables
y_visual = df['visual_verbal']
y_sensing = df['sensing_intuitive']
y_active = df['active_reflective']
y_sequential = df['sequential_global']

# --- 4. TRAIN FOUR CATBOOST MODELS (Enhanced by FCM) AND ONE HNN (Conceptual Benchmark) ---

catboost_models = {}
hybrid_nn_models = {} 

# Define list of models to train
MODEL_TARGETS = [
    ('visual', y_visual), 
    ('sensing', y_sensing), 
    ('active', y_active), 
    ('sequential', y_sequential)
]

# 4.1. CatBoost Training Loop (The Main Predictive Models)
for name, y_data in MODEL_TARGETS:
    print(f"\n--- Training CatBoost model for {name} dimension (FCM Enhanced) ---")
    
    # Check if we have multiple classes in the entire dataset
    unique_classes = np.unique(y_data)
    print(f"Unique classes in {name}: {unique_classes}")
    
    if len(unique_classes) < 2:
        print(f"❌ Skipping {name} dimension: Only one class present in the entire dataset.")
        continue
    
    # Split the ENHANCED data (X_enhanced)
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y_data, test_size=0.2, stratify=y_data, random_state=42
    )
    
    print(f"Training set class distribution for {name}:")
    print(pd.Series(y_train).value_counts().sort_index())
    
    # Apply safe SMOTE resampling
    X_resampled, y_resampled = safe_smote_resample(X_train, y_train)
    
    print(f"After resampling - Class distribution for {name}:")
    print(pd.Series(y_resampled).value_counts().sort_index())

    # Train CatBoost Classifier (Core Model)
    cb_model = CatBoostClassifier(
        iterations=100, 
        depth=6, 
        learning_rate=0.1, 
        verbose=0, 
        loss_function='Logloss', 
        random_seed=42
    )
    cb_model.fit(X_resampled, y_resampled)
    
    # Evaluate the model
    train_score = cb_model.score(X_resampled, y_resampled)
    test_score = cb_model.score(X_test, y_test)
    
    catboost_models[name] = cb_model
    print(f"✅ Core CatBoost Model for {name} trained on FCM features.")
    print(f"   Training Accuracy: {train_score:.4f}")
    print(f"   Test Accuracy: {test_score:.4f}")
    
    # 4.2. Conceptual Hybrid NN Training (The Advanced Benchmark)
    hnn_model = define_hybrid_optimized_nn(X_enhanced.shape[1])
    hybrid_nn_models[name] = hnn_model 
    
    # Check if HNN definition was successful
    if hnn_model is not None:
        print(f"🧠 HNN architecture defined for {name} dimension (Input: {X_enhanced.shape[1]}).")
    elif TENSORFLOW_AVAILABLE:
         print("❌ Error defining HNN architecture.")

# --- 5. SAVE ALL TRAINED MODELS ---

if catboost_models:
    joblib.dump(catboost_models, 'fslsm_catboost_fcm_models../models/.joblib')
    print(f"\n✅ {len(catboost_models)} FSLSM CatBoost models (FCM-enhanced) trained and saved successfully!")
else:
    print("\n❌ No models were trained. Check your data and class distributions.")

if hybrid_nn_models and any(hybrid_nn_models.values()):
    print("Hybrid Optimized NN architectures defined for advanced comparative analysis.")

# Additional diagnostics
print("\n=== TRAINING SUMMARY ===")
print(f"Total dimensions attempted: {len(MODEL_TARGETS)}")
print(f"Dimensions successfully trained: {len(catboost_models)}")
print(f"Enhanced feature dimensions: {X_enhanced.shape[1]}")
=======
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# 1. Load the data
df = pd.read_csv('../data/data_fs1.csv')

# Define the Felder-Silverman learning style dimensions based on your data labels
df['visual_verbal'] = df['learning_style'].apply(lambda x: 1 if x == 0 else 0)
df['sensing_intuitive'] = df['learning_style'].apply(lambda x: 1 if x == 1 or x == 2 else 0)
df['active_reflective'] = df['learning_style'].apply(lambda x: 1 if x == 3 else 0)

# 2. Separate features and target
X = df.drop(['learning_style', 'visual_verbal', 'sensing_intuitive', 'active_reflective'], axis=1)

# The new target variables are the three FSLSM dimensions
y_visual = df['visual_verbal']
y_sensing = df['sensing_intuitive']
y_active = df['active_reflective']

# 3. Train three separate models
models = {}
for name, y_data in [('visual', y_visual), ('sensing', y_sensing), ('active', y_active)]:
    print(f"Training model for {name} dimension...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_data, test_size=0.2, stratify=y_data, random_state=42
    )
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=0, loss_function='Logloss')
    model.fit(X_resampled, y_resampled)
    models[name] = model
    print(f"✅ Model for {name} trained.")

# 4. Save all three trained models
joblib.dump(models, 'fslsm_models../models/.joblib')

print("All FSLSM models trained and saved successfully!")
>>>>>>> a54f0a3a2ff90fb292dab62bad3424f7ccd55b48
