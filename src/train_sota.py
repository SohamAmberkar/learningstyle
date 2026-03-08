import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
import skfuzzy as fuzz
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
import os

# --- HARDCODED CREDENTIALS ---
# (Keeping this placeholder pattern consistent)
api_key = "PASTE_YOUR_GEMINI_API_KEY_HERE"

# --- 0. HELPER FUNCTIONS ---

def apply_fuzzy_c_means(X, n_clusters=8, fuzziness=2.0):
    if X.empty: return X
    data_for_fcm = X.values.astype(np.float64).T
    try:
        cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(data_for_fcm, c=n_clusters, m=fuzziness, error=0.005, maxiter=100)
        fcm_features = pd.DataFrame(u.T, columns=[f'FCM_M_{i+1}' for i in range(n_clusters)], index=X.index)
        return pd.concat([X, fcm_features], axis=1)
    except: return X

def safe_smote_resample(X_train, y_train):
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2: return X_train, y_train
    class_counts = y_train.value_counts()
    if any(class_counts < 2):
        return RandomOverSampler(random_state=42).fit_resample(X_train, y_train)
    return SMOTE(random_state=42).fit_resample(X_train, y_train)

# --- 1. MODEL DEFINITIONS ---

# A. KAN (Kolmogorov-Arnold Network) - Simplified Implementation
class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Learnable weights for edges (approximated by linear for stability in this demo)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5))
        
    def forward(self, x):
        return torch.nn.functional.linear(x, self.base_weight)

class KANClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super().__init__()
        self.layer1 = KANLayer(input_dim, hidden_dim)
        self.layer2 = KANLayer(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.nn.functional.silu(self.layer1(x))
        x = self.layer2(x)
        return x

# B. Neural Additive Model (NAM) - Simplified "Glass Box"
# NAMs learn shape functions f_i(x_i) for each feature and sum them.
class FeatureNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)

class NAMClassifier(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.feature_nns = nn.ModuleList([FeatureNN() for _ in range(num_features)])
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # x shape: [batch, num_features]
        output = self.bias
        for i, feature_nn in enumerate(self.feature_nns):
            # Extract i-th feature column
            feature_input = x[:, i].unsqueeze(1) 
            output = output + feature_nn(feature_input).squeeze(1)
        return output

# C. FT-Transformer (Feature Tokenizer + Transformer)
# Simplified implementation using torch.nn.TransformerEncoder
class FTTransformer(nn.Module):
    def __init__(self, num_numerical_features, embedding_dim=32, depth=3, heads=4):
        super().__init__()
        self.num_numerical_features = num_numerical_features
        
        # Feature Tokenizer (Numerical Embedding)
        # We use a Linear layer to project each scalar feature to 'embedding_dim'
        self.feature_tokenizers = nn.ModuleList([
            nn.Linear(1, embedding_dim) for _ in range(num_numerical_features)
        ])
        
        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=heads, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 2) # Binary classification
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Tokenize features
        tokens = []
        for i, tokenizer in enumerate(self.feature_tokenizers):
            feature = x[:, i].unsqueeze(1) # [batch, 1]
            token = tokenizer(feature).unsqueeze(1) # [batch, 1, emb]
            tokens.append(token)
            
        x_emb = torch.cat(tokens, dim=1) # [batch, num_features, emb]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_emb = torch.cat((cls_tokens, x_emb), dim=1)
        
        # Transformer Pass
        x_out = self.transformer(x_emb)
        
        # Use CLS token for prediction
        cls_out = x_out[:, 0, :]
        return self.head(cls_out)

# --- 2. TRAINING LOGIC ---

def train_pytorch_model(model_class, model_name, X_train, y_train, input_dim):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_name == "NAM":
        model = NAMClassifier(input_dim).to(device)
    elif model_name == "FT-Transformer":
        model = FTTransformer(input_dim).to(device)
    elif model_name == "KAN":
        model = KANClassifier(input_dim).to(device)
    else:
        return None

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss() if model_name != "NAM" else nn.BCEWithLogitsLoss()
    
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train.values).to(device), 
        torch.LongTensor(y_train.values).to(device) if model_name != "NAM" else torch.FloatTensor(y_train.values).to(device)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    print(f"   Training {model_name}...")
    model.train()
    for epoch in range(15): # Short training
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            
            # Simple shape handling for different losses
            if model_name == "NAM":
                loss = criterion(output, batch_y)
            else:
                loss = criterion(output, batch_y)
                
            loss.backward()
            optimizer.step()
            
    return model

# --- 3. MAIN SCRIPT ---

if __name__ == "__main__":
    df = pd.read_csv('../data/data_fs1.csv')
    
    # Feature Engineering
    df['visual_verbal'] = df['learning_style'].apply(lambda x: 1 if x in [0, 1, 2] else 0)
    df['sensing_intuitive'] = df['learning_style'].apply(lambda x: 1 if x in [1, 3, 5] else 0)
    df['active_reflective'] = df['learning_style'].apply(lambda x: 1 if x in [2, 3, 4] else 0)
    df['sequential_global'] = df['learning_style'].apply(lambda x: 1 if x in [0, 4, 5] else 0)

    X_raw = df.drop(['learning_style', 'visual_verbal', 'sensing_intuitive', 'active_reflective', 'sequential_global'], axis=1)
    X_enhanced = apply_fuzzy_c_means(X_raw)
    
    y_targets = {
        'visual': df['visual_verbal'],
        'sensing': df['sensing_intuitive'],
        'active': df['active_reflective'],
        'sequential': df['sequential_global']
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for name, y_data in y_targets.items():
        if len(np.unique(y_data)) < 2: continue
        print(f"\n=== Dimension: {name} ===")
        
        X_train, X_test, y_train, y_test = train_test_split(X_enhanced, y_data, test_size=0.2, stratify=y_data, random_state=42)
        X_resampled, y_resampled = safe_smote_resample(X_train, y_train)
        
        # 1. TabNet
        print("Training TabNet...")
        tabnet = TabNetClassifier(optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2), verbose=0)
        tabnet.fit(X_resampled.values, y_resampled.values, max_epochs=20)
        tabnet.save_model(f'sota_tabnet_{name}')
        
        # 2. KAN
        kan = train_pytorch_model(KANClassifier, "KAN", X_resampled, y_resampled, X_enhanced.shape[1])
        torch.save(kan, f'sota_kan_{name}../models/.pt')
        
        # 3. NAM
        nam = train_pytorch_model(NAMClassifier, "NAM", X_resampled, y_resampled, X_enhanced.shape[1])
        torch.save(nam, f'sota_nam_{name}../models/.pt')
        
        # 4. FT-Transformer
        ftt = train_pytorch_model(FTTransformer, "FT-Transformer", X_resampled, y_resampled, X_enhanced.shape[1])
        torch.save(ftt, f'sota_ftt_{name}../models/.pt')
        
    print("\n✅ All SOTA models (TabNet, KAN, NAM, FT-Transformer) trained and saved.")
