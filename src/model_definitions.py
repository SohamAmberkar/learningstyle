import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd

# --- SOTA MODEL DEFINITIONS ---

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5))
        
    def forward(self, x):
        return torch.nn.functional.linear(x, self.base_weight)

class KANClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, dropout=0.2):
        super().__init__()
        self.layer1 = KANLayer(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.drop1 = nn.Dropout(dropout)
        self.layer2 = KANLayer(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.drop2 = nn.Dropout(dropout)
        self.layer3 = KANLayer(hidden_dim // 2, output_dim)
        
    def forward(self, x):
        x = torch.nn.functional.silu(self.layer1(x))
        x = self.bn1(x)
        x = self.drop1(x)
        x = torch.nn.functional.silu(self.layer2(x))
        x = self.bn2(x)
        x = self.drop2(x)
        x = self.layer3(x)
        return x

class SklearnPyTorchWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_class, input_dim=None, epochs=50, lr=0.01, name="Model", hidden_dim=64, dropout=0.2):
        self.model_class = model_class
        self.input_dim = input_dim
        self.epochs = epochs
        self.lr = lr
        self.name = name
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.model = None
        self.classes_ = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if self.classes_[0] == -1: self.classes_ = self.classes_[1:]

        if isinstance(X, pd.DataFrame): X = X.values
        if isinstance(y, pd.Series): y = y.values
        
        mask = y != -1
        X_train = X[mask]
        y_train = y[mask]

        if self.input_dim is None: self.input_dim = X_train.shape[1]
            
        if self.name == "KAN":
            self.model = self.model_class(
                self.input_dim, 
                hidden_dim=self.hidden_dim, 
                dropout=self.dropout
            ).to(self.device)
            criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.LongTensor(y_train).to(self.device)

        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step()
        return self

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame): X = X.values
        if self.model is None:
             return np.zeros((len(X), len(self.classes_)))
             
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
