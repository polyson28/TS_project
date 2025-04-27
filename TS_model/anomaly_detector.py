from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

class AnomalyDetector:
    """Детектор аномалий методом IsolationForest"""
    def __init__(self, contamination=0.01):
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def fit(self, X):
        # Выберем только числовые колонки для обнаружения аномалий
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        self.numeric_columns = numeric_columns
        X_numeric = X[numeric_columns]

        self.model.fit(X_numeric)
        return self
    
    def detect(self, X):
        #Выберем только числовые колонки для обнаружения аномалий
        X_numeric = X[self.numeric_columns]  
         
        preds = self.model.predict(X_numeric)
        anomaly_mask = (preds == -1)
        return anomaly_mask
    
    def remove_anomalies(self, X, y=None):
        mask = self.detect(X)
        X_clean = X.loc[~mask]
        return X_clean