# anomaly_detection.py
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def calculate_ndvi(data):
    nir = data[:, :, 7]
    red = data[:, :, 3]
    ndvi = (nir - red) / (nir + red + 1e-6)
    return ndvi

def detect_anomalies(ndvi_series):
    flattened = ndvi_series.reshape(ndvi_series.shape[0], -1)
    df = pd.DataFrame(flattened)
    
    clf = IsolationForest(contamination=0.05, random_state=42)
    clf.fit(df)
    anomalies = clf.predict(df)
    
    return anomalies