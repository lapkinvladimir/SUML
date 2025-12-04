import joblib
import numpy as np
import os

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
model = joblib.load(model_path)

def predict(features):
    """features: list or array of 4 numbers"""
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)[0]
    return prediction
