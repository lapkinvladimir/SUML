import json
import os
import streamlit as st
from predict import predict

st.title("Iris Flower Classifier")
st.write("Enter flower measurements and get species prediction.")

# Show model metadata (if exists)
meta_path = os.path.join(os.path.dirname(__file__), "model_meta.json")
if os.path.exists(meta_path):
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        st.caption(
            f"Model: **{meta.get('best_model')}** | "
            f"Version: **{meta.get('version')}** | "
            f"F1-macro: **{meta.get('metrics', {}).get('f1_macro')}** | "
            f"Run ID: `{meta.get('mlflow_run_id')}`"
        )
    except Exception:
        st.caption("Model metadata exists but could not be loaded.")
else:
    st.caption("Model metadata not found (run training to generate model_meta.json).")

# Input fields
sepal_length = st.number_input("Sepal length", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal width", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal length", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal width", 0.0, 10.0, 0.2)

if st.button("Predict"):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    pred = predict(features)

    iris_names = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Predicted species: **{iris_names[pred]}**")
