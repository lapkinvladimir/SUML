import streamlit as st
from predict import predict

st.title("Iris Flower Classifier")
st.write("Enter flower measurements and get species prediction.")

# Input fields
sepal_length = st.number_input("Sepal length", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal width", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal length", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal width", 0.0, 10.0, 0.2)

if st.button("Predict"):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    pred = predict(features)

    # Map class numbers to names (like Iris dataset)
    iris_names = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Predicted species: **{iris_names[pred]}**")
