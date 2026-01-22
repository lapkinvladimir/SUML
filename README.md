pip install -r requirements.txt

pip install kedro kedro-mlflow kedro-viz kedro-datasets

cd iris-kedro

kedro mlflow init

kedro run --pipeline iris_training

появится 2 файла:
app/model.joblib
app/model_meta.json

kedro viz


в другом терминале:
streamlit run app/app.py

