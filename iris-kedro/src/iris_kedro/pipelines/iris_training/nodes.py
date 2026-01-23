from typing import Dict, Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import json
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from typing import Tuple

from sklearn.model_selection import train_test_split


def load_iris_data(iris: pd.DataFrame, test_size: float, random_state: int):
    X = iris.drop(columns=["target"])
    y = iris["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test

def split_features_and_target(iris: pd.DataFrame):
    X = iris.drop(columns=["target"])
    y = iris["target"]
    return X, y


def train_test_split_node(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def train_knn_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_neighbors: int,
) -> Dict:
    """Обучает KNN, логирует метрики в MLflow и возвращает результат."""
    with mlflow.start_run(nested=True):
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        mlflow.log_param("model_type", "KNN")
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        return {
            "model_type": "KNN",
            "model": model,
            "accuracy": acc,
            "f1_macro": f1,
        }

def train_logreg_model(X_train, X_test, y_train, y_test, max_iter: int) -> Dict:
    with mlflow.start_run(nested=True):
        model = LogisticRegression(max_iter=max_iter)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        return {
            "model_type": "LogisticRegression",
            "model": model,
            "accuracy": acc,
            "f1_macro": f1,
        }


def train_svm_model(X_train, X_test, y_train, y_test, C: float, kernel: str) -> Dict:
    with mlflow.start_run(nested=True):
        model = SVC(C=C, kernel=kernel)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        mlflow.log_param("model_type", "SVM")
        mlflow.log_param("C", C)
        mlflow.log_param("kernel", kernel)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        return {
            "model_type": "SVM",
            "model": model,
            "accuracy": acc,
            "f1_macro": f1,
        }


def train_rf_model(X_train, X_test, y_train, y_test, n_estimators: int, random_state: int) -> Dict:
    with mlflow.start_run(nested=True):
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        return {
            "model_type": "RandomForest",
            "model": model,
            "accuracy": acc,
            "f1_macro": f1,
        }


def select_best_model(*model_results: Dict) -> Dict:
    best = max(model_results, key=lambda x: x["f1_macro"])
    mlflow.log_param("best_model_type", best["model_type"])
    mlflow.log_metric("best_f1_macro", best["f1_macro"])
    return best

def save_best_model_locally(best_model_result: Dict, output_path: str) -> str:
    model = best_model_result["model"]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def save_model_metadata(best_model_result: Dict, output_path: str) -> str:
    meta = {
        "model_type": best_model_result["model_type"],
        "accuracy": best_model_result["accuracy"],
        "f1_macro": best_model_result["f1_macro"],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2)
    return output_path