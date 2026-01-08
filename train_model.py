import os
import json
import joblib
import tempfile

import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


VERSION = "v1.0.0"
EXPERIMENT_NAME = "iris-model-zoo"
REGISTERED_MODEL_NAME = "IrisModel"


def _safe_roc_auc(y_true, model, X_test):
    """
    Try to compute multiclass ROC AUC (OVR).
    Returns None if not possible.
    """
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            return roc_auc_score(y_true, proba, multi_class="ovr")
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            return roc_auc_score(y_true, scores, multi_class="ovr")
    except Exception:
        return None
    return None


def _log_confusion_matrix(cm, labels, artifact_name="confusion_matrix.png"):
    """
    Log confusion matrix plot as an artifact to MLflow.
    """
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Add numbers on the matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, artifact_name)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(path)


def _log_text(content: str, filename: str):
    """
    Log text file artifact to MLflow.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        mlflow.log_artifact(path)


def main():
    # Ensure output folder exists
    os.makedirs("app", exist_ok=True)

    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = list(iris.target_names)

    # Split data (fixed random_state for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Define a "model zoo"
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=200, random_state=42)),
            ]
        ),
        "SVM": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", SVC(probability=True, random_state=42)),
            ]
        ),
        "KNN": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
    }

    best = {
        "name": None,
        "f1_macro": -1.0,
        "run_id": None,
        "metrics": None,
        "model": None,
    }

    # Train + evaluate each model in its own MLflow run
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id

            # Tags
            mlflow.set_tag("version", VERSION)
            mlflow.set_tag("model_name", model_name)

            # Log model params (best effort)
            try:
                if isinstance(model, Pipeline):
                    # Log pipeline final estimator params
                    mlflow.log_params(model.get_params())
                else:
                    mlflow.log_params(model.get_params())
            except Exception:
                pass

            # Train
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
            auc = _safe_roc_auc(y_test, model, X_test)

            mlflow.log_metric("accuracy", float(acc))
            mlflow.log_metric("precision_macro", float(prec))
            mlflow.log_metric("recall_macro", float(rec))
            mlflow.log_metric("f1_macro", float(f1m))
            if auc is not None:
                mlflow.log_metric("roc_auc_ovr", float(auc))

            # Artifacts
            cm = confusion_matrix(y_test, y_pred)
            _log_confusion_matrix(cm, class_names)

            report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
            _log_text(report, "classification_report.txt")

            # Log model to MLflow
            # (also registers version in Model Registry)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=REGISTERED_MODEL_NAME,
            )

            print(f"[{model_name}] accuracy={acc:.4f} f1_macro={f1m:.4f} run_id={run_id}")

            # Track best by f1_macro
            if f1m > best["f1_macro"]:
                best["name"] = model_name
                best["f1_macro"] = float(f1m)
                best["run_id"] = run_id
                best["metrics"] = {
                    "accuracy": float(acc),
                    "precision_macro": float(prec),
                    "recall_macro": float(rec),
                    "f1_macro": float(f1m),
                    "roc_auc_ovr": float(auc) if auc is not None else None,
                }
                best["model"] = model

    # Save best model to app/model.joblib
    model_path = os.path.join("app", "model.joblib")
    joblib.dump(best["model"], model_path)

    # Save metadata JSON
    meta = {
        "best_model": best["name"],
        "metrics": best["metrics"],
        "mlflow_run_id": best["run_id"],
        "version": VERSION,
        "experiment": EXPERIMENT_NAME,
        "registered_model_name": REGISTERED_MODEL_NAME,
    }
    meta_path = os.path.join("app", "model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Best model: {best['name']} (f1_macro={best['f1_macro']:.4f})")
    print(f"Saved best model to: {model_path}")
    print(f"Saved metadata to: {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
