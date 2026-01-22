from kedro.pipeline import Pipeline, node

from .nodes import (
    load_iris_data,
    train_knn_model,
    train_logreg_model,
    train_svm_model,
    train_rf_model,
    select_best_model,
    save_best_model_locally,
    save_model_metadata,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=load_iris_data,
                inputs=dict(test_size="params:test_size", random_state="params:random_state"),
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="load_iris_data_node",
            ),
            node(
                func=train_knn_model,
                inputs=dict(
                    X_train="X_train",
                    X_test="X_test",
                    y_train="y_train",
                    y_test="y_test",
                    n_neighbors="params:knn_n_neighbors",
                ),
                outputs="knn_result",
                name="train_knn_node",
            ),
            node(
                func=train_logreg_model,
                inputs=dict(
                    X_train="X_train",
                    X_test="X_test",
                    y_train="y_train",
                    y_test="y_test",
                    max_iter="params:logreg_max_iter",
                ),
                outputs="logreg_result",
                name="train_logreg_node",
            ),
            node(
                func=train_svm_model,
                inputs=dict(
                    X_train="X_train",
                    X_test="X_test",
                    y_train="y_train",
                    y_test="y_test",
                    C="params:svm_C",
                    kernel="params:svm_kernel",
                ),
                outputs="svm_result",
                name="train_svm_node",
            ),
            node(
                func=train_rf_model,
                inputs=dict(
                    X_train="X_train",
                    X_test="X_test",
                    y_train="y_train",
                    y_test="y_test",
                    n_estimators="params:rf_n_estimators",
                    random_state="params:random_state",
                ),
                outputs="rf_result",
                name="train_rf_node",
            ),
            node(
                func=select_best_model,
                inputs=["knn_result", "logreg_result", "svm_result", "rf_result"],
                outputs="best_model_result",
                name="select_best_model_node",
            ),
            node(
                func=save_best_model_locally,
                inputs=dict(
                    best_model_result="best_model_result",
                    output_path="params:model_output_path",
                ),
                outputs="saved_model_path",
                name="save_best_model_node",
            ),
            node(
                func=save_model_metadata,
                inputs=dict(
                    best_model_result="best_model_result",
                    output_path="params:model_meta_output_path",
                ),
                outputs="saved_meta_path",
                name="save_model_metadata_node",
            ),
        ]
    )
