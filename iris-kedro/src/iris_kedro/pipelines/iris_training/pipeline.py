from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    split_features_and_target,
    train_test_split_node,
    train_knn_model,
    train_logreg_model,
    train_rf_model,
    train_svm_model,
    select_best_model,
    save_best_model_locally,
    save_model_metadata,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_features_and_target,
                inputs="iris",
                outputs=["X", "y"],
                name="split_features_and_target",
            ),
            node(
                func=train_test_split_node,
                inputs=["X", "y", "params:test_size", "params:random_state"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="train_test_split_node",
            ),
            node(
                func=train_knn_model,
                inputs=["X_train", "X_test", "y_train", "y_test", "params:knn_n_neighbors"],
                outputs="knn_result",
                name="train_knn_node",
            ),
            node(
                func=train_logreg_model,
                inputs=["X_train", "X_test", "y_train", "y_test", "params:logreg_max_iter"],
                outputs="logreg_result",
                name="train_logreg_node",
            ),
            node(
                func=train_rf_model,
                inputs=["X_train", "X_test", "y_train", "y_test", "params:rf_n_estimators", "params:random_state"],
                outputs="rf_result",
                name="train_rf_node",
            ),
            node(
                func=train_svm_model,
                inputs=["X_train", "X_test", "y_train", "y_test", "params:svm_C", "params:svm_kernel"],
                outputs="svm_result",
                name="train_svm_node",
            ),
            node(
                func=select_best_model,
                inputs=["knn_result", "logreg_result", "svm_result", "rf_result"],
                outputs="best_model_result",
                name="select_best_model_node",
            ),
            node(
                func=save_best_model_locally,
                inputs=["best_model_result", "params:model_output_path"],
                outputs="saved_model_path",
                name="save_best_model_node",
            ),
            node(
                func=save_model_metadata,
                inputs=["best_model_result", "params:model_meta_output_path"],
                outputs="saved_meta_path",
                name="save_model_metadata_node",
            ),
        ]
    )

