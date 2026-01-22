from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

def register_pipelines() -> dict[str, Pipeline]:
    pipelines = find_pipelines()
    pipelines["__default__"] = pipelines["iris_training"]
    return pipelines
