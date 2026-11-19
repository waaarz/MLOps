# MLflow API reference for the lab

Use these MLflow functions with the examples from Lecture 1 and the [MLflow documentation](https://mlflow.org/docs/latest/).

```python
mlflow.set_tracking_uri(...)
mlflow.set_experiment(...)
mlflow.start_run(...)
mlflow.log_params(...)
mlflow.log_metric(...)
mlflow.log_artifact(...)
mlflow.data.from_pandas(...)
mlflow.log_input(...)
infer_signature(...)
mlflow.pytorch.log_model(...)
```

Pass the model name with `name=`.

Use `metadata={...}` in `mlflow.pytorch.log_model(...)` to record how to use the saved model, including the class order. Log `training_data_identity` separately as a Run parameter. The evaluator compares it with the identity of the current training and validation data.

Use `serialization_format="pickle"` to save `BeanCNN` and `code_paths=["src"]` to include the model class so the evaluator can load it. Before logging, move the model and example inputs to CPU and put the model in evaluation mode.

The model signature describes its normalized `float32` input tensors. `NCHW` is the order of the dimensions: batch size, channels, height and width. Before passing images to the model in Phase 2, use the same RGB conversion, resize and normalization as `src/prepare.py` and `src/data.py`. Read the class names in the order recorded in `class_names`.
