# Lab hints

Try the task first. If you get stuck, open the hint for that part of the lab. Start with the question, then use the function names or tests to check your approach.

<details>
<summary>Milestone 0 · Find a concrete issue for the audit</summary>

Imagine that someone clones your repository and tries to reproduce a result. Pick one thing they would need, such as the exact package versions. Where would they look? Can they recover it from the files in the starter?

For each issue in your report, point to a file, setting, or test outcome. Explain what could go wrong and how you would check that your suggested fix works.

</details>

<details>
<summary>Milestone 1 · Find the random number generators</summary>

Open `test_global_seeding_repeats_python_numpy_torch` in `tests/test_reproducibility.py`. Which libraries does the test draw from? Can you make the draws repeat by resetting the seed before each group of draws?

Look up `numpy.random.seed()` and `torch.manual_seed()`. For the CUDA TODO, look up `torch.cuda.is_available()` and check what the `deterministic` and `benchmark` settings under `torch.backends.cudnn` control.

</details>

<details>
<summary>Milestone 1 · Seed shuffling and workers</summary>

Find where `src/data.py` passes a generator and a worker function to the DataLoader. Which controls the shuffle order, and which runs inside a worker?

Look up `torch.Generator` and `torch.initial_seed()`. Check what PyTorch gives each worker at startup and what range NumPy accepts for a seed. The lab loaders use zero worker processes by default. Use the worker test to exercise your function directly.

</details>

<details>
<summary>Milestone 3 · Check folders and counts</summary>

Start with one split. What should happen if it is missing, has an extra class, or contains an empty class folder? Read one corresponding case in `tests/test_validation_contract.py` before moving on.

Look up `Path.iterdir()` and `Path.is_dir()` for finding class folders. For image counts, explore `Path.rglob()`, `Path.is_file()` and `Path.suffix`. Check how the tests provide smaller counts through the existing constants.

</details>

<details>
<summary>Milestone 3 · Detect a corrupt image and keep the report stable</summary>

The corrupt file in `tests/test_data_validation.py` still ends in `.jpg`. Look up `Image.open()` and the image object's `load()` method. At what point does Pillow read the pixels, and what error does the test expect to reach the caller?

Think about which parts of the report could change if the same files were listed in a different order. Which values should stay the same every time the validator runs on the same images?

</details>

<details>
<summary>Milestone 4 · Trace the inputs for one stage</summary>

For one stage, trace the paths and settings used by its Python script. Follow imported helpers too. Each time you find a path, ask whether the stage reads it or creates it. For a setting, ask which key the script or helper actually uses.

For practice, imagine a script that reads `events.csv`, uses the setting `filters.year`, and writes `totals.json`. Where would each one go in a stage definition? Apply the same questions to the lab. [DVC pipeline examples](https://doc.dvc.org/user-guide/pipelines/defining-pipelines)

</details>

<details>
<summary>Milestone 4 · Read the DVC graph</summary>

If the graph differs from the diagram in the guide, pick one unexpected connection. Find the file listed as an output of the upstream stage and see whether the downstream stage declares it as an input. Which line in `dvc.yaml` explains the connection?

</details>

<details>
<summary>Milestone 5 · Collect the Run manifest</summary>

Read the assertions in `tests/test_provenance.py` to see where values belong. Use the supplied `training_data_identity()` helper for the prepared data.

Look up how Git prints the current commit and how it reports working tree changes with `--porcelain`. Try the commands in the project and in a folder outside Git. When you call them with `subprocess.run()`, compare the exit status with what the command printed. Which other fields can you still collect if Git fails?

For versions, look up `platform.python_version()` and `importlib.metadata.version()`. Check which fields describe installed packages and which come from `params` or the current Run.

In `src/train.py`, inspect the object returned by `mlflow.start_run()` to find the identifier that belongs in the manifest. Read `write_run_manifest()` to see what it already writes for you.

</details>

<details>
<summary>Milestone 5 · Log parameters and the training dataset</summary>

Follow one value, such as the learning rate, from `params.yaml` to where the optimizer reads it. Which value should you see in MLflow? Use the same approach for each required parameter.

The code already reads `train_manifest_df`. Inspect its columns and find the one that describes the class. In the MLflow reference, check which function turns a DataFrame into a Dataset object and which function attaches that object to the current Run. [MLflow API reference](MLFLOW_API_REFERENCE.md)

</details>

<details>
<summary>Milestone 5 · Check metrics and files in MLflow</summary>

For the metrics TODO, look at the values appended to `history` in `src/train.py`. Which values should appear once per epoch? Which value is known only after the loop? If a chart has one point per metric, check where the logging call runs and what is passed as `step`.

For the artifact TODO, find where each file is first written. When you inspect a real Run in Milestone 6, can you open those files in MLflow?

</details>

<details>
<summary>Milestone 5 · Build the model signature and input example</summary>

Use `iter()` and `next()` to inspect one batch from `val_loader`. Which part does the model take as input? If you keep one image, check whether the shape still has a batch dimension.

Check that the model and the example are on the same device. Look up what `model.eval()` does to dropout and what `torch.no_grad()` does during the forward pass. Before calling `infer_signature()`, compare the model's output shape with the shape of the predicted class IDs. Which shape will a caller of the saved model receive? [MLflow model signatures](https://mlflow.org/docs/latest/ml/model/signatures/)

</details>

<details>
<summary>Milestone 6 · Explain a prediction that did not match</summary>

Start with the first stage whose observed behavior surprised you. Find it in the terminal output and look for the command DVC ran or the message saying it skipped the stage. Trace the change back to a path or setting in `dvc.yaml`.

For an input produced by another stage, compare the file's contents before and after the check. Use the message and what happened to that input to explain your observation.

</details>

<details>
<summary>Milestone 7 · Explain what the comparison shows</summary>

Open Run A and one candidate in MLflow. Check the saved parameters before interpreting the difference in their scores. If two runs have similar accuracy, could they make errors on different classes? Use that question to justify how you chose your main metric.

Check the final epoch on the metric chart against the value in your table. What might the loss curves tell you that the final score alone cannot?

</details>

<details>
<summary>Milestone 8 · Find why the selected Run was rejected</summary>

Use the error message to locate the check in `src/evaluate.py`. If the message names a setting or a data identity, compare the selected Run's record with the current project. If it names the saved model, open that Run's model record and check the model name and metadata.

For Run checks such as the experiment or data fingerprint, see how `tests/test_handoff.py` uses a fake Run to reproduce the error.

</details>
