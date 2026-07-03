# MLOps Guide — Experiment Tracking & Versioning

This document outlines how Ocular AI uses MLflow and DVC to manage model experiments, datasets, and pipelines.

---

## 1. MLflow Experiment Tracking

The training script `ml_pipeline/train_effnet.py` is configured to log all parameters and metrics to an MLflow server.

### Logging Parameters
We track:
- Optimizer and learning rate scheduler.
- Focal loss parameters (alpha and gamma).
- Batch size, learning rate, and native image dimensions.

### Logging Metrics
We trace:
- Training loss and validation loss per epoch.
- Macro F1-score and ROC-AUC per epoch.
- Final test-set performance metrics (overall accuracy, precision, recall, and specificities per class).

### Logging Artifacts
At the end of a run, we log:
- **Confusion Matrix plot**: Visualizes true vs. predicted positives/negatives.
- **Model Checkpoints**: Logged and registered to the **Model Registry** under the name `Ocular_EfficientNet_B3` when ROC-AUC $\ge 0.75$.

---

## 2. DVC Data & Model Versioning

We version raw inputs and PyTorch checkpoints outside of Git to avoid repository bloat.

### Versioning Datasets
To track the dataset directory:
```bash
dvc add data
```
This updates the `data.dvc` pointer file. You can commit `data.dvc` to Git.

### DVC Pipeline Stages
The pipeline is structured inside `dvc.yaml`:
```yaml
stages:
  train:
    cmd: python ml_pipeline/train_effnet.py
    deps:
      - ml_pipeline/train_effnet.py
      - ml_pipeline/dataset.py
      - ml_pipeline/model_effnet.py
      - ml_pipeline/loss.py
      - data/train.csv
      - data/val.csv
      - data/test.csv
    params:
      - configs/training.yaml:
        - training.epochs
        - training.batch_size
        - training.learning_rate
    outs:
      - checkpoints/best_model.pth
```

### Reproducing the Pipeline
To rerun training in a fully reproducible way based on changed parameters/inputs:
```bash
dvc repro
```
To fetch files tracked on the default remote storage:
```bash
dvc pull
```
