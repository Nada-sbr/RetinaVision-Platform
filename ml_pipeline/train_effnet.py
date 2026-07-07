import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
import yaml
from dataset import ODIRDataset
from loss import MultiLabelFocalLoss
from model_effnet import ODIREfficientNetB3
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from transforms import get_train_transforms, get_val_transforms

# ----------------------------------------------------
# Configuration & Hyperparameters (Loaded from YAML)
# ----------------------------------------------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
config_path = os.path.join(base_dir, "configs", "training.yaml")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        _train_config = yaml.safe_load(f) or {}
else:
    _train_config = {}

# Set variables from config
model_cfg = _train_config.get("model", {})
train_cfg = _train_config.get("training", {})
data_cfg = _train_config.get("data", {})

BATCH_SIZE = train_cfg.get("batch_size", 8)
LEARNING_RATE = float(train_cfg.get("learning_rate", 5e-5))
EPOCHS = train_cfg.get("epochs", 5)
IMAGE_SIZE = model_cfg.get("resolution", 300)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
LABELS = ["N", "D", "G", "C", "A", "H", "M", "O"]


# ----------------------------------------------------
# Helper Functions
# ----------------------------------------------------
def compute_pos_weights(train_csv_path: str) -> torch.Tensor:
    """
    Computes positive class weights for Focal Loss / BCE to handle class imbalance.
    """
    df = pd.read_csv(train_csv_path)
    num_samples = len(df)

    pos_weights = []
    for label in LABELS:
        pos_count = df[label].sum()
        neg_count = num_samples - pos_count
        weight = neg_count / max(pos_count, 1)
        pos_weights.append(weight)

    print("Calculated class weights (pos_weight) to combat imbalance:")
    for label, w in zip(LABELS, pos_weights):
        print(f"  Class {label}: {w:.2f}")

    return torch.tensor(pos_weights, dtype=torch.float32).to(DEVICE)


def calculate_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5):
    """
    Computes multi-label metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC per class and overall.
    """
    y_pred = (y_pred_probs >= threshold).astype(int)
    metrics = {}

    # Global metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )

    try:
        auc_macro = roc_auc_score(y_true, y_pred_probs, average="macro")
        auc_micro = roc_auc_score(y_true, y_pred_probs, average="micro")
    except ValueError:
        auc_macro, auc_micro = 0.5, 0.5

    metrics["overall"] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "auc_macro": auc_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "auc_micro": auc_micro,
    }

    # Per-class metrics
    metrics["per_class"] = {}
    for idx, label in enumerate(LABELS):
        class_true = y_true[:, idx]
        class_pred = y_pred[:, idx]
        class_probs = y_pred_probs[:, idx]

        precision, recall, f1, _ = precision_recall_fscore_support(class_true, class_pred, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(class_true, class_probs)
        except ValueError:
            auc = 0.5

        tn = ((class_true == 0) & (class_pred == 0)).sum()
        fp = ((class_true == 0) & (class_pred == 1)).sum()
        specificity = tn / max((tn + fp), 1)

        metrics["per_class"][label] = {
            "accuracy": accuracy_score(class_true, class_pred),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "auc": auc,
        }

    return metrics


# ----------------------------------------------------
# Loops
# ----------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_outputs.append(probs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    all_targets = np.concatenate(all_targets, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)

    metrics = calculate_metrics(all_targets, all_outputs)
    return epoch_loss, metrics


# ----------------------------------------------------
# Run Pipeline
# ----------------------------------------------------
def run_pipeline():
    print(f"Using device: {DEVICE}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 1. Datasets & Dataloaders
    train_dataset = ODIRDataset("data/train.csv", "preprocessed_images", transform=get_train_transforms(IMAGE_SIZE))
    val_dataset = ODIRDataset("data/val.csv", "preprocessed_images", transform=get_val_transforms(IMAGE_SIZE))
    test_dataset = ODIRDataset("data/test.csv", "preprocessed_images", transform=get_val_transforms(IMAGE_SIZE))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"EfficientNet-B3 DataLoaders initialized: Train batches = {len(train_loader)}, Val batches = {len(val_loader)}")

    # 2. Compute Class Imbalance Weights & Focal Loss Criterion
    pos_weight = compute_pos_weights("data/train.csv")
    # We use Multi-Label Focal Loss with alpha=0.25, gamma=2.0 and positive weights
    criterion = MultiLabelFocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)
    print("Multi-Label Focal Loss initialized with imbalance-aware pos_weights.")

    # 3. Instantiate EfficientNet-B3
    model = ODIREfficientNetB3(num_classes=8, pretrained=True).to(DEVICE)

    # Unfreeze final stages (features stage 7 & 8) and classifier for fine-tuning
    print("Fine-tuning: Freezing lower layers, keeping backbone features[7], features[8] and classifier trainable...")
    for name, param in model.named_parameters():
        # Keep features stage 7 (index 7), stage 8 (index 8) and classifier trainable
        if "backbone.classifier" in name or "backbone.features.7" in name or "backbone.features.8" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")

    # 4. Optimizer, Scheduler & Early Stopping
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2, verbose=True)

    best_val_loss = float("inf")
    early_stopping_patience = 4
    epochs_no_improve = 0

    # 5. Training Loop
    print("\n--- Starting Training (EfficientNet-B3 + Focal Loss) ---")

    from mlflow_tracker import MLflowTracker

    tracker = MLflowTracker()

    with tracker.start_run(run_name="EfficientNet_B3_Training"):
        # Log Hyperparameters to MLflow
        tracker.log_params(
            {
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "image_size": IMAGE_SIZE,
                "device": str(DEVICE),
                "loss_function": train_cfg.get("loss", "FocalLoss"),
                "focal_gamma": train_cfg.get("focal_gamma", 2.0),
                "optimizer": train_cfg.get("optimizer", "Adam"),
                "lr_scheduler": train_cfg.get("lr_scheduler", "StepLR"),
            }
        )

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_metrics = validate(model, val_loader, criterion, DEVICE)

            scheduler.step(val_loss)

            # Log epoch metrics to MLflow
            tracker.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_f1_macro": val_metrics["overall"]["f1_macro"],
                    "val_auc_macro": val_metrics["overall"]["auc_macro"],
                    "val_accuracy": val_metrics["overall"]["accuracy"],
                },
                step=epoch,
            )

            print(f"Epoch {epoch}/{EPOCHS}:")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(
                f"  Val Macro F1: {val_metrics['overall']['f1_macro']:.4f} | Val Macro AUC: {val_metrics['overall']['auc_macro']:.4f}"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"  -> Best model saved to {MODEL_SAVE_PATH}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print(f"  -> Early stopping triggered after {epoch} epochs.")
                    break

        # 6. Evaluation on Test Set
        print("\n--- Starting Final Evaluation on Test Set ---")
        if os.path.exists(MODEL_SAVE_PATH):
            model.load_state_dict(torch.load(MODEL_SAVE_PATH))
            print(f"Loaded best weights from {MODEL_SAVE_PATH}")

        test_loss, test_metrics = validate(model, test_loader, criterion, DEVICE)

        # Log final test set metrics to MLflow
        tracker.log_metrics(
            {
                "test_loss": test_loss,
                "test_accuracy": test_metrics["overall"]["accuracy"],
                "test_f1_macro": test_metrics["overall"]["f1_macro"],
                "test_auc_macro": test_metrics["overall"]["auc_macro"],
            }
        )

        print(f"\nFinal Test Loss: {test_loss:.4f}")
        print("\n================ FINAL TEST SET METRICS ================")
        print(f"Subset Accuracy: {test_metrics['overall']['accuracy'] * 100:.2f}%")
        print(f"Macro F1-score:  {test_metrics['overall']['f1_macro']:.4f}")
        print(f"Macro ROC-AUC:   {test_metrics['overall']['auc_macro']:.4f}")
        print("--------------------------------------------------------")

        for label in LABELS:
            m = test_metrics["per_class"][label]
            # Log per-class metrics to MLflow
            tracker.log_metrics(
                {
                    f"test_class_{label}_accuracy": m["accuracy"],
                    f"test_class_{label}_precision": m["precision"],
                    f"test_class_{label}_recall": m["recall"],
                    f"test_class_{label}_f1": m["f1"],
                    f"test_class_{label}_auc": m["auc"],
                }
            )

            print(f"Class {label}:")
            print(f"  Accuracy:    {m['accuracy'] * 100:.2f}%")
            print(f"  Precision:   {m['precision']:.4f}")
            print(f"  Recall/Sens: {m['recall']:.4f}")
            print(f"  Specificity: {m['specificity']:.4f}")
            print(f"  F1-score:    {m['f1']:.4f}")
            print(f"  ROC-AUC:     {m['auc']:.4f}")
            print("--------------------------------------------------------")

        # Generate and save a confusion matrix plot for the test set
        model.eval()
        all_targets = []
        all_outputs = []
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(probs.cpu().numpy())

        all_targets = np.concatenate(all_targets, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_preds = (all_outputs >= 0.5).astype(int)

        mcm = multilabel_confusion_matrix(all_targets, all_preds)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()

        for idx, label in enumerate(LABELS):
            sns.heatmap(
                mcm[idx],
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=axes[idx],
                cbar=False,
                xticklabels=["Pred -", "Pred +"],
                yticklabels=["True -", "True +"],
            )
            axes[idx].set_title(f"Class {label}")

        plt.suptitle("EfficientNet-B3 Confusion Matrices per Disease Class on Test Set")
        plt.tight_layout()
        plot_path = os.path.join(CHECKPOINT_DIR, "confusion_matrices.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Confusion matrices plot saved to {plot_path}")

        # Log artifacts and model to MLflow registry
        tracker.log_artifact(MODEL_SAVE_PATH, artifact_path="checkpoints")
        tracker.log_artifact(plot_path, artifact_path="plots")

        # Auto register best model if macro AUC is high
        if test_metrics["overall"]["auc_macro"] >= 0.75:
            print("Registering best model to MLflow Model Registry...")
            tracker.log_model(model, artifact_path="model", registered_model_name="Ocular_EfficientNet_B3")


if __name__ == "__main__":
    run_pipeline()
