import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, multilabel_confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import ODIRDataset
from transforms import get_train_transforms, get_val_transforms
from model import ODIRResNet50

# ----------------------------------------------------
# Configuration & Hyperparameters
# ----------------------------------------------------
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
EPOCHS = 5
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
LABELS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

# ----------------------------------------------------
# Helper Functions
# ----------------------------------------------------
def compute_pos_weights(train_csv_path: str) -> torch.Tensor:
    """
    Computes positive class weights for BCEWithLogitsLoss to handle class imbalance.
    pos_weight = negative_samples / positive_samples
    """
    df = pd.read_csv(train_csv_path)
    num_samples = len(df)
    
    pos_weights = []
    for label in LABELS:
        pos_count = df[label].sum()
        neg_count = num_samples - pos_count
        # Prevent division by zero
        weight = neg_count / max(pos_count, 1)
        pos_weights.append(weight)
        
    print(f"Calculated class weights (pos_weight) to combat imbalance:")
    for l, w in zip(LABELS, pos_weights):
        print(f"  Class {l}: {w:.2f}")
        
    return torch.tensor(pos_weights, dtype=torch.float32).to(DEVICE)

def calculate_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5):
    """
    Computes multi-label metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC per class and overall.
    """
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    metrics = {}
    
    # Global metrics (macro averages)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    try:
        auc_macro = roc_auc_score(y_true, y_pred_probs, average='macro')
        auc_micro = roc_auc_score(y_true, y_pred_probs, average='micro')
    except ValueError:
        # Fallback if some classes have no positive samples in the batch/split
        auc_macro, auc_micro = 0.5, 0.5
        
    metrics['overall'] = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'auc_macro': auc_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'auc_micro': auc_micro
    }
    
    # Per-class metrics
    metrics['per_class'] = {}
    for idx, label in enumerate(LABELS):
        class_true = y_true[:, idx]
        class_pred = y_pred[:, idx]
        class_probs = y_pred_probs[:, idx]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            class_true, class_pred, average='binary', zero_division=0
        )
        try:
            auc = roc_auc_score(class_true, class_probs)
        except ValueError:
            auc = 0.5
            
        # Specificity calculation (TN / (TN + FP))
        tn = ((class_true == 0) & (class_pred == 0)).sum()
        fp = ((class_true == 0) & (class_pred == 1)).sum()
        specificity = tn / max((tn + fp), 1)
        
        metrics['per_class'][label] = {
            'accuracy': accuracy_score(class_true, class_pred),
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'auc': auc
        }
        
    return metrics

# ----------------------------------------------------
# Training, Validation, and Testing Loops
# ----------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass & optimize
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
            
            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(outputs)
            
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(probs.cpu().numpy())
            
    epoch_loss = running_loss / len(dataloader.dataset)
    all_targets = np.concatenate(all_targets, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    
    metrics = calculate_metrics(all_targets, all_outputs)
    return epoch_loss, metrics

# ----------------------------------------------------
# Main Pipeline execution
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
    
    print(f"DataLoaders initialized: Train batches = {len(train_loader)}, Val batches = {len(val_loader)}")
    
    # 2. Compute Imbalance Weights & Loss Criterion
    pos_weight = compute_pos_weights("data/train.csv")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 3. Instantiate Model
    model = ODIRResNet50(num_classes=8, pretrained=True).to(DEVICE)
    
    # Freeze lower layers, unfreeze layer4 and fc layer for partial fine-tuning
    print("Fine-tuning: Freezing lower layers, keeping backbone.layer4 and backbone.fc trainable...")
    for name, param in model.named_parameters():
        if "backbone.fc" in name or "backbone.layer4" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    # Print trainable parameters count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")
    
    # 4. Optimizer, Scheduler, and Early Stopping setup
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    best_val_loss = float('inf')
    early_stopping_patience = 4
    epochs_no_improve = 0
    
    # 5. Training Loop
    print("\n--- Starting Training ---")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_metrics = validate(model, val_loader, criterion, DEVICE)
        
        # Step LR Scheduler
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch}/{EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val Macro F1: {val_metrics['overall']['f1_macro']:.4f} | Val Macro AUC: {val_metrics['overall']['auc_macro']:.4f}")
        
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
        # Load best weights
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print(f"Loaded best weights from {MODEL_SAVE_PATH}")
        
    test_loss, test_metrics = validate(model, test_loader, criterion, DEVICE)
    
    print(f"\nFinal Test Loss: {test_loss:.4f}")
    print("\n================ FINAL TEST SET METRICS ================")
    print(f"Subset Accuracy: {test_metrics['overall']['accuracy'] * 100:.2f}%")
    print(f"Macro F1-score:  {test_metrics['overall']['f1_macro']:.4f}")
    print(f"Macro ROC-AUC:   {test_metrics['overall']['auc_macro']:.4f}")
    print("--------------------------------------------------------")
    
    for label in LABELS:
        m = test_metrics['per_class'][label]
        print(f"Class {label}:")
        print(f"  Accuracy:    {m['accuracy'] * 100:.2f}%")
        print(f"  Precision:   {m['precision']:.4f}")
        print(f"  Recall/Sens: {m['recall']:.4f}")
        print(f"  Specifity:   {m['specificity']:.4f}")
        print(f"  F1-score:    {m['f1']:.4f}")
        print(f"  ROC-AUC:     {m['auc']:.4f}")
        print("--------------------------------------------------------")
        
    # Generate and save a confusion matrix plot for the test set
    # Load dataset for final inference
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
    
    # Multilabel confusion matrix
    mcm = multilabel_confusion_matrix(all_targets, all_preds)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for idx, label in enumerate(LABELS):
        sns.heatmap(mcm[idx], annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=['Pred -', 'Pred +'], yticklabels=['True -', 'True +'])
        axes[idx].set_title(f"Class {label}")
        
    plt.suptitle("Confusion Matrices per Disease Class on Test Set")
    plt.tight_layout()
    plot_path = os.path.join(CHECKPOINT_DIR, "confusion_matrices.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Confusion matrices plot saved to {plot_path}")

if __name__ == "__main__":
    run_pipeline()
