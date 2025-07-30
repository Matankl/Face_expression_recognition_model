"""
eval_metrics.py
Evaluate a trained EfficientFER model:
  • accuracy
  • precision, recall, F1 (per-class & macro)
  • confusion-matrix plot

Assumes:
  – best_model.pth was saved by your training code
  – get_train_dataloaders() is available and returns (train_loader, val_loader)
  – EfficientFER architecture and EMA wrapper are unchanged
"""

import torch
from tqdm import tqdm
import torch.nn.functional as F
from ema_pytorch import EMA
from efficientfer import EfficientFER
from dataloader import get_train_dataloaders
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------------------
# Configuration – keep it in sync with the one used in training
# --------------------------------------------------------------------------------------
CONFIG = {
    'data_dir':  r'C:\Users\matan\Desktop\Code\DataSets\Face_expression_recognition',
    'data_dir2': r'C:\Users\matan\Desktop\Code\DataSets\affectnet',
    'num_classes': 7,
    'batch_size': 64,
    'num_workers': 1,
    'image_size': 224,
    'checkpoint_path': Path('checkpoints') / 'best_model.pth',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'cm_fig_path': 'confusion_matrix.png',
    'class_names': ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],  # edit if different
}

# --------------------------------------------------------------------------------------
# Utility – create model, wrap in EMA, and load the checkpoint
# --------------------------------------------------------------------------------------
def load_ema_model(checkpoint_path: Path) -> EMA:
    """
    Rebuild the EfficientFER + EMA wrapper and load weights from a checkpoint.
    Returns the EMA *wrapper* so that `ema_model.ema_model` gives the averaged net.
    """
    # 1. Build the base network
    model = EfficientFER(num_classes=CONFIG['num_classes']).to(CONFIG['device'])

    # 2. Build an EMA wrapper with identical hyper-params to training
    ema_model = EMA(
        model,
        beta=0.999,
        update_after_step=100,
        update_every=1,
    ).to(CONFIG['device'])

    # 3. Load checkpoint (weights, not optimizer/scheduler)
    ckpt = torch.load(checkpoint_path, map_location=CONFIG['device'])
    ema_model.load_state_dict(ckpt['ema_state_dict'])  # 👈 averaged weights
    ema_model.eval()                                   # inference-only
    return ema_model


# --------------------------------------------------------------------------------------
# Evaluation loop
# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(ema_model: EMA, loader):
    """
    Iterate over `loader`, accumulate predictions / labels, and compute metrics.
    """
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="Evaluating", leave=False):
        imgs   = imgs.to(CONFIG['device'])
        labels = labels.to(CONFIG['device'])

        logits = ema_model(imgs)            # EMA wrapper is callable
        preds  = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    preds  = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    # --- global metrics ---
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=range(CONFIG['num_classes']), average=None, zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )

    # --- confusion matrix ---
    cm = confusion_matrix(labels, preds, labels=range(CONFIG['num_classes']))

    return {
        'acc': acc,
        'prec': prec,
        'rec': rec,
        'f1': f1,
        'prec_macro': prec_macro,
        'rec_macro': rec_macro,
        'f1_macro': f1_macro,
        'cm': cm,
    }


# --------------------------------------------------------------------------------------
# Pretty-print & plot helpers
# --------------------------------------------------------------------------------------
def print_metrics(metrics):
    print("\nOverall accuracy: {:.4f}".format(metrics['acc']))
    print("Macro precision : {:.4f}".format(metrics['prec_macro']))
    print("Macro recall    : {:.4f}".format(metrics['rec_macro']))
    print("Macro F1-score  : {:.4f}\n".format(metrics['f1_macro']))

    # Per-class table
    header = "{:<10s} {:>9s} {:>9s} {:>9s}".format("Class", "Precision", "Recall", "F1")
    print(header)
    print("-" * len(header))
    for i, cname in enumerate(CONFIG['class_names']):
        print("{:<10s} {:9.4f} {:9.4f} {:9.4f}".format(
            cname, metrics['prec'][i], metrics['rec'][i], metrics['f1'][i]
        ))

def plot_confusion_matrix(cm: np.ndarray, fig_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CONFIG['class_names'],
        yticklabels=CONFIG['class_names'],
        ax=ax
    )
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    print(f"\nConfusion-matrix figure saved to: {fig_path}")


# --------------------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------------------
def main():
    print("Loading EMA model from:", CONFIG['checkpoint_path'])
    ema_model = load_ema_model(CONFIG['checkpoint_path'])

    print("Building data loader (validation split)…")
    _, val_loader = get_train_dataloaders(
        CONFIG['data_dir'],
        CONFIG['batch_size'],
        CONFIG['num_workers'],
        CONFIG['image_size'],
        CONFIG['data_dir2']
    )

    print("Running evaluation …")
    metrics = evaluate(ema_model, val_loader)

    print_metrics(metrics)
    plot_confusion_matrix(metrics['cm'], CONFIG['cm_fig_path'])


if __name__ == "__main__":
    main()
