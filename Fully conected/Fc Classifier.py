#!/usr/bin/env python
"""
Fully‑connected neural‑network classifier for face expression recognition.
Trains on pre‑extracted features (embeddings or landmarks) using PyTorch.

Example usage:
    python fc_classifier.py --data_dir ./FER2013 --feature emb \
                            --epochs 30 --batch_size 256 --lr 0.0005 \
                            --save_model models/fc_emb.pth
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
from DataLoader import FaceExpressionLandmarksDS
from const import *

# ───────────────────────────────────────────────────────────────────────────────
#  Feature extraction helper
# ───────────────────────────────────────────────────────────────────────────────

def make_feature_dataset(base_ds, feature: str) -> TensorDataset:
    """Extracts the requested feature from the base dataset and wraps it in
    a TensorDataset so we can use the usual PyTorch DataLoader."""
    X, y = [], []
    for i in range(len(base_ds)):
        _, lms, emb, label = base_ds[i]
        vec = emb if feature == "emb" else lms.flatten()
        X.append(vec)
        y.append(label)

    X = torch.stack(X)           # shape [N, D]
    y = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X, y)

# ───────────────────────────────────────────────────────────────────────────────
#  Simple feed‑forward network
# ───────────────────────────────────────────────────────────────────────────────

class FCNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ───────────────────────────────────────────────────────────────────────────────
#  Train / eval loops
# ───────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, output_path, epoch=None, lr=None):
    model.eval()
    all_logits, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    probs = torch.softmax(all_logits, dim=1).numpy()
    y_true = all_labels.numpy()
    y_pred = probs.argmax(axis=1)

    print(classification_report(y_true, y_pred))
    evaluate_and_save(y_true, probs, output_path, epoch, lr)
    acc = accuracy_score(y_true, y_pred)
    return acc


def evaluate_and_save(y_true, y_probs, output_path, epoch=None, lr=None):
    """
    Save classification metrics (acc, recall, precision, f1) for top-1, top-2, and top-3 predictions.
    """
    top_preds = np.argsort(y_probs, axis=1)[:, ::-1]  # Descending: most probable class first

    results = ""
    for k in [1, 2, 3]:
        top_k_preds = top_preds[:, :k]
        correct = np.any(top_k_preds == y_true[:, None], axis=1)
        acc = np.mean(correct)

        results += f"Top-{k} Evaluation:\n"
        results += f"  Accuracy:  {acc * 100:.2f}%\n"

        if k == 1:
            y_pred_top1 = top_k_preds[:, 0]
            precision = precision_score(y_true, y_pred_top1, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred_top1, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred_top1, average='macro', zero_division=0)

            results += f"  Precision: {precision * 100:.2f}%\n"
            results += f"  Recall:    {recall * 100:.2f}%\n"
            results += f"  F1-score:  {f1 * 100:.2f}%\n\n"

    if epoch is not None:
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_epoch{epoch}{ext}"

    if lr is not None:
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_lr{lr}{ext}"

    with open(output_path, "w") as f:
        f.write(results)

    print(f"Evaluation saved to {output_path}")
# ───────────────────────────────────────────────────────────────────────────────
#  CLI
# ───────────────────────────────────────────────────────────────────────────────

def main():
    description = "Fully‑connected network for FER"
    # feature = "emb"  # Default feature type
    feature = "lms"
    lr = [3e-2, 1e-2, 1e-3, 1e-4, 1e-5]  # Default learning rates


    # ──────────────────────────────────────────────────────────────
    #  Train classifier
    # ──────────────────────────────────────────────────────────────
    train_base = FaceExpressionLandmarksDS(DATA_DIR, split="train")
    val_base   = FaceExpressionLandmarksDS(DATA_DIR, split="validation")

    train_ds = make_feature_dataset(train_base, feature)
    val_ds   = make_feature_dataset(val_base,   feature)

    input_dim = train_ds[0][0].shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"


    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    for lr in lr:
        print(f"Training with learning rate: {lr}")
        # set up model, loss function, optimizer
        model = FCNet(input_dim).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best = 0.0
        counter = 0
        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(model, train_loader, criterion, optimizer, device)
            acc  = evaluate(model, val_loader, device, OUT_DIR_FCC, epoch, lr)
            print(f"Epoch {epoch}: loss={loss:.4f}  val_acc={acc*100:.2f}%")



            if acc > best:
                counter = 0
                best = acc
                os.makedirs(os.path.dirname(OUT_DIR_FCC), exist_ok=True)
            else:
                counter += 1
                if counter >= 5:
                    print("Early stopping")
                    break

if __name__ == "__main__":
    main()
