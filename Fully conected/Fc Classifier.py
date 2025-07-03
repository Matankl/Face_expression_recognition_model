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
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        preds.extend(out.argmax(1).cpu().numpy())
        trues.extend(y.cpu().numpy())
    acc = accuracy_score(trues, preds)
    print(classification_report(trues, preds))
    return acc

# ───────────────────────────────────────────────────────────────────────────────
#  CLI
# ───────────────────────────────────────────────────────────────────────────────

def main():
    description = "Fully‑connected network for FER"
    # feature = "emb"  # Default feature type
    feature = "lms"

    parser = argparse.ArgumentParser(description="Fully‑connected network for FER")

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_model", type=str, default=None)
    args = parser.parse_args()

    train_base = FaceExpressionLandmarksDS(args.data_dir, split="train")
    val_base   = FaceExpressionLandmarksDS(args.data_dir, split="val")

    train_ds = make_feature_dataset(train_base, args.feature)
    val_ds   = make_feature_dataset(val_base,   args.feature)

    input_dim = train_ds[0][0].shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FCNet(input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        acc  = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: loss={loss:.4f}  val_acc={acc*100:.2f}%")

        if args.save_model and acc > best:
            best = acc
            os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
            torch.save(model.state_dict(), args.save_model)
            print(f"New best model saved to {args.save_model}")

if __name__ == "__main__":
    main()
