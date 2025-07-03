"""
---------------------------------
End‑to‑end training & evaluation script for facial‑expression recognition
using MediaPipe landmarks as input to a Graph Convolutional Network (GCN).

"""

from __future__ import annotations

import os
import warnings
import logging

# ────────────────────────────────────────────────────────────────────────────────
#  Quiet noisy libraries *before* they import
# ────────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # TensorFlow / MediaPipe
os.environ.setdefault("GLOG_minloglevel", "2")           # Mediapipe C++ backend
warnings.filterwarnings("ignore", category=UserWarning)   # torch pin_memory, etc.
logging.getLogger("tensorflow").setLevel(logging.ERROR)
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass  # absl optional

from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# ────────────────────────────────────────────────────────────────────────────────
#  PyG imports (install with pip install torch_geometric)
# ────────────────────────────────────────────────────────────────────────────────
try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.transforms import KNNGraph
    from torch_geometric.nn import GCNConv, global_mean_pool
except ImportError as e:
    raise SystemExit("torch_geometric is required:  pip install torch-geometric") from e


# ────────────────────────────────────────────────────────────────────────────────
#  Import the dataset class
# ────────────────────────────────────────────────────────────────────────────────
from DataLoader import FaceExpressionLandmarksDS
from const import *


# ────────────────────────────────────────────────────────────────────────────────
#  Dataset adapter: converts landmark tensors into PyG Data objects
# ────────────────────────────────────────────────────────────────────────────────
class LandmarksGraphDS(torch.utils.data.Dataset):
    """Wraps FaceExpressionLandmarksDS and outputs torch_geometric Data objects."""

    def __init__(self, root_dir: str, split: str, transform=None):
        self.base_ds = FaceExpressionLandmarksDS(root_dir, split)
        self.transform = transform

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        _img, lms, features, label = self.base_ds[idx]    # image tensor not used here
        data = Data(x=lms.clone(),              # node features (2‑D coordinates)
                    pos=lms.clone(),            # positional info for KNNGraph
                    y=torch.tensor(label))
        if self.transform is not None:
            data = self.transform(data)
        return data


# ────────────────────────────────────────────────────────────────────────────────
#  Model: simple 3‑layer GCN + global mean pooling
# ────────────────────────────────────────────────────────────────────────────────
class LandmarkGCN(nn.Module):
    def __init__(self, in_channels: int = 2, hidden: int = 64, num_classes: int = 7):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)  # [batch_size, hidden]
        return self.fc(x)


# ────────────────────────────────────────────────────────────────────────────────
#  Training & evaluation routines
# ────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: nn.Module, loader: PyGDataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    y_true, y_pred = [], []
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = criterion(logits, batch.y)
        preds = logits.argmax(dim=1)
        correct += (preds == batch.y).sum().item()

        total += batch.y.size(0)
        loss_sum += loss.item() * batch.y.size(0)
        y_true.extend(batch.y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # print metrics
    print(f"Val Loss: {loss_sum / total:.4f} | acc: {acc:.4f} | precision: {precision:.4f} | recall: {recall:.4f} | f1: {f1:.4f}")

    return loss_sum / total, acc, precision, recall, f1


# ────────────────────────────────────────────────────────────────────────────────
#  Training loop
# ────────────────────────────────────────────────────────────────────────────────
def train_epoch(model: nn.Module, loader: PyGDataLoader, optimizer, device: torch.device) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch), batch.y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch.y.size(0)
    return running_loss / len(loader.dataset)


# ────────────────────────────────────────────────────────────────────────────────
#  Utility: build DataLoaders with KNN edges
# ────────────────────────────────────────────────────────────────────────────────
def make_graph_loaders(data_dir: str, batch_size: int, num_workers: int = 2, k: int = 8):
    knn = KNNGraph(k=k)
    train_ds = LandmarksGraphDS(data_dir, split="train", transform=knn)
    val_ds   = LandmarksGraphDS(data_dir, split="validation",   transform=knn)

    train_loader = PyGDataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=True)
    val_loader   = PyGDataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


# ────────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ────────────────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    train_loader, val_loader = make_graph_loaders(DATA_DIR, BATCH_SIZE,
                                                  WORKERS_NUM, K_NEAREST_NEIGHBOR)
    model = LandmarkGCN(in_channels=2, hidden=HIDDEN, num_classes=7).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"[Epoch {epoch:02d}/{EPOCHS}] "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}%")

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Path(OUT_DIR_GNN) / "best_model.pt")

    print("Training complete. Best val accuracy: {:.2f}%".format(best_val_acc * 100))


if __name__ == "__main__":
    main()
