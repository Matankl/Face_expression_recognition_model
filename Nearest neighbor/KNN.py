#!/usr/bin/env python
"""
K‑Nearest Neighbors classifier for face expression recognition.
Supports training on either ArcFace embeddings or 468‑point facial landmarks.

Example run:
    python knn_classifier.py --data_dir /data/FER --feature lms --k 7
"""

import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from DataLoader import FaceExpressionLandmarksDS
from const import *


# ───────────────────────────────────────────────────────────────────────────────
#  Helper: extract design‑matrix X and label vector y from dataset
# ───────────────────────────────────────────────────────────────────────────────

def _extract(ds: FaceExpressionLandmarksDS, feature: str):
    X, y = [], []
    for i in range(len(ds)):
        _, lms, emb, label = ds[i]
        feat = emb.numpy() if feature == "emb" else lms.flatten().numpy()
        X.append(feat)
        y.append(label)
    return np.stack(X), np.array(y)


# ───────────────────────────────────────────────────────────────────────────────
#  CLI entry‑point
# ───────────────────────────────────────────────────────────────────────────────

def main():
    description = "KNN classifier for FER with landmarks/embeddings"
    # Set up arguments
    # feature = "emb"  # Default feature type
    feature = "lms"
    k = 5
    # weights = "uniform"  # Default weighting scheme
    weights = "distance"  # Default weighting scheme

    # ──────────────────────────────────────────────────────────────
    #  Prepare data
    # ──────────────────────────────────────────────────────────────
    train_ds = FaceExpressionLandmarksDS(DATA_DIR, split="train")
    val_ds   = FaceExpressionLandmarksDS(DATA_DIR, split="validation")

    X_train, y_train = _extract(train_ds, feature)
    X_val,   y_val   = _extract(val_ds,   feature)

    # ──────────────────────────────────────────────────────────────
    #  Train classifier
    # ──────────────────────────────────────────────────────────────
    clf = KNeighborsClassifier(n_neighbors=k, weights=weights, n_jobs=-1)
    clf.fit(X_train, y_train)

    # ──────────────────────────────────────────────────────────────
    #  Evaluate
    # ──────────────────────────────────────────────────────────────
    y_pred = clf.predict_proba(X_val)
    evaluate_and_save(np.array(y_val), y_pred, "evaluation_results.txt")



if __name__ == "__main__":
    main()
