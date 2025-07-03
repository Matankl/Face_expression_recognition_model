#!/usr/bin/env python
"""
Random Forest classifier for face expression recognition using either
DeepFace ArcFace embeddings (default) or MediaPipe landmarks.

Usage example:
    python random_forest_classifier.py \
        --data_dir /path/to/fer_dataset \
        --feature emb \
        --n_estimators 500 \
        --save_model models/rf_emb.joblib
"""

import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from DataLoader import FaceExpressionLandmarksDS
from const import *


def extract_features(dataset: FaceExpressionLandmarksDS, feature_type: str):
    """Iterate through the dataset and return (X, y) numpy arrays."""
    X, y = [], []
    for i in range(len(dataset)):
        _, lms, emb, label = dataset[i]
        feat = emb.numpy() if feature_type == "emb" else lms.flatten().numpy()
        X.append(feat)
        y.append(label)
    return np.stack(X), np.array(y)


def main():
    # Set up arguments
    # feature = "emb"  # Default feature type
    feature = "lms"
    n_estimators = 20  # Default number of trees
    max_depth = None  # Default max depth
    save_model = True

    parser = argparse.ArgumentParser(
        description="Random Forest classifier for face expression recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )


    parser.add_argument("--save_model", type=str, default=None,
                        help="Path to save the trained model (.joblib)")


    # ──────────────────────────────────────────────────────────────
    #  Prepare data
    # ──────────────────────────────────────────────────────────────
    train_ds = FaceExpressionLandmarksDS(DATA_DIR, split="train")
    val_ds   = FaceExpressionLandmarksDS(DATA_DIR, split="validation")

    X_train, y_train = extract_features(train_ds, feature)
    X_val,   y_val   = extract_features(val_ds,   feature)

    # ──────────────────────────────────────────────────────────────
    #  Train classifier
    # ──────────────────────────────────────────────────────────────
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # ──────────────────────────────────────────────────────────────
    #  Evaluate
    # ──────────────────────────────────────────────────────────────
    y_pred = clf.predict_proba(X_val)
    evaluate_and_save(np.array(y_val), y_pred, "evaluation_results.txt")


if __name__ == "__main__":
    main()
