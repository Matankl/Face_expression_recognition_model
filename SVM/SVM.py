#!/usr/bin/env python
"""
Support Vector Machine classifier for face expression recognition.
Choose between training on DeepFace embeddings or MediaPipe landmarks.

Example:
    python svm_classifier.py --data_dir ./FER2013 --feature emb \
                             --kernel rbf --c 10 --gamma scale
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
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


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def main():
    feature = "lms"
    kernel = 'rbf'
    c = 1                    # Regularization parameter
    gamma = "scale"          # Kernel coefficient
    grid_search = False  # Use grid search for hyperparameter tuning


    train_ds = FaceExpressionLandmarksDS(DATA_DIR, split="train")
    val_ds   = FaceExpressionLandmarksDS(DATA_DIR, split="validation")

    X_train, y_train = extract_features(train_ds, feature)
    X_val,   y_val   = extract_features(val_ds,   feature)
    if grid_search:
        print("Running Grid Search...")
        param_grid = {
            'svc__C': [0.1, 1, 10],
            'svc__gamma': ['scale', 'auto'],
            'svc__kernel': ['linear', 'rbf']
        }
        pipe = make_pipeline(StandardScaler(), SVC())
        clf = GridSearchCV(pipe, param_grid, cv=2, verbose=1, n_jobs=-1)
        clf.fit(X_train, y_train)
        print("Best parameters:", clf.best_params_)
    else:
        print("Training SVM classifier...")
        clf = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=c, gamma=gamma, probability=True)
)
        clf.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = clf.predict_proba(X_val)
    evaluate_and_save(np.array(y_val), y_pred, "evaluation_results.txt")

    print("Plotting confusion matrix...")
    class_names = sorted(set(y_val))
    plot_confusion_matrix(y_val, y_pred, class_names)



if __name__ == '__main__':
    main()