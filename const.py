

DATA_DIR = r"C:\Users\matan\Desktop\Code\DataSets\Face_expression_recognition"
DATA_DIR2 = r"C:\Users\matan\Desktop\Code\DataSets\affectnet"
OUT_DIR_GNN = r"/GNN model/Trained_GNN_models"
OUT_DIR_FCC = r"C:\Users\matan\Desktop\Code\Face_expression_recognition_model\Fully conected"
BATCH_SIZE = 64
WORKERS_NUM = 1
K_NEAREST_NEIGHBOR = 8
EPOCHS = 60
lr = 2e-2
HIDDEN = 64


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def evaluate_and_save(y_true, y_probs, output_path):
    """
    Save classification metrics (acc, recall, precision, f1) for top-1, top-2, and top-3 predictions.
    """
    # Ensure y_probs is a probability/distribution matrix
    top_preds = np.argsort(y_probs, axis=1)[:, ::-1]  # Descending order

    results = ""
    for k in [1, 2, 3]:
        top_k_preds = top_preds[:, :k]
        correct = np.any(top_k_preds == y_true[:, None], axis=1)
        y_pred_topk = top_k_preds[:, 0]  # For computing metrics like precision, f1

        acc = np.mean(correct)

        results += f"Top-{k} Evaluation:\n"
        results += f"  Accuracy:  {acc * 100:.2f}%\n"
        if k == 1:
            precision = precision_score(y_true, y_pred_topk, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred_topk, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred_topk, average='macro', zero_division=0)

            results += f"  Precision: {precision * 100:.2f}%\n"
            results += f"  Recall:    {recall * 100:.2f}%\n"
            results += f"  F1-score:  {f1 * 100:.2f}%\n\n"

    with open(output_path, "w") as f:
        f.write(results)

    print(f"Evaluation saved to {output_path}")
