from collections import defaultdict
from DataLoader import FaceExpressionLandmarksDS
import torch
import tqdm

# Path to dataset root (containing "train" and "val" folders)
DATASET_PATH = r"C:\Users\matan\Desktop\Code\DataSets\Face_expression_recognition"

# Load both train and val
splits = ["train", "validation"]
class_counts = defaultdict(int)
detected_counts = defaultdict(int)

for split in splits:
    dataset = FaceExpressionLandmarksDS(DATASET_PATH, split=split)
    idx_to_class = {v: k for k, v in dataset.ds.class_to_idx.items()}  # 0→"angry", etc.
    print("dataset size:", len(dataset))
    # for i in tqdm(range(len(dataset)), desc=f"Processing {split}"):
    for i in range(len(dataset)):
        print(f"Processing {split} image {i+1}/{len(dataset)}")
        _, landmarks, label = dataset[i]
        class_counts[label] += 1

        if torch.any(landmarks != 0):  # At least one nonzero coordinate
            detected_counts[label] += 1

# Print results
print(f"{'Class':<10} {'Total':>6} {'Detected':>9}")
for class_id in sorted(class_counts.keys()):
    class_name = idx_to_class[class_id]
    total = class_counts[class_id]
    detected = detected_counts[class_id]
    print(f"{class_name} {total} {detected}")
