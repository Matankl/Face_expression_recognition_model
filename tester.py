import os
import cv2
from torchvision import datasets
import mediapipe as mp
from tqdm import tqdm

# Path to the dataset folder that contains 'train' and 'val' subfolders
DATA_DIR = r"C:\Users\matan\Desktop\Code\DataSets\Face_expression_recognition"

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)

def count_successful_detections(split):
    # Load images using torchvision
    dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, split))

    total = len(dataset)
    success = 0

    print(f"Checking {split} images...")
    for img_path, _ in tqdm(dataset.imgs, total=total):
        img = cv2.imread(img_path)
        if img is None:
            continue  # skip unreadable images

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            success += 1

    return success, total

if __name__ == "__main__":
    success_train, total_train = count_successful_detections("train")
    success_val, total_val = count_successful_detections("validation")

    total_images = total_train + total_val
    total_success = success_train + success_val
    percentage = 100 * total_success / total_images

    print("\n──── Landmark Detection Summary ────")
    print(f"Train success: {success_train} / {total_train}")
    print(f"Val success  : {success_val} / {total_val}")
    print(f"Total success: {total_success} / {total_images}")
    print(f"Detection success rate: {percentage:.2f}%")
