# face_expression_landmarks_dataset.py
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import mediapipe as mp
import numpy as np
from typing import Tuple

# ────────────────────────────────────────────────────────────────────────────────
#  Helper: initialise MediaPipe Face Mesh once (fast on CPU for 48×48 images)
# ────────────────────────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
_face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,         # run on individual images, not video
    max_num_faces=1,                # dataset has single face per image
    refine_landmarks=False,         # no iris landmarks ––> exactly 468 points
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


class FaceExpressionLandmarksDS(Dataset):
    """
    Loads 48×48 face images and returns:

        img_tensor  : torch.float32, shape [1, 48, 48], values ∈ [-1, 1]
        lms_tensor  : torch.float32, shape [468, 2], (x, y) pixel coords
        label       : int in [0 … 6]

    The class mapping 0-6 follows the folder names in torchvision.datasets.ImageFolder.
    """

    def __init__(self, root_dir: str, split: str = "train"):
        super().__init__()

        # torchvision handles class folders & labels
        self.ds = datasets.ImageFolder(
            root=os.path.join(root_dir, split),
            transform=transforms.Compose([
                transforms.Grayscale(num_output_channels=1),   # FER images are usually 1-ch
                transforms.ToTensor(),                         # (H,W) → [1,H,W] & [0,1]
                transforms.Normalize((0.5,), (0.5,)),          # mean≃0, std≃1
            ])
        )

    # ────────────────────────────────────────────────────────────────────────────
    #  Convert MediaPipe landmarks (relative coords) → tensor [468,2] in pixels
    # ────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _mp_landmarks_to_tensor(results, img_shape: Tuple[int, int]) -> torch.Tensor:
        h, w = img_shape
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]          # first face only
            coords = np.array([[p.x * w, p.y * h] for p in lm.landmark],
                              dtype=np.float32)           # shape (468,2)
        else:                                             # no detection → zeros
            coords = np.zeros((468, 2), dtype=np.float32)

        return torch.from_numpy(coords)                   # → torch.float32

    # ────────────────────────────────────────────────────────────────────────────
    #  Standard dataset methods
    # ────────────────────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        # Get image tensor and label from ImageFolder
        img_tensor, label = self.ds[idx]

        # Read original BGR image for landmark detection
        img_path, _ = self.ds.imgs[idx]
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        h, w, _ = img_bgr.shape

        # Detect landmarks using MediaPipe Face Mesh
        results = _face_mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        lm_tensor = self._mp_landmarks_to_tensor(results, (h, w))  # shape [468, 2]

        # Filter: Keep only expression-relevant landmarks
        expression_indices = list(range(55, 66))  # left eyebrow
        expression_indices += list(range(285, 296))  # right eyebrow
        expression_indices += list(range(33, 42))  # left eye
        expression_indices += list(range(133, 142))  # right eye
        expression_indices += [6, 7, 8, 97, 98, 168, 169, 170, 171, 172]  # nose
        expression_indices += list(range(61, 89))  # outer mouth
        expression_indices += list(range(291, 319))  # inner mouth
        expression_indices = sorted(set(expression_indices))

        lm_tensor = lm_tensor[expression_indices, :]  # shape → [~100, 2]

        print("Landmarks shape:", lm_tensor.shape)

        return img_tensor, lm_tensor, label


# ────────────────────────────────────────────────────────────────────────────────
#  Convenience factory: train & validation DataLoaders
# ────────────────────────────────────────────────────────────────────────────────
def make_loaders(data_dir: str,
                 batch_size: int = 64,
                 num_workers: int = 2):

    train_ds = FaceExpressionLandmarksDS(data_dir, split="train")
    val_ds   = FaceExpressionLandmarksDS(data_dir, split="val")

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)

    print("Class-to-index mapping:", train_ds.ds.class_to_idx)
    return train_loader, val_loader
