"""
Real-time face expression recognition from webcam.

Features
--------
1. Streams live video from the default webcam and displays it in a window.
2. Detects faces in each frame (using MediaPipe) and draws a green bounding box.
3. Every `--interval` seconds it crops the detected face, preprocesses it, and feeds it to
   your EfficientFER classifier to predict the expression.
4. Renders the predicted expression label (with confidence) beside the bounding box.
5. Exits cleanly when the user presses the **q** key.

Requirements
------------
$ pip install torch torchvision opencv-python mediapipe
# plus your EfficientFER implementation

Usage
-----
python realtime_face_expression.py --checkpoint path/to/your_weights.pth --interval 0.5

Press **q** in the video window to quit.
"""

import time
import argparse
from pathlib import Path

import cv2  # OpenCV for video capture and drawing
import torch  # PyTorch for inference
from torchvision import transforms  # Standard image transforms
import mediapipe as mp  # Robust real‑time face detector

# -----------------------------------------------------------------------------
# 1.  Model loading helper -----------------------------------------------------
# -----------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device):
    """Load EfficientFER weights and put the model in eval mode."""

    # Instantiate your network. Replace num_classes if you changed it.
    from efficientfer import EfficientFER  # local import to avoid heavy import at module level

    model = EfficientFER(num_classes=7)  # 7 facial expression classes

    # Load the checkpoint (CPU‑compatible) and extract the state dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    # Populate the network weights (strict=False lets you ignore mismatched heads, if any)
    load_msg = model.load_state_dict(state_dict, strict=False)
    print("[Model] State‑dict loaded (strict=False):", load_msg)

    model.to(device).eval()  # inference‑only
    return model

# -----------------------------------------------------------------------------
# 2.  Main realtime loop -------------------------------------------------------
# -----------------------------------------------------------------------------

def main(checkpoint: Path, interval: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] Using device: {device}")

    # ─── Load expression classifier ───────────────────────────────────────────
    model = load_model(checkpoint, device)

    # Ensure the label order matches your training.
    labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    # Pre‑processing pipeline expected by EfficientFER (RGB 0‑1 range → normalized)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),  # convert NumPy → PIL
        # transforms.Resize((400, 400)),  # resize to 48x48 pixels (model input size)
        transforms.ToTensor(),  # 0‑1 float tensor, shape [C,H,W]
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),            ######## do i neet it? the numbers are correct?
    ])

    # ─── Face detector (MediaPipe) ────────────────────────────────────────────
    mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=0,  # 0: short range (~2m)
        min_detection_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        raise RuntimeError("Unable to open the webcam. Check camera permissions.")

    last_infer_t = 0.0  # wall‑clock time of last model inference
    prev_label, prev_conf = "", 0.0  # keep last prediction to avoid flicker

    try:
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                print("[Warning] Failed to grab frame; skipping…")
                continue

            # MediaPipe expects RGB images
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detection_result = mp_face.process(frame_rgb)

            now = time.time()
            ready_for_inference = (now - last_infer_t) >= interval

            if detection_result.detections:
                # Define expansion factor
                expansion = 1.25
                for det in detection_result.detections:
                    h, w, _ = frame_bgr.shape
                    rel = det.location_data.relative_bounding_box
                    # Convert rel coords to absolute
                    x = rel.xmin * w
                    y = rel.ymin * h
                    bw = rel.width * w
                    bh = rel.height * h

                    # New center‑based box with margin
                    cx = x + bw / 2
                    cy = y + bh / 2
                    new_w = bw * expansion
                    new_h = bh * expansion

                    x1 = int(max(cx - new_w / 2, 0))
                    y1 = int(max(cy - new_h / 2, 0))
                    x2 = int(min(cx + new_w / 2, w))
                    y2 = int(min(cy + new_h / 2, h))

                    face_roi = frame_rgb[y1:y2, x1:x2]

                    # ------------------------------------------------------------------
                    # 1. Run inference at most every `interval` seconds to save compute
                    # ------------------------------------------------------------------
                    if ready_for_inference and face_roi.size != 0:
                        last_infer_t = now

                        # Preprocess and move to device
                        inp = preprocess(face_roi).unsqueeze(0).to(device)

                        with torch.no_grad():
                            logits = model(inp)
                            print("[Model] Logits: ", logits)
                            probs = torch.softmax(logits, dim=1)
                            idx = int(torch.argmax(probs))
                            prev_label = labels[idx]
                            prev_conf = float(probs[0, idx])

                    # ------------------------------------------------------------------
                    # 2. Draw bounding box & (possibly cached) prediction
                    # ------------------------------------------------------------------
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame_bgr,
                        f"{prev_label} {prev_conf:.2f}",
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )

            # ─── Display ───────────────────────────────────────────────────────
            cv2.imshow("Real‑time Face Expression Recognition (press 'q' to quit)", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        mp_face.close()
        print("[System] Webcam and windows have been released.")


# -----------------------------------------------------------------------------
# 3.  CLI entry‑point ----------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    path = r"C:\Users\matan\Desktop\Code\Face_expression_recognition_model\EfficientNetv2\checkpoints\best_model.pth"
    interval = 0.5

    main(path, interval)
