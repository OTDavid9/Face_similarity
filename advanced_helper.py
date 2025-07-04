import base64
import cv2
import numpy as np
from typing import Callable, Optional


def decode_base64_image(base64_str: str) -> Optional[np.ndarray]:
    """Convert base64 string to OpenCV image."""
    try:
        base64_str = base64_str.strip().split(",")[-1]  # Handle data URL prefix
        img_data = base64.b64decode(base64_str + '===')
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"❌ Decoding error: {e}")
        return None


def preprocess_image(img: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """Upscale and apply histogram equalization."""
    try:
        img_resized = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        yuv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        img_eq = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return img_eq
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return img  # fallback


def zoom_crop_center(img: np.ndarray, crop_ratio: float = 0.6) -> np.ndarray:
    """Crop the center portion of the image."""
    try:
        h, w, _ = img.shape
        ch, cw = int(h * crop_ratio), int(w * crop_ratio)
        y1 = (h - ch) // 2
        x1 = (w - cw) // 2
        cropped = img[y1:y1+ch, x1:x1+cw]
        return cv2.resize(cropped, (w, h))  # resize to original
    except Exception as e:
        print(f"❌ Cropping error: {e}")
        return img  # fallback


def get_face_embedding(
    img: np.ndarray,
    detect_fn: Callable[[np.ndarray], Optional[np.ndarray]],
    embed_fn: Callable[[np.ndarray], Optional[np.ndarray]]
) -> Optional[np.ndarray]:
    """
    Generic face embedding extractor using pluggable detector and embedder.

    Args:
        img: Input image (OpenCV BGR format).
        detect_fn: Function that accepts image and returns cropped face (BGR).
        embed_fn: Function that accepts cropped face and returns embedding.

    Returns:
        np.ndarray embedding vector or None.
    """
    attempts = [
        ("Original", img),
        ("Preprocessed", preprocess_image(img)),
        ("CenterCrop", zoom_crop_center(img))
    ]

    for stage, attempt_img in attempts:
        try:
            face_img = detect_fn(attempt_img)
            if face_img is None:
                print(f"❌ No face detected at {stage} stage.")
                continue

            embedding = embed_fn(face_img)
            if embedding is not None:
                print(f"✅ Embedding extracted after {stage} stage.")
                cv2.imwrite(f"face_{stage.lower()}.png", face_img)
                return embedding
            else:
                print(f"❌ Embedding failed after {stage} stage.")
        except Exception as e:
            print(f"🚨 Error at {stage}: {e}")

    print("❌ All embedding attempts failed.")
    return None
