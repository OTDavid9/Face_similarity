import base64
import cv2
import numpy as np
import pandas as pd
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet

# Initialize face detector and embedder once
detector = MTCNN()
embedder = FaceNet()

def decode_base64_image(base64_str):
    """Convert base64 string to OpenCV image."""
    try:
        img_data = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"❌ Decoding error: {e}")
        return None

def preprocess_image(img, scale=2.0):
    """Upscale and apply histogram equalization."""
    try:
        img_resized = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        yuv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        img_eq = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return img_eq
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return img  # fallback to original

def zoom_crop_center(img, crop_ratio=0.6):
    """Crop the center portion of the image."""
    try:
        h, w, _ = img.shape
        ch, cw = int(h * crop_ratio), int(w * crop_ratio)
        y1 = (h - ch) // 2
        x1 = (w - cw) // 2
        cropped = img[y1:y1+ch, x1:x1+cw]
        return cv2.resize(cropped, (w, h))  # Resize back to original size
    except Exception as e:
        print(f"❌ Cropping error: {e}")
        return img  # fallback to original

def get_face_embedding(img):
    """
    Attempt to detect face and return embedding.
    Applies preprocessing and fallback strategies to increase success rate.
    Returns:
        List of 512 floats if successful, else None.
    """
    try:
        attempts = []

        # Try raw image
        attempts.append(("Original", img))

        # Try histogram equalized + resized image
        attempts.append(("Preprocessed", preprocess_image(img)))

        # Try center cropped version
        attempts.append(("CenterCrop", zoom_crop_center(img)))

        for stage, attempt_img in attempts:
            img_rgb = cv2.cvtColor(attempt_img, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(img_rgb)

            if faces:
                print(f"✅ Face detected after {stage}")
                x, y, w, h = faces[0]['box']
                x, y = max(0, x), max(0, y)
                face = attempt_img[y:y+h, x:x+w]
                face = cv2.resize(face, (160, 160))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                # ✅ Save the cropped face image
                cv2.imwrite("face_used_for_embedding.png", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f"face_{stage.lower()}.png", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

                embedding = embedder.embeddings([face])[0]
                return embedding

        print("❌ Face not detected after all attempts.")
        return None
    except Exception as e:
        print(f"🚨 Embedding error: {e}")
        return None
