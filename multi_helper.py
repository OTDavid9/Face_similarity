"""
multi_helper.py  –  inference‑time embeddings for FaceNet, VGG‑Face, ArcFace
----------------------------------------------------------------------------

Returns a dict with any of: {"facenet": [...], "vgg": [...], "arcface": [...]}

• FaceNet (512‑D) uses keras‑facenet and our own MTCNN crop (160×160).
• VGG‑Face (4096‑D) and ArcFace (512‑D) use DeepFace.represent with MTCNN.
• If a model fails, its key is simply absent; the caller can still query
  Pinecone for whatever embeddings were produced.

Environment versions known to work (July 2025):
  numpy<2, tensorflow‑cpu==2.15.1, keras==2.15.1, deepface==0.0.79,
  mtcnn, keras‑facenet, opencv‑python‑headless
"""

from __future__ import annotations
from typing import Dict, List, Optional

import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from deepface import DeepFace

# ──────────────────────────────────────────────────────────────
# Optional: pre‑build VGG‑Face to avoid "sequential has never been called"
# (works on DeepFace <=0.0.92; ignored on later builds)
try:
    from deepface.basemodels import VGGFace
    VGGFace.loadModel().build(input_shape=(None, 224, 224, 3))
except Exception:
    pass  # safe to ignore if module structure changed

# ──────────────────────────────────────────────────────────────
# Initialise detectors / models once at import‑time
# ──────────────────────────────────────────────────────────────
_detector = MTCNN()
_facenet = FaceNet()  # outputs 512‑D vectors

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _crop_biggest_face(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Returns a 160×160 BGR crop of the *largest* detected face, or None.
    Two passes: native size, then 2× up‑scale.
    """
    for attempt in (1, 2):
        search_img = cv2.resize(img, None, fx=2, fy=2) if attempt == 2 else img
        faces = _detector.detect_faces(cv2.cvtColor(search_img, cv2.COLOR_BGR2RGB))
        if faces:
            x, y, w, h = faces[0]["box"]
            x, y = max(0, x), max(0, y)
            face = search_img[y : y + h, x : x + w]
            return cv2.resize(face, (160, 160))
    return None

# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────
def get_face_embedding(rgb_face):
    embeddings = {}

    # FaceNet - stable
    try:
        rep = DeepFace.represent(
            img_path=rgb_face,
            model_name="Facenet",
            enforce_detection=False
        )[0]
        embeddings["facenet"] = rep["embedding"]
    except Exception as e:
        print(f"[FaceNet] {e}")

    # VGG-Face - avoid calling .build()
    try:
        rep = DeepFace.represent(
            img_path=rgb_face,
            model_name="VGG-Face",
            enforce_detection=False
        )[0]
        embeddings["vgg"] = rep["embedding"]
    except Exception as e:
        print(f"[VGG-Face] {e}")

    # ArcFace - unstable
    try:
        rep = DeepFace.represent(
            img_path=rgb_face,
            model_name="ArcFace",
            enforce_detection=False
        )[0]
        embeddings["arcface"] = rep["embedding"]
    except Exception as e:
        print(f"[ArcFace] {e}")

    print(f"✅ extracted: {list(embeddings.keys())}")
    return embeddings