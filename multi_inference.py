import os
import cv2
import base64
import numpy as np
import pandas as pd
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from pinecone import Pinecone
from typing import Optional, List
from multi_helper import get_face_embedding  # <- this must return dict: {"facenet": [...], "vgg": [...], "arcface": [...]}
from verify_identity import verify_identity

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ================== CONFIG ==================
PINECONE_API_KEY = "pcsk_BnCpm_KSFsVygP2bKX8dJWoURGMXj6hSRpLc1maSqR3Q1ZxQiwZqWatyJCu4MwhkJwBNX"
TOP_K = 5

# Index setup
INDEX_CONFIG = {
    "facenet": {"index_name": "facenet-index", "dimension": 512},
    "vgg": {"index_name": "vgg-index", "dimension": 4096},
    "arcface": {"index_name": "arcface-index", "dimension": 512}
}

# Initialize MTCNN and FaceNet
facenet_embedder = FaceNet()
detector = MTCNN()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_map = {
    key: pc.Index(cfg["index_name"])
    for key, cfg in INDEX_CONFIG.items()
}

# ================== HELPERS ==================
def decode_base64_image(base64_str: str) -> Optional[np.ndarray]:
    try:
        base64_str = base64_str.strip().split(",")[-1]
        image_data = base64.b64decode(base64_str + '===')
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"❌ Decoding error: {e}")
        return None

def capture_live_image() -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot access webcam.")
    print("📸 Press 'c' to capture, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Live Capture", frame)
        key = cv2.waitKey(1)
        if key == ord('c'):
            print("✅ Image captured.")
            cap.release()
            cv2.destroyAllWindows()
            return frame
        elif key == ord('q'):
            print("❌ Capture aborted.")
            cap.release()
            cv2.destroyAllWindows()
            return None

def search_embedding_in_pinecone(embedding_dict: dict, top_k: int = TOP_K):
    all_results = {}

    for name, emb in embedding_dict.items():
        try:
            idx = index_map[name]
            response = idx.query(
                vector=emb,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
            matches = response.get("matches", [])
            all_results[name] = matches

            print(f"\n🔍 Top {top_k} matches in {name}-index:")
            for i, match in enumerate(matches, 1):
                print(f"{i}. ID: {match['id']} | Score: {match['score']:.4f} | BVN: {match['metadata'].get('bvn')}")

        except Exception as e:
            print(f"❌ {name}-index search failed: {e}")
            all_results[name] = []

    return all_results

# ================== MAIN ==================
def search_face(input_type=None, base64_input="", file_path=None):
    if input_type == "live":
        img = capture_live_image()
        if img is None:
            return []
        cv2.imwrite("input_image.png", img)

    elif input_type == "base64":
        if not base64_input:
            print("❌ Base64 input not provided.")
            return []
        img = decode_base64_image(base64_input)
        if img is None:
            print("❌ Failed to decode base64 image.")
            return []
        cv2.imwrite("input_image.png", img)

    elif input_type == "file":
        if not file_path:
            print("❌ File path not provided.")
            return []
        img = cv2.imread(file_path)
        if img is None:
            print("❌ Failed to load image from file.")
            return []
        cv2.imwrite("input_image.png", img)

    else:
        print("❌ Invalid input_type. Use 'live', 'base64', or 'file'.")
        return []

    embeddings = get_face_embedding(img)  # must return dict with keys 'facenet', 'vgg', 'arcface'
    print("✅ extracted:", list(embeddings.keys()))

    if not embeddings:
        print("❌ Could not extract embeddings.")
        return []

    return search_embedding_in_pinecone(embeddings)


if __name__ == "__main__":
    # Example static file search
    results = search_face(input_type="file", file_path="unseen_images/image copy 8.png")
    print(results)
    # verify_identity(results)
