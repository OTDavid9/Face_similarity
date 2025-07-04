import os
import cv2
import base64
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from pinecone import Pinecone
from typing import Optional, List, Callable
import pandas as pd
from helper import get_face_embedding
from verify_identity import verify_identity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ ['TF_ENABLE_ONEDNN_OPTS']= '0'

# ========== CONFIG ==========
PINECONE_API_KEY = "pcsk_BnCpm_KSFsVygP2bKX8dJWoURGMXj6hSRpLc1maSqR3Q1ZxQiwZqWatyJCu4MwhkJwBNX"
INDEX_NAME = "facenet-index"
TOP_K = 5

# Initialize MTCNN and FaceNet
detector = MTCNN()
embedder = FaceNet()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ========== HELPERS ==========

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

def search_embedding_in_pinecone(embedding: np.ndarray, top_k: int = TOP_K):
    try:
        response = index.query(
            vector=embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )

        matches = response.get("matches", [])
        
        print(matches)
        print(f"\n🔍 Top {top_k} matches:")
        for i, match in enumerate(matches, 1):
            print(f"{i}. ID: {match['id']} | Score: {match['score']:.4f} | BVN: {match['metadata'].get('bvn')}")

        return matches
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return []

# ========== MAIN ENTRYPOINT ==========

def search_face(input_type=None, base64_input="base64", file_path=None):
    """
    input_type: "live", "base64", or "file"
    base64_input: required for "base64"
    file_path: required for "file"
    """
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

    embedding = get_face_embedding(img)

    if embedding is None:
        print("❌ Could not extract embedding.")
        return []

    return search_embedding_in_pinecone(embedding)




if __name__ == "__main__":
    # Option 1: Live capture
    # search_face(input_type="live")
    # list_of_similar_embeddings = search_face(input_type="live")
    # verify_identity(list_of_similar_embeddings)

    # # # Option 2: Base64 string
    # df = pd.read_csv("output_with_faces_v4.csv")
    # base64_str = df[df['BVN'] == 22374302379]['ImageBase64'].iloc[0]
    # search_face(input_type="base64", base64_input=base64_str)
    # list_of_similar_embeddings = search_face(input_type="base64", base64_input=base64_str)
    # verify_identity(list_of_similar_embeddings)

    # # # Option 3: Static file path
    # list_of_similar_embeddings = search_face(input_type="file", file_path="unseen_images/image copy 2.png")
    # verify_identity(list_of_similar_embeddings)




    