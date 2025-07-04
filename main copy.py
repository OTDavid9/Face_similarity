import cv2
import base64
import numpy as np
import pandas as pd
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

# Initialize detector and embedder
detector = MTCNN()
embedder = FaceNet()

def get_face_embedding(img):
    """Detect face and return 512-dim embedding."""
    try:
        faces = detector.detect_faces(img)
        if not faces:
            print("No face detected.")
            return None

        # Extract bounding box
        x, y, w, h = faces[0]['box']
        x, y = max(0, x), max(0, y)
        face = img[y:y+h, x:x+w]

        # Resize for FaceNet
        face = cv2.resize(face, (160, 160))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Get embedding
        embedding = embedder.embeddings([face])[0]
        return embedding
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None

def capture_live_image():
    """Capture an image from webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot access webcam.")
    print("📸 Press 's' to capture, 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Live Capture", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            print("✅ Image captured.")
            cap.release()
            cv2.destroyAllWindows()
            return frame
        elif key == ord('q'):
            print("❌ Capture aborted.")
            cap.release()
            cv2.destroyAllWindows()
            return None

def load_embeddings(csv_path):
    """Load embeddings and BVNs from CSV."""
    df = pd.read_csv(csv_path)
    df = df[df["FaceEmbedding"].notnull()].copy()
    df["FaceEmbedding"] = df["FaceEmbedding"].apply(eval)
    return df

def find_closest_match(live_embedding, df):
    """Compute cosine similarity and return closest BVN."""
    embeddings = np.vstack(df["FaceEmbedding"].values)
    similarities = cosine_similarity([live_embedding], embeddings)[0]
    best_idx = np.argmax(similarities)
    return df.iloc[best_idx]["BVN"], similarities[best_idx]

# ---- MAIN PIPELINE ----
if __name__ == "__main__":
    live_img = capture_live_image()
    if live_img is None:
        exit()

    live_embedding = get_face_embedding(live_img)
    if live_embedding is None:
        print("❌ Could not extract embedding from live image.")
        exit()

    df_faces = load_embeddings("output_with_faces.csv")
    bvn, similarity = find_closest_match(live_embedding, df_faces)

    print(f"\n🎯 Closest match BVN: {bvn} (Similarity: {similarity:.4f})")
