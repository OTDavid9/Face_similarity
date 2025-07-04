import cv2
from concurrent.futures import ThreadPoolExecutor
from helper import get_face_embedding
from verify_identity import verify_identity
from mtcnn_facenet_helpers import mtcnn_detect, facenet_embed
from retina_arcface_helpers import retina_detect, arcface_embed
from yolo_dlib_helpers import yolo_detect, dlib_embed
from pinecone import Pinecone

# Pinecone Config
PINECONE_API_KEY = "your-key"
INDEX_NAME = "face-index"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

TOP_K = 5

def search_in_pinecone(embedding, model_name):
    try:
        response = index.query(
            vector=embedding.tolist(),
            top_k=TOP_K,
            include_metadata=True,
            include_values=False
        )
        matches = response.get("matches", [])
        print(f"\n🔍 Top matches for {model_name}:")
        for i, match in enumerate(matches, 1):
            print(f"{i}. ID: {match['id']} | Score: {match['score']:.4f} | BVN: {match['metadata'].get('bvn')}")
        return (model_name, matches)
    except Exception as e:
        print(f"❌ {model_name} search failed: {e}")
        return (model_name, [])

def run_pipeline(img, detect_fn, embed_fn, model_name):
    embedding = get_face_embedding(img, detect_fn, embed_fn)
    if embedding is None:
        return (model_name, [])
    return search_in_pinecone(embedding, model_name)

if __name__ == "__main__":
    image_path = "unseen_images/image.png"
    img = cv2.imread(image_path)

    if img is None:
        print("❌ Image not found.")
        exit()

    # Define all model pipelines
    pipelines = [
        ("MTCNN+FaceNet", mtcnn_detect, facenet_embed),
        ("RetinaFace+ArcFace", retina_detect, arcface_embed),
        ("YOLOv8+Dlib", yolo_detect, dlib_embed),
    ]

    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(run_pipeline, img, detect, embed, name)
            for name, detect, embed in pipelines
        ]

        for future in futures:
            model_name, matches = future.result()
            results[model_name] = matches

    # Final decision or fusion logic
    print("\n✅ Final Verification (First model as reference):")
    verify_identity(results["MTCNN+FaceNet"])  # Or average top scores from all
