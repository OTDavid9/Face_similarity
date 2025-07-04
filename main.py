import os
import cv2
import base64
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from pinecone import Pinecone
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ========== CONFIG ==========


# Load environment variables from .env file
load_dotenv()

# ========== CONFIG ==========
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")
TOP_K = int(os.getenv("TOP_K", 3)) 

app = FastAPI(title="Face Search API",
              description="API for searching faces using FaceNet embeddings and Pinecone vector database")

# Initialize components
detector = MTCNN()
embedder = FaceNet()
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
        raise HTTPException(status_code=400, detail=f"Decoding error: {e}")

def get_face_embedding(img: np.ndarray) -> Optional[np.ndarray]:
    try:
        faces = detector.detect_faces(img)
        if not faces:
            return None

        main_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
        x, y, w, h = main_face['box']

        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        face = np.expand_dims(face, axis=0)

        embedding = embedder.embeddings(face)
        return embedding[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face embedding extraction failed: {e}")

def convert_matches_to_dict(matches: List[Any]) -> List[Dict[str, Any]]:
    """Convert Pinecone ScoredVector objects to serializable dictionaries"""
    serializable_matches = []
    for match in matches:
        serializable_match = {
            "id": match.id,
            "score": match.score,
            "metadata": match.metadata if hasattr(match, 'metadata') else {}
        }
        serializable_matches.append(serializable_match)
    return serializable_matches

def search_embedding_in_pinecone(embedding: np.ndarray, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    try:
        response = index.query(
            vector=embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
        # Convert matches to serializable format
        return convert_matches_to_dict(response.matches)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone search failed: {e}")

def verify_identity(matches: List[Dict[str, Any]], threshold: float = 0.7) -> Dict[str, Any]:
    if not matches:
        return {"verified": False, "best_match": None, "message": "No matches found"}
    
    best_match = matches[0]
    if best_match['score'] >= threshold:
        return {
            "verified": True,
            "best_match": best_match,
            "message": "Similarity Threshold met for this user"
        }
    else:
        return {
            "verified": False,
            "best_match": best_match,
            "message": "No match meets the similarity threshold"
        }

# ========== API ENDPOINTS ==========

@app.post("/search/base64/")
async def search_face_base64(base64_str: str = Form(..., description="Base64 encoded image string")):
    try:
        img = decode_base64_image(base64_str)
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode base64 image")
        
        embedding = get_face_embedding(img)
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        matches = search_embedding_in_pinecone(embedding)
        verification = verify_identity(matches)
        
        return {
            "status": "success",
            "matches": matches,
            "verification": verification
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/file/")
async def search_face_file(file: UploadFile = File(..., description="Image file to search")):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to read image file")
        
        embedding = get_face_embedding(img)
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        matches = search_embedding_in_pinecone(embedding)
        verification = verify_identity(matches)
        
        return {
            "status": "success",
            "matches": matches,
            "verification": verification
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)