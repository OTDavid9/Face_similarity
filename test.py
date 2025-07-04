import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from deepface import DeepFace
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_deepface_cache():
    """Clear DeepFace model cache to force fresh download"""
    weights_dir = os.path.expanduser('~/.deepface/weights')
    if os.path.exists(weights_dir):
        for file in os.listdir(weights_dir):
            os.remove(os.path.join(weights_dir, file))
        logger.info("Cleared DeepFace model cache")

def load_and_preprocess_image(image_path):
    """Load image and convert to RGB"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def detect_and_crop_face(img, detector):
    """Detect face using MTCNN and crop it"""
    faces = detector.detect_faces(img)
    if not faces:
        raise ValueError("No faces detected in the image")
    
    # Get the largest face
    face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
    x, y, w, h = face['box']
    
    # Ensure coordinates are within image bounds
    x, y = max(0, x), max(0, y)
    face_img = img[y:y+h, x:x+w]
    
    return cv2.resize(face_img, (224, 224))  # VGG-Face expects 224x224

def get_vggface_embedding(face_img):
    """Extract VGG-Face embedding with robust initialization"""
    try:
        # First verify the model can be loaded
        DeepFace.build_model("VGG-Face")
        
        # Then get representation
        result = DeepFace.represent(
            img_path=face_img,
            model_name="VGG-Face",
            enforce_detection=False,
            detector_backend="mtcnn"
        )
        return result[0]["embedding"]
    except Exception as e:
        logger.error(f"VGG-Face embedding failed: {str(e)}")
        return None

def test_vggface_embedding(image_path):
    """Test pipeline for VGG-Face embeddings"""
    try:
        # Clear cache to prevent corrupted model files
        clear_deepface_cache()
        
        # Initialize components
        detector = MTCNN()
        
        # Load and preprocess image
        img = load_and_preprocess_image(image_path)
        
        # Detect and crop face
        face_img = detect_and_crop_face(img, detector)
        
        # Save the cropped face for debugging
        cv2.imwrite("debug_face.jpg", cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
        logger.info("Saved cropped face to debug_face.jpg")
        
        # Get VGG-Face embedding
        embedding = get_vggface_embedding(face_img)
        
        if embedding is not None:
            logger.info("VGG-Face embedding successful!")
            logger.info(f"Embedding length: {len(embedding)}")
            return embedding
        return None
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return None

if __name__ == "__main__":
    image_path = "unseen_images/image copy 2.png"
    test_vggface_embedding(image_path)