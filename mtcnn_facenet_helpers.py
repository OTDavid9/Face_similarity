from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from helper import get_face_embedding

mtcnn_detector = MTCNN()
facenet_embedder = FaceNet()

def mtcnn_detect(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = mtcnn_detector.detect_faces(img_rgb)
    if not faces:
        return None
    x, y, w, h = faces[0]['box']
    x, y = max(0, x), max(0, y)
    face = img[y:y+h, x:x+w]
    return cv2.resize(face, (160, 160))

def facenet_embed(face_img):
    rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    return facenet_embedder.embeddings([rgb])[0]

# Usage:
embedding = get_face_embedding(img, detect_fn=mtcnn_detect, embed_fn=facenet_embed)
