# retina_arcface_helpers.py
from insightface.app import FaceAnalysis
import cv2

insight_app = FaceAnalysis(name="antelopev2", providers=['CPUExecutionProvider'])
insight_app.prepare(ctx_id=0)

def retina_detect(img):
    faces = insight_app.get(img)
    if not faces:
        return None
    x1, y1, x2, y2 = faces[0].bbox.astype(int)
    face = img[y1:y2, x1:x2]
    return cv2.resize(face, (112, 112))

def arcface_embed(face_img):
    faces = insight_app.get(face_img)
    return faces[0].embedding if faces else None
