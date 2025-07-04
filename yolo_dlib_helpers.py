import cv2
import face_recognition

def yolo_detect(img):
    # For demo, fallback to dlib's default detector
    # Replace this with actual YOLOv8-face if integrated
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    if not boxes:
        return None
    top, right, bottom, left = boxes[0]
    return cv2.resize(img[top:bottom, left:right], (150, 150))

def dlib_embed(face_img):
    rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    return encodings[0] if encodings else None