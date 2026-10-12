import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

def is_valid_image(image_path):
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except Exception:
        return False

def check_resolution(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False
    height, width, _ = img.shape
    return height >= 128 and width >= 128

def detect_face(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img_rgb)
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            if width >= 48 and height >= 48:
                return x, y, width, height
    return None

def detect_blur(face_region):
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance > 40

def check_exposure(face_region):
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return 40 <= mean_brightness <= 220

def filter_image(image_path):
    if not is_valid_image(image_path) or not check_resolution(image_path):
        return False
    image = cv2.imread(image_path)
    if image is None:
        return False
    face_box = detect_face(image)
    if not face_box:
        return False
    x, y, w, h = face_box
    face_region = image[y:y+h, x:x+w]
    if not detect_blur(face_region) or not check_exposure(face_region):
        return False
    return True
