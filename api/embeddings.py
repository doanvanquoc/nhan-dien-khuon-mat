# api/embeddings.py
import face_recognition
import numpy as np

def get_face_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    
    if len(face_encodings) == 0:
        return None
    
    return face_encodings[0]
