# api/routes.py
from flask import Blueprint, request, jsonify
from .embeddings import get_face_encoding
from .faiss_index import FaissIndex
import os
import numpy as np

routes = Blueprint('routes', __name__)

# Tạo index của FAISS
d = 128  # Kích thước vector khuôn mặt
faiss_index = FaissIndex(d)

# Thư mục lưu trữ thông tin người dùng
USER_DATA_DIR = 'user_data'
if not os.path.exists(USER_DATA_DIR):
    os.makedirs(USER_DATA_DIR)

@routes.route('/register', methods=['POST'])
def register():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image_path = os.path.join(USER_DATA_DIR, image_file.filename)
    image_file.save(image_path)

    face_encoding = get_face_encoding(image_path)
    if face_encoding is None:
        return jsonify({"error": "No face detected"}), 400

    user_id = os.path.splitext(image_file.filename)[0]

    # Lưu encoding vào FAISS
    faiss_index.add_embedding(face_encoding)

    # Lưu thông tin user
    np.save(os.path.join(USER_DATA_DIR, f'{user_id}.npy'), face_encoding)

    return jsonify({"message": "User registered successfully"}), 200

@routes.route('/verify', methods=['POST'])
def verify():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image_path = os.path.join(USER_DATA_DIR, image_file.filename)
    image_file.save(image_path)

    face_encoding = get_face_encoding(image_path)
    if face_encoding is None:
        return jsonify({"error": "No face detected"}), 400

    D, I = faiss_index.search_embedding(face_encoding)
    if D[0][0] < 0.6:
        user_id = os.path.splitext(image_file.filename)[0]
        return jsonify({"message": "User verified", "user_id": user_id}), 200
    else:
        return jsonify({"error": "Verification failed"}), 400
