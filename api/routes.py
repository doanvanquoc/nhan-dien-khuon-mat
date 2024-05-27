from flask import Blueprint, request, jsonify
from .embeddings import get_face_encoding
from .faiss_index import FaissIndex
import os
import numpy as np
import json

routes = Blueprint("routes", __name__)

# Tạo index của FAISS
d = 128  # Kích thước vector khuôn mặt
faiss_index = FaissIndex(d)

# Thư mục lưu trữ thông tin người dùng
USER_DATA_DIR = "user_data"
if not os.path.exists(USER_DATA_DIR):
    os.makedirs(USER_DATA_DIR)

# Tệp tin lưu trữ ánh xạ giữa chỉ số FAISS và tên tệp
INDEX_MAPPING_PATH = "index_mapping.json"
FAISS_INDEX_PATH = "faiss_index.bin"

# Tải ánh xạ chỉ số FAISS và tên tệp nếu tệp tồn tại
if os.path.exists(INDEX_MAPPING_PATH):
    with open(INDEX_MAPPING_PATH, "r") as file:
        index_mapping = json.load(file)
else:
    index_mapping = {}

# Tải FAISS index nếu tệp tồn tại
if os.path.exists(FAISS_INDEX_PATH):
    faiss_index.load_index(FAISS_INDEX_PATH)


@routes.route("/register", methods=["POST"])
def register():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    user_id = os.path.splitext(image_file.filename)[0]
    image_path = os.path.join(USER_DATA_DIR, f"{user_id}.jpg")
    image_file.save(image_path)

    face_encoding = get_face_encoding(image_path)
    if face_encoding is None:
        return jsonify({"error": "No face detected"}), 400

    # Lưu encoding vào FAISS và lấy chỉ số
    index = faiss_index.add_embedding(face_encoding)

    # Lưu ánh xạ giữa chỉ số FAISS và tên tệp
    index_mapping[str(index)] = f"{user_id}.jpg"

    # Ghi ánh xạ vào tệp tin
    with open(INDEX_MAPPING_PATH, "w") as file:
        json.dump(index_mapping, file)

    # Lưu FAISS index vào tệp
    faiss_index.save_index(FAISS_INDEX_PATH)

    print(f"User registered: {user_id} with index {index}")

    return jsonify({"message": "User registered successfully"}), 200


@routes.route("/verify", methods=["POST"])
def verify():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]

    # Lưu tạm ảnh vào thư mục temp_images
    temp_dir = "temp_images"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    temp_image_path = os.path.join(temp_dir, image_file.filename)
    image_file.save(temp_image_path)

    # Lấy mã hóa khuôn mặt từ ảnh
    face_encoding = get_face_encoding(temp_image_path)

    # Xóa ảnh tạm sau khi đã lấy mã hóa
    os.remove(temp_image_path)

    if face_encoding is None:
        return jsonify({"error": "No face detected"}), 400

    D, I = faiss_index.search_embedding(face_encoding)
    print(f"Distances: {D}")
    print(f"Indices: {I}")

    if D[0][0] < 0.45:  # Điều chỉnh ngưỡng xác minh nếu cần
        # Lấy tên tệp gốc từ ánh xạ FAISS index
        closest_index = I[0][0]
        print(f"Closest index: {closest_index}")

        original_filename = index_mapping.get(str(closest_index), None)
        print(f"Original filename: {original_filename}")

        if original_filename:
            return (
                jsonify({"message": "User verified", "filename": original_filename}),
                200,
            )
        else:
            return jsonify({"error": "Verification failed - original_filename"}), 400
    else:
        return jsonify({"error": "Verification failed - unverify"}), 400
