from flask import Blueprint, request, jsonify
from .embeddings import get_face_encoding
from .faiss_index import FaissIndex
import os
import json
import speech_recognition as sr
import re
from datetime import datetime

routes = Blueprint("routes", __name__)

d = 128
faiss_index = FaissIndex(d)

USER_DATA_DIR = "user_data"
LOGIN_HISTORY_DIR = "login_history"
if not os.path.exists(USER_DATA_DIR):
    os.makedirs(USER_DATA_DIR)

if not os.path.exists(LOGIN_HISTORY_DIR):
    os.makedirs(LOGIN_HISTORY_DIR)

INDEX_MAPPING_PATH = "index_mapping.json"
FAISS_INDEX_PATH = "faiss_index.bin"

if os.path.exists(INDEX_MAPPING_PATH):
    with open(INDEX_MAPPING_PATH, "r") as file:
        index_mapping = json.load(file)
else:
    index_mapping = {}

number_map = {
    "một": "1",
    "hai": "2",
    "ba": "3",
    "bốn": "4",
    "năm": "5",
    "sáu": "6",
    "bảy": "7",
    "tám": "8",
    "chín": "9",
    "mười": "10",
    "không": "0",
}


def merge_numbers(text):
    return re.sub(r"(\d)\s+(\d)", r"\1\2", text)


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

    index = faiss_index.add_embedding(face_encoding)

    index_mapping[str(index)] = f"{user_id}.jpg"

    with open(INDEX_MAPPING_PATH, "w") as file:
        json.dump(index_mapping, file)

    faiss_index.save_index(FAISS_INDEX_PATH)

    return jsonify({"message": "User registered successfully"}), 200


@routes.route("/verify", methods=["POST"])
def verify():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image_path = os.path.join(LOGIN_HISTORY_DIR, image_file.filename)
    image_file.save(image_path)

    face_encoding = get_face_encoding(image_path)

    if face_encoding is None:
        os.remove(image_path)
        return jsonify({"error": "No face detected"}), 400

    D, I = faiss_index.search_embedding(face_encoding)
    print(f"Khoảng cách giữa 2 vector: {D[0][0]}")
    if D[0][0] < 0.19:
        closest_index = I[0][0]
        original_filename = index_mapping.get(str(closest_index), None)

        if original_filename:
            # Đổi tên file lưu với tên file + ngày giờ
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_image_path = os.path.join(
                LOGIN_HISTORY_DIR,
                f"{os.path.splitext(image_file.filename)[0]}_{timestamp}.jpg",
            )
            os.rename(image_path, new_image_path)

            return (
                jsonify({"message": "User verified", "filename": original_filename}),
                200,
            )
        else:
            os.remove(image_path)
            return jsonify({"error": "Verification failed - original_filename"}), 400
    else:
        os.remove(image_path)
        return jsonify({"error": "Verification failed - unverify"}), 400


# router nhan dien giong noi
@routes.route("/stt", methods=["POST"])
def transcribe():
    audio_file = request.files["file"]
    audio_path = os.path.join("temp.wav")
    audio_file.save(audio_path)

    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data, language="vi-VN")
    except sr.UnknownValueError:
        return (
            jsonify({"error": "Google Speech Recognition could not understand audio"}),
            400,
        )
    except sr.RequestError as e:
        return (
            jsonify(
                {
                    "error": f"Could not request results from Google Speech Recognition service; {e}"
                }
            ),
            500,
        )

    words = text.split()
    converted_text = " ".join([number_map.get(word, word) for word in words])

    final_text = merge_numbers(converted_text)
    while final_text != merge_numbers(final_text):
        final_text = merge_numbers(final_text)

    os.remove(audio_path)

    return jsonify({"text": final_text}), 200
