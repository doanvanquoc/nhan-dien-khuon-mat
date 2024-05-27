import faiss
import numpy as np


class FaissIndex:
    def __init__(self, d):
        self.index = faiss.IndexFlatL2(d)
        self.d = d  # Save the dimension

    def add_embedding(self, embedding):
        embedding = np.array([embedding]).astype("float32")
        self.index.add(embedding)
        return self.index.ntotal - 1  # Trả về chỉ số của embedding vừa thêm

    def search_embedding(self, embedding, k=1):
        embedding = np.array([embedding]).astype("float32")
        D, I = self.index.search(embedding, k)
        return D, I

    def save_index(self, file_path):
        faiss.write_index(self.index, file_path)

    def load_index(self, file_path):
        self.index = faiss.read_index(file_path)
