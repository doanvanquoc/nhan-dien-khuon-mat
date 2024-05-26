# api/faiss_index.py
import faiss
import numpy as np

class FaissIndex:
    def __init__(self, d):
        self.index = faiss.IndexFlatL2(d)
    
    def add_embedding(self, embedding):
        self.index.add(np.array([embedding]))
    
    def search_embedding(self, embedding, k=1):
        return self.index.search(np.array([embedding]), k)
