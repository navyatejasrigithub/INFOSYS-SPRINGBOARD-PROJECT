# src/vector_store/faiss_store.py

import faiss
import numpy as np
import pickle
import os


class FAISSStore:

    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.text_chunks = []
        self.metadata = []

    def add(self, embeddings, chunks, metadata):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        self.index.add(embeddings)
        self.text_chunks.extend(chunks)
        self.metadata.extend(metadata)

    def search(self, query_embedding, k=5):
        if self.index.ntotal == 0:
            return []

        k = min(k, self.index.ntotal)
        if k <= 0:
            return []

        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.text_chunks):
                continue

            distance = float(dist)
            similarity = 1.0 / (1.0 + distance)
            results.append({
                "id": int(idx),
                "text": self.text_chunks[idx],
                "metadata": self.metadata[idx],
                "distance": distance,
                "similarity": similarity
            })

        return results

    def save(self, path="embeddings/faiss_index"):
        os.makedirs(path, exist_ok=True)

        faiss.write_index(self.index, f"{path}/index.faiss")

        with open(f"{path}/meta.pkl", "wb") as f:
            pickle.dump((self.text_chunks, self.metadata), f)

    def load(self, path="embeddings/faiss_index"):
        self.index = faiss.read_index(f"{path}/index.faiss")

        with open(f"{path}/meta.pkl", "rb") as f:
            self.text_chunks, self.metadata = pickle.load(f)
