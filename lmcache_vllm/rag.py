import faiss
import numpy as np


class RAG:
    def __init__(self, embedding_model, use_faiss=False):
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_model.get_sentence_embedding_dimension()
        # Use inner product for cosine
        self.use_faiss = use_faiss
        if use_faiss:
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            self.embeddings = []
        self.context_keys = []
        self.contexts = {}

    def index(self, context_key, context):
        if context_key in self.context_keys:
            print(f"/// [RAG] context_key '{context_key}' already indexed, skipping.")
            return
        print("/// [RAG] indexing context: " + context_key)
        embeddings_np = self.embedding_model.encode(
            [context], normalize_embeddings=True
        )

        if self.use_faiss:
            self.faiss_index.add(embeddings_np.astype(np.float32))
        else:
            self.embeddings.append(embeddings_np[0])

        self.context_keys.append(context_key)
        self.contexts[context_key] = context

    def search(self, question, top_k=5):
        print("/// [RAG] searching for question: " + question)
        question_np = self.embedding_model.encode([question], normalize_embeddings=True)

        if self.use_faiss:
            _, ids = self.faiss_index.search(question_np.astype(np.float32), top_k)
            idx = ids[0][0]
        else:
            # Compute cosine similarity with all stored embeddings
            similarities = []
            question = question_np[0]
            for emb in self.embeddings:
                sim = np.dot(question, emb) / (
                    np.linalg.norm(question) * np.linalg.norm(emb) + 1e-8
                )
                similarities.append(sim)
            # Get top_k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            # Return the best match
            idx = top_indices[0]

        context_key = self.context_keys[idx]
        context = self.contexts[context_key]
        return context_key, context
