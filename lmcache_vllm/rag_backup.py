import faiss
import numpy as np


class RAG:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_model.get_sentence_embedding_dimension()
        # Use inner product for cosine
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
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
        self.faiss_index.add(embeddings_np.astype(np.float32))
        self.context_keys.append(context_key)
        self.contexts[context_key] = context

    def search(self, question, top_k=5):
        print("/// [RAG] searching for question: " + question)
        question_np = self.embedding_model.encode([question], normalize_embeddings=True)
        distances, ids = self.faiss_index.search(question_np.astype(np.float32), top_k)
        context_key = self.context_keys[ids[0][0]]
        context = self.contexts[context_key]
        return context_key, context
