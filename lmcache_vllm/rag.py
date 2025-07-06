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

    def index(self, context_key, context, embeddings_np):
        if context_key in self.context_keys:
            print(f"/// [RAG] context_key '{context_key}' already indexed, skipping.")
            return
        print("/// [RAG] indexing context: " + context_key)
        # embeddings_np = self.encode(context)
        # embeddings_np = self.embedding_model.encode(
        #     [context], normalize_embeddings=True
        # )

        if self.use_faiss:
            self.faiss_index.add(embeddings_np.astype(np.float32))
        else:
            self.embeddings.append(embeddings_np[0])

        self.context_keys.append(context_key)
        self.contexts[context_key] = context

    def search(self, question, question_np, top_k=5):
        print("/// [RAG] searching for question: " + question)
        # question_np = self.encode(question)
        # question_np = self.embedding_model.encode([question], normalize_embeddings=True)

        if self.use_faiss:
            _, ids = self.faiss_index.search(question_np.astype(np.float32), top_k)
            idx = ids[0][0]
        else:
            # Compute cosine similarity with all stored embeddings
            similarities = []
            question = question_np[0]
            for emb in self.embeddings:
                # For each embedding, calculate the cosine similarity between the question and the embedding.
                # Cosine similarity is computed as the dot product of the two vectors divided by the product of their norms.
                # Sim is a value between -1 and 1, where 1 means the vectors are identical in direction.
                sim = np.dot(question, emb) / (
                    np.linalg.norm(question) * np.linalg.norm(emb)
                    + 1e-8  # prevent division by zero
                )
                similarities.append(sim)
            # Get top_k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            # Return the best match
            idx = top_indices[0]

        context_key = self.context_keys[idx]
        context = self.contexts[context_key]
        return context_key, context

    def encode(self, text):
        return self.embedding_model.encode([text], normalize_embeddings=True)
