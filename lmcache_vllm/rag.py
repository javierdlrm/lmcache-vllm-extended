import torch
import faiss
import numpy as np


class RAG:

    def __init__(self, model, tokenizer, device, embedding_dim=768):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.embedding_dim = embedding_dim
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)  # new faiss index
        self.context_keys = []
        self.contexts = {}

    def index(self, context_key, context):
        embeddings = self._get_llm_embeddings(context)
        print("/// -> Embedding dimension: ", len(embeddings))
        embeddings_np = embeddings.cpu().numpy().astype(np.float32)
        self.faiss_index.add(embeddings_np)
        self.context_keys.append(context_key)
        self.contexts[context_key] = context

    def search(self, question, top_k=5):
        question_embedding = self._get_llm_embeddings(question)
        question_np = question_embedding.cpu().numpy().astype(np.float32)
        distances, ids = self.faiss_index.search(question_np, top_k)
        context_key = self.context_keys[ids[0][0]]
        context = self.contexts[context_key]
        return context_key, context

    def _get_llm_embeddings(self, text):
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # Forward pass through the model to get hidden states
            outputs = self.model(**inputs, output_hidden_states=True)
            # Extract the last hidden state (embeddings)
            hidden_states = outputs.hidden_states
            last_hidden_state = hidden_states[-1]
            # Mean pooling for sentence-level embedding
            embeddings = last_hidden_state.mean(dim=1)
        return embeddings
