import torch
import faiss
import numpy as np


class RAG:

    def __init__(self, model, tokenizer, device, embedding_dim=768):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.embedding_dim = embedding_dim
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)  # create faiss index
        self.context_keys = []
        self.contexts = {}

    def index(self, context_key, context):
        print("/// [RAG] indexing context: " + context_key)
        embeddings = self._get_llm_embeddings(context)
        print("/// -> [index] Embedding dimension: ", len(embeddings))
        embeddings_np = embeddings.cpu().numpy().astype(np.float32)
        self.faiss_index.add(embeddings_np)
        self.context_keys.append(context_key)
        self.contexts[context_key] = context

    def search(self, question, top_k=5):
        print("/// [RAG] searching for question: " + question)
        question_embedding = self._get_llm_embeddings(question)
        print("/// -> [search] embedding dimension: ", len(question_embedding))
        question_np = question_embedding.cpu().numpy().astype(np.float32)
        distances, ids = self.faiss_index.search(question_np, top_k)
        context_key = self.context_keys[ids[0][0]]
        context = self.contexts[context_key]
        return context_key, context

    def _get_llm_embeddings(self, text):
        # Tokenize the text chunk
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )

        # Generate embeddings using the model
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract the last hidden state (token-level embeddings)
        last_hidden_state = outputs.last_hidden_state

        # Pool the token embeddings to get a single vector (mean pooling)
        embeddings = last_hidden_state.mean(dim=1).squeeze()

        return embeddings

    def _get_llm_embeddings_2(self, text):
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # Forward pass through the model to get hidden states
            outputs = self.model(**inputs, output_hidden_states=True)
            # Extract the last hidden state (embeddings)
            hidden_states = outputs.hidden_states
            last_hidden_state = hidden_states[-1]
            # Mean pooling for sentence-level embedding
            embeddings = last_hidden_state.mean(dim=1).squeeze()
        return embeddings
