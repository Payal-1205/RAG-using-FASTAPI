import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Embedding dimension
dimension = 384

# Create FAISS index
index = faiss.IndexFlatL2(dimension)

# Store chunks in memory
document_chunks = []


def create_embeddings(text_chunks):

    embeddings = embedding_model.encode(
        text_chunks,
        show_progress_bar=True
    )

    return np.array(embeddings).astype("float32")


def store_embeddings(text_chunks, embeddings):

    index.add(embeddings)

    document_chunks.extend(text_chunks)
    
    print(f"Added {len(text_chunks)} chunks to FAISS index.")
    print(f"Total chunks stored: {len(document_chunks)}")


def search_similar_chunks(question, k=3):

    # Handle empty database
    if len(document_chunks) == 0:
        return ["No documents uploaded yet."]

    # Create question embedding
    question_embedding = embedding_model.encode([question])

    # Search FAISS
    distances, indices = index.search(
        np.array(question_embedding).astype("float32"),
        k=k
    )

    retrieved_chunks = []

    for i in indices[0]:

        # FAISS can return -1
        if i != -1:
            retrieved_chunks.append(document_chunks[i])

    return retrieved_chunks