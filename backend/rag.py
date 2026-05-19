import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from groq import Groq

from backend.vector_store import (
    create_embeddings,
    store_embeddings,
    search_similar_chunks
)

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

print("RAG LOADED...")

def load_pdf(file_path):

    # Load PDF
    loader = PyPDFLoader(file_path)

    docs = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(docs)

    text_chunks = []

    for chunk in chunks:
        text_chunks.append(chunk.page_content)

    # Create embeddings
    embeddings = create_embeddings(text_chunks)

    # Store in FAISS
    store_embeddings(text_chunks, embeddings)

    return len(text_chunks)


def ask_question(question):

    # Retrieve relevant chunks
    retrieved_chunks = search_similar_chunks(question)

    # Combine chunks into context
    context = "\n".join(retrieved_chunks)

    # Prompt
    prompt = f"""
    You are a helpful AI assistant.

    Answer the user's question ONLY using the provided context.

    If the answer is not available in the context, say:
    "I could not find the answer in the uploaded document."

    Context:
    {context}

    User Question:
    {question}

    Answer:
    """

    # LLM Call
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content