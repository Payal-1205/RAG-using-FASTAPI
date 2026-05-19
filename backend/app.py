import os

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from backend.rag import load_pdf, ask_question

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"

print("APP STARTING...")
# Create uploads directory if it does not exist
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def home():
    return {
        "message": "Welcome to the RAG API. Backend is running successfully."
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        chunks = load_pdf(file_path)

        return {
            "message": f"File '{file.filename}' uploaded and processed successfully.",
            "chunks_created": chunks
        }

    except Exception as e:
        return {
            "error": str(e)
        }


@app.post("/ask")
async def ask(data: dict):

    try:
        question = data["question"]

        answer = ask_question(question)

        return {
            "answer": answer
        }

    except Exception as e:
        return {
            "error": str(e)
        }