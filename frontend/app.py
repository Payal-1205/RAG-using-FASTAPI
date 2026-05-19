import streamlit as st

import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.title("RAG with FastAPI and Streamlit")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:

    files = {
    "file": (
        uploaded_file.name,
        uploaded_file.getvalue(),
        uploaded_file.type
    )
}

    response = requests.post(f"{BACKEND_URL}/upload", files=files)

    st.success(response.json()["message"])

question = st.text_input("Ask a question :")

if st.button("Submit"):

    response = requests.post(f"{BACKEND_URL}/ask", json={"question": question})

    answer = response.json()["answer"]

    st.write("Answer:")
    st.write(answer)