import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from scripts.utils.parser import chunk_text

def embed_resume(text: str, user_id: str, role: str):
    path = f"vectorstore/{role}/{user_id}"
    os.makedirs(path, exist_ok=True)

    chunks = chunk_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_texts(chunks, embeddings)
    vectordb.save_local(path)
