
from pathlib import Path

from app.rag.pdf_loader import load_documents
from app.rag.chunker import generate_chunk
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parents[2]
PERSIST_DIRECTORY = BASE_DIR / "db" / "chroma_db"


def create_embeddings(file_path: str):
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    documents=load_documents(file_path)
    chunks = generate_chunk(documents)

    if not chunks:
        raise ValueError("No chunks generated from the PDF files.")

    vector_db = Chroma(
        collection_name="research_papers",
        embedding_function=embedding_model,
        persist_directory=str(PERSIST_DIRECTORY)
    )

    vector_db.add_documents(chunks)

    print(f"Chunks stored: {len(chunks)}")
    print(f"Database location: {PERSIST_DIRECTORY}")

    return vector_db

if __name__ == "__main__":
    create_embeddings(
        "C:/Users/vaish/Documents/ResearchAgent/backend/data/paper"
    )
