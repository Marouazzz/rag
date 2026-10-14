import os
#from langchain_community.document_loaders import DirectoryLoader, TextLoader deprecated !!
from pathlib import Path
import ollama
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma # cuz i can host it locally
from dotenv import load_dotenv #if we use any api key

from langchain_ollama import OllamaEmbeddings

# Load environment variables
load_dotenv() 

import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def load_documents(docs_path="docs"):
    """Load all .txt files from the docs directory."""
    print(f"Loading documents from {docs_path}...")

    if not os.path.isdir(docs_path):
        raise FileNotFoundError(
            f"Directory '{docs_path}' does not exist."
        )

    documents = []

    for file_path in sorted(Path(docs_path).glob("*.txt")):
        try:
            content = file_path.read_text(encoding="utf-8")

            if not content.strip():
                print(f"Skipping empty file: {file_path}")
                continue

            documents.append(
                Document(
                    page_content=content,
                    metadata={"source": str(file_path)}
                )
            )

        except (OSError, UnicodeDecodeError) as error:
            print(f"Could not load {file_path}: {error}")

    if not documents:
        raise FileNotFoundError(
            f"No readable, non-empty .txt files found in '{docs_path}'."
        )

    print(f"Loaded {len(documents)} documents.")
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """Split documents into chunks."""
    print("Splitting documents into chunks...")

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")
    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create a local Chroma vector store using Ollama embeddings."""
    if not chunks:
        raise ValueError("No chunks to ingest.")

    print("Creating local embeddings and storing in ChromaDB...")

    embedding_model = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name="langchain",
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"Vector store saved to: {persist_directory}")
    print(f"Stored chunks: {vectorstore._collection.count()}")

    return vectorstore


def main():
    print("Starting RAG ingestion pipeline...")

    docs_path = "docs"
    persistent_directory = "db/chroma_db"

    embedding_model = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # Reuse an existing populated database.
    # If the database exists but is empty, ingest documents.
    if os.path.isdir(persistent_directory):
        existing_store = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model,
            collection_name="langchain"
        )

        count = existing_store._collection.count()

        if count > 0:
            print(f"Existing vector store contains {count} chunks.")
            print("Skipping ingestion.")
            return existing_store

        print("Existing vector store is empty. Starting ingestion...")

    # 1. Load source documents
    documents = load_documents(docs_path)

    # 2. Split documents
    chunks = split_documents(documents)

    # 3. Generate embeddings and store the chunks
    vectorstore = create_vector_store(
        chunks,
        persistent_directory
    )

    print("\nIngestion complete!")
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks stored: {vectorstore._collection.count()}")

    return vectorstore


if __name__ == "__main__":
    main()
