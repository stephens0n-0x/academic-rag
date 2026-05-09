from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from app.core.config import config


def load_embedding_model() -> HuggingFaceEmbeddings:
    """Initialize and return the sentence embedding model."""
    return HuggingFaceEmbeddings(
        model_name=config["embeddings"]["model_name"],
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Return a text splitter configured for academic paper chunking.
    Splits on paragraph boundaries before falling back to sentences and words.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=config["embeddings"]["chunk_size"],
        chunk_overlap=config["embeddings"]["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def get_vectorstore(embedding_model: HuggingFaceEmbeddings) -> Chroma:
    """Load an existing ChromaDB collection or create one if it does not exist."""
    return Chroma(
        collection_name=config["vectorstore"]["collection_name"],
        embedding_function=embedding_model,
        persist_directory=config["vectorstore"]["persist_directory"],
    )


def add_documents_to_vectorstore(
    documents: List[Document],
    vectorstore: Chroma,
) -> int:
    """
    Split documents into chunks and add them to the vector store.
    Returns the number of chunks indexed.
    """
    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)
    vectorstore.add_documents(chunks)
    return len(chunks)