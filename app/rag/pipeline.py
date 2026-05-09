from typing import List, Tuple
from langchain_core.documents import Document
from langchain_huggingface import HuggingFacePipeline

from app.rag.embeddings import (
    load_embedding_model,
    get_vectorstore,
    add_documents_to_vectorstore,
)
from app.rag.retriever import build_rag_chain, retrieve_with_sources
from app.core.config import config


class RAGPipeline:
    """
    Orchestrates the full RAG pipeline lifecycle:
    embedding model, vector store, LLM, and retrieval chain.

    Instantiated once at application startup and shared across all requests
    via FastAPI dependency injection.
    """

    def __init__(self, llm: HuggingFacePipeline):
        self.embedding_model = load_embedding_model()
        self.vectorstore = get_vectorstore(self.embedding_model)
        self.llm = llm
        self.chain, self.retriever = build_rag_chain(self.vectorstore, self.llm)

        print("[pipeline] RAG pipeline initialized.")

    def index_documents(self, documents: List[Document]) -> int:
        """
        Add a list of LangChain Documents to the vector store.
        Returns the total number of chunks indexed.
        """
        chunk_count = add_documents_to_vectorstore(documents, self.vectorstore)
        # Rebuild chain so retriever reflects newly indexed documents
        self.chain, self.retriever = build_rag_chain(self.vectorstore, self.llm)
        print(f"[pipeline] Indexed {chunk_count} chunks.")
        return chunk_count

    def query(self, question: str) -> Tuple[str, List[dict]]:
        """
        Run a question through the RAG chain.
        Returns the generated answer and a list of source chunk metadata.
        """
        return retrieve_with_sources(question, self.chain, self.retriever)