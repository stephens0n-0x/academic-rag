from pathlib import Path
from typing import List, Tuple

import arxiv
from langchain_community.document_loaders import PyPDFLoader, ArxivLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline

from app.core.config import config


# RAG prompt template — instructs the model to answer strictly from retrieved context
# and acknowledge when the context is insufficient rather than hallucinating
RAG_PROMPT = PromptTemplate.from_template("""
You are a research assistant specializing in academic papers.
Use only the following retrieved context to answer the question.
If the context does not contain enough information, say so explicitly.
Do not fabricate information that is not present in the context.

Context:
{context}

Question:
{question}

Answer:
""")


def load_pdf(file_path: str) -> List[Document]:
    """Load and return pages from a local PDF file as LangChain Documents."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_arxiv_paper(paper_id: str) -> List[Document]:
    """
    Fetch an arXiv paper by ID and return its content as LangChain Documents.
    Accepts both full URLs and bare IDs (e.g. '2305.14314' or 'https://arxiv.org/abs/2305.14314').
    """
    # Strip URL prefix if a full arXiv URL was provided
    paper_id = paper_id.strip()
    if "arxiv.org" in paper_id:
        paper_id = paper_id.split("/")[-1]

    loader = ArxivLoader(
        query=paper_id,
        load_max_docs=1,
    )
    return loader.load()


def search_arxiv(query: str) -> List[dict]:
    """
    Search arXiv by keyword and return paper metadata without downloading full text.
    Used to let users discover relevant papers before indexing them.
    """
    client = arxiv.Client()
    search = arxiv.Search(