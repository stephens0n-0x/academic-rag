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
        query=query,
        max_results=config["arxiv"]["max_results"],
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results = []
    for paper in client.results(search):
        results.append({
            "paper_id": paper.entry_id.split("/")[-1],
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "summary": paper.summary[:300] + "...",
            "published": str(paper.published.date()),
            "url": paper.entry_id,
        })
        if len(results) >= config["arxiv"]["max_results"]:
            break

    return results


def format_retrieved_chunks(chunks: List[Document]) -> str:
    """Concatenate retrieved document chunks into a single context string."""
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, "
        f"Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
        for doc in chunks
    )


def build_rag_chain(vectorstore: Chroma, llm: HuggingFacePipeline):
    """
    Construct and return a LangChain RAG chain.
    Chain flow: question → retriever → format context → prompt → LLM → parse output
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config["vectorstore"]["top_k"]},
    )

    chain = (
        {
            "context": retriever | format_retrieved_chunks,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def retrieve_with_sources(
    question: str,
    chain,
    retriever,
) -> Tuple[str, List[dict]]:
    """
    Run the RAG chain and return both the generated answer and source metadata.
    Sources allow the API to tell the user exactly which chunks informed the answer.
    """
    answer = chain.invoke(question)

    source_chunks = retriever.invoke(question)
    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "N/A"),
            "content_preview": doc.page_content[:150] + "...",
        }
        for doc in source_chunks
    ]

    return answer, sources