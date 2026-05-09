import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.rag.retriever import load_pdf, load_arxiv_paper, search_arxiv
from app.core.config import config


# ── Request / Response schemas ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str

class SourceChunk(BaseModel):
    source: str
    page: str | int
    content_preview: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceChunk]

class IndexResponse(BaseModel):
    message: str
    chunks_indexed: int

class ArxivSearchResult(BaseModel):
    paper_id: str
    title: str
    authors: List[str]
    summary: str
    published: str
    url: str

class ArxivIndexRequest(BaseModel):
    paper_id: str


# ── Router ─────────────────────────────────────────────────────────────────────

router = APIRouter()


def get_pipeline():
    """
    FastAPI dependency providing the shared RAGPipeline instance.
    Overridden at startup in main.py with the live pipeline object.
    """
    raise NotImplementedError("Pipeline dependency not initialized.")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/health")
def health_check():
    """Liveness probe — confirms the API is running."""
    return {"status": "ok", "app": config["app"]["name"]}


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, pipeline=Depends(get_pipeline)):
    """
    Answer a question using retrieved context from indexed documents.
    Returns the generated answer and the source chunks that informed it.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    answer, sources = await run_in_threadpool(pipeline.query, request.question)

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=[SourceChunk(**s) for s in sources],
    )


@router.post("/index/pdf", response_model=IndexResponse)
async def index_pdf(file: UploadFile = File(...), pipeline=Depends(get_pipeline)):
    """
    Upload and index a PDF document.
    The file is saved temporarily, parsed, chunked, and added to the vector store.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    documents = await run_in_threadpool(load_pdf, str(file_path))
    chunk_count = await run_in_threadpool(pipeline.index_documents, documents)

    return IndexResponse(
        message=f"{file.filename} indexed successfully.",
        chunks_indexed=chunk_count,
    )


@router.post("/index/arxiv", response_model=IndexResponse)
async def index_arxiv(request: ArxivIndexRequest, pipeline=Depends(get_pipeline)):
    """
    Fetch an arXiv paper by ID and index it into the vector store.
    Accepts bare IDs (2305.14314) or full arXiv URLs.
    """
    documents = await run_in_threadpool(load_arxiv_paper, request.paper_id)

    if not documents:
        raise HTTPException(
            status_code=404,
            detail=f"Could not fetch paper {request.paper_id} from arXiv."
        )

    chunk_count = await run_in_threadpool(pipeline.index_documents, documents)

    return IndexResponse(
        message=f"Paper {request.paper_id} indexed successfully.",
        chunks_indexed=chunk_count,
    )


@router.get("/search/arxiv", response_model=List[ArxivSearchResult])
async def search_arxiv_papers(query: str):
    """
    Search arXiv by keyword and return paper metadata.
    Does not index papers — use /index/arxiv to add a paper to the vector store.
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    results = await run_in_threadpool(search_arxiv, query)
    return results