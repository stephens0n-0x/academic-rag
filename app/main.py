from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes
from app.core.config import config
from app.core.model import load_llm
from app.rag.pipeline import RAGPipeline


# ── Application lifespan ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage startup and shutdown of expensive resources.
    The LLM and RAG pipeline are initialized once here and shared
    across all requests via dependency injection.
    """
    print("[startup] Loading LLM...")
    llm = load_llm()

    print("[startup] Initializing RAG pipeline...")
    pipeline = RAGPipeline(llm)

    # Override the placeholder dependency in routes with the live pipeline
    routes.get_pipeline = lambda: pipeline

    print("[startup] API ready.")
    yield

    # Shutdown — release resources cleanly
    print("[shutdown] Cleaning up.")


# ── Application factory ────────────────────────────────────────────────────────

app = FastAPI(
    title=config["app"]["name"],
    version=config["app"]["version"],
    description="RAG API for querying academic papers using LangChain and ChromaDB.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router, prefix="/api/v1")