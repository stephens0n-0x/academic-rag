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
    print("[startup] Loading LLM...")
    llm = load_llm()

    print("[startup] Initializing RAG pipeline...")
    pipeline = RAGPipeline(llm)

    # FastAPI's official dependency override mechanism
    # replaces get_pipeline with a function that returns the live pipeline
    app.dependency_overrides[routes.get_pipeline] = lambda: pipeline

    print("[startup] API ready.")
    yield

    app.dependency_overrides.clear()
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