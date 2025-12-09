import os
import io
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from pypdf import PdfReader

# LlamaIndex Core
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core import Settings as LlamaSettings

# LlamaIndex – Ollama Integration
from llama_index.llms.ollama import Ollama as LlamaOllamaLLM
from llama_index.embeddings.ollama import OllamaEmbedding as LlamaOllamaEmbedding

# Qdrant Integration
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import SentenceSplitter


# ---------- Umgebungsvariablen / Settings ----------

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3")

# Timeout für direkte Aufrufe an Ollama (für /embed, /chat, /chat-with-context)
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))

# Timeout für LlamaIndex->Ollama (request_timeout im Ollama-LLM)
OLLAMA_LLM_TIMEOUT = float(os.getenv("OLLAMA_LLM_TIMEOUT", "300"))

# Storage-Verzeichnis für Index-Metadaten & Docstore (nicht die Vektoren!)
INDEX_STORAGE_DIR = os.getenv("INDEX_STORAGE_DIR", "./storage")

# Qdrant Settings
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_collection")

app = FastAPI(title="Local RAG Backend with Ollama + LlamaIndex + Qdrant")


# ---------- Qdrant & LlamaIndex Global Config ----------

# Qdrant-Client
qdrant_client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
)

# VectorStore, der Qdrant nutzt
qdrant_vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=QDRANT_COLLECTION,
)

# LlamaIndex: LLM + Embedding an Ollama binden
LlamaSettings.llm = LlamaOllamaLLM(
    model=CHAT_MODEL,
    base_url=OLLAMA_URL,
    request_timeout=OLLAMA_LLM_TIMEOUT,
)
LlamaSettings.embed_model = LlamaOllamaEmbedding(
    model_name=EMBED_MODEL,
    base_url=OLLAMA_URL,
)

LlamaSettings.node_parser = SentenceSplitter(
    chunk_size=1000,      # statt 300
    chunk_overlap=150,
    paragraph_separator="\n\n",
)

# Globaler Index (Meta + Qdrant-VectorStore)
index: Optional[VectorStoreIndex] = None


# ========== FIXED STORAGE-CONTEXT HANDLING ==========

def create_storage_context(existing: bool) -> StorageContext:
    """
    existing=True  -> wir laden einen bestehenden Index, daher persist_dir verwenden
    existing=False -> wir erzeugen einen neuen StorageContext (ohne docstore.json)
    """
    storage_path = Path(INDEX_STORAGE_DIR)

    if existing:
        if not storage_path.exists():
            return StorageContext.from_defaults(vector_store=qdrant_vector_store)

        return StorageContext.from_defaults(
            vector_store=qdrant_vector_store,
            persist_dir=INDEX_STORAGE_DIR,
        )

    # Neuer Index → persist_dir erst nach Erzeugung anlegen
    storage_path.mkdir(parents=True, exist_ok=True)
    return StorageContext.from_defaults(vector_store=qdrant_vector_store)


def _load_existing_index() -> Optional[VectorStoreIndex]:
    """Versucht, einen bestehenden Index zu laden."""
    storage_path = Path(INDEX_STORAGE_DIR)
    if not storage_path.exists():
        return None

    try:
        storage_context = create_storage_context(existing=True)
        return load_index_from_storage(storage_context)
    except Exception:
        return None


def _ensure_index_loaded():
    """Lädt den Index in die globale Variable, falls möglich."""
    global index
    if index is None:
        index = _load_existing_index()


def _persist_index(idx: VectorStoreIndex):
    """Persistiert Index-Metadaten/Docstore auf Platte (Vektoren liegen in Qdrant)."""
    idx.storage_context.persist(persist_dir=INDEX_STORAGE_DIR)


# ---------- Schemas ----------

class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: List[float]


class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = False


class ChatResponse(BaseModel):
    content: str


class ChatWithContextRequest(BaseModel):
    question: str
    context: Optional[str] = None


class IndexTextRequest(BaseModel):
    text: str
    metadata: Optional[dict] = None


class QueryIndexRequest(BaseModel):
    question: str
    top_k: int = 3


# ---------- Helper: HTTP Client zu Ollama (direkte API-Calls) ----------

async def ollama_post(path: str, payload: dict) -> dict:
    url = f"{OLLAMA_URL}{path}"
    timeout = httpx.Timeout(OLLAMA_TIMEOUT, connect=5.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(url, json=payload)
        except httpx.ReadTimeout:
            raise HTTPException(
                status_code=504,
                detail=f"Ollama request to {path} exceeded {OLLAMA_TIMEOUT} seconds",
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Error while requesting Ollama: {e}",
            )

        if resp.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Ollama error {resp.status_code}: {resp.text}",
            )

        return resp.json()


# ---------- Basis-Endpunkte ----------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ollama_url": OLLAMA_URL,
        "embed_model": EMBED_MODEL,
        "chat_model": CHAT_MODEL,
        "qdrant_host": QDRANT_HOST,
        "qdrant_port": QDRANT_PORT,
        "qdrant_collection": QDRANT_COLLECTION,
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    payload = {
        "model": EMBED_MODEL,
        "prompt": req.text,
    }
    data = await ollama_post("/api/embeddings", payload)

    embedding = data.get("embedding")
    if embedding is None:
        raise HTTPException(status_code=500, detail="No embedding returned from Ollama")

    return EmbedResponse(embedding=embedding)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    payload = {
        "model": CHAT_MODEL,
        "messages": [m.model_dump() for m in req.messages],
        "stream": req.stream,
    }
    data = await ollama_post("/api/chat", payload)

    message = data.get("message", {})
    content = message.get("content", "")

    return ChatResponse(content=content)


@app.post("/chat-with-context", response_model=ChatResponse)
async def chat_with_context(req: ChatWithContextRequest):
    system_prompt = (
        "Du bist ein hilfreicher Assistent. "
        "Nutze den bereitgestellten Kontext, wenn er relevant ist. "
        "Wenn etwas nicht im Kontext steht, sage offen, dass du es nicht weißt."
    )

    context_block = f"\n\nKontext:\n{req.context}" if req.context else ""
    user_content = f"Frage:\n{req.question}{context_block}"

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
    }

    data = await ollama_post("/api/chat", payload)
    message = data.get("message", {})
    content = message.get("content", "")

    return ChatResponse(content=content)


# ---------- LlamaIndex + Qdrant: Text Indexieren & Query ----------

@app.post("/index-text")
async def index_text(req: IndexTextRequest):
    """
    Nimmt Text entgegen, chunked ihn mit LlamaIndex und fügt ihn dem Vektorindex (Qdrant) hinzu.
    """
    global index
    _ensure_index_loaded()

    doc = Document(text=req.text, metadata=req.metadata or {})

    if index is None:
        # neuer Index → kein persist_dir verwenden
        storage_context = create_storage_context(existing=False)
        index = VectorStoreIndex.from_documents(
            [doc],
            storage_context=storage_context,
        )
    else:
        # existierender Index → einfach einfügen
        index.insert(doc)

    _persist_index(index)

    return {"status": "ok", "message": "Text indexed successfully"}


@app.post("/index-pdf")
async def index_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    raw_bytes = await file.read()

    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages_text: List[str] = []

        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(text)

        if not pages_text:
            raise HTTPException(status_code=400, detail="No text found in PDF")

        full_text = "\n\n".join(pages_text)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing PDF: {e}")

    req = IndexTextRequest(
        text=full_text,
        metadata={"filename": file.filename},
    )
    return await index_text(req)


@app.post("/query-index", response_model=ChatResponse)
async def query_index(req: QueryIndexRequest):
    global index
    _ensure_index_loaded()

    if index is None:
        raise HTTPException(
            status_code=400,
            detail="Index is empty. Please index some text or PDFs first.",
        )

    query_engine = index.as_query_engine(similarity_top_k=req.top_k)

    try:
        response = query_engine.query(req.question)
    except httpx.ReadTimeout:
        raise HTTPException(
            status_code=504,
            detail="LLM (Ollama) timed out while answering the query.",
        )

    content = getattr(response, "response", str(response))

    return ChatResponse(content=content)