import os
import io
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from pypdf import PdfReader

# LlamaIndex
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core import Settings as LlamaSettings
from llama_index.llms.ollama import Ollama as LlamaOllamaLLM
from llama_index.embeddings.ollama import OllamaEmbedding as LlamaOllamaEmbedding


# ---------- Umgebungsvariablen / Settings ----------

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3")

# Timeout für Aufrufe an Ollama (Sekunden)
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))

# Verzeichnis, in dem der LlamaIndex persistent gespeichert wird
INDEX_STORAGE_DIR = os.getenv("INDEX_STORAGE_DIR", "./storage")

app = FastAPI(title="Local RAG Backend with Ollama + LlamaIndex")


# ---------- LlamaIndex Global Config ----------

LlamaSettings.llm = LlamaOllamaLLM(
    model=CHAT_MODEL,
    base_url=OLLAMA_URL,
    request_timeout=OLLAMA_TIMEOUT
)
LlamaSettings.embed_model = LlamaOllamaEmbedding(
    model_name=EMBED_MODEL,
    base_url=OLLAMA_URL,
)

# Globaler Index (einfacher Ansatz, ein gemeinsamer Index)
index: Optional[VectorStoreIndex] = None


def _load_existing_index() -> Optional[VectorStoreIndex]:
    """Versucht, einen bestehenden Index vom Dateisystem zu laden."""
    storage_path = Path(INDEX_STORAGE_DIR)
    if storage_path.exists():
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_STORAGE_DIR)
        return load_index_from_storage(storage_context)
    return None


def _ensure_index_loaded():
    """Lädt den Index in die globale Variable, falls möglich."""
    global index
    if index is None:
        index = _load_existing_index()


def _persist_index(idx: VectorStoreIndex):
    """Persistiert den Index auf die Platte."""
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
    """
    Text, der über LlamaIndex gechunked und in den Vektorindex eingefügt werden soll.
    """
    text: str
    metadata: Optional[dict] = None  # z.B. {"filename": "foo.pdf"}


class QueryIndexRequest(BaseModel):
    question: str
    top_k: int = 3


# ---------- Helper: HTTP Client zu Ollama ----------

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
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    """
    Holt ein Embedding für den gegebenen Text von Ollama.
    """
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
    """
    Einfacher Chat-Endpunkt über Ollama /api/chat.
    """
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
    """
    Frage + optionaler Kontext, direkt über Ollama /api/chat.
    Hier könntest du später (wenn du willst) LlamaIndex-Retrieval als Kontext einfügen.
    """
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


# ---------- LlamaIndex: Text Indexieren & Query ----------

@app.post("/index-text")
async def index_text(req: IndexTextRequest):
    """
    Nimmt Text entgegen, chunked ihn mit LlamaIndex und fügt ihn dem Vektorindex hinzu.
    """
    global index
    _ensure_index_loaded()

    doc = Document(text=req.text, metadata=req.metadata or {})

    if index is None:
        # neuen Index erstellen
        index = VectorStoreIndex.from_documents([doc])
    else:
        # existierenden Index erweitern
        index.insert(doc)

    _persist_index(index)

    return {"status": "ok", "message": "Text indexed successfully"}


@app.post("/index-pdf")
async def index_pdf(file: UploadFile = File(...)):
    """
    Nimmt eine PDF-Datei entgegen, extrahiert den Text und indexiert ihn über LlamaIndex.
    """
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
        # bereits geworfen
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing PDF: {e}")

    req = IndexTextRequest(
        text=full_text,
        metadata={"filename": file.filename},
    )
    # Wiederverwendung der Logik aus /index-text
    return await index_text(req)


@app.post("/query-index", response_model=ChatResponse)
async def query_index(req: QueryIndexRequest):
    """
    Führt eine semantische Suche im LlamaIndex durch und gibt eine generierte Antwort zurück.
    """
    global index
    _ensure_index_loaded()

    if index is None:
        raise HTTPException(
            status_code=400,
            detail="Index is empty. Please index some text or PDFs first.",
        )

    # Query-Engine mit top_k konfigurieren
    query_engine = index.as_query_engine(similarity_top_k=req.top_k)
    response = query_engine.query(req.question)

    # je nach LlamaIndex-Version: response.response oder str(response)
    content = getattr(response, "response", str(response))

    return ChatResponse(content=content)