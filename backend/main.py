import os
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3")

app = FastAPI(title="Local RAG Backend with Ollama")


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


# ---------- Helper: HTTP Client ----------

async def ollama_post(path: str, payload: dict) -> dict:
    url = f"{OLLAMA_URL}{path}"
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Ollama error {resp.status_code}: {resp.text}"
            )
        return resp.json()


# ---------- Endpoints ----------

@app.get("/health")
async def health():
    return {"status": "ok", "ollama_url": OLLAMA_URL}


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    """
    Holt ein Embedding für den gegebenen Text von Ollama.
    """
    payload = {
        "model": EMBED_MODEL,
        "prompt": req.text
    }
    data = await ollama_post("/api/embeddings", payload)

    # Ollama-Embeddings-Response hat typischerweise die Form:
    # { "embedding": [ ... ] }
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

    # Für non-streaming: die letzte Antwort im Feld "message" / "content"
    message = data.get("message", {})
    content = message.get("content", "")

    return ChatResponse(content=content)


@app.post("/chat-with-context", response_model=ChatResponse)
async def chat_with_context(req: ChatWithContextRequest):
    """
    Beispiel: Frage + (optionaler) Kontext.
    Hier könntest du später deine Vektor-DB-Abfrage einbauen.
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