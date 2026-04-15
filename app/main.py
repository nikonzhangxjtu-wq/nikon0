"""FastAPI app entrypoint with competition-compatible `/chat` endpoint."""

from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException, status

from app.core.config import settings
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.pipeline import ChatPipeline
from app.services.session_store import ensure_session_id

app = FastAPI(
    title="Multimodal Customer Service Agent - V1 Scaffold",
    version="0.1.0",
)

pipeline = ChatPipeline()


def verify_bearer_token(authorization: str = Header(default="")) -> None:
    """Simple bearer token auth check.

    Expected format:
    Authorization: Bearer <token>
    """

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header.",
        )

    token = authorization.removeprefix("Bearer ").strip()
    if token != settings.api_bearer_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token.",
        )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Lightweight health check endpoint."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    _: None = Depends(verify_bearer_token),
) -> ChatResponse:
    """Competition-compatible chat endpoint.

    Notes for your own implementation:
    - If `stream` is requested, V1 can explicitly reject or degrade to sync.
    - `images` are accepted here; you can integrate multimodal processing later.
    """
    if req.stream:
        # You can later switch to streaming responses.
        # For V1 scaffold, keep behavior explicit.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="`stream=true` is not supported in V1 scaffold.",
        )

    sid = ensure_session_id(req.session_id)
    result = pipeline.run(question=req.question, images=req.images)
    return ChatResponse.success(answer=result.answer, session_id=sid)
