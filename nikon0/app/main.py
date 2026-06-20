"""FastAPI entrypoint for nikon0."""

from __future__ import annotations

from fastapi import FastAPI

from nikon0.app.api.v1.chat import router as chat_router

app = FastAPI(
    title="nikon0 Enterprise Assistant",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "nikon0"}


app.include_router(chat_router)
