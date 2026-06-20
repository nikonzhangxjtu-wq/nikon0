"""Knowledge runtime schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from nikon0.app.schemas.capability import Evidence


class KnowledgeRequest(BaseModel):
    query: str
    product_model: str | None = None
    intent: str = "unknown"
    need_images: bool = False
    images: list[str] = Field(default_factory=list)
    allowed_manual_names: list[str] = Field(default_factory=list)
    knowledge_version: str | None = None
    tenant_id: str | None = None
    user_id: str | None = None
    max_evidence: int = 6


class KnowledgeResult(BaseModel):
    answer_hints: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    backend_trace: list[dict[str, object]] = Field(default_factory=list)
