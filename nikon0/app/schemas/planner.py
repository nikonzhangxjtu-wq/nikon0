"""Planner schemas for intent and capability routing."""

from __future__ import annotations

from pydantic import BaseModel, Field

from nikon0.app.schemas.capability import RiskLevel


class IntentSignal(BaseModel):
    intent: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class CapabilityCandidate(BaseModel):
    kind: str
    name: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class PlanStep(BaseModel):
    step_id: str
    capability: str
    purpose: str
    status: str = "pending"


class PlannerResult(BaseModel):
    intents: list[IntentSignal] = Field(default_factory=list)
    candidates: list[CapabilityCandidate] = Field(default_factory=list)
    steps: list[PlanStep] = Field(default_factory=list)
    recommended_skill: str | None = None
    risk_level: RiskLevel = "low"
    needs_general_handle: bool = True
    is_composite: bool = False
