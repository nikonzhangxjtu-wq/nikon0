"""KnowledgeRuntime with a local structured-manual backend."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from nikon0.app.schemas.capability import Evidence
from nikon0.app.schemas.knowledge import KnowledgeRequest, KnowledgeResult


@dataclass(frozen=True)
class ManualPassage:
    manual_name: str
    text: str
    score: float


class StructuredManualBackend:
    """Small local backend over `手册/*.txt`.

    This is intentionally lightweight: it gives ProductSupportSkill a real,
    traceable knowledge path without requiring Milvus/Ollama during the first
    nikon0 integration.
    """

    def __init__(self, manual_dir: str | Path = "手册") -> None:
        self.manual_dir = Path(manual_dir)

    def query(self, request: KnowledgeRequest) -> KnowledgeResult:
        passages = self._search(
            request.query,
            limit=max(1, request.max_evidence),
            allowed_manual_names=request.allowed_manual_names,
        )
        evidence: list[Evidence] = []
        hints: list[str] = []
        for idx, passage in enumerate(passages, start=1):
            evidence_id = f"manual:{passage.manual_name}:{idx}"
            evidence.append(
                Evidence(
                    evidence_id=evidence_id,
                    source="manual",
                    text=passage.text,
                    payload={"manual_name": passage.manual_name, "score": passage.score},
                    confidence=min(1.0, max(0.1, passage.score / 5.0)),
                )
            )
            hints.append(f"[{passage.manual_name}] {passage.text}")
        return KnowledgeResult(
            answer_hints=hints,
            evidence=evidence,
            backend_trace=[
                {
                    "backend": "structured_manual",
                    "manual_dir": str(self.manual_dir),
                    "hit_count": len(passages),
                }
            ],
        )

    def _search(
        self,
        query: str,
        *,
        limit: int,
        allowed_manual_names: list[str] | None = None,
    ) -> list[ManualPassage]:
        if not self.manual_dir.is_dir():
            return []
        tokens = _query_tokens(query)
        if not tokens:
            return []
        allowed = {name.strip() for name in (allowed_manual_names or []) if name.strip()}
        scored: list[ManualPassage] = []
        for path in sorted(self.manual_dir.glob("*.txt")):
            if allowed and path.stem not in allowed:
                continue
            try:
                text = _read_manual_text(path)
            except OSError:
                continue
            for passage in _split_passages(text):
                score = _score_passage(tokens, passage, path.stem)
                if score > 0:
                    scored.append(ManualPassage(manual_name=path.stem, text=passage, score=score))
        scored.sort(key=lambda item: (-item.score, item.manual_name, item.text[:40]))
        return scored[:limit]


class KnowledgeBackend(Protocol):
    def query(self, request: KnowledgeRequest) -> KnowledgeResult:
        ...


class EnterpriseRagBackend:
    """Adapter over the existing Milvus/BM25/rerank/multimodal RAG pipeline.

    The legacy RAG code already owns indexing and retrieval. nikon0 keeps this
    as a thin governance adapter: permission filtering, evidence normalization,
    backend trace, and fallback to local manuals when the enterprise backend is
    unavailable.
    """

    def __init__(
        self,
        *,
        retriever_factory: Callable[[], Any] | None = None,
        fallback_backend: KnowledgeBackend | None = None,
        manual_name_decider: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self._retriever_factory = retriever_factory or self._default_retriever_factory
        self._retriever: Any | None = None
        self.fallback_backend = fallback_backend or StructuredManualBackend()
        self._manual_name_decider = manual_name_decider or self._manual_name_decision

    def query(self, request: KnowledgeRequest) -> KnowledgeResult:
        try:
            return self._query_enterprise(request)
        except Exception as exc:  # noqa: BLE001
            fallback = self.fallback_backend.query(request)
            fallback.backend_trace.insert(
                0,
                {
                    "backend": "enterprise_rag",
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "fallback": "structured_manual",
                },
            )
            return fallback

    def _query_enterprise(self, request: KnowledgeRequest) -> KnowledgeResult:
        retriever = self._get_retriever()
        allowed_manual_names = [
            name.strip() for name in request.allowed_manual_names if str(name).strip()
        ]
        if allowed_manual_names:
            manual_decision = {
                "manual_name": allowed_manual_names[0] if len(allowed_manual_names) == 1 else "",
                "confidence": 1.0,
                "source": "product_resolver",
                "reason": "restricted by resolved product manuals",
                "should_filter": len(allowed_manual_names) == 1,
            }
            manual_name = allowed_manual_names[0] if len(allowed_manual_names) == 1 else None
        else:
            manual_decision = self._manual_name_decider(request.query)
            manual_name = manual_decision.get("manual_name") if manual_decision.get("should_filter") else None
        raw_chunks = list(
            retriever.retrieve(
                request.query,
                top_k=max(1, request.max_evidence),
                manual_name=manual_name,
                image_inputs=request.images if request.need_images or request.images else [],
            )
        )
        permitted_chunks = self._apply_permission_filter(raw_chunks, request.allowed_manual_names)
        filtered_chunks = self._score_filter(permitted_chunks)[: max(1, request.max_evidence)]
        trace = self._build_retrieval_trace(
            retriever=retriever,
            request=request,
            raw_chunks=raw_chunks,
            filtered_chunks=filtered_chunks,
            manual_decision=manual_decision,
        )

        evidence = [
            self._chunk_to_evidence(chunk, idx=idx, request=request)
            for idx, chunk in enumerate(filtered_chunks, start=1)
        ]
        return KnowledgeResult(
            answer_hints=[self._chunk_hint(chunk) for chunk in filtered_chunks],
            evidence=evidence,
            backend_trace=[
                {
                    "backend": "enterprise_rag",
                    "ok": True,
                    "collection": self._attr(retriever, "collection_name", ""),
                    "image_collection": self._attr(retriever, "image_collection_name", ""),
                    "dense_field": self._attr(retriever, "dense_field", ""),
                    "sparse_enabled": bool(self._attr(retriever, "sparse_enabled", False)),
                    "rerank": "enabled",
                    "knowledge_version": request.knowledge_version,
                    "tenant_id": request.tenant_id,
                    "manual_name_decision": manual_decision,
                    "permission_filter": {
                        "allowed_manual_names": request.allowed_manual_names,
                    },
                    "raw_count": len(raw_chunks),
                    "filtered_count": len(filtered_chunks),
                    "retrieval_trace": trace,
                }
            ],
        )

    def _get_retriever(self) -> Any:
        if self._retriever is None:
            self._retriever = self._retriever_factory()
        return self._retriever

    @staticmethod
    def _default_retriever_factory() -> Any:
        from app.services.retriever import VectorRetriever

        return VectorRetriever()

    @staticmethod
    def _manual_name_decision(query: str) -> dict[str, Any]:
        try:
            from app.services.rag_skill.query_construction import query_construction_decision

            decision = query_construction_decision(query)
            return {
                "manual_name": decision.manual_name,
                "confidence": decision.confidence,
                "source": decision.source,
                "reason": decision.reason,
                "should_filter": decision.should_filter,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "manual_name": "",
                "confidence": 0.0,
                "source": "unavailable",
                "reason": str(exc),
                "should_filter": False,
            }

    @staticmethod
    def _apply_permission_filter(chunks: list[Any], allowed_manual_names: list[str]) -> list[Any]:
        if not allowed_manual_names:
            return chunks
        allowed = {name.strip() for name in allowed_manual_names if name.strip()}
        return [chunk for chunk in chunks if str(getattr(chunk, "manual_name", "")) in allowed]

    @staticmethod
    def _score_filter(chunks: list[Any]) -> list[Any]:
        return sorted(
            [chunk for chunk in chunks if float(getattr(chunk, "score", 0.0)) > 0.0],
            key=lambda item: float(getattr(item, "score", 0.0)),
            reverse=True,
        )

    @staticmethod
    def _build_retrieval_trace(
        *,
        retriever: Any,
        request: KnowledgeRequest,
        raw_chunks: list[Any],
        filtered_chunks: list[Any],
        manual_decision: dict[str, Any],
    ) -> dict[str, Any]:
        if hasattr(retriever, "build_trace"):
            trace = retriever.build_trace(
                query=request.query,
                top_k=max(1, request.max_evidence),
                raw_chunks=raw_chunks,
                filtered_chunks=filtered_chunks,
                manual_name_decisions=[manual_decision.get("manual_name", "")],
            )
            return _model_or_object_to_dict(trace)
        return {
            "query": request.query,
            "top_k": max(1, request.max_evidence),
            "raw_count": len(raw_chunks),
            "filtered_count": len(filtered_chunks),
            "retrieved_chunk_ids": [str(getattr(chunk, "chunk_id", "")) for chunk in raw_chunks],
            "filtered_chunk_ids": [str(getattr(chunk, "chunk_id", "")) for chunk in filtered_chunks],
            "retrieved_manual_names": [str(getattr(chunk, "manual_name", "")) for chunk in raw_chunks],
            "filtered_manual_names": [str(getattr(chunk, "manual_name", "")) for chunk in filtered_chunks],
            "selected_image_ids": [
                str(getattr(image, "image_id", ""))
                for chunk in filtered_chunks
                for image in getattr(chunk, "image_evidence", [])
            ],
        }

    @staticmethod
    def _chunk_to_evidence(chunk: Any, *, idx: int, request: KnowledgeRequest) -> Evidence:
        chunk_id = str(getattr(chunk, "chunk_id", f"chunk_{idx}"))
        score = float(getattr(chunk, "score", 0.0))
        manual_name = str(getattr(chunk, "manual_name", ""))
        image_evidence = [
            _model_or_object_to_dict(item)
            for item in getattr(chunk, "image_evidence", [])
        ]
        payload = {
            "chunk_id": chunk_id,
            "manual_name": manual_name,
            "score": score,
            "image_ids": list(getattr(chunk, "image_ids", []) or []),
            "image_evidence": image_evidence,
            "knowledge_version": request.knowledge_version,
            "tenant_id": request.tenant_id,
        }
        return Evidence(
            evidence_id=f"enterprise_rag:{chunk_id}",
            source="enterprise_rag",
            text=str(getattr(chunk, "text", "")),
            payload=payload,
            confidence=min(1.0, max(0.05, score)),
        )

    @staticmethod
    def _chunk_hint(chunk: Any) -> str:
        manual_name = str(getattr(chunk, "manual_name", ""))
        text = str(getattr(chunk, "text", ""))
        image_parts = [
            str(getattr(item, "prompt_text", "")).strip()
            for item in getattr(chunk, "image_evidence", [])
            if str(getattr(item, "prompt_text", "")).strip()
        ]
        suffix = "\n[图片证据]\n" + "\n".join(image_parts[:2]) if image_parts else ""
        return f"[{manual_name}] {text}{suffix}"

    @staticmethod
    def _attr(obj: Any, name: str, default: Any) -> Any:
        return getattr(obj, name, default)


class KnowledgeRuntime:
    def __init__(self, backend: KnowledgeBackend | None = None) -> None:
        self.backend = backend or EnterpriseRagBackend()

    async def query(self, request: KnowledgeRequest) -> KnowledgeResult:
        return self.backend.query(request)


def _query_tokens(query: str) -> list[str]:
    raw = query.strip().lower()
    english_stop_words = {
        "a", "an", "and", "are", "at", "can", "do", "does", "for", "from", "how",
        "i", "in", "is", "it", "my", "of", "on", "or", "the", "this", "to", "what",
        "when", "with", "you", "your",
    }
    tokens = [
        token for token in re.findall(r"[a-z0-9][a-z0-9/.-]*", raw)
        if len(token) > 1 and token not in english_stop_words
    ]
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", raw)
    for run in cjk_runs:
        # CJK has no whitespace word boundary. Overlapping bi/tri-grams preserve
        # useful concepts such as 清洁、滤网、化油器 instead of treating a whole
        # sentence as one token that can never match a passage.
        for size in (2, 3, 4):
            tokens.extend(run[index:index + size] for index in range(max(0, len(run) - size + 1)))
    return list(dict.fromkeys(tokens))


def _read_manual_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(decoded, list) and decoded and isinstance(decoded[0], str):
        return decoded[0]
    if isinstance(decoded, str):
        return decoded
    return raw


def _split_passages(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n|(?<=[。！？!?])\s*", text) if part.strip()]
    compact: list[str] = []
    for paragraph in paragraphs:
        cleaned = re.sub(r"\s+", " ", paragraph).strip()
        if len(cleaned) < 12:
            continue
        if len(cleaned) <= 500:
            compact.append(cleaned)
            continue
        # Long OCR/manual blocks are split into overlapping windows so a table of
        # contents does not crowd out the actual operating instruction.
        for start in range(0, len(cleaned), 360):
            compact.append(cleaned[start:start + 500])
    return compact[:2000]


def _score_passage(tokens: list[str], passage: str, manual_name: str) -> float:
    haystack = f"{manual_name}\n{passage}".lower()
    score = 0.0
    matched = 0
    for token in tokens:
        if token and token in haystack:
            matched += 1
            score += 1.0 if len(token) >= 3 else 0.35
    if not matched:
        return 0.0
    # Prefer passages with several independent query concepts, and strongly
    # prefer a product-name hit without granting every manual a free score.
    score += min(2.0, matched * 0.2)
    if any(token in manual_name.lower() for token in tokens if len(token) >= 3):
        score += 1.5
    if "table of contents" in haystack or "目录" in haystack:
        score -= 0.75
    return max(0.0, score)


def _model_or_object_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return dict(value)
    out: dict[str, Any] = {}
    for name in (
        "image_id",
        "image_type",
        "match_reason",
        "prompt_text",
        "score",
        "parent_chunk_ids",
        "query",
        "top_k",
        "raw_count",
        "filtered_count",
        "score_threshold",
        "top1_score",
        "retrieved_chunk_ids",
        "filtered_chunk_ids",
        "retrieved_manual_names",
        "filtered_manual_names",
        "source_queries",
        "manual_name_decisions",
        "image_vector_hits",
        "ocr_entity_hits",
        "selected_image_ids",
    ):
        if hasattr(value, name):
            out[name] = getattr(value, name)
    if out:
        return out
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return out
