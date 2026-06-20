"""Resolve ambiguous product references before manual-scoped retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ResolutionStatus = Literal["resolved", "disambiguation_required", "passthrough"]
ResolutionSource = Literal[
    "session",
    "user_choice",
    "strong_signal",
    "retrieval_evidence",
    "passthrough",
    "unresolved",
]


@dataclass(frozen=True)
class CatalogProduct:
    product_id: str
    display_name: str
    manual_names: tuple[str, ...]
    exclusive_aliases: tuple[str, ...]
    identity_aliases: tuple[str, ...]


@dataclass(frozen=True)
class ProductCluster:
    cluster_id: str
    generic_terms: tuple[str, ...]
    products: tuple[CatalogProduct, ...]


@dataclass(frozen=True)
class ProductCatalog:
    generic_terms: frozenset[str]
    clusters: tuple[ProductCluster, ...]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ProductCatalog":
        catalog_path = Path(path) if path else Path(__file__).with_name("product_catalog.json")
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        generic_terms = frozenset(str(item) for item in raw.get("generic_terms", []))
        clusters: list[ProductCluster] = []
        for cluster in raw.get("clusters", []):
            products = tuple(
                CatalogProduct(
                    product_id=str(item["product_id"]),
                    display_name=str(item["display_name"]),
                    manual_names=tuple(str(name) for name in item.get("manual_names", [])),
                    exclusive_aliases=tuple(str(alias) for alias in item.get("exclusive_aliases", [])),
                    identity_aliases=tuple(
                        str(alias) for alias in item.get("identity_aliases", [])
                    ),
                )
                for item in cluster.get("products", [])
            )
            clusters.append(
                ProductCluster(
                    cluster_id=str(cluster["cluster_id"]),
                    generic_terms=tuple(str(term) for term in cluster.get("generic_terms", [])),
                    products=products,
                )
            )
        return cls(generic_terms=generic_terms, clusters=tuple(clusters))

    def product_by_id(self, product_id: str) -> CatalogProduct | None:
        for cluster in self.clusters:
            for product in cluster.products:
                if product.product_id == product_id:
                    return product
        return None


@dataclass(frozen=True)
class ProductResolution:
    status: ResolutionStatus
    source: ResolutionSource
    product_id: str | None = None
    display_name: str | None = None
    manual_names: tuple[str, ...] = ()
    matched_terms: tuple[str, ...] = ()
    disclose_default_product: bool = False
    candidates: tuple[CatalogProduct, ...] = ()
    reason: str = ""

    def to_trace(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source": self.source,
            "product_id": self.product_id,
            "display_name": self.display_name,
            "manual_names": list(self.manual_names),
            "matched_terms": list(self.matched_terms),
            "disclose_default_product": self.disclose_default_product,
            "candidate_product_ids": [item.product_id for item in self.candidates],
            "reason": self.reason,
        }


class ProductResolver:
    def __init__(self, catalog: ProductCatalog | None = None) -> None:
        self.catalog = catalog or ProductCatalog.load()

    def resolve(
        self,
        query: str,
        *,
        session_state: dict[str, Any] | None = None,
    ) -> ProductResolution:
        text = query.strip()
        if not text:
            return ProductResolution(status="passthrough", source="passthrough", reason="empty query")

        support_state = _product_support_state(session_state)
        pending_candidates = _pending_candidates(support_state, self.catalog)
        if pending_candidates:
            chosen = _parse_user_choice(text, pending_candidates)
            if chosen is not None:
                return ProductResolution(
                    status="resolved",
                    source="user_choice",
                    product_id=chosen.product_id,
                    display_name=chosen.display_name,
                    manual_names=chosen.manual_names,
                    matched_terms=(text,),
                    disclose_default_product=False,
                    reason="user selected product after disambiguation",
                )

        explicit = self._resolve_explicit_product_identity(text)
        if explicit is not None:
            return explicit

        # An explicit identity in this user message always outranks the session
        # shortcut. This is essential when a customer switches products.
        for cluster in self.catalog.clusters:
            resolution = self._resolve_cluster(text, cluster)
            if resolution is not None:
                return resolution

        selected_id = str(support_state.get("selected_product_id") or "").strip()
        if selected_id and not _looks_like_product_switch(text):
            product = self.catalog.product_by_id(selected_id)
            if product is not None:
                return ProductResolution(
                    status="resolved",
                    source="session",
                    product_id=product.product_id,
                    display_name=product.display_name,
                    manual_names=product.manual_names,
                    disclose_default_product=False,
                    reason="reuse session-selected product",
                )

        return ProductResolution(status="passthrough", source="passthrough", reason="no catalog cluster matched")

    def _resolve_explicit_product_identity(self, text: str) -> ProductResolution | None:
        """Prefer a named product over a generic technical term from another manual."""
        matches: list[tuple[CatalogProduct, tuple[str, ...]]] = []
        for cluster in self.catalog.clusters:
            for product in cluster.products:
                terms = _matched_aliases(text, product.identity_aliases)
                if product.display_name and product.display_name.lower() in text.lower():
                    terms = tuple(dict.fromkeys((*terms, product.display_name)))
                if terms:
                    matches.append((product, terms))
        if len(matches) != 1:
            return None
        product, terms = matches[0]
        return ProductResolution(
            status="resolved",
            source="strong_signal",
            product_id=product.product_id,
            display_name=product.display_name,
            manual_names=product.manual_names,
            matched_terms=terms,
            disclose_default_product=False,
            reason="user named product identity",
        )

    def resolve_from_retrieval(
        self,
        resolution: ProductResolution,
        manual_names: list[str],
        manual_scores: list[float] | None = None,
    ) -> ProductResolution:
        if resolution.status != "disambiguation_required" or not resolution.candidates:
            return resolution
        scores: dict[str, float] = {}
        counts: dict[str, float] = {}
        first_seen: dict[str, int] = {}
        for index, manual_name in enumerate(manual_names):
            normalized = str(manual_name).strip()
            if not normalized:
                continue
            score = 1.0
            if manual_scores is not None and index < len(manual_scores):
                score = max(0.0, float(manual_scores[index]))
            for product in resolution.candidates:
                if normalized in product.manual_names:
                    counts[product.product_id] = counts.get(product.product_id, 0.0) + 1.0
                    scores[product.product_id] = scores.get(product.product_id, 0.0) + score
                    first_seen.setdefault(product.product_id, index)
        if not scores:
            return resolution
        ranked = sorted(scores.items(), key=lambda item: (-item[1], first_seen.get(item[0], 9999), item[0]))
        if len(ranked) > 1 and (ranked[0][1] - ranked[1][1]) < 0.05:
            return resolution
        product = self.catalog.product_by_id(ranked[0][0])
        if product is None:
            return resolution
        matched_manuals = tuple(
            name for name in manual_names if str(name).strip() in product.manual_names
        )
        return ProductResolution(
            status="resolved",
            source="retrieval_evidence",
            product_id=product.product_id,
            display_name=product.display_name,
            manual_names=product.manual_names,
            matched_terms=matched_manuals,
            disclose_default_product=True,
            reason="inferred product from retrieved manual evidence",
        )

    def _resolve_cluster(self, text: str, cluster: ProductCluster) -> ProductResolution | None:
        if not _text_hits_any(text, cluster.generic_terms) and not any(
            _text_hits_any(text, product.exclusive_aliases) for product in cluster.products
        ):
            return None

        scored: list[tuple[CatalogProduct, float, tuple[str, ...]]] = []
        for product in cluster.products:
            matched = _matched_aliases(text, product.exclusive_aliases)
            if not matched:
                continue
            score = float(len(matched))
            for term in matched:
                if term not in self.catalog.generic_terms:
                    score += 2.0
            scored.append((product, score, matched))

        if scored:
            scored.sort(key=lambda item: (-item[1], item[0].product_id))
            top_product, top_score, top_terms = scored[0]
            second_score = scored[1][1] if len(scored) > 1 else 0.0
            if top_score >= 2.0 and (top_score - second_score) >= 1.0:
                return self._strong_resolution(top_product, top_terms, text)
            if len(scored) == 1 and top_score >= 1.0 and any(
                term not in self.catalog.generic_terms for term in top_terms
            ):
                return self._strong_resolution(top_product, top_terms, text)

        if _text_hits_any(text, cluster.generic_terms) or _text_hits_any(text, self.catalog.generic_terms):
            return ProductResolution(
                status="disambiguation_required",
                source="unresolved",
                candidates=cluster.products,
                reason="generic product terms without exclusive signal",
            )

        return None

    def _strong_resolution(
        self,
        product: CatalogProduct,
        matched_terms: tuple[str, ...],
        text: str,
    ) -> ProductResolution:
        inferred = not _user_named_product_identity(text, product, matched_terms)
        return ProductResolution(
            status="resolved",
            source="strong_signal",
            product_id=product.product_id,
            display_name=product.display_name,
            manual_names=product.manual_names,
            matched_terms=matched_terms,
            disclose_default_product=inferred,
            reason="inferred product from technical signal" if inferred else "user named product identity",
        )


def _user_named_product_identity(
    text: str,
    product: CatalogProduct,
    matched_terms: tuple[str, ...],
) -> bool:
    """True when the user already named the product, not just a technical hint."""
    if product.display_name and product.display_name in text:
        return True
    for alias in product.identity_aliases:
        if alias and (alias in text or alias.lower() in text.lower()):
            return True
    for term in matched_terms:
        if term in product.identity_aliases:
            return True
    return False


def build_disambiguation_answer(candidates: tuple[CatalogProduct, ...]) -> str:
    lines = [
        "您的问题可能对应多个产品，请先确认具体型号，我才能准确查阅对应手册：",
    ]
    for idx, product in enumerate(candidates, start=1):
        manuals = "、".join(product.manual_names)
        lines.append(f"{idx}. {product.display_name}（手册：{manuals}）")
    lines.append("请直接回复序号（如 1），或说明具体型号/关键词（如 EF-S、拍立得）。")
    return "\n".join(lines)


def build_product_disclosure_prefix(resolution: ProductResolution) -> str:
    if not resolution.disclose_default_product or not resolution.display_name:
        return ""
    if resolution.source == "retrieval_evidence":
        lead = "根据召回的手册证据，"
    elif resolution.matched_terms:
        terms = "、".join(resolution.matched_terms[:3])
        lead = f"根据您问题中的「{terms}」，"
    else:
        lead = "根据当前识别到的产品信息，"
    return (
        f"{lead}当前默认按 **{resolution.display_name}** 为您解答；"
        "如您的实际产品不符，请告诉我正确型号。\n\n"
    )


def apply_product_disclosure(answer: str, resolution: ProductResolution) -> str:
    prefix = build_product_disclosure_prefix(resolution)
    if not prefix:
        return answer
    body = answer.strip()
    if not body:
        return prefix.strip()
    return prefix + body


def _product_support_state(session_state: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(session_state, dict):
        return {}
    raw = session_state.get("product_support")
    return raw if isinstance(raw, dict) else {}


def _pending_candidates(
    support_state: dict[str, Any],
    catalog: ProductCatalog,
) -> tuple[CatalogProduct, ...]:
    if not bool(support_state.get("disambiguation_pending")):
        return ()
    ids = support_state.get("disambiguation_candidates") or []
    if not isinstance(ids, list):
        return ()
    products: list[CatalogProduct] = []
    for product_id in ids:
        product = catalog.product_by_id(str(product_id))
        if product is not None:
            products.append(product)
    return tuple(products)


def _parse_user_choice(text: str, candidates: tuple[CatalogProduct, ...]) -> CatalogProduct | None:
    normalized = text.strip()
    if not normalized or not candidates:
        return None
    if normalized.isdigit():
        index = int(normalized)
        if 1 <= index <= len(candidates):
            return candidates[index - 1]
    lowered = normalized.lower()
    for product in candidates:
        if product.display_name in normalized:
            return product
        for alias in product.exclusive_aliases:
            if alias and (alias.lower() in lowered or alias in normalized):
                return product
        for manual_name in product.manual_names:
            if manual_name and manual_name in normalized:
                return product
    return None


def _looks_like_product_switch(text: str) -> bool:
    markers = (
        "换一款",
        "不是这个",
        "另一款",
        "另一台",
        "改成",
        "其实我是",
        "我的是拍立得",
        "我的是单反",
    )
    return any(marker in text for marker in markers)


def _text_hits_any(text: str, terms: tuple[str, ...] | frozenset[str]) -> bool:
    lowered = text.lower()
    for term in terms:
        if not term:
            continue
        if term.lower() in lowered or term in text:
            return True
    return False


def _matched_aliases(text: str, aliases: tuple[str, ...]) -> tuple[str, ...]:
    hits: list[str] = []
    lowered = text.lower()
    for alias in aliases:
        if not alias:
            continue
        if alias.lower() in lowered or alias in text:
            hits.append(alias)
    return tuple(hits)
