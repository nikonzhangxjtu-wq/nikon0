"""检索质量离线评估脚本。

评估当前「单路文本 RAG + 图片证据挂载」架构的检索质量，输出以下指标：

文本 RAG 指标
  - Precision@K / Recall@K / MRR / NDCG@K
  - 手册名命中率（检索结果中正确手册的占比）
  - Top-1 手册准确率

图片证据指标（图片 collection 存在时）
  - 图片证据挂载率（有 image_evidence 的 chunk 占比）
  - 图片-文本手册一致性（图片证据指向的手册与文本 chunk 手册一致的比例）
  - 图片证据独立命中率（图片召回的手册名是否正确）

延迟分解
  - embedding / dense_search / sparse_search / rank_fusion / rerank / image_retrieval 各阶段耗时

数据集构建
  - 从每本手册中自动抽取代表性查询（标题、关键句、产品名+动作词）
  - 正例：查询来源手册的所有 chunk
  - 负例：其他手册的 chunk

用法::

    python -m app.test.test_retrieval_quality
    python -m app.test.test_retrieval_quality --sample-size 200
    python -m app.test.test_retrieval_quality --no-image  # 跳过图片评估
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core.config import settings
from app.services.ingestion import ManualIngestionService, ManualChunk
from app.services.retriever import VectorRetriever, RetrievedChunk, retriever_context_filter


# ═══════════════════════════════════════════════════════════════════════════════
# 评估数据集
# ═══════════════════════════════════════════════════════════════════════════════

# 用于从 chunk 文本中提取可作 query 的模式
_HEADING_RE = re.compile(r"(?m)^#+\s*(.+?)(?:\s*#+\s*)?$")
_STEP_HEADING_RE = re.compile(r"(?:步骤\s*\d+|Step\s*\d+)[：:.\s]*(.+)", re.IGNORECASE)
_WARNING_HEADING_RE = re.compile(r"(?:警告|注意|Warning|Caution|Danger)[：:.\s]*(.+)", re.IGNORECASE)
_ACTION_NOUN_RE = re.compile(
    r"(如何|怎么|怎样)(安装|拆卸|清洁|清洗|更换|设置|操作|使用|启动|关闭|充电|调节|维护|保养|排除|解决|处理)"
)
_EN_ACTION_RE = re.compile(
    r"(how to|how do I|troubleshoot(?:ing)?)\s+([a-z]{3,}(?:\s+[a-z]{3,}){0,5})",
    re.IGNORECASE,
)
_PRODUCT_NAME_RE = re.compile(
    r"(空调|冰箱|洗衣机|洗碗机|烤箱|微波炉|电钻|吹风机|水泵|发电机|键盘|鼠标|耳机|相机|电视|净化器|"
    r"摩托艇|滑雪|健身|温控|电动|牙刷|咖啡|传真|电话|割草|烧烤|安全摄像头|主板|船|"
    r"air\s*fryer|coffee|microwave|camera|vacuum|washer|grill|boat|television|"
    r"earphone|toothbrush|ereader|fax|lawn|mower|snowmobile|security)",
    re.IGNORECASE,
)
_FAULT_RE = re.compile(r"(不转|不工作|不启动|无法|坏了|故障|闪(?:烁|红灯|黄灯|蓝灯)|错误(?:码|代码)?|E\d{1,4})")
_MAINTENANCE_RE = re.compile(r"(清洁|清洗|更换|保养|维护|滤网|电池|充电)")


@dataclass
class EvalQuery:
    """一条评估查询。"""

    query_id: str
    query_text: str
    source_manual: str
    source_chunk_id: str
    query_type: str  # heading / step / fault / maintenance / product_action / general
    language: str  # zh / en


@dataclass
class RetrievalEvalDataset:
    """评估数据集：queries + 每 query 的 ground truth manual。"""

    queries: list[EvalQuery]
    all_manuals: list[str]
    chunks_by_manual: dict[str, list[str]]  # manual_name -> [chunk_id, ...]

    def __len__(self) -> int:
        return len(self.queries)


def _query_type(text: str) -> str:
    if _FAULT_RE.search(text):
        return "fault"
    if _MAINTENANCE_RE.search(text):
        return "maintenance"
    if _STEP_HEADING_RE.search(text):
        return "step"
    if _HEADING_RE.search(text):
        return "heading"
    if _ACTION_NOUN_RE.search(text) or _EN_ACTION_RE.search(text):
        return "product_action"
    return "general"


def _extract_heading(text: str) -> str | None:
    m = _HEADING_RE.search(text)
    if not m:
        return None
    heading = m.group(1).strip()
    # 过滤太短或纯标签
    if len(heading) < 3:
        return None
    skip_prefixes = (
        "user manual", "table of contents", "目录", "使用说明书", "产品简介",
        "note", "important", "caution", "warning", "danger", "注意", "警告", "小心",
        "tip", "提示", "remark", "备注",
    )
    if heading.lower().startswith(skip_prefixes):
        return None
    return heading


def _extract_fault_query(text: str, manual_name: str) -> str | None:
    """从 chunk 中抽故障相关 query。"""
    m = _FAULT_RE.search(text)
    if not m:
        return None
    # 取故障关键词前后的上下文
    start = max(0, m.start() - 10)
    end = min(len(text), m.end() + 40)
    snippet = text[start:end].strip().replace("\n", " ")
    # 加入产品名提示
    product = _product_hint(manual_name)
    if product and product not in snippet:
        return f"{product}{snippet}"
    return snippet


def _extract_maintenance_query(text: str, manual_name: str) -> str | None:
    m = _MAINTENANCE_RE.search(text)
    if not m:
        return None
    start = max(0, m.start() - 5)
    end = min(len(text), m.end() + 30)
    snippet = text[start:end].strip().replace("\n", " ")
    product = _product_hint(manual_name)
    if product and product not in snippet:
        return f"{product}{snippet}"
    return snippet


def _product_hint(manual_name: str) -> str:
    """从手册名提取产品中文名。"""
    mapping = {
        "空调手册": "空调", "冰箱手册": "冰箱", "烤箱手册": "烤箱", "水泵手册": "水泵",
        "电钻手册": "电钻", "吹风机手册": "吹风机", "发电机手册": "发电机",
        "洗碗机手册": "洗碗机", "摩托艇手册": "摩托艇", "空气净化器手册": "空气净化器",
        "VR头显手册": "VR头显", "功能键盘手册": "键盘", "蓝牙激光鼠标手册": "鼠标",
        "健身单车手册": "健身单车", "健身追踪器手册": "健身追踪器",
        "儿童电动摩托车手册": "儿童电动摩托车", "可编程温控器手册": "温控器",
        "人体工学椅手册": "人体工学椅",
        "Airfryer": "Airfryer", "Boat": "Boat", "Coffee_Machine": "Coffee Machine",
        "Cordless_Landline": "Cordless Landline", "DSLR_Camera": "DSLR Camera",
        "Earphones": "Earphones", "Electric_Toothbrush": "Electric Toothbrush",
        "eReader": "eReader", "Fax_Machine": "Fax Machine", "Grill": "Grill",
        "Lawn_Mower": "Lawn Mower", "Microwave_OTR": "Microwave OTR",
        "Motherboard": "Motherboard", "PressureCooker_Airfryer": "Pressure Cooker Airfryer",
        "Security_Camera": "Security Camera", "Snowmobile": "Snowmobile",
        "Television": "Television", "Vacuum_Cleaner": "Vacuum Cleaner",
        "Washing_Machine": "Washing Machine", "WaveRunner_Jetski": "WaveRunner Jetski",
    }
    return mapping.get(manual_name, "")


def build_eval_dataset(chunks: list[ManualChunk], *, max_per_manual: int = 8) -> RetrievalEvalDataset:
    """从手册 chunk 中自动构建评估数据集。

    策略（按优先级）：
    1. 从含标题的 chunk 中提取标题作为 query
    2. 从含故障描述的 chunk 中拼接故障 query
    3. 从含维护关键词的 chunk 中拼接维护 query
    4. 从每个手册均匀采样 general query

    每条 query 的 ground truth 是其来源手册的全部 chunk。
    """
    chunks_by_manual: dict[str, list[ManualChunk]] = defaultdict(list)
    for c in chunks:
        chunks_by_manual[c.manual_name].append(c)

    all_manuals = sorted(chunks_by_manual.keys())
    all_chunk_ids: dict[str, list[str]] = {
        m: [c.chunk_id for c in clist] for m, clist in chunks_by_manual.items()
    }

    queries: list[EvalQuery] = []
    qid = 0

    for manual_name, manual_chunks in chunks_by_manual.items():
        manual_queries: list[EvalQuery] = []
        seen_texts: set[str] = set()

        # 第 1 轮：从标题中提取
        for c in manual_chunks:
            heading = _extract_heading(c.text)
            if heading and heading not in seen_texts and len(manual_queries) < max_per_manual:
                seen_texts.add(heading)
                manual_queries.append(EvalQuery(
                    query_id=f"q{qid:04d}",
                    query_text=heading,
                    source_manual=manual_name,
                    source_chunk_id=c.chunk_id,
                    query_type="heading",
                    language="zh" if _is_chinese(heading) else "en",
                ))
                qid += 1

        # 第 2 轮：从故障描述中提取
        for c in manual_chunks:
            if len(manual_queries) >= max_per_manual:
                break
            q = _extract_fault_query(c.text, manual_name)
            if q and q not in seen_texts:
                seen_texts.add(q)
                manual_queries.append(EvalQuery(
                    query_id=f"q{qid:04d}",
                    query_text=q,
                    source_manual=manual_name,
                    source_chunk_id=c.chunk_id,
                    query_type="fault",
                    language="zh" if _is_chinese(q) else "en",
                ))
                qid += 1

        # 第 3 轮：从维护关键词中提取
        for c in manual_chunks:
            if len(manual_queries) >= max_per_manual:
                break
            q = _extract_maintenance_query(c.text, manual_name)
            if q and q not in seen_texts:
                seen_texts.add(q)
                manual_queries.append(EvalQuery(
                    query_id=f"q{qid:04d}",
                    query_text=q,
                    source_manual=manual_name,
                    source_chunk_id=c.chunk_id,
                    query_type="maintenance",
                    language="zh" if _is_chinese(q) else "en",
                ))
                qid += 1

        # 第 4 轮：用 chunk 首句补足
        for c in manual_chunks:
            if len(manual_queries) >= max_per_manual:
                break
            first_sentence = _first_sentence(c.text)
            if not first_sentence or len(first_sentence) < 6:
                continue
            product = _product_hint(manual_name)
            q = f"{product} {first_sentence}" if product else first_sentence
            if q not in seen_texts:
                seen_texts.add(q)
                manual_queries.append(EvalQuery(
                    query_id=f"q{qid:04d}",
                    query_text=q,
                    source_manual=manual_name,
                    source_chunk_id=c.chunk_id,
                    query_type="general",
                    language="zh" if _is_chinese(q) else "en",
                ))
                qid += 1

        # 如果手册 chunk 太少，至少保证 1 条
        if not manual_queries and manual_chunks:
            c = manual_chunks[0]
            product = _product_hint(manual_name)
            q = f"{product} 使用说明" if product else c.text[:80]
            manual_queries.append(EvalQuery(
                query_id=f"q{qid:04d}",
                query_text=q,
                source_manual=manual_name,
                source_chunk_id=c.chunk_id,
                query_type="general",
                language="zh",
            ))
            qid += 1

        queries.extend(manual_queries)

    return RetrievalEvalDataset(
        queries=queries,
        all_manuals=all_manuals,
        chunks_by_manual=all_chunk_ids,
    )


def _is_chinese(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", text))


def _first_sentence(text: str) -> str:
    """取 chunk 的第一个有意义的句子，跳过 IMG 标签和空行。"""
    cleaned = re.sub(r"<IMG[^>]*>", "", text)
    cleaned = re.sub(r"<PIC>", "", cleaned)
    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            line = line.lstrip("#").strip()
        if len(line) >= 4:
            return line[:120]
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 评估指标计算
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TextRagMetrics:
    """文本 RAG 指标汇总。"""

    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    manual_top1_accuracy: float = 0.0
    manual_top3_accuracy: float = 0.0
    chunk_hit_rate: float = 0.0  # source_chunk_id 在 top-K 中的比例


@dataclass
class ImageEvidenceMetrics:
    """图片证据质量指标。"""

    image_enabled: bool = False
    total_image_collection_size: int = 0
    chunk_with_evidence_rate: float = 0.0  # 有图片证据的 chunk 占比
    avg_images_per_chunk: float = 0.0
    manual_consistency_rate: float = 0.0  # 图片证据手册名与文本 chunk 一致的占比
    image_only_manual_accuracy: float = 0.0  # 仅通过图片证据判断的正确手册率
    entity_match_rate: float = 0.0  # OCR/实体匹配命中率
    semantic_match_rate: float = 0.0  # 语义向量匹配命中率
    image_vector_match_rate: float = 0.0  # 图片向量匹配命中率（仅当有用户图片输入时）


@dataclass
class LatencyMetrics:
    """延迟分解。"""

    embedding_ms: list[float] = field(default_factory=list)
    dense_search_ms: list[float] = field(default_factory=list)
    sparse_search_ms: list[float] = field(default_factory=list)
    rerank_ms: list[float] = field(default_factory=list)
    image_retrieval_ms: list[float] = field(default_factory=list)
    total_ms: list[float] = field(default_factory=list)

    @property
    def embedding_p50(self) -> float: return _percentile(self.embedding_ms, 0.5)
    @property
    def embedding_p95(self) -> float: return _percentile(self.embedding_ms, 0.95)
    @property
    def dense_p50(self) -> float: return _percentile(self.dense_search_ms, 0.5)
    @property
    def dense_p95(self) -> float: return _percentile(self.dense_search_ms, 0.95)
    @property
    def sparse_p50(self) -> float: return _percentile(self.sparse_search_ms, 0.5) if self.sparse_search_ms else 0
    @property
    def sparse_p95(self) -> float: return _percentile(self.sparse_search_ms, 0.95) if self.sparse_search_ms else 0
    @property
    def rerank_p50(self) -> float: return _percentile(self.rerank_ms, 0.5)
    @property
    def rerank_p95(self) -> float: return _percentile(self.rerank_ms, 0.95)
    @property
    def image_p50(self) -> float: return _percentile(self.image_retrieval_ms, 0.5) if self.image_retrieval_ms else 0
    @property
    def image_p95(self) -> float: return _percentile(self.image_retrieval_ms, 0.95) if self.image_retrieval_ms else 0
    @property
    def total_p50(self) -> float: return _percentile(self.total_ms, 0.5)
    @property
    def total_p95(self) -> float: return _percentile(self.total_ms, 0.95)
    @property
    def total_mean(self) -> float: return sum(self.total_ms) / len(self.total_ms) if self.total_ms else 0


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    n = len(xs)
    if n == 1:
        return xs[0]
    rank = p * (n - 1)
    lo, hi = int(rank), min(int(rank) + 1, n - 1)
    frac = rank - lo
    return xs[lo] + frac * (xs[hi] - xs[lo])


def _dcg(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain。"""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2**rel - 1) / (__import__("math").log2(i + 2))
    return dcg


def _ndcg(retrieved_relevances: list[float], ideal_relevances: list[float], k: int) -> float:
    idcg = _dcg(sorted(ideal_relevances, reverse=True), k)
    if idcg == 0:
        return 0.0
    return _dcg(retrieved_relevances, k) / idcg


def _rolling_mrr(results: list[dict], window: int = 20) -> float:
    """最近 N 条结果的滚动 MRR，用于进度显示。"""
    window_results = results[-window:]
    if not window_results:
        return 0.0
    return sum(r["mrr"] for r in window_results) / len(window_results)


# ═══════════════════════════════════════════════════════════════════════════════
# 评估器
# ═══════════════════════════════════════════════════════════════════════════════

class RetrievalEvaluator:
    """检索质量评估器。"""

    def __init__(self) -> None:
        self.retriever = VectorRetriever()
        self.image_enabled = self.retriever.image_retrieval_enabled

    # ── 文本 RAG 评估 ──────────────────────────────────────────────────────

    def evaluate_text_rag(
        self,
        dataset: RetrievalEvalDataset,
        *,
        top_k: int = 6,
        k_values: tuple[int, ...] = (1, 3, 5, 10),
    ) -> tuple[TextRagMetrics, list[dict], LatencyMetrics]:
        """逐 query 检索并计算文本 RAG 指标。

        Returns:
            metrics: 汇总指标
            per_query_details: 每条 query 的明细结果
            latency: 延迟分解
        """
        results: list[dict] = []
        latency = LatencyMetrics()

        max_k = max(k_values)
        actual_k = max(top_k, max_k)

        total_queries = len(dataset.queries)
        t_start_all = time.perf_counter()
        print(f"  [text_rag] 进度: 0/{total_queries} (0.0%)", end="", flush=True)

        for idx, eq in enumerate(dataset.queries):
            t0 = time.perf_counter()

            # 1. embedding
            t_emb = time.perf_counter()
            # 通过 retriever 内部路径做 query embedding
            embedder = (
                self.retriever._embed_zh
                if _is_chinese(eq.query_text)
                else self.retriever._embed_en
            )
            model_name = (
                settings.embed_model_zh
                if _is_chinese(eq.query_text)
                else settings.embed_model_en
            )
            query_vector = self.retriever._embed_query(embedder, model_name, eq.query_text)
            t_emb_end = time.perf_counter()

            # 2. dense search
            t_dense_start = time.perf_counter()
            dense_hits = self.retriever._search_dense(
                query_vector=query_vector, limit=actual_k
            )
            t_dense_end = time.perf_counter()

            # 3. sparse search（如果启用）
            t_sparse_start = time.perf_counter()
            sparse_hits: list[dict] = []
            if self.retriever.sparse_enabled:
                sparse_hits = self.retriever._search_sparse_text(
                    query=eq.query_text, limit=actual_k
                )
            t_sparse_end = time.perf_counter()

            # 4. fusion + rerank
            t_rerank_start = time.perf_counter()
            if self.retriever.sparse_enabled:
                fused = self.retriever._fuse_hits_by_rank(
                    dense_hits=dense_hits, sparse_hits=sparse_hits, top_k=actual_k
                )
            else:
                fused = dense_hits[:actual_k]
            fused = self.retriever._rerank(fused, eq.query_text, top_k=actual_k)
            t_rerank_end = time.perf_counter()

            # 5. 图片证据（如果启用且传入 manual_name 模拟）
            t_image_start = time.perf_counter()
            retrieved_chunks = self._hits_to_chunks(fused)
            if self.image_enabled:
                # 用正确的 manual_name 测试图片检索
                retrieved_chunks = self.retriever._maybe_add_image_evidence(
                    chunks=retrieved_chunks,
                    query=eq.query_text,
                    top_k=top_k,
                    manual_name=eq.source_manual,
                    image_inputs=[],
                )
            t_image_end = time.perf_counter()

            t_total = time.perf_counter()

            latency.embedding_ms.append((t_emb_end - t_emb) * 1000)
            latency.dense_search_ms.append((t_dense_end - t_dense_start) * 1000)
            if self.retriever.sparse_enabled:
                latency.sparse_search_ms.append((t_sparse_end - t_sparse_start) * 1000)
            latency.rerank_ms.append((t_rerank_end - t_rerank_start) * 1000)
            if self.image_enabled:
                latency.image_retrieval_ms.append((t_image_end - t_image_start) * 1000)
            latency.total_ms.append((t_total - t0) * 1000)

            # 计算本 query 的各项指标
            result = self._compute_query_result(
                eq=eq,
                chunks=retrieved_chunks,
                dataset=dataset,
                k_values=k_values,
            )
            results.append(result)

            # 每 10 条打印一次进度
            if (idx + 1) % 10 == 0 or idx == total_queries - 1:
                elapsed = time.perf_counter() - t_start_all
                avg_ms = (elapsed / (idx + 1)) * 1000
                eta_sec = avg_ms * (total_queries - idx - 1) / 1000
                print(
                    f"\r  [text_rag] 进度: {idx + 1}/{total_queries} "
                    f"({(idx + 1) / total_queries * 100:.1f}%) | "
                    f"当前 MRR={_rolling_mrr(results, 20):.3f} | "
                    f"平均 {avg_ms:.0f}ms/条 | 预计剩余 {eta_sec:.0f}s",
                    end="",
                    flush=True,
                )

        print()  # 换行，结束进度行

        metrics = self._aggregate_text_metrics(results, k_values)
        return metrics, results, latency

    def _hits_to_chunks(self, fused_hits: list[dict]) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        for hit in fused_hits:
            entity = hit.get("entity", {}) or {}
            chunks.append(RetrievedChunk(
                chunk_id=str(entity.get("chunk_id", "")),
                text=str(entity.get("text", "")),
                score=float(hit.get("score", hit.get("distance", 0.0))),
                manual_name=str(entity.get("manual_name", "")),
                image_ids=self.retriever._parse_image_ids(entity.get("image_ids")),
            ))
        return chunks

    def _compute_query_result(
        self,
        eq: EvalQuery,
        chunks: list[RetrievedChunk],
        dataset: RetrievalEvalDataset,
        k_values: tuple[int, ...],
    ) -> dict:
        relevant_manuals = {eq.source_manual}
        # 有些手册可能是另一个的子集（如 PressureCooker_Airfryer vs Airfryer），
        # 这种情况只把精确匹配视为相关
        retrieved_manuals = [c.manual_name for c in chunks]
        retrieved_scores = [c.score for c in chunks]

        # 相关性：same manual
        relevances = [1.0 if c.manual_name == eq.source_manual else 0.0 for c in chunks]

        # Precision@K, Recall@K
        precisions: dict[int, float] = {}
        recalls: dict[int, float] = {}
        total_relevant = len(dataset.chunks_by_manual.get(eq.source_manual, []))
        for k in k_values:
            hit_count = sum(1 for r in relevances[:k] if r > 0)
            precisions[k] = hit_count / k if k > 0 else 0.0
            recalls[k] = hit_count / total_relevant if total_relevant > 0 else 0.0

        # MRR
        mrr = 0.0
        for rank, rel in enumerate(relevances, start=1):
            if rel > 0:
                mrr = 1.0 / rank
                break

        # NDCG
        ndcgs: dict[int, float] = {}
        ideal_relevances = [1.0] * min(total_relevant, max(k_values))
        for k in k_values:
            ndcgs[k] = _ndcg(relevances, ideal_relevances, k)

        # Manual accuracy
        manual_top1 = 1.0 if (retrieved_manuals[:1] and retrieved_manuals[0] == eq.source_manual) else 0.0
        manual_top3 = 1.0 if eq.source_manual in retrieved_manuals[:3] else 0.0

        # chunk hit
        chunk_hit = 1.0 if any(c.chunk_id == eq.source_chunk_id for c in chunks[:5]) else 0.0

        # Image evidence
        image_evidence_count = sum(len(c.image_evidence) for c in chunks)
        image_manuals_consistent = 0
        image_manuals_total = 0
        for c in chunks:
            for ev in c.image_evidence:
                image_manuals_total += 1
                # 图片证据的手册通过 parent_chunk_ids 反查？不直接有。用间接信号。
                # 这里用 match_reason 判断来源
                if any(
                    m in ev.match_reason
                    for m in ["OCR/实体", "文本问题-图片语义", "用户上传图片"]
                ):
                    image_manuals_consistent += 1  # 简化：有图片证据就视为补充

        return {
            "query_id": eq.query_id,
            "query_text": eq.query_text,
            "source_manual": eq.source_manual,
            "query_type": eq.query_type,
            "language": eq.language,
            "precision_at_k": precisions,
            "recall_at_k": recalls,
            "mrr": mrr,
            "ndcg_at_k": ndcgs,
            "manual_top1": manual_top1,
            "manual_top3": manual_top3,
            "chunk_hit": chunk_hit,
            "retrieved_chunks": len(chunks),
            "retrieved_manuals": retrieved_manuals[:5],
            "retrieved_scores": retrieved_scores[:5],
            "image_evidence_count": image_evidence_count,
        }

    def _aggregate_text_metrics(
        self, results: list[dict], k_values: tuple[int, ...]
    ) -> TextRagMetrics:
        n = len(results)
        if n == 0:
            return TextRagMetrics()

        precision: dict[int, float] = {}
        recall: dict[int, float] = {}
        ndcg: dict[int, float] = {}
        for k in k_values:
            precision[k] = sum(r["precision_at_k"].get(k, 0.0) for r in results) / n
            recall[k] = sum(r["recall_at_k"].get(k, 0.0) for r in results) / n
            ndcg[k] = sum(r["ndcg_at_k"].get(k, 0.0) for r in results) / n

        mrr = sum(r["mrr"] for r in results) / n
        manual_top1 = sum(r["manual_top1"] for r in results) / n
        manual_top3 = sum(r["manual_top3"] for r in results) / n
        chunk_hit = sum(r["chunk_hit"] for r in results) / n

        return TextRagMetrics(
            precision_at_k=precision,
            recall_at_k=recall,
            mrr=mrr,
            ndcg_at_k=ndcg,
            manual_top1_accuracy=manual_top1,
            manual_top3_accuracy=manual_top3,
            chunk_hit_rate=chunk_hit,
        )

    # ── 图片证据评估 ──────────────────────────────────────────────────────

    def evaluate_image_evidence(self, details: list[dict]) -> ImageEvidenceMetrics:
        """基于文本 RAG 结果，评估图片证据挂载质量。"""
        if not self.image_enabled:
            return ImageEvidenceMetrics(image_enabled=False)

        n = len(details)
        if n == 0:
            return ImageEvidenceMetrics(image_enabled=True)

        total_image_evidence = sum(d.get("image_evidence_count", 0) for d in details)
        total_chunks = sum(d.get("retrieved_chunks", 0) for d in details)

        # 每个 query 的图片挂载统计
        queries_with_evidence = sum(
            1 for d in details if d.get("image_evidence_count", 0) > 0
        )

        return ImageEvidenceMetrics(
            image_enabled=True,
            chunk_with_evidence_rate=queries_with_evidence / n if n > 0 else 0.0,
            avg_images_per_chunk=total_image_evidence / total_chunks if total_chunks > 0 else 0.0,
            manual_consistency_rate=0.0,  # 需要更细粒度的图片 manual 信息
        )

    # ── 独立图片检索评估（仅用图片 collection 检索，不依赖文本 RAG）─────

    def evaluate_image_standalone(
        self, dataset: RetrievalEvalDataset, *, top_k: int = 5
    ) -> dict:
        """独立评估图片 collection 的检索能力。

        对每条 query，直接用文本语义检索图片 collection，
        检查返回图片的 manual_name 是否与 query 来源手册一致。
        """
        if not self.image_enabled:
            return {"enabled": False, "reason": "图片 collection 不存在或未启用"}

        from app.services.multimodal.embeddings import JinaMultimodalEmbeddingClient

        jina = JinaMultimodalEmbeddingClient()
        manual_hits = 0
        total = 0
        per_query: list[dict] = []

        for eq in dataset.queries:
            vec = jina.embed_text(eq.query_text)
            if not vec:
                continue
            total += 1

            try:
                results = self.retriever.client.search(
                    collection_name=settings.multimodal_image_collection,
                    anns_field="semantic_vector",
                    data=[vec],
                    limit=top_k,
                    output_fields=["image_id", "manual_name", "context_intent", "image_type"],
                )
            except Exception as exc:
                print(f"  [WARN] 图片独立检索失败: {exc}")
                continue

            hits = results[0] if results and results[0] else []
            hit_manuals = [
                str((h.get("entity", {}) or {}).get("manual_name", ""))
                for h in hits
            ]
            correct = eq.source_manual in hit_manuals
            if correct:
                manual_hits += 1

            per_query.append({
                "query_id": eq.query_id,
                "query_text": eq.query_text[:80],
                "source_manual": eq.source_manual,
                "hit_manuals": hit_manuals[:3],
                "correct": correct,
                "hit_count": len(hits),
            })

        accuracy = manual_hits / total if total > 0 else 0.0
        return {
            "enabled": True,
            "total_queries": total,
            "manual_accuracy": accuracy,
            "correct_count": manual_hits,
            "per_query": per_query,
        }

    # ── 延迟专项测试 ──────────────────────────────────────────────────────

    def benchmark_latency(
        self, queries: list[str], *, warmup: int = 3, iterations: int = 20
    ) -> LatencyMetrics:
        """对固定 query 集做多轮延迟基准测试。"""
        latency = LatencyMetrics()

        # warmup
        for q in queries[:warmup]:
            self.retriever.retrieve(q, top_k=6)

        for _ in range(iterations):
            for q in queries[: min(len(queries), 10)]:
                t0 = time.perf_counter()
                chunks = self.retriever.retrieve(q, top_k=6)
                latency.total_ms.append((time.perf_counter() - t0) * 1000)

        return latency


# ═══════════════════════════════════════════════════════════════════════════════
# 报告输出
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(
    dataset: RetrievalEvalDataset,
    text_metrics: TextRagMetrics,
    text_details: list[dict],
    image_metrics: ImageEvidenceMetrics,
    image_standalone: dict,
    latency: LatencyMetrics,
    *,
    sample_size: int = 0,
) -> None:
    """打印人类可读的评估报告。"""
    sep = "=" * 72
    sub = "-" * 56

    print(f"\n{sep}")
    print("  检索质量评估报告")
    print(f"{sep}")

    # ── 数据集概况 ──
    print(f"\n📋 数据集")
    print(sub)
    print(f"  总 query 数:            {len(dataset.queries)}")
    if sample_size and sample_size < len(dataset.queries):
        print(f"  (从全部 query 中采样 {sample_size} 条)")
    print(f"  覆盖手册数:             {len(dataset.all_manuals)}")
    qt_counts: dict[str, int] = defaultdict(int)
    for eq in dataset.queries:
        qt_counts[eq.query_type] += 1
    print(f"  Query 类型分布:         {dict(qt_counts)}")
    lang_counts: dict[str, int] = defaultdict(int)
    for eq in dataset.queries:
        lang_counts[eq.language] += 1
    print(f"  语言分布:               {dict(lang_counts)}")
    # 每手册 query 数
    qpm: dict[str, int] = defaultdict(int)
    for eq in dataset.queries:
        qpm[eq.source_manual] += 1
    print(f"  每手册 query 数:         min={min(qpm.values())} max={max(qpm.values())} avg={sum(qpm.values())/len(qpm):.1f}")

    # ── 文本 RAG 指标 ──
    print(f"\n📖 文本 RAG 检索质量")
    print(sub)
    print(f"  {'K':<6} {'Precision':<12} {'Recall':<12} {'NDCG@K':<12}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12}")
    for k in sorted(text_metrics.precision_at_k.keys()):
        print(
            f"  @{k:<5} {text_metrics.precision_at_k[k]:.4f}       "
            f"{text_metrics.recall_at_k[k]:.4f}       "
            f"{text_metrics.ndcg_at_k[k]:.4f}"
        )
    print(f"\n  MRR:                    {text_metrics.mrr:.4f}")
    print(f"  Top-1 手册准确率:       {text_metrics.manual_top1_accuracy:.4f}")
    print(f"  Top-3 手册准确率:       {text_metrics.manual_top3_accuracy:.4f}")
    print(f"  Chunk 命中率 (top-5):   {text_metrics.chunk_hit_rate:.4f}")

    # 按 query 类型分组
    print(f"\n  📊 按 Query 类型分组:")
    by_type: dict[str, list[dict]] = defaultdict(list)
    for d in text_details:
        by_type[d["query_type"]].append(d)
    for qtype in ["heading", "fault", "maintenance", "product_action", "general"]:
        group = by_type.get(qtype, [])
        if not group:
            continue
        n_g = len(group)
        mrr_g = sum(d["mrr"] for d in group) / n_g
        top1_g = sum(d["manual_top1"] for d in group) / n_g
        p5_g = sum(d["precision_at_k"].get(5, 0) for d in group) / n_g
        print(
            f"    {qtype:<18} n={n_g:<4} MRR={mrr_g:.4f}  "
            f"Top1={top1_g:.4f}  P@5={p5_g:.4f}"
        )

    # 按语言分组
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for d in text_details:
        by_lang[d["language"]].append(d)
    print(f"\n  📊 按语言分组:")
    for lang in ["zh", "en"]:
        group = by_lang.get(lang, [])
        if not group:
            continue
        n_g = len(group)
        top1_g = sum(d["manual_top1"] for d in group) / n_g
        mrr_g = sum(d["mrr"] for d in group) / n_g
        print(f"    {lang}: n={n_g}  Top1={top1_g:.4f}  MRR={mrr_g:.4f}")

    # 按手册分组（Top-1 准确率最低的 10 本）
    print(f"\n  📊 手册级 Top-1 准确率 (最低 10 本，需关注):")
    by_manual: dict[str, list[dict]] = defaultdict(list)
    for d in text_details:
        by_manual[d["source_manual"]].append(d)
    manual_stats = []
    for m, group in by_manual.items():
        n_m = len(group)
        top1 = sum(d["manual_top1"] for d in group) / n_m
        mrr_m = sum(d["mrr"] for d in group) / n_m
        manual_stats.append((m, n_m, top1, mrr_m))
    manual_stats.sort(key=lambda x: x[2])  # sort by top1 ascending
    print(f"    {'手册':<30} {'n':>4} {'Top1':>8} {'MRR':>8}")
    print(f"    {'-'*30} {'-'*4} {'-'*8} {'-'*8}")
    for m, n_m, top1, mrr_m in manual_stats[:10]:
        flag = " ⚠" if top1 < 0.5 else ""
        print(f"    {m:<30} {n_m:>4} {top1:>7.4f} {mrr_m:>7.4f}{flag}")
    # 最好的 5 本
    print(f"\n    最好的 5 本:")
    for m, n_m, top1, mrr_m in manual_stats[-5:]:
        print(f"    {m:<30} {n_m:>4} {top1:>7.4f} {mrr_m:>7.4f}")

    # ── 分数校准分析 ──
    print(f"\n  📊 分数校准（高分是否意味着相关？）:")
    all_scores_relevant: list[float] = []
    all_scores_irrelevant: list[float] = []
    for d in text_details:
        manuals = d["retrieved_manuals"]
        scores = d.get("retrieved_scores", [])
        for i, (m, s) in enumerate(zip(manuals, scores)):
            if m == d["source_manual"]:
                all_scores_relevant.append(s)
            else:
                all_scores_irrelevant.append(s)
    if all_scores_relevant and all_scores_irrelevant:
        import statistics
        print(f"    相关 chunk 平均分数:   {statistics.mean(all_scores_relevant):.4f}")
        print(f"    无关 chunk 平均分数:   {statistics.mean(all_scores_irrelevant):.4f}")
        print(f"    分数分离度 (相关-无关): {statistics.mean(all_scores_relevant) - statistics.mean(all_scores_irrelevant):.4f}")
        print(f"    (分离度越大，reranker 区分度越好)")
    else:
        print(f"    (分数数据不足，无法分析)")

    # ── 错误分析 ──
    print(f"\n  🔍 错误分析（Top-1 手册错误，最多展示 15 条）:")
    errors = [d for d in text_details if d["manual_top1"] == 0]
    for d in errors[:15]:
        predicted = d["retrieved_manuals"][0] if d["retrieved_manuals"] else "(none)"
        scores_str = ", ".join(f"{s:.3f}" for s in d.get("retrieved_scores", [])[:3])
        print(
            f"    [{d['query_type']:<12}] Q: {d['query_text'][:55]:<55} "
            f"期望={d['source_manual'][:18]:<18} 实际={predicted:<18} scores=[{scores_str}]"
        )
    if len(errors) > 15:
        print(f"    ... 共 {len(errors)} 条错误")

    # ── 图片证据指标 ──
    print(f"\n🖼️  图片证据挂载质量")
    print(sub)
    if image_metrics.image_enabled:
        print(f"  图片 collection 已启用")
        print(f"  Query 含图片证据比例:   {image_metrics.chunk_with_evidence_rate:.4f}")
        print(f"  平均每 chunk 图片数:    {image_metrics.avg_images_per_chunk:.2f}")
    else:
        print(f"  ⚠ 图片 collection 未启用或不存在，跳过图片证据评估")

    # 独立图片检索
    print(f"\n  独立图片检索（仅 semantic_vector 检索图片 collection）:")
    print(sub)
    if image_standalone.get("enabled"):
        acc = image_standalone.get("manual_accuracy", 0)
        corr = image_standalone.get("correct_count", 0)
        total = image_standalone.get("total_queries", 0)
        print(f"  图片检索手册准确率:     {acc:.4f} ({corr}/{total})")
        # 展示几条图片检索的 case
        per_q = image_standalone.get("per_query", [])
        errors_img = [q for q in per_q if not q.get("correct")]
        if errors_img:
            print(f"  图片检索错误示例 (最多 5):")
            for q in errors_img[:5]:
                hit = q.get("hit_manuals", [])
                print(f"    Q: {q['query_text'][:50]} → 期望={q['source_manual']}, 命中={hit}")
    else:
        print(f"  ⚠ {image_standalone.get('reason', '未启用')}")

    # ── 延迟分析 ──
    print(f"\n⏱️  延迟分解（ms）")
    print(sub)
    if latency.total_ms:
        rows = [
            ("Query Embedding", latency.embedding_p50, latency.embedding_p95),
            ("Dense Search", latency.dense_p50, latency.dense_p95),
            ("Sparse Search", latency.sparse_p50, latency.sparse_p95),
            ("Rerank", latency.rerank_p50, latency.rerank_p95),
            ("Image Retrieval", latency.image_p50, latency.image_p95),
            ("→ Total", latency.total_p50, latency.total_p95),
        ]
        print(f"  {'阶段':<22} {'P50':>8} {'P95':>8}")
        print(f"  {'-'*22} {'-'*8} {'-'*8}")
        for name, p50, p95 in rows:
            marker = " ← 瓶颈" if p50 > 0 and name != "→ Total" and p50 / max(latency.total_p50, 1) > 0.3 else ""
            print(f"  {name:<22} {p50:>7.1f}  {p95:>7.1f}{marker}")
        print(f"\n  平均总延迟: {latency.total_mean:.1f} ms")
        if latency.total_mean > 0:
            print(f"  估算 QPS (单 query):    {1000 / latency.total_mean:.1f}")

    print(f"\n{sep}")
    print("  评估完成 — 建议关注 Top1 准确率 < 0.5 的手册，检查其切块/嵌入质量")
    print(f"{sep}\n")


def export_json(
    dataset: RetrievalEvalDataset,
    text_metrics: TextRagMetrics,
    text_details: list[dict],
    image_metrics: ImageEvidenceMetrics,
    image_standalone: dict,
    latency: LatencyMetrics,
    output_path: Path,
) -> None:
    """导出完整 JSON 结果供后续分析。"""
    from dataclasses import asdict

    report = {
        "dataset": {
            "total_queries": len(dataset.queries),
            "total_manuals": len(dataset.all_manuals),
            "manuals": dataset.all_manuals,
        },
        "text_rag": asdict(text_metrics),
        "image_evidence": asdict(image_metrics),
        "image_standalone": image_standalone,
        "latency": {
            "total_mean_ms": latency.total_mean,
            "total_p50_ms": latency.total_p50,
            "total_p95_ms": latency.total_p95,
            "embedding_p50_ms": latency.embedding_p50,
            "dense_p50_ms": latency.dense_p50,
            "sparse_p50_ms": latency.sparse_p50,
            "rerank_p50_ms": latency.rerank_p50,
            "image_p50_ms": latency.image_p50,
        },
        "per_query": [
            {
                "query_id": d["query_id"],
                "query_text": d["query_text"],
                "query_type": d["query_type"],
                "source_manual": d["source_manual"],
                "mrr": d["mrr"],
                "manual_top1": d["manual_top1"],
                "precision_at_5": d["precision_at_k"].get(5, 0),
                "retrieved_manuals": d["retrieved_manuals"],
                "retrieved_scores": d["retrieved_scores"],
                "image_evidence_count": d["image_evidence_count"],
            }
            for d in text_details
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[export] 完整结果已导出至: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="检索质量离线评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m app.test.test_retrieval_quality
  python -m app.test.test_retrieval_quality --sample-size 100  # 快速采样评估
  python -m app.test.test_retrieval_quality --no-image          # 跳过图片评估
  python -m app.test.test_retrieval_quality --benchmark-only    # 仅做延迟基准
        """,
    )
    parser.add_argument("--sample-size", type=int, default=0,
                        help="采样 query 数量，0 表示全量")
    parser.add_argument("--no-image", action="store_true",
                        help="跳过所有图片相关评估")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="仅运行延迟基准测试")
    parser.add_argument("--export", type=Path, default=None,
                        help="导出 JSON 结果路径")
    parser.add_argument("--top-k", type=int, default=6,
                        help="检索 top-K（默认 6）")
    parser.add_argument("--max-per-manual", type=int, default=8,
                        help="每本手册最多抽取几条 query（默认 8）")
    args = parser.parse_args()

    print("=" * 72)
    print("  检索质量评估工具")
    print("=" * 72)
    print(f"\n[init] 解析手册 chunk ...")

    ingestion = ManualIngestionService(settings.manual_dir)
    all_chunks = ingestion.parse_and_chunk()
    print(f"[init] 共 {len(all_chunks)} 个 chunk，覆盖 {len(set(c.manual_name for c in all_chunks))} 本手册")

    print(f"[init] 构建评估数据集 ...")
    dataset = build_eval_dataset(all_chunks, max_per_manual=args.max_per_manual)
    print(f"[init] 生成 {len(dataset)} 条 query")

    if args.sample_size and args.sample_size < len(dataset.queries):
        import random
        random.seed(42)
        sampled = random.sample(dataset.queries, args.sample_size)
        dataset.queries = sampled
        print(f"[init] 采样 {args.sample_size} 条 query")

    print(f"[init] 初始化检索器 ...")
    evaluator = RetrievalEvaluator()
    print(f"[init] 图片检索: {'已启用' if evaluator.image_enabled else '未启用'}")

    if args.benchmark_only:
        print(f"\n[benchmark] 运行延迟基准测试 ...")
        queries = [eq.query_text for eq in dataset.queries[:20]]
        latency = evaluator.benchmark_latency(queries)
        empty = RetrievalEvalDataset(queries=[], all_manuals=[], chunks_by_manual={})
        print_report(
            empty, TextRagMetrics(), [], ImageEvidenceMetrics(), {}, latency
        )
        return

    k_values = (1, 3, 5, 10)

    # ── 文本 RAG 评估 ──
    print(f"\n[eval] 开始文本 RAG 评估 ({len(dataset.queries)} queries, top_k={args.top_k}) ...")
    text_metrics, text_details, latency = evaluator.evaluate_text_rag(
        dataset, top_k=args.top_k, k_values=k_values
    )

    # ── 图片证据评估 ──
    if args.no_image:
        image_metrics = ImageEvidenceMetrics(image_enabled=False)
        image_standalone = {"enabled": False, "reason": "--no-image"}
    else:
        print(f"\n[eval] 评估图片证据挂载质量 ...")
        image_metrics = evaluator.evaluate_image_evidence(text_details)

        print(f"[eval] 独立图片检索评估 ...")
        image_standalone = evaluator.evaluate_image_standalone(dataset)

    # ── 输出报告 ──
    print_report(
        dataset, text_metrics, text_details,
        image_metrics, image_standalone, latency,
        sample_size=args.sample_size,
    )

    # ── 导出 ──
    if args.export:
        export_json(
            dataset, text_metrics, text_details,
            image_metrics, image_standalone, latency,
            args.export,
        )
    else:
        # 默认导出到 eval/results 目录
        from datetime import datetime, timezone
        default_path = (
            _ROOT / "eval" / "results" /
            f"retrieval_quality_{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        export_json(
            dataset, text_metrics, text_details,
            image_metrics, image_standalone, latency,
            default_path,
        )


if __name__ == "__main__":
    main()
