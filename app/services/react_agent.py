"""ReAct Agent — 多轮迭代检索证据收集。

Agent（本地 qwen2）通过 SEARCH/FINAL_ANSWER 动作循环收集手册证据。
最终答案由 Bailian 根据合并去重后的全部证据生成，本模块不负责最终生成。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum

from app.core.config import settings
from app.services.rag_skill.query_construction import query_construction
from app.services.retriever import RetrievedChunk, VectorRetriever, retriever_context_filter


class AgentAction(str, Enum):
    SEARCH = "SEARCH"
    FINAL_ANSWER = "FINAL_ANSWER"


@dataclass
class SearchEvidence:
    iteration: int
    query: str
    chunks: list[RetrievedChunk]
    summary: str


@dataclass
class MultiSearchResult:
    all_filtered_chunks: list[RetrievedChunk]
    iterations: int
    search_queries: list[str]
    trace_log: list[str] = field(default_factory=list)


_AGENT_SYSTEM = (
    "你是检索助手，你的任务是通过多次 SEARCH 从产品手册知识库中收集信息来回答用户问题。\n"
    "You are a search assistant. Your job is to gather information from a product manual "
    "knowledge base through multiple SEARCH actions to answer the user's question.\n"
    "\n"
    "输出格式（每轮严格 3 行 / Strict 3-line format per round）：\n"
    "THOUGHT: <分析缺失信息 / analyze what info is still missing, ≤40 chars>\n"
    "ACTION: SEARCH（需继续检索 / need more info）或 FINAL_ANSWER（信息已够 / enough info）\n"
    "QUERY: <搜索关键词，中英文均可 / search keywords in English or Chinese, ≤30 chars>\n"
    "\n"
    "规则 / Rules：\n"
    "1. 第 1 轮必须 SEARCH / Round 1 MUST be SEARCH\n"
    "2. 查看 Observation 后判断是否需要更多信息 / After seeing observations, decide if more info is needed\n"
    "3. 信息不足则换角度/换关键词再次 SEARCH / If evidence is insufficient, search again with different keywords\n"
    "4. 信息足够覆盖所有子问题时输出 FINAL_ANSWER / Only output FINAL_ANSWER when all sub-questions are covered\n"
    "5. 达到最大轮数时必须 FINAL_ANSWER / Must output FINAL_ANSWER when max rounds reached\n"
    "6. 不要重复相同的查询词 / Do not repeat identical search queries\n"
)

_RE_THOUGHT = re.compile(r"THOUGHT:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_RE_ACTION = re.compile(r"ACTION:\s*(SEARCH|FINAL_ANSWER)", re.IGNORECASE)
_RE_QUERY = re.compile(r"QUERY:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

# 每条 chunk 在 observation 中的最大字符数
_CHUNK_PREVIEW_CHARS = 100
# 每轮 observation 的最大字符数
_MAX_OBSERVATION_CHARS = 1000


class ReActAgent:
    """ReAct 多轮检索器，仅收集证据，不生成最终答案。"""

    def __init__(
        self,
        retriever: VectorRetriever | None = None,
        agent_model: str | None = None,
    ) -> None:
        self._retriever = retriever or VectorRetriever()
        self._agent_model = (agent_model or settings.react_agent_model).strip()
        self._max_iterations = max(1, settings.react_max_iterations)

    def collect_evidence(self, question: str) -> MultiSearchResult:
        q = (question or "").strip()
        if not q:
            return MultiSearchResult(
                all_filtered_chunks=[],
                iterations=0,
                search_queries=[],
                trace_log=["空输入"],
            )

        evidence_list: list[SearchEvidence] = []
        all_chunks: list[RetrievedChunk] = []
        trace: list[str] = []

        for iteration in range(1, self._max_iterations + 1):
            prompt = self._build_agent_prompt(q, evidence_list, iteration)
            raw = ""
            try:
                raw = self._call_agent(prompt)
            except Exception as exc:
                trace.append(f"iter={iteration} agent_call_failed: {exc}")
                if iteration == 1:
                    chunks = self._execute_search(q, original_question=q)
                    all_chunks.extend(chunks)
                    evidence_list.append(SearchEvidence(
                        iteration=1,
                        query=q,
                        chunks=chunks,
                        summary=self._format_observation(chunks, q),
                    ))
                break

            action, content = self._parse_action(raw)
            trace.append(
                f"iter={iteration} action={action.value} query={content[:80] if content else 'N/A'}"
            )

            if action == AgentAction.SEARCH:
                search_query = content.strip() if content else q
                if any(search_query == e.query for e in evidence_list):
                    trace.append(f"iter={iteration} duplicate_query_skipped")
                    continue

                chunks = self._execute_search(search_query, original_question=q)
                all_chunks.extend(chunks)
                evidence_list.append(SearchEvidence(
                    iteration=iteration,
                    query=search_query,
                    chunks=chunks,
                    summary=self._format_observation(chunks, search_query),
                ))
                continue

            elif action == AgentAction.FINAL_ANSWER:
                trace.append(f"iter={iteration} agent_signaled_done")
                break

            else:
                trace.append(f"iter={iteration} parse_failed raw={raw[:120]}")
                if not evidence_list and not all_chunks:
                    chunks = self._execute_search(q, original_question=q)
                    all_chunks.extend(chunks)
                    evidence_list.append(SearchEvidence(
                        iteration=iteration,
                        query=q,
                        chunks=chunks,
                        summary=self._format_observation(chunks, q),
                    ))
                break

        unique = self._deduplicate_chunks(all_chunks)
        unique.sort(key=lambda c: c.score, reverse=True)

        return MultiSearchResult(
            all_filtered_chunks=unique,
            iterations=len(evidence_list),
            search_queries=[e.query for e in evidence_list],
            trace_log=trace,
        )

    def _build_agent_prompt(
        self,
        question: str,
        evidence: list[SearchEvidence],
        iteration: int,
    ) -> str:
        parts: list[str] = [_AGENT_SYSTEM]
        parts.append(f"\n【第 {iteration}/{self._max_iterations} 轮】\n")
        parts.append(f"用户问题：{question}\n")

        if evidence:
            parts.append("已有证据：")
            for e in evidence:
                parts.append(f"\n--- 第 {e.iteration} 轮检索 [{e.query}] ---")
                parts.append(e.summary)
            parts.append("")
        else:
            parts.append("尚无证据，请执行首次 SEARCH。\n")

        return "\n".join(parts)

    def _call_agent(self, prompt: str) -> str:
        if settings.bailian_api_key:
            return self._call_bailian(prompt)
        return self._call_ollama(prompt)

    def _call_bailian(self, prompt: str) -> str:
        import requests as _req

        resp = _req.post(
            f"{settings.bailian_base_url}/chat/completions",
            json={
                "model": self._agent_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 256,
            },
            headers={
                "Authorization": f"Bearer {settings.bailian_api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        return body.get("choices", [{}])[0].get("message", {}).get("content", "")

    def _call_ollama(self, prompt: str) -> str:
        import requests as _req

        resp = _req.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": self._agent_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 256},
            },
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        return body.get("message", {}).get("content", "")

    @staticmethod
    def _parse_action(raw: str) -> tuple[AgentAction, str]:
        text = (raw or "").strip()

        # 提取 ACTION
        action_match = _RE_ACTION.search(text)
        if action_match:
            action_str = action_match.group(1).upper()
            action = AgentAction.FINAL_ANSWER if action_str == "FINAL_ANSWER" else AgentAction.SEARCH
        else:
            # 启发式：只有明确表达停止意图时才 FINAL_ANSWER，否则安全默认 SEARCH
            final_signals = (
                "FINAL_ANSWER" in text.upper()
                or "足够" in text
            )
            if final_signals:
                action = AgentAction.FINAL_ANSWER
            else:
                action = AgentAction.SEARCH

        # 提取 QUERY
        query = ""
        query_match = _RE_QUERY.search(text)
        if query_match:
            query = query_match.group(1).strip()

        return action, query

    def _execute_search(
        self, query: str, original_question: str = ""
    ) -> list[RetrievedChunk]:
        q = query.strip()
        if not q:
            return []
        # 用原始问题（含完整上下文）识别手册名，Agent 搜索词仅用于语义匹配
        manual_name = query_construction(original_question or q)
        raw = self._retriever.retrieve(q, top_k=6, manual_name=manual_name or None)
        return retriever_context_filter(raw)

    @staticmethod
    def _format_observation(chunks: list[RetrievedChunk], query: str) -> str:
        if not chunks:
            return f"[检索词: \"{query}\"] 无结果。请更换检索词。\n"

        lines = [f"[检索词: \"{query}\"] 找到 {len(chunks)} 条："]
        budget = _MAX_OBSERVATION_CHARS - len(lines[0]) - 10
        for i, c in enumerate(chunks, 1):
            preview = c.text[: _CHUNK_PREVIEW_CHARS].replace("\n", " ")
            line = f"  [{i}] s={c.score:.2f} | {c.manual_name} | {preview}"
            if len(line) > budget and i > 2:
                lines.append(f"  ... 还有 {len(chunks) - i + 1} 条结果（已省略）")
                break
            lines.append(line)
            budget -= len(line)
            if budget <= 0:
                break

        return "\n".join(lines) + "\n"

    @staticmethod
    def _deduplicate_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        seen: set[str] = set()
        out: list[RetrievedChunk] = []
        for c in chunks:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                out.append(c)
        return out
