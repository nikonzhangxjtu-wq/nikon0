"""v4 session issue memory 读取。"""

from __future__ import annotations

import re

from app.services.memory.v4.types import IssueReadRequest, SessionIssueMemory


class IssueReadPlanner:
    def plan(self, *, session_id: str | None, query: str) -> IssueReadRequest:
        sid = (session_id or "").strip()
        q = query or ""
        entities = _extract_entities(q)
        if not sid:
            return IssueReadRequest(session_id="", query=q, read_mode="none", query_entities={}, reason="无 session_id")
        if entities:
            return IssueReadRequest(session_id=sid, query=q, read_mode="specific_issue", query_entities=entities, reason="命中明确实体")
        if _is_pure_howto(q):
            return IssueReadRequest(session_id=sid, query=q, read_mode="none", query_entities=entities, reason="纯手册问题")
        if re.search(r"这个|那个|刚才|还是|我试过|下一步|继续|还不行|怎么办", q):
            return IssueReadRequest(session_id=sid, query=q, read_mode="active_issue", query_entities=entities, reason="上下文延续")
        return IssueReadRequest(session_id=sid, query=q, read_mode="active_issue", query_entities=entities, reason="默认读取 active issue")


class IssueMemoryReader:
    def select_threads(self, memory: SessionIssueMemory, request: IssueReadRequest):
        if request.read_mode == "none":
            return []
        if request.read_mode == "active_issue":
            if memory.active_thread_id and memory.active_thread_id in memory.threads:
                thread = memory.threads[memory.active_thread_id]
                return [] if thread.status in {"resolved", "cancelled"} else [thread]
            return []
        if request.read_mode == "specific_issue":
            thread_ids: list[str] = []
            for kind, values in request.query_entities.items():
                for value in values:
                    thread_ids.extend(memory.entity_index.get(f"{kind}:{value}", []))
            seen = set()
            threads = []
            for tid in thread_ids:
                if tid in seen:
                    continue
                seen.add(tid)
                thread = memory.threads.get(tid)
                if thread and thread.status not in {"resolved", "cancelled"}:
                    threads.append(thread)
            return threads
        return [t for t in memory.threads.values() if t.status not in {"resolved", "cancelled"}]


def _extract_entities(text: str) -> dict[str, list[str]]:
    entities: dict[str, list[str]] = {}
    models = re.findall(r"\b[A-Z]{1,6}[-_]?\d{2,6}[A-Z0-9-]*\b", text or "")
    if models:
        entities["product_model"] = sorted(set(models))
    codes = re.findall(r"\b[A-Z]{1,3}\d{1,4}\b", text or "")
    codes = [code for code in codes if code not in set(models)]
    if codes:
        entities["fault_code"] = sorted(set(codes))
    return entities


def _is_pure_howto(text: str) -> bool:
    return bool(re.search(r"怎么|如何|步骤|清洗|安装|拆卸|使用", text or "")) and not bool(
        re.search(r"已经|试过|还是|不行|报修|退款|投诉|显示|故障|红灯|闪|无法|不能", text or "")
    )
