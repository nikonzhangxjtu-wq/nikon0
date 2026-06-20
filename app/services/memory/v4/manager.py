"""v4 session-only issue memory 总入口。"""

from __future__ import annotations

from app.core.config import settings
from app.services.memory.v3.types import TurnEvidencePacket
from app.services.memory.v4.reader import IssueMemoryReader
from app.services.memory.v4.renderer import IssueSummaryRenderer
from app.services.memory.v4.resolver import IssueThreadResolver
from app.services.memory.v4.state_change import StateChangeDetector
from app.services.memory.v4.store import InMemorySessionIssueMemoryStore, RedisSessionIssueMemoryStore
from app.services.memory.v4.types import IssueMemoryTrace, IssueReadRequest, IssueSummary
from app.services.memory.v4.updater import IssueThreadUpdater

_manager_v4: "MemoryManagerV4 | None" = None


class MemoryManagerV4:
    def __init__(
        self,
        *,
        store: object | None = None,
        detector: object | None = None,
        resolver: IssueThreadResolver | None = None,
        updater: IssueThreadUpdater | None = None,
        reader: IssueMemoryReader | None = None,
        renderer: IssueSummaryRenderer | None = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.store = store or InMemorySessionIssueMemoryStore()
        self.detector = detector or StateChangeDetector()
        self.resolver = resolver or IssueThreadResolver()
        self.updater = updater or IssueThreadUpdater()
        self.reader = reader or IssueMemoryReader()
        self.renderer = renderer or IssueSummaryRenderer()

    def read(self, request: IssueReadRequest) -> IssueSummary:
        if not self.enabled or not request.session_id:
            return IssueSummary(rendered_context="", trace={"disabled": True})
        memory = self.store.load(request.session_id)
        threads = self.reader.select_threads(memory, request)
        summary = self.renderer.render(threads)
        summary.trace.update({"read_mode": request.read_mode, "reason": request.reason})
        return summary

    def observe_and_write(self, packet: TurnEvidencePacket) -> IssueMemoryTrace:
        if not (self.enabled and packet.session_id):
            return IssueMemoryTrace(False, "no_change", "memory disabled or missing session")
        memory = self.store.load(packet.session_id)
        change = self.detector.detect(packet, memory)
        if not change.should_write:
            return IssueMemoryTrace(False, change.change_type, change.reason)
        target_thread_id, create_new = self.resolver.resolve(memory, change, packet)
        trace = self.updater.apply(
            memory,
            change,
            packet=packet,
            target_thread_id=target_thread_id,
            create_new=create_new,
        )
        if trace.should_write:
            self.store.save(memory)
        return trace


def get_memory_manager_v4() -> MemoryManagerV4:
    global _manager_v4
    if _manager_v4 is None:
        _manager_v4 = MemoryManagerV4(
            store=_build_default_store(),
            enabled=settings.memory_enabled,
        )
    return _manager_v4


def reset_memory_manager_v4_for_tests() -> None:
    global _manager_v4
    _manager_v4 = None


def _build_default_store():
    mode = (settings.memory_v4_store or "redis").strip().lower()
    if mode == "memory":
        return InMemorySessionIssueMemoryStore()
    if mode != "redis":
        print(f"[WARN] 未知 MEMORY_V4_STORE={mode!r}，v4 记忆回退内存存储")
        return InMemorySessionIssueMemoryStore()
    try:
        import redis as redis_lib  # type: ignore[import-untyped]
    except ImportError:
        print("[WARN] 未安装 redis 包，v4 记忆回退内存存储。pip install redis")
        return InMemorySessionIssueMemoryStore()
    try:
        client = redis_lib.Redis.from_url(settings.redis_url, decode_responses=True)
        client.ping()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Redis 不可用，v4 记忆回退内存存储: {exc}")
        return InMemorySessionIssueMemoryStore()
    return RedisSessionIssueMemoryStore(
        client,
        key_prefix=settings.redis_conversation_key_prefix,
        ttl_seconds=settings.conversation_ttl_seconds,
    )
