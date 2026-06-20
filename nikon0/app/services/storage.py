"""In-memory trace and transcript stores for the first runtime versions."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from nikon0.app.schemas.storage import StoredTrace, TranscriptEntry
from nikon0.app.schemas.trace import ExecutionTrace


class InMemoryTraceRecorder:
    """Stores execution traces by trace id."""

    def __init__(self) -> None:
        self._traces: dict[str, StoredTrace] = {}
        self._by_session: dict[str, list[str]] = defaultdict(list)

    def record(self, trace: ExecutionTrace) -> StoredTrace:
        stored = StoredTrace(trace_id=trace.trace_id, session_id=trace.session_id, trace=trace)
        self._traces[trace.trace_id] = stored
        self._by_session[trace.session_id].append(trace.trace_id)
        return stored

    def get(self, trace_id: str) -> StoredTrace | None:
        return self._traces.get(trace_id)

    def list_for_session(self, session_id: str) -> list[StoredTrace]:
        return [
            self._traces[trace_id]
            for trace_id in self._by_session.get(session_id, [])
            if trace_id in self._traces
        ]


class JsonlTraceRecorder:
    """Append-only JSONL trace recorder for replayable production diagnostics."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def default(cls) -> "JsonlTraceRecorder":
        return cls(Path("nikon0/infra/runtime/traces.jsonl"))

    def record(self, trace: ExecutionTrace) -> StoredTrace:
        stored = StoredTrace(trace_id=trace.trace_id, session_id=trace.session_id, trace=trace)
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(stored.model_dump_json())
            fp.write("\n")
        return stored

    def get(self, trace_id: str) -> StoredTrace | None:
        for trace in self._iter():
            if trace.trace_id == trace_id:
                return trace
        return None

    def list_for_session(self, session_id: str) -> list[StoredTrace]:
        return [trace for trace in self._iter() if trace.session_id == session_id]

    def _iter(self) -> list[StoredTrace]:
        if not self.path.exists():
            return []
        traces: list[StoredTrace] = []
        with self.path.open("r", encoding="utf-8") as fp:
            for line in fp:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    traces.append(StoredTrace.model_validate(json.loads(raw)))
                except (json.JSONDecodeError, ValueError):
                    continue
        return traces


class InMemoryTranscriptStore:
    """Stores replayable transcript entries by session id."""

    def __init__(self) -> None:
        self._entries: dict[str, list[TranscriptEntry]] = defaultdict(list)

    def append(self, entry: TranscriptEntry) -> None:
        self._entries[entry.session_id].append(entry)

    def list_for_session(self, session_id: str) -> list[TranscriptEntry]:
        return list(self._entries.get(session_id, []))

    def replay_text(self, session_id: str) -> str:
        entries = self.list_for_session(session_id)
        return "\n".join(f"{entry.role}: {entry.content}" for entry in entries)


class JsonlTranscriptStore:
    """Append-only JSONL transcript store.

    The file is intentionally simple so it can be replayed by eval runners and
    inspected without a database.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def default(cls) -> "JsonlTranscriptStore":
        return cls(Path("nikon0/infra/runtime/transcripts.jsonl"))

    def append(self, entry: TranscriptEntry) -> None:
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(entry.model_dump_json())
            fp.write("\n")

    def list_for_session(self, session_id: str) -> list[TranscriptEntry]:
        if not self.path.exists():
            return []
        entries: list[TranscriptEntry] = []
        with self.path.open("r", encoding="utf-8") as fp:
            for line in fp:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    entry = TranscriptEntry.model_validate(json.loads(raw))
                except (json.JSONDecodeError, ValueError):
                    continue
                if entry.session_id == session_id:
                    entries.append(entry)
        return entries

    def replay_text(self, session_id: str) -> str:
        entries = self.list_for_session(session_id)
        return "\n".join(f"{entry.role}: {entry.content}" for entry in entries)
