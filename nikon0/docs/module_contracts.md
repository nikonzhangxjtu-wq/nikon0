# Module Contracts

本文定义 nikon0 第一版核心接口。后续写代码时，优先保证这些边界稳定。

## AgentRuntime Contract

```python
class AgentRuntime:
    async def run(self, request: AgentRequest) -> AgentResponse:
        ...
```

### AgentRequest

```python
class AgentRequest(BaseModel):
    session_id: str
    user_id: str | None = None
    message: str
    images: list[str] = []
    channel: str = "web"
    metadata: dict[str, Any] = {}
```

### AgentResponse

```python
class AgentResponse(BaseModel):
    answer: str
    images: list[str] = []
    state_summary: str = ""
    risk_level: str = "low"
    trace_id: str
    actions: list[AgentActionRecord] = []
    debug: dict[str, Any] = {}
```

## Skill Contract

```python
class Skill(Protocol):
    name: str
    description: str
    risk_level: str

    async def can_handle(self, context: AgentContext) -> SkillMatch:
        ...

    async def run(self, context: AgentContext) -> SkillResult:
        ...
```

### SkillMatch

```python
class SkillMatch(BaseModel):
    matched: bool
    confidence: float
    reason: str
    required_inputs: list[str] = []
```

### SkillResult

```python
class SkillResult(BaseModel):
    status: str                 # success / needs_more_info / failed / handoff_required
    answer_draft: str = ""
    evidence: list[Evidence] = []
    tool_calls: list[ToolCallRequest] = []
    state_updates: list[StateUpdate] = []
    risk_level: str = "low"
    handoff_reason: str | None = None
```

## ToolRuntime Contract

```python
class ToolRuntime:
    async def list_tools(self, service_id: str | None = None) -> list[ToolSpec]:
        ...

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        ...
```

### ToolCallRequest

```python
class ToolCallRequest(BaseModel):
    service_id: str
    tool_name: str
    arguments: dict[str, Any]
    risk_level: str = "low"
    requires_approval: bool = False
```

### ToolCallResult

```python
class ToolCallResult(BaseModel):
    ok: bool
    data: dict[str, Any] = {}
    error_code: str | None = None
    error_message: str | None = None
    raw: dict[str, Any] = {}
```

## KnowledgeRuntime Contract

```python
class KnowledgeRuntime:
    async def query(self, request: KnowledgeRequest) -> KnowledgeResult:
        ...
```

### KnowledgeRequest

```python
class KnowledgeRequest(BaseModel):
    query: str
    product_model: str | None = None
    intent: str = "unknown"     # howto / fault / button / policy / general
    need_images: bool = False
    max_evidence: int = 6
```

### KnowledgeResult

```python
class KnowledgeResult(BaseModel):
    answer_hints: list[str] = []
    evidence: list[Evidence] = []
    backend_trace: list[dict[str, Any]] = []
```

## Safety Contract

```python
class SafetyGate:
    async def check(self, context: AgentContext, result: SkillResult) -> SafetyDecision:
        ...
```

### SafetyDecision

```python
class SafetyDecision(BaseModel):
    allowed: bool
    risk_level: str
    requires_human: bool = False
    reason: str
    blocked_actions: list[str] = []
```

## Design Rule

AgentRuntime 只能依赖这些抽象接口，不直接依赖具体 RAG、Redis、Milvus、MCP 服务或业务数据库。具体实现通过 registry 注入。

