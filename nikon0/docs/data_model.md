# Data Model

nikon0 的数据模型分为运行态、业务态、审计态三类。

## Runtime State

运行态只服务当前一轮 agent 执行。

```python
class AgentContext(BaseModel):
    request: AgentRequest
    session_state: SessionIssueMemory | None
    memory_context: str = ""
    selected_skill: str | None = None
    knowledge_result: KnowledgeResult | None = None
    tool_results: list[ToolCallResult] = []
    trace: ExecutionTrace
```

## Session Issue State

复用当前 v4 的核心思想。

```python
class SessionIssueMemory(BaseModel):
    session_id: str
    active_thread_id: str | None
    threads: dict[str, IssueThread]
    entity_index: dict[str, list[str]]
    turn_count: int
    updated_at: float
```

```python
class IssueThread(BaseModel):
    thread_id: str
    status: str              # open / diagnosing / waiting_user / submitted / resolved / cancelled
    issue_type: str          # howto / fault / repair / refund / complaint / unknown
    product_model: str | None
    facts: dict[str, IssueFact]
    evidence_refs: dict[str, EvidenceRef]
    last_turn_ids: list[str]
    created_at: float
    updated_at: float
```

## Business Records

长期业务事实不由 Agent 自己记忆，而是来自 MCP Gateway 后面的业务系统。

第一批业务对象：

```text
case_ticket
- case_id
- status
- product_model
- issue_description
- missing_slots
- created_at
- updated_at

order
- order_id
- status
- product_model
- purchase_time
- warranty_status

refund_assessment
- order_id
- eligible
- reasons
- requires_human_review
```

## Evidence

所有关键结论都应该能追溯证据。

```python
class Evidence(BaseModel):
    evidence_id: str
    source: str              # user / manual / image / tool / memory / policy
    text: str
    payload: dict[str, Any] = {}
    confidence: float = 1.0
```

## Execution Trace

```python
class ExecutionTrace(BaseModel):
    trace_id: str
    session_id: str
    user_message: str
    selected_skills: list[str]
    knowledge_calls: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    safety_decisions: list[dict[str, Any]]
    memory_updates: list[dict[str, Any]]
    final_risk_level: str
```

trace 是 nikon0 评测和排查的核心数据。

