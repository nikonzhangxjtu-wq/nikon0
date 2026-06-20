# nikon0 Tool Inventory

This document records the Phase 1 tool boundary for the current two business
scenes: product support and case intake.

## Boundary

- Runtime owns tool discovery, permission checks, hooks, trace events, and tool
  result normalization.
- Skill owns task-local orchestration and can call multiple tools inside one
  skill execution.
- Tool owns one observable atomic action. A tool should be reusable by future
  skills such as presales guidance, fault diagnosis, installation guidance,
  order support, and refund intake.
- Workflow owns strong business protocol decisions for high-risk flows, such as
  required slots, approval, handoff, and stop conditions.
- MCP owns external system connectivity only. MCP servers provide tools,
  resources, and prompts; nikon0 still owns business decisions, permission,
  policy, context governance, and trace.

## MCP Connector Layer

MCP capabilities are normalized through `McpCapabilityProvider` before they
enter `ToolRegistry`.

Flow:

1. `McpClient` discovers services and tools from MCP Gateway.
2. `McpCapabilityProvider` normalizes remote tool metadata.
3. `McpToolAdapter` exposes the remote tool as a nikon0 `Tool`.
4. `ToolRuntime` applies permission checks, hooks, policy, and trace.
5. Skills can only use tools that are visible through `ToolRegistry`.

MCP metadata is preserved in tool schemas and trace:

- `x-provider: mcp`
- `x-source-service`
- `x-capability-tags`
- `provider=mcp`
- `source_service`

Local policy can override MCP-provided risk metadata, so external services
cannot lower nikon0 safety requirements.

## Product Support Tools

### product-support.resolve_product

Purpose: resolve product scope from the user message and optional session state.

Input:

- `message`: user query.
- `session_state`: optional memory snapshot.

Output:

- `resolution.status`: `resolved`, `disambiguation_required`, or `passthrough`.
- `resolution.product_id`
- `resolution.display_name`
- `resolution.manual_names`
- `resolution.source`
- `resolution.disclose_default_product`
- `resolution.candidate_product_ids`

Current status: registered, tested, and consumed by `ProductSupportSkill`.

### product-support.search_product_manual

Purpose: retrieve product manual evidence through `KnowledgeRuntime`.

Input:

- `query`
- `product_model`
- `allowed_manual_names`
- `images`
- `tenant_id`
- `user_id`
- `knowledge_version`
- `max_evidence`

Output:

- `answer_hints`
- `evidence`
- `manual_names`
- `backend_trace`

Current status: registered, tested, and consumed by `ProductSupportSkill` with
injectable `KnowledgeRuntime`. Default runtime uses the enterprise RAG backend
through `KnowledgeRuntime()`.

### product-support.validate_answer_grounding

Purpose: first-pass deterministic answer grounding check.

Input:

- `answer`
- `evidence`
- `required_terms`

Output:

- `grounded`
- `missing_terms`
- `evidence_count`
- `token_overlap`
- `reason`

Current status: registered, tested, and consumed by `ProductSupportSkill` as an
audit signal. This is a platform validation primitive, not a final enterprise
judge. Future versions should add LLM-as-judge, fact-level citation checks, and
policy-specific validators.

## Case Intake Tools

### case-intake.extract_case_slots

Purpose: extract preliminary intent and slots before calling strict intake
workflow tools.

Input:

- `message`

Output:

- `intent`: `repair`, `refund`, `complaint`, or `unknown`.
- `slots.contact_phone`
- `slots.order_id`
- `slots.product_model`
- `slots.issue`
- `missing_slots`
- `confidence`

Current status: registered and tested. This is a helper tool; authoritative
case progression still lives in the MCP-backed intake workflow.

## Case Workflow Protocols

`CaseIntakeSkill` now calls `case-intake.extract_case_slots`, then delegates
the business decision to `WorkflowRuntime`.

Built-in protocols:

- `repair_intake`: collects product model, issue, and contact phone.
- `refund_intake`: high-risk refund/return intake; requires approval and does
  not allow automatic refund commitment.
- `complaint_escalation`: high-risk complaint or escalation; requires human
  handoff.
- `case_intake_cancel`: cancels active case-intake collection.

Workflow decisions are emitted to trace as `workflow.select` and
`workflow.decision`. `SafetyGate` reads these decisions before falling back to
keyword-based safety rules.

### case-intake.collect_case_intake

Purpose: authoritative case-intake workflow step through MCP Gateway.

Input:

- `question`
- `session_id`
- `conversation_history`
- `enrichment`

Output: normalized MCP payload with reply text, missing slots, completion
status, and ticket payload.

Current status: registered through `McpGatewayTool` and already consumed by
`CaseIntakeSkill`.

### case-intake.try_cancel_case_intake

Purpose: cancel active case-intake workflow through MCP Gateway.

Current status: registered through `McpGatewayTool` and already consumed by
`CaseIntakeSkill`.

## Memory Tools

### memory.read_session_memory

Purpose: return an explicit memory snapshot to a skill/tool plan.

Input:

- `session_state`

Output:

- `session_state`

Current status: registered and tested. Runtime-native memory injection is still
the source of truth; this tool is a reusable protocol shape for future
agentic tool plans.

### memory.write_session_fact

Purpose: produce an explicit `state_update` patch for Runtime to apply.

Input:

- `key`
- `value`
- `reason`
- `evidence_ids`

Output:

- `state_update`

Current status: registered and tested. It does not mutate memory directly in
Phase 1; mutation remains centralized in Runtime memory handling.

## Product Support Tool-Step Flow

`ProductSupportSkill` now uses explicit tool-step orchestration:

1. `product-support.resolve_product`
2. `product-support.search_product_manual`
3. optional retrieval-evidence product resolution
4. LLM answer generation
5. `product-support.validate_answer_grounding`
6. memory state update emitted back to Runtime

The public answer behavior remains compatible with the earlier implementation,
while product-support decisions are visible in `trace.tool_calls` and
`response.actions`.
