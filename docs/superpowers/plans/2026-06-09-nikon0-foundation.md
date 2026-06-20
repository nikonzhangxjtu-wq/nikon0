# nikon0 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the first nikon0 project foundation for an enterprise product service Agent, with a clean directory skeleton and architecture documentation.

**Architecture:** nikon0 is a self-built Agent framework, not a LangChain/LangGraph wrapper. The core runtime orchestrates skills, tools, knowledge, session issue memory, safety gates, and evaluation harnesses.

**Tech Stack:** FastAPI, Pydantic, Redis, PostgreSQL, MCP Gateway, Milvus as optional knowledge backend, custom AgentRuntime.

---

### Task 1: Create Project Skeleton

**Files:**
- Create: `nikon0/README.md`
- Create directories: `nikon0/app`, `nikon0/agent`, `nikon0/skills`, `nikon0/tools`, `nikon0/knowledge`, `nikon0/memory`, `nikon0/safety`, `nikon0/eval`, `nikon0/infra`, `nikon0/docs`

- [x] **Step 1: Create directories**

Run:

```bash
mkdir -p nikon0/docs nikon0/agent nikon0/skills nikon0/tools nikon0/knowledge nikon0/memory nikon0/safety nikon0/eval nikon0/app nikon0/infra
```

Expected: directories exist under `nikon0/`.

- [x] **Step 2: Write README**

Write `nikon0/README.md` with:

```markdown
# nikon0

nikon0 是一个面向企业产品服务场景的自研 Agent 框架。
```

Expected: README defines positioning, design principles, initial directory layout, and reading order.

### Task 2: Write Architecture Documents

**Files:**
- Create: `nikon0/docs/architecture.md`
- Create: `nikon0/docs/module_contracts.md`
- Create: `nikon0/docs/data_model.md`

- [x] **Step 1: Write architecture overview**

`architecture.md` must define:

```text
AgentRuntime
SkillRegistry
ToolRuntime
KnowledgeRuntime
Session Issue Memory
Safety / HITL
Evaluation Harness
```

Expected: architecture contains a Mermaid flowchart and migration relationship to the existing project.

- [x] **Step 2: Write module contracts**

`module_contracts.md` must define stable first-version contracts for:

```text
AgentRequest / AgentResponse
Skill / SkillMatch / SkillResult
ToolRuntime / ToolCallRequest / ToolCallResult
KnowledgeRuntime / KnowledgeRequest / KnowledgeResult
SafetyGate / SafetyDecision
```

Expected: future implementation can follow these interfaces without inventing new boundaries.

- [x] **Step 3: Write data model**

`data_model.md` must define:

```text
AgentContext
SessionIssueMemory
IssueThread
Business Records
Evidence
ExecutionTrace
```

Expected: the memory and trace model is business-focused, not a generic companion memory.

### Task 3: Write Engineering and Evaluation Baseline

**Files:**
- Create: `nikon0/docs/engineering_baseline.md`
- Create: `nikon0/docs/evaluation.md`
- Create: `nikon0/docs/roadmap.md`

- [x] **Step 1: Write engineering baseline**

Document:

```text
FastAPI API layer
Pydantic schemas
PostgreSQL for persistent records
Redis for session issue memory
MCP Gateway as external service boundary
Milvus only as optional knowledge backend
observability and security baseline
```

Expected: the project has a production-ready direction inspired by the reference FastAPI template, without adopting LangGraph as core runtime.

- [x] **Step 2: Write evaluation design**

Document eval layers:

```text
skill_selection_eval
tool_call_eval
knowledge_eval
memory_state_eval
safety_eval
end_to_end_eval
```

Expected: nikon0 can be measured by Agent behavior, not only answer quality.

- [x] **Step 3: Write roadmap**

Document phases:

```text
Phase 0: architecture foundation
Phase 1: minimal AgentRuntime
Phase 2: migrate existing customer-service skills
Phase 3: KnowledgeRuntime
Phase 4: ToolRuntime + MCP Gateway
Phase 5: Safety / HITL
Phase 6: Evaluation Harness
```

Expected: next conversation can start implementation from Phase 1.

### Task 4: Verify Foundation

**Files:**
- Inspect: `nikon0/`

- [ ] **Step 1: List created files**

Run:

```bash
find nikon0 -maxdepth 3 -type f | sort
```

Expected output includes:

```text
nikon0/README.md
nikon0/docs/architecture.md
nikon0/docs/module_contracts.md
nikon0/docs/data_model.md
nikon0/docs/engineering_baseline.md
nikon0/docs/evaluation.md
nikon0/docs/roadmap.md
```

- [ ] **Step 2: Review for placeholders**

Run:

```bash
rg "TODO|TBD|待定|占位" nikon0
```

Expected: no output.
