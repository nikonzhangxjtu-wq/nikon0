# nikon0 Memory Governance V1 Report

## Goal

Memory V1 changes nikon0 from direct skill-owned state mutation to a governed
runtime pipeline:

```text
Skill StateUpdate
  -> StateUpdateCandidate
  -> MemoryWriteGate
  -> MemoryWriteDecision
  -> IssueThread-aware store write
  -> Redis hot snapshot + MySQL durable audit
```

The runtime, not a Skill or model, is the final authority for persistence.

## Architecture

### Read path

1. Load `SessionIssueMemory` from Redis, or restore it from MySQL on a cache miss.
2. `MemoryReadPlanner` selects a minimal thread scope and lifecycle action.
3. The LLM may recommend a read/thread decision in production. Its JSON output is
   validated against open thread ids and terminal status. Invalid, low-confidence,
   or failed calls fall back to deterministic rules.
4. `MemoryViewBuilder` renders only approved threads and session facts. Raw
   `flat_state` is no longer passed to the model selector.

### Write path

`AgentRuntime` converts each legacy `StateUpdate` to a candidate carrying
provenance, confidence, risk and an idempotency key. `MemoryWriteGate` decides:

| Outcome | Runtime behavior |
|---|---|
| `accept` | Write into the selected thread and persist the audit event. |
| `reject` | Do not persist; emit a trace event. |
| `needs_confirmation` | Preserve prior fact, pause dependent work, ask user to confirm. |
| `no_op` | Avoid duplicate persistence. |

Critical fields include product/model, order identifier, phone, address, ticket
and case identifiers. A lower-trust candidate cannot overwrite a stored critical
fact. Workflow progress such as `collecting -> ready` is intentionally not treated
as a user-fact conflict.

### Thread lifecycle

`IssueThreadLifecycleManager` produces one of:

- `continue_active`
- `switch_open_thread`
- `create_thread`
- `needs_clarification`

Terminal threads (`submitted`, `resolved`, `cancelled`) cannot be reused by
default. Explicit current-message product identity wins over a prior session
product; session product is only a fallback for vague follow-ups.

## Persistence and Availability

Redis is a hot snapshot/cache. MySQL owns durable snapshots, state update events,
write decisions and thread lifecycle events.

MySQL persistence includes:

- `nikon0_memory_sessions.memory_version` for optimistic concurrency;
- `nikon0_state_update_events` for accepted legacy-compatible updates;
- `nikon0_memory_write_decisions` for accept/reject/confirmation/no-op audit;
- `nikon0_memory_thread_events` for lifecycle audit.

Each session write acquires a Redis `SET NX` lock, with a process-local fallback
for minimal/test Redis clients. While holding that lock, it reads the SQL snapshot
and version, applies the update, conditionally persists the next version, then
refreshes Redis. This prevents duplicate initial inserts and lost updates across
runtime instances.

If persistence fails, low-risk requests use an explicitly traced ephemeral-memory
degradation. Medium/high-risk writes do not advance the workflow and return a
handoff-safe response. No silent fallback is used in the runtime debug or trace.

## LLM Boundary

The LLM is permitted to recommend thread selection and read scope only. It cannot
write memory or resolve conflicts. The deterministic gate owns validation,
terminal-thread policy, field authority and final persistence.

## Real Validation Evidence

The configured runtime was verified as:

```text
store_type: RedisMysqlSessionIssueStore
redis_ok: true
mysql_ok: true
sql_dialect: mysql
schema_version: 1
```

### Case 1: Runtime write

Input: `空气炸锅使用后如何清洁？`

- ProductSupport emitted one candidate.
- Gate returned `accept`.
- MySQL recorded one `nikon0_state_update_events` row and one
  `nikon0_memory_write_decisions` row.

### Case 2: Critical conflict

Initial stored fact: `case_intake.order_id=ORD-10001`, source `user`, confidence
`0.95`.

Incoming candidate: `ORD-20002`, source `verified_tool`, confidence `0.62`.

Result: `needs_confirmation`; MySQL replay still contained `ORD-10001`, and the
decision row was persisted. The old fact was not overwritten.

### Case 3: Product thread switch

1. Store an air-conditioner thread.
2. User introduces an airfryer issue: lifecycle returns `create_thread`.
3. User says `刚才的空调问题继续`: lifecycle returns `switch_open_thread` to
   the original air-conditioner thread.

The real MySQL snapshot contained two threads and the switch decision pointed to
the first thread.

## Eval and Load Testing

`nikon0/eval/datasets/memory_governance_cases.jsonl` covers continuation,
product switch, switch-back, key-fact conflict, terminal thread behavior and
ambiguous references. `run_agent_eval` now reports:

- `memory_write_accept_rate`
- `memory_write_reject_rate`
- `memory_confirmation_rate`
- `memory_degraded_write_rate`
- `memory_read_fallback_rate`

The deterministic memory baseline exercised six cases and produced write-decision
metrics; it is a governance regression baseline, not a final LLM quality score.

Real Redis/MySQL store-only load test:

```text
40 independent sessions + 10 concurrent writers to one shared session
operations: 50
error_rate: 0.0
shared_event_count: 10/10
shared_consistent: true
p50: 72.76 ms
p95: 130.04 ms
p99: 144.37 ms
throughput: 331.80 ops/sec
```

The first load-test run exposed duplicate snapshot insertion and stale version
reuse. The final run passed after adding session locking and making the SQL
`memory_version` column authoritative over snapshot JSON.

## Operational Commands

```bash
# Memory-specific Eval baseline
conda run -n kefu python -m nikon0.eval.run_agent_eval \
  --dataset nikon0/eval/datasets/memory_governance_cases.jsonl \
  --output-dir nikon0/eval/reports/memory-governance \
  --runtime-profile production_like

# Real Redis/MySQL 50-operation load test
conda run -n kefu python -m nikon0.eval.memory_load_test \
  --output-dir nikon0/eval/reports/memory-load \
  --independent-sessions 40 \
  --shared-writers 10
```

## Remaining Risks

- A direct production MemoryReadPlanner LLM smoke call succeeded. The complete
  production-like multi-case Eval still needs a scheduled run with observable
  progress and a persisted report artifact; prior whole-run attempts did not
  produce an artifact in this desktop execution channel.
- Full PII is retained by product decision. It is scoped to same-session memory,
  but a future RBAC/encryption phase should govern broader operator access.
- `flat_state` remains as a compatibility/workflow layer. New Skills should use
  thread-backed facts rather than expanding `flat_state`.
- The current load test isolates the store; an LLM/RAG end-to-end load profile
  should be run separately because provider latency dominates it.
