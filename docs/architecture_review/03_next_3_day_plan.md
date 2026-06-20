# nikon0 三天工业级改造方案（03）

> 基于 `01_architecture_review.md` 的真实代码缺口，给出"接下来最该做的一个模块"+三天可落地的工业级方案。
> 原则：现实可行、纯框架内可完成、立即提升可靠性与可审计性、为后续 Safety/Ticket 可靠化打地基。

---

## 1. Most Important Next Module

### 选定：**Memory 写入治理层（StateUpdate Candidate + Write Validator + 冲突检测）**

> 附带一个**半天的前置硬任务**：对齐 eval runtime 与生产 runtime（否则无法可信验证任何改动）。

### 1.1 为什么是它（而不是 Safety / Storage）

文档 01 指出 Safety 与 Storage(JSONL) 风险等级更高（Critical），但它们**三天内做不出可信的工业级成品**：

- **Safety 真正可靠化**需要：RBAC 模型 + 审批通过后的执行回路 + 对接真实业务系统的高风险动作定义。三天只能再造一层关键词/规则，仍是占位，投入产出比低。
- **Storage 可靠化**需要：trace/approval 迁 DB + 并发/事务/索引，工作量大且收益主要在"运维健壮性"，不改变 Agent 行为质量。

而 **Memory 写入治理**满足"最该先做"的全部条件：

1. **缺口明确且纯框架内**：当前 `_upsert_fact` 后写覆盖前写、无校验、无冲突检测（`nikon0/memory/session.py:204-220`），`StateUpdate` 是裸 dict 直接落库（`memory/session.py:53-99`）。
2. **是其他模块可靠化的地基**：Safety 要可信，前提是"会话状态可信"；Ticket 两段式确认、退款风险判定，都依赖"哪些事实是确定的、来自哪条证据、是否与历史冲突"。Memory 写入治理把这层确定性建起来。
3. **三天能交付工业级成品**：候选化 + 校验 + 冲突检测 + trace + eval，闭环完整、可验收。
4. **立刻提升所有 skill**：product_support / case_intake 都经此层落状态，一处加固全局受益。

### 1.2 它解决的根问题

**当前"谁写了什么状态、为什么写、是否与已有状态冲突"完全不可控、不可审计。** skill 写什么就存什么，后写无条件覆盖前写（`session.py:217`），没有置信度仲裁，没有矛盾检测（例如上轮 product_model=Z6III、本轮工具误抽成 Z8，会静默覆盖）。

### 1.3 不做的后果

- 多轮会话中产品/订单/联系方式被错误覆盖，且**事后无法定位是哪一步、依据什么证据写的**（虽有 EvidenceRef，但无"被拒绝的写入"记录）。
- Safety 和 Ticket 建立在不可信状态上，越往上越脆。
- eval 的 fact_coverage 等指标无法归因到"写入错误"还是"检索错误"。

### 1.4 与其他模块的关系

- **Skill**：skill 仍产出 `StateUpdate`，但改为产出 **`StateUpdateCandidate`（带 confidence/source/evidence）**，不再直接落库。
- **Runtime**：runtime 在 `apply_updates` 前插入 **`MemoryWriteValidator`** 做校验 + 冲突检测，仅接受通过的候选。
- **Memory**：`InMemorySessionIssueStore` / `RedisMysqlSessionIssueStore` 接收"已校验候选"，并新增"被拒绝候选"的审计落库。
- **Context**：被拒绝/冲突信息可进 memory section，让模型知道"有未决冲突需澄清"。
- **Tool / Workflow / MCP**：不变（候选 source 标注来自哪个 tool）。
- **Safety**：后续可消费"高风险字段变更需确认"的冲突信号。

---

## 2. Industrial-grade Design

### 2.1 目标

- 所有会话状态写入必须经过**候选化 → 校验 → 冲突检测 → 提交/拒绝/挂起**的统一管线。
- 每次写入（接受或拒绝）都**可追溯**：来源 skill/tool、证据、置信度、冲突对象、决策原因。
- 字段级**校验规则**（格式 + 枚举 + 必要性）+ **冲突策略**（覆盖 / 保留 / 标记待澄清）。
- 向后兼容：现有 skill 不改也能跑（兼容 `StateUpdate`），逐步迁移到 `StateUpdateCandidate`。

### 2.2 非目标（三天内明确不做）

- 不做 LLM-based memory extractor（保留为后续；本期校验器是确定性 + 规则）。
- 不做跨会话长期画像。
- 不重写 Redis/MySQL schema（复用现有 `nikon0_state_update_events`，新增一张 rejected 事件表即可）。
- 不动 Safety 关键词逻辑（仅产出冲突信号供其消费）。
- 不引入新的外部依赖。

### 2.3 核心数据结构

```python
# nikon0/memory/write_gate/types.py（新增）
from typing import Any, Literal
from pydantic import BaseModel, Field

WriteDecision = Literal["accept", "reject", "needs_confirmation", "merge"]
ConflictPolicy = Literal["overwrite", "keep_existing", "flag_for_confirmation", "higher_confidence_wins"]

class StateUpdateCandidate(BaseModel):
    key: str
    value: Any
    source: str                         # "product_support" | "case_intake" | "tool:collect_case_intake" ...
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    reason: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    field_path: str = ""                # 可选：细到字段，如 "ticket_payload.product_model"
    conflict_policy: ConflictPolicy = "higher_confidence_wins"

class WriteValidationIssue(BaseModel):
    code: str                           # "invalid_format" | "missing_required" | "conflict" | "low_confidence"
    field_path: str
    message: str
    existing_value: Any = None
    incoming_value: Any = None

class WriteValidationResult(BaseModel):
    candidate: StateUpdateCandidate
    decision: WriteDecision
    accepted_value: Any = None          # 经合并/仲裁后的最终值
    issues: list[WriteValidationIssue] = Field(default_factory=list)
    reason: str = ""
```

> 兼容层：`StateUpdate`（现有 `schemas/capability.py`）可通过 `StateUpdateCandidate.from_state_update(...)` 适配，默认 confidence=1.0、source=skill 名。

### 2.4 核心接口

```python
# nikon0/memory/write_gate/validator.py（新增）
class MemoryWriteValidator:
    def __init__(self, field_rules: "FieldRuleSet | None" = None) -> None: ...

    def validate(
        self,
        *,
        memory: SessionIssueMemory,             # 当前状态（用于冲突检测）
        candidates: list[StateUpdateCandidate],
    ) -> list[WriteValidationResult]:
        """对每个候选做：格式校验 → 冲突检测 → 决策(accept/reject/needs_confirmation/merge)。"""

# nikon0/memory/write_gate/rules.py（新增）
class FieldRuleSet:
    """字段级规则：正则/枚举/必填。配置化，便于扩展与租户差异。"""
    PHONE = r"^1[3-9]\d{9}$"
    ORDER_ID = r"^[A-Za-z0-9\-]{6,}$"
    PRODUCT_MODEL_REQUIRED_FOR = {"repair"}
    # validate_field(field_path, value) -> WriteValidationIssue | None
```

### 2.5 各层责任

| 层 | 责任 |
| --- | --- |
| **Skill** | 产出 `StateUpdateCandidate`（带 source/confidence/evidence）。不再假设写入一定成功。 |
| **Runtime** | 在 `apply_updates` 前调用 `MemoryWriteValidator.validate`；只把 accept/merge 的候选交给 store；把 reject/needs_confirmation 写入审计 + 注入 memory 冲突信号；写 trace。 |
| **Validator** | 字段校验 + 冲突检测 + 决策仲裁（confidence / policy）。纯函数、可单测。 |
| **Store** | 接收已校验值落库；新增"被拒绝/待确认候选"审计落库。 |

### 2.6 存储方案（复用 + 最小新增）

- 复用 `nikon0_memory_sessions`（快照）、`nikon0_state_update_events`（接受的写入，已存在 `memory/persistence.py:38-48`）。
- **新增表** `nikon0_state_write_decisions`：记录每个候选的 decision/issues/reason（接受与拒绝都记），用于审计与 eval：

```sql
CREATE TABLE nikon0_state_write_decisions (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  session_id VARCHAR(191) NOT NULL INDEX,
  turn_id VARCHAR(191) NOT NULL,
  update_key VARCHAR(191) NOT NULL,
  field_path VARCHAR(191) NOT NULL DEFAULT '',
  decision VARCHAR(32) NOT NULL,        -- accept/reject/needs_confirmation/merge
  source VARCHAR(64) NOT NULL,
  confidence FLOAT NOT NULL,
  issues_json JSON NOT NULL,
  reason TEXT NOT NULL,
  created_at DOUBLE NOT NULL
);
```

- In-Memory store 同步维护一个内存 list 供测试断言。
- SQLite（测试）走 `JSON().with_variant(Text(),"sqlite")`，与现有写法一致（`persistence.py:33`）。

### 2.7 Trace 方案

新增 trace 事件（`ExecutionTrace.add_event`）：
- `memory.write_validate`：候选总数、accept/reject/needs_confirmation 计数。
- `memory.write_rejected`：每个被拒候选的 key/field_path/code/reason/existing vs incoming。
- `memory.write_conflict`：冲突字段 + 采用的 policy + 最终值。

并在 `AgentResponse.debug` 增加 `memory_write_decisions` 列表（与现有 `memory_updates` 并列，`runtime.py:145-146` 附近）。

### 2.8 Eval 指标

在 `EvalRunReport`（`eval/run_agent_eval.py:71-92`）新增：
- `state_write_accept_rate`：accept /(总候选)。
- `state_conflict_detect_rate`：在注入冲突的 case 中被检测出的比例。
- `invalid_field_block_rate`：非法字段（错误电话/订单号）被拦截比例。

新增小数据集 `nikon0/eval/datasets/memory_write_gate_cases.jsonl`（~15 条）：覆盖正常写入、冲突覆盖、非法格式、低置信拒绝、待确认。

### 2.9 Failure modes（必须覆盖）

| 场景 | 期望行为 |
| --- | --- |
| 工具误抽 product_model 与历史冲突 | decision=needs_confirmation（高风险字段不静默覆盖） |
| 非法电话/订单号 | decision=reject + issue.code=invalid_format |
| 低置信候选覆盖高置信历史 | decision=keep_existing（higher_confidence_wins） |
| 同 turn 多候选写同字段 | 取最高 confidence，其余记 reject |
| validator 自身异常 | fail-open 退回旧 `apply_updates` 行为 + trace 告警（不能阻断主流程） |

### 2.10 Migration plan（向后兼容）

1. 适配器：`StateUpdate → StateUpdateCandidate`（confidence=1.0, source=skill 名）。runtime 默认把现有 `result.state_updates` 包成候选过校验。**现有 skill 零改动即可受保护**。
2. 灰度开关：`NIKON0_MEMORY_WRITE_GATE_ENABLED`（默认 true，可一键回退到旧 `apply_updates`）。
3. 逐步迁移 product_support / case_intake 显式产出带 confidence/evidence 的候选。

---

## 3. Three-day Implementation Plan

### Day 1 — 前置对齐 + 数据结构与校验器骨架

**目标**：消除"测假的"风险；落地候选/校验数据结构与确定性校验器（无冲突检测）。

- **前置（上午，半天，必做）对齐 eval/生产 runtime**：
  - 改动 `nikon0/agent/runtime.py:353-362`：移除生产默认 `MockSkill`（或加 `NIKON0_ENABLE_MOCK_SKILL` 开关，默认 false）。
  - 改动 `nikon0/eval/run_agent_eval.py:203-230`：`build_eval_runtime` 注入 `_build_default_context_governance()`，使 eval 与生产 context 路径一致（或提供 `--context-llm` 开关，至少跑一次对齐版基线）。
  - 重跑 `agent_eval_150` 得到"对齐后基线"，作为后续对照。
- **下午**：
  - 新增 `nikon0/memory/write_gate/types.py`（§2.3 数据结构）。
  - 新增 `nikon0/memory/write_gate/rules.py`（`FieldRuleSet`：电话/订单号正则、repair 必填 product_model）。
  - 新增 `nikon0/memory/write_gate/validator.py`：先实现"格式校验 + 决策(accept/reject)"，冲突检测留 Day 2。
- **测试**：`nikon0/app/test/test_memory_write_gate.py`（字段校验单测：合法/非法电话、订单号、缺必填）。
- **完成标准**：对齐后基线报告产出；validator 能对非法字段返回 reject，合法返回 accept；单测全绿。

### Day 2 — 冲突检测 + Runtime 接入 + Trace

**目标**：冲突仲裁完整；runtime 走候选化管线；trace 可见。

- **改动**：
  - `validator.py`：加冲突检测（与 `memory.flat_state` / active_product / thread.facts 比对）+ 4 种 `ConflictPolicy` 仲裁。
  - `nikon0/agent/runtime.py:140-146`：在 `apply_updates` 前插入 validate；只落 accept/merge；reject/needs_confirmation 进 trace + debug。
  - 适配器 `StateUpdate → StateUpdateCandidate`；灰度开关 `NIKON0_MEMORY_WRITE_GATE_ENABLED`。
  - `schemas/trace.py` 无需改结构（用 add_event），`runtime.py` 增 `memory_write_decisions` 到 debug。
- **新增数据结构**：`WriteValidationResult` 落地到 trace/debug；冲突信号注入 memory section（可选，`context/runtime.py` memory 来源）。
- **测试**：`test_memory_write_gate.py` 加冲突场景（覆盖/保留/待确认/置信仲裁）；`test_phase1_runtime.py` 加"runtime 经 gate 后状态正确"。
- **完成标准**：注入冲突的多轮 case 中，高风险字段冲突 → needs_confirmation 且不覆盖；trace 出现 `memory.write_validate` / `memory.write_rejected` / `memory.write_conflict`；开关可回退旧行为。

### Day 3 — 持久化审计 + Eval 指标 + 验收

**目标**：被拒/待确认候选可持久审计；eval 指标上线；端到端验收。

- **改动**：
  - `nikon0/memory/persistence.py`：新增 `nikon0_state_write_decisions` 表 + `append_write_decisions`；`RedisMysqlSessionIssueStore.apply_updates` 调用之。In-Memory store 加内存记录。
  - `eval/run_agent_eval.py:71-92`：`EvalRunReport` 增 3 个指标；`_run_case` 统计写入决策。
  - 新增 `nikon0/eval/datasets/memory_write_gate_cases.jsonl`（~15 条）。
- **测试**：`test_memory_persistence.py` 加"决策事件落库 + 可查询"；跑新数据集 eval。
- **完成标准**：见 §4 验收标准全部满足；产出 Day1 基线 vs Day3 报告对比。

---

## 4. Acceptance Criteria

- **可跑通的 demo case**：
  1. 多轮报修：轮1 缺型号 → 轮2 补型号写入 accept；若轮2 工具误抽冲突型号 → needs_confirmation，历史不被覆盖。
  2. 非法字段：电话"123" → reject，工单不前进。
  3. 置信仲裁：低置信工具值不覆盖高置信用户已确认值。
- **trace 中必须看到**：`memory.write_validate`（计数）、`memory.write_rejected`（含 existing vs incoming）、`memory.write_conflict`（policy + 最终值）。
- **必须持久化**：accept 写入（复用现有事件表）+ 全部决策（新表 `nikon0_state_write_decisions`），可按 session_id 查询回放。
- **必须覆盖的失败场景**：§2.9 全部五种（含 validator 自身异常 fail-open）。
- **toy 可保留 / 必须替换**：
  - 可保留：本期不动 `StructuredManualBackend`、SafetyGate 关键词、JSONL trace/approval（不在本期范围）。
  - 必须替换/移除：生产默认 `MockSkill`（Day1 移除）；eval 与生产 runtime 配置不一致（Day1 对齐）。
- **回归**：对齐后的 `agent_eval_150` 通过率不低于"对齐基线"（写入治理不应降低主指标）。

---

## 5. Risks and Tradeoffs

- **临时折中**：
  - 校验器是确定性规则（正则/枚举/置信），**不是 LLM extractor**——对自然语言抽取的纠错能力有限，作为 v1 可接受。
  - 冲突仲裁默认 `higher_confidence_wins`，但 skill 当前 confidence 多为常量（如 product_support 0.82），早期仲裁信号偏弱；先把管线建起来，confidence 真实化是后续事项。
- **三天内做不到**：跨会话冲突、语义级冲突（同义不同字面）、LLM 仲裁、Safety 与 gate 的联动消费。
- **需后续演进**：confidence 真实化（来自检索分数 / 抽取置信）；冲突信号驱动 Safety 二次确认；字段规则配置化到租户。
- **不要过早复杂化**：不要本期就引入完整规则 DSL / 插件式校验器；先用一个 `FieldRuleSet` 类 + 简单 policy 枚举，够用且可读。

---

## 6. Follow-up Roadmap（三天后，≤2 周）

**Week 1（gate 之后）**
1. **confidence 真实化**：product_support confidence 来自 RAG top1_score；case_intake 来自 slot 抽取置信。让仲裁有意义。
2. **冲突信号 → Safety/Context 联动**：needs_confirmation 字段自动生成澄清话术（复用 disambiguation 模式），并在高风险字段变更时触发 SafetyGate 二次确认。

**Week 2**
3. **Storage 可靠化第一刀**：trace/approval 从 JSONL 迁到 SQL（复用 `SqlMemoryPersistence` 模式），解决多副本并发（文档 01 §3.12 Critical）。
4. **审批执行回路**：approval `approved` 后重放被阻断的 tool_call（补齐 HITL 半截子，文档 01 §3.14）。
5. **grounding 硬阻断（可选灰度）**：`validate_answer_grounding` 不达标时降级为"证据不足"回复而非直接返回（文档 02 §1.6）。

> 顺序原则：先把"状态可信"（本期 Memory gate）→ 再"confidence 真实化 + 冲突联动"→ 再"存储/审批可靠化"→ 最后"答案 grounding 硬约束"。每一步都建立在上一步的确定性之上。
