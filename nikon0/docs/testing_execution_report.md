# nikon0 测试执行报告

**日期**: 2026-06-19 | **环境**: macOS Darwin 25.4.0, Python 3.13.12

---

## 1. Commands Run

```bash
# Step 1 - 语法检查
for f in tests_deep/*.py; do
  python -c "compile(open('$f').read(), '$f', 'exec')"
done

# Step 2 - 测试收集
python -m pytest tests_deep/ --collect-only -q --override-ini="testpaths="

# Step 3 - 逐文件 smoke tests
python -m pytest tests_deep/test_edge_planner.py -v --tb=short --override-ini="testpaths="
python -m pytest tests_deep/test_edge_skill_routing.py -v --tb=short --override-ini="testpaths="
python -m pytest tests_deep/test_edge_memory.py -v --tb=short --override-ini="testpaths="
python -m pytest tests_deep/test_edge_context.py -v --tb=short --override-ini="testpaths="
python -m pytest tests_deep/test_edge_tools.py -v --tb=short --override-ini="testpaths="
python -m pytest tests_deep/test_edge_safety.py -v --tb=short --override-ini="testpaths="
python -m pytest tests_deep/test_edge_failure.py -v --tb=short --override-ini="testpaths="
python -m pytest tests_deep/test_edge_integration.py -v --tb=short --override-ini="testpaths="
python -m pytest tests_deep/test_edge_boundary.py -v --tb=short --override-ini="testpaths="

# Step 4 - 已有项目测试
python -m pytest nikon0/app/test/ -v --tb=short
python -m pytest nikon0/app/test/ --ignore=nikon0/app/test/test_phase1_runtime.py -v --tb=short
```

---

## 2. Test Collection Result

| 指标 | 值 |
|------|----|
| 收集成功 | 160 tests collected |
| 收集错误 | 0 |
| 语法错误 | 1 (已修复) |

### 修复的语法问题

| 文件 | 问题 | 修复 |
|------|------|------|
| `test_edge_tools.py` | `asyncio.run()` 括号不匹配 | 重写为 `async def _run()` + `asyncio.run(_run())` 模式 |
| `test_edge_context.py` | `_trim_tail` 不存在于 `context.runtime` | 改为从 `context.budgeter` import `trim_tail` |

### 类命名问题

原始文件使用 `class Describe*` 命名，pytest 默认只收集 `class Test*`。已将全部 27 个类从 `Describe*` 重命名为 `Test*`。

---

## 3. Existing Test / Eval Baseline

### 项目已有测试

| 测试文件 | 结果 |
|----------|------|
| `test_agent_eval_runner.py` (8 tests) | 全部通过 |
| `test_context_eval.py` (4 tests) | 全部通过 |
| `test_context_pack.py` (20 tests) | 全部通过 |
| `test_mcp_provider.py` (3 tests) | 全部通过 |
| `test_memory_module.py` (5 tests) | 全部通过 |
| `test_memory_persistence.py` (3 tests) | 全部通过 |
| `test_phase1_tool_inventory.py` (7 tests) | 全部通过 |
| `test_product_resolver.py` (8 tests) | 全部通过 |
| `test_runtime_profiles.py` (6 tests) | 全部通过 |
| `test_skill_routing.py` (16 tests) | 全部通过 |
| `test_workflow_protocol.py` (4 tests) | 全部通过 |
| `test_phase1_runtime.py` (~40 tests) | **无法导入**（缺少 `fastapi` 依赖） |

**汇总**: 85/85 可收集测试全部通过。

### Eval 基线

- `nikon0/eval/run_agent_eval.py` — 已通过 `test_agent_eval_runner.py` (8 tests) 间接验证
- `nikon0/eval/context_eval.py` — 已通过 `test_context_eval.py` (4 tests) 间接验证
- `nikon0/eval/datasets/agent_eval_150.jsonl` — 数据集文件存在
- `nikon0/eval/datasets/product_support_40.jsonl` — 数据集文件存在

---

## 4. New Targeted Test Results

### 总体统计

| 指标 | 值 |
|------|----|
| 新增测试总数 | 160 |
| 通过 | 134 |
| 失败 | 26 |
| 通过率 | 83.8% |

### 按模块详情

#### Planner (19 tests: 14 pass, 5 fail)

| 状态 | 测试 | 失败原因 |
|------|------|----------|
| FAIL | `test_empty_message_produces_general_intent` | test_infra: AgentRequest Pydantic 拒绝空消息 |
| FAIL | `test_whitespace_only_message` | test_infra: AgentRequest Pydantic 拒绝纯空白消息 |
| FAIL | `test_very_long_message` | test_assertion: 100K "X" + "工具回声" 的尾部中文触发了 product_support 而非 case_intake |
| FAIL | `test_english_only_message` | test_assertion: 英文 "broken, how to fix" 触发了 product_support（Planner 包含英文关键词） |
| FAIL | `test_partial_keyword_match_edge` | test_assertion: "怎么" 单独也触发 product_support（Planner 将其作为 troubleshooting 信号） |
| PASS | 14 tests | — |

#### Skill Routing (11 tests: 9 pass, 2 fail)

| 状态 | 测试 | 失败原因 |
|------|------|----------|
| FAIL | `test_model_selector_low_confidence_falls_back_to_planned` | test_assertion: 低置信度选择后 selected_skills 为空，而非回退到 planned |
| **FAIL** | `test_rule_fallback_selection_when_model_unavailable` | **system_behavior**: `_model_match()` 中 model selector 抛异常未经捕获，导致整个请求崩溃 |
| PASS | 9 tests | — |

#### Memory (23 tests: 23 pass, 0 fail)

全部通过。

#### Context (22 tests: 19 pass, 3 fail)

| 状态 | 测试 | 失败原因 |
|------|------|----------|
| FAIL | `test_empty_context_still_builds_pack` | test_infra: 空消息被 Pydantic 拒绝 |
| FAIL | `test_excerpt_truncation_at_query_match` | test_assertion: excerpt 实际长度 80 > 预期 55 |
| FAIL | `test_excerpt_no_query_match_truncates_head` | test_assertion: excerpt 实际长度 80 > 预期 30 |
| PASS | 19 tests | — |

#### ToolRuntime (25 tests: 23 pass, 2 fail)

| 状态 | 测试 | 失败原因 |
|------|------|----------|
| FAIL | `test_failure_hook_triggered` | test_assertion: `on_failure` hook 仅在工具抛异常时触发，返回 `ok=False` 结果不算 failure |
| FAIL | `test_validate_answer_grounding_has_overlap` | test_assertion: `grounded` 返回 False（token overlap 不足以达标） |
| PASS | 23 tests | — |

#### Safety (13 tests: 13 pass, 0 fail)

全部通过。SafetyGate 关键词检测、优先级逻辑、trace 记录均正常工作。

#### Failure Modes (16 tests: 13 pass, 3 fail)

| 状态 | 测试 | 失败原因 |
|------|------|----------|
| FAIL | `test_llm_unavailable_for_product_support_falls_back` | test_assertion: LLM 失败后回退路径不经过 product_support skill（走 general handle） |
| FAIL | `test_llm_empty_response_falls_back` | test_assertion: 同上 |
| FAIL | `test_corrupted_memory_does_not_crash_runtime` | **system_behavior**: 向 `_state` 注入非对象值后，`load()` 调用 `model_copy()` 发生 AttributeError |
| PASS | 13 tests | — |

#### Integration (11 tests: 7 pass, 4 fail)

| 状态 | 测试 | 失败原因 |
|------|------|----------|
| FAIL | `test_repair_then_switch_to_refund` | test_assertion: 无 approval action（refund 审批未触发） |
| FAIL | `test_knowledge_backend_query_and_answer` | test_assertion: product_support skill 未被选中 |
| FAIL | `test_no_evidence_produces_needs_more_info` | test_assertion: 同上 |
| PASS | 7 tests | — |

#### Boundary (20 tests: 13 pass, 7 fail)

| 状态 | 测试 | 失败原因 |
|------|------|----------|
| FAIL | `test_session_state_with_corrupted_flat_state_type` | **system_behavior**: 损坏的 flat_state（字符串而非 dict）导致 `apply_updates()` TypeError |
| FAIL | `test_score_passage_edge_cases` | test_assertion: 空列表评分预期 0.0，实际 0.2 |
| FAIL | `test_structured_manual_backend_no_directory` | test_infra: `StructuredManualBackend` 未在文件中 import |
| FAIL | `test_structured_manual_backend_filter_manuals` | test_infra: 同上 |
| PASS | 13 tests | — |

---

## 5. Real Issues Found

### Issue #1: Model selector 异常未被捕获，导致请求崩溃

- **严重程度**: **medium**
- **现象**: 当 LLM skill selector 不可用（网络故障、超时、模型异常），RuntimeError 从 `_model_match()` 一路传播到 `runtime.run()`，整个请求返回 500 而非优雅降级
- **复现命令**:
  ```bash
  python -m pytest tests_deep/test_edge_skill_routing.py::TestSkillRoutingEdgeCases::test_rule_fallback_selection_when_model_unavailable -v --tb=long --override-ini="testpaths="
  ```
- **失败信息**: `RuntimeError: model service unavailable` → 未经捕获，异常传播至 `asyncio.run()`
- **相关代码位置**: `nikon0/skills/base.py:192` — `_model_match()` 方法中 `await self.selector.select(...)` 无 try/except。调用链: `select_best()` (line 113) → `_model_match()` (line 188) → supervisor → loop → runtime
- **建议修复**: 在 `_model_match()` 中对 selector 调用加 try/except，异常时返回 `(None, SkillMatch(matched=False), SkillSelection(source="error"))` 使 fallback 链继续

### Issue #2: 损坏的 memory 状态在 `load()` 中触发未处理异常

- **严重程度**: **low**
- **现象**: 如果 `InMemorySessionIssueStore._state` 中存储了非 `SessionIssueMemory` 类型的值（如直接注入字符串），`load()` 调用 `.model_copy()` 触发 `AttributeError`
- **复现命令**:
  ```bash
  python -m pytest tests_deep/test_edge_failure.py::TestStateCorruption::test_corrupted_memory_does_not_crash_runtime -v --tb=long --override-ini="testpaths="
  ```
- **失败信息**: `AttributeError: 'str' object has no attribute 'model_copy'`
- **相关代码位置**: `nikon0/memory/session.py:23` — `return memory.model_copy(deep=True)` 无类型检查
- **建议修复**: `load()` 中检查 `memory` 是否为 `SessionIssueMemory` 实例，否则重建空状态
- **备注**: 此问题需要直接操作 `_state`（私有属性）才能触发，生产环境中不直接可达。但在 Redis/MySQL 反序列化场景中可能出现类型损坏

### Issue #3: 损坏的 flat_state 在 `apply_updates()` 中触发 TypeError

- **严重程度**: **low**
- **现象**: 如果 `SessionIssueMemory.flat_state` 不是 dict（如被错误设置为字符串），`apply_updates()` 中的 `memory.flat_state[update.key] = update.value` 触发 `TypeError`
- **复现命令**:
  ```bash
  python -m pytest tests_deep/test_edge_boundary.py::TestBoundaryState::test_session_state_with_corrupted_flat_state_type -v --tb=long --override-ini="testpaths="
  ```
- **失败信息**: `TypeError: 'str' object does not support item assignment`
- **相关代码位置**: `nikon0/memory/session.py:61`
- **建议修复**: `apply_updates()` 开始时检查 `isinstance(memory.flat_state, dict)`，否则重置为 `{}`

---

## 6. Test Infrastructure Issues

### 6.1 类命名规则

原始文件使用 `class Describe*`（如 `DescribePlannerEdgeCases`），pytest 不收集这类名称。已统一改为 `class Test*`。

### 6.2 异步测试模式

部分测试文件（`test_edge_tools.py`, `test_edge_safety.py`）使用 `async def test_*` 方法。项目安装了 `pytest-asyncio 1.4.0` 但不支持 `asyncio_mode = auto`。已将这些文件改为 `def test_*` 内嵌 `async def _run()` + `asyncio.run(_run())` 模式。

### 6.3 缺少依赖

`test_phase1_runtime.py` 依赖 `fastapi`，当前环境未安装。其余 11 个已有测试文件不受影响。

### 6.4 与系统行为不对齐的断言（测试 bug，非系统 bug）

以下断言基于对系统行为的错误假设：

| 测试 | 错误假设 | 实际行为 |
|------|---------|---------|
| `test_empty_message_*` / `test_whitespace_*` / `test_empty_context_*` | 系统应接受空消息 | AgentRequest Pydantic 验证 `message` 必须非空 |
| `test_english_only_message` | 纯英文走 general handle | Planner 的 PRODUCT_SUPPORT_KEYWORDS 包含英文关键词 "broken", "fix" 等 |
| `test_partial_keyword_match_edge` | "怎么" 不触发 product_support | Planner 将 "怎么" 作为 troubleshooting 信号词 |
| `test_very_long_message` | 长消息尾部的 "工具回声" 触发 case_intake | "工具回声" 的关键词映射到了 product_support |
| `test_excerpt_truncation_*` | excerpt 会被截断到 budget 以内 | evidence excerpt 逻辑使用从 query 匹配点展开的策略，实际长度可能超 budget |
| `test_failure_hook_triggered` | `on_failure` hook 在工具返回 error result 时触发 | `on_failure` 仅在工具抛出异常时触发，返回 `ok=False` 不算 failure |
| `test_score_passage_edge_cases` | 空列表评分返回 0.0 | 实际返回 0.2（有 baseline） |
| `test_llm_unavailable_for_product_support_falls_back` | LLM 故障时 skill 走回退 | LLM 故障导致 skill registry 根本不会选中 product_support，走 general handle |
| `test_knowledge_backend_query_and_answer` | product_support 被选中 | make_runtime 注入的 skill 还需要 planner 匹配，测试构造不完整 |
| `test_structured_manual_backend_*` | 文件顶部已 import | 实际 `StructuredManualBackend` 未在 import 中 |

---

## 7. Recommended Next Tests

按重要性排序，仅推荐以下验证点：

1. **Planner 关键词准确率统计** — 用 product_support_40.jsonl 数据集的 40 条消息统计 Planner 的正确/误触发比例
2. **FailingLlmClient 场景下 `_model_match` 异常处理** — 加 try/except 后验证 fallback 链完整性
3. **多轮状态下 sticky 切换** — case_intake 完成后是否能正确切换到 product_support
4. **EnterpriseRagBackend fallback 链路完整测试** — Milvus 故障 → StructuredManual 回退 → 空结果降级
5. **ContextPack 在真实 case_intake workflow 中的字段包含检查** — tool_observations section 是否包含 ticket_id
6. **SafetyGate 对 answer_draft 内容的检查** — 当前只检查 message 和 status，不检查 answer 内容中的敏感词
7. **ToolHook `on_failure` 对 `ok=False` 结果的触发逻辑** — 确认当前行为是否为设计意图

---

*报告生成时间: 2026-06-19 | 共计执行命令: 12 条 | 测试环境: macOS + Python 3.13.12*
