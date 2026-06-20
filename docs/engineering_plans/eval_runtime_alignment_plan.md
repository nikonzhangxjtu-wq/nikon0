# Eval / Runtime 对齐与 Eval 模块完善计划

> 阶段 1 产物：只读代码后的实施计划。本文不包含代码改动。
> 目标：让 nikon0 的评测结果代表真实上线行为，成为后续 Memory Write Gate、Safety、Ticket、RAG Grounding 等改造的可信基线。

---

## 1. Current Findings

### 1.1 生产 runtime 当前如何构建

- `AgentRuntime.__init__` 在未传入 `skill_registry` 时会调用 `_build_default_skills()` 构造默认 skill 列表；未传入 `context_governance` 时使用裸 `ContextGovernance()`；未传入 store/runtime 时使用内存实现。位置：`nikon0/agent/runtime.py:42-73`。
- `AgentRuntime.run()` 每轮都会执行 transcript replay、memory load、MemoryView 构建、`context_governance.agovern()`、planner、AgentLoop、SafetyGate、memory apply、trace/transcript 持久化和 response debug 输出。位置：`nikon0/agent/runtime.py:75-231`。
- 生产入口 `build_default_runtime()` 会注入 `_build_default_skill_registry()`、`_build_default_context_governance()`、`build_memory_store_from_env()`、JSONL trace/transcript/approval store。位置：`nikon0/agent/runtime.py:270-281`。
- 生产默认 ContextGovernance 由 `_build_default_context_governance()` 构建：读取 `NIKON0_CONTEXT_*` 配置；若启用 LLM 且有模型，则使用 `LlmContextReadPlanner`、`LlmConversationCompactor`、`LlmEvidenceSpanSelector`；否则回退 deterministic。位置：`nikon0/agent/runtime.py:284-323`。
- 生产默认 skill registry 会根据 `ROUTER_LLM_ENABLED` 和模型配置接入 `LlmSkillSelector`。位置：`nikon0/agent/runtime.py:326-350`。
- 生产默认 skill 列表当前包含 `ToolEchoSkill()`、`CaseIntakeSkill()`、`ProductSupportSkill(...)`、`MockSkill()`。这是最大不一致风险。位置：`nikon0/agent/runtime.py:353-362`。
- 生产默认 ProductSupport 使用 `KnowledgeRuntime(EnterpriseRagBackend())`，没有显式本地 manual fallback。位置：`nikon0/agent/runtime.py:388-389`。
- 生产默认 ToolRuntime 使用 `default_tools()`，其中包含 EchoTool、product-support 三个工具、case-intake 本地抽槽工具、MCP capability discovery；MCP discovery 失败时注册 `McpGatewayTool` fallback。位置：`nikon0/tools/runtime.py:131-152`、`nikon0/tools/runtime.py:275-344`。
- 生产 MemoryStore 来自 `build_memory_store_from_env()`，可由环境切到 Redis/MySQL；默认仍是 memory。位置：`nikon0/agent/runtime.py:277`、`app/core/config.py:263-267`。

### 1.2 eval runtime 当前如何构建

- `run_agent_eval()` 默认调用 `build_eval_runtime(manual_dir, use_real_llm, local_rag)`，除非外部传入 runtime。位置：`nikon0/eval/run_agent_eval.py:95-108`。
- `build_eval_runtime()` 手工构造 ProductSupportSkill、CaseIntakeSkill、ToolEchoSkill，显式不包含 `MockSkill`。位置：`nikon0/eval/run_agent_eval.py:203-230`。
- `build_eval_runtime()` 没有注入 `_build_default_context_governance()`，因此 `AgentRuntime.__init__` 使用裸 `ContextGovernance()`；该路径默认是 deterministic `ContextRuntime()`，不是生产的 LLM context 配置。位置：`nikon0/eval/run_agent_eval.py:226-230`、`nikon0/agent/runtime.py:64`、`nikon0/agent/context_governance.py:12-20`。
- eval ProductSupport 默认使用 `KnowledgeRuntime(EnterpriseRagBackend(fallback_backend=StructuredManualBackend(manual_dir)))`；`--local-rag` 时使用 `StructuredManualBackend`。位置：`nikon0/eval/run_agent_eval.py:233-237`。
- eval 的 case-intake 工具是 `_EvalCaseIntakeTool`，它 deterministic mock 了 `collect_case_intake` 和 `try_cancel_case_intake`。位置：`nikon0/eval/run_agent_eval.py:217-225`、`nikon0/eval/run_agent_eval.py:286-347`。
- eval 的 answer generator 可接真实 LLM；`--no-real-llm` 会关闭。位置：`nikon0/eval/run_agent_eval.py:240-257`、`nikon0/eval/run_agent_eval.py:693-715`。
- eval 的 LLM router 只有在 `use_real_llm=True` 且 `ROUTER_LLM_ENABLED=true` 时启用。位置：`nikon0/eval/run_agent_eval.py:260-283`。
- eval report 当前只包含业务指标，没有记录 runtime profile、context profile、mock usage、context section miss/omitted 等对齐审计字段。位置：`nikon0/eval/run_agent_eval.py:71-92`、`nikon0/eval/run_agent_eval.py:140-150`、`nikon0/eval/reports/agent-eval-full-150/metrics.json`。

### 1.3 二者具体不一致

- 生产有 `MockSkill`，eval 没有。生产默认 mock 可能吞掉未匹配请求；eval 看不到这个行为。位置：`nikon0/agent/runtime.py:353-362` vs `nikon0/eval/run_agent_eval.py:215`。
- 生产默认 context 可启用 LLM planner/compactor/span selector，eval 默认 deterministic context。位置：`nikon0/agent/runtime.py:284-323` vs `nikon0/eval/run_agent_eval.py:226-230`。
- 生产 ToolRuntime 走 `default_tools()`，case-intake 经 MCP discovery / McpGatewayTool；eval 使用 `_EvalCaseIntakeTool` mock。位置：`nikon0/tools/runtime.py:275-344` vs `nikon0/eval/run_agent_eval.py:217-225`、`nikon0/eval/run_agent_eval.py:286-347`。
- 生产 ProductSupport 的 EnterpriseRagBackend 没有显式 StructuredManual fallback；eval 默认有 fallback。位置：`nikon0/agent/runtime.py:388-389` vs `nikon0/eval/run_agent_eval.py:233-237`。
- 生产使用环境配置的 memory/trace/transcript/approval store；eval 默认使用 `AgentRuntime` 内存 store。位置：`nikon0/agent/runtime.py:270-281` vs `nikon0/eval/run_agent_eval.py:226-230`。
- eval 当前报告无法证明是否与生产匹配。architecture review 已明确指出这是最高风险之一。位置：`docs/architecture_review/01_architecture_review.md:51-52`、`docs/architecture_review/01_architecture_review.md:324`、`docs/architecture_review/01_architecture_review.md:421-432`。

### 1.4 哪些 mock 合理保留

- `EchoTool` / `ToolEchoSkill` 可以保留在 deterministic 或 legacy profile，用于验证 ToolRuntime 生命周期，但 production-like profile 不应让它影响业务路由。位置：`nikon0/tools/runtime.py:255-272`、`nikon0/skills/tool_echo.py`。
- `_EvalCaseIntakeTool` 可以保留为 deterministic/legacy baseline，尤其用于无外部 MCP 服务时的稳定回归。位置：`nikon0/eval/run_agent_eval.py:286-347`。
- `StructuredManualBackend` 可以保留为 `--local-rag` 基线或 enterprise RAG fallback 审计，但 production-like report 必须明确记录是否启用、是否实际 fallback。位置：`nikon0/eval/run_agent_eval.py:233-237`。

### 1.5 哪些 mock 会污染评测可信度

- `MockSkill` 不能出现在生产默认 runtime，也不能参与 production-like eval。它是“任何非空消息 0.5 兜底”的 Phase1 skill。位置：`nikon0/skills/mock_skill.py:31-38`、`nikon0/skills/mock_skill.py:40-65`。
- `_EvalCaseIntakeTool` 会污染真实 case-intake / refund / handoff 的上线可信度，因为它不经过真实 MCP/工单系统，只按关键词和“型号+电话”返回固定 payload。位置：`nikon0/eval/run_agent_eval.py:286-347`。
- eval 默认 StructuredManual fallback 会让 enterprise RAG 故障被掩盖；虽然当前 report 有 `rag_fallback_rate`，但 runtime profile 未记录 fallback policy。位置：`nikon0/eval/run_agent_eval.py:233-237`、`nikon0/eval/run_agent_eval.py:471-484`。

### 1.6 当前 eval 指标可信度问题

- `EvalCaseResult.fact_coverage_score` 和 `evidence_alignment_score` 默认值为 1.0；非 product_support case 或缺少检查项时天然满分，不能代表全链路事实质量。位置：`nikon0/eval/run_agent_eval.py:63-64`、`nikon0/eval/run_agent_eval.py:413-433`。
- `_score_case()` 对 product_support 强制要求 enterprise RAG ok 且无 fallback，但 report 没有说明 runtime 是否 production-like、是否 local_rag、是否真实 LLM。位置：`nikon0/eval/run_agent_eval.py:389-402`。
- 现有 `fallback_rate` 和 `guard_rejection_rate` 是从 selection source/selected_skill 推导，不是从统一 trace event 汇总，语义偏窄。位置：`nikon0/eval/run_agent_eval.py:606-607`。
- 当前 agent eval 没有纳入 context section miss、evidence omitted、tool observation omitted、LLM context component usage 等指标；这些只在 `context_eval.py` 里独立存在。位置：`nikon0/eval/context_eval.py:101-134`、`nikon0/eval/context_eval.py:511-630`。
- `agent_eval_150` 当前不能直接作为后续上线改造基线，因为 architecture review 指出 runtime/eval 不一致。位置：`docs/architecture_review/01_architecture_review.md:51-52`、`docs/architecture_review/03_next_3_day_plan.md:202-205`。

### 1.7 context eval 与 agent eval 的关系

- `context_eval.py` 有直接 context module eval 和 end-to-end stress eval 两种模式。位置：`nikon0/eval/context_eval.py:337-456`。
- context direct eval 自建 `ContextRuntime()` 或使用传入 `context_runtime`，不通过生产 `_build_default_context_governance()`。位置：`nikon0/eval/context_eval.py:337-356`、`nikon0/eval/context_eval.py:459-476`。
- context stress eval 复用 `build_eval_runtime()`，因此继承 agent eval 的 runtime/profile 不一致问题。位置：`nikon0/eval/context_eval.py:394-407`。
- agent eval 已经从 response.debug 里可以拿到 `context_debug`，但没有把 context 指标聚合进 `EvalRunReport`。位置：`nikon0/agent/runtime.py:217-230`、`nikon0/eval/run_agent_eval.py:176-200`。

---

## 2. Alignment Goals

P0 目标：

- 生产默认不注册 `MockSkill`；若需要调试，通过 `NIKON0_ENABLE_MOCK_SKILL=true` 显式开启，默认 false。
- eval runtime 支持 runtime profile，默认切到 `production_like`，并能显式选择 deterministic/legacy。
- `production_like` eval 使用与生产一致的 ContextGovernance 构造路径，即复用 `_build_default_context_governance()` 或同等 profile builder。
- eval report 必须记录 `runtime_profile`、`context_profile`、`mock_skill_enabled`、`mock_tool_usage_count`、`context_governance_enabled`、`llm_context_components_enabled`、`eval_runtime_matches_production`。
- agent eval 聚合 context 指标：`context_section_miss_rate`、`evidence_omitted_rate`、`tool_observation_omitted_rate`。
- mock case-intake tool 的使用必须进入 report，而不是隐性混入结果。
- `agent_eval_150` 能产出一份 aligned baseline，用于后续 Memory Write Gate 的对照。

P1 目标：

- 增加 `fallback_rate_by_backend`，区分 enterprise RAG fallback、LLM fallback、skill fallback、tool fallback。
- context eval 和 agent eval 使用同一个 runtime profile 入口。
- eval answers.jsonl 每行记录 runtime/context/tool/mock 审计字段，便于失败 case 归因。

---

## 3. Non-goals

- 不实现 Memory Write Gate。
- 不重构 SafetyGate。
- 不做真实 RBAC / tenant scope / quota。
- 不做真实工单系统。
- 不迁移 JSONL trace/transcript/approval storage。
- 不改变 RAG 建库、embedding、Milvus schema。
- 不引入新外部依赖。
- 不在本阶段改评测数据集内容，除非为 profile metadata 添加非行为字段。
- 不把 `_EvalCaseIntakeTool` 立即删除；只把它 profile 化、显式化、可审计化。

---

## 4. Proposed Code Changes

### `app/core/config.py`

- Change 1：新增 `nikon0_enable_mock_skill: bool = Field(default=False, alias="NIKON0_ENABLE_MOCK_SKILL")`。
- Why：生产默认不应挂 Phase1 `MockSkill`，但保留显式调试开关。
- Risk：现有少量测试可能默认期待 MockSkill，需要改成显式构造。
- Test：新增/更新测试断言生产默认 skill 列表不包含 `mock_enterprise_assistant`，开启开关时才包含。

### `nikon0/agent/runtime.py`

- Change 1：修改 `_build_default_skills()`，默认不注册 `MockSkill`；读取 `NIKON0_ENABLE_MOCK_SKILL` 后才追加。
- Change 2：将 `_build_default_context_governance()` 保持为 production context profile 的唯一来源；必要时导出更明确的 profile metadata helper。
- Change 3：增加 runtime introspection helper，例如 `describe_runtime(runtime, profile_name)` 或在新文件中实现，避免 eval 用私有推断散落。
- Why：生产与 eval 需要共享构造逻辑，减少“测假的”。
- Risk：关闭 MockSkill 后，未命中业务 skill 的回复会走 Supervisor general fallback，而不是 MockSkill 文案；这是预期行为，但旧测试需更新。
- Test：`test_default_runtime_excludes_mock_skill_by_default`、`test_default_runtime_can_enable_mock_skill_with_flag`、`test_default_context_governance_uses_llm_components_when_enabled` 已有类似基础可扩展。

### `nikon0/eval/runtime_profiles.py`（新增）

- Change 1：新增 `EvalRuntimeProfile` 枚举/配置模型：`deterministic`、`production_like`、`production_like_no_llm`、`legacy_eval`。
- Change 2：新增 `build_runtime_for_eval(profile, manual_dir, local_rag, use_real_llm, use_mock_case_intake_tool)`。
- Change 3：新增 `RuntimeProfileAudit` / `ContextProfileAudit` 数据结构，用于 report。
- Why：把 runtime 构造差异显式化，避免 CLI boolean 越堆越多。
- Risk：需要保持 `build_eval_runtime()` 向后兼容，避免现有测试/命令断裂。
- Test：新增 `test_eval_runtime_profiles.py` 覆盖各 profile 的 context、LLM、mock tool、MockSkill 行为。

### `nikon0/eval/run_agent_eval.py`

- Change 1：`build_eval_runtime()` 保留但内部转调 `runtime_profiles.build_runtime_for_eval()`；默认 profile 逐步切到 `production_like`。
- Change 2：`run_agent_eval()` 新增参数 `runtime_profile`、`context_profile` 或统一 `profile`。
- Change 3：`EvalRunReport` 增加 P0 profile/audit 字段。
- Change 4：`EvalCaseResult` / `EvalTurnResult` 增加 context audit 字段：section_names、missing_expected_sections、evidence_section_present、tool_observations_present、llm_context usage。
- Change 5：聚合 `context_section_miss_rate`、`evidence_omitted_rate`、`tool_observation_omitted_rate`、`mock_tool_usage_count`。
- Change 6：`answers.jsonl` 每行写入 `runtime_audit`、`context_audit`、`mock_tool_usage`、`rag_audit`。
- Change 7：CLI 增加 `--runtime-profile`；保留 `--no-real-llm`、`--local-rag`，但在 markdown 中明确它们如何影响 profile。
- Why：report 本身必须说明“这次评测代表什么环境”。
- Risk：metrics schema 扩展会影响读取旧 report 的脚本；使用新增字段保持向后兼容，不删除旧字段。
- Test：更新 `test_agent_eval_runner_outputs_answers_metrics_and_failures` 断言 profile 字段；新增 mock usage 统计测试。

### `nikon0/eval/context_eval.py`

- Change 1：stress eval 改为接受 `runtime_profile` 并复用 `runtime_profiles.build_runtime_for_eval()`。
- Change 2：direct context eval 增加 `context_profile` 参数，可选 deterministic/production_like。
- Change 3：`ContextEvalReport` 增加 runtime/context profile 字段，与 agent eval 对齐。
- Why：context eval 不能和 agent eval 继续割裂，否则 context 指标无法作为上线门禁。
- Risk：真实 LLM context profile 会让 direct context eval 非确定性增强；保留 deterministic profile 做稳定单测。
- Test：更新 `test_context_eval_runner_outputs_metrics_and_debug_report`，新增 `test_context_stress_uses_profile_runtime`。

### `nikon0/eval/agent_metrics.py`

- Change 1：若仍保留旧 harness，补充 profile 字段或标记为 legacy。
- Why：避免两个 eval harness 产生互相矛盾的“权威指标”。
- Risk：低。该文件目前偏旧 harness。
- Test：更新 `test_agent_evaluation_harness_outputs_metrics` 或明确 legacy usage。

### `nikon0/app/test/test_agent_eval_runner.py`

- Change 1：新增/更新 profile 相关断言。
- Change 2：保留 deterministic/local_rag 测试，作为稳定 baseline。
- Change 3：新增 production_like profile 使用 EnterpriseRagBackend、生产 ContextGovernance、无 MockSkill 的断言。

### `nikon0/app/test/test_skill_routing.py`

- Change 1：当前已有 `test_eval_runtime_excludes_mock_skill`，扩展为 production default 也 excludes mock。
- Change 2：保留 `ROUTING_EXCLUDED_SKILLS` 测试。

---

## 5. Runtime Profile Design

### `deterministic`

- ContextGovernance：启用。
- LLM read planner：关闭。
- LLM compactor：关闭。
- LLM span selector：关闭。
- Answer LLM：默认关闭，可通过 `use_real_llm=True` 显式开启。
- Skill router LLM：默认关闭。
- Case-intake tool：使用 `_EvalCaseIntakeTool`。
- MockSkill：不允许。
- RAG：默认 EnterpriseRagBackend + StructuredManual fallback；可 `--local-rag`。
- 适用场景：快速稳定回归、CI、小样本开发验证。

### `production_like`

- ContextGovernance：启用，复用 `_build_default_context_governance()`。
- LLM read planner：按生产配置启用。
- LLM compactor：按生产配置启用。
- LLM span selector：按生产配置启用。
- Answer LLM：默认启用，使用生产配置。
- Skill router LLM：按生产配置启用。
- Case-intake tool：默认使用生产 ToolRuntime/default_tools/MCP 路径；如果显式 `--mock-case-intake-tool` 才使用 eval mock，并在 report 标记。
- MockSkill：不允许，除非显式 `NIKON0_ENABLE_MOCK_SKILL=true`，但 report 必须标红/标记 `eval_runtime_matches_production=false`。
- RAG：默认 EnterpriseRagBackend，fallback 行为与生产尽量一致；若为了本地可跑加入 StructuredManual fallback，report 必须记录。
- 适用场景：上线前基线、回归门禁、真实行为评估。

### `production_like_no_llm`

- ContextGovernance：启用，使用生产结构但强制 deterministic context components。
- LLM read planner：关闭。
- LLM compactor：关闭。
- LLM span selector：关闭。
- Answer LLM：关闭。
- Skill router LLM：关闭。
- Case-intake tool：默认生产 ToolRuntime；可显式 mock 并记录。
- MockSkill：不允许。
- RAG：EnterpriseRagBackend。
- 适用场景：排除 LLM 随机性、验证工具/RAG/memory/trace 结构是否一致。

### `legacy_eval`

- ContextGovernance：使用当前 eval 默认裸 `ContextGovernance()`。
- LLM context components：关闭。
- Answer LLM：遵循旧 `--no-real-llm` 逻辑。
- Skill router LLM：遵循旧 `use_real_llm` 逻辑。
- Case-intake tool：使用 `_EvalCaseIntakeTool`。
- MockSkill：不允许。
- RAG：当前 eval 逻辑，即 EnterpriseRagBackend + StructuredManual fallback 或 `--local-rag`。
- 适用场景：对比历史报告，解释新 baseline 与旧 baseline 的差异。

---

## 6. Eval Report Schema Changes

### P0 字段

- `runtime_profile: str`
- `context_profile: str`
- `runtime_profile_description: dict`
- `mock_skill_enabled: bool`
- `mock_tool_usage_count: int`
- `mock_tool_names: list[str]`
- `context_governance_enabled: bool`
- `llm_context_components_enabled: dict[str, bool]`
- `eval_runtime_matches_production: bool`
- `rag_backend_policy: dict`
- `case_intake_tool_mode: str`
- `context_section_miss_rate: float`
- `evidence_omitted_rate: float`
- `tool_observation_omitted_rate: float`

### P1 字段

- `fallback_rate_by_backend: dict[str, float]`
- `llm_answer_fallback_rate: float`
- `llm_router_usage_rate: float`
- `context_compaction_rate: float`
- `context_span_selector_usage_rate: float`
- `profile_warnings: list[str]`
- `production_mismatch_reasons: list[str]`

### Case-level P0 字段

- `runtime_profile`
- `context_profile`
- `mock_tool_names`
- `context_section_names`
- `expected_context_sections`
- `missing_context_sections`
- `evidence_section_present`
- `tool_observations_present`
- `llm_context_read_plan_used`
- `llm_context_compactor_used`
- `llm_context_span_selector_used`

---

## 7. Tests to Add or Update

### `nikon0/app/test/test_runtime_profiles.py`（新增）

- `test_default_runtime_excludes_mock_skill_by_default`
  - 断言 `_build_default_skill_registry()` / `build_default_runtime()` 默认不含 `mock_enterprise_assistant`。
- `test_default_runtime_can_enable_mock_skill_with_flag`
  - monkeypatch settings，显式开启后才包含 MockSkill。
- `test_production_like_eval_profile_uses_default_context_governance`
  - 断言 production_like runtime 的 context_runtime component 类型与 `_build_default_context_governance()` 一致。
- `test_deterministic_profile_uses_deterministic_context`
  - 断言 deterministic profile 使用 `DeterministicContextReadPlanner`、`ConversationCompactor`、无 LLM span selector。
- `test_production_like_profile_disallows_mock_skill`
  - 断言 production_like 不包含 MockSkill。

### `nikon0/app/test/test_agent_eval_runner.py`（更新）

- 更新 `test_agent_eval_runner_outputs_answers_metrics_and_failures`
  - 增加 `runtime_profile`、`context_profile`、`mock_tool_usage_count`、`eval_runtime_matches_production` 字段断言。
- 新增 `test_eval_report_records_mock_case_intake_tool_usage`
  - 使用 deterministic/legacy profile，跑 case_intake case，断言 mock usage > 0。
- 新增 `test_eval_production_like_report_records_context_profile`
  - production_like profile 下 report 有 context LLM component audit。
- 保留 `test_eval_runtime_uses_enterprise_rag_by_default`、`test_eval_runtime_keeps_local_rag_as_explicit_baseline`，但补充 profile 参数。

### `nikon0/app/test/test_context_eval.py`（更新）

- `test_context_stress_runner_outputs_debug_report`
  - 增加 profile 字段断言。
- 新增 `test_context_eval_can_use_production_like_context_profile`
  - 断言 direct context eval 可以复用 production-like ContextRuntime。

### `nikon0/app/test/test_skill_routing.py`（更新）

- 更新 `test_eval_runtime_excludes_mock_skill`
  - 对所有非 legacy profile 断言 excludes mock。
- 新增 production default excludes mock 测试。

### 最小命令验收

- `conda run -n kefu pytest -q nikon0/app/test/test_runtime_profiles.py nikon0/app/test/test_agent_eval_runner.py nikon0/app/test/test_context_eval.py nikon0/app/test/test_skill_routing.py`
- `conda run -n kefu pytest -q nikon0/app/test`
- aligned baseline：
  - `conda run -n kefu python -m nikon0.eval.run_agent_eval --runtime-profile production_like --dataset /Users/nikonzhang/compeletion/nikon0/eval/datasets/agent_eval_150.jsonl --output-dir /Users/nikonzhang/compeletion/nikon0/eval/reports --run-id agent-eval-150-aligned --progress`

---

## 8. Migration / Backward Compatibility

- 保留现有 `run_agent_eval()` 参数：`use_real_llm`、`local_rag`、`runtime`。
- CLI 保留 `--no-real-llm`、`--local-rag`、`--progress`。
- 新增 `--runtime-profile`，默认建议改为 `production_like`；若担心历史 CI 波动，可短期默认 `legacy_eval`，但文档和命令中明确 aligned baseline 必须使用 `production_like`。
- `build_eval_runtime()` 保留原函数名，内部转调 profile builder，避免现有测试和调用方断裂。
- `_EvalCaseIntakeTool` 不删除，改为 profile/tool mode 的显式选择，并在 report 中记录。
- 旧 report schema 只新增字段，不删除旧字段，保证历史对比脚本可以继续读取基础指标。
- 旧 deterministic eval 命令可迁移为：
  - `--runtime-profile deterministic --no-real-llm --local-rag`
- 旧真实 LLM eval 可迁移为：
  - `--runtime-profile legacy_eval`
  - 或推荐 `--runtime-profile production_like`

---

## 9. Step-by-step Execution Plan

1. 新增 `nikon0/eval/runtime_profiles.py`
   - 定义 profile enum/config/audit。
   - 实现 deterministic、production_like、production_like_no_llm、legacy_eval 的 runtime builder。

2. 关闭生产默认 MockSkill
   - 在 `app/core/config.py` 增加 `NIKON0_ENABLE_MOCK_SKILL=false`。
   - 修改 `_build_default_skills()` 默认不追加 MockSkill。

3. 修改 eval runtime 构建
   - `build_eval_runtime()` 转调 profile builder。
   - `run_agent_eval()` 增加 `runtime_profile` 参数。
   - CLI 增加 `--runtime-profile`。

4. 修改 eval report schema
   - `EvalRunReport` 增加 profile/audit P0 字段。
   - `EvalCaseResult` / `EvalTurnResult` 增加 context/mock audit 字段。
   - `metrics.md` 输出 profile 和 mismatch warnings。

5. 增加 context/evidence/tool observation 指标
   - 从 `response.debug["context_debug"]` 和 `trace.events` 提取 section names、LLM context component usage。
   - 聚合 `context_section_miss_rate`、`evidence_omitted_rate`、`tool_observation_omitted_rate`。

6. 修改 context eval
   - `run_context_stress_eval()` 接收 runtime profile。
   - direct context eval 支持 context profile。
   - `ContextEvalReport` 增加 profile 字段。

7. 增加/更新测试
   - 先测 profile builder。
   - 再测 report schema。
   - 再测 CLI/runner 小数据集。

8. 跑最小测试
   - 先跑新增和相关测试。
   - 再跑全量 `nikon0/app/test`。

9. 跑 aligned baseline
   - 用 `agent_eval_150` 产出 `agent-eval-150-aligned`。
   - 报告中必须能看到 `runtime_profile=production_like` 和 profile audit。

---

## 10. Acceptance Criteria

- 生产默认 `MockSkill` disabled。
- 显式 `NIKON0_ENABLE_MOCK_SKILL=true` 时才注册 MockSkill。
- `production_like` eval 与生产 ContextGovernance 构造路径一致。
- `deterministic` profile 仍可用于稳定回归。
- eval report 中可以看到：
  - `runtime_profile`
  - `context_profile`
  - `mock_skill_enabled`
  - `mock_tool_usage_count`
  - `context_governance_enabled`
  - `llm_context_components_enabled`
  - `eval_runtime_matches_production`
- agent eval 聚合 context 指标：
  - `context_section_miss_rate`
  - `evidence_omitted_rate`
  - `tool_observation_omitted_rate`
- mock case-intake tool 的使用可在 report 和 answers.jsonl 中审计。
- `agent_eval_150` 可以产出 aligned baseline。
- legacy/deterministic eval 仍可运行。
- 所有新增测试通过，全量 `nikon0/app/test` 通过。

---

## Recommended Stage 2 Scope

阶段 2 建议只做 P0，不做 P1：

- runtime profile builder
- 生产默认关闭 MockSkill
- eval/context eval 接入 profile
- report 增加 profile 和 context/mock 审计字段
- 新增测试
- 跑 aligned baseline

P1 的 fallback 分解、LLM fallback rate、context compaction rate 等可在 aligned baseline 稳定后继续补。
