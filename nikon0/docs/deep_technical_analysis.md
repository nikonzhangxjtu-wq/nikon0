# nikon0 深度技术分析报告

> 本报告基于对 nikon0 全部约40个核心源文件的完整阅读形成。分析范围覆盖 Agent Kernel、Skill/Tool/Runtime、Context Governance、Memory、Safety、Knowledge、Workflow、Eval 等全部模块。报告聚焦设计细节、当前不足和可改进方向，不涉及对代码的修改。

---

## 一、总体架构评价

### 1.1 架构亮点

nikon0 的核心架构思想是"自研 Agent Kernel + 能力注册表 + 工具执行管线 + trace/transcript 分离"，这是一个成熟的工程选择：

- **不绑定外部框架**：没有依赖 LangChain/LangGraph 作为核心编排，Runtime 完全自研，给予了最大的控制力和灵活性。
- **清晰的分层抽象**：API Layer → Agent Kernel → Context Governance → Planner → Orchestrator → Act → Verify → Answer → Update 的流水线清晰，每层职责明确。
- **能力注册表快照设计**：AgentRegistry、SkillRegistry、ToolRegistry 在每轮执行开始时生成不可变快照，避免运行中动态增减能力导致行为不可预测。
- **Trace 为一等公民**：ExecutionTrace 不是调试附属品，而是从设计之初就作为核心数据结构，支持评测和回放。
- **Context Governance 独立分层**：上下文治理不是简单的 prompt 拼接，而是有预算管理、证据保真、压缩策略的完整模块。

### 1.2 架构待改进点

1. **多 Agent 协作未实现**：设计文档描述了 SupervisorAgent → SpecialistAgent 的委派模型，但当前只实现了 SupervisorAgent。所有子 Agent（ProductSupportAgent、CaseIntakeAgent、OrderAgent 等）仅作为架构占位存在。

2. **Planner 过于简陋**：RuleBasedPlanner 是基于关键词匹配的确定性路由器，无法处理复杂语义。例如用户输入"我的AC900昨天还能用，今天开机就显示E2，已经试过重启和清洁滤网，还是不行"，planner 仅会匹配到 product_support（因为含"故障"、"E2"关键词），但无法识别"已经尝试过重启和清洁"这一重要信息。

3. **Skill 选择优先级可能产生冲突**：`_recommend_skill` 中 case_intake 优先级为0（最高），这意味着"报修"信号会覆盖 product_support 信号。对于复合意图"AC900 显示 E2，需要报修"，用户可能是想先诊断再决定是否报修，但系统直接走了 case_intake 流程。

4. **缺乏动态上下文重规划**：一轮执行中，Context Governance 只在开始时组装一次上下文。当工具执行后产生大量结果时，没有重新评估上下文预算分配。

---

## 二、模块逐一深度分析

### 2.1 AgentRuntime (`agent/runtime.py`)

#### 设计细节
- `build_default_runtime()` 函数是整个系统的工厂入口，从环境变量读取配置，构建完整的运行时。
- 运行时支持注入所有依赖（registry、store、recorder），这为测试提供了极好的可测试性。
- `max_turns` 硬限制为4轮，防止无限循环。

#### 当前不足
1. **错误恢复粒度粗**：当 skill 抛异常时，SupervisorAgent 的异常处理给出了通用错误消息，但没有区分可恢复错误（如临时网络超时）和不可恢复错误（如配置错误）。
2. **run() 方法过长**（约150行）：包含了 trace 初始化、transcript 追加、session state 加载、context 构建、planner 调用、loop 执行、safety 检查、answer 组装、memory 更新、transcript 追加、trace 持久化、actions 构建。可以考虑拆分为更小的私有方法。
3. **answer 降级链隐式**：当 safety 阻止、result 无 answer_draft、general_handle 三条路径的降级逻辑分散在 run() 方法中，不易追踪和测试。
4. **tool_runtime 注入到 context 但有循环依赖风险**：AgentContext 持有 tool_runtime 引用（Any 类型），然后 Skill.run() 内部通过 `context.tool_runtime.call_step()` 使用。这意味着 Skill 可以任意调用工具，而不受 AgentRuntime 的精细控制。

#### 改进方向
- 引入 `AnswerComposer` 独立模块，统一所有降级路径。
- 将 run() 拆分为 `_init_trace()`, `_load_context()`, `_execute_loop()`, `_apply_safety()`, `_compose_answer()`, `_finalize()`。
- 限制 Skill 对 ToolRuntime 的访问路径：让 ToolRuntime 只在 AgentLoop 中被调用，Skill 只产出 ToolCallRequest。

---

### 2.2 AgentLoop (`agent/loop.py`)

#### 设计细节
- 受 claw-code query.ts 启发，采用 plan/act → observe tools → repeat 的最小循环。
- 每个 turn：选择 Agent → 执行 Agent → 如果没有 tool_calls 就退出，否则执行工具并继续下一轮。
- 支持 tool error 重试（当 `context.retry_tool_errors` 为 True 时）。

#### 当前不足
1. **循环终止条件单一**：仅当 agent 不产出 tool_calls 或达到 max_turns 时停止。缺乏"工具已全部成功"、"证据已足够"、"用户不需要更多操作"等更细腻的终止条件。
2. **没有工具结果去重**：如果第二轮的工具调用产生了和第一轮相同的结果，loop 不会跳过重复执行。
3. **retry_tool_errors 逻辑粗糙**：仅重试一次，且重试时不修改参数（如超时时间），可能重试相同的失败。
4. **工具调用是串行的**：`for tool_call in last_result.tool_calls` 是顺序执行，如果 skill 产出多个独立的工具调用，它们不能并发执行。
5. **loop 不检查 context 变化**：loop 在多轮之间不检查 context 是否有效（如 budget 是否已耗尽）。

#### 改进方向
- 引入 `LoopPolicy`：定义更灵活的终止条件（如 max_tool_calls、min_evidence_count）。
- 工具调用去重：通过 `(service_id, tool_name, arguments_hash)` 判断是否已执行过。
- 并行执行无依赖的工具调用。

---

### 2.3 RuleBasedPlanner (`agent/planner.py`)

#### 设计细节
- 基于硬编码的中文关键词匹配，识别 case_intake、refund、complaint、product_support、tool_echo 五种意图。
- 支持复合意图检测（is_composite）。
- 优先级排序：case_intake(0) > tool_echo(1) > product_support(2)。

#### 当前不足
1. **关键词列表硬编码且不完整**：
   - `product_support` 关键词包含"洗碗机"、"空气净化"、"摩托艇"、"相机"、"单反"、"拍立得"这些具体产品名，但缺少大量其他产品（如"空调"、"微波炉"、"洗衣机"出现在 CUSTOMER_SERVICE_KEYWORDS 周围的条件下才能命中）。
   - 故障码仅硬编码了"e2"、"E2"，没有覆盖其他常见故障码（E1、E3、E4、E5、F0、H1等）。
   - "怎么处理"在第44行，但'怎么处理'需要和前面关键词一起才会生效——单独的"怎么处理"不会触发。
   
2. **复合意图处理存在冲突**：当用户输入"AC900 显示 E2，需要退款"时，同时产生 product_support 和 refund 意图，但推荐的 skill 是 case_intake（因为 refund 的优先级更高，而 refund 被路由到 case_intake）。用户可能想要的是"先了解 E2 是什么问题，再决定是否退款"。

3. **置信度是硬编码的**：所有匹配的置信度都是固定值（0.9、0.88、0.82、0.95），不是基于实际匹配关键词数量的动态计算。

4. **不支持英文用户**：大量中文硬编码关键词使得英文用户或中英混合输入几乎无法被正确路由。

5. **没有模糊匹配**："我的设备坏了"中的"坏了"可以匹配，但"有问题"、"异常"、"不正常"、"出毛病"等口语表达无法匹配。

6. **`needs_general_handle` 语义不清**：当 `recommended_skill is None` 时为 True，但这包括了"所有 skill 都无法处理"和"匹配了 planner 但 registry 中没有对应 skill"两种情况，这两种情况的处理应该不同。

#### 改进方向
- 将关键词提取为可配置的资源文件（如 YAML/JSON），支持热更新。
- 引入轻量级 NLP（如 jieba 分词 + TF-IDF 向量相似度）替代纯关键词匹配。
- 支持英文和多语言。
- 为复合意图设计显式的处理策略（如"先解答技术问题，再询问是否需要售后"）。

---

### 2.4 SupervisorAgent (`agent/supervisor.py`)

#### 设计细节
- 是当前唯一的 Agent 实现，承担了 skill 选择、执行、回退、答案生成的全部职责。
- `_general_handle()` 在 LLM 可用时使用 LLM 生成通用回复，否则使用硬编码的 fallback。
- `_sticky_turn_update()` 实现了 sticky policy 的计数器更新。
- `_apply_fallback_policy()` 根据 skill 的 fallback_policy 决定失败后的行为。

#### 当前不足
1. **`can_handle()` 永远返回 True**：SupervisorAgent 总是声明自己能处理任何请求，这意味着 AgentRegistry.best_match() 不会选择其他 Agent（因为只有 SupervisorAgent 一个）。
2. **`_general_handle` 的 fallback_answer 过于冗长**：`"nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。"` 包含了太多内部实现细节。
3. **confidence_threshold 机制存在不一致**：全局阈值 0.75，但 product_support 被降低到 0.55。这意味着 model selector 只需 0.55 的置信度即可选择 product_support，但 planner/rule-fallback 需要 0.75。这可能导致路由行为不一致。
4. **`_conflict_resolution` 缺失**：当 planner 推荐 case_intake 但 model selector 推荐 product_support 时，当前逻辑是 sticky > model > planned > rule_fallback 的优先级链，没有冲突解决机制。
5. **skill 异常处理过于宽泛**：`except Exception as exc: # noqa: BLE001` 捕获了所有异常，包括 KeyboardInterrupt 和 SystemExit。

#### 改进方向
- 将 answer 生成从 SupervisorAgent 中移出，变为独立的 `AnswerComposer`。
- 引入显式的冲突解决策略而非隐式优先级链。
- 限制异常捕获范围（至少排除 BaseException 子类）。

---

### 2.5 SkillRegistry (`skills/base.py`)

#### 设计细节
- 四层选择策略（优先级从高到低）:
  1. **sticky**: 多轮对话中保持同一 skill
  2. **model**: LLM 驱动的 skill 选择
  3. **planned**: Planner 推荐的 skill
  4. **rule_fallback**: 每个 skill 的 `can_handle()` 守卫
- Sticky policy 支持 `continue_when`、`exit_when`、`max_turns`、`priority` 四个维度。
- 工具依赖验证：如果选中的 skill 需要的工具未在 ToolRuntime 中注册，则拒绝该 skill。
- `SkillSelection` 记录了完整的选择过程（candidates、rejected、source、confidence）。

#### 当前不足
1. **`_sticky_match` 和 `_model_match` 的验证时机**：sticky 和 model 路径先选中 skill，再调用 `_validate_selected_skill` 检查工具依赖。但 `_model_match` 中模型选中未知 skill 会直接修改 selection 的 selected_skill 为 None，这个修改后的 selection 不会被传递给后续的 `_planned_match` 逻辑。
2. **sticky state 读取依赖于 session_state 的 flat_state**：`_sticky_status()` 从 `session_state.flat_state[skill_name]["status"]` 读取状态。这个路径假设所有 skill 都在 flat_state 中以 `{skill_name: {"status": ...}}` 格式存储状态。如果状态格式不匹配（如直接存储字符串），sticky 逻辑会静默失败。
3. **`_sticky_turns` 计数器只增不减**：`SupervisorAgent._sticky_turn_update` 只增加计数器，但从不在退出 sticky 时重置。这可能导致跨会话的计数污染。
4. **selector 的 `build_selection` 方法**：它允许任意构造 SkillSelection，绕过了 model/planner/rule 的正常路径。这在测试中有用，但如果被误用，可能导致不安全的路由。
5. **`select_best` 中 model_match 失败后的处理**：当 model 选中了 unknown skill 时，代码设置了 `source="none"`，但没有清除之前的 sticky 状态。如果之前有 sticky，sticky 应该在 model 失败后被回退。

#### 改进方向
- sticky_turns 在 sticky 退出时重置。
- 当 model 选中 unknown skill 时，记录 warn 级别的事件到 trace。
- 考虑将 selector 接口从 `ManifestDrivenSkillSelector` 重构为更灵活的形式。

---

### 2.6 ProductSupportSkill (`skills/product_support.py`)

#### 设计细节
- 完整的工具编排链：resolve_product → search_product_manual → (可选: retrieval-evidence product resolution) → LLM answer generation → validate_answer_grounding。
- 产品消歧：当 `resolve_from_retrieval` 无法区分时，返回 `disambiguation_required` 状态，提示用户在候选产品中选择。
- 证据过滤：检索结果可以根据 resolved product 的 manual_names 进行过滤。
- `apply_product_disclosure` 在回答前加上产品透明声明。

#### 当前不足
1. **在 skill 内部重新创建 ContextGovernance**（第210行）：`ContextGovernance().govern(context)` 创建了一个新的、默认配置的 ContextGovernance 实例，覆盖了 Runtime 中配置的治理策略。这意味着 Runtime 级别的 budget、section 配置被丢失了。
2. **answer_generator 为 None 时无 LLM 回答**：fallback_answer 是"根据当前商品手册证据，建议如下："+ 直接拼接 evidence 文本，对于复杂问题回答质量很低。
3. **validate_answer_grounding 仅做 token overlap 检测**：`_meaningful_overlap()` 只检查答案和证据的 token 交集，无法判断语义一致性（如 "E2 表示滤网堵塞" vs "滤网有问题"）。
4. **图片处理不完整**：虽然将 `context.request.images` 传给了 SearchProductManualTool，但没有对图片内容做独立的理解和处理。Multimodal 能力仅停留在"传递图片列表"层面。
5. **证据与答案的一致性验证缺失**：validate_answer_grounding 使用了 `required_terms=[]`，空列表意味着跳过任何必需术语的检查。
6. **`_compose_answer` 模板过于通用**：没有利用 product_context 中的产品名称来定制 preamble。

#### 改进方向
- 不要在 skill 内部创建新的 ContextGovernance。
- 为 answer grounding 增加 LLM-as-judge 验证。
- 实现真正的 multimodal 理解（图片 → 结构化故障描述）。
- 设计更丰富的 fallback 答案模板（区分"无证据"、"证据不匹配"、"证据冲突"等场景）。

---

### 2.7 CaseIntakeSkill (`skills/case_intake.py`)

#### 设计细节
- 使用 MCP 工具进行工单收集，不在本地实现业务逻辑。
- 支持四种 workflow：repair_intake、refund_intake、complaint_escalation、case_intake_cancel。
- Sticky policy 配置：`continue_when=["collecting"]`、`exit_when=["ready", "cancelled"]`、`max_turns=6`。
- 取消意图检测通过关键词匹配。
- 风险判断：`_is_risky_intake()` 根据 ticket_payload 中的 intent 和 priority 判断。

#### 当前不足
1. **首轮不产生答案**（第177行）：`answer_draft=""` —— 第一轮 CaseIntakeSkill 只产出 ToolCallRequest，没有给用户任何即时反馈。用户会看到空白回答，直到第二轮 tool result 返回后才得到回复。
2. **`_latest_case_intake_tool_result` 的匹配逻辑依赖于 service_id + tool_name**：如果 tool result 是其他 service 产生的，即使包含了 case intake 数据，也不会被识别。
3. **取消逻辑有竞态条件**：`_should_cancel` 依赖于 `_has_pending_intake` 返回 True，但如果用户在 intake 未开始时就输入"算了"，取消不会生效。
4. **`_workflow_decision` 中对 intent="unknown" 的处理**：当 extract_case_slots 返回 intent="unknown" 时，workflow_runtime.decide 会 fallback 到 repair 协议。但未知意图的用户可能根本不是要报修。
5. **Workflow Decision 中的关键词覆盖逻辑不安全**（workflows/runtime.py 第59-64行）：当 extract_case_slots 返回 intent="repair" 但消息中包含"赔偿"关键字时，会被重新判定为 complaint。这种基于消息文本的二次判断与 extract_case_slots 的 intent 分类可能冲突。
6. **`_cancel_keywords` 过于宽泛**："算了"在很多中文语境中可能只是表达无奈（如"算了，我再试试"），而不是真的取消。
7. **`_MANUAL_PROHIBITION_PATTERNS` 中的"为什么不能用排插"等模式**：这些模式同时出现在 case_intake 的 exclusion 和 product_support 的 routing signals 中，但两个模块对同一模式的处理逻辑不一致。

#### 改进方向
- 第一轮就给出"正在为你收集信息..."的即时反馈。
- 取消操作需要二次确认（"确定要取消当前工单收集吗？"）。
- 将 intent 判断集中到一个独立的 IntentClassifier 模块，避免多处重复判断。

---

### 2.8 ToolRuntime (`tools/runtime.py`)

#### 设计细节
- 完整的工具执行管线：schema validation → permission check → safety preflight → MCP Gateway call → result normalization → post verification → trace record。
- HookRunner 支持 pre_tool、post_tool、on_failure 三类 hook。
- ToolPermissionPolicy 基于 risk_level 和 requires_approval 做访问控制。
- default_tools() 从 MCP Gateway 动态发现工具，失败时 fallback 到静态定义的 McpGatewayTool。

#### 当前不足
1. **`schema validation` 未实现**：ToolCallRequest 的 arguments 不做 schema 校验就直接传给工具。文档中描述的"schema validation"步骤在代码中不存在。
2. **没有超时控制**：工具调用没有 timeout 机制，一个慢响应的 MCP 服务可能阻塞整个 AgentLoop。
3. **没有熔断/降级**：文档描述的"circuit breaker / fallback"机制未实现。
4. **pre_tool hook 的决策没有累积**：如果第一个 pre_hook 返回 allowed=True 但第二个返回 allowed=False，最终决策采用第二个。但如果第一个 hook 返回了带有特定 reason 的拒绝，这个信息在最终决策中被丢失。
5. **`ToolPermissionPolicy.check` 中 `requires_approval=True` 总是返回拒绝**：因为真正的审批流程（HITL）还未实现，所以任何需要审批的工具调用都被直接拒绝。这可能在生产中拦截了合法的高风险但有审批流程的操作。
6. **`call_step` 将结果直接追加到 context.tool_results**：如果同一个 tool 被 skill 多次调用（如在 loop 的不同 turn），context.tool_results 会累积，可能导致内存膨胀。
7. **MCP Gateway 发现失败时的 fallback 工具是硬编码的**：`default_tools()` 中 hardcode 了 case-intake 的三个工具名，如果 MCP Gateway 提供的工具名变了，fallback 不会更新。

#### 改进方向
- 实现基于 JSON Schema 的入参校验。
- 添加 `asyncio.wait_for(timeout=...)` 超时控制。
- 实现基于失败计数的简单熔断器。
- 将 MCP fallback 工具定义移到配置文件中。

---

### 2.9 Session Issue Memory (`memory/session.py`, `memory/persistence.py`, `memory/view.py`)

#### 设计细节
- `SessionIssueMemory` 是正式的数据模型：包含 session 级事实、多线程（issue threads）、实体索引、flat_state。
- 更新机制通过 `StateUpdate` 列表驱动：每个 update 由 key、value、reason、evidence_ids 组成。
- Redis + MySQL 双层持久化：Redis 热缓存 + MySQL 审计事件。
- `MemoryViewBuilder` 将内存状态转换为模型可消费的紧凑视图。
- 隐式结构化字段提取：当 update.key 是 "product_support" 或 "case_intake" 时，自动提取 product_model、workflow 等信息填充到 IssueThread 中。

#### 当前不足
1. **flat_state 与结构化字段的双写**：`_apply_update()` 同时向 flat_state 和 thread.facts 写入数据，但两者的一致性没有保证。如果 flat_state 中的 product_support 状态和 thread.product_model 不一致，后续读取方会得到矛盾的信息。
2. **线程管理模型过于简单**：`_ensure_active_thread()` 总是创建新线程当 active_thread 为 None。但在多轮对话中，如果用户切换了话题，旧线程不会被正确关闭。
3. **`_infer_issue_type` 只检查 case_intake update**：对于 product_support 产生的会话，issue_type 始终是 "unknown"。
4. **MemoryView.render() 的格式不够结构化**：输出是 plain text 格式，包含中文标签（如 `active_issue:`、`session_facts:`）。对于 LLM 消费来说，JSON 格式可能更清晰。
5. **Redis TTL 和 MySQL 持久化的一致性**：如果 Redis 中的 session 过期但 MySQL 中的记录还在，下次 load 时从 MySQL 恢复，但这期间的状态可能已经过期。没有"stale session"检测机制。
6. **`_apply_structured_fields` 中的字段映射隐式耦合**：`_apply_product_support_state` 和 `_apply_case_intake_state` 硬编码了 product_support 和 case_intake 的 state 结构。如果这些 skill 的 state 格式发生变化，memory 模块需要同步修改。
7. **缺少跨 session 的用户维度聚合**：虽然设计文档明确说不做长期用户画像，但缺少哪怕是最基本的 session 列表查询（如"这个用户最近有哪些 session"）。

#### 改进方向
- 引入 state schema version 字段，支持状态格式的演化。
- 为 flat_state 和结构化字段实现 reconciliation 机制。
- 添加线程状态机（open → diagnosing → waiting_user → submitted → resolved/cancelled）的显式验证。
- MemoryView 支持 JSON 渲染模式。

---

### 2.10 Context Governance (`context/runtime.py`, `context/evidence.py`, `context/tool_observation.py`)

#### 设计细节
- `ContextRuntime.build_pack()` 组装7个 section：system_policy、workflow、memory、conversation、tool_observations、evidence、current_user、runtime。
- 总字符预算 9000，各 section 有独立预算。
- `EvidenceContextManager` 不默认总结 RAG chunk，优先保留 raw_excerpt。
- `ToolObservationManager` 将 raw tool result 转换为 prompt-safe 的摘要。

#### 当前不足
1. **`_trim_tail` 策略可能存在信息丢失**：当 section 内容超出预算时，保留的是尾部（最新的）内容。但对于 conversation history，这意味着最久远但可能包含关键上下文的消息被丢弃，而不是基于语义重要性的智能裁剪。
2. **`_system_policy` 硬编码为两行文字**：没有根据当前激活的 skill、risk_level、租户配置动态调整。
3. **`_runtime_context` 包含 `available_tool_count` 但不包含工具名称**：LLM 需要知道有哪些工具才能正确规划，但当前只给了工具数量。
4. **EvidenceContextManager 的 `_dedupe_key` 使用文本前160字符的规范化文本**：这可能导致语义相同但表述不同的证据被当作不同证据，或语义不同但前160字符相同的证据被错误去重。
5. **`_apply_budget` 没有考虑 section 之间的优先级**：当总预算不足时，按 sections 列表顺序裁剪，而不是按 priority 保留重要 section。实际上 section 已经按 priority 排序了，但裁剪策略是先到先得，没有考虑全局优化。
6. **没有 ContextVerifier**：架构文档中描述的 "检查关键事实是否仍有证据支撑" 的 ContextVerifier 模块未实现。
7. **没有 PrivacyFilter**：架构文档中描述的 "过滤身份证、手机号、地址等敏感信息" 的 PrivacyFilter 未实现。
8. **TranscriptCompactor 未实现**：多轮后压缩历史的机制不存在，当前是简单的尾截断。

#### 改进方向
- 实现基于语义重要性的 conversation 裁剪（保留关键决策点、用户诉求、系统承诺）。
- 为不同 risk_level 动态注入不同的 system_policy 内容。
- 实现全局优先级感知的 budget 分配（重要 section 优先保留）。

---

### 2.11 SafetyGate (`safety/gate.py`)

#### 设计细节
- 检查三个维度：workflow decision 中的 handoff_required/requires_approval、消息中的关键词、result.risk_level。
- 生成 HandoffRequest 或 ApprovalRequest。
- 审批和转人工状态通过 ApprovalStore 持久化。

#### 当前不足
1. **关键词检测过于简单**：`handoff_words` 和 `approval_words` 是硬编码的字符串列表。只检查消息中是否包含这些词，不考虑上下文。例如用户说"我不需要退款"仍然会触发退款审批（因为"退款"在消息中）。
2. **handoff 和 approval 的优先级判断**：handoff 条件先于 approval 条件判断，如果消息同时包含"投诉升级"和"退款"，只触发 handoff 而不触发 approval。但在实际场景中，可能两者都需要。
3. **没有对 answer_draft 内容的安全性检查**：只检查输入消息和 result.risk_level，不检查生成的回答内容是否包含风险承诺（如"我们一定会给您退款"）。
4. **`_latest_workflow_decision` 只查找最新的 workflow.decision 事件**：如果 loop 中有多轮 workflow 决策（例如先 repair 后升级为 complaint），旧的决策被忽略。
5. **没有安全级别阈值可配置**：哪些词触发 handoff、哪些触发 approval 是硬编码的，不同租户可能有不同的安全策略。

#### 改进方向
- 对 answer_draft 进行内容安全扫描（检查是否包含承诺性语句）。
- 安全关键词列表可配置化，支持租户级别定制。
- 同时触发 handoff 和 approval 时，生成复合 SafetyDecision。

---

### 2.12 KnowledgeRuntime (`knowledge/runtime.py`)

#### 设计细节
- EnterpriseRagBackend 是对现有 Milvus/BM25/rerank 管道的薄适配层。
- 支持权限过滤（按 allowed_manual_names 过滤检索结果）。
- 失败时 fallback 到 StructuredManualBackend（本地 TXT 文件检索）。
- StructuredManualBackend 实现了简单的 token 匹配 + 段落切分。

#### 当前不足
1. **EnterpriseRagBackend 的 retriever 是惰性初始化的**（`_get_retriever`）：VectorRetriever() 的初始化可能涉及 Milvus 连接、模型加载等耗时操作，第一次 query 会阻塞较长时间。
2. **`manual_name_decider` 和 `allowed_manual_names` 的交互不够清晰**：如果 product resolver 已经确定了 allowed_manual_names，就不会调用 manual_name_decider。但如果 allowed_manual_names 有多个（消歧失败），manual_name_decider 也不会被调用。
3. **StructuredManualBackend 的评分函数过于简单**：`_score_passage` 仅计算 token 在文本中出现的次数，没有 TF-IDF 或 BM25 等更有效的排序机制。
4. **`_split_passages` 按段落和句号切割**：对于中文手册，很多内容没有句号而是换行分隔，可能导致合并了不相关的内容。
5. **`_chunk_to_evidence` 中的 confidence 直接使用 chunk.score**：score 和 confidence 的语义不同（一个是检索相关性分数，一个是可信度），直接映射可能不合理。
6. **异步接口是伪异步**：`KnowledgeRuntime.query()` 声明为 async 但实际调用的是同步的 `self.backend.query()`。`StructuredManualBackend.query()` 中的文件 I/O 操作是同步的。

#### 改进方向
- 将 VectorRetriever 的初始化移到 background task 或 connection pool 中。
- 为 StructuredManualBackend 引入 jieba 分词 + TF-IDF。
- 将文件 I/O 改为 asyncio 的 `run_in_executor`。
- 区分 retrieval score 和 evidence confidence，设计独立的置信度评估逻辑。

---

### 2.13 ProductResolver (`knowledge/product_resolver.py`)

#### 设计细节
- 基于 ProductCatalog（JSON 配置）进行产品识别和消歧。
- 支持三种消歧来源：session（会话记忆）、user_choice（用户明确选择）、strong_signal（技术关键词匹配）。
- `resolve_from_retrieval` 通过检索结果中的 manual_names 分布来推断产品。

#### 当前不足
1. **ProductCatalog 的匹配是严格的字符串包含检查**：`_text_hits_any` 使用 `term.lower() in lowered or term in text`，这意味着 "EF-S" 会匹配 "EF-S18-55mm" 但不匹配 "efs"（因为大小写和连字符的差异）。
2. **`_looks_like_product_switch` 的关键词列表不完整**：例如"我说的是"、"我要问的是"、"是另一个"、"不是 AC900" 等常见的产品切换表达未覆盖。
3. **cluster 匹配的优先级问题**：如果用户说"AC900 和 BC200 都显示 E2 怎么办"，`resolve_cluster` 可能会匹配到 AC900 或 BC200 所在的 cluster，但不一定能正确处理多产品共存的情况。
4. **`disclose_default_product` 的判断条件不够精确**：当 source 是 "strong_signal" 且 `_user_named_product_identity` 返回 False 时，设置 disclose_default_product=True。但某些 strong_signal 可能是非常明确的（如"AC900"这个产品 ID 直接出现），此时不需要 disclaimer。
5. **catalog 是启动时一次性加载的**：如果产品目录发生变化，需要重启服务才能生效。

#### 改进方向
- 支持 fuzzy matching（编辑距离 ≤ 2 的别名也视为匹配）。
- ProductCatalog 支持热重载。
- 增加 `_looks_like_product_switch` 的覆盖范围。

---

### 2.14 WorkflowRuntime (`workflows/runtime.py`)

#### 设计细节
- 声明式 workflow 协议：每个 workflow 定义 intent、risk_level、required_slots、approval/handoff 标志。
- 四种内置协议：repair_intake、refund_intake、complaint_escalation、case_intake_cancel。
- `decide()` 方法结合工具提取的 slots 和消息文本进行最终决策。

#### 当前不足
1. **intent 判断分散在多处**：extract_case_slots 判断一次 intent，workflow_runtime.decide 又基于消息文本重新判断一次。如果两处判断冲突（如 extract 说 repair，但消息包含"退款"），workflow 的二次判断会覆盖。
2. **fallback 到 repair 可能不合适**：当 intent="unknown" 时，默认走 repair_intake，要求用户提供 product_model、issue、contact_phone。但如果用户只是路过问了一个问题（如"你们有实体店吗"），这些必填槽位会让对话体验很差。
3. **required_slots 是静态的**：不同产品的维修流程可能需要不同的信息（如某些产品需要购买凭证，某些需要序列号），但当前 required_slots 是协议级别的硬编码。
4. **`user_message_when_blocked` 是硬编码的**：在 runtime.py 的 AgentRuntime.run() 中没有被使用。安全拦截的回答是由 SafetyGate 决定的，而不是 WorkflowProtocol。
5. **workflow 协议不支持跨协议的 transition**：例如用户从 repair_intake 开始，但中途发现需要退款，当前没有从 repair → refund 的过渡机制。

#### 改进方向
- 将 intent 分类集中到 WorkflowRuntime 中，移除 extract_case_slots 的 intent 判断职责。
- 支持动态 required_slots（基于产品类型）。
- 实现 workflow 间 transition 机制。

---

### 2.15 LLM Integration (`llm/client.py`, `llm/generation.py`, `llm/prompts.py`)

#### 设计细节
- BailianOllamaChatClient 使用百炼优先、Ollama 回退的策略。
- LlmAnswerGenerator 提供了 product_support_answer 和 general_answer 两个生成节点。
- prompt 通过 JSON 格式组织，包含 task、user_message、context_pack、evidence、answer_rules 等。
- 生成失败时有 fallback_answer 兜底。

#### 当前不足
1. **`_complete_sync` 是同步阻塞调用**：在 async 方法中通过 `asyncio.to_thread()` 执行同步调用。这意味着每个 LLM 调用都会占用一个线程。在高并发场景下，线程池可能成为瓶颈。
2. **prompt 中不包含 available_tools**：PRODUCT_SUPPORT_SYSTEM_PROMPT 没有告诉模型有哪些工具可用。模型只能基于 evidence 生成回答，不能主动发起工具调用。
3. **`build_product_support_messages` 中的 context_pack 可能过大**：如果 evidence section 很大，加上 conversation 和 memory，可能超出模型的 context window。
4. **temperature 和 max_tokens 是 hardcoded 的**：在 `_build_default_answer_generator` 中设置 temperature=0.1, max_tokens=1024。不同场景可能需要不同的参数。
5. **没有实现 streaming**：LLM 调用是同步阻塞的，用户需要等待完整回答生成完毕。

#### 改进方向
- 使用真正的异步 HTTP 客户端（如 httpx.AsyncClient）替代 `asyncio.to_thread()`。
- 在 prompt 中注入 skill-relevant 的工具描述。
- 实现 token-level streaming 响应。

---

### 2.16 MCP Gateway Integration (`mcp/provider.py`, `tools/mcp_gateway.py`)

#### 设计细节
- McpCapabilityProvider 从 MCP Gateway 发现服务和工具，并转换为 ToolRuntime 兼容的 ToolSpec。
- McpToolAdapter 将 MCP 工具调用包装为 ToolRuntime 的 call 契约。
- 支持 McpToolPolicy：按服务+工具名覆盖 risk_level、requires_approval、capability_tags。
- MCP 元数据通过 input_schema 的 x-* 字段传递。

#### 当前不足
1. **`McpGatewayTool.call()` 是同步的**：`self._client.call_tool()` 是同步调用，在 async 方法中没有使用 asyncio.to_thread() 或其他异步化手段。
2. **没有连接池或连接复用**：每次发现工具时都创建新的 McpGatewayClient 实例。
3. **`McpCapabilityProvider.discover_tools()` 依赖 search_services 和 list_tools**：如果 MCP Gateway 有 100 个服务，每个服务有 20 个工具，discover 过程需要 1 + 100 次网络调用，可能很慢。
4. **错误处理粒度粗**：MCP 工具调用失败时，只捕获 Exception 并返回 error，没有对不同类型的错误（网络超时、服务不可用、参数错误）做区分。
5. **allowed_tools 白名单是简单的字符串集合**：不支持通配符或正则匹配。

#### 改进方向
- 异步化 MCP 客户端调用。
- 引入连接池和健康检查。
- tool discovery 支持并发和缓存。
- 错误分类和结构化错误码。

---

### 2.17 Eval System (`eval/`)

#### 设计细节
- SkillSelectionHarness 和 AgentEvaluationHarness 两层评测。
- golden_agent_dataset 包含 8+ 个分类的测试用例。
- 评测指标：skill_accuracy、tool_accuracy、safety_accuracy、approval_rate、handoff_rate、avg_loop_turns。

#### 当前不足
1. **评测案例数较少**：golden_agent_dataset 只有约 8-10 个 case，不足以覆盖生产环境的多样性。
2. **没有 trace replay 评测**：架构文档中描述的 trace replay 功能未实现。
3. **评测仅检查行为（选了哪个 skill、调用了哪些 tool），不评估回答质量**：没有对 answer 内容的语义评估。
4. **测试依赖 fake/mock 组件**：FakeCaseIntakeTool、FakeEnterpriseRetriever 等假组件的实现简化了真实系统的复杂性。
5. **没有回归测试基线**：没有保存历史评测结果用于对比。

#### 改进方向
- 将 golden case 扩展到 100+ 条，覆盖更多真实场景。
- 实现 LLM-as-judge 的回答质量评估。
- 建立 CI 中的回归评测流程。

---

### 2.18 API Layer (`app/main.py`, `app/api/v1/chat.py`)

#### 设计细节
- FastAPI 单文件应用，路由在模块加载时初始化 runtime（全局单例）。
- 支持 /chat、/approvals、/handoffs、/health 四个端点。
- ChatRequest 有字段级验证（session_id 和 message 非空）。

#### 当前不足
1. **runtime 是模块级全局变量**：`runtime = build_default_runtime()` 在模块导入时执行，意味着配置错误会在服务启动时（而不是第一次请求时）暴露。同时全局单例使得测试困难。
2. **没有中间件**：缺少请求日志、trace_id 注入、限流、CORS、异常处理中间件。
3. **ChatRequest.metadata 类型是 dict[str, object]**：object 类型太宽泛，应该限制为 JSON-serializable 的值。
4. **没有 session 管理 API**：`GET /api/v1/sessions/{session_id}` 在 engineering_baseline 文档中描述但未实现。
5. **没有 trace 查询 API**：`GET /api/v1/sessions/{session_id}/trace/{trace_id}` 未实现。
6. **没有错误响应标准化**：当请求验证失败时，FastAPI 返回默认的 422 错误，缺少业务友好的错误格式。

#### 改进方向
- 使用 FastAPI dependency injection 管理 runtime 生命周期。
- 添加结构化错误响应模型。
- 实现文档中描述的 sessions 和 trace API。

---

## 三、跨模块问题汇总

### 3.1 隐式耦合

多个模块之间存在未在接口中声明的隐式耦合：

| 耦合关系 | 描述 |
|---------|------|
| memory.session ↔ skills | memory 的 `_apply_structured_fields` 硬编码了 product_support 和 case_intake 的 state 结构 |
| product_support ↔ ContextGovernance | skill 内部创建了新的 ContextGovernance 实例 |
| case_intake ↔ workflow | case_intake 在 run() 中调用 workflow_runtime.decide()，但 workflow decision 又被 safety gate 独立读取 |
| planner → skill registry | planner 推荐的 skill name 必须与 registry 中的 skill name 精确匹配 |

### 3.2 同步/异步混用

项目中大量使用了 `asyncio.to_thread()` 来包装同步调用，这在高并发场景下可能导致性能问题：

- `BailianOllamaChatClient._complete_sync()` 
- `BailianOllamaSkillSelectionClient._complete_sync()`
- `McpGatewayTool.call()` 本身也是同步的但没有 to_thread

### 3.3 错误处理不一致

- AgentRuntime 对 skill 异常的处理：捕获 Exception，生成通用错误回答
- ToolRuntime 对工具异常的处理：捕获 Exception，返回 ToolCallResult(ok=False)
- KnowledgeRuntime 对后端异常的处理：捕获 Exception，fallback 到本地搜索
- LLM 调用的异常处理：捕获 Exception，返回 fallback_answer

这些处理策略各不相同，缺乏统一的错误处理框架。

### 3.4 配置管理

- 配置分散在环境变量、硬编码常量、ProductCatalog JSON 中
- 没有配置变更的运行时热更新
- 部分配置（如 CUSTOMER_SERVICE_KEYWORDS）作为源代码常量而非外部配置

### 3.5 测试覆盖

现有测试覆盖了主路径，但以下方面缺乏测试：
- 并发场景（多个请求同时修改同一 session state）
- 内存泄漏（长时间运行的 transcript/trace 积累）
- 大数据量输入（极长消息、极多图片、极多工具调用）
- 异常恢复后的状态一致性

---

## 四、改进优先级建议

### P0（影响正确性和安全性）

1. **SafetyGate 需要检查 answer_draft 内容**，而不仅是输入消息
2. **ProductSupportSkill 内部不应创建新的 ContextGovernance**
3. **CaseIntakeSkill 第一轮应给出即时反馈**，而非空白回答

### P1（影响用户体验和可靠性）

4. **Planner 升级**：从关键词匹配升级为轻量级 NLP
5. **ToolRuntime 添加超时控制**
6. **Memory 添加 flat_state 与结构化字段的一致性验证**
7. **LLM prompt 注入 available_tools 描述**

### P2（工程质量和可维护性）

8. **统一错误处理框架**
9. **异步化所有 I/O 操作**
10. **实现 trace replay 评测**
11. **添加结构化错误响应模型**
12. **配置外部化和热更新**

### P3（扩展性和性能）

13. **并行执行无依赖的工具调用**
14. **Context Governance 的智能裁剪**
15. **实现真正的多 Agent 协作**
16. **Streaming 响应**

---

## 五、总结

nikon0 作为一个从零开始自研的 Agent 框架，在第一阶段已经建立了清晰的分层架构、完备的数据模型和可测试的运行时。相比直接使用 LangChain/LangGraph，自研方案在可控性、可观测性和适配企业场景方面有明显优势。

当前的短板主要集中在：
- **语义理解层过于简单**（Planner 基于关键词）
- **安全校验不完整**（不检查输出内容）
- **异步 I/O 是伪异步**（同步调用包在线程中）
- **错误处理缺乏统一框架**
- **一些已设计但未实现的模块**（PrivacyFilter、ContextVerifier、多 Agent）

建议按照上述优先级逐步推进改进，P0 和 P1 项应在下一轮迭代中优先完成。
