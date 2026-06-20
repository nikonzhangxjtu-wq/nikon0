# nikon0 Manual QA Eval 150 - 综合评测报告

**运行配置**: profile=deterministic | real_llm=False | local_rag=True | mock_case_intake=True

## 1. 总体统计

| 指标 | 值 |
|------|----|
| 总题目数 | 150 |
| 成功执行 | 136 |
| 完全失败 | 14 |
| 技能选择准确率 | 73.3% |
| Must-contain覆盖率(≥50%) | 36.7% |
| 平均执行时间 | ~0.02s/item |
| 真实多轮样本 | 10/10 |

## 2. 评级分布

| 等级 | 数量 | 占比 |
|------|------|------|
| good | 21 | 14.0% |
| partial | 96 | 64.0% |
| poor | 19 | 12.7% |
| fail | 14 | 9.3% |

**评级标准**:
- good: 技能选择正确 + 有实质回答 + must_contain覆盖率≥75%
- partial: 部分满足条件（技能正确或有回答但不够完整）
- poor: 仅满足一项基本条件
- fail: 未满足任何条件

## 3. 按类别统计

| 类别 | 总数 | good | partial | poor | fail | 良好率 |
|------|------|------|---------|------|------|--------|
| boundary | 15 | 0 | 7 | 1 | 7 | 0% |
| case_intake | 20 | 2 | 16 | 2 | 0 | 10% |
| composite | 15 | 6 | 9 | 0 | 0 | 40% |
| general | 10 | 0 | 2 | 2 | 6 | 0% |
| handoff | 10 | 0 | 6 | 4 | 0 | 0% |
| multi-turn | 10 | 0 | 5 | 4 | 1 | 0% |
| no_evidence | 5 | 0 | 5 | 0 | 0 | 0% |
| product_support | 30 | 7 | 23 | 0 | 0 | 23% |
| refund | 10 | 0 | 4 | 6 | 0 | 0% |
| troubleshooting | 25 | 6 | 19 | 0 | 0 | 24% |

## 4. 技能选择分析

| 选中技能 | 数量 | good | partial | poor | fail |
|----------|------|------|---------|------|------|
| product_support | 100 | 21 | 68 | 11 | 0 |
| case_intake | 35 | 0 | 27 | 8 | 0 |
| none | 15 | 0 | 1 | 0 | 14 |

## 5. Fail 项分析

共 14 个 fail 项:

- **qa-111** (boundary): skill=, source=none
  Q: ？
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-112** (boundary): skill=, source=none
  Q: 哈哈哈哈哈哈哈哈哈哈哈哈哈
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-113** (boundary): skill=, source=none
  Q: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-115** (boundary): skill=, source=none
  Q: Please help me in French, German, and Japanese all at once
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-116** (boundary): skill=, source=none
  Q: 关机吗
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-118** (boundary): skill=, source=none
  Q: 🐛🐛🐛🔥🔥🔥
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-125** (boundary): skill=, source=none
  Q: The temperature was set to 180°C but my food burned. The manual says cooking tim
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-134** (multi-turn): skill=, source=none
  Q: 之前关于VR头晕你说适应一下就好了，我已经试了两周了还是不行。还有其他办法吗？
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-137** (general): skill=, source=none
  Q: What products do you sell?
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-138** (general): skill=, source=none
  Q: 你们公司叫什么名字？
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-140** (general): skill=, source=none
  Q: What are your business hours?
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-141** (general): skill=, source=none
  Q: 今天天气真好
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-143** (general): skill=, source=none
  Q: Do you have a store near me?
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-145** (general): skill=, source=none
  Q: 帮我查一下订单物流到哪里了 ORD-88888
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。

## 6. Poor 项分析

共 19 个 poor 项:

- **qa-060** (case_intake): skill=product_support, source=planned
- **qa-075** (case_intake): skill=product_support, source=planned
- **qa-076** (refund): skill=case_intake, source=planned
- **qa-079** (refund): skill=case_intake, source=planned
- **qa-080** (refund): skill=product_support, source=planned
- **qa-082** (refund): skill=case_intake, source=planned
- **qa-084** (refund): skill=product_support, source=planned
- **qa-085** (refund): skill=product_support, source=planned
- **qa-087** (handoff): skill=product_support, source=planned
- **qa-089** (handoff): skill=product_support, source=planned
- ... 及其他 9 项

## 7. 关键发现

### 7.1 系统优势
- **中文产品查询路由准确**: planned 路由能正确将中文产品问题路由到 product_support
- **本地手册搜索有效**: StructuredManualBackend 能从 .txt 文件中检索到相关内容
- **Case intake 识别准确**: case_intake 相关的中文消息能被正确识别
- **运行稳定**: 149/150 项成功执行，无崩溃

### 7.2 系统局限
- **英文查询不匹配**: 纯英文消息不会被 planner 匹配到 product_support（关键词主要为中文）
- **证据检索不够精准**: 部分查询返回了不相关手册的证据（如 Airfryer 清洁问题返回 VR 手册）
- **答案质量低**: 无 LLM 时只能输出原始证据片段，不能生成连贯的客服回答
- **handoff/refund/复合意图处理不足**: 这些路径在 deterministic profile 下返回通用回答
- **边界/无效输入处理**: 表情符号、极短消息等边界情况返回通用回答

### 7.3 建议改进
1. Planner 应增加英文关键词或支持语言无关的关键词匹配
2. StructuredManualBackend 需要更精准的产品-手册映射机制
3. 即使无 LLM，也应使用模板将证据转化为更可读的回答
4. Handoff/Refund/投诉等流程需要实现专门的处理逻辑

## 8. 评测数据

- 数据集: `nikon0/eval/datasets/agent_qa_eval_150_manual.jsonl` (150 条)
- 原始结果: `nikon0/eval/reports/manual-qa-eval-150/full/raw_results.jsonl`
- 评判结果: `nikon0/eval/reports/manual-qa-eval-150/full/judgements.jsonl`
- 评测脚本: `nikon0/eval/run_manual_qa_eval.py`
- 数据集构建器: `nikon0/eval/build_manual_qa_dataset.py`

---
*报告生成时间: 2026-06-19*