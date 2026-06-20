# nikon0 Manual QA Eval 150 - 综合评测报告

**运行配置**: profile=production_like | real_llm=True | local_rag=False | mock_case_intake=False

## 1. 总体统计

| 指标 | 值 |
|------|----|
| 总题目数 | 150 |
| 成功执行 | 150 |
| 完全失败 | 0 |
| 技能选择准确率 | 72.7% |
| Must-contain覆盖率(≥50%) | 61.3% |
| 平均执行时间 | ~0.02s/item |
| 真实多轮样本 | 10/10 |

## 2. 评级分布

| 等级 | 数量 | 占比 |
|------|------|------|
| good | 45 | 30.0% |
| partial | 80 | 53.3% |
| poor | 25 | 16.7% |
| fail | 0 | 0.0% |

**评级标准**:
- good: 技能选择正确 + 有实质回答 + must_contain覆盖率≥75%
- partial: 部分满足条件（技能正确或有回答但不够完整）
- poor: 仅满足一项基本条件
- fail: 未满足任何条件

## 3. 按类别统计

| 类别 | 总数 | good | partial | poor | fail | 良好率 |
|------|------|------|---------|------|------|--------|
| boundary | 15 | 1 | 7 | 7 | 0 | 7% |
| case_intake | 20 | 1 | 19 | 0 | 0 | 5% |
| composite | 15 | 8 | 7 | 0 | 0 | 53% |
| general | 10 | 0 | 2 | 8 | 0 | 0% |
| handoff | 10 | 0 | 6 | 4 | 0 | 0% |
| multi-turn | 10 | 1 | 7 | 2 | 0 | 10% |
| no_evidence | 5 | 1 | 4 | 0 | 0 | 20% |
| product_support | 30 | 18 | 12 | 0 | 0 | 60% |
| refund | 10 | 0 | 6 | 4 | 0 | 0% |
| troubleshooting | 25 | 15 | 10 | 0 | 0 | 60% |

## 4. 技能选择分析

| 选中技能 | 数量 | good | partial | poor | fail |
|----------|------|------|---------|------|------|
| product_support | 96 | 45 | 47 | 4 | 0 |
| case_intake | 29 | 0 | 24 | 5 | 0 |
| none | 25 | 0 | 9 | 16 | 0 |

## 5. Fail 项分析

共 0 个 fail 项:


## 6. Poor 项分析

共 25 个 poor 项:

- **qa-076** (refund): skill=, source=none
- **qa-079** (refund): skill=, source=none
- **qa-082** (refund): skill=case_intake, source=model
- **qa-085** (refund): skill=case_intake, source=model
- **qa-087** (handoff): skill=case_intake, source=model
- **qa-089** (handoff): skill=product_support, source=planned
- **qa-093** (handoff): skill=case_intake, source=model
- **qa-094** (handoff): skill=case_intake, source=model
- **qa-111** (boundary): skill=, source=none
- **qa-112** (boundary): skill=, source=none
- ... 及其他 15 项

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