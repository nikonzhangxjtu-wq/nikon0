# 得分提升 TODO

> 当前状态：排名 50/93，自评分数 ~4.13/5（qwen2 自评，可能偏高）
> 关键短板：manual_hit_rate 仅 40%（RAG 检索错了手册）、视觉模型空缺、模型指令遵循弱

---

## P0 — 高收益低投入，建议立刻做

### 1. 换更强的生成模型

**现状**：`qwen2:latest`（7B Q4_0），基座模型较旧，指令遵循和推理均弱于 qwen2.5。
**动作**：
```bash
ollama pull qwen2.5:14b        # 首选，14B 推理能力显著提升
# 若显存/内存不够，降级为：
ollama pull qwen2.5:7b         # 仍比当前 qwen2 强 20-30%
```
然后修改 `.env` 或环境变量：
```
GEN_MODEL=qwen2.5:14b
```
**预期收益**：答案质量 +0.3~0.5 分（结构、指令遵循、虚构减少），估计可提升 10-15 名。

### 2. 修复 query_construction 导致的手册命中率仅 40% 的问题

**现状**：`app/services/rag_skill/query_construction.py` 用 LLM 判断用户问题属于哪本手册，正确率仅 40%。手册选错 → 检索结果完全无关 → 答案质量崩塌。
**根因**：query_construction 依赖 LLM 单次决策，没有 fallback；LLM 不知道所有手册名列表。
**动作**（按优先级）：
- **方案 A（推荐，低投入）**：检索时先不过滤手册名（全库搜索），让 BM25 + dense + rerank 自然召回最相关 chunk。在 `pipeline.py:120-122` 去掉 `manual_name` 过滤参数。全库搜索可能引入噪音，但 reranker 会过滤掉。
- **方案 B（中投入）**：在 query_construction prompt 中显式列出所有可用手册名及其内容简介，让 LLM 做选择题而非开放生成。
- **方案 C（高投入）**：用 embedding 相似度匹配手册名（用户 query embedding vs 每个手册的摘要描述 embedding），替代 LLM 调用。
**预期收益**：RAG 答案准确率从 40% → 80%+，估计可提升 15-20 名。

### 3. 强制 max_tokens 控制输出长度

**现状**：prompt 里写了 ≤250 字，但 `generator.py` 没有传 `max_tokens` 参数，模型可能忽略指令。
**动作**：在 `app/services/generator.py:27-31` 的 `ChatOllama` 初始化中加入：
```python
self.client = ChatOllama(
    model=settings.gen_model,
    base_url=settings.ollama_base_url,
    temperature=temp,
    num_predict=512,   # 中文约 350-400 字，留有余量
)
```
同时在 `config.py` 新增 `gen_max_tokens: int = Field(default=512)` 使其可配置。
**预期收益**：杜绝超长回答，结构分 +0.2。

---

## P1 — 中等投入，明确收益

### 4. 补全视觉能力（多模态理解）

**现状**：`vision_model` 默认为空 → 回退到 `qwen2:latest` → qwen2 不支持图片输入 → vision 管线静默失效。当前没有任何多模态模型可用。
**动作**：
```bash
ollama pull minicpm-v:latest     # 轻量多模态，~8B，支持图片理解
# 或
ollama pull qwen2.5-vl:7b        # 通义千问视觉版，对中文场景更好
```
修改 `.env`：
```
VISION_MODEL=minicpm-v:latest
```
同时确认 `app/services/vision.py` 中的 prompt 能输出紧凑摘要（当前 prompt 需检查是否过于啰嗦）。
**注意**：`question_public.csv` 当前没有图片列。如果比赛最终测试集会提供图片 URL/Base64，需要同步修改 `run_batch_submission.py:58` 的 `images=[]` 来传入实际图片。
**预期收益**：多模态题从 2-3 分 → 4-5 分。

### 5. 扩展路由关键词覆盖

**现状**：`app/services/router.py` 的 MANUAL_PHRASES 只有 ~14 条，大量合理的手册问题被错误路由为 `unknown`，触发额外的 LLM 仲裁调用，增加延迟和路由错误概率。
**动作**：在 `router.py` 的 MANUAL_PHRASES 中增加以下高频模式：
```python
# 中文
"怎么安装", "如何设置", "怎么使用", "怎么开机", "怎么关机",
"如何启动", "如何关闭", "怎么连接", "怎么拆", "如何更换",
"怎么清洗", "如何清洁", "维护保养", "故障排除", "常见问题",
"指示灯", "按键功能", "配件", "规格参数", "技术规格",
# 英文
"how to install", "how to use", "how to set up",
"how to clean", "how to replace", "troubleshooting",
"specifications", "features", "components",
```
同时微调 `router_unknown_llm_arbitrate=False`，先关闭 LLM 仲裁看关键词扩展后的效果，如果 unknown 比例从当前下降到可接受范围就直接用启发式路由。
**预期收益**：路由准确率 +5~10%，延迟 -30%。

### 6. 改进检索后置过滤（用真实 reranker 分数）

**现状**：`app/services/rag_skill/rerank.py:177` 将 cross-encoder 分数重新映射为 `(top_k - rank) / top_k` 的排名分数，丢失了真实相关性信号。管道中 `retriever_context_filter_score_threshold=0.3` 作用在这些人工分数上，等价于固定取 top-4，无论实际相关性如何。
**动作**：修改 `rerank.py` 保留原始 cross-encoder logits 或做 softmax 归一化：
```python
# 方案：保留并 softmax 归一化 cross-encoder 原始分数
import numpy as np
logits = [r["score"] for r in ranked]  # 原始 logits
probs = np.exp(logits) / np.sum(np.exp(logits))
for r, prob in zip(ranked, probs):
    r["score"] = float(prob)
```
然后调整 `retriever_context_filter_score_threshold` 到一个有意义的阈值（如 0.1，仅过滤明显无关的 chunk）。
**预期收益**：检索精度 +10%，减少漏召回。

---

## P2 — 值得做但投入较大

### 7. 多轮对话支持

**现状**：`question_public.csv` 中包含多行问题（用 `\n` 分隔），这些是"多子问"合并在一条消息中的场景。真正的多轮（跨消息记忆）目前完全没有。
**动作**：
- **短期**（处理多子问）：在 prompt 中增加"问题中包含多个子问题，请逐一回答，每个 1-2 句"——我上一轮已做了这个。
- **中期**（session 级记忆）：在 `pipeline.py` 的 `ChatPipeline` 中加入 session-store（dict 或 Redis），保存最近 3 轮的 Q&A。在 `PromptContext` 中新增 `history: str` 字段，各 builder 在构建 prompt 时附上历史。路由时也可以参考历史上下文。
**预期收益**：多轮/多子问题从 3-4 分 → 4-5 分。

### 8. ReAct / 反思式生成（仅用于复杂问题）

**现状**：所有问题都走单次生成流程。对于复杂的多诉求投诉类问题（如 #46-63），单次生成容易遗漏子诉求。
**动作**：不是所有问题都 ReAct，而是加一个复杂度判断：
- 问题包含 ≥3 个独立问号或诉求关键词（"同时"、"另外"、"还有"）→ 标记为复杂
- 复杂问题走两步生成：
  1. 先让模型拆解子问题并逐一检索
  2. 再汇总生成最终回答
- 简单问题保持当前单次流程
**预期收益**：复杂题 +0.5 分，简单题不受影响。投入约 2-3 天。

### 9. 规则化的后处理校验

**现状**：生成后的答案没有任何自动质检，可能包含明显的事实错误或格式问题。
**动作**：在 `pipeline.py` 的 `run()` 方法最后增加轻量后处理：
```python
# 1. 检查答案是否包含禁止的模板话术，若有则截断
for banned in ["很高兴为您", "如有任何疑问请随时", "祝您生活愉快"]:
    if banned in answer:
        answer = answer.split(banned)[0].strip()
# 2. 检查答案中引用的 IMG 标记是否在合法列表中（不在则移除）
# 3. 如果答案为空或仅剩模板，触发重试
```
**预期收益**：避免因模板话术扣结构分，+0.1~0.2。

---

## 不建议做的事（及原因）

| 想法 | 不建议的原因 |
|------|-------------|
| 全面 ReAct / Agent 化 | 大多数题目是单步检索即可回答的手册查询，ReAct 增加 2-3 倍延迟，且 qwen2 级别的模型做多步推理容易跑偏 |
| 全异步重写 pipeline | 比赛是批量推理，不需要并发性能；投入产出比低 |
| 自训练 reranker | 需要标注数据，且 bge-reranker-base 已经是中英文通用的强 baseline |
| 复杂多轮记忆系统 | 除非比赛明确考察跨轮对话（session 级），否则只需处理好多子问即可 |

---

## 执行建议

**如果只能做一件事**：先换模型（qwen2.5:14b）+ 去掉 manual_name 过滤（全库检索）。这两个改动加起来约 1 小时，预期可提升 20-30 名。

**如果有 1 天时间**：按 P0 → P1 顺序，做完 1-6 项。预期可进入前 20 名。

**如果想冲击前十**：P2 的第 7、8 项必须做，尤其是复杂问题的多步处理。
