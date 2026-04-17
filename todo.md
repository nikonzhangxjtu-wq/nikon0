# V1 实现待办清单

本文件对应「骨架已搭好、核心由你实现」的路线，按顺序推进即可。

---

## 已完成（框架层）

- [x] `README.md`：安装说明、环境变量、运行方式
- [x] `app/main.py`：`POST /chat`、Bearer 校验、`GET /healthz`
- [x] `app/schemas/chat.py`：请求/响应结构与字段校验
- [x] `app/core/config.py`：从 `.env` 读取配置
- [x] `app/services/router.py`：RAG gate + domain hint（启发式，可替换）
- [x] `app/services/retriever.py`：Milvus 检索占位
- [x] `app/services/generator.py`：LangChain + Ollama `qwen2`
- [x] `app/services/pipeline.py`：gate → retrieve → generate 主链路
- [x] `app/services/ingestion.py`：手册解析/切块占位
- [x] `app/services/session_store.py`：`session_id` 生成
- [x] `app/utils/prompt_builder.py`：检索结果拼 prompt
- [x] `scripts/build_index.py`：建索引入口
- [x] `scripts/run_batch_submission.py`：批量 `id,ret` 导出

---

## 阶段 1：`app/services/ingestion.py`（手册入库）

**目标**：把 `手册/*.txt` 变成可索引的 chunk，并绑定图片 ID。

- [ ] 解析单份手册文件格式（长字符串、`#` 标题、`<PIC>`、尾部 `image_id` 列表）
- [ ] 从正文中提取 `<PIC>` 位置，与尾部列表做顺序/局部对齐（先实现一种简单规则即可）
- [ ] 按「标题 / 步骤 / 故障排查」等边界切块，控制单块长度
- [ ] 输出 `ManualChunk`：`chunk_id`、`manual_name`、`text`、`image_ids`
- [ ] 写 1～2 个单元测试或小脚本：打印某手册前几个 chunk 与绑定的 `image_ids`，人工扫一眼是否合理

**自测建议**：选 `冰箱手册.txt` 或 `健身追踪器手册.txt`，对照原文检查切块是否切断在奇怪位置。

---

## 阶段 2：`scripts/build_index.py` + `app/services/retriever.py`（Milvus + 向量）

**目标**：chunk 写入 Milvus，查询时能 embedding + top-k 召回。

- [ ] 在 `build_index.py` 中：ingestion → 生成 embedding（`nomic-embed-text`）→ 写入 Milvus（含 metadata：manual、chunk_id、image_ids 等）
- [ ] 定义 collection schema 与索引参数（维度与 embed 模型一致）
- [ ] 在 `retriever.py` 中：query 向量化 → `search` → 映射为 `RetrievedChunk`（`score`、`manual_name`、`image_ids`）
- [ ] 删除 `VectorRetriever` 里的占位假数据，改为真实检索
- [ ] 自测：固定几个问题，打印 top-k 文本与分数，确认召回相关段落

---

## 阶段 3：`app/services/pipeline.py` + `app/utils/prompt_builder.py`（质量与幻觉）

**目标**：检索弱时有兜底，生成严格依赖 context。

- [ ] 为检索分数设阈值（例如 top1 低于某值则视为「未命中」）
- [ ] 未命中或 chunk 为空时：保守回答（如引导用户提供型号/订单信息），避免编造手册内容
- [ ] 在 `prompt_builder.py` 中强化 system/user 指令：仅依据 CONTEXT、不足则明确说明
- [ ] 可选：对 `customer_service` 分支单独模板（与 `manual` 的 RAG 路径区分）

---

## 阶段 4：`scripts/run_batch_submission.py`（批量与提交）

**目标**：对 `question_public.csv` 全量跑通并导出可提交文件。

- [x] 确认 `id` 与 `question_public.csv` 一一对应，无遗漏、无重复（脚本内置校验）
- [x] 跑通全量（注意耗时与 Ollama/Milvus 限流，必要时加批量 sleep 或并发控制）
- [x] 输出 `submission_v1.csv`（或你命名的版本），列名为 `id,ret`，与 `submission_example.csv` 一致
- [x] 抽样人工检查：多轮问答类、英文题、纯客服题各若干条（脚本输出抽样条目供人工核查）

---

## 可选后续（V1+）

- [ ] 用户上传图片 → 与 `question` 联合检索或 caption 后再进 pipeline（接口字段已预留）
- [ ] LangGraph：把 gate / retrieve / generate 拆成节点，便于调试与 A/B
- [ ] Reranker：对 top-k 再精排
- [ ] 将 `router.py` 从关键词升级为轻量分类模型或 LLM 分类

---

## 参考：手把手补 `ingestion.py` 的推进顺序（若需要）

1. 只读解析：读入一个 `手册/*.txt`，打印出「正文长度、`<PIC>` 个数、尾部 id 列表长度」
2. 对齐策略：先实现「按 `<PIC>` 顺序依次消费 `image_ids`」的最简版
3. 切块：按 `#` 分段，过长再按段落或固定字符数二次切
4. 再迭代：针对「故障列表」「步骤 1/2/3」单独正则或规则优化

完成阶段 1～2 后，API 的 `/chat` 在 `needs_rag=true` 时才会真正变成「手册 RAG」。
