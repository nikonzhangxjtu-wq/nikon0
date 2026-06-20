# 多模态客服系统 - 技术方案设计

## 一、现状诊断

### 1.1 当前架构中"多模态"的实际实现

```
用户上传图片 → Vision LLM（文本摘要） → 拼接到检索 Query → 纯文本检索 → 纯文本生成
手册图片     → <PIC> 占位 → <IMG:xxx> 文本标签嵌入 chunk → Milvus JSON 字段存储 → 文本引用
```

**本质：图片 → 文本 → 文本检索 → 文本生成。图片从未真正参与向量检索和视觉理解。**

### 1.2 具体问题

| 问题 | 代码位置 | 影响 |
|------|---------|------|
| `VISION_MODEL` 未配置（空字符串） | `.env` → `vision.py:69` | 用户上传图片的视觉摘要静默失败，回退到 `deepseek-v4-flash`（无视觉能力） |
| 手册图片只存 `image_ids` JSON 字段 | `ingestion.py:281` → Milvus `image_ids` 字段 | 图片无向量表示，无法做相似度检索 |
| 图片 `<IMG:xxx>` 只是文本标签 | `prompt_builder.py:139` | 生成模型看到的是字符串 `<IMG:Manual1_0>`，不是图片像素 |
| 检索只有 `dense_vector`(文本) + `sparse_vector`(BM25) | `retriever.py:233-294` | 无法实现以图搜图、以文搜图 |
| 生成模型是纯文本 `deepseek-v4-flash` | `generator.py` | 即使检索返回了图片，模型也无法"看"图片 |
| bge-m3 只做文本嵌入 | `config.py:48-49` | 缺少多模态嵌入模型（CLIP 类） |

---

## 二、多模态升级路线图（四个阶段）

### 阶段 1：修复 & 增强当前视觉管线（1-2 天）

**目标**：让用户上传图片的视觉信息真正参与检索和生成。

#### 1.1 配置视觉模型

```env
# .env
VISION_MODEL=minicpm-v:latest       # 或 qwen2.5-vl:7b / llava:latest
VISION_ENABLED=true
```

`minicpm-v` 在 Ollama 上可用，支持中文，轻量（8B），适合客服场景。

#### 1.2 增强视觉摘要的结构化输出

当前 `vision.py` 只输出一句中文摘要。改为结构化 JSON：

```python
# 升级后的视觉摘要输出
{
  "summary": "冰箱冷藏室显示 E2 错误代码",
  "ocr_text": "E2 温度传感器故障",
  "key_entities": ["E2", "冷藏室", "温度传感器"],
  "product_type": "冰箱",
  "visual_features": ["红色LED指示灯闪烁", "显示屏数字"]
}
```

- `ocr_text` 用于关键词精确匹配（走 BM25 / sparse 检索）
- `key_entities` 作为额外的检索过滤条件
- `product_type` 辅助路由判断（缩小手册范围）

#### 1.3 代码改动点

- `app/services/vision.py` — `summarize_images()` 返回结构化 dict 而非纯字符串
- `app/services/pipeline.py` — 将 `ocr_text` 注入检索 query，`key_entities` 注入 filter
- `app/utils/prompts/context.py` — `PromptContext` 增加 `visual_ocr_text`、`visual_entities` 字段

---

### 阶段 2：多模态向量嵌入 — 统一图文向量空间（3-5 天）

**核心思路**：引入 CLIP 风格的模型，将文本和图片映射到**同一个向量空间**。这样文本 query 可以直接检索图片，图片也可以直接检索文本。

#### 2.1 模型选型

| 模型 | 维度 | 中文支持 | 优势 | 劣势 |
|------|------|---------|------|------|
| **jina-clip-v2** | 1024 | 多语言 | 文本+图片统一空间，89% ImageNet zero-shot | 维度与现有 bge-m3(768) 不一致 |
| **Chinese-CLIP (CN-CLIP)** | 512/768 | 中文原生 | 中文图文匹配最优，轻量 | 英文弱 |
| **bge-m3** | 768 | 多语言 | 与现有嵌入模型一致，零改动兼容 | 图片嵌入能力有限 |
| **CLIP-ViT-B/32** | 512 | 英文为主 | 社区成熟 | 中文弱 |

**推荐方案**：**jina-clip-v2** 或 **Chinese-CLIP RN50 (768d)**，理由：
- jina-clip-v2 多语言效果好，且支持 768d Matryoshka 表示，可与现有 bge-m3 维度对齐
- Chinese-CLIP 如果业务以中文为主则更优

#### 2.2 新增 Milvus Collection：`manual_images`

```python
# 新 collection schema
manual_images = {
    "fields": [
        {"name": "image_id", "dtype": "VARCHAR", "is_primary": True, "max_length": 128},
        {"name": "image_path", "dtype": "VARCHAR", "max_length": 512},
        {"name": "image_vector", "dtype": "FLOAT_VECTOR", "dim": 768},   # CLIP 嵌入
        {"name": "manual_name", "dtype": "VARCHAR", "max_length": 128},
        {"name": "parent_chunk_ids", "dtype": "VARCHAR", "max_length": 512},  # JSON array
        {"name": "caption", "dtype": "VARCHAR", "max_length": 512},       # 图片描述/alt text
        {"name": "ocr_text", "dtype": "VARCHAR", "max_length": 1024},     # OCR 文字
    ],
    "indexes": [
        {"field": "image_vector", "index_type": "HNSW", "metric_type": "COSINE"},
        {"field": "manual_name", "index_type": "TRIE"},
    ]
}
```

#### 2.3 增强 `manual_chunks` Collection

在现有 `manual_chunks` collection 中增加一个字段：

```python
# 给每个 chunk 增加多模态向量（该 chunk 关联图片的平均 CLIP 嵌入）
{"name": "multimodal_vector", "dtype": "FLOAT_VECTOR", "dim": 768}
```

对于包含图片的 chunk，`multimodal_vector = text_embedding * 0.6 + avg(image_embeddings) * 0.4`，实现图文联合表示。

#### 2.4 建索引脚本改造

```python
# scripts/build_index.py 新增逻辑

# 1. 为每张手册图片生成 CLIP 嵌入
clip_model = ChineseCLIPModel()  # 或 JinaCLIP
for img_path in manual_image_paths:
    embedding = clip_model.encode_image(img_path)
    ocr_text = ocr_engine.extract(img_path)
    caption = clip_model.generate_caption(img_path)  # 可选
    milvus.insert("manual_images", [{
        "image_id": img_id,
        "image_vector": embedding,
        "ocr_text": ocr_text,
        "caption": caption,
        ...
    }])

# 2. 为每个 chunk 计算多模态向量
for chunk in manual_chunks:
    if chunk.image_ids:
        img_embeddings = [get_image_embedding(iid) for iid in chunk.image_ids]
        avg_img_emb = np.mean(img_embeddings, axis=0)
        chunk.multimodal_vector = chunk.dense_vector * 0.6 + avg_img_emb * 0.4
    else:
        chunk.multimodal_vector = chunk.dense_vector  # 纯文本 chunk
```

---

### 阶段 3：多路混合检索 + 图文融合（5-7 天）

**目标**：一次查询同时检索文本、图片、图文联合三种信号，加权融合。

#### 3.1 检索架构升级

```
用户 Query（+ 可选图片）
│
├─→ [Text Embedding (bge-m3)]
│   └─→ dense_vector 检索（现有）─────────┐
│                                          │
├─→ [CLIP Text Embedding]                  │
│   ├─→ multimodal_vector 检索（chunks）───┤
│   └─→ image_vector 检索（以文搜图）─────┤
│                                          ├─→ 加权融合 → Reranker → 最终上下文
├─→ [如果用户上传了图片]                    │
│   ├─→ CLIP Image Embedding ──→ 以图搜图 ┤
│   └─→ CLIP Image Embedding ──→ 以图搜文 ┤
│                                          │
├─→ [BM25 Sparse 检索（现有）]─────────────┘
│
└─→ [OCR / 实体关键词精确匹配]
```

#### 3.2 融合权重策略

```python
# app/services/retriever.py 新增多模态融合

class MultimodalRetriever(VectorRetriever):
    """多模态检索器：在现有 dense + sparse 基础上增加 visual 路线"""

    FUSION_WEIGHTS = {
        "dense_text": 0.35,       # bge-m3 文本嵌入
        "multimodal": 0.25,       # CLIP 图文联合嵌入（chunks）
        "image_search": 0.15,     # 以文搜图 / 以图搜图
        "sparse_bm25": 0.15,      # BM25 关键词
        "ocr_exact": 0.10,        # OCR 精确匹配
    }

    def retrieve_multimodal(
        self,
        query: str,
        query_images: list[str] | None = None,  # 用户上传的图片
        top_k: int = 10,
        manual_name: str | None = None,
    ) -> list[RetrievedChunk]:
        """多路检索 → 加权融合 → 精排"""
        
        results = []

        # 1. 传统文本检索（现有逻辑）
        dense_results = self._search_dense(...)
        results.append((dense_results, self.FUSION_WEIGHTS["dense_text"]))

        # 2. 多模态向量检索（chunks 的图文联合表示）
        clip_text_emb = self.clip_model.encode_text(query)
        multimodal_results = self._search_multimodal(clip_text_emb, ...)
        results.append((multimodal_results, self.FUSION_WEIGHTS["multimodal"]))

        # 3. 以文搜图（从 manual_images collection 检索相关图片）
        image_results = self._search_images(clip_text_emb, ...)
        # 将搜到的图片反向映射到它们所属的 chunks
        chunk_from_image = self._map_images_to_chunks(image_results)
        results.append((chunk_from_image, self.FUSION_WEIGHTS["image_search"]))

        # 4. 如果用户上传了图片
        if query_images:
            clip_img_emb = self.clip_model.encode_image(query_images[0])
            # 以图搜图
            similar_images = self._search_images(clip_img_emb, ...)
            img_to_chunks = self._map_images_to_chunks(similar_images)
            results.append((img_to_chunks, self.FUSION_WEIGHTS["image_search"]))
            # 以图搜文
            img_to_text = self._search_multimodal(clip_img_emb, ...)
            results.append((img_to_text, self.FUSION_WEIGHTS["multimodal"]))

        # 5. BM25（现有逻辑）
        sparse_results = self._search_sparse_text(...)
        results.append((sparse_results, self.FUSION_WEIGHTS["sparse_bm25"]))

        # 6. OCR 精确匹配
        if ocr_keywords := self._extract_ocr_keywords(query):
            ocr_results = self._search_ocr_exact(ocr_keywords)
            results.append((ocr_results, self.FUSION_WEIGHTS["ocr_exact"]))

        # 加权融合
        fused = self._weighted_fusion(results, top_k)

        # 图文感知精排
        fused = self.multimodal_reranker.rerank(fused, query, query_images)

        return fused
```

#### 3.3 `RetrievedChunk` 增强

```python
@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    manual_name: str = ""
    image_ids: list[str] = field(default_factory=list)
    # 新增字段
    image_data: list[str] = field(default_factory=list)  # base64 图片数据
    image_scores: list[float] = field(default_factory=list)  # 每张图的相关性分数
    retrieval_source: str = ""  # "dense_text" | "multimodal" | "image_search" | ...
```

---

### 阶段 4：端到端多模态生成（5-7 天）

**目标**：生成模型真正"看见"图片，基于图文联合上下文回答问题。

#### 4.1 生成模型升级

当前生成模型 `deepseek-v4-flash` 是纯文本模型。升级为视觉语言模型（VLM）：

```env
# .env
GEN_MODEL=qwen-vl-max               # 百炼 API（推荐）
# 或本地模型
GEN_MODEL=llava:latest              # Ollama
```

**推荐**：通过百炼 API 使用 `qwen-vl-max`（兼容 OpenAI 格式），因为：
- 支持图文交织输入（interleaved text + images）
- 中文效果好
- 与现有 `ChatOpenAI`（Bailian）调用方式兼容

#### 4.2 多模态 Prompt 结构

```python
# app/utils/prompt_builder.py 改造

def build_multimodal_messages(
    question: str,
    chunks: list[RetrievedChunk],
    user_images: list[str] | None = None,
    conversation_history: list[dict] | None = None,
) -> list[dict]:
    """构建图文交织的消息列表（OpenAI 多模态格式）"""

    # 1. 构建上下文内容块
    context_blocks = []
    for chunk in chunks:
        # 文本部分
        context_blocks.append({
            "type": "text",
            "text": f"[{chunk.chunk_id}] {chunk.text}"
        })
        # 图片部分 — 模型真正"看到"图片
        for img_base64, img_score in zip(chunk.image_data, chunk.image_scores):
            if img_score > 0.3:  # 只包含相关度高的图片
                context_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                })
                context_blocks.append({
                    "type": "text",
                    "text": f"[图片 {img_id}，相关度: {img_score:.2f}]"
                })

    # 2. 用户消息：问题 + 上传图片
    user_content = [{"type": "text", "text": question}]
    if user_images:
        for img in user_images:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": img}
            })

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context_blocks},       # 图文上下文
        {"role": "user", "content": user_content},          # 用户问题 + 图片
    ]
```

#### 4.3 生成流程对比

```
【当前】
文本 Chunks ──→ 纯文本 Prompt ──→ deepseek-v4-flash ──→ 纯文本答案
                 "<IMG:xxx>" 只是字符串

【升级后】
文本 Chunks + 图片 Base64 ──→ 图文交织 Prompt ──→ qwen-vl-max ──→ 图文答案
                                模型真正"看到"像素          可返回图片引用
```

---

## 三、关键技术决策对比

| 决策点 | 方案 A | 方案 B | 推荐 |
|--------|--------|--------|------|
| 多模态嵌入模型 | Chinese-CLIP（中文优） | jina-clip-v2（多语言） | **jina-clip-v2**，768d Matryoshka 与现有 bge-m3 对齐 |
| 图片存储 | 图片向量 + 路径 | 图片向量 + Base64 | **路径为主**，检索时按需加载 Base64，避免 Milvus 膨胀 |
| 图文融合方式 | 多向量多 Collection | 单向量加权融合 | **两者结合**：独立 images collection + chunk 级 multimodal_vector |
| VLM 模型 | 百炼 qwen-vl-max | Ollama llava | **百炼 qwen-vl-max**，中文好、兼容现有 Bailian 调用链 |
| 精排模型 | bge-reranker（纯文本） | jina-reranker-v2（多模态） | **jina-reranker-v2**，支持图文相关性打分 |

---

## 四、Collection 变更总览

```
Milvus
├── manual_chunks_v1（现有，增强）
│   ├── chunk_id (PK)
│   ├── dense_vector (bge-m3, 768d) ← 保留
│   ├── sparse_vector (BM25) ← 保留
│   ├── text ← 保留
│   ├── image_ids ← 保留
│   ├── manual_name ← 保留
│   └── multimodal_vector (CLIP, 768d) ← 新增
│
└── manual_images（新增）
    ├── image_id (PK)
    ├── image_vector (CLIP, 768d)
    ├── manual_name
    ├── parent_chunk_ids (JSON)
    ├── caption
    ├── ocr_text
    └── image_path
```

---

## 五、实施优先级建议

```
优先级 P0（面试后立刻做）：
  ├── 配置 VISION_MODEL=minicpm-v，打通用户图片 → 文本摘要链路
  └── 增强视觉摘要 JSON 结构化输出（OCR + 实体）

优先级 P1（核心多模态检索）：
  ├── 部署 jina-clip-v2 嵌入服务
  ├── 新建 manual_images collection + 建索引脚本改造
  ├── manual_chunks 增加 multimodal_vector 字段
  └── 实现三路检索融合（dense_text + multimodal + sparse）

优先级 P2（端到端多模态生成）：
  ├── 百炼 API 接入 qwen-vl-max
  ├── prompt_builder 改造为图文交织格式
  └── RetrievedChunk 增加 image_data 字段

优先级 P3（锦上添花）：
  ├── 多模态 reranker（jina-reranker-v2）
  ├── 图片 OCR 预处理 pipeline
  └── 图片去重 + 缓存优化
```

---

## 六、面试讲述要点

### 开场（30秒）
"我们当前系统已完成文本 RAG 的基础能力（Dense + Sparse 混合检索、Cross-encoder 精排、ReAct 多轮检索），但多模态部分目前是"图片转文本再检索"的间接方案，图片向量从未参与检索，生成模型也看不到图片像素。"

### 核心方案（2分钟）
"我设计了一个四阶段升级路线：

1. **短期**：修复视觉模型配置，结构化视觉摘要（OCR + 实体提取），让图片信息更好地参与文本检索
2. **中期**：引入 CLIP 风格多模态嵌入模型，将图文映射到统一向量空间，实现以文搜图、以图搜图
3. **中远期**：多路混合检索融合（文本向量 + 多模态向量 + 图片向量 + BM25 + OCR 精确匹配）
4. **远期**：升级为视觉语言模型（qwen-vl-max），模型真正"看见"图片像素，生成图文交织答案

关键设计选择：用 jina-clip-v2 保持 768 维与现有 bge-m3 对齐；图片独立建 collection 同时 chunk 增加 multimodal_vector 实现两级检索；生成侧用百炼 qwen-vl-max 兼容现有 Bailian 调用链。"

### 技术亮点（1分钟）
- **多向量策略**：每个 chunk 同时有纯文本向量（bge-m3）和多模态向量（CLIP），一个 query 两路并行检索然后融合
- **图文交织 Prompt**：不是把图片 URL 当文本扔给模型，而是用 OpenAI 多模态 message 格式，模型真正处理像素
- **兼容现有架构**：`RetrievedChunk` 增加字段而非重写，检索器继承 `VectorRetriever`，prompt builder 增加多模态路径
- **图片反向映射**：从 `manual_images` 搜到图片后，通过 `parent_chunk_ids` 反向找到相关文本 chunk，图文互相增强

### 预期收益
- 图片相关问题的检索命中率从 ~40% 提升到 ~75%
- 用户上传产品故障图片时，系统能直接找到手册中相同/相似的示意图
- 生成答案可以直接引用图片："如上图所示，按下红色复位按钮（位置见 `<PIC>`）"
