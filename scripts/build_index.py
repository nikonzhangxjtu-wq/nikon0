"""从手册构建向量索引：解析切块 → Ollama 嵌入 → 写入 Milvus。

嵌入：英文手册 stem（见 ``english_manual_naming`` 白名单）用 ``EMBED_MODEL_EN``，
其余中文手册用 ``EMBED_MODEL_ZH``；两模型输出维数必须一致，否则无法写入同一
``dense_vector`` 字段。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 允许 `python scripts/build_index.py` 从项目根运行
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from langchain_ollama import OllamaEmbeddings
from llama_index.core import Document
from pymilvus import MilvusClient

from app.core.config import settings
from app.services.ingestion import ManualChunk, ManualIngestionService
from app.services.milvus_create import build_collection, _create_vector_index
from app.utils.manual_lang import manual_stem_uses_english_embedding
# 与切块默认上限对齐并留余量（VARCHAR 上限受 Milvus 版本限制，一般 ≤ 65535）
TEXT_MAX_LEN = 16384
IMAGE_IDS_MAX_LEN = 8192
EMBED_BATCH = 32


def _milvus_client() -> MilvusClient:
    kwargs: dict = {"uri": settings.milvus_uri, "db_name": settings.milvus_db_name}
    if settings.milvus_token:
        kwargs["token"] = settings.milvus_token
    return MilvusClient(**kwargs)


def _probe_dim(embeddings: OllamaEmbeddings) -> int:
    v = embeddings.embed_query(".")
    return len(v)


def _chunks_to_documents(chunks: list[ManualChunk]) -> list[Document]:
    return [
        Document(
            text=c.text,
            metadata={
                "chunk_id": c.chunk_id,
                "manual_name": c.manual_name,
                "image_ids": c.image_ids,
            },
        )
        for c in chunks
    ]


def _embed_batch_bilingual(
    embeddings_en: OllamaEmbeddings,
    embeddings_zh: OllamaEmbeddings,
    batch_chunks: list[ManualChunk],
    texts: list[str],
) -> list[list[float]]:
    """同一 batch 内按 chunk.manual_name 分流到两个嵌入客户端（中英不同模型时使用）。"""
    if len(batch_chunks) != len(texts):
        raise ValueError("chunks 与 texts 长度不一致")
    n = len(texts)
    out: list[list[float] | None] = [None] * n
    en_indices = [i for i, c in enumerate(batch_chunks) if manual_stem_uses_english_embedding(c.manual_name)]
    zh_indices = [i for i, c in enumerate(batch_chunks) if not manual_stem_uses_english_embedding(c.manual_name)]
    if en_indices:
        vecs = embeddings_en.embed_documents([texts[i] for i in en_indices])
        for k, idx in enumerate(en_indices):
            out[idx] = vecs[k]
    if zh_indices:
        vecs = embeddings_zh.embed_documents([texts[i] for i in zh_indices])
        for k, idx in enumerate(zh_indices):
            out[idx] = vecs[k]
    if any(v is None for v in out):
        raise RuntimeError("中英嵌入分流未覆盖全部切片")
    return [v for v in out]  # type: ignore[misc]


def _truncate(text: str, max_len: int, label: str, warned: list[bool]) -> str:
    if len(text) <= max_len:
        return text
    if not warned[0]:
        print(f"[WARN] 部分 {label} 超过 schema 长度 {max_len}，已截断（仅提示一次）")
        warned[0] = True
    return text[:max_len]

def main() -> None:
    embeddings_en = OllamaEmbeddings(
        model=settings.embed_model_en,
        base_url=settings.ollama_base_url,
    )
    embeddings_zh = OllamaEmbeddings(
        model=settings.embed_model_zh,
        base_url=settings.ollama_base_url,
    )
    dim_en = _probe_dim(embeddings_en)
    dim_zh = _probe_dim(embeddings_zh)
    if dim_en != dim_zh:
        print(
            f"[ERROR] EMBED_MODEL_EN 维数={dim_en} 与 EMBED_MODEL_ZH 维数={dim_zh} 不一致，"
            "无法在单一 Milvus dense_vector 字段中共存；请改用输出维数相同的两个模型，"
            "或拆成两套 collection。"
        )
        return
    dim = dim_en
    print(
        f"[INFO] 双路嵌入 dim={dim} | EN={settings.embed_model_en} | ZH={settings.embed_model_zh}"
    )

    ingestion = ManualIngestionService(settings.manual_dir)
    chunks = ingestion.parse_and_chunk()

    print(f"[INFO] 已解析切片数: {len(chunks)}")
    if not chunks:
        print("[WARN] 未解析到任何切片，请检查 `手册/` 下是否有可解析的 `.txt`。")
        return

    documents = _chunks_to_documents(chunks)

    client = _milvus_client()
    build_collection(client, vector_dim=dim)

    # 建完库立刻做一次 schema 自检：避免「create 像是成功了，但服务端实际没有 dense_vector
    # 字段」的沉默失败（在 Milvus Lite 上开启 BM25 Function 时曾观察到此类状态）。
    if hasattr(client, "describe_collection"):
        try:
            desc = client.describe_collection(collection_name=settings.milvus_collection)
            fields = desc.get("fields") if isinstance(desc, dict) else getattr(desc, "fields", None)
            field_names: list[str] = []
            if fields:
                for f in fields:
                    if isinstance(f, dict):
                        field_names.append(str(f.get("name", "")))
                    else:
                        field_names.append(str(getattr(f, "name", "")))
            print(f"[INFO] 集合字段: {field_names}")
            if not field_names:
                print("[WARN] describe_collection 未返回字段列表，跳过建库字段自检")
            elif "dense_vector" not in field_names:
                raise RuntimeError(
                    "集合缺少 dense_vector 字段，build_collection 未按预期创建。"
                    "在 Milvus Lite 上请确认 MILVUS_ENABLE_BM25=False，并删除旧的 .db 文件后重建。"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] 建库自检失败: {exc}")
            return
    else:
        print("[INFO] Milvus 客户端不支持 describe_collection，跳过建库自检")

    success = 0
    failed = 0
    truncate_warned = [False]

    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start : start + EMBED_BATCH]
        batch_docs = documents[start : start + EMBED_BATCH]
        texts = [d.text for d in batch_docs]
        try:
            vectors = _embed_batch_bilingual(embeddings_en, embeddings_zh, batch, texts)
        except Exception as exc:
            failed += len(batch)
            print(f"[ERROR] 批量嵌入失败（本批 {len(batch)} 条）: {exc}")
            continue

        if len(vectors) != len(batch):
            failed += len(batch)
            print("[ERROR] 嵌入返回条数与批次不一致，跳过本批。")
            continue

        rows: list[dict] = []
        for chunk, vec in zip(batch, vectors):
            if len(vec) != dim:
                failed += 1
                print(f"[ERROR] 向量维数异常 chunk_id={chunk.chunk_id}: {len(vec)} != {dim}")
                continue
            text = _truncate(chunk.text, TEXT_MAX_LEN, "text", truncate_warned)
            ids_json = json.dumps(chunk.image_ids, ensure_ascii=False)
            ids_json = _truncate(ids_json, IMAGE_IDS_MAX_LEN, "image_ids", truncate_warned)
            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "dense_vector": vec,
                    "text": text,
                    "image_ids": ids_json,
                    "manual_name": chunk.manual_name,
                }
            )
        if not rows:
            continue
        try:
            client.insert(collection_name=settings.milvus_collection, data=rows)
            success += len(rows)
        except Exception as exc:
            failed += len(rows)
            print(f"[ERROR] Milvus insert 失败: {exc}")

    index_ok = True
    try:
        _create_vector_index(client)
    except Exception as exc:  # noqa: BLE001
        index_ok = False
        print(f"[ERROR] 创建向量索引或 load 失败（数据可能已写入）: {exc}")

    load_ok = False
    if index_ok:
        try:
            state = None
            try:
                state = client.get_load_state(collection_name=settings.milvus_collection)
            except Exception:
                state = "<unknown>"
            print(f"[INFO] collection 已建索引并 load, state={state}")
            load_ok = True
        except Exception as exc:  # noqa: BLE001
            print(
                f"[ERROR] load_collection 失败: {exc}. "
                "常见原因：某个 vector 字段未建索引（Milvus 2.6 要求所有向量字段都建索引才能 load）。"
            )

    total = success + failed
    print(f"[INFO] 集合名: {settings.milvus_collection} | Milvus: {settings.milvus_uri}")
    print(
        f"[INFO] 写入成功: {success} 条，失败: {failed} 条 | index_ok={index_ok} | load_ok={load_ok}"
    )
    if total > 0:
        print(f"[INFO] 失败占比: {failed / total:.2%}")


if __name__ == "__main__":
    main()
