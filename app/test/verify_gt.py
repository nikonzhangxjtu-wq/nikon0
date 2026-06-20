"""用 LLM 验证并纠正官方问题数据集的 Ground Truth。

流程：
1. 加载 official_chunk_queries.json
2. 对每条问题，用 retriever（带 manual_name 过滤）获取 Top-5 chunk
3. 将问题 + 候选 chunk 批量发给 LLM 判定最佳 chunk
4. 输出修正后的数据集
"""

import json, re, sys, time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from app.core.config import settings
from app.services.ingestion import ManualIngestionService
from app.services.retriever import VectorRetriever

BATCH_SIZE = 8  # 每批发给 LLM 的问题数


def build_llm_client():
    """构建 LLM 客户端。"""
    from openai import OpenAI

    if settings.bailian_api_key:
        return OpenAI(
            api_key=settings.bailian_api_key,
            base_url=settings.bailian_base_url,
        ), settings.simple_llm_model
    else:
        return OpenAI(
            api_key="ollama",
            base_url=f"{settings.ollama_base_url}/v1",
        ), settings.simple_llm_model


def verify_batch(client, model: str, batch: list[dict]) -> list[dict | None]:
    """让 LLM 判定一批问题的正确 chunk。

    每项: {question_id, question_text, candidates: [{chunk_id, text_preview}]}
    返回: [{question_id, selected_chunk_id, confidence, reason}] 或 None
    """
    # 构建 prompt
    parts = ["你是一个检索质量评估专家。对每个问题，从候选 chunk 中选择最能回答该问题的 chunk。"]
    parts.append("如果没有 chunk 能回答，请选 'none'。\n")
    parts.append("格式：对每个问题用以下 JSON 格式回答：")
    parts.append('{"qid": "问题ID", "chunk_id": "选中的chunk_id 或 none", "confidence": "high/medium/low", "brief_reason": "一句话理由"}\n')

    for i, item in enumerate(batch):
        parts.append(f"--- 问题 {i+1} ---")
        parts.append(f"ID: {item['question_id']}")
        parts.append(f"问题: {item['question_text']}")
        parts.append(f"候选 chunk:")
        for j, c in enumerate(item['candidates']):
            parts.append(f"  [{j}] chunk_id={c['chunk_id']}")
            # 截断过长文本
            text = c['text'][:500].replace('\n', ' ')
            parts.append(f"      内容: {text}")
        parts.append("")

    prompt = "\n".join(parts)
    prompt += "\n请按 JSON 数组格式返回所有问题的判定结果。"

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048,
            timeout=60,
        )
        content = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"  LLM 调用失败: {e}")
        return [None] * len(batch)

    # 解析 JSON 数组
    results = []
    # 尝试提取 JSON 部分
    json_matches = re.findall(r'\{[^}]+\}', content)
    for item in batch:
        # 查找匹配的 JSON
        found = None
        for m in json_matches:
            try:
                obj = json.loads(m)
                if obj.get("qid") == item['question_id']:
                    found = obj
                    break
            except json.JSONDecodeError:
                continue
        if found:
            results.append({
                "question_id": item['question_id'],
                "selected_chunk_id": found.get("chunk_id", "none"),
                "confidence": found.get("confidence", "low"),
                "brief_reason": found.get("brief_reason", ""),
            })
        else:
            results.append(None)

    return results


def main():
    # 加载数据集
    dataset_path = Path("./eval/dataset/official_chunk_queries.json")
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    queries = data["queries"]
    print(f"加载 {len(queries)} 条问题")

    # 加载 chunks
    ingestion = ManualIngestionService()
    all_chunks = ingestion.parse_and_chunk()
    chunk_by_id = {c.chunk_id: c for c in all_chunks}
    chunks_by_manual = defaultdict(list)
    for c in all_chunks:
        chunks_by_manual[c.manual_name].append(c)

    # 初始化 retriever & LLM client
    retriever = VectorRetriever()
    client, model = build_llm_client()
    print(f"LLM: {model}")

    # 第一步：对每条问题，用 manual_name filter 检索 Top-5
    print("\n第1步：为每条问题检索候选 chunk...")
    enriched = []
    for i, q in enumerate(queries):
        manual = q["source_manual"]
        if manual not in chunks_by_manual:
            continue
        try:
            retrieved = retriever.retrieve(q["query_text"], top_k=5, manual_name=manual)
        except Exception as e:
            print(f"  [{q['query_id']}] 检索失败: {e}")
            continue

        candidates = []
        for c in retrieved:
            if c.chunk_id in chunk_by_id:
                candidates.append({
                    "chunk_id": c.chunk_id,
                    "text": chunk_by_id[c.chunk_id].text[:600],
                    "score": round(c.score, 4),
                })

        enriched.append({
            "question_id": q["query_id"],
            "question_text": q["query_text"],
            "source_manual": manual,
            "language": q.get("language", "zh"),
            "candidates": candidates,
            "old_gt": q.get("ground_truth_chunk_ids", [None])[0],
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(queries)}]")

    print(f"  共 {len(enriched)} 条有效问题")

    # 第二步：分批发给 LLM 验证
    print(f"\n第2步：LLM 逐批验证 GT（每批 {BATCH_SIZE} 条）...")
    verified_results = []

    for batch_start in range(0, len(enriched), BATCH_SIZE):
        batch = enriched[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(enriched) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  批次 {batch_num}/{total_batches} ({batch_start+1}-{batch_start+len(batch)})...", end=" ", flush=True)

        try:
            results = verify_batch(client, model, batch)
        except Exception as e:
            print(f"失败: {e}")
            results = [None] * len(batch)

        ok = sum(1 for r in results if r is not None)
        print(f"解析成功 {ok}/{len(batch)}")

        for item, result in zip(batch, results):
            if result and result["selected_chunk_id"] != "none" and result["selected_chunk_id"]:
                verified_results.append({
                    "query_id": item["question_id"],
                    "query_text": item["question_text"],
                    "source_manual": item["source_manual"],
                    "ground_truth_chunk_ids": [result["selected_chunk_id"]],
                    "query_type": "official_verified",
                    "language": item["language"],
                    "llm_confidence": result["confidence"],
                    "llm_reason": result["brief_reason"],
                    "old_gt_chunk_id": item["old_gt"],
                })
            else:
                # LLM 认为没找到合适 chunk，保留旧 GT 但标记
                verified_results.append({
                    "query_id": item["question_id"],
                    "query_text": item["question_text"],
                    "source_manual": item["source_manual"],
                    "ground_truth_chunk_ids": [item["old_gt"]] if item["old_gt"] else [],
                    "query_type": "official_verified",
                    "language": item["language"],
                    "llm_confidence": "none",
                    "llm_reason": result["brief_reason"] if result else "parse_failed",
                    "old_gt_chunk_id": item["old_gt"],
                })

    # 第三步：统计变更
    changed = sum(1 for v in verified_results if v["ground_truth_chunk_ids"][0] != v["old_gt_chunk_id"])
    none_count = sum(1 for v in verified_results if v["llm_confidence"] == "none")
    print(f"\n第3步：统计")
    print(f"  GT 变更: {changed}/{len(verified_results)} ({changed/len(verified_results):.0%})")
    print(f"  LLM 认为无合适 chunk: {none_count}")

    # 第四步：导出
    out_path = Path("./eval/dataset/official_chunk_queries_verified.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "description": "Chunk-level eval from official questions, GT verified by LLM",
        "total_queries": len(verified_results),
        "gt_changed_count": changed,
        "queries": verified_results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n验证后数据集导出: {out_path}")

    # 打印变更示例
    print("\n=== GT 变更示例 ===")
    showed = 0
    for v in verified_results:
        if v["ground_truth_chunk_ids"][0] != v["old_gt_chunk_id"] and showed < 10:
            showed += 1
            old_text = chunk_by_id.get(v["old_gt_chunk_id"], None) if v["old_gt_chunk_id"] else None
            new_text = chunk_by_id.get(v["ground_truth_chunk_ids"][0], None)
            print(f"\n  [{v['query_id']}] {v['source_manual']}")
            print(f"  Q: {v['query_text'][:100]}")
            print(f"  旧GT: {v['old_gt_chunk_id']} | {old_text.text[:80].replace(chr(10), ' ') if old_text else '?'}")
            print(f"  新GT: {v['ground_truth_chunk_ids'][0]} | {new_text.text[:80].replace(chr(10), ' ') if new_text else '?'}")
            print(f"  置信度: {v['llm_confidence']} | 理由: {v['llm_reason']}")


if __name__ == "__main__":
    main()
