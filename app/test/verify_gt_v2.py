"""用 LLM 严谨验证并纠正官方问题数据集的 Ground Truth — V2。

改进相对 V1：
1. 修正手册映射错误（PressureCooker_Airfryer vs Airfryer）
2. 检索 Top-15 候选 chunk（非 Top-5）
3. 发送完整 chunk 文本给 LLM（不截断）
4. LLM 可选定多个 chunk（答案可能横跨多个 chunk）
5. 所有答案必须存在，confidence 必须为 high
6. 更详细的 prompt 指导 LLM 仔细判定
"""

import json, re, sys, time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from app.core.config import settings
from app.services.ingestion import ManualIngestionService
from app.services.retriever import VectorRetriever

# === 手工修正手册映射 ===
MANUAL_MAPPING_FIXES = {
    # 这些问题是关于 "multi-use pressure cooker and air fryer" 的，
    # 应该映射到 PressureCooker_Airfryer 而不是 Airfryer
    "oq0387": "PressureCooker_Airfryer",
    "oq0388": "PressureCooker_Airfryer",
    "oq0389": "PressureCooker_Airfryer",
    "oq0390": "PressureCooker_Airfryer",
    "oq0391": "PressureCooker_Airfryer",
    "oq0392": "PressureCooker_Airfryer",
    "oq0393": "PressureCooker_Airfryer",
    "oq0394": "PressureCooker_Airfryer",
    "oq0395": "PressureCooker_Airfryer",
    "oq0396": "PressureCooker_Airfryer",
    "oq0397": "PressureCooker_Airfryer",
    "oq0398": "PressureCooker_Airfryer",
    "oq0399": "PressureCooker_Airfryer",
    "oq0400": "PressureCooker_Airfryer",
    # oq0291 是关于在电视上查看相机图像的，源手册可能是 DSLR_Camera 或 Television
    # 先保留 Television，让 LLM 来判定
}

BATCH_SIZE = 5  # 减少批量大小，因为每块文本更长了
RETRIEVAL_K = 15


def build_llm_client():
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


def format_chunks_for_llm(candidates: list[dict], chunk_by_id: dict) -> str:
    """格式化候选 chunk 供 LLM 审阅。返回完整文本。"""
    parts = []
    for i, c in enumerate(candidates):
        chunk_id = c["chunk_id"]
        full_chunk = chunk_by_id.get(chunk_id)
        if full_chunk:
            text = full_chunk.text
        else:
            text = c.get("text", "")
        parts.append(f"┌─ Chunk [{i}] chunk_id={chunk_id} ─────────────────")
        parts.append(text)
        parts.append("└" + "─" * 50)
    return "\n".join(parts)


def verify_batch(client, model: str, batch: list[dict], chunk_by_id: dict) -> list[dict | None]:
    """让 LLM 严谨判定一批问题的正确 chunk(s)。

    返回: [{question_id, selected_chunk_ids: [str], confidence, brief_reason}] 或 None
    """
    prompt_parts = [
        "你是一个产品手册检索质量评估专家。下面会给你多个用户问题，以及从产品手册中检索到的候选 chunk。",
        "",
        "你的任务：",
        "1. 仔细阅读每个 chunk 的完整内容",
        "2. 判断哪些 chunk 能回答用户的问题（可能有多个 chunk，答案可能分布在不同章节）",
        "3. 如果问题涉及产品特性、操作步骤、安全警告、故障排除等，需要在手册中找到明确对应的内容",
        "",
        "重要规则：",
        "- 可以选定 0 个、1 个或 多个 chunk（用 chunk_id 列表表示）",
        "- 如果选多个 chunk，说明这多个 chunk 共同覆盖了完整答案",
        "- 所有被选中的 chunk 必须确实包含问题相关的信息",
        "- 如果候选 chunk 中找不到答案，chunk_ids 填 []",
        "- 每个答案必须是高置信度的（你真的在 chunk 中读到了相关内容）",
        "",
        "按以下 JSON 格式回答（每个问题一个 JSON 对象，用数组包裹）：",
        '[{"qid": "问题ID", "chunk_ids": ["chunk_id1", "chunk_id2"], "confidence": "high", "reason": "说明在哪个/哪些 chunk 中找到的答案，具体内容是什么"}]',
        "",
    ]

    for i, item in enumerate(batch):
        prompt_parts.append("=" * 70)
        prompt_parts.append(f"问题 {i+1}：")
        prompt_parts.append(f"  ID: {item['question_id']}")
        prompt_parts.append(f"  手册: {item['source_manual']}")
        prompt_parts.append(f"  问题: {item['question_text']}")
        prompt_parts.append("")
        prompt_parts.append(f"候选 chunk（共 {len(item['candidates'])} 个）：")
        prompt_parts.append(format_chunks_for_llm(item["candidates"], chunk_by_id))
        prompt_parts.append("")

    prompt_parts.append("=" * 70)
    prompt_parts.append("请仔细分析后，按 JSON 数组格式返回所有问题的判定结果。")
    prompt_parts.append("记住：如果候选 chunk 中真正有答案，confidence 才是 high；如果没有答案，chunk_ids 填 []。")

    prompt = "\n".join(prompt_parts)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4096,
            timeout=120,
        )
        content = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"  LLM 调用失败: {e}")
        return [None] * len(batch)

    # 解析 JSON
    results = []
    # 尝试多种解析策略
    json_objects = re.findall(r'\{[^{}]*"qid"[^{}]*\}', content)
    # 更宽松的匹配：匹配花括号包围的 JSON
    if not json_objects:
        json_objects = re.findall(r'\{[^{}]*\}', content)

    for item in batch:
        found = None
        for m in json_objects:
            try:
                obj = json.loads(m)
                if obj.get("qid") == item["question_id"]:
                    found = obj
                    break
            except json.JSONDecodeError:
                continue

        if found:
            chunk_ids = found.get("chunk_ids", [])
            if isinstance(chunk_ids, str):
                chunk_ids = [chunk_ids] if chunk_ids and chunk_ids != "none" else []
            results.append({
                "question_id": item["question_id"],
                "selected_chunk_ids": chunk_ids,
                "confidence": found.get("confidence", "high"),
                "brief_reason": found.get("reason", ""),
            })
        else:
            results.append(None)

    return results


def main():
    print("=" * 60)
    print("GT 验证 V2：严谨版数据集构建")
    print("=" * 60)

    # 加载数据集
    dataset_path = Path("./eval/dataset/official_chunk_queries.json")
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    queries = data["queries"]
    print(f"加载 {len(queries)} 条问题")

    # 应用手册映射修正
    fixed_count = 0
    for q in queries:
        if q["query_id"] in MANUAL_MAPPING_FIXES:
            old = q["source_manual"]
            q["source_manual"] = MANUAL_MAPPING_FIXES[q["query_id"]]
            print(f"  修正映射: {q['query_id']} {old} → {q['source_manual']}")
            fixed_count += 1
    print(f"共修正 {fixed_count} 条手册映射")

    # 加载 chunks
    ingestion = ManualIngestionService()
    all_chunks = ingestion.parse_and_chunk()
    chunk_by_id = {c.chunk_id: c for c in all_chunks}
    chunks_by_manual = defaultdict(list)
    for c in all_chunks:
        chunks_by_manual[c.manual_name].append(c)
    print(f"加载 {len(all_chunks)} 个 chunk，涵盖 {len(chunks_by_manual)} 本手册")

    # 初始化 retriever & LLM client
    retriever = VectorRetriever()
    client, model = build_llm_client()
    print(f"LLM: {model}")

    # 第一步：对每条问题，用 manual_name filter 检索 Top-K 候选
    print(f"\n第1步：检索候选 chunk（Top-{RETRIEVAL_K}，manual_name 过滤）...")
    enriched = []
    missing_manuals = set()
    for i, q in enumerate(queries):
        manual = q["source_manual"]
        if manual not in chunks_by_manual:
            missing_manuals.add(manual)
            print(f"  ⚠ {q['query_id']}: 手册 '{manual}' 不存在！跳过")
            continue

        try:
            retrieved = retriever.retrieve(q["query_text"], top_k=RETRIEVAL_K, manual_name=manual)
        except Exception as e:
            print(f"  [{q['query_id']}] 检索失败: {e}")
            continue

        candidates = []
        seen_ids = set()
        for c in retrieved:
            if c.chunk_id in seen_ids:
                continue
            seen_ids.add(c.chunk_id)
            candidates.append({
                "chunk_id": c.chunk_id,
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

    if missing_manuals:
        print(f"\n  缺失手册: {missing_manuals}")

    print(f"  共 {len(enriched)} 条有效问题")

    # 第二步：分批发给 LLM 验证
    print(f"\n第2步：LLM 逐批严谨验证 GT（每批 {BATCH_SIZE} 条）...")
    verified_results = []
    parse_failures = 0

    for batch_start in range(0, len(enriched), BATCH_SIZE):
        batch = enriched[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(enriched) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n  批次 {batch_num}/{total_batches} ({batch_start+1}-{batch_start+len(batch)})...", flush=True)

        try:
            results = verify_batch(client, model, batch, chunk_by_id)
        except Exception as e:
            print(f"  失败: {e}")
            results = [None] * len(batch)

        ok = sum(1 for r in results if r is not None)
        print(f"  解析成功 {ok}/{len(batch)}")

        for item, result in zip(batch, results):
            if result is None:
                parse_failures += 1
                # 解析失败，保留旧 GT
                verified_results.append({
                    "query_id": item["question_id"],
                    "query_text": item["question_text"],
                    "source_manual": item["source_manual"],
                    "ground_truth_chunk_ids": [item["old_gt"]] if item["old_gt"] else [],
                    "query_type": "official_verified",
                    "language": item["language"],
                    "llm_confidence": "parse_failed",
                    "llm_reason": "LLM 响应解析失败",
                    "old_gt_chunk_id": item["old_gt"],
                })
                print(f"    ⚠ {item['question_id']}: 解析失败，保留旧 GT")
            elif not result["selected_chunk_ids"]:
                # LLM 没找到答案
                verified_results.append({
                    "query_id": item["question_id"],
                    "query_text": item["question_text"],
                    "source_manual": item["source_manual"],
                    "ground_truth_chunk_ids": [],
                    "query_type": "official_verified",
                    "language": item["language"],
                    "llm_confidence": "none",
                    "llm_reason": result.get("brief_reason", "未找到答案"),
                    "old_gt_chunk_id": item["old_gt"],
                })
                print(f"    ⚠ {item['question_id']}: LLM 未找到答案 - {result.get('brief_reason', '')[:80]}")
            else:
                verified_results.append({
                    "query_id": item["question_id"],
                    "query_text": item["question_text"],
                    "source_manual": item["source_manual"],
                    "ground_truth_chunk_ids": result["selected_chunk_ids"],
                    "query_type": "official_verified",
                    "language": item["language"],
                    "llm_confidence": result["confidence"],
                    "llm_reason": result.get("brief_reason", ""),
                    "old_gt_chunk_id": item["old_gt"],
                })

    # 第三步：统计
    multi_chunk = sum(1 for v in verified_results if len(v["ground_truth_chunk_ids"]) > 1)
    none_count = sum(1 for v in verified_results if v["llm_confidence"] == "none")
    parse_fail = sum(1 for v in verified_results if v["llm_confidence"] == "parse_failed")
    high_count = sum(1 for v in verified_results if v["llm_confidence"] == "high")
    changed = sum(
        1 for v in verified_results
        if v["ground_truth_chunk_ids"] != ([v["old_gt_chunk_id"]] if v["old_gt_chunk_id"] else [])
    )

    print(f"\n{'='*60}")
    print(f"第3步：统计")
    print(f"  总条目: {len(verified_results)}")
    print(f"  High confidence: {high_count}")
    print(f"  None (未找到答案): {none_count}")
    print(f"  解析失败: {parse_fail}")
    print(f"  多 chunk 条目: {multi_chunk}")
    print(f"  GT 变更（相对旧版本）: {changed}")

    # 第四步：导出
    out_path = Path("./eval/dataset/official_chunk_queries_verified.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "description": "Chunk-level eval from official questions, GT verified by LLM V2",
        "total_queries": len(verified_results),
        "gt_changed_count": changed,
        "multi_chunk_count": multi_chunk,
        "high_confidence_count": high_count,
        "none_confidence_count": none_count,
        "queries": verified_results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n验证后数据集导出: {out_path}")

    # 打印各类示例
    print(f"\n=== 多 Chunk 示例（共 {multi_chunk} 条）===")
    showed = 0
    for v in verified_results:
        if len(v["ground_truth_chunk_ids"]) > 1 and showed < 10:
            showed += 1
            print(f"\n  [{v['query_id']}] {v['source_manual']}")
            print(f"  Q: {v['query_text'][:120]}")
            print(f"  Chunk IDs: {v['ground_truth_chunk_ids']}")
            print(f"  Reason: {v['llm_reason'][:150]}")

    if none_count > 0:
        print(f"\n=== 未找到答案的条目（共 {none_count} 条）===")
        for v in verified_results:
            if v["llm_confidence"] == "none":
                print(f"\n  [{v['query_id']}] {v['source_manual']}")
                print(f"  Q: {v['query_text'][:120]}")
                print(f"  Reason: {v['llm_reason'][:200]}")


if __name__ == "__main__":
    main()
