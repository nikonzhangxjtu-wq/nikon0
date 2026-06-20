"""修复 V2 验证后剩余的 16 个问题条目。

策略：
1. 对 parse 失败的 3 条，重新用更小的 batch 发送
2. 对 "none" 的 13 条，尝试：
   a. 跨手册检索（可能是手册映射错误）
   b. 检索全部 chunk（不是 top-K）
   c. 用英文 embedding 重试中文问题（或反过来）
3. 确实找不到答案的，记录原因
"""

import json, sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from app.core.config import settings
from app.services.ingestion import ManualIngestionService
from app.services.retriever import VectorRetriever

# 可能有跨手册映射问题的问题
CROSS_MANUAL_CANDIDATES = {
    "oq0291": ["DSLR_Camera", "Television"],  # camera → TV 查看，可能在相机手册中
}

# 需要检索全手册的问题
FULL_MANUAL_SEARCH = [
    "oq0249", "oq0296", "oq0311", "oq0314", "oq0320",
    "oq0376", "oq0388", "oq0401", "oq0408", "oq0419",
    "oq0420", "oq0426",
]


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


def format_chunks_for_llm(chunks: list[dict], chunk_by_id: dict) -> str:
    parts = []
    for i, c in enumerate(chunks):
        chunk_id = c["chunk_id"]
        full_chunk = chunk_by_id.get(chunk_id)
        text = full_chunk.text if full_chunk else c.get("text", "")
        parts.append(f"┌─ Chunk [{i}] {chunk_id} ─────────────────")
        parts.append(text)
        parts.append("└" + "─" * 50)
    return "\n".join(parts)


def verify_single_question(client, model, question: dict, candidates: list[dict], chunk_by_id: dict) -> dict | None:
    """让 LLM 仔细判定单个问题的答案。"""
    prompt = f"""你是一个产品手册检索质量评估专家。请仔细阅读下面的用户问题和候选 chunk 的内容。

用户问题：{question['question_text']}
来源手册：{question['source_manual']}

候选 chunk（共 {len(candidates)} 个）：

{format_chunks_for_llm(candidates, chunk_by_id)}

请判断哪些 chunk 回答了用户的问题。记住：
- 答案可能分布在一个或多个 chunk 中
- 仔细阅读每个 chunk 的内容，不要因为标题不匹配就跳过
- 如果问题涉及某个操作，请查看是否有 chunk 包含该操作或其前提/后续步骤
- 只有确实在 chunk 中读到了相关内容才选中

请用 JSON 格式回答：
{{"chunk_ids": ["id1", "id2"], "confidence": "high", "reason": "具体说明在哪个 chunk 中找到什么内容"}}

如果候选 chunk 中确实找不到答案，chunk_ids 填 []。"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
            timeout=120,
        )
        content = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"  LLM 调用失败: {e}")
        return None

    import re
    # 提取 JSON 对象
    json_match = re.search(r'\{[^{}]*"chunk_ids"[^{}]*\}', content)
    if not json_match:
        # 更宽松的匹配
        json_match = re.search(r'\{[^{}]*\}', content)

    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            chunk_ids = obj.get("chunk_ids", [])
            if isinstance(chunk_ids, str):
                chunk_ids = [chunk_ids] if chunk_ids and chunk_ids != "none" else []
            return {
                "selected_chunk_ids": chunk_ids,
                "confidence": obj.get("confidence", "high"),
                "brief_reason": obj.get("reason", ""),
            }
        except json.JSONDecodeError:
            pass

    print(f"  解析失败，原始响应: {content[:200]}")
    return None


def main():
    print("修复剩余问题条目\n")

    # 加载 V2 验证结果
    verified_path = Path("./eval/dataset/official_chunk_queries_verified.json")
    data = json.loads(verified_path.read_text(encoding="utf-8"))
    queries = data["queries"]

    # 加载原始问题数据
    original_path = Path("./eval/dataset/official_chunk_queries.json")
    original_data = json.loads(original_path.read_text(encoding="utf-8"))
    orig_by_id = {q["query_id"]: q for q in original_data["queries"]}

    # 加载 chunks
    ingestion = ManualIngestionService()
    all_chunks = ingestion.parse_and_chunk()
    chunk_by_id = {c.chunk_id: c for c in all_chunks}
    chunks_by_manual = defaultdict(list)
    for c in all_chunks:
        chunks_by_manual[c.manual_name].append(c)

    retriever = VectorRetriever()
    client, model = build_llm_client()

    # 找出需要修复的条目
    to_fix = []
    for q in queries:
        if q.get("llm_confidence") in ("none", "parse_failed"):
            to_fix.append(q)

    print(f"需要修复: {len(to_fix)} 条\n")

    fixed = 0
    for i, q in enumerate(to_fix):
        qid = q["query_id"]
        manual = q["source_manual"]
        qtext = q["query_text"]
        print(f"[{i+1}/{len(to_fix)}] {qid}: {manual} | {qtext[:80]}")

        # 确定要检索的手册
        manuals_to_check = CROSS_MANUAL_CANDIDATES.get(qid, [manual])

        found_result = None
        for check_manual in manuals_to_check:
            if check_manual not in chunks_by_manual:
                print(f"  ⚠ 手册 '{check_manual}' 不存在")
                continue

            # 确定检索策略
            if qid in FULL_MANUAL_SEARCH:
                # 检索该手册所有 chunk
                all_manual_chunks = chunks_by_manual[check_manual]
                candidates = [{"chunk_id": c.chunk_id, "score": 0.0} for c in all_manual_chunks]
                print(f"  检索全部 {len(candidates)} 个 chunk（全手册）")
            else:
                # 检索 Top-25
                try:
                    retrieved = retriever.retrieve(qtext, top_k=25, manual_name=check_manual)
                except Exception as e:
                    print(f"  检索失败: {e}")
                    continue
                seen = set()
                candidates = []
                for c in retrieved:
                    if c.chunk_id not in seen:
                        seen.add(c.chunk_id)
                        candidates.append({"chunk_id": c.chunk_id, "score": round(c.score, 4)})
                print(f"  检索 {len(candidates)} 个候选")

            result = verify_single_question(client, model, q, candidates, chunk_by_id)

            if result and result["selected_chunk_ids"]:
                found_result = result
                if check_manual != manual:
                    # 更换了手册
                    print(f"  ✅ 在 {check_manual} 中找到答案！(原映射: {manual})")
                    q["source_manual"] = check_manual
                break
            elif check_manual != manual:
                print(f"  在 {check_manual} 中未找到")

        if found_result:
            q["ground_truth_chunk_ids"] = found_result["selected_chunk_ids"]
            q["llm_confidence"] = found_result["confidence"]
            q["llm_reason"] = found_result["brief_reason"]
            fixed += 1
            print(f"  ✅ 修复成功: {found_result['selected_chunk_ids']}")
            print(f"     {found_result['brief_reason'][:120]}")
        else:
            # 仍然找不到，保留空 chunk_ids
            print(f"  ❌ 仍然找不到答案")

    print(f"\n修复完成: {fixed}/{len(to_fix)}")

    # 更新统计
    high_count = sum(1 for q in queries if q.get("llm_confidence") == "high")
    none_count = sum(1 for q in queries if q.get("llm_confidence") == "none")
    multi_chunk = sum(1 for q in queries if len(q.get("ground_truth_chunk_ids", [])) > 1)
    parse_fail = sum(1 for q in queries if q.get("llm_confidence") == "parse_failed")

    data["high_confidence_count"] = high_count
    data["none_confidence_count"] = none_count
    data["multi_chunk_count"] = multi_chunk
    data["gt_changed_count"] = sum(
        1 for q in queries
        if q.get("ground_truth_chunk_ids") != ([q.get("old_gt_chunk_id")] if q.get("old_gt_chunk_id") else [])
    )

    verified_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n最终统计:")
    print(f"  High: {high_count}, None: {none_count}, Multi-chunk: {multi_chunk}, Parse fail: {parse_fail}")

    if none_count > 0:
        print(f"\n仍然无法找到答案的问题:")
        for q in queries:
            if q.get("llm_confidence") == "none":
                print(f"  {q['query_id']}: {q['source_manual']} | {q['query_text'][:100]}")


if __name__ == "__main__":
    main()
