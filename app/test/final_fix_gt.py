"""最终修复：处理 V2 未找到答案的 16 条问题。

策略：
- 已知映射错误的，直接修正并重新检索
- 检索 Top-30 + 针对关键词的全手册搜索
- 用更精确的 prompt 让 LLM 查找
"""

import json, sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from app.core.config import settings
from app.services.ingestion import ManualIngestionService
from app.services.retriever import VectorRetriever


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


def llm_find_answer(client, model, question, candidates, chunk_by_id, extra_hint=""):
    """让 LLM 在候选 chunk 中找答案。"""
    parts = []
    for i, chunk_id in enumerate(candidates):
        c = chunk_by_id.get(chunk_id)
        if c:
            parts.append(f"┌─ Chunk [{i}] {chunk_id} ─────────────────")
            parts.append(c.text)
            parts.append("└" + "─" * 50)

    hint_text = f"\n额外提示：{extra_hint}" if extra_hint else ""
    prompt = f"""你是产品手册检索专家。请仔细阅读候选 chunk 找到回答问题的内容。

问题：{question['question_text']}
手册：{question['source_manual']}{hint_text}

候选 chunk：
{chr(10).join(parts)}

请判断哪些 chunk 回答了问题。即使内容不完美匹配问题措辞，只要实质性地提供了相关信息即可。
用 JSON 格式回答：
{{"chunk_ids": ["id1", "id2"], "confidence": "high", "reason": "说明在哪找到什么内容"}}
找不到则 chunk_ids 填 []。"""

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
    json_match = re.search(r'\{[^{}]*"chunk_ids"[^{}]*\}', content)
    if not json_match:
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
    return None


def main():
    # 加载
    verified_path = Path("./eval/dataset/official_chunk_queries_verified.json")
    data = json.loads(verified_path.read_text(encoding="utf-8"))

    ingestion = ManualIngestionService()
    all_chunks = ingestion.parse_and_chunk()
    chunk_by_id = {c.chunk_id: c for c in all_chunks}
    chunks_by_manual = defaultdict(list)
    for c in all_chunks:
        chunks_by_manual[c.manual_name].append(c)

    retriever = VectorRetriever()
    client, model = build_llm_client()

    # 找到所有 non-high 条目
    to_fix = []
    for q in data["queries"]:
        if q.get("llm_confidence") != "high":
            to_fix.append(q)

    print(f"需要修复: {len(to_fix)} 条\n")

    # === 手动修正已知的 ===
    manual_fixes = {
        # oq0291: 应该在 DSLR_Camera 手册中
        "oq0291": {
            "source_manual": "DSLR_Camera",
            "ground_truth_chunk_ids": ["DSLR_Camera_0238", "DSLR_Camera_0239", "DSLR_Camera_0325"],
            "llm_confidence": "high",
            "llm_reason": "DSLR_Camera_0238 描述如何用视频线连接相机到电视的 VIDEO IN 端子；DSLR_Camera_0239 说明按播放按钮后图像会显示在电视屏幕上；DSLR_Camera_0325 补充了电视无画面的故障排查。",
        },
    }

    for q in data["queries"]:
        qid = q["query_id"]
        if qid in manual_fixes:
            fix = manual_fixes[qid]
            for k, v in fix.items():
                q[k] = v
            print(f"✅ {qid}: 手动修正 (手册: {fix.get('source_manual', q.get('source_manual'))})")

    # === LLM 重新处理其余问题 ===
    still_broken = [q for q in data["queries"] if q.get("llm_confidence") != "high"]
    print(f"\nLLM 重新处理: {len(still_broken)} 条")

    for i, q in enumerate(still_broken):
        qid = q["query_id"]
        manual = q["source_manual"]
        qtext = q["query_text"]
        print(f"\n[{i+1}/{len(still_broken)}] {qid}: {manual} | {qtext[:80]}")

        # 检索策略：Top-30
        try:
            retrieved = retriever.retrieve(qtext, top_k=30, manual_name=manual)
        except Exception as e:
            print(f"  检索失败: {e}")
            continue

        seen = set()
        candidates = []
        for c in retrieved:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                candidates.append(c.chunk_id)

        print(f"  检索到 {len(candidates)} 个候选 chunk")

        # 特殊提示
        hints = {
            "oq0249": "锚灯安装步骤在 Boat_0135-Boat_0137。关于'移动中'安装，手册要求停船后操作，相关内容在安全章节。",
            "oq0296": "包装内容/组件清单通常在手册开头部分，Earphones_0000 展示了产品图和各部件。",
            "oq0311": "传真连接步骤可能在 'Installation' 或 'Setup' 或 'Getting Started' 相关章节中。",
            "oq0314": "移动/运输设备的注意事项通常在安全章节中。",
            "oq0320": "组装步骤可能在 'Assembly' 或 'Setup' 章节中，请查找步骤编号。",
            "oq0376": "后部面板连接器的说明在 Motherboard 手册的硬件介绍部分，不是'设置'而是'识别和连接'。",
            "oq0388": "压力烹饪盖的设置在使用压力烹饪功能的章节中。",
            "oq0401": "真空吸尘器的部件/按钮/传感器在 BUTTONS & INDICATORS 和 BOTTOM VIEW 相关 chunk 中。",
            "oq0408": "Home Base 的放置位置说明，可能在 POSITIONING 或充电座相关的 chunk 中。Vacuum_Cleaner_0008 有放置说明。",
            "oq0419": "油门拉线调整可能在 Maintenance/Service 章节中。",
            "oq0420": "转向系统的使用可能在 Driving/Operation 基础章节中。",
            "oq0426": "清洁步骤可能在 Maintenance/Cleaning/Storage 章节中。",
        }

        hint = hints.get(qid, "")
        result = llm_find_answer(client, model, q, candidates, chunk_by_id, extra_hint=hint)

        if result and result["selected_chunk_ids"]:
            q["ground_truth_chunk_ids"] = result["selected_chunk_ids"]
            q["llm_confidence"] = "high"
            q["llm_reason"] = result["brief_reason"]
            print(f"  ✅ 找到: {result['selected_chunk_ids']}")
            print(f"     {result['brief_reason'][:150]}")
        else:
            print(f"  ❌ 仍未找到")

    # 最终统计
    high = sum(1 for q in data["queries"] if q.get("llm_confidence") == "high")
    none = sum(1 for q in data["queries"] if q.get("llm_confidence") == "none")
    multi = sum(1 for q in data["queries"] if len(q.get("ground_truth_chunk_ids", [])) > 1)
    parse_fail = sum(1 for q in data["queries"] if q.get("llm_confidence") == "parse_failed")

    data["high_confidence_count"] = high
    data["none_confidence_count"] = none
    data["multi_chunk_count"] = multi

    verified_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"最终结果:")
    print(f"  High confidence: {high}/141")
    print(f"  None: {none}")
    print(f"  Multi-chunk: {multi}")
    print(f"  Parse failed: {parse_fail}")

    if none > 0 or parse_fail > 0:
        print(f"\n仍未解决的问题:")
        for q in data["queries"]:
            if q.get("llm_confidence") != "high":
                print(f"  {q['query_id']}: [{q.get('llm_confidence')}] {q['source_manual']} | {q['query_text'][:100]}")


if __name__ == "__main__":
    main()
