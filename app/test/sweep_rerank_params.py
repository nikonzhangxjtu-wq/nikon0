"""Search for optimal reranker parameters (temperature, prior_weight) efficiently.

Caches CrossEncoder scores per query, then sweeps parameter combinations.
"""

import json, math, sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from app.core.config import settings
from app.services.rag_skill.rerank import _load_cross_encoder


def evaluate_with_params(
    queries, ce_scores_cache, temperature, prior_weight, gt_mode="any"
):
    """Evaluate with given parameters. gt_mode: 'any' = any GT chunk counts as hit."""
    hits = 0
    mrr_total = 0.0

    for q in queries:
        qid = q["query_id"]
        gt_ids = set(q["ground_truth_chunk_ids"])
        cache = ce_scores_cache[qid]
        raw_scores = cache["raw_scores"]
        fusion_scores = cache["fusion_scores"]
        chunk_ids = cache["chunk_ids"]

        if not raw_scores:
            # No valid chunks, count as miss
            continue

        # Adaptive confidence
        raw_vals = [float(s) for s in raw_scores]
        ce_mean = sum(raw_vals) / max(len(raw_vals), 1)
        ce_var = sum((v - ce_mean) ** 2 for v in raw_vals) / max(len(raw_vals), 1)
        ce_confidence = min(ce_var / 0.5, 1.0)
        pw = (1 - ce_confidence) * prior_weight

        # Sort by raw CE score
        scored = list(zip(range(len(raw_scores)), raw_scores, fusion_scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # CE softmax
        clamped = [max(min(float(x[1]), 50.0), -50.0) for x in scored]
        max_raw_val = max(clamped)
        exp_scores = [math.exp((s - max_raw_val) / temperature) for s in clamped]
        exp_sum = sum(exp_scores)
        ce_softmax = [es / exp_sum if exp_sum > 0 else 0.0 for es in exp_scores]

        # Fusion softmax
        fusion_temp = max(temperature, 0.3)
        f_clamped = [max(min(float(x[2]), 50.0), -50.0) for x in scored]
        max_f = max(f_clamped)
        f_exp = [math.exp((f - max_f) / fusion_temp) for f in f_clamped]
        f_exp_sum = sum(f_exp)
        fusion_softmax = [fe / f_exp_sum if f_exp_sum > 0 else 0.0 for fe in f_exp]

        # Blend
        blended = [
            (scored[i][0], (1 - pw) * ce_softmax[i] + pw * fusion_softmax[i])
            for i in range(len(scored))
        ]
        blended.sort(key=lambda x: x[1], reverse=True)

        # Check hit
        ranked_chunk_ids = [chunk_ids[blended[j][0]] for j in range(len(blended))]

        hit = False
        best_rank = len(ranked_chunk_ids) + 1
        for rank, cid in enumerate(ranked_chunk_ids, 1):
            if cid in gt_ids:
                hit = True
                best_rank = min(best_rank, rank)
                if rank == 1:
                    hits += 1
                break

        if hit:
            mrr_total += 1.0 / best_rank

    n = len(queries)
    return {
        "hit_at_1": hits / n if n > 0 else 0.0,
        "mrr": mrr_total / n if n > 0 else 0.0,
        "temperature": temperature,
        "prior_weight": prior_weight,
    }


def main():
    # Load dataset
    ds_path = Path("./eval/dataset/official_chunk_queries_verified.json")
    data = json.loads(ds_path.read_text(encoding="utf-8"))
    queries = data["queries"]
    print(f"Loaded {len(queries)} queries")

    # Initialize retriever & CrossEncoder
    from app.services.retriever import VectorRetriever

    retriever = VectorRetriever()
    ce_model = _load_cross_encoder(settings.rerank_model_name)
    if ce_model is None:
        print("ERROR: Cannot load CrossEncoder")
        sys.exit(1)

    # Disable reranking for caching — we want raw fusion scores + raw CE scores
    orig_rerank_enabled = settings.rerank_enabled
    settings.rerank_enabled = False

    # Cache: for each query, retrieve top-20 and compute CE scores
    print("Caching retrieval results (rerank disabled) and CrossEncoder scores...")
    ce_scores_cache = {}
    for i, q in enumerate(queries):
        qid = q["query_id"]
        manual = q.get("source_manual", "")
        print(f"  [{i+1}/{len(queries)}] {qid}", end="\r")

        # Retrieve (rerank disabled, so scores are raw fusion scores)
        try:
            retrieved = retriever.retrieve(q["query_text"], top_k=5, manual_name=manual)
        except Exception as e:
            print(f"\n  WARN: {qid} retrieval failed: {e}")
            ce_scores_cache[qid] = {
                "raw_scores": [],
                "fusion_scores": [],
                "chunk_ids": [],
            }
            continue

        # Extract texts, fusion scores, chunk IDs
        texts = []
        fusion_scores = []
        chunk_ids = []
        for hit in retrieved:
            text = str(hit.text).strip() if hit.text else ""
            if not text:
                continue
            texts.append(text)
            fusion_scores.append(hit.score)
            chunk_ids.append(hit.chunk_id)

        if not texts:
            ce_scores_cache[qid] = {
                "raw_scores": [],
                "fusion_scores": [],
                "chunk_ids": [],
            }
            continue

        # CrossEncoder raw scores
        try:
            raw_scores = ce_model.predict(list(zip([q["query_text"]] * len(texts), texts)))
            if hasattr(raw_scores, "tolist"):
                raw_scores = raw_scores.tolist()
            else:
                raw_scores = list(raw_scores)
        except Exception:
            raw_scores = [0.0] * len(texts)

        ce_scores_cache[qid] = {
            "raw_scores": [float(s) for s in raw_scores],
            "fusion_scores": [float(s) for s in fusion_scores],
            "chunk_ids": chunk_ids,
        }

    # Restore rerank setting
    settings.rerank_enabled = orig_rerank_enabled

    print(f"\nCached scores for {len(ce_scores_cache)} queries\n")

    # Parameter sweep
    temperatures = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    prior_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    print(f"{'Temp':>6} {'Prior':>6} {'Hit@1':>8} {'MRR':>8}")
    print("-" * 35)

    best = None
    results = []
    for temp in temperatures:
        for pw in prior_weights:
            r = evaluate_with_params(queries, ce_scores_cache, temp, pw)
            results.append(r)
            marker = ""
            if best is None or r["hit_at_1"] > best["hit_at_1"]:
                best = r
                marker = " <<"
            print(f"{temp:>6.2f} {pw:>6.2f} {r['hit_at_1']:>7.1%} {r['mrr']:>7.4f}{marker}")

    print(f"\nBest: temp={best['temperature']}, prior={best['prior_weight']}, "
          f"Hit@1={best['hit_at_1']:.1%}, MRR={best['mrr']:.4f}")

    # Also show CE-only stats (raw score ordering, no softmax)
    ce_raw_hits = 0
    ce_raw_mrr = 0.0
    for q in queries:
        cache = ce_scores_cache.get(q["query_id"], {})
        raw = cache.get("raw_scores", [])
        cids = cache.get("chunk_ids", [])
        gt_ids = set(q["ground_truth_chunk_ids"])
        if not raw:
            continue
        scored = sorted(zip(range(len(raw)), raw, cids), key=lambda x: x[1], reverse=True)
        for rank, (_, _, cid) in enumerate(scored, 1):
            if cid in gt_ids:
                ce_raw_mrr += 1.0 / rank
                if rank == 1:
                    ce_raw_hits += 1
                break

    n = len(queries)
    print(f"\nCE raw score (no normalization): Hit@1={ce_raw_hits/n:.1%}, MRR={ce_raw_mrr/n:.4f}")
    print(f"Current best (softmax temp={best['temperature']}): Hit@1={best['hit_at_1']:.1%}")


if __name__ == "__main__":
    main()
