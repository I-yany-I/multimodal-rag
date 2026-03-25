"""
evaluate.py
-----------
评估多模态 RAG 系统的检索性能。

评估指标：
- Recall@K: 前 K 个结果中包含正确答案的比例
- MRR (Mean Reciprocal Rank): 正确结果排名的倒数均值
- NDCG@K: 归一化折扣累积增益（考虑排序质量）

运行方式: python evaluate.py
"""

import json
import os
import time
from typing import List, Dict, Tuple

import numpy as np
import yaml

from src.encoder import CLIPEncoder
from src.retriever import FAISSRetriever


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Recall@K: 前 K 个结果中命中相关图像的比例。"""
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(top_k & relevant) / len(relevant)


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """计算单个 query 的 Reciprocal Rank。"""
    relevant = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """NDCG@K: 考虑排序位置的检索质量指标。"""
    relevant = set(relevant_ids)
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in relevant:
            dcg += 1.0 / np.log2(i + 1)
    # ideal DCG: 所有相关结果都排在最前面
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def build_eval_queries(metadata: List[Dict], n_queries: int = 100) -> List[Dict]:
    """
    从已有元数据构造评估 query 集合。
    策略：对每张图像，以其 caption 作为文本 query，
    正确答案是该图像本身（filename）。
    仅选取有 caption 的图像。
    """
    queries = []
    for item in metadata:
        if item.get("caption"):
            queries.append({
                "query_text": item["caption"],
                "relevant_filename": item["filename"],
            })
        if len(queries) >= n_queries:
            break
    return queries


def main():
    config_path = "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ret_cfg = cfg["retrieval"]
    clip_cfg = cfg["clip"]

    # 加载索引和元数据
    retriever = FAISSRetriever(dim=512)
    retriever.load(ret_cfg["index_path"], ret_cfg["metadata_path"])

    encoder = CLIPEncoder(model_name=clip_cfg["model_name"], device=clip_cfg["device"])

    metadata = retriever.metadata
    queries = build_eval_queries(metadata, n_queries=100)

    if not queries:
        print("[Eval] 元数据中没有 caption，无法自动评估。")
        print("       请提供带标注的数据集（如 COCO）或手动构建 eval_queries.json。")
        return

    print(f"[Eval] 共 {len(queries)} 条评估 query")
    print("=" * 50)

    ks = [1, 3, 5]
    metrics = {f"Recall@{k}": [] for k in ks}
    metrics["MRR"] = []
    metrics.update({f"NDCG@{k}": [] for k in ks})

    latencies = []

    for q in queries:
        t0 = time.time()
        query_vec = encoder.encode_text(q["query_text"])
        results = retriever.search(query_vec, top_k=max(ks))
        latencies.append(time.time() - t0)

        retrieved_filenames = [r["filename"] for r in results]
        relevant = [q["relevant_filename"]]

        for k in ks:
            metrics[f"Recall@{k}"].append(recall_at_k(retrieved_filenames, relevant, k))
            metrics[f"NDCG@{k}"].append(ndcg_at_k(retrieved_filenames, relevant, k))
        metrics["MRR"].append(reciprocal_rank(retrieved_filenames, relevant))

    print("\n===== 检索评估结果 =====")
    for name, vals in metrics.items():
        print(f"  {name:<12}: {np.mean(vals):.4f}")
    print(f"\n  平均检索延迟: {np.mean(latencies)*1000:.1f} ms")
    print(f"  评估样本数  : {len(queries)}")

    # 保存结果
    result_path = "data/eval_results.json"
    os.makedirs("data", exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: float(np.mean(v)) for k, v in metrics.items()},
            f, indent=2, ensure_ascii=False
        )
    print(f"\n[Eval] 结果已保存至 {result_path}")


if __name__ == "__main__":
    main()
