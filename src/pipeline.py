"""
pipeline.py
-----------
端到端多模态 RAG Pipeline：
Encode Query → Retrieve Images → Generate Answer
将 encoder / retriever / generator 串联为统一接口。
"""

import os
from typing import Any, List, Dict, Optional, Tuple
import yaml
import numpy as np
from PIL import Image

from src.encoder import make_encoder
from src.retriever import FAISSRetriever
from src.generator import Qwen2VLGenerator


def merge_candidates(result_groups: List[List[Dict]], limit: int) -> List[Dict]:
    """按 image_path 去重合并多路召回，并保留每张图的最高检索分。"""
    merged = {}
    order = []
    for group in result_groups:
        for item in group:
            key = item.get("image_path") or item.get("filename")
            if not key:
                continue
            score = float(item.get("score", 0.0))
            if key not in merged:
                merged[key] = dict(item)
                order.append(key)
                continue
            if score > float(merged[key].get("score", 0.0)):
                updated = dict(merged[key])
                updated.update(item)
                updated["score"] = score
                merged[key] = updated
    ranked = [merged[key] for key in order]
    ranked.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return ranked[:limit]


def rerank_and_filter(
    encoder: Any,
    query_vec: np.ndarray,
    items: List[Dict],
    final_k: int,
    *,
    rerank: bool,
    rerank_visual_weight: float,
    min_retrieval_score: Optional[float],
    caption_query_vec: Optional[np.ndarray] = None,
) -> List[Dict]:
    """用 CLIP 文本-查询相关度与视觉分加权，并可选按阈值过滤（与线上一致）。"""
    if not items or not rerank:
        out = list(items)[:final_k]
        if min_retrieval_score is not None:
            out = [it for it in out if float(it.get("score", 0.0)) >= min_retrieval_score]
        if not out and items:
            return [dict(it) for it in items[:final_k]]
        return out[:final_k]

    alpha = rerank_visual_weight
    text_query = caption_query_vec if caption_query_vec is not None else query_vec
    qflat = text_query.reshape(-1).astype(np.float32)
    qflat = qflat / (np.linalg.norm(qflat) + 1e-8)

    captions = [(it.get("caption") or "").strip() for it in items]
    non_empty_idx = [i for i, c in enumerate(captions) if c]
    text_sims = [0.0] * len(items)
    if non_empty_idx:
        to_encode = [captions[i] for i in non_empty_idx]
        cap_embs = encoder.encode_texts(to_encode)
        cap_embs = cap_embs.astype(np.float32)
        for row, idx in enumerate(non_empty_idx):
            c = cap_embs[row].reshape(-1)
            c = c / (np.linalg.norm(c) + 1e-8)
            text_sims[idx] = float(np.dot(qflat, c))

    scored = []
    for it, t_sim in zip(items, text_sims):
        v = float(it.get("score", 0.0))
        c = (it.get("caption") or "").strip()
        if c:
            combined = alpha * v + (1.0 - alpha) * t_sim
        else:
            combined = v
        row = dict(it)
        row["rerank_score"] = combined
        row["text_match"] = t_sim if c else None
        scored.append(row)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    if min_retrieval_score is not None:
        scored = [it for it in scored if it["rerank_score"] >= min_retrieval_score]
    if not scored and items:
        return [dict(it) for it in items[:final_k]]
    for it in scored:
        it["score"] = float(it["rerank_score"])
    return scored[:final_k]


def retrieve_text_query(
    encoder: Any,
    retriever: FAISSRetriever,
    text: str,
    *,
    metrics_top_k: int,
    candidate_k: int,
    rerank: bool,
    rerank_visual_weight: float,
    min_score: Optional[float],
    query_rewrite: bool = False,
    generator: Optional[Qwen2VLGenerator] = None,
    rewrite_chinese_only: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """
    与 MultimodalRAGPipeline.query_by_text 的检索阶段一致（不含 LLM 生成）。
    返回:
        raw_faiss: 合并去重后按 FAISS 视觉分排序，截断至 metrics_top_k（用于 raw 指标）
        with_rerank: 经 rerank / min_score 后的最终列表（与线上一致）
    """
    query_vec = encoder.encode_text(text)
    pool_k = max(metrics_top_k, int(candidate_k))
    result_groups = [retriever.search(query_vec, top_k=pool_k)]
    rewrite_vec = None
    if query_rewrite and generator is not None:
        rewrite_text = generator.rewrite_query_for_retrieval(
            text,
            chinese_only=rewrite_chinese_only,
        )
        if rewrite_text and rewrite_text.strip().lower() != text.strip().lower():
            rewrite_vec = encoder.encode_text(rewrite_text)
            result_groups.append(retriever.search(rewrite_vec, top_k=pool_k))

    merged = merge_candidates(result_groups, limit=pool_k)
    caption_query_vec = rewrite_vec if rewrite_vec is not None else query_vec
    raw_faiss = merged[:metrics_top_k]
    with_rerank = rerank_and_filter(
        encoder,
        query_vec,
        merged,
        metrics_top_k,
        rerank=rerank,
        rerank_visual_weight=rerank_visual_weight,
        min_retrieval_score=min_score,
        caption_query_vec=caption_query_vec,
    )
    return raw_faiss, with_rerank


class MultimodalRAGPipeline:
    """
    多模态 RAG 完整流程：
    1. encode:   用 CLIP 将用户 query（文本/图像）编码为向量
    2. retrieve: 用 FAISS 检索最相似的 top-k 图像
    3. generate: 用 Qwen2-VL 基于检索图像生成回答

    CLIP 与 Qwen2-VL 只构造一次；COCO / 个人图库各用独立 FAISS，按 query 的 library 切换，
    避免切换图库时重复加载大模型导致显存翻倍。
    """

    def __init__(self, config_path: str = "config.yaml", library: Optional[str] = None):
        """
        library 参数已弃用（保留仅为兼容旧代码），检索目标由 query_by_* 的 library 参数指定。
        """
        _ = library  # 兼容 MultimodalRAGPipeline(..., library="coco")
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        clip_cfg = self.cfg["clip"]
        gen_cfg = self.cfg["generator"]
        ret_cfg = self.cfg["retrieval"]
        pl = self.cfg.get("personal_library") or {}

        self.encoder = make_encoder(clip_cfg)
        enc_dim = self.encoder.embedding_dim

        self._retrievers: Dict[str, Optional[FAISSRetriever]] = {
            "coco": None,
            "personal": None,
        }

        coco_ip, coco_mp = ret_cfg["index_path"], ret_cfg["metadata_path"]
        if os.path.isfile(coco_ip) and os.path.isfile(coco_mp):
            r = FAISSRetriever()
            r.load(coco_ip, coco_mp)
            if r.dim is not None and r.dim != enc_dim:
                raise ValueError(
                    f"COCO 索引维度与当前 CLIP 不一致: encoder={enc_dim}, index={r.dim}. "
                    "请用同一模型重建 build_index.py。"
                )
            self._retrievers["coco"] = r

        pers_ip = pl.get("index_path", "data/personal_faiss.index")
        pers_mp = pl.get("metadata_path", "data/personal_metadata.json")
        if os.path.isfile(pers_ip) and os.path.isfile(pers_mp):
            r = FAISSRetriever()
            r.load(pers_ip, pers_mp)
            if r.dim is not None and r.dim != enc_dim:
                raise ValueError(
                    f"个人图库索引维度与当前 CLIP 不一致: encoder={enc_dim}, index={r.dim}. "
                    "请用同一模型重建 build_personal_index.py。"
                )
            self._retrievers["personal"] = r

        self.generator = Qwen2VLGenerator(
            model_name=gen_cfg["model_name"],
            device=gen_cfg["device"],
            load_in_4bit=gen_cfg["load_in_4bit"],
            max_new_tokens=gen_cfg["max_new_tokens"],
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg.get("top_p", 0.9),
            max_image_long_side=int(gen_cfg.get("max_image_long_side", 768)),
        )
        self.top_k = ret_cfg["top_k"]
        self.candidate_k = int(ret_cfg.get("candidate_k", max(self.top_k, 10)))
        self.rerank = bool(ret_cfg.get("rerank", False))
        self.rerank_visual_weight = float(ret_cfg.get("rerank_visual_weight", 0.62))
        self.query_rewrite = bool(ret_cfg.get("query_rewrite", False))
        self.rewrite_chinese_only = bool(ret_cfg.get("rewrite_chinese_only", True))
        self.min_retrieval_score = ret_cfg.get("min_score", None)
        if self.min_retrieval_score is not None:
            self.min_retrieval_score = float(self.min_retrieval_score)

    def _retriever(self, library: str) -> FAISSRetriever:
        key = (library or "coco").strip().lower()
        if key not in self._retrievers:
            key = "coco"
        r = self._retrievers.get(key)
        if r is None:
            if key == "personal":
                raise RuntimeError(
                    "个人图库索引未就绪：请将照片放入 data/personal_images/ 后运行 "
                    "python build_personal_index.py"
                )
            raise RuntimeError("COCO 演示索引未就绪：请先运行 python build_index.py")
        return r

    def _merge_candidates(self, result_groups: List[List[Dict]], limit: int) -> List[Dict]:
        return merge_candidates(result_groups, limit)

    def _rerank_and_filter(
        self,
        query_vec: np.ndarray,
        items: List[Dict],
        final_k: int,
        caption_query_vec: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        return rerank_and_filter(
            self.encoder,
            query_vec,
            items,
            final_k,
            rerank=self.rerank,
            rerank_visual_weight=self.rerank_visual_weight,
            min_retrieval_score=self.min_retrieval_score,
            caption_query_vec=caption_query_vec,
        )

    def query_by_text(
        self,
        text: str,
        top_k: Optional[int] = None,
        max_images: Optional[int] = None,
        library: str = "coco",
    ) -> Tuple[str, List[Dict]]:
        """
        文本问答主入口。
        library: coco | personal
        返回: (answer, retrieved_items)
        """
        retriever = self._retriever(library)
        effective_top_k = top_k if top_k is not None else self.top_k
        effective_max_images = max_images if max_images is not None else effective_top_k
        query_vec = self.encoder.encode_text(text)
        pool_k = max(effective_top_k, self.candidate_k)
        rewrite_text = None
        rewrite_vec = None
        result_groups = [retriever.search(query_vec, top_k=pool_k)]
        if self.query_rewrite:
            rewrite_text = self.generator.rewrite_query_for_retrieval(
                text,
                chinese_only=self.rewrite_chinese_only,
            )
            if rewrite_text and rewrite_text.strip().lower() != text.strip().lower():
                rewrite_vec = self.encoder.encode_text(rewrite_text)
                result_groups.append(retriever.search(rewrite_vec, top_k=pool_k))

        merged = self._merge_candidates(result_groups, limit=pool_k)
        caption_query_vec = rewrite_vec if rewrite_vec is not None else query_vec
        retrieved = self._rerank_and_filter(
            query_vec,
            merged,
            effective_top_k,
            caption_query_vec=caption_query_vec,
        )
        answer = self.generator.generate(text, retrieved, max_images=effective_max_images)
        return answer, retrieved

    def query_by_image(
        self,
        image: Image.Image,
        question: str = "请描述这张图像的内容。",
        top_k: Optional[int] = None,
        max_images: Optional[int] = None,
        library: str = "coco",
    ) -> Tuple[str, List[Dict]]:
        """
        以图搜图 + 问答：先用图像检索相似图，再用 LLM 回答。
        """
        retriever = self._retriever(library)
        effective_top_k = top_k if top_k is not None else self.top_k
        effective_max_images = max_images if max_images is not None else effective_top_k
        query_vec = self.encoder.encode_image_single(image)
        pool_k = max(effective_top_k, self.candidate_k)
        raw = retriever.search(query_vec, top_k=pool_k)
        caption_query_vec = None
        if question.strip() and self.query_rewrite:
            rewrite_question = self.generator.rewrite_query_for_retrieval(
                question,
                chinese_only=self.rewrite_chinese_only,
            )
            if rewrite_question:
                caption_query_vec = self.encoder.encode_text(rewrite_question)
        elif question.strip():
            caption_query_vec = self.encoder.encode_text(question)

        retrieved = self._rerank_and_filter(
            query_vec,
            raw,
            effective_top_k,
            caption_query_vec=caption_query_vec,
        )
        answer = self.generator.generate(question, retrieved, max_images=effective_max_images)
        return answer, retrieved
