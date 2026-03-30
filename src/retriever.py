"""
retriever.py
------------
基于 FAISS 的向量检索模块。
FAISS (Facebook AI Similarity Search) 支持高效的近似最近邻搜索，
在亿级向量规模下仍能毫秒级返回结果。
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import faiss
import numpy as np


class FAISSRetriever:
    """
    向量数据库封装：
    - build():    从向量矩阵构建 FAISS 索引（IndexFlatIP = 内积/cosine 搜索）
    - save():     持久化索引 + 元数据到磁盘
    - load():     从磁盘加载已有索引
    - search():   输入查询向量，返回 top-k 图像元数据
    """

    def __init__(self, dim: Optional[int] = None):
        self.dim = dim
        self.index = None
        self.metadata: List[Dict] = []  # 与向量一一对应的图像元数据

    def build(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        """
        构建 FAISS 内积索引（向量已 L2 归一化，内积等价于 cosine 相似度）。
        IndexFlatIP: 暴力精确搜索，适合 <100万 向量规模。
        """
        assert embeddings.shape[0] == len(metadata), "向量数量与元数据数量不匹配"
        self.metadata = metadata
        self.dim = int(embeddings.shape[1])  # 以数据为单一事实来源
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings.astype(np.float32))
        print(f"[Retriever] 索引构建完成，共 {self.index.ntotal} 条向量，维度 {self.dim}")

    def save(self, index_path: str, metadata_path: str) -> None:
        """持久化索引和元数据。"""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        print(f"[Retriever] 已保存索引 -> {index_path}")
        print(f"[Retriever] 已保存元数据 -> {metadata_path}")

    def load(self, index_path: str, metadata_path: str) -> None:
        """从磁盘加载已构建的索引。"""
        self.index = faiss.read_index(index_path)
        self.dim = int(self.index.d)  # 从索引回写真实维度
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"[Retriever] 已加载索引，共 {self.index.ntotal} 条向量，维度 {self.dim}")

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        近似最近邻搜索。
        返回: List[dict]，每个 dict 包含 image_path, caption, score 等字段。
        """
        if self.index is None:
            raise RuntimeError("请先调用 build() 或 load() 初始化索引")
        query_vec = query_vec.astype(np.float32)
        if query_vec.ndim == 1:
            query_vec = query_vec[np.newaxis, :]
            
        if query_vec.shape[1] != self.index.d:
            raise ValueError(
                f"查询向量维度 {query_vec.shape[1]} 与索引维度 {self.index.d} 不一致。"
                "请检查编码器与索引是否匹配，必要时重建索引。"
            )
            
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results
