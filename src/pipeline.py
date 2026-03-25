"""
pipeline.py
-----------
端到端多模态 RAG Pipeline：
Encode Query → Retrieve Images → Generate Answer
将 encoder / retriever / generator 串联为统一接口。
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from src.encoder import CLIPEncoder
from src.retriever import FAISSRetriever
from src.generator import Qwen2VLGenerator


class MultimodalRAGPipeline:
    """
    多模态 RAG 完整流程：
    1. encode:   用 CLIP 将用户 query（文本/图像）编码为向量
    2. retrieve: 用 FAISS 检索最相似的 top-k 图像
    3. generate: 用 Qwen2-VL 基于检索图像生成回答
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        clip_cfg = self.cfg["clip"]
        gen_cfg = self.cfg["generator"]
        ret_cfg = self.cfg["retrieval"]

        self.encoder = CLIPEncoder(
            model_name=clip_cfg["model_name"],
            device=clip_cfg["device"],
        )
        self.retriever = FAISSRetriever(dim=512)
        self.retriever.load(
            index_path=ret_cfg["index_path"],
            metadata_path=ret_cfg["metadata_path"],
        )
        self.generator = Qwen2VLGenerator(
            model_name=gen_cfg["model_name"],
            device=gen_cfg["device"],
            load_in_4bit=gen_cfg["load_in_4bit"],
            max_new_tokens=gen_cfg["max_new_tokens"],
            temperature=gen_cfg["temperature"],
        )
        self.top_k = ret_cfg["top_k"]

    def query_by_text(self, text: str) -> Tuple[str, List[Dict]]:
        """
        文本问答主入口。
        返回: (answer, retrieved_items)
        """
        query_vec = self.encoder.encode_text(text)
        retrieved = self.retriever.search(query_vec, top_k=self.top_k)
        answer = self.generator.generate(text, retrieved)
        return answer, retrieved

    def query_by_image(self, image: Image.Image, question: str = "请描述这张图像的内容。") -> Tuple[str, List[Dict]]:
        """
        以图搜图 + 问答：先用图像检索相似图，再用 LLM 回答。
        """
        query_vec = self.encoder.encode_image_single(image)
        retrieved = self.retriever.search(query_vec, top_k=self.top_k)
        answer = self.generator.generate(question, retrieved)
        return answer, retrieved
