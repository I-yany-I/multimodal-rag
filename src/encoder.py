"""
encoder.py
----------
使用 CLIP 对图像和文本分别编码为向量。
CLIP 将图像和文本映射到同一向量空间，使得语义相似的图文向量距离更近。
"""

import os
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


class CLIPEncoder:
    """
    CLIP 双塔编码器：
    - encode_images(): 批量将图像编码为 L2 归一化向量
    - encode_text():   将文本 query 编码为 L2 归一化向量
    同一语义的图文向量点积（cosine 相似度）更高，是 RAG 检索的基础。
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"[Encoder] 加载 CLIP 模型: {model_name}  设备: {self.device}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode_images(self, image_paths: List[str], batch_size: int = 64) -> np.ndarray:
        """
        批量编码图像，返回 shape=(N, 512) 的 float32 归一化向量。
        batch_size 过大会 OOM，根据显存调整。
        """
        all_embeddings = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="编码图像"):
            batch_paths = image_paths[i: i + batch_size]
            images = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                except Exception as e:
                    print(f"[Encoder] 跳过损坏图像 {p}: {e}")
                    images.append(Image.new("RGB", (224, 224)))  # 占位
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            feats = self.model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)   # L2 归一化
            all_embeddings.append(feats.cpu().float().numpy())
        return np.vstack(all_embeddings)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """
        编码单条文本 query，返回 shape=(1, 512) 的 float32 归一化向量。
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        feats = self.model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()

    @torch.no_grad()
    def encode_image_single(self, image: Image.Image) -> np.ndarray:
        """
        编码单张 PIL 图像，用于以图搜图场景。
        """
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()
