"""
encoder.py
----------
使用 CLIP 对图像和文本分别编码为向量。
CLIP 将图像和文本映射到同一向量空间，使得语义相似的图文向量距离更近。
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from transformers import SiglipModel, SiglipProcessor


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

    @property
    def embedding_dim(self) -> int:
        """
        获取当前 CLIP 模型的输出向量维度。
        尝试从 text_projection 获取，若失败则进行一次最小推理兜底。
        """
        proj = getattr(self.model, "text_projection", None)
        if proj is not None and hasattr(proj, "out_features"):
            return int(proj.out_features)
        # 兜底方案：做一次最小推理以获知形状
        return int(self.encode_text("a").shape[1])

    def _get_image_embeds(self, inputs: dict) -> torch.Tensor:
        try:
            feats = self.model.get_image_features(**inputs)
            if isinstance(feats, torch.Tensor):
                return feats
        except Exception:
            pass

        vision_outputs = self.model.vision_model(pixel_values=inputs["pixel_values"], return_dict=True)
        pooled = getattr(vision_outputs, "pooler_output", None)
        if pooled is None:
            raise TypeError(f"CLIP vision pooler_output 缺失，输出类型: {type(vision_outputs)}")
        proj = getattr(self.model, "visual_projection", None)
        if proj is not None:
            in_features = getattr(proj, "in_features", None)
            if in_features is None or pooled.shape[-1] == in_features:
                return proj(pooled)
        return pooled

    def _get_text_embeds(self, inputs: dict) -> torch.Tensor:
        try:
            feats = self.model.get_text_features(**inputs)
            if isinstance(feats, torch.Tensor):
                return feats
        except Exception:
            pass

        text_outputs = self.model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            return_dict=True,
        )
        pooled = getattr(text_outputs, "pooler_output", None)
        if pooled is None:
            raise TypeError(f"CLIP text pooler_output 缺失，输出类型: {type(text_outputs)}")
        proj = getattr(self.model, "text_projection", None)
        if proj is not None:
            in_features = getattr(proj, "in_features", None)
            if in_features is None or pooled.shape[-1] == in_features:
                return proj(pooled)
        return pooled

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
            feats = self._get_image_embeds(inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)   # L2 归一化
            all_embeddings.append(feats.cpu().float().numpy())
        return np.vstack(all_embeddings)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """
        编码单条文本 query，返回 shape=(1, D) 的 float32 归一化向量。
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        feats = self._get_text_embeds(inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """批量编码多条文本，返回 shape=(N, D)。"""
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        feats = self._get_text_embeds(inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()

    @torch.no_grad()
    def encode_image_single(self, image: Image.Image) -> np.ndarray:
        """
        编码单张 PIL 图像，用于以图搜图场景。
        """
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        feats = self._get_image_embeds(inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()


class SiglipEncoder:
    """
    SigLIP 图文编码器（transformers），与 CLIPEncoder 相同接口以便检索 pipeline 复用。
    更换 encoder 后必须重新运行 build_index.py。
    """

    def __init__(self, model_name: str = "google/siglip-base-patch16-224", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"[Encoder] 加载 SigLIP 模型: {model_name}  设备: {self.device}")
        self.model = SiglipModel.from_pretrained(model_name).to(self.device)
        self.processor = SiglipProcessor.from_pretrained(model_name)
        self.model.eval()

    @property
    def embedding_dim(self) -> int:
        proj = getattr(self.model.config, "projection_dim", None)
        if proj is not None:
            return int(proj)
        return int(self.encode_text("a").shape[1])

    @torch.no_grad()
    def encode_images(self, image_paths: List[str], batch_size: int = 64) -> np.ndarray:
        all_embeddings = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="编码图像"):
            batch_paths = image_paths[i : i + batch_size]
            images = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                except Exception as e:
                    print(f"[Encoder] 跳过损坏图像 {p}: {e}")
                    images.append(Image.new("RGB", (224, 224)))
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            feats = self.model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeddings.append(feats.cpu().float().numpy())
        return np.vstack(all_embeddings)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        feats = self.model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        feats = self.model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()

    @torch.no_grad()
    def encode_image_single(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()


def make_encoder(clip_cfg: dict) -> Union[CLIPEncoder, SiglipEncoder]:
    """根据 config 的 encoder_family 构造编码器。"""
    family = str(clip_cfg.get("encoder_family", "clip")).lower()
    if family == "siglip":
        return SiglipEncoder(
            model_name=clip_cfg["model_name"],
            device=clip_cfg["device"],
        )
    return CLIPEncoder(
        model_name=clip_cfg["model_name"],
        device=clip_cfg["device"],
    )
