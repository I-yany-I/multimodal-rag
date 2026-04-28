"""
build_personal_index.py
-----------------------
为「个人图库 / 本地生活相册」构建独立 FAISS 索引（与 COCO 演示库分离）。

使用方式：
1. 将照片放入 data/personal_images/（支持子目录；格式 jpg/jpeg/png/webp）
2. （可选）复制 data/personal_notes.example.json 为 data/personal_notes.json，
   按文件名 basename 填写你的文字备注，增强「用自然语言回忆照片」的检索效果
3. 运行：
   python build_personal_index.py
   或指定目录与数量：
   python build_personal_index.py --images_dir data/personal_images --max_images 500

隐私说明：默认仅本地索引；请勿将含隐私的照片目录提交到公开仓库。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import yaml
from tqdm import tqdm

from src.encoder import make_encoder
from src.retriever import FAISSRetriever
from src.utils import collect_image_paths


def _load_notes(notes_path: str) -> Dict[str, str]:
    if not notes_path or not os.path.exists(notes_path):
        return {}
    with open(notes_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in data.items():
        if k is None or v is None:
            continue
        key = str(k).strip()
        val = str(v).strip()
        if key and val:
            out[key] = val
    return out


def _extensions() -> tuple:
    return (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")


def build_personal_metadata(image_paths: List[str], notes: Dict[str, str]) -> List[dict]:
    """为个人图库构造 metadata：仅在有 personal_notes 时写入 caption，供重排与 VLM 证据。"""
    meta: List[dict] = []
    for path in image_paths:
        fname = os.path.basename(path)
        note = notes.get(fname, "").strip()
        caption = f"备注：{note}" if note else ""
        meta.append(
            {
                "image_path": os.path.abspath(path),
                "filename": fname,
                "caption": caption,
                "source": "personal",
            }
        )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="为个人图库构建 CLIP + FAISS 索引")
    parser.add_argument("--images_dir", type=str, default=None, help="个人照片根目录")
    parser.add_argument("--max_images", type=int, default=None, help="最多入库张数")
    parser.add_argument("--notes", type=str, default=None, help="备注 JSON 路径，默认读 config")
    args = parser.parse_args()

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pl = cfg.get("personal_library") or {}
    images_dir = args.images_dir or pl.get("images_dir", "data/personal_images")
    max_images = args.max_images if args.max_images is not None else int(pl.get("max_images", 2000))
    index_path = pl.get("index_path", "data/personal_faiss.index")
    metadata_path = pl.get("metadata_path", "data/personal_metadata.json")
    notes_path = args.notes or pl.get("notes_path", "data/personal_notes.json")

    os.makedirs(images_dir, exist_ok=True)

    print("=" * 50)
    print("个人图库索引：收集图像路径")
    print("=" * 50)
    paths = collect_image_paths(images_dir, extensions=_extensions())
    if not paths:
        print(f"[Error] 目录中没有找到图像：{images_dir}")
        print("请将照片放入该目录后重试（支持 jpg/png/webp 等）。")
        sys.exit(1)
    if len(paths) > max_images:
        paths = paths[:max_images]
        print(f"[Build] 图像数量超过 max_images={max_images}，已截断。")

    notes = _load_notes(notes_path)
    if notes:
        print(f"[Build] 已加载备注文件 {notes_path}，共 {len(notes)} 条键。")
    else:
        print(f"[Build] 未找到备注文件或为空：{notes_path}（可选）")

    metadata = build_personal_metadata(paths, notes)

    print("\n" + "=" * 50)
    print("CLIP 编码个人图库")
    print("=" * 50)
    encoder = make_encoder(cfg["clip"])
    embeddings = encoder.encode_images(paths, batch_size=cfg["clip"]["batch_size"])
    print(f"[Build] 编码完成，向量矩阵形状: {embeddings.shape}")

    manifest_path = os.path.join(os.path.dirname(index_path) or ".", "personal_index_manifest.json")
    manifest = {
        "kind": "personal_library",
        "model_name": cfg["clip"]["model_name"],
        "embedding_dim": int(embeddings.shape[1]),
        "num_images": len(paths),
        "images_dir": os.path.abspath(images_dir),
        "notes_path": notes_path if os.path.exists(notes_path) else None,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[Build] 已写入索引清单 -> {manifest_path}")

    print("\n" + "=" * 50)
    print("构建 FAISS 索引")
    print("=" * 50)
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    retriever = FAISSRetriever()
    retriever.build(embeddings, metadata)
    retriever.save(index_path, metadata_path)

    print("\n[Build] 个人图库索引完成")
    print(f"  图像数量 : {len(paths)}")
    print(f"  向量维度 : {embeddings.shape[1]}")
    print(f"  索引路径 : {index_path}")
    print(f"  元数据   : {metadata_path}")
    print("\n接下来：在 Gradio 中选择「个人图库」，或运行 app.py 后切换库。")


if __name__ == "__main__":
    main()
