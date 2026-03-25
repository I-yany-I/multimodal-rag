"""
build_index.py
--------------
离线建库脚本：
1. 自动下载 COCO 2017 val 子集（500 张演示用）
2. 用 CLIP 批量编码所有图像
3. 构建 FAISS 索引并持久化到 data/ 目录

运行方式: python build_index.py
"""

import json
import os
import shutil
import sys
import urllib.request
import zipfile

import numpy as np
import yaml
from tqdm import tqdm

from src.encoder import CLIPEncoder
from src.retriever import FAISSRetriever
from src.utils import collect_image_paths, load_coco_captions


# ---------- COCO 下载工具 ----------

COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# 使用一批固定的 COCO val2017 图像 URL（避免下载整个 5GB zip）
# 这里提供一个轻量替代方案：下载 COCO val2017 前 500 张图像的文件名列表
SAMPLE_IMAGES_LIST_URL = (
    "https://raw.githubusercontent.com/nightrome/cocostuff/master/dataset/imageLists/val2017.txt"
)


def _load_filenames_from_coco_annotations(max_images: int) -> list:
    ann_dir = os.path.join("data", "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    ann_file = os.path.join(ann_dir, "captions_val2017.json")
    if not os.path.exists(ann_file):
        zip_path = os.path.join(ann_dir, "annotations_trainval2017.zip")
        print("[Build] 正在下载 COCO annotations（用于获取稳定文件名列表）...")
        urllib.request.urlretrieve(COCO_ANNOTATIONS_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extract("annotations/captions_val2017.json", path=ann_dir)
        extracted = os.path.join(ann_dir, "annotations", "captions_val2017.json")
        os.replace(extracted, ann_file)
        shutil.rmtree(os.path.join(ann_dir, "annotations"), ignore_errors=True)
        if os.path.exists(zip_path):
            os.remove(zip_path)
    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    filenames = [img["file_name"] for img in data["images"][:max_images]]
    return filenames


def download_sample_images(images_dir: str, max_images: int = 500) -> list:
    """
    从 COCO 下载 max_images 张样例图像到 images_dir。
    若网络不可用，跳过并使用本地已有图像。
    """
    os.makedirs(images_dir, exist_ok=True)
    existing = collect_image_paths(images_dir)
    if len(existing) >= max_images:
        print(f"[Build] 本地已有 {len(existing)} 张图像，跳过下载")
        return existing[:max_images]

    print(f"[Build] 正在从 COCO 下载 {max_images} 张样例图像...")
    base_url = "http://images.cocodataset.org/val2017/"

    try:
        with urllib.request.urlopen(SAMPLE_IMAGES_LIST_URL, timeout=10) as resp:
            lines = resp.read().decode().strip().split("\n")
        filenames = []
        for line in lines:
            value = line.strip()
            if not value:
                continue
            fname = os.path.basename(value)
            if not fname.lower().endswith(".jpg"):
                fname = f"{fname}.jpg"
            filenames.append(fname)
            if len(filenames) >= max_images:
                break
    except Exception:
        print("[Build] 远端文件名列表不可用，切换为 COCO annotations 文件名列表...")
        filenames = _load_filenames_from_coco_annotations(max_images)

    downloaded = []
    failed = 0
    for fname in tqdm(filenames, desc="下载图像", unit="img"):
        save_path = os.path.join(images_dir, fname)
        if os.path.exists(save_path):
            downloaded.append(save_path)
            continue
        url = base_url + fname
        try:
            urllib.request.urlretrieve(url, save_path)
            downloaded.append(save_path)
        except Exception:
            failed += 1
            continue

    print(f"[Build] 下载完成，共 {len(downloaded)} 张图像，失败 {failed} 张")
    return downloaded


def build_metadata(image_paths: list, captions: dict = None) -> list:
    """为每张图像生成元数据字典。"""
    metadata = []
    for path in image_paths:
        fname = os.path.basename(path)
        caption = ""
        if captions:
            caption = captions.get(fname, "")
        metadata.append({
            "image_path": path,
            "filename": fname,
            "caption": caption,
        })
    return metadata


def main():
    config_path = "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    images_dir = cfg["data"]["images_dir"]
    max_images = cfg["data"]["max_images"]
    index_path = cfg["retrieval"]["index_path"]
    metadata_path = cfg["retrieval"]["metadata_path"]

    # 1. 准备图像
    print("=" * 50)
    print("阶段 1/3：准备图像数据")
    print("=" * 50)
    image_paths = download_sample_images(images_dir, max_images)

    if not image_paths:
        print("[Error] 没有找到任何图像，请手动将图像放入 data/images/ 目录后重试。")
        sys.exit(1)

    # 尝试加载 COCO 标注（可选）
    captions = {}
    ann_file = "data/annotations/captions_val2017.json"
    if os.path.exists(ann_file):
        captions = load_coco_captions(ann_file)
        print(f"[Build] 已加载 COCO 标注，共 {len(captions)} 条描述")

    metadata = build_metadata(image_paths, captions)

    # 2. CLIP 编码
    print("\n" + "=" * 50)
    print("阶段 2/3：CLIP 图像编码")
    print("=" * 50)
    encoder = CLIPEncoder(
        model_name=cfg["clip"]["model_name"],
        device=cfg["clip"]["device"],
    )
    embeddings = encoder.encode_images(image_paths, batch_size=cfg["clip"]["batch_size"])
    print(f"[Build] 编码完成，向量矩阵形状: {embeddings.shape}")

    # 3. 构建 FAISS 索引
    print("\n" + "=" * 50)
    print("阶段 3/3：构建 FAISS 索引")
    print("=" * 50)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    retriever = FAISSRetriever(dim=embeddings.shape[1])
    retriever.build(embeddings, metadata)
    retriever.save(index_path, metadata_path)

    print("\n[Build] 全部完成！")
    print(f"  图像数量 : {len(image_paths)}")
    print(f"  向量维度 : {embeddings.shape[1]}")
    print(f"  索引路径 : {index_path}")
    print(f"  元数据   : {metadata_path}")
    print("\n接下来运行: python app.py  启动 Gradio 演示界面")


if __name__ == "__main__":
    main()
