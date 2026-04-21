"""
utils.py
--------
通用工具函数。
"""

import json
import os
from pathlib import Path
from typing import List, Dict

import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_image_paths(
    images_dir: str,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"),
) -> List[str]:
    """递归收集目录下所有图像路径。"""
    paths = []
    for root, _, files in os.walk(images_dir):
        for fname in sorted(files):
            if fname.lower().endswith(extensions):
                paths.append(os.path.join(root, fname))
    return paths


def load_coco_captions(annotation_file: str) -> Dict[str, str]:
    """
    解析 COCO annotations/captions_val2017.json，
    返回 {filename: first_caption} 映射。
    """
    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    id2filename = {img["id"]: img["file_name"] for img in data["images"]}
    filename2caption = {}
    for ann in data["annotations"]:
        fname = id2filename[ann["image_id"]]
        if fname not in filename2caption:
            filename2caption[fname] = ann["caption"]
    return filename2caption
