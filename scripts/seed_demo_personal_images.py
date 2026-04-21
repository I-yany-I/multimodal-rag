"""
从网络拉取「演示用占位图」到 data/personal_images/，用于没有真机照片时跑通个人图库索引。

图片来源：picsum.photos 固定 ID（Lorem Picsum），仅作视觉占位，不代表任何真实人物或隐私场景。
请勿将本脚本误解为「爬取个人生活照」；真实个人图库请使用自己拍摄/有权使用的照片。

用法：
  python scripts/seed_demo_personal_images.py
  python scripts/seed_demo_personal_images.py --count 12

之后执行：python build_personal_index.py
"""

from __future__ import annotations

import argparse
import os
import urllib.request

# 固定 ID，便于复现；可自由增删（对应 picsum 图库中的不同图）
DEFAULT_IDS = (237, 433, 582, 1027, 1084, 111, 292, 866, 825, 64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/personal_images")
    parser.add_argument("--count", type=int, default=10, help="下载张数，循环使用 ID 表")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    ids = list(DEFAULT_IDS)
    n = max(1, int(args.count))
    print(f"[seed] 目标目录: {out_dir} ，共 {n} 张（占位图）")

    for i in range(n):
        pid = ids[i % len(ids)]
        url = f"https://picsum.photos/id/{pid}/640/480.jpg"
        fname = f"demo_picsum_{i:03d}_id{pid}.jpg"
        dest = os.path.join(out_dir, fname)
        if os.path.exists(dest):
            print(f"  跳过已存在: {fname}")
            continue
        print(f"  下载 {url} -> {fname}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(f"  [warn] 失败 {fname}: {e}")

    print("[seed] 完成。请按需编辑 data/personal_notes.json，然后运行: python build_personal_index.py")


if __name__ == "__main__":
    main()
