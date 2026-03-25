"""
app.py
------
Gradio 演示界面：多模态 RAG 图像问答系统
支持：
  1. 文本 → 检索图像 + LLM 回答
  2. 图像 → 以图搜图 + LLM 回答
"""

import os
import sys
from pathlib import Path
from typing import Optional

import gradio as gr
from PIL import Image

# 延迟加载 pipeline，避免启动时就占满显存
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.pipeline import MultimodalRAGPipeline
        _pipeline = MultimodalRAGPipeline(config_path="config.yaml")
    return _pipeline


def check_index_ready() -> bool:
    import yaml
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    return (
        os.path.exists(cfg["retrieval"]["index_path"]) and
        os.path.exists(cfg["retrieval"]["metadata_path"])
    )


# ---------- 核心推理函数 ----------

def text_query_fn(query: str, top_k: int):
    """文本问答入口。"""
    if not query.strip():
        return "请输入问题。", [], ""
    if not check_index_ready():
        return "请先运行 `python build_index.py` 构建图像索引！", [], ""

    pipeline = get_pipeline()
    answer, retrieved = pipeline.query_by_text(
        query,
        top_k=top_k,
        max_images=top_k,
    )

    images_out = []
    captions_out = []
    for item in retrieved:
        img_path = item["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
            images_out.append(img)
            captions_out.append(
                f"[{item['filename']}]  相似度: {item['score']:.3f}\n{item.get('caption', '')}"
            )
        except Exception:
            pass

    retrieved_text = "\n\n".join(captions_out)
    return answer, images_out, retrieved_text


def image_query_fn(image: Optional[Image.Image], question: str, top_k: int):
    """以图搜图 + 问答入口。"""
    if image is None:
        return "请上传一张图像。", [], ""
    if not check_index_ready():
        return "请先运行 `python build_index.py` 构建图像索引！", [], ""

    if not question.strip():
        question = "请描述图像中的主要内容。"

    pipeline = get_pipeline()
    answer, retrieved = pipeline.query_by_image(
        image,
        question,
        top_k=top_k,
        max_images=top_k,
    )

    images_out = []
    captions_out = []
    for item in retrieved:
        img_path = item["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
            images_out.append(img)
            captions_out.append(
                f"[{item['filename']}]  相似度: {item['score']:.3f}\n{item.get('caption', '')}"
            )
        except Exception:
            pass

    retrieved_text = "\n\n".join(captions_out)
    return answer, images_out, retrieved_text


# ---------- Gradio UI ----------

with gr.Blocks(
    title="多模态 RAG 图像问答系统",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # 🖼️ 多模态 RAG 图像问答系统
        **Multimodal RAG for Visual Question Answering**

        基于 **CLIP**（图文向量检索）+ **Qwen2-VL**（多模态大语言模型）构建，
        支持文本检索图像库并生成回答，也支持以图搜图。

        > 技术栈：CLIP · FAISS · Qwen2-VL-2B · Gradio · FastAPI
        """
    )

    with gr.Tab("📝 文本问答"):
        gr.Markdown("输入自然语言问题，系统检索图像库并由 LLM 生成回答。")
        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="输入问题",
                    placeholder="例如：一只猫坐在沙发上",
                    lines=3,
                )
                top_k_slider = gr.Slider(1, 5, value=3, step=1, label="检索图像数量 (top-k)")
                text_btn = gr.Button("🔍 检索 + 生成回答", variant="primary")
            with gr.Column(scale=2):
                text_answer = gr.Textbox(label="LLM 生成回答", lines=5, interactive=False)
                text_gallery = gr.Gallery(label="检索到的图像", columns=3, height=300)
                text_retrieved = gr.Textbox(label="检索详情", lines=4, interactive=False)

        text_btn.click(
            fn=text_query_fn,
            inputs=[text_input, top_k_slider],
            outputs=[text_answer, text_gallery, text_retrieved],
        )

        gr.Examples(
            examples=[
                ["一只猫坐在沙发上", 3],
                ["海边的日落风景", 3],
                ["运动员在比赛中", 3],
                ["一个小孩在公园玩耍", 3],
            ],
            inputs=[text_input, top_k_slider],
        )

    with gr.Tab("🖼️ 以图搜图"):
        gr.Markdown("上传图像，系统检索相似图像并回答你的问题。")
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="上传查询图像", type="pil")
                image_question = gr.Textbox(
                    label="关于图像的问题（可选）",
                    placeholder="请描述这张图像中的内容。",
                    value="请描述图像中的主要内容。",
                )
                top_k_slider2 = gr.Slider(1, 5, value=3, step=1, label="检索图像数量 (top-k)")
                image_btn = gr.Button("🔍 以图搜图 + 回答", variant="primary")
            with gr.Column(scale=2):
                image_answer = gr.Textbox(label="LLM 生成回答", lines=5, interactive=False)
                image_gallery = gr.Gallery(label="检索到的相似图像", columns=3, height=300)
                image_retrieved = gr.Textbox(label="检索详情", lines=4, interactive=False)

        image_btn.click(
            fn=image_query_fn,
            inputs=[image_input, image_question, top_k_slider2],
            outputs=[image_answer, image_gallery, image_retrieved],
        )

    gr.Markdown(
        """
        ---
        **使用前请先运行**: `python build_index.py` 构建图像向量索引

        **系统架构**：用户 Query → CLIP 编码 → FAISS 检索 top-k 图像 → Qwen2-VL 多模态生成
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
