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

_UI_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_lg,
)

_UI_CSS = """
/* 页头：轻渐变 + 细边框 */
.mm-hero {
    padding: 1.35rem 1.5rem 1.5rem;
    margin-bottom: 0.5rem;
    border-radius: 14px;
    background: linear-gradient(135deg,
        color-mix(in srgb, var(--primary-500) 14%, transparent) 0%,
        color-mix(in srgb, var(--neutral-200) 55%, transparent) 48%,
        color-mix(in srgb, var(--secondary-500) 10%, transparent) 100%);
    border: 1px solid color-mix(in srgb, var(--neutral-400) 35%, transparent);
    box-shadow: 0 8px 28px color-mix(in srgb, var(--neutral-950) 6%, transparent);
}
.mm-hero h1 {
    margin: 0 0 0.35rem 0;
    font-weight: 700;
    letter-spacing: -0.02em;
    font-size: 1.65rem !important;
    line-height: 1.2;
}
.mm-hero .mm-sub {
    margin: 0 0 0.75rem 0;
    opacity: 0.92;
    font-size: 0.98rem;
}
.mm-badges { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.35rem; }
.mm-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    background: color-mix(in srgb, var(--primary-500) 18%, transparent);
    color: var(--neutral-950);
    border: 1px solid color-mix(in srgb, var(--primary-600) 22%, transparent);
}

/* 侧栏输入区更聚拢 */
.mm-panel { padding: 0.15rem 0; }

/* 结果区：回答框略突出 */
.mm-answer textarea, .mm-answer .wrap {
    font-size: 0.98rem !important;
    line-height: 1.55 !important;
}

/* 图库：圆角与留白 */
.mm-gallery .grid-wrap, .mm-gallery img {
    border-radius: 10px !important;
}

/* 页脚说明 */
.mm-foot {
    margin-top: 1.25rem;
    padding: 1rem 1.15rem;
    border-radius: 12px;
    background: color-mix(in srgb, var(--neutral-300) 45%, transparent);
    border: 1px solid color-mix(in srgb, var(--neutral-400) 30%, transparent);
    font-size: 0.88rem;
    line-height: 1.55;
}
.mm-foot code {
    font-size: 0.82rem !important;
}
"""

with gr.Blocks(
    title="多模态 RAG · 图像问答",
    theme=_UI_THEME,
    css=_UI_CSS,
) as demo:
    gr.Markdown(
        """
<div class="mm-hero">

# 多模态 RAG 图像问答

<p class="mm-sub">CLIP 向量检索 · Qwen2-VL 多模态理解 — 文本搜图或以图搜图，回答均基于离线图库检索结果。</p>

<div class="mm-badges">
<span class="mm-badge">CLIP</span>
<span class="mm-badge">FAISS</span>
<span class="mm-badge">Qwen2-VL</span>
<span class="mm-badge">Gradio</span>
</div>

</div>
        """
    )

    with gr.Tab("文本问答"):
        gr.Markdown("输入自然语言问题，系统将检索最相关的图块并由多模态模型汇总作答。")
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=280):
                with gr.Group():
                    text_input = gr.Textbox(
                        label="问题",
                        placeholder="例如：滑雪的人在怎样的环境里？",
                        lines=4,
                        show_label=True,
                    )
                    top_k_slider = gr.Slider(
                        1, 5, value=3, step=1,
                        label="检索图像数量 (top-k)",
                    )
                text_btn = gr.Button("检索并生成回答", variant="primary", size="lg")

            with gr.Column(scale=7, min_width=320):
                with gr.Group():
                    text_answer = gr.Textbox(
                        label="模型回答",
                        lines=7,
                        interactive=False,
                        elem_classes=["mm-answer"],
                    )
                    text_gallery = gr.Gallery(
                        label="检索到的图像",
                        columns=3,
                        height=340,
                        object_fit="contain",
                        show_label=True,
                        elem_classes=["mm-gallery"],
                    )
                    text_retrieved = gr.Textbox(
                        label="检索详情（文件名 · 分数 · 标注）",
                        lines=6,
                        interactive=False,
                    )

        text_btn.click(
            fn=text_query_fn,
            inputs=[text_input, top_k_slider],
            outputs=[text_answer, text_gallery, text_retrieved],
        )

        gr.Examples(
            label="示例一键填入",
            examples=[
                ["一只猫坐在沙发上", 3],
                ["海边的日落风景", 3],
                ["运动员在比赛中", 3],
                ["一个小孩在公园玩耍", 3],
            ],
            inputs=[text_input, top_k_slider],
        )

    with gr.Tab("以图搜图"):
        gr.Markdown("上传查询图，在图库中寻找语义相近的图像，并可结合你的问题生成说明。")
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=280):
                with gr.Group():
                    image_input = gr.Image(
                        label="查询图像",
                        type="pil",
                        height=320,
                    )
                    image_question = gr.Textbox(
                        label="附加问题（可选）",
                        placeholder="例如：画面里的人在做什么？",
                        value="请描述图像中的主要内容。",
                        lines=3,
                    )
                    top_k_slider2 = gr.Slider(
                        1, 5, value=3, step=1,
                        label="检索图像数量 (top-k)",
                    )
                image_btn = gr.Button("以图搜图并生成回答", variant="primary", size="lg")

            with gr.Column(scale=7, min_width=320):
                with gr.Group():
                    image_answer = gr.Textbox(
                        label="模型回答",
                        lines=7,
                        interactive=False,
                        elem_classes=["mm-answer"],
                    )
                    image_gallery = gr.Gallery(
                        label="检索到的相似图像",
                        columns=3,
                        height=340,
                        object_fit="contain",
                        elem_classes=["mm-gallery"],
                    )
                    image_retrieved = gr.Textbox(
                        label="检索详情（文件名 · 分数 · 标注）",
                        lines=6,
                        interactive=False,
                    )

        image_btn.click(
            fn=image_query_fn,
            inputs=[image_input, image_question, top_k_slider2],
            outputs=[image_answer, image_gallery, image_retrieved],
        )

    gr.Markdown(
        """
<div class="mm-foot">

**使用前**请在本项目目录执行 `python build_index.py` 构建向量索引（首次或更换 CLIP 模型后需重建）。

**数据流**：Query → CLIP 编码 → FAISS 召回 top-k → Qwen2-VL 读图与标注并生成文本（不生成新图像）。

</div>
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
