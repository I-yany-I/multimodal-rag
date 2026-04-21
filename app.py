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
from typing import Dict, Optional

import gradio as gr
import yaml
from PIL import Image

# 延迟加载 pipeline（按图库分实例），避免启动时重复加载
_pipelines: Dict[str, object] = {}


def _normalize_library(library: Optional[str]) -> str:
    key = (library or "coco").strip().lower()
    return key if key in ("coco", "personal") else "coco"


def get_pipeline(library: Optional[str] = None):
    """library: coco（COCO 演示库） | personal（本地个人图库，独立索引）"""
    key = _normalize_library(library)
    global _pipelines
    if key not in _pipelines:
        from src.pipeline import MultimodalRAGPipeline

        _pipelines[key] = MultimodalRAGPipeline(config_path="config.yaml", library=key)
    return _pipelines[key]


def check_index_ready(library: Optional[str] = None) -> bool:
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    key = _normalize_library(library)
    if key == "personal":
        pl = cfg.get("personal_library") or {}
        ip = pl.get("index_path", "data/personal_faiss.index")
        mp = pl.get("metadata_path", "data/personal_metadata.json")
        return os.path.exists(ip) and os.path.exists(mp)
    return os.path.exists(cfg["retrieval"]["index_path"]) and os.path.exists(cfg["retrieval"]["metadata_path"])


def _run_text_query(query: str, top_k: int, library: str):
    """文本检索 + 生成；返回 (answer, images_out, retrieved_text)。"""
    lib = _normalize_library(library)
    if not check_index_ready(lib):
        if lib == "personal":
            msg = (
                "个人图库索引未就绪：请将照片放入 `data/personal_images/` 后运行 "
                "`python build_personal_index.py`。"
            )
        else:
            msg = "请先运行 `python build_index.py` 构建 COCO 演示索引！"
        return msg, [], ""

    pipeline = get_pipeline(lib)
    answer, retrieved = pipeline.query_by_text(
        query.strip(),
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


# ---------- 核心推理函数 ----------

def text_query_fn(query: str, top_k: int, library: str):
    """文本问答入口。"""
    if not query.strip():
        return "请输入问题。", [], ""
    return _run_text_query(query, top_k, library)


def image_query_fn(image: Optional[Image.Image], question: str, top_k: int, library: str):
    """以图搜图 + 问答入口。"""
    if image is None:
        return "请上传一张图像。", [], ""
    lib = _normalize_library(library)
    if not check_index_ready(lib):
        if lib == "personal":
            return (
                "个人图库索引未就绪：请将照片放入 `data/personal_images/` 后运行 "
                "`python build_personal_index.py`。",
                [],
                "",
            )
        return "请先运行 `python build_index.py` 构建 COCO 演示索引！", [], ""

    if not question.strip():
        question = "请描述图像中的主要内容。"

    pipeline = get_pipeline(lib)
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


def voice_query_fn(
    audio_path: Optional[str],
    supplement: str,
    top_k: int,
    library: str,
):
    """
    语音问答：ASR 转写后与「文本问答」共用检索与生成。
    audio_path: Gradio Audio(type='filepath') 返回的临时 wav 路径。
    """
    if not audio_path or not str(audio_path).strip():
        return "", "请先录制或上传一段音频。", [], ""

    try:
        from src.speech_asr import transcribe_file

        transcript = transcribe_file(str(audio_path), config_path="config.yaml")
    except RuntimeError as e:
        return "", f"语音识别不可用：{e}", [], ""
    except Exception as e:
        return "", f"语音识别失败（可检查 ffmpeg 是否已安装）：{e}", [], ""

    if not (transcript or "").strip():
        return "", "未识别到有效语音内容，请重试或检查麦克风/音量。", [], ""

    query = transcript.strip()
    if supplement and supplement.strip():
        query = f"{query}。补充说明：{supplement.strip()}"

    answer, images_out, retrieved_text = _run_text_query(query, top_k, library)
    return transcript, answer, images_out, retrieved_text


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
<span class="mm-badge">faster-whisper</span>
</div>

</div>
        """
    )

    gr.Markdown(
        """
**图库选择**：「COCO 演示库」用于复现论文式检索；「个人图库」用于本地生活照片（独立索引，默认不提交照片到 Git）。**语音**：本地 ASR 转写后走同一套检索与 Qwen2-VL 生成（需 `pip install faster-whisper`，建议系统安装 **ffmpeg**）。
"""
    )
    library_radio = gr.Radio(
        choices=[
            ("COCO 演示库（默认）", "coco"),
            ("个人图库（本地生活向）", "personal"),
        ],
        value="coco",
        label="当前检索图库",
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
            inputs=[text_input, top_k_slider, library_radio],
            outputs=[text_answer, text_gallery, text_retrieved],
        )

        gr.Examples(
            label="示例一键填入",
            examples=[
                ["一只猫坐在沙发上", 3, "coco"],
                ["海边的日落风景", 3, "coco"],
                ["运动员在比赛中", 3, "coco"],
                ["一个小孩在公园玩耍", 3, "coco"],
            ],
            inputs=[text_input, top_k_slider, library_radio],
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
            inputs=[image_input, image_question, top_k_slider2, library_radio],
            outputs=[image_answer, image_gallery, image_retrieved],
        )

    with gr.Tab("语音问答"):
        gr.Markdown(
            "录制或上传音频，系统先用 **faster-whisper** 转写成文本，再按当前所选图库检索并生成回答。"
            " 可在下方补充文字，拼接到识别结果后一起参与检索。"
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=280):
                with gr.Group():
                    voice_audio = gr.Audio(
                        label="语音输入（麦克风或上传文件）",
                        sources=["microphone", "upload"],
                        type="filepath",
                    )
                    voice_supplement = gr.Textbox(
                        label="补充说明（可选，会拼在识别文字之后）",
                        placeholder="例如：重点找有猫的照片",
                        lines=2,
                    )
                    top_k_slider3 = gr.Slider(
                        1, 5, value=3, step=1,
                        label="检索图像数量 (top-k)",
                    )
                voice_btn = gr.Button("识别语音并检索回答", variant="primary", size="lg")

            with gr.Column(scale=7, min_width=320):
                with gr.Group():
                    voice_transcript = gr.Textbox(
                        label="识别文本",
                        lines=3,
                        interactive=False,
                    )
                    voice_answer = gr.Textbox(
                        label="模型回答",
                        lines=6,
                        interactive=False,
                        elem_classes=["mm-answer"],
                    )
                    voice_gallery = gr.Gallery(
                        label="检索到的图像",
                        columns=3,
                        height=320,
                        object_fit="contain",
                        show_label=True,
                        elem_classes=["mm-gallery"],
                    )
                    voice_retrieved = gr.Textbox(
                        label="检索详情（文件名 · 分数 · 标注）",
                        lines=5,
                        interactive=False,
                    )

        voice_btn.click(
            fn=voice_query_fn,
            inputs=[voice_audio, voice_supplement, top_k_slider3, library_radio],
            outputs=[voice_transcript, voice_answer, voice_gallery, voice_retrieved],
        )

    gr.Markdown(
        """
<div class="mm-foot">

**使用前**：COCO 库执行 `python build_index.py`；个人图库将照片放入 `data/personal_images/` 后执行 `python build_personal_index.py`（可选 `personal_notes.json` 备注）。语音依赖 `faster-whisper` 与 **ffmpeg**（见 README）。

**数据流（文本/语音）**：Query → CLIP 文本编码 → FAISS 召回 → 重排/阈值 → Qwen2-VL；**语音**先经 ASR 得到 Query。

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
