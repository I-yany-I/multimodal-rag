"""
speech_asr.py
-------------
本地语音转写（ASR），用于「语音 → 文本 → 再走 CLIP 检索 + Qwen2-VL」链路。

依赖：pip install faster-whisper
常见格式（wav/mp3 等）建议在系统 PATH 中安装 ffmpeg：https://ffmpeg.org/download.html
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from src.utils import load_config

_model_cache: Optional[Any] = None
_model_cache_key: Optional[Tuple[str, str, str]] = None


def _speech_cfg(config_path: str) -> dict:
    cfg = load_config(config_path)
    return cfg.get("speech") or {}


def get_whisper_model(config_path: str = "config.yaml"):
    """懒加载 faster-whisper 模型（按 model_size + device + compute_type 缓存）。"""
    global _model_cache, _model_cache_key
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError(
            "未安装 faster-whisper，请执行: pip install faster-whisper"
        ) from e

    sc = _speech_cfg(config_path)
    if not sc.get("enabled", True):
        raise RuntimeError("config.yaml 中 speech.enabled 为 false，已关闭语音识别")

    model_size = str(sc.get("model_size", "base"))
    device = str(sc.get("device", "cuda"))
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    compute_type = str(sc.get("compute_type", "float16"))
    if device == "cpu":
        compute_type = str(sc.get("compute_type_cpu", "int8"))

    key = (model_size, device, compute_type)
    if _model_cache is not None and _model_cache_key == key:
        return _model_cache

    _model_cache = WhisperModel(model_size, device=device, compute_type=compute_type)
    _model_cache_key = key
    return _model_cache


def transcribe_file_with_meta(audio_path: str, config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    转写并返回结构化信息：全文、检测语种、分段时间轴（用于可解释展示与 API）。
    说明：仍以「语音→文本→CLIP 检索」为主，不引入与图像不对齐的音频塔嵌入。
    """
    if not audio_path or not os.path.isfile(audio_path):
        raise ValueError("无效的音频文件路径")

    sc = _speech_cfg(config_path)
    model = get_whisper_model(config_path)

    language = sc.get("language")
    if language in ("", "auto", "null", None):
        language = None
    else:
        language = str(language)

    beam_size = int(sc.get("beam_size", 5))
    vad_filter = bool(sc.get("vad_filter", True))
    want_ts = bool(sc.get("return_timestamps", True))

    segments_iter, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )

    parts: List[str] = []
    seg_rows: List[Dict[str, Any]] = []
    for seg in segments_iter:
        parts.append(seg.text)
        if want_ts:
            seg_rows.append(
                {
                    "start": round(float(seg.start), 2),
                    "end": round(float(seg.end), 2),
                    "text": (seg.text or "").strip(),
                }
            )

    text = "".join(parts).strip()
    lang = getattr(info, "language", None)
    lang_p = getattr(info, "language_probability", None)
    try:
        lang_p_f = float(lang_p) if lang_p is not None else None
    except (TypeError, ValueError):
        lang_p_f = None

    return {
        "text": text,
        "detected_language": lang,
        "language_probability": lang_p_f,
        "segments": seg_rows,
    }


def format_transcription_display(meta: Dict[str, Any], config_path: str = "config.yaml") -> str:
    """Gradio 用：拼装「语种 + 全文 + 分段时间轴」。"""
    sc = _speech_cfg(config_path)
    show_lang = bool(sc.get("show_language_line", True))
    lines: List[str] = []
    if show_lang and meta.get("detected_language"):
        lp = meta.get("language_probability")
        if lp is not None:
            lines.append(f"【检测语种】{meta['detected_language']}（confidence≈{float(lp):.2f}）")
        else:
            lines.append(f"【检测语种】{meta['detected_language']}")
    lines.append("【全文】\n" + (meta.get("text") or "").strip())
    segs = meta.get("segments") or []
    if segs:
        lines.append("\n【分段时间轴】")
        for s in segs:
            lines.append(f"[{s['start']:.2f}s – {s['end']:.2f}s] {s.get('text', '')}")
    return "\n".join(lines).strip()


def transcribe_file(audio_path: str, config_path: str = "config.yaml") -> str:
    """仅返回全文；内部复用 transcribe_file_with_meta。"""
    return transcribe_file_with_meta(audio_path, config_path)["text"]
