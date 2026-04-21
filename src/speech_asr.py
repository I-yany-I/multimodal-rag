"""
speech_asr.py
-------------
本地语音转写（ASR），用于「语音 → 文本 → 再走 CLIP 检索 + Qwen2-VL」链路。

依赖：pip install faster-whisper
常见格式（wav/mp3 等）建议在系统 PATH 中安装 ffmpeg：https://ffmpeg.org/download.html
"""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

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


def transcribe_file(audio_path: str, config_path: str = "config.yaml") -> str:
    """
    将音频文件转写为文本（自动语言检测，也可在 config 中指定 speech.language）。
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

    segments, _info = model.transcribe(
        audio_path,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )
    parts = [seg.text for seg in segments]
    text = "".join(parts).strip()
    return text
