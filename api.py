"""
api.py
------
FastAPI RESTful 接口，对外暴露多模态 RAG 能力。
支持文本问答和以图搜图两个端点。

启动方式: uvicorn api:app --host 0.0.0.0 --port 8000 --reload

接口文档: http://localhost:8000/docs  (Swagger UI)
"""

import base64
import io
import os
import time
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

app = FastAPI(
    title="多模态 RAG 图像问答 API",
    description="基于 CLIP + Qwen2-VL 的多模态 RAG 系统，支持文本检索图像和以图搜图",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 延迟初始化 pipeline（按图库分实例）
_pipelines: Dict[str, object] = {}


def _normalize_library(library: Optional[str]) -> str:
    key = (library or "coco").strip().lower()
    return key if key in ("coco", "personal") else "coco"


def get_pipeline(library: Optional[str] = None):
    key = _normalize_library(library)
    global _pipelines
    if key not in _pipelines:
        from src.pipeline import MultimodalRAGPipeline

        _pipelines[key] = MultimodalRAGPipeline(config_path="config.yaml", library=key)
    return _pipelines[key]


# ---------- 数据模型 ----------

class TextQueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=10)
    include_base64: bool = False
    library: str = Field(
        default="coco",
        description="coco=COCO 演示索引；personal=个人图库独立索引（需先 build_personal_index）",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "一只猫坐在沙发上",
                "top_k": 3,
                "include_base64": False,
                "library": "coco",
            }
        }


class RetrievedImage(BaseModel):
    filename: str
    image_path: str
    caption: str
    score: float
    image_base64: Optional[str] = None  # 可选：返回 base64 图像


class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_images: List[RetrievedImage]
    latency_ms: float
    transcript: Optional[str] = None  # 语音接口：ASR 原文；文本接口为 null
    asr_detected_language: Optional[str] = None
    asr_language_probability: Optional[float] = None
    asr_segments: Optional[List[Dict[str, Any]]] = None  # {start, end, text} 分段，可解释 ASR


# ---------- 工具函数 ----------

def image_to_base64(image_path: str, max_size: int = 512) -> Optional[str]:
    """将图像转为 base64 字符串（用于 API 返回）。"""
    try:
        img = Image.open(image_path).convert("RGB")
        # 缩放避免响应体过大
        img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def format_retrieved(items: List[dict], include_base64: bool = False) -> List[RetrievedImage]:
    result = []
    for item in items:
        b64 = image_to_base64(item["image_path"]) if include_base64 else None
        result.append(RetrievedImage(
            filename=item["filename"],
            image_path=item["image_path"],
            caption=item.get("caption", ""),
            score=item["score"],
            image_base64=b64,
        ))
    return result


# ---------- API 端点 ----------

@app.get("/", summary="健康检查")
def root():
    return {"status": "ok", "message": "多模态 RAG API 运行中"}


@app.get("/health", summary="服务健康状态")
def health():
    import yaml
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    coco_ready = os.path.exists(cfg["retrieval"]["index_path"]) and os.path.exists(
        cfg["retrieval"]["metadata_path"]
    )
    pl = cfg.get("personal_library") or {}
    p_ip = pl.get("index_path", "data/personal_faiss.index")
    p_mp = pl.get("metadata_path", "data/personal_metadata.json")
    personal_ready = os.path.exists(p_ip) and os.path.exists(p_mp)
    try:
        import faster_whisper  # noqa: F401
        asr_ready = True
    except Exception:
        asr_ready = False
    return {
        "status": "ok" if (coco_ready or personal_ready) else "index_not_ready",
        "index_ready_coco": coco_ready,
        "index_ready_personal": personal_ready,
        "asr_ready": asr_ready,
        "message": "至少一个图库索引就绪" if (coco_ready or personal_ready) else "请运行 build_index.py 或 build_personal_index.py",
    }


@app.post("/query/text", response_model=QueryResponse, summary="文本问答")
def query_by_text(request: TextQueryRequest):
    """
    输入自然语言问题，检索图像库并由 Qwen2-VL 生成回答。

    - **query**: 用户问题，如"一只猫坐在沙发上"
    - **top_k**: 检索的图像数量（1-5）
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")

    t0 = time.time()
    try:
        pipeline = get_pipeline(request.library)
        answer, retrieved = pipeline.query_by_text(
            request.query,
            top_k=request.top_k,
            max_images=request.top_k,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    latency_ms = (time.time() - t0) * 1000
    return QueryResponse(
        query=request.query,
        answer=answer,
        retrieved_images=format_retrieved(retrieved, include_base64=request.include_base64),
        latency_ms=round(latency_ms, 1),
    )


@app.post("/query/image", response_model=QueryResponse, summary="以图搜图问答")
def query_by_image(
    file: UploadFile = File(..., description="上传查询图像（jpg/png）"),
    question: str = Form(default="请描述图像中的主要内容。", description="关于图像的问题"),
    top_k: int = Form(default=3, description="检索图像数量"),
    include_base64: bool = Form(default=False, description="是否返回 base64 图像"),
    library: str = Form(default="coco", description="coco 或 personal"),
):
    """
    上传图像，检索相似图像并回答问题。
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图像文件")
    if not (1 <= top_k <= 10):
        raise HTTPException(status_code=400, detail="top_k 必须在 1 到 10 之间")

    try:
        image_bytes = file.file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图像解析失败: {e}")

    t0 = time.time()
    try:
        pipeline = get_pipeline(library)
        answer, retrieved = pipeline.query_by_image(
            image,
            question,
            top_k=top_k,
            max_images=top_k,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    latency_ms = (time.time() - t0) * 1000
    return QueryResponse(
        query=question,
        answer=answer,
        retrieved_images=format_retrieved(retrieved, include_base64=include_base64),
        latency_ms=round(latency_ms, 1),
    )


@app.get("/stats", summary="索引统计信息")
def stats(library: str = "coco"):
    """返回指定图库的向量索引统计信息。library=coco|personal"""
    import yaml
    from src.retriever import FAISSRetriever
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    key = _normalize_library(library)
    if key == "personal":
        pl = cfg.get("personal_library") or {}
        index_path = pl.get("index_path", "data/personal_faiss.index")
        metadata_path = pl.get("metadata_path", "data/personal_metadata.json")
    else:
        ret_cfg = cfg["retrieval"]
        index_path = ret_cfg["index_path"]
        metadata_path = ret_cfg["metadata_path"]
    if not os.path.exists(index_path):
        raise HTTPException(status_code=503, detail=f"索引未构建: {index_path}")
    retriever = FAISSRetriever()
    retriever.load(index_path, metadata_path)
    return {
        "library": key,
        "total_images": retriever.index.ntotal,
        "vector_dim": retriever.index.d,
        "index_path": index_path,
        "has_captions": sum(1 for m in retriever.metadata if m.get("caption")),
    }


@app.post("/query/voice", response_model=QueryResponse, summary="语音问答（ASR + 文本检索）")
async def query_by_voice(
    file: UploadFile = File(..., description="音频文件 wav/mp3/webm 等"),
    supplement: str = Form(default="", description="可选，拼在识别文字之后"),
    top_k: int = Form(default=3, ge=1, le=10),
    include_base64: bool = Form(default=False),
    library: str = Form(default="coco"),
):
    """
    上传音频 → faster-whisper 转写 → 与 `/query/text` 相同检索与生成链路。
    """
    try:
        import tempfile

        from src.speech_asr import transcribe_file_with_meta
    except ImportError as e:
        raise HTTPException(status_code=501, detail=f"语音识别依赖未就绪: {e}") from e

    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    tmp_path = None
    t_all = time.time()
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        tmp.close()
        asr_meta = transcribe_file_with_meta(tmp_path, config_path="config.yaml")
        transcript = (asr_meta.get("text") or "").strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"语音识别失败: {e}") from e
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    if not (transcript or "").strip():
        raise HTTPException(status_code=400, detail="未识别到有效语音内容")

    query = transcript.strip()
    if supplement and supplement.strip():
        query = f"{query}。补充说明：{supplement.strip()}"

    try:
        pipeline = get_pipeline(library)
        answer, retrieved = pipeline.query_by_text(
            query,
            top_k=top_k,
            max_images=top_k,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    latency_ms = (time.time() - t_all) * 1000
    return QueryResponse(
        query=query,
        answer=answer,
        retrieved_images=format_retrieved(retrieved, include_base64=include_base64),
        latency_ms=round(latency_ms, 1),
        transcript=transcript.strip(),
        asr_detected_language=asr_meta.get("detected_language"),
        asr_language_probability=asr_meta.get("language_probability"),
        asr_segments=asr_meta.get("segments"),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
