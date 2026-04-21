# 多模态 RAG：GPU 运行环境骨架（Linux + NVIDIA Container Toolkit）
# 构建：docker build -t multimodal-rag:local .
# 运行需 --gpus all，且宿主机已安装 NVIDIA 驱动

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt

COPY . .

# 默认启动 API；若需 Gradio 可改为 python app.py
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
