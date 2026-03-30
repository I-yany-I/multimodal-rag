# 多模态 RAG 图像问答系统

> 基于 **CLIP + Qwen2-VL** 构建的多模态检索增强生成（RAG）系统，支持自然语言检索图像库并生成上下文相关回答，同时支持以图搜图问答。

## 系统架构

```
用户输入（文本/图像）
        ↓
   CLIP 编码器
   （图文统一向量空间）
        ↓
  FAISS 向量数据库
  （cosine 相似度检索）
        ↓
  top-k 相关图像
        ↓
  Qwen2-VL-2B 生成器
  （多模态大语言模型）
        ↓
   自然语言回答
```

## 技术栈

| 模块 | 技术 | 说明 |
|------|------|------|
| 图文编码 | CLIP (ViT-B/32) | OpenAI 对比学习预训练，512维向量 |
| 向量数据库 | FAISS (IndexFlatIP) | 内积检索等价cosine，毫秒级响应 |
| 多模态LLM | Qwen2-VL-2B-Instruct | 阿里开源，4bit量化后约3GB显存 |
| 演示界面 | Gradio | 文本问答 + 以图搜图双Tab |
| REST API | FastAPI + Uvicorn | 自动生成Swagger文档 |
| 框架 | PyTorch 2.x + CUDA | RTX 4070 Laptop 验证 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 构建图像索引

```bash
python build_index.py
```

脚本会自动：
- 从 COCO 2017 val 下载 5000 张样例图像
- 用 CLIP 批量编码为 512 维向量
- 构建 FAISS 索引并保存到 `data/`

可在 `config.yaml` 的 `data.max_images` 调整下载数量，或运行 `python build_index.py --max_images 5000` 覆盖配置。

### 3. 启动 Gradio 演示

```bash
python app.py
```

访问 http://localhost:7860

### 4. 启动 FastAPI 接口（可选）

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

API 文档：http://localhost:8000/docs

## 功能演示

### 文本问答

```
输入：一只猫坐在沙发上
检索：返回图库中最相关的 top-3 图像
生成：Qwen2-VL 基于检索图像给出描述性回答
```

### 以图搜图

```
上传：任意图像
检索：CLIP 编码后在图库中找相似图像
回答：LLM 综合多张图像回答用户问题
```

## API 接口

### POST `/query/text`

```json
{
  "query": "海边的日落风景",
  "top_k": 3
}
```

响应：
```json
{
  "query": "海边的日落风景",
  "answer": "这些图像展示了...",
  "retrieved_images": [...],
  "latency_ms": 1240.5
}
```

### POST `/query/image`

```
multipart/form-data:
  file: <图像文件>
  question: 请描述图像内容
  top_k: 3
```

### GET `/health`

```json
{
  "status": "ok",
  "index_ready": true
}
```

## 评估结果

在 COCO val2017 数据集（100条query，以caption检索对应图像）：

| 指标 | 数值 |
|------|------|
| Recall@1 | ~0.42 |
| Recall@3 | ~0.68 |
| Recall@5 | ~0.78 |
| MRR | ~0.52 |
| NDCG@5 | ~0.61 |
| 平均检索延迟 | ~15 ms |

运行评估：
```bash
python evaluate.py
```

## 项目结构

```
multimodal-rag/
├── src/
│   ├── encoder.py      # CLIP 双塔编码器
│   ├── retriever.py    # FAISS 向量检索
│   ├── generator.py    # Qwen2-VL 多模态生成
│   ├── pipeline.py     # 端到端 RAG 流程
│   └── utils.py        # 工具函数
├── app.py              # Gradio 演示界面
├── api.py              # FastAPI REST 接口
├── build_index.py      # 离线建库脚本
├── evaluate.py         # 检索评估脚本
├── config.yaml         # 配置文件
└── requirements.txt
```

## 核心技术要点

### CLIP 对比学习
CLIP 通过图文配对数据训练，使语义相似的图像和文字在向量空间中距离更近。
本项目将图库图像预编码为向量存入 FAISS，查询时只需对 query 实时编码，
做一次向量相似度计算即可完成检索，效率远高于逐图比较。

### RAG 流程
传统 RAG（检索增强生成）用于文本领域；本项目将其扩展至多模态：
用检索到的图像作为 LLM 的"上下文"，使模型能基于实际视觉内容回答问题，
而非凭空生成，显著提升回答的准确性和可信度。

### 4bit 量化
Qwen2-VL-7B 原始需要约 14GB 显存，通过 BitsAndBytes 的 NF4 量化，
压缩至约 4GB，可在消费级 8GB 显卡上流畅运行，精度损失极小。

## 硬件要求

- GPU：NVIDIA 8GB+ 显存（RTX 3070/4060/4070 及以上）
- RAM：16GB+
- 存储：10GB+（模型 + 数据）

## 相关项目

- [项目一：基于 BERT + LoRA 的文本分类 RAG 系统](https://github.com/I-yany-I/llm-text-classification-system)

---

*作者：学生简历项目  |  技术栈：PyTorch · CLIP · FAISS · Qwen2-VL · FastAPI · Gradio*
