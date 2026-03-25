"""
generator.py
------------
使用 Qwen2-VL 多模态大语言模型，基于检索到的图像生成自然语言回答。
Qwen2-VL 支持多图输入，可同时理解多张检索图像并综合回答用户问题。
"""

from typing import List, Dict
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig


class Qwen2VLGenerator:
    """
    多模态生成器：
    - 输入: 用户 query + 检索到的 top-k 图像路径
    - 输出: 大模型生成的自然语言回答
    4bit 量化后约占用 3-4GB 显存，RTX 4070 Laptop 完全可用。
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        print(f"[Generator] 加载 Qwen2-VL: {model_name}  4bit={load_in_4bit}")

        quantization_config = None
        if load_in_4bit and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        print("[Generator] 模型加载完成")

    def generate(self, query: str, retrieved_items: List[Dict], max_images: int = 3) -> str:
        """
        基于检索结果生成回答。
        构造多模态 prompt: [图像1][图像2]... + 用户问题
        """
        # 加载检索图像（最多 max_images 张）
        images = []
        valid_items = []
        for item in retrieved_items[:max_images]:
            img_path = item.get("image_path", "")
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                valid_items.append(item)
            except Exception:
                continue

        if not images:
            return "抱歉，未能找到相关图像，无法回答该问题。"

        # 构造消息格式（Qwen2-VL chat template）
        content = []
        for i, img in enumerate(images):
            content.append({"type": "image", "image": img})
        content.append({
            "type": "text",
            "text": (
                f"你是一个图像问答助手。我为你提供了 {len(images)} 张与问题相关的图像，"
                f"请根据这些图像内容回答用户的问题。\n\n"
                f"用户问题：{query}"
            )
        })

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        # 只取新生成的部分（去掉 prompt）
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_len:]
        answer = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return answer.strip()
