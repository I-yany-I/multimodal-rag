"""
generator.py
------------
使用 Qwen2-VL 多模态大语言模型，基于检索到的图像生成自然语言回答。
Qwen2-VL 支持多图输入，可同时理解多张检索图像并综合回答用户问题。
"""

from typing import List, Dict, Optional

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
        max_new_tokens: int = 384,
        temperature: float = 0.35,
        top_p: float = 0.9,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._rewrite_cache: Dict[str, str] = {}

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

    def _generate_from_messages(
        self,
        messages: List[Dict],
        images: Optional[List[Image.Image]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        processor_kwargs = {
            "text": [text],
            "return_tensors": "pt",
        }
        if images:
            processor_kwargs["images"] = images

        inputs = self.processor(**processor_kwargs)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
        }
        effective_temperature = self.temperature if temperature is None else temperature
        effective_top_p = self.top_p if top_p is None else top_p
        if effective_temperature and effective_temperature > 0:
            gen_kwargs["temperature"] = effective_temperature
            gen_kwargs["top_p"] = effective_top_p
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_len:]
        answer = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return answer.strip()

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in text)

    def rewrite_query_for_retrieval(self, query: str, chinese_only: bool = True) -> str:
        """
        将自然语言问题改写为适合 CLIP 检索的英文短语。
        主要用于中文 query 对英文图文空间检索效果不足的场景。
        """
        normalized = " ".join(query.strip().split())
        if not normalized:
            return ""
        if chinese_only and not self._contains_cjk(normalized):
            return normalized
        if normalized in self._rewrite_cache:
            return self._rewrite_cache[normalized]

        rewrite_prompt = (
            "Rewrite the user query into a short English phrase for image retrieval.\n"
            "Keep only the key visual entities, actions, scene, and attributes.\n"
            "Output one English line only, no explanation, no quotes, 4 to 14 words.\n\n"
            f"User query: {normalized}"
        )
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": rewrite_prompt}],
            }
        ]
        rewritten = self._generate_from_messages(
            messages,
            images=None,
            max_new_tokens=32,
            temperature=0.0,
        )
        rewritten = " ".join(
            rewritten.replace('"', " ").replace("'", " ").splitlines()[0].split()
        ).strip()
        if not rewritten:
            rewritten = normalized
        self._rewrite_cache[normalized] = rewritten
        return rewritten

    def generate(self, query: str, retrieved_items: List[Dict], max_images: int = 3) -> str:
        """
        基于检索结果生成回答。
        构造多模态 prompt: [图像1][图像2]... + 检索说明 + 用户问题
        """
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

        evidence_lines = []
        for i, it in enumerate(valid_items, start=1):
            cap = (it.get("caption") or "").strip()
            sc = it.get("score")
            sc_s = f"{sc:.3f}" if isinstance(sc, (int, float)) else "—"
            fname = it.get("filename", "")
            if cap:
                evidence_lines.append(
                    f"- 图{i} ({fname})，检索相关度约 {sc_s}；数据集标注摘要：{cap}"
                )
            else:
                evidence_lines.append(f"- 图{i} ({fname})，检索相关度约 {sc_s}。")
        evidence_block = "\n".join(evidence_lines)

        instruction = (
            "你是多模态 RAG 助手。上方多张图来自向量检索，可能与用户问题只有部分相关。\n"
            "请优先依据图像中的可见内容作答；可结合「数据集标注摘要」作补充，但若与图像明显矛盾则以图像为准。\n"
            "若检索图不足以支持结论，请明确说明依据不足，不要编造细节。\n"
            "如果多张图之间信息不一致，请优先回答它们的共同点，再补充说明不确定之处。\n"
            "回答用中文，先给简短结论，再给1到2句依据，不要空泛复述问题。"
        )

        user_text = (
            f"{instruction}\n\n"
            f"【检索证据】\n{evidence_block}\n\n"
            f"【用户问题】\n{query.strip()}"
        )

        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": user_text})

        messages = [{"role": "user", "content": content}]
        return self._generate_from_messages(messages, images=images)
