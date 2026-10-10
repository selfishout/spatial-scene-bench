"""Model adapters. Each exposes ``generate(images, prompt) -> str``.

* ``random``              chance-level baseline (uses the answer type, never the answer)
* ``hf:<model_id>``       any Hugging Face image-text-to-text model
                          (Qwen2.5-VL, Qwen3-VL, InternVL3-hf, LLaVA-OneVision, Gemma 3, ...)
* ``openai:<model>``      OpenAI-compatible chat endpoint (OpenAI, vLLM ``serve``, SGLang, ...)
"""

from __future__ import annotations

import base64
import io
import os
import random

from PIL import Image


class VLM:
    name = "base"

    def generate(self, images: list[Image.Image], prompt: str, record: dict | None = None) -> str:
        raise NotImplementedError


class RandomBaseline(VLM):
    name = "random"

    def __init__(self, seed: int = 0, max_count: int = 8):
        self.rng = random.Random(seed)
        self.max_count = max_count

    def generate(self, images, prompt, record=None):
        at = record["answer_type"] if record else "yesno"
        if at == "count":
            return str(self.rng.randint(0, self.max_count))
        if at == "yesno":
            return self.rng.choice(["yes", "no"])
        if at == "choice":
            return self.rng.choice("ABCD")
        return f"{self.rng.uniform(0.5, 6.0):.2f}"


class HFVLM(VLM):
    def __init__(self, model_id: str, max_new_tokens: int = 512, dtype: str = "auto", device_map: str = "auto"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.name = model_id
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, device_map=device_map
        ).eval()
        self.max_new_tokens = max_new_tokens

    def generate(self, images, prompt, record=None):
        content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]


class OpenAIVLM(VLM):
    def __init__(self, model: str, base_url: str | None = None, max_tokens: int = 1024):
        from openai import OpenAI

        self.name = model
        self.client = OpenAI(base_url=base_url or os.environ.get("OPENAI_BASE_URL"))
        self.max_tokens = max_tokens

    @staticmethod
    def _encode(im: Image.Image) -> str:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def generate(self, images, prompt, record=None):
        content = [{"type": "image_url", "image_url": {"url": self._encode(im)}} for im in images]
        content.append({"type": "text", "text": prompt})
        resp = self.client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": content}],
            max_tokens=self.max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""


def load_model(spec: str, **kwargs) -> VLM:
    if spec == "random":
        return RandomBaseline()
    kind, _, name = spec.partition(":")
    if kind == "hf":
        return HFVLM(name, **kwargs)
    if kind == "openai":
        return OpenAIVLM(name, **kwargs)
    raise ValueError(f"Unknown model spec {spec!r}; expected random | hf:<id> | openai:<model>")
