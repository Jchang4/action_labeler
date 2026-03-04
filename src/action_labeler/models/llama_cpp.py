"""LlamaCpp model that connects to a running llama-server instance.

Expects a llama.cpp server to be running with multimodal support.
Example startup script: generative/scripts/start_vlm_server.sh

    ./build/bin/llama-server -m model.gguf --mmproj mmproj.gguf \\
        --port 5000 --host 0.0.0.0 --jinja

Uses the OpenAI-compatible /v1/chat/completions endpoint.
"""

import base64
from io import BytesIO

import requests
from PIL import Image

from .base import BaseModel


class LlamaCpp(BaseModel):
    def __init__(
        self,
        base_url: str = "http://localhost:5000",
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int = 1024,
    ):
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens

    def load_image(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")

    def predict(self, system: str, user: str, images: list[Image.Image]) -> str:
        user_content = []
        for image in images:
            b64 = self._encode_image(image)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        user_content.append({"type": "text", "text": user})

        payload: dict = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
