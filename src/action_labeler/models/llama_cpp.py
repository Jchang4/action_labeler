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
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
        presence_penalty: float = 1.5,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
        seed: int = -1,
        max_tokens: int = 32768,
    ):
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.repeat_penalty = repeat_penalty
        self.seed = seed
        self.max_tokens = max_tokens

    def load_image(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")

    def predict(self, system: str, user: str, images: list[Image.Image]) -> str:
        user_content = []
        for image in images:
            b64 = self._encode_image(image)
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
        user_content.append({"type": "text", "text": user})

        payload: dict = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repeat_penalty": self.repeat_penalty,
            "seed": self.seed,
        }

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        if not content:
            raise ValueError(f"Model returned empty response: {data}")
        return content

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
