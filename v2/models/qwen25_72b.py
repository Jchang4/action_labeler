import base64
from io import BytesIO
from time import sleep

from openai import OpenAI
from PIL import Image

from .base import BaseVisionLanguageModel


def pil_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    buffered = BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class Qwen25VL72B(BaseVisionLanguageModel):
    client: OpenAI

    def __init__(self) -> None:
        OPENROUTER_API_KEY = (
            "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"
        )

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

    def predict(self, prompt: str, images: list[Image.Image]) -> str:
        completion = self.client.chat.completions.create(
            model="qwen/qwen2.5-vl-72b-instruct:free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *[
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{pil_to_base64(image)}",
                                },
                            }
                            for image in images
                        ],
                    ],
                }
            ],
        )
        sleep(3)
        return completion.choices[0].message.content
