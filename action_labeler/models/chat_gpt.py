import base64
import io
from time import sleep

from PIL import Image

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "ChatGPT requires the openai package. Please install it with `pip install openai`."
    )

from .base import BaseVisionLanguageModel


class ChatGPT(BaseVisionLanguageModel):
    """
    ChatGPT model wrapper.

    Args:
        model: The model to use.
        system_prompt: The system prompt to use.
        max_tokens: The maximum number of tokens to generate.
        sleep_time: The time to sleep after generating a response.
    """

    model: str
    client: OpenAI
    max_tokens: int
    sleep_time: int

    def __init__(
        self,
        model: str,
        system_prompt: str,
        max_tokens: int = 1024,
        sleep_time: int = 5,
    ) -> None:
        super().__init__(system_prompt)
        self.model = model
        self.client = OpenAI()
        self.max_tokens = max_tokens
        self.sleep_time = sleep_time

    def predict(self, prompt: str, images: list[Image.Image]) -> str:
        images = [
            image.convert("RGB") if image.mode != "RGB" else image for image in images
        ]

        img_byte_arrs = []
        for image in images:
            # Resize so largest dimension is 1024px
            max_dim = max(image.size)
            scale = 1024 / max_dim
            image = image.resize(
                (int(image.size[0] * scale), int(image.size[1] * scale))
            )
            # Save to byte array
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="JPEG")
            img_byte_arrs.append(img_byte_arr.getvalue())

        response = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.system_prompt.strip().strip("\n")}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *[
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64.b64encode(img_byte_arr).decode('utf-8')}",
                                    "detail": "low",
                                },
                            }
                            for img_byte_arr in img_byte_arrs
                        ],
                    ],
                },
            ],
        )

        sleep(self.sleep_time)
        return response.choices[0].message.content.strip()
