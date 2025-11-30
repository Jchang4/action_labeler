import base64
import io

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "LlamaCpp requires the openai package. Please install it with `pip install openai`."
    )

from PIL import Image

from .base import BaseVisionLanguageModel


def encode_image(image_path: str | Image.Image) -> str:
    if isinstance(image_path, Image.Image):
        image_path = image_path.convert("RGB")

        img_byte_arr = io.BytesIO()
        image_path.save(img_byte_arr, format="JPEG")
        img_byte_arr = img_byte_arr.getvalue()
        return base64.b64encode(img_byte_arr).decode("utf-8")
    else:
        # Handle file path
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


class LlamaCpp(BaseVisionLanguageModel):
    client: OpenAI
    max_tokens: int

    def __init__(self, system_prompt: str, max_tokens: int = 4096) -> None:
        super().__init__(system_prompt)
        self.client = OpenAI(base_url="http://127.0.0.1:5000/v1/")
        self.max_tokens = max_tokens

    def predict(self, prompt: str, images: list[Image.Image]) -> str:
        encoded_images = [encode_image(image) for image in images]
        response = self.client.chat.completions.create(
            model="llama-cpp-local",
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *[
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            }
                            for base64_image in encoded_images
                        ],
                    ],
                },
            ],
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content
