import torch
from PIL import Image
from transformers import Pipeline, pipeline

from .base import BaseVisionLanguageModel

# https://huggingface.co/google/gemma-3-27b-it


class Gemma3(BaseVisionLanguageModel):
    model: Pipeline

    def __init__(self) -> None:
        self.model = pipeline(
            "image-text-to-text",
            model="google/gemma-3-27b-it",
            torch_dtype=torch.bfloat16,
            model_kwargs={
                "load_in_4bit": True,
            },
        )

    def predict(self, prompt: str, images: list[Image.Image]) -> str:
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant that can identify the action of each person in an image. You are thorough and detailed. You only choose actions from the list below. You only choose actions where you can see the person doing the action and the object they are interacting with. You do not choose actions where the person is not doing anything or the object is not visible.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "image", "image": image} for image in images]
                + [{"type": "text", "text": prompt}],
            },
        ]

        output = self.model(text=messages, max_new_tokens=1024)
        return output[0]["generated_text"][-1]["content"]
