import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

from .base import BaseVisionLanguageModel


class PaliGemma10b448(BaseVisionLanguageModel):
    model: PaliGemmaForConditionalGeneration
    processor: PaliGemmaProcessor

    def __init__(self) -> None:
        model_id = "google/paligemma2-10b-pt-448"

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor = PaliGemmaProcessor.from_pretrained(model_id)

    def predict(self, prompt: str, images: list[Image.Image]) -> str:
        image_prefix = "\n".join(["<image>"] * len(images))

        model_inputs = (
            self.processor(
                # The model is a text completion model, so we need to provide the "Answer:" suffix
                text=f"{image_prefix}\n{prompt}\n\nResponse:",
                images=images,
                return_tensors="pt",
            )
            .to(torch.bfloat16)
            .to(self.model.device)
        )
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs, max_new_tokens=512, do_sample=False
            )
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            return decoded.strip()
