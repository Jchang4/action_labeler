from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from .base import BaseVisionLanguageModel


class Phi4VL(BaseVisionLanguageModel):
    model: AutoModelForCausalLM
    processor: AutoProcessor
    generation_config: GenerationConfig

    def __init__(self) -> None:
        model_path = "microsoft/Phi-4-multimodal-instruct"

        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).cuda()

        # Load generation config
        self.generation_config = GenerationConfig.from_pretrained(model_path)

    def predict(self, prompt: str, images: list[Image.Image]) -> str:
        # Define prompt structure
        user_prompt = "<|user|>"
        assistant_prompt = "<|assistant|>"
        prompt_suffix = "<|end|>"

        prompt = f"{user_prompt}{self.get_image_prompt(images)}{prompt}{prompt_suffix}{assistant_prompt}"

        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(
            "cuda:0"
        )

        # Generate response
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1000,
            generation_config=self.generation_config,
        )
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response

    def get_image_prompt(self, images: list[Image.Image]) -> str:
        image_prompt = ""
        for i in range(len(images)):
            image_prompt += f"<|image_{i+1}|>"
        return image_prompt
