import torch
from PIL import Image
from transformers import AutoModelForCausalLM

from .base import BaseVisionLanguageModel

# https://huggingface.co/AIDC-AI/Ovis2-34B


class Ovis234B(BaseVisionLanguageModel):
    model: AutoModelForCausalLM

    def __init__(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            "AIDC-AI/Ovis2-34B",
            # torch_dtype=torch.bfloat16,
            torch_dtype=torch.float16,
            multimodal_max_length=32768,
            trust_remote_code=True,
            load_in_4bit=True,
        ).eval()
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()

    def predict(self, prompt: str, images: list[Image.Image]) -> str:
        cot_suffix = "Provide a step-by-step solution to the problem, and conclude with 'the action is' followed by the final solution."
        max_partition = 9
        query = f"{self.get_images_query(images)}\n{prompt}\n{cot_suffix}"

        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
            query, images, max_partition=max_partition
        )
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(
                dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device
            )
        pixel_values = [pixel_values]

        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True,
            )
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **gen_kwargs,
            )[0]
            output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
            return output

    def get_images_query(self, images: list[Image.Image]) -> str:
        if len(images) == 1:
            return "<image>"
        return "\n".join([f"Image {i+1}: <image>" for i in range(len(images))])
