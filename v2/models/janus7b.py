import torch
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseVisionLanguageModel


class Janus7B(BaseVisionLanguageModel):
    vl_chat_processor: VLChatProcessor
    vl_gpt: MultiModalityCausalLM
    tokenizer: AutoTokenizer

    def __init__(self) -> None:
        # specify the path to the model
        model_path = "deepseek-ai/Janus-Pro-7B"
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
            model_path
        )
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()

    def predict(self, prompt: str, images: list[Image.Image]) -> str:
        content_str = ""
        for _ in range(len(images)):
            content_str += f"<image_placeholder>\n"
        content_str += f"{prompt}"

        conversation = [
            {
                "role": "<|User|>",
                "content": content_str,
                "images": images,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # load images and prepare for inputs
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=images, force_batchify=True
        ).to(self.vl_gpt.device)

        # # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # # run the model to get the response
        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

        answer = self.tokenizer.decode(
            outputs[0].cpu().tolist(), skip_special_tokens=True
        )
        return answer
