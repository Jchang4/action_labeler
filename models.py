from PIL import Image
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaOnevisionForConditionalGeneration,
    BitsAndBytesConfig,
)

from .base import BaseClassificationModel


class LlavaOnevision7B(BaseClassificationModel):
    processor: AutoProcessor
    model: LlavaOnevisionForConditionalGeneration

    def __init__(self, load_in_4bit: bool = True):
        self.load_in_4bit = load_in_4bit
        # model_id = "llava-hf/llava-onevision-qwen2-7b-si-hf"  # Single image
        model_id = "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf"  # image-text interleaved input and video input
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_flash_attention_2=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch.float16,
            ),
        ).to(0)

    def predict(self, image: Image.Image, prompt: str) -> str:
        # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
        # Each value in "content" has to be a list of dicts with types ("text", "image")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            0, torch.float16
        )

        output = self.model.generate(**inputs, max_new_tokens=500, do_sample=False)
        response = self.processor.decode(output[0][2:], skip_special_tokens=True)
        return response.split("assistant")[1]


class Ovis10B(BaseClassificationModel):
    model: AutoModelForCausalLM

    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "AIDC-AI/Ovis1.6-Gemma2-9B",
            torch_dtype=torch.bfloat16,
            multimodal_max_length=8192,
            trust_remote_code=True,
            load_in_4bit=True,
        ).to(0)

    def predict(self, image: Image.Image, prompt: str) -> str:
        text_tokenizer = self.model.get_text_tokenizer()
        visual_tokenizer = self.model.get_visual_tokenizer()

        # enter image path and prompt
        query = f"<image>\n{prompt}"

        # format conversation
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, [image])
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        pixel_values = [
            pixel_values.to(
                dtype=visual_tokenizer.dtype, device=visual_tokenizer.device
            )
        ]

        # generate output
        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=text_tokenizer.pad_token_id,
                use_cache=True,
            )
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **gen_kwargs,
            )[0]
            output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
            return output


class Molmo7B(BaseClassificationModel):
    processor: AutoProcessor
    model: AutoModelForCausalLM

    def __init__(self, load_in_4bit: bool = False):
        self.processor = AutoProcessor.from_pretrained(
            "allenai/Molmo-7B-D-0924",
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "allenai/Molmo-7B-D-0924",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=load_in_4bit,
        )

    def predict(self, images: list[Image.Image], prompt: str) -> str:
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            inputs = self.processor.process(
                images=[
                    image.convert("RGB") if image.mode != "RGB" else image
                    for image in images
                ],
                text=prompt,
            )

            # move inputs to the correct device and make a batch of size 1
            inputs = {
                k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()
            }

            # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=500, stop_strings="<|endoftext|>"),
                tokenizer=self.processor.tokenizer,
            )

            # only get generated tokens; decode them to text
            generated_tokens = output[0, inputs["input_ids"].size(1) :]
            generated_text = self.processor.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )

            return generated_text


class Pixtral12B(BaseClassificationModel):
    model: LlavaForConditionalGeneration
    processor: AutoProcessor

    def __init__(self) -> None:
        model_id = "mistral-community/pixtral-12b"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            load_in_4bit=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_id, device_map="auto")

    def predict(self, image: Image, prompt: str) -> str:
        # with torch.autocast("cuda", enabled=True, dtype=torch.float16):
        PROMPT = f"<s>[INST]{prompt}\n[IMG][/INST]"

        inputs = self.processor(text=PROMPT, images=[image], return_tensors="pt").to(
            "cuda"
        )
        generate_ids = self.model.generate(**inputs, max_new_tokens=500)
        output = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output.replace(prompt, "").strip()
