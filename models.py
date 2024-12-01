import base64
import io
import json
from time import sleep
from typing import Any

import torch
from openai import OpenAI
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    LlavaOnevisionForConditionalGeneration,
)

from .base import BaseClassificationModel


class Gpt4oMini(BaseClassificationModel):
    client: OpenAI
    max_tokens: int
    sleep_time: int

    def __init__(self, max_tokens: int = 1024, sleep_time: int = 5) -> None:
        self.client = OpenAI()
        self.max_tokens = max_tokens
        self.sleep_time = sleep_time

    def predict(self, images: list[Image.Image], prompt: str) -> str:
        images = [
            image.convert("RGB") if image.mode != "RGB" else image for image in images
        ]

        img_byte_arrs = []
        for image in images:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="JPEG")
            img_byte_arrs.append(img_byte_arr.getvalue())

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=self.max_tokens,
            messages=[
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
        return response.choices[0].message.content


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
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=load_in_4bit,
        )

    def predict(self, images: list[Image.Image], prompt: str) -> str:
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
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

            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=1024, stop_strings="<|endoftext|>"),
                tokenizer=self.processor.tokenizer,
            )

            # only get generated tokens; decode them to text
            generated_tokens = output[0, inputs["input_ids"].size(1) :]
            generated_text = self.processor.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )

            return json.dumps({"description": generated_text.strip()})


class Ovis9B(BaseClassificationModel):
    model: AutoModelForCausalLM
    text_tokenizer: Any
    visual_tokenizer: Any
    llm: Any

    def __init__(self) -> None:
        # load model
        model = AutoModelForCausalLM.from_pretrained(
            "AIDC-AI/Ovis1.6-Gemma2-9B",
            torch_dtype=torch.bfloat16,
            multimodal_max_length=8192,
            trust_remote_code=True,
        ).cuda()
        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()

        self.model = model
        self.text_tokenizer = text_tokenizer
        self.visual_tokenizer = visual_tokenizer
        self.llm = self.model.get_llm()

    def predict(self, images: list[Image.Image], prompt: str) -> str:
        # enter image path and prompt
        image = images[0] if len(images) == 1 else images[1]
        text = prompt
        query = f"<image>\n{text}"

        # format conversation
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, [image])
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        pixel_values = [
            pixel_values.to(
                dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device
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
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True,
            )
            self.llm._cache = None
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **gen_kwargs,
            )[0]
            output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
            return json.dumps({"description": output.strip()})


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


# NOTE: not good, cannot follow instructions.
# class Llama11BVisionInstruct(BaseClassificationModel):
#     model: MllamaForConditionalGeneration
#     processor: AutoProcessor

#     def __init__(self, load_in_4bit: bool = False, load_in_8bit: bool = False):
#         model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

#         self.model = MllamaForConditionalGeneration.from_pretrained(
#             model_id,
#             torch_dtype=torch.bfloat16,
#             device_map="auto",
#             load_in_4bit=load_in_4bit,
#             load_in_8bit=load_in_8bit,
#         )
#         self.processor = AutoProcessor.from_pretrained(model_id)

#     def predict(self, images: list[Image.Image], prompt: str) -> str:
#         image = images[0] if len(images) == 1 else images[1]
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image"},
#                     {"type": "text", "text": prompt},
#                 ],
#             }
#         ]
#         input_text = self.processor.apply_chat_template(
#             messages, add_generation_prompt=True
#         )
#         inputs = self.processor(
#             image, input_text, add_special_tokens=False, return_tensors="pt"
#         ).to(self.model.device)

#         output = self.model.generate(**inputs, max_new_tokens=1024)
#         return (
#             self.processor.decode(output[0])
#             .split("assistant<|end_header_id|>")[-1]
#             .replace("<|eot_id|>", "")
#             .strip()
#         )


class Pixtral12B(BaseClassificationModel):
    model: LlavaForConditionalGeneration
    processor: AutoProcessor

    def __init__(self) -> None:
        model_id = "mistral-community/pixtral-12b"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            use_flash_attention_2=True,
        )

    def predict(self, images: list[Image.Image], prompt: str) -> str:
        images_query = "".join(["[IMG]"] * len(images))
        PROMPT = f"<s>[INST]{prompt}\n{images_query}[/INST]"

        inputs = self.processor(text=PROMPT, images=images, return_tensors="pt").to(
            "cuda"
        )
        generate_ids = self.model.generate(**inputs, max_new_tokens=500)
        output = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output.split("string: confidence in the prediction from 0 to 1.")[1]


class Llava34B(BaseClassificationModel):
    model: LlavaNextForConditionalGeneration
    processor: LlavaNextProcessor

    def __init__(self) -> None:
        self.processor = LlavaNextProcessor.from_pretrained(
            "llava-hf/llava-v1.6-34b-hf"
        )

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-34b-hf",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_4bit=True,
            use_flash_attention_2=True,
        )
        self.model.to("cuda:0")

    def predict(self, images: list[Image.Image], prompt: str) -> str:
        # Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
        # Each value in "content" has to be a list of dicts with types ("text", "image")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ]
                + [{"type": "image"} for _ in range(len(images))],
            },
        ]
        prompt_input = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs = self.processor(
            images=images, text=prompt_input, return_tensors="pt"
        ).to("cuda:0")

        output = self.model.generate(**inputs, max_new_tokens=512)

        answer = self.processor.decode(output[0], skip_special_tokens=True)
        answer = answer.split("<|im_start|> assistant")[1].strip()
        return answer


class QwenVL72B(BaseClassificationModel):

    def __init__(self) -> None:
        pass

    def predict(self, images: list[Image.Image], prompt: str) -> str:
        pass
