import base64
import io
import json
from time import sleep
from typing import Any

import torch
from openai import OpenAI
from PIL import Image, ImageStat
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
    Pipeline,
    Qwen2VLForConditionalGeneration,
    pipeline,
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
                images=[self.get_image(image) for image in images],
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

    @staticmethod
    def get_image(image: Image.Image) -> Image.Image:
        image = image.copy()
        gray_image = image.convert("L")  # Convert to grayscale

        # Calculate the average brightness
        stat = ImageStat.Stat(gray_image)
        average_brightness = stat.mean[0]  # Get the average value

        # Define background color based on brightness (threshold can be adjusted)
        bg_color = (0, 0, 0) if average_brightness > 127 else (255, 255, 255)

        # Create a new image with the same size as the original, filled with the background color
        new_image = Image.new("RGB", image.size, bg_color)

        # Paste the original image on top of the background (use image as a mask if needed)
        new_image.paste(image, (0, 0), image if image.mode == "RGBA" else None)

        return new_image


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


class Qwen2VL7B(BaseClassificationModel):
    model: Qwen2VLForConditionalGeneration
    processor: AutoProcessor

    def __init__(self) -> None:
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct-AWQ", torch_dtype="auto", device_map="auto"
        )

        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct-AWQ",
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    def predict(self, images: list[Image.Image], prompt: str) -> str:
        messages = [{"type": "image"} for _ in images]
        messages.append({"type": "text", "text": prompt})
        conversation = [
            {
                "role": "user",
                "content": messages,
            }
        ]

        # Preprocess the inputs
        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

        inputs = self.processor(
            text=[text_prompt], images=[images], padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        output_ids = self.model.generate(**inputs, max_new_tokens=2048)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        return json.dumps({"description": output_text[0]})


class HuggingFaceImageClassificationModel(BaseClassificationModel):
    pipeline: Pipeline

    def __init__(self) -> None:
        HF_MODEL_DIR = "./datasets/human_action_classification_finetuned"

        self.pipeline = pipeline(
            "image-classification", model=HF_MODEL_DIR, device="cuda"
        )

    def predict(self, images: list[Image.Image], prompt: str) -> str:
        # Pipeline returns a list of lists of dicts with keys "label" and "score"
        results = self.pipeline(images)
        # Average the scores
        combined_results = {}
        for result in results:
            for data in result:
                label = data["label"]
                score = data["score"]
                combined_results[label] = combined_results.get(label, 0) + score
        # Best Label and Score
        best_label = max(combined_results, key=combined_results.get)
        best_score = combined_results[best_label]

        return json.dumps(
            {
                "action": best_label,
                "confidence": best_score,
                **combined_results,
            }
        )
