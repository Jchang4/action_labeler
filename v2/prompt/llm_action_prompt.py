from pathlib import Path

import supervision as sv
from PIL import Image

from action_labeler.helpers import load_pickle, xyxy_to_xywh

from .base import BaseActionPrompt

LLM_ACTION_PROMPT_TEMPLATE = """Image Caption: "{description}"

Classify the action of the person in the bounding box. \
Some examples of classifications are: cooking, cleaning_dishes, using_phone, using_computer, standing, walking, etc. \
We want to capture what the person is doing and what objects they are interacting with.

Output Format:
- Only respond with "action: ..."
- Do not include any other text
- Do not provide explanations
- If none of the actions apply, respond with "action: none"
- If multiple actions apply, choose the most specific action.
"""


class LLMActionPrompt(BaseActionPrompt):
    description_file_name: str

    def __init__(
        self,
        description_file_name: str,
        numbered_actions: bool = False,
    ):
        super().__init__(
            LLM_ACTION_PROMPT_TEMPLATE,
            [],
            numbered_actions=numbered_actions,
        )
        self.description_file_name = description_file_name

    def prompt(
        self,
        image: Image.Image,
        index: int,
        detections: sv.Detections,
        image_path: Path,
    ) -> str:
        return self.template.format(
            description=self.get_description_text(
                image,
                index,
                detections,
                image_path,
            ),
        )

    def get_description_text(
        self,
        image: Image.Image,
        index: int,
        detections: sv.Detections,
        image_path: Path,
    ) -> str:
        descriptions = load_pickle(image_path.parent.parent, self.description_file_name)
        xywh = xyxy_to_xywh(Image.open(image_path), detections.xyxy[index])
        box = " ".join(map(str, xywh))

        if str(image_path) not in descriptions:
            raise ValueError(f"No description found for {image_path}")
        elif box not in descriptions[str(image_path)]:
            raise ValueError(f"No description found for {box} in {image_path}")
        return descriptions[str(image_path)][box].strip()
