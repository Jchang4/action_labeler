from pathlib import Path

import supervision as sv
from PIL import Image

from action_labeler.helpers import load_pickle, xyxy_to_xywh

from .base import BaseActionPrompt

ACTION_FROM_DESCRIPTION_PROMPT_TEMPLATE = """Image Caption: "{description}"

What is the person in the purple bounding box actively doing? \
Classify the image into **one** of the following actions. \
If multiple actions apply, choose the one with the highest priority. \
The actions are sorted by priority.

Actions:

{actions}

Output Format:
- Only respond with "action: ..."
- Do not include any other text
- Do not provide explanations
- If none of the actions apply, respond with "action: none"
"""


class DescriptionActionPrompt(BaseActionPrompt):
    description_file_name: str

    def __init__(
        self,
        description_file_name: str,
        actions: list[str],
        numbered_actions: bool = False,
    ):
        super().__init__(
            ACTION_FROM_DESCRIPTION_PROMPT_TEMPLATE,
            actions,
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
            actions=self.format_actions(),
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
