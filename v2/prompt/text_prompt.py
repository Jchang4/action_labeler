from pathlib import Path

import supervision as sv
from PIL import Image

from action_labeler.v2.base import IPrompt


class TextPrompt(IPrompt):
    text_prompt: str

    def __init__(self, text_prompt: str):
        self.text_prompt = text_prompt

    def prompt(
        self,
        image: Image.Image,
        index: int,
        detections: sv.Detections,
        image_path: Path,
    ) -> str:
        return self.text_prompt


"""
Example usage
prompt = Prompt(
    text_prompt="Describe the image",
    actions=["action1", "action2", "action3"],
)

prompt.prompt(
    Image.open(
        "datasets/human/calling/images/0abdba8a3d67f2e5b66e068626f6117dc488d1ed8c259b54890368b54c1114dc.jpg"
    ),
    0,
    sv.Detections.empty(),
)
"""
