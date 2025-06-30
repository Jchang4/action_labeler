from pathlib import Path

from action_labeler.helpers import Detection, load_pickle

from .base import BasePrompt

DESCRIPTION_ACTION_PROMPT_TEMPLATE = """Image Caption: "{description}"

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


class DescriptionActionPrompt(BasePrompt):
    description_file_name: str

    def __init__(
        self,
        description_file_name: str,
        classes: list[str],
        numbered_classes: bool = False,
    ):
        super().__init__(
            DESCRIPTION_ACTION_PROMPT_TEMPLATE,
            classes,
            numbered_classes=numbered_classes,
        )
        self.description_file_name = description_file_name

    def prompt(
        self,
        index: int,
        detections: Detection,
        image_path: Path,
    ) -> str:
        return self.template.format(
            description=self.get_description_text(
                index,
                detections,
                image_path,
            ),
            actions=self.format_classes(),
        )

    def get_description_text(
        self,
        index: int,
        detections: Detection,
        image_path: Path,
    ) -> str:
        descriptions = load_pickle(image_path.parent.parent, self.description_file_name)
        xywh = detections.xywhn[index]
        box = " ".join(map(str, xywh))

        if str(image_path) not in descriptions:
            raise ValueError(f"No description found for {image_path}")
        elif box not in descriptions[str(image_path)]:
            raise ValueError(f"No description found for {box} in {image_path}")
        return descriptions[str(image_path)][box].strip()
