from pathlib import Path

from action_labeler.helpers import Detection, load_pickle

from .base import BasePrompt

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


class DescriptionOnlyPrompt(BasePrompt):
    description_file_name: str

    def __init__(
        self,
        description_file_name: str,
        numbered_classes: bool = False,
    ):
        super().__init__(
            LLM_ACTION_PROMPT_TEMPLATE,
            classes=[],
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
