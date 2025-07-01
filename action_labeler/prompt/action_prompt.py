from pathlib import Path

from action_labeler.detections.detection import Detection
from action_labeler.helpers import load_pickle

from .base import BasePrompt

ACTION_PROMPT_TEMPLATE = """Classify the action of the person in the bounding box. \
Some examples of classifications are: cooking, cleaning_dishes, using_phone, using_computer, standing, walking, etc. \
We want to capture what the person is doing and what objects they are interacting with.

Output Format:
- Only respond with "action: ..."
- Do not include any other text
- Do not provide explanations
- If none of the actions apply, respond with "action: none"
- If multiple actions apply, choose the most specific action.
"""


class ActionPrompt(BasePrompt):
    def prompt(
        self,
        index: int,
        detections: Detection,
        image_path: Path,
    ) -> str:
        return f"{self.template}\n\nActions:\n\n{self.format_classes()}"
