from pathlib import Path

from action_labeler.detections.detection import Detection

from .base import BasePrompt


class TextPrompt(BasePrompt):
    """
    A prompt that returns a text string.

    No classes are used. This is similar to asking an LLM a question.

    Args:
        text_prompt (str): The text prompt to return.
    """

    text_prompt: str

    def __init__(self, text_prompt: str):
        super().__init__(text_prompt, classes=[], numbered_classes=False)
        self.text_prompt = text_prompt

    def prompt(
        self,
        index: int,
        detections: Detection,
        image_path: Path,
    ) -> str:
        return self.text_prompt
