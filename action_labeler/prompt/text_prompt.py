from pathlib import Path

from action_labeler.detections.detection import Detection

from .base import BasePrompt


class TextPrompt(BasePrompt):
    """
    A prompt that returns a text string.
    This is usefulf or static prompts that do not have classes.

    NOTE: No classes are used in this prompt. This is similar to asking an LLM a question.

    Args:
        prompt (str): The text prompt to return.
    """

    def prompt(
        self,
        _index: int,
        _detections: Detection,
        _image_path: Path,
    ) -> str:
        """
        Return the template.
        """
        return self.template
