from abc import ABC
from pathlib import Path

from action_labeler.detections.detection import Detection


class BasePrompt(ABC):
    """
    Base class for prompts that are used to label actions in images.

    Args:
        template: The template for the prompt.
        classes: The classes for the prompt.
        numbered_classes: Whether to number the classes in the prompt.
    """

    template: str
    classes: list[str]
    numbered_classes: bool = False

    def __init__(
        self, template: str, classes: list[str], numbered_classes: bool = False
    ):
        self.template = template
        self.classes = classes
        self.numbered_classes = numbered_classes

    def format_classes(self) -> str:
        """
        Format the classes for the prompt.

        If numbered_classes is True, the classes will be numbered.
        For example:
        ```
        1. "class1"
        2. "class2"
        3. "class3"
        ```

        If numbered_classes is False, the classes will be unnumbered.
        For example:
        ```
        - "class1"
        - "class2"
        - "class3"
        ```

        Returns:
            str: The formatted classes.
        """
        if self.numbered_classes:
            return "\n".join(
                f'{i+1}. "{class_name}"' for i, class_name in enumerate(self.classes)
            )
        return "\n".join(f'- "{class_name}"' for class_name in self.classes)

    def prompt(
        self,
        _index: int,  # The index of the detection
        _detections: Detection,  # The detections for the image
        _image_path: Path,  # The path to the image
    ) -> str:
        """
        Generate the prompt for the action labeling.

        Args:
            index: The index of the detection.
            detections: The detections for the image.
            image_path: The path to the image.
        """
        return self.template.format(classes=self.format_classes())
