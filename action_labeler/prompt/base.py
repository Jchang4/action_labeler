from abc import ABC, abstractmethod
from pathlib import Path

from action_labeler.detections.detection import Detection


class BasePrompt(ABC):
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
        if self.numbered_classes:
            return "\n".join(
                f'{i+1}. "{class_name}"' for i, class_name in enumerate(self.classes)
            )
        return "\n".join(f'- "{class_name}"' for class_name in self.classes)

    @abstractmethod
    def prompt(
        self,
        index: int,  # The index of the detection
        detections: Detection,  # The detections for the image
        image_path: Path,  # The path to the image
    ) -> str: ...
