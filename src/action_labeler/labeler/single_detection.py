from PIL import Image
from pydantic import BaseModel

from .base import ActionLabeler
from ..types import Detection


class SingleDetectionLabeler(ActionLabeler):
    """Labels each detection individually with a separate VLM call."""

    def label(
        self, image: Image.Image, detections: list[Detection]
    ) -> list[BaseModel | str]:
        responses = []
        system = self.prompt.format_system()
        user = self.prompt.format_user()
        for det in detections:
            images = self._apply_preprocessors(image, [det])
            text = self.model.predict(system, user, images)
            responses.append(self.prompt.parse(text))
        return responses
