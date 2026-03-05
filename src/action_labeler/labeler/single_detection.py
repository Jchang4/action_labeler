from PIL import Image

from .base import ActionLabeler
from ..types import Detection, LabelResult


class SingleDetectionLabeler(ActionLabeler):
    """Labels each detection individually with a separate VLM call."""

    def label(
        self, image: Image.Image, detections: list[Detection]
    ) -> list[LabelResult]:
        results = []
        system = self.prompt.format_system()
        user = self.prompt.format_user()
        for det in detections:
            images = self._apply_preprocessors(image, [det])
            text = self.model.predict(system, user, images)
            results.append(self._make_result(self.prompt.parse(text)))
        return results
