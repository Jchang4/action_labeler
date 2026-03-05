from PIL import Image

from ..types import Detection, LabelResult
from .base import ActionLabeler


class AllAtOnceLabeler(ActionLabeler):
    """Send all detections to the VLM in a single call.

    The prompt's response_model should be ``list[ActionResponse]`` so that
    the parsed result contains one ActionResponse per detection.
    """

    def label(
        self, image: Image.Image, detections: list[Detection]
    ) -> list[LabelResult]:
        images = self._apply_preprocessors(image, detections)
        system = self.prompt.format_system()
        user = self.prompt.format_user()
        text = self.model.predict(system, user, images)
        parsed = self.prompt.parse(text)

        if isinstance(parsed, str):
            return [self._make_result(parsed)] * len(detections)

        if isinstance(parsed, list):
            return [self._make_result(r) for r in parsed]

        return [self._make_result(parsed)]
