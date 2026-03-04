from PIL import Image
from pydantic import BaseModel

from ..types import Detection
from .base import ActionLabeler


class AllAtOnceLabeler(ActionLabeler):
    """Send all detections to the VLM in a single call.

    The prompt's response_model should contain a list field holding one
    item per detection. After parsing, the labeler extracts individual
    items from that field using ``response_field``.

    Args:
        response_field: Name of the list attribute on the parsed response
            model that holds per-detection results.
        **kwargs: Forwarded to :class:`ActionLabeler`.
    """

    def __init__(self, *, response_field: str = "actions", **kwargs):
        super().__init__(**kwargs)
        self.response_field = response_field

    def label(
        self, image: Image.Image, detections: list[Detection]
    ) -> list[BaseModel | str]:
        images = self._apply_preprocessors(image, detections)
        system = self.prompt.format_system()
        user = self.prompt.format_user()
        text = self.model.predict(system, user, images)
        parsed = self.prompt.parse(text)

        if isinstance(parsed, str):
            return [parsed] * len(detections)

        return list(getattr(parsed, self.response_field))
