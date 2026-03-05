from PIL import Image

from .base import ActionLabeler
from ..filters.base import BaseFilter
from ..models.base import BaseModel as BaseVLM
from ..preprocessors.base import BasePreprocessor
from ..prompts import Prompt
from ..types import Detection, LabelResult


class MultiViewLabeler(ActionLabeler):
    """Labels each detection with multiple preprocessed views of the image.

    Unlike SingleDetectionLabeler, this class requires multiple preprocessor
    chains -- each chain produces a different view of the image (e.g. cropped,
    masked, with bounding box overlay). All views are sent together in a
    single VLM call per detection.

    Args:
        preprocessors: Required. Must contain at least 2 chains.
            Each chain produces one image view per detection.
    """

    def __init__(
        self,
        model: BaseVLM,
        prompt: Prompt,
        preprocessors: list[list[BasePreprocessor]],
        filters: list[BaseFilter] | None = None,
    ):
        if len(preprocessors) < 2:
            raise ValueError(
                "MultiViewLabeler requires at least 2 preprocessor chains, "
                f"got {len(preprocessors)}"
            )
        super().__init__(
            model=model, prompt=prompt, preprocessors=preprocessors, filters=filters,
        )

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
