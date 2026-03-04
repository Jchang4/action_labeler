from unittest.mock import MagicMock, patch

from PIL import Image
from pydantic import BaseModel

from action_labeler.labeler.all_at_once import AllAtOnceLabeler
from action_labeler.types import Detection


def _make_detection(**kwargs):
    defaults = dict(
        class_id=0,
        x_center=0.5,
        y_center=0.5,
        width=0.3,
        height=0.4,
        image_width=64,
        image_height=64,
    )
    defaults.update(kwargs)
    return Detection(**defaults)


def _make_image():
    return Image.new("RGB", (64, 64), color="red")


class ActionItem(BaseModel):
    label: str


class BatchResponse(BaseModel):
    actions: list[ActionItem]


class TestAllAtOnceLabeler:
    def _build_labeler(self, *, parsed_response, response_field="actions"):
        model = MagicMock()
        model.predict.return_value = "raw model text"

        prompt = MagicMock()
        prompt.format_system.return_value = "system msg"
        prompt.format_user.return_value = "user msg"
        prompt.parse.return_value = parsed_response

        labeler = AllAtOnceLabeler(
            model=model,
            prompt=prompt,
            response_field=response_field,
        )
        return labeler, model, prompt

    def test_calls_apply_preprocessors(self):
        """label() delegates to _apply_preprocessors with image and all detections."""
        detections = [_make_detection(), _make_detection(x_center=0.2)]
        parsed = BatchResponse(
            actions=[ActionItem(label="walk"), ActionItem(label="sit")]
        )
        labeler, model, prompt = self._build_labeler(parsed_response=parsed)
        image = _make_image()

        with patch.object(
            labeler, "_apply_preprocessors", return_value=[image]
        ) as mock_pp:
            labeler.label(image, detections)
            mock_pp.assert_called_once_with(image, detections)

    def test_model_predict_called_once_with_preprocessed_images(self):
        """model.predict is called exactly once with all preprocessed images."""
        detections = [_make_detection(), _make_detection()]
        parsed = BatchResponse(
            actions=[ActionItem(label="a"), ActionItem(label="b")]
        )
        labeler, model, prompt = self._build_labeler(parsed_response=parsed)
        image = _make_image()

        preprocessed = [_make_image(), _make_image()]
        with patch.object(
            labeler, "_apply_preprocessors", return_value=preprocessed
        ):
            labeler.label(image, detections)

        model.predict.assert_called_once_with("system msg", "user msg", preprocessed)

    def test_extracts_per_detection_items_via_response_field(self):
        """Parsed response items are extracted from the response_field attribute."""
        detections = [_make_detection(), _make_detection(), _make_detection()]
        items = [
            ActionItem(label="walk"),
            ActionItem(label="sit"),
            ActionItem(label="run"),
        ]
        parsed = BatchResponse(actions=items)
        labeler, model, prompt = self._build_labeler(parsed_response=parsed)
        image = _make_image()

        with patch.object(
            labeler, "_apply_preprocessors", return_value=[image]
        ):
            result = labeler.label(image, detections)

        assert result == items
        assert len(result) == len(detections)

    def test_custom_response_field(self):
        """A custom response_field name is used to extract items."""

        class CustomResponse(BaseModel):
            results: list[ActionItem]

        detections = [_make_detection()]
        items = [ActionItem(label="jump")]
        parsed = CustomResponse(results=items)
        labeler, model, prompt = self._build_labeler(
            parsed_response=parsed, response_field="results"
        )
        image = _make_image()

        with patch.object(
            labeler, "_apply_preprocessors", return_value=[image]
        ):
            result = labeler.label(image, detections)

        assert result == items

    def test_raw_string_fallback(self):
        """When prompt.parse returns a str, it is replicated for each detection."""
        detections = [_make_detection(), _make_detection()]
        labeler, model, prompt = self._build_labeler(
            parsed_response="standing"
        )
        image = _make_image()

        with patch.object(
            labeler, "_apply_preprocessors", return_value=[image]
        ):
            result = labeler.label(image, detections)

        assert result == ["standing", "standing"]
        assert len(result) == len(detections)

    def test_prompt_format_and_parse_called(self):
        """format_system, format_user, and parse are all invoked correctly."""
        detections = [_make_detection()]
        parsed = BatchResponse(actions=[ActionItem(label="x")])
        labeler, model, prompt = self._build_labeler(parsed_response=parsed)
        image = _make_image()

        with patch.object(
            labeler, "_apply_preprocessors", return_value=[image]
        ):
            labeler.label(image, detections)

        prompt.format_system.assert_called_once_with()
        prompt.format_user.assert_called_once_with()
        prompt.parse.assert_called_once_with("raw model text")
