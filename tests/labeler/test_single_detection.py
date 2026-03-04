from unittest.mock import MagicMock, call, patch

from PIL import Image
from pydantic import BaseModel

from action_labeler.labeler.single_detection import SingleDetectionLabeler
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


class TestSingleDetectionLabeler:
    def _build_labeler(self, *, predict_side_effect=None):
        model = MagicMock()
        if predict_side_effect is not None:
            model.predict.side_effect = predict_side_effect
        else:
            model.predict.return_value = "raw model text"

        prompt = MagicMock()
        prompt.format_system.return_value = "system msg"
        prompt.format_user.return_value = "user msg"
        prompt.parse.return_value = "parsed"

        labeler = SingleDetectionLabeler(
            model=model,
            prompt=prompt,
        )
        return labeler, model, prompt

    def test_model_predict_called_once_per_detection(self):
        """model.predict is called once for each detection."""
        detections = [_make_detection(), _make_detection(), _make_detection()]
        labeler, model, prompt = self._build_labeler()
        image = _make_image()

        with patch.object(
            labeler, "_apply_preprocessors", return_value=[image]
        ):
            labeler.label(image, detections)

        assert model.predict.call_count == 3

    def test_apply_preprocessors_called_with_single_detection(self):
        """_apply_preprocessors is called with [det] for each detection."""
        d1 = _make_detection(x_center=0.2)
        d2 = _make_detection(x_center=0.8)
        detections = [d1, d2]
        labeler, model, prompt = self._build_labeler()
        image = _make_image()

        with patch.object(
            labeler, "_apply_preprocessors", return_value=[image]
        ) as mock_pp:
            labeler.label(image, detections)

            assert mock_pp.call_count == 2
            mock_pp.assert_any_call(image, [d1])
            mock_pp.assert_any_call(image, [d2])

    def test_responses_collected_in_order(self):
        """Parsed responses are returned in detection order."""
        detections = [_make_detection(), _make_detection(), _make_detection()]
        labeler, model, prompt = self._build_labeler(
            predict_side_effect=["text_a", "text_b", "text_c"]
        )
        prompt = labeler.prompt
        items = [
            ActionItem(label="walk"),
            ActionItem(label="sit"),
            ActionItem(label="run"),
        ]
        prompt.parse.side_effect = items
        image = _make_image()

        with patch.object(
            labeler, "_apply_preprocessors", return_value=[image]
        ):
            result = labeler.label(image, detections)

        assert result == items
        assert len(result) == 3

    def test_prompt_format_and_parse_called(self):
        """format_system/format_user called once; parse called per detection."""
        detections = [_make_detection(), _make_detection()]
        labeler, model, prompt = self._build_labeler()
        image = _make_image()

        with patch.object(
            labeler, "_apply_preprocessors", return_value=[image]
        ):
            labeler.label(image, detections)

        prompt.format_system.assert_called_once_with()
        prompt.format_user.assert_called_once_with()
        assert prompt.parse.call_count == 2
        prompt.parse.assert_has_calls([call("raw model text"), call("raw model text")])

    def test_predict_receives_preprocessed_images(self):
        """Each model.predict call receives images from _apply_preprocessors."""
        detections = [_make_detection(), _make_detection()]
        labeler, model, prompt = self._build_labeler()
        image = _make_image()

        img_a = _make_image()
        img_b = _make_image()
        preprocessed_batches = [[img_a], [img_b]]

        with patch.object(
            labeler, "_apply_preprocessors", side_effect=preprocessed_batches
        ):
            labeler.label(image, detections)

        model.predict.assert_any_call("system msg", "user msg", [img_a])
        model.predict.assert_any_call("system msg", "user msg", [img_b])

    def test_with_real_preprocessors(self):
        """Preprocessors are applied per-detection via the real _apply_preprocessors."""
        preprocessor = MagicMock()
        processed_img = _make_image()
        preprocessor.process.return_value = processed_img

        labeler, model, prompt = self._build_labeler()
        labeler.preprocessors = [[preprocessor]]
        image = _make_image()

        d1 = _make_detection(x_center=0.2)
        d2 = _make_detection(x_center=0.8)

        labeler.label(image, [d1, d2])

        assert preprocessor.process.call_count == 2
        # Each call gets a single-element detection list
        for c in preprocessor.process.call_args_list:
            det_arg = c[0][1]
            assert len(det_arg) == 1

    def test_empty_detections_returns_empty_list(self):
        """No detections means no calls and an empty result."""
        labeler, model, prompt = self._build_labeler()
        image = _make_image()

        result = labeler.label(image, [])

        assert result == []
        model.predict.assert_not_called()
