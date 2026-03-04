from unittest.mock import MagicMock

import pytest
from PIL import Image

from action_labeler.labeler.multi_view import MultiViewLabeler
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


def _make_prompt_mock():
    prompt = MagicMock()
    prompt.format_system.return_value = "system"
    prompt.format_user.return_value = "user"
    prompt.parse.side_effect = lambda text: text
    return prompt


def _make_model_mock():
    model = MagicMock()
    model.predict.return_value = "response"
    return model


def _make_preprocessor_chain():
    """Create a single preprocessor mock that returns a new image."""
    p = MagicMock()
    p.process.side_effect = lambda img, dets: img.copy()
    return [p]


class TestMultiViewLabelerInit:
    def test_raises_with_zero_chains(self):
        with pytest.raises(ValueError, match="at least 2 preprocessor chains"):
            MultiViewLabeler(
                model=_make_model_mock(),
                prompt=_make_prompt_mock(),
                preprocessors=[],
            )

    def test_raises_with_one_chain(self):
        with pytest.raises(ValueError, match="at least 2 preprocessor chains"):
            MultiViewLabeler(
                model=_make_model_mock(),
                prompt=_make_prompt_mock(),
                preprocessors=[_make_preprocessor_chain()],
            )

    def test_accepts_two_chains(self):
        labeler = MultiViewLabeler(
            model=_make_model_mock(),
            prompt=_make_prompt_mock(),
            preprocessors=[_make_preprocessor_chain(), _make_preprocessor_chain()],
        )
        assert len(labeler.preprocessors) == 2


class TestMultiViewLabelerLabel:
    def test_predict_called_once_per_detection(self):
        model = _make_model_mock()
        labeler = MultiViewLabeler(
            model=model,
            prompt=_make_prompt_mock(),
            preprocessors=[_make_preprocessor_chain(), _make_preprocessor_chain()],
        )
        image = Image.new("RGB", (64, 64), color="red")
        detections = [_make_detection(), _make_detection()]

        labeler.label(image, detections)

        assert model.predict.call_count == 2

    def test_predict_receives_multiple_images_per_call(self):
        model = _make_model_mock()
        chains = [
            _make_preprocessor_chain(),
            _make_preprocessor_chain(),
            _make_preprocessor_chain(),
        ]
        labeler = MultiViewLabeler(
            model=model,
            prompt=_make_prompt_mock(),
            preprocessors=chains,
        )
        image = Image.new("RGB", (64, 64), color="red")
        detections = [_make_detection()]

        labeler.label(image, detections)

        call_args = model.predict.call_args_list[0]
        images_arg = call_args[0][2]  # third positional arg
        assert len(images_arg) == 3
        assert all(isinstance(img, Image.Image) for img in images_arg)

    def test_responses_collected_in_order(self):
        model = MagicMock()
        model.predict.side_effect = ["first", "second", "third"]
        labeler = MultiViewLabeler(
            model=model,
            prompt=_make_prompt_mock(),
            preprocessors=[_make_preprocessor_chain(), _make_preprocessor_chain()],
        )
        image = Image.new("RGB", (64, 64), color="red")
        detections = [_make_detection(), _make_detection(), _make_detection()]

        responses = labeler.label(image, detections)

        assert responses == ["first", "second", "third"]

    def test_two_detections_three_chains(self):
        model = _make_model_mock()
        chains = [
            _make_preprocessor_chain(),
            _make_preprocessor_chain(),
            _make_preprocessor_chain(),
        ]
        labeler = MultiViewLabeler(
            model=model,
            prompt=_make_prompt_mock(),
            preprocessors=chains,
        )
        image = Image.new("RGB", (64, 64), color="red")
        detections = [_make_detection(), _make_detection()]

        responses = labeler.label(image, detections)

        # 2 predict calls (one per detection)
        assert model.predict.call_count == 2
        # Each call receives 3 images (one per chain)
        for call in model.predict.call_args_list:
            images_arg = call[0][2]
            assert len(images_arg) == 3
        # 2 responses returned
        assert len(responses) == 2

    def test_prompt_format_called(self):
        prompt = _make_prompt_mock()
        labeler = MultiViewLabeler(
            model=_make_model_mock(),
            prompt=prompt,
            preprocessors=[_make_preprocessor_chain(), _make_preprocessor_chain()],
        )
        image = Image.new("RGB", (64, 64), color="red")
        detections = [_make_detection()]

        labeler.label(image, detections)

        prompt.format_system.assert_called()
        prompt.format_user.assert_called()
        prompt.parse.assert_called_once_with("response")
