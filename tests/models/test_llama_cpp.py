import base64
from unittest.mock import MagicMock, patch

from PIL import Image

from action_labeler.models import LlamaCpp


def _make_image(mode="RGB", size=(64, 64)):
    return Image.new(mode, size, color="red")


def _mock_response(text="test response"):
    mock = MagicMock()
    mock.json.return_value = {
        "choices": [{"message": {"content": text}}],
    }
    return mock


class TestLoadImage:
    def test_converts_rgba_to_rgb(self):
        model = LlamaCpp()
        image = _make_image("RGBA")
        result = model.load_image(image)
        assert result.mode == "RGB"

    def test_rgb_stays_rgb(self):
        model = LlamaCpp()
        image = _make_image("RGB")
        result = model.load_image(image)
        assert result.mode == "RGB"

    def test_converts_grayscale_to_rgb(self):
        model = LlamaCpp()
        image = _make_image("L")
        result = model.load_image(image)
        assert result.mode == "RGB"


class TestPredict:
    @patch("action_labeler.models.llama_cpp.requests.post")
    def test_returns_response_text(self, mock_post):
        mock_post.return_value = _mock_response("sitting")
        model = LlamaCpp()
        result = model.predict("system", "What action?", [_make_image()])
        assert result == "sitting"

    @patch("action_labeler.models.llama_cpp.requests.post")
    def test_sends_to_correct_url(self, mock_post):
        mock_post.return_value = _mock_response()
        model = LlamaCpp(base_url="http://myhost:9000")
        model.predict("system", "prompt", [_make_image()])
        mock_post.assert_called_once()
        assert mock_post.call_args[0][0] == "http://myhost:9000/v1/chat/completions"

    @patch("action_labeler.models.llama_cpp.requests.post")
    def test_strips_trailing_slash_from_base_url(self, mock_post):
        mock_post.return_value = _mock_response()
        model = LlamaCpp(base_url="http://localhost:5000/")
        model.predict("system", "prompt", [_make_image()])
        url = mock_post.call_args[0][0]
        assert "//" not in url.split("://")[1]

    @patch("action_labeler.models.llama_cpp.requests.post")
    def test_sends_system_message(self, mock_post):
        mock_post.return_value = _mock_response()
        model = LlamaCpp()
        model.predict("You are a classifier.", "describe this", [_make_image()])
        payload = mock_post.call_args[1]["json"]
        messages = payload["messages"]
        assert messages[0] == {"role": "system", "content": "You are a classifier."}

    @patch("action_labeler.models.llama_cpp.requests.post")
    def test_payload_contains_prompt_and_images(self, mock_post):
        mock_post.return_value = _mock_response()
        model = LlamaCpp()
        model.predict("system", "describe this", [_make_image(), _make_image()])
        payload = mock_post.call_args[1]["json"]
        content = payload["messages"][1]["content"]
        # 2 images + 1 text
        assert len(content) == 3
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "image_url"
        assert content[2] == {"type": "text", "text": "describe this"}

    @patch("action_labeler.models.llama_cpp.requests.post")
    def test_images_are_base64_encoded(self, mock_post):
        mock_post.return_value = _mock_response()
        model = LlamaCpp()
        model.predict("system", "prompt", [_make_image()])
        content = mock_post.call_args[1]["json"]["messages"][1]["content"]
        url = content[0]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")
        b64_data = url.split(",", 1)[1]
        base64.b64decode(b64_data)  # should not raise

    @patch("action_labeler.models.llama_cpp.requests.post")
    def test_default_sampling_params(self, mock_post):
        mock_post.return_value = _mock_response()
        model = LlamaCpp()
        model.predict("system", "prompt", [_make_image()])
        payload = mock_post.call_args[1]["json"]
        assert payload["max_tokens"] == 1024
        assert "temperature" not in payload
        assert "top_p" not in payload
        assert "top_k" not in payload

    @patch("action_labeler.models.llama_cpp.requests.post")
    def test_custom_sampling_params(self, mock_post):
        mock_post.return_value = _mock_response()
        model = LlamaCpp(temperature=0.5, top_p=0.9, top_k=40, max_tokens=512)
        model.predict("system", "prompt", [_make_image()])
        payload = mock_post.call_args[1]["json"]
        assert payload["temperature"] == 0.5
        assert payload["top_p"] == 0.9
        assert payload["top_k"] == 40
        assert payload["max_tokens"] == 512

    @patch("action_labeler.models.llama_cpp.requests.post")
    def test_calls_raise_for_status(self, mock_post):
        mock_post.return_value = _mock_response()
        model = LlamaCpp()
        model.predict("system", "prompt", [_make_image()])
        mock_post.return_value.raise_for_status.assert_called_once()
