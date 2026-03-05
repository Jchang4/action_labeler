import json

import pytest
from pydantic import BaseModel, ValidationError

from action_labeler.prompts import Prompt


class ActionLabel(BaseModel):
    action: str
    confidence: float


class TestFormatSystem:
    def test_returns_system_as_is_without_response_model(self):
        prompt = Prompt(system="You are a classifier.", user="classify")
        assert prompt.format_system() == "You are a classifier."

    def test_appends_example_with_response_model(self):
        prompt = Prompt(
            system="You are a classifier.",
            user="classify",
            response_model=ActionLabel,
        )
        result = prompt.format_system()
        assert result.startswith("You are a classifier.\n\n")
        assert "Respond with JSON using exactly this format:" in result
        assert '"action"' in result
        assert '"confidence"' in result

    def test_example_is_valid_json(self):
        prompt = Prompt(system="sys", user="usr", response_model=ActionLabel)
        result = prompt.format_system()
        example_str = result.split("Respond with JSON using exactly this format:\n")[1]
        parsed = json.loads(example_str)
        assert "action" in parsed
        assert "confidence" in parsed

    def test_example_uses_type_placeholders(self):
        prompt = Prompt(system="sys", user="usr", response_model=ActionLabel)
        result = prompt.format_system()
        example_str = result.split("Respond with JSON using exactly this format:\n")[1]
        parsed = json.loads(example_str)
        assert parsed["action"] == "<action>"
        assert parsed["confidence"] == 0.0


class TestFormatUser:
    def test_renders_template(self):
        prompt = Prompt(system="sys", user="Box {box_id} action?")
        assert prompt.format_user(box_id=3) == "Box 3 action?"

    def test_multiple_placeholders(self):
        prompt = Prompt(system="sys", user="{a} and {b}")
        assert prompt.format_user(a="hello", b="world") == "hello and world"

    def test_no_placeholders(self):
        prompt = Prompt(system="sys", user="What action?")
        assert prompt.format_user() == "What action?"

    def test_missing_kwarg_raises(self):
        prompt = Prompt(system="sys", user="Box {box_id}")
        with pytest.raises(KeyError):
            prompt.format_user()


class TestParse:
    def test_returns_raw_string_without_response_model(self):
        prompt = Prompt(system="sys", user="usr")
        assert prompt.parse("some text") == "some text"

    def test_parses_clean_json(self):
        prompt = Prompt(system="sys", user="usr", response_model=ActionLabel)
        result = prompt.parse('{"action": "sitting", "confidence": 0.95}')
        assert isinstance(result, ActionLabel)
        assert result.action == "sitting"
        assert result.confidence == 0.95

    def test_extracts_json_from_markdown_code_block(self):
        prompt = Prompt(system="sys", user="usr", response_model=ActionLabel)
        text = '```json\n{"action": "running", "confidence": 0.8}\n```'
        result = prompt.parse(text)
        assert result.action == "running"

    def test_extracts_json_from_plain_code_block(self):
        prompt = Prompt(system="sys", user="usr", response_model=ActionLabel)
        text = '```\n{"action": "running", "confidence": 0.8}\n```'
        result = prompt.parse(text)
        assert result.action == "running"

    def test_extracts_json_from_surrounding_text(self):
        prompt = Prompt(system="sys", user="usr", response_model=ActionLabel)
        text = 'Here is the result: {"action": "walking", "confidence": 0.7} hope that helps!'
        result = prompt.parse(text)
        assert result.action == "walking"

    def test_raises_on_invalid_json(self):
        prompt = Prompt(system="sys", user="usr", response_model=ActionLabel)
        with pytest.raises(ValidationError):
            prompt.parse("not json at all")

    def test_raises_on_wrong_schema(self):
        prompt = Prompt(system="sys", user="usr", response_model=ActionLabel)
        with pytest.raises(ValidationError):
            prompt.parse('{"wrong_field": "value"}')


class TestListResponseModel:
    """Tests for list[BaseModel] as response_model."""

    def test_format_system_generates_array_example(self):
        prompt = Prompt(system="Classify.", user="usr", response_model=list[ActionLabel])
        result = prompt.format_system()
        assert "Respond with JSON using exactly this format:" in result
        example_str = result.split("Respond with JSON using exactly this format:\n")[1]
        parsed = json.loads(example_str)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert "action" in parsed[0]
        assert "confidence" in parsed[0]

    def test_parse_json_array(self):
        prompt = Prompt(system="sys", user="usr", response_model=list[ActionLabel])
        text = '[{"action": "sitting", "confidence": 0.9}, {"action": "standing", "confidence": 0.8}]'
        result = prompt.parse(text)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].action == "sitting"
        assert result[1].action == "standing"

    def test_parse_array_from_markdown_code_block(self):
        prompt = Prompt(system="sys", user="usr", response_model=list[ActionLabel])
        text = '```json\n[{"action": "running", "confidence": 0.7}]\n```'
        result = prompt.parse(text)
        assert isinstance(result, list)
        assert result[0].action == "running"

    def test_parse_array_from_surrounding_text(self):
        prompt = Prompt(system="sys", user="usr", response_model=list[ActionLabel])
        text = 'Here: [{"action": "walking", "confidence": 0.6}] done.'
        result = prompt.parse(text)
        assert isinstance(result, list)
        assert result[0].action == "walking"

    def test_parse_raises_on_invalid_items(self):
        prompt = Prompt(system="sys", user="usr", response_model=list[ActionLabel])
        with pytest.raises(ValidationError):
            prompt.parse('[{"wrong": "schema"}]')

    def test_parse_empty_array(self):
        prompt = Prompt(system="sys", user="usr", response_model=list[ActionLabel])
        result = prompt.parse("[]")
        assert result == []
