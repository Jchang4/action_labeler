import json
import re
from typing import get_args, get_origin

from pydantic import BaseModel, TypeAdapter


class Prompt:
    """Prompt with separate system/user messages and optional Pydantic parsing.

    Args:
        system: System prompt text.
        user: User prompt template. Use {placeholders} for dynamic values.
        response_model: Optional Pydantic model or ``list[Model]`` for structured
            output parsing. When a list type is given, the VLM is instructed to
            return a JSON array and parsing produces a list of model instances.
    """

    def __init__(
        self,
        system: str,
        user: str,
        response_model: type[BaseModel] | type[list[BaseModel]] | None = None,
    ):
        self.system = system
        self.user = user
        self.response_model = response_model

        # Detect list[BaseModel] and store the inner model + adapter
        self._is_list = (
            get_origin(response_model) is list
            and len(get_args(response_model)) == 1
            and isinstance(get_args(response_model)[0], type)
            and issubclass(get_args(response_model)[0], BaseModel)
        )
        if self._is_list:
            self._inner_model: type[BaseModel] = get_args(response_model)[0]
            self._type_adapter = TypeAdapter(response_model)
        else:
            self._inner_model = None
            self._type_adapter = None

    def format_system(self) -> str:
        """Return the system prompt, appending JSON format instructions if response_model is set."""
        if self.response_model is None:
            return self.system

        if self._is_list:
            example = self._build_example(self._inner_model)
            example = json.dumps([json.loads(example)], indent=2)
        else:
            example = self._build_example(self.response_model)
        return (
            f"{self.system}\n\n"
            f"Respond with JSON using exactly this format:\n{example}"
        )

    @staticmethod
    def _build_example(model: type[BaseModel]) -> str:
        """Build a JSON example from a Pydantic model's field names and types."""
        placeholders: dict[str, object] = {}
        for name, field_info in model.model_fields.items():
            placeholders[name] = Prompt._example_value(name, field_info.annotation)
        return json.dumps(placeholders, indent=2)

    @staticmethod
    def _example_value(name: str, annotation: type | None) -> object:
        """Return a placeholder value for a single type annotation."""
        if annotation is str:
            return f"<{name}>"
        if annotation is int:
            return 0
        if annotation is float:
            return 0.0
        if annotation is bool:
            return False
        if annotation is list:
            return []

        origin = getattr(annotation, "__origin__", None)
        args = getattr(annotation, "__args__", ())

        if origin is list and args:
            item = Prompt._example_value(name, args[0])
            return [item]

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return {
                n: Prompt._example_value(n, f.annotation)
                for n, f in annotation.model_fields.items()
            }

        return f"<{name}>"

    def format_user(self, **kwargs: object) -> str:
        """Render the user template with the given variables."""
        return self.user.format(**kwargs)

    def parse(self, text: str) -> list[BaseModel] | BaseModel | str:
        """Parse model output into a Pydantic instance (or list) or return raw text.

        Handles common VLM quirks: triple-backtick code blocks and
        JSON embedded in surrounding text.
        """
        if self.response_model is None:
            return text

        json_str = self._extract_json(text)
        if self._is_list:
            return self._type_adapter.validate_json(json_str)
        return self.response_model.model_validate_json(json_str)

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract a JSON object from text that may contain markdown or extra prose."""
        # Try markdown code block first: ```json ... ``` or ``` ... ```
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try to find a top-level JSON object or array
        match = re.search(r"[\{\[].*[\}\]]", text, re.DOTALL)
        if match:
            return match.group(0)

        # Fall back to raw text — let Pydantic raise the validation error
        return text
