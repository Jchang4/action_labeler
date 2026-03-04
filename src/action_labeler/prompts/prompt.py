import json
import re

from pydantic import BaseModel


class Prompt:
    """Prompt with separate system/user messages and optional Pydantic parsing.

    Args:
        system: System prompt text.
        user: User prompt template. Use {placeholders} for dynamic values.
        response_model: Optional Pydantic model for structured output parsing.
    """

    def __init__(
        self,
        system: str,
        user: str,
        response_model: type[BaseModel] | None = None,
    ):
        self.system = system
        self.user = user
        self.response_model = response_model

    def format_system(self) -> str:
        """Return the system prompt, appending JSON format instructions if response_model is set."""
        if self.response_model is None:
            return self.system

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
            annotation = field_info.annotation
            if annotation is str:
                placeholders[name] = f"<{name}>"
            elif annotation is int:
                placeholders[name] = 0
            elif annotation is float:
                placeholders[name] = 0.0
            elif annotation is bool:
                placeholders[name] = False
            elif annotation is list:
                placeholders[name] = []
            else:
                placeholders[name] = f"<{name}>"
        return json.dumps(placeholders, indent=2)

    def format_user(self, **kwargs: object) -> str:
        """Render the user template with the given variables."""
        return self.user.format(**kwargs)

    def parse(self, text: str) -> BaseModel | str:
        """Parse model output into a Pydantic instance or return raw text.

        Handles common VLM quirks: triple-backtick code blocks and
        JSON embedded in surrounding text.
        """
        if self.response_model is None:
            return text

        json_str = self._extract_json(text)
        return self.response_model.model_validate_json(json_str)

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract a JSON object from text that may contain markdown or extra prose."""
        # Try markdown code block first: ```json ... ``` or ``` ... ```
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try to find a top-level JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)

        # Fall back to raw text — let Pydantic raise the validation error
        return text
