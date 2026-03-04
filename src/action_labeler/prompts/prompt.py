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
        """Return the system prompt, appending JSON schema instructions if response_model is set."""
        if self.response_model is None:
            return self.system

        schema = json.dumps(self.response_model.model_json_schema())
        return (
            f"{self.system}\n\n"
            f"Respond with JSON matching this schema:\n{schema}"
        )

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
