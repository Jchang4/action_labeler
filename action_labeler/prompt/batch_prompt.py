"""Base class for batch prompts that label multiple detections simultaneously.

Batch prompts define:
1. How to format prompts for multiple detections
2. The expected output format (JSON schema)
3. How to parse responses back to individual detection labels
"""

from abc import abstractmethod
from pathlib import Path
from typing import Any

from action_labeler.detections.detection import Detection
from action_labeler.prompt.base import BasePrompt


class BatchPrompt(BasePrompt):
    """Base class for prompts that process multiple detections at once.

    Batch prompts need to:
    - Define the detection identifier key (e.g., "person", "id", "detection")
    - Define the label key (e.g., "action", "label", "classification")
    - Parse the model's batch response into individual detection labels

    Args:
        system_prompt: System prompt for the model
        template: Template for the prompt
        classes: List of valid class labels
        numbered_classes: Whether to number classes in the prompt
        detection_id_key: Key used to identify detections in output (e.g., "person")
        label_key: Key used for the label/action in output (e.g., "action")
    """

    detection_id_key: str = "person"
    label_key: str = "action"

    def __init__(
        self,
        system_prompt: str,
        template: str,
        classes: list[str],
        numbered_classes: bool = False,
        detection_id_key: str = "person",
        label_key: str = "action",
    ):
        super().__init__(system_prompt, template, classes, numbered_classes)
        self.detection_id_key = detection_id_key
        self.label_key = label_key

    def prompt(
        self,
        index: int,
        detections: Detection,
        image_path: Path,
    ) -> str:
        """Generate batch prompt for all detections.

        For batch mode, index is typically 0 or None since we process all detections.
        Override this to customize the prompt format.

        Args:
            index: Detection index (ignored in batch mode)
            detections: All detections in the image
            image_path: Path to the image

        Returns:
            Formatted prompt string
        """
        num_detections = len(detections.xyxy)

        prompt = self.template.format(
            classes=self.format_classes(),
            num_detections=num_detections,
        )

        return prompt.strip()

    @abstractmethod
    def parse_batch_response(
        self, raw_response: str, detections: Detection
    ) -> dict[int, str]:
        """Parse the model's batch response into individual detection labels.

        This method is responsible for:
        1. Parsing the model's output format (JSON, text, etc.)
        2. Extracting the detection ID and label for each detection
        3. Mapping detection IDs back to detection indices

        Args:
            raw_response: Raw string response from the model
            detections: All detections in the image (for validation)

        Returns:
            Dictionary mapping detection index -> label
            Example: {0: "walking", 1: "sitting", 2: "standing"}

        Raises:
            ValueError: If response cannot be parsed or is invalid
        """
        raise NotImplementedError("Subclasses must implement parse_batch_response()")

    def get_output_format_description(self) -> str:
        """Get a description of the expected output format.

        This can be used in the prompt to tell the model what format to use.
        Override this to customize the format description.

        Returns:
            Description of output format
        """
        return f"""[
    {{"{self.detection_id_key}": <id>, "{self.label_key}": <class>}},
    ...
]"""

    def validate_parsed_response(
        self,
        parsed: dict[int, str],
        detections: Detection,
    ) -> tuple[bool, str | None]:
        """Validate the parsed response.

        Checks:
        - All detection indices are valid
        - All labels are in the valid classes (optional, based on validator)

        Args:
            parsed: Parsed response mapping indices to labels
            detections: All detections in the image

        Returns:
            Tuple of (is_valid, error_message)
        """
        num_detections = len(detections.xyxy)

        # Check all indices are valid
        for idx in parsed.keys():
            if idx < 0 or idx >= num_detections:
                return (
                    False,
                    f"Invalid detection index: {idx} (max: {num_detections-1})",
                )

        # Check we got labels for all detections
        if len(parsed) != num_detections:
            return False, f"Expected {num_detections} labels, got {len(parsed)}"

        return True, None


class JSONBatchPrompt(BatchPrompt):
    """Batch prompt that expects JSON array output.

    Expected format:
    [
        {"person": 0, "action": "walking"},
        {"person": 1, "action": "sitting"},
        ...
    ]

    Example usage:
        prompt = JSONBatchPrompt(
            system_prompt="You are a vision AI...",
            template="Classify people in the numbered boxes.\n\nActions:\n{classes}\n\nReturn JSON array.",
            classes=["walking", "sitting", "standing"],
            detection_id_key="person",  # Could be "id", "detection", etc.
            label_key="action",          # Could be "label", "classification", etc.
        )
    """

    def parse_batch_response(
        self, raw_response: str, detections: Detection
    ) -> dict[int, str]:
        """Parse JSON array response.

        Args:
            raw_response: Raw JSON response from model
            detections: All detections for validation

        Returns:
            Dictionary mapping detection index -> label

        Raises:
            ValueError: If JSON is invalid or missing required fields
        """
        import json

        # Clean up response (remove code fences)
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Parse JSON
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")

        # Ensure it's a list
        if not isinstance(parsed, list):
            raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")

        # Extract labels
        result = {}
        for item in parsed:
            if not isinstance(item, dict):
                raise ValueError(f"Expected object in array, got {type(item).__name__}")

            # Get detection ID
            if self.detection_id_key not in item:
                raise ValueError(f"Missing '{self.detection_id_key}' key in: {item}")
            detection_id = item[self.detection_id_key]

            # Get label
            if self.label_key not in item:
                raise ValueError(f"Missing '{self.label_key}' key in: {item}")
            label = item[self.label_key]

            # Convert detection_id to int if needed
            try:
                detection_idx = int(detection_id)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid detection ID (must be integer): {detection_id}"
                )

            result[detection_idx] = label

        # Validate
        is_valid, error = self.validate_parsed_response(result, detections)
        if not is_valid:
            raise ValueError(error)

        return result

    def get_output_format_description(self) -> str:
        """Get JSON format description."""
        return f"""[
    {{"{self.detection_id_key}": 0, "{self.label_key}": "<class>"}},
    {{"{self.detection_id_key}": 1, "{self.label_key}": "<class>"}},
    ...
]"""


class TextBatchPrompt(BatchPrompt):
    """Batch prompt that expects simple text format.

    Expected format:
    0: walking
    1: sitting
    2: standing

    Or:
    Person 0: walking
    Person 1: sitting
    """

    def parse_batch_response(
        self, raw_response: str, detections: Detection
    ) -> dict[int, str]:
        """Parse text format response.

        Supports formats like:
        - "0: walking"
        - "Person 0: walking"
        - "Detection 0: walking"

        Args:
            raw_response: Raw text response
            detections: All detections for validation

        Returns:
            Dictionary mapping detection index -> label

        Raises:
            ValueError: If format is invalid
        """
        import re

        result = {}

        # Try to match lines with patterns like:
        # "0: walking" or "Person 0: walking" or "Detection 0: walking"
        pattern = r"(?:" + re.escape(self.detection_id_key) + r"\s*)?(\d+)\s*:\s*(.+)"

        for line in raw_response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                detection_idx = int(match.group(1))
                label = match.group(2).strip()
                result[detection_idx] = label

        # Validate
        is_valid, error = self.validate_parsed_response(result, detections)
        if not is_valid:
            raise ValueError(error)

        return result

    def get_output_format_description(self) -> str:
        """Get text format description."""
        return f"""0: <class>
1: <class>
..."""
