"""Composable processing pipeline for labeling workflow.

This module provides a flexible pipeline that chains together:
filters → preprocessors → model → validation
"""

from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from action_labeler.detections.detection import Detection
from action_labeler.filters.base import IFilter
from action_labeler.labeler.core.processing_modes import LabelResult, ProcessingUnit
from action_labeler.models.base import IVisionLanguageModel
from action_labeler.preprocessors.base import IPreprocessor
from action_labeler.prompt.base import BasePrompt


@dataclass
class ModelResponse:
    """Structured response from a vision-language model.

    Provides a standardized format for model outputs, replacing raw strings.

    Attributes:
        label: The primary label/classification
        confidence: Optional confidence score (0-1)
        raw_response: The complete raw response from the model
        metadata: Additional model-specific data (e.g., token count, latency)
        is_valid: Whether the response passed validation
        validation_error: Error message if validation failed
    """

    label: str
    raw_response: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True
    validation_error: str | None = None

    def __post_init__(self) -> None:
        """Validate confidence score if provided."""
        if self.confidence is not None:
            if not 0 <= self.confidence <= 1:
                raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


class ProcessingPipeline:
    """Composable pipeline for labeling workflow.

    The pipeline executes these stages in order:
    1. Filtering - Decide if a detection should be processed
    2. Preprocessing - Transform the image (crop, mask, etc.)
    3. Prompting - Generate the prompt for the model
    4. Model Inference - Query the VLM
    5. Response Parsing - Structure the model output
    6. Validation - Verify the label is valid

    Each stage can be customized with different implementations.
    """

    def __init__(
        self,
        model: IVisionLanguageModel,
        prompt: BasePrompt,
        filters: list[IFilter] | None = None,
        preprocessors: list[IPreprocessor] | None = None,
        response_parser: "IResponseParser | None" = None,
        label_validator: "ILabelValidator | None" = None,
    ):
        """Initialize the processing pipeline.

        Args:
            model: Vision-language model for labeling
            prompt: Prompt generator
            filters: Optional list of filters (detections that don't pass are skipped)
            preprocessors: Optional list of preprocessors (applied sequentially)
            response_parser: Optional parser to structure model output
            label_validator: Optional validator to check labels
        """
        self.model = model
        self.prompt = prompt
        self.filters = filters or []
        self.preprocessors = preprocessors or []
        self.response_parser = response_parser or DefaultResponseParser()
        self.label_validator = label_validator

    def should_process(self, unit: ProcessingUnit) -> bool:
        """Check if a processing unit should be labeled.

        Applies all filters in sequence. If any filter returns False,
        the unit is skipped.

        Args:
            unit: Processing unit to check

        Returns:
            True if unit should be processed, False to skip
        """
        # If no detection index, we're processing all detections (batch mode)
        # In this case, we might want to apply image-level filters
        if unit.detection_index is None:
            # For batch mode, could add image-level filters here
            # For now, always process batch units
            return True

        # Apply filters for single detection
        for filter_obj in self.filters:
            if not filter_obj.is_valid(
                unit.image, unit.detection_index, unit.detection
            ):
                return False

        return True

    def preprocess(self, unit: ProcessingUnit) -> Image.Image:
        """Apply preprocessing to a processing unit.

        Args:
            unit: Processing unit to preprocess

        Returns:
            Preprocessed image
        """
        image = unit.image

        # Only preprocess if we have a specific detection index
        # Batch mode typically uses the original image
        if unit.detection_index is None or not self.preprocessors:
            return image

        # Apply preprocessors sequentially
        for preprocessor in self.preprocessors:
            image = preprocessor.preprocess(
                image, unit.detection_index, unit.detection
            )

        return image

    def generate_prompt(self, unit: ProcessingUnit) -> str:
        """Generate prompt for a processing unit.

        Args:
            unit: Processing unit to generate prompt for

        Returns:
            Prompt string
        """
        # Get image path from metadata
        image_path = unit.metadata.get("image_path", "") if unit.metadata else ""

        # For batch mode (no detection index), we need a different prompt strategy
        # For now, use the first detection index as a placeholder
        # TODO: Extend BasePrompt to support batch prompting
        detection_index = unit.detection_index if unit.detection_index is not None else 0

        return self.prompt.prompt(detection_index, unit.detection, image_path)

    def query_model(self, image: Image.Image, prompt: str) -> str:
        """Query the vision-language model.

        Args:
            image: Preprocessed image
            prompt: Generated prompt

        Returns:
            Raw model response string
        """
        return self.model.predict(prompt, [image])

    def parse_response(self, raw_response: str, unit: ProcessingUnit) -> ModelResponse:
        """Parse and structure the model response.

        Args:
            raw_response: Raw string from model
            unit: Processing unit that was labeled

        Returns:
            Structured ModelResponse
        """
        return self.response_parser.parse(raw_response, unit)

    def validate_label(self, response: ModelResponse) -> ModelResponse:
        """Validate the parsed label.

        Args:
            response: Parsed model response

        Returns:
            ModelResponse with validation status updated
        """
        if self.label_validator is None:
            return response

        is_valid, error = self.label_validator.validate(
            response.label, self.prompt.classes
        )

        response.is_valid = is_valid
        if not is_valid:
            response.validation_error = error

        return response

    def process(self, unit: ProcessingUnit) -> ModelResponse | None:
        """Process a single unit through the complete pipeline.

        Args:
            unit: Processing unit to label

        Returns:
            ModelResponse if successful, None if filtered out
        """
        # Stage 1: Filtering
        if not self.should_process(unit):
            return None

        # Stage 2: Preprocessing
        preprocessed_image = self.preprocess(unit)

        # Stage 3: Prompting
        prompt = self.generate_prompt(unit)

        # Stage 4: Model Inference
        raw_response = self.query_model(preprocessed_image, prompt)

        # Stage 5: Response Parsing
        response = self.parse_response(raw_response, unit)

        # Stage 6: Validation
        response = self.validate_label(response)

        return response

    def process_batch(
        self, units: list[ProcessingUnit]
    ) -> list[tuple[ProcessingUnit, ModelResponse | None]]:
        """Process multiple units through the pipeline.

        Args:
            units: List of processing units

        Returns:
            List of (unit, response) tuples. Response is None if filtered.
        """
        results = []

        for unit in units:
            response = self.process(unit)
            results.append((unit, response))

        return results


class IResponseParser:
    """Interface for parsing model responses into structured format."""

    def parse(self, raw_response: str, unit: ProcessingUnit) -> ModelResponse:
        """Parse raw model response into structured format.

        Args:
            raw_response: Raw string from model
            unit: Processing unit that was labeled

        Returns:
            Structured ModelResponse
        """
        raise NotImplementedError


class DefaultResponseParser(IResponseParser):
    """Default parser that uses the raw response as the label.

    This maintains backward compatibility with the current behavior
    where model output is stored directly as the action label.
    """

    def parse(self, raw_response: str, unit: ProcessingUnit) -> ModelResponse:
        """Parse response by using it directly as label.

        Args:
            raw_response: Raw string from model
            unit: Processing unit that was labeled

        Returns:
            ModelResponse with label = raw_response
        """
        # Clean up response (strip whitespace, newlines)
        label = raw_response.strip()

        return ModelResponse(
            label=label,
            raw_response=raw_response,
            confidence=None,  # Default parser doesn't extract confidence
        )


class ILabelValidator:
    """Interface for validating labels."""

    def validate(self, label: str, valid_classes: list[str]) -> tuple[bool, str | None]:
        """Validate a label.

        Args:
            label: The label to validate
            valid_classes: List of valid class names

        Returns:
            Tuple of (is_valid, error_message)
            error_message is None if valid
        """
        raise NotImplementedError


class StrictClassValidator(ILabelValidator):
    """Validator that requires exact match with class names (case-insensitive)."""

    def validate(self, label: str, valid_classes: list[str]) -> tuple[bool, str | None]:
        """Validate label is in valid classes.

        Args:
            label: The label to validate
            valid_classes: List of valid class names

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Case-insensitive comparison
        label_lower = label.lower().strip()
        valid_lower = [c.lower() for c in valid_classes]

        if label_lower in valid_lower:
            return True, None
        else:
            return (
                False,
                f"Label '{label}' not in valid classes: {valid_classes}",
            )


class FlexibleClassValidator(ILabelValidator):
    """Validator that allows labels if they contain a valid class name."""

    def validate(self, label: str, valid_classes: list[str]) -> tuple[bool, str | None]:
        """Validate label contains a valid class.

        Args:
            label: The label to validate
            valid_classes: List of valid class names

        Returns:
            Tuple of (is_valid, error_message)
        """
        label_lower = label.lower()

        # Check if any valid class is contained in the label
        for valid_class in valid_classes:
            if valid_class.lower() in label_lower:
                return True, None

        return (
            False,
            f"Label '{label}' does not contain any valid class: {valid_classes}",
        )
