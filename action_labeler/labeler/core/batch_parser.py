"""Response parser for batch mode that handles multiple detections per response.

In batch mode, the model returns labels for multiple detections in a single response.
This parser uses the prompt's parsing logic to extract individual labels.
"""

from action_labeler.labeler.core.processing_modes import ProcessingUnit
from action_labeler.labeler.core.processing_pipeline import (
    IResponseParser,
    ModelResponse,
)
from action_labeler.prompt.batch_prompt import BatchPrompt


class BatchResponseParser(IResponseParser):
    """Parser for batch mode responses.

    This parser:
    1. Uses the prompt's parse_batch_response() method to extract labels
    2. Returns a ModelResponse with metadata about all detections
    3. Stores the parsed mapping in metadata for later processing

    The parsed response (dict[int, str]) is stored in metadata["batch_labels"]
    so that the labeler can split it into individual detection labels.

    Args:
        prompt: The BatchPrompt that knows how to parse its output format
    """

    def __init__(self, prompt: BatchPrompt):
        """Initialize the parser.

        Args:
            prompt: BatchPrompt instance with parsing logic
        """
        self.prompt = prompt

    def parse(self, raw_response: str, unit: ProcessingUnit) -> ModelResponse:
        """Parse batch response using the prompt's parsing logic.

        Args:
            raw_response: Raw string from model
            unit: Processing unit (should have detection_index=None for batch)

        Returns:
            ModelResponse with batch_labels in metadata
        """
        try:
            # Use the prompt's parsing logic
            batch_labels = self.prompt.parse_batch_response(
                raw_response, unit.detection
            )

            # For batch mode, we store all labels in metadata
            # The label field contains a summary
            num_detections = len(batch_labels)
            label_summary = f"Batch: {num_detections} detections labeled"

            return ModelResponse(
                label=label_summary,
                raw_response=raw_response,
                confidence=None,
                metadata={
                    "batch_labels": batch_labels,  # dict[int, str]
                    "num_detections": num_detections,
                    "batch_mode": True,
                },
                is_valid=True,
            )

        except ValueError as e:
            # Parsing failed
            return ModelResponse(
                label="",
                raw_response=raw_response,
                confidence=None,
                metadata={
                    "batch_mode": True,
                    "parsing_error": str(e),
                },
                is_valid=False,
                validation_error=f"Failed to parse batch response: {e}",
            )
        except Exception as e:
            # Unexpected error
            return ModelResponse(
                label="",
                raw_response=raw_response,
                confidence=None,
                metadata={
                    "batch_mode": True,
                    "error": str(e),
                },
                is_valid=False,
                validation_error=f"Unexpected error parsing batch response: {e}",
            )


class HybridResponseParser(IResponseParser):
    """Parser that switches between single and batch parsing based on the unit.

    This is useful when using hybrid mode or when you want a single pipeline
    to handle both single and batch detections.

    Args:
        batch_parser: Parser for batch mode (uses BatchPrompt)
        single_parser: Parser for single detection mode
    """

    def __init__(
        self,
        batch_parser: BatchResponseParser,
        single_parser: IResponseParser,
    ):
        """Initialize the hybrid parser.

        Args:
            batch_parser: Parser for batch responses
            single_parser: Parser for single detection responses
        """
        self.batch_parser = batch_parser
        self.single_parser = single_parser

    def parse(self, raw_response: str, unit: ProcessingUnit) -> ModelResponse:
        """Parse response based on the processing unit type.

        If unit.detection_index is None, uses batch parser.
        Otherwise, uses single detection parser.

        Args:
            raw_response: Raw string from model
            unit: Processing unit

        Returns:
            Parsed ModelResponse
        """
        if unit.detection_index is None:
            # Batch mode
            return self.batch_parser.parse(raw_response, unit)
        else:
            # Single detection mode
            return self.single_parser.parse(raw_response, unit)
