class DatasetColumns:
    """Column name constants — single source of truth for schema."""

    IMAGE_PATH = "image_path"
    DETECTION_INDEX = "detection_index"
    DETECTION = "detection"
    ACTION = "action"
    RESPONSE = "response"

    REQUIRED = {IMAGE_PATH, DETECTION_INDEX, DETECTION, ACTION, RESPONSE}
