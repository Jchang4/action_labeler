from enum import Enum


class DetectionType(str, Enum):
    DETECT = "detect"
    SEGMENT = "segment"
