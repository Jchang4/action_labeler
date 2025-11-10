"""Abstraction for providing images and detections to labelers.

This module decouples the data source from the labeling logic,
making it easy to switch between folders, databases, APIs, etc.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image

from action_labeler.detections.detection import Detection
from action_labeler.helpers.detections_helpers import image_to_txt_path
from action_labeler.helpers.general import get_image_paths, load_image


@dataclass
class ImageData:
    """Container for an image and its associated data.

    Attributes:
        image_path: Path or identifier for the image
        image: PIL Image
        detections: Detection objects for this image
        metadata: Additional metadata (e.g., dataset split, quality score)
    """

    image_path: str
    image: Image.Image
    detections: Detection
    metadata: dict | None = None


class IImageProvider(ABC):
    """Interface for providing images and detections to labelers.

    Implementations can load from various sources:
    - Local folder with YOLO .txt files
    - Database with stored detections
    - API endpoints
    - Cached/preprocessed data
    """

    @abstractmethod
    def __iter__(self) -> Iterator[ImageData]:
        """Iterate over all images and their detections.

        Yields:
            ImageData for each image
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get total number of images available.

        Returns:
            Number of images
        """
        pass

    @abstractmethod
    def get_progress_description(self) -> str:
        """Get a description for progress tracking.

        Returns:
            String description (e.g., "Processing images", "Labeling dataset")
        """
        pass


class FolderImageProvider(IImageProvider):
    """Provides images from a folder with YOLO detection .txt files.

    This is the standard provider that loads:
    - Images from a folder (jpg, png, etc.)
    - Detections from corresponding .txt files
    """

    def __init__(
        self,
        folder: str | Path,
        detection_type: str = "detect",
        image_extensions: list[str] | None = None,
    ):
        """Initialize folder provider.

        Args:
            folder: Path to folder containing images and .txt files
            detection_type: "detect", "segment", or "pose"
            image_extensions: List of valid image extensions (default: jpg, png, etc.)
        """
        self.folder = Path(folder)
        if not self.folder.exists():
            raise ValueError(f"Folder does not exist: {folder}")

        self.detection_type = detection_type
        self.image_paths = get_image_paths(self.folder)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in folder: {folder}")

    def _get_txt_path(self, image_path: Path) -> Path:
        """Get the corresponding .txt file for an image.

        Handles YOLO folder structure where:
        - Images are in folder/images/*.jpg
        - Labels are in folder/{detection_type}/*.txt

        Args:
            image_path: Path to image file

        Returns:
            Path to .txt file
        """
        return image_to_txt_path(image_path, detection_type=self.detection_type)

    def __iter__(self) -> Iterator[ImageData]:
        """Iterate over images in the folder.

        Yields:
            ImageData for each image with valid detections
        """
        for image_path in self.image_paths:
            # Load image
            try:
                image = load_image(image_path)
            except Exception as e:
                # Skip images that can't be loaded
                print(f"Warning: Failed to load {image_path}: {e}")
                continue

            # Load detections
            txt_path = self._get_txt_path(Path(image_path))
            if not txt_path.exists():
                # Skip images without detection files
                continue

            try:
                detections = Detection.from_text_path(txt_path, image)
            except Exception as e:
                # Skip images with invalid detections
                print(f"Warning: Failed to load detections for {image_path}: {e}")
                continue

            # Skip images with no detections
            if len(detections.xyxy) == 0:
                continue

            yield ImageData(
                image_path=str(image_path),
                image=image,
                detections=detections,
                metadata={"source": "folder", "folder": str(self.folder)},
            )

    def __len__(self) -> int:
        """Get number of images (approximation - some may be skipped).

        Returns:
            Number of image files found
        """
        return len(self.image_paths)

    def get_progress_description(self) -> str:
        """Get progress description for this provider.

        Returns:
            Progress description string
        """
        return "Processing images"


class FilteredImageProvider(IImageProvider):
    """Wraps another provider and filters images based on criteria.

    This allows filtering at the image level (before loading detections),
    which is more efficient than filtering at the detection level.

    Example use cases:
    - Process only images from a specific dataset split
    - Skip images below a quality threshold
    - Process only images with certain metadata
    """

    def __init__(
        self,
        base_provider: IImageProvider,
        filter_func: Callable[[ImageData], bool],
    ):
        """Initialize filtered provider.

        Args:
            base_provider: Underlying image provider
            filter_func: Function that returns True to keep image, False to skip
        """
        self.base_provider = base_provider
        self.filter_func = filter_func

    def __iter__(self) -> Iterator[ImageData]:
        """Iterate over filtered images.

        Yields:
            ImageData that passes the filter
        """
        for image_data in self.base_provider:
            if self.filter_func(image_data):
                yield image_data

    def __len__(self) -> int:
        """Get approximate length (actual may be less due to filtering).

        Returns:
            Length of base provider (upper bound)
        """
        return len(self.base_provider)

    def get_progress_description(self) -> str:
        """Get progress description from base provider.

        Returns:
            Progress description string
        """
        return self.base_provider.get_progress_description()


class CachedImageProvider(IImageProvider):
    """Wraps another provider and caches loaded images in memory.

    Useful for:
    - Multiple passes over the same dataset
    - Reducing I/O when images are accessed multiple times
    - Experimentation workflows that iterate over same images

    Warning: Can use significant memory for large datasets.
    """

    def __init__(
        self, base_provider: IImageProvider, max_cache_size: int | None = None
    ):
        """Initialize cached provider.

        Args:
            base_provider: Underlying image provider
            max_cache_size: Maximum number of images to cache (None = unlimited)
        """
        self.base_provider = base_provider
        self.max_cache_size = max_cache_size
        self._cache: list[ImageData] = []
        self._is_cached = False

    def _load_cache(self) -> None:
        """Load all images into cache."""
        if self._is_cached:
            return

        for i, image_data in enumerate(self.base_provider):
            if self.max_cache_size and i >= self.max_cache_size:
                break
            self._cache.append(image_data)

        self._is_cached = True

    def __iter__(self) -> Iterator[ImageData]:
        """Iterate over cached images.

        Yields:
            ImageData from cache
        """
        self._load_cache()
        yield from self._cache

    def __len__(self) -> int:
        """Get number of cached images.

        Returns:
            Number of images in cache or base provider
        """
        if self._is_cached:
            return len(self._cache)
        return len(self.base_provider)

    def get_progress_description(self) -> str:
        """Get progress description from base provider.

        Returns:
            Progress description string
        """
        return self.base_provider.get_progress_description()


class SubsetImageProvider(IImageProvider):
    """Provides a subset of images from another provider.

    Useful for:
    - Testing on a small sample
    - Splitting dataset into chunks for parallel processing
    - Resuming from a checkpoint
    """

    def __init__(
        self,
        base_provider: IImageProvider,
        start: int = 0,
        end: int | None = None,
    ):
        """Initialize subset provider.

        Args:
            base_provider: Underlying image provider
            start: Starting index (inclusive)
            end: Ending index (exclusive), None for all remaining
        """
        self.base_provider = base_provider
        self.start = start
        self.end = end if end is not None else len(base_provider)

        if self.start < 0:
            raise ValueError(f"start must be non-negative, got {self.start}")
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")

    def __iter__(self) -> Iterator[ImageData]:
        """Iterate over subset of images.

        Yields:
            ImageData in the specified range
        """
        for i, image_data in enumerate(self.base_provider):
            if i < self.start:
                continue
            if i >= self.end:
                break
            yield image_data

    def __len__(self) -> int:
        """Get size of subset.

        Returns:
            Number of images in subset
        """
        return min(self.end, len(self.base_provider)) - self.start

    def get_progress_description(self) -> str:
        """Get progress description from base provider.

        Returns:
            Progress description string
        """
        return self.base_provider.get_progress_description()
