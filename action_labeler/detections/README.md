# Detections Module

This module handles YOLO format detections for all three supported types: bounding boxes, segmentation, and pose estimation.

## Overview

The `Detection` class is a unified container for YOLO detections that automatically handles three different formats:

1. **Bounding Box (BBOX)** - Object detection with rectangular boxes
2. **Segmentation (SEGMENT)** - Instance segmentation with polygon masks  
3. **Pose Estimation (POSE)** - Human/animal pose with keypoints

All helper functions are colocated in `helpers.py` to keep the module self-contained.

## Quick Start

### Loading Detections from File

```python
from action_labeler.detections import Detection
from PIL import Image

# Load image
image = Image.open("image.jpg")

# Automatically detect format from .txt file
detection = Detection.from_text_path("labels/image.txt", image)

# Or specify the format explicitly
bbox_detection = Detection.from_bbox_text_path("labels/image.txt", image)
segment_detection = Detection.from_segment_text_path("labels/image.txt", image)

# For pose, you must specify the number of keypoints
pose_detection = Detection.from_pose_text_path("labels/image.txt", image, num_keypoints=17)
```

**Note:** The API accepts `image: Image.Image` directly rather than `image_size: tuple`. This design eliminates potential width/height confusion since PIL uses `(width, height)` order while numpy and many other libraries use `(height, width)`. By passing the image object, the Detection class extracts the dimensions internally in the correct format, making the API less error-prone.

### Creating Empty Detection

```python
from PIL import Image

# Empty detection (no objects found)
image = Image.new("RGB", (1920, 1080))
empty = Detection.empty(image)

# Empty detection without an image (creates 0x0 placeholder)
empty_no_image = Detection.empty()
```

### Accessing Detection Data

```python
# Bounding boxes in pixel coordinates
xyxys = detection.xyxy  # numpy array, shape (N, 4)

# Normalized xywh coordinates (for YOLO format)
xywhs = detection.xywh  # list of tuples

# Class IDs
class_ids = detection.class_id  # numpy array, shape (N,)

# Segmentation polygons (normalized coordinates)
polygons = detection.segmentation_points  # list of lists

# Keypoints (for pose format only)
if detection.keypoints.size > 0:
    keypoints = detection.keypoints  # numpy array, shape (N, K, 2)
    # Each keypoint: [x, y] in normalized coordinates [0-1]

# Check if empty
if detection.is_empty():
    print("No detections found")
```

### Working with Individual Detections

```python
# Get a single detection by index
single_detection = detection.get_index(0)

# Make a copy
detection_copy = detection.copy()
```

## YOLO Format Reference

Based on [Ultralytics YOLO documentation](https://docs.ultralytics.com/datasets/detect/)

### Bounding Box Format
```
class_id x_center y_center width height
```
- 5 values per line
- All coordinates normalized (0-1)
- Example: `0 0.5 0.5 0.3 0.4`

### Segmentation Format  
```
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```
- Variable number of coordinate pairs (polygon vertices)
- All coordinates normalized (0-1)
- Example: `0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4`

### Pose Format
```
class_id x_center y_center width height px1 py1 px2 py2 ... pxn pyn
```
- 5 bbox values + 2 values per keypoint (x, y)
- All coordinates normalized (0-1)
- Example for 3 keypoints: `0 0.5 0.5 0.3 0.4 0.6 0.3 0.5 0.4 0.4 0.3`

## DetectionManager

Utility class for running YOLO detection on image directories.

```python
from action_labeler.detections import DetectionManager

# Create detector once
detector = DetectionManager(
    model_name="yolo11n.pt",
    batch=32,
    classes=[0],  # person class
    conf=0.5
)

# Run on multiple directories
detector.detect("datasets/classA/")
detector.detect("datasets/classB/")

# Creates directory structure:
# datasets/classA/
#   images/          # Your images
#   detect/          # Generated .txt files
```

### Features
- ✅ Reusable across multiple image directories
- ✅ Lazy loads model (only loads once)
- ✅ Uses UUID subfolders to avoid conflicts
- ✅ Automatic cleanup after each run
- ✅ Progress tracking with tqdm
- ✅ Skips existing detections

## Architecture

### Module Structure

```
detections/
├── __init__.py           # Exports: Detection, DetectionManager
├── detection.py          # Detection class (main container)
├── detection_manager.py  # YOLO detection runner
├── helpers.py            # Coordinate conversion helpers
└── README.md             # This file
```

### Key Design Decisions

1. **Self-Contained Module** - All helpers are colocated to avoid dependencies on global `action_labeler.helpers`

2. **Unified Detection Class** - Single class handles all three formats rather than separate classes

3. **Automatic Format Detection** - `from_text_path()` auto-detects format based on number of values

4. **Normalized Coordinates** - YOLO uses normalized coords (0-1), but Detection stores pixel coords for easier processing

5. **Numpy for Performance** - Uses numpy arrays for bounding boxes, keypoints, and class IDs

## Examples

### Example 1: Process Detections from YOLO

```python
from action_labeler.detections import DetectionManager, Detection
from PIL import Image
from pathlib import Path

# Step 1: Run YOLO detection
detector = DetectionManager(model_name="yolo11n.pt", classes=[0])
detector.detect("datasets/people/")

# Step 2: Load and process detections
images_dir = Path("datasets/people/images")
detect_dir = Path("datasets/people/detect")

for image_path in images_dir.glob("*.jpg"):
    # Load image
    image = Image.open(image_path)
    
    # Load corresponding detections
    txt_path = detect_dir / image_path.with_suffix(".txt").name
    detection = Detection.from_text_path(txt_path, image)
    
    # Process each detection
    for i in range(len(detection.xyxy)):
        bbox = detection.xyxy[i]
        class_id = detection.class_id[i]
        print(f"Found object {class_id} at {bbox}")
```

### Example 2: Work with Pose Keypoints

```python
from action_labeler.detections import Detection
from PIL import Image

# Load pose detections (17 keypoints for COCO format)
image = Image.open("person.jpg")
detection = Detection.from_pose_text_path(
    "labels/person.txt",
    image,
    num_keypoints=17
)

# Access keypoints
for i in range(len(detection.keypoints)):
    keypoints = detection.keypoints[i]  # Shape: (17, 2)

    # Check each keypoint
    for kp_idx, (x, y) in enumerate(keypoints):
        print(f"Keypoint {kp_idx}: ({x:.3f}, {y:.3f})")
```

### Example 3: Filter Detections

```python
from action_labeler.detections import Detection
from PIL import Image
import numpy as np

image = Image.open("crowded_scene.jpg")
detection = Detection.from_text_path("labels/crowded_scene.txt", image)

# Filter by class (e.g., only persons, class_id=0)
person_mask = detection.class_id == 0
filtered_detection = Detection(
    xyxy=detection.xyxy[person_mask],
    segmentation_points=[detection.segmentation_points[i] for i, m in enumerate(person_mask) if m],
    keypoints=detection.keypoints[person_mask] if detection.keypoints.size > 0 else np.array([]),
    class_id=detection.class_id[person_mask],
    image=image,
)

print(f"Filtered from {len(detection.xyxy)} to {len(filtered_detection.xyxy)} persons")
```

## Migration from Old Code

If you were using the old global helpers, here's how to migrate:

### Before
```python
from action_labeler.helpers.detections_helpers import xywhs_to_xyxys
from action_labeler.helpers.yolov8_dataset import yolov8_labels_to_row

rows = yolov8_labels_to_row("labels.txt")
xywhs = [row[1:5] for row in rows]
xyxys = xywhs_to_xyxys(xywhs, image_size)
```

### After
```python
from action_labeler.detections import Detection
from PIL import Image

image = Image.open("image.jpg")
detection = Detection.from_text_path("labels.txt", image)
xyxys = detection.xyxy  # Already in pixel coordinates
xywhs = detection.xywh  # Normalized coordinates
```

The `Detection` class handles all coordinate conversions internally, making your code simpler and less error-prone.

## Testing

```python
from PIL import Image

# Create test image (or load from file)
test_image = Image.new("RGB", (1920, 1080))

# Test with bbox format
bbox_detection = Detection.from_bbox_text_path("test_bbox.txt", test_image)
assert len(bbox_detection.xyxy) > 0
assert bbox_detection.keypoints.size == 0

# Test with segmentation format
seg_detection = Detection.from_segment_text_path("test_seg.txt", test_image)
assert all(len(seg) > 0 for seg in seg_detection.segmentation_points)

# Test with pose format
pose_detection = Detection.from_pose_text_path("test_pose.txt", test_image, num_keypoints=17)
assert pose_detection.keypoints.shape[1] == 17
assert pose_detection.keypoints.shape[2] == 2
```

## References

- [Ultralytics YOLO Detection Format](https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format)
- [Ultralytics YOLO Segmentation Format](https://docs.ultralytics.com/datasets/segment/)
- [Ultralytics YOLO Pose Format](https://docs.ultralytics.com/datasets/pose/)

