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
from action_labeler.detections import Detection, DetectionFormat
from PIL import Image

# Load image to get size
image = Image.open("image.jpg")
image_size = image.size  # (width, height)

# Automatically detect format from .txt file
detection = Detection.from_text_path("labels/image.txt", image_size)

# Or specify the format explicitly
bbox_detection = Detection.from_bbox_text_path("labels/image.txt", image_size)
segment_detection = Detection.from_segment_text_path("labels/image.txt", image_size)

# For pose, you must specify the number of keypoints
pose_detection = Detection.from_pose_text_path("labels/image.txt", image_size, num_keypoints=17)
```

### Creating Empty Detection

```python
# Empty detection (no objects found)
empty = Detection.empty(image_size=(1920, 1080))

# Empty detection with specific format
empty_pose = Detection.empty(
    image_size=(1920, 1080), 
    detection_format=DetectionFormat.POSE
)
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
if detection.format == DetectionFormat.POSE:
    keypoints = detection.keypoints  # numpy array, shape (N, K, 3)
    # Each keypoint: [x, y, visibility]
    # visibility: 0=not labeled, 1=labeled but not visible, 2=labeled and visible

# Check format
print(detection.format)  # DetectionFormat.BBOX, SEGMENT, or POSE

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
class_id x_center y_center width height px1 py1 v1 px2 py2 v2 ... pxn pyn vn
```
- 5 bbox values + 3 values per keypoint (x, y, visibility)
- All coordinates normalized (0-1)
- Visibility values: 0, 1, or 2
- Example for 3 keypoints: `0 0.5 0.5 0.3 0.4 0.6 0.3 2 0.5 0.4 2 0.4 0.3 1`

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
├── __init__.py           # Exports: Detection, DetectionFormat, DetectionManager
├── detection.py          # Detection class (main container)
├── detection_manager.py  # YOLO detection runner
├── helpers.py            # Coordinate conversion helpers
└── README.md            # This file
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
    detection = Detection.from_text_path(txt_path, image.size)
    
    # Process each detection
    for i in range(len(detection.xyxy)):
        bbox = detection.xyxy[i]
        class_id = detection.class_id[i]
        print(f"Found object {class_id} at {bbox}")
```

### Example 2: Work with Pose Keypoints

```python
from action_labeler.detections import Detection, DetectionFormat
from PIL import Image

# Load pose detections (17 keypoints for COCO format)
image = Image.open("person.jpg")
detection = Detection.from_pose_text_path(
    "labels/person.txt", 
    image.size,
    num_keypoints=17
)

# Access keypoints
for i in range(len(detection.keypoints)):
    keypoints = detection.keypoints[i]  # Shape: (17, 3)
    
    # Check each keypoint
    for kp_idx, (x, y, visibility) in enumerate(keypoints):
        if visibility == 2:  # Labeled and visible
            print(f"Keypoint {kp_idx}: ({x:.3f}, {y:.3f})")
```

### Example 3: Filter Detections

```python
from action_labeler.detections import Detection
from PIL import Image
import numpy as np

image = Image.open("crowded_scene.jpg")
detection = Detection.from_text_path("labels/crowded_scene.txt", image.size)

# Filter by class (e.g., only persons, class_id=0)
person_mask = detection.class_id == 0
filtered_detection = Detection(
    xyxy=detection.xyxy[person_mask],
    segmentation_points=[detection.segmentation_points[i] for i, m in enumerate(person_mask) if m],
    keypoints=detection.keypoints[person_mask] if detection.keypoints.size > 0 else np.array([]),
    class_id=detection.class_id[person_mask],
    image_size=detection.image_size,
    detection_format=detection.format
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

detection = Detection.from_text_path("labels.txt", image_size)
xyxys = detection.xyxy  # Already in pixel coordinates
xywhs = detection.xywh  # Normalized coordinates
```

The `Detection` class handles all coordinate conversions internally, making your code simpler and less error-prone.

## Testing

```python
# Test with bbox format
bbox_detection = Detection.from_bbox_text_path("test_bbox.txt", (1920, 1080))
assert bbox_detection.format == DetectionFormat.BBOX
assert len(bbox_detection.xyxy) > 0
assert bbox_detection.keypoints.size == 0

# Test with segmentation format
seg_detection = Detection.from_segment_text_path("test_seg.txt", (1920, 1080))
assert seg_detection.format == DetectionFormat.SEGMENT
assert all(len(seg) > 0 for seg in seg_detection.segmentation_points)

# Test with pose format
pose_detection = Detection.from_pose_text_path("test_pose.txt", (1920, 1080), num_keypoints=17)
assert pose_detection.format == DetectionFormat.POSE
assert pose_detection.keypoints.shape[1] == 17
assert pose_detection.keypoints.shape[2] == 3
```

## References

- [Ultralytics YOLO Detection Format](https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format)
- [Ultralytics YOLO Segmentation Format](https://docs.ultralytics.com/datasets/segment/)
- [Ultralytics YOLO Pose Format](https://docs.ultralytics.com/datasets/pose/)

