# Action Labeler

An autodistillation framework that uses vision-language models (VLMs) to label human actions in images. Given YOLO detections, Action Labeler classifies what each person is doing — walking, sitting, running, etc. — and produces clean datasets ready for YOLOv8 training.

## Overview

Action Labeler sits between your object detector and your action classifier. The pipeline:

1. Run a pre-trained YOLO model to get bounding boxes and segmentation masks
2. Feed each detection through configurable preprocessors and filters
3. A VLM classifies the action for each detection
4. Results are collected into a Dataset for analysis, balancing, and export

## Installation

```bash
uv pip install -e ".[dev]"
```

## Quick Start

```python
from pathlib import Path
from action_labeler import ActionResponse, Dataset
from action_labeler.models import LlamaCpp
from action_labeler.preprocessors import Resize
from action_labeler.filters import AspectRatioFilter
from action_labeler.prompts import Prompt
from action_labeler.labeler import SingleDetectionLabeler

# Define a structured response model
class MyAction(ActionResponse):
    action: str
    confidence: str

# Set up the labeling pipeline
labeler = SingleDetectionLabeler(
    model=LlamaCpp("http://localhost:5000"),
    prompt=Prompt(
        system="Classify the action of the person in the image.",
        user="What is this person doing?",
        response_model=MyAction,
    ),
    preprocessors=[[Resize(768)]],
    filters=[AspectRatioFilter(min_ratio=0.3)],
    save_every=50,
    save_path=Path("results.pkl"),
)

# Run labeling — auto-resumes if save_path exists
dataset = labeler.run(Path("my_dataset/"))

# Post-process and export
dataset.remove_class("unknown")
dataset.balance(seed=42)
dataset.export_yolov8(Path("yolo_dataset"), val_ratio=0.2, seed=42)
```

## Labeling Strategies

Three strategies for how the VLM processes detections:

**SingleDetectionLabeler** — One VLM call per detection. Best for accuracy when you need isolated context per person.

```python
from action_labeler.labeler import SingleDetectionLabeler

labeler = SingleDetectionLabeler(model=model, prompt=prompt)
```

**AllAtOnceLabeler** — One VLM call per image, all detections at once. Faster for crowded scenes. Use `list[ActionResponse]` as your response model.

```python
from action_labeler.labeler import AllAtOnceLabeler

labeler = AllAtOnceLabeler(model=model, prompt=prompt)
```

**MultiViewLabeler** — One VLM call per detection with multiple preprocessed views (e.g., a cropped view and a full-image view). Requires at least two preprocessor chains.

```python
from action_labeler.labeler import MultiViewLabeler
from action_labeler.preprocessors import BoundingBox, Resize

labeler = MultiViewLabeler(
    model=model,
    prompt=prompt,
    preprocessors=[
        [Resize(768)],                    # full image, resized
        [BoundingBox(), Resize(768)],     # annotated with bounding boxes
    ],
)
```

## Filters

Filters reject images before they reach the VLM. All filters are composable — pass a list and all must pass.

```python
from action_labeler.filters import (
    AspectRatioFilter,
    DetectionCountFilter,
    OverlapFilter,
)

filters = [
    AspectRatioFilter(min_ratio=0.3, max_ratio=3.0),  # skip extreme aspect ratios
    DetectionCountFilter(min_count=1, max_count=5),     # skip empty or crowded images
    OverlapFilter(max_iou=0.5),                         # skip heavily overlapping detections
]
```

## Preprocessors

Preprocessors transform images before VLM inference. They are configured as chains — each chain produces one image sent to the model.

```python
from action_labeler.preprocessors import BoundingBox, Resize

# Single chain: resize only
preprocessors = [[Resize(768)]]

# Two chains: one clean view, one annotated — produces two images per call
preprocessors = [
    [Resize(768)],
    [BoundingBox(line_width=2, font_size=16), Resize(768)],
]
```

## Prompts

Prompts support structured output via Pydantic response models. The `Prompt` class handles JSON format instructions and parsing automatically.

```python
from action_labeler import ActionResponse
from action_labeler.prompts import Prompt

class DetailedAction(ActionResponse):
    action: str
    pose: str
    confidence: str

prompt = Prompt(
    system="Classify the action of the highlighted person.",
    user="What is person {detection_index} doing?",
    response_model=DetailedAction,
)

# Template variables are filled at inference time
formatted = prompt.format_user(detection_index=0)
```

## Dataset Management

The `Dataset` class wraps a pandas DataFrame with one row per (image, detection) pair.

```python
from action_labeler import Dataset

# Load a saved dataset
dataset = Dataset.load(Path("results.pkl"))

# Inspect
print(dataset.df.head())
dataset.plot_distribution()
dataset.plot_grid(n=16, action="walking")

# Clean up classes
dataset.remove_class("unknown")
dataset.rename_class("jog", "running")

# Balance class distribution
dataset.balance(seed=42)

# Combine multiple datasets
combined = Dataset.combine(dataset1, dataset2)

# Export to YOLOv8 format with stratified train/val split
combined.export_yolov8(Path("yolo_dataset"), val_ratio=0.2, seed=42)
```

## Expected Input Format

Action Labeler expects a dataset directory with YOLO-format detections:

```
dataset_folder/
  images/    # .jpg, .jpeg, .png, etc.
  detect/    # bounding boxes (class_id x_center y_center width height, normalized 0-1)
  segments/  # segmentation masks (ultralytics format)
```

Each detection file shares a filename stem with its corresponding image.

## Architecture

```
src/action_labeler/
  types.py              # Detection, LabelResult, ActionResponse
  models/               # VLM integrations (base interface + llama.cpp)
  preprocessors/        # Image transforms (bounding_box, resize)
  filters/              # Rejection filters (aspect_ratio, detection_count, overlap)
  prompts/              # Prompt template with structured response parsing
  labeler/              # Orchestration (base + 3 strategies)
    base.py             # Shared run() loop, filtering, resume, auto-save
    all_at_once.py      # One VLM call per image
    single_detection.py # One VLM call per detection
    multi_view.py       # Multiple views per detection
  dataset/              # pandas DataFrame wrapper
    dataset.py          # Construction, save/load, combine
    filter.py           # balance, remove_class, rename_class
    plot.py             # plot_grid, plot_distribution
    export.py           # export_yolov8 with stratified split
```
