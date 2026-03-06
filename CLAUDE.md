# CLAUDE.md

## Important

Do not use `yolo/action_labeler` as a reference. We are creating a new package for a reason.

## Project Overview

ActionLabeler is an autodistillation framework that uses VLMs to label the actions of people in images.

## Pipeline

1. User runs a pre-trained YOLO model (ultralytics) to get detections and segmentation masks
2. Pre-computed bounding boxes, image preprocessors, filters, and a prompt are used to classify actions via a VLM
3. The VLM classifies the actions of each bounding box/segmentation mask

## Expected Dataset Structure

```
dataset_folder/
  images/    # jpgs, jpegs, pngs, etc.
  detect/    # bounding box detections (ultralytics format: class_id, x_center, y_center, box_width, box_height normalized 0-1)
  segments/  # segmentation masks (ultralytics format)
```

Examples: `../datasets/human/sitting/`

## Architecture

```
src/action_labeler/
  types.py              # Detection, LabelResult, ActionResponse dataclasses/models
  models/               # VLM integrations (base interface + llama.cpp)
  preprocessors/        # Image transforms before VLM inference (bounding_box, resize)
  filters/              # Reject detections/images (aspect_ratio, detection_count, overlap)
  prompts/              # Prompt template with system/user messages and response parsing
  labeler/              # Orchestration layer (abstract base + 3 strategies)
    base.py             # ActionLabeler — shared run() loop, filtering, resume, auto-save
    all_at_once.py      # One VLM call per image, all detections at once
    single_detection.py # One VLM call per detection
    multi_view.py       # One VLM call per detection, multiple preprocessed views
  dataset/              # Output container (pandas DataFrame wrapper)
    columns.py          # Column name constants (single source of truth)
    dataset.py          # Dataset class — construction, save/load, add_rows, has_row, combine
    filter.py           # DatasetFilterMixin — balance, remove_class, rename_class
    plot.py             # DatasetPlotMixin — plot_grid, plot_distribution
    export.py           # DatasetExportMixin — export_yolov8 with stratified split
```

## Core Components

- **Detection** (`types.py`): Dataclass for a single YOLO-format detection with normalized coordinates and pixel-space properties (x1, y1, x2, y2, xyxy). Has `from_yolo()` and `load_txt()` class methods
- **ActionResponse** (`types.py`): Base Pydantic model for VLM responses. All prompt response models should inherit from this to ensure `action` field is present
- **LabelResult** (`types.py`): Pairs the extracted action string with the full VLM response. Returned by `label()` methods
- **Model** (`models/`): VLM integration — `predict(system, user, images)` returns raw text. Currently: `LlamaCppModel`
- **Preprocessors** (`preprocessors/`): Transform images before VLM inference. Configured as chains: `list[list[BasePreprocessor]]` where each inner list produces one image
- **Filters** (`filters/`): Accept/reject an image and its detections. Return `True` to keep, `False` to skip
- **Prompt** (`prompts/`): Template with `format_system()`, `format_user()`, and `parse(text)` for response parsing (supports Pydantic response models)
- **ActionLabeler** (`labeler/`): Abstract base with shared `run()` loop. Subclasses implement `label()` returning `list[LabelResult]`. Supports `save_every`/`save_path` for periodic checkpointing and auto-resume from existing save files
- **Dataset** (`dataset/`): Output container wrapping a pandas DataFrame. One row per (image, detection) pair. Supports save/load (pickle), combine, filtering (balance, remove_class, rename_class), plotting (plot_grid, plot_distribution), and export (export_yolov8)

## Usage Pattern

```python
labeler = SingleDetectionLabeler(
    model=LlamaCppModel(...),
    prompt=Prompt(system="...", user="...", response_model=MyAction),
    preprocessors=[[crop, resize]],
    filters=[AspectRatioFilter(min_ratio=0.3)],
    save_every=50,
    save_path=Path("results.pkl"),
)

# Auto-resumes if save_path exists; saves every 50 images + at the end
dataset = labeler.run(dataset_path)

# Post-processing
dataset.remove_class("unknown")
dataset.rename_class("old_name", "new_name")
dataset.balance(seed=42)

# Combine multiple datasets
combined = Dataset.combine(dataset1, dataset2)

# Export to YOLOv8 format
combined.export_yolov8(Path("yolo_dataset"), val_ratio=0.2, seed=42)

# Visualization
dataset.plot_distribution()
dataset.plot_grid(n=16, action="walking")
```

## Commands

```bash
uv pip install -e ".[dev]" --python /home/justin/machine_learning/yolo/.venv/bin/python
```

## Testing

Tests use **pytest** and live in `tests/`, mirroring the `src/action_labeler/` structure.

```bash
pytest tests/           # Run all tests
pytest tests/models/    # Run tests for a specific module
pytest -v -k "test_name"  # Run a specific test
```
